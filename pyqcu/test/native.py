"""Native KUPL/CUDA full-mode runners for the formal :mod:`pyqcu.test` port.

The tier-0 cases of the port are single-rank smoke gates on a small lattice.
This module adds the *full mode* the production backends really run:

* an MPI rank grid validated against the communicator (``prod(grid) == size``)
  and against the lattice (each local extent must be even, because both native
  backends split a parity-compact odd-Schur field);
* one thread count and one shared-memory-halo decision per process, applied
  through the documented environment variables;
* a backend probe for the KUPL ``ctypes`` library and the CUDA Cython bridge;
* one native Wilson/Clover dslash, one native solve and one native Legacy-MG
  solve with ``mass``/``kappa``/``csw``/``atol``/``max_iter`` reaching the ABI;
* an **independent** true residual: ``||D x - b|| / ||b||`` is reconstructed
  from the raw solution with the torch full-lattice Wilson-Clover operator
  (``pyqcu.torch.dslash``), never from the solver's own recurrence.

Execution topology
------------------
One Python process is one MPI rank; a CUDA process owns exactly one device and
``device_id``/``--device`` selects it.  Cross-device work is MPI partitioning of
the four-dimensional lattice, never several devices in one process.  When more
than one rank runs, the verification gathers the rank-local blocks into the
global field on rank zero and computes the reference there (the native rank
order is ``t`` fastest, ``x`` slowest -- ``lattice_geometry.cpp``), so the
residual is still an *independent* number and not a native re-application of
the same operator.

Skips (environmental absence only, always with a reason)
-------------------------------------------------------
``mpi_grid``          ``prod(lattice_grid) != MPI size``
``mpi_single_rank``   the case needs one rank where the job has more
``kupl_library``      ``pyqcu.kupl.qcu``/``libqcu.so`` is missing or incomplete
``cuda_bridge``       ``pyqcu.cuda.qcu`` is not built in this checkout
``cuda_device``       ``torch.cuda.is_available()`` is False or the device is
                      not visible
``shm_unavailable``   the shared halo was requested but cannot be used here
                      (single rank, more than one node, non-CPU backend)
``device_unavailable````--device`` names something the backend cannot own

Anything else (a contradictory ``--solver``/``--dslash``, a grid that does not
tile the lattice, a bad ``atol``) is a usage error, and a numeric mismatch is a
plain observation the harness gates on -- this module never decides PASS/FAIL.

Lifecycle and ABI ownership
---------------------------
Every native call runs the documented ordinary transaction::

    select a free slot in set_ptrs[0:30)
        -> applyInitQcu
        -> operation (and same-slot getters)
        -> applyEndQcu
        -> _SET_INDEX_ = (slot + 1) % 30

``_SET_INDEX_`` is never advanced before ``applyEndQcu``.  The control arrays
(``params`` int32[58], ``argv`` real[8], ``set_ptrs`` int64[100]) are a private
:class:`pyqcu.cuda._runtime.QcuRuntime` instance, so the module-level arrays of
``pyqcu.cuda.define``/``pyqcu.kupl.define`` are never passed to the library and
every runner verifies afterwards that they are unchanged.  Slots from index 30
onward belong to the multigrid asset table (``[30:60)`` coarse, ``[60:80)``
strict, ``80`` strict hierarchy) and are only written through the documented
Legacy-MG layout.

Import policy
-------------
Only the standard library and the stdlib-only siblings (:mod:`pyqcu.test.common`,
:mod:`pyqcu.test.policy`, :mod:`pyqcu.test.cache`) are imported at module import
time.  ``torch``, ``numpy``, ``h5py``, ``mpi4py`` and the backend modules are
imported inside the runners, so ``python -m pyqcu.test --list`` never pays for
them.
"""

from __future__ import annotations

import math
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, NoReturn, Sequence

from . import cache as cache_module
from . import policy
from .common import Skip, mpi_info, time_samples

__all__ = [
    "BACKENDS",
    "DEFAULT_ATOL_C128",
    "DEFAULT_ATOL_C64",
    "DEFAULT_BACKEND",
    "DEFAULT_CSW",
    "DEFAULT_LATTICE",
    "DEFAULT_MASS",
    "DEFAULT_MG_ATOL_C128",
    "DEFAULT_MG_ATOL_C64",
    "DEFAULT_MG_GRID",
    "BackendProbe",
    "NativeCase",
    "NativeEnvironment",
    "SKIP_TOKENS",
    "apply_environment",
    "case",
    "local_lattice",
    "mpi_rank",
    "mpi_size",
    "probe_backend",
    "process_grid",
    "rank_coordinate",
    "resolve_shm",
    "resolve_threads",
    "run_case",
    "run_dslash",
    "run_mg",
    "run_solve",
    "shm_available",
]

#: Backends this module drives.
BACKENDS: tuple[str, ...] = ("kupl", "cuda")
DEFAULT_BACKEND = "kupl"
#: Project correctness lattice (``AGENTS.md``, ``pyqcu.test.policy``).
DEFAULT_LATTICE: tuple[int, int, int, int] = policy.QUICK_LATTICE
DEFAULT_DTYPE: str = policy.QUICK_DTYPE
#: Tier-0 mass (``pyqcu.test.kupl_qcu.DEFAULT_MASS``).
DEFAULT_MASS = 0.17
#: Wilson default: no clover term, so the operator *is* the Wilson operator.
DEFAULT_CSW = 0.0
#: Native solve tolerance per dtype (the ``native_solve`` gate needs these).
DEFAULT_ATOL_C64 = 1.0e-6
DEFAULT_ATOL_C128 = 1.0e-10
#: Legacy-MG acceptance used for the *native* full-residual check is looser than
#: a solve: the C++ MG aborts the process when its internal full residual misses
#: the target, so the default is a value a c64 two-level solve reliably reaches.
DEFAULT_MG_ATOL_C64 = 1.0e-5
DEFAULT_MG_ATOL_C128 = 1.0e-9
#: Smallest native tolerance the solvers are asked for.
MIN_ATOL = 1.0e-13
DEFAULT_MAX_ITER = 1000
DEFAULT_MG_MAX_ITER = 500
DEFAULT_SEED = 42
DEFAULT_GAUGE_SIGMA = 0.1
DEFAULT_SOURCE_SEED = 7
DEFAULT_GAUGE_SEED = 42
#: ``define._SET_PTRS_COARSE_BASE_``: first multigrid asset slot.
SLOT_LIMIT = 30
#: ``define._SET_PTRS_SIZE_``: the whole pointer table.
SET_PTRS_SIZE = 100
#: ``kupl/define.h``: set_ptrs[99] selects the Legacy-MG fine layout.
LEGACY_LAYOUT_SLOT = 99
LEGACY_LAYOUT_ODD_SCHUR = 1
#: Legacy-MG coarse level parameter block (``define._MG_LEVEL*_``).
MG_LEVEL_BASE = 19
MG_LEVEL_STRIDE = 8
MG_LEVEL_MAX_ITER = 5
MG_LEVEL_DATA_TYPE = 6
MG_LEVEL_NUM_RESTART = 7
#: ``define._MG_LEVEL1_ATOL_``: first coarse tolerance in ``argv``.
ARGV_MG_LEVEL_BASE = 3
#: Default coarsening: 12 fine components, 12 coarse basis vectors.
DEFAULT_DOF: tuple[int, int] = (12, 12)
DEFAULT_MG_GRID: tuple[int, int, int, int] = (2, 2, 2, 2)
DEFAULT_NV_ITERS = 2
#: Null-vector inverse-iteration acceptance (the native builder's own default).
NV_TOLERANCE = 1.0e-2
#: Stencil build window/batch (the native builder's tuned defaults).
STENCIL_WINDOW = 8
STENCIL_BATCH = 8
#: Upper bound on the gathered verification workspace (fields are re-created on
#: rank zero; a production L24/L32 lattice would need tens of GiB per tensor).
REFERENCE_BUDGET_BYTES = 2 << 30
#: Environment variables this module owns (applied per run, then restored).
THREAD_VARIABLES: tuple[str, ...] = ("OMP_NUM_THREADS", "KUPL_OMP_THREADS")
SHM_VARIABLE = "KUPL_SHM_HALO"
#: OpenMP placement variables that conflict with the library's own binding.
PLACEMENT_VARIABLES: tuple[str, ...] = (
    "OMP_PROC_BIND", "OMP_PLACES", "GOMP_CPU_AFFINITY", "KMP_AFFINITY")
#: ``PYQCU_*`` variables the runners read (and, for the process grid, neutralise).
PYQCU_VARIABLES: tuple[str, ...] = (
    "PYQCU_KUPL_LIB", "PYQCU_KUPL_PROFILE", "PYQCU_PROCESS_GRID",
    "PYQCU_NULLVEC_CACHE")
PROCESS_GRID_VARIABLE = "PYQCU_PROCESS_GRID"
_SHM_TRUE = {"1", "true", "yes", "on"}
_SHM_FALSE = {"", "0", "false", "no", "off"}

#: Skip tokens this module can raise (``shm_unavailable`` is not in
#: ``policy.SKIP_TOKENS`` yet; it is carried as a ``"<token>: <reason>"`` prefix
#: so the reason stays self-describing and ``Result.skip_token`` stays valid).
SKIP_TOKENS: tuple[str, ...] = (
    "mpi_grid", "mpi_single_rank", "kupl_library", "cuda_bridge",
    "cuda_device", "shm_unavailable", "device_unavailable")

_SPACE = 4
_CASES = ("dslash", "solve", "mg")
_DSLASH_ALIASES = {"auto": "auto", "all": "auto", "both": "auto",
                   "wilson": "wilson", "wilson_dslash": "wilson",
                   "clover": "clover", "clover_dslash": "clover"}
_SOLVER_ALIASES = {
    "auto": "auto", "all": "auto",
    "cg": "cg", "conjugate_gradient": "cg", "normal_equations": "cg",
    "bistabcg": "bistabcg", "bicgstab": "bistabcg", "bistab": "bistabcg",
    "mg": "mg", "multigrid": "mg", "legacy": "mg",
    "legacy_multigrid": "mg", "multigrid_legacy": "mg",
}
_DTYPE_ALIASES = {"c64": "c64", "complex64": "c64", "float32": "c64",
                  "single": "c64", "c128": "c128", "complex128": "c128",
                  "float64": "c128", "double": "c128"}
_SOURCE_ALIASES = {"random": "random", "rand": "random",
                   "gaussian": "random", "point": "point", "delta": "point",
                   "ones": "ones", "unit": "ones", "one": "ones",
                   "wall": "ones"}
_GAUGE_ALIASES = {"random": "random", "rand": "random", "gaussian": "random",
                  "unit": "unit", "identity": "unit", "ones": "unit",
                  "one": "unit"}
#: ABI entry points each case kind needs before the first native call.
REQUIRED_SYMBOLS: dict[str, tuple[str, ...]] = {
    "dslash": ("applyInitQcu", "applyEndQcu", "applyWilsonDslashQcu"),
    "solve": ("applyInitQcu", "applyEndQcu", "applyCloversQcu",
              "applyWilsonBistabCgQcu", "applyCloverBistabCgQcu",
              "getCloverBistabCgIterationsQcu"),
    "mg": ("applyInitQcu", "applyEndQcu", "applyCloversQcu",
           "applyMultigridBuildNullVectorsQcu",
           "applyMultigridBuildStencilLocalQcu", "applyCloverMultigridQcu",
           "getCloverBistabCgIterationsQcu"),
}


# ---------------------------------------------------------------------------
# skips
# ---------------------------------------------------------------------------
def _token_is_known(token: str) -> bool:
    """True when ``policy`` already declares ``token`` in its skip vocabulary."""
    try:
        return bool(policy.is_skip_token(token))
    except Exception:  # noqa: BLE001 - a probe must not raise
        return False


def _skip(reason: Any, token: str) -> NoReturn:
    """Raise :class:`Skip` with a non-empty reason and a self-describing token.

    A token the policy already knows is reported through ``Skip.token``; a new
    one (``shm_unavailable``) is prefixed to the message so the reason names it
    even when the harness cannot classify it.
    """
    text = str(reason).strip() or "native backend is unavailable"
    if _token_is_known(token):
        raise Skip(text, token=token)
    raise Skip(f"{token}: {text}")


# ---------------------------------------------------------------------------
# MPI size/rank and the process grid
# ---------------------------------------------------------------------------
def mpi_size() -> int:
    """MPI world size, without importing a backend when that is avoidable.

    The launcher environment is authoritative enough to reject a wrong grid
    before any import; once ``pyqcu.cuda.define`` (the module the native ABI
    control arrays are built from) is loaded, its value wins because it is what
    the library will use.
    """
    return _bound_mpi()[0]


def mpi_rank() -> int:
    """MPI world rank of this process."""
    return _bound_mpi()[1]


def _bound_mpi() -> tuple[int, int]:
    module = sys.modules.get("pyqcu.cuda.define")
    if module is not None:
        size = getattr(module, "size", None)
        rank = getattr(module, "rank", None)
        if isinstance(size, int) and isinstance(rank, int) and size > 0:
            return int(size), int(rank)
    return mpi_info()


def _int4(values: Any, name: str) -> tuple[int, int, int, int]:
    """Normalize a four-extent value (mapping, string spelling or sequence)."""
    if isinstance(values, Mapping):
        values = tuple(values.values())
    if isinstance(values, (str, bytes)):
        text = str(values).strip().strip("()[]")
        parts: list[Any] = [part for part in text.replace(",", " ").split() if part]
    else:
        try:
            parts = list(values)
        except TypeError as error:
            raise ValueError(
                f"{name} must contain four integers, got {values!r}") from error
    try:
        extents = tuple(int(value) for value in parts)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must contain four integers, got {values!r}") from error
    if len(extents) != 4:
        raise ValueError(f"{name} must contain four extents, got {extents}")
    return extents  # type: ignore[return-value]


def process_grid(lattice: Sequence[int], grid: Any = None, *,
                 size: int | None = None) -> tuple[int, int, int, int]:
    """Validate the four-dimensional rank grid and return it.

    ``prod(grid) != MPI size`` is an *environmental* mismatch (the job was not
    launched with the ranks the case describes), so it raises the ``mpi_grid``
    skip exactly like the tier-0 native runners do.  A grid that does not tile
    the lattice, or that leaves an odd local extent, is a *usage* error: no
    launch could ever make it valid.
    """
    shape = policy.validate_lattice(lattice)
    world = mpi_size() if size is None else int(size)
    requested = _int4((1, 1, 1, 1) if grid is None else grid, "lattice_grid")
    if any(value <= 0 for value in requested):
        raise ValueError(f"lattice_grid extents must be positive, got {requested}")
    product = math.prod(requested)
    if product != world:
        _skip(f"lattice_grid {requested} describes {product} rank(s) but "
              f"MPI_COMM_WORLD has {world}; the native ABI needs one process "
              f"grid matching the world size", token="mpi_grid")
    for axis in range(_SPACE):
        if shape[axis] % requested[axis]:
            raise ValueError(
                f"lattice_grid {requested} does not tile lattice {shape}: "
                f"extent {shape[axis]} on axis {axis} is not divisible by "
                f"{requested[axis]}")
    local = tuple(shape[axis] // requested[axis] for axis in range(_SPACE))
    if any(extent % 2 for extent in local):
        raise ValueError(
            f"lattice_grid {requested} gives the odd local lattice {local}; "
            "both native backends split an even parity-compact field, so every "
            "local extent must be even")
    return requested


def local_lattice(lattice: Sequence[int],
                  grid: Sequence[int]) -> tuple[int, int, int, int]:
    """Rank-local extents of ``lattice`` under ``grid``."""
    shape = _int4(lattice, "lattice")
    requested = _int4(grid, "lattice_grid")
    return tuple(shape[axis] // requested[axis]
                 for axis in range(_SPACE))  # type: ignore[return-value]


def rank_coordinate(rank: int, grid: Sequence[int]) -> tuple[int, int, int, int]:
    """Rank coordinate in the native rank order (``x`` slowest, ``t`` fastest).

    ``cpp/kupl/qcu/src/lattice_geometry.cpp`` derives
    ``coord[dim] = (rank / stride) % grid[dim]`` with ``stride`` accumulated
    from ``t`` upward, i.e. ``rank = ((x*Gy + y)*Gz + z)*Gt + t``.  The
    verification pass relies on exactly this order when it reassembles the
    global field from the rank-local blocks.
    """
    requested = _int4(grid, "lattice_grid")
    value = int(rank)
    if value < 0 or value >= math.prod(requested):
        raise ValueError(f"rank {value} is outside grid {requested}")
    coordinate = []
    for axis in range(_SPACE - 1, -1, -1):
        coordinate.append(value % requested[axis])
        value //= requested[axis]
    return tuple(reversed(coordinate))  # type: ignore[return-value]


def _require_single_rank(*, size: int | None = None, reason: str) -> int:
    """Skip ``mpi_single_rank`` when the case can only be truthful on one rank."""
    world = mpi_size() if size is None else int(size)
    if world != 1:
        _skip(f"{reason}; MPI_COMM_WORLD has {world} rank(s)",
              token="mpi_single_rank")
    return world


# ---------------------------------------------------------------------------
# threads, shared halo, environment
# ---------------------------------------------------------------------------
def _first_int_environment(names: Sequence[str]) -> int | None:
    """Return the first integer launcher variable that is set, else ``None``."""
    for name in names:
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            return int(raw)
        except ValueError:
            continue
    return None


def resolve_threads(threads: Any = None) -> int:
    """Resolve the per-rank thread count (explicit wins, then the environment)."""
    if threads is not None:
        if isinstance(threads, bool) or not isinstance(threads, int):
            try:
                threads = int(threads)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"threads must be a positive integer, got {threads!r}") from error
        value = int(threads)
    else:
        value = 0
        for name in ("KUPL_OMP_THREADS", "OMP_NUM_THREADS"):
            raw = os.environ.get(name, "").strip()
            if raw.isdigit() and int(raw) > 0:
                value = int(raw)
                break
        if value <= 0:
            value = int(os.cpu_count() or 1)
    if value <= 0:
        raise ValueError(f"threads must be positive, got {value}")
    return value


def shm_available(*, size: int | None = None, backend: str = "kupl",
                  platform_name: str | None = None) -> tuple[bool, str]:
    """Decide whether the KUPL shared-memory halo can be used here.

    ``KUPL_SHM_HALO`` needs at least two ranks, all of them inside one shared
    memory domain (``MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)`` must return
    the whole communicator), and it only exists in the KUPL CPU backend.  When
    the launcher exposes the node-local size, a multi-node launch is rejected
    instead of silently falling back to the MPI exchange.
    """
    world = mpi_size() if size is None else int(size)
    system = sys.platform if platform_name is None else str(platform_name)
    if str(backend).strip().lower() != "kupl":
        return False, f"the shared halo is a KUPL CPU feature, not {backend}"
    if world < 2:
        return False, ("the shared halo exchanges faces between ranks; "
                       "MPI_COMM_WORLD has a single rank")
    if system != "linux":
        return False, f"shared-memory windows need Linux, this is {system}"
    local = _first_int_environment((
        "OMPI_COMM_WORLD_LOCAL_SIZE", "MV2_COMM_WORLD_LOCAL_SIZE",
        "SLURM_NTASKS_PER_NODE", "I_MPI_LOCAL_SIZE"))
    if local is not None and local < world:
        return False, (f"{world} ranks span at least two nodes "
                       f"(local size {local}); the shared window only covers "
                       "ranks of one node")
    return True, ""


def resolve_shm(shm: Any = None, *, size: int | None = None,
                backend: str = "kupl") -> bool:
    """Resolve the shared-halo request; skip ``shm_unavailable`` if unusable."""
    if shm is None:
        raw = os.environ.get(SHM_VARIABLE)
        if raw is None:
            return False
        text = raw.strip().lower()
        requested = text in _SHM_TRUE
        if not requested and text not in _SHM_FALSE:
            raise ValueError(
                f"{SHM_VARIABLE}={raw!r} is not a boolean spelling")
    else:
        requested = bool(shm)
    if not requested:
        return False
    available, reason = shm_available(size=size, backend=backend)
    if not available:
        _skip(f"the shared halo ({SHM_VARIABLE}=1) cannot be used: {reason}",
              token="shm_unavailable")
    return True


@dataclass(frozen=True)
class NativeEnvironment:
    """The process environment a native run applies, and what it replaced."""

    threads: int
    shm: bool
    values: Mapping[str, str] = field(default_factory=dict)
    previous: Mapping[str, str | None] = field(default_factory=dict)
    removed: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe report of the applied environment."""
        return {
            "threads": int(self.threads),
            "shm": bool(self.shm),
            "environment": dict(self.values),
            "environment_previous": {
                name: value for name, value in self.previous.items()},
            "environment_removed": list(self.removed),
            "environment_notes": list(self.notes),
        }


@contextmanager
def apply_environment(*, threads: Any = None, shm: Any = None,
                      size: int | None = None, backend: str = "kupl",
                      single_rank_reference: bool = True
                      ) -> Iterator[NativeEnvironment]:
    """Apply the thread/shared-halo environment for one run and restore it.

    ``OMP_NUM_THREADS`` and ``KUPL_OMP_THREADS`` are set *before* the first
    native call, which is the only moment a libgomp process still reads them;
    ``KUPL_SHM_HALO`` is set to the decided value so a run that asked for the
    MPI exchange is not silently given the shared window (the library enables
    it by default on one node).  OpenMP placement variables that fight the
    library's own NUMA binding are removed, and ``PYQCU_PROCESS_GRID`` is
    removed for a single-rank run so the torch reference operator cannot treat
    the gathered global lattice as partitioned.
    """
    world = mpi_size() if size is None else int(size)
    thread_count = resolve_threads(threads)
    shared = resolve_shm(shm, size=world, backend=backend)
    notes: list[str] = []
    values: dict[str, str] = {
        "OMP_NUM_THREADS": str(thread_count),
        "KUPL_OMP_THREADS": str(thread_count),
        "OMP_DYNAMIC": "FALSE",
        SHM_VARIABLE: "1" if shared else "0",
    }
    removed: list[str] = []
    if _environment_enabled("KUPL_NUMA_BIND"):
        conflicting = [name for name in PLACEMENT_VARIABLES
                       if os.environ.get(name) not in (None, "")]
        for name in conflicting:
            removed.append(name)
        if conflicting:
            notes.append(
                "removed OpenMP placement variables that conflict with the "
                f"library NUMA binding: {', '.join(conflicting)}")
    if world == 1 and single_rank_reference:
        override = os.environ.get(PROCESS_GRID_VARIABLE)
        if override is not None:
            removed.append(PROCESS_GRID_VARIABLE)
            notes.append(
                f"removed {PROCESS_GRID_VARIABLE}={override!r} so the "
                "full-lattice reference operator is not confused with a "
                "distributed process grid")
    previous: dict[str, str | None] = {}
    try:
        for name, value in values.items():
            previous[name] = os.environ.get(name)
            os.environ[name] = value
        for name in removed:
            previous[name] = os.environ.pop(name, None)
        environment = NativeEnvironment(
            threads=thread_count, shm=shared, values=dict(values),
            previous=previous, removed=tuple(removed), notes=tuple(notes))
        yield environment
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _environment_enabled(name: str) -> bool:
    """True when ``name`` is set to a truthy non-empty spelling."""
    raw = os.environ.get(name)
    return raw is not None and raw.strip().lower() in _SHM_TRUE


# ---------------------------------------------------------------------------
# backend probe
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BackendProbe:
    """What a runner needs once the backend prerequisites have passed."""

    backend: str
    module: Any
    define: Any
    library: str
    device: str
    torch_device: str
    size: int
    rank: int
    symbols: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe report of the probed backend."""
        return {
            "backend": self.backend,
            "library": self.library,
            "device": self.device,
            "torch_device": self.torch_device,
            "mpi_size": int(self.size),
            "mpi_rank": int(self.rank),
            "abi_symbols": list(self.symbols),
        }


def _resolve_backend(backend: Any) -> str:
    """Resolve ``auto``/``kupl``/``cuda`` (and the ``all`` spelling)."""
    text = "auto" if backend is None else str(backend).strip().lower()
    if text in ("auto", "", "all", "cpu"):
        return DEFAULT_BACKEND
    if text in BACKENDS:
        return text
    raise ValueError(
        f"backend must be one of auto/{'/'.join(BACKENDS)}, got {backend!r}")


def _library_text(module: Any) -> str:
    """Best-effort library path/module file of a backend binding."""
    for attribute in ("library_path",):
        getter = getattr(module, attribute, None)
        if callable(getter):
            try:
                return str(getter())
            except BaseException:  # noqa: BLE001 - diagnostics only
                pass
    library = getattr(module, "_LIBRARY", None)
    name = getattr(library, "_name", None)
    if name:
        return str(name)
    return str(getattr(module, "__file__", "") or "<unknown>")


def _import_kupl() -> Any:
    """Import the KUPL ctypes binding, skipping ``kupl_library`` when absent."""
    try:
        from pyqcu.kupl import qcu as module
    except (FileNotFoundError, OSError, ImportError) as error:
        _skip(f"the KUPL QCU library is unavailable: "
              f"{type(error).__name__}: {error}", token="kupl_library")
    return module


def _import_torch(token: str) -> Any:
    try:
        import torch
    except ImportError as error:
        _skip(f"torch is not installed: {type(error).__name__}: {error}",
              token=token)
    return torch


def _cuda_device(requested: Any = None, device_id: Any = None
                 ) -> tuple[Any, str]:
    """Return ``(torch, "cuda:N")`` or skip ``cuda_device``."""
    torch = _import_torch("cuda_device")
    try:
        available = bool(torch.cuda.is_available())
    except Exception as error:  # noqa: BLE001 - a probe must stay a SKIP
        _skip(f"torch.cuda.is_available() raised "
              f"{type(error).__name__}: {error}", token="cuda_device")
    if not available:
        _skip("torch.cuda.is_available() is False; the native CUDA bridge "
              "needs one visible device per process", token="cuda_device")
    if requested is not None and str(requested).strip().lower() not in (
            "", "auto", "gpu", "cuda"):
        candidate = str(requested).strip()
    elif device_id is not None:
        candidate = f"cuda:{int(device_id)}"
    elif requested is not None and str(requested).strip().lower() == "gpu":
        candidate = "cuda:0"
    else:
        candidate = os.environ.get("QCU_DEVICE", "cuda:0")
    try:
        selected = torch.device(candidate)
    except (RuntimeError, ValueError) as error:
        raise ValueError(
            f"invalid CUDA device {candidate!r}: {error}") from error
    if selected.type != "cuda":
        _skip(f"device {candidate!r} is not a CUDA device; this backend owns "
              "one CUDA device per process", token="device_unavailable")
    index = 0 if selected.index is None else int(selected.index)
    try:
        count = int(torch.cuda.device_count())
    except Exception as error:  # noqa: BLE001 - a probe must stay a SKIP
        _skip(f"torch.cuda.device_count() raised "
              f"{type(error).__name__}: {error}", token="cuda_device")
    if index < 0 or index >= count:
        _skip(f"device {index} is not visible; torch reports {count} CUDA "
              "device(s)", token="cuda_device")
    return torch, f"cuda:{index}"


def _import_cuda_bridge() -> tuple[Any, str]:
    """Import ``pyqcu.cuda.qcu``, skipping ``cuda_bridge`` when it is not built."""
    try:
        from pyqcu.cuda import qcu as module
    except ImportError as error:
        _skip(f"pyqcu.cuda.qcu is not importable: "
              f"{type(error).__name__}: {error}", token="cuda_bridge")
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        _skip("pyqcu.cuda.qcu resolves to a namespace package (no compiled "
              "qcu*.so): the Cython bridge is not built in this checkout",
              token="cuda_bridge")
    return module, str(module_file)


def _require_symbols(module: Any, names: Sequence[str], backend: str,
                     purpose: str) -> tuple[str, ...]:
    """Skip when the binding lacks an entry point the case needs."""
    missing = [name for name in names if not callable(getattr(module, name, None))]
    if missing:
        token = "cuda_bridge" if backend == "cuda" else "kupl_library"
        _skip(f"the {backend} binding does not export {', '.join(missing)} "
              f"(required for {purpose})", token=token)
    return tuple(names)


def probe_backend(backend: Any = "auto", *, device: Any = None,
                  device_id: Any = None, kind: str = "solve") -> BackendProbe:
    """Probe one native backend for one case kind.

    KUPL is a CPU backend: a non-CPU ``--device`` is ``device_unavailable``.
    CUDA requires a visible device *and* a built bridge, in that order, so an
    unbuilt checkout reports ``cuda_device`` only when the machine really has
    no device and ``cuda_bridge`` otherwise.
    """
    name = _resolve_backend(backend)
    if kind not in REQUIRED_SYMBOLS:
        raise ValueError(f"unknown native case kind {kind!r}; "
                         f"choose from {', '.join(_CASES)}")
    if name == "kupl":
        text = "cpu" if device is None else str(device).strip().lower()
        base = text.partition(":")[0]
        if base not in ("", "auto", "cpu", "host"):
            _skip(f"the KUPL backend is CPU-only; device {device!r} is not "
                  "available (supported: cpu)", token="device_unavailable")
        module = _import_kupl()
        from pyqcu.cuda import define  # private control-array definitions
        symbols = _require_symbols(module, REQUIRED_SYMBOLS[kind], name, kind)
        size = int(getattr(define, "size", 1))
        rank = int(getattr(define, "rank", 0))
        return BackendProbe(
            backend=name, module=module, define=define,
            library=_library_text(module), device="cpu",
            torch_device="cpu", size=size, rank=rank, symbols=symbols)

    _, torch_device = _cuda_device(device, device_id)
    module, module_file = _import_cuda_bridge()
    from pyqcu.cuda import define
    symbols = _require_symbols(module, REQUIRED_SYMBOLS[kind], name, kind)
    size = int(getattr(define, "size", 1))
    rank = int(getattr(define, "rank", 0))
    return BackendProbe(
        backend=name, module=module, define=define, library=module_file,
        device=torch_device, torch_device=torch_device, size=size, rank=rank,
        symbols=symbols)


# ---------------------------------------------------------------------------
# private ABI controls and the ordinary transaction
# ---------------------------------------------------------------------------
def _global_abi_snapshot() -> dict[str, Any]:
    """Clone the module-level ABI control arrays of the native packages."""
    snapshot: dict[str, Any] = {}
    for module_name in ("pyqcu.cuda.define", "pyqcu.kupl.define"):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        for attribute in ("params", "argv", "set_ptrs"):
            value = getattr(module, attribute, None)
            clone = getattr(value, "clone", None)
            if callable(clone):
                snapshot[f"{module_name}.{attribute}"] = clone()
    return snapshot


def _changed_global_arrays(snapshot: Mapping[str, Any]) -> list[str]:
    """Names of the global ABI arrays that changed since ``snapshot``."""
    import torch

    changed: list[str] = []
    for name, before in snapshot.items():
        module_name, _, attribute = name.rpartition(".")
        module = sys.modules.get(module_name)
        current = getattr(module, attribute, None) if module else None
        if current is None or not torch.equal(before, current):
            changed.append(name)
    return changed


class _Operation:
    """Operation handle valid only inside :meth:`_Controls.transaction`."""

    def __init__(self, controls: "_Controls", name: str) -> None:
        self.controls = controls
        self.name = name

    def __call__(self, *operands: Any) -> Any:
        """Run the transaction's own entry point on ``operands``."""
        return self.call(self.name, *operands)

    def call(self, name: str, *operands: Any) -> Any:
        """Run any entry point of the current slot (assets build, verify, solve)."""
        return self.invoke(name, operands)

    def invoke(self, name: str, operands: Sequence[Any],
               trailing: Sequence[Any] = ()) -> Any:
        """Call an entry point whose ``(set_ptrs, params)`` pair is not last.

        The Legacy-MG setup entry points take integer geometry *after*
        ``set_ptrs``/``params``, so the control pair is spliced in at its
        documented position instead of being appended.
        """
        self.controls.check_operands(operands)
        self.controls.check_operands(trailing)
        function = getattr(self.controls.module, name)
        return function(*operands, self.controls.set_ptrs, self.controls.params,
                        *trailing)

    def read(self, name: str, *operands: Any) -> Any:
        """Read a same-slot getter (iteration counts) before the slot is freed."""
        self.controls.check_operands(operands)
        getter = getattr(self.controls.module, name)
        return getter(*operands, self.controls.set_ptrs, self.controls.params)

    def check(self, operands: Sequence[Any],
              shapes: Sequence[Sequence[int] | None]) -> None:
        """Validate operands and their shapes before the native call."""
        self.controls.check_operands(operands)
        self.controls.check_shapes(operands, shapes)


class _Controls:
    """Own one private set of ABI controls and run ordinary transactions.

    The control arrays come from :meth:`pyqcu.cuda._runtime.QcuRuntime.from_lattice`,
    which validates the lattice/grid/dtype and never touches the module-level
    arrays.  The transaction is local (instead of ``QcuRuntime.call``) because
    the Legacy-MG setup entry points carry integer geometry arguments that the
    tensor-only ``QcuRuntime`` API cannot express, and because this port follows
    the documented ``applyEndQcu`` *then* ``_SET_INDEX_ += 1`` order.
    """

    def __init__(self, probe: BackendProbe, *, lattice: Sequence[int],
                 grid: Sequence[int], dtype: str, mass: float, csw: float,
                 atol: float, max_iter: int, plan: int, parity: int,
                 seed: int) -> None:
        from pyqcu.cuda import QcuRuntime

        self.probe = probe
        self.module = probe.module
        self.define = probe.define
        self.dtype_name = dtype
        self.torch_dtype = _torch_dtype(dtype)
        self.lattice = tuple(int(value) for value in lattice)
        self.grid = tuple(int(value) for value in grid)
        self.slot_limit = SLOT_LIMIT
        self.runtime = QcuRuntime.from_lattice(
            self.lattice, mass=float(mass), csw=float(csw), atol=float(atol),
            sigma=DEFAULT_GAUGE_SIGMA, max_iter=int(max_iter),
            dtype=self.torch_dtype, device=probe.torch_device, plan=int(plan),
            parity=int(parity), seed=int(seed), verbose=0, grid=self.grid,
            qcu_module=probe.module, slot_limit=SLOT_LIMIT)
        self.params = self.runtime.params
        self.argv = self.runtime.argv
        self.set_ptrs = self.runtime.set_ptrs
        self._index: int | None = None
        self.init_calls = 0
        self.end_calls = 0

    # -- controls ----------------------------------------------------------
    @property
    def kappa(self) -> float:
        return float(self.runtime.kappa)

    def configure(self, **values: Any) -> None:
        """Update the private controls without rebuilding the object."""
        self.runtime.configure(**values)

    def tensor(self, shape: Sequence[int], *, dtype: Any = None) -> Any:
        """Allocate a private operand with the runtime dtype/device."""
        import torch

        return torch.zeros(tuple(int(value) for value in shape),
                           dtype=self.torch_dtype if dtype is None else dtype,
                           device=self.runtime.device)

    def empty(self, shape: Sequence[int]) -> Any:
        """Allocate an uninitialized private operand (output buffers)."""
        import torch

        return torch.empty(tuple(int(value) for value in shape),
                           dtype=self.torch_dtype,
                           device=self.runtime.device)

    @property
    def control_values(self) -> dict[str, Any]:
        """Effective physics controls of the private arrays."""
        define = self.define
        return {
            "argv_mass": float(self.argv[define._MASS_]),
            "argv_atol": float(self.argv[define._ATOL_]),
            "argv_csw": float(self.argv[define._CSW_]),
            "params_max_iter": int(self.params[define._MAX_ITER_]),
            "params_plan": int(self.params[define._SET_PLAN_]),
            "params_parity": int(self.params[define._PARITY_]),
            "params_data_type": int(self.params[define._DATA_TYPE_]),
            "params_seed": int(self.params[define._SEED_]),
            "params_lattice": [int(self.params[index]) for index in (
                define._LAT_X_, define._LAT_Y_, define._LAT_Z_, define._LAT_T_)],
            "params_grid": [int(self.params[index]) for index in (
                define._GRID_X_, define._GRID_Y_, define._GRID_Z_, define._GRID_T_)],
            "params_node_rank": int(self.params[define._NODE_RANK_]),
            "params_node_size": int(self.params[define._NODE_SIZE_]),
        }

    def controls_applied(self, *, mass: float, csw: float, atol: float,
                         max_iter: int, grid: Sequence[int],
                         lattice: Sequence[int]) -> dict[str, bool]:
        """Whether each requested option really reached the ABI control array."""
        values = self.control_values
        resolution = 1e-6 if self.dtype_name == "c64" else 1e-12

        def _same(actual: float, expected: float) -> bool:
            return abs(float(actual) - float(expected)) <= resolution * max(
                1.0, abs(float(expected)))

        return {
            "mass_applied": _same(values["argv_mass"], mass),
            "csw_applied": _same(values["argv_csw"], csw),
            "atol_applied": _same(values["argv_atol"], atol),
            "max_iter_applied": values["params_max_iter"] == int(max_iter),
            "grid_applied": tuple(values["params_grid"]) == tuple(
                int(value) for value in grid),
            "lattice_applied": tuple(values["params_lattice"]) == tuple(
                int(value) for value in lattice),
            "dtype_applied": values["params_data_type"] == (
                self.define._LAT_C128_ if self.dtype_name == "c128"
                else self.define._LAT_C64_),
        }

    def check_operands(self, operands: Sequence[Any]) -> None:
        """Reject an operand the native ABI would abort on.

        Only array-like operands are checked; the Legacy-MG setup calls mix
        tensors with plain integers, which pass through untouched.
        """
        for position, value in enumerate(operands):
            pointer = getattr(value, "data_ptr", None)
            if not callable(pointer):
                # The Legacy-MG setup entry points mix tensors with plain
                # integer geometry and one real tolerance argument.
                if isinstance(value, (int, float)) and not isinstance(
                        value, bool):
                    continue
                if isinstance(value, bool):
                    continue
                raise TypeError(
                    f"native operand {position} must be a tensor or a scalar, "
                    f"got {type(value).__name__}")
            if getattr(value, "dtype", None) != self.torch_dtype:
                raise TypeError(
                    f"native operand {position} must have dtype "
                    f"{self.torch_dtype}, got {getattr(value, 'dtype', None)}")
            device = getattr(value, "device", None)
            if device is not None and str(device) != str(self.runtime.device):
                raise ValueError(
                    f"native operand {position} must live on "
                    f"{self.runtime.device}, got {device}")
            contiguous = getattr(value, "is_contiguous", None)
            if callable(contiguous) and not bool(contiguous()):
                raise ValueError(f"native operand {position} is not contiguous")

    def check_shapes(self, operands: Sequence[Any],
                     shapes: Sequence[Sequence[int] | None]) -> None:
        """Validate operand shapes (``None`` skips that position)."""
        for position, expected in enumerate(shapes):
            if expected is None:
                continue
            actual = tuple(int(value) for value in operands[position].shape)
            if actual != tuple(int(value) for value in expected):
                raise ValueError(
                    f"native operand {position} must have shape "
                    f"{tuple(expected)}, got {actual}")

    # -- lifecycle ---------------------------------------------------------
    def _allocate_slot(self) -> int:
        for index in range(self.slot_limit):
            if int(self.set_ptrs[index]) == 0:
                return index
        raise RuntimeError("all ordinary native set_ptrs slots are busy")

    @contextmanager
    def transaction(self, name: str) -> Iterator[_Operation]:
        """Run one ``init -> operation/getter -> end -> next slot`` transaction."""
        function_owner = getattr(self.module, name, None)
        if not callable(function_owner):
            raise AttributeError(
                f"the {self.probe.backend} binding has no entry point {name!r}")
        index = self._allocate_slot()
        self._index = index
        self.params[self.define._SET_INDEX_] = index
        self.init_calls += 1
        try:
            self.module.applyInitQcu(self.set_ptrs, self.params, self.argv)
        except BaseException:
            # No slot was initialized; release the pointer defensively but do
            # not advance _SET_INDEX_ (it may only move after a real end).
            try:
                self.module.applyEndQcu(self.set_ptrs, self.params)
            except BaseException:
                pass
            self.set_ptrs[index] = 0
            self._index = None
            raise
        try:
            yield _Operation(self, name)
        finally:
            try:
                self.module.applyEndQcu(self.set_ptrs, self.params)
                self.end_calls += 1
            finally:
                self.set_ptrs[index] = 0
                self.params[self.define._SET_INDEX_] = (
                    (index + 1) % self.slot_limit)

    @property
    def set_index(self) -> int | None:
        return self._index

    @property
    def next_index(self) -> int:
        return int(self.params[self.define._SET_INDEX_])

    @property
    def slot_released(self) -> bool:
        index = self._index
        if index is None:
            return True
        return int(self.set_ptrs[index]) == 0

    def lifecycle(self) -> dict[str, Any]:
        """Evidence that every slot was opened and released exactly once."""
        return {
            "set_index": self.set_index,
            "set_index_after": self.next_index,
            "slot_released": self.slot_released,
            "init_calls": self.init_calls,
            "end_calls": self.end_calls,
        }


# ---------------------------------------------------------------------------
# input construction (gauge, source, Clover terms)
# ---------------------------------------------------------------------------
def _torch_dtype(dtype_name: str) -> Any:
    torch = _import_torch("kupl_library")
    return torch.complex128 if dtype_name == "c128" else torch.complex64


def _real_dtype(torch: Any, complex_dtype: Any) -> Any:
    return torch.float64 if complex_dtype is torch.complex128 else torch.float32


def _compact_shape(lattice: Sequence[int]) -> tuple[int, int, int, int]:
    """Parity-compact ``[X, Y, Z, T/2]`` extents of the ABI tensors."""
    shape = tuple(int(value) for value in lattice)
    return (*shape[:3], shape[3] // 2)


def _gauge_shape(lattice: Sequence[int]) -> tuple[int, ...]:
    return (2, 3, 3, 4, *_compact_shape(lattice))


def _clover_shape(lattice: Sequence[int]) -> tuple[int, ...]:
    return (4, 3, 4, 3, *_compact_shape(lattice))


def _fermion_shape(lattice: Sequence[int]) -> tuple[int, ...]:
    return (2, 4, 3, *_compact_shape(lattice))


@contextmanager
def _preserve_torch_rng(torch: Any) -> Iterator[None]:
    """Restore the process-wide torch RNG state around gauge generation."""
    state = torch.random.get_rng_state()
    try:
        yield
    finally:
        torch.random.set_rng_state(state)


def _build_gauge(case: "NativeCase", torch: Any, dtype: Any
                 ) -> dict[str, Any]:
    """Build the parity-compact gauge field this rank owns.

    ``unit`` is exactly SU(3) (the identity links), ``random`` uses the project
    generator with the case seed and ``gauge_sigma`` disorder.  The generated
    tensor is stored in the compact ``[2, 3, 3, 4, X, Y, Z, T/2]`` ABI layout
    and the generator's RNG state is restored, so a runner cannot perturb the
    caller's stream.
    """
    import pyqcu.tools as tools

    compact = _compact_shape(case.lattice)
    full_shape = (3, 3, 4, *case.lattice)
    if case.gauge_kind == "unit":
        index = torch.arange(3)
        full = torch.zeros(full_shape, dtype=dtype)
        full[index, index, :] = 1.0 + 0.0j
    else:
        from pyqcu import lattice as lattice_module

        full = torch.empty(full_shape, dtype=dtype)
        with _preserve_torch_rng(torch):
            lattice_module.generate_gauge_field(
                full, sigma=float(case.gauge_sigma),
                seed=int(case.gauge_seed), verbose=False)
    full = full.contiguous()
    compact_gauge = tools.oooxyzt2poooxyzt(full).contiguous()
    expected = _gauge_shape(case.lattice)
    if tuple(compact_gauge.shape) != expected:
        raise RuntimeError(
            f"compact gauge shape {tuple(compact_gauge.shape)} != {expected}")
    su3 = None
    try:
        from pyqcu import lattice as lattice_module

        su3 = bool(lattice_module.check_su3(full, tol=1e-3, verbose=False))
    except BaseException:  # noqa: BLE001 - diagnostics only
        su3 = None
    return {
        "full": full,
        "compact": compact_gauge,
        "kind": case.gauge_kind,
        "sigma": (float(case.gauge_sigma) if case.gauge_kind == "random"
                  else None),
        "seed": (int(case.gauge_seed) if case.gauge_kind == "random" else None),
        "su3": su3,
        "compact_shape": tuple(compact),
    }


def _build_source(case: "NativeCase", torch: Any, dtype: Any
                  ) -> dict[str, Any]:
    """Build the two-parity compact source ``[2, 4, 3, X, Y, Z, T/2]``.

    ``ones`` fills every component with one (a wall source), ``point`` places a
    unit charge at ``(spin, color, site)``, ``random`` uses a private generator
    seeded by ``source_seed`` so the process-wide stream is untouched.
    """
    compact = _compact_shape(case.lattice)
    shape = (2, 4, 3, *compact)
    real_dtype = _real_dtype(torch, dtype)
    if case.source_kind == "point":
        source = torch.zeros(shape, dtype=dtype)
        source[(0, case.source_spin, case.source_color, *case.source_site)] = 1.0
    elif case.source_kind == "ones":
        source = torch.ones(shape, dtype=dtype)
    else:
        generator = torch.Generator().manual_seed(int(case.source_seed))
        single = torch.complex(
            torch.randn((4, 3, *compact), generator=generator,
                        dtype=real_dtype),
            torch.randn((4, 3, *compact), generator=generator,
                        dtype=real_dtype),
        ).to(dtype)
        source = torch.stack((single, single)).contiguous()
    source = source.contiguous()
    if tuple(source.shape) != shape:
        raise RuntimeError(f"source shape {tuple(source.shape)} != {shape}")
    return {
        "compact": source,
        "kind": case.source_kind,
        "site": list(case.source_site),
        "spin": int(case.source_spin),
        "color": int(case.source_color),
        "seed": (int(case.source_seed) if case.source_kind == "random" else None),
    }


def _native_clover_terms(controls: _Controls, gauge: Any, torch: Any
                         ) -> tuple[list[Any], list[Any]]:
    """Build the native Clover term and its inverse for both parities.

    One ``applyCloversQcu`` transaction per parity: the parity is a control read
    by ``applyInitQcu``, so it is written before the transaction opens and stays
    fixed inside it.  The previous plan is restored afterwards.
    """
    define = controls.define
    shape = _clover_shape(controls.lattice)
    previous_plan = int(controls.params[define._SET_PLAN_])
    controls.configure(plan=define._SET_PLAN2_)
    terms: list[Any] = []
    inverses: list[Any] = []
    try:
        for parity in (0, 1):
            term = controls.empty(shape)
            inverse = controls.empty(shape)
            controls.params[define._PARITY_] = parity
            with controls.transaction("applyCloversQcu") as operation:
                operation.check((term, inverse, gauge),
                                       (shape, shape, None))
                operation(term, inverse, gauge)
            terms.append(term)
            inverses.append(inverse)
    finally:
        controls.configure(plan=previous_plan)
    return terms, inverses


# ---------------------------------------------------------------------------
# verification: gathered global fields and the torch reference operator
# ---------------------------------------------------------------------------
class VerificationUnavailable(RuntimeError):
    """The independent check cannot run at this size (not a numeric failure)."""


def _comm(*, size: int) -> Any:
    """Return ``MPI.COMM_WORLD``, verified against the probed world size."""
    if int(size) <= 1:
        return None
    try:
        from mpi4py import MPI
    except ImportError as error:
        _skip(f"the distributed verification gathers the rank-local blocks and "
              f"needs mpi4py: {type(error).__name__}: {error}; run this case on "
              "one rank", token="mpi_single_rank")
    comm = MPI.COMM_WORLD
    if int(comm.Get_size()) != int(size):
        raise RuntimeError(
            f"mpi4py reports {comm.Get_size()} ranks but the native ABI was "
            f"built for {size}; the launcher environment is inconsistent")
    return comm


def _to_numpy(torch: Any, value: Any) -> Any:
    """Detach a tensor to a contiguous CPU numpy array."""
    import numpy as np

    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
        for attribute in ("resolve_conj", "resolve_neg"):
            method = getattr(value, attribute, None)
            if callable(method):
                value = method()
        value = value.cpu()
        numpy_method = getattr(value, "numpy", None)
        if callable(numpy_method):
            return np.ascontiguousarray(numpy_method())
    return np.ascontiguousarray(value)


def _global_compact(torch: Any, value: Any, *, lattice: Sequence[int],
                    grid: Sequence[int], size: int, purpose: str,
                    budget: int = REFERENCE_BUDGET_BYTES) -> Any:
    """Reassemble a rank-local compact field into the global field on rank 0.

    The native rank order is ``t`` fastest and ``x`` slowest, so the local
    blocks tile the global array exactly in that order.  Only rank 0 returns a
    tensor; every other rank gets ``None``.  The gathered workspace is bounded
    by ``budget`` so a production lattice cannot silently allocate tens of GiB.
    """
    import numpy as np

    shape = _int4(lattice, "lattice")
    requested = _int4(grid, "lattice_grid")
    if int(size) <= 1 or math.prod(requested) == 1:
        return value
    comm = _comm(size=size)
    local = _to_numpy(torch, value)
    elements = math.prod(shape) * max(1, int(local.size // max(
        1, math.prod(local.shape[-4:]))))
    estimated = int(elements) * int(local.dtype.itemsize) * 4
    if estimated > int(budget):
        raise VerificationUnavailable(
            f"gathering the {tuple(shape)} {purpose} needs about "
            f"{estimated / 2**30:.1f} GiB of reference workspace, above the "
            f"{int(budget) / 2**30:.1f} GiB budget")
    gathered = comm.gather(local, root=0)
    if comm.Get_rank() != 0:
        return None
    local_extents = tuple(int(extent) for extent in local.shape[-_SPACE:])
    lead = tuple(int(extent) for extent in local.shape[:-_SPACE])
    global_extents = tuple(local_extents[axis] * requested[axis]
                           for axis in range(_SPACE))
    output = np.zeros((*lead, *global_extents), dtype=local.dtype)
    for rank in range(int(size)):
        coordinate = rank_coordinate(rank, requested)
        selection = [slice(None)] * len(lead) + [
            slice(coordinate[axis] * local_extents[axis],
                  (coordinate[axis] + 1) * local_extents[axis])
            for axis in range(_SPACE)
        ]
        output[tuple(selection)] = gathered[rank]
    return torch.from_numpy(output)


def _expand_parity(tools: Any, value: Any) -> Any:
    """Expand a compact ``[2, ...X, Y, Z, T/2]`` field to the full lattice."""
    return tools.poooxyzt2oooxyzt(value.contiguous())


def _full_fields(torch: Any, tools: Any, *, solution: Any, source: Any,
                 gauge: Any, lattice: Sequence[int], grid: Sequence[int],
                 size: int, purpose: str) -> dict[str, Any] | None:
    """Gather and expand the solution/source/gauge needed by a reference run."""
    gathered: dict[str, Any] = {}
    # Every rank must enter *all* collectives in the same order: a non-root
    # rank that returned early would leave the others blocked in ``gather``.
    for name, value in (("solution", solution), ("source", source),
                        ("gauge", gauge)):
        gathered[name] = _global_compact(
            torch, value, lattice=lattice, grid=grid, size=size,
            purpose=f"{purpose} {name}")
    if any(value is None for value in gathered.values()):
        return None
    return {
        "solution": _expand_parity(tools, gathered["solution"]),
        "source": _expand_parity(tools, gathered["source"]),
        "gauge": _expand_parity(tools, gathered["gauge"]),
    }


def _operator_reference(torch: Any, dslash: Any, *, gauge: Any, kappa: float,
                        csw: float) -> Callable[[Any], Any]:
    """Return ``x -> D x`` for the full Wilson-Clover operator.

    ``D = A - kappa * hopping`` with ``A = I + csw * T``.  ``give_wilson`` with
    ``with_I=True`` is ``I - kappa * hopping`` and ``give_clover`` applies the
    pointwise ``T`` block, so the sum is the complete operator.  This is an
    independent path: it is built from ``pyqcu.torch.dslash`` on the raw
    solution, never from the native solver's recurrence.
    """
    real_dtype = _real_dtype(torch, gauge.dtype)
    kappa_tensor = torch.tensor([float(kappa)], dtype=real_dtype)
    clover_term = None
    if float(csw) != 0.0:
        clover_term = dslash.make_clover(
            gauge, kappa=kappa_tensor, csw=float(csw), verbose=False)

    def apply(vector: Any) -> Any:
        result = dslash.give_wilson(vector, gauge, kappa_tensor, with_I=True)
        if clover_term is not None:
            result = result + dslash.give_clover(vector, clover_term)
        return result

    return apply


def _true_residual(torch: Any, *, solution: Any, source: Any, gauge: Any,
                   lattice: Sequence[int], grid: Sequence[int], size: int,
                   kappa: float, csw: float, purpose: str,
                   budget: int = REFERENCE_BUDGET_BYTES) -> dict[str, Any]:
    """Independent ``||D x - b|| / ||b||`` on the full Wilson-Clover operator.

    ``solution``/``source``/``gauge`` are the rank-local compact fields; the
    result is reduced to every rank so every rank reports the same number.
    """
    import pyqcu.tools as tools
    from pyqcu.torch import dslash

    comm = _comm(size=size)
    started = time.perf_counter()
    fields = _full_fields(torch, tools, solution=solution, source=source,
                         gauge=gauge, lattice=lattice, grid=grid, size=size,
                         purpose=purpose)
    value: float | None = None
    source_norm = residual_norm = solution_norm = None
    if fields is not None:
        operator = _operator_reference(torch, dslash, gauge=fields["gauge"],
                                       kappa=kappa, csw=csw)
        applied = operator(fields["solution"])
        residual = applied - fields["source"]
        denominator = float(torch.linalg.vector_norm(fields["source"]))
        numerator = float(torch.linalg.vector_norm(residual))
        value = (math.inf if denominator == 0.0
                 else float(numerator / denominator))
        source_norm = denominator
        residual_norm = numerator
        solution_norm = float(torch.linalg.vector_norm(fields["solution"]))
    if comm is not None:
        value = comm.bcast(value, root=0)
        source_norm = comm.bcast(source_norm, root=0)
        residual_norm = comm.bcast(residual_norm, root=0)
        solution_norm = comm.bcast(solution_norm, root=0)
    return {
        "relative_residual": value,
        "source_norm": source_norm,
        "residual_norm": residual_norm,
        "solution_norm": solution_norm,
        "reference_seconds": time.perf_counter() - started,
        "residual_source": (
            "independent torch full Wilson-Clover reconstruction"
            + ("" if size <= 1 else
               " of the gathered global field (MPI all-blocks on rank 0)")),
    }


def _bare_hopping_reference(torch: Any, dslash: Any, *, source: Any,
                            gauge: Any, target: int) -> Any:
    """Bare hopping sum onto ``target`` parity, from the opposite parity.

    The native ``applyWilsonDslashQcu`` applies exactly this operator (the
    hopping sum without ``I`` and without ``kappa``): ``give_wilson_eo/oe``
    return ``-kappa * hopping``, so ``kappa = 1`` with a sign flip is the bare
    sum.  ``target`` 0 is the even destination fed by the odd source.
    """
    one = torch.tensor([1.0], dtype=_real_dtype(torch, gauge.dtype))
    if int(target) == 0:
        return -dslash.give_wilson_eo(src_o=source, U_eo=gauge, kappa=one)
    return -dslash.give_wilson_oe(src_e=source, U_eo=gauge, kappa=one)


def _relative_error(torch: Any, actual: Any, expected: Any) -> float:
    """``max|actual - expected| / max|expected|`` (``inf`` on shape mismatch)."""
    if tuple(actual.shape) != tuple(expected.shape):
        return math.inf
    if int(expected.numel()) == 0:
        return 0.0
    scale = float(expected.abs().max())
    difference = float((actual - expected).abs().max())
    if scale == 0.0:
        return difference
    return difference / scale


# ---------------------------------------------------------------------------
# normalized run options
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NativeCase:
    """One validated native case (everything the runners act on)."""

    kind: str = "solve"
    backend: str = DEFAULT_BACKEND
    lattice: tuple[int, int, int, int] = DEFAULT_LATTICE
    lattice_grid: tuple[int, int, int, int] | None = None
    dtype: str = DEFAULT_DTYPE
    seed: int = DEFAULT_SEED
    repeat: int = 1
    warmup: int = 1
    mass: float = DEFAULT_MASS
    kappa: float = 1.0 / (2.0 * DEFAULT_MASS + 8.0)
    csw: float = DEFAULT_CSW
    atol: float = DEFAULT_ATOL_C64
    max_iter: int = DEFAULT_MAX_ITER
    threads: int | None = None
    shm: bool | None = None
    device: str | None = None
    device_id: int | None = None
    source_kind: str = "ones"
    source_site: tuple[int, int, int, int] = (0, 0, 0, 0)
    source_spin: int = 0
    source_color: int = 0
    source_seed: int = DEFAULT_SOURCE_SEED
    gauge_kind: str = "random"
    gauge_sigma: float = DEFAULT_GAUGE_SIGMA
    gauge_seed: int = DEFAULT_GAUGE_SEED
    dslash: str = "auto"
    solver: str = "auto"
    mg_grid: tuple[int, int, int, int] = DEFAULT_MG_GRID
    mg_levels: int = 2
    dof: tuple[int, ...] = DEFAULT_DOF
    nv_iters: int = DEFAULT_NV_ITERS
    restart: int = 5
    coarse_max_iter: int = 200
    coarse_tol_factor: float = 1.0e4
    deflate: bool = True
    verify: bool = True
    cache_policy: str = "auto"
    cache: Path | None = None
    cache_directory: Path | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)

    @property
    def compact(self) -> tuple[int, int, int, int]:
        """Parity-compact ``[X, Y, Z, T/2]`` extents."""
        return _compact_shape(self.lattice)

    @property
    def fermion_shape(self) -> tuple[int, ...]:
        """Two-parity compact fermion shape of the ABI."""
        return _fermion_shape(self.lattice)

    @property
    def clover_shape(self) -> tuple[int, ...]:
        """Parity compact Clover-term shape of the ABI."""
        return _clover_shape(self.lattice)

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe description of the request (reported by every runner)."""
        return {
            "kind": self.kind,
            "backend": self.backend,
            "lattice": list(self.lattice),
            "lattice_grid": (None if self.lattice_grid is None
                             else list(self.lattice_grid)),
            "dtype": self.dtype,
            "seed": int(self.seed),
            "repeat": int(self.repeat),
            "warmup": int(self.warmup),
            "mass": float(self.mass),
            "kappa": float(self.kappa),
            "csw": float(self.csw),
            "atol": float(self.atol),
            "max_iter": int(self.max_iter),
            "threads": None if self.threads is None else int(self.threads),
            "shm": self.shm,
            "device": self.device,
            "device_id": self.device_id,
            "source": self.source_kind,
            "source_site": list(self.source_site),
            "gauge": self.gauge_kind,
            "gauge_sigma": float(self.gauge_sigma),
            "gauge_seed": int(self.gauge_seed),
            "source_seed": int(self.source_seed),
            "dslash": self.dslash,
            "solver": self.solver,
            "mg_grid": list(self.mg_grid),
            "mg_levels": int(self.mg_levels),
            "dof": list(self.dof),
            "nv_iters": int(self.nv_iters),
            "restart": int(self.restart),
            "coarse_max_iter": int(self.coarse_max_iter),
            "coarse_tol_factor": float(self.coarse_tol_factor),
            "deflate": bool(self.deflate),
            "verify": bool(self.verify),
            "cache_policy": self.cache_policy,
            "cache": None if self.cache is None else str(self.cache),
            "cache_directory": (None if self.cache_directory is None
                                else str(self.cache_directory)),
            "extra_options": dict(self.extras),
        }


def _container_value(container: Any, name: str) -> Any:
    """Read ``name`` from an options container (mapping or object)."""
    if container is None:
        return None
    if isinstance(container, Mapping):
        return container.get(name)
    return getattr(container, name, None)


def _pick(explicit: Any, container: Any, name: str, default: Any) -> Any:
    """Explicit keyword wins, then the options container, then the default."""
    if explicit is not None:
        return explicit
    value = _container_value(container, name)
    return default if value is None else value


def _normalize_dtype(value: Any) -> str:
    if value is None:
        return DEFAULT_DTYPE
    name = getattr(value, "__name__", None) or str(value)
    name = name.replace("torch.", "").strip().lower()
    try:
        return _DTYPE_ALIASES[name]
    except KeyError:
        raise ValueError(
            f"native backends support c64/c128 (complex64/complex128), "
            f"got {value!r}") from None


def _normalize_choice(value: Any, aliases: Mapping[str, str], name: str) -> str:
    if value is None:
        return "auto"
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return aliases[text]
    except KeyError:
        choices = ", ".join(sorted(set(aliases.values())))
        raise ValueError(f"{name} must be one of {choices}, got {value!r}") from None


def _resolve_mass_kappa(mass: Any, kappa: Any) -> tuple[float, float]:
    """Resolve ``(mass, kappa)`` with ``kappa = 1/(2*mass + 8)``."""
    if kappa is not None:
        value = float(kappa)
        if not math.isfinite(value) or value == 0.0:
            raise ValueError(f"kappa must be finite and non-zero, got {value}")
        derived = 0.5 / value - 4.0
        if mass is not None and not math.isclose(
                float(mass), derived, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(
                f"mass={float(mass)!r} and kappa={value!r} are inconsistent: "
                f"kappa={value!r} implies mass={derived!r}")
        return derived, value
    value = DEFAULT_MASS if mass is None else float(mass)
    if not math.isfinite(value):
        raise ValueError(f"mass must be finite, got {value}")
    denominator = 2.0 * value + 8.0
    if denominator == 0.0:
        raise ValueError(
            f"mass={value!r} is degenerate: kappa = 1/(2m+8) is infinite")
    return value, 1.0 / denominator


def _resolve_atol(atol: Any, *, dtype: str, kind: str) -> float:
    """Resolve the solver tolerance for ``kind`` and ``dtype``."""
    if atol is not None:
        value = float(atol)
    elif kind == "mg":
        value = (DEFAULT_MG_ATOL_C128 if dtype == "c128"
                 else DEFAULT_MG_ATOL_C64)
    else:
        value = DEFAULT_ATOL_C128 if dtype == "c128" else DEFAULT_ATOL_C64
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"atol must be finite and positive, got {value}")
    if value < MIN_ATOL:
        raise ValueError(
            f"atol={value!r} is below the smallest native tolerance "
            f"({MIN_ATOL:g}); a native solver error aborts the process")
    if value > 1.0:
        raise ValueError(
            f"atol={value!r} is not a relative residual tolerance (<= 1)")
    return value


def _resolve_max_iter(max_iter: Any, *, kind: str) -> int:
    if max_iter is None:
        value = DEFAULT_MG_MAX_ITER if kind == "mg" else DEFAULT_MAX_ITER
    else:
        value = int(max_iter)
    if value <= 0:
        raise ValueError(f"max_iter must be positive, got {value}")
    return value


def _resolve_int(value: Any, name: str, *, minimum: int = 0) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer, got {value!r}") from error
    if number < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {number}")
    return number


def _options(case: Any = None, *, kind: str = "solve", **explicit: Any
             ) -> NativeCase:
    """Merge a ``Case``/``Options`` container with explicit keywords.

    An explicit keyword always wins, then the container (``options``/``ctx`` or
    a ``Case``), then the default; an unset option therefore keeps the runner's
    own default instead of being overridden with ``None``.
    """
    if kind not in _CASES:
        raise ValueError(f"kind must be one of {', '.join(_CASES)}, got {kind!r}")
    container = explicit.pop("options", None)
    extras = dict(explicit.pop("extras", None) or {})
    if case is not None:
        if isinstance(case, Mapping):
            merged = dict(case)
            for name in ("lattice", "dtype", "seed"):
                if explicit.get(name) is None and name in merged:
                    explicit[name] = merged.pop(name)
            rest = {key: value for key, value in merged.items()
                    if value is not None}
            if rest and container is None:
                container = rest
        elif isinstance(case, (tuple, list)):
            if explicit.get("lattice") is None:
                explicit["lattice"] = case
        else:
            for name in ("lattice", "dtype", "seed"):
                if explicit.get(name) is None:
                    explicit[name] = getattr(case, name, None)
            if container is None:
                container = case

    def _get(name: str, default: Any = None) -> Any:
        return _pick(explicit.get(name), container, name, default)

    lattice = policy.validate_lattice(
        DEFAULT_LATTICE if _get("lattice") is None else _get("lattice"))
    dtype = _normalize_dtype(_get("dtype"))
    seed = _resolve_int(DEFAULT_SEED if _get("seed") is None else _get("seed"),
                        "seed")
    repeat = _resolve_int(_get("repeat", 1), "repeat", minimum=1)
    warmup = _resolve_int(_get("warmup", 1), "warmup", minimum=1)
    mass, kappa = _resolve_mass_kappa(_get("mass"), _get("kappa"))
    csw = float(_get("csw", DEFAULT_CSW))
    if not math.isfinite(csw):
        raise ValueError(f"csw must be finite, got {csw}")
    atol = _resolve_atol(_get("atol"), dtype=dtype, kind=kind)
    max_iter = _resolve_max_iter(_get("max_iter"), kind=kind)
    grid_value = _get("lattice_grid")
    lattice_grid = None if grid_value is None else _int4(grid_value, "lattice_grid")
    threads = _get("threads")
    threads = None if threads is None else resolve_threads(threads)
    shm = _get("shm")
    if isinstance(shm, str):
        spelling = shm.strip().lower()
        if spelling in _SHM_TRUE:
            shm = True
        elif spelling in _SHM_FALSE:
            shm = False
        else:
            raise ValueError(f"shm={shm!r} is not a boolean spelling")
    shm = None if shm is None else bool(shm)
    device = _get("device")
    device = None if device in (None, "") else str(device)
    device_id = _get("device_id")
    device_id = None if device_id is None else _resolve_int(device_id, "device_id")
    gauge_sigma = float(_get("gauge_sigma", DEFAULT_GAUGE_SIGMA))
    if not math.isfinite(gauge_sigma) or gauge_sigma < 0.0:
        raise ValueError(
            f"gauge_sigma must be finite and non-negative, got {gauge_sigma}")
    site = _int4(_get("source_site", (0, 0, 0, 0)), "source_site")
    for axis in range(_SPACE):
        if site[axis] < 0 or site[axis] >= lattice[axis]:
            raise ValueError(f"source_site {site} is outside lattice {lattice}")
    spin = _resolve_int(_get("source_spin", 0), "source_spin")
    color = _resolve_int(_get("source_color", 0), "source_color")
    if spin >= 4:
        raise ValueError(f"source_spin must be in [0, 4), got {spin}")
    if color >= 3:
        raise ValueError(f"source_color must be in [0, 3), got {color}")
    mg_grid = _int4(_get("mg_grid", DEFAULT_MG_GRID), "mg_grid")
    if any(value <= 0 for value in mg_grid):
        raise ValueError(f"mg_grid extents must be positive, got {mg_grid}")
    mg_levels = _resolve_int(_get("mg_levels", 2), "mg_levels", minimum=1)
    if mg_levels > 2:
        raise ValueError(
            f"the Legacy-MG port drives one coarse transition, so mg_levels "
            f"must be 1 or 2, got {mg_levels}")
    dof_value = _get("dof", DEFAULT_DOF)
    dof = tuple(_resolve_int(value, "dof", minimum=1) for value in dof_value)
    if len(dof) < 2 or dof[0] != 12:
        raise ValueError(
            f"dof must start with the 12-component fine field and list at "
            f"least two levels, got {dof}")
    nv_iters = _resolve_int(_get("nv_iters", DEFAULT_NV_ITERS), "nv_iters",
                            minimum=1)
    restart = _resolve_int(_get("restart", 5), "restart")
    coarse_max_iter = _resolve_int(_get("coarse_max_iter", 200),
                                    "coarse_max_iter", minimum=1)
    coarse_tol_factor = float(_get("coarse_tol_factor", 1.0e4))
    if not math.isfinite(coarse_tol_factor) or coarse_tol_factor <= 0.0:
        raise ValueError(
            f"coarse_tol_factor must be finite and positive, got "
            f"{coarse_tol_factor}")
    cache_policy = str(_get("cache_policy", "auto")).strip().lower()
    if cache_policy not in ("auto", "off", "require"):
        raise ValueError(
            f"cache_policy must be auto/off/require, got {cache_policy!r}")
    cache_value = _get("cache")
    cache_directory = _get("cache_directory")
    return NativeCase(
        kind=kind,
        backend=_resolve_backend(_get("backend")),
        lattice=lattice,
        lattice_grid=lattice_grid,
        dtype=dtype,
        seed=seed,
        repeat=repeat,
        warmup=warmup,
        mass=mass,
        kappa=kappa,
        csw=csw,
        atol=atol,
        max_iter=max_iter,
        threads=threads,
        shm=shm,
        device=device,
        device_id=device_id,
        source_kind=_normalize_choice(_get("source", "ones"), _SOURCE_ALIASES,
                                      "source"),
        source_site=site,
        source_spin=spin,
        source_color=color,
        source_seed=_resolve_int(_get("source_seed", DEFAULT_SOURCE_SEED),
                                 "source_seed"),
        gauge_kind=_normalize_choice(_get("gauge", "random"), _GAUGE_ALIASES,
                                     "gauge"),
        gauge_sigma=gauge_sigma,
        gauge_seed=_resolve_int(_get("gauge_seed", DEFAULT_GAUGE_SEED),
                                "gauge_seed"),
        dslash=_normalize_choice(_get("dslash"), _DSLASH_ALIASES, "dslash"),
        solver=_normalize_choice(_get("solver"), _SOLVER_ALIASES, "solver"),
        mg_grid=mg_grid,
        mg_levels=mg_levels,
        dof=dof,
        nv_iters=nv_iters,
        restart=restart,
        coarse_max_iter=coarse_max_iter,
        coarse_tol_factor=coarse_tol_factor,
        deflate=bool(_get("deflate", True)),
        verify=bool(_get("verify", True)),
        cache_policy=cache_policy,
        cache=(None if cache_value is None else Path(str(cache_value)).expanduser()),
        cache_directory=(None if cache_directory is None
                         else Path(str(cache_directory)).expanduser()),
        extras=extras,
    )


def _require_operator(option: NativeCase, *, dslash: str | None = None,
                      solver: str | None = None) -> None:
    """Reject a ``dslash``/``solver`` request that contradicts the runner."""
    if dslash is not None and option.dslash not in ("auto", dslash):
        raise ValueError(
            f"this runner applies the {dslash} operator, but "
            f"dslash={option.dslash!r} was requested")
    if solver is not None and option.solver not in ("auto", solver):
        raise ValueError(
            f"this runner uses the {solver} solver, but "
            f"solver={option.solver!r} was requested")


# ---------------------------------------------------------------------------
# shared observation blocks
# ---------------------------------------------------------------------------
def _common_observations(option: NativeCase, probe: BackendProbe,
                         grid: Sequence[int], environment: NativeEnvironment,
                         controls: _Controls, started: float) -> dict[str, Any]:
    """Observation block shared by every native runner."""
    return {
        "backend": probe.backend,
        "device": probe.device,
        "library": probe.library,
        "native_kind": option.kind,
        "lattice": list(option.lattice),
        "local_lattice": list(local_lattice(option.lattice, grid)),
        "lattice_grid": list(grid),
        "grid_coordinate": list(rank_coordinate(probe.rank, grid)),
        "mpi_size": int(probe.size),
        "mpi_rank": int(probe.rank),
        "dtype": option.dtype,
        "seed": int(option.seed),
        "mass": float(option.mass),
        "kappa": controls.kappa,
        "csw": float(option.csw),
        "atol": float(option.atol),
        "max_iter": int(option.max_iter),
        "gauge_kind": option.gauge_kind,
        "gauge_sigma": float(option.gauge_sigma),
        "gauge_seed": int(option.gauge_seed),
        "source_kind": option.source_kind,
        "source": {
            "kind": option.source_kind,
            "site": list(option.source_site),
            "spin": int(option.source_spin),
            "color": int(option.source_color),
            "seed": (int(option.source_seed)
                     if option.source_kind == "random" else None),
        },
        "repeat": int(option.repeat),
        "warmup": int(option.warmup),
        "dslash_option": option.dslash,
        "solver_option": option.solver,
        "mg_grid": list(option.mg_grid),
        "mg_levels": int(option.mg_levels),
        "dof": list(option.dof),
        "nv_iters": int(option.nv_iters),
        "restart": int(option.restart),
        "coarse_max_iter": int(option.coarse_max_iter),
        "coarse_tol_factor": float(option.coarse_tol_factor),
        "deflate": bool(option.deflate),
        "extra_options": dict(option.extras),
        "case_seconds": time.perf_counter() - started,
        **environment.as_dict(),
        **controls.control_values,
    }


def _finish(observations: dict[str, Any], controls: _Controls,
            global_before: Mapping[str, Any], samples: Sequence[float],
            immutability: Mapping[str, bool]) -> dict[str, Any]:
    """Add timing, lifecycle and global-ABI evidence, then return."""
    import statistics

    changed = _changed_global_arrays(global_before)
    observations.update(
        seconds=float(statistics.median(samples)),
        samples_seconds=list(samples),
        **controls.lifecycle(),
        global_abi_unchanged=not changed,
        changed_global_arrays=changed,
    )
    observations.update(immutability)
    if immutability:
        observations["inputs_unchanged"] = all(
            bool(value) for value in immutability.values())
    return observations


# ---------------------------------------------------------------------------
# runner core
# ---------------------------------------------------------------------------
def _run_native(option: NativeCase) -> dict[str, Any]:
    """Resolve the environment, probe the backend and run one case."""
    started = time.perf_counter()
    size = mpi_size()
    rank = mpi_rank()
    grid = process_grid(option.lattice, option.lattice_grid, size=size)
    declared = _resolve_backend(option.backend)
    if option.kind == "dslash":
        _require_single_rank(
            size=size,
            reason="the native dslash comparison needs the whole lattice for "
                   "its full-lattice reference")
    elif option.kind == "mg":
        # The coarse assets are built per rank by the *local* 33-stencil
        # builder, which is a serial geometry: it cannot represent the coarse
        # couplings across a rank boundary (the production driver builds the
        # stencils on a serial overlap patch or reads a ranked cache, neither of
        # which this port writes).  A multi-rank Legacy-MG case would therefore
        # run with a wrong coarse operator, so it is refused instead of
        # reported as a pass.
        _require_single_rank(
            size=size,
            reason="the Legacy-MG coarse assets are built per rank by the "
                   "serial local 33-stencil builder, which cannot represent "
                   "coarse couplings across a rank boundary")
    # Decide the shared halo before any import so an unusable request is a SKIP
    # and not a half-initialised backend.
    resolve_shm(option.shm, size=size, backend=declared)
    with apply_environment(threads=option.threads, shm=option.shm, size=size,
                           backend=declared) as environment:
        probe = probe_backend(option.backend, device=option.device,
                              device_id=option.device_id, kind=option.kind)
        if probe.size != size:
            raise RuntimeError(
                f"the backend reports MPI size {probe.size} but the launcher "
                f"environment says {size} (rank {rank}); refusing to run with "
                "inconsistent process grids")
        if option.kind == "dslash":
            return _dslash_case(option, probe, grid, environment, started)
        if option.kind == "solve":
            return _solve_case(option, probe, grid, environment, started)
        return _mg_case(option, probe, grid, environment, started)


def _dslash_case(option: NativeCase, probe: BackendProbe,
                 grid: Sequence[int], environment: NativeEnvironment,
                 started: float) -> dict[str, Any]:
    """Apply the native Wilson/Clover dslash once per parity and compare."""
    torch = _import_torch("kupl_library")
    from pyqcu.torch import dslash as dslash_module

    controls = _Controls(
        probe, lattice=option.lattice, grid=grid, dtype=option.dtype,
        mass=option.mass, csw=option.csw, atol=option.atol,
        max_iter=option.max_iter, plan=controls_plan(probe, "dslash"),
        parity=0, seed=option.seed)
    gauge = _build_gauge(option, torch, controls.torch_dtype)
    source = _build_source(option, torch, controls.torch_dtype)
    parity_shape = (4, 3, *option.compact)
    # The ABI operand is one parity block: the even target is fed by the odd
    # source and the odd target by the even source.
    hop_source = {0: source["compact"][1].contiguous(),
                  1: source["compact"][0].contiguous()}
    outputs = {0: controls.empty(parity_shape), 1: controls.empty(parity_shape)}
    clover = option.dslash == "clover"
    inverses = None
    if clover:
        _, inverses = _native_clover_terms(controls, gauge["compact"], torch)
    entry = "applyCloverDslashQcu" if clover else "applyWilsonDslashQcu"
    global_before = _global_abi_snapshot()
    gauge_before = gauge["compact"].clone()
    source_before = source["compact"].clone()

    def apply_parities() -> None:
        for target in (0, 1):
            # The parity is copied by applyInitQcu, so it is written before the
            # transaction opens and stays fixed inside it.
            controls.params[controls.define._PARITY_] = target
            with controls.transaction(entry) as operation:
                operation.check(
                    (outputs[target], hop_source[target], gauge["compact"]),
                    (parity_shape, parity_shape, None))
                operation(outputs[target], hop_source[target],
                          gauge["compact"])

    samples = time_samples(apply_parities, repeat=option.repeat,
                           warmup=option.warmup)

    errors: dict[int, float] = {}
    output_norm = 0.0
    for target in (0, 1):
        reference = _bare_hopping_reference(
            torch, dslash_module, source=hop_source[target],
            gauge=gauge["compact"], target=target)
        if inverses is not None:
            reference = dslash_module.give_clover(
                src=reference, clover_term=inverses[target], verbose=False)
        errors[target] = _relative_error(torch, outputs[target], reference)
        output_norm = max(output_norm,
                          float(torch.linalg.vector_norm(outputs[target])))
    relative = max(errors.values())
    observations = _common_observations(option, probe, grid, environment,
                                        controls, started)
    observations.update(
        check="dslash_clover" if clover else "dslash_wilson",
        abi_symbol=entry,
        equation=("clover_inverse_times_wilson_hopping" if clover
                  else "wilson_hopping"),
        csw_used_by_operator=bool(clover),
        mass_used_by_operator=False,
        source_parity={"even_target": 1, "odd_target": 0},
        relative=relative,
        relative_error=relative,
        relative_error_even_target=errors[0],
        relative_error_odd_target=errors[1],
        true_residual=relative,
        output_norm=output_norm,
        output_finite=bool(torch.isfinite(outputs[0]).all()
                           and torch.isfinite(outputs[1]).all()),
        reference=("give_clover(bare_hopping, native_clover_inverse)"
                   if clover else "-give_wilson_eo/oe(kappa=1)"),
        notes=[
            ("applyCloverDslashQcu applies A^-1 * hopping; the reference "
             "consumes the native A^-1, so the gated number isolates every "
             "native contribution except the Clover-term construction"
             if clover else
             "applyWilsonDslashQcu applies the bare hopping sum: mass/kappa "
             "reach argv[_MASS_] but not the operator"),
        ],
        **controls.controls_applied(
            mass=option.mass, csw=option.csw, atol=option.atol,
            max_iter=option.max_iter, grid=grid, lattice=option.lattice),
    )
    return _finish(observations, controls, global_before, samples, {
        "gauge_unchanged": bool(torch.equal(gauge["compact"], gauge_before)),
        "source_unchanged": bool(torch.equal(source["compact"], source_before)),
    })


def _solve_case(option: NativeCase, probe: BackendProbe,
                grid: Sequence[int], environment: NativeEnvironment,
                started: float) -> dict[str, Any]:
    """Run one native solve and reconstruct the true residual independently."""
    torch = _import_torch("kupl_library")
    controls = _Controls(
        probe, lattice=option.lattice, grid=grid, dtype=option.dtype,
        mass=option.mass, csw=option.csw, atol=option.atol,
        max_iter=option.max_iter, plan=controls_plan(probe, "solve"),
        parity=1, seed=option.seed)
    gauge = _build_gauge(option, torch, controls.torch_dtype)
    source = _build_source(option, torch, controls.torch_dtype)
    solution = controls.tensor(option.fermion_shape)
    # The CUDA ABI has no iteration getter on the Wilson solver, so a zero
    # Clover term is used there: with csw = 0 the Clover solver *is* the Wilson
    # operator (the documented tier-0 convention).
    clover = bool(option.csw != 0.0 or probe.backend == "cuda")
    terms: list[Any] | None = None
    inverses: list[Any] | None = None
    if clover:
        terms, inverses = _native_clover_terms(controls, gauge["compact"], torch)
    entry = "applyCloverBistabCgQcu" if clover else "applyWilsonBistabCgQcu"
    global_before = _global_abi_snapshot()
    gauge_before = gauge["compact"].clone()
    source_before = source["compact"].clone()
    state: dict[str, Any] = {"iterations": 0}

    def solve_once() -> None:
        with controls.transaction(entry) as operation:
            if clover:
                assert terms is not None and inverses is not None
                operation.check(
                    (solution, source["compact"], gauge["compact"]),
                    (option.fermion_shape, option.fermion_shape, None))
                operation(solution, source["compact"], gauge["compact"],
                          terms[0], terms[1], inverses[0], inverses[1])
            else:
                operation(solution, source["compact"], gauge["compact"])
            state["iterations"] = int(
                operation.read("getCloverBistabCgIterationsQcu"))

    samples = time_samples(solve_once, repeat=option.repeat,
                           warmup=option.warmup)
    iterations = int(state["iterations"])
    residual = _true_residual(
        torch, solution=solution, source=source["compact"],
        gauge=gauge["compact"], lattice=option.lattice, grid=grid,
        size=probe.size, kappa=controls.kappa, csw=option.csw,
        purpose="solve")
    value = residual["relative_residual"]
    observations = _common_observations(option, probe, grid, environment,
                                        controls, started)
    observations.update(
        check="solve_bistabcg",
        equation="wilson" if not clover else "wilson_clover",
        abi_symbol=entry,
        iteration_getter="getCloverBistabCgIterationsQcu",
        csw_used_by_operator=bool(clover),
        mass_used_by_operator=True,
        parity=1,
        iterations=iterations,
        iterations_available=iterations > 0,
        hit_max_iter=iterations >= int(option.max_iter),
        converged=bool(value is not None and value <= float(option.atol)),
        relative_residual=value,
        relative_error=value,
        true_residual=value,
        source_norm=residual["source_norm"],
        residual_norm=residual["residual_norm"],
        solution_norm=residual["solution_norm"],
        residual_source=residual["residual_source"],
        residual_verified=bool(value is not None),
        reference_seconds=residual["reference_seconds"],
        notes=[
            "the native solver fails closed (a native error aborts the "
            "process), so atol/max_iter are validated before the transaction",
        ],
        **controls.controls_applied(
            mass=option.mass, csw=option.csw, atol=option.atol,
            max_iter=option.max_iter, grid=grid, lattice=option.lattice),
    )
    return _finish(observations, controls, global_before, samples, {
        "gauge_unchanged": bool(torch.equal(gauge["compact"], gauge_before)),
        "source_unchanged": bool(torch.equal(source["compact"], source_before)),
    })


def controls_plan(probe: BackendProbe, kind: str) -> int:
    """``params[_SET_PLAN_]`` of one case kind (plain dslash vs solver)."""
    define = probe.define
    return int(define._SET_PLAN0_) if kind == "dslash" else int(
        define._SET_PLAN1_)


# ---------------------------------------------------------------------------
# Legacy-MG assets (cache reuse or native build) and wiring
# ---------------------------------------------------------------------------
def _device_tensor(torch: Any, value: Any, probe: BackendProbe,
                   dtype: Any) -> Any:
    """Move a cached host array onto the backend device/dtype."""
    import numpy as np

    array = np.ascontiguousarray(value)
    return torch.from_numpy(array).to(device=probe.torch_device, dtype=dtype)


def _build_mg_assets(option: NativeCase, controls: _Controls, gauge: Any,
                     terms: Sequence[Any], inverses: Sequence[Any],
                     shapes: Mapping[str, Sequence[int]]) -> dict[str, Any]:
    """Build the four coarse tensors with the native Legacy-MG builders.

    The null-vector inverse iteration and the 33-stencil build are the same
    collective entry points the production driver uses
    (``applyMultigridBuildNullVectorsQcu``/``applyMultigridBuildStencilLocalQcu``).
    Both live in one ordinary transaction each, which is what the wrappers
    expect; the parity is fixed at 1 (the odd-Schur fine field) before them.
    """
    define = controls.define
    fine_odd = _compact_shape(option.lattice)
    coarse = tuple(int(fine_odd[axis] // option.mg_grid[axis])
                   for axis in range(_SPACE))
    coarse_vectors, fine_components = int(option.dof[1]), 12
    assets = {name: controls.empty(shapes[name]) for name in cache_module.ASSET_NAMES}
    previous_plan = int(controls.params[define._SET_PLAN_])
    controls.configure(plan=define._SET_PLAN1_)
    controls.params[define._PARITY_] = 1
    controls.params[define._MG_NUM_LEVEL_] = int(option.mg_levels)
    try:
        with controls.transaction("applyMultigridBuildNullVectorsQcu") as operation:
            operation.check(
                (assets["lonv"], gauge["compact"], terms[0], terms[1],
                 inverses[0], inverses[1]),
                (shapes["lonv"], None, None, None, None, None))
            status = operation.invoke(
                "applyMultigridBuildNullVectorsQcu",
                (assets["lonv"], gauge["compact"], terms[0], terms[1],
                 inverses[0], inverses[1]),
                (coarse_vectors, fine_components, *fine_odd, *coarse,
                 int(option.nv_iters), int(option.seed), NV_TOLERANCE))
        if int(status) != 0:
            raise RuntimeError(
                f"the native null-vector build failed with status {status}")
        with controls.transaction("applyMultigridBuildStencilLocalQcu") as operation:
            operation.check(
                (assets["sit"], assets["hnn"], assets["hdg"], assets["lonv"]),
                (shapes["sit"], shapes["hnn"], shapes["hdg"], shapes["lonv"]))
            status = operation.invoke(
                "applyMultigridBuildStencilLocalQcu",
                (assets["sit"], assets["hnn"], assets["hdg"], assets["lonv"],
                 gauge["compact"], terms[0], terms[1], inverses[0],
                 inverses[1]),
                (coarse_vectors, fine_components, *fine_odd, *coarse,
                 STENCIL_WINDOW, STENCIL_BATCH))
        if int(status) != 0:
            raise RuntimeError(
                f"the native stencil build failed with status {status}")
    finally:
        controls.configure(plan=previous_plan)
    return assets


def _mg_identity(option: NativeCase, controls: _Controls, gauge: Any,
                 torch: Any, grid: Sequence[int]) -> dict[str, Any]:
    """Cache identity of this case (gauge digest only when it is global)."""
    digest = None
    if int(controls.probe.size) == 1:
        digest = cache_module.array_sha256(
            _to_numpy(torch, gauge["full"]))
    return cache_module.mg_identity(
        lattice=option.lattice, mg_grid=option.mg_grid, dtype=option.dtype,
        dof=tuple(option.dof[:option.mg_levels]), nv_iters=option.nv_iters,
        seed=option.seed, mass=option.mass, csw=option.csw,
        gauge_sha256=digest, lattice_grid=tuple(int(value) for value in grid),
        mpi_size=int(controls.probe.size))


def _resolve_mg_assets(option: NativeCase, controls: _Controls, gauge: Any,
                       terms: Sequence[Any], inverses: Sequence[Any],
                       torch: Any, grid: Sequence[int]
                       ) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    """Return the four coarse tensors, reusing or publishing a cache.

    ``cache_policy`` selects the behaviour: ``auto`` reuses a valid cache and
    otherwise builds (publishing the result), ``off`` always builds, and
    ``require`` refuses to build.  A single-file cache describes one rank's
    coarse slab, so a multi-rank request is a usage error instead of a silent
    cross-identity hit.
    """
    identity = _mg_identity(option, controls, gauge, torch, grid)
    shapes = cache_module.asset_shapes(identity)
    notes: list[str] = []
    path = (option.cache if option.cache is not None
            else cache_module.cache_path(identity, option.cache_directory))
    if option.cache_policy != "off":
        if int(controls.probe.size) > 1:
            raise ValueError(
                "a single-file Legacy-MG cache stores one rank's coarse slab; "
                "run one rank or set cache_policy='off' for a multi-rank MG "
                f"case (mpi_size={controls.probe.size})")
        try:
            entry = cache_module.validate_cache(path, identity=identity)
        except cache_module.CacheMiss:
            entry = None
        except cache_module.CacheError as error:
            notes.append(f"ignored unusable cache: {error}")
            entry = None
        if entry is not None:
            arrays = cache_module.read_cache(path, identity=identity,
                                             validate=False)
            assets = {
                name: _device_tensor(torch, arrays[name], controls.probe,
                                     controls.torch_dtype)
                for name in cache_module.ASSET_NAMES
            }
            return assets, entry, {
                "cache_reused": True, "cache_seconds": None,
                "cache_notes": notes, "cache_shapes": shapes}
        if option.cache_policy == "require":
            raise ValueError(
                f"cache_policy='require' but no valid Legacy-MG cache at {path}"
                + (f" ({notes[-1]})" if notes else ""))
    build_started = time.perf_counter()
    assets = _build_mg_assets(option, controls, gauge, terms, inverses, shapes)
    build_seconds = time.perf_counter() - build_started
    entry = None
    if option.cache_policy != "off" and int(controls.probe.size) == 1:
        entry = cache_module.create_cache(
            path, assets, identity=identity,
            setup_method="cxx-null-vector+local-33-stencil",
            setup_backend=f"{controls.probe.backend}-qcu",
            setup_seconds=build_seconds, notes=tuple(notes))
    return assets, entry, {
        "cache_reused": False, "cache_seconds": build_seconds,
        "cache_notes": notes, "cache_shapes": shapes}


def _configure_legacy_mg(option: NativeCase, controls: _Controls,
                         assets: Mapping[str, Any]) -> dict[str, Any]:
    """Wire the coarse assets into ``params``/``argv``/``set_ptrs``.

    The parameter block is the documented Legacy-MG one: level 0 is the fine
    odd-Schur field (``kVector`` components), every further level reads its
    ``E`` and coarse extents from ``params`` and its four asset pointers from
    ``set_ptrs[30 + 4*(level-1) : ... ]``.  ``set_ptrs[99]`` selects the
    odd-Schur fine layout, which is what the cache stores.
    """
    define = controls.define
    params, argv, set_ptrs = controls.params, controls.argv, controls.set_ptrs
    params[define._MG_NUM_LEVEL_] = int(option.mg_levels)
    params[define._MG_USE_DEFLATE_] = 1 if option.deflate else 0
    levels: list[dict[str, Any]] = []
    for level in range(1, int(option.mg_levels)):
        block = tuple(int(value) for value in assets["lonv"].shape)
        base = MG_LEVEL_BASE + (level - 1) * MG_LEVEL_STRIDE
        params[base + 0] = block[0]
        params[base + 1] = block[2]
        params[base + 2] = block[4]
        params[base + 3] = block[6]
        params[base + 4] = block[8]
        params[base + MG_LEVEL_MAX_ITER] = int(option.coarse_max_iter)
        params[base + MG_LEVEL_DATA_TYPE] = int(params[define._DATA_TYPE_])
        params[base + MG_LEVEL_NUM_RESTART] = int(option.restart)
        argv[ARGV_MG_LEVEL_BASE + level - 1] = float(
            option.atol * option.coarse_tol_factor * (3 ** (level - 1)))
        slot = SLOT_LIMIT + 4 * (level - 1)
        set_ptrs[slot + 0] = int(assets["lonv"].data_ptr())
        set_ptrs[slot + 1] = int(assets["hnn"].data_ptr())
        set_ptrs[slot + 2] = int(assets["hdg"].data_ptr())
        set_ptrs[slot + 3] = int(assets["sit"].data_ptr())
        levels.append({
            "level": int(level),
            "E": block[0],
            "coarse": [block[2], block[4], block[6], block[8]],
            "slot": int(slot),
            "coarse_max_iter": int(option.coarse_max_iter),
            "restart": int(option.restart),
            "atol": float(argv[ARGV_MG_LEVEL_BASE + level - 1]),
        })
    set_ptrs[LEGACY_LAYOUT_SLOT] = LEGACY_LAYOUT_ODD_SCHUR
    return {
        "mg_layout": "odd-schur",
        "mg_deflate": bool(option.deflate),
        "mg_level_parameters": levels,
    }


def _mg_case(option: NativeCase, probe: BackendProbe, grid: Sequence[int],
             environment: NativeEnvironment, started: float) -> dict[str, Any]:
    """Run one native Legacy-MG solve with a cache or a native asset build."""
    torch = _import_torch("kupl_library")
    controls = _Controls(
        probe, lattice=option.lattice, grid=grid, dtype=option.dtype,
        mass=option.mass, csw=option.csw, atol=option.atol,
        max_iter=option.max_iter, plan=controls_plan(probe, "mg"),
        parity=1, seed=option.seed)
    gauge = _build_gauge(option, torch, controls.torch_dtype)
    source = _build_source(option, torch, controls.torch_dtype)
    terms, inverses = _native_clover_terms(controls, gauge["compact"], torch)
    assets: dict[str, Any] = {}
    cache_entry = None
    cache_info: dict[str, Any] = {"cache_reused": False, "cache_seconds": None,
                                  "cache_notes": ["no coarse transition: "
                                                  "mg_levels=1"]}
    if int(option.mg_levels) > 1:
        assets, cache_entry, cache_info = _resolve_mg_assets(
            option, controls, gauge, terms, inverses, torch, grid)
    mg_parameters = _configure_legacy_mg(option, controls, assets)
    solution = controls.tensor(option.fermion_shape)
    global_before = _global_abi_snapshot()
    gauge_before = gauge["compact"].clone()
    source_before = source["compact"].clone()
    state: dict[str, Any] = {"iterations": 0, "verify_status": None}

    def solve_once() -> None:
        with controls.transaction("applyCloverMultigridQcu") as operation:
            operation.check(
                (solution, source["compact"], gauge["compact"], terms[0],
                 terms[1], inverses[0], inverses[1]),
                (option.fermion_shape, option.fermion_shape, None, None, None,
                 None, None))
            if option.verify:
                state["verify_status"] = int(operation.call(
                    "verifyCloverMultigridQcu", solution, source["compact"],
                    gauge["compact"], terms[0], terms[1], inverses[0],
                    inverses[1]))
            operation(solution, source["compact"], gauge["compact"], terms[0],
                      terms[1], inverses[0], inverses[1])
            state["iterations"] = int(
                operation.read("getCloverBistabCgIterationsQcu"))

    samples = time_samples(solve_once, repeat=option.repeat,
                           warmup=option.warmup)
    iterations = int(state["iterations"])
    residual = _true_residual(
        torch, solution=solution, source=source["compact"],
        gauge=gauge["compact"], lattice=option.lattice, grid=grid,
        size=probe.size, kappa=controls.kappa, csw=option.csw, purpose="MG")
    value = residual["relative_residual"]
    observations = _common_observations(option, probe, grid, environment,
                                        controls, started)
    observations.update(
        check="mg_legacy",
        variant="legacy",
        equation="wilson_clover",
        abi_symbol="applyCloverMultigridQcu",
        verify_symbol="verifyCloverMultigridQcu" if option.verify else None,
        verify_status=state["verify_status"],
        csw_used_by_operator=True,
        mass_used_by_operator=True,
        parity=1,
        iterations=iterations,
        iterations_available=iterations > 0,
        hit_max_iter=iterations >= int(option.max_iter),
        converged=bool(
            value is not None and value <= float(option.atol)
            and (state["verify_status"] in (None, 0))),
        relative_residual=value,
        relative_error=value,
        true_residual=value,
        source_norm=residual["source_norm"],
        residual_norm=residual["residual_norm"],
        solution_norm=residual["solution_norm"],
        residual_source=residual["residual_source"],
        residual_verified=bool(value is not None),
        reference_seconds=residual["reference_seconds"],
        asset_source=("cache" if cache_entry is not None
                      and cache_info.get("cache_reused") else "native"),
        asset_shapes={name: list(value_.shape)
                      for name, value_ in assets.items()},
        asset_names=sorted(assets),
        **mg_parameters,
        **cache_info,
        notes=[
            "the Legacy-MG entry point aborts the process when its internal "
            "full residual misses the target; atol is the acceptance used for "
            "both the native check and this independent reconstruction",
        ],
        **controls.controls_applied(
            mass=option.mass, csw=option.csw, atol=option.atol,
            max_iter=option.max_iter, grid=grid, lattice=option.lattice),
    )
    if cache_entry is not None:
        observations.update(cache_entry.as_dict())
    return _finish(observations, controls, global_before, samples, {
        "gauge_unchanged": bool(torch.equal(gauge["compact"], gauge_before)),
        "source_unchanged": bool(torch.equal(source["compact"], source_before)),
    })


# ---------------------------------------------------------------------------
# public runners
# ---------------------------------------------------------------------------
def run_dslash(case: Any = None, *, lattice: Any = None, lattice_grid: Any = None,
               dtype: Any = None, seed: Any = None, repeat: Any = None,
               warmup: Any = None, mass: Any = None, kappa: Any = None,
               csw: Any = None, atol: Any = None, max_iter: Any = None,
               source: Any = None, gauge: Any = None, gauge_sigma: Any = None,
               device: Any = None, device_id: Any = None, threads: Any = None,
               shm: Any = None, dslash: Any = None, solver: Any = None,
               gauge_seed: Any = None, source_seed: Any = None,
               source_site: Any = None, source_spin: Any = None,
               source_color: Any = None, backend: Any = None,
               options: Any = None, **extras: Any) -> dict[str, Any]:
    """Apply one native Wilson (default) or Clover dslash per parity.

    The reference is the independent torch path: the bare hopping sum
    ``-give_wilson_eo/oe(kappa=1)`` for ``dslash="wilson"``, and the native
    ``A^-1`` applied to it for ``dslash="clover"`` (the same convention as the
    tier-0 Clover case: the gated number isolates every native contribution
    except the Clover-term construction, which the MG/solve cases exercise).

    Required observation for an ``operator`` gate: ``relative_error``.
    The dslash comparison needs the whole lattice, so a multi-rank job is
    ``SKIP(mpi_single_rank)``; the solve and MG runners do support several
    ranks.
    """
    option = _options(
        case, kind="dslash", lattice=lattice, lattice_grid=lattice_grid,
        dtype=dtype, seed=seed, repeat=repeat, warmup=warmup, mass=mass,
        kappa=kappa, csw=csw, atol=atol, max_iter=max_iter, source=source,
        gauge=gauge, gauge_sigma=gauge_sigma, device=device,
        device_id=device_id, threads=threads, shm=shm, dslash=dslash,
        solver=solver, gauge_seed=gauge_seed, source_seed=source_seed,
        source_site=source_site, source_spin=source_spin,
        source_color=source_color, backend=backend, options=options,
        extras=extras)
    return _run_native(option)


def run_solve(case: Any = None, *, lattice: Any = None, lattice_grid: Any = None,
              dtype: Any = None, seed: Any = None, repeat: Any = None,
              warmup: Any = None, mass: Any = None, kappa: Any = None,
              csw: Any = None, atol: Any = None, max_iter: Any = None,
              source: Any = None, gauge: Any = None, gauge_sigma: Any = None,
              device: Any = None, device_id: Any = None, threads: Any = None,
              shm: Any = None, dslash: Any = None, solver: Any = None,
              gauge_seed: Any = None, source_seed: Any = None,
              source_site: Any = None, source_spin: Any = None,
              source_color: Any = None, backend: Any = None,
              options: Any = None, **extras: Any) -> dict[str, Any]:
    """Run one native BiCGStab solve and report the independent true residual.

    The operator is Wilson when ``csw == 0`` (the KUPL Wilson solver or, on
    CUDA, the Clover solver with a zero Clover term) and Wilson-Clover
    otherwise.  ``||D x - b|| / ||b||`` is reconstructed from the raw solution
    on the full lattice with ``pyqcu.torch.dslash``; several ranks are supported
    by gathering the rank-local blocks on rank zero.

    Required observations for a ``native_solver`` gate: ``relative_residual``,
    ``iterations``, ``max_iter``.
    """
    option = _options(
        case, kind="solve", lattice=lattice, lattice_grid=lattice_grid,
        dtype=dtype, seed=seed, repeat=repeat, warmup=warmup, mass=mass,
        kappa=kappa, csw=csw, atol=atol, max_iter=max_iter, source=source,
        gauge=gauge, gauge_sigma=gauge_sigma, device=device,
        device_id=device_id, threads=threads, shm=shm, dslash=dslash,
        solver=solver, gauge_seed=gauge_seed, source_seed=source_seed,
        source_site=source_site, source_spin=source_spin,
        source_color=source_color, backend=backend, options=options,
        extras=extras)
    _require_operator(option, solver="bistabcg")
    return _run_native(option)


def run_mg(case: Any = None, *, lattice: Any = None, lattice_grid: Any = None,
           mg_grid: Any = None, dtype: Any = None, seed: Any = None,
           repeat: Any = None, warmup: Any = None, mass: Any = None,
           kappa: Any = None, csw: Any = None, atol: Any = None,
           max_iter: Any = None, source: Any = None, gauge: Any = None,
           gauge_sigma: Any = None, device: Any = None, device_id: Any = None,
           threads: Any = None, shm: Any = None, dslash: Any = None,
           solver: Any = None, gauge_seed: Any = None, source_seed: Any = None,
           source_site: Any = None, source_spin: Any = None,
           source_color: Any = None, backend: Any = None, mg_levels: Any = None,
           dof: Any = None, nv_iters: Any = None, restart: Any = None,
           coarse_max_iter: Any = None, coarse_tol_factor: Any = None,
           deflate: Any = None, verify: Any = None, cache: Any = None,
           cache_policy: Any = None, cache_directory: Any = None,
           options: Any = None, **extras: Any) -> dict[str, Any]:
    """Run one native Legacy-MG solve (``applyCloverMultigridQcu``).

    The four coarse tensors come from a validated cache
    (:mod:`pyqcu.test.cache`) when one matches the case identity, otherwise
    from the native null-vector/33-stencil builders; the run then publishes the
    result so the next identical case reuses it.  ``cache_policy="off"`` always
    builds, ``"require"`` refuses to build.

    Required observations for an ``mg`` gate: ``converged``,
    ``relative_residual``; ``iterations`` is reported for the diagnostic
    comparison and is not the coarse-correction count.
    """
    option = _options(
        case, kind="mg", lattice=lattice, lattice_grid=lattice_grid,
        mg_grid=mg_grid, dtype=dtype, seed=seed, repeat=repeat, warmup=warmup,
        mass=mass, kappa=kappa, csw=csw, atol=atol, max_iter=max_iter,
        source=source, gauge=gauge, gauge_sigma=gauge_sigma, device=device,
        device_id=device_id, threads=threads, shm=shm, dslash=dslash,
        solver=solver, gauge_seed=gauge_seed, source_seed=source_seed,
        source_site=source_site, source_spin=source_spin,
        source_color=source_color, backend=backend, options=options,
        mg_levels=mg_levels, dof=dof, nv_iters=nv_iters, restart=restart,
        coarse_max_iter=coarse_max_iter,
        coarse_tol_factor=coarse_tol_factor, deflate=deflate, verify=verify,
        cache=cache, cache_policy=cache_policy,
        cache_directory=cache_directory, extras=extras)
    _require_operator(option, solver="mg")
    return _run_native(option)


_RUNNERS: dict[str, Callable[..., dict[str, Any]]] = {
    "dslash": run_dslash,
    "solve": run_solve,
    "mg": run_mg,
}


def run_case(kind: str, case: Any = None, **options: Any) -> dict[str, Any]:
    """Dispatch to :func:`run_dslash`/:func:`run_solve`/:func:`run_mg`."""
    try:
        runner = _RUNNERS[str(kind).strip().lower()]
    except KeyError:
        raise ValueError(
            f"unknown native case kind {kind!r}; choose from "
            f"{', '.join(_RUNNERS)}") from None
    return runner(case, **options)


# ---------------------------------------------------------------------------
# harness integration
# ---------------------------------------------------------------------------
def _default_expect(kind: str) -> Mapping[str, Mapping[str, Any]]:
    """Gate group of each case kind (never a bare threshold)."""
    if kind == "dslash":
        return dict(policy.RESIDUAL_GATES["operator"])
    if kind == "solve":
        return dict(policy.RESIDUAL_GATES["native_solver"])
    gates = dict(policy.RESIDUAL_GATES["mg"])
    # The variant is explicit: a runner that silently used another MG variant
    # must not pass as the Legacy check (same rule as ``units.py``).
    gates["variant"] = {"eq": "legacy"}
    return gates


def _default_requires(kind: str, backend: str) -> tuple[str, ...]:
    """Skip tokens a case declares for ``--list`` and the report.

    Only tokens the policy vocabulary knows are listed; ``shm_unavailable`` is
    deliberately absent because it is conditional on the launch (it is carried
    in the reason instead).
    """
    if backend == "cuda":
        tokens = ["cuda_bridge", "cuda_device"]
    else:
        tokens = ["kupl_library"]
    if kind == "dslash":
        tokens.append("mpi_single_rank")
    return tuple(tokens)


def case(kind: str, *, backend: Any = DEFAULT_BACKEND, tag: str | None = None,
         tier: int = 1, expect: Mapping[str, Mapping[str, Any]] | None = None,
         timeout: float | None = 600.0, **options: Any) -> Any:
    """Build a :class:`pyqcu.test.common.Case` for one native runner.

    The unmodified harness can then run the native matrix without touching
    ``units.py``::

        from pyqcu.test import Test
        from pyqcu.test.native import case

        Test(cases=(case("mg", backend="kupl", mass=-0.277, csw=1.0),
                    case("solve", backend="kupl"))).run()

    ``tag`` defaults to ``"<backend>/native/<kind>"`` and must start with the
    resolved backend name, which is exactly what ``Case`` enforces.
    """
    from .common import Case

    resolved = _resolve_backend(backend)
    option = _options(None, kind=kind, backend=resolved, **options)
    return Case(
        tag=tag or f"{resolved}/native/{kind}",
        backend=resolved,
        kind=str(kind).strip().lower(),
        runner=f"native:run_{str(kind).strip().lower()}",
        lattice=option.lattice,
        dtype=option.dtype,
        tier=int(tier),
        requires=_default_requires(str(kind).strip().lower(), resolved),
        expect=_default_expect(str(kind).strip().lower()) if expect is None
        else expect,
        timeout=timeout,
    )
