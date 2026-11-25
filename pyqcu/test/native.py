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
  (``pyqcu.torch.dslash``), never from the solver's own recurrence;
* a **bounded-memory** reference: :func:`reference_plan` declares the peak
  workspace of every decomposition *before* a tensor is allocated, rank-local
  blocks use a distributed ``D x - b`` (production layout, MPI halo from
  ``pyqcu.torch.tools._mpi_roll``, Allreduce of ``||r||^2``/``||b||^2``),
  whole-lattice fields use a tiled ``t``-slab reference, and
  :class:`VerificationUnavailable` is reserved for a reference that really does
  not fit the declared ``reference_budget``.

Execution topology
-----------------
One Python process is one MPI rank; a CUDA process owns exactly one device and
``device_id``/``--device`` selects it.  Cross-device work is MPI partitioning of
the four-dimensional lattice, never several devices in one process.  The ABI is
**always** told the rank-local lattice (``params[_LAT_X_.._LAT_T_]``) and the
process grid (``params[_GRID_*]``); every operand - gauge, source, solution,
Clover term - has the rank-local extent and ``prod(local) * prod(grid) ==
prod(global)`` is asserted before the first native call
(:func:`local_geometry`, :class:`LocalGeometry`).  The *global* lattice only
appears in the independent residual, in the identity of a global HDF5 input
file and in the observations.

When more than one rank runs, the residual keeps the same physical definition
(the full Wilson-Clover operator on the periodic lattice) but no longer gathers
the global field on rank zero: the native rank order is ``t`` fastest, ``x``
slowest (``lattice_geometry.cpp``), and each rank applies the operator to its
own block with the two faces of every partitioned axis exchanged through
``pyqcu.torch.tools._mpi_roll``.  ``||r||^2`` and ``||b||^2`` are Allreduced, so
every rank reports the same number while no rank holds the global solution,
source or gauge field - let alone the whole-lattice Clover term (about 11 GiB
for L48 c64), which is never materialised.


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
``reference_workspace`` the independent residual does not fit the declared
                      ``reference_budget``; taken **before** the native call so
                      a successful solve is never turned into an error

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
    "REFERENCE_BUDGET_BYTES",
    "REFERENCE_MODES",
    "BackendProbe",
    "LocalGeometry",
    "NativeCase",
    "NativeEnvironment",
    "SKIP_TOKENS",
    "VerificationUnavailable",
    "apply_environment",
    "case",
    "local_geometry",
    "local_lattice",
    "mpi_rank",
    "mpi_size",
    "operand_plan",
    "probe_backend",
    "process_grid",
    "rank_coordinate",
    "reference_plan",
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
#: Level count the KUPL Legacy-MG accepts (``num_levels_ < 1 || > 5`` in
#: ``cpp/kupl/qcu/src/lattice_clover_multigrid.cpp:195``).
MG_MAX_LEVELS = 5
#: Smallest coarse extent a *new* transition may produce.  The solver warns
#: below this ("too small", ``lattice_clover_multigrid.cpp:288``) and its block
#: validation then has nothing left to halve.
MG_COARSEST_EXTENT = 4
#: Components of the odd-Schur fine field (``kVector``); every transition's
#: ``lonv`` keeps the parent components as its second axis.
MG_FINE_COMPONENTS = 12
#: ``lonv``/``hnn``/``hdg``/``sit``: the four tensors of one coarse transition
#: (:data:`pyqcu.test.cache.ASSET_NAMES`, the cache format's dataset names).
MG_TRANSITION_ASSETS: tuple[str, ...] = tuple(cache_module.ASSET_NAMES)
#: Default coarsening: 12 fine components, 12 coarse basis vectors.
DEFAULT_DOF: tuple[int, int] = (12, 12)
DEFAULT_MG_GRID: tuple[int, int, int, int] = (2, 2, 2, 2)
DEFAULT_NV_ITERS = 2
#: Null-vector inverse-iteration acceptance (the native builder's own default).
NV_TOLERANCE = 1.0e-2
#: Stencil build window/batch (the native builder's tuned defaults).
STENCIL_WINDOW = 8
STENCIL_BATCH = 8
#: Declared peak workspace of the independent reference, in bytes.
#:
#: The default is one rank's budget, not rank 0's: the distributed path is
#: designed to stay inside it at the production L48 geometries (about 1.3 GiB
#: per rank at 32 ranks c64, 2.6 GiB at 16 ranks c64, from
#: :func:`reference_plan`), while the two allocations this module must not
#: repeat - the rank-0 gather of the three global fields (about 9.5 GiB for L48
#: c64) and the whole-lattice Clover term (about 11.4 GiB) - are both refused
#: with the estimate.  A case that needs more declares it through
#: ``reference_budget`` instead of silently allocating.
REFERENCE_BUDGET_BYTES = 4 << 30
#: Environment variables this module owns (applied per run, then restored).
THREAD_VARIABLES: tuple[str, ...] = ("OMP_NUM_THREADS", "KUPL_OMP_THREADS")
SHM_VARIABLE = "KUPL_SHM_HALO"
#: OpenMP placement variables that conflict with the library's own binding.
PLACEMENT_VARIABLES: tuple[str, ...] = (
    "OMP_PROC_BIND", "OMP_PLACES", "GOMP_CPU_AFFINITY", "KMP_AFFINITY")
#: Binding/hugepage variables the runner does not set but must record.  On the
#: Kunpeng nodes the same configuration measured 186.08 s without these and
#: 2.06 s with them (2026-11-23, L48 c64, 16 ranks), so a report that omits
#: them cannot be compared with one that has them.
BINDING_VARIABLES: tuple[str, ...] = (
    "KUPL_NUMA_BIND", "KUPL_MEMORY_NODE", "KUPL_HUGEPAGE")
#: Escape hatch for a deliberate bound-vs-unbound A/B.  A real Kunpeng
#: multi-rank run must never take it by accident: an unbound run is ~95x slower
#: and its numbers are not comparable with anything else.
ALLOW_UNBOUND_VARIABLE = "PYQCU_ALLOW_UNBOUND_KUPL"
#: ``PYQCU_*`` variables the runners read (and, for the process grid, neutralise).
PYQCU_VARIABLES: tuple[str, ...] = (
    "PYQCU_KUPL_LIB", "PYQCU_KUPL_PROFILE", "PYQCU_PROCESS_GRID",
    "PYQCU_NULLVEC_CACHE")
PROCESS_GRID_VARIABLE = "PYQCU_PROCESS_GRID"
_SHM_TRUE = {"1", "true", "yes", "on"}
_SHM_FALSE = {"", "0", "false", "no", "off"}

#: Skip tokens this module can raise (``shm_unavailable`` and
#: ``reference_workspace`` are not in ``policy.SKIP_TOKENS`` yet; they are
#: carried as a ``"<token>: <reason>"`` prefix so the reason stays
#: self-describing and ``Result.skip_token`` stays valid).
SKIP_TOKENS: tuple[str, ...] = (
    "mpi_grid", "mpi_single_rank", "kupl_library", "cuda_bridge",
    "cuda_device", "shm_unavailable", "device_unavailable",
    "reference_workspace")

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
    environment = mpi_info()
    module = sys.modules.get("pyqcu.cuda.define")
    if module is not None:
        size = getattr(module, "size", None)
        rank = getattr(module, "rank", None)
        if isinstance(size, int) and isinstance(rank, int) and size > 0:
            # ``pyqcu.test`` deliberately does not import mpi4py.  A stale
            # ``define`` imported before the launcher initialised PMIx keeps
            # its module defaults (size=1), while ``drun -mpi pmix`` exports
            # PMIX_RANK/PMIX_SIZE.  Prefer the live launcher environment when
            # it says this is a multi-rank world; real mpi4py-derived values
            # still win when they agree or when no launcher size is present.
            if size > 1 or environment[0] <= 1:
                return int(size), int(rank)
    return environment


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


@dataclass(frozen=True)
class LocalGeometry:
    """The global lattice, the rank grid and the rank-local extents.

    The native ABI is told the **local** lattice: ``params[_LAT_X_.._LAT_T_]``
    is what one rank owns and ``params[_GRID_*]`` is how the ranks tile the
    physical lattice (``pyqcu/testing/simple/kupl_multigrid.py::main`` builds
    ``make_params(local_lattice, ..., grid=grid)`` and allocates every operand
    from ``local_lattice``; ``pyqcu/cuda/_runtime.py::from_lattice`` writes its
    ``lattice`` argument straight into those slots).  Handing it the global
    extents - which this module did before - makes every rank read and write
    ``prod(grid)`` times its own slab.

    The invariant is one line::

        prod(local) * prod(grid) == prod(global)

    together with positive and **even** local extents.  Evenness is not
    cosmetic: both native backends split an odd-Schur parity-compact field
    (:func:`_compact_shape` halves the last axis), and because every local
    extent is even, every rank origin is even as well - which is exactly what
    makes the parity-compact layout separable per rank (the local checkerboard
    parity equals the global one on every rank).
    """

    lattice: tuple[int, int, int, int]
    local: tuple[int, int, int, int]
    grid: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        for name in ("lattice", "local", "grid"):
            values = tuple(int(value) for value in getattr(self, name))
            if len(values) != _SPACE or any(value <= 0 for value in values):
                raise ValueError(
                    f"{name} must be four positive extents, got {values}")
            object.__setattr__(self, name, values)
        for axis in range(_SPACE):
            if self.local[axis] % 2:
                raise ValueError(
                    f"local lattice {self.local} has the odd extent "
                    f"{self.local[axis]} on axis {axis}; both native backends "
                    "split an even parity-compact field, so every rank-local "
                    "extent must be even")
        if self.local_volume * self.grid_volume != self.volume:
            raise RuntimeError(
                f"rank geometry is inconsistent: prod(local) * prod(grid) = "
                f"{self.local_volume} * {self.grid_volume} != prod(global) = "
                f"{self.volume} (local {self.local}, grid {self.grid}, global "
                f"{self.lattice})")

    @property
    def volume(self) -> int:
        """Sites of the physical (global) lattice."""
        return math.prod(self.lattice)

    @property
    def local_volume(self) -> int:
        """Sites one rank owns."""
        return math.prod(self.local)

    @property
    def grid_volume(self) -> int:
        """Number of ranks in the grid."""
        return math.prod(self.grid)

    @property
    def partitioned(self) -> bool:
        """True when the grid really splits the lattice."""
        return self.grid_volume > 1 and self.local != self.lattice

    def check(self) -> "LocalGeometry":
        """Re-assert the invariants (cheap, used before every native call)."""
        if self.local_volume * self.grid_volume != self.volume:
            raise RuntimeError(
                f"prod(local)*prod(grid) != prod(global) for {self.as_dict()}")
        return self

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe report (the observations carry it next to ``lattice``)."""
        return {
            "lattice": [int(value) for value in self.lattice],
            "local_lattice": [int(value) for value in self.local],
            "lattice_grid": [int(value) for value in self.grid],
            "volume": int(self.volume),
            "local_volume": int(self.local_volume),
            "grid_volume": int(self.grid_volume),
            "local_even": True,
            "volume_check": bool(
                self.local_volume * self.grid_volume == self.volume),
            "partitioned": bool(self.partitioned),
        }


def local_geometry(lattice: Sequence[int], grid: Any = None, *,
                   size: int | None = None) -> LocalGeometry:
    """Validate the rank grid and return the rank-local geometry.

    `grid` is validated by :func:`process_grid` (so an ``mpi_grid`` mismatch is
    still an environmental SKIP and a non-tiling/odd grid is still a usage
    error), then the ``prod(local) * prod(grid) == prod(global)`` invariant and
    the evenness of every local extent are asserted.  Pure arithmetic: no
    torch, numpy or MPI import, so a report can ask "which slab would this rank
    own?" without a backend.
    """
    shape = policy.validate_lattice(lattice)
    requested = process_grid(shape, grid, size=size)
    local = local_lattice(shape, requested)
    geometry = LocalGeometry(lattice=shape, local=local, grid=requested)
    return geometry.check()


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
        "SLURM_NTASKS_PER_NODE", "I_MPI_LOCAL_SIZE", "PMIX_LOCAL_SIZE"))
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
    binding: Mapping[str, str] = field(default_factory=dict)

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
            "binding_environment": dict(self.binding),
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
    _require_kunpeng_binding(backend, world)
    notes: list[str] = []
    if (str(backend) == "kupl" and world > 1
            and not _environment_enabled("KUPL_NUMA_BIND")):
        notes.append(
            f"running unbound because {ALLOW_UNBOUND_VARIABLE} is set: these "
            "timings are ~95x slower than a NUMA-bound run and are not "
            "comparable with one")
    values: dict[str, str] = {
        "OMP_NUM_THREADS": str(thread_count),
        "KUPL_OMP_THREADS": str(thread_count),
        "OMP_DYNAMIC": "FALSE",
        SHM_VARIABLE: "1" if shared else "0",
    }
    removed: list[str] = []
    binding = {name: os.environ[name] for name in BINDING_VARIABLES
               if os.environ.get(name) not in (None, "")}
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
            previous=previous, removed=tuple(removed), notes=tuple(notes),
            binding=binding)
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


def _require_kunpeng_binding(backend: str, world: int) -> None:
    """Refuse a multi-rank KUPL run that is not fully bound on Kunpeng.

    The three variables are not interchangeable and the docstring used to claim
    all three while checking only the first: every measured Kunpeng number in
    this repo was taken with ``KUPL_NUMA_BIND=1 KUPL_MEMORY_NODE=local+16
    KUPL_HUGEPAGE=1`` together, and the identical L48 c64 16-rank solve took
    196.00 s unbound against 2.06 s fully configured (2026-11-23, jobs 2236451
    vs 2236485, i.e. 95x).  A run that keeps NUMA_BIND but loses the memory
    policy or hugepages is a configuration nobody has measured, so it is refused
    rather than recorded as if it were comparable.

    Single-rank runs and non-KUPL backends are untouched: the tier-0 gate runs
    one rank on ordinary hardware.  ``PYQCU_ALLOW_UNBOUND_KUPL=1`` exists only
    for a deliberate bound-vs-unbound A/B and is reported in the row notes.
    """
    if str(backend) != "kupl" or int(world) <= 1:
        return
    if _environment_enabled(ALLOW_UNBOUND_VARIABLE):
        return
    missing = [
        name for name in ("KUPL_NUMA_BIND", "KUPL_MEMORY_NODE", "KUPL_HUGEPAGE")
        if not (os.environ.get(name) or "").strip()
    ]
    # NUMA_BIND is a flag (truthy spelling), the other two are values.
    if not _environment_enabled("KUPL_NUMA_BIND") and "KUPL_NUMA_BIND" not in missing:
        missing.append("KUPL_NUMA_BIND")
    if not missing:
        return
    raise RuntimeError(
        "a multi-rank KUPL run must set KUPL_NUMA_BIND=1, "
        "KUPL_MEMORY_NODE=local+16 and KUPL_HUGEPAGE=1 on the Kunpeng nodes; "
        f"missing or false: {', '.join(sorted(missing))}.  The same "
        "configuration measured 196.00 s without the binding and 2.06 s with it "
        "(95x, 2026-11-23), and a partial binding is a configuration nothing in "
        f"this repo has measured. Set {ALLOW_UNBOUND_VARIABLE}=1 only for a "
        "deliberate unbound control.")


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
    arrays.  ``lattice`` is the **rank-local** lattice - the extents the ABI is
    told about and the shape every operand has - and ``global_lattice`` is the
    physical one; ``prod(local) * prod(grid) == prod(global)`` is asserted here,
    at the boundary where the two used to be confused.  The transaction is local
    (instead of ``QcuRuntime.call``) because the Legacy-MG setup entry points
    carry integer geometry arguments that the tensor-only ``QcuRuntime`` API
    cannot express, and because this port follows the documented
    ``applyEndQcu`` *then* ``_SET_INDEX_ += 1`` order.
    """

    def __init__(self, probe: BackendProbe, *, lattice: Sequence[int],
                 grid: Sequence[int], dtype: str, mass: float, csw: float,
                 atol: float, max_iter: int, plan: int, parity: int,
                 seed: int, global_lattice: Sequence[int] | None = None) -> None:
        from pyqcu.cuda import QcuRuntime

        self.probe = probe
        self.module = probe.module
        self.define = probe.define
        self.dtype_name = dtype
        self.torch_dtype = _torch_dtype(dtype)
        self.lattice = tuple(int(value) for value in lattice)
        self.grid = tuple(int(value) for value in grid)
        self.global_lattice = (self.lattice if global_lattice is None
                               else tuple(int(v) for v in global_lattice))
        if math.prod(self.lattice) * math.prod(self.grid) != math.prod(
                self.global_lattice):
            raise RuntimeError(
                f"the ABI would be told the local lattice {self.lattice} on a "
                f"{self.grid} grid, which does not tile the "
                f"{self.global_lattice} global lattice: "
                f"{math.prod(self.lattice)} * {math.prod(self.grid)} != "
                f"{math.prod(self.global_lattice)}")
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

    @property
    def geometry(self) -> dict[str, Any]:
        """What the ABI was told: local extents, grid and the volume check."""
        return {
            "abi_lattice": list(self.lattice),
            "global_lattice": list(self.global_lattice),
            "lattice_grid": list(self.grid),
            "abi_local_volume": int(math.prod(self.lattice)),
            "abi_volume_check": bool(
                math.prod(self.lattice) * math.prod(self.grid)
                == math.prod(self.global_lattice)),
        }

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


def _su3_flag(full: Any) -> bool | None:
    """``check_su3`` of a full-lattice gauge field (``None`` when unavailable)."""
    try:
        from pyqcu import lattice as lattice_module

        return bool(lattice_module.check_su3(full, tol=1e-3, verbose=False))
    except BaseException:  # noqa: BLE001 - diagnostics only
        return None


def _hdf5_identity(role: str, path: Path, lattice: Sequence[int],
                   dtype: str) -> dict[str, Any]:
    """Validate one ``role`` artifact and return its identity/observations.

    ``pyqcu.test.io`` is imported here (lazily: it is the module that imports
    ``h5py``) and it is the only reader: the file is opened, its dataset/shape/
    dtype/attributes are checked and real elements are read before it is
    accepted, and its SHA256 is streamed.  A file for another lattice or
    another complex precision is rejected with both values in the message
    instead of being cast silently - the 2026-11 matrix lost a whole L32 block
    to a dtype taken from a file *name*.
    """
    from . import io

    record = io.describe(path, role=role)
    declared = tuple(int(value) for value in record["lattice"])
    if declared != tuple(int(value) for value in lattice):
        raise ValueError(
            f"{role} file {path} holds lattice {list(declared)}, this case asks "
            f"{list(int(value) for value in lattice)}")
    wanted = "complex128" if str(dtype) == "c128" else "complex64"
    if str(record["dtype"]) != wanted:
        raise ValueError(
            f"{role} file {path} holds {record['dtype']} but the case asks "
            f"{dtype} ({wanted}); convert the file (pyqcu.test.io.load_"
            f"{role}(..., dtype=...)) instead of letting the read cast it")
    return {
        "origin": "hdf5",
        "kind": "file",
        "path": str(path),
        "layout": str(record["layout"]),
        # ``compact`` is the ABI tensor in this module's dicts, so the file
        # layout flag is named after the file, not after the operand.
        "file_compact": bool(record["compact"]),
        "lattice": [int(value) for value in declared],
        "dtype": str(record["dtype"]),
        "bytes": int(record["bytes"]),
        "file_bytes": int(record["file_bytes"]),
        "sha256": io.sha256_file(path),
    }


def _mpi_comm() -> Any:
    """The MPI communicator for a collective HDF5 read, or ``None``.

    ``pyqcu.tools`` already owns the capability probe (``h5py.get_config().mpi``
    plus a live ``driver='mpio'`` open), so this only reuses the flag; the
    communicator itself always comes from ``mpi4py`` (``pyqcu.tools`` does not
    re-export it, and importing it from there raised ``ImportError`` on every
    rank, which the blanket ``except`` turned into "MPI is unavailable").  Only
    ``ImportError`` is caught here: a broken *name* must surface instead of
    silently selecting the serial path.

    A one-rank communicator resolves to ``None``: with nothing to coordinate, a
    plain serial open is correct and cheaper, and handing the comm back would
    turn every single-rank artifact read into a collective ``driver='mpio'``
    open - a new failure mode where a serial open already works.

    On more than one rank the decision is allgathered: one rank entering a
    collective open while another reads independently would hang, so every rank
    must agree before any of them opens the file.  ``None`` means "read
    independently", never "skip the read".

    Nothing on this path returns ``None`` because of a local failure: only
    ``ImportError`` (no mpi4py at all, which is uniform across ranks) and a world
    size of one do.  An MPI that imported but cannot be sized raises, because
    that condition is process-local and silently choosing the serial path is how
    one rank ends up outside a collective the others are already inside.
    """
    try:
        from mpi4py import MPI
        from pyqcu.tools import HAS_MPI_SUPPORT
    except ImportError:  # no mpi4py/capability probe is a normal single-node case
        return None
    try:
        comm = MPI.COMM_WORLD
        size = int(comm.Get_size())
    except Exception as error:  # noqa: BLE001 - see below
        # Do NOT return None here.  This decision is process-local, so a rank
        # whose Get_size() raised would take the serial path while the others
        # entered the allgather below and the collective HDF5 open after it -
        # a hang with no message.  MPI that imported but cannot be sized is
        # broken in this process, so failing loudly is the only outcome that
        # cannot be mistaken for "read it independently".
        raise RuntimeError(
            "mpi4py imported but MPI_COMM_WORLD could not be sized "
            f"({type(error).__name__}: {error}); refusing to guess whether to "
            "use a collective HDF5 read, because other ranks may already be in "
            "one") from error
    if size <= 1:
        return None
    votes = comm.allgather(1 if HAS_MPI_SUPPORT else 0)
    if len(set(votes)) != 1:
        raise RuntimeError(
            "h5py MPI support differs between ranks; a collective HDF5 read "
            "must be uniform, so refusing instead of risking a hang")
    return comm if votes[0] else None


def _read_block(path: Path, *, geometry: LocalGeometry, rank: int,
                compact: bool, dtype: str) -> tuple[Any, str]:
    """Read **only this rank's lattice block** from a field artifact.

    ``io.load_gauge``/``io.load_source`` return the whole global field, and the
    caller then slices the rank's block out of it.  On N ranks that reads N
    times the file: the L64x128 c64 gauge is 9.66 GB, so 128 ranks moved 1.24 TB
    and held ~9.5 GB RSS each (job 2236885, 2026-11-23).

    This reads the block directly as an HDF5 hyperslab.  The lattice axes are
    the last four in both layouts - ``[3,3,4,X,Y,Z,T]`` / ``[4,3,X,Y,Z,T]`` and
    the parity-compact ``[2,3,3,4,X,Y,Z,T/2]`` / ``[2,4,3,X,Y,Z,T/2]`` - so the
    selection is one slice per axis and the byte count becomes the block size
    (75 MB for the case above) instead of the file size.

    With a parallel HDF5 the open is collective (``driver='mpio'``, world
    communicator): the ranks' slabs tile the file exactly once, which lets
    MPI-IO coalesce 128 interleaved regular selections into large sequential
    reads.  That matters here because the artifacts are stored **contiguously**
    (``chunks=None``), so one rank's slab is ~1.18e6 short runs along ``t`` and
    an independent serial read of it is far worse than a collective one.

    Returns ``(array, mode)`` with ``mode`` one of ``"mpio"``, ``"slab"`` or
    ``"full"`` so the report records which path produced the numbers.
    """
    import h5py
    import numpy as np

    coordinate = rank_coordinate(rank, geometry.grid)
    comm = _mpi_comm()
    mode = "mpio" if comm is not None else "slab"
    with (h5py.File(path, "r", driver="mpio", comm=comm)
          if comm is not None else h5py.File(path, "r")) as h:
        dataset = h["data"]
        selection = [slice(None)] * (dataset.ndim - _SPACE)
        for axis in range(_SPACE):
            extent = geometry.local[axis]
            if compact and axis == _SPACE - 1:
                extent //= 2
            start = coordinate[axis] * extent
            selection.append(slice(start, start + extent))
        block = dataset[tuple(selection)]
    array = np.ascontiguousarray(block)
    wanted = np.complex128 if str(dtype) == "c128" else np.complex64
    if array.dtype != wanted:
        array = array.astype(wanted)
    return array, mode


def _build_gauge(case: "NativeCase", torch: Any, dtype: Any,
                 geometry: LocalGeometry, rank: int = 0) -> dict[str, Any]:
    """Build the parity-compact gauge field **this rank owns**.

    The ABI operand is the rank-local slab ``[2, 3, 3, 4, X, Y, Z, T/2]`` of
    ``geometry.local``, never the global field: the extents the ABI is told are
    the ones the operands really have.

    ``unit`` is exactly SU(3) (the identity links), ``random`` uses the project
    generator with the case seed and ``gauge_sigma`` disorder.  A ``random``
    gauge is drawn **per rank** (``seed + rank``) so a partitioned run does not
    give every rank the same periodic block; on one rank the seed is the case
    seed, so a single-rank run is bit-identical to before.  A
    ``case.gauge_file`` (canonical :mod:`pyqcu.test.io` HDF5, full or
    parity-compressed) is **read** instead - validated against the *global*
    lattice and dtype the file describes - and its origin/path/layout/SHA256
    are returned plus ``local_slice`` describing the cut; there is no fallback
    path, so an unreadable, mismatching or truncated file raises instead of
    silently producing a generated field.  The generator's RNG state is
    restored, so a runner cannot perturb the caller's stream.
    """
    import pyqcu.tools as tools

    local = tuple(int(value) for value in geometry.local)
    compact = _compact_shape(local)
    full_shape = (3, 3, 4, *local)
    cut = {
        "local_slice": bool(geometry.partitioned),
        "source_lattice": list(local),
        "file_lattice": (list(geometry.lattice) if geometry.partitioned
                         else list(local)),
    }
    if case.gauge_file is not None:
        path = Path(case.gauge_file).expanduser()
        identity = _hdf5_identity("gauge", path, geometry.lattice, case.dtype)
        from . import io

        array, read_mode = _read_block(
            path, geometry=geometry, rank=rank,
            compact=bool(identity["file_compact"]), dtype=str(case.dtype))
        if identity["file_compact"]:
            # ``[2, 3, 3, 4, X, Y, Z, T/2]`` is the ABI layout already; the
            # block read above is exactly this rank's slice of it.
            compact_gauge = torch.from_numpy(array).to(dtype).contiguous()
            array = None
            full = tools.poooxyzt2oooxyzt(compact_gauge).contiguous()
        else:
            full = torch.from_numpy(array).to(dtype).contiguous()
            array = None
            compact_gauge = tools.oooxyzt2poooxyzt(full).contiguous()
        expected = _gauge_shape(local)
        if tuple(compact_gauge.shape) != expected:
            raise RuntimeError(
                f"gauge file {path} compacts to {tuple(compact_gauge.shape)}, "
                f"the ABI needs {expected}")
        return {
            "full": full,
            "compact": compact_gauge,
            "compact_shape": tuple(compact),
            "su3": _su3_flag(full),
            "sigma": None,
            "seed": None,
            "read_mode": read_mode,
            **identity,
            **cut,
        }

    seed = int(case.gauge_seed)
    if geometry.partitioned:
        # A partitioned random gauge is drawn per rank: the assembled global
        # field is the juxtaposition of the blocks, and giving every rank the
        # same seed would make it periodic with the *local* extent instead.
        seed += int(rank)
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
                seed=seed, verbose=False)
    full = full.contiguous()
    compact_gauge = tools.oooxyzt2poooxyzt(full).contiguous()
    expected = _gauge_shape(local)
    if tuple(compact_gauge.shape) != expected:
        raise RuntimeError(
            f"compact gauge shape {tuple(compact_gauge.shape)} != {expected}")
    return {
        "full": full,
        "compact": compact_gauge,
        "kind": case.gauge_kind,
        "origin": case.gauge_kind,
        "path": None,
        "layout": None,
        "dtype": str(dtype).replace("torch.", ""),
        "bytes": int(full.numel() * full.element_size()),
        "sha256": None,
        "sigma": (float(case.gauge_sigma) if case.gauge_kind == "random"
                  else None),
        "seed": seed,
        "su3": _su3_flag(full),
        "compact_shape": tuple(compact),
        **cut,
    }


def _build_source(case: "NativeCase", torch: Any, dtype: Any,
                  geometry: LocalGeometry, rank: int = 0) -> dict[str, Any]:
    """Build the two-parity compact source ``[2, 4, 3, X, Y, Z, T/2]``.

    The extents are the rank-local ones (see :func:`_build_gauge`), so the
    operand is what the ABI is told the *local* lattice is.  ``ones`` fills
    every component with one (a wall source, identical on every rank because
    the assembled global field is all ones), ``point`` places a unit charge at
    the **global** ``(spin, color, site)`` in the parity block the global
    checkerboard demands - only the owning rank sets it, and a rank that does
    not own the site returns an all-zero slab - and ``random`` uses a private
    generator seeded by ``source_seed`` (``+ rank`` on a partitioned run) so the
    process-wide stream is untouched.  A ``case.source_file`` (canonical
    :mod:`pyqcu.test.io` HDF5, full or parity-compressed) is **read** instead
    and recorded with its SHA256; a missing or mismatching file raises rather
    than falling back to a generator.
    """
    import pyqcu.tools as tools

    local = tuple(int(value) for value in geometry.local)
    compact = _compact_shape(local)
    shape = (2, 4, 3, *compact)
    real_dtype = _real_dtype(torch, dtype)
    cut = {
        "local_slice": bool(geometry.partitioned),
        "source_lattice": list(local),
        "file_lattice": (list(geometry.lattice) if geometry.partitioned
                         else list(local)),
    }
    site = [int(value) for value in case.source_site]
    parity = sum(site) % 2
    if case.source_file is not None:
        path = Path(case.source_file).expanduser()
        identity = _hdf5_identity("source", path, geometry.lattice, case.dtype)
        from . import io

        array, read_mode = _read_block(
            path, geometry=geometry, rank=rank,
            compact=bool(identity["file_compact"]), dtype=str(case.dtype))
        if identity["file_compact"]:
            source = torch.from_numpy(array).to(dtype).contiguous()
            array = None
        else:
            full = torch.from_numpy(array).to(dtype).contiguous()
            array = None
            source = tools.oooxyzt2poooxyzt(full).contiguous()
            full = None
        if tuple(source.shape) != shape:
            raise RuntimeError(
                f"source file {path} compacts to {tuple(source.shape)}, the ABI "
                f"needs {shape}")
        return {
            "compact": source,
            "kind": "file",
            "site": site,
            "spin": int(case.source_spin),
            "color": int(case.source_color),
            "seed": None,
            "read_mode": read_mode,
            **identity,
            **cut,
        }
    if case.source_kind == "point":
        source = torch.zeros(shape, dtype=dtype)
        origin = (rank_coordinate(rank, geometry.grid)
                  if geometry.partitioned else (0, 0, 0, 0))
        local_site = tuple(site[axis] - origin[axis] * local[axis]
                           for axis in range(_SPACE))
        owned = all(0 <= local_site[axis] < local[axis] for axis in range(_SPACE))
        if owned:
            source[(parity, case.source_spin, case.source_color,
                    *local_site)] = 1.0
    elif case.source_kind == "ones":
        source = torch.ones(shape, dtype=dtype)
    else:
        seed = int(case.source_seed)
        if geometry.partitioned:
            seed += int(rank)
        generator = torch.Generator().manual_seed(seed)
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
        "origin": case.source_kind,
        "path": None,
        "layout": None,
        "dtype": str(dtype).replace("torch.", ""),
        "bytes": int(source.numel() * source.element_size()),
        "sha256": None,
        "site": site,
        "parity": int(parity),
        "spin": int(case.source_spin),
        "color": int(case.source_color),
        "seed": (int(case.source_seed) if case.source_kind == "random" else None),
        **cut,
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
# verification: bounded-memory independent residuals
# ---------------------------------------------------------------------------
class VerificationUnavailable(RuntimeError):
    """The independent check cannot run inside the declared budget.

    This is never a numeric verdict: it says the *reference* could not be
    computed at this size, which is why the reasoning (the estimate and the
    budget) always travels in the message, and why the runners report it as an
    unavailable observation instead of a converged/not-converged decision.
    """


#: Complex elements per lattice site one *rank-local* reference holds at once.
#:
#: A declared upper bound, not a measurement.  A rank-local stage keeps the
#: local gauge field (36 complex numbers per site) and its adjoint (36), the
#: Clover term (144), the solution and the right-hand side (12 each), the
#: destination plus the four hopping terms of one ward (about 84), one rolled
#: copy of the solution (12) and - because the caller keeps them alive - the
#: three parity-compact operands (60).  Measuring a real L48 run is the
#: operator's job; this number only decides whether the path is attempted.
REFERENCE_DISTRIBUTED_SITE_ELEMENTS = 480
#: Complex elements per site one *tiled* slab holds: the same stages over a
#: ``t``-slab that is extended by one compact index (two sites) on each side.
REFERENCE_TILED_SITE_ELEMENTS = 480
#: Complex elements per site the historical rank-0 gather materialises *before*
#: the operator runs (the gathered per-rank blocks plus the assembled global
#: solution, source and gauge arrays).
REFERENCE_GATHER_SITE_ELEMENTS = 120
#: Complex elements per site of one whole-lattice reference: the gather, the
#: parity expansion, the Clover term and the operator temporaries.
REFERENCE_WHOLE_SITE_ELEMENTS = 480
#: The Clover term of the whole lattice (``[4,3,4,3,X,Y,Z,T]``, 144 per site).
REFERENCE_CLOVER_SITE_ELEMENTS = 144
#: Accepted ``reference_mode`` values.  ``auto`` uses the rank-local distributed
#: path for rank-local blocks and the tiled path for whole-lattice fields,
#: ``distributed``/``tiled`` force one of them, ``gather`` keeps the historical
#: rank-0 gather (small lattices; also the oracle the tests compare against).
REFERENCE_MODES: tuple[str, ...] = ("auto", "distributed", "tiled", "gather")
#: Byte multipliers accepted by :func:`_resolve_reference_budget`.
_BUDGET_UNITS: Mapping[str, int] = {
    "": 1, "b": 1,
    "k": 10 ** 3, "kb": 10 ** 3, "ki": 2 ** 10, "kib": 2 ** 10,
    "m": 10 ** 6, "mb": 10 ** 6, "mi": 2 ** 20, "mib": 2 ** 20,
    "g": 10 ** 9, "gb": 10 ** 9, "gi": 2 ** 30, "gib": 2 ** 30,
    "t": 10 ** 12, "tb": 10 ** 12, "ti": 2 ** 40, "tib": 2 ** 40,
}


def _element_bytes(dtype_name: str) -> int:
    """Bytes of one complex element of ``dtype_name`` (``c64``/``c128``)."""
    return 16 if _normalize_dtype(dtype_name) == "c128" else 8


def _resolve_reference_mode(value: Any) -> str:
    """Normalize ``reference_mode`` (``auto``/``distributed``/``tiled``/``gather``)."""
    text = "auto" if value is None else str(value).strip().lower()
    if text not in REFERENCE_MODES:
        raise ValueError(
            f"reference_mode must be one of {', '.join(REFERENCE_MODES)}, "
            f"got {value!r}")
    return text


def _resolve_reference_budget(value: Any) -> int:
    """Normalize the reference budget in bytes (``"4GiB"``, ``4 << 30``, ``4e9``)."""
    if value is None:
        return int(REFERENCE_BUDGET_BYTES)
    if isinstance(value, bool):
        raise ValueError(f"reference_budget must be a byte count, got {value!r}")
    if isinstance(value, (int, float)):
        number = int(value)
    elif isinstance(value, str):
        text = value.strip().lower().replace("_", "").replace(" ", "")
        if not text:
            raise ValueError("reference_budget must not be empty")
        suffix = ""
        while text and not (text[-1].isdigit() or text[-1] == "."):
            suffix = text[-1] + suffix
            text = text[:-1]
        try:
            number = int(float(text) * _BUDGET_UNITS[suffix])
        except (KeyError, ValueError) as error:
            raise ValueError(
                f"reference_budget={value!r} is not a byte count "
                f"(accepted suffixes: {', '.join(sorted(set(_BUDGET_UNITS)))})"
            ) from error
    else:
        raise ValueError(f"reference_budget must be a byte count, got {value!r}")
    if number <= 0:
        raise ValueError(f"reference_budget must be positive, got {value!r}")
    return number


def _tiled_extent(shape: Sequence[int], element: int, budget: int) -> tuple[int, int]:
    """Largest even ``t`` slab (and its count) whose workspace fits ``budget``.

    The extent must divide the global ``t`` extent so every slab is the same
    size.  ``(0, 0)`` means not even the smallest slab fits.
    """
    per_site = REFERENCE_TILED_SITE_ELEMENTS * int(element)
    per_t = per_site * int(shape[0]) * int(shape[1]) * int(shape[2])
    total = int(shape[3])
    if per_t * total <= int(budget):
        return total, 1
    best = 0
    for extent in range(2, total, 2):
        if total % extent:
            continue
        # One compact halo index on each side keeps the expansion parity exact
        # (two extra sites per side), so the slab workspace is extent + 4.
        if per_t * (extent + 4) <= int(budget):
            best = extent
    if best <= 0:
        return 0, 0
    return best, total // best


def reference_plan(lattice: Sequence[int], dtype: str = DEFAULT_DTYPE, *,
                   lattice_grid: Sequence[int] = (1, 1, 1, 1),
                   size: int | None = None,
                   local_lattice: Sequence[int] | None = None,
                   budget: Any = None, mode: Any = "auto") -> dict[str, Any]:
    """Declare which independent reference fits the budget, and its peak.

    The function is pure arithmetic (no torch/numpy), so a report and a test can
    both ask "what would the residual cost here?" without running anything:

    * ``distributed_peak_bytes`` - one rank's workspace for the rank-local
      ``D x - b`` with the MPI halo and an Allreduce of ``||r||^2``/``||b||^2``;
    * ``tiled_peak_bytes`` - rank 0's workspace for one ``t`` slab of the
      whole-lattice residual (``tiled_slab_extent_t`` sites, ``tiled_slabs``
      slabs, halo included);
    * ``gather_peak_bytes`` - the historical rank-0 gather *before* the operator
      runs (the 7.6 GiB class of allocation this module must not repeat);
    * ``whole_lattice_peak_bytes`` - the whole historical reference (gather,
      parity expansion, Clover term, operator temporaries);
    * ``global_clover_bytes`` - the single Clover tensor of the whole lattice
      (about 11 GiB for L48 c64), which the tiled path never allocates.

    ``mode`` is ``auto`` (distributed for rank-local blocks, tiled for
    whole-lattice fields), ``distributed``, ``tiled`` or ``gather``.  When the
    chosen path does not fit, ``mode`` becomes ``"unavailable"`` and ``reason``
    carries the estimate and the budget.
    """
    shape = policy.validate_lattice(lattice)
    grid = _int4((1, 1, 1, 1) if lattice_grid is None else lattice_grid,
                 "lattice_grid")
    if any(value <= 0 for value in grid):
        raise ValueError(f"lattice_grid extents must be positive, got {grid}")
    for axis in range(_SPACE):
        if shape[axis] % grid[axis]:
            raise ValueError(
                f"lattice_grid {grid} does not tile lattice {shape}: extent "
                f"{shape[axis]} on axis {axis} is not divisible by {grid[axis]}")
    world = math.prod(grid) if size is None else int(size)
    derived = tuple(shape[axis] // grid[axis] for axis in range(_SPACE))
    local = derived if local_lattice is None else _int4(local_lattice,
                                                        "local_lattice")
    name = _normalize_dtype(dtype)
    element = _element_bytes(name)
    limited = (REFERENCE_BUDGET_BYTES if budget is None
               else _resolve_reference_budget(budget))
    requested = _resolve_reference_mode(mode)
    volume = math.prod(shape)
    distributed = REFERENCE_DISTRIBUTED_SITE_ELEMENTS * math.prod(local) * element
    gathered = REFERENCE_GATHER_SITE_ELEMENTS * volume * element
    whole = REFERENCE_WHOLE_SITE_ELEMENTS * volume * element
    clover = REFERENCE_CLOVER_SITE_ELEMENTS * volume * element
    extent, slabs = _tiled_extent(shape, element, limited)
    per_site = REFERENCE_TILED_SITE_ELEMENTS * element
    tiled = (per_site * shape[0] * shape[1] * shape[2]
             * (shape[3] if slabs <= 1 else extent + 4)) if extent else 0
    partitioned = math.prod(grid) > 1 and tuple(local) != tuple(shape)
    if requested == "auto":
        chosen = "distributed" if partitioned else "tiled"
    elif requested == "distributed":
        if not partitioned:
            raise ValueError(
                "reference_mode='distributed' needs rank-local reference "
                f"blocks, but the fields cover the whole {tuple(shape)} "
                f"lattice while lattice_grid={grid} declares "
                f"{math.prod(grid)} rank(s); pass the rank-local block or use "
                "reference_mode='auto'")
        chosen = "distributed"
    else:
        chosen = requested
    estimate = {"distributed": distributed, "tiled": tiled,
                "gather": whole}[chosen]
    fits = bool(estimate) and int(estimate) <= limited
    reason = ""
    if not fits:
        if chosen == "tiled" and not extent:
            smallest = (per_site * shape[0] * shape[1] * shape[2] * 6)
            reason = (
                f"not even one even t slab of the {tuple(shape)} lattice fits "
                f"the {limited / 2 ** 20:.0f} MiB reference budget "
                f"({smallest / 2 ** 30:.2f} GiB for a six-site slab); raise "
                "reference_budget")
        else:
            scope = ("per rank" if chosen == "distributed" else "per t slab")
            reason = (
                f"the {chosen} independent reference needs about "
                f"{int(estimate) / 2 ** 30:.2f} GiB {scope}, above the declared "
                f"{limited / 2 ** 30:.2f} GiB budget; the whole-lattice gather "
                f"would need {whole / 2 ** 30:.2f} GiB and its Clover term "
                f"{clover / 2 ** 30:.2f} GiB. Raise reference_budget only with "
                "a matching memory plan")
        chosen_mode = "unavailable"
    else:
        chosen_mode = chosen
    return {
        "mode": chosen_mode,
        "requested_mode": requested,
        "partitioned": bool(partitioned),
        "reason": reason,
        "dtype": name,
        "bytes_per_element": int(element),
        "lattice": [int(value) for value in shape],
        "lattice_grid": [int(value) for value in grid],
        "local_lattice": [int(value) for value in local],
        "mpi_size": int(world),
        "volume": int(volume),
        "budget_bytes": int(limited),
        "peak_bytes": int(estimate) if fits else 0,
        "distributed_peak_bytes": int(distributed),
        "distributed_fits": bool(distributed <= limited),
        "tiled_peak_bytes": int(tiled),
        "tiled_slab_extent_t": int(extent),
        "tiled_slabs": int(slabs),
        "tiled_fits": bool(tiled and tiled <= limited),
        "gather_peak_bytes": int(gathered),
        "gather_fits": bool(whole <= limited),
        "whole_lattice_peak_bytes": int(whole),
        "global_clover_bytes": int(clover),
    }


def operand_plan(lattice: Sequence[int], lattice_grid: Any = None, *,
                 dtype: str = DEFAULT_DTYPE, size: int | None = None
                 ) -> dict[str, Any]:
    """Declared ABI operand footprint of **one rank**, in bytes.

    Pure arithmetic (no torch/numpy), so a size budget can be checked before a
    single tensor is allocated.  The shapes are the parity-compact ABI ones of
    :func:`_gauge_shape`/:func:`_fermion_shape`/:func:`_clover_shape` at the
    *rank-local* lattice, and ``global_bytes`` is what the same roles would cost
    if the ABI were handed the global lattice - the difference is exactly the
    ``prod(grid)``-fold overshoot the local-slab fix removes.

    ``peak_bytes`` counts what :func:`_solve_case` really keeps alive at once
    with a Clover solve: the compact gauge, the source and the solution plus the
    two Clover terms *and* their two inverses (``applyCloversQcu`` writes a term
    and its inverse per parity).
    """
    geometry = local_geometry(lattice, lattice_grid, size=size)
    element = _element_bytes(dtype)
    local = geometry.local
    shapes = {
        "gauge": _gauge_shape(local),
        "source": _fermion_shape(local),
        "solution": _fermion_shape(local),
        "clover": _clover_shape(local),
    }
    global_shapes = {
        "gauge": _gauge_shape(geometry.lattice),
        "source": _fermion_shape(geometry.lattice),
        "solution": _fermion_shape(geometry.lattice),
        "clover": _clover_shape(geometry.lattice),
    }
    per_role = {role: math.prod(shape) * element
                for role, shape in shapes.items()}
    global_per_role = {role: math.prod(shape) * element
                       for role, shape in global_shapes.items()}
    persistent = (per_role["gauge"] + per_role["source"]
                  + per_role["solution"])
    peak = persistent + 4 * per_role["clover"]
    global_peak = (global_per_role["gauge"] + global_per_role["source"]
                   + global_per_role["solution"]
                   + 4 * global_per_role["clover"])
    return {
        "lattice": list(geometry.lattice),
        "local_lattice": list(local),
        "lattice_grid": list(geometry.grid),
        "dtype": _normalize_dtype(dtype),
        "bytes_per_element": int(element),
        "shapes": {role: list(shape) for role, shape in shapes.items()},
        "bytes": per_role,
        "persistent_bytes": int(persistent),
        "clover_terms": 4,
        "peak_bytes": int(peak),
        "budget_bytes": int(REFERENCE_BUDGET_BYTES),
        "fits_budget": bool(peak <= REFERENCE_BUDGET_BYTES),
        "global_shapes": {role: list(shape)
                          for role, shape in global_shapes.items()},
        "global_bytes": global_per_role,
        "global_peak_bytes": int(global_peak),
        "global_fits_budget": bool(global_peak <= REFERENCE_BUDGET_BYTES),
        "overshoot": (float(global_peak) / float(peak)) if peak else math.inf,
    }


def _reference_plan(torch: Any, *, solution: Any, source: Any, gauge: Any,
                    lattice: Sequence[int], grid: Sequence[int], size: int,
                    budget: Any = None, mode: Any = "auto") -> dict[str, Any]:
    """Validate the field geometry and return the reference plan (no alloc).

    Split out of :func:`_true_residual` so a runner can ask "can the
    independent residual run at this size?" **before** the native call: the
    answer only needs the shapes, never a tensor.
    """
    shape = _int4(lattice, "lattice")
    requested = _int4(grid, "lattice_grid")
    world = int(size)
    observed = _field_lattice(gauge)
    for name, value in (("solution", solution), ("source", source)):
        if _field_lattice(value) != observed:
            raise ValueError(
                f"the reference {name} covers {_field_lattice(value)} but the "
                f"gauge covers {observed}; the residual needs one geometry")
    if tuple(observed) == tuple(shape):
        local = shape
    else:
        expected = tuple(shape[axis] // requested[axis] for axis in range(_SPACE))
        if tuple(observed) != expected:
            raise ValueError(
                f"the reference fields cover {tuple(observed)}, which is "
                f"neither the {tuple(shape)} whole lattice nor its "
                f"{expected} rank-local block under lattice_grid {requested}")
        local = expected
    return reference_plan(shape, _dtype_name(torch, gauge),
                          lattice_grid=requested, size=world,
                          local_lattice=local, budget=budget, mode=mode)


def _preflight_reference(*, dtype: str, lattice: Sequence[int],
                         local: Sequence[int], grid: Sequence[int],
                         size: int, budget: Any = None,
                         mode: Any = "auto") -> dict[str, Any]:
    """Decide the independent reference **before** the native call.

    A reference that does not fit the declared ``reference_budget`` is an
    explicit ``SKIP(reference_workspace)`` carrying the estimate, taken before
    the solve - it must never surface as an ``ERROR`` after a successful solve.
    For the distributed path the MPI reduction is probed here as well, so a
    missing or unusable ``mpi4py`` is the same kind of early skip instead of a
    failure in the middle of the verification.

    Only shapes are needed (no tensor): the rank-local block is what the ABI
    owns, so ``local`` is the geometry a partitioned run plans for, while a
    whole-lattice field plans with the global extents.
    """
    name = _normalize_dtype(dtype)
    shape = _int4(lattice, "lattice")
    rank_local = _int4(local, "local_lattice")
    requested = _int4(grid, "lattice_grid")
    plan = reference_plan(shape, name, lattice_grid=requested, size=int(size),
                          local_lattice=rank_local, budget=budget, mode=mode)
    if plan["mode"] == "unavailable":
        _skip(f"the independent residual cannot run at this size: "
              f"{plan['reason']}", token="reference_workspace")
    if plan["mode"] == "distributed":
        _comm(size=int(plan["mpi_size"]))
    return plan


def _residual_unavailable(reason: Any, plan: Mapping[str, Any] | None,
                          started: float) -> dict[str, Any]:
    """Observation block of a reference that could not be computed.

    The native call already happened (this is only used *after* a solve), so
    the failure is reported, not raised: ``relative_residual`` stays ``None``,
    ``residual_verified`` is ``False`` and ``residual_unavailable`` carries the
    reason the ledger has to show.
    """
    observation = _residual_observation(
        numerator=0.0, denominator=0.0, solution_norm=0.0,
        plan=(plan if plan is not None else {
            "mode": "unavailable", "partitioned": False, "budget_bytes": 0,
            "peak_bytes": 0, "tiled_slabs": 0}),
        started=started, source="unavailable")
    observation.update(
        relative_residual=None,
        residual_norm=None,
        source_norm=None,
        solution_norm=None,
        residual_verified=False,
        residual_unavailable=str(reason),
    )
    return observation


def _field_lattice(value: Any) -> tuple[int, int, int, int]:
    """Whole-lattice extents of a parity-compact ``[2, ...X, Y, Z, T/2]`` field."""
    shape = tuple(int(extent) for extent in value.shape)
    if len(shape) < _SPACE + 1 or shape[0] != 2:
        raise ValueError(
            "a parity-compact reference field must have shape "
            f"[2, ..., X, Y, Z, T/2], got {shape}")
    if any(extent <= 0 for extent in shape):
        raise ValueError(f"a reference field must have positive extents, got {shape}")
    extents = shape[-_SPACE:]
    return (extents[0], extents[1], extents[2], extents[3] * 2)


def _dtype_name(torch: Any, value: Any) -> str:
    """``c64``/``c128`` of a torch complex tensor."""
    return "c128" if value.dtype == torch.complex128 else "c64"


def _mpi(size: int) -> tuple[Any, Any]:
    """``(mpi4py.MPI, MPI.COMM_WORLD)`` for a distributed reduction."""
    comm = _comm(size=size)
    if comm is None:
        _skip("the distributed reference needs at least two MPI ranks",
              token="mpi_single_rank")
    from mpi4py import MPI

    return MPI, comm


def _comm(*, size: int) -> Any:
    """Return ``MPI.COMM_WORLD``, verified against the probed world size."""
    if int(size) <= 1:
        return None
    try:
        from mpi4py import MPI
    except ImportError as error:
        _skip(f"the distributed reference exchanges rank-local halo faces and "
              f"reduces its norms with mpi4py: {type(error).__name__}: "
              f"{error}; run this case on one rank",
              token="mpi_single_rank")
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
                 size: int, purpose: str,
                 budget: int = REFERENCE_BUDGET_BYTES) -> dict[str, Any] | None:
    """Gather and expand the solution/source/gauge needed by a reference run."""
    gathered: dict[str, Any] = {}
    # Every rank must enter *all* collectives in the same order: a non-root
    # rank that returned early would leave the others blocked in ``gather``.
    for name, value in (("solution", solution), ("source", source),
                        ("gauge", gauge)):
        gathered[name] = _global_compact(
            torch, value, lattice=lattice, grid=grid, size=size,
            purpose=f"{purpose} {name}", budget=int(budget))
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


def _sum_squares(torch: Any, value: Any) -> float:
    """``sum |value|^2`` accumulated in float64 without a float64 copy."""
    real = value.real
    imag = value.imag
    return (float(torch.sum(real * real, dtype=torch.float64))
            + float(torch.sum(imag * imag, dtype=torch.float64)))


def _reduced_norms(torch: Any, *, residual: Any, source: Any, solution: Any,
                   comm: Any = None) -> tuple[float, float, float]:
    """``(||r||, ||b||, ||x||)``, Allreduced over ``comm`` when it is set.

    All ranks compute the same three numbers, so the runner reports one value
    even though no rank ever holds the global residual.
    """
    residual_sum = _sum_squares(torch, residual)
    source_sum = _sum_squares(torch, source)
    solution_sum = _sum_squares(torch, solution)
    if comm is not None:
        MPI, _ = _mpi(size=int(comm.Get_size()))
        residual_sum = float(comm.allreduce(residual_sum, op=MPI.SUM))
        source_sum = float(comm.allreduce(source_sum, op=MPI.SUM))
        solution_sum = float(comm.allreduce(solution_sum, op=MPI.SUM))
    return (math.sqrt(max(residual_sum, 0.0)),
            math.sqrt(max(source_sum, 0.0)),
            math.sqrt(max(solution_sum, 0.0)))


def _clover_term(torch: Any, roll: Callable[[Any, int, int], Any], *,
                 gauge: Any, kappa: float, csw: float) -> Any:
    """Clover term ``-0.125 * kappa * csw * sum_{mu<nu} sigma F`` (or ``None``).

    The definition is :func:`pyqcu.torch.dslash.make_clover`'s (the production
    reference this module is checked against), with the periodic shift injected
    as ``roll(value, shifts, dims)``: ``torch.roll`` for a whole-lattice slab
    and the MPI halo roll for a rank-local block.  ``csw == 0`` returns ``None``
    so the operator is the plain Wilson operator, exactly like the reference.
    """
    from pyqcu import lattice as lattice_module

    if float(csw) == 0.0:
        return None
    dtype, device = gauge.dtype, gauge.device
    keys = ("xy", "xz", "xt", "yz", "yt", "zt")
    sigma = {key: lattice_module.gamma_gamma[
        lattice_module.ward_wards[key]["ward"]].to(device).type(dtype)
        for key in keys}
    dagger = gauge.permute(1, 0, 2, 3, 4, 5, 6).conj()
    clover = torch.zeros((4, 3, 4, 3, *gauge.shape[-_SPACE:]),
                         dtype=dtype, device=device)
    factor = -0.125 * float(kappa) * float(csw)
    for key in keys:
        mu = lattice_module.ward_wards[key]["mu"]
        nu = lattice_module.ward_wards[key]["nu"]
        U_mu = gauge[..., mu, :, :, :, :]
        U_nu = gauge[..., nu, :, :, :, :]
        U_dag_mu = dagger[..., mu, :, :, :, :]
        U_dag_nu = dagger[..., nu, :, :, :, :]
        field = torch.zeros(U_mu.shape, dtype=dtype, device=device)
        first = torch.einsum(
            "abxyzt,bcxyzt->acxyzt", U_mu, roll(U_nu, -1, mu))
        second = torch.einsum(
            "abxyzt,bcxyzt->acxyzt", first, roll(U_dag_mu, -1, nu))
        field = field + torch.einsum(
            "abxyzt,bcxyzt->acxyzt", second, U_dag_nu)
        first = torch.einsum(
            "abxyzt,bcxyzt->acxyzt", U_nu,
            roll(roll(U_dag_mu, 1, mu), -1, nu))
        second = torch.einsum(
            "abxyzt,bcxyzt->acxyzt", first, roll(U_dag_nu, 1, mu))
        field = field + torch.einsum(
            "abxyzt,bcxyzt->acxyzt", second, roll(U_mu, 1, mu))
        first = torch.einsum(
            "abxyzt,bcxyzt->acxyzt", roll(U_dag_mu, 1, mu),
            roll(roll(U_dag_nu, 1, mu), 1, nu))
        second = torch.einsum(
            "abxyzt,bcxyzt->acxyzt", first,
            roll(roll(U_mu, 1, mu), 1, nu))
        field = field + torch.einsum(
            "abxyzt,bcxyzt->acxyzt", second, roll(U_nu, 1, nu))
        first = torch.einsum(
            "abxyzt,bcxyzt->acxyzt", roll(U_dag_nu, 1, nu),
            roll(U_mu, 1, nu))
        second = torch.einsum(
            "abxyzt,bcxyzt->acxyzt", first,
            roll(roll(U_nu, -1, mu), 1, nu))
        field = field + torch.einsum(
            "abxyzt,bcxyzt->acxyzt", second, U_dag_mu)
        field = field - field.permute(1, 0, 2, 3, 4, 5).conj()
        clover = clover + factor * torch.einsum(
            "Ss,Ccxyzt->SCscxyzt", sigma[key], field)
        del first, second, field
    return clover


def _wilson_clover_apply(torch: Any, roll: Callable[[Any, int, int], Any], *,
                         vector: Any, gauge: Any, clover: Any,
                         kappa: float) -> Any:
    """``D x = (I - kappa * hopping) x + T x`` on one lattice block.

    This is exactly ``dslash.give_wilson(x, U, kappa, with_I=True) +
    dslash.give_clover(x, clover_term)`` - the reference composition of
    :func:`_operator_reference` - with the four periodic shifts injected as
    ``roll(value, shifts, dims)`` so a rank-local block can exchange MPI faces
    instead of wrapping inside the block.
    """
    from pyqcu import lattice as lattice_module
    from pyqcu.torch import dslash

    dtype, device = vector.dtype, vector.device
    identity = lattice_module.I.to(device).type(dtype)
    dagger = gauge.permute(1, 0, 2, 3, 4, 5, 6).conj()
    dest = vector.clone()
    factor = float(kappa)
    for key in lattice_module.ward_keys:
        ward = lattice_module.wards[key]
        gamma = lattice_module.gamma[ward].to(device).type(dtype)
        U_mu = gauge[..., ward, :, :, :, :]
        U_dag_mu = dagger[..., ward, :, :, :, :]
        forward = torch.einsum(
            "Ccxyzt,scxyzt->sCxyzt", U_mu, roll(vector, -1, ward))
        backward = torch.einsum(
            "Ccxyzt,scxyzt->sCxyzt", roll(U_dag_mu, 1, ward),
            roll(vector, 1, ward))
        hopping = (torch.einsum("Ss,sCxyzt->SCxyzt",
                                identity - gamma, forward)
                   + torch.einsum("Ss,sCxyzt->SCxyzt",
                                  identity + gamma, backward))
        dest = dest - factor * hopping
    if clover is not None:
        dest = dest + dslash.give_clover(vector, clover)
    return dest


def _residual_observation(*, numerator: float, denominator: float,
                          solution_norm: float, plan: Mapping[str, Any],
                          started: float, source: str,
                          notes: Sequence[str] = ()) -> dict[str, Any]:
    """The observation block a runner stores next to ``relative_residual``."""
    value = (math.inf if denominator == 0.0
             else float(numerator / denominator))
    observation = {
        "relative_residual": value,
        "source_norm": float(denominator),
        "residual_norm": float(numerator),
        "solution_norm": float(solution_norm),
        "reference_seconds": time.perf_counter() - started,
        "residual_source": source,
        "residual_mode": plan["mode"],
        "residual_partitioned": bool(plan["partitioned"]),
        "residual_budget_bytes": int(plan["budget_bytes"]),
        "residual_peak_bytes": int(plan["peak_bytes"]),
        "residual_slabs": int(plan["tiled_slabs"]),
        "residual_plan": dict(plan),
    }
    if notes:
        observation["residual_notes"] = list(notes)
    return observation


def _residual_fields(residual: Mapping[str, Any]) -> dict[str, Any]:
    """The observation keys every runner copies out of a residual block.

    One place decides what a reference contributes, so an unavailable reference
    (:func:`_residual_unavailable`) reaches the ledger with the same keys as a
    computed one - ``relative_residual=None``, ``residual_verified=False`` and
    the reason - instead of a KeyError in the middle of a successful solve.
    """
    value = residual["relative_residual"]
    fields: dict[str, Any] = {
        "relative_residual": value,
        "relative_error": value,
        "true_residual": value,
        "source_norm": residual["source_norm"],
        "residual_norm": residual["residual_norm"],
        "solution_norm": residual["solution_norm"],
        "residual_source": residual["residual_source"],
        "residual_mode": residual["residual_mode"],
        "residual_partitioned": residual["residual_partitioned"],
        "residual_budget_bytes": residual["residual_budget_bytes"],
        "residual_peak_bytes": residual["residual_peak_bytes"],
        "residual_slabs": residual["residual_slabs"],
        "residual_plan": residual["residual_plan"],
        "residual_verified": bool(value is not None),
        "reference_seconds": residual["reference_seconds"],
    }
    if residual.get("residual_unavailable"):
        fields["residual_unavailable"] = residual["residual_unavailable"]
    return fields


def _residual_distributed(torch: Any, tools: Any, *, solution: Any, source: Any,
                          gauge: Any, plan: Mapping[str, Any], kappa: float,
                          csw: float, started: float) -> dict[str, Any]:
    """Rank-local ``D x - b`` with the production layout and the MPI halo.

    Each rank holds its parity-compact block, expands it to the local full
    lattice, applies the whole Wilson-Clover operator with the periodic shift
    provided by :mod:`pyqcu.torch.tools._mpi_roll`, and contributes its local
    ``||r||^2``/``||b||^2`` to an Allreduce.  No rank ever holds the global
    field, so the rank-0 gather (and its multi-GiB workspace) is gone.
    """
    from pyqcu.torch.tools import _mpi_roll

    comm = _comm(size=int(plan["mpi_size"]))
    if comm is None:
        raise VerificationUnavailable(
            "the rank-local reference needs MPI_COMM_WORLD, but this run has "
            "one rank")

    def roll(value: Any, shifts: int, dims: int) -> Any:
        return _mpi_roll.roll(value, shifts=shifts, dims=dims)

    fields = {
        "solution": tools.poooxyzt2oooxyzt(solution.contiguous()),
        "source": tools.poooxyzt2oooxyzt(source.contiguous()),
        "gauge": tools.poooxyzt2oooxyzt(gauge.contiguous()),
    }
    with _mpi_roll.distributed_roll(
            comm=comm, process_grid=plan["lattice_grid"],
            local_extents=plan["local_lattice"]):
        clover = _clover_term(torch, roll, gauge=fields["gauge"], kappa=kappa,
                              csw=csw)
        applied = _wilson_clover_apply(
            torch, roll, vector=fields["solution"], gauge=fields["gauge"],
            clover=clover, kappa=kappa)
    del clover
    residual = applied - fields["source"]
    numerator, denominator, solution_norm = _reduced_norms(
        torch, residual=residual, source=fields["source"],
        solution=fields["solution"], comm=comm)
    return _residual_observation(
        numerator=numerator, denominator=denominator,
        solution_norm=solution_norm, plan=plan, started=started,
        source=("independent torch full Wilson-Clover reconstruction of the "
                "rank-local blocks (MPI halo from pyqcu.torch.tools._mpi_roll, "
                "allreduce of ||Dx-b||^2 and ||b||^2)"))


def _residual_tiled(torch: Any, tools: Any, *, solution: Any, source: Any,
                    gauge: Any, plan: Mapping[str, Any], kappa: float,
                    csw: float, started: float) -> dict[str, Any]:
    """Whole-lattice residual computed one bounded ``t`` slab at a time.

    Every slab carries one compact halo index on each side (two sites), which
    keeps the parity expansion exact (the slab's ``t`` origin stays even) and
    supplies the ``t`` neighbours the operator and the Clover term need; the
    operator itself is the same injected-roll Wilson-Clover apply.  The norms
    are accumulated in float64, so the peak is one slab - never the 11 GiB
    Clover term of the whole lattice.
    """
    def roll(value: Any, shifts: int, dims: int) -> Any:
        return torch.roll(value, shifts=shifts, dims=dims)

    shape = tuple(int(value) for value in plan["lattice"])
    extent = int(plan["tiled_slab_extent_t"])
    slabs = int(plan["tiled_slabs"])
    count = extent // 2
    compact_t = shape[3] // 2
    residual_sum = source_sum = solution_sum = 0.0
    for index in range(max(slabs, 1)):
        if slabs <= 1:
            gutter: Any = slice(None)
            interior = slice(None)
        else:
            indices = [(index * count - 1 + offset) % compact_t
                       for offset in range(count + 2)]
            gutter = torch.as_tensor(indices, dtype=torch.long,
                                     device=solution.device)
            interior = slice(2, 2 + extent)
        gauge_slab = tools.poooxyzt2oooxyzt(gauge[..., gutter].contiguous())
        vector = tools.poooxyzt2oooxyzt(solution[..., gutter].contiguous())
        right = tools.poooxyzt2oooxyzt(
            source[..., index * count:(index + 1) * count].contiguous())
        clover = _clover_term(torch, roll, gauge=gauge_slab, kappa=kappa,
                              csw=csw)
        applied = _wilson_clover_apply(
            torch, roll, vector=vector, gauge=gauge_slab, clover=clover,
            kappa=kappa)
        residual = applied[..., interior] - right
        residual_sum += _sum_squares(torch, residual)
        source_sum += _sum_squares(torch, right)
        solution_sum += _sum_squares(torch, vector[..., interior])
        del gauge_slab, vector, right, clover, applied, residual
    return _residual_observation(
        numerator=math.sqrt(max(residual_sum, 0.0)),
        denominator=math.sqrt(max(source_sum, 0.0)),
        solution_norm=math.sqrt(max(solution_sum, 0.0)),
        plan=plan, started=started,
        source=(f"independent torch full Wilson-Clover reconstruction of the "
                f"{tuple(shape)} lattice in {max(slabs, 1)} t slab(s) of "
                f"{extent} site(s) (no rank-0 gather of the global field)"))


def _residual_gathered(torch: Any, tools: Any, *, solution: Any, source: Any,
                       gauge: Any, plan: Mapping[str, Any], kappa: float,
                       csw: float, purpose: str,
                       started: float) -> dict[str, Any]:
    """Historical rank-0 gather; kept for small lattices and as the test oracle."""
    from pyqcu.torch import dslash

    comm = _comm(size=int(plan["mpi_size"]))
    fields = _full_fields(
        torch, tools, solution=solution, source=source, gauge=gauge,
        lattice=plan["lattice"], grid=plan["lattice_grid"],
        size=int(plan["mpi_size"]), purpose=purpose,
        budget=int(plan["budget_bytes"]))
    value: float | None = None
    numerator = denominator = solution_norm = None
    if fields is not None:
        operator = _operator_reference(torch, dslash, gauge=fields["gauge"],
                                       kappa=kappa, csw=csw)
        residual = operator(fields["solution"]) - fields["source"]
        denominator = float(torch.linalg.vector_norm(fields["source"]))
        numerator = float(torch.linalg.vector_norm(residual))
        solution_norm = float(torch.linalg.vector_norm(fields["solution"]))
        value = (math.inf if denominator == 0.0
                 else float(numerator / denominator))
    if comm is not None:
        value = comm.bcast(value, root=0)
        numerator = comm.bcast(numerator, root=0)
        denominator = comm.bcast(denominator, root=0)
        solution_norm = comm.bcast(solution_norm, root=0)
    observation = _residual_observation(
        numerator=float(numerator or 0.0), denominator=float(denominator or 0.0),
        solution_norm=float(solution_norm or 0.0), plan=plan, started=started,
        source=("independent torch full Wilson-Clover reconstruction of the "
                "gathered global field (MPI all-blocks on rank 0)"),
        notes=["residual_mode='gather' keeps the historical rank-0 path; use "
               "'auto' for the bounded reference"])
    observation["relative_residual"] = value
    observation["residual_norm"] = numerator
    observation["source_norm"] = denominator
    observation["solution_norm"] = solution_norm
    return observation


def _true_residual(torch: Any, *, solution: Any, source: Any, gauge: Any,
                   lattice: Sequence[int], grid: Sequence[int], size: int,
                   kappa: float, csw: float, purpose: str,
                   budget: Any = None,
                   mode: Any = "auto",
                   plan: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Independent ``||D x - b|| / ||b||`` inside a declared memory budget.

    The physical definition never changes: the *full* Wilson-Clover operator
    ``D = I + csw*T - kappa*hopping`` applied to the raw solution on the whole
    (periodic) lattice, never the solver's recurrence.  Only the decomposition
    does, and :func:`reference_plan` decides it before a single tensor is
    allocated:

    * rank-local blocks (the production layout, one block per rank) use the
      distributed path - ``_mpi_roll`` halo, Allreduce of ``||r||^2``/``||b||^2``;
      this is the bounded path and the one a multi-rank solve takes;
    * whole-lattice fields (one rank owning everything) use the tiled path on
      every rank, with no collective at all;
    * ``mode="gather"`` keeps the historical rank-0 gather for small lattices.

    ``plan`` is the already-decided :func:`reference_plan`; a runner passes it
    in so the decision was taken *before* the native call
    (:func:`_preflight_reference`).  Without one the plan is computed here, and
    ``VerificationUnavailable`` is raised when the chosen path really does not
    fit the budget (or when no path exists), carrying the estimate, the budget
    and the alternative costs.
    """
    import pyqcu.tools as tools

    started = time.perf_counter()
    if plan is None:
        plan = _reference_plan(
            torch, solution=solution, source=source, gauge=gauge,
            lattice=lattice, grid=grid, size=size, budget=budget, mode=mode)
    if plan["mode"] == "unavailable":
        raise VerificationUnavailable(plan["reason"])
    if plan["mode"] == "distributed":
        return _residual_distributed(
            torch, tools, solution=solution, source=source, gauge=gauge,
            plan=plan, kappa=kappa, csw=csw, started=started)
    if plan["mode"] == "gather":
        return _residual_gathered(
            torch, tools, solution=solution, source=source, gauge=gauge,
            plan=plan, kappa=kappa, csw=csw, purpose=purpose, started=started)
    return _residual_tiled(
        torch, tools, solution=solution, source=source, gauge=gauge,
        plan=plan, kappa=kappa, csw=csw, started=started)


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
    #: Rank-local extents the ABI is told about, derived from ``lattice`` and
    #: ``lattice_grid`` (``None``/``1x1x1x1`` mean "the whole lattice").  A
    #: multi-rank case must declare a grid: every native operand has this shape,
    #: while ``lattice`` stays the physical lattice the independent residual is
    #: defined on.
    local_lattice: tuple[int, int, int, int] | None = None
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
    #: Whether the caller actually supplied ``dof``.  ``dof`` alone cannot say:
    #: it defaults to :data:`DEFAULT_DOF`, so "declared the default" and
    #: "declared nothing" are the same tuple, and the ranked-cache guard has to
    #: tell them apart to enforce the requested E (see the guard in
    #: ``_load_ranked_assets``).
    dof_declared: bool = False
    nv_iters: int = DEFAULT_NV_ITERS
    restart: int = 5
    coarse_max_iter: int = 200
    #: Fine iterations before the first coarse correction (params[kMgMuPre]);
    #: 0 keeps the solver default.  Accepted by the CLI and dropped for years.
    mg_nu_pre: int = 0
    coarse_tol_factor: float = 1.0e4
    deflate: bool = True
    verify: bool = True
    cache_policy: str = "auto"
    cache: Path | None = None
    cache_directory: Path | None = None
    #: Cache roots of the deeper coarse transitions, in hierarchy order: entry
    #: ``k`` publishes transition ``k + 2`` (``--coarse-cache-level`` number
    #: ``k + 1``, the first one being transition 2).  ``cache``/``cache_directory``
    #: keep naming transition 1 only, because that is the one transition this
    #: port can also *build*; a deeper transition has no builder and must come
    #: from one of these roots.
    coarse_cache_level: tuple[Path | None, ...] = ()
    #: Independent-reference policy.  ``reference_mode`` is ``auto`` (the
    #: distributed path for rank-local blocks, the tiled path for whole-lattice
    #: fields), ``distributed``, ``tiled`` or ``gather``; ``reference_budget``
    #: is the declared peak workspace of the reference in bytes.  A case whose
    #: reference cannot fit raises :class:`VerificationUnavailable` carrying the
    #: estimate instead of allocating it silently.
    reference_mode: str = "auto"
    reference_budget: int = REFERENCE_BUDGET_BYTES
    #: Explicit HDF5 gauge/source artifacts (:mod:`pyqcu.test.io`).  When one is
    #: set the field is *read from it* instead of generated, and the origin,
    #: path, layout and SHA256 are reported - never a silent fallback to the
    #: ``gauge``/``source`` generator.
    gauge_file: Path | None = None
    source_file: Path | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)

    @property
    def operand_lattice(self) -> tuple[int, int, int, int]:
        """Extents of every ABI operand: the rank-local slab, not the global.

        ``prod(operand_lattice) * prod(lattice_grid) == prod(lattice)`` holds by
        construction (:func:`_declared_local`); the runners re-derive the same
        numbers from the live MPI grid (:func:`local_geometry`) and refuse to
        run if the two disagree.
        """
        return (tuple(self.lattice) if self.local_lattice is None
                else tuple(self.local_lattice))

    @property
    def compact(self) -> tuple[int, int, int, int]:
        """Parity-compact ``[X, Y, Z, T/2]`` extents of the ABI operands."""
        return _compact_shape(self.operand_lattice)

    @property
    def fermion_shape(self) -> tuple[int, ...]:
        """Two-parity compact fermion shape of the ABI."""
        return _fermion_shape(self.operand_lattice)

    @property
    def clover_shape(self) -> tuple[int, ...]:
        """Parity compact Clover-term shape of the ABI."""
        return _clover_shape(self.operand_lattice)

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe description of the request (reported by every runner)."""
        return {
            "kind": self.kind,
            "backend": self.backend,
            "lattice": list(self.lattice),
            "lattice_grid": (None if self.lattice_grid is None
                             else list(self.lattice_grid)),
            "local_lattice": list(self.operand_lattice),
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
            "coarse_cache_level": [None if path is None else str(path)
                                   for path in self.coarse_cache_level],
            "gauge_file": (None if self.gauge_file is None
                           else str(self.gauge_file)),
            "source_file": (None if self.source_file is None
                            else str(self.source_file)),
            "reference_mode": self.reference_mode,
            "reference_budget_bytes": int(self.reference_budget),
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


def _path_option(value: Any) -> Path | None:
    """Normalise an optional HDF5 input path (``None``/``''`` stay ``None``)."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text).expanduser()


def _declared_local(lattice: Sequence[int], lattice_grid: Any
                    ) -> tuple[int, int, int, int]:
    """Rank-local extents a *declared* grid implies (pure arithmetic).

    Option resolution must not ask MPI anything (``--list`` and ``case()`` run
    without a launcher), so the local extents come from the declared grid here;
    :func:`local_geometry` re-derives them from the live grid before the first
    native call and the two are required to agree.  A grid that does not tile
    the lattice is a usage error at both places - no launch could make it valid.
    """
    shape = tuple(int(value) for value in lattice)
    if lattice_grid is None:
        return shape  # type: ignore[return-value]
    requested = _int4(lattice_grid, "lattice_grid")
    if any(value <= 0 for value in requested):
        raise ValueError(f"lattice_grid extents must be positive, got {requested}")
    for axis in range(_SPACE):
        if shape[axis] % requested[axis]:
            raise ValueError(
                f"lattice_grid {requested} does not tile lattice {shape}: "
                f"extent {shape[axis]} on axis {axis} is not divisible by "
                f"{requested[axis]}")
    return tuple(shape[axis] // requested[axis]
                 for axis in range(_SPACE))  # type: ignore[return-value]


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
    local_lattice = _declared_local(lattice, lattice_grid)
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
    if mg_levels > MG_MAX_LEVELS:
        # This *is* a solver limit: the KUPL Legacy-MG throws for a level count
        # outside 1..5 (``num_levels_ < 1 || num_levels_ > 5`` in
        # cpp/kupl/qcu/src/lattice_clover_multigrid.cpp:195).
        raise ValueError(
            f"the KUPL Legacy-MG accepts 1..{MG_MAX_LEVELS} levels "
            f"(cpp/kupl/qcu/src/lattice_clover_multigrid.cpp:195), got "
            f"{mg_levels}")
    dof_value = _get("dof", DEFAULT_DOF)
    dof = tuple(_resolve_int(value, "dof", minimum=1) for value in dof_value)
    # ``_get`` folds an absent ``dof`` into DEFAULT_DOF, so record the supply
    # separately: the cache guard must reject a declared E that the artifact
    # cannot provide, and must NOT reject a case that never asked.
    dof_declared = (
        explicit.get("dof") is not None
        or (container is not None and getattr(container, "dof", None) is not None)
    )
    if len(dof) < 2 or dof[0] != 12:
        raise ValueError(
            f"dof must start with the 12-component fine field and list at "
            f"least two levels, got {dof}")
    if mg_levels > 2:
        # Derive the hierarchy now: every level of a Legacy-MG case is built in
        # one solver constructor (lattice_clover_multigrid.cpp:231-275), so an
        # unrepresentable depth is a usage error here rather than a half-wired
        # hierarchy later.  The derivation names the level and the extent.
        _mg_level_chain(lattice, mg_grid, mg_levels, dof)
    nv_iters = _resolve_int(_get("nv_iters", DEFAULT_NV_ITERS), "nv_iters",
                            minimum=1)
    restart = _resolve_int(_get("restart", 5), "restart")
    coarse_max_iter = _resolve_int(_get("coarse_max_iter", 200),
                                    "coarse_max_iter", minimum=1)
    mg_nu_pre = _resolve_int(_get("mg_nu_pre", 0), "mg_nu_pre", minimum=0)
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
    # Deeper transition roots, in order (``--coarse-cache-level``).  A position
    # is kept even when it is empty so a gap names the transition it leaves
    # out instead of shifting the next root down a level.
    level_values = _get("coarse_cache_level") or ()
    if isinstance(level_values, (str, bytes, Path)):
        level_values = (level_values,)
    coarse_cache_level = tuple(
        None if value in (None, "") else Path(str(value)).expanduser()
        for value in level_values)
    reference_mode = _resolve_reference_mode(_get("reference_mode"))
    reference_budget = _resolve_reference_budget(_get("reference_budget"))
    gauge_file = _path_option(_get("gauge_file"))
    source_file = _path_option(_get("source_file"))
    return NativeCase(
        kind=kind,
        backend=_resolve_backend(_get("backend")),
        lattice=lattice,
        lattice_grid=lattice_grid,
        local_lattice=local_lattice,
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
        dof_declared=dof_declared,
        nv_iters=nv_iters,
        restart=restart,
        coarse_max_iter=coarse_max_iter,
        mg_nu_pre=mg_nu_pre,
        coarse_tol_factor=coarse_tol_factor,
        deflate=bool(_get("deflate", True)),
        verify=bool(_get("verify", True)),
        cache_policy=cache_policy,
        cache=(None if cache_value is None else Path(str(cache_value)).expanduser()),
        cache_directory=(None if cache_directory is None
                         else Path(str(cache_directory)).expanduser()),
        coarse_cache_level=coarse_cache_level,
        reference_mode=reference_mode,
        reference_budget=reference_budget,
        gauge_file=gauge_file,
        source_file=source_file,
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
def _input_observations(gauge: Mapping[str, Any] | None,
                        source: Mapping[str, Any] | None) -> dict[str, Any]:
    """Origin/path/hash of the two inputs (``hdf5`` or the generator name).

    The row ledger has to answer "which bytes did this run consume?", so the
    origin is never inferred from a file name: ``origin="hdf5"`` always comes
    with the validated path, layout, dtype and the streamed SHA256, and a
    generated field names its generator instead (``random``/``unit``/
    ``ones``/``point``) with no hash to mistake for evidence.
    """
    observations: dict[str, Any] = {}
    for role, built in (("gauge", gauge), ("source", source)):
        if built is None:
            continue
        observations.update({
            f"{role}_origin": built.get("origin"),
            f"{role}_path": built.get("path"),
            f"{role}_sha256": built.get("sha256"),
            f"{role}_layout": built.get("layout"),
            f"{role}_file_dtype": built.get("dtype"),
            # ``bytes`` is the payload size; an HDF5 input also reports its
            # real ``st_size``, which is what a "file_bytes" key promises.
            f"{role}_file_bytes": built.get("file_bytes", built.get("bytes")),
            # Which reader produced the block: "mpio", "slab" or "full".
            # Without this a per-rank full read and a rank-slab read cannot be
            # told apart in the ledger (2026-11-23, 1.24 TB vs 9.66 GB).
            f"{role}_read_mode": built.get("read_mode"),
        })
    return observations


def _case_geometry(option: NativeCase, grid: Sequence[int],
                   geometry: LocalGeometry | None = None) -> LocalGeometry:
    """The rank geometry of ``option`` under the live ``grid``, cross-checked.

    The runners get the geometry from :func:`_run_native`; a direct call to a
    case function re-derives it from the option and the grid.  Either way the
    option's own ``operand_lattice`` must be the same slab, otherwise the
    ledger would describe one geometry while the native call used another.
    """
    shape = policy.validate_lattice(option.lattice)
    requested = _int4(grid, "lattice_grid")
    if geometry is None:
        geometry = local_geometry(shape, requested,
                                  size=math.prod(requested))
    geometry.check()
    if tuple(option.operand_lattice) != geometry.local:
        raise RuntimeError(
            f"the case declares the local lattice "
            f"{list(option.operand_lattice)} but the {list(geometry.grid)} grid "
            f"owns {list(geometry.local)}; refusing to run with two geometries")
    return geometry


def _common_observations(option: NativeCase, probe: BackendProbe,
                         grid: Sequence[int], environment: NativeEnvironment,
                         controls: _Controls, started: float, *,
                         gauge: Mapping[str, Any] | None = None,
                         source: Mapping[str, Any] | None = None
                         ) -> dict[str, Any]:
    """Observation block shared by every native runner.

    ``lattice`` is always the physical (global) lattice and ``local_lattice``
    the slab this rank owns; ``abi_geometry`` restates what the ABI was told
    (``params[_LAT_*]``/``params[_GRID_*]``) with the volume check, so a ledger
    row can tell "the operators ran on this rank's slab" from "the operators
    were handed the global lattice".
    """
    geometry = _case_geometry(option, grid)
    return {
        "backend": probe.backend,
        "device": probe.device,
        "library": probe.library,
        "native_kind": option.kind,
        "lattice": list(option.lattice),
        "local_lattice": list(geometry.local),
        "lattice_grid": list(grid),
        "global_volume": int(geometry.volume),
        "local_volume": int(geometry.local_volume),
        "abi_geometry": controls.geometry,
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
        **_input_observations(gauge, source),
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
    # The ABI/operand geometry follows the *live* grid; the declared grid the
    # options already turned into ``local_lattice`` must agree with it.
    geometry = local_geometry(option.lattice, grid, size=size)
    if tuple(option.operand_lattice) != geometry.local:
        raise RuntimeError(
            f"the case declares the local lattice "
            f"{list(option.operand_lattice)} but the {list(grid)} grid in this "
            f"job owns {list(geometry.local)}; refusing to run with two "
            "geometries")
    declared = _resolve_backend(option.backend)
    if option.kind == "dslash":
        _require_single_rank(
            size=size,
            reason="the native dslash comparison needs the whole lattice for "
                   "its full-lattice reference")
    elif option.kind == "mg":
        # A ranked cache (the production 16-rank/two-node format) supplies each
        # rank's coarse slab, so multi-rank MG is valid there.  Without one,
        # the serial local 33-stencil builder cannot represent coarse
        # couplings across a rank boundary; keep the old fail-closed guard.
        if _ranked_mg_cache_path(option) is None:
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
            return _dslash_case(option, probe, grid, environment, started,
                                geometry=geometry)
        if option.kind == "solve":
            return _solve_case(option, probe, grid, environment, started,
                               geometry=geometry)
        return _mg_case(option, probe, grid, environment, started,
                        geometry=geometry)


def _dslash_case(option: NativeCase, probe: BackendProbe,
                 grid: Sequence[int], environment: NativeEnvironment,
                 started: float, *,
                 geometry: LocalGeometry | None = None) -> dict[str, Any]:
    """Apply the native Wilson/Clover dslash once per parity and compare."""
    torch = _import_torch("kupl_library")
    from pyqcu.torch import dslash as dslash_module

    geometry = _case_geometry(option, grid, geometry)
    local = geometry.local
    controls = _Controls(
        probe, lattice=local, grid=grid, dtype=option.dtype,
        mass=option.mass, csw=option.csw, atol=option.atol,
        max_iter=option.max_iter, plan=controls_plan(probe, "dslash"),
        parity=0, seed=option.seed, global_lattice=geometry.lattice)
    gauge = _build_gauge(option, torch, controls.torch_dtype, geometry,
                         probe.rank)
    source = _build_source(option, torch, controls.torch_dtype, geometry,
                           probe.rank)
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
                                        controls, started, gauge=gauge,
                                        source=source)
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
            max_iter=option.max_iter, grid=grid, lattice=local),
    )
    return _finish(observations, controls, global_before, samples, {
        "gauge_unchanged": bool(torch.equal(gauge["compact"], gauge_before)),
        "source_unchanged": bool(torch.equal(source["compact"], source_before)),
    })


def _solve_case(option: NativeCase, probe: BackendProbe,
                grid: Sequence[int], environment: NativeEnvironment,
                started: float, *,
                geometry: LocalGeometry | None = None) -> dict[str, Any]:
    """Run one native solve and reconstruct the true residual independently.

    Every operand is the rank-local slab of ``geometry.local`` - the extents the
    ABI is told - and the independent residual is planned **before** the native
    call (:func:`_preflight_reference`), so a reference that cannot fit the
    budget is an explicit ``SKIP(reference_workspace)`` instead of an
    ``ERROR`` after a successful solve.  On more than one rank the residual is
    the bounded distributed, halo-exchanged ``D x - b`` of
    :func:`_residual_distributed`; a failure that still happens after the solve
    is reported as ``residual_verified=False`` + ``residual_unavailable`` and
    never raised.
    """
    torch = _import_torch("kupl_library")
    geometry = _case_geometry(option, grid, geometry)
    local = geometry.local
    plan = _preflight_reference(
        dtype=option.dtype, lattice=geometry.lattice, local=local,
        grid=grid, size=probe.size, budget=option.reference_budget,
        mode=option.reference_mode)
    controls = _Controls(
        probe, lattice=local, grid=grid, dtype=option.dtype,
        mass=option.mass, csw=option.csw, atol=option.atol,
        max_iter=option.max_iter, plan=controls_plan(probe, "solve"),
        parity=1, seed=option.seed, global_lattice=geometry.lattice)
    gauge = _build_gauge(option, torch, controls.torch_dtype, geometry,
                         probe.rank)
    source = _build_source(option, torch, controls.torch_dtype, geometry,
                           probe.rank)
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
    residual_started = time.perf_counter()
    try:
        residual = _true_residual(
            torch, solution=solution, source=source["compact"],
            gauge=gauge["compact"], lattice=geometry.lattice, grid=grid,
            size=probe.size, kappa=controls.kappa, csw=option.csw,
            purpose="solve", budget=option.reference_budget,
            mode=option.reference_mode, plan=plan)
    except Exception as error:  # noqa: BLE001 - never an ERROR after a solve
        residual = _residual_unavailable(
            f"the independent residual failed after the native solve: "
            f"{type(error).__name__}: {error}", plan, residual_started)
    value = residual["relative_residual"]
    observations = _common_observations(option, probe, grid, environment,
                                        controls, started, gauge=gauge,
                                        source=source)
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
        **_residual_fields(residual),
        notes=[
            "the native solver fails closed (a native error aborts the "
            "process), so atol/max_iter are validated before the transaction",
        ],
        **controls.controls_applied(
            mass=option.mass, csw=option.csw, atol=option.atol,
            max_iter=option.max_iter, grid=grid, lattice=local),
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
# Legacy-MG hierarchy (per-level geometry and per-transition shapes)
# ---------------------------------------------------------------------------
def _mg_level_extents(lattice: Sequence[int], mg_grid: Sequence[int],
                      limit: int = MG_MAX_LEVELS,
                      ) -> tuple[tuple[tuple[int, ...], ...],
                                 dict[str, Any] | None]:
    """Derive a Legacy-MG hierarchy, stopping at the first unrepresentable level.

    Level 0 is the odd-Schur fine field (``X, Y, Z, T/2``); level 1 is the one
    transition the cache format stores, ``fine_odd // mg_grid``
    (:func:`pyqcu.tools._legacy_mg_cache.legacy_mg_geometry`); every further
    level halves its parent.  Halving is the finest coarsening the CUDA/KUPL
    block validation accepts for a second transition
    (``cpp/kupl/qcu/src/lattice_clover_multigrid.cpp:238-244``: each coarse
    extent must divide its parent), and the solver only ever builds a block
    factor of ``parent // coarse`` (line 285).

    Returns ``(extents, blocked)``: ``extents`` holds the extents of levels
    ``0..n-1`` that *can* be built, ``blocked`` describes the first level that
    cannot (``level``/``axis``/``extent`` of the parent/``half`` and the
    ``reason``, ``"odd"`` or ``"small"``), or ``None`` when ``limit`` was the
    stopping reason.  A level-1 grid that does not divide the fine odd field is
    a usage error, not an early stop: there is no hierarchy at all without the
    first transition.
    """
    extents = tuple(int(value) for value in lattice)
    grid = tuple(int(value) for value in mg_grid)
    if len(extents) != _SPACE or len(grid) != _SPACE:
        raise ValueError(
            f"lattice {extents} and mg_grid {grid} must both be 4-D")
    if any(value <= 0 for value in extents):
        raise ValueError(f"lattice extents must be positive, got {extents}")
    if extents[3] % 2:
        raise ValueError(
            f"the odd-Schur Legacy-MG needs an even time extent, got {extents}")
    if any(value <= 0 for value in grid):
        raise ValueError(f"mg_grid extents must be positive, got {grid}")
    fine = (extents[0], extents[1], extents[2], extents[3] // 2)
    coarse = []
    for axis in range(_SPACE):
        if fine[axis] % grid[axis]:
            raise ValueError(
                f"level 1 (axis {axis}): the fine odd extent {fine[axis]} is "
                f"not divisible by mg_grid {grid[axis]} (fine {fine}, mg_grid "
                f"{grid}); the first transition has no valid coarse extent")
        coarse.append(fine[axis] // grid[axis])
    levels: list[tuple[int, ...]] = [fine, tuple(coarse)]
    blocked: dict[str, Any] | None = None
    for level in range(2, max(int(limit), 2)):
        parent = levels[-1]
        child = []
        for axis in range(_SPACE):
            extent = parent[axis]
            half = extent // 2
            if extent % 2 or half < MG_COARSEST_EXTENT:
                blocked = {"level": level, "axis": axis, "extent": extent,
                           "half": half,
                           "reason": "odd" if extent % 2 else "small"}
                break
            child.append(half)
        if blocked is not None:
            break
        levels.append(tuple(child))
    return tuple(levels), blocked


def _mg_level_chain(lattice: Sequence[int], mg_grid: Sequence[int],
                    mg_levels: int, dof: Sequence[int] | None = None,
                    ) -> tuple[dict[str, Any], ...]:
    """Return the per-level geometry and component counts of the hierarchy.

    ``dof`` supplies one component count per level: ``dof[0]`` is the
    odd-Schur fine field (:data:`MG_FINE_COMPONENTS`), ``dof[level]`` the coarse
    basis of that level (the first transition's ``out_dof`` is what the cache
    format stores, ``pyqcu/tools/_legacy_mg_cache.py:103-128``).  Every level
    record is ``{"level", "E", "dims", "parent", "block"}``: ``block`` is the
    per-axis ratio to the parent, which is what ``lonv`` stores as its inner
    extents.

    A depth the halving chain cannot reach raises :class:`ValueError` naming the
    level and the extent it would have had - never a silently truncated
    hierarchy, because the C++ then keeps ``num_levels_`` levels and would read
    uninitialised ``params`` blocks for the missing ones.
    """
    count = int(mg_levels)
    if count < 1:
        raise ValueError(f"mg_levels must be >= 1, got {count}")
    if count > MG_MAX_LEVELS:
        raise ValueError(
            f"the KUPL Legacy-MG accepts 1..{MG_MAX_LEVELS} levels "
            f"(cpp/kupl/qcu/src/lattice_clover_multigrid.cpp:195), got {count}")
    values = tuple(int(value) for value in dof) if dof is not None else ()
    if values:
        if values[0] != MG_FINE_COMPONENTS:
            raise ValueError(
                f"dof must start with the {MG_FINE_COMPONENTS}-component fine "
                f"field, got {values[0]} (dof={values})")
        if len(values) < count:
            raise ValueError(
                f"mg_levels={count} needs one dof component count per level; "
                f"dof={values} stops at level {len(values) - 1}.  dof[0] is "
                f"the fine field and dof[level] the coarse basis of that level")
    extents, blocked = _mg_level_extents(lattice, mg_grid, count)
    if blocked is not None:
        # ``_mg_level_extents`` only ever blocks a level below ``count``: the
        # loop stops there, so a blocked hierarchy is always too shallow for the
        # requested depth and must never be truncated to fit.
        if blocked["reason"] == "odd":
            raise ValueError(
                f"level {blocked['level']} (axis {blocked['axis']}) cannot be "
                f"built: level {blocked['level'] - 1} is "
                f"{blocked['extent']} on that axis and an odd extent has no "
                f"integer half (it would be {blocked['half']}).  "
                f"mg_levels={count} does not fit lattice {tuple(lattice)} with "
                f"mg_grid {tuple(mg_grid)}; the hierarchy supports at most "
                f"{blocked['level']} level(s) here")
        raise ValueError(
            f"level {blocked['level']} (axis {blocked['axis']}) would have "
            f"extent {blocked['half']}, below the coarsest extent "
            f"{MG_COARSEST_EXTENT}: level {blocked['level'] - 1} is "
            f"{blocked['extent']} and halves to {blocked['half']}.  "
            f"mg_levels={count} does not fit lattice {tuple(lattice)} with "
            f"mg_grid {tuple(mg_grid)}; the hierarchy supports at most "
            f"{blocked['level']} level(s) here")
    chain: list[dict[str, Any]] = []
    for level in range(count):
        dims = extents[level]
        parent = None
        block = None
        if level:
            parent = level - 1
            previous = extents[level - 1]
            block = tuple(previous[axis] // dims[axis] for axis in range(_SPACE))
        chain.append({
            "level": level,
            "E": (values[level] if level < len(values) else None),
            "dims": dims,
            "parent": parent,
            "block": block,
        })
    return tuple(chain)


def _mg_level_shapes(identity: Mapping[str, Any],
                     chain: Sequence[Mapping[str, Any]],
                     ) -> list[dict[str, tuple[int, ...]]]:
    """Return the four tensor shapes of every coarse transition.

    Transition 1 is the canonical cache shape
    (:func:`pyqcu.test.cache.asset_shapes`), so a mg-2 case keeps exactly the
    shapes and the cache identity it always had.  A deeper transition uses the
    same documented layout with its own component counts and extents
    (``pyqcu/tools/_legacy_mg_cache.py:103-128``): ``lonv`` is
    ``(E, E_parent, X, bx, Y, by, Z, bz, T, bt)`` with the coarse extents and
    the block factors of that transition, ``hnn``/``hdg`` keep their direction
    axes (``[2,4]``/``[2,2,6]``) and ``sit`` is the diagonal block.

    These are the shapes :func:`_configure_legacy_mg` cross-checks its inputs
    against.  KUPL builds the fine transition only (its two builders take the fine
    gauge plus Clover terms), so every transition above the first must be supplied
    as a cache root built beforehand by
    ``pyqcu.testing.simple.mg_transition`` - a checkout-only developer tool.  A
    missing root is refused by :func:`_missing_transition_root`, naming the
    transition; nothing here allocates or builds.
    """
    first = cache_module.asset_shapes(identity)
    shapes = [{name: tuple(int(value) for value in first[name])
               for name in MG_TRANSITION_ASSETS}]
    for level in range(2, len(chain)):
        child = chain[level]
        parent = chain[level - 1]
        dims = tuple(int(value) for value in child["dims"])
        block = tuple(int(value) for value in child["block"])
        components = int(child["E"])
        parent_components = int(parent["E"])
        shapes.append({
            "lonv": (components, parent_components, dims[0], block[0],
                     dims[1], block[1], dims[2], block[2], dims[3], block[3]),
            "hnn": (2, 4, components, components, *dims),
            "hdg": (2, 2, 6, components, components, *dims),
            "sit": (components, components, *dims),
        })
    return shapes


def _mg_chain_record(chain: Sequence[Mapping[str, Any]],
                     shapes: Sequence[Mapping[str, Sequence[int]]],
                     sources: Sequence[str]) -> list[dict[str, Any]]:
    """Row description of the hierarchy: one record per coarse transition.

    ``dims``/``block`` are the *global* hierarchy derived from the physical
    lattice and ``mg_grid``, which is the level a usage error is judged at.
    The ABI block of each level is the rank-local slab - the shapes of the
    tensors actually pointed at - and is recorded separately in
    ``mg_level_parameters`` by :func:`_configure_legacy_mg`.
    """
    record: list[dict[str, Any]] = []
    for level in range(1, len(chain)):
        transition = level - 1
        entry: dict[str, Any] = {
            "level": level,
            "parent": level - 1,
            "E": chain[level]["E"],
            "E_parent": chain[level - 1]["E"],
            "coarse": [int(value) for value in chain[level]["dims"]],
            "block": [int(value) for value in chain[level]["block"]],
            "source": (sources[transition]
                       if transition < len(sources) else "unknown"),
        }
        if transition < len(shapes):
            entry["shape"] = [int(value) for value in shapes[transition]["lonv"]]
        record.append(entry)
    return record


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
    fine_odd = _compact_shape(option.operand_lattice)
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


def _mg_transition_parent(chain: Sequence[Mapping[str, Any]], transition: int,
                          ) -> tuple[tuple[int, ...], tuple[int, int]]:
    """Return ``(lattice, (E_parent, E_child))`` of one coarse transition.

    A cache root publishes exactly one transition, so every identity field is a
    property of *that* transition and not of the case.  The operator the root
    stores acts on the parent level's field, so ``lattice`` is the parent
    level's box in the cache's even-time convention
    (``legacy_mg_geometry`` halves the time extent before blocking) and ``dof``
    is the pair of component counts the transition couples: ``E_parent`` on the
    parent field, ``E_child`` the coarse basis dimension it publishes
    (``lonv`` axis 0 -> ``params[kMgLevel1E]``).

    For transition 1 the parent is level 0, so the pair is the case's own
    lattice and ``option.dof[:2]`` - byte for byte the identity this port wrote
    when it only had one transition.  Transition ``k`` describes
    ``chain[k - 1]``'s coarse box at twice that level's odd-Schur time extent,
    which is the lattice :func:`pyqcu.testing.simple.mg_transition.
    transition_geometry` publishes as ``child_lattice`` and
    :func:`pyqcu.testing.simple.mpi.solve_case._root_lattice_chain` plans as
    ``root_lattice``.  ``E`` is never divided by the process grid.
    """
    parent = chain[transition - 1]
    dims = tuple(int(value) for value in parent["dims"])
    lattice = (dims[0], dims[1], dims[2], 2 * dims[3])
    return lattice, (int(parent["E"]), int(chain[transition]["E"]))


def _mg_identity(option: NativeCase, controls: _Controls, gauge: Any,
                 torch: Any, grid: Sequence[int], *,
                 lattice: Sequence[int] | None = None,
                 dof: Sequence[int] | None = None) -> dict[str, Any]:
    """Cache identity of one transition (gauge digest only when it is global).

    ``lattice``/``dof`` describe the transition being resolved
    (:func:`_mg_transition_parent`); the defaults are the fine transition, i.e.
    the identity this port has always written, so a caller without a hierarchy
    keeps that byte for byte.
    """
    digest = None
    if int(controls.probe.size) == 1:
        digest = cache_module.array_sha256(
            _to_numpy(torch, gauge["full"]))
    return cache_module.mg_identity(
        lattice=option.lattice if lattice is None else lattice,
        mg_grid=option.mg_grid, dtype=option.dtype,
        dof=(tuple(option.dof[:option.mg_levels]) if dof is None
             else tuple(int(value) for value in dof)),
        nv_iters=option.nv_iters,
        seed=option.seed, mass=option.mass, csw=option.csw,
        gauge_sha256=digest, lattice_grid=tuple(int(value) for value in grid),
        mpi_size=int(controls.probe.size))


def _ranked_mg_root(path: Path | None) -> Path | None:
    """Return ``path`` when it is a production ranked MG cache root."""
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    try:
        from pyqcu.tools._legacy_mg_cache import is_legacy_mg_ranked_cache
        return path if is_legacy_mg_ranked_cache(path) else None
    except Exception:  # noqa: BLE001 - an unreadable cache is not a path hit
        return None


def _ranked_mg_cache_path(option: NativeCase) -> Path | None:
    """Return ``option.cache`` when it is a production ranked MG cache root."""
    return _ranked_mg_root(option.cache)


def _load_ranked_mg_assets(
        option: NativeCase, controls: _Controls, torch: Any,
        grid: Sequence[int], path: Path, *,
        lattice: Sequence[int] | None = None,
        dof: Sequence[int] | None = None,
        transition: int = 1,
        ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load coarse tensors from a ranked Legacy-MG cache.

    The cached tensors describe the global coarse operator; the process grid
    and local slabs that published them are only storage metadata.  Each live
    rank therefore computes its own coarse window from the live grid and reads
    the covering slabs.  This permits a cache built by 16 ranks to serve 32 or
    64 ranks when the cache identity (gauge, lattice, dtype, mass, csw,
    ``mg_grid``, seed) still matches.  A single-file cache or a rebuilt serial
    stencil is never used as a fallback; an identity mismatch is an ERROR.

    ``lattice``/``dof``/``transition`` name the transition this root must
    describe (:func:`_mg_transition_parent`).  The default is the fine
    transition with the case's own lattice and ``option.dof[:2]``, which is what
    a two-level case has always compared against; a deeper root is checked
    against *its* parent lattice, never against the fine one.
    """
    from pyqcu.tools._legacy_mg_cache import (
        canonical_dtype, file_sha256, legacy_mg_geometry,
        legacy_mg_local_shapes, load_legacy_mg_cache_slab,
        read_legacy_mg_ranked_manifest,
    )

    lattice = tuple(int(value) for value in
                    (option.lattice if lattice is None else lattice))
    if dof is None:
        requested_dof = tuple(int(value) for value in option.dof)
    else:
        requested_dof = tuple(int(value) for value in dof)
    if len(requested_dof) < 2:
        raise ValueError(
            f"transition {transition} needs (E_parent, E_child), got "
            f"dof={list(requested_dof)}")
    requested_pair = (requested_dof[0], requested_dof[1])
    label = (f"transition {transition}" if transition > 1
             else "the coarse transition")
    root_flag = ("--cache-dir" if transition == 1
                 else f"--coarse-cache-level number {transition - 1}")
    manifest = read_legacy_mg_ranked_manifest(path)
    identity = manifest["identity"]
    # Compared fields.  `nv_builder` and `nv_iters` belong here as much as mass
    # and csw do: they decide what the stored basis IS.  Without `nv_builder` an
    # artifact written by the old `v -= S^-1(S v)` loop - which stored the inner
    # solver's residual, i.e. noise - was accepted for a case asking for the
    # current inverse iteration, because the comparison never looked.  A manifest
    # that records no revision normalises to the "unspecified" sentinel and is
    # therefore refused rather than trusted.
    expected = {
        "lattice": [int(value) for value in lattice],
        "dtype": str(canonical_dtype(option.dtype)),
        "mass": float(option.mass),
        "csw": float(option.csw),
        "mg_grid": list(option.mg_grid),
        "seed": int(option.seed),
        "nv_iters": int(option.nv_iters),
        "nv_builder": str(cache_module.NULL_VECTOR_BUILDER),
    }
    actual = {
        "lattice": list(identity.get("lattice", [])),
        "dtype": str(canonical_dtype(identity.get("dtype"))),
        "mass": float(identity.get("mass", 0.0)),
        "csw": float(identity.get("csw", 0.0)),
        "mg_grid": list(identity.get("mg_grid", [])),
        "seed": int(identity.get("seed", -1)),
        "nv_iters": int(identity.get("nv_iters", -1)),
        "nv_builder": str(identity.get("nv_builder")
                          or cache_module.UNSPECIFIED_NULL_VECTOR_BUILDER),
    }
    # nv_iters describes the KUPL C++ null-vector builder's inverse iteration, so
    # it is compared for transition 1 only.  A deeper transition is the Python
    # probe, which records its own pass count - one per vector, by construction
    # (mg_transition.transition_identity) - and that number says nothing about the
    # case's --nv-iters, which governs the C++ builder.  Comparing them was the
    # only thing that refused a correct transition 2, whose every other field
    # matched.  nv_builder still is compared for every transition: it names which
    # basis the artifact holds, and that must never be taken on trust.
    if int(transition) > 1:
        expected.pop("nv_iters")
        actual.pop("nv_iters")
    if actual != expected:
        raise ValueError(
            f"ranked Legacy-MG cache identity of {label} does not match this "
            f"case ({path}): {actual} != {expected}")
    # E, the coarse basis dimension per coarse site (``dof[1]``), is the
    # capacity of the coarse space: ``capture_bound = sqrt(E / (e * block))``
    # (cpp/kupl/qcu/src/lattice_clover_multigrid.cpp:1019-1023,1092).  It
    # travels *inside* the artifact - ``lonv``'s first axis and
    # ``identity['dof']`` - and ``_configure_legacy_mg`` copies it into
    # ``params[kMgLevel1E]`` (=19), so a stale cache silently pins E to whatever
    # it was built with no matter what the case declares.  That is how every
    # earlier run ended up at E=2.  A case that asks for a different ``dof``
    # must get that E or an error, never the cache's E.
    cached_dof = tuple(int(value) for value in identity.get("dof", ()))
    if len(cached_dof) < 2 or cached_dof[1] <= 0:
        raise ValueError(
            f"ranked Legacy-MG cache {path} ({label}) records no usable dof/E: "
            f"{list(cached_dof)}")
    cached_e = int(cached_dof[1])
    # ``(E_parent, E_child)`` is the pair this transition couples, so both
    # entries are compared: a root built for the neighbouring transition has the
    # same lattice whenever the block is uniform, and only the pair says which
    # level it publishes.  E_parent is only adoptable for the fine transition -
    # a deeper transition's parent E is the level below, which this resolve loop
    # already fixed - so an undeclared ``dof`` keeps the documented "adopt the
    # artifact's E" behaviour for exactly the case that has one transition.
    if (option.dof_declared or transition > 1) \
            and (cached_dof[0], cached_e) != requested_pair:
        raise ValueError(
            f"ranked Legacy-MG cache dof={list(cached_dof)} of {label} "
            f"({path}) does not match the requested dof pair "
            f"{list(requested_pair)}.  (E_parent, E_child) is a property of the "
            f"artifact (E_child is lonv axis 0 -> params[kMgLevel1E]=19) and "
            f"cannot be changed at load time: build a dedicated cache with "
            f"`--dof {' '.join(str(v) for v in requested_pair)}` (pyqcu/testing/"
            f"simple/kupl_multigrid.py --write-coarse-cache) or pass "
            f"{root_flag} to one that already has it.")
    if option.gauge_file is not None and identity.get("gauge_file_sha256"):
        digest = file_sha256(option.gauge_file)
        if digest != identity["gauge_file_sha256"]:
            raise ValueError(
                "ranked Legacy-MG cache gauge_file_sha256 does not match "
                f"--gauge-file {option.gauge_file}")
    # The cached tensors describe the global coarse operator, not the rank
    # partition that happened to publish them.  A live grid may therefore read
    # its own coarse window from a cache built by another process grid, as long
    # as the global geometry and identity match.  The geometry is the one of
    # *this* transition: ``lattice`` is its parent field, so the coarse box is
    # the level this root publishes.
    _, coarse = legacy_mg_geometry(lattice, option.mg_grid)
    if list(manifest.get("global_coarse", [])) != list(coarse):
        raise ValueError(
            f"ranked Legacy-MG cache global_coarse "
            f"{manifest.get('global_coarse')} of {label} != live coarse "
            f"{list(coarse)}")
    live_grid = tuple(int(value) for value in grid)
    if any(coarse[axis] % live_grid[axis] for axis in range(4)):
        raise ValueError(
            f"coarse geometry {coarse} of {label} is not divisible by live "
            f"grid {live_grid}")
    local_coarse = tuple(
        coarse[axis] // live_grid[axis] for axis in range(4))
    rank = int(controls.probe.rank)
    if math.prod(live_grid) != int(controls.probe.size):
        raise ValueError(
            f"live grid {live_grid} does not match MPI size {controls.probe.size}")
    coordinates = [0, 0, 0, 0]
    remainder = rank
    for axis in range(3, -1, -1):
        coordinates[axis] = remainder % live_grid[axis]
        remainder //= live_grid[axis]
    starts = tuple(
        coordinates[axis] * local_coarse[axis] for axis in range(4))
    values = load_legacy_mg_cache_slab(
        path, option.dtype, starts=starts, local_shape=local_coarse,
        names=("lonv", "hnn", "hdg", "sit"))
    assets = {
        name: _device_tensor(torch, values[name], controls.probe,
                             controls.torch_dtype)
        for name in ("lonv", "hnn", "hdg", "sit")
    }
    local_shapes = legacy_mg_local_shapes(
        manifest["global_shapes"], local_coarse)
    repartitioned = (
        list(manifest.get("process_grid", [])) != list(live_grid)
        or int(len(manifest.get("ranks", []))) != int(controls.probe.size))
    notes = [
        (f"ranked Legacy-MG cache of {label} repartitioned to live process grid"
         if repartitioned
         else f"ranked Legacy-MG cache of {label} loaded; no coarse rebuild"),
        f"coarse basis E={cached_e} (e={MG_FINE_COMPONENTS}) "
        f"dof={list(cached_dof)} -> params[kMgLevel1E]={cached_e}",
    ]
    entry = {
        "cache_reused": True,
        "cache_seconds": None,
        "cache_notes": notes,
        "cache_shapes": local_shapes,
        "cache_identity": identity,
        "cache_manifest": str(path),
        "cache_rank": rank,
        "cache_starts": list(starts),
        "cache_local_coarse": list(local_coarse),
        #: The coarse basis dimension actually in force for this run.  It is the
        #: argument of ``capture_bound`` and therefore the one number a reader
        #: of the row needs in order to know which coarse space was measured.
        "mg_e": cached_e,
        "mg_dof": [int(value) for value in cached_dof],
    }
    return assets, entry, dict(entry)


def _missing_transition_root(option: NativeCase, transition: int,
                             depth: int) -> ValueError:
    """Refusal for the first transition whose cache root was not supplied."""
    flag = (f"--coarse-cache-level number {transition - 1}"
            if transition > 1 else "--cache-dir")
    return ValueError(
        f"mg_levels={depth} needs a cache root for multigrid transition "
        f"{transition} (level {transition - 1} -> level {transition}), but "
        f"{flag} was not supplied.  A level-N hierarchy consumes N-1 "
        f"per-transition roots in order: --cache-dir names transition 1 (the "
        f"gauge-built 33-tensor stencil) and every --coarse-cache-level names "
        f"the next one, so the first --coarse-cache-level is transition 2.  "
        f"KUPL has no coarse-to-coarser builder - "
        f"applyMultigridBuildNullVectorsQcu and applyMultigridBuildStencilLocalQcu "
        f"take the fine gauge plus Clover terms "
        f"(cpp/kupl/qcu/src/apply_multigrid.cpp:232,281) - so transition "
        f"{transition} must come from a root built beforehand by "
        f"pyqcu.testing.simple.mg_transition (a developer tool, not shipped in "
        f"pyqcu.test).")


def _mg_transition_roots(option: NativeCase,
                         depth: int) -> list[Path | None]:
    """Cache root of every coarse transition, in hierarchy order.

    ``[0]`` is transition 1, which ``--cache-dir`` names; ``None`` there means
    "no explicit root", so the runner resolves or builds it exactly as it
    always has (:func:`_resolve_mg_transition`).  ``[k]`` for ``k >= 1`` is
    ``--coarse-cache-level`` number ``k`` and must be present: a deeper
    transition has no builder in this port, so a missing root is refused here -
    before any tensor is allocated or any inverse iteration runs - and never
    falls back to the coarser root the caller did supply.
    """
    roots: list[Path | None] = [None] * max(0, int(depth) - 1)
    if depth <= 2:
        return roots
    supplied = tuple(getattr(option, "coarse_cache_level", ()) or ())
    # A hierarchy of `depth` levels has `depth - 1` transitions; transition 1 is
    # --cache-dir, so the ones that need --coarse-cache-level are 2..depth-1
    # INCLUSIVE.  `depth + 1` here asked a three-level case for a transition 3
    # that cannot exist, which the run then refused.
    for transition in range(2, int(depth)):
        index = transition - 2
        value = supplied[index] if index < len(supplied) else None
        if value is None or not str(value).strip():
            raise _missing_transition_root(option, transition, int(depth))
        path = Path(str(value)).expanduser()
        if not path.is_file():
            raise ValueError(
                f"mg_levels={depth} needs a cache root for multigrid "
                f"transition {transition}, and --coarse-cache-level number "
                f"{transition - 1} names {path}, which is not a file.  There is "
                f"no coarser root to fall back to: each root publishes exactly "
                f"one transition (pyqcu/tools/_legacy_mg_cache.py:103-128), so "
                f"transition {transition} cannot be resolved from the root of "
                f"transition {transition - 1}.")
        roots[transition - 1] = path
    return roots


def _mg_transition_record(transition: int, chain: Sequence[Mapping[str, Any]],
                          root: Path | None, entry: Any,
                          info: Mapping[str, Any]) -> dict[str, Any]:
    """Row/cache description of one resolved transition.

    ``identity`` is the cache identity that was actually used for this
    transition - the transition's own parent lattice and ``(E_parent, E_child)``
    - and ``root`` is the artifact it was resolved from, so a reader can see
    which level each root supplies without re-deriving the hierarchy.
    """
    lattice, dof = _mg_transition_parent(chain, transition)
    record: dict[str, Any] = {
        "transition": int(transition),
        "parent_level": int(transition - 1),
        "child_level": int(transition),
        "lattice": [int(value) for value in lattice],
        "dof": [int(value) for value in dof],
        "root": None if root is None else str(root),
        "source": "cache" if info.get("cache_reused") else "native",
        "cache_reused": bool(info.get("cache_reused")),
    }
    if entry is not None:
        values = (entry.as_dict() if hasattr(entry, "as_dict") else dict(entry))
        for name in ("cache_manifest", "cache_identity", "mg_e", "mg_dof"):
            if name in values:
                record[name] = values[name]
    return record


def _resolve_mg_assets(option: NativeCase, controls: _Controls, gauge: Any,
                       terms: Sequence[Any], inverses: Sequence[Any],
                       torch: Any, grid: Sequence[int], *,
                       chain: Sequence[Mapping[str, Any]] | None = None
                       ) -> tuple[list[dict[str, Any]], Any, dict[str, Any]]:
    """Resolve every coarse transition of the hierarchy, in order.

    The list holds one four-tensor dict per coarse transition, in hierarchy
    order (``[0]`` is level 0 -> level 1), which is the indexing
    :func:`_configure_legacy_mg` places into ``params``/``set_ptrs``.  A
    ``mg_levels`` of 1 or 2 resolves to exactly the single transition this port
    has always used, through :func:`_resolve_mg_transition`.

    Transition 1 keeps its cache-or-native behaviour.  Every deeper transition
    is read from its own root (``--coarse-cache-level`` number ``transition-1``)
    because the cache format publishes one transition per root
    (``pyqcu/tools/_legacy_mg_cache.py:103-128``) and KUPL's two builders take
    the *fine* gauge plus Clover terms
    (``cpp/kupl/qcu/src/apply_multigrid.cpp:232,281``), so nothing in this port
    (or in C++) can build a coarse-to-coarser operator.  A missing root is
    refused by :func:`_mg_transition_roots` before any work happens; a root whose
    identity does not describe its own transition is refused by
    :func:`_load_ranked_mg_assets`; nothing is ever built as a fallback.
    """
    depth = int(option.mg_levels)
    if chain is None:
        chain = _mg_level_chain(option.lattice, option.mg_grid, depth,
                                option.dof)
    roots = _mg_transition_roots(option, depth)
    assets: list[dict[str, Any]] = []
    entries: list[Any] = []
    infos: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for transition in range(1, depth):
        transition_assets, entry, info = _resolve_mg_transition(
            option, controls, gauge, terms, inverses, torch, grid,
            transition=transition, chain=chain,
            root=roots[transition - 1])
        assets.append(transition_assets)
        entries.append(entry)
        infos.append(info)
        records.append(_mg_transition_record(
            transition, chain, roots[transition - 1], entry, info))
    sources = ["cache" if info.get("cache_reused") else "native"
               for info in infos]
    if depth == 2:
        # One transition: the row keeps exactly the fields it has always had.
        info = dict(infos[0])
        info["mg_transition_sources"] = sources
        return assets, entries[0], info
    merged = dict(infos[0])
    merged["cache_reused"] = all(info.get("cache_reused") for info in infos)
    merged["cache_seconds"] = None
    merged["cache_notes"] = [
        f"transition {record['transition']} ({record['source']}): {note}"
        for record, info in zip(records, infos)
        for note in (info.get("cache_notes") or ())]
    merged["mg_transition_sources"] = sources
    merged["mg_transition_caches"] = records
    return assets, merged, merged


def _resolve_mg_transition(option: NativeCase, controls: _Controls, gauge: Any,
                           terms: Sequence[Any], inverses: Sequence[Any],
                           torch: Any, grid: Sequence[int], *,
                           transition: int = 1,
                           chain: Sequence[Mapping[str, Any]] | None = None,
                           root: Path | None = None
                           ) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    """Resolve the four tensors of one coarse transition.

    ``cache_policy`` selects the behaviour of transition 1: ``auto`` reuses a
    valid cache and otherwise builds (publishing the result), ``off`` always
    builds, and ``require`` refuses to build.  A single-file cache describes one
    rank's coarse slab, so a multi-rank request is a usage error instead of a
    silent cross-identity hit.

    A transition above 1 is resolved from ``root`` and never built: this port
    has no coarse-to-coarser builder, and the root was already checked to exist
    by :func:`_mg_transition_roots`, so an unusable deeper root is an error
    rather than a rebuild.  Its identity is built from *that* transition's
    parent lattice and ``(E_parent, E_child)``
    (:func:`_mg_transition_parent`), never from ``option.lattice``.
    """
    lattice: Sequence[int] | None = None
    dof: Sequence[int] | None = None
    if chain is not None:
        lattice, dof = _mg_transition_parent(chain, transition)
    identity = _mg_identity(option, controls, gauge, torch, grid,
                            lattice=lattice, dof=dof)
    if transition == 1:
        ranked = _ranked_mg_cache_path(option)
        source = option.cache
    else:
        ranked = _ranked_mg_root(root)
        source = root
    if ranked is not None:
        return _load_ranked_mg_assets(
            option, controls, torch, grid, ranked, lattice=lattice, dof=dof,
            transition=transition)
    shapes = cache_module.asset_shapes(identity)
    notes: list[str] = []
    path = (source if source is not None
            else cache_module.cache_path(identity, option.cache_directory))
    if option.cache_policy != "off":
        if int(controls.probe.size) > 1:
            raise ValueError(
                "a single-file Legacy-MG cache stores one rank's coarse slab; "
                "run one rank or set cache_policy='off' for a multi-rank MG "
                f"case (mpi_size={controls.probe.size}, transition "
                f"{transition}, {path})")
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
    if transition > 1:
        # Nothing in pyqcu.test can build this operator, and a placeholder
        # would make the guarded ABI call abort the process, so an unusable root
        # is refused with the transition and the flag that supplied it.
        raise ValueError(
            f"the cache root of multigrid transition {transition} "
            f"(--coarse-cache-level number {transition - 1} = {path}) is not a "
            f"usable Legacy-MG cache for this case: it must be a ranked root "
            f"(`<root>` + `<root>.d/` + `<root>.json`) or a single-file "
            f"canonical cache whose identity is {identity}"
            + (f"; {notes[-1]}" if notes else "")
            + ".  It cannot be built here: KUPL's builders take the fine gauge "
              "plus Clover terms (cpp/kupl/qcu/src/apply_multigrid.cpp:232,281) "
              "and the coarse-to-coarser builder is the developer tool "
              "pyqcu.testing.simple.mg_transition, which is not shipped in "
              "pyqcu.test.")
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


def _mg_local_chain(chain: Sequence[Mapping[str, Any]],
                    grid: Sequence[int]) -> list[dict[str, Any]]:
    """Return the hierarchy with each level's extents divided by ``grid``.

    ``_configure_legacy_mg`` cross-checks the tensors it points at, and those are
    the rank-local slabs: ``lonv`` stores ``(E, E_parent, coarse/G, block)``.
    The row-level hierarchy (:func:`_mg_level_chain` and
    :func:`_mg_chain_record`) stays the global one - a usage error is judged
    against the physical lattice - so the cross-check gets this derived view
    instead.  ``E`` (a per-coarse-site count) and the block factors are not
    divided.  With one rank the two views are identical.
    """
    live = tuple(int(value) for value in grid)
    local: list[dict[str, Any]] = []
    for level in chain:
        dims = tuple(int(value) for value in level["dims"])
        if any(dims[axis] % live[axis] for axis in range(4)):
            raise ValueError(
                f"level {level['level']} extents {dims} of the Legacy-MG "
                f"hierarchy are not divisible by the live process grid {live}")
        local.append(dict(level, dims=tuple(dims[axis] // live[axis]
                                           for axis in range(4))))
    return local


def _configure_legacy_mg(option: NativeCase, controls: _Controls,
                         assets: Sequence[Mapping[str, Any]],
                         chain: Sequence[Mapping[str, Any]] | None = None
                         ) -> dict[str, Any]:
    """Wire the coarse assets into ``params``/``argv``/``set_ptrs``.

    The parameter block is the documented Legacy-MG one: level 0 is the fine
    odd-Schur field (``kVector`` components), every further level ``L`` reads its
    ``E`` and coarse extents from ``params[MG_LEVEL_BASE + MG_LEVEL_STRIDE*L-1 :
    ...]`` and its four asset pointers from
    ``set_ptrs[SLOT_LIMIT + 4*(L-1) : ...]``; its tolerance is
    ``argv[ARGV_MG_LEVEL_BASE + L - 1]``.  ``assets`` holds one four-tensor dict
    per coarse transition (``assets[L-1]`` is level ``L-1 -> L``), which is the
    same indexing the C++ uses for its own per-level loop
    (``cpp/kupl/qcu/src/lattice_clover_multigrid.cpp:231-275``).
    ``set_ptrs[99]`` selects the odd-Schur fine layout, which is what the cache
    stores.

    The terminal block shape is read from the transition's own ``lonv`` tensor,
    so the recorded geometry is the geometry the solver will read; when
    ``chain`` is given (:func:`_mg_level_chain`) it is used to cross-check the
    tensor against the derived hierarchy and a mismatch is an error rather than a
    silently wrong level.
    """
    define = controls.define
    params, argv, set_ptrs = controls.params, controls.argv, controls.set_ptrs
    depth = int(option.mg_levels)
    params[define._MG_NUM_LEVEL_] = depth
    params[define._MG_USE_DEFLATE_] = 1 if option.deflate else 0
    # ``smoothing_prefix`` in the solver reads params[kMgMuPre] (=56); nothing
    # in Python wrote it, so --mg-nu-pre was accepted, recorded in the row and
    # then dropped.  Leave it at 0 unless the case asks, which keeps the
    # solver-side default (a quarter of the interval).
    params[define._MG_MU_PRE_] = int(getattr(option, "mg_nu_pre", 0) or 0)
    if len(assets) < depth - 1:
        raise ValueError(
            f"mg_levels={depth} needs {depth - 1} coarse transitions, got "
            f"{len(assets)} asset set(s)")
    levels: list[dict[str, Any]] = []
    for level in range(1, depth):
        level_assets = assets[level - 1]
        block = tuple(int(value) for value in level_assets["lonv"].shape)
        if len(block) != 10:
            raise ValueError(
                f"level {level} lonv must be "
                f"(E, E_parent, X, bx, Y, by, Z, bz, T, bt), got {block}")
        if chain is not None:
            expected = chain[level]
            parent = chain[level - 1]
            coarse = [block[2], block[4], block[6], block[8]]
            factors = [block[3], block[5], block[7], block[9]]
            derived = [int(value) for value in expected["dims"]]
            derived_block = [int(value) for value in expected["block"]]
            if (coarse != derived or factors != derived_block
                    or block[0] != int(expected["E"])
                    or block[1] != int(parent["E"])):
                raise ValueError(
                    f"level {level} lonv {block} does not match the derived "
                    f"hierarchy (E={expected['E']}, E_parent={parent['E']}, "
                    f"coarse={derived}, block={derived_block})")
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
        for offset, name in enumerate(MG_TRANSITION_ASSETS):
            set_ptrs[slot + offset] = int(level_assets[name].data_ptr())
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
             environment: NativeEnvironment, started: float, *,
             geometry: LocalGeometry | None = None) -> dict[str, Any]:
    """Run one native Legacy-MG solve with a cache or a native asset build.

    Same operand geometry as :func:`_solve_case` (the rank-local slab), with the
    independent reference planned before the native call and never turned into
    an ``ERROR`` afterwards.
    """
    torch = _import_torch("kupl_library")
    geometry = _case_geometry(option, grid, geometry)
    local = geometry.local
    plan = _preflight_reference(
        dtype=option.dtype, lattice=geometry.lattice, local=local,
        grid=grid, size=probe.size, budget=option.reference_budget,
        mode=option.reference_mode)
    controls = _Controls(
        probe, lattice=local, grid=grid, dtype=option.dtype,
        mass=option.mass, csw=option.csw, atol=option.atol,
        max_iter=option.max_iter, plan=controls_plan(probe, "mg"),
        parity=1, seed=option.seed, global_lattice=geometry.lattice)
    gauge = _build_gauge(option, torch, controls.torch_dtype, geometry,
                         probe.rank)
    source = _build_source(option, torch, controls.torch_dtype, geometry,
                           probe.rank)
    terms, inverses = _native_clover_terms(controls, gauge["compact"], torch)
    assets: list[dict[str, Any]] = []
    chain: tuple[dict[str, Any], ...] | None = None
    cache_entry = None
    cache_info: dict[str, Any] = {"cache_reused": False, "cache_seconds": None,
                                  "cache_notes": ["no coarse transition: "
                                                  "mg_levels=1"],
                                  "mg_transition_sources": []}
    if int(option.mg_levels) > 1:
        chain = _mg_level_chain(option.lattice, option.mg_grid,
                                int(option.mg_levels), option.dof)
        assets, cache_entry, cache_info = _resolve_mg_assets(
            option, controls, gauge, terms, inverses, torch, grid, chain=chain)
    sources = list(cache_info.get("mg_transition_sources") or [])
    mg_parameters = _configure_legacy_mg(
        option, controls, assets, None if chain is None or len(chain) < 3
        else _mg_local_chain(chain, grid))
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
    residual_started = time.perf_counter()
    try:
        residual = _true_residual(
            torch, solution=solution, source=source["compact"],
            gauge=gauge["compact"], lattice=geometry.lattice, grid=grid,
            size=probe.size, kappa=controls.kappa, csw=option.csw, purpose="MG",
            budget=option.reference_budget, mode=option.reference_mode,
            plan=plan)
    except Exception as error:  # noqa: BLE001 - never an ERROR after a solve
        residual = _residual_unavailable(
            f"the independent residual failed after the native MG solve: "
            f"{type(error).__name__}: {error}", plan, residual_started)
    value = residual["relative_residual"]
    observations = _common_observations(option, probe, grid, environment,
                                        controls, started, gauge=gauge,
                                        source=source)
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
        **_residual_fields(residual),
        asset_source=("cache" if cache_entry is not None
                      and cache_info.get("cache_reused") else "native"),
        asset_shapes={name: list(value_.shape)
                      for name, value_ in (assets[0] if assets else {}).items()},
        asset_names=sorted(assets[0] if assets else {}),
        mg_level_chain=_mg_chain_record(
            chain or (), list(cache_info.get("mg_transition_shapes") or []),
            sources),
        **mg_parameters,
        **cache_info,
        notes=[
            "the Legacy-MG entry point aborts the process when its internal "
            "full residual misses the target; atol is the acceptance used for "
            "both the native check and this independent reconstruction",
        ],
        **controls.controls_applied(
            mass=option.mass, csw=option.csw, atol=option.atol,
            max_iter=option.max_iter, grid=grid, lattice=local),
    )
    if cache_entry is not None:
        observations.update(cache_entry.as_dict()
                            if hasattr(cache_entry, "as_dict")
                            else dict(cache_entry))
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
               gauge_file: Any = None, source_file: Any = None,
               reference_budget: Any = None, reference_mode: Any = None,
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
        source_color=source_color, backend=backend, gauge_file=gauge_file,
        source_file=source_file, reference_budget=reference_budget,
        reference_mode=reference_mode, options=options, extras=extras)
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
              gauge_file: Any = None, source_file: Any = None,
              reference_budget: Any = None, reference_mode: Any = None,
              options: Any = None, **extras: Any) -> dict[str, Any]:
    """Run one native BiCGStab solve and report the independent true residual.

    The operator is Wilson when ``csw == 0`` (the KUPL Wilson solver or, on
    CUDA, the Clover solver with a zero Clover term) and Wilson-Clover
    otherwise.  ``||D x - b|| / ||b||`` is reconstructed from the raw solution
    with ``pyqcu.torch.dslash`` on this rank's slab, with the halo exchanged
    through ``pyqcu.torch.tools._mpi_roll`` and ``||r||^2``/``||b||^2``
    Allreduced, so no rank ever holds the global field.  A reference that does
    not fit ``reference_budget`` skips with ``reference_workspace`` *before* the
    solve.

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
        source_color=source_color, backend=backend, gauge_file=gauge_file,
        source_file=source_file, reference_budget=reference_budget,
        reference_mode=reference_mode, options=options, extras=extras)
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
           mg_nu_pre: Any = None,
           deflate: Any = None, verify: Any = None, cache: Any = None,
           cache_policy: Any = None, cache_directory: Any = None,
           coarse_cache_level: Any = None,
           gauge_file: Any = None, source_file: Any = None,
           reference_budget: Any = None, reference_mode: Any = None,
           options: Any = None, **extras: Any) -> dict[str, Any]:
    """Run one native Legacy-MG solve (``applyCloverMultigridQcu``).

    The four coarse tensors come from a validated cache
    (:mod:`pyqcu.test.cache`) when one matches the case identity, otherwise
    from the native null-vector/33-stencil builders; the run then publishes the
    result so the next identical case reuses it.  ``cache_policy="off"`` always
    builds, ``"require"`` refuses to build.

    One cache root publishes one transition, so a hierarchy of ``mg_levels``
    needs ``mg_levels - 1`` roots: ``cache`` (``--cache-dir``) names transition 1
    and ``coarse_cache_level`` (repeatable ``--coarse-cache-level``, first entry
    = transition 2) names the rest, in order.  Only transition 1 can be built
    here; a deeper root must already exist and a missing one is an error naming
    the transition.

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
        source_color=source_color, backend=backend, gauge_file=gauge_file,
        source_file=source_file, reference_budget=reference_budget,
        reference_mode=reference_mode, options=options,
        mg_levels=mg_levels, dof=dof, nv_iters=nv_iters, restart=restart,
        coarse_max_iter=coarse_max_iter, mg_nu_pre=mg_nu_pre,
        coarse_tol_factor=coarse_tol_factor, deflate=deflate, verify=verify,
        cache=cache, cache_policy=cache_policy,
        cache_directory=cache_directory,
        coarse_cache_level=coarse_cache_level, extras=extras)
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


#: Runner options a ``Case`` has to carry explicitly: ``Case`` only declares
#: the cross-runner fields, so anything else would be validated by ``_options``
#: and then discarded, making the runner fall back to its own default.
_CARRIED_OPTIONS: tuple[str, ...] = (
    "restart", "mg_levels", "mg_grid", "coarse_tol_factor", "coarse_max_iter",
    "nv_iters", "dof", "deflate", "verify", "cache", "cache_policy",
    "coarse_cache_level",
)


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
    built = Case(
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
    # ``Case`` carries only the cross-runner fields, so a runner-specific knob
    # passed here would be validated and then dropped: the runner would fall
    # back to its own default and a parameter sweep would silently run the same
    # point N times (2026-11-24, every MG sweep row recorded restart=5/1e4
    # whatever the command line said).  Attach what the caller actually set;
    # ``_options`` reads it back through getattr.
    for name in _CARRIED_OPTIONS:
        value = options.get(name)
        if value is None:
            continue
        try:
            setattr(built, name, value)
        except AttributeError:  # frozen Case
            object.__setattr__(built, name, value)
    return built
