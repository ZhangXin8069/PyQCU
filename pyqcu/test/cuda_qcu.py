"""CUDA backend runners for the formal :mod:`pyqcu.test` port.

The module is import-light: only the standard library (and the stdlib-only
sibling :mod:`pyqcu.test.common`/:mod:`pyqcu.test.policy`) is imported at
module import time.  ``torch``, ``mpi4py`` (reached through
``pyqcu.cuda.define``) and the Cython bridge ``pyqcu.cuda.qcu`` are imported
inside the runner functions, so ``python -m pyqcu.test --list`` never pays for
a backend import.

Contract
--------
Each ``run_*`` function returns a *plain observation dict*: numbers, booleans,
strings and lists only.  It never decides PASS/FAIL and never compares a number
against a tolerance; :mod:`pyqcu.test.units` declares the ``expect`` mapping
and :mod:`pyqcu.test.common` evaluates it.  The observations carry the
established key spellings (``relative_error``, ``relative_residual``,
``iterations``, ``max_iter``) plus the aliases accepted by
``policy.OBSERVATION_ALIASES``.

Run options
-----------
Every numerical runner accepts and forwards the configuration the CLI
advertises: ``lattice``, ``lattice_grid``, ``dtype``, ``seed``, ``repeat``,
``warmup``, ``mass``/``kappa``, ``csw``, ``gauge_sigma``, ``source``,
``gauge``, ``max_iter``, ``atol``, ``device`` and ``device_id``.  They are
normalised once into :class:`_RunOptions`; the gauge/source builders and
:func:`_make_runtime` consume that object, and every observation reports the
values that actually ran (``mass``, ``kappa``, ``csw``, ``gauge_sigma``,
``source``, ``gauge``, ``max_iter``, ``atol``, ``device``,
``lattice_grid``).  ``controls_applied`` additionally proves that the private
ABI arrays carry those values; a mismatch raises ``RuntimeError`` (an ERROR)
instead of reporting a green case whose options were ignored.

``mass`` reaches ``argv[_MASS_]``/``kappa = 1/(2*mass + 8)`` and the torch
reference.  The bare Wilson dslash symbol is mass- and clover-independent, so
the Wilson runner reports ``mass_used_by_operator=False`` (the full-lattice
reference with the run's ``kappa`` still exercises the value) and
``csw_used_by_operator=False``; the clover dslash and both solvers consume
``csw`` through the native clover term.  ``atol``/``max_iter`` reach the ABI
solver controls; they are inert for the dslash cases and reported as such.

Probes and skips
----------------
Both prerequisites are probed *before* any native call:

``torch.cuda.is_available()``
    otherwise ``raise Skip("cuda_device: ...")``;
``pyqcu.cuda.qcu`` imports as a compiled extension and exposes ``applyInitQcu``
    otherwise ``raise Skip("cuda_bridge: ...")``.  A source directory that
    shadows the extension imports as a *namespace package* (``__file__`` is
    ``None``, no symbols); that is the expected state of an unbuilt checkout
    and is reported as ``cuda_bridge``, never as a pass and never as a
    numerical failure.

A requested device that is not a usable CUDA device skips ``device_unavailable``
(``--device cpu`` on a GPU runner is a skip, not an error).  A multi-rank MPI
job skips ``mpi_single_rank`` and a ``lattice_grid`` whose product is not the
MPI size skips ``mpi_grid``: the tier-0 CUDA cases are one rank, one device,
one process.

The bridge is never built, imported for side effects or monkey-patched here.

Lifecycle
---------
Every native call follows the documented ordinary transaction::

    select a free slot in set_ptrs[0:30)
        -> applyInitQcu
        -> operation (and any same-slot getter)
        -> applyEndQcu

``_SET_INDEX_`` is never advanced before ``applyEndQcu``.  All ABI control
arrays (``params`` int32[58], ``argv`` real[8], ``set_ptrs`` int64[100]) belong
to a private :class:`pyqcu.cuda._runtime.QcuRuntime` created by
``QcuRuntime.from_lattice``; the module-level arrays in ``pyqcu.cuda.define``
are never passed to the library and every runner verifies afterwards that they
are unchanged (``globals_unchanged``).  Slots from index 30 onward are
reserved for the multigrid wrappers and are never touched.

Layout
------
Parity-split ABI tensors are the project layout ``poooxyzt``: fermions are
``[2, 4, 3, X, Y, Z, T/2]``, gauge links ``[2, 3, 3, 4, X, Y, Z, T/2]`` and
Clover terms ``[4, 3, 4, 3, X, Y, Z, T/2]`` (``T`` halved, ``T/2`` fastest).
``pyqcu.tools`` converts to/from the full ``oooxyzt`` lattice, and the torch
reference is built from the same full-lattice operator
(``pyqcu.torch.dslash.give_wilson``) so the comparison never trusts the
backend's own residual.

Operators
---------
``cuda/dslash/wilson``
    ``applyWilsonDslashQcu`` (the bare hopping sum) against the torch
    even/odd helpers *and* the full-lattice Wilson operator at the run's
    ``kappa``.
``cuda/dslash/clover``
    ``applyCloverDslashQcu`` applies ``A^-1 * D_W`` with
    ``A = I + csw * T``.  The reference rebuilds the bare hopping with torch
    and applies the **native** ``A^-1`` produced by ``applyCloversQcu``; the
    native-vs-torch clover construction is reported ungated because c128
    agreement (~1e-9) is above ``policy``'s 1e-10 ``compare`` gate.
``cuda/solve/bistabcg``
    ``applyCloverBistabCgQcu`` with the clover term at the requested ``csw``
    (``csw = 0`` is the Wilson operator).  It reports a native iteration count
    (``getCloverBistabCgIterationsQcu``) and its ``argv[_ATOL_]`` is a
    relative tolerance with a round-off floor.  The independent true residual
    is always reconstructed from the full clover/Wilson operator, never from
    the solver recurrence.
``cuda/solve/cg``
    ``applyWilsonCgQcu`` performs the whole native CG solve.  The CUDA ABI
    exposes no CG iteration getter (only the Clover BiCGStab counter), and the
    C++ solver prints its final ``##RANK:<r>##LOOP:<n>##Residual(norm2):...``
    line to file descriptor 1.  The runner captures that single line for the
    duration of the call and reports ``iterations = max_loop + 1``; if the
    line is absent it raises (ERROR) instead of inventing a count.  The true
    residual is reconstructed with the full Wilson operator.  The symbol has
    no clover term, so ``csw`` reaches ``argv[_CSW_]`` but does not enter the
    operator (``csw_used_by_operator=False``).

No runner is a permanent stub: every declared CUDA case either performs its
native operation or skips for a real, probed reason.
"""

from __future__ import annotations

import math
import os
import re
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, NoReturn

from . import policy
from .common import (Skip, grid_product, relative_error, relative_residual,
                     time_samples)

__all__ = [
    "BACKEND",
    "DEFAULT_CUDA_DEVICE",
    "QUICK_LATTICE",
    "SLOT_LIMIT",
    "probe_cuda",
    "run_dslash_wilson",
    "run_smoke_init_end",
    "run_solve_bistabcg",
]

#: Backend name used in case tags and reports.
BACKEND = "cuda"
#: Project correctness lattice (``AGENTS.md``, ``pyqcu/testing/_config.py``).
QUICK_LATTICE: tuple[int, int, int, int] = (8, 8, 8, 16)
#: ``define._SET_PTRS_COARSE_BASE_``: first index reserved for multigrid.
SLOT_LIMIT = 30
#: Device used when ``QCU_DEVICE`` is not set in the environment.
DEFAULT_CUDA_DEVICE = "cuda:0"

#: Mass of the tier-0 CUDA cases when neither ``mass`` nor ``kappa`` is given
#: (``pyqcu._selftest`` uses the same value).
DEFAULT_MASS = 0.17
#: ``csw`` when a direct caller provides no option: Wilson (zero clover).
DEFAULT_CSW = 0.0
#: Relative solver tolerance (``argv[_ATOL_]``) per dtype.
DEFAULT_ATOL_C64 = 1.0e-6
DEFAULT_ATOL_C128 = 1.0e-10
#: Runaway guard for the native solver; the ``iterations < max_iter`` gate
#: needs the requested limit as an observation.
DEFAULT_MAX_ITER = 200
#: Gauge-field disorder used to build the random fixed input.
DEFAULT_GAUGE_SIGMA = 0.1
#: Source/gauge defaults for a direct API call (the CLI forwards its own).
DEFAULT_SOURCE = "random"
DEFAULT_GAUGE = "random"
#: Point-source location/spin/colour (the CLI source has no site options).
DEFAULT_SOURCE_SITE: tuple[int, int, int, int] = (0, 0, 0, 0)
DEFAULT_SOURCE_SPIN = 0
DEFAULT_SOURCE_COLOR = 0
#: Largest tolerated ``atol``: the ABI value is a relative tolerance.
MAX_ATOL = 1.0

_DEVICE_ENVIRONMENT = "QCU_DEVICE"
_BRIDGE_TOKEN = "cuda_bridge"
_DEVICE_TOKEN = "cuda_device"
_UNAVAILABLE_TOKEN = "device_unavailable"
_RANK_TOKEN = "mpi_single_rank"
_GRID_TOKEN = "mpi_grid"

_SMOKE_SYMBOLS = ("applyInitQcu", "applyEndQcu")
_DSLASH_SYMBOLS = _SMOKE_SYMBOLS + ("applyWilsonDslashQcu",)
_CLOVER_SYMBOLS = _SMOKE_SYMBOLS + (
    "applyCloverDslashQcu",
    "applyCloversQcu",
)
_SOLVE_SYMBOLS = _SMOKE_SYMBOLS + (
    "applyCloversQcu",
    "applyCloverBistabCgQcu",
    "getCloverBistabCgIterationsQcu",
)
_CG_SYMBOLS = _SMOKE_SYMBOLS + ("applyWilsonCgQcu",)

#: The C++ CG prints one ``##RANK:<r>##LOOP:<n>##Residual(norm2):...`` line
#: when its iteration loop ends.  ``loop`` is zero based, so the iteration
#: count is ``max(loop) + 1``.
_NATIVE_LOOP = re.compile(r"##LOOP:(\d+)")

#: Physics controls compared before/after a runner.  ``params[_SET_INDEX_]`` is
#: deliberately absent: it is the runtime's slot cursor and is expected to move
#: inside the ordinary ``set_ptrs[0:30)`` window.
_CONTROL_PARAM_NAMES = (
    "_LAT_X_", "_LAT_Y_", "_LAT_Z_", "_LAT_T_", "_LAT_XYZT_",
    "_GRID_X_", "_GRID_Y_", "_GRID_Z_", "_GRID_T_",
    "_PARITY_", "_NODE_RANK_", "_NODE_SIZE_", "_DAGGER_",
    "_MAX_ITER_", "_DATA_TYPE_", "_SET_PLAN_",
)
_CONTROL_ARGV_NAMES = ("_MASS_", "_ATOL_", "_SIGMA_", "_CSW_")
_GLOBAL_ARRAY_NAMES = ("params", "argv", "set_ptrs")


# ---------------------------------------------------------------------------
# skips
# ---------------------------------------------------------------------------
def _skip(message: Any, token: str) -> NoReturn:
    """Raise the harness :class:`Skip` with a mandatory ``"<token>: <reason>"``.

    The token is carried in the message (and not duplicated in the ``token``
    argument) so that ``str(error)`` is self-describing for a standalone
    caller while :func:`pyqcu.test.common.run_case` still splits it back into
    ``Result.skip_token`` and ``Result.reason``.  An empty reason is an ERROR
    in the harness, so the text is never allowed to be empty.
    """
    text = str(message).strip() or "CUDA backend unavailable"
    raise Skip(f"{token}: {text}")


# ---------------------------------------------------------------------------
# run options
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _RunOptions:
    """Validated subset of ``pyqcu.test.common.Options`` a runner consumes."""

    shape: tuple[int, int, int, int]
    lattice_grid: tuple[int, int, int, int]
    dtype: str
    seed: int
    repeat: int
    warmup: int
    mass: float
    kappa: float
    csw: float
    gauge_sigma: float
    source: str
    gauge: str
    source_site: tuple[int, int, int, int]
    source_spin: int
    source_color: int
    max_iter: int
    atol: float
    device: str | None
    device_id: int | None
    dslash: str
    solver: str

    @property
    def compact(self) -> tuple[int, int, int, int]:
        """Parity-compact ``[X, Y, Z, T/2]`` extents of the ABI tensors."""
        return (*self.shape[:3], self.shape[3] // 2)


def _container_value(container: Any, name: str) -> Any:
    """Read ``name`` from an options container (mapping or object)."""
    if container is None:
        return None
    if isinstance(container, dict):
        return container.get(name)
    return getattr(container, name, None)


def _pick(explicit: Any, container: Any, name: str, default: Any) -> Any:
    """Explicit keyword wins, then the options container, then the default."""
    if explicit is not None:
        return explicit
    value = _container_value(container, name)
    return default if value is None else value


def _resolve_mass_kappa(mass: Any, kappa: Any) -> tuple[float, float]:
    """Resolve ``(mass, kappa)`` with ``kappa = 1/(2*mass + 8)``.

    ``kappa`` alone is enough (``mass = 1/(2*kappa) - 4``); giving both is
    accepted only when they describe the same operator, because a silent
    mismatch would mean the native controls and the torch reference disagree.
    """
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


def _resolve_atol(atol: Any, dtype_name: str) -> float:
    """Resolve the relative solver tolerance for ``dtype_name``."""
    if atol is None:
        value = (DEFAULT_ATOL_C128 if dtype_name == "c128"
                 else DEFAULT_ATOL_C64)
    else:
        value = float(atol)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"atol must be finite and positive, got {value}")
    if value > MAX_ATOL:
        raise ValueError(
            f"atol={value!r} is not a relative tolerance (<= {MAX_ATOL:g})")
    return value


def _resolve_max_iter(max_iter: Any) -> int:
    value = DEFAULT_MAX_ITER if max_iter is None else int(max_iter)
    if value < 1:
        raise ValueError(f"max_iter must be >= 1, got {value}")
    return value


def _resolve_site(values: Any,
                  shape: tuple[int, int, int, int]
                  ) -> tuple[int, int, int, int]:
    """Validate the source site against the lattice extents."""
    site = tuple(int(value) for value in values)
    if len(site) != 4:
        raise ValueError(
            f"source_site must contain four coordinates, got {site}")
    for coordinate, extent in zip(site, shape):
        if coordinate < 0 or coordinate >= extent:
            raise ValueError(f"source site {site} is outside lattice {shape}")
    return site  # type: ignore[return-value]


def _options(*, lattice: Any = None, lattice_grid: Any = None,
             dtype: Any = None, seed: Any = None, repeat: Any = None,
             warmup: Any = None, atol: Any = None, source: Any = None,
             gauge: Any = None, gauge_sigma: Any = None, mass: Any = None,
             kappa: Any = None, csw: Any = None, solver: Any = None,
             dslash: Any = None, device: Any = None, max_iter: Any = None,
             device_id: Any = None, source_site: Any = None,
             source_spin: Any = None, source_color: Any = None,
             options: Any = None) -> _RunOptions:
    """Normalize the runner arguments without importing a backend."""
    shape = policy.validate_lattice(
        _pick(lattice, options, "lattice", QUICK_LATTICE))
    grid = policy.validate_lattice_grid(
        _pick(lattice_grid, options, "lattice_grid",
              policy.DEFAULT_LATTICE_GRID))
    dtype_name = policy.validate_dtype(
        _pick(dtype, options, "dtype", policy.QUICK_DTYPE))
    seed_value = int(_pick(seed, options, "seed", policy.QUICK_SEED))
    if seed_value < 0:
        raise ValueError(f"seed must be non-negative, got {seed_value}")
    repeat_value = int(_pick(repeat, options, "repeat", policy.DEFAULT_REPEAT))
    warmup_value = int(_pick(warmup, options, "warmup", policy.DEFAULT_WARMUP))
    if repeat_value < 1:
        raise ValueError(f"repeat must be >= 1, got {repeat_value}")
    if warmup_value < 1:
        raise ValueError(f"warmup must be >= 1, got {warmup_value}")
    mass_value, kappa_value = _resolve_mass_kappa(
        _pick(mass, options, "mass", None),
        _pick(kappa, options, "kappa", None))
    csw_value = float(_pick(csw, options, "csw", DEFAULT_CSW))
    if not math.isfinite(csw_value) or csw_value < 0.0:
        raise ValueError(
            f"csw must be finite and non-negative, got {csw_value}")
    sigma_value = float(
        _pick(gauge_sigma, options, "gauge_sigma", DEFAULT_GAUGE_SIGMA))
    if not math.isfinite(sigma_value) or sigma_value < 0.0:
        raise ValueError(
            f"gauge_sigma must be finite and non-negative, got {sigma_value}")
    source_value = policy.validate_source(
        _pick(source, options, "source", DEFAULT_SOURCE))
    gauge_value = policy.validate_gauge(
        _pick(gauge, options, "gauge", DEFAULT_GAUGE))
    site = _resolve_site(
        _pick(source_site, options, "source_site", DEFAULT_SOURCE_SITE),
        shape)
    spin = int(_pick(source_spin, options, "source_spin",
                     DEFAULT_SOURCE_SPIN))
    color = int(_pick(source_color, options, "source_color",
                      DEFAULT_SOURCE_COLOR))
    if not 0 <= spin < 4:
        raise ValueError(f"source_spin must be in [0, 4), got {spin}")
    if not 0 <= color < 3:
        raise ValueError(f"source_color must be in [0, 3), got {color}")
    dslash_name = policy.validate_dslash(
        _pick(dslash, options, "dslash", policy.DEFAULT_DSLASH))
    solver_name = policy.validate_solver(
        _pick(solver, options, "solver", policy.DEFAULT_SOLVER))
    device_value = _pick(device, options, "device", None)
    if device_value is not None:
        device_value = str(device_value).strip() or None
    device_id_value = _pick(device_id, options, "device_id", None)
    if device_id_value is not None:
        device_id_value = int(device_id_value)
        if device_id_value < 0:
            raise ValueError(
                f"device_id must be >= 0, got {device_id_value}")
    return _RunOptions(
        shape=shape,
        lattice_grid=grid,
        dtype=dtype_name,
        seed=seed_value,
        repeat=repeat_value,
        warmup=warmup_value,
        mass=mass_value,
        kappa=kappa_value,
        csw=csw_value,
        gauge_sigma=sigma_value,
        source=source_value,
        gauge=gauge_value,
        source_site=site,
        source_spin=spin,
        source_color=color,
        max_iter=_resolve_max_iter(
            _pick(max_iter, options, "max_iter", None)),
        atol=_resolve_atol(_pick(atol, options, "atol", None), dtype_name),
        device=device_value,
        device_id=device_id_value,
        dslash=dslash_name,
        solver=solver_name,
    )


def _require_dslash(options: _RunOptions, expected: str) -> None:
    """Reject a ``--dslash`` choice that contradicts the runner.

    ``all``/``both`` are the "no filter" spellings; naming the matching
    operator is accepted so one option set can drive the whole matrix.  Naming
    a different operator is a usage error: silently running the wrong operator
    would make the observation meaningless.
    """
    if options.dslash not in ("all", "both", expected):
        raise ValueError(
            f"this runner applies the {expected} operator, but "
            f"dslash={options.dslash!r} was requested")


def _require_solver(options: _RunOptions, expected: str) -> None:
    """Reject a ``--solver`` choice that contradicts the runner."""
    if options.solver not in ("all", expected):
        raise ValueError(
            f"this runner uses the {expected} solver, but "
            f"solver={options.solver!r} was requested")


# ---------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _CudaProbe:
    """Everything a runner needs once both prerequisites have passed."""

    torch: Any
    device: Any
    device_name: str
    capability: str
    bridge: Any
    bridge_file: str
    define: Any
    mpi_size: int
    mpi_rank: int
    globals_snapshot: dict[str, Any]


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        _skip(f"torch is not installed ({type(error).__name__}: {error})",
              _DEVICE_TOKEN)
    return torch


def _requested_device(device: Any, device_id: Any) -> str:
    """Resolve ``--device``/``--device-id``/$QCU_DEVICE to one device string."""
    if device is not None and str(device).strip():
        return str(device).strip()
    if device_id is not None:
        return f"cuda:{int(device_id)}"
    return os.environ.get(_DEVICE_ENVIRONMENT) or DEFAULT_CUDA_DEVICE


def _probe_device(device: Any = None, device_id: Any = None
                  ) -> tuple[Any, Any, str, str]:
    """Return ``(torch, torch.device, name, capability)`` or skip."""
    torch = _import_torch()
    try:
        available = bool(torch.cuda.is_available())
    except Exception as error:  # noqa: BLE001 - probe must stay a SKIP
        _skip(f"torch.cuda.is_available() raised "
              f"{type(error).__name__}: {error}", _DEVICE_TOKEN)
    if not available:
        _skip("torch.cuda.is_available() is False; the native CUDA bridge "
              "needs a visible device", _DEVICE_TOKEN)
    requested = _requested_device(device, device_id)
    try:
        selected = torch.device(str(requested))
    except (RuntimeError, ValueError) as error:
        _skip(f"device {requested!r} is not a usable CUDA device: {error}",
              _UNAVAILABLE_TOKEN)
    if selected.type != "cuda":
        _skip(f"device {requested!r} is a {selected.type!r} device; the CUDA "
              "backend requires a cuda device", _UNAVAILABLE_TOKEN)
    index = 0 if selected.index is None else int(selected.index)
    try:
        count = int(torch.cuda.device_count())
    except Exception as error:  # noqa: BLE001 - probe must stay a SKIP
        _skip(f"torch.cuda.device_count() raised "
              f"{type(error).__name__}: {error}", _DEVICE_TOKEN)
    if index < 0 or index >= count:
        _skip(f"device {index} is not visible; torch reports {count} "
              f"CUDA device(s)", _UNAVAILABLE_TOKEN)
    try:
        name = str(torch.cuda.get_device_name(index))
        capability = ".".join(
            str(value) for value in torch.cuda.get_device_capability(index))
    except Exception as error:  # noqa: BLE001 - probe must stay a SKIP
        _skip(f"cannot query CUDA device {index}: "
              f"{type(error).__name__}: {error}", _DEVICE_TOKEN)
    return torch, torch.device(f"cuda:{index}"), name, capability


def _probe_bridge() -> tuple[Any, str]:
    """Import ``pyqcu.cuda.qcu`` and require ``applyInitQcu``/``applyEndQcu``."""
    try:
        from pyqcu.cuda import qcu as bridge
    except ImportError as error:
        _skip(f"pyqcu.cuda.qcu is not importable: "
              f"{type(error).__name__}: {error}", _BRIDGE_TOKEN)
    module_file = getattr(bridge, "__file__", None)
    if module_file is None:
        _skip("pyqcu.cuda.qcu resolves to a namespace package "
              "(no compiled qcu*.so): the Cython bridge is not built in this "
              "checkout", _BRIDGE_TOKEN)
    _require_symbols(bridge, _SMOKE_SYMBOLS, "the init/end lifecycle")
    return bridge, str(module_file)


def _require_symbols(bridge: Any, names: tuple[str, ...],
                     purpose: str) -> None:
    """Skip ``cuda_bridge`` when the built bridge lacks a needed entry point."""
    missing = [name for name in names
               if not callable(getattr(bridge, name, None))]
    if missing:
        _skip(f"pyqcu.cuda.qcu does not export {', '.join(missing)} "
              f"(required for {purpose})", _BRIDGE_TOKEN)


def _import_define() -> Any:
    """Import the ABI constant module (``mpi4py`` + ``torch`` behind it)."""
    try:
        from pyqcu.cuda import define
    except ImportError as error:
        _skip(f"pyqcu.cuda.define is unavailable: "
              f"{type(error).__name__}: {error}", _BRIDGE_TOKEN)
    return define


def _global_snapshot(define: Any) -> dict[str, Any]:
    """Clone the module-level ABI control arrays (never mutated by a runner)."""
    snapshot: dict[str, Any] = {}
    for name in _GLOBAL_ARRAY_NAMES:
        value = getattr(define, name, None)
        clone = getattr(value, "clone", None)
        if not callable(clone):
            raise AttributeError(
                f"pyqcu.cuda.define.{name} is missing; the ABI contract "
                "changed")
        snapshot[name] = clone()
    return snapshot


def _changed_globals(probe: _CudaProbe) -> list[str]:
    """Names of ``pyqcu.cuda.define`` control arrays that changed."""
    changed: list[str] = []
    for name, before in probe.globals_snapshot.items():
        current = getattr(probe.define, name, None)
        if current is None or not probe.torch.equal(before, current):
            changed.append(f"pyqcu.cuda.define.{name}")
    return changed


def _require_cuda(device: Any = None, device_id: Any = None) -> _CudaProbe:
    """Probe device, bridge, define and MPI size; raise :class:`Skip` on absence.

    The device is checked first and the bridge second, so a machine without a
    GPU never imports the extension and an unbuilt checkout never allocates a
    tensor.  Only after every probe passes may a runner issue a native call.
    """
    torch, selected, name, capability = _probe_device(device, device_id)
    bridge, bridge_file = _probe_bridge()
    define = _import_define()
    try:
        mpi_size = int(define.size)
        mpi_rank = int(define.rank)
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError(
            "pyqcu.cuda.define must expose integer 'size' and 'rank' "
            f"({type(error).__name__}: {error})") from error
    if mpi_size != 1:
        _skip(f"native CUDA cases run on exactly one MPI rank; "
              f"MPI_COMM_WORLD size is {mpi_size}", _RANK_TOKEN)
    return _CudaProbe(
        torch=torch,
        device=selected,
        device_name=name,
        capability=capability,
        bridge=bridge,
        bridge_file=bridge_file,
        define=define,
        mpi_size=mpi_size,
        mpi_rank=mpi_rank,
        globals_snapshot=_global_snapshot(define),
    )


def _require_grid(probe: _CudaProbe, options: _RunOptions) -> None:
    """Skip ``mpi_grid`` when ``prod(lattice_grid)`` is not the MPI size."""
    size = int(grid_product(options.lattice_grid))
    if size != probe.mpi_size:
        _skip(f"lattice_grid {list(options.lattice_grid)} describes "
              f"{size} rank(s) but MPI_COMM_WORLD has {probe.mpi_size}",
              _GRID_TOKEN)


def probe_cuda(device: Any = None) -> dict[str, Any]:
    """Report CUDA availability without raising and without a native call.

    Returns the device/bridge/MPI facts plus ``ready`` and the skip reasons
    (``"<token>: <reason>"``).  A runner must still go through
    :func:`_require_cuda`, which raises the skip.
    """
    report: dict[str, Any] = {
        "backend": BACKEND,
        "device_available": False,
        "device": None,
        "device_name": None,
        "capability": None,
        "bridge_available": False,
        "bridge_file": None,
        "bridge_built": False,
        "mpi_size": None,
        "mpi_rank": None,
        "ready": False,
        "skip_reasons": [],
        "skip_tokens": [],
    }

    def _record(error: BaseException) -> None:
        reason = str(getattr(error, "reason", "") or error).strip()
        token = getattr(error, "token", None)
        if not isinstance(token, str) or not policy.is_skip_token(token):
            head, separator, tail = reason.partition(":")
            if separator and tail.strip() and policy.is_skip_token(head.strip()):
                token = head.strip()
                reason = tail.strip()
        text = f"{token}: {reason}" if isinstance(token, str) and reason \
            else (reason or "CUDA backend unavailable")
        report["skip_reasons"].append(text)
        if isinstance(token, str) and policy.is_skip_token(token):
            report["skip_tokens"].append(token)

    try:
        torch, selected, name, capability = _probe_device(device)
    except Skip as error:
        _record(error)
        return report
    except Exception as error:  # noqa: BLE001 - the probe must not raise
        _record(error)
        return report
    report.update(
        device_available=True,
        device=str(selected),
        device_name=name,
        capability=capability,
    )
    try:
        bridge, bridge_file = _probe_bridge()
    except Skip as error:
        _record(error)
        return report
    except Exception as error:  # noqa: BLE001 - the probe must not raise
        _record(error)
        return report
    report.update(
        bridge_available=True,
        bridge_file=bridge_file,
        bridge_built=True,
    )
    try:
        define = _import_define()
    except Skip as error:
        _record(error)
        return report
    except Exception as error:  # noqa: BLE001 - the probe must not raise
        _record(error)
        return report
    try:
        report["mpi_size"] = int(define.size)
        report["mpi_rank"] = int(define.rank)
    except (AttributeError, TypeError, ValueError) as error:
        _record(RuntimeError(
            f"pyqcu.cuda.define size/rank are invalid: {error}"))
        return report
    if report["mpi_size"] != 1:
        _record(Skip(f"{_RANK_TOKEN}: native CUDA cases run on exactly one "
                     f"MPI rank; MPI_COMM_WORLD size is "
                     f"{report['mpi_size']}", _RANK_TOKEN))
        return report
    report["ready"] = bool(report["device_available"]
                           and report["bridge_available"])
    return report


# ---------------------------------------------------------------------------
# private ABI runtime
# ---------------------------------------------------------------------------
def _torch_dtype(torch: Any, dtype_name: str) -> Any:
    if dtype_name == "c128":
        return torch.complex128
    if dtype_name == "c64":
        return torch.complex64
    raise ValueError(f"unsupported dtype {dtype_name!r}")


def _real_tensor_dtype(torch: Any, complex_dtype: Any) -> Any:
    return torch.float64 if complex_dtype == torch.complex128 else torch.float32


def _make_runtime(probe: _CudaProbe, options: _RunOptions, *,
                  plan: int, parity: int) -> Any:
    """Build one private :class:`QcuRuntime` for this case (never a global)."""
    from pyqcu.cuda._runtime import QcuRuntime

    return QcuRuntime.from_lattice(
        options.shape,
        mass=options.mass,
        csw=options.csw,
        atol=options.atol,
        sigma=options.gauge_sigma,
        dtype=_torch_dtype(probe.torch, options.dtype),
        device=str(probe.device),
        plan=plan,
        parity=parity,
        max_iter=options.max_iter,
        seed=options.seed,
        verbose=0,
        grid=options.lattice_grid,
        qcu_module=probe.bridge,
    )


def _controls(params: Any, argv: Any, define: Any) -> dict[str, Any]:
    """Snapshot the physics controls of one runtime (no ``_SET_INDEX_``)."""
    values: dict[str, Any] = {}
    for name in _CONTROL_PARAM_NAMES:
        values[f"params{name}"] = int(params[getattr(define, name)])
    for name in _CONTROL_ARGV_NAMES:
        values[f"argv{name}"] = float(argv[getattr(define, name)])
    return values


def _controls_applied(probe: _CudaProbe, runtime: Any,
                      options: _RunOptions) -> dict[str, bool]:
    """Prove that the private ABI arrays carry the requested run options.

    ``argv`` is float32 for c64, so an exact comparison against the Python
    float would reject every value that is not binary exact (``0.17``,
    ``0.1``, ``1e-6``).  The tolerance is one float32 epsilon band for c64 and
    a tight float64 band for c128.
    """
    define = probe.define
    tolerance = (1e-12 if runtime.argv.dtype == probe.torch.float64
                 else 1e-6)

    def _close(name: str, target: float) -> bool:
        actual = float(runtime.argv[getattr(define, name)])
        target_value = float(target)
        return math.isclose(
            actual, target_value, rel_tol=tolerance,
            abs_tol=tolerance * max(1.0, abs(target_value)))

    grid = tuple(int(runtime.params[getattr(define, name)]) for name in (
        "_GRID_X_", "_GRID_Y_", "_GRID_Z_", "_GRID_T_"))
    applied = {
        "mass": _close("_MASS_", options.mass),
        "csw": _close("_CSW_", options.csw),
        "gauge_sigma": _close("_SIGMA_", options.gauge_sigma),
        "atol": _close("_ATOL_", options.atol),
        "max_iter": int(runtime.params[define._MAX_ITER_])
        == int(options.max_iter),
        "lattice": tuple(int(value) for value in runtime.lattice)
        == tuple(options.shape),
        "lattice_grid": grid == tuple(options.lattice_grid),
        "device": str(runtime.device) == str(probe.device),
        "dtype": int(runtime.params[define._DATA_TYPE_])
        == int(define.epytd(_torch_dtype(probe.torch, options.dtype)).item()),
    }
    if not all(applied.values()):
        raise RuntimeError(
            "the private QCU controls do not match the run options: "
            f"{applied}")
    return applied


def _set_index(runtime: Any, define: Any) -> int:
    return int(runtime.params[define._SET_INDEX_])


def _slot_pointer(runtime: Any, index: int) -> int:
    return int(runtime.set_ptrs[int(index)])


def _synchronize(probe: _CudaProbe) -> None:
    """Block until the device queue is empty (no-op on a non-CUDA device)."""
    if probe.device.type == "cuda":
        probe.torch.cuda.synchronize(probe.device)


def _slot_values(runtime: Any, limit: int = SLOT_LIMIT) -> list[int]:
    """Values of the ordinary scratch window ``set_ptrs[0:limit)``."""
    count = min(int(limit), int(runtime.set_ptrs.numel()))
    return [int(runtime.set_ptrs[index]) for index in range(count)]


def _slots_released(runtime: Any, limit: int = SLOT_LIMIT) -> bool:
    return all(value == 0 for value in _slot_values(runtime, limit))


def _neighbour_slots_free(runtime: Any, active: int,
                          limit: int = SLOT_LIMIT) -> bool:
    values = _slot_values(runtime, limit)
    return all(value == 0 for index, value in enumerate(values)
               if index != int(active))


def _end_quietly(bridge: Any, runtime: Any) -> bool:
    """Release a slot from an error path without masking the original error."""
    try:
        bridge.applyEndQcu(runtime.set_ptrs, runtime.params)
        return True
    except Exception:  # noqa: BLE001 - teardown must not mask a failure
        return False


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------
def _measure(call: Callable[[], Any], *, repeat: int,
             warmup: int) -> tuple[float, Any]:
    """Run ``call`` ``warmup`` + ``repeat`` times; return median and last value."""
    last: list[Any] = [None]

    def _once() -> None:
        last[0] = call()

    samples = time_samples(_once, repeat=int(repeat), warmup=int(warmup))
    return float(statistics.median(samples)), last[0]


# ---------------------------------------------------------------------------
# C-level stdout capture (native CG iteration record)
# ---------------------------------------------------------------------------
@contextmanager
def _captured_stdout() -> Iterator[dict[str, Any]]:
    """Capture file descriptor 1 for the enclosed native call.

    The CUDA CG solver reports its final loop index only as a C++ ``printf``
    (``##RANK:<r>##LOOP:<n>##Residual(norm2):...``); there is no ABI getter.
    Redirecting fd 1 for the duration of one synchronous call is the only way
    to read that record without editing the core C++ or guessing a count.  The
    captured payload is one short line (the verbosity switch is off), so the
    pipe cannot fill; on a platform without ``dup2`` the sink stays empty and
    the caller reports an ERROR rather than a fabricated iteration count.
    """
    sink: dict[str, Any] = {"data": b""}
    try:
        sys.stdout.flush()
        saved = os.dup(1)
        read_fd, write_fd = os.pipe()
    except (AttributeError, OSError):
        yield sink
        return
    try:
        os.dup2(write_fd, 1)
    except OSError:
        os.close(write_fd)
        os.close(read_fd)
        os.close(saved)
        yield sink
        return
    os.close(write_fd)
    try:
        yield sink
    finally:
        try:
            sys.stdout.flush()
        except (OSError, ValueError):
            pass
        try:
            os.dup2(saved, 1)
        finally:
            os.close(saved)
        chunks: list[bytes] = []
        while True:
            try:
                data = os.read(read_fd, 65536)
            except OSError:
                break
            if not data:
                break
            chunks.append(data)
        os.close(read_fd)
        sink["data"] = b"".join(chunks)


def _native_loop_count(sink: dict[str, Any]) -> int:
    """Parse the native CG loop record into an iteration count."""
    data = sink.get("data") or b""
    text = data.decode("utf-8", "replace")
    loops = [int(value) for value in _NATIVE_LOOP.findall(text)]
    if not loops:
        raise RuntimeError(
            "the native CUDA CG solver produced no ##LOOP iteration record; "
            f"captured stdout={text[:400]!r}")
    return max(loops) + 1


# ---------------------------------------------------------------------------
# deterministic inputs (random but fixed gauge/source)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Inputs:
    """Gauge and source in both Python layouts."""

    full_gauge: Any
    gauge: Any
    source: Any
    gauge_su3: bool | None
    gauge_seed: int | None
    source_seed: int | None
    gauge_kind: str
    source_kind: str
    source_site: tuple[int, int, int, int]
    source_spin: int
    source_color: int


@contextmanager
def _preserved_rng(torch: Any) -> Iterator[None]:
    """Restore the global CPU/CUDA RNG state after a seeded generation.

    ``pyqcu.lattice.generate_gauge_field`` calls ``torch.manual_seed`` and
    draws from the global generator.  A test runner must not advance the
    process-wide random stream, so both states are saved and restored.
    """
    cpu_state = torch.random.get_rng_state()
    cuda_states = None
    if torch.cuda.is_available():
        try:
            cuda_states = torch.cuda.get_rng_state_all()
        except Exception:  # noqa: BLE001 - best effort, never fatal
            cuda_states = None
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            try:
                torch.cuda.set_rng_state_all(cuda_states)
            except Exception:  # noqa: BLE001 - best effort, never fatal
                pass


def _build_gauge(probe: _CudaProbe, options: _RunOptions
                 ) -> tuple[Any, Any, bool | None, int | None]:
    """Build the full and parity-compact gauge fields for ``options.gauge``.

    ``random`` uses the project generator with an explicit seed and a disorder
    of ``gauge_sigma`` (CPU generation, then a copy, so the value does not
    depend on the device RNG); ``unit`` builds identity links; ``ones`` fills
    every link with one (a deliberately non-SU(3) stress input, reported by
    ``gauge_su3=False`` and never gated).
    """
    torch = probe.torch
    from pyqcu import lattice as lattice_module
    from pyqcu import tools

    shape = options.shape
    dtype = _torch_dtype(torch, options.dtype)
    full_gauge = torch.empty((3, 3, 4, *shape), dtype=dtype, device="cpu")
    gauge_seed: int | None = None
    if options.gauge == "unit":
        index = torch.arange(3)
        full_gauge.zero_()
        full_gauge[index, index, :] = 1.0 + 0.0j
    elif options.gauge == "ones":
        full_gauge.fill_(1.0 + 0.0j)
    else:
        gauge_seed = int(options.seed)
        with _preserved_rng(torch):
            lattice_module.generate_gauge_field(
                full_gauge, sigma=float(options.gauge_sigma),
                seed=gauge_seed, verbose=False)
    full_gauge = full_gauge.contiguous()
    try:
        gauge_su3: bool | None = bool(
            lattice_module.check_su3(full_gauge, tol=1e-3, verbose=False))
    except BaseException:  # noqa: BLE001 - diagnostics only
        gauge_su3 = None
    device_gauge = full_gauge.to(device=probe.device).contiguous()
    gauge = tools.oooxyzt2poooxyzt(device_gauge).contiguous()
    return device_gauge, gauge, gauge_su3, gauge_seed


def _build_source(probe: _CudaProbe, options: _RunOptions
                  ) -> tuple[Any, int | None]:
    """Build the two-parity compact source ``[2, 4, 3, X, Y, Z, T/2]``.

    ``random`` uses a private CPU generator (reproducible and it does not
    consume the process-wide stream); ``point`` places a unit charge at
    ``(spin, color, site)``; ``ones`` fills every component with one.
    """
    torch = probe.torch
    dtype = _torch_dtype(torch, options.dtype)
    full_shape = (2, 4, 3, *options.compact)
    source_seed: int | None = None
    if options.source == "point":
        source = torch.zeros(full_shape, dtype=dtype, device="cpu")
        site = (0, options.source_spin, options.source_color,
                *options.source_site)
        source[site] = 1.0 + 0.0j
    elif options.source == "ones":
        source = torch.ones(full_shape, dtype=dtype, device="cpu")
    else:
        source_seed = int(options.seed) + 1
        generator = torch.Generator(device="cpu").manual_seed(source_seed)
        source = torch.randn(
            full_shape, generator=generator, dtype=dtype, device="cpu")
    return source.to(device=probe.device).contiguous(), source_seed


def _make_inputs(probe: _CudaProbe, options: _RunOptions) -> _Inputs:
    """Build the fixed gauge field and source described by ``options``."""
    full_gauge, gauge, gauge_su3, gauge_seed = _build_gauge(probe, options)
    source, source_seed = _build_source(probe, options)
    return _Inputs(
        full_gauge=full_gauge,
        gauge=gauge,
        source=source,
        gauge_su3=gauge_su3,
        gauge_seed=gauge_seed,
        source_seed=source_seed,
        gauge_kind=options.gauge,
        source_kind=options.source,
        source_site=options.source_site,
        source_spin=options.source_spin,
        source_color=options.source_color,
    )


def _source_operand(inputs: _Inputs, target: int) -> Any:
    """Compact operand fed to a dslash target parity (the opposite block)."""
    return inputs.source[1 - target].contiguous()


# ---------------------------------------------------------------------------
# torch references (independent of the native recurrence)
# ---------------------------------------------------------------------------
def _compact_wilson_reference(torch: Any, compact_source: Any, gauge: Any,
                              parity: int) -> Any:
    """``-kappa * hopping`` reference on the compact parity layout.

    ``applyWilsonDslashQcu`` applies the bare hopping sum (no ``I`` term and no
    ``-kappa`` factor) from the opposite compact parity; ``give_wilson_eo``/
    ``give_wilson_oe`` return the negated hopping sum at ``kappa = 1``.
    """
    from pyqcu.torch import dslash

    kappa_one = torch.tensor([1.0])
    if parity == 0:
        return -dslash.give_wilson_eo(
            src_o=compact_source, U_eo=gauge, kappa=kappa_one)
    return -dslash.give_wilson_oe(
        src_e=compact_source, U_eo=gauge, kappa=kappa_one)


def _full_lattice_wilson_reference(probe: _CudaProbe, inputs: _Inputs,
                                   compact_source: Any, parity: int,
                                   kappa: float) -> Any:
    """Bare hopping implied by the full-lattice Wilson operator.

    The compact source is embedded in the opposite parity of a full field and
    the full Wilson operator is applied with the run's ``kappa``.  On the
    target parity the field is zero, so
    ``hopping(x_other) = -D_W x|_target / kappa``.  This exercises
    ``mass``/``kappa`` numerically instead of hiding them behind a hard-coded
    ``kappa = 1`` reference.
    """
    from pyqcu import tools
    from pyqcu.torch import dslash

    torch = probe.torch
    compact = inputs.source.shape[-4:]
    full_source = torch.zeros(
        (2, 4, 3, *compact), dtype=compact_source.dtype,
        device=compact_source.device)
    full_source[1 - parity] = compact_source
    full_field = tools.poooxyzt2oooxyzt(full_source)
    kappa_tensor = torch.tensor(
        [float(kappa)],
        dtype=_real_tensor_dtype(torch, compact_source.dtype))
    applied = dslash.give_wilson(
        full_field, inputs.full_gauge, kappa_tensor, with_I=True)
    target_block = tools.oooxyzt2poooxyzt(applied)[parity]
    return (-target_block / float(kappa)).contiguous()


def _relative_l2(actual: Any, expected: Any) -> float:
    """``||actual - expected|| / ||expected||`` (diagnostic cross-check)."""
    import torch

    left = actual.detach().to("cpu", torch.complex128)
    right = expected.detach().to("cpu", torch.complex128)
    denominator = float(torch.linalg.vector_norm(right))
    difference = float(torch.linalg.vector_norm(left - right))
    if denominator == 0.0:
        return difference
    return difference / denominator


def _native_clover_terms(probe: _CudaProbe, runtime: Any,
                         options: _RunOptions, gauge: Any
                         ) -> tuple[list[Any], list[Any]]:
    """Build the native clover term and its inverse for both parities.

    One ``applyCloversQcu`` transaction per parity; the parity is a control
    read by ``applyInitQcu``, so it is written before the transaction opens
    and left untouched inside it.  The runtime's previous plan is restored.
    """
    torch = probe.torch
    define = probe.define
    shape = (4, 3, 4, 3, *options.compact)
    previous_plan = int(runtime.params[define._SET_PLAN_])
    runtime.configure(plan=define._SET_PLAN2_)
    terms: list[Any] = []
    inverses: list[Any] = []
    try:
        for parity in (0, 1):
            term = torch.empty(shape, dtype=gauge.dtype,
                               device=gauge.device)
            inverse = torch.empty_like(term)
            runtime.params[define._PARITY_] = parity
            runtime.call("applyCloversQcu", term, inverse, gauge)
            terms.append(term)
            inverses.append(inverse)
    finally:
        runtime.configure(plan=previous_plan)
    return terms, inverses


def _torch_clover_terms(probe: _CudaProbe, inputs: _Inputs, kappa: float,
                        csw: float) -> dict[str, Any]:
    """Torch-side clover term ``I + T`` and its inverse, both parities.

    Returned in the ABI layout ``[4, 3, 4, 3, X, Y, Z, T/2]`` per parity so a
    caller can compare the native ``applyCloversQcu`` output directly.  This is
    the independent convention: it comes from ``pyqcu.torch.dslash`` only.
    """
    torch = probe.torch
    from pyqcu import tools
    from pyqcu.torch import dslash

    shape = tuple(int(value) for value in inputs.full_gauge.shape[-4:])
    compact = (*shape[:3], shape[3] // 2)
    kappa_tensor = torch.tensor(
        [float(kappa)], dtype=_real_tensor_dtype(torch, inputs.full_gauge.dtype))
    term = dslash.add_I(
        dslash.make_clover(inputs.full_gauge, kappa=kappa_tensor,
                           csw=float(csw), verbose=False),
        verbose=False)
    inverse = dslash.inverse(term, verbose=False)
    split_term = tools.oooxyzt2poooxyzt(term.reshape(12, 12, *shape))
    split_inverse = tools.oooxyzt2poooxyzt(inverse.reshape(12, 12, *shape))
    terms = [split_term[parity].reshape(4, 3, 4, 3, *compact).contiguous()
             for parity in (0, 1)]
    inverses = [split_inverse[parity].reshape(4, 3, 4, 3, *compact)
                .contiguous() for parity in (0, 1)]
    return {
        "term": terms,
        "inverse": inverses,
        "term_abs_max": max(float(value.abs().max()) for value in terms),
        "inverse_abs_max": max(float(value.abs().max()) for value in inverses),
    }


def _clover_term_comparison(probe: _CudaProbe, native_terms: list[Any],
                            native_inverses: list[Any],
                            torch_terms: dict[str, Any]) -> dict[str, Any]:
    """Compare native and torch clover terms and their inverse identity."""
    torch = probe.torch
    term_relative = max(
        relative_error(native_terms[parity], torch_terms["term"][parity])
        for parity in (0, 1))
    inverse_relative = max(
        relative_error(native_inverses[parity],
                       torch_terms["inverse"][parity])
        for parity in (0, 1))
    identity = torch.zeros(12, 12, dtype=torch_terms["term"][0].dtype,
                           device=torch_terms["term"][0].device)
    index = torch.arange(12, device=identity.device)
    identity[index, index] = 1.0
    identity_max_abs = 0.0
    for parity in (0, 1):
        product = torch.einsum(
            "ijx,jkx->ikx",
            native_terms[parity].reshape(12, 12, -1),
            native_inverses[parity].reshape(12, 12, -1))
        reference = identity.reshape(12, 12, 1).expand_as(product)
        identity_max_abs = max(
            identity_max_abs, float((product - reference).abs().max()))
    return {
        "clover_term_relative": term_relative,
        "clover_inverse_relative": inverse_relative,
        "clover_identity_max_abs": identity_max_abs,
    }


def _full_clover_operator(probe: _CudaProbe, inputs: _Inputs, kappa: float,
                          csw: float) -> tuple[Any, Any, Any]:
    """Return ``(clover_term, kappa_tensor, matvec)`` for the full operator.

    ``D = A - kappa * hop`` with ``A = I + csw*T``;
    ``give_wilson(with_I=False)`` is ``-kappa * hop``, so
    ``D x = give_clover(x, A) + give_wilson(x, with_I=False)``.
    """
    torch = probe.torch
    from pyqcu.torch import dslash

    kappa_tensor = torch.tensor(
        [float(kappa)], dtype=_real_tensor_dtype(torch, inputs.full_gauge.dtype))
    clover_term = dslash.add_I(
        dslash.make_clover(inputs.full_gauge, kappa=kappa_tensor,
                           csw=float(csw), verbose=False),
        verbose=False)

    def _matvec(vector: Any) -> Any:
        return (dslash.give_clover(src=vector, clover_term=clover_term,
                                   verbose=False)
                + dslash.give_wilson(vector, inputs.full_gauge, kappa_tensor,
                                     with_I=False))

    return clover_term, kappa_tensor, _matvec


# ---------------------------------------------------------------------------
# shared observation blocks
# ---------------------------------------------------------------------------
def _common_observations(probe: _CudaProbe, options: _RunOptions,
                         runtime: Any, *, check: str,
                         inputs: _Inputs | None = None,
                         source_parity: Any = None) -> dict[str, Any]:
    """Observation block shared by every CUDA runner."""
    observations: dict[str, Any] = {
        "backend": BACKEND,
        "check": check,
        "device": str(probe.device),
        "device_requested": options.device,
        "device_id": options.device_id,
        "device_name": probe.device_name,
        "capability": probe.capability,
        "lattice": list(options.shape),
        "lattice_grid": list(options.lattice_grid),
        "dtype": options.dtype,
        "seed": int(options.seed),
        "mass": float(options.mass),
        "kappa": float(runtime.kappa),
        "csw": float(options.csw),
        "gauge_sigma": float(options.gauge_sigma),
        "max_iter": int(options.max_iter),
        "atol": float(options.atol),
        "source": (inputs.source_kind if inputs is not None else None),
        "source_kind": (inputs.source_kind if inputs is not None else None),
        "source_seed": (inputs.source_seed if inputs is not None else None),
        "source_parity": source_parity,
        "gauge": (inputs.gauge_kind if inputs is not None else None),
        "gauge_kind": (inputs.gauge_kind if inputs is not None else None),
        "gauge_seed": (inputs.gauge_seed if inputs is not None else None),
        "gauge_su3": (inputs.gauge_su3 if inputs is not None else None),
        "mpi_size": int(probe.mpi_size),
        "mpi_rank": int(probe.mpi_rank),
        "bridge_file": probe.bridge_file,
        "bridge_built": True,
        "dslash_option": options.dslash,
        "solver_option": options.solver,
        "repeat": int(options.repeat),
        "warmup": int(options.warmup),
        "controls_applied": _controls_applied(probe, runtime, options),
    }
    if inputs is not None:
        observations["source_spec"] = {
            "kind": inputs.source_kind,
            "parity": source_parity,
            "spin": int(inputs.source_spin),
            "color": int(inputs.source_color),
            "site": list(inputs.source_site),
            "seed": inputs.source_seed,
        }
    return observations


def _finish(observations: dict[str, Any], probe: _CudaProbe, runtime: Any,
            *, seconds: float, prepare_seconds: float,
            immutability: dict[str, bool]) -> dict[str, Any]:
    """Add timing, lifecycle and global-state evidence, then return."""
    changed = _changed_globals(probe)
    observations.update(
        seconds=float(seconds),
        prepare_seconds=float(prepare_seconds),
        slots_released=_slots_released(runtime),
        params_index_after=_set_index(runtime, probe.define),
        globals_unchanged=not changed,
        changed_global_arrays=changed,
    )
    observations.update(immutability)
    if immutability:
        observations["inputs_unchanged"] = all(
            bool(value) for value in immutability.values())
    return observations


# ---------------------------------------------------------------------------
# runner: bridge/device init/end smoke test
# ---------------------------------------------------------------------------
def run_smoke_init_end(lattice: Any = None, lattice_grid: Any = None,
                       dtype: Any = None, seed: Any = None,
                       repeat: Any = None, warmup: Any = None,
                       atol: Any = None, source: Any = None,
                       gauge: Any = None, gauge_sigma: Any = None,
                       mass: Any = None, kappa: Any = None, csw: Any = None,
                       solver: Any = None, dslash: Any = None,
                       device: Any = None, max_iter: Any = None,
                       device_id: Any = None, source_site: Any = None,
                       source_spin: Any = None, source_color: Any = None,
                       options: Any = None) -> dict[str, Any]:
    """Exercise ``applyInitQcu``/``applyEndQcu`` on private slot tables.

    Observations: ``init_ok`` (the gate), ``slot_isolation``, ``session_ok``,
    the two allocated slot indices, the release/neighbour facts, the
    ``globals_unchanged`` evidence and the timing.  No gauge or fermion field
    is touched, so this case isolates the lifecycle from the numerics.
    """
    opts = _options(
        lattice=lattice, lattice_grid=lattice_grid, dtype=dtype, seed=seed,
        repeat=repeat, warmup=warmup, atol=atol, source=source, gauge=gauge,
        gauge_sigma=gauge_sigma, mass=mass, kappa=kappa, csw=csw,
        solver=solver, dslash=dslash, device=device, max_iter=max_iter,
        device_id=device_id, source_site=source_site,
        source_spin=source_spin, source_color=source_color, options=options)
    probe = _require_cuda(opts.device, opts.device_id)
    _require_grid(probe, opts)
    _require_symbols(probe.bridge, _SMOKE_SYMBOLS, "the init/end smoke case")
    started = time.perf_counter()
    main = _make_runtime(probe, opts, plan=0, parity=0)
    partner = _make_runtime(probe, opts, plan=0, parity=0)
    prepare_seconds = time.perf_counter() - started
    define = probe.define
    bridge = probe.bridge

    def _sequence() -> dict[str, Any]:
        main_before = _controls(main.params, main.argv, define)
        partner_before = _controls(partner.params, partner.argv, define)
        # Explicit slot selection: the main table uses slot 0, the partner
        # deliberately slot 1, so the selector is exercised and both peers must
        # keep their own neighbours untouched.
        main.params[define._SET_INDEX_] = 0
        partner.params[define._SET_INDEX_] = 1
        main_slot = _set_index(main, define)
        partner_slot = _set_index(partner, define)
        main_live = False
        partner_live = False
        try:
            bridge.applyInitQcu(main.set_ptrs, main.params, main.argv)
            main_live = True
            bridge.applyInitQcu(partner.set_ptrs, partner.params,
                                partner.argv)
            partner_live = True
            main_pointer = _slot_pointer(main, main_slot)
            partner_pointer = _slot_pointer(partner, partner_slot)
            partner_slot0_free = _slot_pointer(partner, 0) == 0
            distinct_pointers = (main_pointer != 0
                                 and main_pointer != partner_pointer)
            # Ending the first slot must not disturb the second table.
            bridge.applyEndQcu(main.set_ptrs, main.params)
            main_live = False
            main_released = _slot_pointer(main, main_slot) == 0
            partner_survives = _slot_pointer(partner, partner_slot) \
                == partner_pointer
            neighbour_free = _neighbour_slots_free(partner, partner_slot)
            bridge.applyEndQcu(partner.set_ptrs, partner.params)
            partner_live = False
        finally:
            if partner_live:
                _end_quietly(bridge, partner)
            if main_live:
                _end_quietly(bridge, main)
        partner_released = _slot_pointer(partner, partner_slot) == 0
        # The wrapper's own transaction must round-trip as well.
        with main.session() as session:
            session_slot = int(session.set_index)
            session_pointer = _slot_pointer(main, session_slot)
        session_released = _slot_pointer(main, session_slot) == 0
        session_ok = bool(session_pointer != 0 and session_released)
        controls_unchanged = (_controls(main.params, main.argv, define)
                              == main_before
                              and _controls(partner.params, partner.argv,
                                            define) == partner_before)
        changed = _changed_globals(probe)
        slots_released = (_slots_released(main)
                          and _slots_released(partner))
        init_ok = bool(
            main_pointer != 0
            and partner_pointer != 0
            and distinct_pointers
            and main_slot == 0
            and partner_slot == 1
            and partner_slot0_free
            and main_released
            and partner_released
            and partner_survives
            and neighbour_free
            and session_ok
            and controls_unchanged
            and slots_released
            and not changed
        )
        return {
            "init_ok": init_ok,
            "slot_isolation": bool(distinct_pointers and partner_survives
                                   and neighbour_free and partner_slot0_free),
            "session_ok": session_ok,
            "slots": [int(main_slot), int(partner_slot)],
            "slot_pointer_nonzero": bool(main_pointer != 0
                                         and partner_pointer != 0),
            "slot_pointer_distinct": bool(distinct_pointers),
            "slot_released_after_end": bool(main_released
                                            and partner_released),
            "neighbour_slots_free": bool(neighbour_free),
            "partner_survives_peer_end": bool(partner_survives),
            "session_slot": int(session_slot),
            "set_ptrs_0_30": _slot_values(main),
            "slots_released": bool(slots_released),
            "controls_unchanged": bool(controls_unchanged),
            "globals_unchanged": not changed,
            "changed_global_arrays": changed,
            "params_index_after": _set_index(main, define),
        }

    seconds, observation = _measure(
        _sequence, repeat=opts.repeat, warmup=opts.warmup)
    if not isinstance(observation, dict):
        raise RuntimeError("init/end smoke transaction returned no observation")
    observations = _common_observations(
        probe, opts, main, check="smoke_init_end")
    observations.update(observation)
    return _finish(observations, probe, main, seconds=seconds,
                   prepare_seconds=prepare_seconds, immutability={})


# ---------------------------------------------------------------------------
# runner: Wilson dslash against the torch reference
# ---------------------------------------------------------------------------
def run_dslash_wilson(lattice: Any = None, lattice_grid: Any = None,
                      dtype: Any = None, seed: Any = None,
                      repeat: Any = None, warmup: Any = None,
                      atol: Any = None, source: Any = None,
                      gauge: Any = None, gauge_sigma: Any = None,
                      mass: Any = None, kappa: Any = None, csw: Any = None,
                      solver: Any = None, dslash: Any = None,
                      device: Any = None, max_iter: Any = None,
                      device_id: Any = None, source_site: Any = None,
                      source_spin: Any = None, source_color: Any = None,
                      options: Any = None) -> dict[str, Any]:
    """Apply ``applyWilsonDslashQcu`` to both parity targets and compare.

    The input is the opposite compact parity of the configured source and the
    gauge field is the configured field; the reference is computed by torch
    (the compact even/odd helper *and* the full-lattice Wilson operator with
    the run's ``kappa``, which is the independent path).  ``mass``/``kappa``
    reach ``argv[_MASS_]`` and the full-Wilson reference; the bare hopping
    symbol itself is mass- and clover-independent, and that is reported rather
    than hidden.  Observations: ``relative_error`` (gate alias ``relative``),
    ``relative_error_full_wilson``, ``reference_spread``, input immutability,
    lifecycle release and timing.
    """
    opts = _options(
        lattice=lattice, lattice_grid=lattice_grid, dtype=dtype, seed=seed,
        repeat=repeat, warmup=warmup, atol=atol, source=source, gauge=gauge,
        gauge_sigma=gauge_sigma, mass=mass, kappa=kappa, csw=csw,
        solver=solver, dslash=dslash, device=device, max_iter=max_iter,
        device_id=device_id, source_site=source_site,
        source_spin=source_spin, source_color=source_color, options=options)
    _require_dslash(opts, "wilson")
    probe = _require_cuda(opts.device, opts.device_id)
    _require_grid(probe, opts)
    _require_symbols(probe.bridge, _DSLASH_SYMBOLS, "the Wilson dslash case")
    started = time.perf_counter()
    torch = probe.torch
    define = probe.define
    inputs = _make_inputs(probe, opts)
    operands = {target: _source_operand(inputs, target)
                for target in (0, 1)}
    parity = 1
    runtime = _make_runtime(probe, opts, plan=0, parity=parity)
    outputs = {target: torch.zeros_like(operands[target])
               for target in (0, 1)}
    source_before = inputs.source.clone()
    gauge_before = inputs.gauge.clone()
    controls_before = _controls(runtime.params, runtime.argv, define)
    prepare_seconds = time.perf_counter() - started

    def _apply_all() -> None:
        for target in (0, 1):
            runtime.params[define._PARITY_] = target
            runtime.call("applyWilsonDslashQcu", outputs[target],
                         operands[target], inputs.gauge)

    seconds, _ = _measure(
        _apply_all, repeat=opts.repeat, warmup=opts.warmup)
    runtime.configure(parity=parity)
    errors: dict[int, float] = {}
    full_errors: dict[int, float] = {}
    compact_references: dict[int, Any] = {}
    full_references: dict[int, Any] = {}
    reference_norm = 0.0
    output_norm = 0.0
    for target in (0, 1):
        compact_references[target] = _compact_wilson_reference(
            torch, operands[target], inputs.gauge, target)
        full_references[target] = _full_lattice_wilson_reference(
            probe, inputs, operands[target], target, runtime.kappa)
        errors[target] = relative_error(
            outputs[target], compact_references[target])
        full_errors[target] = relative_error(
            outputs[target], full_references[target])
        reference_norm = max(
            reference_norm,
            float(torch.linalg.vector_norm(compact_references[target])))
        output_norm = max(output_norm,
                          float(torch.linalg.vector_norm(outputs[target])))
    relative = max(errors.values())
    reference_spread = max(
        relative_error(compact_references[target], full_references[target])
        for target in (0, 1))
    observations = _common_observations(
        probe, opts, runtime, check="dslash_wilson", inputs=inputs,
        source_parity={"even_target": 1, "odd_target": 0})
    observations.update(
        equation="wilson_hopping",
        abi_symbol="applyWilsonDslashQcu",
        csw_used_by_operator=False,
        mass_used_by_operator=False,
        mass_used_by_reference=True,
        relative=relative,
        relative_error=relative,
        relative_error_even_target=errors[0],
        relative_error_odd_target=errors[1],
        relative_error_full_wilson=max(full_errors.values()),
        relative_error_full_wilson_even=full_errors[0],
        relative_error_full_wilson_odd=full_errors[1],
        true_residual=relative,
        output_norm=output_norm,
        reference_norm=reference_norm,
        reference_spread=reference_spread,
        output_finite=bool(torch.isfinite(outputs[0]).all()
                           and torch.isfinite(outputs[1]).all()),
        notes=[
            "applyWilsonDslashQcu applies the bare hopping sum: mass/kappa "
            "reach argv[_MASS_] and the full-Wilson reference, csw only "
            "reaches argv[_CSW_]",
        ],
    )
    return _finish(
        observations, probe, runtime, seconds=seconds,
        prepare_seconds=prepare_seconds, immutability={
            "gauge_unchanged": bool(torch.equal(inputs.gauge, gauge_before)),
            "source_unchanged": bool(torch.equal(inputs.source, source_before)),
            "controls_unchanged": (
                _controls(runtime.params, runtime.argv, define)
                == controls_before),
        })


# ---------------------------------------------------------------------------
# runner: clover dslash against a torch reference
# ---------------------------------------------------------------------------
def run_solve_bistabcg(lattice: Any = None, lattice_grid: Any = None,
                       dtype: Any = None, seed: Any = None,
                       repeat: Any = None, warmup: Any = None,
                       atol: Any = None, source: Any = None,
                       gauge: Any = None, gauge_sigma: Any = None,
                       mass: Any = None, kappa: Any = None, csw: Any = None,
                       solver: Any = None, dslash: Any = None,
                       device: Any = None, max_iter: Any = None,
                       device_id: Any = None, source_site: Any = None,
                       source_spin: Any = None, source_color: Any = None,
                       options: Any = None) -> dict[str, Any]:
    """Run one full native BiCGStab solve and reconstruct its true residual.

    The native equation is the Clover operator at the configured ``csw``
    (``csw = 0`` is the Wilson operator).  ``applyCloverBistabCgQcu`` reports
    a native iteration count through ``getCloverBistabCgIterationsQcu`` and
    interprets ``argv[_ATOL_]`` as a relative tolerance.  ``relative_residual``
    is recomputed from the full clover/Wilson operator applied to the returned
    solution, never from the solver recurrence.  Gate observations:
    ``relative_residual``, ``iterations``, ``max_iter``.
    """
    opts = _options(
        lattice=lattice, lattice_grid=lattice_grid, dtype=dtype, seed=seed,
        repeat=repeat, warmup=warmup, atol=atol, source=source, gauge=gauge,
        gauge_sigma=gauge_sigma, mass=mass, kappa=kappa, csw=csw,
        solver=solver, dslash=dslash, device=device, max_iter=max_iter,
        device_id=device_id, source_site=source_site,
        source_spin=source_spin, source_color=source_color, options=options)
    _require_solver(opts, "bicgstab")
    probe = _require_cuda(opts.device, opts.device_id)
    _require_grid(probe, opts)
    _require_symbols(probe.bridge, _SOLVE_SYMBOLS, "the native solve case")
    started = time.perf_counter()
    torch = probe.torch
    define = probe.define
    from pyqcu import tools
    from pyqcu.api import BiCGStabControl
    from pyqcu.cuda.linear_operator import CudaLinearOperator
    from pyqcu.solver.krylov import Krylov

    inputs = _make_inputs(probe, opts)
    parity = 1
    runtime = _make_runtime(probe, opts, plan=1, parity=parity)
    gauge_before = inputs.gauge.clone()
    source_before = inputs.source.clone()
    clover_even, clover_even_inv = runtime.make_clover(
        inputs.gauge, with_inverse=True, parity=0)
    clover_odd, clover_odd_inv = runtime.make_clover(
        inputs.gauge, with_inverse=True, parity=1)
    clover_abs_max = max(
        float(clover_even.abs().max()), float(clover_odd.abs().max()))
    clover_inv_abs_max = max(
        float(clover_even_inv.abs().max()), float(clover_odd_inv.abs().max()))
    operator = CudaLinearOperator(
        runtime=runtime, gauge=inputs.gauge, parity=parity,
        clover=(clover_even, clover_odd, clover_even_inv, clover_odd_inv))
    control = BiCGStabControl(rtol=opts.atol, max_iter=opts.max_iter)
    solver = Krylov(
        backend=BACKEND, method="bicgstab", operator=operator,
        control=control)
    # ``make_clover`` leaves plan 2 in the private controls; the solve driver
    # selects plan 1, so snapshot after restoring the plan the solve uses.
    runtime.configure(plan=define._SET_PLAN1_, parity=parity)
    controls_before = _controls(runtime.params, runtime.argv, define)
    prepare_seconds = time.perf_counter() - started

    def _solve() -> Any:
        result = solver.solve_result(inputs.source)
        _synchronize(probe)
        return result

    seconds, result = _measure(
        _solve, repeat=opts.repeat, warmup=opts.warmup)
    if result is None:
        raise RuntimeError("native CUDA solve returned no result")
    clover_full, _, matvec = _full_clover_operator(
        probe, inputs, runtime.kappa, opts.csw)
    full_source = tools.poooxyzt2oooxyzt(inputs.source)
    full_solution = tools.poooxyzt2oooxyzt(result.x)
    true_residual = relative_residual(matvec, full_solution, full_source)
    iterations = (None if result.krylov_iterations is None
                  else int(result.krylov_iterations))
    status = getattr(result.status, "value", str(result.status))
    scope = getattr(result.residual_scope, "value",
                    str(result.residual_scope))
    observations = _common_observations(
        probe, opts, runtime, check="solve_bistabcg", inputs=inputs,
        source_parity=parity)
    observations.update(
        equation=("clover" if opts.csw != 0.0 else "wilson"),
        abi_symbol="applyCloverBistabCgQcu",
        iteration_getter="getCloverBistabCgIterationsQcu",
        csw_used_by_operator=True,
        mass_used_by_operator=True,
        clover_abs_max=clover_abs_max,
        clover_inverse_abs_max=clover_inv_abs_max,
        clover_term_abs_max=float(clover_full.abs().max()),
        relative_residual=true_residual,
        true_residual=true_residual,
        relative_error=true_residual,
        iterations=iterations,
        iterations_available=iterations is not None,
        hit_max_iter=bool(iterations is not None
                          and iterations >= control.max_iter),
        native_reported_converged=bool(result.converged),
        solver_status=status,
        residual_scope=scope,
        source_norm=float(torch.linalg.vector_norm(full_source)),
        solution_norm=float(torch.linalg.vector_norm(full_solution)),
        solve_seconds=(None if result.solve_seconds is None
                       else float(result.solve_seconds)),
        notes=[
            "the independent true residual uses the full Clover operator at "
            f"csw={opts.csw:g} (csw=0 is the Wilson operator)",
        ],
    )
    return _finish(
        observations, probe, runtime, seconds=seconds,
        prepare_seconds=prepare_seconds, immutability={
            "gauge_unchanged": bool(torch.equal(inputs.gauge, gauge_before)),
            "source_unchanged": bool(torch.equal(inputs.source, source_before)),
            "controls_unchanged": (
                _controls(runtime.params, runtime.argv, define)
                == controls_before),
        })


# ---------------------------------------------------------------------------
# runner: full native conjugate-gradient solve
# ---------------------------------------------------------------------------
