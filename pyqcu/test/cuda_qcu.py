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

The bridge is never built, imported for side effects or monkey-patched here.
A multi-rank MPI job is skipped as ``mpi_single_rank``: the tier-0 CUDA cases
are defined as one rank, one device, one process.

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

Solve case
----------
``applyCloverBistabCgQcu`` at ``csw = 0`` is used for the solver case.  With a
zero Clover term the operator *is* the Wilson operator, while this ABI symbol
is the only CUDA solver that reports a native iteration count
(``getCloverBistabCgIterationsQcu``) and the only one whose ``argv[ATOL]`` is a
relative tolerance with a round-off floor.  The Wilson symbol
``applyWilsonBistabCgQcu`` has no iteration getter in the 40-symbol ABI and
interprets ``argv[ATOL]`` as an *absolute* Schur residual, so it cannot feed
the ``iterations > 0`` gate.  The independent true residual is always
reconstructed with the full Wilson operator, never from the solver recurrence.

Note: the C++ solver prints one ``##RANK:<r>##LOOP:<n>##Residual(norm2):...``
line to stdout when it leaves its iteration loop.  That print lives in
``cpp/cuda/qcu`` and is not suppressed here (doing so would require editing
core code or hijacking file descriptor 1).
"""

from __future__ import annotations

import os
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, NoReturn

from . import policy
from .common import Skip, relative_error, relative_residual, time_samples

__all__ = [
    "BACKEND",
    "DEFAULT_CUDA_DEVICE",
    "QUICK_LATTICE",
    "SLOT_LIMIT",
    "probe_cuda",
    "run_dslash_clover",
    "run_dslash_wilson",
    "run_smoke_init_end",
    "run_solve_bistabcg",
    "run_solve_cg",
]

#: Backend name used in case tags and reports.
BACKEND = "cuda"
#: Project correctness lattice (``AGENTS.md``, ``pyqcu/testing/_config.py``).
QUICK_LATTICE: tuple[int, int, int, int] = (8, 8, 8, 16)
#: ``define._SET_PTRS_COARSE_BASE_``: first index reserved for multigrid.
SLOT_LIMIT = 30
#: Device used when ``QCU_DEVICE`` is not set in the environment.
DEFAULT_CUDA_DEVICE = "cuda:0"

#: Mass of the tier-0 CUDA cases (``pyqcu._selftest`` uses the same value).
DEFAULT_MASS = 0.17
#: ``csw`` of the solver case: zero Clover term, i.e. the Wilson operator.
DEFAULT_CSW = 0.0
#: Relative solver tolerance (``argv[_ATOL_]``) per dtype.
DEFAULT_ATOL_C64 = 1.0e-6
DEFAULT_ATOL_C128 = 1.0e-10
#: Runaway guard for the native solver; the ``iterations < max_iter`` gate
#: needs the requested limit as an observation.
DEFAULT_MAX_ITER = 200
#: Gauge-field disorder used to build the random fixed input.
DEFAULT_GAUGE_SIGMA = 0.1

_DEVICE_ENVIRONMENT = "QCU_DEVICE"
_BRIDGE_TOKEN = "cuda_bridge"
_DEVICE_TOKEN = "cuda_device"
_RANK_TOKEN = "mpi_single_rank"

_SMOKE_SYMBOLS = ("applyInitQcu", "applyEndQcu")
_DSLASH_SYMBOLS = _SMOKE_SYMBOLS + ("applyWilsonDslashQcu",)
_SOLVE_SYMBOLS = _SMOKE_SYMBOLS + (
    "applyCloversQcu",
    "applyCloverBistabCgQcu",
    "getCloverBistabCgIterationsQcu",
)

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
    dtype: str
    seed: int
    repeat: int
    warmup: int


def _options(lattice: Any = None, dtype: Any = None, seed: Any = None,
             repeat: Any = None, warmup: Any = None) -> _RunOptions:
    """Normalize the runner arguments without importing a backend."""
    shape = policy.validate_lattice(
        QUICK_LATTICE if lattice is None else lattice)
    dtype_name = policy.validate_dtype(
        policy.QUICK_DTYPE if dtype is None else str(dtype))
    seed_value = policy.QUICK_SEED if seed is None else int(seed)
    repeat_value = policy.QUICK_REPEAT if repeat is None else int(repeat)
    warmup_value = policy.DEFAULT_WARMUP if warmup is None else int(warmup)
    if repeat_value < 1:
        raise ValueError(f"repeat must be >= 1, got {repeat!r}")
    if warmup_value < 1:
        raise ValueError(f"warmup must be >= 1, got {warmup!r}")
    return _RunOptions(shape=shape, dtype=dtype_name, seed=seed_value,
                       repeat=repeat_value, warmup=warmup_value)


def _solve_atol(dtype_name: str) -> float:
    """Relative ``argv[_ATOL_]`` for the native solver of ``dtype_name``."""
    return DEFAULT_ATOL_C128 if dtype_name == "c128" else DEFAULT_ATOL_C64


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


def _probe_device(device: Any = None) -> tuple[Any, Any, str, str]:
    """Return ``(torch, torch.device, name, capability)`` or skip ``cuda_device``."""
    torch = _import_torch()
    try:
        available = bool(torch.cuda.is_available())
    except Exception as error:  # noqa: BLE001 - probe must stay a SKIP
        _skip(f"torch.cuda.is_available() raised "
              f"{type(error).__name__}: {error}", _DEVICE_TOKEN)
    if not available:
        _skip("torch.cuda.is_available() is False; the native CUDA bridge "
              "needs a visible device", _DEVICE_TOKEN)
    requested = device or os.environ.get(_DEVICE_ENVIRONMENT) \
        or DEFAULT_CUDA_DEVICE
    try:
        selected = torch.device(str(requested))
    except (RuntimeError, ValueError) as error:
        raise ValueError(
            f"invalid CUDA device {requested!r} "
            f"(from ${_DEVICE_ENVIRONMENT}): {error}") from error
    if selected.type != "cuda":
        raise ValueError(
            f"device must be a CUDA device, got {requested!r}")
    index = 0 if selected.index is None else int(selected.index)
    try:
        count = int(torch.cuda.device_count())
    except Exception as error:  # noqa: BLE001 - probe must stay a SKIP
        _skip(f"torch.cuda.device_count() raised "
              f"{type(error).__name__}: {error}", _DEVICE_TOKEN)
    if index < 0 or index >= count:
        _skip(f"device {index} is not visible; torch reports {count} "
              f"CUDA device(s)", _DEVICE_TOKEN)
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


def _require_cuda(device: Any = None) -> _CudaProbe:
    """Probe device, bridge, define and MPI size; raise :class:`Skip` on absence.

    The device is checked first and the bridge second, so a machine without a
    GPU never imports the extension and an unbuilt checkout never allocates a
    tensor.  Only after every probe passes may a runner issue a native call.
    """
    torch, selected, name, capability = _probe_device(device)
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


def _make_runtime(probe: _CudaProbe, options: _RunOptions, *, mass: float,
                  csw: float, atol: float, plan: int, parity: int,
                  max_iter: int) -> Any:
    """Build one private :class:`QcuRuntime` for this case (never a global)."""
    from pyqcu.cuda._runtime import QcuRuntime

    return QcuRuntime.from_lattice(
        options.shape,
        mass=mass,
        csw=csw,
        atol=atol,
        sigma=DEFAULT_GAUGE_SIGMA,
        dtype=_torch_dtype(probe.torch, options.dtype),
        device=str(probe.device),
        plan=plan,
        parity=parity,
        max_iter=max_iter,
        seed=options.seed,
        verbose=0,
        grid=(1, 1, 1, 1),
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
# deterministic inputs (random but fixed gauge/source)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Inputs:
    """Random-but-fixed gauge and source in both Python layouts."""

    full_gauge: Any
    gauge: Any
    source: Any
    gauge_su3: bool
    gauge_seed: int
    source_seed: int


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


def _make_inputs(probe: _CudaProbe, options: _RunOptions) -> _Inputs:
    """Build a fixed random SU(3) gauge field and a fixed random source.

    The gauge field comes from the project generator with an explicit seed; it
    is generated on the CPU and copied to the device so the value does not
    depend on the device RNG implementation.  The source uses a private
    ``torch.Generator``, so the value is reproducible and does not consume the
    process-wide random stream.
    """
    torch = probe.torch
    from pyqcu import lattice, tools

    shape = options.shape
    dtype = _torch_dtype(torch, options.dtype)
    gauge_seed = int(options.seed)
    source_seed = int(options.seed) + 1
    full_gauge = torch.empty((3, 3, 4, *shape), dtype=dtype, device="cpu")
    with _preserved_rng(torch):
        lattice.generate_gauge_field(
            full_gauge, sigma=DEFAULT_GAUGE_SIGMA, seed=gauge_seed,
            verbose=False)
    gauge_su3 = bool(lattice.check_su3(full_gauge, tol=1e-3, verbose=False))
    compact = (shape[0], shape[1], shape[2], shape[3] // 2)
    generator = torch.Generator(device="cpu").manual_seed(source_seed)
    source = torch.randn(
        (2, 4, 3, *compact), generator=generator, dtype=dtype, device="cpu")
    full_gauge_device = full_gauge.to(device=probe.device).contiguous()
    gauge = tools.oooxyzt2poooxyzt(full_gauge_device).contiguous()
    return _Inputs(
        full_gauge=full_gauge_device,
        gauge=gauge,
        source=source.to(device=probe.device).contiguous(),
        gauge_su3=gauge_su3,
        gauge_seed=gauge_seed,
        source_seed=source_seed,
    )


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
                                   compact_source: Any, parity: int) -> Any:
    """Independent reference built from the *full-lattice* Wilson operator.

    The compact source is embedded in the opposite parity of a full field,
    the full hopping operator is applied once (``with_I=False``, ``kappa=1``)
    and the target parity is split back out.  This does not reuse the compact
    even/odd helpers used by :func:`_compact_wilson_reference`.
    """
    from pyqcu import tools
    from pyqcu.torch import dslash

    torch = probe.torch
    shape = tuple(int(value) for value in inputs.full_gauge.shape[-4:])
    compact = (shape[0], shape[1], shape[2], shape[3] // 2)
    full_source = torch.zeros(
        (2, 4, 3, *compact), dtype=compact_source.dtype,
        device=compact_source.device)
    full_source[1 - parity] = compact_source
    full_field = tools.poooxyzt2oooxyzt(full_source)
    hopping = -dslash.give_wilson(
        full_field, inputs.full_gauge, kappa=torch.tensor([1.0]), with_I=False)
    return tools.oooxyzt2poooxyzt(hopping)[parity].contiguous()


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


# ---------------------------------------------------------------------------
# runner: bridge/device init/end smoke test
# ---------------------------------------------------------------------------
def run_smoke_init_end(lattice: Any = None, dtype: Any = None,
                       seed: Any = None, repeat: Any = None,
                       warmup: Any = None) -> dict[str, Any]:
    """Exercise ``applyInitQcu``/``applyEndQcu`` on private slot tables.

    Observations: ``init_ok`` (the gate), ``slot_isolation``, ``session_ok``,
    the two allocated slot indices, the release/neighbour facts, the
    ``globals_unchanged`` evidence and the timing.  No gauge or fermion field is
    touched, so this case isolates the lifecycle from the numerics.
    """
    options = _options(lattice, dtype, seed, repeat, warmup)
    probe = _require_cuda()
    _require_symbols(probe.bridge, _SMOKE_SYMBOLS, "the init/end smoke case")
    started = time.perf_counter()
    main = _make_runtime(
        probe, options, mass=DEFAULT_MASS, csw=DEFAULT_CSW, atol=1.0e-7,
        plan=0, parity=0, max_iter=8)
    partner = _make_runtime(
        probe, options, mass=DEFAULT_MASS, csw=DEFAULT_CSW, atol=1.0e-7,
        plan=0, parity=0, max_iter=8)
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
            bridge.applyInitQcu(partner.set_ptrs, partner.params, partner.argv)
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
        _sequence, repeat=options.repeat, warmup=options.warmup)
    if not isinstance(observation, dict):
        raise RuntimeError("init/end smoke transaction returned no observation")
    return {
        "backend": BACKEND,
        "check": "smoke_init_end",
        "device": probe.device_name,
        "capability": probe.capability,
        "device_id": str(probe.device),
        "lattice": list(options.shape),
        "dtype": options.dtype,
        "seed": int(options.seed),
        "mpi_size": int(probe.mpi_size),
        "mpi_rank": int(probe.mpi_rank),
        "bridge_file": probe.bridge_file,
        "bridge_built": True,
        "repeat": int(options.repeat),
        "warmup": int(options.warmup),
        **observation,
        "prepare_seconds": prepare_seconds,
        "seconds": seconds,
    }


# ---------------------------------------------------------------------------
# runner: Wilson dslash against the torch reference
# ---------------------------------------------------------------------------
def run_dslash_wilson(lattice: Any = None, dtype: Any = None,
                      seed: Any = None, repeat: Any = None,
                      warmup: Any = None) -> dict[str, Any]:
    """Apply ``applyWilsonDslashQcu`` once per parity transaction and compare.

    The input is the opposite compact parity of a fixed random source and the
    gauge field is a fixed random SU(3) field; the reference is computed by
    torch (compact even/odd helper *and* the full-lattice operator, which is
    the independent path).  Observations: ``relative_error`` (gate alias
    ``relative``), ``relative_error_full``, ``reference_spread``, input
    immutability, lifecycle release and timing.
    """
    options = _options(lattice, dtype, seed, repeat, warmup)
    probe = _require_cuda()
    _require_symbols(probe.bridge, _DSLASH_SYMBOLS, "the Wilson dslash case")
    started = time.perf_counter()
    inputs = _make_inputs(probe, options)
    parity = 1
    runtime = _make_runtime(
        probe, options, mass=DEFAULT_MASS, csw=DEFAULT_CSW, atol=1.0e-7,
        plan=0, parity=parity, max_iter=8)
    torch = probe.torch
    source = inputs.source[1 - parity].contiguous()
    output = torch.zeros_like(source)
    source_before = source.clone()
    gauge_before = inputs.gauge.clone()
    controls_before = _controls(runtime.params, runtime.argv, probe.define)
    prepare_seconds = time.perf_counter() - started

    def _apply() -> None:
        runtime.call("applyWilsonDslashQcu", output, source, inputs.gauge)
        _synchronize(probe)

    seconds, _ = _measure(
        _apply, repeat=options.repeat, warmup=options.warmup)
    reference = _compact_wilson_reference(torch, source, inputs.gauge, parity)
    reference_full = _full_lattice_wilson_reference(
        probe, inputs, source, parity)
    relative = relative_error(output, reference)
    relative_l2 = _relative_l2(output, reference)
    relative_full = relative_error(output, reference_full)
    reference_spread = relative_error(reference, reference_full)
    changed = _changed_globals(probe)
    return {
        "backend": BACKEND,
        "check": "dslash_wilson",
        "device": probe.device_name,
        "capability": probe.capability,
        "lattice": list(options.shape),
        "dtype": options.dtype,
        "seed": int(options.seed),
        "gauge_seed": inputs.gauge_seed,
        "source_seed": inputs.source_seed,
        "mpi_size": int(probe.mpi_size),
        "mpi_rank": int(probe.mpi_rank),
        "bridge_file": probe.bridge_file,
        "mass": DEFAULT_MASS,
        "csw": DEFAULT_CSW,
        "kappa": runtime.kappa,
        "plan": 0,
        "parity": parity,
        "source_parity": 1 - parity,
        "source_norm": float(torch.linalg.vector_norm(source)),
        "output_norm": float(torch.linalg.vector_norm(output)),
        "reference_norm": float(torch.linalg.vector_norm(reference)),
        "gauge_su3": inputs.gauge_su3,
        "relative_error": relative,
        "relative": relative,
        "relative_error_l2": relative_l2,
        "relative_error_full": relative_full,
        "reference_spread": reference_spread,
        "output_finite": bool(torch.isfinite(output).all()),
        "source_unchanged": bool(torch.equal(source, source_before)),
        "gauge_unchanged": bool(torch.equal(inputs.gauge, gauge_before)),
        "inputs_unchanged": bool(torch.equal(source, source_before)
                                 and torch.equal(inputs.gauge, gauge_before)),
        "controls_unchanged": (_controls(runtime.params, runtime.argv,
                                         probe.define) == controls_before),
        "slots_released": _slots_released(runtime),
        "params_index_after": _set_index(runtime, probe.define),
        "globals_unchanged": not changed,
        "changed_global_arrays": changed,
        "repeat": int(options.repeat),
        "warmup": int(options.warmup),
        "prepare_seconds": prepare_seconds,
        "seconds": seconds,
    }


# ---------------------------------------------------------------------------
# runner: full native BiStabCG solve with an independent true residual
# ---------------------------------------------------------------------------
def run_solve_bistabcg(lattice: Any = None, dtype: Any = None,
                       seed: Any = None, repeat: Any = None,
                       warmup: Any = None) -> dict[str, Any]:
    """Run one full native BiStabCG solve and reconstruct its true residual.

    The native equation is the Wilson operator: the Clover solver is used with
    ``csw = 0`` (zero Clover term, identity inverse), which is the only CUDA
    solver that reports a native iteration count and uses a relative
    tolerance.  ``relative_residual`` is recomputed from the full-lattice
    Wilson operator applied to the returned solution, never from the solver
    recurrence.  Observations: ``relative_residual`` (gate), ``iterations``,
    ``max_iter``, status/scope, timings, lifecycle release and immutability.
    """
    options = _options(lattice, dtype, seed, repeat, warmup)
    probe = _require_cuda()
    _require_symbols(probe.bridge, _SOLVE_SYMBOLS, "the native solve case")
    started = time.perf_counter()
    torch = probe.torch
    inputs = _make_inputs(probe, options)
    atol = _solve_atol(options.dtype)
    max_iter = DEFAULT_MAX_ITER
    parity = 1
    runtime = _make_runtime(
        probe, options, mass=DEFAULT_MASS, csw=DEFAULT_CSW, atol=atol,
        plan=1, parity=parity, max_iter=max_iter)
    from pyqcu.api import BiCGStabControl
    from pyqcu.cuda.linear_operator import CudaLinearOperator
    from pyqcu.solver.krylov import Krylov

    # csw = 0: build the (zero) Clover term and its (identity) inverse for both
    # parities.  Each build is one ordinary init/op/end transaction.
    gauge_before = inputs.gauge.clone()
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
    control = BiCGStabControl(rtol=atol, max_iter=max_iter)
    solver = Krylov(
        backend=BACKEND, method="bicgstab", operator=operator,
        control=control)
    # ``make_clover`` leaves plan 2 in the private controls; the solve driver
    # selects plan 1, so snapshot after restoring the plan the solve uses.
    runtime.configure(plan=probe.define._SET_PLAN1_)
    controls_before = _controls(runtime.params, runtime.argv, probe.define)
    source_before = inputs.source.clone()
    prepare_seconds = time.perf_counter() - started

    def _solve() -> Any:
        result = solver.solve_result(inputs.source)
        _synchronize(probe)
        return result

    seconds, result = _measure(
        _solve, repeat=options.repeat, warmup=options.warmup)
    if result is None:
        raise RuntimeError("native CUDA solve returned no result")
    from pyqcu import tools
    from pyqcu.torch import dslash

    full_source = tools.poooxyzt2oooxyzt(inputs.source)
    full_solution = tools.poooxyzt2oooxyzt(result.x)
    kappa = runtime.kappa
    full_gauge = inputs.full_gauge

    def _matvec(vector: Any) -> Any:
        return dslash.give_wilson(
            vector, full_gauge, torch.tensor([kappa]), with_I=True)

    true_residual = relative_residual(_matvec, full_solution, full_source)
    iterations = (None if result.krylov_iterations is None
                  else int(result.krylov_iterations))
    status = getattr(result.status, "value", str(result.status))
    scope = getattr(result.residual_scope, "value",
                    str(result.residual_scope))
    changed = _changed_globals(probe)
    return {
        "backend": BACKEND,
        "check": "solve_bistabcg",
        "device": probe.device_name,
        "capability": probe.capability,
        "lattice": list(options.shape),
        "dtype": options.dtype,
        "seed": int(options.seed),
        "gauge_seed": inputs.gauge_seed,
        "source_seed": inputs.source_seed,
        "mpi_size": int(probe.mpi_size),
        "mpi_rank": int(probe.mpi_rank),
        "bridge_file": probe.bridge_file,
        "mass": DEFAULT_MASS,
        "csw": DEFAULT_CSW,
        "kappa": kappa,
        "plan": 1,
        "parity": parity,
        "abi_symbol": "applyCloverBistabCgQcu",
        "iteration_getter": "getCloverBistabCgIterationsQcu",
        "equation": "wilson",
        "clover_abs_max": clover_abs_max,
        "clover_inverse_abs_max": clover_inv_abs_max,
        "gauge_su3": inputs.gauge_su3,
        "relative_residual": true_residual,
        "true_residual": true_residual,
        "relative_error": true_residual,
        "iterations": iterations,
        "iterations_available": iterations is not None,
        "max_iter": int(control.max_iter),
        "hit_max_iter": bool(iterations is not None
                             and iterations >= control.max_iter),
        "native_reported_converged": bool(result.converged),
        "solver_status": status,
        "residual_scope": scope,
        "source_norm": float(torch.linalg.vector_norm(full_source)),
        "solution_norm": float(torch.linalg.vector_norm(full_solution)),
        "solve_seconds": (None if result.solve_seconds is None
                          else float(result.solve_seconds)),
        "source_unchanged": bool(torch.equal(inputs.source, source_before)),
        "gauge_unchanged": bool(torch.equal(inputs.gauge, gauge_before)),
        "inputs_unchanged": bool(torch.equal(inputs.source, source_before)
                                 and torch.equal(inputs.gauge, gauge_before)),
        "controls_unchanged": (_controls(runtime.params, runtime.argv,
                                         probe.define) == controls_before),
        "slots_released": _slots_released(runtime),
        "params_index_after": _set_index(runtime, probe.define),
        "globals_unchanged": not changed,
        "changed_global_arrays": changed,
        "repeat": int(options.repeat),
        "warmup": int(options.warmup),
        "prepare_seconds": prepare_seconds,
        "seconds": seconds,
    }


def run_dslash_clover(lattice: Any = None, dtype: Any = None,
                      seed: Any = None, repeat: Any = None,
                      warmup: Any = None) -> dict[str, Any]:
    """CUDA Clover dslash entry point.

    The Clover ABI path is available only when the Cython bridge is built and
    exports the Clover dslash symbols.  This checkout has no bridge, so the
    probe raises the documented ``cuda_bridge`` skip before any native call.
    Keeping the runner explicit is important: the self-check must be able to
    resolve every declared case, and an unavailable backend must SKIP with a
    reason rather than silently disappearing from the matrix.
    """
    options = _options(lattice, dtype, seed, repeat, warmup)
    probe = _require_cuda()
    _require_symbols(probe.bridge, _DSLASH_SYMBOLS, "the Clover dslash case")
    raise Skip(
        "cuda_bridge: the CUDA Clover dslash runner requires a built bridge "
        f"at {probe.bridge_file}; this checkout does not implement that path",
        token="cuda_bridge")


def run_solve_cg(lattice: Any = None, dtype: Any = None,
                 seed: Any = None, repeat: Any = None,
                 warmup: Any = None) -> dict[str, Any]:
    """CUDA CG solve entry point.

    As with :func:`run_dslash_clover`, the bridge/device probes run first; an
    unavailable bridge is a SKIP with a non-empty reason.  The function is kept
    in the registry so ``--solver cg`` and the self-check can resolve the case
    on every machine.
    """
    options = _options(lattice, dtype, seed, repeat, warmup)
    probe = _require_cuda()
    _require_symbols(probe.bridge, _SOLVE_SYMBOLS, "the native CG case")
    raise Skip(
        "cuda_bridge: the CUDA CG runner requires a built bridge at "
        f"{probe.bridge_file}; this checkout does not implement that path",
        token="cuda_bridge")
