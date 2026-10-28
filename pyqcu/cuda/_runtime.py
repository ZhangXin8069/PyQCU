"""Lifecycle-safe wrapper around the flat QCU Cython/ctypes ABI.

Every call crosses the backend boundary through three contiguous tensors:

``params`` (int32[58])
    Geometry, MPI rank grid, dtype, plan, parity, solver controls, and the
    current ``_SET_INDEX_`` slot.
``argv`` (real[8])
    Mass, tolerance, sigma, four multigrid tolerances, and ``csw`` in slot 7.
``set_ptrs`` (int64[100])
    Backend-owned resource pointers for the selected slot. Slots ``[0:30)``
    are ordinary transactional scratch. ``[30:60)`` is reserved for regular
    multigrid coarse assets, ``[60:80)`` for strict per-level assets, and slot
    80 for the strict hierarchy. Code outside those dedicated wrappers must
    treat all indices from 30 onward as reserved.

An ordinary direct-ABI transaction is ``applyInitQcu``, operation and any
same-slot getters, ``applyEndQcu``, then selection of the next slot. The index
must not advance before ``applyEndQcu`` because that call releases the slot at
the current index. Repeated solves may instead use ``QcuRuntime.session()`` to
keep one ordinary slot initialized across multiple calls. Strict multigrid is
the exception: one fixed private slot remains initialized across its setup,
solve, getters, and teardown calls.

Parity-split ABI tensors use ``[X, Y, Z, T/2]`` as their final four axes. Their
leading dimensions are parity/spin/color in ABI order: fermions are
``[2, 4, 3, X, Y, Z, T/2]``, gauge links are
``[2, 3, 3, 4, X, Y, Z, T/2]``, and Clover terms are
``[4, 3, 4, 3, X, Y, Z, T/2]``. ``params[_PARITY_]`` selects the target even
or odd parity where an operation is parity-specific.

The production CUDA topology is one Python process, one MPI rank, and one
device. Distributed work must use MPI rank partitioning of the four-dimensional
lattice rather than multiple device workers in one process. The explicit
multi-GPU test driver is not a production execution model.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import torch

from pyqcu import tools
from pyqcu.cuda import define


_FORBIDDEN_TRANSACTION_OPERATIONS = frozenset({
    "applyInitQcu",
    "applyEndQcu",
    "applyMultigridStrictInitQcu",
    "applyMultigridStrictEndQcu",
})


@dataclass
class QcuRuntime:
    """Own one QCU control state and execute lifecycle-safe calls.

    ``call()`` performs the supported transaction shape for ordinary QCU
    operations::

        select free slot -> applyInitQcu -> operation/getter -> applyEndQcu

    The selected index is recorded before initialization, the slot remains live
    only inside ``operation()``, and ``applyEndQcu`` clears it in ``finally``.
    Only slots ``[0:30)`` are considered recyclable ordinary scratch, so coarse
    pointers from index 30 onward are never overwritten. Multi-call strict-MG
    lifecycles instead keep one fixed private slot and use their dedicated
    wrapper for the full setup-to-teardown sequence.
    """

    params: torch.Tensor
    argv: torch.Tensor
    set_ptrs: torch.Tensor
    device: torch.device
    lattice: tuple[int, int, int, int]
    mass: float
    csw: float
    qcu_module: Any = None
    slot_limit: int = define._SET_PTRS_COARSE_BASE_
    _slot_cursor: int = field(default=0, init=False, repr=False)
    _operation_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        _require_control_array(
            self.params, "params", torch.int32, define._PARAMS_SIZE_)
        _require_control_array(
            self.set_ptrs, "set_ptrs", torch.int64, define._SET_PTRS_SIZE_)
        expected_argv_dtype = _argv_dtype_from_params(self.params)
        _require_control_array(
            self.argv, "argv", expected_argv_dtype, define._ARGV_SIZE_)
        if not 1 <= int(self.slot_limit) <= define._SET_PTRS_COARSE_BASE_:
            raise ValueError(
                "slot_limit must be in "
                f"[1, {define._SET_PTRS_COARSE_BASE_}] for ordinary scratch")

    @classmethod
    def from_lattice(
        cls,
        lattice: Sequence[int],
        *,
        mass: float,
        csw: float = 1.0,
        atol: float = 1.0e-6,
        sigma: float = 0.1,
        dtype: torch.dtype = torch.complex64,
        device: str | torch.device = "cuda:0",
        plan: int = 0,
        parity: int = 0,
        max_iter: int = 1000,
        seed: int = 42,
        verbose: int = 0,
        grid: Sequence[int] | None = None,
        qcu_module: Any = None,
        slot_limit: int = define._SET_PTRS_COARSE_BASE_,
    ) -> "QcuRuntime":
        """Create independent ABI arrays for one lattice and MPI rank grid.

        ``grid`` must match the process's MPI rank grid. The resulting arrays
        are private by default and begin at set slot zero. For production CUDA
        execution, ``device`` denotes the single device owned by this
        Python/MPI rank.
        """
        lat = tuple(int(value) for value in lattice)
        if len(lat) != 4 or any(value <= 0 or value % 2 for value in lat):
            raise ValueError("lattice dimensions must be four positive even integers")
        if dtype not in (torch.complex64, torch.complex128):
            raise TypeError("QCU runtime supports complex64 or complex128")

        data_type = int(define.epytd(dtype).item())
        params = torch.zeros(define._PARAMS_SIZE_, dtype=torch.int32)
        params[define._LAT_X_] = lat[0]
        params[define._LAT_Y_] = lat[1]
        params[define._LAT_Z_] = lat[2]
        params[define._LAT_T_] = lat[3]
        params[define._LAT_XYZT_] = int(torch.tensor(lat).prod())
        if grid is None:
            grid = tools.give_grid_size()
        grid_tuple = tuple(int(value) for value in grid)
        if len(grid_tuple) != 4 or any(value <= 0 for value in grid_tuple):
            raise ValueError("grid must contain four positive integers")
        if int(torch.tensor(grid_tuple).prod()) != define.size:
            raise ValueError(
                f"grid {grid_tuple} contains {int(torch.tensor(grid_tuple).prod())} "
                f"ranks, but MPI_COMM_WORLD has {define.size}")
        if not 1 <= int(slot_limit) <= define._SET_PTRS_COARSE_BASE_:
            raise ValueError(
                "slot_limit must be in "
                f"[1, {define._SET_PTRS_COARSE_BASE_}] for ordinary scratch")
        params[define._GRID_X_], params[define._GRID_Y_], \
            params[define._GRID_Z_], params[define._GRID_T_] = grid_tuple
        params[define._PARITY_] = int(parity)
        params[define._NODE_RANK_] = int(define.rank)
        params[define._NODE_SIZE_] = int(define.size)
        params[define._DAGGER_] = 0
        params[define._MAX_ITER_] = int(max_iter)
        params[define._DATA_TYPE_] = data_type
        params[define._SET_INDEX_] = 0
        params[define._SET_PLAN_] = int(plan)
        params[define._VERBOSE_] = int(verbose)
        params[define._SEED_] = int(seed)
        params[define._TEST_IN_CPU_] = 0
        params = params.contiguous()

        argv = define.make_argv(mass, atol=atol, sigma=sigma, csw=csw,
                                data_type=data_type).contiguous()
        set_ptrs = torch.zeros(define._SET_PTRS_SIZE_, dtype=torch.int64).contiguous()
        runtime = cls(
            params=params,
            argv=argv,
            set_ptrs=set_ptrs,
            device=torch.device(device),
            lattice=lat,
            mass=float(mass),
            csw=float(csw),
            qcu_module=qcu_module,
            slot_limit=int(slot_limit),
        )
        runtime.configure(mass=mass, csw=csw, atol=atol, parity=parity,
                          max_iter=max_iter)
        return runtime

    @property
    def kappa(self) -> float:
        return 1.0 / (2.0 * self.mass + 8.0)

    @property
    def last_set_index(self) -> int:
        return int(self.params[define._SET_INDEX_])

    def configure(self, *, mass: float | None = None, csw: float | None = None,
                  atol: float | None = None, plan: int | None = None,
                  parity: int | None = None, max_iter: int | None = None) -> None:
        """Update physical controls without rebuilding the runtime object."""
        if mass is not None:
            value = float(mass)
            if not math.isfinite(value) or 2.0 * value + 8.0 == 0.0:
                raise ValueError("mass must be finite and must not make kappa infinite")
            self.mass = value
            self.argv[define._MASS_] = self.mass
        if csw is not None:
            value = float(csw)
            if not math.isfinite(value):
                raise ValueError("csw must be finite")
            self.csw = value
            self.argv[define._CSW_] = self.csw
        if atol is not None:
            value = float(atol)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("atol must be finite and positive")
            self.argv[define._ATOL_] = value
        if plan is not None:
            self.params[define._SET_PLAN_] = int(plan)
        if parity is not None:
            value = int(parity)
            if value not in (0, 1):
                raise ValueError("parity must be 0 or 1")
            self.params[define._PARITY_] = value
        if max_iter is not None:
            value = int(max_iter)
            if value < 0:
                raise ValueError("max_iter must be nonnegative")
            self.params[define._MAX_ITER_] = value

    def _validate_operands(self, values: Sequence[torch.Tensor]) -> None:
        data_type = int(self.params[define._DATA_TYPE_].item())
        if data_type == define._LAT_C64_:
            expected_dtype = torch.complex64
        elif data_type == define._LAT_C128_:
            expected_dtype = torch.complex128
        else:
            raise ValueError(f"unsupported QCU data type: {data_type}")
        for index, value in enumerate(values):
            name = f"operand {index}"
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if value.device != self.device:
                raise ValueError(f"{name} must be on {self.device}")
            if value.dtype != expected_dtype:
                raise TypeError(f"{name} must have dtype {expected_dtype}")
            if not value.is_contiguous():
                raise ValueError(f"{name} must be contiguous")

    def _qcu(self) -> Any:
        if self.qcu_module is None:
            from pyqcu.cuda import qcu as qcu_module
            self.qcu_module = qcu_module
        return self.qcu_module

    def _allocate_slot(self) -> int:
        count = int(self.slot_limit)
        if count < 1 or count > define._SET_PTRS_COARSE_BASE_:
            raise ValueError(
                "slot_limit must be in "
                f"[1, {define._SET_PTRS_COARSE_BASE_}] for ordinary scratch")
        for offset in range(count):
            index = (self._slot_cursor + offset) % count
            if int(self.set_ptrs[index]) == 0:
                self.params[define._SET_INDEX_] = index
                self._slot_cursor = (index + 1) % count
                return index
        raise RuntimeError("all QCU set_ptrs slots are busy")

    @contextmanager
    def operation(self, name: str, *,
                  synchronize: bool = True) -> Iterator["QcuOperation"]:
        """Open one ordinary QCU slot for an operation and same-slot getters.

        Perform the operation and read any resulting getter state before leaving
        this context. Exiting calls ``applyEndQcu`` while ``_SET_INDEX_`` still
        names the active slot, then clears that slot for reuse. Reads after the
        context exits are invalid. Strict-MG multiple-call lifecycles belong to
        their dedicated setup/solve/close wrapper, which keeps its slot fixed.
        """
        _require_transaction_operation(name)
        try:
            acquired = self._operation_lock.acquire(blocking=False)
        except BaseException:
            acquired = False
        if not acquired:
            raise RuntimeError(
                "QcuRuntime instances are not reentrant or thread-safe")
        try:
            if self.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.set_device(self.device)
            qcu = self._qcu()
            function = getattr(qcu, name)
            index = self._allocate_slot()
            try:
                qcu.applyInitQcu(self.set_ptrs, self.params, self.argv)
            except BaseException:
                try:
                    qcu.applyEndQcu(self.set_ptrs, self.params)
                except BaseException:
                    pass
                self.set_ptrs[index] = 0
                raise
            try:
                yield QcuOperation(self, function, synchronize=synchronize)
            finally:
                try:
                    qcu.applyEndQcu(self.set_ptrs, self.params)
                finally:
                    self.set_ptrs[index] = 0
        finally:
            self._operation_lock.release()

    def session(self, *, synchronize: bool = True) -> "QcuSession":
        """Reuse one initialized ordinary slot across multiple backend calls.

        Use this only while the copied C++ controls remain unchanged: lattice,
        process grid, dtype, plan, parity, mass, CSW, tolerances, iteration
        limits, and multigrid mode. The session snapshots those controls and
        fails closed if they, the owner thread, or the active slot pointer
        drift. A native operation or getter exception also poisons the session.
        The session holds the runtime's non-reentrant lock and restores the
        snapshot before releasing the slot through ``applyEndQcu`` in
        ``__exit__``.
        """
        return QcuSession(self, synchronize=synchronize)

    def call(self, name: str, *args: torch.Tensor,
             synchronize: bool = True) -> Any:
        """Invoke one ``*Qcu`` operation with automatic init/end bookkeeping."""
        _require_transaction_operation(name)
        with self.operation(name, synchronize=synchronize) as operation:
            return operation(*args)

    def make_clover(self, gauge: torch.Tensor, *, with_inverse: bool = False,
                    parity: int | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Build a parity Clover field using the runtime's ``mass`` and ``csw``."""
        if parity is not None:
            self.params[define._PARITY_] = int(parity)
        self.params[define._SET_PLAN_] = define._SET_PLAN2_
        compact = define.lat_shape(self.params)
        shape = (4, 3, 4, 3, *compact)
        clover = torch.empty(shape, dtype=gauge.dtype, device=gauge.device)
        if not with_inverse:
            self.call("applyCloverQcu", clover, gauge)
            return clover
        clover_inv = torch.empty_like(clover)
        self.call("applyCloversQcu", clover, clover_inv, gauge)
        return clover, clover_inv


@dataclass
class QcuOperation:
    """Operation handle valid only inside :meth:`QcuRuntime.operation`.

    The handle always targets the slot named by the runtime's current
    ``params[_SET_INDEX_]``. Use :meth:`read` immediately after an operation
    for getters whose values live in that slot.
    """

    runtime: QcuRuntime
    function: Any
    synchronize: bool = True

    def __call__(self, *args: torch.Tensor) -> Any:
        """Run the selected operation on the currently initialized slot."""
        self.runtime._validate_operands(args)
        result = self.function(*args, self.runtime.set_ptrs, self.runtime.params)
        if (self.synchronize and self.runtime.device.type == "cuda"
                and torch.cuda.is_available()):
            torch.cuda.synchronize(self.runtime.device)
        return result

    def read(self, name: str, *args: torch.Tensor) -> Any:
        """Read getter state before the surrounding operation context exits.

        Getters such as solver iteration counts are stored in the backend
        object referenced by the current ``set_ptrs`` slot. ``applyEndQcu``
        releases that object when the context exits, so the getter cannot be
        deferred outside the context.
        """
        getter = getattr(self.runtime._qcu(), name)
        return getter(*args, self.runtime.set_ptrs, self.runtime.params)


class QcuSession:
    """Persistent ordinary-slot context returned by :meth:`QcuRuntime.session`.

    A session performs exactly one ``applyInitQcu`` and one ``applyEndQcu``.
    Calls made through ``session(name, ...)`` reuse the same native scratch
    allocation. QCU copies its controls during ``applyInitQcu``, so subsequent
    Python-side control changes cannot affect the initialized backend object.
    The session detects such changes and fails closed instead of treating them
    as effective controls. Native operation/getter failures also poison the
    session even when the caller catches the exception.
    """

    def __init__(self, runtime: QcuRuntime, *, synchronize: bool = True) -> None:
        self.runtime = runtime
        self.synchronize = bool(synchronize)
        self._qcu: Any = None
        self._index: int | None = None
        self._device: torch.device | None = None
        self._active = False
        self._owner_thread: int | None = None
        self._params: torch.Tensor | None = None
        self._argv: torch.Tensor | None = None
        self._set_ptrs: torch.Tensor | None = None
        self._params_snapshot: torch.Tensor | None = None
        self._argv_snapshot: torch.Tensor | None = None
        self._slot_pointer: int | None = None
        self._poison_reason: str | None = None

    @property
    def active(self) -> bool:
        return self._active

    @property
    def set_index(self) -> int | None:
        return self._index

    @property
    def poisoned(self) -> bool:
        return self._poison_reason is not None

    def __enter__(self) -> "QcuSession":
        if self._active:
            raise RuntimeError("QCU session is already active")
        runtime = self.runtime
        try:
            acquired = runtime._operation_lock.acquire(blocking=False)
        except BaseException:
            acquired = False
        if not acquired:
            raise RuntimeError(
                "QcuRuntime instances are not reentrant or thread-safe")
        try:
            if runtime.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.set_device(runtime.device)
            qcu = runtime._qcu()
            index = runtime._allocate_slot()
            params = runtime.params
            argv = runtime.argv
            set_ptrs = runtime.set_ptrs
            try:
                qcu.applyInitQcu(set_ptrs, params, argv)
            except BaseException:
                self._discard_initialization(qcu, index, set_ptrs, params)
                raise
            try:
                params_snapshot = params.clone()
                argv_snapshot = argv.clone()
                slot_pointer = int(set_ptrs[index].item())
            except BaseException:
                self._discard_initialization(qcu, index, set_ptrs, params)
                raise
            self._qcu = qcu
            self._index = index
            self._device = runtime.device
            self._owner_thread = threading.get_ident()
            self._params = params
            self._argv = argv
            self._set_ptrs = set_ptrs
            self._params_snapshot = params_snapshot
            self._argv_snapshot = argv_snapshot
            self._slot_pointer = slot_pointer
            self._poison_reason = None
            self._active = True
            return self
        except BaseException:
            self._clear_state()
            runtime._operation_lock.release()
            raise

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if not self._active:
            return False
        runtime = self.runtime
        index = int(self._index)  # type: ignore[arg-type]
        params = self._params
        argv = self._argv
        set_ptrs = self._set_ptrs
        params_snapshot = self._params_snapshot
        argv_snapshot = self._argv_snapshot
        slot_pointer = self._slot_pointer
        try:
            # ``applyEndQcu`` releases the original slot with the controls used
            # by init. Restore object identities and values first so a drift
            # detected during a call cannot redirect teardown.
            runtime.params = params  # type: ignore[assignment]
            runtime.argv = argv  # type: ignore[assignment]
            runtime.set_ptrs = set_ptrs  # type: ignore[assignment]
            if runtime.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.set_device(runtime.device)
            params.copy_(params_snapshot)  # type: ignore[union-attr]
            argv.copy_(argv_snapshot)  # type: ignore[union-attr]
            params[define._SET_INDEX_] = index  # type: ignore[index]
            set_ptrs[index] = int(slot_pointer)  # type: ignore[index]
            self._qcu.applyEndQcu(set_ptrs, params)
        finally:
            try:
                set_ptrs[index] = 0  # type: ignore[index]
            finally:
                self._clear_state()
                runtime._operation_lock.release()
        return False

    def __call__(self, name: str, *args: torch.Tensor) -> Any:
        _require_transaction_operation(name)
        self._require_ready()
        runtime = self.runtime
        if runtime.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.set_device(runtime.device)
        runtime._validate_operands(args)
        try:
            result = getattr(self._qcu, name)(
                *args, runtime.set_ptrs, runtime.params)
            if (self.synchronize and runtime.device.type == "cuda"
                    and torch.cuda.is_available()):
                torch.cuda.synchronize(runtime.device)
            return result
        except BaseException:
            self._poison(f"native operation {name} raised")
            raise

    def read(self, name: str, *args: torch.Tensor) -> Any:
        self._require_ready()
        try:
            return getattr(self._qcu, name)(
                *args, self.runtime.set_ptrs, self.runtime.params)
        except BaseException:
            self._poison(f"native getter {name} raised")
            raise

    @staticmethod
    def _discard_initialization(
            qcu: Any, index: int, set_ptrs: torch.Tensor,
            params: torch.Tensor) -> None:
        try:
            params[define._SET_INDEX_] = index
            qcu.applyEndQcu(set_ptrs, params)
        except BaseException:
            pass
        finally:
            set_ptrs[index] = 0

    def _require_ready(self) -> None:
        if not self._active or self._index is None:
            raise RuntimeError("QCU session is not active")
        if self._poison_reason is not None:
            raise RuntimeError(
                f"QCU session is poisoned: {self._poison_reason}")
        if threading.get_ident() != self._owner_thread:
            self._fail_closed(
                "used from a thread other than its owner thread")

        runtime = self.runtime
        if runtime.params is not self._params:
            self._fail_closed("params tensor was replaced")
        if runtime.argv is not self._argv:
            self._fail_closed("argv tensor was replaced")
        if runtime.set_ptrs is not self._set_ptrs:
            self._fail_closed("set_ptrs tensor was replaced")
        if runtime.device != self._device:
            self._fail_closed("device changed")
        try:
            index_changed = int(runtime.params[define._SET_INDEX_]) != self._index
            params_changed = not torch.equal(
                runtime.params, self._params_snapshot)
            argv_changed = not torch.equal(runtime.argv, self._argv_snapshot)
            pointer_changed = (
                int(runtime.set_ptrs[self._index]) != self._slot_pointer)
        except BaseException as exc:
            self._fail_closed("control snapshot check failed", exc)
        if index_changed:
            self._fail_closed("_SET_INDEX_ changed")
        if params_changed:
            self._fail_closed("params changed")
        if argv_changed:
            self._fail_closed("argv changed")
        if pointer_changed:
            self._fail_closed("active slot pointer changed")

    def _poison(self, reason: str) -> None:
        self._poison_reason = reason

    def _fail_closed(self, reason: str, cause: BaseException | None = None) -> None:
        self._poison(reason)
        error = RuntimeError(f"QCU session poisoned: {reason}")
        if cause is None:
            raise error
        raise error from cause

    def _clear_state(self) -> None:
        self._active = False
        self._qcu = None
        self._index = None
        self._owner_thread = None
        self._device = None
        self._params = None
        self._argv = None
        self._set_ptrs = None
        self._params_snapshot = None
        self._argv_snapshot = None
        self._slot_pointer = None
        self._poison_reason = None


def _argv_dtype_from_params(params: torch.Tensor) -> torch.dtype:
    data_type = int(params[define._DATA_TYPE_].item())
    if data_type == define._LAT_C64_:
        return torch.float32
    if data_type == define._LAT_C128_:
        return torch.float64
    raise ValueError(f"unsupported QCU data type: {data_type}")


def _require_transaction_operation(name: str) -> None:
    if name in _FORBIDDEN_TRANSACTION_OPERATIONS:
        raise ValueError(
            f"{name} manages the QCU lifecycle and cannot be nested in a "
            "QcuRuntime transaction or session")


def _require_control_array(
        value: Any, name: str, dtype: torch.dtype, numel: int) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU tensor")
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if value.ndim != 1 or int(value.numel()) != int(numel):
        raise ValueError(f"{name} must be a flat {numel}-element tensor")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
