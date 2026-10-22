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
the current index. Strict multigrid is the exception: one fixed private slot
remains initialized across its setup, solve, getters, and teardown calls.

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
        if not 1 <= int(slot_limit) <= define._SET_PTRS_SIZE_:
            raise ValueError(
                f"slot_limit must be in [1, {define._SET_PTRS_SIZE_}]")
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

    def _qcu(self) -> Any:
        if self.qcu_module is None:
            from pyqcu.cuda import qcu as qcu_module
            self.qcu_module = qcu_module
        return self.qcu_module

    def _allocate_slot(self) -> int:
        count = int(self.slot_limit)
        if count < 1 or count > int(self.set_ptrs.numel()):
            raise ValueError(
                f"slot_limit must be in [1, {int(self.set_ptrs.numel())}]")
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
                finally:
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

    def call(self, name: str, *args: torch.Tensor,
             synchronize: bool = True) -> Any:
        """Invoke one ``*Qcu`` operation with automatic init/end bookkeeping."""
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
