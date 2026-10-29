from __future__ import annotations

import os
from typing import Any, Sequence, Tuple

import numpy as np
from mpi4py import MPI

import pyqcu.cann as _torch


_BATCH_REDUCTIONS_ENV = "PYQCU_TORCH_BATCH_REDUCTIONS"


def _communicator(comm: Any = None) -> Any:
    """Return the active distributed context communicator when available."""
    if comm is not None:
        return comm
    # Local import avoids a tools-package import cycle.  Strict-MG executes
    # reductions inside this context so an explicit communicator is honored.
    from . import _mpi_roll

    context = _mpi_roll.current_context()
    if context is not None and hasattr(context.comm, "Allreduce"):
        return context.comm
    return MPI.COMM_WORLD


def _local_vdot_pair(pair: Tuple[Any, Any]) -> Any:
    try:
        left, right = pair
    except (TypeError, ValueError) as exc:
        raise TypeError("vdot_many 的每一项必须是 (a, b) 二元组") from exc
    if left.device != right.device:
        raise ValueError("vdot_many 的 a/b 必须位于同一设备")
    return _torch.vdot(left.flatten(), right.flatten())


def _allreduce_scalars(
        local_values: Sequence[Any], comm: Any = None) -> Tuple[Any, ...]:
    if not local_values:
        return ()
    device = local_values[0].device
    dtype = local_values[0].dtype
    if any(value.device != device or value.dtype != dtype
           for value in local_values[1:]):
        raise ValueError("批量归约的标量必须具有相同 device/dtype")
    sendbuf = (
        _torch.stack(list(local_values)).detach().cpu().contiguous().numpy())
    recvbuf = np.empty_like(sendbuf)
    _communicator(comm).Allreduce(sendbuf=sendbuf, recvbuf=recvbuf, op=MPI.SUM)
    result = _torch.as_tensor(recvbuf, dtype=dtype, device=device)
    return tuple(result[index] for index in range(len(local_values)))


def vdot(a: Any, b: Any, *, comm: Any = None) -> Any:
    """
    Multi-process dot product using mpi4py with buffer mode.
    Args:
        a, b: local tensors on GPU (per process)
    Returns:
        global dot product as a complex scalar (torch.Tensor)
    """
    assert a.device == b.device, "a and b must be on the same device"
    # OPT 2026-07-28 R2: removed redundant Barrier() — Allreduce is already blocking.
    # Each BiCGStab iteration calls vdot ~5 times; removing these saves ~10
    # unnecessary global synchronizations per iteration.
    return _allreduce_scalars([_local_vdot_pair((a, b))], comm=comm)[0]


def vdot_many(pairs: Sequence[Tuple[Any, Any]], *,
              comm: Any = None) -> Tuple[Any, ...]:
    """Evaluate independent global inner products with one collective.

    ``PYQCU_TORCH_BATCH_REDUCTIONS=0`` restores the scalar-by-scalar path for
    A/B measurements.  Callers remain responsible for using only products that
    are mathematically independent at that point in the recurrence.
    """
    items = tuple(pairs)
    if not items:
        return ()
    if os.environ.get(_BATCH_REDUCTIONS_ENV, "1").strip() == "0":
        return tuple(vdot(left, right, comm=comm) for left, right in items)
    communicator = _communicator(comm)
    return _allreduce_scalars(
        [_local_vdot_pair(pair) for pair in items], comm=communicator)


def norm(a: Any) -> float:
    """
    Multi-process norm with buffer mode.
    Args:
        a: local tensor
    Returns:
        global norm
    """
    return float(_torch.torch.sqrt(vdot(a=a, b=a).real).item())


def norm_many(values: Sequence[Any]) -> Tuple[float, ...]:
    """Evaluate independent global norms with at most one collective."""
    items = tuple(values)
    if not items:
        return ()
    squared = vdot_many(tuple((value, value) for value in items))
    return tuple(
        float(_torch.torch.sqrt(value.real).item()) for value in squared)


__all__ = ["norm", "norm_many", "vdot", "vdot_many"]
