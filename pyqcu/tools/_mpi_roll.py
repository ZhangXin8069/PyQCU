"""Distributed, periodic ``roll`` for the last four lattice axes.

The setup reference code historically used ``torch.roll`` on rank-local
tensors.  That is correct only when the lattice is not partitioned.  This
module provides a small thread-local context that turns the same operation
into a global periodic roll, while moving only the two face slabs per axis.

The implementation deliberately stages the face slabs through CPU memory.
This is a setup-time reference path, not a solver hot loop.  Peak host
traffic is proportional to the tensor surface, never to its volume.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from math import prod
from numbers import Integral
from threading import local
from typing import Any, Iterator, Optional, Sequence, Tuple

import numpy as np

import pyqcu.cann as _torch


Shape4 = Tuple[int, int, int, int]
_SPACE_DIMS = 4
_LOW_FACE_TAG = 0x2100
_HIGH_FACE_TAG = 0x2200
_STATE = local()


class _SerialCommunicator:
    """Minimal mpi4py-like communicator used when no MPI binding is needed."""

    @staticmethod
    def Get_size() -> int:
        return 1

    @staticmethod
    def Get_rank() -> int:
        return 0


@dataclass
class RollContext:
    """Thread-local geometry shared by all distributed roll calls."""

    comm: Any
    process_grid: Shape4
    rank_coordinate: Shape4
    local_extents: Optional[Shape4]
    size: int
    rank: int

    @property
    def distributed(self) -> bool:
        return self.size > 1 and any(value > 1 for value in self.process_grid)

    def global_shape(self, local_shape: Sequence[int]) -> Shape4:
        shape = _shape4(local_shape, "local_shape")
        return tuple(
            int(shape[axis]) * int(self.process_grid[axis])
            for axis in range(4)
        )  # type: ignore[return-value]

    def rank_origin(self, local_shape: Sequence[int]) -> Shape4:
        shape = _shape4(local_shape, "local_shape")
        return tuple(
            int(self.rank_coordinate[axis]) * int(shape[axis])
            for axis in range(4)
        )  # type: ignore[return-value]

    def local_coordinate(self, global_coordinate: Sequence[int],
                         local_shape: Sequence[int]) -> Optional[Shape4]:
        shape = _shape4(local_shape, "local_shape")
        coordinate = _coordinate4(global_coordinate, "global_coordinate")
        origin = self.rank_origin(shape)
        local = tuple(
            int(coordinate[axis]) - int(origin[axis]) for axis in range(4)
        )
        if any(local[axis] < 0 or local[axis] >= shape[axis]
               for axis in range(4)):
            return None
        return local  # type: ignore[return-value]


def _shape4(value: Sequence[int], name: str) -> Shape4:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} 必须包含四个正整数")
    try:
        result = tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} 必须包含四个正整数") from exc
    if len(result) != 4:
        raise ValueError(f"{name} 必须包含四个正整数，得到 {result}")
    if any(item <= 0 for item in result):
        raise ValueError(f"{name} 的每一项必须为正数，得到 {result}")
    return result  # type: ignore[return-value]


def _coordinate4(value: Sequence[int], name: str) -> Shape4:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} 必须包含四个非负整数")
    try:
        result = tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} 必须包含四个非负整数") from exc
    if len(result) != 4:
        raise ValueError(f"{name} 必须包含四个非负整数，得到 {result}")
    if any(item < 0 for item in result):
        raise ValueError(f"{name} 的每一项必须为非负数，得到 {result}")
    return result  # type: ignore[return-value]


def _rank_coordinate(rank: int, grid: Shape4) -> Shape4:
    remainder = int(rank)
    coordinate = []
    for axis in range(4):
        stride = prod(grid[axis + 1:])
        coordinate.append(remainder // stride)
        remainder %= stride
    return tuple(coordinate)  # type: ignore[return-value]


def _linear_rank(coordinate: Shape4, grid: Shape4) -> int:
    rank = 0
    for axis in range(4):
        value = int(coordinate[axis])
        if value < 0 or value >= grid[axis]:
            raise ValueError(
                f"rank_coordinate={coordinate} 超出 process_grid={grid}")
        rank = rank * grid[axis] + value
    return rank


def _context_stack() -> list[RollContext]:
    stack = getattr(_STATE, "stack", None)
    if stack is None:
        stack = []
        _STATE.stack = stack
    return stack


def current_context() -> Optional[RollContext]:
    """Return the innermost active context, or ``None``."""
    stack = _context_stack()
    return stack[-1] if stack else None


def activate(comm: Any = None, process_grid: Sequence[int] = (1, 1, 1, 1),
             local_extents: Optional[Sequence[int]] = None) -> RollContext:
    """Push a distributed roll context onto the current thread's stack."""
    communicator = _SerialCommunicator() if comm is None else comm
    if not hasattr(communicator, "Get_size"):
        raise TypeError("comm 必须提供 mpi4py 风格的 Get_size/Get_rank")
    if not hasattr(communicator, "Get_rank"):
        raise TypeError("comm 必须提供 mpi4py 风格的 Get_size/Get_rank")
    grid = _shape4(process_grid, "process_grid")
    size = int(communicator.Get_size())
    rank = int(communicator.Get_rank())
    if int(prod(grid)) != size:
        raise ValueError(
            f"process_grid={grid} 的乘积 {prod(grid)} != comm size {size}")
    if rank < 0 or rank >= size:
        raise ValueError(f"rank={rank} 超出 communicator size={size}")
    extents = (
        None if local_extents is None
        else _shape4(local_extents, "local_extents")
    )
    context = RollContext(
        comm=communicator,
        process_grid=grid,
        rank_coordinate=_rank_coordinate(rank, grid),
        local_extents=extents,
        size=size,
        rank=rank,
    )
    _context_stack().append(context)
    return context


def deactivate(context: Optional[RollContext] = None) -> None:
    """Pop the innermost context, optionally checking the exact token."""
    stack = _context_stack()
    if not stack:
        raise RuntimeError("没有活动的分布式 roll 上下文")
    if context is not None and stack[-1] is not context:
        raise RuntimeError("分布式 roll 上下文退出顺序不一致")
    stack.pop()


def set_local_extents(local_extents: Sequence[int]) -> None:
    """Update the active context for the current hierarchy level."""
    context = current_context()
    if context is None:
        raise RuntimeError("设置 local_extents 需要活动的分布式 roll 上下文")
    context.local_extents = _shape4(local_extents, "local_extents")


@contextmanager
def distributed_roll(
        comm: Any = None, process_grid: Sequence[int] = (1, 1, 1, 1),
        local_extents: Optional[Sequence[int]] = None
) -> Iterator[RollContext]:
    """Context-manager wrapper around :func:`activate`/:func:`deactivate`."""
    context = activate(comm, process_grid, local_extents)
    try:
        yield context
    finally:
        deactivate(context)


def global_shape(local_shape: Sequence[int]) -> Shape4:
    """Return the global shape for the active context, otherwise local shape."""
    shape = _shape4(local_shape, "local_shape")
    context = current_context()
    return shape if context is None else context.global_shape(shape)


def local_coordinate(global_coordinate: Sequence[int],
                     local_shape: Sequence[int]) -> Optional[Shape4]:
    """Map a global coordinate to the current rank, or return ``None``."""
    context = current_context()
    if context is None:
        coordinate = _coordinate4(global_coordinate, "global_coordinate")
        shape = _shape4(local_shape, "local_shape")
        if any(coordinate[axis] >= shape[axis] for axis in range(4)):
            return None
        return coordinate
    return context.local_coordinate(global_coordinate, local_shape)


def is_distributed() -> bool:
    context = current_context()
    return bool(context is not None and context.distributed)


def _face_tag(axis: int, face: str) -> int:
    if face == "low":
        return _LOW_FACE_TAG + int(axis)
    if face == "high":
        return _HIGH_FACE_TAG + int(axis)
    raise ValueError(f"未知 face={face!r}")


def face_tag(axis: int, face: str) -> int:
    """Return the stable tag used for one axis face exchange."""
    return _face_tag(axis, face)


def _neighbor_rank(context: RollContext, axis: int, delta: int) -> int:
    coordinate = list(context.rank_coordinate)
    coordinate[axis] = (
        int(coordinate[axis]) + int(delta)) % int(context.process_grid[axis])
    return _linear_rank(tuple(coordinate), context.process_grid)


def _face_slice(
        ndim: int, axis: int, index: int | slice) -> tuple[slice, ...]:
    dim = ndim - _SPACE_DIMS + int(axis)
    result = [slice(None)] * ndim
    result[dim] = index
    return tuple(result)


def _face_buffer(tensor: Any, axis: int, index: int) -> np.ndarray:
    value = tensor[_face_slice(tensor.ndim, axis, index)]
    value = value.detach()
    if hasattr(value, "resolve_conj"):
        value = value.resolve_conj()
    if hasattr(value, "resolve_neg"):
        value = value.resolve_neg()
    return np.ascontiguousarray(value.cpu().numpy())


def _tensor_from_buffer(buffer: np.ndarray, tensor: Any) -> Any:
    return _torch.as_tensor(
        buffer, dtype=tensor.dtype, device=tensor.device)


def _roll_axis_once(tensor: Any, axis: int, direction: int) -> Any:
    """Roll one position by ``direction`` on one distributed axis."""
    context = current_context()
    assert context is not None
    direction = 1 if direction >= 0 else -1
    local_extent = int(tensor.shape[tensor.ndim - _SPACE_DIMS + axis])
    if local_extent <= 0:
        raise ValueError("分布式 roll 的 local extent 必须为正数")
    result = tensor.clone()
    if direction > 0:
        if local_extent > 1:
            source = _face_slice(result.ndim, axis, slice(0, local_extent - 1))
            target = _face_slice(result.ndim, axis, slice(1, local_extent))
            result[target] = tensor[source]
    else:
        if local_extent > 1:
            source = _face_slice(result.ndim, axis, slice(1, local_extent))
            target = _face_slice(result.ndim, axis, slice(0, local_extent - 1))
            result[target] = tensor[source]
    low = _face_buffer(tensor, axis, 0)
    high = _face_buffer(tensor, axis, local_extent - 1)
    recv_low = np.zeros_like(low)
    recv_high = np.zeros_like(high)
    previous = _neighbor_rank(context, axis, -1)
    following = _neighbor_rank(context, axis, +1)

    if direction > 0:
        # Rank r sends its original high face to r+1 and receives r-1's
        # original high face into its low ghost.
        context.comm.Sendrecv(
            sendbuf=high,
            dest=following,
            sendtag=face_tag(axis, "high"),
            recvbuf=recv_low,
            source=previous,
            recvtag=face_tag(axis, "high"),
        )
        result[_face_slice(result.ndim, axis, 0)] = _tensor_from_buffer(
            recv_low, result)
    else:
        # Rank r sends its original low face to r-1 and receives r+1's
        # original low face into its high ghost.
        context.comm.Sendrecv(
            sendbuf=low,
            dest=previous,
            sendtag=face_tag(axis, "low"),
            recvbuf=recv_high,
            source=following,
            recvtag=face_tag(axis, "low"),
        )
        result[
            _face_slice(result.ndim, axis, local_extent - 1)
        ] = _tensor_from_buffer(recv_high, result)
    return result


def roll(tensor: Any, shifts: int | Sequence[int],
         dims: int | Sequence[int]) -> Any:
    """Global periodic roll with ``torch.roll``-equivalent semantics.

    The final four axes are interpreted as ``x,y,z,t``.  Calls with no active
    context, with a single rank, or on an axis whose process-grid extent is
    one fall back exactly to :func:`pyqcu.cann.roll`.
    """
    if not hasattr(tensor, "ndim"):
        raise TypeError("roll 需要 torch-like tensor")
    shift_values = (shifts,) if isinstance(shifts, Integral) else tuple(shifts)
    dim_values = (dims,) if isinstance(dims, Integral) else tuple(dims)
    if len(shift_values) != len(dim_values):
        raise ValueError("shifts 与 dims 长度必须一致")
    if not shift_values:
        return tensor

    context = current_context()
    result = tensor
    for raw_dim, raw_shift in zip(dim_values, shift_values):
        ndim = int(result.ndim)
        dim = int(raw_dim)
        if dim < 0:
            dim += ndim
        if dim < 0 or dim >= ndim:
            raise IndexError(f"roll dim={raw_dim} 超出 ndim={ndim}")
        shift = int(raw_shift)
        axis = dim - (ndim - _SPACE_DIMS)
        if (context is None or not context.distributed or
                axis < 0 or axis >= _SPACE_DIMS or
                int(context.process_grid[axis]) == 1):
            result = _torch.roll(result, shifts=shift, dims=dim)
            continue

        local_extent = int(result.shape[dim])
        global_extent = local_extent * int(context.process_grid[axis])
        count = abs(shift) % global_extent
        direction = 1 if shift >= 0 else -1
        for _ in range(count):
            result = _roll_axis_once(result, axis, direction)
    return result


__all__ = [
    "RollContext",
    "activate",
    "current_context",
    "deactivate",
    "distributed_roll",
    "face_tag",
    "global_shape",
    "is_distributed",
    "local_coordinate",
    "roll",
    "set_local_extents",
]
