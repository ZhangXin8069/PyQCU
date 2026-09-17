"""Halo-backed fine operators for distributed Strict-MG setup.

The fine operator in the Python Strict-MG reference is a nearest-neighbour
stencil.  If every rank extends its local lattice by one ghost layer only on
the decomposed directions, fills those layers with the true neighbour data,
and applies the ordinary periodic operator on that padded lattice, then the
restriction of the result to the original local volume is exactly the action
of the distributed fine operator.

The argument is local: a nearest-neighbour stencil reads only the point itself
and its four forward/backward neighbours.  Therefore every local output entry
depends only on the local volume and the corresponding adjacent ghost faces;
the padded periodic boundary cannot reach the local volume.  Onsite Clover
terms do not need a ghost value at all, but the padded tensor keeps an
exchange-filled ghost layer so that the operator shape and dtype stay uniform.

No global tensor is gathered.  Gauge, Clover, and source fields are exchanged
with face ``Sendrecv`` operations in :mod:`_mpi_roll`, and the padded operator
is constructed once and cached.
"""

from __future__ import annotations

from math import prod
from typing import Any, Optional, Sequence, Tuple

import numpy as np

import pyqcu.cann as _torch
from pyqcu import dslash

from . import _mpi_roll


Shape4 = Tuple[int, int, int, int]


def _shape4(value: Sequence[int], name: str) -> Shape4:
    try:
        result = tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} 必须包含四个正整数") from exc
    if len(result) != 4:
        raise ValueError(f"{name} 必须包含四个正整数，得到 {result}")
    if any(item <= 0 for item in result):
        raise ValueError(f"{name} 的每一项必须为正数，得到 {result}")
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
        if coordinate[axis] < 0 or coordinate[axis] >= grid[axis]:
            raise ValueError(
                f"rank_coordinate={coordinate} 超出 process_grid={grid}")
        rank = rank * grid[axis] + coordinate[axis]
    return rank


def _neighbor_rank(coordinate: Shape4, grid: Shape4, axis: int,
                   delta: int) -> int:
    value = list(coordinate)
    value[axis] = (value[axis] + int(delta)) % int(grid[axis])
    return _linear_rank(tuple(value), grid)


def _face_slice(ndim: int, axis: int, index: int) -> tuple[slice, ...]:
    result = [slice(None)] * ndim
    result[ndim - 4 + axis] = int(index)
    return tuple(result)


def _face_buffer(tensor: Any, axis: int, index: int) -> np.ndarray:
    value = tensor[_face_slice(tensor.ndim, axis, index)].detach()
    if hasattr(value, "resolve_conj"):
        value = value.resolve_conj()
    if hasattr(value, "resolve_neg"):
        value = value.resolve_neg()
    return np.ascontiguousarray(value.cpu().numpy())


def _buffer_tensor(buffer: np.ndarray, tensor: Any) -> Any:
    return _torch.as_tensor(
        buffer, dtype=tensor.dtype, device=tensor.device)


def _exchange_padded_axis(tensor: Any, axis: int, local_shape: Shape4,
                          grid: Shape4, coordinate: Shape4,
                          comm: Any) -> None:
    """Fill both ghost layers along one decomposed axis in place."""
    extent = int(local_shape[axis])
    low_index = 1
    high_index = 1 + extent - 1
    low = _face_buffer(tensor, axis, low_index)
    high = _face_buffer(tensor, axis, high_index)
    recv_low = np.zeros_like(low)
    recv_high = np.zeros_like(high)
    previous = _neighbor_rank(coordinate, grid, axis, -1)
    following = _neighbor_rank(coordinate, grid, axis, +1)

    # Exchange the high faces first: the previous rank's high face fills our
    # low ghost.  The low-face exchange then fills our high ghost.
    comm.Sendrecv(
        sendbuf=high,
        dest=following,
        sendtag=_mpi_roll.face_tag(axis, "high"),
        recvbuf=recv_low,
        source=previous,
        recvtag=_mpi_roll.face_tag(axis, "high"),
    )
    tensor[_face_slice(tensor.ndim, axis, 0)] = _buffer_tensor(
        recv_low, tensor)
    comm.Sendrecv(
        sendbuf=low,
        dest=previous,
        sendtag=_mpi_roll.face_tag(axis, "low"),
        recvbuf=recv_high,
        source=following,
        recvtag=_mpi_roll.face_tag(axis, "low"),
    )
    tensor[_face_slice(tensor.ndim, axis, 1 + extent)] = _buffer_tensor(
        recv_high, tensor)


def halo_pad_last4(tensor: Any, local_shape: Sequence[int],
                   process_grid: Sequence[int], comm: Any,
                   *, rank: Optional[int] = None) -> Any:
    """Pad only decomposed spacetime axes and exchange their ghost faces.

    正确性论证：时空是最后四轴，Dirac 算子是最近邻模板；padded 格点上
    按周期边界作用后，本地体积内每个输出只读取本地点和一层 ghost。把
    ghost 填成真实邻 rank 的 face 后，结果限制回本地体积，逐点等于分布式
    算子的 action。padded 边界自身如何周期闭合不影响本地体积，因此无须
    allgather 整个场。
    """
    shape = _shape4(local_shape, "local_shape")
    grid = _shape4(process_grid, "process_grid")
    if tuple(int(x) for x in tensor.shape[-4:]) != shape:
        raise ValueError(
            f"halo tensor 的末四轴应为 {shape}，"
            f"得到 {tuple(tensor.shape[-4:])}")
    if int(prod(grid)) != int(comm.Get_size()):
        raise ValueError(
            f"process_grid={grid} 的乘积 {prod(grid)} != "
            f"comm size {comm.Get_size()}")
    rank = int(comm.Get_rank()) if rank is None else int(rank)
    coordinate = _rank_coordinate(rank, grid)

    padded_shape = list(tensor.shape)
    for axis, width in enumerate(grid):
        if width > 1:
            padded_shape[-4 + axis] += 2
    padded = _torch.zeros(
        size=padded_shape, dtype=tensor.dtype, device=tensor.device)
    slices = [slice(None)] * tensor.ndim
    for axis, width in enumerate(grid):
        start = 1 if width > 1 else 0
        slices[-4 + axis] = slice(start, start + shape[axis])
    padded[tuple(slices)] = tensor

    for axis, width in enumerate(grid):
        if width > 1:
            _exchange_padded_axis(
                padded, axis, shape, grid, coordinate, comm)
    return padded


def _crop_last4(tensor: Any, local_shape: Shape4,
                process_grid: Shape4) -> Any:
    slices = [slice(None)] * tensor.ndim
    for axis, width in enumerate(process_grid):
        start = 1 if width > 1 else 0
        slices[-4 + axis] = slice(start, start + local_shape[axis])
    return tensor[tuple(slices)].contiguous()


class DistributedFineOperator:
    """Nearest-neighbour fine dslash with explicit periodic ghost faces.

    The wrapper avoids the MPI path built into :class:`pyqcu.dslash.hopping`.
    The hopping matrices are rebuilt against the padded gauge field and the
    underlying operator is switched to a single-rank periodic action.  This
    is deliberate: the supplied ``process_grid`` may differ from the
    auto-factorised ``tools.give_grid_size()`` mapping, so relying on the
    existing hopping communicator would exchange the wrong axis.
    """

    def __init__(
        self,
        *,
        U: Any,
        clover_term: Any = None,
        kappa: Any = None,
        u_0: Any = None,
        lat_size: Sequence[int],
        process_grid: Sequence[int],
        comm: Any,
        verbose: bool = False,
    ):
        self.local_shape = _shape4(lat_size, "lat_size")
        self.process_grid = _shape4(process_grid, "process_grid")
        self.comm = comm
        if int(prod(self.process_grid)) != int(comm.Get_size()):
            raise ValueError(
                f"process_grid={self.process_grid} 的乘积 "
                f"{prod(self.process_grid)} != comm size {comm.Get_size()}")
        if tuple(int(x) for x in U.shape[-4:]) != self.local_shape:
            raise ValueError(
                f"U 的末四轴应为 {self.local_shape}，"
                f"得到 {tuple(U.shape[-4:])}")
        if clover_term is not None and tuple(
                int(x) for x in clover_term.shape[-4:]) != self.local_shape:
            raise ValueError(
                f"clover_term 的末四轴应为 {self.local_shape}，"
                f"得到 {tuple(clover_term.shape[-4:])}")
        self.rank_coordinate = _rank_coordinate(
            int(comm.Get_rank()), self.process_grid)
        self.padded_shape = tuple(
            self.local_shape[axis] +
            (2 if self.process_grid[axis] > 1 else 0)
            for axis in range(4)
        )

        padded_U = halo_pad_last4(
            U, self.local_shape, self.process_grid, comm)
        padded_clover = (
            None if clover_term is None else
            halo_pad_last4(
                clover_term, self.local_shape, self.process_grid, comm)
        )
        # Constructing with U=None is important: the standard hopping
        # constructor would perform MPI exchanges using the global MPI rank
        # factorisation, which need not equal the requested process_grid.
        self._operator = dslash.operator(
            U=None,
            clover_term=padded_clover,
            kappa=kappa,
            u_0=u_0,
            verbose=verbose,
        )
        hopper = self._operator.hopping
        hopper.U = padded_U
        hopper.grid_size = [1, 1, 1, 1]
        hopper.M_plus_list = [
            dslash.give_hopping_plus(
                ward=ward, U=padded_U, kappa=kappa, u_0=u_0,
                verbose=verbose)
            for ward in range(4)
        ]
        hopper.M_minus_list = [
            dslash.give_hopping_minus(
                ward=ward, U=padded_U, U_head=None, kappa=kappa,
                u_0=u_0, verbose=verbose)
            for ward in range(4)
        ]
        self.is_distributed = any(value > 1 for value in self.process_grid)
        self.dof = 12
        self.spin = 4
        self.color = 3
        self.shape = self.local_shape

    @property
    def padded_operator(self) -> Any:
        return self._operator

    def _validate_value(self, value: Any) -> None:
        if tuple(int(x) for x in value.shape[-4:]) != self.local_shape:
            raise ValueError(
                f"fine field 的末四轴应为 {self.local_shape}，"
                f"得到 {tuple(value.shape[-4:])}")

    def apply(self, value: Any) -> Any:
        self._validate_value(value)
        padded = halo_pad_last4(
            value, self.local_shape, self.process_grid, self.comm)
        image = self._operator.matvec(padded)
        return _crop_last4(image, self.local_shape, self.process_grid)

    matvec = apply

    def batch_apply(self, value: Any) -> Any:
        if value.ndim < 5:
            raise ValueError(
                "batch fine field 至少需要 batch、DOF 与四个时空轴")
        self._validate_value(value)
        padded = halo_pad_last4(
            value, self.local_shape, self.process_grid, self.comm)
        image = self._operator.matvec_batch(padded)
        return _crop_last4(image, self.local_shape, self.process_grid)

    matvec_batch = batch_apply
    batch_matvec = batch_apply

    def memory_report(self) -> dict[str, int]:
        """Return setup-only padded storage and face-buffer estimates."""
        padded_elements = prod(self.padded_shape)
        padded_gauge = 3 * 3 * 4 * padded_elements
        padded_clover = 0
        clover = getattr(self._operator.sitting, "clover_term", None)
        if clover is not None:
            padded_clover = int(clover.numel())
        face_elements = sum(
            prod(self.padded_shape) // self.padded_shape[axis]
            for axis, width in enumerate(self.process_grid)
            if width > 1
        )
        return {
            "padded_gauge_elements": int(padded_gauge),
            "padded_clover_elements": int(padded_clover),
            "face_staging_elements_per_axis": int(face_elements),
        }


__all__ = [
    "DistributedFineOperator",
    "halo_pad_last4",
]
