"""CPU tests for distributed Strict-MG setup primitives."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from mpi4py import MPI

from pyqcu import dslash
from pyqcu.tools import _mpi_roll
from pyqcu.tools._distributed_setup import (
    DistributedFineOperator,
    halo_pad_last4,
)
from pyqcu.tools._strict_galerkin import build_strict_galerkin
from pyqcu.solver import QudaTransfer


class _SerialRingComm:
    """Two-rank mpi4py-like communicator for a deterministic CPU test.

    The two ranks are executed sequentially.  ``world`` holds the local
    tensor owned by each rank, which is enough to model the simultaneous
    ``Sendrecv`` pair used by the face exchange.
    """

    def __init__(self, rank: int, world: dict[int, torch.Tensor]):
        self.rank = int(rank)
        self.world = world

    def Get_size(self) -> int:
        return 2

    def Get_rank(self) -> int:
        return self.rank

    @staticmethod
    def _face(tag: int) -> tuple[int, str]:
        for axis in range(4):
            if tag == _mpi_roll.face_tag(axis, "low"):
                return axis, "low"
            if tag == _mpi_roll.face_tag(axis, "high"):
                return axis, "high"
        raise AssertionError(f"unknown face tag {tag}")

    @staticmethod
    def _slice(tensor: torch.Tensor, axis: int, face: str):
        slices = [slice(None)] * tensor.ndim
        slices[tensor.ndim - 4 + axis] = 0 if face == "low" else -1
        return tuple(slices)

    def Sendrecv(self, *, sendbuf, dest, sendtag, recvbuf, source, recvtag):
        send_axis, send_face = self._face(sendtag)
        recv_axis, recv_face = self._face(recvtag)
        assert send_axis == recv_axis
        send_owner = self.world[self.rank]
        expected = send_owner[self._slice(send_owner, send_axis, send_face)]
        np.testing.assert_allclose(sendbuf, expected.numpy())
        peer_axis, peer_face = self._face(recvtag)
        assert peer_axis == recv_axis
        peer = self.world[int(source)]
        np.copyto(
            recvbuf,
            peer[self._slice(peer, peer_axis, peer_face)].cpu().numpy())


def test_roll_without_context_is_exact_torch_roll():
    value = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    for shift in (-7, -1, 0, 1, 4, 9):
        assert torch.equal(
            _mpi_roll.roll(value, shifts=shift, dims=1),
            torch.roll(value, shifts=shift, dims=1),
        )


def test_global_roll_wraps_two_serial_ranks_on_both_faces():
    local = {
        0: torch.arange(2, dtype=torch.int64).reshape(2, 1, 1, 1),
        1: torch.arange(2, 4, dtype=torch.int64).reshape(2, 1, 1, 1),
    }
    for rank in (0, 1):
        comm = _SerialRingComm(rank, local)
        with _mpi_roll.distributed_roll(
                comm, process_grid=(2, 1, 1, 1),
                local_extents=(2, 1, 1, 1)):
            positive = _mpi_roll.roll(
                local[rank], shifts=1, dims=0)
            negative = _mpi_roll.roll(
                local[rank], shifts=-1, dims=0)
        global_value = torch.arange(4, dtype=torch.int64).reshape(4, 1, 1, 1)
        expected_positive = torch.roll(
            global_value, shifts=1, dims=0)[rank * 2:(rank + 1) * 2]
        expected_negative = torch.roll(
            global_value, shifts=-1, dims=0)[rank * 2:(rank + 1) * 2]
        expected_positive = expected_positive.reshape(2, 1, 1, 1)
        expected_negative = expected_negative.reshape(2, 1, 1, 1)
        assert torch.equal(positive, expected_positive)
        assert torch.equal(negative, expected_negative)


def test_non_decomposed_axes_use_local_periodic_roll():
    value = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    comm = _SerialRingComm(0, {0: value, 1: value})
    with _mpi_roll.distributed_roll(
            comm, process_grid=(2, 1, 1, 1),
            local_extents=(2, 3, 4, 5)):
        for axis in (1, 2, 3):
            assert torch.equal(
                _mpi_roll.roll(value, shifts=1, dims=axis),
                torch.roll(value, shifts=1, dims=axis),
            )


def test_context_cleanup_restores_previous_context_after_exception():
    value = torch.arange(12)
    comm = _SerialRingComm(0, {0: value, 1: value})
    with _mpi_roll.distributed_roll(
            comm, process_grid=(2, 1, 1, 1),
            local_extents=(12, 1, 1, 1)):
        outer = _mpi_roll.current_context()
        assert outer is not None
        with pytest.raises(RuntimeError, match="fixture"):
            with _mpi_roll.distributed_roll(
                    comm, process_grid=(2, 1, 1, 1),
                    local_extents=(12, 1, 1, 1)):
                assert _mpi_roll.current_context() is not outer
                raise RuntimeError("fixture")
        assert _mpi_roll.current_context() is outer
    assert _mpi_roll.current_context() is None


def test_halo_pad_matches_global_periodic_reference():
    global_value = torch.arange(4, dtype=torch.float64).reshape(4, 1, 1, 1)
    local_values = {
        0: global_value[0:2].clone(),
        1: global_value[2:4].clone(),
    }
    for rank in (0, 1):
        comm = _SerialRingComm(rank, local_values)
        padded = halo_pad_last4(
            local_values[rank],
            local_shape=(2, 1, 1, 1),
            process_grid=(2, 1, 1, 1),
            comm=comm,
        )
        start = rank * 2
        expected_x = torch.stack([
            global_value[(start - 1) % 4],
            global_value[(start + 0) % 4],
            global_value[(start + 1) % 4],
            global_value[(start + 2) % 4],
        ]).reshape(4, 1, 1, 1)
        assert torch.equal(padded, expected_x)


@pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() != 1,
    reason="single-process global reference test")
def test_distributed_fine_operator_matches_global_periodic_operator():
    torch.manual_seed(20260917)
    global_shape = (4, 2, 2, 2)
    local_shape = (2, 2, 2, 2)
    global_U = torch.zeros(3, 3, 4, *global_shape, dtype=torch.complex64)
    for color in range(3):
        global_U[color, color] = 1.0
    global_U = global_U + 0.01 * torch.randn_like(global_U)
    global_field = torch.randn(
        12, *global_shape, dtype=torch.complex64)
    reference = dslash.operator(
        U=global_U, kappa=torch.tensor([0.1]), u_0=torch.ones(1),
        verbose=False).matvec(global_field)

    local_values = {
        0: global_U[..., 0:2, :, :, :].clone(),
        1: global_U[..., 2:4, :, :, :].clone(),
    }
    local_fields = {
        0: global_field[..., 0:2, :, :, :].clone(),
        1: global_field[..., 2:4, :, :, :].clone(),
    }
    for rank in (0, 1):
        comm = _SerialRingComm(rank, local_values)
        operator = DistributedFineOperator(
            U=local_values[rank],
            clover_term=None,
            kappa=torch.tensor([0.1]),
            u_0=torch.ones(1),
            lat_size=local_shape,
            process_grid=(2, 1, 1, 1),
            comm=comm,
            verbose=False,
        )
        comm.world = local_fields
        result = operator.matvec(local_fields[rank])
        expected = reference[..., rank * 2:(rank + 1) * 2, :, :, :]
        torch.testing.assert_close(result, expected, rtol=2e-5, atol=2e-5)


def test_distributed_galerkin_rejects_unmarked_local_operator():
    shape = (4, 4, 4, 4)
    null = torch.randn(2, 4, 3, *shape, dtype=torch.complex64)
    transfer = QudaTransfer(
        null, shape, fine_spin=4, fine_color=3,
        coarse_spin=2, block_size=(2, 2, 2, 2), verbose=False)
    comm = _SerialRingComm(0, {0: torch.empty(1), 1: torch.empty(1)})
    with _mpi_roll.distributed_roll(
            comm, process_grid=(2, 1, 1, 1),
            local_extents=shape):
        with pytest.raises(RuntimeError, match="分布式 fine 算子"):
            build_strict_galerkin(
                transfer, lambda value: value,
                site_batch_size=1, check_fine_support=False,
                verbose=False)
