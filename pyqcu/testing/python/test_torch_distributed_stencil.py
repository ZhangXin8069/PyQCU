"""Distributed Torch Legacy stencil regressions for one, two, or four ranks."""

from __future__ import annotations

from mpi4py import MPI

import pyqcu.cann as _torch
from pyqcu.torch.solver._legacy_multigrid import _CachedLegacyCoarseOperator


def test_cached_legacy_stencil_uses_remote_periodic_face() -> None:
    comm = MPI.COMM_WORLD
    size = int(comm.Get_size())
    if size == 1:
        return
    if size not in (2, 4):
        raise RuntimeError("run this regression with one, two, or four ranks")
    rank = int(comm.Get_rank())
    process_grid = (2, 1, 1, 1) if size == 2 else (2, 2, 1, 1)
    local_extents = (
        2, 1 if size == 2 else 2, 1, 1)
    shape = (1, *local_extents)
    rank_x = rank // (process_grid[1] * process_grid[2] * process_grid[3])

    cpu = _torch.device("cpu")
    source = _torch.zeros(*shape, dtype=_torch.complex64, device=cpu)
    for local_x in range(local_extents[0]):
        global_x = rank_x * local_extents[0] + local_x
        source[0, local_x, 0, 0, 0] = complex(
            global_x + 1, 0.0)

    hop_nn = _torch.zeros(
        (2, 4, 1, 1, *local_extents),
        dtype=_torch.complex64, device=cpu)
    hop_nn[1, 0, 0, 0, :, :, :, :] = 1.0
    hop_diag = _torch.zeros(
        (2, 2, 6, 1, 1, *local_extents),
        dtype=_torch.complex64, device=cpu)
    sitting = _torch.zeros(
        (1, 1, *local_extents), dtype=_torch.complex64, device=cpu)

    operator = _CachedLegacyCoarseOperator(
        hop_nn, hop_diag, sitting, comm=comm,
        process_grid=process_grid, local_extents=local_extents)
    result = operator.matvec(source)

    global_volume = local_extents[0] * process_grid[0]
    for local_x in range(local_extents[0]):
        global_x = rank_x * local_extents[0] + local_x
        expected = (global_x - 1) % global_volume + 1
        assert result[0, local_x, 0, 0, 0].real.item() == float(expected)


if __name__ == "__main__":
    test_cached_legacy_stencil_uses_remote_periodic_face()
    if int(MPI.COMM_WORLD.Get_rank()) == 0:
        print("torch distributed stencil: PASS")
