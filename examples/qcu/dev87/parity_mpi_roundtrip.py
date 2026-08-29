"""MPI 奇偶压缩布局的切分/重建回环测试。

该测试专门守护 ``global_parity_to_local`` 的边界约定：全局奇偶布局不能
直接按最后的 ``T/2`` 轴切块，因为每个奇偶页的压缩索引依赖物理坐标。
测试先恢复全局物理格点，再按 rank 的物理 block 切分，最后将各 rank
重建回全局并重新压缩；这样可以同时覆盖 X/Y/Z/T 方向的 rank 原点。

示例（不需要 CUDA）：
    mpirun --allow-run-as-root --oversubscribe -np 2 \
      python examples/qcu/dev87/parity_mpi_roundtrip.py --grid 2 1 1 1
    mpirun --allow-run-as-root --oversubscribe -np 4 \
      python examples/qcu/dev87/parity_mpi_roundtrip.py --grid 1 1 2 2
"""

import argparse

import numpy as np
import torch
from mpi4py import MPI

from pyqcu import tools

from common import global_parity_to_local, local_geometry, process_grid


def _make_global_parity(lat, dtype):
    X, Y, Z, T = (int(v) for v in lat)
    if T % 2:
        raise ValueError(f"global T must be even for parity layout: {lat}")
    shape = (2, 4, 3, X, Y, Z, T // 2)
    count = int(np.prod(shape))
    # Use a coordinate-dependent, non-symmetric field so an incorrect block
    # origin or parity page cannot cancel in the norm check.
    index = torch.arange(count, dtype=torch.float64).reshape(shape)
    value = torch.sin(0.013 * index + 0.17) + 1j * torch.cos(
        0.019 * index - 0.11)
    return value.to(dtype=dtype).contiguous()


def _physical_slice(starts, sizes):
    # ``poooxyzt2oooxyzt`` removes the parity axis; the physical field has
    # only the spin/color prefix ``[4, 3]``.
    return (slice(None),) * 2 + tuple(
        slice(int(starts[d]), int(starts[d] + sizes[d])) for d in range(4)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat", type=int, nargs=4, default=[8, 8, 8, 16])
    parser.add_argument("--grid", type=int, nargs=4, required=True)
    parser.add_argument("--dtype", choices=("c64", "c128"), default="c64")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    grid = process_grid(args.grid)
    if int(np.prod(grid)) != comm.Get_size():
        raise ValueError(
            f"grid={grid} has {int(np.prod(grid))} ranks, "
            f"MPI world has {comm.Get_size()}"
        )
    dtype = torch.complex64 if args.dtype == "c64" else torch.complex128
    lat = [int(v) for v in args.lat]
    local_lat, starts, _ = local_geometry(
        lat, grid=grid, rank=rank, require_even=True
    )

    global_parity = _make_global_parity(lat, dtype)
    local_parity = global_parity_to_local(
        global_parity, lat, grid=grid, rank=rank, device="cpu", dtype=dtype
    )
    expected_local_shape = (2, 4, 3, *local_lat[:3], local_lat[3] // 2)
    if tuple(local_parity.shape) != expected_local_shape:
        raise AssertionError(
            f"rank {rank}: local parity shape {tuple(local_parity.shape)} != "
            f"{expected_local_shape}"
        )

    # First check the local conversion independently of MPI gathering.
    local_again = tools.oooxyzt2poooxyzt(
        tools.poooxyzt2oooxyzt(local_parity)
    )
    local_diff = (local_again - local_parity).reshape(-1)
    local_max = float(local_diff.abs().max().item())

    # Gather physical blocks, not compressed pages.  The latter is precisely
    # the unsafe operation this test is intended to catch.
    local_full = tools.poooxyzt2oooxyzt(local_parity).contiguous()
    blocks = comm.gather((rank, local_full), root=0)
    if rank == 0:
        global_full = torch.empty((4, 3, *lat), dtype=dtype, device="cpu")
        for block_rank, block in blocks:
            block_lat, block_starts, _ = local_geometry(
                lat, grid=grid, rank=block_rank, require_even=True
            )
            global_full[_physical_slice(block_starts, block_lat)] = block
        reconstructed = tools.oooxyzt2poooxyzt(global_full)
        diff = (reconstructed - global_parity).reshape(-1)
        denominator = max(float(global_parity.abs().max().item()), 1e-30)
        rel_max = float(diff.abs().max().item()) / denominator
        global_max = float(diff.abs().max().item())
        all_local_max = comm.gather(local_max, root=0)
        result = {
            "grid": grid,
            "lat": lat,
            "local_lat": local_lat,
            "dtype": args.dtype,
            "local_max_abs": max(all_local_max),
            "global_max_abs": global_max,
            "global_rel_max": rel_max,
        }
        print(result, flush=True)
        tolerance = 2e-6 if dtype == torch.complex64 else 2e-14
        if rel_max >= tolerance or max(all_local_max) >= tolerance:
            raise AssertionError(f"MPI parity round-trip failed: {result}")
    else:
        comm.gather(local_max, root=0)


if __name__ == "__main__":
    main()
