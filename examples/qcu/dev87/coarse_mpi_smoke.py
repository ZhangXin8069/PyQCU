"""分布式 33 点粗算子通信 smoke test。

该脚本不依赖 HDF5 缓存：它构造一个确定性的全局 stencil/vector，切成
MPI rank-local block 后调用 C++ ``applyMultigridCoarseDslashWideQcu``，
最后与 Python 周期参考逐元素比较。因而可单独覆盖 32 方向 halo、周期
自邻居、host-staging、CUDA-aware MPI 选择以及 interior/boundary overlap。

示例（从仓库根目录执行）：
    source ./env.sh
    PYQCU_MPI_DEVICE_AWARE=0 PYQCU_MPI_OVERLAP=0 \
      mpirun --allow-run-as-root --oversubscribe -np 2 \
      python examples/qcu/dev87/coarse_mpi_smoke.py
    PYQCU_MPI_DEVICE_AWARE=0 PYQCU_MPI_OVERLAP=1 \
      mpirun --allow-run-as-root --oversubscribe -np 2 \
      python examples/qcu/dev87/coarse_mpi_smoke.py

``PYQCU_MPI_DEVICE_AWARE=1`` 可用于验证“请求设备感知、能力不足时安全
回退 pinned-host”的策略；实际是否走 device MPI 由编译期 MPI 能力决定。
"""

import argparse
import os

import torch
from mpi4py import MPI

from pyqcu import tools
from pyqcu.cuda import define, qcu


def _slice_last4(starts, sizes):
    return (slice(None),) + tuple(
        slice(int(starts[d]), int(starts[d] + sizes[d])) for d in range(4)
    )


def _rank_coord(rank, grid):
    # This is the row-major rank map used by LatticeSet::init().
    coord = [0, 0, 0, 0]
    value = int(rank)
    for d in range(3, -1, -1):
        coord[d] = value % int(grid[d])
        value //= int(grid[d])
    return coord


def _make_global_data(E, dims):
    """Return deterministic CPU tensors and an input vector."""
    X, Y, Z, T = dims
    volume = X * Y * Z * T
    # Keep every coefficient in complex64: mixing a float64 site coordinate
    # into the broadcast would promote the stencil to complex128 on CPU.
    site = torch.arange(volume, dtype=torch.float32).reshape(1, X, Y, Z, T)
    channel = torch.arange(E, dtype=torch.float32).reshape(E, 1, 1, 1, 1)
    x = (torch.sin(0.017 * site + 0.13 * channel) +
         1j * torch.cos(0.023 * site - 0.07 * channel)).to(torch.complex64)

    eye = torch.eye(E, dtype=torch.complex64).reshape(E, E, 1, 1, 1, 1)
    site6 = site.reshape(1, 1, X, Y, Z, T)
    sit = eye * (0.7 + 0.001 * site6)

    hop_nn = torch.empty((2, 4, E, E, X, Y, Z, T),
                         dtype=torch.complex64)
    for pm in range(2):
        for direction in range(4):
            coefficient = (0.011 * (1 + direction) +
                           0.003 * (1 + pm))
            hop_nn[pm, direction] = eye * coefficient

    hop_diag = torch.empty((2, 2, 6, E, E, X, Y, Z, T),
                           dtype=torch.complex64)
    for s1 in range(2):
        for s2 in range(2):
            for pair in range(6):
                coefficient = (0.001 * (1 + pair) +
                               0.0002 * (1 + s1) +
                               0.0001 * (1 + s2))
                hop_diag[s1, s2, pair] = eye * coefficient
    return x.contiguous(), sit.contiguous(), hop_nn.contiguous(), hop_diag.contiguous()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, nargs=4, default=[2, 1, 1, 1])
    parser.add_argument("--E", type=int, default=2)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    grid = [int(v) for v in args.grid]
    if len(grid) != 4 or any(v <= 0 for v in grid):
        raise ValueError(f"invalid process grid: {grid}")
    if size != int(torch.tensor(grid).prod().item()):
        raise ValueError(f"MPI size={size} does not match grid={grid}")

    # The checked-in test build deliberately enables
    # _TEST_SINGLE_GPU_MULTI_RANK_, so C++ binds every MPI rank to device 0.
    # Match that contract here; otherwise rank-local tensors would be created
    # on a different GPU from the one used by the C++ LatticeSet.
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the coarse MPI smoke test")
    device_id = 0
    torch.cuda.set_device(device_id)
    device = torch.device("cuda", device_id)

    E = int(args.E)
    global_dims = [4, 2, 2, 2]
    if any(global_dims[d] % grid[d] for d in range(4)):
        raise ValueError(f"global dimensions {global_dims} not divisible by {grid}")
    local_dims = [global_dims[d] // grid[d] for d in range(4)]
    coord = _rank_coord(rank, grid)
    starts = [coord[d] * local_dims[d] for d in range(4)]
    local_slice = _slice_last4(starts, local_dims)

    x_global, sit_global, hnn_global, hdg_global = _make_global_data(
        E, global_dims)
    reference = tools.apply_stencil(hnn_global, hdg_global, sit_global,
                                    x_global)

    x_local = x_global[local_slice].to(device=device).contiguous()
    sit_local = sit_global[(slice(None), slice(None)) + tuple(
        slice(starts[d], starts[d] + local_dims[d]) for d in range(4)
    )].to(device=device).contiguous()
    hnn_local = hnn_global[(slice(None), slice(None), slice(None), slice(None)) + tuple(
        slice(starts[d], starts[d] + local_dims[d]) for d in range(4)
    )].to(device=device).contiguous()
    hdg_local = hdg_global[(slice(None), slice(None), slice(None),
                            slice(None), slice(None)) + tuple(
        slice(starts[d], starts[d] + local_dims[d]) for d in range(4)
    )].to(device=device).contiguous()
    y_local = torch.empty_like(x_local)

    params = define.params.clone()
    argv = define.argv.clone()
    set_ptrs = define.set_ptrs.clone()
    params[define._LAT_X_] = local_dims[0]
    params[define._LAT_Y_] = local_dims[1]
    params[define._LAT_Z_] = local_dims[2]
    # LatticeSet's checkerboard communication geometry uses the full T slot;
    # the public coarse vector itself uses the level-1 T slot below.
    params[define._LAT_T_] = 2 * local_dims[3]
    params[define._LAT_XYZT_] = 2 * int(torch.tensor(local_dims).prod().item())
    params[define._GRID_X_] = grid[0]
    params[define._GRID_Y_] = grid[1]
    params[define._GRID_Z_] = grid[2]
    params[define._GRID_T_] = grid[3]
    params[define._NODE_RANK_] = rank
    params[define._NODE_SIZE_] = size
    params[define._DATA_TYPE_] = define._LAT_C64_
    params[define._SET_INDEX_] = 0
    params[define._SET_PLAN_] = 1
    params[define._VERBOSE_] = 0
    params[define._MG_NUM_LEVEL_] = 2
    params[define._MG_LEVEL1_E_] = E
    params[define._MG_LEVEL1_X_] = local_dims[0]
    params[define._MG_LEVEL1_Y_] = local_dims[1]
    params[define._MG_LEVEL1_Z_] = local_dims[2]
    params[define._MG_LEVEL1_T_] = local_dims[3]
    params[define._MAX_ITER_] = 1
    argv[define._MASS_] = 0.0
    argv[define._ATOL_] = 1e-6

    initialized = False
    try:
        qcu.applyInitQcu(set_ptrs, params, argv)
        initialized = True
        qcu.applyMultigridCoarseDslashWideQcu(
            y_local, x_local, sit_local, hnn_local, hdg_local,
            set_ptrs, params)
        torch.cuda.synchronize()
        blocks = comm.gather((rank, y_local.cpu()), root=0)
    finally:
        if initialized:
            qcu.applyEndQcu(set_ptrs, params)

    if rank == 0:
        joined = torch.empty_like(reference)
        for block_rank, block in blocks:
            block_coord = _rank_coord(block_rank, grid)
            block_starts = [block_coord[d] * local_dims[d] for d in range(4)]
            joined[_slice_last4(block_starts, local_dims)] = block
        difference = (joined - reference).reshape(-1)
        denominator = max(float(reference.abs().max().item()), 1e-30)
        rel_max = float(difference.abs().max().item()) / denominator
        payload = {
            "grid": grid,
            "global_dims": global_dims,
            "local_dims": local_dims,
            "device_aware_request": os.environ.get("PYQCU_MPI_DEVICE_AWARE", "auto"),
            "overlap": os.environ.get("PYQCU_MPI_OVERLAP", "on"),
            "rel_max": rel_max,
        }
        print(payload, flush=True)
        if rel_max >= 3e-5:
            raise AssertionError(f"distributed coarse dslash failed: {payload}")


if __name__ == "__main__":
    main()
