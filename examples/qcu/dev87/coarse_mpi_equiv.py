"""逐元素验证分布式 33 点粗算子与全局周期参考的一致性。

示例（source ./env.sh 后）：
  mpirun --allow-run-as-root -np 2 python examples/qcu/dev87/coarse_mpi_equiv.py

每个 rank 只持有一个局部粗格向量；C++ 宽粗 dslash 通过 host-staging halo
交换计算局部输出，随后在 rank 0 重建全局输出并与 ``tools.apply_stencil``
的完整周期结果比较。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from mpi4py import MPI

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import (load_stencil, load_stencil_local, local_geometry,
                    parse_complex_dtype, process_grid)
from pyqcu import tools
from pyqcu.cuda import qcu
import pyqcu.cuda.define as define
from pyqcu.cuda.define import (argv as mod_argv, params as mod_params,
                               set_ptrs as mod_set_ptrs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=int, nargs=4, default=[8, 8, 8, 16])
    ap.add_argument("--grid", type=int, nargs=4, default=None)
    ap.add_argument("--E", type=int, default=12)
    ap.add_argument("--nvi", type=int, default=1)
    ap.add_argument("--dtype", choices=("c64", "c128"), default="c64")
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    grid = process_grid(args.grid)
    if int(np.prod(grid)) != comm.Get_size():
        raise ValueError(f"grid={grid} 与 MPI size={comm.Get_size()} 不一致")
    dtype, data_type = parse_complex_dtype(args.dtype)

    lonv_g, hnn_g, hdg_g, sit_g = load_stencil(
        args.lat, args.E, args.nvi, device="cpu", dtype=dtype, level=1)
    del lonv_g
    coarse_g = [int(sit_g.shape[2]), int(sit_g.shape[3]),
                int(sit_g.shape[4]), int(sit_g.shape[5])]
    local_coarse, starts, _ = local_geometry(
        coarse_g, grid=grid, rank=rank, require_even=False)
    _, hnn_l, hdg_l, sit_l = load_stencil_local(
        args.lat, args.E, args.nvi, grid=grid, rank=rank, device="cuda",
        dtype=dtype, level=1)

    n_global = args.E * int(np.prod(coarse_g))
    base = torch.arange(n_global, dtype=torch.float64).reshape(args.E, *coarse_g)
    x_global = (torch.sin(base * 0.013) + 1j * torch.cos(base * 0.017)).to(dtype)
    local_slice = (slice(None),) + tuple(
        slice(starts[d], starts[d] + local_coarse[d]) for d in range(4))
    x_local = x_global[local_slice].contiguous().cuda()
    y_local = torch.empty_like(x_local)

    p = mod_params.clone()
    a = mod_argv.clone().to(dtype=torch.float64 if dtype == torch.complex128
                             else torch.float32)
    s = mod_set_ptrs.clone()
    p[define._LAT_X_] = local_coarse[0]
    p[define._LAT_Y_] = local_coarse[1]
    p[define._LAT_Z_] = local_coarse[2]
    p[define._LAT_T_] = 2 * local_coarse[3]
    p[define._LAT_XYZT_] = int(np.prod(local_coarse) * 2)
    for idx, value in zip((define._GRID_X_, define._GRID_Y_,
                           define._GRID_Z_, define._GRID_T_), grid):
        p[idx] = value
    p[define._NODE_RANK_] = rank
    p[define._NODE_SIZE_] = comm.Get_size()
    p[define._DATA_TYPE_] = data_type
    p[define._SET_INDEX_] = 0
    p[define._SET_PLAN_] = 1
    p[define._PARITY_] = 0
    p[define._VERBOSE_] = 0
    p[define._MG_LEVEL1_E_] = args.E
    p[define._MG_LEVEL1_X_] = local_coarse[0]
    p[define._MG_LEVEL1_Y_] = local_coarse[1]
    p[define._MG_LEVEL1_Z_] = local_coarse[2]
    p[define._MG_LEVEL1_T_] = local_coarse[3]

    qcu.applyInitQcu(s, p, a)
    try:
        qcu.applyMultigridCoarseDslashWideQcu(
            y_local, x_local, sit_l, hnn_l, hdg_l, s, p)
        torch.cuda.synchronize()
        ref_global = tools.apply_stencil(hnn_g, hdg_g, sit_g, x_global)
        ref_local = ref_global[local_slice].contiguous()
        diff = (y_local.cpu() - ref_local).reshape(-1)
        local_den = max(float(torch.linalg.norm(ref_local).item()), 1e-30)
        local_result = (float(torch.linalg.norm(diff).item()) / local_den,
                        float(diff.abs().max().item()))
        results = comm.gather(local_result, root=0)
        blocks = comm.gather((rank, y_local.cpu()), root=0)
        if rank == 0:
            joined = torch.empty_like(ref_global)
            for block_rank, block in blocks:
                block_local, block_starts, _ = local_geometry(
                    coarse_g, grid=grid, rank=block_rank, require_even=False)
                block_slice = (slice(None),) + tuple(
                    slice(block_starts[d], block_starts[d] + block_local[d])
                    for d in range(4))
                joined[block_slice] = block
            global_diff = (joined - ref_global).reshape(-1)
            global_rel = float(torch.linalg.norm(global_diff).item()) / max(
                float(torch.linalg.norm(ref_global).item()), 1e-30)
            global_max = float(global_diff.abs().max().item())
            print({"grid": grid, "coarse_global": coarse_g,
                   "local_errors": results, "global_l2_rel": global_rel,
                   "global_max_abs": global_max}, flush=True)
            if global_rel >= (3e-5 if dtype == torch.complex64 else 1e-11):
                raise AssertionError("distributed coarse dslash equivalence failed")
    finally:
        qcu.applyEndQcu(s, p)


if __name__ == "__main__":
    main()
