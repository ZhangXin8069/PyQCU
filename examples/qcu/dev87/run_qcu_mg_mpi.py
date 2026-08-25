"""dev87 MPI 多 rank MG 冒烟：验证分布式下相对停机/真残差刷新路径。

用法（source ./env.sh 后）：
  mpirun --allow-run-as-root -np 2 python examples/qcu/dev87/run_qcu_mg_mpi.py
两 rank 可共享同一 GPU（仅正确性冒烟，不做性能声明）。
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QCU_LOG_DIR", str(Path(__file__).resolve().parents[2] / "logs" / "dev87_mpi"))
Path(os.environ["QCU_LOG_DIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_gauge_h5, load_stencil, MASS_DEFAULT
from pyqcu.cuda._multi_gpu import _SET_PTRS_COARSE_BASE_

from pyqcu import tools
import pyqcu.cuda.define as define
from pyqcu.cuda.define import params as mod_params, argv as mod_argv, set_ptrs as mod_set_ptrs
from pyqcu.cuda import qcu


def main():
    lat = [8, 8, 8, 16]
    mass = MASS_DEFAULT
    Lx, Ly, Lz, Lt = lat
    p = mod_params.clone(); av = mod_argv.clone(); s = mod_set_ptrs.clone()
    dt = define._LAT_C64_
    p[define._LAT_X_] = Lx; p[define._LAT_Y_] = Ly; p[define._LAT_Z_] = Lz; p[define._LAT_T_] = Lt
    p[define._LAT_XYZT_] = Lx * Ly * Lz * Lt
    gx, gy, gz, gt = tools.give_grid_size()
    p[define._GRID_X_] = gx; p[define._GRID_Y_] = gy; p[define._GRID_Z_] = gz; p[define._GRID_T_] = gt
    p[define._NODE_RANK_] = define.rank
    p[define._NODE_SIZE_] = define.size
    p[define._DATA_TYPE_] = dt
    av[define._MASS_] = mass; av[define._ATOL_] = 1e-6; av[define._SIGMA_] = 0.1

    g = load_gauge_h5(lat, mass, device="cuda")
    lonv, hnn, hdg, sit = load_stencil(lat, 12, 1, device="cuda")
    s[_SET_PTRS_COARSE_BASE_ + 0] = lonv.contiguous().data_ptr()
    s[_SET_PTRS_COARSE_BASE_ + 1] = hnn.contiguous().data_ptr()
    s[_SET_PTRS_COARSE_BASE_ + 2] = hdg.contiguous().data_ptr()
    s[_SET_PTRS_COARSE_BASE_ + 3] = sit.contiguous().data_ptr()

    from common import make_clover_tensors
    ce, cei, coo, coi, s, p, av = make_clover_tensors(g, lat, mass)

    npz = np.load(Path(__file__).resolve().parent / "out" / "qcu_clover_solve.npz")
    b_eo = torch.from_numpy(npz["b_eo"]).to("cuda")

    idx = int(p[define._SET_INDEX_].item())
    p[define._SET_INDEX_] = idx; p[define._SET_PLAN_] = 1
    p[define._PARITY_] = 0; p[define._MAX_ITER_] = 1000; p[define._VERBOSE_] = 1
    p[define._MG_NUM_LEVEL_] = 2
    p[define._MG_LEVEL1_E_] = 12
    p[define._MG_LEVEL1_X_] = Lx // 2; p[define._MG_LEVEL1_Y_] = Ly // 2
    p[define._MG_LEVEL1_Z_] = Lz // 2; p[define._MG_LEVEL1_T_] = Lt // (2 * 2)
    p[define._MG_LEVEL1_MAX_ITER_] = 200
    p[define._MG_LEVEL1_DATA_TYPE_] = dt
    p[define._MG_LEVEL1_NUM_RESTART_] = 5
    av[define._MG_LEVEL1_ATOL_] = 1e-6 * 3000.0

    qcu.applyInitQcu(s, p, av)
    x_mg = torch.zeros_like(b_eo)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    qcu.applyCloverMultigridQcu(x_mg, b_eo, g, ce, coo, cei, coi, s, p)
    torch.cuda.synchronize()
    if define.rank == 0:
        print(f"[mpi-mg] wall={time.perf_counter()-t0:.3f}s", flush=True)
    p[define._SET_INDEX_] = idx
    try:
        qcu.applyEndQcu(s, p)
    except Exception:
        pass


if __name__ == "__main__":
    main()
