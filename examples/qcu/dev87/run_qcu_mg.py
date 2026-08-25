"""dev87 PyQCU C++ 多线程 MultiGrid 端到端运行器（G8/G10 PyQCU 侧）。

复用 data/ 统一 gauge 与 33-tensor stencil 缓存；源 b 与 run_qcu_ops 的
clover_solve 案例完全一致（同 seed），以便与 quda/PyQUDA 的 MG 解对照。
用法（source ./env.sh 后）：
  python examples/qcu/dev87/run_qcu_mg.py [--levels 2] [--E 12] ...
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QCU_LOG_DIR", str(Path(__file__).resolve().parents[2] / "logs" / "dev87"))
Path(os.environ["QCU_LOG_DIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ATOL_DEFAULT, LAT_DEFAULT, MASS_DEFAULT, load_gauge_h5, load_stencil, pick_v100, save_result
from pyqcu.cuda import define, qcu
from pyqcu.cuda._multi_gpu import _SET_PTRS_COARSE_BASE_

OUT = Path(__file__).resolve().parent / "out"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=int, nargs=4, default=LAT_DEFAULT)
    ap.add_argument("--mass", type=float, default=MASS_DEFAULT)
    ap.add_argument("--atol", type=float, default=ATOL_DEFAULT)
    ap.add_argument("--levels", type=int, default=2)
    ap.add_argument("--E", type=int, default=12)
    ap.add_argument("--nvi", type=int, default=1)
    ap.add_argument("--mg-grid", type=int, nargs=4, default=[2, 2, 2, 2])
    ap.add_argument("--restart", type=int, default=5)
    ap.add_argument("--cmi", type=int, default=200)
    ap.add_argument("--ctf", type=float, default=3000.0)
    args = ap.parse_args()

    dev = pick_v100()
    print(f"[dev87-mg] device={torch.cuda.get_device_name(dev)}")
    lat = args.lat
    Lx, Ly, Lz, Lt = lat
    g = load_gauge_h5(lat, args.mass, device="cuda")
    stencil1 = load_stencil(lat, args.E, args.nvi, device="cuda")

    p = define.params.clone(); av = define.argv.clone(); s = define.set_ptrs.clone()
    dt = define._LAT_C64_
    p[define._LAT_X_] = Lx; p[define._LAT_Y_] = Ly; p[define._LAT_Z_] = Lz; p[define._LAT_T_] = Lt
    p[define._LAT_XYZT_] = Lx * Ly * Lz * Lt
    p[define._GRID_X_] = p[define._GRID_Y_] = p[define._GRID_Z_] = p[define._GRID_T_] = 1
    p[define._NODE_RANK_] = 0; p[define._NODE_SIZE_] = 1
    p[define._DATA_TYPE_] = dt
    av[define._MASS_] = args.mass; av[define._ATOL_] = args.atol; av[define._SIGMA_] = 0.1
    p[define._MG_NUM_LEVEL_] = args.levels
    p[define._MG_LEVEL1_E_] = args.E
    p[define._MG_LEVEL1_X_] = Lx // args.mg_grid[0]
    p[define._MG_LEVEL1_Y_] = Ly // args.mg_grid[1]
    p[define._MG_LEVEL1_Z_] = Lz // args.mg_grid[2]
    p[define._MG_LEVEL1_T_] = Lt // (2 * args.mg_grid[3])
    p[define._MG_LEVEL1_MAX_ITER_] = args.cmi
    p[define._MG_LEVEL1_DATA_TYPE_] = dt
    p[define._MG_LEVEL1_NUM_RESTART_] = args.restart
    av[define._MG_LEVEL1_ATOL_] = args.atol * args.ctf

    # Clover 张量（复用 common 的生命周期封装，槽位从当前 index 起）
    from common import make_clover_tensors
    ce, cei, coo, coi, s, p, av = make_clover_tensors(g, lat, args.mass)

    # 粗算子填槽（level1）
    lonv, hnn, hdg, sit = stencil1
    base = _SET_PTRS_COARSE_BASE_
    s[base + 0] = lonv.contiguous().data_ptr()
    s[base + 1] = hnn.contiguous().data_ptr()
    s[base + 2] = hdg.contiguous().data_ptr()
    s[base + 3] = sit.contiguous().data_ptr()

    # dev87 形状守卫：基线 npz 与目标格子不一致时自动经 run_qcu_ops 重建
    import subprocess
    expect = [Lx, Ly, Lz, Lt // 2]
    npz_path = OUT / "qcu_clover_solve.npz"
    need = True
    if npz_path.exists():
        zz = np.load(npz_path)
        need = list(zz["b_eo"].shape[-4:]) != expect
    if need:
        print("[dev87-mg] 基线形状不匹配 -> 重跑 run_qcu_ops clover_solve", flush=True)
        r = subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "run_qcu_ops.py"),
                            "--case", "clover_solve", "--lat", *[str(v) for v in lat],
                            "--mass", str(args.mass)], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("baseline rebuild failed: " + r.stderr[-800:])
    npz = np.load(npz_path)
    assert list(npz["b_eo"].shape[-4:]) == expect, "baseline shape still mismatched"
    b_eo = torch.from_numpy(npz["b_eo"]).to("cuda")
    x_ref = torch.from_numpy(npz["x_eo"]).to("cuda")

    s[:_SET_PTRS_COARSE_BASE_] = 0   # dev87: 清除已结束集合的陈旧句柄
    idx = 2
    p[define._SET_INDEX_] = idx; p[define._SET_PLAN_] = 1
    p[define._PARITY_] = 0; p[define._MAX_ITER_] = 1000; p[define._VERBOSE_] = 0
    qcu.applyInitQcu(s, p, av)
    x_mg = torch.zeros_like(b_eo)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    qcu.applyCloverMultigridQcu(x_mg, b_eo, g, ce, coo, cei, coi, s, p)
    torch.cuda.synchronize(); mg_time = time.perf_counter() - t0
    for j in (idx,):
        p[define._SET_INDEX_] = j
        try:
            qcu.applyEndQcu(s, p)
        except Exception:
            pass

    rel = float((torch.linalg.norm((x_mg - x_ref).ravel())
                 / torch.linalg.norm(x_ref.ravel())).item())
    res = save_result("qcu_clover_mg", {
        "lat": lat, "mass": args.mass, "atol": args.atol,
        "levels": args.levels, "E": args.E, "nvi": args.nvi,
        "restart": args.restart, "coarse_max_iter": args.cmi,
        "coarse_tol_factor": args.ctf,
        "mg_wall_s": mg_time, "rel_diff_vs_bistabcg": rel,
    })
    np.savez_compressed(OUT / "qcu_clover_mg.npz", x_eo=x_mg.cpu().numpy())


if __name__ == "__main__":
    main()
