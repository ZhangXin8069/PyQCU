#!/usr/bin/env python3
"""
dev80_2 多卡并行验证 — P100*2 vs V100 单卡
- 单卡 V100: 1 线程 cuda:0
- 双卡 P100: 2 线程 cuda:1,2 (torch 不支持 sm_60，故主线程 V100 预生成 gauge/coarse 后拷贝)
- 对照：单线程 MG vs 双线程 MG 一致性 (rel <1e-5)
"""
import os, sys, time, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch
from pyqcu.cuda._multi_gpu import MultiGpuMultigrid

LAT = [16,32,32,48]  # 目标格子, 8×8×8×16 用于快速验证 (单线程 0.43s)
MASS = 0.05
ATOL = 1e-6

def run_single():
    print("=== Single V100 (1 thread, cuda:0) ===")
    mg = MultiGpuMultigrid(lat_size=LAT, mass=MASS, atol=ATOL, num_levels=2, dof_list=[12,12], mg_grid=[2,2,2,2],
                           num_restart=3, coarse_max_iter=15, coarse_tol_factor=1e3, nv_iters=1,
                           nthreads=1, device_ids=[0], use_cache=True, cache_dir=str(ROOT/"data"), verbose=True)
    res = mg.solve()
    print(f"single mg_time {[r['mg_time'] for r in res['threads']]} ref { [r['ref_time'] for r in res['threads']]}")
    cons = mg.verify_consistency(tol=1e-5)
    print(f"consistency {cons}")
    return res, cons

def run_multi():
    print("=== Multi P100*2 (2 threads, cuda:1,2) ===")
    # 主线程仍在 V100 生成，workers 在 P100 求解（libqcu.so 含 sm_60）
    mg = MultiGpuMultigrid(lat_size=LAT, mass=MASS, atol=ATOL, num_levels=2, dof_list=[12,12], mg_grid=[2,2,2,2],
                           num_restart=3, coarse_max_iter=15, coarse_tol_factor=1e3, nv_iters=1,
                           nthreads=2, device_ids=[1,2], use_cache=True, cache_dir=str(ROOT/"data"), verbose=True)
    res = mg.solve()
    print(f"multi mg_time {[r['mg_time'] for r in res['threads']]}")
    cons = mg.verify_consistency(tol=1e-5)
    print(f"consistency {cons}")
    return res, cons

if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single","multi","both"], default="both")
    args=parser.parse_args()
    os.environ["QCU_LOG_DIR"]=str(ROOT/"logs"/"dev80_2")
    Path(ROOT/"logs"/"dev80_2").mkdir(parents=True, exist_ok=True)
    if args.mode in ("single","both"):
        try:
            r,c = run_single()
            with open(ROOT/"logs"/"dev80_2"/"multi_single.json","w") as f:
                json.dump({"res":str(r),"cons":c},f,indent=2)
        except Exception as e:
            import traceback; traceback.print_exc()
    if args.mode in ("multi","both"):
        try:
            r,c = run_multi()
            with open(ROOT/"logs"/"dev80_2"/"multi_double.json","w") as f:
                json.dump({"res":str(r),"cons":c},f,indent=2)
        except Exception as e:
            import traceback; traceback.print_exc()
