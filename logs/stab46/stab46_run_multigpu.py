"""dev87 多线程多卡 MultiGrid 对照与性能运行器。

默认配置与项目约定一致：V100 单线程作为基线，P100×2（一线程一卡）
作为并行配置；两者使用同一 gauge、同一 E=12 33-tensor 缓存和同一
MultiGpuMultigrid 参数。结果写入 ``out/multigpu.json``。

用法：
  source ./env.sh
  python examples/qcu/dev87/run_multigpu.py
  python examples/qcu/dev87/run_multigpu.py --devices 1 2 --single-device 0

该脚本测的是并行吞吐/线程隔离，不把不同 GPU 型号的单卡时间误称为
纯算法加速；报告同时保存单线程和每线程真实计时。
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

from common import DATA_DIR, LAT_DEFAULT, MASS_DEFAULT  # noqa: E402
from pyqcu.cuda._multi_gpu import MultiGpuMultigrid  # noqa: E402


OUT = HERE / "out"


def _find_devices():
    v100 = []
    p100 = []
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        if "V100" in name:
            v100.append(i)
        if "P100" in name:
            p100.append(i)
    return v100, p100


def _run_one(label, device_ids, args):
    """在一个 MultiGpuMultigrid 实例中运行并采集所有数值指标。"""
    t0 = time.perf_counter()
    mg = MultiGpuMultigrid(
        lat_size=list(args.lat), mass=args.mass, atol=args.atol,
        num_levels=2, dof_list=[12, 12], mg_grid=[2, 2, 2, 2],
        num_restart=args.num_restart, coarse_max_iter=args.coarse_max_iter,
        coarse_tol_factor=args.coarse_tol_factor, nv_iters=1,
        nthreads=len(device_ids), device_ids=list(device_ids),
        use_cache=True, cache_dir=str(args.cache_dir), verbose=args.verbose,
        seed=args.seed,
    )
    result = mg.solve()
    wall = time.perf_counter() - t0
    consistency = mg.verify_consistency(tol=args.consistency_tol)
    threads = []
    for item in result["threads"]:
        ref = item["ref"]
        sol = item["mg"]
        max_ref = float(ref.abs().max().item())
        max_diff = float((sol - ref).abs().max().item())
        threads.append({
            "tid": int(item["tid"]),
            "device": int(item["device"]),
            "device_name": torch.cuda.get_device_name(int(item["device"])),
            "mg_s": float(item["mg_time"]),
            "ref_s": float(item["ref_time"]),
            "mg_vs_ref_max_abs": max_diff,
            "mg_vs_ref_rel_max": max_diff / max(max_ref, 1e-30),
        })
    row = {
        "label": label,
        "device_ids": [int(d) for d in device_ids],
        "wall_s": float(wall),
        "mg_parallel_wall_s": max(t["mg_s"] for t in threads),
        "ref_parallel_wall_s": max(t["ref_s"] for t in threads),
        "threads": threads,
        "consistency": consistency,
    }
    # Drop CPU result tensors and force the C++/PyTorch allocations to be
    # released before the next GPU configuration runs in this process.
    del mg, result
    gc.collect()
    for dev_id in device_ids:
        with torch.cuda.device(int(dev_id)):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=int, nargs=4, default=LAT_DEFAULT)
    ap.add_argument("--mass", type=float, default=MASS_DEFAULT)
    ap.add_argument("--atol", type=float, default=1e-6)
    ap.add_argument("--single-device", type=int, default=None)
    ap.add_argument("--devices", type=int, nargs="+", default=None)
    ap.add_argument("--num-restart", type=int, default=3)
    ap.add_argument("--coarse-max-iter", type=int, default=15)
    ap.add_argument("--coarse-tol-factor", type=float, default=1e3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--consistency-tol", type=float, default=1e-5)
    ap.add_argument("--cache-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--only", choices=("all", "single", "multi"), default="all")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA 设备")
    v100, p100 = _find_devices()
    single = ([args.single_device] if args.single_device is not None
              else v100[:1])
    multi = list(args.devices) if args.devices is not None else p100[:2]
    if args.only in ("all", "single") and len(single) != 1:
        raise RuntimeError(f"需要一张 V100 作为基线，发现 {v100}")
    if args.only in ("all", "multi") and len(multi) < 2:
        raise RuntimeError(f"需要两张 P100 进行多卡测试，发现 {p100}")
    if list(args.lat) != LAT_DEFAULT:
        # 只允许使用已有的 E=12 缓存；若用户切换格子，应显式先准备缓存。
        print(f"[multigpu] lat={args.lat}; 将从 {args.cache_dir} 查找对应缓存",
              flush=True)

    report = {
        "lat": list(args.lat), "mass": args.mass, "atol": args.atol,
        "num_levels": 2, "dof_list": [12, 12], "mg_grid": [2, 2, 2, 2],
        "num_restart": args.num_restart,
        "coarse_max_iter": args.coarse_max_iter,
        "coarse_tol_factor": args.coarse_tol_factor,
        "seed": args.seed,
        "cache_dir": str(args.cache_dir),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": {},
    }
    if args.only in ("all", "single"):
        print(f"[multigpu] single V100 device={single[0]}", flush=True)
        report["results"]["single_v100"] = _run_one("single_v100", single, args)
    if args.only == "all":
        # 同一 P100 型号的单卡基线才可用于判断线程并行效率；V100
        # 单卡仅作为跨型号参考，不能替代这个基线。
        p100_single = [multi[0]]
        print(f"[multigpu] single P100 device={p100_single[0]}", flush=True)
        report["results"]["single_p100"] = _run_one(
            "single_p100", p100_single, args)
    if args.only in ("all", "multi"):
        print(f"[multigpu] multi GPU devices={multi}", flush=True)
        report["results"]["multi_p100"] = _run_one("multi_p100", multi, args)
    if "single_p100" in report["results"] and "multi_p100" in report["results"]:
        a = report["results"]["single_p100"]["mg_parallel_wall_s"]
        b = report["results"]["multi_p100"]["mg_parallel_wall_s"]
        report["parallel_speedup_single_p100_over_p100x2"] = a / max(b, 1e-30)
    if "single_v100" in report["results"] and "multi_p100" in report["results"]:
        a = report["results"]["single_v100"]["mg_parallel_wall_s"]
        b = report["results"]["multi_p100"]["mg_parallel_wall_s"]
        report["cross_model_ratio_v100_over_p100x2"] = a / max(b, 1e-30)
        report["mg_vs_ref_speedup_single_v100"] = (
            report["results"]["single_v100"]["ref_parallel_wall_s"] / max(a, 1e-30))
    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "multigpu.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print(f"[result] {path}")
    ok = all(v["consistency"]["all_pass"] for v in report["results"].values())
    return report, ok


if __name__ == "__main__":
    _, success = main()
    raise SystemExit(0 if success else 1)
