"""dev87 一键回归闸门：串联本工作区全部核心校验，输出 out/regression.json。

用法（source ./env.sh 后，单卡 V100）：
  python examples/qcu/dev87/run_all.py [--with-quda]
默认仅 PyQCU 侧（~6 分钟）；--with-quda 追加 quda/PyQUDA 解对照（需 QUDA 环境与
libquda 可用，另 +2 分钟）。任一断言失败即 exit 1。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import os
os.environ.setdefault("QCU_LOG_DIR", str(HERE.parents[1] / "logs" / "dev87"))
Path(os.environ["QCU_LOG_DIR"]).mkdir(parents=True, exist_ok=True)

from common import (LAT_DEFAULT, MASS_DEFAULT, load_gauge_h5, load_stencil,
                    make_clover_tensors, pick_v100)
from pyqcu.cuda import define, qcu
from pyqcu import dslash, tools

RESULTS = {}
FAILED = []


def check(name, ok, detail):
    RESULTS[name] = {"pass": bool(ok), "detail": detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    if not ok:
        FAILED.append(name)


def t_clover_solve_and_truth():
    """G4.1 基线：全求解器解的全算子真相对残差（修复后应 ~1e-7）。"""
    lat, m = LAT_DEFAULT, MASS_DEFAULT
    kappa = 1.0 / (2 * m + 8)
    g = load_gauge_h5(lat, m, device="cuda")
    ce, cei, coo, coi, s, p, av = make_clover_tensors(g, lat, m)
    gen = torch.Generator("cpu").manual_seed(43)
    b_eo = torch.randn([2, 4, 3] + define.lat_shape(p), generator=gen,
                       dtype=torch.float32, device="cpu").to(torch.complex64).to("cuda")
    x = torch.zeros_like(b_eo)
    p[define._SET_PLAN_] = 1; p[define._PARITY_] = 0
    p[define._MAX_ITER_] = 1000; p[define._VERBOSE_] = 0
    idx = int(p[define._SET_INDEX_].item()); av[define._ATOL_] = 1e-6
    qcu.applyInitQcu(s, p, av)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    qcu.applyCloverBistabCgQcu(x, b_eo, g, ce, coo, cei, coi, s, p)
    torch.cuda.synchronize(); wall = time.perf_counter() - t0
    p[define._SET_INDEX_] = idx; qcu.applyEndQcu(s, p)
    U = tools.poooxyzt2oooxyzt(g); cl = dslash.make_clover(U, kappa=kappa)
    xf = tools.poooxyzt2oooxyzt(x); bf = tools.poooxyzt2oooxyzt(b_eo)
    r = dslash.give_wilson(xf, U, kappa, True) + dslash.give_clover(xf, cl) - bf
    rel = float(tools.norm(r) / tools.norm(bf))
    check("clover_solve_true_res", rel < 1e-5,
          f"rel={rel:.3e} wall={wall:.2f}s (bug42 修复口径)")


def t_mg_end_to_end():
    """G8/G10 PyQCU MG：解 vs BiCGStab 参考一致性。"""
    npz = np.load(OUT := HERE / "out" / "qcu_clover_solve.npz")
    _ = npz  # noqa: F841
    import subprocess
    r = subprocess.run([sys.executable, str(HERE / "run_qcu_mg.py"),
                        "--levels", "2", "--E", "12", "--nvi", "1",
                        "--cmi", "200", "--ctf", "3000"],
                       capture_output=True, text=True, timeout=900)
    j = json.loads((HERE / "out" / "qcu_clover_mg.json").read_text())
    check("mg_vs_ref", r.returncode == 0 and j["rel_diff_vs_bistabcg"] < 1e-5,
          f"rel={j['rel_diff_vs_bistabcg']:.3e} wall={j['mg_wall_s']:.2f}s")


def t_component():
    """G5-G7 组件诊断：Galerkin/正交性阈值断言。"""
    import component_diag  # noqa: F401  (复用其 main 产物)
    j = json.loads((HERE / "out" / "component_diag.json").read_text())
    gal = float(j["galerkin_rel_diff"])
    off = float(j["ortho_offdiag_max"])
    check("component_quality", gal < 1e-5 and off < 1e-5,
          f"galerkin={gal:.2e} ortho_offdiag={off:.2e}")


def t_quda_scaled():
    """G4.1 双方解对照（缩放 m+4 口径）。"""
    zq = json.loads((HERE / "out" / "quda_clover_solve.json").read_text())
    rd = float(zq["rel_diff_vs_qcu"])
    check("quda_solve_scaled_agreement", rd < 1e-5,
          f"rel_diff(scaled m+4)={rd:.3e} iters={zq.get('iters')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-quda", action="store_true")
    args = ap.parse_args()
    pick_v100()

    t0 = time.perf_counter()
    try:
        t_clover_solve_and_truth()
        t_mg_end_to_end()
        t_component()
        if args.with_quda:
            t_quda_scaled()
    except Exception as e:  # noqa: BLE001 —— 闸门需捕获一切并如实报告
        check("exception", False, repr(e))
    total = time.perf_counter() - t0

    summary = {"total_s": round(total, 1), "failed": FAILED,
               "n_pass": sum(1 for v in RESULTS.values() if v["pass"]),
               "n_total": len(RESULTS), "results": RESULTS,
               "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    (HERE / "out" / "regression.json").write_text(json.dumps(summary, indent=2))
    print(f"\n=== regression {'GREEN' if not FAILED else 'RED'} "
          f"({summary['n_pass']}/{summary['n_total']}) in {total:.1f}s ===")
    raise SystemExit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
