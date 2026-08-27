"""dev87 一键回归闸门：串联本工作区全部核心校验，输出 out/regression.json。

用法（source ./env.sh 后，单卡 V100）：
  python examples/qcu/dev87/run_all.py [--with-quda]
默认仅 PyQCU 侧（当前约 30 秒）；--with-quda 追加 quda/PyQUDA 解对照（需 QUDA
环境与 libquda 可用，当前无调优缓存时约另需 7 分钟）。任一断言失败即 exit 1。
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


def _quda_process_env():
    """为 QUDA 子进程显式选择 dev87 的库，隔离 env.sh 的旧库前缀。"""
    env = os.environ.copy()
    install = env.get("QUDA_INSTALL")
    script = HERE / "quda_env.sh"
    if not install and script.exists():
        for line in script.read_text().splitlines():
            prefix = "export QUDA_INSTALL="
            if line.startswith(prefix):
                install = line[len(prefix):].strip().strip('"').strip("'")
                break
    if install:
        env["QUDA_INSTALL"] = install
        env["QUDA_PATH"] = install
        old_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{install}/lib:{old_ld}"
        env.setdefault("DEV87_REDUCE_SYNC", "1")
    return env


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
    import subprocess
    out_dir = HERE / "out"
    mg_json = out_dir / "qcu_clover_mg.json"
    before_ns = mg_json.stat().st_mtime_ns if mg_json.exists() else -1
    r = subprocess.run([sys.executable, str(HERE / "run_qcu_mg.py"),
                        "--levels", "2", "--E", "12", "--nvi", "1",
                        "--cmi", "200", "--ctf", "3000"],
                       capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        tail = (r.stderr or r.stdout)[-500:].replace("\n", " ")
        check("mg_vs_ref", False, f"rc={r.returncode} 子进程失败: {tail}")
        return
    if not mg_json.exists() or mg_json.stat().st_mtime_ns <= before_ns:
        check("mg_vs_ref", False, "子进程成功但未生成本次 qcu_clover_mg.json（拒绝读取旧结果）")
        return
    try:
        j = json.loads(mg_json.read_text())
        lat = list(j["lat"])
        expected = list(LAT_DEFAULT)
        rel = float(j["rel_diff_vs_bistabcg"])
        ok = lat == expected and np.isfinite(rel) and rel < 1e-5
        detail = f"rc=0 lat={lat} rel={rel:.3e} wall={float(j['mg_wall_s']):.2f}s"
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
        check("mg_vs_ref", False, f"结果文件无效: {e!r}")
        return
    hint = "" if ok else (
        " | 直连运行器历史‘驱动上下文损伤’假设已由 MultiGpu 与直连复验排除；"
        "本次失败应按 run_qcu_mg 的桥接序列和本次日志定位")
    check("mg_vs_ref", ok,
          detail + hint)


def t_component():
    """G5-G7 组件诊断：Galerkin/正交性阈值断言。"""
    import component_diag
    path = HERE / "out" / "component_diag.json"
    before_ns = path.stat().st_mtime_ns if path.exists() else -1
    component_diag.main()
    if not path.exists() or path.stat().st_mtime_ns <= before_ns:
        check("component_quality", False, "组件诊断未生成本次结果（拒绝读取旧结果）")
        return
    try:
        j = json.loads(path.read_text())
        lat = list(j["lat"])
        gal = float(j["galerkin_rel_diff"])
        off = float(j["ortho_offdiag_max"])
        ok = (lat == list(LAT_DEFAULT) and np.isfinite(gal) and
              np.isfinite(off) and gal < 1e-5 and off < 1e-5)
        check("component_quality", ok,
              f"lat={lat} galerkin={gal:.2e} ortho_offdiag={off:.2e}")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
        check("component_quality", False, f"组件结果无效: {e!r}")


def t_quda_scaled():
    """G4.1 双方真实运行与解对照（缩放 m+4 口径）。"""
    import subprocess
    out_dir = HERE / "out"
    solve_json = out_dir / "quda_clover_solve.json"
    mg_json = out_dir / "quda_clover_mg.json"
    before = {p: p.stat().st_mtime_ns if p.exists() else -1
              for p in (solve_json, mg_json)}
    r = subprocess.run([sys.executable, str(HERE / "run_quda_py.py"),
                        "--case", "all", "--lat", *map(str, LAT_DEFAULT),
                        "--mass", str(MASS_DEFAULT), "--nvec", "12",
                        "--block", "2", "2", "2", "2"],
                       env=_quda_process_env(),
                       capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        tail = (r.stderr or r.stdout)[-700:].replace("\n", " ")
        check("quda_solve_scaled_agreement", False,
              f"QUDA/PyQUDA 子进程失败 rc={r.returncode}: {tail}")
        return
    stale = [str(p.name) for p, old in before.items()
             if not p.exists() or p.stat().st_mtime_ns <= old]
    if stale:
        check("quda_solve_scaled_agreement", False,
              "QUDA 子进程成功但结果未刷新: " + ", ".join(stale))
        return
    try:
        zq = json.loads(solve_json.read_text())
        zmg = json.loads(mg_json.read_text())
        lat_ok = list(zq["lat"]) == list(LAT_DEFAULT) and list(zmg["lat"]) == list(LAT_DEFAULT)
        rd = float(zq["rel_diff_vs_qcu"])
        ok = lat_ok and np.isfinite(rd) and rd < 1e-5
        detail = f"lat={zq['lat']} rel_diff(scaled m+4)={rd:.3e} iters={zq.get('iters')}"
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
        check("quda_solve_scaled_agreement", False, f"QUDA 结果文件无效: {e!r}")
        return
    check("quda_solve_scaled_agreement", ok, detail)

    # MG 的 QUDA 端解也必须与同一 b、同一布局下的 PyQCU MG 解比较；只
    # 看求解器迭代数会遗漏 transfer/coarse-op 的系统性布局错误。
    try:
        from run_quda_py import reconstruct_full_b
        qcu_x = reconstruct_full_b(np.load(out_dir / "qcu_clover_mg.npz")["x_eo"])
        quda_x = np.load(out_dir / "quda_clover_mg.npz")["x_scxyzt"]
        # reconstruct_full_b 返回合并 spin/color 的 [12,X,Y,Z,T]；
        # PyQUDA 导出保留 [4,3,X,Y,Z,T]。两者是同一内存语义，比较前
        # 只展开/合并这两个内部自由度，不能把形状差误判为布局差。
        if quda_x.ndim == 6 and quda_x.shape[:2] == (4, 3):
            qcu_cmp = qcu_x.reshape(quda_x.shape)
        else:
            qcu_cmp = qcu_x
        if quda_x.shape != qcu_cmp.shape:
            raise ValueError(f"MG 解 shape 不同: QCU={qcu_x.shape}, QUDA={quda_x.shape}")
        scale = float(MASS_DEFAULT + 4.0)
        mg_rel = float(np.linalg.norm((quda_x * scale - qcu_cmp).ravel()) /
                       max(np.linalg.norm(qcu_cmp.ravel()), 1e-30))
        mg_ok = np.isfinite(mg_rel) and mg_rel < 1e-4
        check("quda_mg_scaled_agreement", mg_ok,
              f"rel_diff(scaled m+4)={mg_rel:.3e} qcu_wall={json.loads((out_dir / 'qcu_clover_mg.json').read_text())['mg_wall_s']:.2f}s quda_setup={zmg.get('setup_s'):.2f}s quda_solve={zmg.get('wall_s'):.2f}s")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
        check("quda_mg_scaled_agreement", False, f"MG 解对照失败: {e!r}")


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
