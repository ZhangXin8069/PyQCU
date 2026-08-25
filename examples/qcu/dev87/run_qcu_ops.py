"""dev87 PyQCU 侧基线运行器：把对照单测所需的功能输出与计时落盘到 out/。

用法（须先 source ./env.sh）：
  python examples/qcu/dev87/run_qcu_ops.py --case all
案例：
  schur_dslash : C++ Schur 奇偶算子单次作用 S·v（G4 依赖 + G7 对照输入）
  clover_solve : applyCloverBistabCgQcu 全求解器（G4.1 基线，quda --inv-type bicgstab 对标）
同时导出 quda 侧所需的 QDP 序 gauge npy 与元数据。
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("QCU_LOG_DIR", str(Path(__file__).resolve().parents[2] / "logs" / "dev87"))
Path(os.environ["QCU_LOG_DIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ATOL_DEFAULT, DATA_DIR, LAT_DEFAULT, MASS_DEFAULT, SEED_DEFAULT,
                    full_gauge_numpy, full_to_qdp, gauge_tag, load_gauge_h5,
                    make_clover_tensors, pick_v100, save_result)
from pyqcu.cuda import define, qcu

OUT = Path(__file__).resolve().parent / "out"


def export_qdp_gauge(g_dev, lat, mass):
    """全格点 -> QDP 序 npy（供 pyquda loadGauge），并写元数据。"""
    u_np = full_gauge_numpy(g_dev)
    qdp = full_to_qdp(u_np)
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "gauge_qdp_c64.npy"
    np.save(f, qdp.astype(np.complex64))
    meta = {"lat_xyzt": list(lat), "mass": mass,
            "kappa": float(1.0 / (2 * mass + 8)), "sigma": 0.1, "seed": SEED_DEFAULT,
            "layout": "(mu,t,z,y,row,col) QDP x-fastest; src=PyQCU h5 parity->poooxyzt2oooxyzt",
            "file": str(f), "bytes": int(f.stat().st_size)}
    save_result("qdp_gauge_meta", meta)
    return meta


def case_schur_dslash(lat, mass, g_dev, ce, coo, cei, coi, s, p, av):
    dt = define._LAT_C64_
    ls = define.lat_shape(p)
    gen = torch.Generator(device=g_dev.device); gen.manual_seed(SEED_DEFAULT)
    src_o = torch.randn([4, 3] + ls, generator=gen, dtype=torch.float32,
                        device=g_dev.device).to(torch.complex64)
    out_o = torch.zeros_like(src_o)
    p[define._SET_PLAN_] = 1
    p[define._PARITY_] = 0
    idx = int(p[define._SET_INDEX_].item())
    qcu.applyInitQcu(s, p, av)
    for _ in range(3):
        qcu.applyCloverBistabCgDslashQcu(out_o, src_o, g_dev, ce, coo, cei, coi, s, p)
    torch.cuda.synchronize()
    times = []
    for _ in range(20):
        st, ed = torch.cuda.Event(True), torch.cuda.Event(True)
        st.record()
        qcu.applyCloverBistabCgDslashQcu(out_o, src_o, g_dev, ce, coo, cei, coi, s, p)
        ed.record()
        torch.cuda.synchronize()
        times.append(st.elapsed_time(ed))
    p[define._SET_INDEX_] = idx
    qcu.applyEndQcu(s, p)
    p[define._SET_INDEX_] = idx + 1
    np.savez_compressed(OUT / "qcu_schur_dslash.npz",
                        src_o=src_o.cpu().numpy(), out_o=out_o.cpu().numpy())
    res = save_result("qcu_schur_dslash", {
        "lat": lat, "mass": mass, "matvec_ms_median": float(np.median(times)),
        "out_norm2": float(tools_norm(out_o)),
    })
    return res


def tools_norm(t):
    from pyqcu import tools
    return tools.norm(t)


def case_clover_solve(lat, mass, g_dev, ce, coo, cei, coi, s, p, av, atol=ATOL_DEFAULT):
    ls = define.lat_shape(p)
    gen = torch.Generator(device="cpu"); gen.manual_seed(SEED_DEFAULT + 1)
    b_eo = torch.randn([2, 4, 3] + ls, generator=gen, dtype=torch.float32,
                       device="cpu").to(torch.complex64).to(g_dev.device)
    x_eo = torch.zeros_like(b_eo)
    p[define._SET_PLAN_] = 1
    p[define._PARITY_] = 0
    p[define._MAX_ITER_] = 1000
    idx = int(p[define._SET_INDEX_].item())
    av[define._ATOL_] = atol
    qcu.applyInitQcu(s, p, av)
    torch.cuda.synchronize()
    st = torch.cuda.Event(True); ed = torch.cuda.Event(True)
    st.record()
    qcu.applyCloverBistabCgQcu(x_eo, b_eo, g_dev, ce, coo, cei, coi, s, p)
    ed.record()
    torch.cuda.synchronize()
    solve_ms = st.elapsed_time(ed)
    r_full = b_eo.clone()
    p[define._SET_INDEX_] = idx
    qcu.applyEndQcu(s, p)
    p[define._SET_INDEX_] = idx + 1
    np.savez_compressed(OUT / "qcu_clover_solve.npz",
                        b_eo=b_eo.cpu().numpy(), x_eo=x_eo.cpu().numpy())
    hist = parse_history()
    return save_result("qcu_clover_solve", {
        "lat": lat, "mass": mass, "atol": atol, "solve_ms": float(solve_ms),
        "iters": hist.get("iters"), "final_rn2": hist.get("final"),
        "history_head": hist.get("head", [])[:20],
    })


def parse_history():
    """从 QCU_LOG_DIR 下最新 clover_multigrid*.log 提取 CONVERGENCE_HISTORY。"""
    import glob
    import re
    logs = sorted(glob.glob(os.environ["QCU_LOG_DIR"] + "/**/*.log"), key=os.path.getmtime)
    for f in reversed(logs):
        txt = open(f, errors="ignore").read()
        m = re.findall(r"CONVERGENCE_HISTORY:\s*\[([^\]]*)\]", txt)
        if m:
            vals = [float(v) for v in m[-1].split(",") if v.strip()]
            return {"iters": len(vals), "final": vals[-1] if vals else None, "head": vals}
        it = re.findall(r"([0-9]+) iterations?", txt)
        if it:
            return {"iters": int(it[-1]), "final": None, "head": []}
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="all", choices=["schur_dslash", "clover_solve", "all"])
    ap.add_argument("--lat", type=int, nargs=4, default=LAT_DEFAULT)
    ap.add_argument("--mass", type=float, default=MASS_DEFAULT)
    args = ap.parse_args()

    dev = pick_v100()
    print(f"[dev87] device={torch.cuda.get_device_name(dev)}")
    g_dev = load_gauge_h5(args.lat, args.mass, device="cuda")
    ce, cei, coo, coi, s, p, av = make_clover_tensors(g_dev, args.lat, args.mass)
    export_qdp_gauge(g_dev, args.lat, args.mass)
    if args.case in ("schur_dslash", "all"):
        print(case_schur_dslash(args.lat, args.mass, g_dev, ce, coo, cei, coi, s, p, av))
    if args.case in ("clover_solve", "all"):
        print(case_clover_solve(args.lat, args.mass, g_dev, ce, coo, cei, coi, s, p, av))


if __name__ == "__main__":
    main()
