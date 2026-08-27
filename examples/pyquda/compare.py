"""阶段 3（聚合进程）：pyqcu vs pyquda 结果对比 + 性能对比 + 作图。

用法：python examples/pyquda/compare.py [--lat 8 8 8 16] [--mass 0.05] [--csw 1.0]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from common import (
    DATA_DIR, KAPPA_PYQCU, LAT_DEFAULT, MASS, OUT_DIR,
    linreg_scale, quda_fermion_to_pyqcu, rel_diff, save_json,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"figure.dpi": 110, "font.size": 10})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=int, nargs=4, default=LAT_DEFAULT)
    ap.add_argument("--mass", type=float, default=MASS)
    ap.add_argument("--csw", type=float, default=1.0)
    args = ap.parse_args()

    lat = list(args.lat)
    tag = "x".join(map(str, lat))
    data_dir = DATA_DIR / tag
    scale = args.mass + 4.0  # D_mass = (m+4)*D_kappa（dev87 G2 锚定）

    # ---------------- 数据
    import h5py

    with h5py.File(data_dir / "input.h5", "r") as f:
        b_p = np.asarray(f["b_p"])
    with h5py.File(data_dir / "pyqcu.h5", "r") as f:
        x_p = np.asarray(f["x_p"])
        y_p = np.asarray(f["y_p"]) if "y_p" in f else None
        y_hop = np.asarray(f["y_hop"]) if "y_hop" in f else None
    with h5py.File(data_dir / "pyquda.h5", "r") as f:
        x_q = np.asarray(f["x_q"])
        q_hist = np.asarray(f["iter_hist"]) if "iter_hist" in f else None
    x_q_p = quda_fermion_to_pyqcu(x_q, lat)  # (2,T,Z,Y,X/2,4,3) -> (4,3,X,Y,Z,T)

    pj = json.loads((OUT_DIR / f"pyqcu_{tag}.json").read_text())
    qj = json.loads((OUT_DIR / f"pyquda_{tag}.json").read_text())
    qpj = json.loads((OUT_DIR / f"pyquda_perf_{tag}.json").read_text())

    # ---------------- 结果对比
    c, rd_scale = linreg_scale(x_p, x_q_p)
    rd_raw = rel_diff(x_p, x_q_p)
    rd_mass = rel_diff(x_p, x_q_p * scale)
    print(f"[compare] x_p vs x_q: rel_diff(raw)={rd_raw:.3e} "
          f"regress c={c:.6f} (expect {scale:.4f}) rel={rd_scale:.3e}")
    print(f"[compare] x_p vs (m+4)*x_q: rel_diff={rd_mass:.3e}")

    # 两侧各自"代入原方程"残差（相同归一化的 D_mass 方程）
    # pyqcu 侧残差由 run_pyqcu 在 D_kappa 下计算；统一换算到 D_mass 后应与 quda true_res 同量级
    rp = pj.get("rel_res_full", float("nan"))
    rq = qj.get("true_res", float("nan"))
    print(f"[compare] rel_res: pyqcu(D x_p-b)/|b|={rp:.3e}  quda true_res={rq:.3e}")

    res = {
        "lat": lat, "mass": args.mass, "kappa_pyqcu": KAPPA_PYQCU,
        "scale_m_plus_4": scale,
        "rel_diff_raw": rd_raw, "regress_c": c, "regress_rel": rd_scale,
        "rel_diff_scaled": rd_mass,
        "rel_res_pyqcu": rp, "true_res_quda": rq,
        "iters_pyqcu": pj.get("iters"), "iters_quda": qj.get("iters"),
        "wall_pyqcu_s": pj.get("wall_s"), "wall_quda_s": qj.get("wall_s"),
        "secs_quda_internal": qj.get("secs"),
    }

    # ---------------- 算子级中间量对比（dslash 跳跃部分）
    if y_hop is not None:
        with h5py.File(data_dir / "pyquda_dslash.h5", "r") as f:
            y_q = np.asarray(f["y_q"])
        y_q_p = quda_fermion_to_pyqcu(y_q, lat)
        c_hop, r_hop = linreg_scale(y_hop, y_q_p)
        res["hop_regress_c"] = c_hop
        res["hop_regress_rel"] = r_hop
        print(f"[compare] hop y_p vs y_q: regress c={c_hop:.6f} rel={r_hop:.3e}")

    # ---------------- Clover 解对比
    try:
        with h5py.File(data_dir / "pyqcu_clover.h5", "r") as f:
            xc_p = np.asarray(f["x_cl_p"])
        with h5py.File(data_dir / "pyquda_clover.h5", "r") as f:
            xc_q = np.asarray(f["xc_q"])
            qc_hist = np.asarray(f["iter_hist"]) if "iter_hist" in f else None
        xc_q_p = quda_fermion_to_pyqcu(xc_q, lat)
        c_c, rd_c = linreg_scale(xc_p, xc_q_p)
        rd_cm = rel_diff(xc_p, xc_q_p * scale)
        pcj = json.loads((OUT_DIR / f"pyqcu_clover_{tag}.json").read_text())
        qcj = json.loads((OUT_DIR / f"pyquda_clover_{tag}.json").read_text())
        res["clover_rel_diff_scaled"] = rd_cm
        res["clover_regress_c"] = c_c
        res["clover_regress_rel"] = rd_c
        res["clover_iters_pyqcu"] = pcj.get("iters")
        res["clover_iters_quda"] = qcj.get("iters")
        res["clover_rel_res_pyqcu"] = pcj.get("rel_res_full")
        res["clover_true_res_quda"] = qcj.get("true_res")
        print(f"[compare] clover: regress c={c_c:.6f} rel={rd_c:.3e} "
              f"scaled rel_diff={rd_cm:.3e}")
    except FileNotFoundError:
        print("[compare] clover data missing, skip")

    save_json(f"compare_{tag}", res)

    # ---------------- 性能对比 + 作图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    # 逐迭代残差曲线（Wilson + Clover）
    hist_p = pj.get("hist", [])
    ax = axes[0]
    if hist_p:
        ax.semilogy(np.arange(1, len(hist_p) + 1), hist_p, "o-", ms=3, label="pyqcu BiCGStab (Wilson)")
    if q_hist is not None and len(q_hist):
        ax.semilogy(q_hist[:, 0], q_hist[:, 1], "s-", ms=3, label="pyquda CG (Wilson)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("residual")
    ax.set_title(f"residual vs iteration ({tag})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    # Clover 残差曲线
    ax = axes[1]
    try:
        pcj = json.loads((OUT_DIR / f"pyqcu_clover_{tag}.json").read_text())
        with h5py.File(data_dir / "pyquda_clover.h5", "r") as f:
            qc_hist = np.asarray(f["iter_hist"]) if "iter_hist" in f else None
        if pcj.get("hist"):
            ax.semilogy(np.arange(1, len(pcj["hist"]) + 1), pcj["hist"], "o-", ms=3,
                        label="pyqcu BiCGStab (Clover)")
        if qc_hist is not None and len(qc_hist):
            ax.semilogy(qc_hist[:, 0], qc_hist[:, 1], "s-", ms=3, label="pyquda CG (Clover)")
    except FileNotFoundError:
        pass
    ax.set_xlabel("iteration")
    ax.set_ylabel("residual")
    ax.set_title(f"clover residual vs iteration ({tag})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    # 迭代次数 + 耗时柱状
    ax = axes[2]
    labels = ["pyqcu\nBiCGStab", "pyquda\nCG"]
    iters = [pj.get("iters", 0), qj.get("iters", 0)]
    walls = [pj.get("wall_s", 0), qj.get("wall_s", 0)]
    xpos = np.arange(2)
    w = 0.35
    b1 = ax.bar(xpos - w / 2, iters, w, label="iterations")
    ax2 = ax.twinx()
    b2 = ax2.bar(xpos + w / 2, walls, w, color="tab:orange", label="wall time (s)")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("iterations")
    ax2.set_ylabel("wall time (s)")
    ax.set_title(f"performance ({tag})")
    for b in list(b1) + list(b2):
        ax.annotate(f"{b.get_height():.2g}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=8)
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    png = OUT_DIR / f"compare_{tag}.png"
    fig.savefig(png)
    print(f"[compare] plot -> {png}")

    # ---------------- 汇总报告 md
    md = OUT_DIR / f"compare_{tag}.md"
    md.write_text(f"""# pyqcu vs pyquda 对比报告（lat={'x'.join(map(str,lat))}, m={args.mass}）

## 结果对比
- 解互比（pyqcu x_p vs pyquda x_q）：raw rel_diff = {rd_raw:.3e}
- 回归定标：x_p ≈ {c:.6f} · x_q（期望 {scale:.4f} = m+4，dev87 G2 锚定），回归残差 {rd_scale:.3e}
- 归一化后（x_p vs (m+4)·x_q）：rel_diff = {rd_mass:.3e}
- 代入原方程残差：pyqcu rel_res = {rp:.3e}；pyquda true_res = {rq:.3e}

## 性能对比
| 指标 | pyqcu BiCGStab | pyquda CG |
|---|---|---|
| 迭代次数 | {pj.get('iters')} | {qj.get('iters')} |
| 总耗时 (s) | {pj.get('wall_s'):.3f} | {qj.get('wall_s'):.3f} |
| 平均每步耗时 (s) | {pj.get('avg_matvec_s', 0):.4f}（单次 matvec） | {qj.get('secs') / qj.get('iters', 1):.4f}（QUDA 内部） |

图：`compare_{tag}.png`（逐迭代残差曲线 + 迭代/耗时柱状）。
""")
    print(f"[compare] report -> {md}")


if __name__ == "__main__":
    main()