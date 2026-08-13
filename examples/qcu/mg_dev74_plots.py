#!/usr/bin/env python3
"""dev74 —— 生成图表（性能 + 资源 + 收敛）。

图（logs/dev74_*.png）：
  * dev74_speedup.png —— 加速比 vs 格点体积（本机实测 + dev73_5 V100 参考）
  * dev74_vram.png    —— 峰值显存 vs 格点体积（实测 cold/warm + 校准模型外推）
  * dev74_time.png    —— ref/MG 耗时 vs 格点体积
  * dev74_conv_*.png  —— MG 收敛历史（每配置）
  * dev74_budget.png  —— 集群格子预算预测（cold/warm vs 32G 极限线）
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG_DIR = "/root/PyQCU/logs"


def main():
    with open(os.path.join(LOG_DIR, "dev74_results.json")) as f:
        data = json.load(f)
    results = data["results"]
    V = [r["lattice"][0] * r["lattice"][1] * r["lattice"][2] * r["lattice"][3]
         for r in results]
    sp = [r.get("speedup_min") for r in results]
    ref_ms = [r.get("ref_min_ms") for r in results]
    mg_ms = [r.get("mg_min_ms") for r in results]
    cold = [r.get("peak_vram_cold_mb") for r in results]
    warm = [r.get("peak_vram_warm_mb") for r in results]
    labels = [r["label"] for r in results]

    # ---- speedup vs V ----
    plt.figure(figsize=(7, 4.5))
    plt.plot(V, sp, "o-", label="dev74 本机 RTX4060 (c64, 2L)")
    # dev73_5 V100 参考点
    ref_v = [8 * 8 * 8 * 16, 8 * 16 * 16 * 16, 16 * 16 * 16 * 16, 8 * 16 * 16 * 32]
    ref_sp = [2.43, 1.16, 0.81, 1.11]
    plt.plot(ref_v, ref_sp, "s--", label="dev73_5 V100-32G 参考")
    plt.axhline(1.0, color="gray", ls=":", lw=0.8)
    plt.xscale("log")
    plt.xlabel("lattice volume V")
    plt.ylabel("speedup (ref/MG)")
    plt.title("dev74: speedup vs V (local RTX4060)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, "dev74_speedup.png"), dpi=130)

    # ---- vram vs V（实测 + 模型外推）----
    plt.figure(figsize=(7, 4.5))
    plt.plot(V, cold, "o-", label="cold 实测（含粗算子构建）")
    plt.plot(V, warm, "o-", label="warm 实测（缓存命中求解）")
    Vs = np.logspace(np.log10(8e3), np.log10(2e6), 40)
    model_cold = (24192 + 30.83 * 1024) * Vs / 1e6 - 27
    model_warm = (24192 + 2.8 * 1024) * Vs / 1e6 - 27
    plt.plot(Vs, model_cold, "--", color="C0", alpha=0.6, label="cold 模型 53KB/V")
    plt.plot(Vs, model_warm, "--", color="C1", alpha=0.6, label="warm 模型 27KB/V")
    plt.axhline(32 * 1024, color="red", ls="--", lw=1.2, label="32GB 显存极限")
    plt.xscale("log")
    plt.xlabel("lattice volume V")
    plt.ylabel("peak VRAM (MB)")
    plt.title("dev74: peak VRAM vs V (measured+extrapolated)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, "dev74_vram.png"), dpi=130)

    # ---- time vs V ----
    plt.figure(figsize=(7, 4.5))
    plt.plot(V, ref_ms, "o-", label="BiStabCG (min)")
    plt.plot(V, mg_ms, "s-", label="MG (min)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("lattice volume V")
    plt.ylabel("time (ms, log)")
    plt.title("dev74: solve time (local)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, "dev74_time.png"), dpi=130)

    # ---- 收敛历史 ----
    for r in results:
        conv = r.get("conv_mg")
        if not conv:
            continue
        plt.figure(figsize=(6.5, 4))
        plt.semilogy(conv, "o-", ms=3, label="MG")
        rh = r.get("ref_hist")
        if rh:
            plt.semilogy(rh, "s-", ms=3, label="BiStabCG (Python 复现)")
        plt.xlabel("iteration")
        plt.ylabel("residual norm")
        plt.title(f"dev74: {r['label']}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(LOG_DIR,
                                 f"dev74_conv_{r['label'].split('_')[0]}.png"),
                    dpi=130)

    # ---- 集群预算 ----
    bp = os.path.join(LOG_DIR, "dev74_budget_cluster.json")
    if os.path.exists(bp):
        with open(bp) as f:
            budget = json.load(f)
        rows = budget["rows"]
        Vc = [r["V"] for r in rows]
        cold_gb = [r["pred_vram_mb"] / 1024 for r in rows]
        warm_gb = [r["pred_vram_warm_mb"] / 1024 for r in rows]
        x = np.arange(len(rows))
        plt.figure(figsize=(7, 4.5))
        plt.bar(x - 0.2, cold_gb, 0.4, label="cold（首次构建）")
        plt.bar(x + 0.2, warm_gb, 0.4, label="warm（缓存命中）")
        plt.axhline(32, color="red", ls="--", lw=1.2, label="32GB 极限")
        plt.xticks(x, [f"{r['V']//1024}k" for r in rows])
        plt.ylabel("VRAM (GB)")
        plt.title("dev74: cluster lattice budget (calibrated)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(LOG_DIR, "dev74_budget.png"), dpi=130)

    print(f"wrote dev74_*.png ({len([f for f in os.listdir(LOG_DIR) if f.startswith('dev74') and f.endswith('.png')])} figures)")


if __name__ == "__main__":
    main()
