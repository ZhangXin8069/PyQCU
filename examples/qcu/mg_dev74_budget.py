#!/usr/bin/env python3
"""dev74 —— 内存/显存/磁盘预算模型（大格子可运行性预测）。

物理依据（c64 = 每复元素 8B，V = Lx·Ly·Lz·Lt 全格点数，奇偶分割后每子格 V/2）：

  张量                     形状                         每格点字节
  -------------------------------------------------------------------------
  g  规范场                 [2,3,3,4,V/2]               288
  fi/fo_ref/fo_mg (×3)      [2,4,3,V/2] ×3              288
  ce/coo/cei/coi (×4)       [4,3,4,3,V/2] ×4            2304
  lonv (L1)                 [48,12,V/2]                 2304
  hnn (L1)                  [2,4,48,48,Vc], Vc=V/32     4608
  hdg (L1)                  [2,2,6,48,48,Vc]            13824
  sit (L1)                  [48,48,Vc]                  576
  -------------------------------------------------------------------------
  小计                                                    24192 ≈ 23.6 KB/V

C++ LatticeSet scratch（每 set_index：device_vec0/1/2 等 ~6 份 SC 场 + r/b/p/v/s/t/
x_o 等）与求解中间量按实测校准系数 α 外推：VRAM(V) = α·V + β。

集群目标（32GB 显存 / 512GB 内存）格子选择依据：V ≤ (32GB - β)/α 且留安全余量。

用法：
    python examples/qcu/mg_dev74_budget.py            # 打印预算表
    python examples/qcu/mg_dev74_budget.py --fit     # 用 dev74_bench.json 实测校准 α/β
"""
import os, sys, json

LOG_DIR = "/root/PyQCU/logs"
# 按格点体积的常量（c64, 2L, E=48）
CONST_PER_V = 24192.0   # bytes per lattice point (Python-side tensors, above)
# 每 CudaSchurOp 实例（LatticeSet scratch）的额外开销 —— 由实测校准
ALPHA_DEFAULT = 2.6     # KB/V 综合校准系数（含 C++ scratch 与求解中间量）实测拟合
BETA_DEFAULT = 512.0    # MB 固定开销（CUDA context、缓存器、Python 环境）

# 2L/3L 粗算子张量（E=48）的磁盘缓存大小（每 level）
def disk_cache_bytes(V, levels=2, E=48):
    """nullvec 缓存（lonv/hnn/hdg/sit, CPU 保存）磁盘占用。"""
    Vc = V / 32.0
    lonv = E * 12 * V / 2 * 8
    hnn = 2 * 4 * E * E * Vc * 8
    hdg = 2 * 2 * 6 * E * E * Vc * 8
    sit = E * E * Vc * 8
    return (lonv + hnn + hdg + sit) * levels


def vram_model(v, alpha_kb_per_v=ALPHA_DEFAULT, beta_mb=BETA_DEFAULT):
    """预测峰值显存（MB）——cold（含粗算子构建）峰值。"""
    return CONST_PER_V * v / 1e6 + alpha_kb_per_v * v / 1024.0 + beta_mb


def vram_model_warm(v, alpha_kb_per_v=ALPHA_DEFAULT, beta_mb=BETA_DEFAULT):
    """预测峰值显存（MB）——warm（nullvec 缓存命中，仅求解）。

    常驻粗算子张量（lonv/hnn/hdg/sit, 24.2 KB/V）+ 求解中间量
    （实测 ref 阶段 ~3 KB/V），叠加 C++ scratch 校准项。
    """
    return (CONST_PER_V * v / 1e6 +
            (alpha_kb_per_v / 11.0) * v / 1024.0 + beta_mb)


def rss_model(v, alpha_ram_kb_per_v=5.0, beta_ram_mb=1200.0):
    """预测进程峰值内存 RSS（MB）—— 粗算子 CPU 副本 + Python/PyTorch 环境。"""
    return alpha_ram_kb_per_v * v / 1024.0 + beta_ram_mb


LATTICES = {
    "local":  [(8, 8, 8, 16), (8, 16, 16, 16), (16, 16, 16, 16)],
    "cluster": [(16, 32, 32, 32), (16, 32, 32, 64), (24, 32, 32, 64)],
}


def fit_from_bench(bench_path=None):
    """从 dev74_bench.json 实测（peak_vram_mb, V）线性拟合 α/β。

    返回 (alpha_kb_per_v, beta_mb) 或 None（无数据）。
    """
    path = bench_path or os.path.join(LOG_DIR, "dev74_bench.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    pts = [(r["lattice"], r.get("peak_vram_mb")) for r in data.get("results", [])
           if r.get("peak_vram_mb")]
    if len(pts) < 2:
        return None
    import numpy as np
    Vs = np.array([L[0] * L[1] * L[2] * L[3] for L, _ in pts], dtype=float)
    y = np.array([m for _, m in pts], dtype=float)
    # VRAM = a*V + b（MB）；a = CONST_PER_V/1e6 + alpha/1024
    a, b = np.polyfit(Vs, y, 1)
    alpha = (a * 1e6 - CONST_PER_V) / (1e6 / 1024.0)
    return (float(alpha), float(b))


def budget_table(mode="cluster", alpha=None, beta=None):
    if alpha is None:
        alpha = ALPHA_DEFAULT
    if beta is None:
        beta = BETA_DEFAULT
    rows = []
    for L in LATTICES[mode]:
        V = L[0] * L[1] * L[2] * L[3]
        vram = vram_model(V, alpha, beta)
        vram_warm = vram_model_warm(V, alpha, beta)
        rss = rss_model(V)
        disk = disk_cache_bytes(V, levels=2)
        rows.append({"lattice": list(L), "V": V,
                     "pred_vram_mb": round(vram), "pred_vram_warm_mb": round(vram_warm),
                     "pred_rss_mb": round(rss),
                     "pred_disk_mb": round(disk / 1e6, 1),
                     "vram_frac_32g": round(vram / (32 * 1024), 3),
                     "vram_warm_frac_32g": round(vram_warm / (32 * 1024), 3),
                     "rss_frac_512g": round(rss / (512 * 1024), 3)})
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="cluster", choices=["local", "cluster"])
    ap.add_argument("--fit", action="store_true")
    args = ap.parse_args()
    alpha, beta = ALPHA_DEFAULT, BETA_DEFAULT
    if args.fit:
        f = fit_from_bench()
        if f:
            alpha, beta = f
            print(f"[fit] alpha={alpha:.2f} KB/V  beta={beta:.0f} MB "
                  f"(从 dev74_bench.json 实测)")
        else:
            print("[fit] dev74_bench.json 无实测数据，使用默认系数")
    rows = budget_table(args.mode, alpha, beta)
    print(f"{'lattice':20s} {'V':>9s} {'VRAM_cold':>10s} {'VRAM_warm':>10s} "
          f"{'RSS(MB)':>9s} {'disk(MB)':>9s} {'cold/32G':>9s} {'warm/32G':>9s}")
    for r in rows:
        print(f"{'x'.join(map(str, r['lattice'])):20s} {r['V']:9d} "
              f"{r['pred_vram_mb']:10d} {r['pred_vram_warm_mb']:10d} "
              f"{r['pred_rss_mb']:9d} "
              f"{r['pred_disk_mb']:9.1f} {r['vram_frac_32g']:9.3f} "
              f"{r['vram_warm_frac_32g']:9.3f}")
    out = os.path.join(LOG_DIR, f"dev74_budget_{args.mode}.json")
    with open(out, "w") as f:
        json.dump({"alpha_kb_per_v": alpha, "beta_mb": beta, "rows": rows},
                  f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
