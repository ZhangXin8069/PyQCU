"""显存/内存/磁盘预算模型（大格子可运行性预测）。

整合自 examples/qcu/dev74/mg_dev74_budget.py（主本，含物理依据）与
logs/test11/main.py::vram_model/rss_model/disk_cache_bytes/budget_table/fit_from_bench
（16/32GB 档位分数并入 vram_gb 参数；E 已参数化，最新套件 DOF_LIST 用 E=24 时显式传入）。

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
"""
import json
import os
from typing import Dict, List, Optional, Tuple

# 按格点体积的常量（c64, 2L, E=48）
CONST_PER_V = 24192.0   # bytes per lattice point（Python 侧张量，见上表）
ALPHA_DEFAULT = 2.6     # KB/V 综合校准系数（含 C++ scratch 与求解中间量）实测拟合
BETA_DEFAULT = 512.0    # MB 固定开销（CUDA context、缓存器、Python 环境）

LATTICES = {
    "local": [(8, 8, 8, 16), (8, 16, 16, 16), (16, 16, 16, 16)],
    "cluster": [(16, 32, 32, 32), (16, 32, 32, 64), (24, 32, 32, 64)],
}


def disk_cache_bytes(v: float, levels: int = 2, E: int = 48,
                     E_prev: int = 12) -> float:
    """nullvec 缓存（lonv/hnn/hdg/sit，CPU 保存）磁盘占用预测（bytes）。

    lonv 形状 [E,E_prev,V/2]；hnn/hdg/sit 为 33-tensor stencil（Vc=V/32）。
    """
    Vc = v / 32.0
    lonv = E * E_prev * v / 2 * 8
    hnn = 2 * 4 * E * E * Vc * 8
    hdg = 2 * 2 * 6 * E * E * Vc * 8
    sit = E * E * Vc * 8
    return (lonv + hnn + hdg + sit) * levels


def vram_model(v: float, alpha_kb_per_v: float = ALPHA_DEFAULT,
               beta_mb: float = BETA_DEFAULT) -> float:
    """预测峰值显存（MB）—— cold（含粗算子构建）峰值。"""
    return CONST_PER_V * v / 1e6 + alpha_kb_per_v * v / 1024.0 + beta_mb


def vram_model_warm(v: float, alpha_kb_per_v: float = ALPHA_DEFAULT,
                    beta_mb: float = BETA_DEFAULT) -> float:
    """预测峰值显存（MB）—— warm（nullvec 缓存命中，仅求解）。

    常驻粗算子张量（lonv/hnn/hdg/sit，24.2 KB/V）+ 求解中间量
    （实测 ref 阶段 ~α/11 KB/V），叠加 C++ scratch 校准项。
    """
    return (CONST_PER_V * v / 1e6 +
            (alpha_kb_per_v / 11.0) * v / 1024.0 + beta_mb)


def rss_model(v: float, alpha_ram_kb_per_v: float = 5.0,
              beta_ram_mb: float = 1200.0) -> float:
    """预测进程峰值内存 RSS（MB）—— 粗算子 CPU 副本 + Python/PyTorch 环境。"""
    return alpha_ram_kb_per_v * v / 1024.0 + beta_ram_mb


def fit_from_bench(bench_path: Optional[str] = None,
                   log_dir: Optional[str] = None) -> Optional[Tuple[float, float]]:
    """从 bench 实测 json（results[].lattice/.peak_vram_mb）线性拟合 α。

    返回 (alpha_kb_per_v, beta_mb)；数据不足两个点或文件缺失返回 None。
    """
    path = bench_path or os.path.join(
        log_dir or os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), "logs"),
        "bench.json")
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
    # VRAM(MB) = a*V + b；a = CONST_PER_V/1e6 + alpha/1024 → 反解 alpha
    a, b = np.polyfit(Vs, y, 1)
    alpha = (a * 1e6 - CONST_PER_V) / (1e6 / 1024.0)
    return (float(alpha), float(b))


def budget_table(mode: str = "cluster", vram_gb: int = 32,
                 alpha: Optional[float] = None,
                 beta: Optional[float] = None,
                 levels: int = 2, E: int = 48) -> List[Dict]:
    """预算表：各格子的显存/RSS/磁盘预测与占档比例。"""
    if alpha is None:
        alpha = ALPHA_DEFAULT
    if beta is None:
        beta = BETA_DEFAULT
    rows = []
    for L in LATTICES[mode]:
        V = L[0] * L[1] * L[2] * L[3]
        vram_cold = vram_model(V, alpha, beta)
        vram_warm = vram_model_warm(V, alpha, beta)
        rss = rss_model(V)
        disk = disk_cache_bytes(V, levels=levels, E=E)
        rows.append({"lattice": list(L), "V": V,
                     "pred_vram_mb": round(vram_cold),
                     "pred_vram_warm_mb": round(vram_warm),
                     "pred_rss_mb": round(rss),
                     "pred_disk_mb": round(disk / 1e6, 1),
                     "vram_frac": round(vram_cold / (vram_gb * 1024), 3),
                     "vram_warm_frac": round(vram_warm / (vram_gb * 1024), 3),
                     "rss_frac_512g": round(rss / (512 * 1024), 3),
                     "vram_gb": vram_gb})
    return rows
