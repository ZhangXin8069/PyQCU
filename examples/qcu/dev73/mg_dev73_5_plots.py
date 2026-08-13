#!/usr/bin/env python3
"""dev73_5 —— 图表生成（收敛历史 / 热点 / 加速比 / 耗时对照）。

使用 dataviz 技能验证过的亮色分类/顺序调色板（light mode）：
  series 顺序: blue #2a78d6 → green #008300 → magenta #e87ba4 → yellow #eda100
              → aqua #1baf7a → orange #eb6834 → violet #4a3aa7 → red #e34948
  ink: 主 #0b0b0b / 次 #52514e / 弱 #898781；网格 #e1e0d9；面 #fcfcfb

输入：logs/dev73/dev73_5_bench.json, logs/dev73/dev73_5_ref_conv.json, logs/dev73/dev73_5_verify_*.json
输出：logs/dev73/dev73_5_*.png
"""
import json, os, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

# 注册 CJK 字体（中文标签），Droid Sans Fallback 同时覆盖拉丁字符
for _f in ["/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
           "/usr/share/fonts/truetype/arphic/uming.ttc"]:
    try:
        fm.fontManager.addfont(_f)
    except Exception:
        pass
plt.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback",
                               "AR PL UMing CN", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

LOG_DIR = os.path.expanduser("~/PyQCU/logs/dev73")

# ---- validated palette (dataviz skill, light mode) ----
C = {
    "blue": "#2a78d6", "green": "#008300", "magenta": "#e87ba4",
    "yellow": "#eda100", "aqua": "#1baf7a", "orange": "#eb6834",
    "violet": "#4a3aa7", "red": "#e34948",
}
INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"; GRID = "#e1e0d9"
SURF = "#fcfcfb"
SEQUENTIAL_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
                   "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
                   "#184f95", "#104281"]


def _style(ax):
    ax.set_facecolor(SURF)
    for s in ax.spines.values():
        s.set_color(MUTED)
        s.set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, alpha=0.9)
    ax.grid(False, axis="x")
    ax.tick_params(grid_color=GRID)


def load(path):
    with open(path) as f:
        return json.load(f)


def fig_save(fig, name, dpi=150):
    path = os.path.join(LOG_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print(f"saved {path}")


def plot_conv(results):
    """逐配置收敛历史：MG（蓝）vs 参考 BiStabCG（绿）。"""
    # 按 (lattice, precision) 分组，一个格子一张图，每张图内多个 MG 配置
    groups = {}
    for r in results:
        key = (tuple(r["lattice"]), r["precision"])
        groups.setdefault(key, []).append(r)
    for (lat, prec), rs in groups.items():
        ref_hist = next((r["ref_hist"] for r in rs if r.get("ref_hist")), [])
        fig, ax = plt.subplots(figsize=(8, 4.6))
        _style(ax)
        lat_s = "×".join(str(x) for x in lat)
        lat_f = "x".join(str(x) for x in lat)
        ax.set_title(f"收敛历史  lattice={lat_s}  {prec}  (mass=0.05, atol=1e-6)",
                     color=INK, fontsize=11)
        ax.set_xlabel("迭代次数", color=INK2)
        ax.set_ylabel("Schur 残差 ||r||", color=INK2)
        # 参考
        if ref_hist:
            ax.plot(range(len(ref_hist)), ref_hist, color=C["green"],
                    lw=2.0, label="BiStabCG (参考)", marker="o", ms=3, zorder=3)
        # MG 各配置
        for i, r in enumerate(rs):
            conv = r.get("conv_mg")
            if not conv:
                continue
            color = SEQUENTIAL_BLUE[(i + 2) % len(SEQUENTIAL_BLUE)]
            lbl = (f"MG r={r['restart']} ct={r['ct']:.0e}"
                   f" cmi={r['cmi']} L{r['levels']}")
            ax.plot(range(len(conv)), conv, color=color, lw=1.6,
                    label=lbl, marker="s", ms=2.5, zorder=3)
        ax.set_yscale("log")
        ax.set_ylim(1e-7, 1e3)
        ax.axhline(1e-6, color=MUTED, lw=1, ls="--", zorder=1)
        ax.text(0.99, 1e-6, "atol=1e-6", color=MUTED, fontsize=8,
                ha="right", va="bottom", transform=ax.get_yaxis_transform())
        ax.legend(fontsize=8, frameon=False, loc="best")
        fig.tight_layout()
        fig_save(fig, f"dev73_5_conv_{lat_f}_{prec}.png")


def plot_hotspot(results):
    """PROF_SECTIONS 热点堆叠条形图：fine_iter / vcycle / coarse_solve。"""
    fields = [("fine_iter", "细层迭代 fine_iter", C["blue"]),
              ("vcycle", "V-cycle 修正", C["green"]),
              ("coarse_solve", "粗层求解 coarse_solve", C["orange"]),
              ("coarse_dslash", "粗层 dslash", C["violet"])]
    rs = [r for r in results if r.get("prof")]
    labels = [r["label"] for r in rs]
    vals = {k: [r.get("prof", {}).get(k, 0.0) for r in rs] for k, _, _ in fields}
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    _style(ax)
    ax.set_title("MG 计算热点分解 PROF_SECTIONS (ms)", color=INK, fontsize=11)
    y = np.arange(len(rs))[::-1]
    left = np.zeros(len(rs))
    for k, name, col in fields:
        v = np.array(vals[k])
        ax.barh(y, v, left=left, color=col, label=name, height=0.62,
                edgecolor=SURF, linewidth=0.5)
        left += v
    total = [r.get("mg_med_ms") or r.get("mg_min_ms") or 0 for r in rs]
    for i, yy in enumerate(y):
        t = total[i]
        if t and t > 0:
            ax.text(t + 2, yy, f"{t:.0f}ms", color=INK2, fontsize=7.5,
                    va="center")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("耗时 (ms)", color=INK2)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    ax.grid(False, axis="y")
    fig.tight_layout()
    fig_save(fig, "dev73_5_hotspot.png")


def plot_speedup(results):
    """各配置干净加速比（min 与 median 误差棒）条形图。"""
    rs = sorted([r for r in results if r.get("speedup_min") is not None],
                key=lambda r: r["speedup_min"])
    labels = [r["label"] for r in rs]
    sp = [r["speedup_min"] for r in rs]
    n = len(rs)
    colors = [SEQUENTIAL_BLUE[6 - (n - 1 - i) * 5 // max(n - 1, 1)]
              if n > 1 else C["blue"]
              for i in range(n)]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    _style(ax)
    ax.set_title("MultiGrid vs BiStabCG 加速比（干净测量，min of 5 对）",
                 color=INK, fontsize=11)
    y = np.arange(n)[::-1]
    ax.barh(y, sp, color=colors, height=0.62, edgecolor=SURF, linewidth=0.5)
    for i, yy in enumerate(y):
        r = rs[i]
        lo = r.get("speedup_med")
        ax.text(sp[i] + 0.02, yy,
                f"{sp[i]:.2f}x  (MG {r['mg_min_ms']:.0f} / ref {r['ref_min_ms']:.0f} ms, "
                f"{r['mg_iters']}/{r['ref_iters']} it)", color=INK2, fontsize=7.5,
                va="center")
        if lo is not None and abs(lo - sp[i]) > 0.03:
            ax.plot([sp[i], lo], [yy, yy], color=MUTED, lw=0.8, marker="_",
                    markersize=6)
    ax.axvline(1.0, color=MUTED, lw=1.2, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("加速比 (ref/min / mg/min)", color=INK2)
    ax.set_xlim(0, max(sp) * 1.35 + 0.1)
    fig.tight_layout()
    fig_save(fig, "dev73_5_speedup.png")


def plot_time(results):
    """MG 与参考干净耗时分组条形图（min）。"""
    rs = sorted([r for r in results if r.get("mg_min_ms") is not None],
                key=lambda r: r["ref_min_ms"] - r["mg_min_ms"])
    labels = [r["label"] for r in rs]
    ref = [r["ref_min_ms"] for r in rs]
    mg = [r["mg_min_ms"] for r in rs]
    y = np.arange(len(rs))[::-1]
    h = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    _style(ax)
    ax.set_title("求解耗时对照（干净 min）：BiStabCG（参考） vs MultiGrid",
                 color=INK, fontsize=11)
    ax.barh(y + h / 2, ref, height=h, color=C["green"], label="BiStabCG 参考")
    ax.barh(y - h / 2, mg, height=h, color=C["blue"], label="MultiGrid")
    for i, yy in enumerate(y):
        ax.text(ref[i] + 2, yy + h / 2, f"{ref[i]:.0f}", color=INK2, fontsize=7,
                va="center")
        ax.text(mg[i] + 2, yy - h / 2, f"{mg[i]:.0f}", color=INK2, fontsize=7,
                va="center")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("耗时 (ms)", color=INK2)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig_save(fig, "dev73_5_time.png")


def plot_sweep_curves(results):
    """参数扫描曲线：对默认格子，分别画 r / ct / cmi 对加速比的影响（干净 min）。"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, key, xlabel, title in [
        (axes[0], "restart", "V-cycle 频率 r", "V-cycle 频率扫描 (ct=1e5, cmi=15)"),
        (axes[1], "ct", "最粗层容差因子 ct", "最粗层收敛条件扫描 (r=10, cmi=15)"),
        (axes[2], "cmi", "最粗层最大迭代 cmi", "最粗层迭代上限扫描 (r=10, ct=1e5)"),
    ]:
        _style(ax)
        pts = [r for r in results if tuple(r["lattice"]) == (8, 16, 16, 16)
               and r["precision"] == "c64" and r["levels"] == 2
               and r.get("speedup_min") is not None]
        seen = {}
        for r in pts:
            if key == "restart" and (r["ct"] != 1e5 or r["cmi"] != 15):
                continue
            if key == "ct" and (r["restart"] != 10 or r["cmi"] != 15):
                continue
            if key == "cmi" and (r["restart"] != 10 or r["ct"] != 1e5):
                continue
            seen[r[key]] = r["speedup_min"]
        xs = sorted(seen)
        ax.plot(xs, [seen[x] for x in xs], color=C["blue"], marker="o",
                ms=5, lw=2)
        for x in xs:
            ax.annotate(f"{seen[x]:.2f}", (x, seen[x]), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7.5, color=INK2)
        ax.set_xlabel(xlabel, color=INK2)
        ax.set_ylabel("加速比", color=INK2)
        ax.set_title(title, color=INK, fontsize=10)
        if key == "ct":
            ax.set_xscale("log")
        ax.axhline(1.0, color=MUTED, lw=1, ls="--")
        ax.grid(True, axis="y", color=GRID, linewidth=0.6)
        ax.grid(False, axis="x")
    fig.tight_layout()
    fig_save(fig, "dev73_5_sweep.png")


def main():
    data = load(os.path.join(LOG_DIR, "dev73_5_results.json"))
    results = data["results"]
    if not results:
        print("no results yet")
        return
    plot_conv(results)
    plot_hotspot(results)
    plot_speedup(results)
    plot_time(results)
    plot_sweep_curves(results)


if __name__ == "__main__":
    main()
