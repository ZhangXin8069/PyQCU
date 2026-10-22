"""基准图表共用层（收敛历史 / 热点分解 / 加速比 / 耗时对照 / 参数扫描）。

整合来源（2026-10-15 收敛到本模块；四处原副本为同一实现的复制，随历史套件删除）：

- `pyqcu/testing/qcu/multigrid/legacy/make_plots.py`（dev73_5，最完整的 5 类图实现，本模块以它为准）；
- `pyqcu/testing/qcu/multigrid/scaling/{make_plots,validation_plots}.py`（dev74，同族复制）；
- `pyqcu/testing/qcu/multigrid/volume24/*/generate_assets.py`（同调色板 / `_style` / 字体注册的复制，
  其中 `final` 保留为规范套件并改为调用本模块的样式层）。

设计约束：

- **顶层不得 `import matplotlib`** —— `pyqcu.tools` 在全库（含未装 matplotlib 的 GPU 节点）被导入；
  绘图依赖全部惰性化在函数内（同 `pyqcu/torch/solver/_legacy_multigrid.py::plot` 的做法）。
- 调色板、图形尺寸、字号与原实现逐字一致，接线方（`volume24/final/generate_assets.py`）既有
  证据图不产生像素级变化。

记录模式（record，`dict`，键按需可选）：

| 键 | 含义 |
|---|---|
| `label` | 条形图 y 轴标签；缺省由 `levels/restart/ct/cmi` 生成 |
| `lattice` / `precision` | 4 元整数格子 / `"c64"`、`"c128"` |
| `levels` / `restart` / `ct` / `cmi` | 多重网格扫描参数 |
| `conv_mg` / `ref_hist` | MG 与参考求解器残差历史（list[float]） |
| `prof` | PROF_SECTIONS 分解（ms）：`fine_iter`/`vcycle`/`coarse_solve`/`coarse_dslash` |
| `mg_min_ms`/`mg_med_ms`/`ref_min_ms` | 干净计时（ms） |
| `speedup_min`/`speedup_med` | 加速比（min / median） |
| `mg_iters`/`ref_iters`/`ref_times_ms` | 迭代数与参考计时序列 |

调色板出处：dataviz 技能验证过的亮色分类/顺序调色板（light mode）。
"""
import json
import os

# ---- validated palette (dataviz skill, light mode) ----
BENCH_PALETTE = {
    "blue": "#2a78d6", "green": "#008300", "magenta": "#e87ba4",
    "yellow": "#eda100", "aqua": "#1baf7a", "orange": "#eb6834",
    "violet": "#4a3aa7", "red": "#e34948",
}
BENCH_SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
                  "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
                  "#184f95", "#104281"]
BENCH_INK = "#0b0b0b"
BENCH_INK2 = "#52514e"
BENCH_MUTED = "#898781"
BENCH_GRID = "#e1e0d9"
BENCH_SURF = "#fcfcfb"

_CJK_FONT_PATHS = ("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
                   "/usr/share/fonts/truetype/arphic/uming.ttc")
_CJK_FONT_FAMILY = ["DejaVu Sans", "Droid Sans Fallback",
                    "AR PL UMing CN", "Noto Sans CJK SC"]

_MPL_READY = False


def _mpl():
    """惰性导入 matplotlib（Agg）+ 注册 CJK 字体；返回 `(plt, np)`。"""
    global _MPL_READY
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    if not _MPL_READY:
        import matplotlib.font_manager as fm
        for path in _CJK_FONT_PATHS:
            try:
                fm.fontManager.addfont(path)
            except Exception:
                pass
        plt.rcParams["font.family"] = _CJK_FONT_FAMILY
        plt.rcParams["axes.unicode_minus"] = False
        _MPL_READY = True
    return plt, np


def mpl_cjk_setup():
    """注册 CJK 字体并设置 rcParams（供自行作图的调用方复用同一字体约定）。"""
    _mpl()


def style_axes(ax):
    """统一坐标轴样式（网格 / 脊线 / 刻度配色）。"""
    ax.set_facecolor(BENCH_SURF)
    for s in ax.spines.values():
        s.set_color(BENCH_MUTED)
        s.set_linewidth(0.8)
    ax.tick_params(colors=BENCH_INK2, labelsize=9)
    ax.grid(True, axis="y", color=BENCH_GRID, linewidth=0.6, alpha=0.9)
    ax.grid(False, axis="x")
    ax.tick_params(grid_color=BENCH_GRID)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_figure(fig, path, dpi=150):
    """保存并关闭图形；自动创建父目录，返回写入路径。"""
    path = str(path)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=BENCH_SURF)
    plt, _ = _mpl()
    plt.close(fig)
    print(f"saved {path}")
    return path


def short_label(record, keys=("levels", "restart", "ct", "cmi")):
    """紧凑显示名：`L2 r10 ct1e+05 cmi15`（按 keys 取舍，ct 用科学计数法）。"""
    prefixes = {"levels": "L", "restart": "r", "ct": "ct", "cmi": "cmi"}
    parts = []
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        prefix = prefixes.get(key, key)
        if key == "ct":
            parts.append(f"{prefix}{value:.0e}")
        else:
            parts.append(f"{prefix}{value}")
    return " ".join(parts)


def label_of(record, keys=("levels", "restart", "ct", "cmi")):
    return record.get("label") or short_label(record, keys)


def plot_convergence(results, out_dir, prefix="bench", lattice=None,
                     label_fontsize=8.0, dpi=150):
    """逐 (lattice, precision) 组的收敛历史：MG（顺序蓝）vs 参考 BiStabCG（绿）。"""
    plt, _ = _mpl()
    lattice = tuple(lattice) if lattice is not None else None
    groups = {}
    for r in results:
        key = (tuple(r["lattice"]), r["precision"])
        if lattice is not None and key[0] != lattice:
            continue
        groups.setdefault(key, []).append(r)
    saved = []
    for (lat, prec), rs in groups.items():
        ref_hist = next((r["ref_hist"] for r in rs if r.get("ref_hist")), [])
        fig, ax = plt.subplots(figsize=(8, 4.6))
        style_axes(ax)
        lat_s = "×".join(str(x) for x in lat)
        lat_f = "x".join(str(x) for x in lat)
        ax.set_title(f"收敛历史  lattice={lat_s}  {prec}  (mass=0.05, atol=1e-6)",
                     color=BENCH_INK, fontsize=11)
        ax.set_xlabel("迭代次数", color=BENCH_INK2)
        ax.set_ylabel("Schur 残差 ||r||", color=BENCH_INK2)
        if ref_hist:
            ax.plot(range(len(ref_hist)), ref_hist, color=BENCH_PALETTE["green"],
                    lw=2.0, label="BiStabCG (参考)", marker="o", ms=3, zorder=3)
        for i, r in enumerate(rs):
            conv = r.get("conv_mg")
            if not conv:
                continue
            color = BENCH_SEQ_BLUE[(i + 2) % len(BENCH_SEQ_BLUE)]
            ax.plot(range(len(conv)), conv, color=color, lw=1.6,
                    label=label_of(r), marker="s", ms=2.5, zorder=3)
        ax.set_yscale("log")
        ax.set_ylim(1e-7, 1e3)
        ax.axhline(1e-6, color=BENCH_MUTED, lw=1, ls="--", zorder=1)
        ax.text(0.99, 1e-6, "atol=1e-6", color=BENCH_MUTED, fontsize=8,
                ha="right", va="bottom", transform=ax.get_yaxis_transform())
        ax.legend(fontsize=8, frameon=False, loc="best")
        fig.tight_layout()
        saved.append(save_figure(fig, os.path.join(out_dir, f"{prefix}_conv_{lat_f}_{prec}.png"), dpi))
    return saved


def plot_hotspot(results, out_dir, prefix="bench", label_fontsize=8.0, dpi=150):
    """PROF_SECTIONS 热点堆叠条形图：fine_iter / vcycle / coarse_solve / coarse_dslash。"""
    plt, np = _mpl()
    fields = [("fine_iter", "细层迭代 fine_iter", BENCH_PALETTE["blue"]),
              ("vcycle", "V-cycle 修正", BENCH_PALETTE["green"]),
              ("coarse_solve", "粗层求解 coarse_solve", BENCH_PALETTE["orange"]),
              ("coarse_dslash", "粗层 dslash", BENCH_PALETTE["violet"])]
    rs = [r for r in results if r.get("prof")]
    if not rs:
        return []
    labels = [label_of(r) for r in rs]
    vals = {k: [r.get("prof", {}).get(k, 0.0) for r in rs] for k, _, _ in fields}
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    style_axes(ax)
    ax.set_title("MG 计算热点分解 PROF_SECTIONS (ms)", color=BENCH_INK, fontsize=11)
    y = np.arange(len(rs))[::-1]
    left = np.zeros(len(rs))
    for k, name, col in fields:
        v = np.array(vals[k])
        ax.barh(y, v, left=left, color=col, label=name, height=0.62,
                edgecolor=BENCH_SURF, linewidth=0.5)
        left += v
    total = [r.get("mg_med_ms") or r.get("mg_min_ms") or 0 for r in rs]
    for i, yy in enumerate(y):
        t = total[i]
        if t and t > 0:
            ax.text(t + 2, yy, f"{t:.0f}ms", color=BENCH_INK2, fontsize=7.5,
                    va="center")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=label_fontsize)
    ax.set_xlabel("耗时 (ms)", color=BENCH_INK2)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.grid(True, axis="x", color=BENCH_GRID, linewidth=0.6)
    ax.grid(False, axis="y")
    fig.tight_layout()
    return [save_figure(fig, os.path.join(out_dir, f"{prefix}_hotspot.png"), dpi)]


def plot_speedup(results, out_dir, prefix="bench", pairs=None,
                 label_fontsize=8.0, dpi=150):
    """各配置干净加速比（min 为条形，median 以误差短线叠加）。"""
    plt, np = _mpl()
    rs = sorted([r for r in results if r.get("speedup_min") is not None],
                key=lambda r: r["speedup_min"])
    if not rs:
        return []
    labels = [label_of(r) for r in rs]
    sp = [r["speedup_min"] for r in rs]
    n = len(rs)
    colors = [BENCH_SEQ_BLUE[6 - (n - 1 - i) * 5 // max(n - 1, 1)]
              if n > 1 else BENCH_PALETTE["blue"]
              for i in range(n)]
    if pairs is None:
        pairs = max((len(r.get("ref_times_ms", [])) for r in rs), default=3)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    style_axes(ax)
    ax.set_title(f"MultiGrid vs BiStabCG 加速比（干净测量，min of {pairs} 对）",
                 color=BENCH_INK, fontsize=11)
    y = np.arange(n)[::-1]
    ax.barh(y, sp, color=colors, height=0.62, edgecolor=BENCH_SURF, linewidth=0.5)
    for i, yy in enumerate(y):
        r = rs[i]
        lo = r.get("speedup_med")
        ax.text(sp[i] + 0.02, yy,
                f"{sp[i]:.2f}x  (MG {r['mg_min_ms']:.0f} / ref {r['ref_min_ms']:.0f} ms, "
                f"{r['mg_iters']}/{r['ref_iters']} it)", color=BENCH_INK2, fontsize=7.5,
                va="center")
        if lo is not None and abs(lo - sp[i]) > 0.03:
            ax.plot([sp[i], lo], [yy, yy], color=BENCH_MUTED, lw=0.8, marker="_",
                    markersize=6)
    ax.axvline(1.0, color=BENCH_MUTED, lw=1.2, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=label_fontsize)
    ax.set_xlabel("加速比 (ref/min / mg/min)", color=BENCH_INK2)
    ax.set_xlim(0, max(sp) * 1.35 + 0.1)
    fig.tight_layout()
    return [save_figure(fig, os.path.join(out_dir, f"{prefix}_speedup.png"), dpi)]


def plot_time(results, out_dir, prefix="bench", label_fontsize=8.0, dpi=150):
    """MG 与参考干净耗时（min）分组条形图。"""
    plt, np = _mpl()
    rs = sorted([r for r in results if r.get("mg_min_ms") is not None],
                key=lambda r: r["ref_min_ms"] - r["mg_min_ms"])
    if not rs:
        return []
    labels = [label_of(r) for r in rs]
    ref = [r["ref_min_ms"] for r in rs]
    mg = [r["mg_min_ms"] for r in rs]
    y = np.arange(len(rs))[::-1]
    h = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    style_axes(ax)
    ax.set_title("求解耗时对照（干净 min）：BiStabCG（参考） vs MultiGrid",
                 color=BENCH_INK, fontsize=11)
    ax.barh(y + h / 2, ref, height=h, color=BENCH_PALETTE["green"], label="BiStabCG 参考")
    ax.barh(y - h / 2, mg, height=h, color=BENCH_PALETTE["blue"], label="MultiGrid")
    for i, yy in enumerate(y):
        ax.text(ref[i] + 2, yy + h / 2, f"{ref[i]:.0f}", color=BENCH_INK2, fontsize=7,
                va="center")
        ax.text(mg[i] + 2, yy - h / 2, f"{mg[i]:.0f}", color=BENCH_INK2, fontsize=7,
                va="center")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=label_fontsize)
    ax.set_xlabel("耗时 (ms)", color=BENCH_INK2)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    return [save_figure(fig, os.path.join(out_dir, f"{prefix}_time.png"), dpi)]


def plot_sweep_curves(results, out_dir, prefix="bench", lattice=None,
                      precision="c64", levels=2, ref_points=None, dpi=150):
    """参数扫描曲线：restart / ct / cmi 对加速比的影响（干净 min）。

    基线点 `ref_points` 缺省为 `{"restart": 10, "ct": 1e5, "cmi": 15}`（dev73/dev74 口径）；
    改扫其它基线时显式传入。
    """
    plt, _ = _mpl()
    if ref_points is None:
        ref_points = {"restart": 10, "ct": 1e5, "cmi": 15}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, key, xlabel, title in [
        (axes[0], "restart", "V-cycle 频率 r",
         f"V-cycle 频率扫描 (ct={ref_points['ct']:.0e}, cmi={ref_points['cmi']})"),
        (axes[1], "ct", "最粗层容差因子 ct",
         f"最粗层收敛条件扫描 (r={ref_points['restart']}, cmi={ref_points['cmi']})"),
        (axes[2], "cmi", "最粗层最大迭代 cmi",
         f"最粗层迭代上限扫描 (r={ref_points['restart']}, ct={ref_points['ct']:.0e})"),
    ]:
        style_axes(ax)
        pts = [r for r in results
               if (lattice is None or tuple(r["lattice"]) == tuple(lattice))
               and r["precision"] == precision and r["levels"] == levels
               and r.get("speedup_min") is not None]
        fixed = {k: v for k, v in ref_points.items() if k != key}
        seen = {}
        for r in pts:
            if any(r.get(k) != v for k, v in fixed.items()):
                continue
            seen[r[key]] = r["speedup_min"]
        xs = sorted(seen)
        ax.plot(xs, [seen[x] for x in xs], color=BENCH_PALETTE["blue"], marker="o",
                ms=5, lw=2)
        for x in xs:
            ax.annotate(f"{seen[x]:.2f}", (x, seen[x]), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7.5, color=BENCH_INK2)
        ax.set_xlabel(xlabel, color=BENCH_INK2)
        ax.set_ylabel("加速比", color=BENCH_INK2)
        ax.set_title(title, color=BENCH_INK, fontsize=10)
        if key == "ct":
            ax.set_xscale("log")
        ax.axhline(1.0, color=BENCH_MUTED, lw=1, ls="--")
        ax.grid(True, axis="y", color=BENCH_GRID, linewidth=0.6)
        ax.grid(False, axis="x")
    fig.tight_layout()
    return [save_figure(fig, os.path.join(out_dir, f"{prefix}_sweep.png"), dpi)]


def plot_bench_report(results, out_dir, prefix="bench", lattice=None,
                      label_fontsize=8.0, dpi=150):
    """按 dev73_5 顺序生成全套图（conv → hotspot → speedup → time → sweep），返回写入路径。"""
    saved = []
    saved += plot_convergence(results, out_dir, prefix, lattice=lattice,
                              label_fontsize=label_fontsize, dpi=dpi)
    saved += plot_hotspot(results, out_dir, prefix, label_fontsize=label_fontsize, dpi=dpi)
    saved += plot_speedup(results, out_dir, prefix, label_fontsize=label_fontsize, dpi=dpi)
    saved += plot_time(results, out_dir, prefix, label_fontsize=label_fontsize, dpi=dpi)
    saved += plot_sweep_curves(results, out_dir, prefix,
                               lattice=lattice or (8, 16, 16, 16), dpi=dpi)
    return saved
