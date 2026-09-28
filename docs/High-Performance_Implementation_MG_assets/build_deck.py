#!/usr/bin/env python3
"""Build the 17-slide PyQCU MultiGrid presentation.

The deck is rendered as 16:9 PNG slides, then packaged as a standards-compliant
PPTX with one full-slide image per page.  A PDF and speaker notes are generated
from the same immutable slide images so visual review and delivery stay aligned.
"""

from __future__ import annotations

import csv
import html
import math
import statistics
import textwrap
import zipfile
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
ASSET_DIR = Path(__file__).resolve().parent
SLIDE_DIR = ASSET_DIR / "slides"
PPTX_PATH = DOCS / "High-Performance Implementation of Multigrid Solver for Lattice QCD.pptx"
PDF_PATH = DOCS / "High-Performance Implementation of Multigrid Solver for Lattice QCD_ppt_render.pdf"
NOTES_PATH = DOCS / "High-Performance Implementation of Multigrid Solver for Lattice QCD_speaker_notes.md"
ALG_BICGSTAB = ASSET_DIR / "algorithm_build" / "algorithm_bicgstab.png"
ALG_VFGMRES = ASSET_DIR / "algorithm_build" / "algorithm_vcycle_fgmres.png"
REPORT_508_IMAGE25 = (
    DOCS / "张鑫 508应用测试报告-PyQCU" / "word" / "media" / "image25.png"
)
PERF_DIR = ROOT / "data" / "report_multigrid_comprehensive_20260928" / "final_protocol"
UNITS_CSV = PERF_DIR / "units.csv"
STAGES_CSV = PERF_DIR / "stages.csv"

W, H = 16.0, 9.0
DPI = 120
PX_W, PX_H = int(W * DPI), int(H * DPI)

FONT_PATH = Path("/usr/share/fonts/truetype/arphic/uming.ttc")
if FONT_PATH.exists():
    font_manager.fontManager.addfont(str(FONT_PATH))

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["AR PL UMing CN", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "mathtext.fontset": "dejavusans",
        "savefig.facecolor": "white",
    }
)

NAVY = "#12355B"
BLUE = "#245BA3"
TEAL = "#0B7285"
CYAN = "#53A6B8"
CORAL = "#C94C5A"
AMBER = "#D08A1F"
GREEN = "#3A7D5C"
INK = "#17212B"
MUTED = "#5E6B78"
LIGHT = "#EEF3F6"
PALE_BLUE = "#E7F0F7"
PALE_TEAL = "#E5F2F1"
PALE_AMBER = "#F8EFD9"
PALE_CORAL = "#F7E8EA"
WHITE = "#FFFFFF"
GRID = "#D7E0E6"


def wrap_text(text: str, width: int) -> str:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph:
            lines.append("")
            continue
        lines.extend(textwrap.wrap(paragraph, width=width, break_long_words=False))
    return "\n".join(lines)


def safe_median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def load_perf() -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    converted = {
        "ratio_quda_over_pyqcu",
        "pyqcu_seconds",
        "quda_seconds",
        "pyqcu_iterations",
        "quda_iterations",
        "pyqcu_true_residual",
        "quda_true_residual",
        "pyqcu_steady_sampler_peak_bytes",
        "quda_steady_sampler_peak_bytes",
    }
    with UNITS_CSV.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, object] = dict(raw)
            for key in converted:
                if row.get(key, "") not in ("", None):
                    row[key] = float(row[key])
            row["levels"] = int(row["levels"])
            rows.append(row)

    mg = [r for r in rows if str(r["solver"]).startswith("mg")]
    bicg = [r for r in rows if str(r["solver"]).startswith("bicg")]
    ratios = [float(r["ratio_quda_over_pyqcu"]) for r in rows]
    mg_ratios = [float(r["ratio_quda_over_pyqcu"]) for r in mg]
    mg_off = [r for r in mg if r["trace"] == "off"]
    bicg_ratios = [float(r["ratio_quda_over_pyqcu"]) for r in bicg]
    summary = {
        "n": len(rows),
        "overall_median": safe_median(ratios),
        "overall_range": (min(ratios), max(ratios)),
        "overall_wins": sum(v > 1 for v in ratios),
        "mg_n": len(mg),
        "mg_wins": sum(v > 1 for v in mg_ratios),
        "mg_median": safe_median(mg_ratios),
        "mg_off_n": len(mg_off),
        "mg_off_wins": sum(float(r["ratio_quda_over_pyqcu"]) > 1 for r in mg_off),
        "bi_n": len(bicg),
        "bi_wins": sum(v > 1 for v in bicg_ratios),
        "bi_median": safe_median(bicg_ratios),
        "py_res_max": max(float(r["pyqcu_true_residual"]) for r in rows),
        "qu_res_max": max(float(r["quda_true_residual"]) for r in rows),
        "py_mem_max_mib": max(float(r["pyqcu_steady_sampler_peak_bytes"]) for r in rows) / 2**20,
        "qu_mem_max_mib": max(float(r["quda_steady_sampler_peak_bytes"]) for r in rows) / 2**20,
    }
    return rows, summary


def stage_fractions() -> dict[str, dict[str, float]]:
    fields = [
        "pre_smoother_seconds",
        "post_smoother_seconds",
        "restrict_seconds",
        "prolongate_seconds",
        "coarse_solver_seconds",
        "other_seconds",
    ]
    grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    with STAGES_CSV.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["trace"] != "on":
                continue
            key = (row["unit_id"], row["side"])
            for field in fields:
                grouped[key][field] += float(row[field])

    result: dict[str, dict[str, float]] = {}
    for side in ("pyqcu", "quda"):
        shares = []
        for (unit_id, row_side), values in grouped.items():
            if row_side != side:
                continue
            total = sum(values.values())
            if total > 0:
                shares.append({field: values[field] / total for field in fields})
        result[side] = {
            field: safe_median([share[field] for share in shares]) for field in fields
        }
    return result


ROWS, SUMMARY = load_perf()
STAGE_SHARES = stage_fractions()


class Slide:
    def __init__(self, number: int, title: str, subtitle: str = "", kicker: str = "") -> None:
        self.number = number
        self.fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=WHITE)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis("off")
        self.ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=WHITE, edgecolor="none", zorder=-20))
        self.ax.add_patch(Rectangle((0.045, 0.955), 0.91, 0.004, color=BLUE, zorder=10))
        if kicker:
            self.text(0.055, 0.934, kicker.upper(), size=11.5, color=TEAL, weight="bold")
        self.text(0.055, 0.884 if subtitle else 0.902, title, size=27, color=NAVY, weight="bold")
        if subtitle:
            self.text(0.055, 0.837, subtitle, size=14, color=MUTED)

    def text(
        self,
        x: float,
        y: float,
        text: str,
        size: float = 16,
        color: str = INK,
        weight: str = "normal",
        ha: str = "left",
        va: str = "top",
        zorder: int = 20,
        linespacing: float = 1.32,
        style: str = "normal",
    ) -> None:
        self.ax.text(
            x,
            y,
            text,
            fontsize=size,
            color=color,
            fontweight=weight,
            fontstyle=style,
            ha=ha,
            va=va,
            zorder=zorder,
            linespacing=linespacing,
        )

    def box(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        face: str = WHITE,
        edge: str = GRID,
        radius: float = 0.015,
        linewidth: float = 1.1,
        zorder: int = 1,
    ) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.008,rounding_size={radius}",
            facecolor=face,
            edgecolor=edge,
            linewidth=linewidth,
            zorder=zorder,
        )
        self.ax.add_patch(patch)

    def card(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        heading: str,
        body: str,
        accent: str = BLUE,
        heading_size: float = 17,
        body_size: float = 13.5,
        wrap: int = 32,
    ) -> None:
        self.box(x, y, w, h, face=WHITE, edge=GRID)
        self.ax.add_patch(Rectangle((x + 0.008, y + 0.012), 0.006, h - 0.024, color=accent, zorder=3))
        self.text(x + 0.026, y + h - 0.018, heading, size=heading_size, color=accent, weight="bold")
        self.text(
            x + 0.026,
            y + h - 0.062,
            wrap_text(body, wrap),
            size=body_size,
            color=INK,
            linespacing=1.38,
        )

    def bullets(
        self,
        x: float,
        y: float,
        items: list[str],
        width: int = 40,
        size: float = 14.5,
        accent: str = BLUE,
        gap: float = 0.072,
    ) -> None:
        for i, item in enumerate(items):
            yy = y - i * gap
            self.ax.add_patch(
                FancyBboxPatch(
                    (x, yy - 0.012),
                    0.012,
                    0.012,
                    boxstyle="round,pad=0.002,rounding_size=0.004",
                    facecolor=accent,
                    edgecolor="none",
                    zorder=5,
                )
            )
            self.text(x + 0.024, yy, wrap_text(item, width), size=size, color=INK, linespacing=1.32)

    def formula(
        self,
        x: float,
        y: float,
        formula: str,
        size: float = 22,
        color: str = NAVY,
        ha: str = "left",
    ) -> None:
        self.text(x, y, formula, size=size, color=color, ha=ha, va="center")

    def arrow(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        color: str = BLUE,
        width: float = 1.8,
        connectionstyle: str = "arc3",
    ) -> None:
        patch = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=width,
            color=color,
            connectionstyle=connectionstyle,
            zorder=8,
        )
        self.ax.add_patch(patch)

    def image(self, path: Path, x: float, y: float, w: float, h: float) -> None:
        image = plt.imread(path)
        self.ax.imshow(
            image,
            extent=[x, x + w, y, y + h],
            aspect="auto",
            interpolation="lanczos",
            zorder=3,
        )

    def footer(self, source: str, page: int | None = None) -> None:
        page = self.number if page is None else page
        self.ax.plot([0.045, 0.955], [0.052, 0.052], color=GRID, linewidth=0.8, zorder=1)
        self.text(0.055, 0.037, wrap_text(source, 148), size=8.6, color=MUTED, va="top")
        self.text(0.95, 0.037, f"{page:02d}", size=10, color=NAVY, weight="bold", ha="right")

    def save(self) -> Path:
        SLIDE_DIR.mkdir(parents=True, exist_ok=True)
        target = SLIDE_DIR / f"slide_{self.number:02d}.png"
        self.fig.savefig(target, dpi=DPI, facecolor=WHITE, bbox_inches=None)
        plt.close(self.fig)
        return target


def add_kpi(
    slide: Slide,
    x: float,
    y: float,
    w: float,
    h: float,
    value: str,
    label: str,
    accent: str = TEAL,
    value_size: float = 26,
) -> None:
    slide.box(x, y, w, h, face=LIGHT, edge=WHITE)
    slide.text(x + 0.018, y + h - 0.042, value, size=value_size, color=accent, weight="bold")
    slide.text(x + 0.018, y + 0.026, label, size=11.2, color=MUTED)


def draw_508_strong_scaling(slide: Slide, x: float, y: float, w: float, h: float) -> None:
    axis = slide.ax.inset_axes([x, y, w, h])
    processes = [64, 128, 256, 512, 1024]
    wilson = [1.000, 1.451, 2.001, 4.739, 3.119]
    clover = [1.000, 1.657, 2.687, 4.670, 7.193]
    axis.plot(processes, wilson, marker="o", color=BLUE, linewidth=2.0, label="Wilson")
    axis.plot(processes, clover, marker="^", color=CORAL, linewidth=2.0, label="Clover")
    axis.fill_between(processes, wilson, clover, color=PALE_BLUE, alpha=0.45, zorder=0)
    axis.scatter([1024], [7.193], s=70, facecolors="white", edgecolors=CORAL, linewidths=1.5, zorder=4)
    axis.annotate(
        "7.19×",
        (1024, 7.193),
        xytext=(850, 7.6),
        fontsize=10,
        color=CORAL,
        arrowprops={"arrowstyle": "-", "color": CORAL, "lw": 0.8},
    )
    axis.set_xscale("log", base=2)
    axis.set_xticks(processes, [str(value) for value in processes])
    axis.set_ylim(0.5, 8.2)
    axis.set_xlabel("processes / threads", fontsize=9)
    axis.set_ylabel("reported speedup", fontsize=9)
    axis.grid(color=GRID, linewidth=0.6, alpha=0.7)
    axis.tick_params(labelsize=9)
    axis.legend(fontsize=9, ncol=2, loc="upper left", frameon=False)


def draw_508_weak_scaling(slide: Slide, x: float, y: float, w: float, h: float) -> None:
    axis = slide.ax.inset_axes([x, y, w, h])
    volume = [524288, 1048576, 2097152, 4194304, 8388608]
    delta = [14.166, 26.247, 58.044, 99.751, 112.188]
    axis.plot(volume, delta, marker="s", color=TEAL, linewidth=2.1)
    axis.fill_between(volume, 0, delta, color=PALE_TEAL, alpha=0.75, zorder=0)
    axis.scatter(volume, delta, s=34, facecolors="white", edgecolors=TEAL, linewidths=1.3, zorder=4)
    axis.set_xscale("log", base=2)
    axis.set_xticks(volume, ["0.5M", "1M", "2M", "4M", "8M"])
    axis.set_ylim(0, 125)
    axis.set_xlabel("equivalent single-process volume", fontsize=9)
    axis.set_ylabel("Clover-Wilson iteration (ms)", fontsize=9)
    axis.grid(color=GRID, linewidth=0.6, alpha=0.7)
    axis.tick_params(labelsize=9)
    axis.annotate(
        "112.2 ms",
        (8388608, 112.188),
        xytext=(1.1e6, 106),
        fontsize=10,
        color=TEAL,
        arrowprops={"arrowstyle": "-", "color": TEAL, "lw": 0.8},
    )


def draw_flow(
    slide: Slide,
    nodes: list[tuple[float, float, float, float, str]],
    arrows: list[tuple[int, int]],
    face: str = PALE_BLUE,
    edge: str = BLUE,
    size: float = 12.2,
) -> None:
    centers: list[tuple[float, float]] = []
    for x, y, w, h, label in nodes:
        slide.box(x, y, w, h, face=face, edge=edge)
        slide.text(x + w / 2, y + h / 2, label, size=size, color=NAVY, weight="bold", ha="center", va="center")
        centers.append((x + w / 2, y + h / 2))
    for start, end in arrows:
        x1, y1 = centers[start]
        x2, y2 = centers[end]
        if abs(y1 - y2) < 0.01:
            sx = x1 + nodes[start][2] / 2
            ex = x2 - nodes[end][2] / 2
            slide.arrow((sx, y1), (ex, y2))
        else:
            sy = y1 - nodes[start][3] / 2 if y1 > y2 else y1 + nodes[start][3] / 2
            ey = y2 + nodes[end][3] / 2 if y1 > y2 else y2 - nodes[end][3] / 2
            slide.arrow((x1, sy), (x2, ey))


def draw_box_grid(slide: Slide, values: list[list[str]], x: float, y: float, w: float, h: float) -> None:
    rows = len(values)
    cols = len(values[0])
    cell_w = w / cols
    cell_h = h / rows
    for row in range(rows):
        for col in range(cols):
            fill = PALE_BLUE if (row + col) % 2 == 0 else WHITE
            if row == rows - 1:
                fill = PALE_AMBER
            slide.box(
                x + col * cell_w,
                y + (rows - row - 1) * cell_h,
                cell_w,
                cell_h,
                face=fill,
                edge=WHITE,
                radius=0.006,
            )
            slide.text(
                x + (col + 0.5) * cell_w,
                y + (rows - row - 0.5) * cell_h,
                values[row][col],
                size=11.5,
                color=NAVY,
                weight="bold",
                ha="center",
                va="center",
            )


def slide_01() -> Path:
    s = Slide(1, "", "")
    s.ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=WHITE, edgecolor="none", zorder=-20))
    for i in range(13):
        x = 0.68 + (i % 5) * 0.062
        y = 0.18 + (i // 5) * 0.064
        s.ax.add_patch(Rectangle((x, y), 0.05, 0.05, facecolor=PALE_BLUE, edgecolor=CYAN, linewidth=0.8))
    s.ax.add_patch(Rectangle((0, 0), 0.013, 1, color=BLUE, zorder=10))
    s.text(0.065, 0.87, "PYQCU · LATTICE QCD", size=15, color=TEAL, weight="bold")
    s.text(
        0.065,
        0.765,
        "High-Performance Implementation\nof Multigrid Solver for Lattice QCD",
        size=31,
        color=NAVY,
        weight="bold",
        linespacing=1.08,
    )
    s.text(0.068, 0.615, "面向传播子求解的 Clover–Wilson 多重网格实现", size=18, color=INK)
    s.text(0.068, 0.555, "CUDA-C++ 后端 · Strict QUDA-style MG · 66 个精确对照单元", size=13.5, color=MUTED)
    s.box(0.065, 0.16, 0.46, 0.30, face=LIGHT, edge=WHITE)
    s.text(0.087, 0.414, "张鑫", size=18, color=NAVY, weight="bold")
    s.text(0.087, 0.37, "中国科学院近代物理研究所", size=14, color=INK)
    s.text(0.087, 0.322, "2026-10-11  15:05–15:20", size=13, color=MUTED)
    s.text(0.087, 0.277, "汇报人：张鑫", size=13, color=MUTED)
    s.text(0.087, 0.222, "数据基线：git tag test27", size=12, color=CORAL, weight="bold")
    s.text(
        0.59,
        0.43,
        "项目与证据",
        size=14,
        color=TEAL,
        weight="bold",
    )
    s.bullets(
        0.59,
        0.375,
        [
            "GitHub: github.com/zhangxin8069/PyQCU",
            "Gitee: gitee.com/zhangxin8069/PyQCU",
            "Docs: .../PyQCU/tree/main/docs",
            "Tags: .../PyQCU/tags",
        ],
        width=48,
        size=11.8,
        gap=0.054,
    )
    s.text(0.065, 0.09, "独立实现 · 更扁平结构 · 自主可控 · 面向真实 GPU 性能", size=12, color=MUTED)
    return s.save()


def slide_02() -> Path:
    s = Slide(
        2,
        "Wilson 与 Clover：先固定可持续求解的细网格算子",
        "同一源码约定下定义质量项、hopping 核与 onsite Clover 块。",
        "Propagator · action",
    )
    s.box(0.055, 0.17, 0.43, 0.60, face=PALE_BLUE, edge=WHITE)
    s.text(0.078, 0.735, "Wilson 作用量和 Dslash", size=18, color=BLUE, weight="bold")
    s.formula(0.085, 0.665, r"$(H\psi)_x=\sum_\mu[(1-\gamma_\mu)U_{x,\mu}\psi_{x+\hat\mu}$", size=19)
    s.formula(0.105, 0.612, r"$+(1+\gamma_\mu)U^\dagger_{x-\hat\mu,\mu}\psi_{x-\hat\mu}]$", size=19)
    s.formula(0.085, 0.535, r"$S_W=\sum_x\bar\psi_x[(m_0+4)-\frac{1}{2}H]\psi_x$", size=20)
    s.formula(0.085, 0.462, r"$D_W=(m_0+4)(I-\kappa H)$", size=21)
    s.formula(0.085, 0.407, r"$\kappa=\frac{1}{2m_0+8},\qquad m_0+4=\frac{1}{2\kappa}$", size=19)
    s.text(0.085, 0.315, "PyQCU 求解器使用归一化形式：", size=12.5, color=MUTED)
    s.formula(0.085, 0.267, r"$D_{\rm pc}=I-(\kappa/u_0)H,\qquad u_0=1$", size=20, color=NAVY)
    s.text(0.085, 0.207, "C++ `applyWilsonDslashQcu` 返回裸核 H；无单位项、无 -κ。", size=11.5, color=CORAL)

    s.box(0.515, 0.17, 0.43, 0.60, face=PALE_TEAL, edge=WHITE)
    s.text(0.538, 0.735, "Clover 改进与 onsite 块", size=18, color=TEAL, weight="bold")
    s.formula(0.545, 0.665, r"$P_{\mu\nu}=U_\mu U_\nu U^\dagger_\mu U^\dagger_\nu$", size=19)
    s.formula(0.545, 0.612, r"$C_{\mu\nu}=Q_{\mu\nu}-Q^\dagger_{\mu\nu}$", size=19)
    s.formula(0.545, 0.548, r"$T_p=-\frac{\kappa c_{\rm sw}}{8u_0}\sum_{\mu<\nu}\gamma_\mu\gamma_\nu C_{\mu\nu}$", size=17.5)
    s.formula(0.545, 0.485, r"$A_p=I_p+T_p$", size=21)
    s.formula(0.545, 0.423, r"$D^C_{\rm pc}=I-(\kappa/u_0)H+T$", size=19.5)
    s.formula(0.545, 0.362, r"$D^C_{\rm phys}=(m_0+4)I-\frac{1}{2}H$", size=18)
    s.formula(0.565, 0.312, r"$-\frac{c_{\rm sw}}{16u_0}\sum_{\mu<\nu}\gamma_\mu\gamma_\nu C_{\mu\nu}$", size=17)
    s.text(0.545, 0.235, "当前 C++ 生产路径：u0=1，有效 csw=1。", size=12.3, color=TEAL, weight="bold")
    s.text(0.545, 0.195, "csw→0 时 Clover 路径精确退化为 Wilson。", size=11.8, color=MUTED)
    s.footer(
        "来源：pyqcu/dslash/_wilson.py:16；pyqcu/dslash/_clover.py:118；cpp/cuda/qcu/include/lattice_set.h:522；"
        "QUDA 差分对照 report.md:21-23。"
    )
    return s.save()


def slide_03() -> Path:
    s = Slide(
        3,
        "原始 Dslash：八个邻居上的矩阵自由稀疏作用",
        "每个格点读取四个正向与四个反向旋量，并在飞行中完成投影和规范平移。",
        "Propagator · raw dslash",
    )
    s.box(0.055, 0.45, 0.45, 0.34, face=PALE_BLUE, edge=WHITE)
    s.text(0.078, 0.745, "裸跃迁核 H", size=18, color=BLUE, weight="bold")
    s.formula(
        0.08,
        0.675,
        r"$(H\psi)_x=\sum_{\mu=0}^{3}[(1-\gamma_\mu)U_{x,\mu}\psi_{x+\hat\mu}$",
        size=17,
    )
    s.formula(
        0.10,
        0.617,
        r"$+(1+\gamma_\mu)U^\dagger_{x-\hat\mu,\mu}\psi_{x-\hat\mu}]$",
        size=17,
    )
    s.text(0.08, 0.545, "D_W = (m0+4)I - (1/2)H", size=20, color=NAVY, weight="bold")
    s.text(0.08, 0.495, "H 本身不是完整算子；入口语义必须显式区分。", size=12, color=CORAL)

    s.box(0.055, 0.17, 0.45, 0.235, face=WHITE, edge=GRID)
    s.text(0.078, 0.36, "入口返回对象", size=15.5, color=NAVY, weight="bold")
    table = [
        ("applyWilsonDslashQcu", "H"),
        ("give_wilson_eo/oe", "-(κ/u0) Hpq"),
        ("give_wilson(with_I=True)", "I-(κ/u0)H"),
        ("applyCloverDslashQcu", "A_p^-1 Hpq"),
    ]
    for i, (left, right) in enumerate(table):
        y = 0.318 - i * 0.041
        s.text(0.083, y, left, size=11.2, color=INK)
        s.text(0.405, y, right, size=11.8, color=TEAL, weight="bold", ha="right")
    nodes = [
        (0.56, 0.58, 0.11, 0.12, "邻居旋量\n读取"),
        (0.70, 0.58, 0.11, 0.12, "1±γ\n投影"),
        (0.84, 0.58, 0.10, 0.12, "SU(3)\n平移"),
        (0.70, 0.38, 0.11, 0.12, "八方向\n累加"),
        (0.84, 0.38, 0.10, 0.12, "H·ψ\n输出"),
        (0.56, 0.38, 0.11, 0.12, "Parity\n布局"),
    ]
    draw_flow(s, nodes, [(0, 1), (1, 2), (2, 4), (0, 3), (3, 4), (5, 3)], face=PALE_TEAL, edge=TEAL)
    s.card(
        0.545,
        0.17,
        0.41,
        0.16,
        "为何是性能核心",
        "最近邻模板天然 memory/bandwidth-bound；4→2 spin projection、SU(3) 重用和 xyzt 尾轴布局共同决定有效带宽。",
        accent=AMBER,
        body_size=12.4,
        wrap=39,
    )
    s.footer(
        "来源：cpp/cuda/qcu/src/wilson_dslash.cu:55-218；apply_wilson_dslash.cu:20；"
        "pyqcu/testing/qcu/single_function_common.py:325；508 报告优化 1-4。"
    )
    return s.save()


def slide_04() -> Path:
    s = Slide(
        4,
        "奇偶消元：外层 Krylov 维度减半，解仍可精确重建",
        "PyQCU 将 Level-0 求解放在奇数子格 Schur 算子 S_o 上。",
        "Propagator · even-odd",
    )
    s.box(0.055, 0.48, 0.40, 0.31, face=PALE_BLUE, edge=WHITE)
    s.text(0.078, 0.755, "块矩阵与 Schur 补", size=17.5, color=BLUE, weight="bold")
    s.formula(
        0.08,
        0.686,
        r"$D_{\rm pc}=[A_e,\;-\kappa H_{eo};\;-\kappa H_{oe},\;A_o]$",
        size=18,
    )
    s.formula(0.08, 0.588, r"$S_o=A_o-\kappa^2H_{oe}A_e^{-1}H_{eo}$", size=20)
    s.text(0.08, 0.515, "Wilson: A_e=A_o=I", size=13, color=TEAL, weight="bold")

    s.box(0.055, 0.17, 0.40, 0.27, face=WHITE, edge=GRID)
    s.text(0.078, 0.405, "求解与重建", size=17, color=NAVY, weight="bold")
    s.formula(0.08, 0.342, r"$b_o^{\rm pc}=b_o+\kappa H_{oe}A_e^{-1}b_e$", size=17.5)
    s.formula(0.08, 0.280, r"$S_ox_o=b_o^{\rm pc}$", size=19)
    s.formula(0.08, 0.219, r"$x_e=A_e^{-1}(b_e+\kappa H_{eo}x_o)$", size=17.5)

    nodes = [
        (0.53, 0.64, 0.12, 0.13, "Full field\n(b_e,b_o)"),
        (0.70, 0.64, 0.12, 0.13, "Prepare\nb_o^pc"),
        (0.87, 0.64, 0.08, 0.13, "Solve\nS_o x_o"),
        (0.70, 0.42, 0.12, 0.13, "Reconstruct\nx_e"),
        (0.53, 0.42, 0.12, 0.13, "Full\n(x_e,x_o)"),
        (0.87, 0.42, 0.08, 0.13, "True\nresidual"),
    ]
    draw_flow(s, nodes, [(0, 1), (1, 2), (2, 3), (3, 4), (3, 5), (5, 4)], face=PALE_TEAL, edge=TEAL)
    s.card(
        0.515,
        0.17,
        0.44,
        0.19,
        "数值不变量",
        "Schur 求解只改变未知量组织，不改变原方程；D_ee 局部块批量求逆，最终使用 full Wilson/Clover 真残差复位。",
        accent=AMBER,
        body_size=12.6,
        wrap=43,
    )
    s.footer(
        "来源：pyqcu/dslash/_operator.py:406；cpp/cuda/qcu/include/lattice_wilson_bistabcg.h:98；"
        "lattice_clover_bistabcg.h:116；cpp/cuda/qcu/src/bistabcg.cu:68-90。"
    )
    return s.save()


def slide_05() -> Path:
    s = Slide(
        5,
        "MultiGrid 的实质：粗空间消除低模，平滑器压低高模",
        "自适应聚集不是“另做一次求解”，而是构造一个小得多的 Galerkin 问题。",
        "MultiGrid · algorithm",
    )
    s.box(0.055, 0.46, 0.43, 0.34, face=PALE_BLUE, edge=WHITE)
    s.text(0.078, 0.75, "四件数值资产", size=18, color=BLUE, weight="bold")
    s.bullets(
        0.08,
        0.69,
        [
            "近零模 B_l：A_l B_l ≈ 0，逼近低频子空间。",
            "局部聚集与 QR：V_l^H V_l=I，块内正交。",
            "转移对：P_l 延拓，R_l=P_l^H 限制。",
        ],
        width=39,
        size=13.2,
        gap=0.062,
    )
    s.formula(0.08, 0.515, r"$A_{l+1}=R_lA_lP_l,\qquad R_l=P_l^\dagger$", size=19)

    nodes = [
        (0.54, 0.70, 0.11, 0.11, "Pre-smooth\nν_1"),
        (0.69, 0.70, 0.11, 0.11, "Restrict\nR_l"),
        (0.84, 0.70, 0.11, 0.11, "Coarse\nsolve"),
        (0.69, 0.49, 0.11, 0.11, "Prolong\nP_l"),
        (0.54, 0.49, 0.11, 0.11, "Post-smooth\nν_2"),
        (0.84, 0.49, 0.11, 0.11, "Updated\nx"),
    ]
    draw_flow(s, nodes, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], face=PALE_TEAL, edge=TEAL)
    s.arrow((0.895, 0.70), (0.895, 0.60), color=AMBER, connectionstyle="arc3")
    s.text(0.90, 0.65, "递归", size=11.5, color=AMBER, weight="bold")
    s.text(0.54, 0.42, "V-cycle 作为预条件器；外层 Krylov 负责全局收敛。", size=13.5, color=NAVY, weight="bold")

    s.card(
        0.055,
        0.17,
        0.43,
        0.22,
        "局部正交为何关键",
        "aggregate 支撑互不重叠，因此 R 不需要全局 Gram 求逆；P 的局部块投影能保留规范背景和 spin-color 耦合。",
        accent=TEAL,
        body_size=12.5,
        wrap=43,
    )
    s.card(
        0.515,
        0.17,
        0.435,
        0.22,
        "收敛图像",
        "平滑器处理高频误差，粗层代表低模误差；两者互补，目标是让外层迭代次数弱依赖格点体积。",
        accent=AMBER,
        body_size=12.5,
        wrap=43,
    )
    s.footer(
        "来源：docs/report_pyqcu_mg_operator_construction_20260830.tex:80-148,194-266；"
        "cpp/cuda/qcu/include/lattice_multigrid.h:11-25。"
    )
    return s.save()


def slide_06() -> Path:
    s = Slide(
        6,
        "预条件 Bi-CGStab：粗层小系统与热启动的核心迭代",
        "以 M 为预条件器；下表给出可直接核对的完整伪代码。",
        "MultiGrid · algorithm 1",
    )
    s.box(0.055, 0.705, 0.29, 0.09, face=PALE_BLUE, edge=WHITE)
    s.formula(0.075, 0.75, r"$M^{-1}Ax=M^{-1}b$", size=18, color=BLUE)
    s.box(0.355, 0.705, 0.29, 0.09, face=PALE_TEAL, edge=WHITE)
    s.formula(0.375, 0.75, r"$M\hat p=p^{(i)},\quad M\hat s=s$", size=16.5, color=TEAL)
    s.box(0.655, 0.705, 0.29, 0.09, face=PALE_AMBER, edge=WHITE)
    s.formula(0.675, 0.75, r"$\omega_i=\frac{t^\dagger s}{t^\dagger t}$", size=18, color=AMBER)
    s.image(ALG_BICGSTAB, 0.14, 0.12, 0.72, 0.57)
    s.footer(
        "算法语义：report_multigrid_quda_pyqcu_20260928.tex；实现："
        "cpp/cuda/qcu/include/lattice_{wilson,clover}_bistabcg.h；当前粗层 BiCGStab 与 MR 回退路径。"
    )
    return s.save()


def slide_07() -> Path:
    s = Slide(
        7,
        "PyQCU Strict 特化：D = X + H，并把 QCD 约束嵌入每一层",
        "细层是 target-parity compact 问题；粗层始终保留 full geometry 和 X/Y/Yhat 资产。",
        "PyQCU · algorithm",
    )
    s.box(0.055, 0.51, 0.41, 0.29, face=PALE_BLUE, edge=WHITE)
    s.text(0.078, 0.755, "层级分解与 Galerkin 投影", size=17.5, color=BLUE, weight="bold")
    s.formula(0.08, 0.682, r"$D_l=X_l+H_l$", size=21)
    s.formula(0.08, 0.615, r"$\widehat D_l=X_l^{-1}D_l$", size=21)
    s.formula(0.08, 0.546, r"$D_{l+1}=R_l\widehat D_lP_l$", size=21)
    s.text(0.08, 0.522, "先做 onsite 左预条件，再投影；顺序不可交换。", size=11.5, color=CORAL)

    s.box(0.055, 0.17, 0.41, 0.30, face=PALE_TEAL, edge=WHITE)
    s.text(0.078, 0.425, "运行资产", size=17.5, color=TEAL, weight="bold")
    s.text(0.08, 0.377, "X, X^-1 : onsite / zero-displacement block", size=13.2, color=INK)
    s.text(0.08, 0.331, "Y_f, Y_b : forward / backward coarse links", size=13.2, color=INK)
    s.formula(0.08, 0.275, r"$\widehat Y_f=X^{-1}Y_f,\quad \widehat Y_b=Y_bX^{-\dagger}$", size=17)
    s.text(0.08, 0.216, "raw Y 默认只用于 setup/诊断；常规 solve 驻留 Yhat 与 X/X^-1。", size=11.5, color=MUTED)

    nodes = [
        (0.53, 0.66, 0.12, 0.12, "Compact fine\ntarget parity"),
        (0.69, 0.66, 0.12, 0.12, "P: blocked\nfull coarse"),
        (0.85, 0.66, 0.10, 0.12, "Full coarse\nX/Y/Yhat"),
        (0.69, 0.45, 0.12, 0.12, "R=P^H:\nback to fine"),
        (0.53, 0.45, 0.12, 0.12, "MATPC\nresidual"),
        (0.85, 0.45, 0.10, 0.12, "Recursive\nV-cycle"),
    ]
    draw_flow(s, nodes, [(0, 1), (1, 2), (2, 5), (5, 3), (3, 4)], face=PALE_AMBER, edge=AMBER)
    s.card(
        0.515,
        0.17,
        0.435,
        0.20,
        "QCD 的一致性边界",
        "coarse_spin=2·nvec；粗层不检查板化。P/R 只在 full-coarse 与 compact target-parity fine 之间切换，MATPC 保持原算子语义。",
        accent=CORAL,
        heading_size=15.5,
        body_size=12.3,
        wrap=43,
    )
    s.footer(
        "来源：cpp/cuda/qcu/src/apply_multigrid_strict.cu:1499-1615,2926-3430；"
        "pyqcu/tools/_strict_galerkin.py:428-719；skills/qcu/SKILL.md。"
    )
    return s.save()


def slide_08() -> Path:
    s = Slide(
        8,
        "V-cycle 在里、FGMRES 在外：MG 预条件的算法闭环",
        "左表给出递归预条件器，右表给出 flexible right-preconditioned GMRES。",
        "PyQCU · algorithm 2",
    )
    s.box(0.055, 0.70, 0.43, 0.095, face=PALE_BLUE, edge=WHITE)
    s.formula(
        0.075,
        0.745,
        r"$z_l=S_{\rm post}^{\nu_2}(S_{\rm pre}^{\nu_1}(r_l)+P_l M_{l+1}^{-1}R_l r_{l+1})$",
        size=14.5,
        color=BLUE,
    )
    s.box(0.515, 0.70, 0.43, 0.095, face=PALE_TEAL, edge=WHITE)
    s.formula(
        0.535,
        0.745,
        r"$z_j=M^{-1}v_j,\qquad x\leftarrow x+\sum_jz_jy_j$",
        size=15.5,
        color=TEAL,
    )
    s.image(ALG_VFGMRES, 0.055, 0.285, 0.89, 0.40)
    s.card(
        0.055,
        0.13,
        0.43,
        0.12,
        "迭代口径",
        "最细层迭代只记外层 FGMRES；smoother 与 coarse solver 不相加。",
        accent=AMBER,
        heading_size=13.2,
        body_size=10.8,
        wrap=43,
    )
    s.card(
        0.515,
        0.13,
        0.43,
        0.12,
        "工作区",
        "(2m+5)B_f+2B_c；相同几何与 restart 下跨 solve 复用。",
        accent=CORAL,
        heading_size=13.2,
        body_size=10.8,
        wrap=43,
    )
    s.footer(
        "来源：cpp/cuda/qcu/src/apply_multigrid_strict.cu:2926-3430,4579-4683；"
        "report_multigrid_quda_pyqcu_20260928.tex；skills/qcu/SKILL.md 的工作区契约。"
    )
    return s.save()


def slide_09() -> Path:
    s = Slide(
        9,
        "CUDA-C++ 优化从寄存器、流、显存延伸到 MPI",
        "细层解决带宽，粗层解决启动/同步，多 rank 解决 halo；任何单点优化都不足以实用。",
        "PyQCU · CUDA-C++",
    )
    cards = [
        (
            0.055,
            "细层 kernel",
            BLUE,
            "1±γ 四→二 spin projection；SU(3) link 重用；xyzt 尾轴布局促发合并访存。",
        ),
        (
            0.285,
            "粗层延迟",
            TEAL,
            "device_vals 标量；check-stride；融合 β/ρ 与 r/x 更新；CUDA Graph 8 迭代段。",
        ),
        (
            0.515,
            "setup / 显存",
            AMBER,
            "colored/batched Galerkin、workspace 预算、持久 hierarchy、schema-v2 SHA256 cache。",
        ),
        (
            0.745,
            "通信",
            CORAL,
            "rank-local 32 方向粗 halo；face-only 非阻塞 vector halo；local/remote 分离并融合远端修正。",
        ),
    ]
    for x, title, color, body in cards:
        s.card(x, 0.43, 0.20, 0.28, title, body, accent=color, heading_size=15.2, body_size=11.7, wrap=21)
    s.text(0.055, 0.365, "执行路径", size=15, color=NAVY, weight="bold")
    nodes = [
        (0.06, 0.22, 0.12, 0.10, "Python\norchestration"),
        (0.21, 0.22, 0.12, 0.10, "Cython\nnogil bridge"),
        (0.36, 0.22, 0.12, 0.10, "C ABI\nparams/set_ptrs"),
        (0.51, 0.22, 0.12, 0.10, "C++/CUDA\nkernels"),
        (0.66, 0.22, 0.12, 0.10, "MPI halo\n/reduction"),
        (0.81, 0.22, 0.12, 0.10, "Full true\nresidual"),
    ]
    draw_flow(s, nodes, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], face=LIGHT, edge=NAVY, size=10.6)
    s.box(0.055, 0.10, 0.875, 0.075, face=PALE_CORAL, edge=WHITE)
    s.text(
        0.08,
        0.137,
        "边界：c64 分布式 vector halo 默认 overlap；c128 因浮点累加顺序敏感默认关闭，"
        "仅在显式开关并重验真残差后启用。",
        size=11.8,
        color=CORAL,
        weight="bold",
    )
    s.footer(
        "来源：cpp/cuda/qcu/include/lattice_clover_multigrid.h:24-30,755-789,3584-3806；"
        "src/apply_multigrid_strict.cu:1691-2673；pyqcu/cuda/_strict_cache.py:32-1074。"
    )
    return s.save()


def slide_10() -> Path:
    s = Slide(
        10,
        "对比口径先行：66 个精确单元覆盖设备、精度、层级与 trace",
        "所有正式性能结论来自 git tag test27；trace-on 只用于诊断与阶段归因。",
        "Performance · protocol",
    )
    matrix = [
        ["设备", "V100 单卡 sm70", "双 P100 sm60 / 2×1×1×1"],
        ["精度", "c64", "c128"],
        ["求解", "BiCGStab ref", "MG-2 / MG-3"],
        ["trace", "off（正式）", "on（诊断）"],
        ["phase", "1 cold", "2 warmup + 5 steady"],
    ]
    draw_box_grid(s, matrix, 0.055, 0.49, 0.47, 0.30)
    s.text(0.055, 0.445, "请求矩阵与严格完成口径", size=15, color=NAVY, weight="bold")
    add_kpi(s, 0.055, 0.255, 0.105, 0.15, "72", "请求 combined", BLUE, 24)
    add_kpi(s, 0.17, 0.255, 0.105, 0.15, "66", "精确 combined", TEAL, 24)
    add_kpi(s, 0.285, 0.255, 0.105, 0.15, "132", "精确 side", GREEN, 24)
    add_kpi(s, 0.40, 0.255, 0.125, 0.15, "6", "容量替代项", CORAL, 24)

    s.box(0.56, 0.44, 0.385, 0.35, face=PALE_BLUE, edge=WHITE)
    s.text(0.585, 0.75, "统一公平性与验收门", size=17, color=BLUE, weight="bold")
    s.bullets(
        0.585,
        0.695,
        [
            "config hash 66/66；input bundle 66/66。",
            "每侧 1 cold + 2 warmup + 5 steady。",
            "独立 full-operator 真残差 < 5×10^-6。",
            "最大真残差：PyQCU 7.40×10^-7，QUDA 7.97×10^-7。",
            "速度比定义 R=t_QUDA/t_PyQCU；R>1 表示 PyQCU 更快。",
        ],
        width=38,
        size=12.1,
        gap=0.059,
    )
    s.box(0.56, 0.20, 0.385, 0.20, face=PALE_AMBER, edge=WHITE)
    s.text(0.585, 0.355, "替换不是精确结果", size=15, color=AMBER, weight="bold")
    s.text(
        0.585,
        0.307,
        wrap_text(
            "双 P100 c64 16×32×32×48 的 6 个请求项被显式替代为 16^4；"
            "替代项只作容量趋势，不进入原大格加速比。",
            39,
        ),
        size=11.3,
        color=INK,
        linespacing=1.26,
    )
    s.footer(
        "数据来源：git tag test27；docs/report_multigrid_quda_pyqcu_20260928.tex:34-112；"
        "data/report_multigrid_comprehensive_20260928/final_protocol/units.csv。"
    )
    return s.save()


def slide_11() -> Path:
    s = Slide(
        11,
        "66 个精确单元并非均匀胜出：总中位 1.025，MG 为 35/44",
        "先看全分布和长尾，再决定哪些场景可以宣称优势。",
        "Performance · distribution",
    )
    add_kpi(s, 0.055, 0.69, 0.175, 0.14, f"{SUMMARY['overall_median']:.3f}", "全部精确单元中位 R", BLUE, 24)
    add_kpi(s, 0.245, 0.69, 0.175, 0.14, f"{SUMMARY['mg_wins']}/{SUMMARY['mg_n']}", "MG-2/3 有利", TEAL, 24)
    add_kpi(s, 0.435, 0.69, 0.175, 0.14, f"{SUMMARY['mg_off_wins']}/{SUMMARY['mg_off_n']}", "MG trace-off 有利", GREEN, 24)
    add_kpi(s, 0.625, 0.69, 0.175, 0.14, f"{SUMMARY['bi_wins']}/{SUMMARY['bi_n']}", "BiCGStab 有利", CORAL, 24)
    add_kpi(s, 0.815, 0.69, 0.13, 0.14, f"{SUMMARY['overall_range'][1]:.1f}×", "最大单点 R", AMBER, 23)

    inset = s.ax.inset_axes([0.065, 0.16, 0.87, 0.50])
    categories = [
        ("V100 / c64", [r for r in ROWS if r["device"] == "v100" and r["precision"] == "c64"]),
        ("V100 / c128", [r for r in ROWS if r["device"] == "v100" and r["precision"] == "c128"]),
        ("P100 / c64", [r for r in ROWS if r["device"] == "p100" and r["precision"] == "c64"]),
        ("P100 / c128", [r for r in ROWS if r["device"] == "p100" and r["precision"] == "c128"]),
    ]
    solver_colors = {"bicgstab-reference": CORAL, "mg-2": TEAL, "mg-3": BLUE}
    for x, (label, rows) in enumerate(categories):
        for row in rows:
            solver = str(row["solver"])
            ratio = float(row["ratio_quda_over_pyqcu"])
            trace = str(row["trace"])
            jitter = ((float(row["levels"]) - 2.5) * 0.035) + (0.025 if trace == "on" else -0.025)
            face = solver_colors[solver]
            if trace == "on":
                face = "white"
            inset.scatter(
                x + jitter,
                ratio,
                s=38 if trace == "off" else 32,
                facecolors=face,
                edgecolors=solver_colors[solver],
                linewidths=1.2,
                alpha=0.9,
                zorder=3,
            )
        off = [float(r["ratio_quda_over_pyqcu"]) for r in rows if r["trace"] == "off"]
        if off:
            inset.plot([x - 0.18, x + 0.18], [safe_median(off)] * 2, color=NAVY, linewidth=2.2, zorder=4)
    inset.axhline(1.0, color=INK, linewidth=1.1, linestyle="--")
    inset.set_yscale("log")
    inset.set_ylim(0.15, 18)
    inset.set_xticks(range(len(categories)), [x[0] for x in categories])
    inset.set_ylabel("QUDA / PyQCU")
    inset.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.6)
    inset.tick_params(labelsize=10)
    for spine in inset.spines.values():
        spine.set_color(GRID)
    inset.scatter([], [], facecolors=TEAL, edgecolors=TEAL, label="MG-2")
    inset.scatter([], [], facecolors=BLUE, edgecolors=BLUE, label="MG-3")
    inset.scatter([], [], facecolors=CORAL, edgecolors=CORAL, label="BiCGStab")
    inset.scatter([], [], facecolors="white", edgecolors=MUTED, label="空心=trace-on")
    inset.legend(loc="upper left", ncol=4, fontsize=9, frameon=False)
    inset.annotate("14.63×", (1.05, 14.634), xytext=(1.18, 12.0), fontsize=10, color=AMBER)
    inset.annotate("0.177×", (0.02, 0.1766), xytext=(-0.28, 0.22), fontsize=10, color=CORAL)
    s.footer(
        "数据：tag test27 units.csv / summary_analysis.json。实线为各设备-精度组的 trace-off 中位数；"
        "min/max 是单点极值，不是置信区间。"
    )
    return s.save()


def group_median(rows: list[dict[str, object]], device: str, precision: str, solver: str, trace: str) -> float:
    values = [
        float(r["ratio_quda_over_pyqcu"])
        for r in rows
        if r["device"] == device
        and r["precision"] == precision
        and r["solver"] == solver
        and r["trace"] == trace
    ]
    return safe_median(values)


def slide_12() -> Path:
    s = Slide(
        12,
        "设备与精度决定趋势：V100 明显领先，P100 在 trace-off 总体有利",
        "同一 MG 优势不能外推到 BiCGStab；trace-on 还包含诊断同步开销。",
        "Performance · grouped",
    )
    inset = s.ax.inset_axes([0.07, 0.17, 0.53, 0.62])
    row_labels = ["V100 / c64", "V100 / c128", "P100 / c64", "P100 / c128"]
    col_labels = ["MG-2", "MG-3", "BiCG ref"]
    matrix = []
    for device, precision in [("v100", "c64"), ("v100", "c128"), ("p100", "c64"), ("p100", "c128")]:
        matrix.append(
            [
                group_median(ROWS, device, precision, "mg-2", "off"),
                group_median(ROWS, device, precision, "mg-3", "off"),
                group_median(ROWS, device, precision, "bicgstab-reference", "off"),
            ]
        )
    norm = TwoSlopeNorm(vmin=0.2, vcenter=1.0, vmax=5.0)
    image = inset.imshow(matrix, cmap="RdYlGn", norm=norm, aspect="auto")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            color = "white" if value < 0.45 or value > 3.3 else INK
            inset.text(j, i, f"{value:.2f}×", ha="center", va="center", fontsize=13, color=color, fontweight="bold")
    inset.set_xticks(range(3), col_labels)
    inset.set_yticks(range(4), row_labels)
    inset.tick_params(labelsize=11)
    for spine in inset.spines.values():
        spine.set_color(GRID)
    cb = s.fig.colorbar(image, ax=inset, fraction=0.045, pad=0.03)
    cb.ax.tick_params(labelsize=8)
    cb.set_label("R", size=9)

    s.box(0.63, 0.54, 0.315, 0.25, face=PALE_BLUE, edge=WHITE)
    s.text(0.655, 0.755, "亮点", size=16, color=BLUE, weight="bold")
    s.bullets(
        0.655,
        0.705,
        [
            "V100 c64 MG-3: 4.74×",
            "V100 c128 MG-3: 3.22×",
            "P100 c64 MG-2/3: 1.78× / 1.79×",
            "P100 c128 MG-2: 1.25×",
        ],
        width=28,
        size=12.1,
        gap=0.052,
    )
    s.box(0.63, 0.18, 0.315, 0.32, face=PALE_CORAL, edge=WHITE)
    s.text(0.655, 0.465, "必须保留的边界", size=16, color=CORAL, weight="bold")
    s.bullets(
        0.655,
        0.415,
        [
            "MG 总体 35/44 有利，但范围 0.54×-14.63×。",
            "BiCGStab 0/22，QUDA 明确更快。",
            "trace-on 部分 P100 组退到 1× 以下。",
            "P100/c64 每组 n=2，解释需保守。",
        ],
        width=30,
        size=11.7,
        accent=CORAL,
        gap=0.057,
    )
    s.footer(
        "数据：tag test27 overlap_matrix.csv 与 units.csv。R 按组中位数计算；极值不对应置信区间。"
    )
    return s.save()


def slide_13() -> Path:
    s = Slide(
        13,
        "阶段与容量证据把优化目标指向粗解、同步和 setup",
        "trace-on 仅用于归因；容量替代必须与原格点性能严格分栏。",
        "Performance · bottleneck",
    )
    inset = s.ax.inset_axes([0.06, 0.42, 0.53, 0.37])
    fields = [
        "pre_smoother_seconds",
        "post_smoother_seconds",
        "restrict_seconds",
        "prolongate_seconds",
        "coarse_solver_seconds",
        "other_seconds",
    ]
    labels = ["pre", "post", "R", "P", "coarse", "other"]
    colors = [CYAN, BLUE, AMBER, "#E2B05D", CORAL, "#8D99AE"]
    sides = ["pyqcu", "quda"]
    left = [0.0, 0.0]
    for field, label, color in zip(fields, labels, colors):
        values = [STAGE_SHARES[side][field] * 100 for side in sides]
        inset.barh([1, 0], values, left=left, color=color, edgecolor="white", height=0.46, label=label)
        left = [left[i] + values[i] for i in range(2)]
    inset.set_yticks([1, 0], ["PyQCU", "QUDA"])
    inset.set_xlim(0, 100)
    inset.set_xlabel("median share of recorded stage time (%)", fontsize=9)
    inset.tick_params(labelsize=10)
    inset.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.6)
    inset.legend(loc="lower center", bbox_to_anchor=(0.5, 1.05), ncol=6, fontsize=8, frameon=False)
    for spine in inset.spines.values():
        spine.set_color(GRID)
    s.text(
        0.08,
        0.35,
        "PyQCU 记录中，coarse solve + other 约占 95%；QUDA 的具名 coarse 约占 69%。",
        size=12.2,
        color=INK,
        weight="bold",
    )
    s.text(0.08, 0.30, "层级同名不代表同一职责，跨库只比较总时间，不叠加同名阶段。", size=10.8, color=MUTED)

    s.box(0.64, 0.42, 0.305, 0.37, face=PALE_CORAL, edge=WHITE)
    s.text(0.665, 0.75, "6 个双 P100 大格请求被替代", size=15.5, color=CORAL, weight="bold")
    s.text(0.665, 0.695, "请求：16×32×32×48 c64", size=12.3, color=INK)
    attempts = [
        ("默认 overlap", "15.9 GB", "93.1%"),
        ("OVERLAP=0", "14.8 GB", "86.2%"),
        ("CPU staging", "12.4 GB", "72.4%"),
    ]
    for i, (mode, mem, ratio) in enumerate(attempts):
        y = 0.625 - i * 0.055
        s.text(0.668, y, mode, size=11.2, color=INK)
        s.text(0.82, y, mem, size=11.2, color=NAVY, weight="bold")
        s.text(0.90, y, ratio, size=11.2, color=CORAL, ha="right")
    s.text(0.665, 0.435, "替代为 16^4；CPU staging 仍超过 570 s。", size=11.2, color=MUTED)

    add_kpi(s, 0.06, 0.14, 0.16, 0.13, "7.40e-7", "PyQCU 最大真残差", TEAL, 20)
    add_kpi(s, 0.235, 0.14, 0.16, 0.13, "7.97e-7", "QUDA 最大真残差", BLUE, 20)
    add_kpi(s, 0.41, 0.14, 0.16, 0.13, "11,907 MiB", "PyQCU 最大观测显存", GREEN, 18)
    add_kpi(s, 0.585, 0.14, 0.16, 0.13, "23,063 MiB", "QUDA 最大观测显存", AMBER, 18)
    s.text(0.76, 0.215, "真残差门：5×10^-6", size=13, color=NAVY, weight="bold")
    s.text(0.76, 0.17, "显存为 device-wide sampler；全部保留样本低于 85%。", size=10.8, color=MUTED)
    s.footer(
        "数据：tag test27 stages.csv, mg_memory.csv, capacity_guard_events.json；"
        "stage 分解仅使用 trace-on，容量替代不得混入原大格速度比。"
    )
    return s.save()


def slide_14() -> Path:
    s = Slide(
        14,
        "当前结论：先兑现 MG 专项优势，再攻克粗层与通信",
        "优势已可验证，但范围、成熟度和容量边界必须和性能数字一起呈现。",
        "Performance · conclusion",
    )
    s.card(
        0.055,
        0.46,
        0.275,
        0.34,
        "优势",
        "扁平单仓库与自主可控；Clover-Wilson 聚焦明确；Strict full-coarse、P/R、X/Y/Yhat 和真残差门完整；"
        "trace-off MG 21/22 有利；V100 中位优势显著。",
        accent=TEAL,
        heading_size=18,
        body_size=11.5,
        wrap=22,
    )
    s.card(
        0.362,
        0.46,
        0.275,
        0.34,
        "不足",
        "总中位仅 1.025；BiCGStab 0/22；trace-on 同步开销高且方差大；P100 大格 c64 触发容量/时限替代；"
        "主要验证拓扑有限，功能覆盖仍偏窄。",
        accent=CORAL,
        heading_size=18,
        body_size=11.5,
        wrap=22,
    )
    s.card(
        0.67,
        0.46,
        0.275,
        0.34,
        "下一阶段",
        "通信：持久 request / device-aware；\n"
        "粗解：减少同步、融合小内核；\n"
        "setup：流式资产与显存预算；\n"
        "扩展：作用量、混合精度、国产卡。",
        accent=AMBER,
        heading_size=18,
        body_size=11.5,
        wrap=20,
    )
    s.box(0.055, 0.17, 0.89, 0.20, face=PALE_BLUE, edge=WHITE)
    s.text(0.08, 0.325, "可验证的项目主张", size=15.5, color=BLUE, weight="bold")
    s.formula(
        0.08,
        0.247,
        r"$R=t_{\rm QUDA}/t_{\rm PyQCU}>1:\ \mathrm{MG\ 35/44;\ trace\!-\!off\ 21/22}$",
        size=18.5,
        color=NAVY,
    )
    s.text(0.08, 0.194, "不宣称任意格点、任意层数或所有求解器全面领先。", size=12, color=CORAL, weight="bold")
    s.footer(
        "来源：tag test27；docs/report_multigrid_optimized_20260927.tex:327-342；"
        "docs/report_multigrid_quda_pyqcu_20260928.tex:346-364。"
    )
    return s.save()


def slide_15() -> Path:
    s = Slide(
        15,
        "参考实现与复现入口：先跑最小闸门，再进入公平矩阵",
        "PyQCU 的算法来源、对照实现和对内证据均可追溯到源码或 tag。",
        "Supplement · code",
    )
    s.box(0.055, 0.40, 0.43, 0.40, face=PALE_BLUE, edge=WHITE)
    s.text(0.08, 0.755, "参考文献与代码", size=17, color=BLUE, weight="bold")
    refs = [
        ("quantum-mg", "github.com/weinbe2/quantum-mg", "Evan Weinberg / NVIDIA"),
        ("QUDA", "github.com/lattice/quda", "MG / staging / coarse operator"),
        ("DDalphaAMG", "github.com/mrottmann/DDalphaAMG", "Matthias Rottmann"),
        ("源码快照", "refer/git-rep/**", "本地可核验参考树"),
    ]
    for i, (name, url, note) in enumerate(refs):
        y = 0.695 - i * 0.065
        s.text(0.083, y, name, size=12.3, color=NAVY, weight="bold")
        s.text(0.22, y, url, size=11.5, color=TEAL)
        s.text(0.22, y - 0.024, note, size=9.4, color=MUTED)

    s.box(0.515, 0.40, 0.43, 0.40, face=PALE_TEAL, edge=WHITE)
    s.text(0.54, 0.755, "最短可靠复现接口", size=17, color=TEAL, weight="bold")
    commands = [
        "source ./env.sh",
        "bash ./build.sh && bash ./install.sh",
        "python -B .../run_strict_fast.py --list",
        "python -B .../run_strict_fast.py --fail-fast --json strict-fast.json",
        "mpirun -np 2 python .../strict_mpi_solve_probe.py ...",
        "python -B .../bench_mg_matrix_full.py --list",
    ]
    for i, command in enumerate(commands):
        s.text(0.54, 0.704 - i * 0.046, command, size=10.4, color=INK)

    s.card(
        0.055,
        0.17,
        0.43,
        0.18,
        "主矩阵与聚合器",
        "bench_mg_matrix_full.py → bench_strict_vs_quda.py → assemble_final_matrix.py → build_mg_report.py",
        accent=AMBER,
        heading_size=15.2,
        body_size=11.2,
        wrap=52,
    )
    s.card(
        0.515,
        0.17,
        0.43,
        0.18,
        "平台状态",
        "CUDA C++ 已实测；Torch CPU 部分；DCU/CANN 历史或仿真；TileLang smoke；大规模数据见下一页。",
        accent=GREEN,
        heading_size=15.2,
        body_size=11.2,
        wrap=52,
    )
    s.footer(
        "项目：github.com/zhangxin8069/PyQCU；镜像：gitee.com/zhangxin8069/PyQCU；"
        "当前迭代较快，正式运行应以 main 最新入口与 provenance 为准。"
    )
    return s.save()


def slide_16() -> Path:
    s = Slide(
        16,
        "先前成果：508 报告完成 1024 进程规模的大格扩展测试",
        "图表数据取自《张鑫 508应用测试报告-PyQCU》；用于展示早期 DCU 大规模成果，不与 test27 的 MG 速度比混合。",
        "Prior results · 508 report",
    )
    add_kpi(s, 0.055, 0.705, 0.19, 0.115, "1024", "进程/线程规模", CORAL, 22)
    add_kpi(s, 0.26, 0.705, 0.19, 0.115, "3.12×", "Wilson 报告加速比", BLUE, 22)
    add_kpi(s, 0.465, 0.705, 0.19, 0.115, "7.19×", "Clover 报告加速比", TEAL, 22)
    add_kpi(s, 0.67, 0.705, 0.275, 0.115, "≈2048 GiB", "64 卡池峰值估算", AMBER, 19)

    draw_508_strong_scaling(s, 0.065, 0.40, 0.39, 0.255)
    draw_508_weak_scaling(s, 0.505, 0.40, 0.44, 0.255)
    s.text(
        0.08,
        0.345,
        "强扩展：64→1024；数据来自 TEST8/TEST9 汇总。",
        size=10.7,
        color=NAVY,
    )
    s.text(
        0.52,
        0.345,
        "弱扩展差异：Clover-Wilson 单次迭代耗时随等效体积增长。",
        size=10.7,
        color=NAVY,
    )

    s.image(REPORT_508_IMAGE25, 0.055, 0.185, 0.42, 0.125)
    s.text(0.055, 0.32, "508 报告原始表：TEST9 强扩展数据", size=10.2, color=MUTED)
    s.card(
        0.505,
        0.15,
        0.44,
        0.18,
        "如何解读",
        "该成果证明早期 DCU/CUDA 路径在多进程大格测试中具备正确性与扩展趋势；"
        "但 MG 当时仍属前期验证，不能替代 test27 的 Strict-MG 正式对照。",
        accent=AMBER,
        heading_size=14.2,
        body_size=10.9,
        wrap=45,
    )
    s.footer(
        "来源：docs/张鑫 508应用测试报告-PyQCU/word/document.xml §1.5-1.6 与原始表 image25.png；"
        "环境：4×Pre-Wukong DCU、200Gb 网络、complex64/c128、Mass=0.05、tol(x_o)=1e-12。"
    )
    return s.save()


def slide_17() -> Path:
    s = Slide(17, "", "")
    s.ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=WHITE, edgecolor="none", zorder=-20))
    s.ax.add_patch(Rectangle((0, 0), 0.013, 1, color=TEAL, zorder=10))
    s.text(0.10, 0.69, "谢谢", size=42, color=NAVY, weight="bold")
    s.text(0.105, 0.58, "PyQCU Strict MultiGrid", size=22, color=TEAL, weight="bold")
    s.text(
        0.105,
        0.49,
        "下一步：把粗层同步、通信重叠和 setup 显存做成可复验的持续收益。",
        size=18,
        color=INK,
    )
    s.text(0.105, 0.38, "项目：github.com/zhangxin8069/PyQCU", size=13, color=MUTED)
    s.text(0.105, 0.335, "数据基线：git tag test27 · 2026-09-28", size=13, color=MUTED)
    s.text(0.105, 0.24, "Q & A", size=15, color=CORAL, weight="bold")
    return s.save()


def build_contact_sheet(paths: list[Path]) -> Path:
    thumb_w, thumb_h = 480, 270
    cols, rows = 3, 6
    sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype(str(FONT_PATH), 22)
    except Exception:
        font = ImageFont.load_default()
    for idx, path in enumerate(paths):
        image = Image.open(path).convert("RGB").resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x = (idx % cols) * thumb_w
        y = (idx // cols) * thumb_h
        sheet.paste(image, (x, y))
        draw.rectangle((x + 6, y + 6, x + 58, y + 38), fill=(18, 53, 91))
        draw.text((x + 16, y + 9), f"{idx + 1:02d}", fill="white", font=font)
    target = ASSET_DIR / "contact_sheet.png"
    sheet.save(target)
    return target


def build_pdf(paths: list[Path]) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    images[0].save(PDF_PATH, save_all=True, append_images=images[1:], resolution=DPI)


def build_notes() -> None:
    notes = """# High-Performance Implementation of Multigrid Solver for Lattice QCD

## 1. 标题页（20 秒）
介绍题目、汇报人和证据基线。强调本报告只主张 MultiGrid，不把结果外推到全部求解器。

## 2. Wilson 与 Clover（35 秒）
先给 H、D_W、κ，再给 Clover onsite 块。说明 C++ 返回的裸核与完整算子不是一个对象。

## 3. 原始 Dslash（30 秒）
按“八邻居、spin projection、SU(3)、累加”讲带宽瓶颈；指出 xyzt 尾轴布局和投影重用。

## 4. 奇偶消元（30 秒）
块矩阵、Schur 补、RHS 与重建。强调只改变未知量组织，真残差仍回到 full operator。

## 5. 通用 MG（35 秒）
null vector 近似低模，局部 QR 建 P/R，Galerkin 生成粗算子，V-cycle 作预条件器。

## 6. 预条件 Bi-CGStab（35 秒）
按表逐行说明 rho、p、v、alpha、s、omega 和 x 更新。强调预条件器 M 与热启动路径。

## 7. PyQCU Strict（35 秒）
D=X+H、Dhat=X^-1 D、D_c=R Dhat P。细层 compact target parity，粗层 full X/Y/Yhat。

## 8. V-cycle 与 FGMRES（35 秒）
左侧是递归 MG 预条件器，右侧是 flexible right-preconditioned GMRES。说明迭代不跨层相加。

## 9. CUDA-C++ 优化（40 秒）
从寄存器、流融合、device scalar、CUDA Graph、setup/cache 讲到 MPI overlap；c128 默认回退。

## 10. 对比协议（30 秒）
72 请求、66 精确、132 side record；1+2+5 phase；真残差与 85% 显存门。

## 11. 全分布（35 秒）
总中位 1.025，但 MG 35/44，trace-off MG 21/22；长尾和 BiCG 0/22 必须一起讲。

## 12. 分组中位（35 秒）
V100 MG-3 最突出，P100 trace-off 温和有利；trace-on 只作诊断。

## 13. 瓶颈与容量（40 秒）
PyQCU 记录中 coarse+other 主导；双 P100 大格 6 项容量替代，不能混入原大格速度比。

## 14. 优势、不足与下一步（35 秒）
优势是 MG 专项、自主可控、显存；不足是总能力和通信成熟度；下一步聚焦粗层同步、通信和 setup。

## 15. 参考文献与复现（25 秒）
列出 quantum-mg、QUDA、DDalphaAMG；快速闸门先于完整矩阵，并简述平台适配状态。

## 16. 508 报告大规模测试（30 秒）
展示 508 报告的 64→1024 进程强扩展曲线、弱扩展差异曲线和 TEST9 原始表。
说明这是早期 DCU 路径成果，MG 当时仍属前期验证，不能与 test27 的 Strict-MG 速度比混合。

## 17. 致谢（10 秒）
一句总结和一个明确行动项，然后进入问答。
"""
    NOTES_PATH.write_text(notes, encoding="utf-8")


def build_pptx(paths: list[Path], output: Path) -> None:
    ns_p = "http://schemas.openxmlformats.org/presentationml/2006/main"
    ns_a = "http://schemas.openxmlformats.org/drawingml/2006/main"
    ns_r = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

    def slide_xml() -> str:
        return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:p="{ns_p}" xmlns:a="{ns_a}" xmlns:r="{ns_r}">
  <p:cSld name="">
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name=""/>
        <p:cNvGrpSpPr/>
        <p:nvPr/>
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>
          <a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/>
        </a:xfrm>
      </p:grpSpPr>
      <p:pic>
        <p:nvPicPr>
          <p:cNvPr id="2" name="Slide image" descr="Rendered PyQCU MultiGrid slide"/>
          <p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr>
          <p:nvPr/>
        </p:nvPicPr>
        <p:blipFill>
          <a:blip r:embed="rId1"/>
          <a:stretch><a:fillRect/></a:stretch>
        </p:blipFill>
        <p:spPr>
          <a:xfrm><a:off x="0" y="0"/><a:ext cx="12192000" cy="6858000"/></a:xfrm>
          <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
        </p:spPr>
      </p:pic>
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>
"""

    n = len(paths)
    slide_ids = "".join(
        f'<p:sldId id="{256 + i}" r:id="rId{i + 1}"/>' for i in range(n)
    )
    master_rid = n + 1
    theme_rid = n + 2
    presentation = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:p="{ns_p}" xmlns:a="{ns_a}" xmlns:r="{ns_r}">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId{master_rid}"/></p:sldMasterIdLst>
  <p:sldIdLst>{slide_ids}</p:sldIdLst>
  <p:sldSz cx="12192000" cy="6858000" type="screen16x9"/>
  <p:notesSz cx="6858000" cy="9144000"/>
  <p:defaultTextStyle/>
</p:presentation>
"""
    pres_rels = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
    ]
    for i in range(n):
        pres_rels.append(
            f'<Relationship Id="rId{i + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i + 1}.xml"/>'
        )
    pres_rels.append(
        f'<Relationship Id="rId{master_rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>'
    )
    pres_rels.append(
        f'<Relationship Id="rId{theme_rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>'
    )
    pres_rels.append("</Relationships>")

    slide_master = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:p="{ns_p}" xmlns:a="{ns_a}" xmlns:r="{ns_r}">
  <p:cSld name="PyQCU MultiGrid">
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
    </p:spTree>
  </p:cSld>
  <p:clrMap accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" bg1="lt1" bg2="lt2" folHlink="folHlink" hlink="hlink" tx1="dk1" tx2="dk2"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="1" r:id="rId1"/></p:sldLayoutIdLst>
  <p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>
</p:sldMaster>
"""
    slide_layout = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:p="{ns_p}" xmlns:a="{ns_a}" xmlns:r="{ns_r}" type="blank" preserve="1">
  <p:cSld name="Blank">
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>
"""
    theme = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="{ns_a}" name="PyQCU">
  <a:themeElements>
    <a:clrScheme name="PyQCU">
      <a:dk1><a:srgbClr val="17212B"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1>
      <a:dk2><a:srgbClr val="12355B"/></a:dk2><a:lt2><a:srgbClr val="EEF3F6"/></a:lt2>
      <a:accent1><a:srgbClr val="245BA3"/></a:accent1><a:accent2><a:srgbClr val="0B7285"/></a:accent2>
      <a:accent3><a:srgbClr val="D08A1F"/></a:accent3><a:accent4><a:srgbClr val="C94C5A"/></a:accent4>
      <a:accent5><a:srgbClr val="53A6B8"/></a:accent5><a:accent6><a:srgbClr val="3A7D5C"/></a:accent6>
      <a:hlink><a:srgbClr val="245BA3"/></a:hlink><a:folHlink><a:srgbClr val="C94C5A"/></a:folHlink>
    </a:clrScheme>
    <a:fontScheme name="PyQCU">
      <a:majorFont><a:latin typeface="AR PL UMing CN"/><a:ea typeface="AR PL UMing CN"/><a:cs typeface=""/></a:majorFont>
      <a:minorFont><a:latin typeface="AR PL UMing CN"/><a:ea typeface="AR PL UMing CN"/><a:cs typeface=""/></a:minorFont>
    </a:fontScheme>
    <a:fmtScheme name="PyQCU">
      <a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:gradFill rotWithShape="1"><a:gsLst><a:gs pos="0"><a:schemeClr val="phClr"><a:lumMod val="110000"/><a:satMod val="105000"/></a:schemeClr></a:gs><a:gs pos="100000"><a:schemeClr val="phClr"><a:lumMod val="90000"/><a:satMod val="105000"/></a:schemeClr></a:gs></a:gsLst><a:lin ang="5400000" scaled="0"/></a:gradFill><a:gradFill rotWithShape="1"><a:gsLst><a:gs pos="0"><a:schemeClr val="phClr"><a:lumMod val="110000"/><a:satMod val="105000"/></a:schemeClr></a:gs><a:gs pos="100000"><a:schemeClr val="phClr"><a:lumMod val="90000"/><a:satMod val="105000"/></a:schemeClr></a:gs></a:gsLst><a:lin ang="5400000" scaled="0"/></a:gradFill></a:fillStyleLst>
      <a:lnStyleLst><a:ln w="6350" cap="flat" cmpd="sng" algn="ctr"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:prstDash val="solid"/></a:ln><a:ln w="12700" cap="flat" cmpd="sng" algn="ctr"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:prstDash val="solid"/></a:ln><a:ln w="19050" cap="flat" cmpd="sng" algn="ctr"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:prstDash val="solid"/></a:ln></a:lnStyleLst>
      <a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst>
      <a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst>
    </a:fmtScheme>
  </a:themeElements>
  <a:objectDefaults/><a:extraClrSchemeLst/>
</a:theme>
"""
    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Default Extension="png" ContentType="image/png"/>',
        '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>',
        '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>',
        '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>',
        '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>',
    ]
    for i in range(1, n + 1):
        content_types.append(
            f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        )
    content_types.extend(
        [
            '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
            '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
            "</Types>",
        ]
    )
    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""
    core = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>High-Performance Implementation of Multigrid Solver for Lattice QCD</dc:title>
  <dc:creator>张鑫</dc:creator><cp:lastModifiedBy>PyQCU</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">2026-09-28T00:00:00Z</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">2026-09-28T00:00:00Z</dcterms:modified>
</cp:coreProperties>
"""
    app = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Python OOXML generator</Application>
  <PresentationFormat>On-screen Show (16:9)</PresentationFormat>
  <Slides>{n}</Slides>
  <Company>PyQCU</Company>
</Properties>
"""
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "\n".join(content_types))
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("docProps/core.xml", core)
        archive.writestr("docProps/app.xml", app)
        archive.writestr("ppt/presentation.xml", presentation)
        archive.writestr("ppt/_rels/presentation.xml.rels", "\n".join(pres_rels))
        archive.writestr("ppt/slideMasters/slideMaster1.xml", slide_master)
        archive.writestr(
            "ppt/slideMasters/_rels/slideMaster1.xml.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>
""",
        )
        archive.writestr("ppt/slideLayouts/slideLayout1.xml", slide_layout)
        archive.writestr(
            "ppt/slideLayouts/_rels/slideLayout1.xml.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>
""",
        )
        archive.writestr("ppt/theme/theme1.xml", theme)
        for i, path in enumerate(paths, start=1):
            archive.write(path, f"ppt/media/image{i}.png")
            archive.writestr(f"ppt/slides/slide{i}.xml", slide_xml())
            archive.writestr(
                f"ppt/slides/_rels/slide{i}.xml.rels",
                f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/image{i}.png"/>
</Relationships>
""",
            )


def validate_outputs(paths: list[Path], pptx_path: Path) -> None:
    assert len(paths) == 17, f"expected 17 slides, got {len(paths)}"
    for path in paths:
        with Image.open(path) as image:
            assert image.size == (PX_W, PX_H), (path, image.size)
    assert PDF_PATH.exists() and PDF_PATH.stat().st_size > 0
    assert NOTES_PATH.exists() and NOTES_PATH.stat().st_size > 0
    with zipfile.ZipFile(pptx_path) as archive:
        assert archive.testzip() is None
        names = set(archive.namelist())
        assert "ppt/presentation.xml" in names
        for i in range(1, 18):
            slide_name = f"ppt/slides/slide{i}.xml"
            rel_name = f"ppt/slides/_rels/slide{i}.xml.rels"
            image_name = f"ppt/media/image{i}.png"
            assert slide_name in names
            assert rel_name in names
            assert image_name in names
            ET.fromstring(archive.read(slide_name))
            ET.fromstring(archive.read(rel_name))
        for name in names:
            if name.endswith((".xml", ".rels")):
                ET.fromstring(archive.read(name))
    with PDF_PATH.open("rb") as handle:
        assert handle.read(5) == b"%PDF-"


def main() -> None:
    SLIDE_DIR.mkdir(parents=True, exist_ok=True)
    builders = [
        slide_01,
        slide_02,
        slide_03,
        slide_04,
        slide_05,
        slide_06,
        slide_07,
        slide_08,
        slide_09,
        slide_10,
        slide_11,
        slide_12,
        slide_13,
        slide_14,
        slide_15,
        slide_16,
        slide_17,
    ]
    paths = [build() for build in builders]
    build_contact_sheet(paths)
    build_pdf(paths)
    build_notes()
    build_pptx(paths, PPTX_PATH)
    validate_outputs(paths, PPTX_PATH)
    print(f"slides={len(paths)}")
    print(f"pptx={PPTX_PATH}")
    print(f"pdf={PDF_PATH}")
    print(f"notes={NOTES_PATH}")


if __name__ == "__main__":
    main()
