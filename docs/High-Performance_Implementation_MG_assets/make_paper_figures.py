#!/usr/bin/env python3
"""Generate publication-style vector figures for the MultiGrid report."""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "report_multigrid_comprehensive_20260928" / "final_protocol"
UNITS_CSV = DATA_DIR / "units.csv"
STAGES_CSV = DATA_DIR / "stages.csv"
OUT_DIR = Path(__file__).resolve().parent / "paper_assets"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 10,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9,
        "axes.linewidth": 0.75,
        "axes.edgecolor": "black",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)

COLORS = {
    "mg2": "#2E6DB4",
    "mg3": "#D9772A",
    "bicg": "#168C8C",
    "pyqcu": "#2E6DB4",
    "quda": "#D35B5B",
    "stage": ["#92C5DE", "#4393C3", "#E38C52", "#D35B5B", "#4A4A4A", "#B9B9B9"],
}


def solver_color(solver: str) -> str:
    return {
        "mg-2": COLORS["mg2"],
        "mg-3": COLORS["mg3"],
        "bicgstab-reference": COLORS["bicg"],
    }[solver]


def load_units() -> list[dict[str, object]]:
    keys = {
        "ratio_quda_over_pyqcu",
        "pyqcu_true_residual",
        "quda_true_residual",
        "pyqcu_steady_sampler_peak_bytes",
        "quda_steady_sampler_peak_bytes",
    }
    rows: list[dict[str, object]] = []
    with UNITS_CSV.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, object] = dict(raw)
            for key in keys:
                if row.get(key, "") not in ("", None):
                    row[key] = float(row[key])
            row["levels"] = int(row["levels"])
            rows.append(row)
    return rows


def load_stage_shares() -> dict[str, dict[str, float]]:
    fields = {
        "pre_smoother_seconds": "pre",
        "post_smoother_seconds": "post",
        "restrict_seconds": "restrict",
        "prolongate_seconds": "prolong",
        "coarse_solver_seconds": "coarse",
        "other_seconds": "other",
    }
    grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    with STAGES_CSV.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["trace"] != "on":
                continue
            key = (row["unit_id"], row["side"])
            for source, target in fields.items():
                grouped[key][target] += float(row[source])
    result: dict[str, dict[str, float]] = {}
    for side in ("pyqcu", "quda"):
        shares = []
        for (_, row_side), values in grouped.items():
            if row_side != side:
                continue
            total = sum(values.values())
            if total:
                shares.append({name: values[name] / total for name in fields.values()})
        result[side] = {
            name: statistics.median([share[name] for share in shares])
            for name in fields.values()
        }
    return result


def save(fig: plt.Figure, name: str) -> None:
    OUTPUT = OUT_DIR / name
    fig.savefig(OUTPUT, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(OUTPUT)


def fig_ratio_distribution(rows: list[dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    categories = [
        ("V100, c64", "v100", "c64"),
        ("V100, c128", "v100", "c128"),
        ("P100x2, c64", "p100", "c64"),
        ("P100x2, c128", "p100", "c128"),
    ]
    marker_for = {"mg-2": "o", "mg-3": "^", "bicgstab-reference": "s"}
    for index, (label, device, precision) in enumerate(categories):
        selected = [
            row
            for row in rows
            if row["device"] == device and row["precision"] == precision
        ]
        for row in selected:
            solver = str(row["solver"])
            trace = str(row["trace"])
            fill = solver_color(solver) if trace == "off" else "white"
            edge = solver_color(solver)
            size = 28 if trace == "off" else 24
            offset = -0.12 if str(row["levels"]) == "2" else 0.12
            if solver == "bicgstab-reference":
                offset = 0.0
            ax.scatter(
                index + offset,
                float(row["ratio_quda_over_pyqcu"]),
                marker=marker_for[solver],
                s=size,
                facecolors=fill,
                edgecolors=edge,
                linewidths=0.65,
                zorder=3,
            )
        median = statistics.median(
            float(row["ratio_quda_over_pyqcu"])
            for row in selected
            if row["trace"] == "off"
        )
        ax.plot(
            [index - 0.27, index + 0.27],
            [median, median],
            color="0.15",
            linewidth=1.4,
            zorder=4,
        )
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yscale("log")
    ax.set_ylim(0.15, 18.0)
    ax.set_xticks(range(4), [item[0] for item in categories])
    ax.set_ylabel(r"$R=t_{\rm QUDA}/t_{\rm PyQCU}$")
    ax.grid(axis="y", color="0.86", linewidth=0.6)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", color=COLORS["mg2"], label="MG-2"),
        plt.Line2D([], [], marker="^", linestyle="none", color=COLORS["mg3"], label="MG-3"),
        plt.Line2D([], [], marker="s", linestyle="none", color=COLORS["bicg"], label="BiCGStab"),
        plt.Line2D([], [], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor="black", label="open: trace-on"),
    ]
    ax.legend(handles=handles, ncol=2, fontsize=8, loc="upper left")
    ax.annotate(
        "14.63",
        (1.12, 14.634),
        xytext=(1.25, 11.0),
        fontsize=8,
        arrowprops={"arrowstyle": "-", "linewidth": 0.6, "color": "black"},
    )
    ax.annotate(
        "0.177",
        (0.0, 0.17657),
        xytext=(-0.34, 0.22),
        fontsize=8,
        arrowprops={"arrowstyle": "-", "linewidth": 0.6, "color": "black"},
    )
    save(fig, "fig_ratio_distribution.pdf")


def fig_group_medians(rows: list[dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    configs = [
        ("V100, c64", "v100", "c64"),
        ("V100, c128", "v100", "c128"),
        ("P100x2, c64", "p100", "c64"),
        ("P100x2, c128", "p100", "c128"),
    ]
    solvers = [
        ("MG-2", "mg-2", COLORS["mg2"], "o"),
        ("MG-3", "mg-3", COLORS["mg3"], "^"),
        ("BiCGStab", "bicgstab-reference", COLORS["bicg"], "s"),
    ]
    width = 0.22
    x = list(range(len(configs)))
    for solver_index, (label, solver, face, marker) in enumerate(solvers):
        values = []
        for _, device, precision in configs:
            sample = [
                float(row["ratio_quda_over_pyqcu"])
                for row in rows
                if row["device"] == device
                and row["precision"] == precision
                and row["solver"] == solver
                and row["trace"] == "off"
            ]
            values.append(statistics.median(sample))
        positions = [value + (solver_index - 1) * width for value in x]
        ax.scatter(
            positions,
            values,
            marker=marker,
            s=42,
            facecolors=face,
            edgecolors="black",
            linewidths=0.7,
            label=label,
            zorder=3,
        )
        for position, value in zip(positions, values):
            ax.vlines(position, 0.15, value, color="0.75", linewidth=0.6, zorder=1)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yscale("log")
    ax.set_ylim(0.15, 6.0)
    ax.set_xticks(x, [item[0] for item in configs])
    ax.set_ylabel(r"median $R$")
    ax.grid(axis="y", color="0.88", linewidth=0.55)
    ax.legend(ncol=3, fontsize=8, loc="upper right")
    save(fig, "fig_group_medians.pdf")


def fig_stage_shares(shares: dict[str, dict[str, float]]) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    stages = ["pre", "post", "restrict", "prolong", "coarse", "other"]
    labels = ["pre", "post", "R", "P", "coarse", "other"]
    x = list(range(len(stages)))
    width = 0.34
    py = [100.0 * shares["pyqcu"][name] for name in stages]
    qu = [100.0 * shares["quda"][name] for name in stages]
    ax.bar(
        [value - width / 2 for value in x],
        py,
        width,
        label="PyQCU",
        color=COLORS["stage"],
        edgecolor="black",
        linewidth=0.65,
    )
    ax.bar(
        [value + width / 2 for value in x],
        qu,
        width,
        label="QUDA",
        facecolor="white",
        edgecolor="black",
        linewidth=0.65,
        hatch="//",
    )
    ax.set_xticks(x, labels)
    ax.set_ylabel("share of recorded stage time (%)")
    ax.set_ylim(0, 82)
    ax.legend(ncol=2, fontsize=8)
    ax.grid(axis="y", color="0.88", linewidth=0.55)
    save(fig, "fig_stage_shares.pdf")


def fig_residual_memory(rows: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.7))
    sides = ["PyQCU", "QUDA"]
    residual = [
        max(float(row["pyqcu_true_residual"]) for row in rows),
        max(float(row["quda_true_residual"]) for row in rows),
    ]
    memory = [
        max(float(row["pyqcu_steady_sampler_peak_bytes"]) for row in rows) / 2**20,
        max(float(row["quda_steady_sampler_peak_bytes"]) for row in rows) / 2**20,
    ]
    axes[0].bar(
        sides,
        residual,
        color=[COLORS["pyqcu"], COLORS["quda"]],
        edgecolor="black",
        linewidth=0.75,
        hatch=["//", ""],
    )
    axes[0].axhline(5e-6, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_yscale("log")
    axes[0].set_ylim(1e-8, 1e-5)
    axes[0].set_ylabel("maximum true residual")
    axes[0].text(
        1.0,
        5e-6 * 1.15,
        "gate",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    axes[1].bar(
        sides,
        memory,
        color=[COLORS["pyqcu"], COLORS["quda"]],
        edgecolor="black",
        linewidth=0.75,
    )
    axes[1].set_ylabel("maximum observed memory (MiB)")
    for axis in axes:
        axis.grid(axis="y", color="0.88", linewidth=0.55)
    fig.tight_layout()
    save(fig, "fig_residual_memory.pdf")


def fig_508_scaling() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.9))
    processes = [64, 128, 256, 512, 1024]
    wilson = [1.000, 1.451, 2.001, 4.739, 3.119]
    clover = [1.000, 1.657, 2.687, 4.670, 7.193]
    axes[0].plot(
        processes,
        wilson,
        marker="o",
        color=COLORS["pyqcu"],
        linewidth=1.3,
        label="Wilson",
    )
    axes[0].plot(
        processes,
        clover,
        marker="^",
        color=COLORS["mg3"],
        linewidth=1.3,
        label="Clover",
    )
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(processes, [str(value) for value in processes])
    axes[0].set_ylim(0.5, 8.0)
    axes[0].set_xlabel("processes / threads")
    axes[0].set_ylabel("reported speedup")
    axes[0].grid(axis="y", color="0.88", linewidth=0.55)
    axes[0].legend(ncol=2, fontsize=8)

    x = [0.0, 1.0]
    width = 0.34
    axes[1].bar(
        [value - width / 2 for value in x],
        [3.119, 2.211],
        width,
        color=COLORS["pyqcu"],
        edgecolor="black",
        linewidth=0.65,
        label="Wilson",
    )
    axes[1].bar(
        [value + width / 2 for value in x],
        [7.193, 4.258],
        width,
        color=COLORS["mg3"],
        edgecolor="black",
        linewidth=0.65,
        label="Clover",
    )
    for position, value in zip(
        [x[0] - width / 2, x[0] + width / 2,
         x[1] - width / 2, x[1] + width / 2],
        [3.119, 7.193, 2.211, 4.258],
    ):
        axes[1].text(
            position, value + 0.13, f"{value:.3f}",
            ha="center", va="bottom", fontsize=8,
        )
    axes[1].set_xticks(x, ["TEST8 c64", "TEST9 c128"])
    axes[1].set_ylim(0, 8.4)
    axes[1].set_ylabel("reported speedup at 1024 ranks")
    axes[1].grid(axis="y", color="0.88", linewidth=0.55)
    axes[1].legend(ncol=2, fontsize=8, loc="upper right")
    fig.tight_layout()
    save(fig, "fig_508_scaling.pdf")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_units()
    stages = load_stage_shares()
    fig_ratio_distribution(rows)
    fig_group_medians(rows)
    fig_stage_shares(stages)
    fig_residual_memory(rows)
    fig_508_scaling()


if __name__ == "__main__":
    main()
