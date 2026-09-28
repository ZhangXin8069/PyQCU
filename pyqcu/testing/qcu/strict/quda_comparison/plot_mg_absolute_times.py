#!/usr/bin/env python3
"""Build readable absolute-time figures from formal MG comparison JSON."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


UNIT_RE = re.compile(
    r"(?:pyqcu__)?(?P<device>p100|v100)__(?P<lattice>[^_]+)__"
    r"(?P<precision>c64|c128)__l(?P<levels>[123])__trace-(?P<trace>on|off)"
)
STAGE_FIELDS = (
    ("pre_smoother_seconds", "pre"),
    ("post_smoother_seconds", "post"),
    ("restriction_seconds", "restrict"),
    ("prolongation_seconds", "prolong"),
    ("coarse_solver_seconds", "coarse"),
    ("other_seconds", "other"),
)
STAGE_COLORS = {
    "pre": "#245ba3",
    "post": "#5f9bd6",
    "restrict": "#b5483f",
    "prolong": "#df7a52",
    "coarse": "#59666f",
    "other": "#c4c9cd",
}
SIDE_COLORS = {"pyqcu": "#245ba3", "quda": "#be4137"}
SIDE_LABELS = {"pyqcu": "PyQCU", "quda": "QUDA"}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _median(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        raise ValueError("median of an empty sequence")
    return float(statistics.median(items))


def _format_seconds(value: float) -> str:
    if value >= 10.0:
        return f"{value:.1f}"
    if value >= 1.0:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _format_ratio(value: float) -> str:
    if value >= 10.0:
        return f"{value:.1f}x"
    return f"{value:.3f}x"


def _load_unit(path: Path) -> dict[str, Any]:
    match = UNIT_RE.fullmatch(path.stem)
    if match is None:
        raise ValueError(f"unexpected unit filename: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("state") != "complete":
        raise ValueError(f"{path} is not complete")
    dimensions = match.groupdict()
    levels = int(dimensions["levels"])
    return {
        **dimensions,
        "levels": levels,
        "solver": "bicgstab" if levels == 1 else f"mg-{levels}",
        "document": document,
        "path": path,
    }


def load_units(input_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(input_dir.glob("*.json"))
    if not paths:
        raise ValueError(f"no combined JSON records under {input_dir}")
    return [_load_unit(path) for path in paths]


def _active_side(unit: Mapping[str, Any], side: str) -> Mapping[str, Any]:
    side_document = unit["document"]["sides"][side]
    if side_document.get("status") != "ok":
        raise ValueError(f"{unit['path']} has non-ok {side} status")
    return side_document


def steady_seconds(unit: Mapping[str, Any], side: str) -> float:
    side_document = _active_side(unit, side)
    if int(unit["levels"]) == 1:
        return _float(
            side_document["reference_solver"]["steady"]["median_seconds"]
        )
    return _float(side_document["timing"]["steady"]["median_seconds"])


def final_true_residual(unit: Mapping[str, Any], side: str) -> float:
    residual = _active_side(unit, side).get("true_residual") or {}
    samples = residual.get("samples_rel") or []
    if samples:
        return max(_float(value) for value in samples)
    return _float(residual.get("max_rel"))


def steady_outer_iterations(unit: Mapping[str, Any], side: str) -> float:
    side_document = _active_side(unit, side)
    if int(unit["levels"]) == 1:
        iterations = side_document["reference_solver"]["iterations"]
        return _float(
            iterations.get("median_seconds", iterations.get("median"))
        )
    return _float(side_document["iterations"]["median"])


def stage_rows(unit: Mapping[str, Any], side: str) -> list[dict[str, Any]]:
    """Return active MG level rows, excluding all-zero trailing placeholders."""

    raw_rows = _active_side(unit, side).get("mg_levels") or []
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        values = {
            name: _float(raw.get(field))
            for field, name in STAGE_FIELDS
        }
        total = sum(values.values())
        if total <= 0.0:
            continue
        rows.append(
            {
                "level": int(raw.get("level", len(rows))),
                "seconds": values,
                "total_seconds": total,
            }
        )
    if not rows:
        raise ValueError(
            f"{unit['path']}: no nonzero MG stage rows for {side}"
        )
    rows.sort(key=lambda item: item["level"])
    return rows


def _case_label(unit: Mapping[str, Any]) -> str:
    return (
        f"{unit['precision']} {unit['lattice']} "
        f"{unit['solver'].upper()} {unit['trace']}"
    )


def _group_label(unit: Mapping[str, Any]) -> str:
    return (
        f"{unit['device']}|{unit['precision']}|"
        f"{unit['solver']}|{unit['trace']}"
    )


def _save(
    fig: plt.Figure,
    outdir: Path,
    stem: str,
    *,
    metadata: Mapping[str, str] | None = None,
) -> None:
    kwargs: dict[str, Any] = {"bbox_inches": "tight"}
    if metadata is not None:
        kwargs["metadata"] = metadata
    fig.savefig(outdir / f"{stem}.pdf", format="pdf", **kwargs)
    fig.savefig(outdir / f"{stem}.svg", format="svg", **kwargs)
    plt.close(fig)


def _add_pair_legend(fig: plt.Figure, y: float = 0.01) -> None:
    handles = [
        Patch(facecolor=SIDE_COLORS["pyqcu"], label="PyQCU"),
        Patch(facecolor=SIDE_COLORS["quda"], label="QUDA"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=2,
        frameon=False,
        fontsize=9,
    )


def plot_time_tradeoff(units: Sequence[Mapping[str, Any]], outdir: Path) -> None:
    fig, axis = plt.subplots(figsize=(9.0, 8.0))
    marker_by_solver = {"bicgstab": "o", "mg-2": "s", "mg-3": "^"}
    color_by_device = {"v100": "#245ba3", "p100": "#be4137"}
    pyqtcu_times = []
    quda_times = []
    for unit in units:
        pyqcu = steady_seconds(unit, "pyqcu")
        quda = steady_seconds(unit, "quda")
        pyqtcu_times.append(pyqcu)
        quda_times.append(quda)
        axis.scatter(
            pyqcu,
            quda,
            marker=marker_by_solver[str(unit["solver"])],
            facecolors=(
                color_by_device[str(unit["device"])]
                if unit["trace"] == "off"
                else "white"
            ),
            edgecolors=color_by_device[str(unit["device"])],
            linewidths=1.0,
            s=42,
            alpha=0.9,
        )
    lower = max(1e-3, min(pyqtcu_times + quda_times) * 0.7)
    upper = max(pyqtcu_times + quda_times) * 1.35
    axis.plot([lower, upper], [lower, upper], color="#333333", linewidth=1.0)
    axis.plot(
        [lower, upper],
        [4.0 * lower, 4.0 * upper],
        color="#777777",
        linestyle="--",
        linewidth=0.8,
    )
    axis.plot(
        [lower, upper],
        [0.25 * lower, 0.25 * upper],
        color="#777777",
        linestyle=":",
        linewidth=0.8,
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_xlabel("PyQCU steady solve time (s, log scale)")
    axis.set_ylabel("QUDA steady solve time (s, log scale)")
    axis.set_title("Absolute steady solve times for all 66 exact units")
    axis.grid(True, which="both", alpha=0.18, linewidth=0.5)

    solver_handles = [
        plt.Line2D(
            [],
            [],
            linestyle="none",
            marker=marker_by_solver[name],
            markerfacecolor="white",
            markeredgecolor="black",
            label=label,
        )
        for name, label in (
            ("bicgstab", "BiCGStab"),
            ("mg-2", "MG-2"),
            ("mg-3", "MG-3"),
        )
    ]
    device_handles = [
        Patch(facecolor=color, label=device.upper())
        for device, color in color_by_device.items()
    ]
    line_handles = [
        plt.Line2D([], [], color="#333333", label="QUDA = PyQCU"),
        plt.Line2D(
            [], [], color="#777777", linestyle="--", label="QUDA = 4x PyQCU"
        ),
        plt.Line2D(
            [], [], color="#777777", linestyle=":", label="QUDA = 0.25x PyQCU"
        ),
    ]
    fig.legend(
        handles=solver_handles + device_handles + line_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        frameon=False,
        fontsize=8.5,
    )
    fig.subplots_adjust(bottom=0.24)
    _save(
        fig,
        outdir,
        "mg_time_tradeoff",
        metadata={"Title": "MG absolute steady-time tradeoff"},
    )


def _plot_runtime_panel(
    axis: plt.Axes,
    units: Sequence[Mapping[str, Any]],
) -> None:
    ordered = sorted(
        units,
        key=lambda unit: (
            str(unit["precision"]),
            str(unit["lattice"]),
            str(unit["trace"]),
        ),
    )
    labels = []
    pyqcu_values = []
    quda_values = []
    ratios = []
    for unit in ordered:
        pyqcu = steady_seconds(unit, "pyqcu")
        quda = steady_seconds(unit, "quda")
        labels.append(f"{unit['precision']} {unit['lattice']}")
        pyqcu_values.append(pyqcu)
        quda_values.append(quda)
        ratios.append(quda / pyqcu)

    positions = list(range(len(ordered)))
    axis.barh(
        [position - 0.19 for position in positions],
        pyqcu_values,
        height=0.34,
        color=SIDE_COLORS["pyqcu"],
    )
    axis.barh(
        [position + 0.19 for position in positions],
        quda_values,
        height=0.34,
        color=SIDE_COLORS["quda"],
    )
    for position, pyqcu, quda, ratio in zip(
        positions, pyqcu_values, quda_values, ratios
    ):
        axis.text(
            pyqcu,
            position - 0.19,
            f" {_format_seconds(pyqcu)}",
            va="center",
            ha="left",
            fontsize=6.5,
        )
        axis.text(
            quda,
            position + 0.19,
            f" {_format_seconds(quda)}",
            va="center",
            ha="left",
            fontsize=6.5,
        )
        axis.text(
            0.98,
            position,
            _format_ratio(ratio),
            transform=axis.get_yaxis_transform(),
            va="center",
            ha="right",
            fontsize=6.5,
            color="#333333",
        )
    axis.set_yticks(positions)
    axis.set_yticklabels(labels, fontsize=7)
    axis.invert_yaxis()
    axis.set_xscale("log")
    axis.set_xlabel("steady solve time (s, log scale)")
    axis.grid(True, axis="x", which="both", alpha=0.2, linewidth=0.5)


def plot_runtime_traceoff(
    units: Sequence[Mapping[str, Any]], outdir: Path
) -> None:
    selected = [
        unit
        for unit in units
        if unit["trace"] == "off" and int(unit["levels"]) in (2, 3)
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 10.5))
    panels = (
        ("v100", 2),
        ("v100", 3),
        ("p100", 2),
        ("p100", 3),
    )
    for axis, (device, levels) in zip(axes.flat, panels):
        panel_units = [
            unit
            for unit in selected
            if unit["device"] == device and int(unit["levels"]) == levels
        ]
        _plot_runtime_panel(axis, panel_units)
        axis.set_title(
            f"{device.upper()} MG-{levels} trace-off "
            f"(n={len(panel_units)})",
            fontsize=11,
        )
    fig.suptitle(
        "Absolute trace-off MultiGrid steady solve times",
        fontsize=13,
        y=0.99,
    )
    _add_pair_legend(fig, y=0.01)
    fig.subplots_adjust(hspace=0.33, wspace=0.28, bottom=0.10, top=0.95)
    _save(
        fig,
        outdir,
        "mg_runtime_traceoff",
        metadata={"Title": "Absolute trace-off MG steady times"},
    )


def plot_reference_runtime(
    units: Sequence[Mapping[str, Any]], outdir: Path
) -> None:
    selected = [
        unit
        for unit in units
        if unit["trace"] == "off" and int(unit["levels"]) == 1
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4))
    for axis, device in zip(axes, ("v100", "p100")):
        panel_units = [unit for unit in selected if unit["device"] == device]
        _plot_runtime_panel(axis, panel_units)
        axis.set_title(
            f"{device.upper()} BiCGStab trace-off (n={len(panel_units)})",
            fontsize=11,
        )
    fig.suptitle(
        "Absolute BiCGStab reference steady solve times",
        fontsize=13,
        y=0.99,
    )
    _add_pair_legend(fig, y=0.01)
    fig.subplots_adjust(wspace=0.33, bottom=0.18, top=0.87)
    _save(
        fig,
        outdir,
        "mg_reference_runtime",
        metadata={"Title": "Absolute BiCGStab steady times"},
    )


def _stage_page(
    units: Sequence[Mapping[str, Any]],
    device: str,
    levels: int,
    outdir: Path,
) -> None:
    selected = [
        unit
        for unit in units
        if unit["trace"] == "on"
        and unit["device"] == device
        and int(unit["levels"]) == levels
    ]
    selected.sort(key=lambda unit: (str(unit["precision"]), str(unit["lattice"])))
    fig, axes = plt.subplots(3, 2, figsize=(13.0, 14.0), squeeze=False)
    for axis, unit in zip(axes.flat, selected):
        display_rows: list[tuple[str, dict[str, float], bool]] = []
        for side in ("pyqcu", "quda"):
            rows = stage_rows(unit, side)
            for index, row in enumerate(rows):
                is_coarsest = index == len(rows) - 1
                label = f"{'P' if side == 'pyqcu' else 'Q'} L{row['level']}"
                if is_coarsest:
                    label += "/coarsest"
                display_rows.append(
                    (label, row["seconds"], is_coarsest)
                )
        positions = list(range(len(display_rows)))
        for position, (_, values, _) in zip(positions, display_rows):
            left = 0.0
            for field, name in STAGE_FIELDS:
                value = float(values[name])
                if value <= 0.0:
                    continue
                axis.barh(
                    position,
                    value,
                    left=left,
                    color=STAGE_COLORS[name],
                    height=0.68,
                )
                left += value
            axis.text(
                left,
                position,
                f"  {_format_seconds(left)}",
                va="center",
                ha="left",
                fontsize=6.0,
            )
        axis.set_yticks(positions)
        axis.set_yticklabels(
            [label for label, _, _ in display_rows],
            fontsize=7,
        )
        axis.invert_yaxis()
        axis.set_xlabel("nested stage time (s)", fontsize=8)
        axis.set_title(
            f"{unit['device'].upper()} {unit['precision']} "
            f"{unit['lattice']} MG-{unit['levels']} trace-on",
            fontsize=10,
        )
        axis.grid(True, axis="x", alpha=0.2, linewidth=0.5)
    for axis in axes.flat[len(selected):]:
        axis.axis("off")
    handles = [
        Patch(facecolor=STAGE_COLORS[name], label=name)
        for _, name in STAGE_FIELDS
    ]
    fig.legend(
        handles=handles,
        labels=[name for _, name in STAGE_FIELDS],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=6,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        "Per-level absolute stage times; /coarsest marks the last active "
        "level. Level totals are nested and must not be added across rows.",
        fontsize=12,
        y=0.995,
    )
    fig.subplots_adjust(
        top=0.96,
        bottom=0.08,
        left=0.06,
        right=0.98,
        hspace=0.30,
        wspace=0.25,
    )
    _save(
        fig,
        outdir,
        f"mg_stage_{device}_mg{levels}",
        metadata={"Title": f"Absolute MG stage times, {device} MG-{levels}"},
    )


def plot_stage_pages(
    units: Sequence[Mapping[str, Any]], outdir: Path
) -> None:
    for device in ("v100", "p100"):
        for levels in (2, 3):
            _stage_page(units, device, levels, outdir)


def _steady_residual_curve(
    unit: Mapping[str, Any], side: str
) -> list[tuple[float, float]]:
    side_document = _active_side(unit, side)
    by_phase = side_document.get("mg_levels_by_phase") or {}
    steady_levels = by_phase.get("steady") or []
    if not steady_levels:
        return []
    sequence = steady_levels[0].get("residual_sequence") or []
    by_solve: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for point in sequence:
        by_solve[int(point.get("solve_index", 0))].append(point)
    if not by_solve:
        return []
    solve_index = max(by_solve)
    points: dict[int, tuple[float, float, int]] = {}
    for order, point in enumerate(by_solve[solve_index]):
        iteration = int(point.get("iteration", -1))
        relative = _float(point.get("relative"))
        if iteration < 0 or relative <= 0.0:
            continue
        kind = str(point.get("kind", ""))
        priority = 1 if "true" in kind else 0
        previous = points.get(iteration)
        if previous is None or priority >= previous[2]:
            points[iteration] = (float(iteration), relative, priority)
    ordered = sorted(points.values())
    if not ordered:
        return []
    first = ordered[0][1]
    return [(iteration, relative / first) for iteration, relative, _ in ordered]


def plot_residual_overlay(
    units: Sequence[Mapping[str, Any]], outdir: Path
) -> None:
    fig, axis = plt.subplots(figsize=(10.0, 10.5))
    styles = {
        ("pyqcu", 2): ("#245ba3", "-"),
        ("pyqcu", 3): ("#245ba3", "--"),
        ("quda", 2): ("#be4137", "-"),
        ("quda", 3): ("#be4137", "--"),
    }
    plotted = 0
    for unit in units:
        if unit["trace"] != "on" or int(unit["levels"]) not in (2, 3):
            continue
        for side in ("pyqcu", "quda"):
            curve = _steady_residual_curve(unit, side)
            if not curve:
                continue
            color, linestyle = styles[(side, int(unit["levels"]))]
            axis.plot(
                [point[0] for point in curve],
                [point[1] for point in curve],
                color=color,
                linestyle=linestyle,
                linewidth=0.75,
                alpha=0.42,
            )
            plotted += 1
    axis.set_yscale("log")
    axis.set_xlabel("outer iteration")
    axis.set_ylabel("relative residual")
    axis.set_title(
        "Steady-state finest-level residual curves "
        f"({plotted} side/case curves)"
    )
    axis.grid(True, which="both", alpha=0.18, linewidth=0.5)
    handles = [
        plt.Line2D([], [], color=styles[key][0], linestyle=styles[key][1])
        for key in (
            ("pyqcu", 2),
            ("pyqcu", 3),
            ("quda", 2),
            ("quda", 3),
        )
    ]
    labels = [
        "PyQCU MG-2",
        "PyQCU MG-3",
        "QUDA MG-2",
        "QUDA MG-3",
    ]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        frameon=False,
        fontsize=10,
    )
    fig.subplots_adjust(bottom=0.16)
    _save(
        fig,
        outdir,
        "mg_finest_residual",
        metadata={"Title": "Finest residual curves, legend below plot"},
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _unit_time_rows(
    units: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "unit_id": unit["path"].stem,
            "device": unit["device"],
            "precision": unit["precision"],
            "lattice": unit["lattice"],
            "levels": unit["levels"],
            "solver": unit["solver"],
            "trace": unit["trace"],
            "pyqcu_steady_seconds": steady_seconds(unit, "pyqcu"),
            "quda_steady_seconds": steady_seconds(unit, "quda"),
            "ratio_quda_over_pyqcu": (
                steady_seconds(unit, "quda") / steady_seconds(unit, "pyqcu")
            ),
            "pyqcu_outer_iterations": steady_outer_iterations(unit, "pyqcu"),
            "quda_outer_iterations": steady_outer_iterations(unit, "quda"),
            "pyqcu_true_residual_max": final_true_residual(unit, "pyqcu"),
            "quda_true_residual_max": final_true_residual(unit, "quda"),
        }
        for unit in units
    ]


def _group_time_rows(
    units: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for unit in units:
        grouped[
            (
                str(unit["device"]),
                str(unit["precision"]),
                str(unit["solver"]),
                str(unit["trace"]),
            )
        ].append(unit)
    rows = []
    for key, selected in sorted(grouped.items()):
        device, precision, solver, trace = key
        pyqcu = _median(steady_seconds(unit, "pyqcu") for unit in selected)
        quda = _median(steady_seconds(unit, "quda") for unit in selected)
        ratios = [
            steady_seconds(unit, "quda") / steady_seconds(unit, "pyqcu")
            for unit in selected
        ]
        rows.append(
            {
                "device": device,
                "precision": precision,
                "solver": solver,
                "trace": trace,
                "samples": len(selected),
                "pyqcu_median_seconds": pyqcu,
                "quda_median_seconds": quda,
                "ratio_median": _median(ratios),
                "ratio_min": min(ratios),
                "ratio_max": max(ratios),
            }
        )
    return rows


def _stage_rows_csv(
    units: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for unit in units:
        if int(unit["levels"]) not in (2, 3):
            continue
        for side in ("pyqcu", "quda"):
            for row in stage_rows(unit, side):
                output = {
                    "unit_id": unit["path"].stem,
                    "device": unit["device"],
                    "precision": unit["precision"],
                    "lattice": unit["lattice"],
                    "levels": unit["levels"],
                    "trace": unit["trace"],
                    "side": side,
                    "level": row["level"],
                }
                output.update(
                    {
                        f"{name}_seconds": row["seconds"][name]
                        for _, name in STAGE_FIELDS
                    }
                )
                output["summed_stage_seconds"] = row["total_seconds"]
                rows.append(output)
    return rows


def _coarsest_rows(
    units: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for unit in units:
        if unit["trace"] != "on" or int(unit["levels"]) not in (2, 3):
            continue
        grouped[
            (
                str(unit["device"]),
                str(unit["precision"]),
                int(unit["levels"]),
            )
        ].append(unit)
    rows = []
    for (device, precision, levels), selected in sorted(grouped.items()):
        pyqcu = _median(
            stage_rows(unit, "pyqcu")[-1]["seconds"]["coarse"]
            for unit in selected
        )
        quda = _median(
            stage_rows(unit, "quda")[-1]["seconds"]["coarse"]
            for unit in selected
        )
        rows.append(
            {
                "device": device,
                "precision": precision,
                "levels": levels,
                "samples": len(selected),
                "pyqcu_coarsest_seconds": pyqcu,
                "quda_coarsest_seconds": quda,
                "ratio_quda_over_pyqcu": quda / pyqcu,
            }
        )
    return rows


def _stage_component_rows(
    units: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str, int], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for unit in units:
        if unit["trace"] != "on" or int(unit["levels"]) not in (2, 3):
            continue
        for side in ("pyqcu", "quda"):
            for row in stage_rows(unit, side):
                grouped[
                    (
                        str(unit["device"]),
                        str(unit["precision"]),
                        int(unit["levels"]),
                        side,
                        int(row["level"]),
                    )
                ].append(row)
    rows = []
    for key, selected in sorted(grouped.items()):
        device, precision, levels, side, level = key
        rows.append(
            {
                "device": device,
                "precision": precision,
                "levels": levels,
                "side": side,
                "level": level,
                "samples": len(selected),
                "pre_seconds": _median(
                    row["seconds"]["pre"] for row in selected
                ),
                "post_seconds": _median(
                    row["seconds"]["post"] for row in selected
                ),
                "restrict_seconds": _median(
                    row["seconds"]["restrict"] for row in selected
                ),
                "prolong_seconds": _median(
                    row["seconds"]["prolong"] for row in selected
                ),
                "coarse_seconds": _median(
                    row["seconds"]["coarse"] for row in selected
                ),
                "other_seconds": _median(
                    row["seconds"]["other"] for row in selected
                ),
            }
        )
    return rows


def _ma(labels: Sequence[str]) -> str:
    return " & ".join(labels) + r" \\"


def write_latex_tables(
    group_rows: Sequence[Mapping[str, Any]],
    coarsest_rows: Sequence[Mapping[str, Any]],
    stage_component_rows: Sequence[Mapping[str, Any]],
    outdir: Path,
) -> None:
    group_lines = [
        r"\begin{longtable}{@{}lllrrrrr@{}}",
        r"\caption{分组绝对 steady 耗时与时间比（trace-on/off 分开统计）}"
        r"\label{tab:group-times}\\",
        r"\toprule",
        _ma(
            (
                "设备",
                "精度",
                "求解",
                "trace",
                r"$n$",
                "PyQCU 中位 s",
                "QUDA 中位 s",
                r"$R$ 中位",
            )
        ),
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        _ma(
            (
                "设备",
                "精度",
                "求解",
                "trace",
                r"$n$",
                "PyQCU 中位 s",
                "QUDA 中位 s",
                r"$R$ 中位",
            )
        ),
        r"\midrule",
        r"\endhead",
    ]
    for row in group_rows:
        group_lines.append(
            _ma(
                (
                    str(row["device"]).upper(),
                    str(row["precision"]),
                    str(row["solver"]).upper(),
                    str(row["trace"]),
                    str(row["samples"]),
                    f"{_float(row['pyqcu_median_seconds']):.3f}",
                    f"{_float(row['quda_median_seconds']):.3f}",
                    f"{_float(row['ratio_median']):.4f}",
                )
            )
        )
    group_lines.extend((r"\bottomrule", r"\end{longtable}"))
    (outdir / "group_times.tex").write_text(
        "\n".join(group_lines) + "\n", encoding="utf-8"
    )

    coarsest_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{最粗层 solve 的绝对中位耗时；QUDA 的最后有效层已显式取回}"
        r"\label{tab:coarsest-times}",
        r"\small",
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        _ma(
            (
                "设备",
                "精度/层数",
                r"$n$",
                "PyQCU s",
                "QUDA s",
                r"$R=Q/P$",
            )
        ),
        r"\midrule",
    ]
    for row in coarsest_rows:
        coarsest_lines.append(
            _ma(
                (
                    str(row["device"]).upper(),
                    f"{row['precision']} / MG-{row['levels']}",
                    str(row["samples"]),
                    f"{_float(row['pyqcu_coarsest_seconds']):.3f}",
                    f"{_float(row['quda_coarsest_seconds']):.3f}",
                    f"{_float(row['ratio_quda_over_pyqcu']):.4f}",
                )
            )
        )
    coarsest_lines.extend(
        (r"\bottomrule", r"\end{tabular}", r"\end{table}")
    )
    (outdir / "coarsest_times.tex").write_text(
        "\n".join(coarsest_lines) + "\n", encoding="utf-8"
    )

    stage_lines = [
        r"\begin{longtable}{@{}lllrrrrrrrrrr@{}}",
        r"\caption{MG level component medians (seconds; trace-on, nested)}"
        r"\label{tab:stage-components}\\",
        r"\toprule",
        _ma(
            (
                "Device",
                "Prec",
                "MG",
                "Side",
                "L",
                r"$n$",
                "Pre",
                "Post",
                "Restrict",
                "Prolong",
                "Coarse",
                "Other",
                "Total",
            )
        ),
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        _ma(
            (
                "Device",
                "Prec",
                "MG",
                "Side",
                "L",
                r"$n$",
                "Pre",
                "Post",
                "Restrict",
                "Prolong",
                "Coarse",
                "Other",
                "Total",
            )
        ),
        r"\midrule",
        r"\endhead",
    ]
    for row in stage_component_rows:
        total = sum(
            _float(row[name])
            for name in (
                "pre_seconds",
                "post_seconds",
                "restrict_seconds",
                "prolong_seconds",
                "coarse_seconds",
                "other_seconds",
            )
        )
        stage_lines.append(
            _ma(
                (
                    str(row["device"]).upper(),
                    str(row["precision"]),
                    f"MG-{row['levels']}",
                    SIDE_LABELS[str(row["side"])],
                    f"L{row['level']}",
                    str(row["samples"]),
                    f"{_float(row['pre_seconds']):.3f}",
                    f"{_float(row['post_seconds']):.3f}",
                    f"{_float(row['restrict_seconds']):.3f}",
                    f"{_float(row['prolong_seconds']):.3f}",
                    f"{_float(row['coarse_seconds']):.3f}",
                    f"{_float(row['other_seconds']):.3f}",
                    f"{total:.3f}",
                )
            )
        )
    stage_lines.extend((r"\bottomrule", r"\end{longtable}"))
    (outdir / "stage_components.tex").write_text(
        "\n".join(stage_lines) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_source_manifest(
    units: Sequence[Mapping[str, Any]],
    input_dir: Path,
    source_tag: str,
    source_commit: str,
    outdir: Path,
) -> None:
    source_dir = input_dir.parent
    source_files = {
        "units_csv": source_dir / "units.csv",
        "stages_csv": source_dir / "stages.csv",
    }
    manifest = {
        "schema": "pyqcu.mg-absolute-report-source/v1",
        "source_tag": source_tag,
        "source_commit": source_commit,
        "combined_records": len(units),
        "combined_paths": [str(unit["path"]) for unit in units],
        "source_sha256": {
            name: _sha256(path)
            for name, path in source_files.items()
            if path.exists()
        },
    }
    (outdir / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build(
    input_dir: Path,
    outdir: Path,
    source_tag: str,
    source_commit: str,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    units = load_units(input_dir)
    unit_rows = _unit_time_rows(units)
    group_rows = _group_time_rows(units)
    stage_rows_output = _stage_rows_csv(units)
    coarsest_rows = _coarsest_rows(units)
    stage_component_rows = _stage_component_rows(units)
    _write_csv(outdir / "unit_times.csv", unit_rows)
    _write_csv(outdir / "group_times.csv", group_rows)
    _write_csv(outdir / "stage_times.csv", stage_rows_output)
    _write_csv(outdir / "coarsest_times.csv", coarsest_rows)
    _write_csv(
        outdir / "stage_component_medians.csv", stage_component_rows
    )
    write_latex_tables(
        group_rows, coarsest_rows, stage_component_rows, outdir
    )
    plot_time_tradeoff(units, outdir)
    plot_runtime_traceoff(units, outdir)
    plot_reference_runtime(units, outdir)
    plot_stage_pages(units, outdir)
    plot_residual_overlay(units, outdir)
    write_source_manifest(
        units, input_dir, source_tag, source_commit, outdir
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build absolute-time MG report figures from combined JSON"
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--source-tag", default="test27")
    parser.add_argument("--source-commit", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    build(args.input_dir, args.outdir, args.source_tag, args.source_commit)
    print((args.outdir / "group_times.csv").resolve())
    print((args.outdir / "mg_time_tradeoff.pdf").resolve())
    print((args.outdir / "mg_stage_v100_mg3.pdf").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
