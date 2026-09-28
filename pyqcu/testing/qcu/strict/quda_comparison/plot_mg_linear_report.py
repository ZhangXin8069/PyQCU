#!/usr/bin/env python3
"""Build the linear-axis MG comparison report from test27 combined JSON."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import plot_mg_absolute_times as absolute


STAGE_COLORS = absolute.STAGE_COLORS
STAGE_FIELDS = absolute.STAGE_FIELDS
SIDE_COLORS = absolute.SIDE_COLORS
SIDE_LABELS = absolute.SIDE_LABELS


def _protocol(unit: Mapping[str, Any]) -> Mapping[str, Any]:
    return unit["document"].get("protocol") or {}


def _mass(unit: Mapping[str, Any]) -> float:
    return absolute._float(_protocol(unit).get("mass"))


def _kappa(unit: Mapping[str, Any]) -> float:
    return absolute._float(_protocol(unit).get("kappa"))


def _block(unit: Mapping[str, Any]) -> tuple[int, int, int, int]:
    block = _protocol(unit).get("block_xyzt") or [2, 2, 2, 2]
    return tuple(int(value) for value in block)


def _nvec(unit: Mapping[str, Any]) -> int:
    return int(_protocol(unit).get("nvec") or 0)


def _lattice(unit: Mapping[str, Any]) -> tuple[int, int, int, int]:
    lattice = _protocol(unit).get("lattice_xyzt") or []
    return tuple(int(value) for value in lattice)


def _metadata_label(units: Sequence[Mapping[str, Any]]) -> str:
    if not units:
        return ""
    reference = units[0]
    mass = _mass(reference)
    kappa = _kappa(reference)
    block = _block(reference)
    nvec = _nvec(reference)
    block_text = "x".join(str(value) for value in block)
    return (
        f"mass={mass:.3f}, kappa={kappa:.6f}, "
        f"block={block_text}, nvec={nvec}"
    )


def _unit_case_label(unit: Mapping[str, Any]) -> str:
    lattice = "x".join(str(value) for value in _lattice(unit))
    return (
        f"{unit['precision']} {lattice} {unit['solver'].upper()} "
        f"{unit['trace']} | mass={_mass(unit):.3f}"
    )


def _panel_title(
    units: Sequence[Mapping[str, Any]],
    prefix: str,
) -> str:
    if not units:
        return prefix
    return f"{prefix}\n{_metadata_label(units)}"


def _save(
    fig: plt.Figure,
    outdir: Path,
    stem: str,
    title: str,
) -> None:
    absolute._save(
        fig,
        outdir,
        stem,
        metadata={"Title": title},
    )


def _paired_legend(fig: plt.Figure, y: float = 0.01) -> None:
    absolute._add_pair_legend(fig, y)


def _linear_limits(values: Sequence[float]) -> tuple[float, float]:
    upper = max(values) if values else 1.0
    if upper <= 0.0:
        upper = 1.0
    return 0.0, upper * 1.32


def _paired_runtime_axis(
    axis: plt.Axes,
    units: Sequence[Mapping[str, Any]],
    *,
    show_case_metadata: bool,
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
        pyqcu = absolute.steady_seconds(unit, "pyqcu")
        quda = absolute.steady_seconds(unit, "quda")
        lattice = "x".join(str(value) for value in _lattice(unit))
        label = (
            f"{unit['precision']} {lattice} {unit['trace']}\n"
            f"R={quda / pyqcu:.3f}x"
        )
        if show_case_metadata:
            label += f"; mass={_mass(unit):.3f}"
        labels.append(label)
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
    lower, upper = _linear_limits(pyqcu_values + quda_values)
    for position, pyqcu, quda in zip(
        positions, pyqcu_values, quda_values
    ):
        axis.text(
            min(pyqcu, upper * 0.985),
            position - 0.19,
            f" {absolute._format_seconds(pyqcu)}",
            va="center",
            ha="left",
            fontsize=6.5,
        )
        axis.text(
            min(quda, upper * 0.985),
            position + 0.19,
            f" {absolute._format_seconds(quda)}",
            va="center",
            ha="left",
            fontsize=6.5,
        )
    axis.set_xlim(lower, upper)
    axis.set_yticks(positions)
    axis.set_yticklabels(labels, fontsize=6.8)
    axis.invert_yaxis()
    axis.set_xlabel("steady solve time (s, linear scale)", fontsize=8.5)
    axis.grid(True, axis="x", alpha=0.20, linewidth=0.5)


def plot_group_ratio_linear(
    units: Sequence[Mapping[str, Any]], outdir: Path
) -> None:
    rows = absolute._group_time_rows(units)
    rows.sort(key=lambda row: float(row["ratio_median"]))
    labels = [
        f"{str(row['device']).upper()} {row['precision']} "
        f"{str(row['solver']).upper()} {row['trace']}"
        for row in rows
    ]
    medians = [absolute._float(row["ratio_median"]) for row in rows]
    lows = [absolute._float(row["ratio_min"]) for row in rows]
    highs = [absolute._float(row["ratio_max"]) for row in rows]
    errors = [
        [median - low for median, low in zip(medians, lows)],
        [high - median for median, high in zip(medians, highs)],
    ]
    fig, axis = plt.subplots(figsize=(12.5, 11.0))
    positions = list(range(len(rows)))
    axis.barh(
        positions,
        medians,
        height=0.68,
        color=[
            SIDE_COLORS["pyqcu"] if ratio >= 1.0 else SIDE_COLORS["quda"]
            for ratio in medians
        ],
        xerr=errors,
        error_kw={"ecolor": "#444444", "elinewidth": 0.8, "capsize": 2},
    )
    axis.axvline(1.0, color="black", linestyle="--", linewidth=0.9)
    axis.set_xlim(0.0, max(highs) * 1.16)
    axis.set_yticks(positions)
    axis.set_yticklabels(labels, fontsize=7.5)
    axis.invert_yaxis()
    axis.set_xlabel("R = QUDA / PyQCU steady median (linear scale)")
    axis.set_title(
        "Group median time ratio with min-max range\n"
        f"{_metadata_label(units)}",
        fontsize=12,
    )
    axis.grid(True, axis="x", alpha=0.2, linewidth=0.5)
    for position, median in zip(positions, medians):
        axis.text(
            median,
            position,
            f"  {median:.3f}x",
            va="center",
            ha="left",
            fontsize=6.8,
        )
    fig.tight_layout()
    _save(fig, outdir, "mg_group_ratio_linear", "Linear group time ratio")


def plot_absolute_time_matrix_page(
    units: Sequence[Mapping[str, Any]],
    device: str,
    outdir: Path,
) -> None:
    selected = [unit for unit in units if unit["device"] == device]
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 14.5))
    solvers = (("bicgstab", "BiCGStab"), ("mg-2", "MG-2"), ("mg-3", "MG-3"))
    for axis, (solver, label) in zip(axes, solvers):
        panel_units = [unit for unit in selected if unit["solver"] == solver]
        _paired_runtime_axis(
            axis,
            panel_units,
            show_case_metadata=True,
        )
        axis.set_title(
            _panel_title(
                panel_units,
                f"{device.upper()} absolute {label} steady times",
            ),
            fontsize=10.5,
        )
    _paired_legend(fig, y=0.005)
    fig.subplots_adjust(
        top=0.965,
        bottom=0.055,
        left=0.30,
        right=0.985,
        hspace=0.34,
    )
    _save(
        fig,
        outdir,
        f"mg_absolute_times_{device}",
        f"{device} absolute time matrix",
    )


def plot_runtime_traceoff_linear(
    units: Sequence[Mapping[str, Any]], outdir: Path
) -> None:
    selected = [
        unit
        for unit in units
        if unit["trace"] == "off" and int(unit["levels"]) in (2, 3)
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 12.0))
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
        _paired_runtime_axis(
            axis,
            panel_units,
            show_case_metadata=True,
        )
        axis.set_title(
            _panel_title(
                panel_units,
                f"{device.upper()} MG-{levels} trace-off absolute times",
            ),
            fontsize=10,
        )
    _paired_legend(fig, y=0.005)
    fig.subplots_adjust(
        top=0.955,
        bottom=0.075,
        left=0.17,
        right=0.985,
        hspace=0.34,
        wspace=0.42,
    )
    _save(
        fig,
        outdir,
        "mg_runtime_traceoff_linear",
        "Linear trace-off MG runtimes",
    )


def plot_reference_runtime_linear(
    units: Sequence[Mapping[str, Any]], outdir: Path
) -> None:
    selected = [
        unit
        for unit in units
        if unit["trace"] == "off" and int(unit["levels"]) == 1
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 7.0))
    for axis, device in zip(axes, ("v100", "p100")):
        panel_units = [unit for unit in selected if unit["device"] == device]
        _paired_runtime_axis(
            axis,
            panel_units,
            show_case_metadata=True,
        )
        axis.set_title(
            _panel_title(
                panel_units,
                f"{device.upper()} BiCGStab trace-off absolute times",
            ),
            fontsize=10.5,
        )
    _paired_legend(fig, y=0.005)
    fig.subplots_adjust(
        top=0.84,
        bottom=0.12,
        left=0.15,
        right=0.985,
        wspace=0.48,
    )
    _save(
        fig,
        outdir,
        "mg_reference_runtime_linear",
        "Linear BiCGStab runtimes",
    )


def _stage_page_linear(
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
    fig, axes = plt.subplots(3, 2, figsize=(14.0, 14.5), squeeze=False)
    for axis, unit in zip(axes.flat, selected):
        display_rows = []
        for side in ("pyqcu", "quda"):
            rows = absolute.stage_rows(unit, side)
            for index, row in enumerate(rows):
                label = f"{'P' if side == 'pyqcu' else 'Q'} L{row['level']}"
                if index == len(rows) - 1:
                    label += "/coarsest"
                display_rows.append((label, row["seconds"]))
        positions = list(range(len(display_rows)))
        for position, (_, values) in zip(positions, display_rows):
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
                f"  {absolute._format_seconds(left)}",
                va="center",
                ha="left",
                fontsize=6.0,
            )
        axis.set_yticks(positions)
        axis.set_yticklabels(
            [label for label, _ in display_rows],
            fontsize=7,
        )
        axis.invert_yaxis()
        axis.set_xlabel("nested stage time (s, linear scale)", fontsize=8)
        axis.set_title(
            f"{unit['device'].upper()} {unit['precision']} "
            f"{unit['lattice']} MG-{unit['levels']} trace-on\n"
            f"mass={_mass(unit):.3f}, kappa={_kappa(unit):.6f}, "
            f"block={'x'.join(str(v) for v in _block(unit))}, "
            f"nvec={_nvec(unit)}",
            fontsize=9.2,
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
        "Per-level absolute stage times, linear axes; /coarsest is the last "
        "active level and level totals are nested.",
        fontsize=12,
        y=0.995,
    )
    fig.subplots_adjust(
        top=0.955,
        bottom=0.075,
        left=0.06,
        right=0.985,
        hspace=0.34,
        wspace=0.26,
    )
    _save(
        fig,
        outdir,
        f"mg_stage_linear_{device}_mg{levels}",
        f"Linear stage times {device} MG-{levels}",
    )


def plot_stage_pages_linear(
    units: Sequence[Mapping[str, Any]], outdir: Path
) -> None:
    for device in ("v100", "p100"):
        for levels in (2, 3):
            _stage_page_linear(units, device, levels, outdir)


def _memory_rows(memory_csv: Path) -> list[dict[str, Any]]:
    rows = []
    with memory_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            peak = absolute._float(row.get("steady_sampler_peak_bytes"))
            if peak <= 0.0:
                continue
            case_id = str(row["case_id"])
            device = "v100" if case_id.startswith("v100__") else "p100"
            rows.append(
                {
                    **row,
                    "device_key": device,
                    "levels_int": int(row["levels"]),
                    "peak_mib": peak / (1 << 20),
                }
            )
    return rows


def plot_peak_memory_linear(
    memory_csv: Path,
    outdir: Path,
) -> None:
    rows = _memory_rows(memory_csv)
    grouped: dict[tuple[str, str, int, str, str], list[float]] = defaultdict(
        list
    )
    for row in rows:
        key = (
            row["device_key"],
            row["precision"],
            row["levels_int"],
            row["trace"],
            row["side"],
        )
        grouped[key].append(float(row["peak_mib"]))
    keys = sorted({key[:4] for key in grouped})
    fig, axes = plt.subplots(2, 1, figsize=(15.0, 12.0))
    for axis, device in zip(axes, ("p100", "v100")):
        device_keys = [key for key in keys if key[0] == device]
        positions = list(range(len(device_keys)))
        width = 0.38
        maxima = []
        for side, offset in (("pyqcu", -width / 2), ("quda", width / 2)):
            values = []
            for key in device_keys:
                samples = grouped.get((*key, side), [])
                values.append(absolute._median(samples) if samples else 0.0)
            maxima.extend(values)
            axis.bar(
                [position + offset for position in positions],
                values,
                width=width * 0.9,
                color=SIDE_COLORS[side],
                label=SIDE_LABELS[side],
            )
            for position, value in zip(positions, values):
                axis.text(
                    position + offset,
                    value,
                    f"{value:.0f}",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=5.8,
                )
        axis.set_ylim(0.0, max(maxima) * 1.22 if maxima else 1.0)
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [
                f"{precision} L{levels} {trace}"
                for _, precision, levels, trace in device_keys
            ],
            rotation=35,
            ha="right",
            fontsize=7.5,
        )
        axis.set_ylabel("median device-wide peak (MiB, linear scale)")
        axis.set_title(
            f"{device.upper()} peak memory; mass=0.050, kappa=0.123457, "
            "block=2x2x2x2, nvec=12\n"
            "lattice set: P100={8x8x8x16,16x16x16x16,16x16x32x32(c128)}, "
            "V100={8x8x8x16,16x16x16x16,16x16x32x32,16x32x32x48(c64)}",
            fontsize=10,
        )
        axis.grid(True, axis="y", alpha=0.2, linewidth=0.5)
        axis.legend(fontsize=8)
    fig.subplots_adjust(
        top=0.925,
        bottom=0.08,
        left=0.09,
        right=0.99,
        hspace=0.52,
    )
    _save(fig, outdir, "mg_peak_memory_linear", "Linear peak memory")


def _residual_panel(
    axis: plt.Axes,
    unit: Mapping[str, Any],
) -> None:
    for side in ("pyqcu", "quda"):
        curve = absolute._steady_residual_curve(unit, side)
        axis.plot(
            [point[0] for point in curve],
            [point[1] for point in curve],
            color=SIDE_COLORS[side],
            linewidth=1.1,
            label=SIDE_LABELS[side],
        )
    axis.set_yscale("log")
    axis.set_xlabel("outer iteration", fontsize=8)
    axis.set_ylabel("relative residual", fontsize=8)
    axis.set_title(
        f"{unit['device'].upper()} {unit['precision']} {unit['lattice']} "
        f"MG-{unit['levels']} trace-on\n"
        f"mass={_mass(unit):.3f}, block=2x2x2x2, nvec={_nvec(unit)}",
        fontsize=9,
    )
    axis.grid(True, which="both", alpha=0.18, linewidth=0.5)
    axis.legend(fontsize=7)


def _residual_page(
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
        _residual_panel(axis, unit)
    for axis in axes.flat[len(selected):]:
        axis.axis("off")
    fig.suptitle(
        f"Finest residual curves with logarithmic residual norm only: "
        f"{device.upper()} MG-{levels}",
        fontsize=12,
        y=0.995,
    )
    fig.subplots_adjust(
        top=0.955,
        bottom=0.055,
        left=0.075,
        right=0.985,
        hspace=0.38,
        wspace=0.26,
    )
    _save(
        fig,
        outdir,
        f"mg_residual_linear_{device}_mg{levels}",
        f"Residual panels {device} MG-{levels}",
    )


def plot_residual_pages(
    units: Sequence[Mapping[str, Any]], outdir: Path
) -> None:
    for device in ("v100", "p100"):
        for levels in (2, 3):
            _residual_page(units, device, levels, outdir)


def plot_residual_overlay_linear(
    units: Sequence[Mapping[str, Any]], outdir: Path
) -> None:
    fig, axis = plt.subplots(figsize=(10.5, 10.5))
    styles = {
        ("pyqcu", 2): (SIDE_COLORS["pyqcu"], "-"),
        ("pyqcu", 3): (SIDE_COLORS["pyqcu"], "--"),
        ("quda", 2): (SIDE_COLORS["quda"], "-"),
        ("quda", 3): (SIDE_COLORS["quda"], "--"),
    }
    for unit in units:
        if unit["trace"] != "on" or int(unit["levels"]) not in (2, 3):
            continue
        for side in ("pyqcu", "quda"):
            curve = absolute._steady_residual_curve(unit, side)
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
    axis.set_yscale("log")
    axis.set_xlabel("outer iteration")
    axis.set_ylabel("relative residual")
    axis.set_title(
        "Steady finest-level residual curves\n"
        f"{_metadata_label(units)}; only the residual norm uses a log axis",
        fontsize=11,
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
    fig.legend(
        handles,
        ("PyQCU MG-2", "PyQCU MG-3", "QUDA MG-2", "QUDA MG-3"),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        frameon=False,
        fontsize=9.5,
    )
    fig.subplots_adjust(bottom=0.16)
    _save(
        fig,
        outdir,
        "mg_finest_residual_linear",
        "Residual overlay with log residual norm",
    )


def _write_source_manifest(
    units: Sequence[Mapping[str, Any]],
    input_dir: Path,
    outdir: Path,
    source_commit: str,
) -> None:
    absolute.write_source_manifest(
        units,
        input_dir,
        "test27",
        source_commit,
        outdir,
    )


def build(
    input_dir: Path,
    memory_csv: Path,
    outdir: Path,
    source_commit: str,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    units = absolute.load_units(input_dir)
    absolute._write_csv(
        outdir / "unit_times.csv",
        absolute._unit_time_rows(units),
    )
    absolute._write_csv(
        outdir / "group_times.csv",
        absolute._group_time_rows(units),
    )
    absolute._write_csv(
        outdir / "stage_times.csv",
        absolute._stage_rows_csv(units),
    )
    absolute._write_csv(
        outdir / "coarsest_times.csv",
        absolute._coarsest_rows(units),
    )
    absolute._write_csv(
        outdir / "stage_component_medians.csv",
        absolute._stage_component_rows(units),
    )
    absolute.write_latex_tables(
        absolute._group_time_rows(units),
        absolute._coarsest_rows(units),
        absolute._stage_component_rows(units),
        outdir,
    )
    plot_group_ratio_linear(units, outdir)
    plot_absolute_time_matrix_page(units, "v100", outdir)
    plot_absolute_time_matrix_page(units, "p100", outdir)
    plot_runtime_traceoff_linear(units, outdir)
    plot_stage_pages_linear(units, outdir)
    plot_peak_memory_linear(memory_csv, outdir)
    plot_residual_pages(units, outdir)
    plot_residual_overlay_linear(units, outdir)
    plot_reference_runtime_linear(units, outdir)
    _write_source_manifest(units, input_dir, outdir, source_commit)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build linear-axis MG report figures from test27 JSON"
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--memory-csv", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    build(
        args.input_dir,
        args.memory_csv,
        args.outdir,
        args.source_commit,
    )
    print((args.outdir / "mg_group_ratio_linear.pdf").resolve())
    print((args.outdir / "mg_residual_linear_v100_mg3.pdf").resolve())
    print((args.outdir / "mg_peak_memory_linear.pdf").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
