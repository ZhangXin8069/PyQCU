#!/usr/bin/env python3
"""Generate compact final speedup and stage-breakdown figures."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


UNIT_RE = re.compile(
    r"pyqcu__(?P<device>p100|v100)__(?P<lattice>[^_]+)__"
    r"(?P<precision>c64|c128)__l(?P<levels>[123])__trace-(?P<trace>on|off)")
PHASES = (
    ("pre_smoother_seconds", "pre"),
    ("post_smoother_seconds", "post"),
    ("restriction_seconds", "restrict"),
    ("prolongation_seconds", "prolong"),
    ("coarse_solver_seconds", "coarse"),
    ("other_seconds", "other"),
)
COLORS = ("#245ba3", "#4f8fcb", "#be4137", "#df7a52", "#5a6973", "#b8bec3")


def _unit(document: Mapping[str, Any], path: Path) -> dict[str, Any]:
    match = UNIT_RE.fullmatch(path.stem)
    if match is None:
        raise ValueError(path)
    return {**match.groupdict(), "document": document}


def _speedup(units: list[dict[str, Any]], outdir: Path) -> None:
    rows = []
    for unit in units:
        comparison = unit["document"].get("comparison") or {}
        ratio = comparison.get("speedup_pyqcu_over_quda")
        if not isinstance(ratio, (int, float)):
            continue
        rows.append((path_name(unit), float(ratio), unit))
    rows.sort(key=lambda item: item[1])
    fig, axis = plt.subplots(figsize=(10.5, 8.5))
    values = [row[1] for row in rows]
    colors = [
        "#be4137" if row[2]["levels"] == "1"
        else "#245ba3" if row[2]["device"] == "v100"
        else "#4f8fcb"
        for row in rows
    ]
    axis.barh(range(len(rows)), values, color=colors, height=0.82)
    axis.axvline(1.0, color="black", linewidth=0.8, linestyle="--")
    axis.set_xscale("log")
    axis.set_yticks(range(len(rows)))
    axis.set_yticklabels([row[0] for row in rows], fontsize=4.6)
    axis.set_xlabel("QUDA / PyQCU steady median (log scale)")
    axis.grid(True, axis="x", alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(outdir / "mg_speedup_units.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(outdir / "mg_speedup_units.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def path_name(unit: Mapping[str, Any]) -> str:
    return (
        f"{unit['device']} {unit['precision']} {unit['lattice']} "
        f"L{unit['levels']} {unit['trace']}")


def _stage_breakdown(units: list[dict[str, Any]], outdir: Path) -> None:
    wanted = (
        ("p100", "c64", "16x32x32x48", "2", "on"),
        ("p100", "c64", "16x32x32x48", "3", "on"),
        ("v100", "c128", "16x16x32x32", "2", "on"),
        ("v100", "c128", "16x16x32x32", "3", "on"),
    )
    selected = []
    for key in wanted:
        for unit in units:
            if tuple(unit[name] for name in (
                    "device", "precision", "lattice", "levels", "trace")) == key:
                selected.append(unit)
                break
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), squeeze=False)
    for axis, unit in zip(axes.flat, selected):
        document = unit["document"]
        labels = []
        bottoms = [0.0] * len(PHASES)
        for side, color_offset in (("pyqcu", 0), ("quda", 0)):
            levels = document["sides"][side].get("mg_levels") or []
            for level in levels:
                label = f"{side} L{level['level']}"
                labels.append(label)
                values = [float(level.get(key, 0.0)) for key, _ in PHASES]
                left = 0.0
                for index, value in enumerate(values):
                    axis.barh(
                        len(labels) - 1, value, left=left,
                        color=COLORS[index], height=0.7)
                    left += value
        axis.set_yticks(range(len(labels)))
        axis.set_yticklabels(labels, fontsize=7)
        axis.set_xlabel("nested stage seconds")
        axis.set_title(path_name(unit), fontsize=9)
        axis.grid(True, axis="x", alpha=0.2, linewidth=0.5)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS[index])
        for index in range(len(PHASES))]
    fig.legend(
        handles, [label for _, label in PHASES], ncol=6,
        loc="upper center", fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(
        outdir / "mg_stage_breakdown.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(
        outdir / "mg_stage_breakdown.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", type=Path,
        default=Path("data/mg_matrix_final_combined/units"))
    parser.add_argument(
        "--outdir", type=Path,
        default=Path("data/mg_matrix_final_report"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    units = []
    for path in sorted(args.input_dir.glob("*.json")):
        units.append(_unit(json.loads(path.read_text(encoding="utf-8")), path))
    _speedup(units, args.outdir)
    _stage_breakdown(units, args.outdir)
    print((args.outdir / "mg_speedup_units.pdf").resolve())
    print((args.outdir / "mg_stage_breakdown.pdf").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
