#!/usr/bin/env python3
"""Plot finest-level MG/outer-solver residual curves from combined JSON."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


UNIT_RE = re.compile(
    r"pyqcu__(?P<device>p100|v100)__(?P<lattice>[^_]+)__"
    r"(?P<precision>c64|c128)__l(?P<levels>[123])__trace-(?P<trace>on|off)")


def _curve(record: Mapping[str, Any]) -> tuple[list[float], list[float]]:
    levels = record.get("mg_levels")
    if not isinstance(levels, list) or not levels:
        return [], []
    sequence = levels[0].get("residual_sequence")
    if not isinstance(sequence, list):
        return [], []
    points: list[tuple[float, float]] = []
    for index, item in enumerate(sequence):
        if not isinstance(item, Mapping):
            continue
        value = item.get("relative")
        if not isinstance(value, (int, float)) or not (value > 0.0):
            continue
        iteration = item.get("iteration")
        x = float(iteration) if isinstance(iteration, (int, float)) else float(index)
        points.append((x, float(value)))
    points.sort()
    return [value[0] for value in points], [value[1] for value in points]


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

    cases = []
    for path in sorted(args.input_dir.glob("*.json")):
        match = UNIT_RE.fullmatch(path.stem)
        if match is None or match.group("trace") != "on":
            continue
        levels = int(match.group("levels"))
        if levels not in (2, 3):
            continue
        import json
        document = json.loads(path.read_text(encoding="utf-8"))
        py_curve = _curve(document.get("sides", {}).get("pyqcu", {}))
        quda_curve = _curve(document.get("sides", {}).get("quda", {}))
        cases.append((
            match.group("device"),
            match.group("precision"),
            match.group("lattice"),
            levels,
            py_curve,
            quda_curve,
        ))

    if not cases:
        raise SystemExit("no trace-on MG-2/3 residual curves found")
    cases.sort(key=lambda item: (item[0], item[1], item[3], item[2]))
    columns = 3
    rows = (len(cases) + columns - 1) // columns
    fig, axes = plt.subplots(
        rows, columns, figsize=(12.5, 2.8 * rows), squeeze=False)
    for axis, (device, precision, lattice, levels, py_curve, quda_curve) in zip(
            axes.flat, cases):
        if py_curve[0]:
            axis.semilogy(
                py_curve[0], py_curve[1], color="#245ba3", linewidth=1.2,
                label="PyQCU")
        if quda_curve[0]:
            axis.semilogy(
                quda_curve[0], quda_curve[1], color="#be4137", linewidth=1.2,
                label="QUDA")
        axis.set_title(
            f"{device.upper()} {precision} {lattice} MG-{levels}",
            fontsize=9)
        axis.set_xlabel("outer iteration", fontsize=8)
        axis.set_ylabel(r"$r/r_0$", fontsize=8)
        axis.grid(True, which="both", alpha=0.25, linewidth=0.5)
        axis.tick_params(labelsize=7)
        axis.legend(fontsize=7)
    for axis in axes.flat[len(cases):]:
        axis.axis("off")
    fig.tight_layout()
    pdf = args.outdir / "mg_residual_curves.pdf"
    svg = args.outdir / "mg_residual_curves.svg"
    fig.savefig(pdf, format="pdf", bbox_inches="tight")
    fig.savefig(svg, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(pdf.resolve())
    print(svg.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
