#!/usr/bin/env python3
"""Compare side records from the overlap campaign against QUDA side records."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(path: Path) -> Mapping[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _record_side(path: Path, side: str) -> Mapping[str, Any] | None:
    document = _load(path)
    if document is None:
        return None
    record = (document.get("sides") or {}).get(side)
    return record if isinstance(record, Mapping) else None


def _steady(record: Mapping[str, Any]) -> float | None:
    value = ((record.get("timing") or {}).get("steady") or {}).get(
        "median_seconds")
    return float(value) if isinstance(value, (int, float)) else None


def _iterations(record: Mapping[str, Any]) -> float | None:
    value = (record.get("iterations") or {}).get("median")
    return float(value) if isinstance(value, (int, float)) else None


def _residual(record: Mapping[str, Any]) -> float | None:
    value = (record.get("true_residual") or {}).get("max_rel")
    return float(value) if isinstance(value, (int, float)) else None


def _index(root: Path, side: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(root.glob("*.json")):
        name = path.name
        prefix = f"{side}__"
        if name.startswith(prefix):
            result[name] = path
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pyqcu-root", type=Path, action="append", required=True)
    parser.add_argument("--quda-root", type=Path, action="append", required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--allow-config-mismatch", action="store_true")
    args = parser.parse_args()

    pyqcu: dict[str, Path] = {}
    for root in args.pyqcu_root:
        pyqcu.update(_index(root, "pyqcu"))
    quda: dict[str, Path] = {}
    for root in args.quda_root:
        quda.update(_index(root, "quda"))

    rows: list[dict[str, Any]] = []
    mismatches: list[dict[str, str]] = []
    for pyqcu_name, pyqcu_path in pyqcu.items():
        suffix = pyqcu_name.removeprefix("pyqcu__")
        quda_path = quda.get(f"quda__{suffix}")
        if quda_path is None:
            continue
        py_record = _record_side(pyqcu_path, "pyqcu")
        qu_record = _record_side(quda_path, "quda")
        if py_record is None or qu_record is None:
            continue
        if py_record.get("status") != "ok" or qu_record.get("status") != "ok":
            continue
        py_seconds = _steady(py_record)
        qu_seconds = _steady(qu_record)
        if py_seconds is None or qu_seconds is None or py_seconds <= 0:
            continue
        py_config = (py_record.get("config_hash") or "")
        qu_config = (qu_record.get("config_hash") or "")
        if py_config != qu_config and not args.allow_config_mismatch:
            mismatches.append({
                "unit_id": suffix.removesuffix(".json"),
                "pyqcu_config_hash": str(py_config),
                "quda_config_hash": str(qu_config),
            })
            continue
        parts = suffix.removesuffix(".json").split("__")
        device, lattice, precision, level_tag, trace_tag = parts
        levels = int(level_tag.removeprefix("l"))
        trace = trace_tag.removeprefix("trace-")
        ratio = qu_seconds / py_seconds
        rows.append({
            "unit_id": suffix.removesuffix(".json"),
            "device": device,
            "lattice": lattice,
            "precision": precision,
            "levels": levels,
            "trace": trace,
            "solver": "bicgstab-reference" if levels == 1 else f"mg-{levels}",
            "pyqcu_seconds": py_seconds,
            "quda_seconds": qu_seconds,
            "ratio_quda_over_pyqcu": ratio,
            "pyqcu_iterations": _iterations(py_record),
            "quda_iterations": _iterations(qu_record),
            "pyqcu_true_residual": _residual(py_record),
            "quda_true_residual": _residual(qu_record),
        })

    groups: dict[str, list[float]] = {}
    for row in rows:
        key = "|".join((
            row["device"], row["precision"], row["solver"], row["trace"]))
        groups.setdefault(key, []).append(row["ratio_quda_over_pyqcu"])
    group_summary = {
        key: {
            "count": len(values),
            "median": statistics.median(values),
            "wins": sum(value > 1.0 for value in values),
            "losses": sum(value < 1.0 for value in values),
        }
        for key, values in sorted(groups.items())
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "pyqcu_roots": [str(path) for path in args.pyqcu_root],
        "quda_roots": [str(path) for path in args.quda_root],
        "matched_units": len(rows),
        "config_mismatches": mismatches,
        "groups": group_summary,
        "units": rows,
    }
    (args.outdir / "overlap_matrix_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    fieldnames = list(rows[0]) if rows else [
        "unit_id", "device", "lattice", "precision", "levels", "trace",
        "solver", "pyqcu_seconds", "quda_seconds",
        "ratio_quda_over_pyqcu", "pyqcu_iterations", "quda_iterations",
        "pyqcu_true_residual", "quda_true_residual",
    ]
    with (args.outdir / "overlap_matrix.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if rows:
        labels = list(group_summary)
        values = [group_summary[label]["median"] for label in labels]
        figure, axes = plt.subplots(
            1, 2, figsize=(12.5, 5.2),
            gridspec_kw={"width_ratios": [1.35, 1.0]})
        axes[0].barh(labels, values, color="#245ba3")
        axes[0].axvline(1.0, color="black", linestyle="--", linewidth=1)
        axes[0].set_xlabel("QUDA / PyQCU steady median")
        axes[0].set_title("Group medians")
        per_unit = sorted(rows, key=lambda item: item["ratio_quda_over_pyqcu"])
        axes[1].scatter(
            [item["ratio_quda_over_pyqcu"] for item in per_unit],
            list(range(len(per_unit))),
            c=["#245ba3" if item["ratio_quda_over_pyqcu"] >= 1.0
               else "#be4137" for item in per_unit],
            s=18)
        axes[1].axvline(1.0, color="black", linestyle="--", linewidth=1)
        axes[1].set_yticks([])
        axes[1].set_xlabel("QUDA / PyQCU steady median")
        axes[1].set_title(f"Per-unit ratios (n={len(rows)})")
        for axis in axes:
            axis.grid(alpha=0.2)
        figure.tight_layout()
        for suffix in ("pdf", "svg"):
            figure.savefig(args.outdir / f"overlap_matrix.{suffix}")
        plt.close(figure)
    print(json.dumps({
        "matched_units": len(rows),
        "config_mismatch_count": len(mismatches),
        "groups": group_summary,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
