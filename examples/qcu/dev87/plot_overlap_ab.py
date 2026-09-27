#!/usr/bin/env python3
"""Summarize and plot an overlap-on/overlap-off Strict-MG A/B pair."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _side(path: Path, side: str = "pyqcu") -> Mapping[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    record = (document.get("sides") or {}).get(side)
    if not isinstance(record, Mapping) or record.get("status") != "ok":
        raise ValueError(f"{path}: side {side!r} is not successful")
    return record


def _metrics(path: Path) -> dict[str, Any]:
    record = _side(path)
    timing = record["timing"]
    steady = timing["steady"]
    iterations = record["iterations"]
    residual = record["true_residual"]
    return {
        "path": str(path),
        "steady_seconds": float(steady["median_seconds"]),
        "steady_mad_seconds": float(steady["mad_seconds"]),
        "steady_samples_seconds": list(steady["samples_seconds"]),
        "iterations_median": float(iterations["median"]),
        "iterations_samples": list(iterations["samples"]),
        "max_true_residual_rel": float(residual["max_rel"]),
        "true_residual_gate": float(residual["gate"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--off", type=Path, action="append", required=True)
    parser.add_argument("--on", type=Path, action="append", required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    def aggregate(paths: list[Path]) -> dict[str, Any]:
        runs = [_metrics(path) for path in paths]
        samples = [
            value for run in runs for value in run["steady_samples_seconds"]]
        sample_medians = [
            run["steady_seconds"] for run in runs]
        return {
            "runs": runs,
            "steady_samples_seconds": samples,
            "steady_run_medians_seconds": sample_medians,
            "steady_seconds": float(statistics.median(sample_medians)),
            "steady_mad_seconds": float(statistics.median(
                run["steady_mad_seconds"] for run in runs)),
            "iterations_median": float(statistics.median(
                run["iterations_median"] for run in runs)),
            "max_true_residual_rel": max(
                run["max_true_residual_rel"] for run in runs),
            "true_residual_gate": min(
                run["true_residual_gate"] for run in runs),
        }

    off = aggregate(args.off)
    on = aggregate(args.on)
    improvement = 100.0 * (
        1.0 - on["steady_seconds"] / off["steady_seconds"])
    summary = {
        "off": off,
        "on": on,
        "improvement_percent": improvement,
        "speedup_off_over_on": off["steady_seconds"] / on["steady_seconds"],
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    json_path = args.outdir / "overlap_ab.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    labels = ["overlap off", "overlap on"]
    colors = ["#be4137", "#245ba3"]
    axes[0].bar(labels, [off["steady_seconds"], on["steady_seconds"]],
                color=colors)
    axes[0].set_ylabel("steady median (s)")
    axes[0].set_title(f"Runtime: {improvement:.2f}% lower")
    axes[1].bar(labels, [off["iterations_median"], on["iterations_median"]],
                color=colors)
    axes[1].set_ylabel("outer iterations (median)")
    axes[1].set_title("Outer iterations")
    axes[2].bar(
        labels,
        [off["max_true_residual_rel"], on["max_true_residual_rel"]],
        color=colors)
    axes[2].axhline(off["true_residual_gate"], color="black", linestyle="--")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("max true relative residual")
    axes[2].set_title("Correctness gate")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    for suffix in ("pdf", "svg"):
        figure.savefig(args.outdir / f"overlap_ab.{suffix}")
    plt.close(figure)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
