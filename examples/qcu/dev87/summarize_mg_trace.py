#!/usr/bin/env python3
"""Aggregate the diagnostic MultiGrid trace into report-ready tables."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


REPO = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO / "data" / "strict_trace_stage_3l_20260916.json"
DEFAULT_OUTPUT = REPO / "data" / "strict_trace_summary_3l_20260916.json"


def _median(values: Iterable[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot take median of empty sequence")
    return statistics.median(ordered)


def _mad(values: Iterable[float]) -> float:
    ordered = [float(value) for value in values]
    center = _median(ordered)
    return statistics.median(abs(value - center) for value in ordered)


def _rows(items: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, str], List[float]] = defaultdict(list)
    for item in items:
        level = int(item.get("level", -1))
        phase = str(item.get("phase", item.get("name", "unknown")))
        seconds = float(item["seconds"])
        if seconds > 0.0:
            grouped[(level, phase)].append(seconds)
    result = []
    for (level, phase), values in sorted(grouped.items()):
        result.append({
            "level": level,
            "phase": phase,
            "samples": len(values),
            "median_seconds": _median(values),
            "mad_seconds": _mad(values),
            "total_seconds": sum(values),
        })
    return result


def _residual_rows(items: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, str], List[Tuple[float, float]]] = defaultdict(list)
    for item in items:
        level = int(item.get("level", -1))
        phase = str(item.get("name", item.get("phase", "unknown")))
        absolute = float(item.get("absolute", item.get("rn", 0.0)))
        relative = float(item.get("relative", 0.0))
        grouped[(level, phase)].append((absolute, relative))
    result = []
    for (level, phase), values in sorted(grouped.items()):
        result.append({
            "level": level,
            "phase": phase,
            "samples": len(values),
            "median_absolute_r": _median(value[0] for value in values),
            "median_relative_r": _median(value[1] for value in values),
        })
    return result


def _quda_rows(solve: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]],
                                                   List[Dict[str, Any]]]:
    mg = solve.get("mg_trace")
    if not isinstance(mg, Mapping):
        return [], []
    stages: List[Mapping[str, Any]] = []
    residuals: List[Mapping[str, Any]] = []
    for cycle in mg.get("cycles", []):
        if not isinstance(cycle, Mapping):
            continue
        begin = cycle.get("begin", {})
        level = int(begin.get("level", -1))
        for stage in cycle.get("stages", []):
            if stage.get("seconds", 0.0) > 0.0:
                stages.append({**stage, "level": level})
        for residual in cycle.get("residuals", []):
            residuals.append({**residual, "level": level})
    return [dict(item) for item in stages], [dict(item) for item in residuals]


def summarize(document: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "schema": {"name": "pyqcu.mg-trace-summary", "version": 1},
        "protocol": document.get("protocol"),
        "sides": {},
    }
    for side in ("pyqcu", "quda"):
        solves = document["sides"][side]["steady"]
        stages: List[Mapping[str, Any]] = []
        residuals: List[Mapping[str, Any]] = []
        for solve in solves:
            stages.extend(solve.get("stages", []))
            residuals.extend(solve.get("residuals", []))
        if side == "quda":
            native_stages = []
            native_residuals = []
            for solve in solves:
                stage_rows, residual_rows = _quda_rows(solve)
                native_stages.extend(stage_rows)
                native_residuals.extend(residual_rows)
            stages = native_stages
            residuals = native_residuals
        result["sides"][side] = {
            "steady_solves": len(solves),
            "iterations": [int(solve["iterations"]) for solve in solves],
            "stage_rows": _rows(stages),
            "residual_rows": _residual_rows(residuals),
        }
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize PyQCU/QUDA MultiGrid trace JSON")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    document = json.loads(args.input.read_text(encoding="utf-8"))
    summary = summarize(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    for side in ("pyqcu", "quda"):
        _write_csv(
            args.output.with_name(f"{args.output.stem}_{side}_stages.csv"),
            summary["sides"][side]["stage_rows"])
        _write_csv(
            args.output.with_name(f"{args.output.stem}_{side}_residuals.csv"),
            summary["sides"][side]["residual_rows"])
    print(json.dumps({
        "output": str(args.output.resolve()),
        "pyqcu_stage_rows": len(summary["sides"]["pyqcu"]["stage_rows"]),
        "quda_stage_rows": len(summary["sides"]["quda"]["stage_rows"]),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
