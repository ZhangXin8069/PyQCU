#!/usr/bin/env python3
"""Plot normalized large-volume QCU/QUDA solver matrix records."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from pyqcu import tools


REQUIRED_FIELDS = (
    "backend", "solver", "ranks", "grid", "wall_seconds",
    "iterations", "true_residual")


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path}: records must be a non-empty list")
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"{path}: records[{index}] must be an object")
        missing = [name for name in REQUIRED_FIELDS if name not in record]
        if missing:
            raise ValueError(f"{path}: records[{index}] missing {missing}")
    return payload


def _with_scaling(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    baselines: dict[tuple[str, str], float] = {}
    for record in records:
        if int(record["ranks"]) == 1:
            baselines[(str(record["backend"]), str(record["solver"]))] = \
                float(record["wall_seconds"])
    rows = []
    for record in records:
        row = dict(record)
        key = (str(row["backend"]), str(row["solver"]))
        baseline = baselines.get(key)
        row["strong_speedup"] = (
            baseline / float(row["wall_seconds"])
            if baseline is not None else None)
        row["parallel_efficiency"] = (
            row["strong_speedup"] / int(row["ranks"])
            if row["strong_speedup"] is not None else None)
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="large_volume")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args(argv)

    payload = _load(args.input.expanduser())
    rows = _with_scaling(payload["records"])
    output = args.out_dir.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    summary = {**payload, "records": rows}
    (output / f"{args.prefix}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    _write_csv(output / f"{args.prefix}_summary.csv", rows)

    figures = []
    figures += tools.plot_rank_metric(
        rows, "wall_seconds", str(output), f"{args.prefix}_wall",
        "steady solve wall (s)", "L32x64 steady solve wall by MPI ranks",
        log_y=True, dpi=args.dpi)
    figures += tools.plot_rank_metric(
        rows, "strong_speedup", str(output), f"{args.prefix}_speedup",
        "speedup vs 1 rank", "L32x64 strong scaling", reference=1.0,
        dpi=args.dpi)
    figures += tools.plot_rank_metric(
        rows, "parallel_efficiency", str(output),
        f"{args.prefix}_efficiency", "parallel efficiency",
        "L32x64 parallel efficiency", reference=1.0, dpi=args.dpi)
    figures += tools.plot_rank_metric(
        rows, "iterations", str(output), f"{args.prefix}_iterations",
        "iterations", "L32x64 solver iterations", dpi=args.dpi)
    figures += tools.plot_rank_metric(
        rows, "true_residual", str(output), f"{args.prefix}_residual",
        "relative true residual", "L32x64 independent true residual",
        log_y=True, reference=1.0e-6, dpi=args.dpi)
    print(json.dumps({"status": "PASS", "figures": figures,
                      "summary": str(output / f"{args.prefix}_summary.json")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
