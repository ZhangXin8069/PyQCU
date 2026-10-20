#!/usr/bin/env python3
"""Run the KUPL C++ operator benchmark repeatedly and report stable medians."""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from ._common import DATA_DIR, resolve_path, write_json
except ImportError:
    from _common import DATA_DIR, resolve_path, write_json


LINE = re.compile(
    r"precision=(?P<precision>\S+) operator=(?P<operator>\S+) "
    r"ranks=(?P<ranks>\d+) grid=(?P<grid>\S+) local=(?P<local>\S+) "
    r"threads_per_rank=(?P<threads>\d+) "
    r"(?:iterations=(?P<iterations>\d+) )? ?seconds=(?P<seconds>[0-9.eE+-]+)"
    r"(?: global_sites=(?P<sites>\d+) "
    r"estimated_gflops=(?P<gflops>[0-9.eE+-]+))?"
)


def parse_output(text: str) -> dict[tuple[str, str], dict[str, float]]:
    records: dict[tuple[str, str], dict[str, float]] = {}
    for line in text.splitlines():
        match = LINE.search(line)
        if not match:
            continue
        values: dict[str, Any] = match.groupdict()
        key = (values["precision"], values["operator"])
        record = {
            "seconds": float(values["seconds"]),
            "threads": int(values["threads"]),
            "ranks": int(values["ranks"]),
        }
        if values["iterations"] is not None:
            record["iterations"] = int(values["iterations"])
            record["global_sites"] = int(values["sites"])
            record["estimated_gflops"] = float(values["gflops"])
        records[key] = record
    return records


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default=None)
    parser.add_argument("--lat", nargs=4, type=int, default=(16, 16, 16, 32))
    parser.add_argument("--grid", nargs=4, type=int, default=(1, 1, 1, 1))
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--output", default="performance_kupl_benchmark.json")
    args = parser.parse_args(argv)
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    root = Path(__file__).resolve().parents[3]
    binary = Path(args.binary).expanduser() if args.binary else \
        root / "cpp/kupl/qcu/build/kupl_qcu_bench"
    if not binary.is_file():
        print(json.dumps({"status": "SKIP", "reason": f"missing {binary}"}))
        return 0
    command = [
        str(binary), "--lattice", *(str(value) for value in args.lat),
        "--grid", *(str(value) for value in args.grid),
        "--iterations", str(args.iterations),
    ]
    environment = os.environ.copy()
    environment["KUPL_OMP_THREADS"] = str(args.threads)
    environment["OMP_NUM_THREADS"] = str(args.threads)
    environment.setdefault("OMP_PROC_BIND", "close")
    environment.setdefault("OMP_PLACES", "cores")
    samples: dict[tuple[str, str], list[dict[str, float]]] = {}
    runs = []
    for repeat in range(args.repeats):
        completed = subprocess.run(
            command, text=True, capture_output=True, timeout=args.timeout,
            check=False, env=environment)
        parsed = parse_output(completed.stdout)
        runs.append({
            "repeat": repeat + 1, "returncode": completed.returncode,
            "stdout": completed.stdout, "stderr": completed.stderr,
        })
        if completed.returncode != 0:
            break
        for key, record in parsed.items():
            samples.setdefault(key, []).append(record)

    successful = all(run["returncode"] == 0 for run in runs) and \
        len(runs) == args.repeats
    results = {}
    if successful:
        for (precision, operator), values in sorted(samples.items()):
            seconds = [value["seconds"] for value in values]
            median = statistics.median(seconds)
            mad = statistics.median(abs(value - median) for value in seconds)
            result = {
                "samples": len(values), "seconds_median": median,
                "seconds_mad": mad, "seconds_min": min(seconds),
                "seconds_max": max(seconds),
            }
            if "iterations" in values[0]:
                iterations = int(values[0]["iterations"])
                sites = int(values[0]["global_sites"])
                result["iterations"] = iterations
                result["global_sites"] = sites
                result["estimated_gflops_median"] = \
                    values[0]["estimated_gflops"] * values[0]["seconds"] / median
            results.setdefault(precision, {})[operator] = result
    report = {
        "status": "PASS" if successful else "FAIL",
        "binary": str(binary), "command": command,
        "environment": {
            "KUPL_OMP_THREADS": environment["KUPL_OMP_THREADS"],
            "OMP_NUM_THREADS": environment["OMP_NUM_THREADS"],
            "OMP_PROC_BIND": environment["OMP_PROC_BIND"],
            "OMP_PLACES": environment["OMP_PLACES"],
            "KUPL_DISABLE_BLOCK_DSLASH": environment.get(
                "KUPL_DISABLE_BLOCK_DSLASH"),
        },
        "repeats": args.repeats, "runs": runs, "results": results,
    }
    output = resolve_path(args.output, DATA_DIR / args.output)
    write_json(output, report)
    print(json.dumps({
        "status": report["status"], "output": str(output),
        "repeats": len(runs), "results": results,
    }, ensure_ascii=False))
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
