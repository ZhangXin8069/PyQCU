#!/usr/bin/env python3
"""Run the reproducible PyQCU/QUDA MultiGrid comparison matrix.

The matrix deliberately uses the formal collector for every executable case;
smoke mode only relaxes the collector's repeat/tolerance contract.  Cases with
missing canonical/QIO assets are reported as ``skipped`` instead of silently
falling back to random null vectors.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BENCH = HERE / "bench_strict_vs_quda.py"
TRACE = HERE / "trace_strict_vs_quda.py"
DEFAULT_OUTPUT = REPO / "data" / "mg_matrix_20260916"

CASES: tuple[Dict[str, Any], ...] = (
    {
        "id": "small-c64-l2",
        "lattice": [8, 8, 8, 16],
        "levels": 2,
        "precision": "c64",
    },
    {
        "id": "small-c64-l3",
        "lattice": [8, 8, 8, 16],
        "levels": 3,
        "precision": "c64",
    },
    {
        "id": "small-c128-l3",
        "lattice": [8, 8, 8, 16],
        "levels": 3,
        "precision": "c128",
        "quda_mg_supported": False,
    },
    {
        "id": "medium-c64-l3",
        "lattice": [16, 16, 16, 16],
        "levels": 3,
        "precision": "c64",
    },
    {
        "id": "medium-c128-l3",
        "lattice": [16, 16, 16, 16],
        "levels": 3,
        "precision": "c128",
        "quda_mg_supported": False,
    },
    {
        "id": "formal-c64-l2",
        "lattice": [16, 32, 32, 48],
        "levels": 2,
        "precision": "c64",
    },
    {
        "id": "formal-c64-l3",
        "lattice": [16, 32, 32, 48],
        "levels": 3,
        "precision": "c64",
    },
    {
        "id": "formal-c128-l3",
        "lattice": [16, 32, 32, 48],
        "levels": 3,
        "precision": "c128",
        "quda_mg_supported": False,
    },
)


def _lattice_tag(lattice: Sequence[int]) -> str:
    return "x".join(str(int(value)) for value in lattice)


def _asset_paths(lattice: Sequence[int]) -> Dict[str, Path]:
    tag = _lattice_tag(lattice)
    qio_prefix = REPO / "data" / f"L{tag}_nvec12_quda"
    return {
        "gauge": REPO / "data" / f"gauge_{tag}_m0.05_seed42_c64.h5",
        "null_vectors": REPO / "data" / f"L{tag}_nvec12_full_c64.h5",
        "qio_prefix": qio_prefix,
        "qio_artifact": Path(f"{qio_prefix}_level_0_nvec_12"),
        "qio_manifest": REPO / "data" / f"L{tag}_nvec12_quda.conversion.json",
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _case_command(
        case: Mapping[str, Any], *, profile: str, repeats: int,
        timeout: float, output: Path, side: str = "both",
        extra: Sequence[str] = (),
) -> List[str]:
    assets = _asset_paths(case["lattice"])
    return [
        sys.executable, "-B", str(BENCH),
        "--profile", profile,
        "--side", side,
        "--lattice", *[str(value) for value in case["lattice"]],
        "--levels", str(case["levels"]),
        "--precision", str(case["precision"]),
        "--repeats", str(repeats),
        "--timeout", str(timeout),
        "--gauge-path", str(assets["gauge"]),
        "--nullvec-path", str(assets["null_vectors"]),
        "--quda-nullvec-prefix", str(assets["qio_prefix"]),
        "--quda-nullvec-manifest", str(assets["qio_manifest"]),
        "--output", str(output),
        *extra,
    ]


def _run_case(
        case: Mapping[str, Any], *, profile: str, repeats: int,
        timeout: float, output_dir: Path, trace: bool,
        allow_c128_quda: bool = False,
) -> Dict[str, Any]:
    case_dir = output_dir / str(case["id"])
    case_dir.mkdir(parents=True, exist_ok=True)
    benchmark_output = case_dir / "benchmark.json"
    assets = _asset_paths(case["lattice"])
    missing = [
        str(path) for key, path in assets.items()
        if key != "qio_prefix"
        if not path.exists()
    ]
    if missing:
        return {
            "id": case["id"],
            "status": "skipped",
            "reason": {"code": "missing_assets", "detail": missing},
            "case": dict(case),
        }

    runs: List[Dict[str, Any]] = []
    partial_reason = None
    side = "both"
    if (str(case["precision"]) == "c128" and
            case.get("quda_mg_supported", True) is False and
            not allow_c128_quda):
        side = "pyqcu"
        partial_reason = (
            "QUDA 1.1.0's upstream recursive MultiGrid is single-precision "
            "only (GPU_MULTIGRID_DOUBLE disabled); this case records the "
            "PyQCU c128 result and does not fabricate a cross-library speedup.")
    policies = (
        ("miss", "hit") if profile == "formal" else ("any",)
    )
    for policy in policies:
        target = benchmark_output if policy == policies[-1] else (
            case_dir / f"benchmark_{policy}.json")
        command = _case_command(
            case, profile=profile, repeats=repeats, timeout=timeout,
            output=target, side=side,
            extra=("--cache-expect", policy))
        started = time.perf_counter()
        completed = subprocess.run(
            command, cwd=str(REPO), text=True, capture_output=True,
            timeout=max(30.0, timeout * 3.0), check=False)
        record = {
            "cache_expect": policy,
            "returncode": completed.returncode,
            "elapsed_seconds": time.perf_counter() - started,
            "command": command,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
            "output": str(target),
        }
        if target.is_file():
            record["sha256"] = _sha256_file(target)
            try:
                document = json.loads(target.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                record["json_error"] = repr(exc)
            else:
                record["status"] = document.get("status")
                comparison = document.get("comparison")
                if isinstance(comparison, Mapping):
                    record["speedup"] = comparison.get(
                        "speedup_pyqcu_over_quda")
                    record["fair"] = comparison.get("fair")
        runs.append(record)
        if completed.returncode != 0:
            break

    result: Dict[str, Any] = {
        "id": case["id"],
        "case": dict(case),
        "assets": {key: str(value) for key, value in assets.items()},
        "runs": runs,
        "status": (
            "partial" if partial_reason and runs and
            runs[-1]["returncode"] == 0 else
            "ok" if runs and runs[-1]["returncode"] == 0 else "failed"
        ),
    }
    if partial_reason is not None:
        result["reason"] = {
            "code": "quda_c128_mg_unavailable",
            "detail": partial_reason,
        }
    if trace and result["status"] == "ok":
        trace_output = case_dir / "trace.json"
        assets = _asset_paths(case["lattice"])
        trace_command = [
            sys.executable, "-B", str(TRACE),
            "--output", str(trace_output),
            "--plot", str(case_dir / "trace.svg"),
            "--reference", str(benchmark_output),
            "--repeats", str(repeats),
            "--lattice", *[str(value) for value in case["lattice"]],
            "--levels", str(case["levels"]),
            "--precision", str(case["precision"]),
            "--gauge-path", str(assets["gauge"]),
            "--nullvec-path", str(assets["null_vectors"]),
            "--quda-nullvec-prefix", str(assets["qio_prefix"]),
            "--quda-nullvec-manifest", str(assets["qio_manifest"]),
        ]
        completed = subprocess.run(
            trace_command, cwd=str(REPO), text=True, capture_output=True,
            timeout=max(30.0, timeout), check=False)
        result["trace"] = {
            "returncode": completed.returncode,
            "command": trace_command,
            "output": str(trace_output),
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }
        if trace_output.is_file():
            result["trace"]["sha256"] = _sha256_file(trace_output)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the PyQCU/QUDA MultiGrid comparison matrix")
    parser.add_argument(
        "--profile", choices=("formal", "smoke"), default="smoke")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--allow-c128-quda", action="store_true")
    parser.add_argument("--list", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    selected = [
        case for case in CASES
        if not args.only or case["id"] in set(args.only)
    ]
    if args.only and set(args.only) - {case["id"] for case in CASES}:
        raise ValueError(
            f"unknown case(s): {sorted(set(args.only) - {case['id'] for case in CASES})}")
    if args.list:
        print(json.dumps(list(selected), ensure_ascii=False, indent=2))
        return 0
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [
        _run_case(
            case, profile=args.profile, repeats=args.repeats,
            timeout=args.timeout, output_dir=args.output_dir,
            trace=bool(args.trace),
            allow_c128_quda=bool(args.allow_c128_quda))
        for case in selected
    ]
    document = {
        "schema": {"name": "pyqcu.mg-matrix", "version": 1},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "repeats": args.repeats,
        "cases": results,
        "summary": {
            "total": len(results),
            "ok": sum(result["status"] == "ok" for result in results),
            "partial": sum(
                result["status"] == "partial" for result in results),
            "skipped": sum(result["status"] == "skipped" for result in results),
            "failed": sum(result["status"] == "failed" for result in results),
        },
    }
    output = args.output_dir / f"matrix_{args.profile}.json"
    output.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "output": str(output.resolve()),
        "summary": document["summary"],
    }, ensure_ascii=False, indent=2))
    return 0 if document["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"bench_mg_matrix: {exc}", file=sys.stderr)
        raise SystemExit(1)
