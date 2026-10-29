#!/usr/bin/env python3
"""Orchestrate separate MPI QCU and QUDA BiCGStab child jobs."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from pyqcu.testing.simple._common import DEFAULT_LIME
from pyqcu.testing.simple.mpi._compare import (
    child_report,
    child_command,
    require_orchestrator_singleton,
    run_jobs,
    validate_rank_grid,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gauge", default=str(DEFAULT_LIME))
    parser.add_argument("--source", default=None)
    parser.add_argument("--lat", nargs=4, type=int, default=(8, 8, 8, 16))
    parser.add_argument("--mass", type=float, default=None)
    parser.add_argument("--csw", type=float, default=None)
    parser.add_argument("--atol", type=float, default=1.0e-6)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qcu-ranks", type=int, default=2)
    parser.add_argument("--quda-ranks", type=int, default=2)
    parser.add_argument("--grid", nargs=4, type=int, default=None)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--report", default="compare_bistabcg_mpi.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    require_orchestrator_singleton(parser)

    qcu_ranks, qcu_grid = validate_rank_grid(
        parser, args.lat, args.qcu_ranks, args.grid)
    quda_ranks, quda_grid = validate_rank_grid(
        parser, args.lat, args.quda_ranks, args.grid)
    common = [
        "--gauge", args.gauge, "--lat", *(str(value) for value in args.lat),
        "--max-iter", str(args.max_iter),
        "--warmups", str(args.warmups), "--repeats", str(args.repeats),
        "--seed", str(args.seed),
    ]
    if args.source is not None:
        common.extend(("--source", args.source))
    if args.mass is not None:
        common.extend(("--mass", str(args.mass)))
    if args.csw is not None:
        common.extend(("--csw", str(args.csw)))
    jobs = (
        ("qcu", child_command("qcu_bistabcg.py", [
            *common, "--atol", str(args.atol),
            "--grid", *(str(value) for value in qcu_grid),
            "--ranks", str(qcu_ranks),
            "--report", child_report(args.report, "qcu")])),
        ("quda", child_command("quda_bistabcg.py", [
            *common, "--tol", str(args.atol),
            "--grid", *(str(value) for value in quda_grid),
            "--ranks", str(quda_ranks),
            "--report", child_report(args.report, "quda")])),
    )
    return run_jobs(
        jobs, report=args.report, dry_run=args.dry_run, timeout=args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
