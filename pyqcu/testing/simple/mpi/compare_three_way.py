#!/usr/bin/env python3
"""Orchestrate isolated MPI CUDA-QCU, KUPL and QUDA benchmark jobs."""

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
    parser.add_argument("--lat", nargs=4, type=int, default=(8, 8, 16, 16))
    parser.add_argument("--mass", type=float, default=None)
    parser.add_argument("--csw", type=float, default=None)
    parser.add_argument("--atol", type=float, default=1.0e-6)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backends", default="cuda,kupl,quda")
    parser.add_argument("--ranks", type=int, default=None,
                        help="set all backend rank counts")
    parser.add_argument("--cuda-ranks", type=int, default=2)
    parser.add_argument("--kupl-ranks", type=int, default=2)
    parser.add_argument("--quda-ranks", type=int, default=2)
    parser.add_argument("--cuda-grid", nargs=4, type=int, default=None)
    parser.add_argument("--kupl-grid", nargs=4, type=int, default=None)
    parser.add_argument("--quda-grid", nargs=4, type=int, default=None)
    parser.add_argument("--kupl-threads", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--report", default="compare_three_way_mpi.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    require_orchestrator_singleton(parser)
    if args.ranks is not None:
        args.cuda_ranks = args.kupl_ranks = args.quda_ranks = args.ranks

    selected = {value.strip() for value in args.backends.split(",") if value.strip()}
    if "all" in selected:
        selected = {"cuda", "kupl", "quda"}
    unknown = selected.difference({"cuda", "kupl", "quda"})
    if unknown:
        parser.error(f"unknown backends: {', '.join(sorted(unknown))}")
    cuda_ranks, cuda_grid = (
        validate_rank_grid(parser, args.lat, args.cuda_ranks, args.cuda_grid)
        if "cuda" in selected else (args.cuda_ranks, list(args.cuda_grid or ())))
    kupl_ranks, kupl_grid = (
        validate_rank_grid(parser, args.lat, args.kupl_ranks, args.kupl_grid)
        if "kupl" in selected else (args.kupl_ranks, list(args.kupl_grid or ())))
    quda_ranks, quda_grid = (
        validate_rank_grid(parser, args.lat, args.quda_ranks, args.quda_grid)
        if "quda" in selected else (args.quda_ranks, list(args.quda_grid or ())))
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

    jobs: list[tuple[str, Sequence[str]]] = []
    if "cuda" in selected:
        jobs.append(("cuda-qcu", child_command("qcu_bistabcg.py", [
            *common, "--atol", str(args.atol),
            "--grid", *(str(value) for value in cuda_grid),
            "--ranks", str(cuda_ranks),
            "--report", child_report(args.report, "cuda-qcu")])))
    if "kupl" in selected:
        jobs.append(("kupl-qcu", child_command("kupl_bistabcg.py", [
            *common, "--atol", str(args.atol),
            "--grid", *(str(value) for value in kupl_grid),
            "--ranks", str(kupl_ranks),
            "--threads", str(args.kupl_threads),
            "--report", child_report(args.report, "kupl-qcu")])))
    if "quda" in selected:
        jobs.append(("quda", child_command("quda_bistabcg.py", [
            *common, "--tol", str(args.atol),
            "--grid", *(str(value) for value in quda_grid),
            "--ranks", str(quda_ranks),
            "--report", child_report(args.report, "quda")])))
    return run_jobs(
        jobs, report=args.report, dry_run=args.dry_run, timeout=args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
