#!/usr/bin/env python3
"""Orchestrate isolated MPI CUDA-QCU, KUPL and QUDA benchmark jobs."""

from __future__ import annotations

import argparse
import contextlib
import io
import os
from collections.abc import Sequence

_mpi_rc_initialize = os.environ.get("MPI4PY_RC_INITIALIZE")
os.environ["MPI4PY_RC_INITIALIZE"] = "0"
try:
    with contextlib.redirect_stdout(io.StringIO()):
        from pyqcu.testing.simple._common import DEFAULT_LIME
        from pyqcu.testing.simple.mpi._compare import (
            child_report,
            child_command,
            require_orchestrator_singleton,
            run_jobs,
            validate_rank_grid,
        )
finally:
    if _mpi_rc_initialize is None:
        os.environ.pop("MPI4PY_RC_INITIALIZE", None)
    else:
        os.environ["MPI4PY_RC_INITIALIZE"] = _mpi_rc_initialize


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gauge", default=str(DEFAULT_LIME))
    parser.add_argument("--source", default=None)
    parser.add_argument("--lat", nargs=4, type=int, default=(8, 8, 8, 16))
    parser.add_argument("--mass", type=float, default=None)
    parser.add_argument("--csw", type=float, default=None)
    parser.add_argument("--dtype", choices=("c64", "c128"), default="c64")
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
    parser.add_argument(
        "--matrix", action="store_true",
        help="use the bounded CUDA-QCU/QUDA 1/2/4-rank manifest runner")
    parser.add_argument("--rank-case", action="append", default=[],
                        help="matrix rank:gx,gy,gz,gt case")
    parser.add_argument("--result-root", default=None)
    parser.add_argument("--wall-limit", type=float, default=570.0)
    parser.add_argument("--slurm-time-limit", type=float, default=600.0)
    parser.add_argument("--child-timeout", type=float, default=180.0)
    parser.add_argument("--mpirun", default="mpirun")
    parser.add_argument("--defer-independent-residual", action="store_true")
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
    if args.matrix:
        if "kupl" in selected or not selected:
            parser.error("--matrix supports only cuda and/or quda")
        from pyqcu.testing.simple.mpi.cuda_quda_matrix import main as matrix_main

        matrix_backends = [
            "qcu" if backend == "cuda" else "quda"
            for backend in ("cuda", "quda") if backend in selected]
        matrix_arguments = [
            "--gauge", args.gauge,
            "--lat", *(str(value) for value in args.lat),
            "--dtype", args.dtype, "--atol", str(args.atol),
            "--max-iter", str(args.max_iter), "--warmups", str(args.warmups),
            "--repeats", str(args.repeats), "--seed", str(args.seed),
            "--backends", ",".join(matrix_backends),
            "--wall-limit", str(args.wall_limit),
            "--slurm-time-limit", str(args.slurm_time_limit),
            "--child-timeout", str(args.child_timeout),
            "--mpirun", args.mpirun,
        ]
        if args.source is not None:
            matrix_arguments.extend(("--source", args.source))
        if args.mass is not None:
            matrix_arguments.extend(("--mass", str(args.mass)))
        if args.csw is not None:
            matrix_arguments.extend(("--csw", str(args.csw)))
        if args.result_root is not None:
            matrix_arguments.extend(("--result-root", args.result_root))
        for case in args.rank_case:
            matrix_arguments.extend(("--rank-case", case))
        if args.ranks is not None and not args.rank_case:
            matrix_arguments.extend(("--rank-case", f"{args.ranks}:"
                                     + ",".join("1" for _ in range(3))
                                     + f",{args.ranks}"))
        if args.dry_run:
            matrix_arguments.append("--dry-run")
        if args.defer_independent_residual:
            matrix_arguments.append("--defer-independent-residual")
        return int(matrix_main(matrix_arguments) or 0)
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
