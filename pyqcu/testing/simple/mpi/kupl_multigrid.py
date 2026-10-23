#!/usr/bin/env python3
"""KUPL MPI multigrid entry using the established simple MPI worker."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from pyqcu.testing.simple.mpi.solve_case import main as solve_case_main


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Arguments are parsed by the shared simple MPI solve_case.py")
    parser.add_argument("--gauge", default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--lat", nargs=4, type=int, default=(8, 8, 16, 16))
    parser.add_argument("--mass", type=float, default=None)
    parser.add_argument("--csw", type=float, default=None)
    parser.add_argument("--atol", type=float, default=1.0e-6)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--dtype", choices=("c64", "c128"), default="c64")
    parser.add_argument("--levels", type=int, choices=(1, 2), default=2)
    parser.add_argument("--dof", nargs="+", type=int, default=(12, 12))
    parser.add_argument("--mg-grid", nargs=4, type=int, default=(2, 2, 2, 2))
    parser.add_argument("--restart", type=int, default=5)
    parser.add_argument("--coarse-max-iter", type=int, default=200)
    parser.add_argument("--coarse-tol-factor", type=float, default=1.0e4)
    parser.add_argument("--nv-iters", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ranks", type=int, default=2)
    parser.add_argument("--grid", nargs=4, type=int, default=None)
    parser.add_argument("--omp-threads", type=int, default=1)
    parser.add_argument("--bind-to", choices=("numa", "none"), default=None)
    parser.add_argument("--map-by", default=None)
    parser.add_argument("--oversubscribe", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--allow-run-as-root", action=argparse.BooleanOptionalAction,
                        default=None)
    parser.add_argument("--mpirun", default="mpirun")
    parser.add_argument("--coarse-cache", default=None)
    parser.add_argument("--coarse-cache-level", action="append", default=[])
    parser.add_argument("--rebuild-coarse-cache", action="store_true")
    parser.add_argument("--skip-independent-residual", action="store_true")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    if len(args.dof) < args.levels:
        parser.error("--dof must provide one value per requested level")
    forwarded = [
        "--backend", "kupl",
        "--lat", *(str(value) for value in args.lat),
        "--atol", str(args.atol),
        "--max-iter", str(args.max_iter),
        "--dtype", args.dtype,
        "--levels", str(args.levels),
        "--dof", *(str(value) for value in args.dof[:args.levels]),
        "--mg-grid", *(str(value) for value in args.mg_grid),
        "--restart", str(args.restart),
        "--coarse-max-iter", str(args.coarse_max_iter),
        "--coarse-tol-factor", str(args.coarse_tol_factor),
        "--nv-iters", str(args.nv_iters),
        "--seed", str(args.seed),
        "--ranks", str(args.ranks),
        "--omp-threads", str(args.omp_threads),
        "--mpirun", str(args.mpirun),
    ]
    if args.mass is not None:
        forwarded.extend(("--mass", str(args.mass)))
    if args.csw is not None:
        forwarded.extend(("--csw", str(args.csw)))
    for flag, value in (
            ("--gauge", args.gauge), ("--source", args.source),
            ("--grid", args.grid), ("--bind-to", args.bind_to),
            ("--map-by", args.map_by), ("--coarse-cache", args.coarse_cache),
            ("--work-dir", args.work_dir), ("--output", args.output),
            ("--report", args.report)):
        if value is None:
            continue
        if flag == "--grid":
            forwarded.extend((flag, *(str(item) for item in value)))
        else:
            forwarded.extend((flag, str(value)))
    if args.allow_run_as_root is not None:
        forwarded.append("--allow-run-as-root" if args.allow_run_as_root
                         else "--no-allow-run-as-root")
    forwarded.append("--oversubscribe" if args.oversubscribe
                     else "--no-oversubscribe")
    if args.rebuild_coarse_cache:
        forwarded.append("--rebuild-coarse-cache")
    if args.skip_independent_residual:
        forwarded.append("--skip-independent-residual")
    if args.dry_run:
        forwarded.append("--dry-run")
    return int(solve_case_main(forwarded) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
