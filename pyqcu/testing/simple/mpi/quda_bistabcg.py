#!/usr/bin/env python3
"""Launch the native MPI QUDA/PyQUDA Clover BiCGStab benchmark."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from pyqcu.testing.simple._common import DEFAULT_LIME
from pyqcu.testing.simple.mpi._entrypoint import (
    add_mpi_job_options,
    finalize_mpi_job,
    run_delegated_mpi,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gauge", default=str(DEFAULT_LIME))
    parser.add_argument("--source", default=None)
    parser.add_argument("--mass", type=float, default=None)
    parser.add_argument("--csw", type=float, default=None)
    parser.add_argument("--dtype", choices=("c64", "c128"), default="c64")
    parser.add_argument("--tol", type=float, default=1.0e-6)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    add_mpi_job_options(
        parser, lattice=(8, 8, 8, 16), ranks=2,
        report="quda_bistabcg_mpi.json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = finalize_mpi_job(parser, parser.parse_args(argv))
    forwarded = [
        "--gauge", str(args.gauge),
        "--solver", "bicgstab",
        "--lat", *(str(value) for value in args.lat),
        "--dtype", args.dtype,
        "--tol", str(args.tol),
        "--max-iter", str(args.max_iter),
        "--warmups", str(args.warmups),
        "--repeats", str(args.repeats),
        "--seed", str(args.seed),
    ]
    if args.source is not None:
        forwarded.extend(("--source", str(args.source)))
    if args.mass is not None:
        forwarded.extend(("--mass", str(args.mass)))
    if args.csw is not None:
        forwarded.extend(("--csw", str(args.csw)))
    return run_delegated_mpi(
        "pyqcu.testing.quda.benchmark_quda_lime_mpi", args, forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
