#!/usr/bin/env python3
"""Exercise the distributed KUPL true-residual reduction on a synthetic case."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from pyqcu.testing.simple._common import write_json
from pyqcu.testing.simple.kupl_residual import distributed_relative_residual
from pyqcu.testing.simple.mpi._common import local_geometry, resolve_grid
from pyqcu.testing.simple.mpi.mpi_ranks_power_of_two import validate_rank_count


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lat", nargs=4, type=int, default=(4, 4, 4, 4))
    parser.add_argument("--grid", nargs=4, type=int, default=None)
    parser.add_argument("--ranks", type=int, default=None,
                        help="expected MPI_COMM_WORLD size")
    parser.add_argument("--report", default="kupl_residual_mpi.json")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    from mpi4py import MPI

    args = _parse_args(argv)
    comm = MPI.COMM_WORLD
    rank = int(comm.Get_rank())
    try:
        size = validate_rank_count(int(comm.Get_size()))
        if args.ranks is not None and args.ranks != size:
            raise ValueError(
                f"--ranks={args.ranks} does not match MPI_COMM_WORLD size {size}")
        grid = resolve_grid(args.lat, size, args.grid)
        local, starts, coordinate = local_geometry(args.lat, grid, rank)
    except ValueError as exc:
        if rank == 0:
            print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        if rank == 0:
            print(json.dumps({
                "status": "DRY-RUN",
                "mode": "native-mpi",
                "operation": "distributed_relative_residual",
                "ranks": size,
                "grid": list(grid),
                "lattice": list(args.lat),
                "report": str(args.report),
            }, ensure_ascii=False))
        return 0

    half_time = local[3] // 2
    solution = np.zeros((2, 12, *local[:3], half_time), dtype=np.complex64)
    source = np.zeros_like(solution)
    gauge = np.zeros((2, 3, 3, 4, *local[:3], half_time),
                     dtype=np.complex64)
    clover = np.zeros((4, 3, 4, 3, *local[:3], half_time),
                      dtype=np.complex64)
    residual = distributed_relative_residual(
        solution, source, gauge, clover, clover, 0.1,
        comm, coordinate, grid)
    finite = bool(np.isfinite(residual))
    failed = int(comm.allreduce(
        int(not finite or residual != 0.0), op=MPI.MAX))
    status = "PASS" if not failed else "FAIL"
    report = {
        "status": status,
        "mode": "synthetic-native-mpi",
        "operation": "distributed_relative_residual",
        "ranks": size,
        "grid": list(grid),
        "lattice": list(args.lat),
        "local_lattice": list(local),
        "local_starts": list(starts),
        "residual": float(residual),
        "note": (
            "Synthetic zero field verifies rank geometry, periodic halo "
            "sendrecv symmetry, and Allreduce; it is not a solver benchmark."),
    }
    if rank == 0:
        output = Path(args.report).expanduser()
        temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
        write_json(temporary, report)
        os.replace(temporary, output)
        print(json.dumps(report, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
