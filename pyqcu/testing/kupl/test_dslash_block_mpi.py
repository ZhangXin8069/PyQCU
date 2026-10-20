#!/usr/bin/env python3
"""Direct MPI regression for the KUPL block Wilson-dslash fast path."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "cpp/kupl/qcu/tests"))
import test_new_abi as reference  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--lattice", nargs=4, type=int, default=(4, 4, 8, 8))
    parser.add_argument("--grid", nargs=4, type=int, default=(2, 1, 1, 1))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--solve", action="store_true")
    parser.add_argument("--precision", choices=("c64", "c128"), default="c64")
    args = parser.parse_args()
    comm = MPI.COMM_WORLD
    if int(np.prod(args.grid)) != comm.size:
        raise RuntimeError("grid product must equal MPI size")
    rank = comm.Get_rank()
    coordinate = tuple(
        int(value) for value in np.unravel_index(rank, tuple(args.grid), order="C")
    )
    grid = reference.Grid(tuple(args.lattice), tuple(args.grid), coordinate,
                          rank, comm.size)
    library = reference.bind_library(str(args.library))
    global_gauge, global_source = reference.make_global_state(
        grid, np.dtype(np.complex128))
    dtype = np.dtype(np.complex64 if args.precision == "c64" else np.complex128)
    data_type = 2 if args.precision == "c64" else 3
    tolerance = 5.0e-4 if args.precision == "c64" else 2.0e-9
    global_gauge = global_gauge.astype(dtype)
    global_source = global_source.astype(dtype)
    gauge = reference.gauge_to_local(global_gauge, grid)
    source = reference.fermion_to_local(global_source, grid)
    target_parity = 1
    params, argv, set_ptrs = reference.params_for(
        grid, data_type, 0.1, 0.0, 0, target_parity)
    library.applyInitQcu(reference.pointer(set_ptrs),
                         reference.pointer(params),
                         reference.pointer(argv))
    output = np.zeros_like(source[1 - target_parity])
    try:
        library.applyWilsonDslashQcu(
            reference.pointer(output),
            reference.pointer(source[1 - target_parity]),
            reference.pointer(gauge),
            reference.pointer(set_ptrs),
            reference.pointer(params),
        )
    finally:
        library.applyEndQcu(reference.pointer(set_ptrs),
                            reference.pointer(params))
    output_global = reference.local_compact_to_global(
        output, grid, target_parity).reshape((-1, 12))
    expected = reference.bare_wilson(
        global_source, global_gauge, target_parity).reshape((-1, 12))
    mask = (np.indices(grid.global_shape).sum(axis=0) & 1).reshape(-1) == \
        target_parity
    expected[~mask] = 0
    difference = output_global - expected
    numerator = comm.allreduce(
        float(np.vdot(difference, difference).real), op=MPI.SUM)
    denominator = comm.allreduce(
        float(np.vdot(expected, expected).real), op=MPI.SUM)
    relative = math.sqrt(numerator / max(denominator, 1.0e-30))

    params[16] = 1
    schur_input = np.ascontiguousarray(source[target_parity])
    schur_output = np.zeros_like(schur_input)
    normal_output = np.zeros_like(schur_input)
    library.applyInitQcu(reference.pointer(set_ptrs),
                         reference.pointer(params),
                         reference.pointer(argv))
    try:
        library.applyWilsonBistabCgDslashQcu(
            reference.pointer(schur_output),
            reference.pointer(schur_input),
            reference.pointer(gauge),
            reference.pointer(set_ptrs),
            reference.pointer(params),
        )
        library.applyWilsonCgDslashQcu(
            reference.pointer(normal_output),
            reference.pointer(schur_input),
            reference.pointer(gauge),
            reference.pointer(set_ptrs),
            reference.pointer(params),
        )
    finally:
        library.applyEndQcu(reference.pointer(set_ptrs),
                            reference.pointer(params))

    input_global = reference.local_compact_to_global(
        schur_input, grid, target_parity).reshape(
            (*grid.global_shape, 4, 3))
    schur_expected_global = reference.wilson_schur_global(
        input_global, global_gauge, target_parity, 0.1)
    normal_first = reference.wilson_schur_global(
        input_global, global_gauge, target_parity, 0.1)
    normal_expected_global = reference.wilson_schur_dagger_global(
        normal_first, global_gauge, target_parity, 0.1)

    def compact_error(output_local, expected_global):
        actual = reference.local_compact_to_global(
            output_local, grid, target_parity).reshape((-1, 12))
        expected = expected_global.reshape((-1, 12)).copy()
        expected[~mask] = 0
        diff = actual - expected
        num = comm.allreduce(float(np.vdot(diff, diff).real), op=MPI.SUM)
        den = comm.allreduce(
            float(np.vdot(expected, expected).real), op=MPI.SUM)
        return math.sqrt(num / max(den, 1.0e-30))

    schur_relative = compact_error(schur_output, schur_expected_global)
    normal_relative = compact_error(normal_output, normal_expected_global)
    solve_relative = 0.0
    if args.solve:
        params[13] = 1000
        params[16] = 1
        solution = np.zeros_like(source.reshape(-1))
        library.applyInitQcu(reference.pointer(set_ptrs),
                             reference.pointer(params),
                             reference.pointer(argv))
        try:
            library.applyWilsonBistabCgQcu(
                reference.pointer(solution),
                reference.pointer(source),
                reference.pointer(gauge),
                reference.pointer(set_ptrs),
                reference.pointer(params),
            )
        finally:
            library.applyEndQcu(reference.pointer(set_ptrs),
                                reference.pointer(params))
        solution_global = reference.local_full_to_global(
            solution, grid).reshape((*grid.global_shape, 4, 3))
        applied = reference.full_wilson(solution_global, global_gauge, 0.1)
        residual = applied.reshape((-1, 12)) - global_source.reshape((-1, 12))
        num = comm.allreduce(float(np.vdot(residual, residual).real),
                             op=MPI.SUM)
        den = comm.allreduce(
            float(np.vdot(global_source.reshape((-1, 12)),
                          global_source.reshape((-1, 12))).real), op=MPI.SUM)
        solve_relative = math.sqrt(num / max(den, 1.0e-30))
    passed = max(relative, schur_relative, normal_relative,
                 solve_relative) < tolerance
    if rank == 0:
        print(f"status={'PASS' if passed else 'FAIL'} "
              f"ranks={comm.size} threads={args.threads} "
              f"dslash={relative:.9e} schur={schur_relative:.9e} "
              f"normal={normal_relative:.9e} solve={solve_relative:.9e} "
              f"precision={args.precision}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
