#!/usr/bin/env python3
"""Validate KUPL multigrid against QUDA BiStabCG on one common system."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    from ._common import (DATA_DIR, DEFAULT_ATOL, read_h5, resolve_path,
                          write_h5, write_json)
    from .compare_kupl_bistabcg import (load_report, random_su3_gauge,
                                        relative_difference, run_backend)
except ImportError:
    from _common import (DATA_DIR, DEFAULT_ATOL, read_h5, resolve_path,
                         write_h5, write_json)
    from compare_kupl_bistabcg import (load_report, random_su3_gauge,
                                       relative_difference, run_backend)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gauge", default=None)
    parser.add_argument("--lat", nargs=4, type=int, default=(4, 4, 4, 4))
    parser.add_argument("--mass", type=float, default=0.17)
    parser.add_argument("--csw", type=float, default=0.8)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--levels", type=int, choices=(1, 2), default=2)
    parser.add_argument("--dof", nargs="+", type=int, default=(12, 2))
    parser.add_argument("--mg-grid", nargs=4, type=int, default=(2, 2, 2, 2))
    parser.add_argument("--restart", type=int, default=5)
    parser.add_argument("--coarse-max-iter", type=int, default=200)
    parser.add_argument("--nv-iters", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kupl-threads", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--solution-rtol", type=float, default=5.0e-3)
    parser.add_argument("--output", default="performance_simple_kupl_quda_multigrid.json")
    args = parser.parse_args(argv)
    directory = Path(__file__).resolve().parent
    output = resolve_path(args.output, DATA_DIR / args.output)
    tag = f"{os.getpid()}-{time.time_ns()}"
    gauge_path = output.with_name(f"{output.stem}.{tag}.gauge.h5") \
        if args.gauge is None else resolve_path(args.gauge, resolve_path(None, args.gauge))
    if args.gauge is None:
        write_h5(gauge_path, random_su3_gauge(
            tuple(args.lat), args.seed), metadata={
                "layout": "[3,3,4,x,y,z,t]", "csw": args.csw,
                "reference_mass": args.mass, "seed": args.seed,
                "generator": "independent NumPy SU(3) links",
            })
    source_path = output.with_name(f"{output.stem}.{tag}.source.h5")
    rng = np.random.default_rng(args.seed + 1)
    source = (rng.normal(size=(4, 3, *args.lat)) +
              1j * rng.normal(size=(4, 3, *args.lat))).astype(np.complex64)
    write_h5(source_path, source, metadata={
        "layout": "[spin,color,x,y,z,t]", "seed": args.seed + 1})

    kupl_report = output.with_name(f"{output.stem}.{tag}.kupl.json")
    quda_report = output.with_name(f"{output.stem}.{tag}.quda.json")
    kupl_solution = output.with_name(f"{output.stem}.{tag}.kupl.h5")
    quda_solution = output.with_name(f"{output.stem}.{tag}.quda.h5")
    common = [
        "--lat", *(str(value) for value in args.lat),
        "--gauge", str(gauge_path), "--source", str(source_path),
        "--mass", str(args.mass), "--csw", str(args.csw),
        "--seed", str(args.seed),
    ]
    environment = os.environ.copy()
    environment["KUPL_OMP_THREADS"] = str(args.kupl_threads)
    environment["OMP_NUM_THREADS"] = str(args.kupl_threads)
    kupl_result = run_backend(
        directory / "kupl_multigrid.py",
        [sys.executable, "-B", str(directory / "kupl_multigrid.py"), *common,
         "--dtype", "c64", "--atol", str(args.atol),
         "--max-iter", str(args.max_iter), "--levels", str(args.levels),
         "--dof", *(str(value) for value in args.dof),
         "--mg-grid", *(str(value) for value in args.mg_grid),
         "--restart", str(args.restart),
         "--coarse-max-iter", str(args.coarse_max_iter),
         "--nv-iters", str(args.nv_iters), "--report", str(kupl_report),
         "--output", str(kupl_solution)],
        args.timeout, environment)
    quda_result = run_backend(
        directory / "quda_bistabcg.py",
        [sys.executable, "-B", str(directory / "quda_bistabcg.py"), *common,
         "--tol", str(args.atol), "--max-iter", str(args.max_iter),
         "--report", str(quda_report), "--output", str(quda_solution)],
        args.timeout)

    errors: list[str] = []
    kupl_data = quda_data = None
    if kupl_result["returncode"] == 0:
        try:
            kupl_data = load_report(kupl_report)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            errors.append(f"KUPL report unavailable: {exc}")
    else:
        errors.append(f"KUPL process failed with code {kupl_result['returncode']}")
    if quda_result["returncode"] == 0:
        try:
            quda_data = load_report(quda_report)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            errors.append(f"QUDA report unavailable: {exc}")
    else:
        errors.append(f"QUDA process failed with code {quda_result['returncode']}")

    compatibility = {}
    difference = None
    if kupl_data is not None and quda_data is not None:
        for key in ("lattice", "mass", "csw", "mass_normalization"):
            left, right = kupl_data.get(key), quda_data.get(key)
            if key == "lattice":
                same = list(left or []) == list(right or [])
            elif key == "mass_normalization":
                same = left == right
            elif left is None or right is None:
                same = False
            else:
                same = math.isclose(float(left), float(right), rel_tol=1e-7,
                                    abs_tol=1e-9)
            compatibility[key] = bool(same)
            if not same:
                errors.append(f"{key} mismatch: KUPL={left!r}, QUDA={right!r}")
        if kupl_data.get("status") != "PASS":
            errors.append(f"KUPL status={kupl_data.get('status')!r}")
        if kupl_data.get("verify_status") not in (None, 0):
            errors.append(f"KUPL MG verify={kupl_data.get('verify_status')!r}")
        if quda_data.get("status") != "PASS":
            errors.append(f"QUDA status={quda_data.get('status')!r}")
        try:
            difference = relative_difference(
                read_h5(kupl_solution), read_h5(quda_solution))
            if difference > args.solution_rtol:
                errors.append(
                    f"solution mismatch: relative={difference:.6e} "
                    f"> {args.solution_rtol:.6e}")
        except (FileNotFoundError, OSError, ValueError) as exc:
            errors.append(f"solution comparison unavailable: {exc}")

    status = "PASS" if not errors and difference is not None else "FAIL"
    payload = {
        "status": status, "compatible": all(compatibility.values()),
        "compatibility": compatibility, "errors": errors,
        "input": {"lattice": list(args.lat), "gauge": str(gauge_path),
                  "source": str(source_path), "mass": args.mass,
                  "csw": args.csw, "atol": args.atol,
                  "max_iter": args.max_iter, "levels": args.levels,
                  "dof": args.dof[:args.levels],
                  "mg_grid": args.mg_grid, "seed": args.seed},
        "kupl": {"process": kupl_result, "report": kupl_data},
        "quda": {"process": quda_result, "report": quda_data},
        "solution_relative_difference": difference,
    }
    write_json(output, payload)
    print(json.dumps({
        "status": status, "compatible": payload["compatible"],
        "solution_relative_difference": difference,
        "kupl_returncode": kupl_result["returncode"],
        "quda_returncode": quda_result["returncode"],
        "output": str(output), "errors": errors,
    }, ensure_ascii=False))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
