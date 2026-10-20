#!/usr/bin/env python3
"""Run the KUPL and QUDA Clover BiStabCG examples against one common input."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from ._common import (DATA_DIR, DEFAULT_ATOL, read_h5, resolve_path,
                          write_h5, write_json)
except ImportError:
    from _common import (DATA_DIR, DEFAULT_ATOL, read_h5, resolve_path,
                         write_h5, write_json)


def random_su3_gauge(lattice: tuple[int, int, int, int],
                     seed: int) -> np.ndarray:
    """Generate deterministic Haar-like SU(3) links without a backend."""
    rng = np.random.default_rng(seed)
    x_dim, y_dim, z_dim, t_dim = lattice
    output = np.empty((3, 3, 4, *lattice), dtype=np.complex64)
    for x in range(x_dim):
        for y in range(y_dim):
            for z in range(z_dim):
                for t in range(t_dim):
                    for direction in range(4):
                        vector = rng.normal(size=(3, 3)) + \
                            1j * rng.normal(size=(3, 3))
                        hermitian = vector + vector.conj().T
                        hermitian -= np.trace(hermitian) / 3.0 * np.eye(3)
                        eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
                        link = (eigenvectors @ np.diag(np.exp(1j * eigenvalues))
                                @ eigenvectors.conj().T)
                        output[:, :, direction, x, y, z, t] = link
    return output


def run_backend(script: Path, command: list[str], timeout: float,
                environment: dict[str, str] | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command, cwd=script.parent, text=True, capture_output=True,
            timeout=timeout, check=False, env=environment)
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command, "returncode": 124,
            "elapsed_seconds": time.perf_counter() - started,
            "stdout": exc.stdout or "",
            "stderr": f"timeout after {timeout:g} seconds\n{exc.stderr or ''}",
        }
    return {
        "command": command, "returncode": int(completed.returncode),
        "elapsed_seconds": time.perf_counter() - started,
        "stdout": completed.stdout, "stderr": completed.stderr,
    }


def load_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def relative_difference(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError(f"solution shape mismatch: {left.shape} != {right.shape}")
    numerator = np.linalg.norm(left.astype(np.complex128) -
                               right.astype(np.complex128))
    denominator = np.linalg.norm(right.astype(np.complex128))
    return float(numerator / max(float(denominator), np.finfo(float).tiny))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gauge", default=None,
                        help="shared .lime or QCU HDF5 gauge; generated if omitted")
    parser.add_argument("--lat", nargs=4, type=int, default=(4, 4, 4, 4))
    parser.add_argument("--mass", type=float, default=0.17)
    parser.add_argument("--csw", type=float, default=0.8)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kupl-threads", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--solution-rtol", type=float, default=5.0e-3)
    parser.add_argument("--output", default="performance_simple_kupl_quda_bistabcg.json")
    args = parser.parse_args(argv)
    directory = Path(__file__).resolve().parent
    output = resolve_path(args.output, DATA_DIR / args.output)
    run_tag = f"{os.getpid()}-{time.time_ns()}"
    gauge_path = output.with_name(f"{output.stem}.{run_tag}.gauge.h5") \
        if args.gauge is None else resolve_path(args.gauge, resolve_path(None, args.gauge))
    if args.gauge is None:
        write_h5(gauge_path, random_su3_gauge(
            tuple(args.lat), args.seed), metadata={
                "layout": "[3,3,4,x,y,z,t]", "csw": args.csw,
                "reference_mass": args.mass, "seed": args.seed,
                "generator": "independent NumPy SU(3) links",
            })

    source_path = output.with_name(f"{output.stem}.{run_tag}.source.h5")
    rng = np.random.default_rng(args.seed + 1)
    source = (rng.normal(size=(4, 3, *args.lat)) +
              1j * rng.normal(size=(4, 3, *args.lat))).astype(np.complex64)
    write_h5(source_path, source, metadata={
        "layout": "[spin,color,x,y,z,t]", "seed": args.seed + 1})

    kupl_report = output.with_name(f"{output.stem}.{run_tag}.kupl.json")
    quda_report = output.with_name(f"{output.stem}.{run_tag}.quda.json")
    kupl_solution = output.with_name(f"{output.stem}.{run_tag}.kupl.h5")
    quda_solution = output.with_name(f"{output.stem}.{run_tag}.quda.h5")
    common = [
        "--lat", *(str(value) for value in args.lat),
        "--gauge", str(gauge_path), "--source", str(source_path),
        "--mass", str(args.mass), "--csw", str(args.csw),
        "--seed", str(args.seed),
    ]
    kupl_command = [
        sys.executable, "-B", str(directory / "kupl_bistabcg.py"), *common,
        "--dtype", "c64", "--atol", str(args.atol),
        "--max-iter", str(args.max_iter), "--report", str(kupl_report),
        "--output", str(kupl_solution),
    ]
    environment = os.environ.copy()
    environment["KUPL_OMP_THREADS"] = str(args.kupl_threads)
    environment["OMP_NUM_THREADS"] = str(args.kupl_threads)
    kupl_result = run_backend(directory / "kupl_bistabcg.py",
                              kupl_command, args.timeout, environment)
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

    compatibility: dict[str, bool] = {}
    solution_difference = None
    numerical_match = False
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
        for label, data, expected in (
                ("KUPL", kupl_data, "kupl"), ("QUDA", quda_data, "quda")):
            if data.get("backend") != expected:
                errors.append(f"{label} report backend={data.get('backend')!r}")
            if data.get("status") != "PASS":
                errors.append(f"{label} status={data.get('status')!r}")
        try:
            kupl_array = read_h5(kupl_solution)
            quda_array = read_h5(quda_solution)
            solution_difference = relative_difference(kupl_array, quda_array)
            numerical_match = solution_difference <= args.solution_rtol
            if not numerical_match:
                errors.append(
                    f"solution mismatch: relative={solution_difference:.6e} "
                    f"> {args.solution_rtol:.6e}")
        except (FileNotFoundError, OSError, ValueError) as exc:
            errors.append(f"solution comparison unavailable: {exc}")

    status = "PASS" if not errors and numerical_match else "FAIL"
    payload = {
        "status": status, "compatible": all(compatibility.values()),
        "compatibility": compatibility, "errors": errors,
        "input": {
            "lattice": list(args.lat), "gauge": str(gauge_path),
            "source": str(source_path), "mass": args.mass, "csw": args.csw,
            "atol": args.atol, "max_iter": args.max_iter,
            "seed": args.seed, "solution_rtol": args.solution_rtol,
            "kupl_threads": args.kupl_threads,
        },
        "kupl": {"process": kupl_result, "report": kupl_data},
        "quda": {"process": quda_result, "report": quda_data},
        "numerical_match": numerical_match,
        "solution_relative_difference": solution_difference,
        "solutions": {"kupl": str(kupl_solution), "quda": str(quda_solution)},
    }
    write_json(output, payload)
    print(json.dumps({
        "status": status, "compatible": payload["compatible"],
        "solution_relative_difference": solution_difference,
        "kupl_returncode": kupl_result["returncode"],
        "quda_returncode": quda_result["returncode"],
        "output": str(output), "errors": errors,
    }, ensure_ascii=False))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
