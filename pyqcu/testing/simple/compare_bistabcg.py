#!/usr/bin/env python3
"""Run the minimal QCU and QUDA Clover BiCGStab examples in isolation."""

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


def _load_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _run_backend(script: Path, command: list[str], timeout: float) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command, cwd=script.parent, text=True, capture_output=True,
            timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": 124,
            "stdout": exc.stdout or "",
            "stderr": f"timeout after {timeout:g} seconds\n{exc.stderr or ''}",
        }
    return {
        "command": command,
        "returncode": int(completed.returncode),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _relative_solution_difference(
        qcu_solution: np.ndarray, quda_solution: np.ndarray) -> float:
    if qcu_solution.shape != quda_solution.shape:
        raise ValueError(
            f"solution shape mismatch: QCU={qcu_solution.shape}, "
            f"QUDA={quda_solution.shape}")
    numerator = float(np.linalg.norm(
        qcu_solution.astype(np.complex128) -
        quda_solution.astype(np.complex128)))
    denominator = float(np.linalg.norm(quda_solution.astype(np.complex128)))
    return numerator / max(denominator, np.finfo(np.float64).tiny)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gauge", required=True,
                        help="shared .lime or QCU HDF5 gauge configuration")
    parser.add_argument("--lat", nargs=4, type=int, required=True)
    parser.add_argument("--mass", type=float, default=None)
    parser.add_argument("--csw", type=float, default=None)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--qcu-device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--solution-rtol", type=float, default=5.0e-3,
                        help="maximum QCU/QUDA relative solution difference")
    parser.add_argument("--output", default="performance_simple_bistabcg_pair.json")
    args = parser.parse_args(argv)

    directory = Path(__file__).resolve().parent
    output = resolve_path(args.output, DATA_DIR / args.output)
    run_tag = f"{os.getpid()}-{time.time_ns()}"
    qcu_report = output.with_name(f"{output.stem}.{run_tag}.qcu.json")
    quda_report = output.with_name(f"{output.stem}.{run_tag}.quda.json")
    source_path = output.with_name(f"{output.stem}.{run_tag}.source.h5")
    qcu_solution = output.with_name(f"{output.stem}.{run_tag}.qcu.h5")
    quda_solution = output.with_name(f"{output.stem}.{run_tag}.quda.h5")
    rng = np.random.default_rng(args.seed)
    source = (
        rng.normal(size=(4, 3, *args.lat)) +
        1j * rng.normal(size=(4, 3, *args.lat))
    ).astype(np.complex64)
    write_h5(
        source_path, source,
        metadata={"layout": "[spin,color,x,y,z,t]", "seed": args.seed})
    common = [
        "--lat", *(str(value) for value in args.lat),
        "--gauge", str(resolve_path(args.gauge, resolve_path(None, args.gauge))),
        "--source", str(source_path),
        "--seed", str(args.seed),
    ]
    if args.mass is not None:
        common.extend(["--mass", str(args.mass)])
    if args.csw is not None:
        common.extend(["--csw", str(args.csw)])

    qcu_result = _run_backend(
        directory / "qcu_bistabcg.py",
        [sys.executable, "-B", str(directory / "qcu_bistabcg.py"), *common,
         "--atol", str(args.atol), "--max-iter", str(args.max_iter),
         "--device", args.qcu_device, "--report", str(qcu_report),
         "--output", str(qcu_solution)],
        args.timeout)
    quda_result = _run_backend(
        directory / "quda_bistabcg.py",
        [sys.executable, "-B", str(directory / "quda_bistabcg.py"), *common,
         "--tol", str(args.atol), "--max-iter", str(args.max_iter),
         "--report", str(quda_report), "--output", str(quda_solution)],
        args.timeout)

    errors = []
    qcu_data = quda_data = None
    if qcu_result["returncode"] == 0:
        try:
            qcu_data = _load_report(qcu_report)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            errors.append(f"QCU report unavailable: {exc}")
    else:
        errors.append(f"QCU process failed with code {qcu_result['returncode']}")
    if quda_result["returncode"] == 0:
        try:
            quda_data = _load_report(quda_report)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            errors.append(f"QUDA report unavailable: {exc}")
    else:
        errors.append(f"QUDA process failed with code {quda_result['returncode']}")

    compatible = False
    solution_difference = None
    numerical_match = False
    if qcu_data is not None and quda_data is not None:
        for key in ("lattice", "mass", "csw"):
            left, right = qcu_data.get(key), quda_data.get(key)
            if key == "lattice":
                same = list(left or []) == list(right or [])
            elif left is None or right is None:
                same = False
            else:
                same = math.isclose(float(left), float(right), rel_tol=1.0e-7,
                                    abs_tol=1.0e-9)
            if not same:
                errors.append(f"{key} mismatch: QCU={left!r}, QUDA={right!r}")
        compatible = not errors
        expected_backend = {"qcu": "qcu", "quda": "quda"}
        for label, data in (("qcu", qcu_data), ("quda", quda_data)):
            if data.get("backend") != expected_backend[label]:
                errors.append(
                    f"{label} report backend={data.get('backend')!r}")
            if data.get("status") == "SKIP":
                errors.append(f"{label} unexpectedly returned SKIP")
        try:
            qcu_array = read_h5(qcu_solution)
            quda_array = read_h5(quda_solution)
            solution_difference = _relative_solution_difference(
                qcu_array, quda_array)
            numerical_match = solution_difference <= args.solution_rtol
            if not numerical_match:
                errors.append(
                    f"solution mismatch: relative={solution_difference:.6e} "
                    f"> {args.solution_rtol:.6e}")
        except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
            errors.append(f"solution comparison unavailable: {exc}")

    qcu_ok = (qcu_result["returncode"] == 0
              and qcu_data is not None
              and qcu_data.get("status") == "PASS")
    quda_ok = (quda_result["returncode"] == 0
               and quda_data is not None
               and quda_data.get("status") == "PASS")
    compatible = compatible and not errors
    status = ("PASS" if qcu_ok and quda_ok and compatible and numerical_match
              else "FAIL")
    payload = {
        "status": status,
        "compatible": compatible,
        "input": {
            "lattice": list(args.lat),
            "gauge": str(resolve_path(args.gauge, resolve_path(None, args.gauge))),
            "mass": args.mass,
            "csw": args.csw,
            "atol": args.atol,
            "max_iter": args.max_iter,
            "seed": args.seed,
            "solution_rtol": args.solution_rtol,
            "source": str(source_path),
        },
        "qcu": {"process": qcu_result, "report": qcu_data},
        "quda": {"process": quda_result, "report": quda_data},
        "errors": errors,
        "numerical_match": numerical_match,
        "solution_relative_difference": solution_difference,
        "solutions": {"qcu": str(qcu_solution), "quda": str(quda_solution)},
        "reports": {"qcu": str(qcu_report), "quda": str(quda_report)},
    }
    write_json(output, payload)
    print(json.dumps({
        "status": status, "compatible": compatible, "output": str(output),
        "solution_relative_difference": solution_difference,
        "qcu_returncode": qcu_result["returncode"],
        "quda_returncode": quda_result["returncode"], "errors": errors,
    }, ensure_ascii=False))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
