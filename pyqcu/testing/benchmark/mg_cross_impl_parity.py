#!/usr/bin/env python3
"""Run the Torch/CUDA Legacy and Strict MG cross-implementation contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


REPO = Path(__file__).resolve().parents[3]
WORKER = REPO / "pyqcu/testing/benchmark/qcu_six_way_worker.py"
HARNESS = REPO / "pyqcu/testing/qcu/multigrid/benchmarks/legacy_vs_strict.py"
DEFAULT_CASE = REPO / "data/qcu_16cubed_quick_20261023"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _logical_dataset_sha256(path: Path, dataset: str) -> str:
    import h5py
    import numpy as np

    with h5py.File(path, "r") as handle:
        value = handle[dataset]
        digest = hashlib.sha256()
        digest.update(json.dumps({
            "dataset": dataset,
            "shape": tuple(int(x) for x in value.shape),
            "dtype": str(np.dtype(value.dtype)),
        }, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        if value.ndim == 0:
            digest.update(np.ascontiguousarray(value[()]).tobytes())
        else:
            for index in range(int(value.shape[0])):
                digest.update(np.ascontiguousarray(value[index]).tobytes())
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)


def _canonical_recipe_sha256(value: dict[str, Any]) -> str:
    data = json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def build_contract(args: argparse.Namespace) -> dict[str, Any]:
    gauge = Path(args.gauge).expanduser().resolve()
    source = Path(args.source).expanduser().resolve()
    nullvec = Path(args.nullvec).expanduser().resolve()
    legacy_cache = Path(args.legacy_cache).expanduser().resolve()
    strict_recipe = {
        "canonical_null_dataset_sha256": _logical_dataset_sha256(
            nullvec, args.null_dataset),
        "nvec": int(args.nvec),
        "levels": int(args.levels),
        "block": list(args.block),
        "transfer": "StrictTransfer(block CGS, n_block_ortho=2)",
        "coarse_operator": "R D P (strict MATPC hierarchy)",
    }
    return {
        "schema": "pyqcu.mg-cross-impl-parity-contract.v1",
        "trace": {
            "enabled": False,
            "required_unset": [
                "PYQCU_STRICT_TRACE_FILE", "QUDA_MG_TRACE_FILE"],
            "quda_tuning": "QUDA_ENABLE_TUNING=0",
        },
        "input": {
            "gauge": {
                "path": str(gauge),
                "file_sha256": _sha256_file(gauge),
                "dataset": "data",
                "dataset_sha256": _logical_dataset_sha256(gauge, "data"),
            },
            "source": {
                "path": str(source),
                "file_sha256": _sha256_file(source),
                "dataset": "data",
                "dataset_sha256": _logical_dataset_sha256(source, "data"),
            },
            "canonical_null_basis": {
                "path": str(nullvec),
                "file_sha256": _sha256_file(nullvec),
                "dataset": args.null_dataset,
                "dataset_sha256": _logical_dataset_sha256(
                    nullvec, args.null_dataset),
            },
        },
        "physics": {
            "lattice_xyzt": list(args.lat),
            "precision": "c64",
            "mass": float(args.mass),
            "csw": float(args.csw),
            "kappa": 1.0 / (2.0 * float(args.mass) + 8.0),
        },
        "coarse_assets": {
            "legacy": {
                "path": str(legacy_cache),
                "file_sha256": _sha256_file(legacy_cache),
                "datasets": {
                    name: _logical_dataset_sha256(legacy_cache, name)
                    for name in ("lonv", "hnn", "hdg", "sit")
                },
                "layout": "odd-Schur blocked lonv/hnn/hdg/sit",
            },
            "strict": {
                "serialized": False,
                "recipe": strict_recipe,
                "recipe_sha256": _canonical_recipe_sha256(strict_recipe),
                "shared_by": ["torch-cuda", "torch-cpu", "cuda-qcu"],
            },
        },
        "algorithms": {
            "legacy": {
                "levels": int(args.levels),
                "block_xyzt": list(args.block),
                "nvec": int(args.nvec),
                "coarse_dof": int(args.nvec),
                "outer_solver": "BiCGStab",
                "smoother": "coarse BiCGStab (CUDA log label 'CG' is legacy)",
                "correction": "one-shot deflation + calibrated periodic V-cycle; reject and roll back harmful corrections",
                "restart": int(args.correction_interval),
                "coarse_max_iter": int(args.coarse_max_iter),
                "coarse_tolerance": float(args.coarse_tolerance),
                "stop_criterion": "relative Schur candidate plus relative full Wilson/Clover residual gate",
                "initial_guess": "zero",
            },
            "strict": {
                "levels": int(args.levels),
                "block_xyzt": list(args.block),
                "nvec": int(args.nvec),
                "coarse_dof": 2 * int(args.nvec),
                "outer_solver": "right-preconditioned FGMRES",
                "smoother": "MR, nu_pre=nu_post=1 per MATPC level",
                "coarse_solver": "FGMRES on coarsest MATPC",
                "restart": int(args.restart),
                "coarse_max_iter": int(args.coarse_max_iter),
                "coarse_tolerance": float(args.coarse_tolerance),
                "stop_criterion": "relative MATPC residual plus full reconstruction gate",
                "initial_guess": "zero",
            },
        },
        "acceptance": {
            "full_relative_residual_max": float(args.tolerance),
            "legacy_iteration_relative_delta_max": 0.20,
            "strict_iteration_delta_max": 2,
            "solution_relative_l2_diff_max": 1.0e-4,
        },
    }


def _command_cuda_legacy(args: argparse.Namespace, out: Path) -> list[str]:
    return [
        sys.executable, str(HARNESS), "legacy",
        "--gauge", str(args.gauge), "--source", str(args.source),
        "--lat", *map(str, args.lat), "--mass", str(args.mass),
        "--csw", str(args.csw), "--device", args.cuda_device,
        "--tol", str(args.tolerance), "--max-iter", str(args.max_iter),
        "--warmups", "0", "--repeats", "1", "--cache", str(args.legacy_cache),
        "--levels", str(args.levels), "--coarse-dof", str(args.nvec),
        "--coarse-max-iter", str(args.coarse_max_iter),
        "--coarse-restart", str(args.correction_interval),
        "--coarse-tol-factor", str(args.coarse_tolerance / args.tolerance),
        "--deflate", "--log-dir", str(out / "logs" / "cuda-legacy"),
        "--solution-output", str(out / "solutions" / "cuda-legacy.h5"),
        "--output", str(out / "cuda-legacy.json"),
    ]


def _command_cuda_strict(args: argparse.Namespace, out: Path) -> list[str]:
    return [
        sys.executable, str(HARNESS), "strict",
        "--gauge", str(args.gauge), "--source", str(args.source),
        "--lat", *map(str, args.lat), "--mass", str(args.mass),
        "--csw", str(args.csw), "--device", args.cuda_device,
        "--tol", str(args.tolerance), "--max-iter", str(args.max_iter),
        "--warmups", "0", "--repeats", "1",
        "--nullvec", str(args.nullvec), "--null-dataset", args.null_dataset,
        "--levels", str(args.levels), "--nvec", str(args.nvec),
        "--restart", str(args.restart),
        "--coarse-max-iter", str(args.coarse_max_iter),
        "--coarse-tol-factor", str(args.coarse_tolerance / args.tolerance),
        "--max-krylov-bytes", str(args.max_workspace_bytes),
        "--solution-output", str(out / "solutions" / "cuda-strict.h5"),
        "--output", str(out / "cuda-strict.json"),
    ]


def _command_torch(args: argparse.Namespace, out: Path, variant: str,
                   device: str) -> list[str]:
    strategy = variant
    label = f"torch-{strategy}-{'cpu' if device == 'cpu' else 'cuda'}"
    command = [
        sys.executable, str(WORKER), "--variant", f"torch-{strategy}",
        "--gauge", str(args.gauge), "--source", str(args.source),
        "--lat", *map(str, args.lat), "--mass", str(args.mass),
        "--csw", str(args.csw), "--tolerance", str(args.tolerance),
        "--coarse-tolerance", str(args.coarse_tolerance),
        "--max-iter", str(args.max_iter),
        "--coarse-max-iter", str(args.coarse_max_iter),
        "--levels", str(args.levels), "--block", *map(str, args.block),
        "--nvec", str(args.nvec), "--nu", "1", "--restart", str(args.restart),
        "--correction-interval", str(args.correction_interval),
        "--warmups", "0", "--repeats", "0", "--seed", "42",
        "--device", device,
        "--solution-output", str(out / "solutions" / f"{label}.h5"),
        "--output", str(out / f"{label}.json"),
    ]
    if strategy == "legacy":
        command.extend(["--legacy-cache", str(args.legacy_cache)])
    else:
        command.extend([
            "--nullvec", str(args.nullvec),
            "--null-dataset", args.null_dataset,
        ])
    return command


def _run(command: Sequence[str], env: dict[str, str], stdout: Path,
         timeout: int) -> int:
    stdout.parent.mkdir(parents=True, exist_ok=True)
    try:
        with stdout.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                list(command), cwd=REPO, env=env, stdout=handle,
                stderr=subprocess.STDOUT, check=False, timeout=timeout)
        return int(completed.returncode)
    except subprocess.TimeoutExpired:
        return 124
    except OSError as error:
        stdout.write_text(f"{type(error).__name__}: {error}\n", encoding="utf-8")
        return 127


def _solution_difference(left: Path, right: Path) -> dict[str, float]:
    import h5py
    import numpy as np

    with h5py.File(left, "r") as handle:
        lhs = np.asarray(handle["data"][...])
    with h5py.File(right, "r") as handle:
        rhs = np.asarray(handle["data"][...])
    if lhs.shape != rhs.shape:
        raise ValueError(f"solution shapes differ: {lhs.shape} != {rhs.shape}")
    difference = lhs - rhs
    return {
        "relative_l2": float(
            np.linalg.norm(difference.reshape(-1)) /
            max(float(np.linalg.norm(rhs.reshape(-1))), 1.0e-30)),
        "relative_max": float(
            np.max(np.abs(difference)) /
            max(float(np.max(np.abs(rhs))), 1.0e-30)),
    }


def _normalize_torch(raw: dict[str, Any], device: str) -> dict[str, Any]:
    report = raw.get("coarse_execution", {}).get("last_solve_report", {})
    corrections = report.get("corrections", [])
    native_residual = report.get("native_relative_residual")
    if native_residual is None:
        native_residual = raw.get("residual", {}).get("native_relative")
    if native_residual is None:
        native_residual = report.get("full_relative_residual")
    tolerance_absolute = report.get("tolerance_absolute")
    tolerance_relative = raw.get("configuration", {}).get("tolerance")
    if (native_residual is not None and tolerance_absolute is not None and
            tolerance_relative is not None and
            float(native_residual) > float(tolerance_relative)):
        native_residual = (
            float(report.get("final_residual", native_residual)) /
            max(float(tolerance_absolute), 1e-30) *
            float(tolerance_relative))
    return {
        "backend": "torch",
        "device": device,
        "status": raw.get("status"),
        "iterations": raw.get("iterations", {}).get("samples", [None])[-1],
        "iteration_samples": raw.get("iterations", {}).get("samples", []),
        "stop_reason": raw.get("iterations", {}).get("stop_reasons", [None])[-1],
        "native_residual": native_residual,
        "independent_full_residual": raw.get("residual", {}).get(
            "independent_relative"),
        "phase_timings_seconds": raw.get("timing", {}),
        "coarse_execution": raw.get("coarse_execution", {}),
        "counts": {
            "vcycles": report.get("vcycles"),
            "coarse_iterations": report.get("coarse_iterations", 0),
            "corrections": len(corrections),
            "deflation": report.get("deflation") is not None,
            "top_level_vcycles": report.get("vcycle_calls", {}).get(
                "top_level"),
        },
        "solver_report": report,
        "device_info": raw.get("device", {}),
    }


def _normalize_cuda_legacy(raw: dict[str, Any]) -> dict[str, Any]:
    iterations = raw.get("iterations", [])
    native = raw.get("native_true_residual")
    independent = raw.get("independent_true_residual")
    return {
        "backend": "cuda-qcu",
        "device": "cuda:0",
        "status": raw.get("status"),
        "iterations": iterations[-1] if iterations else None,
        "iteration_samples": iterations,
        "stop_reason": (
            "native_tolerance" if iterations and
            int(iterations[-1]) < int(raw.get("max_iter", 0)) else
            "iteration_limit"),
        "native_residual": native,
        "independent_full_residual": independent,
        "phase_timings_seconds": {
            "input_prepare": raw.get("fine_prepare_seconds"),
            "cache_load": raw.get("cache_load_seconds"),
            "runtime_init": raw.get("mg_init_seconds"),
            "solve": raw.get("steady_median_seconds"),
        },
        "coarse_execution": raw.get("coarse_execution", {}),
        "counts": {
            "vcycles": raw.get("coarse_execution", {}).get(
                "attempted_vcycles"),
            "coarse_iterations": raw.get("coarse_execution", {}).get(
                "coarse_iterations"),
            "corrections": None,
            "deflation": bool(raw.get("deflate")),
        },
        "device_info": raw.get("memory", {}).get("device"),
    }


def _normalize_cuda_strict(raw: dict[str, Any]) -> dict[str, Any]:
    iterations = raw.get("iterations", [])
    native_samples = raw.get("backend_final_residuals", [])
    return {
        "backend": "cuda-qcu",
        "device": "cuda:0",
        "status": raw.get("status"),
        "iterations": iterations[-1] if iterations else None,
        "iteration_samples": iterations,
        "stop_reason": (
            "native_tolerance" if iterations and
            int(iterations[-1]) < int(raw.get("max_iter", 0)) else
            "iteration_limit"),
        "native_residual": native_samples[-1] if native_samples else None,
        "native_residual_scope": "matpc_absolute",
        "independent_full_residual": raw.get("final_true_residual"),
        "phase_timings_seconds": {
            "input_prepare": raw.get("fine_prepare_seconds"),
            "null_basis_load": raw.get("null_vector_load_seconds"),
            "hierarchy_setup": raw.get("hierarchy_setup_seconds"),
            "runtime_init": raw.get("runtime_init_seconds"),
            "solve": raw.get("steady_median_seconds"),
        },
        "coarse_execution": {
            "top_level_vcycles": iterations[-1] if iterations else None,
            "per_solve_lower_bound": iterations[-1] if iterations else None,
        },
        "counts": {
            "top_level_vcycles": iterations[-1] if iterations else None,
            "coarse_iterations": None,
        },
        "device_info": raw.get("memory", {}).get("torch_allocator"),
    }


def _side_pass(side: dict[str, Any], tolerance: float) -> bool:
    residual = side.get("independent_full_residual")
    return (
        side.get("status") == "PASS" and residual is not None and
        float(residual) <= tolerance)


def _comparison(left: dict[str, Any], right: dict[str, Any],
                solutions: tuple[Path, Path], tolerance: float,
                contract: dict[str, Any], strategy: str) -> dict[str, Any]:
    iteration_delta = (
        abs(int(left["iterations"]) - int(right["iterations"]))
        if left.get("iterations") is not None and
        right.get("iterations") is not None else None)
    try:
        solution_delta = _solution_difference(*solutions)
    except (FileNotFoundError, OSError, ValueError, KeyError) as error:
        solution_delta = {"error": f"{type(error).__name__}: {error}"}
    if strategy == "strict":
        iteration_limit = int(
            contract["acceptance"]["strict_iteration_delta_max"])
    else:
        observed_min = min(int(left["iterations"]), int(right["iterations"]))
        relative_limit = float(
            contract["acceptance"]["legacy_iteration_relative_delta_max"])
        iteration_limit = max(2, math.ceil(observed_min * relative_limit))
    solution_limit = float(
        contract["acceptance"]["solution_relative_l2_diff_max"])
    passed = (
        iteration_delta is not None and iteration_delta <= iteration_limit and
        "relative_l2" in solution_delta and
        solution_delta["relative_l2"] <= solution_limit)
    return {
        "left_backend": left["backend"],
        "right_backend": right["backend"],
        "iterations": [left.get("iterations"), right.get("iterations")],
        "iteration_delta": iteration_delta,
        "iteration_delta_max": iteration_limit,
        "iteration_policy": (
            "absolute" if strategy == "strict" else
            "max(2, ceil(0.20 * min_iterations))"),
        "solution_relative_difference": solution_delta,
        "solution_relative_l2_max": solution_limit,
        "status": "PASS" if passed else "FAIL",
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--gauge", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--nullvec", type=Path)
    parser.add_argument("--legacy-cache", type=Path)
    parser.add_argument("--null-dataset", default="null")
    parser.add_argument("--lat", nargs=4, type=int, default=(16, 16, 16, 16))
    parser.add_argument("--mass", type=float, default=0.05)
    parser.add_argument("--csw", type=float, default=1.0)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--coarse-tolerance", type=float, default=1e-4)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--coarse-max-iter", type=int, default=200)
    parser.add_argument("--levels", type=int, default=2)
    parser.add_argument("--block", nargs=4, type=int, default=(2, 2, 2, 2))
    parser.add_argument("--nvec", type=int, default=12)
    parser.add_argument("--restart", type=int, default=20)
    parser.add_argument("--correction-interval", type=int, default=5)
    parser.add_argument("--max-workspace-bytes", type=int, default=512 << 20)
    parser.add_argument("--cuda-device", default="cuda:0")
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument(
        "--out-dir", type=Path,
        default=REPO / "logs/qcu_16cubed_quick_20261023/parity")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--contract-only", action="store_true")
    parser.add_argument("--skip-torch-cpu", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    case = args.case_dir.expanduser().resolve()
    args.gauge = (args.gauge or case / "case/gauge_16x16x16x16_c64.h5").resolve()
    args.source = (args.source or case / "case/source_16x16x16x16_c64.h5").resolve()
    args.nullvec = (args.nullvec or case / "L16x16x16x16_nvec12_full_c64.h5").resolve()
    args.legacy_cache = (
        args.legacy_cache or case / "legacy_cache_cuda_16cubed_E12_c64.h5"
    ).resolve()
    args.lat = tuple(int(value) for value in args.lat)
    args.block = tuple(int(value) for value in args.block)
    out = args.out_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    contract = build_contract(args)
    _write_json(out / "parity_contract.json", contract)
    if args.contract_only:
        print(json.dumps(contract, ensure_ascii=False))
        return 0

    env = dict(os.environ)
    env.pop("PYQCU_STRICT_TRACE_FILE", None)
    env.pop("QUDA_MG_TRACE_FILE", None)
    env["QUDA_ENABLE_TUNING"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    env["OMP_NUM_THREADS"] = str(args.torch_threads)

    jobs: list[tuple[str, list[str], Path]] = [
        ("cuda-legacy", _command_cuda_legacy(args, out),
         out / "cuda-legacy.stdout"),
        ("cuda-strict", _command_cuda_strict(args, out),
         out / "cuda-strict.stdout"),
        ("torch-legacy-cuda", _command_torch(args, out, "legacy", "cuda:0"),
         out / "torch-legacy-cuda.stdout"),
        ("torch-strict-cuda", _command_torch(args, out, "strict", "cuda:0"),
         out / "torch-strict-cuda.stdout"),
    ]
    if not args.skip_torch_cpu:
        jobs.extend([
            ("torch-legacy-cpu", _command_torch(args, out, "legacy", "cpu"),
             out / "torch-legacy-cpu.stdout"),
            ("torch-strict-cpu", _command_torch(args, out, "strict", "cpu"),
             out / "torch-strict-cpu.stdout"),
        ])
    exits: dict[str, int] = {}
    for name, command, stdout in jobs:
        output_json = out / f"{name}.json"
        if args.resume and output_json.is_file():
            exit_file = out / f"{name}.exit_code.txt"
            exits[name] = (
                int(exit_file.read_text(encoding="utf-8").strip())
                if exit_file.is_file() else 0)
            continue
        exits[name] = _run(command, env, stdout, args.timeout_seconds)
        (out / f"{name}.exit_code.txt").write_text(
            f"{exits[name]}\n", encoding="utf-8")

    sides: dict[str, dict[str, Any]] = {}
    if (out / "cuda-legacy.json").is_file():
        sides["cuda-legacy"] = _normalize_cuda_legacy(_json(out / "cuda-legacy.json"))
    if (out / "cuda-strict.json").is_file():
        sides["cuda-strict"] = _normalize_cuda_strict(_json(out / "cuda-strict.json"))
    for name in ("torch-legacy-cuda", "torch-strict-cuda",
                 "torch-legacy-cpu", "torch-strict-cpu"):
        path = out / f"{name}.json"
        if path.is_file():
            sides[name] = _normalize_torch(
                _json(path), "cuda:0" if name.endswith("cuda") else "cpu")

    comparisons: dict[str, Any] = {}
    for strategy in ("legacy", "strict"):
        reference = sides.get(f"cuda-{strategy}")
        if reference is None:
            continue
        for device in ("cuda", "cpu"):
            candidate = sides.get(f"torch-{strategy}-{device}")
            if candidate is None:
                continue
            comparisons[f"{strategy}-torch-{device}-vs-cuda"] = _comparison(
                candidate, reference,
                (out / "solutions" / f"torch-{strategy}-{device}.h5",
                 out / "solutions" / f"cuda-{strategy}.h5"),
                args.tolerance, contract, strategy)

    side_pass = all(
        _side_pass(side, args.tolerance) for side in sides.values()
    ) and len(sides) == len(jobs)
    comparison_pass = all(
        item["status"] == "PASS" for item in comparisons.values())
    blocked = any(code != 0 for code in exits.values())
    summary = {
        "schema": "pyqcu.mg-cross-impl-parity-result.v1",
        "status": (
            "PASS" if side_pass and comparison_pass and not blocked else "FAIL"),
        "contract": str(out / "parity_contract.json"),
        "trace_off": {
            key: key not in env for key in (
                "PYQCU_STRICT_TRACE_FILE", "QUDA_MG_TRACE_FILE")
        },
        "platform": {
            "hostname": platform.node(),
            "python": sys.version,
            "torch_threads": args.torch_threads,
            "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
        },
        "commands": {name: command for name, command, _ in jobs},
        "exit_codes": exits,
        "sides": sides,
        "comparisons": comparisons,
    }
    _write_json(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
