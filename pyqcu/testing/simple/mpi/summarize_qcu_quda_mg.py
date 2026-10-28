#!/usr/bin/env python3
"""Validate and summarize CUDA-QCU/QUDA multigrid comparison evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import h5py


_RANK_GRIDS = {1: [1, 1, 1, 1], 2: [2, 1, 1, 1], 4: [2, 2, 1, 1]}


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--qcu-attempts", nargs="+", required=True,
                        help="per-rank attempts, e.g. 1:39,64,65 2:41,42,64")
    parser.add_argument("--quda-attempts", nargs="+", required=True,
                        help="per-rank formal attempts, e.g. 1:61 2:62 4:62")
    parser.add_argument("--qcu-residual-attempts", nargs="+", required=True,
                        help="one externally checked QCU attempt per rank")
    parser.add_argument("--qcu-abba-tsv", type=Path, default=None)
    parser.add_argument("--solution-crosscheck", type=Path, default=None)
    parser.add_argument("--excluded-runs", nargs="*", default=[],
                        help="audited but non-representative runs and reasons")
    parser.add_argument("--libqcu", type=Path,
                        default=Path("cpp/cuda/qcu/libqcu.so"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, default=None)
    return parser.parse_args(argv)


def _parse_attempt_map(values: Sequence[str]) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    for value in values:
        match = re.fullmatch(r"([124])(?::|=)(.+)", value)
        if match is None:
            raise ValueError(f"invalid rank attempt mapping: {value}")
        rank = int(match.group(1))
        attempts = [int(item) for item in match.group(2).split(",") if item]
        if not attempts or rank in result:
            raise ValueError(f"duplicate or empty rank attempt mapping: {value}")
        result[rank] = attempts
    if set(result) != set(_RANK_GRIDS):
        raise ValueError("attempt mappings must cover ranks 1, 2, and 4")
    return result


def _parse_single_attempt_map(values: Sequence[str]) -> dict[int, int]:
    parsed = _parse_attempt_map(values)
    if any(len(attempts) != 1 for attempts in parsed.values()):
        raise ValueError("residual attempt mapping must contain one attempt per rank")
    return {rank: attempts[0] for rank, attempts in parsed.items()}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _sha256_h5_dataset(path: Path) -> str:
    digest = hashlib.sha256()
    with h5py.File(path, "r") as handle:
        dataset = handle["data"]
        for start in range(0, dataset.shape[0], 16):
            digest.update(dataset[start:start + 16].tobytes(order="C"))
    return digest.hexdigest()


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(float(value) for value in values))


def _mad(values: Sequence[float]) -> float:
    center = _median(values)
    return float(statistics.median(abs(float(value) - center) for value in values))


def _qcu_profile(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    patterns = {
        "iterations": r"Total iterations:\s*(\d+)",
        "n_vcycles": r"n_vcycles=(\d+)",
        "coarse_iterations": r"coarse_iters=(\d+)",
    }
    result: dict[str, int] = {}
    for name, pattern in patterns.items():
        matches = [int(value) for value in re.findall(pattern, text)]
        if not matches:
            raise ValueError(f"{path}: missing {name}")
        result[name] = matches[-1]
    return result


def _qcu_paths(root: Path, ranks: int, attempt: int) -> dict[str, Path]:
    case = root / "cases" / (
        f"qcu-multigrid-c64-r{ranks}-buildcuda-e6-a{attempt}")
    return {
        "report": case / "report.json",
        "stdout": case / "stdout.txt",
        "environment": case / "environment.txt",
        "run": case / "run.json",
        "solution": case / "solution.h5",
        "residual": case / "independent-residual.json",
    }


def _quda_paths(root: Path, ranks: int, attempt: int) -> dict[str, Path]:
    case = root / "cases" / f"quda-multigrid-c64-r{ranks}-e6-a{attempt}"
    return {
        "report": case / "report.json",
        "run": case / "run.json",
        "solution": case / "solution.h5",
        "residual": case / "independent-residual.json",
    }


def _run_guard_errors(run: dict[str, Any], label: str) -> list[str]:
    errors: list[str] = []
    if int(run.get("returncode", -1)) != 0:
        errors.append(f"{label} returncode is not zero")
    if bool(run.get("timeout_10_minutes")):
        errors.append(f"{label} hit the 10-minute guard")
    elapsed = float(run.get("elapsed_seconds", 0.0))
    if elapsed >= 600.0:
        errors.append(f"{label} elapsed {elapsed:.1f}s exceeds 600s")
    if elapsed > 300.0 and not bool(run.get("five_minute_audit")):
        errors.append(f"{label} exceeded 300s without a 5-minute audit")
    return errors


def _validate_qcu(
    paths: dict[str, Path], ranks: int, canonical_source_sha: dict[str, str],
    canonical_cache_path: Path,
) -> dict[str, Any]:
    required_names = ("report", "stdout", "environment", "run", "solution")
    missing = [str(paths[name]) for name in required_names
               if not paths[name].is_file()]
    if missing:
        raise FileNotFoundError("missing QCU evidence:\n" + "\n".join(missing))
    report = _json(paths["report"])
    run = _json(paths["run"])
    profile = _qcu_profile(paths["stdout"])
    errors = _run_guard_errors(run, f"QCU r{ranks}")
    expected_grid = _RANK_GRIDS[ranks]
    if report.get("status") != "PASS":
        errors.append("QCU report is not PASS")
    if report.get("grid") != expected_grid or report.get("mpi_size") != ranks:
        errors.append("QCU rank/grid mismatch")
    if report.get("levels") != 2:
        errors.append("QCU level count is not 2")
    if report.get("coarse_cache") != str(canonical_cache_path):
        errors.append("QCU coarse-cache path mismatch")
    if int(report.get("launcher", {}).get("exit_code", -1)) != 0:
        errors.append("QCU launcher exit code is not zero")
    manifest = _json(Path(report["launcher"]["case_manifest"]))
    if manifest["source"]["sha256"] != canonical_source_sha["file_sha256"]:
        errors.append("QCU source-file SHA mismatch")
    if manifest["gauge"]["sha256"] != canonical_source_sha["gauge_file_sha256"]:
        errors.append("QCU gauge-file SHA mismatch")
    stdout = paths["stdout"].read_text(encoding="utf-8", errors="replace")
    environment = paths["environment"].read_text(
        encoding="utf-8", errors="replace")
    host_match = re.search(r"^host=(.+)$", environment, re.MULTILINE)
    job_match = re.search(r"^slurm_job_id=(.+)$", environment, re.MULTILINE)
    cache_hit = "PYQCU::CUDA::MG_SETUP: C++ stencil start" not in stdout
    if not cache_hit:
        errors.append("QCU solve rebuilt the coarse cache")
    residual = float(report["final_true_residual"])
    acceptance = float(report["residual_acceptance"])
    if residual > acceptance:
        errors.append("QCU native full residual exceeds acceptance")
    solution = paths["solution"]
    return {
        "attempt": int(run["attempt"]),
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "wall_seconds": float(report["wall_seconds"]),
        "iterations": profile["iterations"],
        "n_vcycles": profile["n_vcycles"],
        "coarse_iterations": profile["coarse_iterations"],
        "native_true_residual": residual,
        "residual_acceptance": acceptance,
        "cache_hit": cache_hit,
        "host": None if host_match is None else host_match.group(1).strip(),
        "slurm_job_id": None if job_match is None else job_match.group(1).strip(),
        "solution_sha256": _sha256_file(solution),
        "solution_size_bytes": solution.stat().st_size,
        "five_minute_audit": bool(run.get("five_minute_audit")),
        "timeout_10_minutes": bool(run.get("timeout_10_minutes")),
    }


def _validate_quda(
    paths: dict[str, Path], ranks: int, canonical_source_sha: str,
    canonical_lime_sha: str,
) -> dict[str, Any]:
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing QUDA evidence:\n" + "\n".join(missing))
    report = _json(paths["report"])
    run = _json(paths["run"])
    residual_report = _json(paths["residual"])
    errors = _run_guard_errors(run, f"QUDA r{ranks}")
    expected_grid = _RANK_GRIDS[ranks]
    if report.get("status") != "PASS":
        errors.append("QUDA report is not PASS")
    config = report.get("config", {})
    if (config.get("process_grid_xyzt") != expected_grid or
            report.get("mpi", {}).get("rank_count") != ranks):
        errors.append("QUDA rank/grid mismatch")
    parameter_contract = report.get("mpi", {}).get("ranks", [{}])[0].get(
        "parameter_contract", {})
    if parameter_contract.get("levels", config.get("levels")) != 2:
        errors.append("QUDA level count is not 2")
    if int(config.get("warmups", 0)) < 1 or int(config.get("repeats", 0)) < 3:
        errors.append("QUDA requires at least 1 warmup and 3 repeats")
    if report.get("input", {}).get("source", {}).get("global_sha256") != canonical_source_sha:
        errors.append("QUDA source SHA mismatch")
    if report.get("input", {}).get("lime_sha256") != canonical_lime_sha:
        errors.append("QUDA LIME SHA mismatch")
    steady = report.get("phases", {}).get("steady", {})
    samples = list(steady.get("samples", []))
    if len(samples) < 3:
        errors.append("QUDA steady sample count is below 3")
    sample_times = [float(sample["max_wall_seconds"]) for sample in samples]
    native_residuals = [float(sample["true_res_max"]) for sample in samples]
    if any(value > float(config.get("tol", 0.0)) for value in native_residuals):
        errors.append("QUDA native residual exceeds tolerance")
    median = _median(sample_times)
    reported_median = float(steady["distributed_wall"]["median_seconds"])
    if abs(median - reported_median) > 1.0e-12:
        errors.append("QUDA reported median does not match samples")
    residual = float(residual_report["relative_true_residual"])
    acceptance = float(residual_report["acceptance"])
    if residual_report.get("status") != "PASS" or residual > acceptance:
        errors.append("QUDA independent residual failed")
    if residual_report.get("ranks") != ranks or residual_report.get("grid") != expected_grid:
        errors.append("QUDA independent residual rank/grid mismatch")
    solution_record = report["mpi"]["ranks"][0]["solution_output"]
    solution = paths["solution"]
    solution_sha = _sha256_file(solution)
    if solution_sha != solution_record["sha256"]:
        errors.append("QUDA solution SHA mismatch")
    return {
        "attempt": int(run["attempt"]),
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "samples_seconds": sample_times,
        "median_seconds": median,
        "mad_seconds": _mad(sample_times),
        "iterations": [int(sample["iterations_by_rank"][0]) for sample in samples],
        "native_true_residuals": native_residuals,
        "independent_true_residual": residual,
        "residual_acceptance": acceptance,
        "solution_sha256": solution_sha,
        "solution_size_bytes": solution.stat().st_size,
        "nvec_resolved": report["mpi"]["ranks"][0]["parameter_contract"][
            "nvec_resolved"],
        "array_backend": report["mpi"]["ranks"][0].get("array_backend"),
        "host": report["mpi"]["ranks"][0].get("host"),
        "five_minute_audit": bool(run.get("five_minute_audit")),
        "timeout_10_minutes": bool(run.get("timeout_10_minutes")),
    }


def _validate_qcu_independent(path: Path, ranks: int) -> dict[str, Any]:
    report = _json(path)
    residual = float(report["relative_true_residual"])
    acceptance = float(report["acceptance"])
    errors: list[str] = []
    if report.get("status") != "PASS" or residual > acceptance:
        errors.append("QCU independent residual failed")
    if report.get("ranks") != ranks or report.get("grid") != _RANK_GRIDS[ranks]:
        errors.append("QCU independent residual rank/grid mismatch")
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "relative_true_residual": residual,
        "acceptance": acceptance,
        "clover_source": report.get("clover_source"),
        "elapsed_seconds": float(report.get("elapsed_seconds", 0.0)),
    }


def _aggregate_rank(
    ranks: int, qcu_trials: Sequence[dict[str, Any]],
    quda_trials: Sequence[dict[str, Any]], qcu_independent: dict[str, Any],
) -> dict[str, Any]:
    qcu_times = [trial["wall_seconds"] for trial in qcu_trials]
    quda_times = [value for trial in quda_trials for value in trial["samples_seconds"]]
    qcu_median = _median(qcu_times)
    quda_median = _median(quda_times)
    status = all(
        trial["status"] == "PASS" for trial in [*qcu_trials, *quda_trials]
    ) and qcu_independent["status"] == "PASS"
    return {
        "ranks": ranks,
        "grid": _RANK_GRIDS[ranks],
        "status": "PASS" if status else "FAIL",
        "qcu": {
            "attempts": [trial["attempt"] for trial in qcu_trials],
            "hosts": [trial["host"] for trial in qcu_trials],
            "median_seconds": qcu_median,
            "mad_seconds": _mad(qcu_times),
            "samples_seconds": qcu_times,
            "iterations": [trial["iterations"] for trial in qcu_trials],
            "n_vcycles": [trial["n_vcycles"] for trial in qcu_trials],
            "coarse_iterations": [
                trial["coarse_iterations"] for trial in qcu_trials],
            "native_true_residual_max": max(
                trial["native_true_residual"] for trial in qcu_trials),
            "independent_true_residual": qcu_independent[
                "relative_true_residual"],
            "independent_residual_attempt": next(
                trial["attempt"] for trial in qcu_trials
                if trial.get("has_independent_residual")),
            "all_cache_hits": all(trial["cache_hit"] for trial in qcu_trials),
        },
        "quda": {
            "attempts": [trial["attempt"] for trial in quda_trials],
            "hosts": [trial["host"] for trial in quda_trials],
            "median_seconds": quda_median,
            "mad_seconds": _mad(quda_times),
            "samples_seconds": quda_times,
            "iterations": [
                value for trial in quda_trials for value in trial["iterations"]],
            "native_true_residual_max": max(
                value for trial in quda_trials
                for value in trial["native_true_residuals"]),
            "independent_true_residual_max": max(
                trial["independent_true_residual"] for trial in quda_trials),
            "nvec_resolved": quda_trials[0]["nvec_resolved"],
            "array_backend": quda_trials[0]["array_backend"],
        },
        "quda_over_qcu_speedup": quda_median / qcu_median,
        "qcu_over_quda_speedup": qcu_median / quda_median,
    }


def _qcu_optimization(path: Path, root: Path) -> dict[str, Any]:
    grouped: dict[int, dict[str, list[dict[str, Any]]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            ranks = int(row["ranks"])
            attempt = int(row["attempt"])
            report_path = root / "cases" / (
                f"qcu-multigrid-c64-r{ranks}-buildcuda-e6-a{attempt}") / "report.json"
            report = _json(report_path)
            grouped.setdefault(ranks, {}).setdefault(row["label"], []).append({
                "attempt": attempt,
                "wall_seconds": float(report["wall_seconds"]),
                "libqcu_sha256": row["libqcu_sha256"],
                "status": report["status"],
            })
    rows: list[dict[str, Any]] = []
    for ranks in (1, 2, 4):
        baseline = grouped[ranks]["baseline"]
        optimized = grouped[ranks]["optimized"]
        baseline_median = _median([item["wall_seconds"] for item in baseline])
        optimized_median = _median([item["wall_seconds"] for item in optimized])
        rows.append({
            "ranks": ranks,
            "baseline": baseline,
            "optimized": optimized,
            "baseline_median_seconds": baseline_median,
            "optimized_median_seconds": optimized_median,
            "improvement_percent": 100.0 * (
                baseline_median - optimized_median) / baseline_median,
        })
    return {
        "source": str(path),
        "source_sha256": _sha256_file(path),
        "ranks": rows,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# CUDA QCU vs QUDA Legacy-MultiGrid",
        "",
        "| ranks | grid | QCU samples | QCU median (s) | QCU MAD (s) | QUDA samples | QUDA median (s) | QUDA MAD (s) | QUDA/QCU | QCU scaling | QUDA scaling | QCU ext. residual | QUDA ext. residual |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["ranks"]:
        lines.append(
            f"| {row['ranks']} | {row['grid']} | {len(row['qcu']['samples_seconds'])} | "
            f"{row['qcu']['median_seconds']:.9f} | {row['qcu']['mad_seconds']:.3e} | "
            f"{len(row['quda']['samples_seconds'])} | {row['quda']['median_seconds']:.9f} | "
            f"{row['quda']['mad_seconds']:.3e} | {row['quda_over_qcu_speedup']:.4f}x | "
            f"{100*row['qcu_strong_scaling_efficiency']:.1f}% | "
            f"{100*row['quda_strong_scaling_efficiency']:.1f}% | "
            f"{row['qcu']['independent_true_residual']:.3e} | "
            f"{row['quda']['independent_true_residual_max']:.3e} |")
    lines.extend([
        "",
        "`QUDA/QCU > 1` means QCU is faster; `< 1` means QUDA is faster.",
        "QCU uses Legacy-MultiGrid; QUDA uses GCR-MG. Internal cycles and smoothing differ,",
        "so this is an entry-point comparison, not a universal solver ranking.",
    ])
    if report.get("qcu_optimization") is not None:
        lines.extend([
            "",
            "## QCU MPI Optimization ABBA",
            "",
            "| ranks | baseline median (s) | optimized median (s) | improvement |",
            "|---:|---:|---:|---:|",
        ])
        for row in report["qcu_optimization"]["ranks"]:
            lines.append(
                f"| {row['ranks']} | {row['baseline_median_seconds']:.9f} | "
                f"{row['optimized_median_seconds']:.9f} | "
                f"{row['improvement_percent']:.2f}% |")
    if report.get("solution_crosscheck") is not None:
        lines.extend([
            "",
            "## Solution Cross-Check",
            "",
            "| ranks | QCU attempt | QUDA attempt | relative L2 difference |",
            "|---:|---:|---:|---:|",
        ])
        for row in report["solution_crosscheck"]["comparisons"]:
            lines.append(
                f"| {row['ranks']} | {row['qcu_attempt']} | "
                f"{row['quda_attempt']} | "
                f"{row['relative_solution_difference']:.3e} |")
    if report.get("excluded_runs"):
        lines.extend(["", "## Audited Exclusions", ""])
        lines.extend(f"- {reason}" for reason in report["excluded_runs"])
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    root = args.result_root.resolve()
    qcu_attempts = _parse_attempt_map(args.qcu_attempts)
    quda_attempts = _parse_attempt_map(args.quda_attempts)
    residual_attempts = _parse_single_attempt_map(args.qcu_residual_attempts)

    first_qcu = _json(_qcu_paths(root, 1, qcu_attempts[1][0])["report"])
    first_quda = _json(_quda_paths(root, 1, quda_attempts[1][0])["report"])
    case_manifest_path = Path(first_qcu["launcher"]["case_manifest"])
    case_manifest = _json(case_manifest_path)
    source_path = Path(case_manifest["source"]["path"])
    cache_path = Path(first_qcu["coarse_cache"])
    cache_manifest = _json(cache_path.with_suffix(".json"))
    canonical_source = {
        "array_sha256": _sha256_h5_dataset(source_path),
        "file_sha256": case_manifest["source"]["sha256"],
        "gauge_file_sha256": case_manifest["gauge"]["sha256"],
    }
    canonical_lime_sha = first_quda["input"]["lime_sha256"]

    qcu_trials: dict[int, list[dict[str, Any]]] = {}
    quda_trials: dict[int, list[dict[str, Any]]] = {}
    qcu_independent: dict[int, dict[str, Any]] = {}
    for ranks in (1, 2, 4):
        rank_qcu: list[dict[str, Any]] = []
        for attempt in qcu_attempts[ranks]:
            trial = _validate_qcu(
                _qcu_paths(root, ranks, attempt), ranks,
                canonical_source, cache_path)
            trial["has_independent_residual"] = attempt == residual_attempts[ranks]
            rank_qcu.append(trial)
        qcu_trials[ranks] = rank_qcu
        quda_trials[ranks] = [
            _validate_quda(
                _quda_paths(root, ranks, attempt), ranks,
                canonical_source["array_sha256"], canonical_lime_sha)
            for attempt in quda_attempts[ranks]
        ]
        residual_path = _qcu_paths(
            root, ranks, residual_attempts[ranks])["residual"]
        qcu_independent[ranks] = _validate_qcu_independent(residual_path, ranks)

    rank_rows = [
        _aggregate_rank(
            ranks, qcu_trials[ranks], quda_trials[ranks],
            qcu_independent[ranks])
        for ranks in (1, 2, 4)
    ]
    qcu_baseline = rank_rows[0]["qcu"]["median_seconds"]
    quda_baseline = rank_rows[0]["quda"]["median_seconds"]
    for row in rank_rows:
        row["qcu_strong_scaling_efficiency"] = (
            qcu_baseline / (row["ranks"] * row["qcu"]["median_seconds"]))
        row["quda_strong_scaling_efficiency"] = (
            quda_baseline / (row["ranks"] * row["quda"]["median_seconds"]))

    errors = [
        f"r{row['ranks']}: {trial['errors']}"
        for row in rank_rows for trial in [*qcu_trials[row["ranks"]],
                                           *quda_trials[row["ranks"]]]
        if trial["status"] != "PASS"
    ]
    if any(not row["qcu"]["all_cache_hits"] for row in rank_rows):
        errors.append("at least one formal QCU solve rebuilt the coarse cache")
    if any(len(row["qcu"]["samples_seconds"]) < 2 for row in rank_rows):
        errors.append("QCU formal sample count is below 2")
    if any(len(row["quda"]["samples_seconds"]) < 3 for row in rank_rows):
        errors.append("QUDA formal sample count is below 3")

    report = {
        "schema_version": "pyqcu.cuda-qcu-vs-quda-mg.v2",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "lattice": [24, 24, 24, 64],
        "mass": -0.277,
        "csw": 1.160920226,
        "atol": 1.0e-6,
        "levels": 2,
        "qcu": {"dof": [12, 6], "mg_grid": [2, 2, 2, 1]},
        "quda": {"nvec": 6, "mg_block": [2, 2, 2, 2], "p2p": 0},
        "provenance": {
            "case_manifest": str(case_manifest_path),
            "source": {**canonical_source, "path": str(source_path)},
            "lime_sha256": canonical_lime_sha,
            "coarse_cache": {
                "path": str(cache_path),
                "sha256": cache_manifest["output_sha256"],
                "size_bytes": cache_manifest["output_size_bytes"],
                "setup_seconds": cache_manifest["setup_seconds"],
                "galerkin_relative_error": cache_manifest[
                    "setup_validation"]["galerkin_relative_error"],
                "gram_max_abs_error": cache_manifest[
                    "setup_validation"]["gram_max_abs_error"],
            },
            "libqcu": {
                "path": str(args.libqcu.resolve()),
                "sha256": _sha256_file(args.libqcu.resolve()),
            },
            "libquda": first_quda["libquda"],
            "device_memory_bytes": first_quda["mpi"]["ranks"][0]["gpu"][
                "total_global_memory_bytes"],
        },
        "algorithm_note": (
            "QCU Legacy-MultiGrid and QUDA GCR-MG use different internal "
            "cycles, smoothing, and stopping paths. The timing ratio compares "
            "the actual public entry points and must not be generalized into a "
            "universal solver ranking."),
        "qcu_trials": qcu_trials,
        "quda_trials": quda_trials,
        "qcu_independent_residual": qcu_independent,
        "excluded_runs": list(args.excluded_runs),
        "ranks": rank_rows,
    }
    if args.qcu_abba_tsv is not None:
        report["qcu_optimization"] = _qcu_optimization(
            args.qcu_abba_tsv.resolve(), root)
    if args.solution_crosscheck is not None:
        crosscheck = _json(args.solution_crosscheck.resolve())
        report["solution_crosscheck"] = crosscheck
        if crosscheck.get("status") != "PASS":
            errors.append("QCU/QUDA solution cross-check failed")
            report["status"] = "FAIL"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "errors": report["errors"],
        "output": str(args.output.resolve()),
        "markdown": None if args.markdown is None else str(args.markdown.resolve()),
    }, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
