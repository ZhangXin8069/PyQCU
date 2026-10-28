#!/usr/bin/env python3
"""Warm, independent samples for the CUDA Legacy pair-reduction A/B study.

The ``run`` command is launched once per MPI process group.  It reuses the
production ``worker._run_cuda`` path for one discarded warmup and at least five
timed solves.  Every call creates a fresh QcuRuntime/LatticeSet lifecycle.
The ``aggregate`` and ``verify-residuals`` commands are CPU-only.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np


SCHEMA = "pyqcu.cuda_pair_reduce.samples.v1"
ABBA_ORDER = ("A", "B", "B", "A")
REQUIRED_REDUCTION_KEYS = (
    "mpi_logical_complex_reductions",
    "mpi_pair_reductions",
    "mpi_scalar_reductions",
    "mpi_real_reductions",
    "mpi_allreduce_calls",
    "mpi_unmodified_pair_reductions",
    "mpi_candidate_logical_reductions",
    "mpi_candidate_pair_reductions",
    "mpi_candidate_scalar_reductions",
    "mpi_pair_fine_dot_host",
    "mpi_pair_coarse_dot_slot",
    "mpi_pair_coarse_dot_host",
    "mpi_pair_dot_coarse",
)

_PROFILE_PREFIX = "PROF_SECTIONS:"
_PROFILE_TOKEN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")
_INTEGER = re.compile(r"^[+-]?\d+$")
_FLOAT = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?:ms)?$")


def _number(token: str) -> int | float | str:
    value = token[:-2] if token.endswith("ms") else token
    if _INTEGER.match(value):
        return int(value)
    if _FLOAT.match(token):
        return float(value)
    return token


def parse_native_profile(text: str) -> dict[str, Any]:
    """Parse the final native solve/profile record from one rank's stdout."""
    iterations_matches = re.findall(
        r"Total iterations:\s*(\d+)", text)
    residual_matches = re.findall(
        r"FINAL TRUE residual \(full-op\)\s*=\s*"
        r"([0-9.eE+-]+)\s+relative\s*=\s*([0-9.eE+-]+)", text)
    profile_matches = re.findall(
        rf"{re.escape(_PROFILE_PREFIX)}\s*([^\r\n]+)", text)

    profile: dict[str, int | float | str] = {}
    if profile_matches:
        for key, token in _PROFILE_TOKEN.findall(profile_matches[-1]):
            profile[key] = _number(token)

    iterations = int(iterations_matches[-1]) if iterations_matches else None
    native_abs = float(residual_matches[-1][0]) if residual_matches else None
    native_relative = (
        float(residual_matches[-1][1]) if residual_matches else None)
    return {
        "iterations": iterations,
        "native_true_residual": native_abs,
        "native_relative_residual": native_relative,
        "profile": profile,
    }


def logical_hdf5_sha256(path: str | os.PathLike[str],
                        dataset: str = "data") -> dict[str, Any]:
    """Hash the logical C-order array, not HDF5 layout or metadata."""
    digest = hashlib.sha256()
    with h5py.File(path, "r") as handle:
        array = np.ascontiguousarray(handle[dataset][...])
        digest.update(memoryview(array).cast("B"))
        shape = [int(value) for value in array.shape]
        dtype = str(array.dtype)
    return {"sha256": digest.hexdigest(), "shape": shape, "dtype": dtype}


def summary(values: Sequence[int | float]) -> dict[str, Any]:
    samples = [float(value) for value in values]
    if not samples:
        return {
            "samples": 0, "median": None, "mad": None,
            "min": None, "max": None,
        }
    median = float(statistics.median(samples))
    mad = float(statistics.median(abs(value - median) for value in samples))
    return {
        "samples": len(samples),
        "median": median,
        "mad": mad,
        "min": min(samples),
        "max": max(samples),
    }


def atomic_write_json(path: str | os.PathLike[str],
                      payload: Mapping[str, Any]) -> None:
    """Publish JSON atomically and remove a partial file on any failure."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(
        f".{output.name}.{os.getpid()}.partial")
    try:
        partial.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8")
        os.replace(partial, output)
    finally:
        try:
            partial.unlink()
        except FileNotFoundError:
            pass


def cleanup_failed_sample(sample_dir: str | os.PathLike[str]) -> list[str]:
    """Remove only files published by a failed sampler invocation."""
    directory = Path(sample_dir)
    removed: list[str] = []
    names = ("solution.h5", "report.json")
    candidates = [directory / name for name in names]
    candidates.extend(directory.glob(".*.partial"))
    candidates.extend(directory.glob("*.partial"))
    for path in candidates:
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        removed.append(str(path))
    return removed


@contextmanager
def redirected_process_output(stdout_path: str | os.PathLike[str],
                              stderr_path: str | os.PathLike[str]):
    """Redirect process-level stdout/stderr so native printf is per-sample."""
    stdout = Path(stdout_path)
    stderr = Path(stderr_path)
    stdout.parent.mkdir(parents=True, exist_ok=True)
    stderr.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    old_stdout = os.dup(1)
    old_stderr = os.dup(2)
    try:
        with stdout.open("wb") as out_handle, stderr.open("wb") as err_handle:
            os.dup2(out_handle.fileno(), 1)
            os.dup2(err_handle.fileno(), 2)
            try:
                yield
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
                try:
                    ctypes.CDLL(None).fflush(None)
                except (AttributeError, OSError):
                    pass
                os.dup2(old_stdout, 1)
                os.dup2(old_stderr, 2)
    finally:
        os.close(old_stdout)
        os.close(old_stderr)


def _worker_args(args: argparse.Namespace, output: Path,
                 report: Path) -> argparse.Namespace:
    from pyqcu.testing.simple.mpi import worker

    command = [
        "--backend", "cuda",
        "--gauge", args.gauge,
        "--source", args.source,
        "--lat", *(str(value) for value in args.lat),
        "--grid", *(str(value) for value in args.grid),
        "--mass", str(args.mass),
        "--csw", str(args.csw),
        "--atol", str(args.atol),
        "--max-iter", str(args.max_iter),
        "--dtype", args.dtype,
        "--levels", str(args.levels),
        "--dof", *(str(value) for value in args.dof),
        "--mg-grid", *(str(value) for value in args.mg_grid),
        "--restart", str(args.restart),
        "--coarse-max-iter", str(args.coarse_max_iter),
        "--coarse-tol-factor", str(args.coarse_tol_factor),
        "--nv-iters", str(args.nv_iters),
        "--seed", str(args.seed),
        "--output", str(output),
        "--report", str(report),
        "--skip-independent-residual",
    ]
    if args.coarse_cache:
        command.extend(["--coarse-cache", args.coarse_cache])
    for cache in args.coarse_cache_level:
        command.extend(["--coarse-cache-level", cache])
    return worker._parse_args(command)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_sample(args: argparse.Namespace, comm: Any, phase: str,
                index: int) -> dict[str, Any]:
    from pyqcu.testing.simple.mpi import worker

    rank = int(comm.Get_rank())
    run_dir = (
        Path(args.output_root) / f"warmup-{index:03d}"
        if phase == "warmup" else
        Path(args.output_root) / "samples" / f"sample-{index:03d}"
    )
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    solution = run_dir / "solution.h5"
    report_path = run_dir / "report.json"
    stdout_path = run_dir / f"rank-{rank}.stdout.txt"
    stderr_path = run_dir / f"rank-{rank}.stderr.txt"
    worker_args = _worker_args(args, solution, report_path)

    started = time.perf_counter()
    local_error = ""
    return_code = 2
    try:
        with redirected_process_output(stdout_path, stderr_path):
            return_code = int(worker._run_cuda(worker_args, comm))
    except BaseException:
        local_error = traceback.format_exc()
    elapsed = time.perf_counter() - started

    codes = comm.allgather(return_code)
    errors = comm.allgather(local_error)
    if any(code != 0 for code in codes) or any(errors):
        comm.Barrier()
        if rank == 0:
            removed = cleanup_failed_sample(run_dir)
            atomic_write_json(run_dir / "failure.json", {
                "phase": phase,
                "index": index,
                "return_codes": codes,
                "errors": errors,
                "removed_artifacts": removed,
            })
        raise RuntimeError(
            f"sample {phase}[{index}] failed: codes={codes}, "
            f"errors={[error for error in errors if error]}")

    record: dict[str, Any] | None = None
    if rank == 0:
        report = _read_json(report_path)
        native = parse_native_profile(
            stdout_path.read_text(encoding="utf-8", errors="replace"))
        logical = logical_hdf5_sha256(solution)
        record = {
            "phase": phase,
            "index": int(index),
            "wall_seconds": float(report["wall_seconds"]),
            "native_relative_residual": native["native_relative_residual"],
            "native_true_residual": native["native_true_residual"],
            "iterations": native["iterations"],
            "logical_solution_sha256": logical["sha256"],
            "logical_solution_shape": logical["shape"],
            "logical_solution_dtype": logical["dtype"],
            "report_status": report.get("status"),
            "converged": bool(report.get("converged")),
            "atol": float(report.get("atol", args.atol)),
            "ranks": report.get("ranks", []),
            "profile": native["profile"],
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "solution": str(solution),
            "report": str(report_path),
            "launcher_elapsed_seconds": elapsed,
        }
    return dict(comm.bcast(record, root=0))


def _expected_timed_samples(groups: Sequence[Mapping[str, Any]]) -> list[dict]:
    records: list[dict[str, Any]] = []
    for group in groups:
        label = str(group.get("label"))
        variant = str(group.get("variant"))
        slot = int(group.get("slot", -1))
        for sample in group.get("samples", []):
            record = dict(sample)
            record.update({
                "group_label": label,
                "variant": variant,
                "group_slot": slot,
                "sample_index": int(record.get("index", -1)),
            })
            records.append(record)
    return records


def _metric_values(records: Iterable[Mapping[str, Any]],
                   key: str) -> list[float]:
    values = []
    for record in records:
        value = record.get(key)
        if value is not None:
            values.append(float(value))
    return values


def _profile_values(records: Iterable[Mapping[str, Any]],
                    key: str) -> list[int | None]:
    values: list[int | None] = []
    for record in records:
        profile = record.get("profile", {})
        value = profile.get(key) if isinstance(profile, Mapping) else None
        values.append(None if value is None else int(value))
    return values


def _reduction_evidence(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = REQUIRED_REDUCTION_KEYS + ("mpi_pair_mixed_dot_host",)
    values = {key: _profile_values(records, key) for key in keys}
    available = all(
        all(value is not None for value in key_values)
        for key, key_values in values.items() if key in REQUIRED_REDUCTION_KEYS)
    return {
        "available": available,
        "counts_per_sample": values,
        "logical_complex_reductions": summary(
            [value for value in values[
                "mpi_logical_complex_reductions"] if value is not None]),
        "pair_reductions": summary(
            [value for value in values["mpi_pair_reductions"]
             if value is not None]),
        "scalar_reductions": summary(
            [value for value in values["mpi_scalar_reductions"]
             if value is not None]),
    }


def _group_summary(group: Mapping[str, Any]) -> dict[str, Any]:
    samples = list(group.get("samples", []))
    ranks: list[dict[str, Any]] = []
    rank_count = max(
        (len(sample.get("ranks", [])) for sample in samples), default=0)
    for rank in range(rank_count):
        values = [
            float(sample["ranks"][rank]["solve_seconds"])
            for sample in samples
            if rank < len(sample.get("ranks", [])) and
            sample["ranks"][rank].get("solve_seconds") is not None
        ]
        ranks.append({"rank": rank, "solve_seconds": summary(values)})
    return {
        "label": group.get("label"),
        "variant": group.get("variant"),
        "slot": group.get("slot"),
        "wall_seconds": summary(_metric_values(samples, "wall_seconds")),
        "native_relative_residual": summary(
            _metric_values(samples, "native_relative_residual")),
        "iterations": summary(_metric_values(samples, "iterations")),
        "logical_solution_sha256": sorted({
            str(sample["logical_solution_sha256"]) for sample in samples
            if sample.get("logical_solution_sha256")}),
        "per_rank": ranks,
        "reduction_evidence": _reduction_evidence(samples),
    }


def _variant_summary(groups: Sequence[Mapping[str, Any]],
                     variant: str) -> dict[str, Any]:
    selected = [group for group in groups if group.get("variant") == variant]
    samples = [
        sample for group in selected for sample in group.get("samples", [])]
    return {
        "groups": [group.get("label") for group in selected],
        "wall_seconds": summary(_metric_values(samples, "wall_seconds")),
        "native_relative_residual": summary(
            _metric_values(samples, "native_relative_residual")),
        "iterations": summary(_metric_values(samples, "iterations")),
        "logical_solution_sha256": sorted({
            str(sample["logical_solution_sha256"]) for sample in samples
            if sample.get("logical_solution_sha256")}),
        "reduction_evidence": _reduction_evidence(samples),
    }


def _paired_differences(groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_slot = {int(group.get("slot", -1)): group for group in groups}
    pair_specs = (
        (0, 1, "A0_vs_B1"),
        (3, 2, "A3_vs_B2"),
    )
    pairs = []
    all_differences: list[float] = []
    all_relative: list[float] = []
    for baseline_slot, optimized_slot, label in pair_specs:
        baseline_group = by_slot.get(baseline_slot)
        optimized_group = by_slot.get(optimized_slot)
        if baseline_group is None or optimized_group is None:
            continue
        baseline = {
            int(sample["index"]): sample
            for sample in baseline_group.get("samples", [])}
        optimized = {
            int(sample["index"]): sample
            for sample in optimized_group.get("samples", [])}
        differences = []
        for sample_index in sorted(set(baseline) & set(optimized)):
            before = float(baseline[sample_index]["wall_seconds"])
            after = float(optimized[sample_index]["wall_seconds"])
            difference = after - before
            relative = difference / before if before else None
            differences.append({
                "sample_index": sample_index,
                "baseline_seconds": before,
                "optimized_seconds": after,
                "optimized_minus_baseline_seconds": difference,
                "relative_change": relative,
            })
            all_differences.append(difference)
            if relative is not None:
                all_relative.append(relative)
        pairs.append({
            "label": label,
            "baseline_group": baseline_group.get("label"),
            "optimized_group": optimized_group.get("label"),
            "samples": differences,
            "difference_seconds": summary(
                [item["optimized_minus_baseline_seconds"]
                 for item in differences]),
            "relative_change": summary(
                [item["relative_change"] for item in differences
                 if item["relative_change"] is not None]),
        })
    return {
        "signed": "optimized minus baseline",
        "pairs": pairs,
        "all_samples": {
            "difference_seconds": summary(all_differences),
            "relative_change": summary(all_relative),
        },
    }


def _same_variant_variability(groups: Sequence[Mapping[str, Any]]) -> dict:
    result: dict[str, Any] = {}
    for variant in ("A", "B"):
        selected = sorted(
            [group for group in groups if group.get("variant") == variant],
            key=lambda group: int(group.get("slot", -1)))
        medians = [
            float(np.median(_metric_values(group.get("samples", []),
                                           "wall_seconds")))
            for group in selected if group.get("samples")
        ]
        result[variant] = {
            "group_labels": [group.get("label") for group in selected],
            "group_medians_seconds": medians,
            "group_median_spread_seconds": (
                max(medians) - min(medians) if medians else None),
            "group_median_relative_spread": (
                (max(medians) - min(medians)) / statistics.median(medians)
                if medians and statistics.median(medians) else None),
        }
    return result


def _close_all(values: Sequence[float], *, rtol: float = 1.0e-12,
               atol: float = 1.0e-15) -> bool:
    if not values:
        return False
    reference = values[0]
    return all(math.isclose(value, reference, rel_tol=rtol, abs_tol=atol)
               for value in values[1:])


def aggregate_group_reports(
        groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(groups, key=lambda group: int(group.get("slot", -1)))
    records = _expected_timed_samples(ordered)
    residues = _metric_values(records, "native_relative_residual")
    iterations = [int(record["iterations"]) for record in records
                  if record.get("iterations") is not None]
    hashes = {
        str(record["logical_solution_sha256"]) for record in records
        if record.get("logical_solution_sha256")}
    route_by_variant = {}
    for variant in ("A", "B"):
        variant_records = [
            record for record in records if record["variant"] == variant]
        route_pass = bool(variant_records)
        for record in variant_records:
            profile = record.get("profile", {})
            logical = profile.get("mpi_logical_complex_reductions")
            paired = profile.get("mpi_pair_reductions")
            scalar = profile.get("mpi_scalar_reductions")
            real = profile.get("mpi_real_reductions")
            allreduce_calls = profile.get("mpi_allreduce_calls")
            unchanged_pair = profile.get(
                "mpi_unmodified_pair_reductions")
            candidate_logical = profile.get(
                "mpi_candidate_logical_reductions")
            candidate_pair = profile.get(
                "mpi_candidate_pair_reductions")
            candidate_scalar = profile.get(
                "mpi_candidate_scalar_reductions")
            site_pair_calls = sum(int(profile.get(key, -1)) for key in (
                "mpi_pair_fine_dot_host",
                "mpi_pair_coarse_dot_slot",
                "mpi_pair_coarse_dot_host",
                "mpi_pair_dot_coarse",
                "mpi_pair_mixed_dot_host",
            ))
            fields = (
                logical, paired, scalar, real, allreduce_calls,
                unchanged_pair, candidate_logical, candidate_pair,
                candidate_scalar)
            if any(value is None for value in fields):
                route_pass = False
                break
            logical = int(logical)
            paired = int(paired)
            scalar = int(scalar)
            real = int(real)
            allreduce_calls = int(allreduce_calls)
            unchanged_pair = int(unchanged_pair)
            candidate_logical = int(candidate_logical)
            candidate_pair = int(candidate_pair)
            candidate_scalar = int(candidate_scalar)
            route_pass = route_pass and logical > 0 and unchanged_pair > 0
            route_pass = route_pass and scalar % 2 == 0
            route_pass = route_pass and logical == paired + scalar // 2
            route_pass = route_pass and (
                allreduce_calls == paired + scalar + real)
            route_pass = route_pass and candidate_logical > 0
            route_pass = route_pass and candidate_scalar % 2 == 0
            route_pass = route_pass and candidate_logical == (
                candidate_pair + candidate_scalar // 2)
            if variant == "B":
                route_pass = route_pass and candidate_pair == candidate_logical
                route_pass = route_pass and candidate_scalar == 0
                route_pass = route_pass and site_pair_calls == candidate_pair
            else:
                route_pass = route_pass and candidate_pair == 0
                route_pass = route_pass and candidate_scalar == (
                    2 * candidate_logical)
                route_pass = route_pass and site_pair_calls == 0
        route_by_variant[variant] = route_pass

    variant_summaries = {
        "A": _variant_summary(ordered, "A"),
        "B": _variant_summary(ordered, "B"),
    }
    gates = {
        "four_groups": len(ordered) == 4,
        "abba_order": tuple(group.get("variant") for group in ordered)
        == ABBA_ORDER,
        "one_warmup_each": all(len(group.get("warmups", [])) >= 1
                               for group in ordered),
        "five_timed_samples_each": all(
            len(group.get("samples", [])) >= 5 for group in ordered),
        "group_solve_status_pass": all(
            group.get("status") == "PASS" for group in ordered),
        "sample_reports_converged": all(
            record.get("converged") is True and
            record.get("report_status") == "PASS" for record in records),
        "native_residual_below_atol": all(
            record.get("native_relative_residual") is not None and
            float(record["native_relative_residual"]) <= float(record["atol"])
            for record in records),
        "native_residual_stable": _close_all(residues),
        "iterations_stable": bool(iterations) and len(set(iterations)) == 1,
        "logical_solution_identical": len(hashes) == 1,
        "baseline_scalar_route": route_by_variant["A"],
        "optimized_pair_route": route_by_variant["B"],
    }
    gates["required_reduction_fields_present"] = all(
        all(record.get("profile", {}).get(key) is not None
            for key in REQUIRED_REDUCTION_KEYS)
        for record in records)
    status = "PASS" if gates and all(gates.values()) else "INCOMPLETE"
    return {
        "schema": SCHEMA,
        "kind": "abba_aggregate",
        "status": status,
        "generated_at_unix": time.time(),
        "group_order": [group.get("label") for group in ordered],
        "groups": [_group_summary(group) for group in ordered],
        "variants": variant_summaries,
        "same_library_variability": _same_variant_variability(ordered),
        "paired_differences": _paired_differences(ordered),
        "gates": gates,
    }


def aggregate_residual_reports(
        input_root: str | os.PathLike[str],
        residual_root: str | os.PathLike[str]) -> dict[str, Any]:
    groups = load_group_reports(input_root)
    records = _expected_timed_samples(sorted(
        groups, key=lambda group: int(group.get("slot", -1))))
    values = []
    rows = []
    for record in records:
        relative_path = (
            Path(record["group_label"]) / "samples" /
            f"sample-{int(record['sample_index']):03d}" / "residual.json")
        report_path = Path(residual_root) / relative_path
        report = _read_json(report_path)
        value = float(report["relative_true_residual"])
        values.append(value)
        rows.append({
            "group_label": record["group_label"],
            "variant": record["variant"],
            "sample_index": record["sample_index"],
            "logical_solution_sha256": record["logical_solution_sha256"],
            "status": report.get("status"),
            "converged": bool(report.get("converged")),
            "relative_true_residual": value,
            "report": str(report_path),
        })
    passed = all(
        row["status"] == "PASS" and row["converged"] for row in rows)
    hashes = {row["logical_solution_sha256"] for row in rows}
    gates = {
        "four_groups_present": len(groups) == 4,
        "twenty_timed_samples_present": len(rows) == 20,
        "all_residual_reports_present": len(rows) == len(records),
        "all_residual_reports_pass": passed,
        "logical_solution_identical": len(hashes) == 1,
        "residual_stable": _close_all(values),
    }
    return {
        "schema": SCHEMA,
        "kind": "cpu_residual_aggregate",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "residual_summary": summary(values),
        "gates": gates,
        "samples": rows,
    }


def load_group_reports(root: str | os.PathLike[str]) -> list[dict[str, Any]]:
    root_path = Path(root)
    reports = []
    for path in root_path.glob("*/group.json"):
        report = _read_json(path)
        if report.get("schema") == SCHEMA and report.get("kind") == "group":
            reports.append(report)
    return sorted(reports, key=lambda report: int(report.get("slot", -1)))


def _run_group(args: argparse.Namespace) -> int:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = int(comm.Get_rank())
    size = int(comm.Get_size())
    if size != int(np.prod(args.grid)):
        raise ValueError(
            f"grid {tuple(args.grid)} requires {int(np.prod(args.grid))} "
            f"MPI ranks, got {size}")
    output_root = Path(args.output_root)
    if rank == 0:
        output_root.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    warmups = []
    samples = []
    for index in range(args.warmups):
        warmups.append(_run_sample(args, comm, "warmup", index))
    for index in range(args.samples):
        samples.append(_run_sample(args, comm, "timed", index))

    report: dict[str, Any] | None = None
    if rank == 0:
        report = {
            "schema": SCHEMA,
            "kind": "group",
            "status": (
                "PASS" if len(warmups) >= args.warmups and
                len(samples) >= args.samples else "FAIL"),
            "label": args.label,
            "variant": args.variant,
            "slot": int(args.slot),
            "library": {
                "path": str(Path(args.library).resolve()),
                "sha256": args.library_sha256,
                "ldd_resolved_path": str(Path(args.ldd_library).resolve()),
            },
            "environment": {
                "host": os.uname().nodename,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
                "device_metadata_dir": args.device_metadata_dir,
            },
            "config": {
                "gauge": str(Path(args.gauge).resolve()),
                "source": str(Path(args.source).resolve()),
                "coarse_cache": (
                    str(Path(args.coarse_cache).resolve())
                    if args.coarse_cache else None),
                "lattice": list(args.lat),
                "grid": list(args.grid),
                "ranks": size,
                "dtype": args.dtype,
                "mass": args.mass,
                "csw": args.csw,
                "atol": args.atol,
                "max_iter": args.max_iter,
                "levels": args.levels,
                "dof": list(args.dof),
                "mg_grid": list(args.mg_grid),
                "restart": args.restart,
                "coarse_max_iter": args.coarse_max_iter,
                "coarse_tol_factor": args.coarse_tol_factor,
                "warmups": args.warmups,
                "samples": args.samples,
            },
            "warmups": warmups,
            "samples": samples,
        }
        atomic_write_json(output_root / "group.json", report)
    comm.Barrier()
    return 0 if report is None or report["status"] == "PASS" else 1


def _aggregate_command(args: argparse.Namespace) -> int:
    groups = load_group_reports(args.input_root)
    report = aggregate_group_reports(groups)
    atomic_write_json(args.output, report)
    if report["status"] == "PASS":
        print(f"ABBA aggregate PASS: {args.output}")
        return 0
    failed = [key for key, value in report["gates"].items() if not value]
    print(f"ABBA aggregate {report['status']}: {', '.join(failed)}",
          file=sys.stderr)
    return 3


def _verify_residuals_command(args: argparse.Namespace) -> int:
    report = aggregate_residual_reports(args.input_root, args.residual_root)
    atomic_write_json(args.output, report)
    print(f"CPU residual aggregate {report['status']}: {args.output}")
    return 0 if report["status"] == "PASS" else 1


def _add_common_solve_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gauge", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--coarse-cache")
    parser.add_argument("--coarse-cache-level", action="append", default=[])
    parser.add_argument("--lat", nargs=4, type=int, default=(8, 8, 8, 16))
    parser.add_argument("--grid", nargs=4, type=int, default=(2, 1, 1, 1))
    parser.add_argument("--mass", type=float, default=-0.277)
    parser.add_argument("--csw", type=float, default=1.160920226)
    parser.add_argument("--atol", type=float, default=1.0e-6)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--dtype", choices=("c64", "c128"), default="c64")
    parser.add_argument("--levels", type=int, default=2)
    parser.add_argument("--dof", nargs="+", type=int, default=(12, 6))
    parser.add_argument("--mg-grid", nargs=4, type=int, default=(2, 2, 2, 1))
    parser.add_argument("--restart", type=int, default=5)
    parser.add_argument("--coarse-max-iter", type=int, default=200)
    parser.add_argument("--coarse-tol-factor", type=float, default=1.0e4)
    parser.add_argument("--nv-iters", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run one warmed MPI sample group")
    _add_common_solve_args(run)
    run.add_argument("--output-root", required=True)
    run.add_argument("--variant", choices=("A", "B"), required=True)
    run.add_argument("--label", required=True)
    run.add_argument("--slot", type=int, required=True)
    run.add_argument("--library", required=True)
    run.add_argument("--ldd-library", required=True)
    run.add_argument("--library-sha256", required=True)
    run.add_argument("--device-metadata-dir", default=None)
    run.add_argument("--warmups", type=int, default=1)
    run.add_argument("--samples", type=int, default=5)

    aggregate = commands.add_parser(
        "aggregate", help="aggregate four CPU-visible group reports")
    aggregate.add_argument("--input-root", required=True)
    aggregate.add_argument("--output", required=True)

    verify = commands.add_parser(
        "verify-residuals", help="aggregate CPU true-residual reports")
    verify.add_argument("--input-root", required=True)
    verify.add_argument("--residual-root", required=True)
    verify.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        if args.warmups < 1:
            parser.error("--warmups must be at least 1")
        if args.samples < 5:
            parser.error("--samples must be at least 5")
        if args.slot not in range(4):
            parser.error("--slot must be in [0, 3]")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "run":
        return _run_group(args)
    if args.command == "aggregate":
        return _aggregate_command(args)
    return _verify_residuals_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
