#!/usr/bin/env python3
"""Build a fail-closed CUDA-QCU/QUDA matrix summary from a run manifest."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "pyqcu.cuda-qcu-vs-quda-matrix.v1"
_QCU_NATIVE_RE = re.compile(r"Relative\(norm2\):\s*([0-9.eE+-]+)")
_QUDA_NATIVE_RE = re.compile(
    r"iterated\s*=\s*([0-9.eE+-]+)\s*,\s*true\s*=\s*([0-9.eE+-]+)")


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.partial")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)


def _median(values: Sequence[float]) -> float | None:
    selected = [float(value) for value in values
                if value is not None and math.isfinite(float(value))]
    return float(statistics.median(selected)) if selected else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        selected = float(value)
    except (TypeError, ValueError):
        return None
    return selected if math.isfinite(selected) else None


def _same_number(left: Any, right: Any) -> bool:
    first = _number(left)
    second = _number(right)
    return (first is not None and second is not None and
            math.isclose(first, second, rel_tol=1.0e-12, abs_tol=1.0e-15))


def _integer(value: Any) -> int | None:
    number = _number(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _same_integer(left: Any, right: Any) -> bool:
    first = _integer(left)
    second = _integer(right)
    return first is not None and second is not None and first == second


def _nonempty_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    selected = value.strip()
    return selected or None


def _same_present(left: Any, right: Any) -> bool:
    first = _nonempty_text(left)
    second = _nonempty_text(right)
    return first is not None and second is not None and first == second


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _same_dtype(left: Any, right: Any) -> bool:
    aliases = {"c64": "complex64", "c128": "complex128"}
    first = _nonempty_text(left)
    second = _nonempty_text(right)
    if first is None or second is None:
        return False
    return aliases.get(first, first) == aliases.get(second, second)


def _resolved_text(value: Any) -> str | None:
    text = _nonempty_text(value)
    if text is None:
        return None
    try:
        return str(Path(text).expanduser().resolve())
    except OSError:
        return str(Path(text).expanduser().absolute())


def _same_path(left: Any, right: Any) -> bool:
    first = _resolved_text(left)
    second = _resolved_text(right)
    return first is not None and second is not None and first == second


def _same_lattice(*values: Any) -> bool:
    lattices: list[tuple[int, ...]] = []
    for value in values:
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return False
        try:
            selected = tuple(int(item) for item in value)
        except (TypeError, ValueError):
            return False
        if any(item <= 0 for item in selected):
            return False
        lattices.append(selected)
    return bool(lattices) and all(item == lattices[0] for item in lattices[1:])


def _rank_ids_complete(ranks: Sequence[Mapping[str, Any]], expected: int) -> bool:
    try:
        identifiers = [int(rank["rank"]) for rank in ranks]
    except (KeyError, TypeError, ValueError):
        return False
    return identifiers == list(range(int(expected)))


def _uuids_complete_unique(values: Sequence[Any], expected: int) -> bool:
    selected = [_normalise_uuid(value) for value in values]
    return (int(expected) > 0 and len(selected) == int(expected) and
            all(value is not None for value in selected) and
            len(set(selected)) == int(expected))


def _solver_acceptance(atol: Any, dtype: Any) -> float | None:
    tolerance = _number(atol)
    if tolerance is None or tolerance <= 0.0:
        return None
    if dtype == "c64":
        epsilon = float.fromhex("0x1p-23")
        multiplier = 2048
    elif dtype == "c128":
        epsilon = float.fromhex("0x1p-52")
        multiplier = 128
    else:
        return None
    return max(tolerance, epsilon * multiplier)


def _normalise_uuid(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text.startswith("hip:"):
        text = text[4:]
    elif text.startswith("rocm-smi:"):
        text = text.split(":", 1)[1]
    compact = re.sub(r"[^0-9a-f]", "", text)
    if not compact or int(compact, 16) == 0:
        return None
    return compact


def _rank_monitor_record(
    run: Mapping[str, Any], local_rank: int,
) -> tuple[Mapping[str, Any], str | None]:
    devices = run.get("devices", {})
    if not isinstance(devices, Mapping):
        devices = {}
    tokens = run.get("visible_device_tokens", [])
    if not isinstance(tokens, list):
        tokens = []
    monitor_key = str(int(local_rank))
    token_uuid: str | None = None
    if 0 <= int(local_rank) < len(tokens):
        token = str(tokens[int(local_rank)]).strip()
        if token.isdigit():
            monitor_key = str(int(token))
        elif token:
            token_uuid = token
    monitor = devices.get(monitor_key, {})
    return (monitor if isinstance(monitor, Mapping) else {}), token_uuid


def _last_float(pattern: re.Pattern[str], text: str, group: int = 1) -> float | None:
    matches = list(pattern.finditer(text))
    return _number(matches[-1].group(group)) if matches else None


def _stage_rank_samples(
    stage: Mapping[str, Any], phase: str, field: str,
) -> list[Any]:
    return list(stage.get(phase, {}).get("samples", []))


def _distributed_sample_metric(
    ranks: Sequence[Mapping[str, Any]], stage_name: str, phase: str,
    field: str,
) -> list[float]:
    per_rank = [
        _stage_rank_samples(rank["timings"][stage_name], phase, field)
        for rank in ranks
    ]
    if not per_rank:
        return []
    counts = {len(samples) for samples in per_rank}
    if len(counts) != 1:
        return []
    values: list[float] = []
    for index in range(next(iter(counts))):
        samples = [_number(rank_samples[index].get(field))
                   for rank_samples in per_rank]
        selected = [value for value in samples if value is not None]
        if selected:
            values.append(max(selected))
    return values


def _normalise_qcu(
    report: Mapping[str, Any], stdout_path: Path, run: Mapping[str, Any],
) -> dict[str, Any]:
    ranks = sorted(
        [dict(item) for item in report.get("ranks", [])],
        key=lambda item: int(item["rank"]))
    qcu_execution = _mapping(report.get("execution"))
    solve_stage = "clover_schur_bistabcg_solve"
    reuse_stage = "clover_schur_bistabcg_solve_reused"
    use_reused = bool(ranks) and all(
        isinstance(rank.get("timings", {}).get(reuse_stage), Mapping) and
        isinstance(
            rank["timings"][reuse_stage].get("reuse", {}).get(
                "amortized_lifecycle_seconds"), (int, float))
        for rank in ranks)
    performance_stage = reuse_stage if use_reused else solve_stage
    steady_operation = _distributed_sample_metric(
        ranks, performance_stage, "steady", "operation_seconds")
    nonreused_lifecycle = _distributed_sample_metric(
        ranks, solve_stage, "steady", "lifecycle_seconds")
    if use_reused:
        steady_lifecycle = [max(
            float(rank["timings"][reuse_stage]["reuse"][
                "amortized_lifecycle_seconds"])
            for rank in ranks)]
    else:
        steady_lifecycle = _distributed_sample_metric(
            ranks, solve_stage, "steady", "lifecycle_seconds")
    setup_rank_values: list[float] = []
    setup_lifecycle_rank_values: list[float] = []
    for rank in ranks:
        operation = 0.0
        lifecycle = 0.0
        for stage in ("clover_setup_even", "clover_setup_odd"):
            timing = rank["timings"][stage]["steady"]
            operation += float(timing["operation_seconds"]["median"])
            lifecycle += float(timing["lifecycle_seconds"]["median"])
        setup_rank_values.append(operation)
        setup_lifecycle_rank_values.append(lifecycle)
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
    native_reported = _last_float(_QCU_NATIVE_RE, stdout)
    independent_values = [
        _number(rank.get("independent_true_residual", {}).get("value"))
        for rank in ranks
    ]
    acceptance_values = [
        _number(rank.get("independent_true_residual", {}).get("acceptance"))
        for rank in ranks
    ]
    independent_sources = [
        _nonempty_text(rank.get("independent_true_residual", {}).get("source"))
        for rank in ranks
    ]
    device_records = [dict(rank.get("device", {})) for rank in ranks]
    rank_metrics: list[dict[str, Any]] = []
    for rank in ranks:
        timing = rank["timings"][performance_stage]["steady"]
        lifecycle_seconds = (
            float(rank["timings"][reuse_stage]["reuse"][
                "amortized_lifecycle_seconds"])
            if use_reused else
            float(timing["lifecycle_seconds"]["median"]))
        solver_setup_seconds = (
            float(rank["timings"][reuse_stage]["reuse"]["setup_seconds"])
            if use_reused else None)
        solver_close_seconds = (
            float(rank["timings"][reuse_stage]["reuse"]["close_seconds"])
            if use_reused else None)
        device = rank.get("device", {})
        local_rank = int(rank.get("local_rank", rank.get("rank", 0)))
        monitor, token_uuid = _rank_monitor_record(run, local_rank)
        setup_operation = sum(
            float(rank["timings"][stage]["steady"]["operation_seconds"]["median"])
            for stage in ("clover_setup_even", "clover_setup_odd"))
        setup_lifecycle = sum(
            float(rank["timings"][stage]["steady"]["lifecycle_seconds"]["median"])
            for stage in ("clover_setup_even", "clover_setup_odd"))
        independent = _number(
            rank.get("independent_true_residual", {}).get("value"))
        acceptance = _number(
            rank.get("independent_true_residual", {}).get("acceptance"))
        independent_block = rank.get("independent_true_residual", {})
        rank_metrics.append({
            "rank": int(rank["rank"]),
            "local_rank": local_rank,
            "operation_seconds": float(
                timing["operation_seconds"]["median"]),
            "setup_seconds": float(setup_operation),
            "setup_lifecycle_seconds": float(setup_lifecycle),
            "solver_reuse_setup_seconds": solver_setup_seconds,
            "solver_reuse_close_seconds": solver_close_seconds,
            "lifecycle_seconds": lifecycle_seconds,
            "steady_seconds": lifecycle_seconds,
            "iterations": [
                int(sample["iterations"]) for sample in timing["samples"]
                if sample.get("iterations") is not None],
            "native_residual": native_reported,
            "native_residual_semantics": (
                "largest final QCU Relative(norm2) printed by the solver"),
            "native_converged": bool(
                rank.get("solver_native_reused" if use_reused else
                     "solver_native", {}).get("all_samples_converged")),
            "independent_true_residual": independent,
            "residual_acceptance": acceptance,
            "independent_residual_verified": (
                independent_block.get("verified") is True),
            "independent_residual_pass": independent_block.get("pass") is True,
            "independent_residual_source": _nonempty_text(
                independent_block.get("source")),
            "device_uuid": (
                device.get("uuid") or monitor.get("uuid") or token_uuid),
            "device_memory": {
                "total_bytes": (
                    _number(device.get("total_memory_bytes")) or
                    _number(monitor.get("total_bytes"))),
                "peak_used_bytes": _number(monitor.get("peak_used_bytes")),
                "source": "rocm-smi monitor plus QCU device report",
            },
        })
    return {
        "status": report.get("status"),
        "initial_guess": qcu_execution.get("initial_guess"),
        "operation_seconds": _median([
            item["operation_seconds"] for item in rank_metrics]),
        "setup_seconds": max(setup_rank_values) if setup_rank_values else None,
        "setup_lifecycle_seconds": (
            max(setup_lifecycle_rank_values)
            if setup_lifecycle_rank_values else None),
        "lifecycle_seconds": _median([
            item["lifecycle_seconds"] for item in rank_metrics]),
        "steady_seconds": _median(steady_lifecycle),
        "mpi_max_rank_wall_seconds": (
            max(steady_lifecycle) if steady_lifecycle else None),
        "operation_samples_seconds": steady_operation,
        "lifecycle_samples_seconds": steady_lifecycle,
        "nonreused_lifecycle_seconds": _median(nonreused_lifecycle),
        "runtime_reuse": {
            "enabled": use_reused,
            "timing_stage": performance_stage,
            "lifecycle_scope": (
                "applyInitQcu + cold/warmup/steady operations + applyEndQcu, "
                "amortized over all persistent-session solves"
                if use_reused else "per-solve ordinary transaction"),
        },
        "iterations": [
            value for item in rank_metrics for value in item["iterations"]],
        "native_residual": native_reported,
        "native_residual_semantics": (
            "largest final QCU Relative(norm2) printed by the solver"),
        "native_converged": all(item["native_converged"] for item in rank_metrics),
        "independent_true_residual": max(
            [value for value in independent_values if value is not None],
            default=None),
        "residual_acceptance": max(
            [value for value in acceptance_values if value is not None],
            default=None),
        "independent_true_residual_verified": all(
            item["independent_residual_verified"] and
            item["independent_residual_pass"]
            for item in rank_metrics),
        "independent_true_residual_source": (
            independent_sources[0]
            if independent_sources and all(
                source == independent_sources[0]
                for source in independent_sources) else None),
        "library_sha256": report.get("execution", {}).get(
            "libqcu", {}).get("sha256"),
        "device_uuids": [
            item["device_uuid"] for item in rank_metrics],
        "device_uuids_complete_and_unique": (
            bool(device_records) and _uuids_complete_unique(
                [item["device_uuid"] for item in rank_metrics], len(ranks))),
        "rank_records_complete": _rank_ids_complete(ranks, len(ranks)),
        "device_memory": {
            "total_bytes": max([
                item["device_memory"]["total_bytes"] for item in rank_metrics
                if item["device_memory"]["total_bytes"] is not None], default=None),
            "peak_used_bytes": max([
                item["device_memory"]["peak_used_bytes"] for item in rank_metrics
                if item["device_memory"]["peak_used_bytes"] is not None], default=None),
        },
        "rank_metrics": rank_metrics,
        "report": None,
    }


def _normalise_quda(
    report: Mapping[str, Any], stdout_path: Path, run: Mapping[str, Any],
    independent_report: Mapping[str, Any],
) -> dict[str, Any]:
    ranks = sorted(
        [dict(item) for item in report.get("mpi", {}).get("ranks", [])],
        key=lambda item: int(item["rank"]))
    quda_config = _mapping(report.get("config"))
    steady = list(report.get("phases", {}).get("steady", {}).get("samples", []))
    operation_samples = [
        max(float(value) for value in sample["quda_solver_seconds_by_rank"])
        for sample in steady]
    lifecycle_samples = [
        float(sample["max_wall_seconds"]) for sample in steady]
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
    native_iterated = _last_float(_QUDA_NATIVE_RE, stdout, 1)
    report_true = max(
        [float(sample["true_res_max"]) for sample in steady], default=None)
    independent_true = _number(independent_report.get(
        "relative_true_residual"))
    independent_acceptance = _number(independent_report.get("acceptance"))
    independent_source = _nonempty_text(
        independent_report.get("residual_source"))
    independent_verified = (
        independent_report.get("status") == "PASS" and
        independent_true is not None and independent_acceptance is not None and
        independent_source is not None and
        "independent" in independent_source.lower())
    rank_metrics: list[dict[str, Any]] = []
    for rank in ranks:
        device = dict(rank.get("gpu", {}))
        local_rank = int(rank.get("device_selection", {}).get(
            "device", rank.get("rank", 0)))
        monitor, token_uuid = _rank_monitor_record(run, local_rank)
        rank_steady = list(rank.get("steady", []))
        rank_metrics.append({
            "rank": int(rank["rank"]),
            "local_rank": local_rank,
            "operation_seconds": _median([
                sample.get("quda_solver_seconds") for sample in rank_steady]),
            "setup_seconds": float(rank["gauge_load_setup_seconds"]),
            "setup_lifecycle_seconds": float(
                rank["gauge_load_setup_seconds"]),
            "lifecycle_seconds": _median([
                sample.get("wall_seconds") for sample in rank_steady]),
            "steady_seconds": _median([
                sample.get("wall_seconds") for sample in rank_steady]),
            "iterations": [
                int(sample["iterations"]) for sample in rank_steady],
            "native_iterated_residual": native_iterated,
            "native_residual": report_true,
            "native_residual_semantics": (
                "QudaInvertParam.true_res reported by QUDA; native only"),
            "native_converged": all(
                float(sample["true_res"]) <= float(
                    report.get("solver", {}).get("true_res_tolerance", 0.0))
                for sample in rank_steady),
            "independent_true_residual": independent_true,
            "residual_acceptance": independent_acceptance,
            "independent_residual_source": independent_source,
            "independent_residual_verified": independent_verified,
            "device_uuid": (
                device.get("uuid") or monitor.get("uuid") or token_uuid),
            "device_memory": {
                "total_bytes": (
                    _number(device.get("total_global_memory_bytes")) or
                    _number(monitor.get("total_bytes"))),
                "peak_used_bytes": _number(monitor.get("peak_used_bytes")),
                "source": "rocm-smi monitor plus QUDA GPU report",
            },
        })
    setup_values = [
        float(rank["gauge_load_setup_seconds"]) for rank in ranks]
    return {
        "status": report.get("status"),
        "initial_guess": quda_config.get("use_init_guess"),
        "operation_seconds": _median([
            item["operation_seconds"] for item in rank_metrics]),
        "setup_seconds": max(setup_values) if setup_values else None,
        "setup_lifecycle_seconds": max(setup_values) if setup_values else None,
        "lifecycle_seconds": _median(lifecycle_samples),
        "steady_seconds": _median(lifecycle_samples),
        "mpi_max_rank_wall_seconds": _median(lifecycle_samples),
        "operation_samples_seconds": operation_samples,
        "lifecycle_samples_seconds": lifecycle_samples,
        "iterations": [
            value for item in rank_metrics for value in item["iterations"]],
        "native_iterated_residual": native_iterated,
        "native_residual": report_true,
        "native_true_residual": report_true,
        "native_residual_semantics": (
            "QudaInvertParam.true_res reported by QUDA; native only, not independent"),
        "native_converged": all(
            item["native_converged"] for item in rank_metrics),
        "independent_true_residual": independent_true,
        "residual_acceptance": independent_acceptance,
        "independent_true_residual_source": independent_source,
        "independent_true_residual_verified": independent_verified,
        "library_sha256": report.get("libquda", {}).get("sha256"),
        "device_uuids": [item["device_uuid"] for item in rank_metrics],
        "device_uuids_complete_and_unique": (
            bool(ranks) and _uuids_complete_unique(
                [item["device_uuid"] for item in rank_metrics], len(ranks))),
        "rank_records_complete": _rank_ids_complete(ranks, len(ranks)),
        "device_memory": {
            "total_bytes": max([
                item["device_memory"]["total_bytes"] for item in rank_metrics
                if item["device_memory"]["total_bytes"] is not None], default=None),
            "peak_used_bytes": max([
                item["device_memory"]["peak_used_bytes"] for item in rank_metrics
                if item["device_memory"]["peak_used_bytes"] is not None], default=None),
        },
        "rank_metrics": rank_metrics,
        "report": None,
    }


def _pair_gates(
    ranks: int, grid: Sequence[int], qcu_report: Mapping[str, Any],
    quda_report: Mapping[str, Any], independent_report: Mapping[str, Any],
    qcu: Mapping[str, Any], quda: Mapping[str, Any],
    run_pair: Mapping[str, Any], manifest_config: Mapping[str, Any],
) -> dict[str, bool]:
    qcu_config = _mapping(qcu_report.get("physics"))
    qcu_execution = _mapping(qcu_report.get("execution"))
    qcu_inputs = _mapping(qcu_report.get("inputs"))
    quda_config = _mapping(quda_report.get("config"))
    quda_input = _mapping(quda_report.get("input"))
    quda_source = _mapping(quda_input.get("source"))
    qcu_source = _mapping(qcu_inputs.get("source"))
    quda_mpi = _mapping(quda_report.get("mpi"))
    quda_ranks = quda_mpi.get("ranks", [])
    if not isinstance(quda_ranks, list):
        quda_ranks = []
    quda_contracts = [
        dict(_mapping(rank.get("parameter_contract")))
        for rank in quda_ranks if isinstance(rank, Mapping)]
    expected_grid = [int(value) for value in grid]
    trace_clean = all(
        isinstance(run_pair.get(backend), Mapping) and
        isinstance(run_pair.get(backend, {}).get("trace_environment"), Mapping) and
        not run_pair.get(backend, {}).get("trace_environment", {})
        for backend in ("qcu", "quda"))
    qcu_uuids = [_normalise_uuid(value)
                 for value in qcu.get("device_uuids", [])]
    quda_uuids = [_normalise_uuid(value)
                  for value in quda.get("device_uuids", [])]
    metrics_complete = all(
        _number(block.get(name)) is not None
        for block in (qcu, quda)
        for name in (
            "operation_seconds", "setup_seconds", "lifecycle_seconds",
            "steady_seconds", "mpi_max_rank_wall_seconds"))
    native_residuals_pass = all(
        block.get("native_converged") is True and
        _number(block.get("native_residual")) is not None and
        _number(block.get("residual_acceptance")) is not None and
        float(block["native_residual"]) <=
        float(block["residual_acceptance"])
        for block in (qcu, quda))
    expected_acceptance = _solver_acceptance(
        qcu_config.get("atol"), qcu_config.get("dtype"))
    common_acceptance = (
        expected_acceptance is not None and
        _same_number(qcu.get("residual_acceptance"), expected_acceptance) and
        _same_number(quda.get("residual_acceptance"), expected_acceptance))
    residual_sources_independent = all(
        source is not None and "independent" in source.lower()
        for source in (
            _nonempty_text(qcu.get("independent_true_residual_source")),
            _nonempty_text(quda.get("independent_true_residual_source")))
    )
    qcu_independent_verified = (
        qcu.get("independent_true_residual_verified") is True)
    quda_independent_verified = (
        quda.get("independent_true_residual_verified") is True)
    independent_values_within_acceptance = all(
            _number(block.get("independent_true_residual")) is not None and
            _number(block.get("residual_acceptance")) is not None and
            float(block["independent_true_residual"]) <=
            float(block["residual_acceptance"])
            for block in (qcu, quda))
    independent_residuals_pass = (
        residual_sources_independent and qcu_independent_verified and
        quda_independent_verified and independent_values_within_acceptance
    )
    residuals_pass = (
        native_residuals_pass and common_acceptance and
        independent_residuals_pass)
    memory_complete = all(
        _number(block.get("device_memory", {}).get("total_bytes")) not in (None, 0)
        and _number(block.get("device_memory", {}).get("peak_used_bytes")) not in (None, 0)
        for block in (qcu, quda))
    qcu_reuse_required = (
        _mapping(manifest_config.get("qcu")).get("reuse_runtime") is True)
    qcu_reuse_enabled = (
        _mapping(qcu.get("runtime_reuse")).get("enabled") is True)
    independent_inputs = _mapping(independent_report.get("inputs"))
    solution = _mapping(quda_report.get("solution"))
    independent_identity = (
        independent_report.get("status") == "PASS" and
        _same_integer(independent_report.get("ranks"), ranks) and
        independent_report.get("grid") == expected_grid and
        _same_lattice(
            independent_report.get("lattice"), qcu_config.get("lattice")) and
        _same_dtype(
            independent_report.get("dtype"), qcu_config.get("dtype")) and
        _same_number(independent_report.get("mass"), qcu_config.get("mass")) and
        _same_number(independent_report.get("csw"), qcu_config.get("csw")) and
        _same_path(independent_inputs.get("gauge"),
                   _mapping(qcu_inputs.get("gauge")).get("path")) and
        _same_path(independent_inputs.get("source"), qcu_source.get("path")) and
        _same_path(independent_inputs.get("solution"), solution.get("path")))
    quda_rhs_file_sha256 = _nonempty_text(quda_source.get("file_sha256"))
    quda_rhs_array_sha256 = _nonempty_text(quda_source.get("global_sha256"))
    qcu_rhs_sha256 = _nonempty_text(qcu_source.get("sha256"))
    manifest_rhs_sha256 = _nonempty_text(manifest_config.get("source_sha256"))
    if quda_rhs_file_sha256 is not None:
        same_rhs_sha256 = (
            qcu_rhs_sha256 is not None and
            qcu_rhs_sha256 == quda_rhs_file_sha256 and
            manifest_rhs_sha256 == quda_rhs_file_sha256)
        rhs_identity = "source file SHA256"
    else:
        same_rhs_sha256 = (
            qcu_rhs_sha256 is not None and
            quda_rhs_array_sha256 is not None and
            qcu_rhs_sha256 == quda_rhs_array_sha256 and
            manifest_rhs_sha256 == quda_rhs_array_sha256)
        rhs_identity = "canonical source array SHA256"
    qcu_gauge_sha256 = _nonempty_text(
        _mapping(qcu_inputs.get("gauge")).get("sha256"))
    quda_gauge_sha256 = _nonempty_text(quda_input.get("lime_sha256"))
    manifest_gauge_sha256 = _nonempty_text(manifest_config.get("gauge_sha256"))
    qcu_initial = _mapping(qcu.get("initial_guess"))
    cold_start_aligned = (
        qcu_initial.get("use_init_guess") is False and
        qcu_initial.get("params_value") == 0 and
        quda.get("initial_guess") is False)
    return {
        "run_records_pass": all(
            isinstance(run_pair.get(backend), Mapping) and
            run_pair[backend].get("status") == "PASS" and
            run_pair[backend].get("within_wall_limit") is True
            for backend in ("qcu", "quda")),
        "both_reports_pass": (
            qcu.get("status") == "PASS" and quda.get("status") == "PASS"),
        "same_rank_grid": (
            _same_integer(qcu_execution.get("mpi_size"), ranks) and
            _same_integer(quda_config.get("mpi_rank_count"), ranks) and
            qcu_execution.get("grid") == expected_grid and
            quda_config.get("process_grid_xyzt") == expected_grid and
            qcu.get("rank_records_complete") is True and
            quda.get("rank_records_complete") is True),
        "same_lattice": (
            _same_lattice(
                qcu_config.get("lattice"),
                quda_config.get("global_lattice_xyzt"),
                manifest_config.get("lattice_xyzt"))),
        "same_gauge_sha256": (
            qcu_gauge_sha256 is not None and
            qcu_gauge_sha256 == quda_gauge_sha256 and
            qcu_gauge_sha256 == manifest_gauge_sha256),
        "same_rhs_sha256": same_rhs_sha256,
        "same_mass_csw_dtype_tolerance": (
            _same_number(qcu_config.get("mass"), quda_config.get("mass")) and
            _same_number(qcu_config.get("csw"), quda_config.get("csw")) and
            _same_dtype(qcu_config.get("dtype"), quda_config.get("dtype")) and
            _same_number(qcu_config.get("atol"), quda_config.get("tol"))),
        "same_iteration_budget": (
            _same_integer(qcu_config.get("max_iter"),
                          quda_config.get("max_iter")) and
            _same_integer(qcu_execution.get("warmups"),
                          quda_config.get("warmups")) and
            _same_integer(qcu_execution.get("repeats"),
                          quda_config.get("repeats"))),
        "same_cold_start": cold_start_aligned,
        "quda_tuning_disabled": (
            bool(quda_contracts) and all(
                contract.get("environment_tuning_disabled") is True and
                contract.get("tuning_enabled") is False
                for contract in quda_contracts)),
        "trace_disabled": trace_clean,
        "native_and_independent_residual_pass": residuals_pass,
        "native_residual_pass": native_residuals_pass,
        "independent_residual_source": residual_sources_independent,
        "qcu_independent_residual_verified": qcu_independent_verified,
        "quda_independent_residual_verified": quda_independent_verified,
        "independent_values_within_acceptance": (
            independent_values_within_acceptance),
        "independent_residual_pass": independent_residuals_pass,
        "common_residual_acceptance": common_acceptance,
        "quda_independent_report_identity": independent_identity,
        "rhs_identity_is_file_sha256": rhs_identity == "source file SHA256",
        "timing_metrics_complete": metrics_complete,
        "qcu_persistent_runtime": (
            not qcu_reuse_required or qcu_reuse_enabled),
        "library_sha256_present": (
            _nonempty_text(qcu.get("library_sha256")) is not None and
            _nonempty_text(quda.get("library_sha256")) is not None),
        "qcu_device_uuid_count_nonempty_unique": (
            qcu.get("device_uuids_complete_and_unique") is True and
            _uuids_complete_unique(qcu_uuids, ranks)),
        "quda_device_uuid_count_nonempty_unique": (
            quda.get("device_uuids_complete_and_unique") is True and
            _uuids_complete_unique(quda_uuids, ranks)),
        "device_uuid_present_and_matched": (
            _uuids_complete_unique(qcu_uuids, ranks) and
            _uuids_complete_unique(quda_uuids, ranks) and
            qcu_uuids == quda_uuids),
        "device_memory_present": memory_complete,
    }


def _manifest_coverage(manifest: Mapping[str, Any]) -> dict[str, Any]:
    requested_ranks_raw = manifest.get("requested_ranks", [])
    requested_backends_raw = manifest.get("backends", [])
    runs = manifest.get("runs", [])
    structure_errors: list[str] = []
    if not isinstance(requested_ranks_raw, list):
        requested_ranks_raw = []
        structure_errors.append("requested_ranks is not a list")
    if not isinstance(requested_backends_raw, list):
        requested_backends_raw = []
        structure_errors.append("backends is not a list")
    if not isinstance(runs, list):
        runs = []
        structure_errors.append("runs is not a list")
    try:
        requested_ranks = sorted({int(value) for value in requested_ranks_raw})
    except (TypeError, ValueError):
        requested_ranks = []
        structure_errors.append("requested_ranks contains invalid entries")
    requested_backends = sorted({str(value) for value in requested_backends_raw})
    if len(requested_ranks) != len(requested_ranks_raw):
        structure_errors.append("requested_ranks contains duplicates")
    if len(requested_backends) != len(requested_backends_raw):
        structure_errors.append("backends contains duplicates")
    if any(value not in (1, 2, 4) for value in requested_ranks):
        structure_errors.append("requested_ranks contains unsupported ranks")
    if any(value not in ("qcu", "quda") for value in requested_backends):
        structure_errors.append("backends contains unsupported entries")

    run_ranks: list[int] = []
    actual_cases: list[tuple[int, str]] = []
    run_records: list[Mapping[str, Any]] = []
    for index, pair in enumerate(runs):
        if not isinstance(pair, Mapping):
            structure_errors.append(f"run {index} is not an object")
            continue
        try:
            run_rank = int(pair["ranks"])
        except (KeyError, TypeError, ValueError):
            structure_errors.append(f"run {index} has invalid ranks")
            continue
        run_ranks.append(run_rank)
        run_records.append(pair)
        for backend in ("qcu", "quda"):
            if backend in pair:
                actual_cases.append((run_rank, backend))
        for backend in pair:
            if backend not in ("ranks", "grid", "qcu", "quda"):
                actual_cases.append((run_rank, str(backend)))

    requested_cases = {
        (rank, backend)
        for rank in requested_ranks for backend in requested_backends}
    actual_case_set = set(actual_cases)
    actual_counts = {
        case: actual_cases.count(case) for case in actual_case_set}
    duplicate_cases = sorted(
        case for case, count in actual_counts.items() if count > 1)
    missing_cases = sorted(requested_cases - actual_case_set)
    extra_cases = sorted(actual_case_set - requested_cases)
    run_rank_set = set(run_ranks)
    missing_ranks = sorted(set(requested_ranks) - run_rank_set)
    extra_ranks = sorted(run_rank_set - set(requested_ranks))
    both_backends_requested = set(requested_backends) == {"qcu", "quda"}
    case_coverage_exact = (
        not structure_errors and both_backends_requested and
        not duplicate_cases and not missing_cases and not extra_cases and
        not missing_ranks and not extra_ranks and len(run_ranks) == len(set(run_ranks)))
    return {
        "requested_ranks": requested_ranks,
        "requested_backends": requested_backends,
        "requested_cases": requested_cases,
        "actual_cases": actual_cases,
        "run_records": run_records,
        "missing_ranks": missing_ranks,
        "extra_ranks": extra_ranks,
        "missing_cases": missing_cases,
        "extra_cases": extra_cases,
        "duplicate_cases": duplicate_cases,
        "both_backends_requested": both_backends_requested,
        "case_coverage_exact": case_coverage_exact,
        "structure_errors": structure_errors,
    }


def summarize_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = _json(manifest_path)
    manifest_config = manifest.get("configuration", {})
    if not isinstance(manifest_config, Mapping):
        manifest_config = {}
    coverage = _manifest_coverage(manifest)
    rows: list[dict[str, Any]] = []
    for pair in coverage["run_records"]:
        try:
            ranks = int(pair["ranks"])
            grid = [int(value) for value in pair["grid"]]
        except (KeyError, TypeError, ValueError):
            continue
        qcu_report: dict[str, Any] = {}
        quda_report: dict[str, Any] = {}
        independent_report: dict[str, Any] = {}
        qcu: dict[str, Any] = {}
        quda: dict[str, Any] = {}
        qcu_run = pair.get("qcu", {})
        quda_run = pair.get("quda", {})
        try:
            if isinstance(qcu_run, Mapping) and qcu_run.get("report"):
                qcu_report = _json(Path(qcu_run["report"]))
                qcu = _normalise_qcu(
                    qcu_report, Path(qcu_run["stdout"]), qcu_run)
                qcu["report"] = str(Path(qcu_run["report"]))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            qcu.setdefault("errors", []).append(f"QCU summary parse: {exc}")
        try:
            if isinstance(quda_run, Mapping) and quda_run.get("report"):
                quda_report = _json(Path(quda_run["report"]))
                independent_record = quda_run.get("independent_residual", {})
                if (isinstance(independent_record, Mapping) and
                        independent_record.get("report")):
                    independent_report = _json(
                        Path(independent_record["report"]))
                quda = _normalise_quda(
                    quda_report, Path(quda_run["stdout"]), quda_run,
                    independent_report)
                quda["report"] = str(Path(quda_run["report"]))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            quda.setdefault("errors", []).append(f"QUDA summary parse: {exc}")
        complete = bool(qcu and quda)
        gates = (
            _pair_gates(
                ranks, grid, qcu_report, quda_report, independent_report,
                qcu, quda, pair, manifest_config)
            if complete else {"both_reports_available": False})
        passed = complete and all(gates.values())
        rows.append({
            "ranks": ranks,
            "grid": grid,
            "status": "PASS" if passed else ("INCOMPLETE" if not complete else "FAIL"),
            "gates": gates,
            "qcu": qcu or None,
            "quda": quda or None,
            "speedup": None,
        })
    records = [
        pair.get(backend)
        for pair in coverage["run_records"]
        for backend in coverage["requested_backends"]
        if isinstance(pair.get(backend), Mapping)]
    limits = manifest.get("limits", {})
    if not isinstance(limits, Mapping):
        limits = {}
    wall_limit = _number(limits.get("internal_wall_limit_seconds"))
    manifest_elapsed = _number(manifest.get("elapsed_seconds"))
    residual_policy = _nonempty_text(
        _mapping(manifest_config.get("quda")).get(
            "independent_residual_policy")) or ""
    residual_deferred = residual_policy.startswith("deferred")
    residual_limit = _number(manifest.get("residual_phase_limit_seconds"))
    residual_elapsed = _number(manifest.get("residual_phase_elapsed_seconds"))
    residual_within_limit = (
        manifest.get("residual_phase_within_wall_limit") is True and
        residual_limit is not None and residual_limit > 0.0 and
        residual_elapsed is not None and residual_elapsed < residual_limit)
    manifest_gates = {
        "manifest_complete": manifest.get("status") == "COMPLETE",
        "manifest_within_wall_limit": (
            manifest.get("within_wall_limit") is True and
            wall_limit is not None and wall_limit > 0.0 and
            manifest_elapsed is not None and manifest_elapsed < wall_limit),
        "all_runs_within_wall_limit": (
            bool(records) and len(records) == len(coverage["actual_cases"]) and
            all(record.get("within_wall_limit") is True for record in records)),
        "deferred_residual_within_wall_limit": (
            not residual_deferred or residual_within_limit),
        "requested_backends_and_ranks_covered": coverage["case_coverage_exact"],
        "no_missing_or_extra_cases": (
            not coverage["missing_cases"] and not coverage["extra_cases"] and
            not coverage["missing_ranks"] and not coverage["extra_ranks"] and
            not coverage["duplicate_cases"]),
    }
    complete_matrix = (
        coverage["case_coverage_exact"] and bool(rows) and all(
            row["qcu"] is not None and row["quda"] is not None for row in rows))
    all_gates = bool(
        complete_matrix and rows and all(row["status"] == "PASS" for row in rows) and
        all(manifest_gates.values()))
    if all_gates:
        for row in rows:
            qcu = row["qcu"]
            quda = row["quda"]
            qcu_steady = float(qcu["steady_seconds"])
            quda_steady = float(quda["steady_seconds"])
            qcu_operation = float(qcu["operation_seconds"])
            quda_operation = float(quda["operation_seconds"])
            row["speedup"] = {
                "qcu_over_quda_steady": quda_steady / qcu_steady,
                "qcu_over_quda_operation": quda_operation / qcu_operation,
                "qcu_over_quda_solve_wall": quda_steady / qcu_operation,
                "acceptance_metric": "qcu_over_quda_solve_wall",
                "acceptance_pass": bool(quda_steady > qcu_operation),
            }
    requested_cases = sorted(coverage["requested_cases"])
    completed_runs = sum(
        1 for pair in coverage["run_records"]
        for backend in coverage["requested_backends"]
        if isinstance(pair.get(backend), Mapping) and
        pair[backend].get("status") == "PASS" and
        pair[backend].get("within_wall_limit") is True)
    requested = {
        "ranks": coverage["requested_ranks"],
        "backends": coverage["requested_backends"],
        "total_runs": len(requested_cases),
        "completed_runs": completed_runs,
    }
    status = "PASS" if all_gates else (
        "PARTIAL" if not complete_matrix else "FAIL")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "complete_matrix": complete_matrix,
        "all_gates_pass": all_gates,
        "manifest": str(manifest_path.resolve()),
        "manifest_status": manifest.get("status"),
        "manifest_gates": manifest_gates,
        "coverage": {
            "requested_ranks": coverage["requested_ranks"],
            "requested_backends": coverage["requested_backends"],
            "missing_ranks": coverage["missing_ranks"],
            "extra_ranks": coverage["extra_ranks"],
            "missing_cases": [
                {"ranks": ranks, "backend": backend}
                for ranks, backend in coverage["missing_cases"]],
            "extra_cases": [
                {"ranks": ranks, "backend": backend}
                for ranks, backend in coverage["extra_cases"]],
            "duplicate_cases": [
                {"ranks": ranks, "backend": backend}
                for ranks, backend in coverage["duplicate_cases"]],
            "structure_errors": coverage["structure_errors"],
            "exact": coverage["case_coverage_exact"],
        },
        "requested": requested,
        "algorithm": manifest.get("configuration", {}).get("algorithm"),
        "configuration": manifest.get("configuration", {}),
        "rows": rows,
        "acceptance": {
            "metric": "steady distributed max-rank wall",
            "rule": "QCU is accepted only when qcu_over_quda_steady > 1",
            "speedup_emitted_only_when_all_gates_pass": True,
            "residual_metric": "global relative L2 ||A*x-b||_2 / ||b||_2",
            "residual_acceptance": (
                "max(matrix atol, real_epsilon * 2048 for c64 or 128 for c128)"),
            "quda_native_true_res_is_independent_evidence": False,
        },
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = summarize_manifest(args.manifest.expanduser().resolve())
    _atomic_json(args.output.expanduser().resolve(), summary)
    print(json.dumps({
        "status": summary["status"],
        "complete_matrix": summary["complete_matrix"],
        "all_gates_pass": summary["all_gates_pass"],
        "output": str(args.output.expanduser().resolve()),
    }, ensure_ascii=False))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
