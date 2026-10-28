#!/usr/bin/env python3
"""Run a bounded, same-input CUDA-QCU versus QUDA MPI matrix."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pyqcu.testing.simple._common import DEFAULT_LIME, REPO
from pyqcu.testing.simple.mpi._entrypoint import benchmark_environment
from pyqcu.testing.simple.mpi.summarize_cuda_quda_matrix import (
    _atomic_json,
    summarize_manifest,
)


QCU_ENTRY = REPO / "pyqcu/testing/qcu/benchmark_qcu_lime_mpi.py"
QUDA_ENTRY = REPO / "pyqcu/testing/quda/benchmark_quda_lime_mpi.py"
QUDA_RESIDUAL_ENTRY = REPO / "pyqcu/testing/simple/mpi/quda_residual.py"
DEFAULT_WALL_LIMIT = 570.0
DEFAULT_CHILD_LIMIT = 180.0
AUDIT_SECONDS = 300.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_rank_cases(parser: argparse.ArgumentParser,
                      values: Sequence[str]) -> list[tuple[int, list[int]]]:
    parsed: list[tuple[int, list[int]]] = []
    seen: set[int] = set()
    for value in values:
        match = re.fullmatch(r"([124]):([1-9][0-9]*),([1-9][0-9]*),"
                             r"([1-9][0-9]*),([1-9][0-9]*)", value)
        if match is None:
            parser.error(
                f"invalid rank case {value!r}; expected 1|2|4:gx,gy,gz,gt")
        ranks = int(match.group(1))
        grid = [int(match.group(index)) for index in range(2, 6)]
        if ranks in seen:
            parser.error(f"duplicate rank case for r{ranks}")
        if math.prod(grid) != ranks:
            parser.error(f"rank case {value!r} grid product != {ranks}")
        seen.add(ranks)
        parsed.append((ranks, grid))
    if not parsed:
        parser.error("at least one --rank-case is required")
    return sorted(parsed)


def _parse_backends(parser: argparse.ArgumentParser, value: str) -> list[str]:
    selected: list[str] = []
    for item in value.split(","):
        backend = item.strip().lower()
        if backend not in ("qcu", "quda"):
            parser.error(f"unsupported matrix backend: {backend!r}")
        if backend not in selected:
            selected.append(backend)
    if not selected:
        parser.error("--backends must select qcu and/or quda")
    return selected


def _sha256_file(path: Path) -> str | None:
    import hashlib

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_source_array(path: Path, shape: tuple[int, ...]) -> Any:
    import numpy as np

    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "data" not in archive:
                raise ValueError(f"{path} has no 'data' array")
            array = np.asarray(archive["data"])
    elif suffix in (".h5", ".hdf5"):
        import h5py

        with h5py.File(path, "r") as handle:
            name = next(
                (candidate for candidate in ("data", "source", "src", "b")
                 if candidate in handle), None)
            if name is None:
                raise ValueError(f"{path} has no source dataset")
            array = np.asarray(handle[name])
    else:
        raise ValueError("--source must be .npy, .npz, .h5, or .hdf5")
    if array.shape != shape:
        raise ValueError(f"source shape {array.shape} does not match {shape}")
    if not np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError(f"source dtype must be complex, got {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError("source contains non-finite values")
    return array


def _materialise_source(
    args: argparse.Namespace, result_root: Path,
) -> Path:
    import numpy as np
    import h5py

    dtype = np.dtype(
        np.complex64 if args.dtype == "c64" else np.complex128)
    shape = (4, 3, *(int(value) for value in args.lat))
    if args.source is None:
        generator = np.random.Generator(np.random.PCG64(int(args.seed)))
        real = generator.standard_normal(shape)
        imag = generator.standard_normal(shape)
        source = np.ascontiguousarray(real + 1j * imag, dtype=dtype)
        origin = "seeded-global-pcg64"
    else:
        input_path = Path(args.source).expanduser().resolve()
        source = np.ascontiguousarray(
            _read_source_array(input_path, shape), dtype=dtype)
        origin = f"file:{input_path}"
    path = result_root / "source.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with h5py.File(temporary, "w") as handle:
            handle.create_dataset("data", data=source)
            handle.attrs["format"] = "pyqcu.simple.v1"
            handle.attrs["layout"] = "[spin,color,x,y,z,t]"
            handle.attrs["lattice"] = list(args.lat)
            handle.attrs["origin"] = origin
            handle.attrs["seed"] = int(args.seed)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _mpirun_prefix(mpirun: str) -> list[str]:
    command = [
        mpirun,
        "--allow-run-as-root",
        "--oversubscribe",
        "--bind-to", "none",
        "--map-by", "slot",
    ]
    return command


def _backend_command(
    backend: str, ranks: int, grid: Sequence[int], args: argparse.Namespace,
    report: Path,
) -> list[str]:
    command = [*_mpirun_prefix(args.mpirun), "-np", str(ranks), sys.executable,
               "-B"]
    common = [
        "--gauge", str(Path(args.gauge).expanduser()),
        "--lat", *(str(value) for value in args.lat),
        "--grid", *(str(value) for value in grid),
        "--dtype", args.dtype,
        "--max-iter", str(args.max_iter),
        "--warmups", str(args.warmups),
        "--repeats", str(args.repeats),
        "--seed", str(args.seed),
        "--report", str(report),
    ]
    source = getattr(args, "matrix_source", args.source)
    if source is not None:
        common.extend(("--source", str(Path(source).expanduser())))
    if args.mass is not None:
        common.extend(("--mass", str(args.mass)))
    if args.csw is not None:
        common.extend(("--csw", str(args.csw)))
    if backend == "qcu":
        command.extend((
            str(QCU_ENTRY), *common, "--atol", str(args.atol),
            "--reuse-runtime"))
    else:
        command.extend((
            str(QUDA_ENTRY), *common,
            "--solver", "bicgstab", "--tol", str(args.atol),
            "--solution-output", str(report.with_name("solution.h5"))))
    return command


def _run_environment() -> tuple[dict[str, str], dict[str, str]]:
    environment = benchmark_environment()
    requested = {
        name: value for name, value in os.environ.items() if "TRACE" in name}
    # benchmark_environment() removes every current/future trace variable.
    environment["QUDA_ENABLE_TUNING"] = "0"
    environment.setdefault("QUDA_ENABLE_P2P", "0")
    environment.setdefault("PYQCU_MPI_DEVICE_AWARE", "0")
    environment.setdefault("OMP_NUM_THREADS", "1")
    if not environment.get("CUDA_VISIBLE_DEVICES"):
        environment["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    return environment, requested


def _visible_device_count(environment: Mapping[str, str]) -> int | None:
    configured = environment.get("CUDA_VISIBLE_DEVICES")
    if configured:
        entries = [item for item in configured.split(",") if item.strip()]
        if entries:
            return len(entries)
    slurm_gres = os.environ.get("SLURM_GPUS_ON_NODE", "")
    match = re.search(r"(?:dcu:)?(\d+)$", slurm_gres)
    return int(match.group(1)) if match else None


def _visible_device_tokens(environment: Mapping[str, str]) -> list[str]:
    configured = environment.get("CUDA_VISIBLE_DEVICES", "")
    return [item.strip() for item in configured.split(",") if item.strip()]


def _integer(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).replace(",", "").strip()
    match = re.search(r"-?\d+", text)
    return int(match.group(0)) if match else None


def _parse_rocm_smi(text: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    json_start = text.find("{")
    if json_start >= 0:
        try:
            payload = json.loads(text[json_start:])
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, Mapping):
            for key, value in payload.items():
                if not isinstance(value, Mapping):
                    continue
                card = re.search(r"(\d+)", str(key))
                if card is None:
                    continue
                item = records.setdefault(str(int(card.group(1))), {})
                for name, raw in value.items():
                    lowered = str(name).lower()
                    if "unique" in lowered or lowered == "uuid":
                        item["uuid"] = raw
                    elif "total memory" in lowered and "used" not in lowered:
                        item["total_bytes"] = _integer(raw)
                    elif "used memory" in lowered:
                        item["used_bytes"] = _integer(raw)
                    elif "memory use" in lowered:
                        item["used_percent"] = _integer(raw)
            if records:
                return records
    current: str | None = None
    for line in text.splitlines():
        card = re.search(r"DCU\[(\d+)\]|card(\d+)", line, re.IGNORECASE)
        if card:
            index = card.group(1) or card.group(2)
            current = str(int(index))
            records.setdefault(current, {})
        if current is None:
            continue
        for key, pattern in (
                ("uuid", r"Unique ID\s*:\s*([^\"',}\s]+)"),
                ("total_bytes", r"VRAM Total Memory \(B\)\s*:\s*([0-9.eE+-]+)"),
                ("used_bytes", r"VRAM Total Used Memory \(B\)\s*:\s*([0-9.eE+-]+)"),
                ("used_percent", r"DCU memory use \(%\)\s*:\s*([0-9.eE+-]+)")):
            match = re.search(pattern, line, re.IGNORECASE)
            if match is None:
                continue
            value: Any = match.group(1)
            if key != "uuid":
                value = _integer(value)
            records[current][key] = value
    return records


def _query_devices() -> tuple[dict[str, dict[str, Any]], str | None]:
    try:
        completed = subprocess.run(
            ["rocm-smi", "--showuniqueid", "--showmeminfo", "vram", "--json"],
            text=True, capture_output=True, timeout=8, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {}, f"{type(exc).__name__}: {exc}"
    records = _parse_rocm_smi(completed.stdout + "\n" + completed.stderr)
    if records:
        return records, None
    return {}, f"rocm-smi did not expose device records (rc={completed.returncode})"


def _merge_memory_peak(
    recorded: dict[str, dict[str, Any]],
    current: Mapping[str, Mapping[str, Any]],
) -> None:
    for index, values in current.items():
        target = recorded.setdefault(index, {})
        for key in ("uuid", "total_bytes"):
            if values.get(key) is not None and target.get(key) is None:
                target[key] = values[key]
        used = _integer(values.get("used_bytes"))
        if used is not None:
            target["peak_used_bytes"] = max(
                int(target.get("peak_used_bytes", 0)), used)


def _terminate_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def _run_captured_command(
    command: Sequence[str], *, timeout_seconds: float, deadline: float,
    stdout_path: Path, stderr_path: Path, environment: Mapping[str, str],
) -> tuple[int, bool, float]:
    started = time.monotonic()
    timed_out = False
    with stdout_path.open("w", encoding="utf-8") as out_handle, \
            stderr_path.open("w", encoding="utf-8") as err_handle:
        process = subprocess.Popen(
            command, cwd=REPO, env=dict(environment), text=True,
            stdout=out_handle, stderr=err_handle, start_new_session=True)
        while process.poll() is None:
            now = time.monotonic()
            if now >= deadline or now - started >= timeout_seconds:
                timed_out = True
                _terminate_process(process)
                break
            time.sleep(0.1)
        returncode = int(process.wait())
    return returncode, timed_out, time.monotonic() - started


def _quda_residual_command(
    *, mpirun: str, ranks: int, lattice: Sequence[int], grid: Sequence[int], gauge: Path,
    source: Path, solution: Path, dtype: str, mass: float, csw: float,
    atol: float, report: Path,
) -> list[str]:
    """Build the CPU-side independent QUDA residual command."""
    return [
        *_mpirun_prefix(mpirun),
        "-np", str(ranks), sys.executable, "-B",
        str(QUDA_RESIDUAL_ENTRY),
        "--gauge", str(gauge),
        "--source", str(source),
        "--solution", str(solution),
        "--lat", *(str(value) for value in lattice),
        "--grid", *(str(value) for value in grid),
        "--ranks", str(ranks),
        "--dtype", dtype,
        "--mass", str(mass),
        "--csw", str(csw),
        "--atol", str(atol),
        "--report", str(report),
    ]


def _execute_one(
    *,
    backend: str,
    ranks: int,
    grid: Sequence[int],
    args: argparse.Namespace,
    result_root: Path,
    environment: dict[str, str],
    trace_requested: Mapping[str, str],
    deadline: float,
    started_monotonic: float,
    audit_path: Path,
) -> dict[str, Any]:
    case_dir = result_root / f"r{ranks}" / backend
    case_dir.mkdir(parents=True, exist_ok=True)
    report = case_dir / "report.json"
    stdout = case_dir / "stdout.txt"
    stderr = case_dir / "stderr.txt"
    environment_path = case_dir / "environment.txt"
    devices_path = case_dir / "devices.json"
    solution = case_dir / "solution.h5"
    residual_report = case_dir / "independent-residual.json"
    residual_stdout = case_dir / "independent-residual.stdout.txt"
    residual_stderr = case_dir / "independent-residual.stderr.txt"
    if report.is_file():
        os.replace(report, case_dir / "report.previous.json")
    if residual_report.is_file():
        os.replace(
            residual_report,
            case_dir / "independent-residual.previous.json")
    command = _backend_command(backend, ranks, grid, args, report)
    remaining = deadline - time.monotonic()
    timeout_seconds = min(float(args.child_timeout), remaining - 5.0)
    record: dict[str, Any] = {
        "backend": backend,
        "ranks": ranks,
        "grid": list(grid),
        "status": "SKIPPED-BUDGET" if timeout_seconds < 10.0 else "RUNNING",
        "command": command,
        "report": str(report),
        "solution": str(solution),
        "stdout": str(stdout),
        "stderr": str(stderr),
        "environment": str(environment_path),
        "devices_file": str(devices_path),
        "timeout_seconds": max(0.0, timeout_seconds),
        "trace_environment": {
            name: value for name, value in environment.items() if "TRACE" in name},
        "trace_variables_removed": sorted(trace_requested),
        "device_aware": environment.get("PYQCU_MPI_DEVICE_AWARE"),
        "quda_enable_tuning": environment.get("QUDA_ENABLE_TUNING"),
        "visible_device_tokens": _visible_device_tokens(environment),
        "five_minute_audit": False,
        "independent_residual": {
            "required": backend == "quda",
            "status": (
                "DEFERRED" if backend == "quda" and
                args.defer_independent_residual else
                "PENDING" if backend == "quda" else "NOT-REQUIRED"),
            "deferred": bool(
                backend == "quda" and args.defer_independent_residual),
            "execution_node": (
                "CPU partition" if backend == "quda" and
                args.defer_independent_residual else
                "same node" if backend == "quda" else None),
            "report": str(residual_report) if backend == "quda" else None,
            "stdout": str(residual_stdout) if backend == "quda" else None,
            "stderr": str(residual_stderr) if backend == "quda" else None,
        },
    }
    if timeout_seconds < 10.0:
        record["within_wall_limit"] = False
        return record
    environment_path.write_text(
        "\n".join([
            f"host={platform.node()}",
            f"slurm_job_id={os.environ.get('SLURM_JOB_ID', '')}",
            f"backend={backend}",
            f"ranks={ranks}",
            f"grid={','.join(str(value) for value in grid)}",
            f"CUDA_VISIBLE_DEVICES={environment.get('CUDA_VISIBLE_DEVICES', '')}",
            f"QUDA_ENABLE_TUNING={environment.get('QUDA_ENABLE_TUNING', '')}",
            f"PYQCU_MPI_DEVICE_AWARE={environment.get('PYQCU_MPI_DEVICE_AWARE', '')}",
            f"trace_environment={record['trace_environment']}",
        ]) + "\n", encoding="utf-8")
    memory: dict[str, dict[str, Any]] = {}
    memory_errors: list[str] = []
    began = time.monotonic()
    with stdout.open("w", encoding="utf-8") as out_handle, \
            stderr.open("w", encoding="utf-8") as err_handle:
        process = subprocess.Popen(
            command, cwd=REPO, env=environment, text=True,
            stdout=out_handle, stderr=err_handle, start_new_session=True)
        next_memory = 0.0
        timed_out = False
        while process.poll() is None:
            now = time.monotonic()
            elapsed_job = now - started_monotonic
            if elapsed_job >= AUDIT_SECONDS and not record["five_minute_audit"]:
                record["five_minute_audit"] = True
                _atomic_json(audit_path, {
                    "status": "AUDIT",
                    "elapsed_seconds": elapsed_job,
                    "completed_matrix": [],
                    "message": "five-minute audit; hard matrix deadline follows",
                })
            if now >= deadline or now - began >= timeout_seconds:
                timed_out = True
                _terminate_process(process)
                break
            if now >= next_memory:
                snapshot, error = _query_devices()
                _merge_memory_peak(memory, snapshot)
                if error and error not in memory_errors:
                    memory_errors.append(error)
                next_memory = now + 5.0
            time.sleep(0.25)
        returncode = int(process.wait())
    residual_timed_out = False
    if (backend == "quda" and not args.defer_independent_residual and
            not timed_out and returncode == 0 and report.is_file()):
        independent = record["independent_residual"]
        try:
            quda_payload = json.loads(report.read_text(encoding="utf-8"))
            quda_config = quda_payload["config"]
            remaining = deadline - time.monotonic()
            residual_timeout = min(
                float(args.child_timeout), remaining - 5.0)
            independent["timeout_seconds"] = max(0.0, residual_timeout)
            if residual_timeout < 10.0:
                independent["status"] = "SKIPPED-BUDGET"
            else:
                source = Path(getattr(args, "matrix_source")).expanduser()
                residual_command = _quda_residual_command(
                    mpirun=args.mpirun, ranks=ranks, lattice=args.lat,
                    grid=grid, gauge=Path(args.gauge).expanduser(),
                    source=source, solution=solution, dtype=args.dtype,
                    mass=float(quda_config["mass"]),
                    csw=float(quda_config["csw"]), atol=args.atol,
                    report=residual_report)
                independent["command"] = residual_command
                residual_returncode, residual_timed_out, residual_elapsed = (
                    _run_captured_command(
                        residual_command,
                        timeout_seconds=residual_timeout,
                        deadline=deadline,
                        stdout_path=residual_stdout,
                        stderr_path=residual_stderr,
                        environment=environment,
                    ))
                independent.update({
                    "returncode": (
                        124 if residual_timed_out else residual_returncode),
                    "elapsed_seconds": residual_elapsed,
                })
                report_status = None
                if residual_report.is_file():
                    try:
                        report_status = json.loads(
                            residual_report.read_text(encoding="utf-8")
                        ).get("status")
                    except (OSError, ValueError, TypeError):
                        report_status = None
                passed = (
                    not residual_timed_out and residual_returncode == 0 and
                    report_status == "PASS" and solution.is_file())
                independent["report_status"] = report_status
                independent["solution_published"] = solution.is_file()
                independent["status"] = "PASS" if passed else (
                    "TIMEOUT" if residual_timed_out else "FAIL")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            independent["status"] = "FAIL"
            independent["error"] = f"{type(exc).__name__}: {exc}"
    final_snapshot, final_error = _query_devices()
    _merge_memory_peak(memory, final_snapshot)
    if final_error and final_error not in memory_errors:
        memory_errors.append(final_error)
    _atomic_json(devices_path, {
        "devices": memory,
        "errors": memory_errors,
    })
    elapsed = time.monotonic() - began
    solve_status = "TIMEOUT" if timed_out else (
        "PASS" if returncode == 0 and report.is_file() else "FAIL")
    final_status = solve_status
    if (backend == "quda" and not args.defer_independent_residual and
            solve_status == "PASS"):
        independent_status = record["independent_residual"].get("status")
        if independent_status != "PASS":
            final_status = (
                "TIMEOUT" if independent_status == "TIMEOUT" else "FAIL")
    within_wall_limit = (
        not timed_out and not residual_timed_out and
        time.monotonic() <= deadline and elapsed < float(args.wall_limit))
    record.update({
        "returncode": 124 if timed_out else returncode,
        "elapsed_seconds": elapsed,
        "status": final_status,
        "within_wall_limit": within_wall_limit,
        "devices": memory,
        "device_memory_errors": memory_errors,
        "report_published": report.is_file(),
    })
    return record


def _manifest_payload(
    args: argparse.Namespace, rank_cases: Sequence[tuple[int, list[int]]],
    backends: Sequence[str], result_root: Path,
) -> dict[str, Any]:
    matrix_source = Path(getattr(
        args, "matrix_source", result_root / "source.h5")).expanduser()
    return {
        "schema_version": "pyqcu.cuda-qcu-vs-quda-run-manifest.v1",
        "created_utc": _utc_now(),
        "status": "RUNNING",
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "requested_ranks": [ranks for ranks, _ in rank_cases],
        "backends": list(backends),
        "result_root": str(result_root),
        "limits": {
            "slurm_time_limit_seconds": args.slurm_time_limit,
            "internal_wall_limit_seconds": args.wall_limit,
            "child_timeout_seconds": args.child_timeout,
            "five_minute_audit_seconds": AUDIT_SECONDS,
            "hard_termination_seconds": args.slurm_time_limit,
        },
        "configuration": {
            "algorithm": "clover-schur-bicgstab",
            "gauge": str(Path(args.gauge).expanduser().resolve()),
            "gauge_sha256": _sha256_file(Path(args.gauge).expanduser()),
            "source": (
                str(matrix_source.resolve())),
            "source_input": (
                None if args.source is None else
                str(Path(args.source).expanduser().resolve())),
            "source_sha256": _sha256_file(matrix_source),
            "rhs_policy": (
                "same canonical HDF5 source consumed by QCU, QUDA, and "
                "the independent QUDA residual checker"),
            "seed": args.seed,
            "lattice_xyzt": list(args.lat),
            "mass": args.mass,
            "csw": args.csw,
            "dtype": args.dtype,
            "tolerance": args.atol,
            "max_iter": args.max_iter,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "mpi_topology": "one Python process = one MPI rank = one DCU",
            "trace": "disabled; all environment names containing TRACE removed",
            "quda": {
                "tuning": "disabled (QUDA_ENABLE_TUNING=0)",
                "p2p": "disabled by default (QUDA_ENABLE_P2P=0)",
                "solver": "QUDA_BICGSTAB_INVERTER",
                "independent_residual_policy": (
                    "deferred to a CPU partition" if
                    args.defer_independent_residual else "same phase"),
            },
            "qcu": {
                "prefix": "CUDA-QCU",
                "reuse_runtime": True,
                "steady_timing": (
                    "persistent applyInit/applyEnd lifecycle; report both "
                    "steady operation and amortized lifecycle"),
                "clover_plan": {
                    "set_plan": 2,
                    "symbol": "_SET_PLAN2_",
                    "operation": "applyCloversQcu",
                },
                "solver_plan": {
                    "set_plan": 1,
                    "symbol": "_SET_PLAN1_",
                    "operation": "applyCloverBistabCgQcu",
                },
            },
            "grid_semantics": {
                "process_grid_order": "xyzt",
                "lattice_order": "xyzt",
                "time_axis": 3,
                "time_axis_rule": "last axis is t for CUDA-QCU and QUDA",
                "mg_grid_xyzt": None,
                "qcu_mg_grid_applicable": False,
                "quda_mg_block_applicable": False,
                "note": (
                    "The performance pair is plain Clover BiCGStab, not MG; "
                    "process-grid time remains the last (t) axis. A future MG "
                    "row must compare QCU mg_grid_xyzt with QUDA mg_block_xyzt "
                    "without swapping or reversing the t axis."),
            },
            "environment": {
                "trace_variables_removed": sorted(
                    name for name in os.environ if "TRACE" in name),
            },
        },
        "runs": [
            {"ranks": ranks, "grid": list(grid)}
            for ranks, grid in rank_cases
        ],
    }


def _update_status(manifest: dict[str, Any]) -> None:
    records = [
        pair[backend]
        for pair in manifest["runs"] for backend in ("qcu", "quda")
        if backend in pair
    ]
    if any(item.get("status") == "RUNNING" for item in records):
        manifest["status"] = "RUNNING"
    elif any(item.get("status") == "TIMEOUT" for item in records):
        manifest["status"] = "TIMEOUT"
    elif any(
            isinstance(pair.get("quda"), Mapping) and
            pair["quda"].get("independent_residual", {}).get("required") is True
            and pair["quda"].get("independent_residual", {}).get("status")
            != "PASS"
            for pair in manifest.get("runs", [])):
        manifest["status"] = "PARTIAL"
    elif any(item.get("status") != "PASS" for item in records):
        manifest["status"] = "PARTIAL"
    else:
        manifest["status"] = "COMPLETE"
    manifest["updated_utc"] = _utc_now()


def _run_deferred_residuals(
    args: argparse.Namespace, manifest_path: Path,
) -> int:
    """Run deferred NumPy residuals on a CPU partition and finalize summary."""
    if os.environ.get("SLURM_JOB_PARTITION") == "kshdnormal":
        raise SystemExit(
            "refusing CPU-only independent residuals on the DCU partition")
    manifest_path = manifest_path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise SystemExit(f"invalid matrix manifest: {manifest_path}")
    config = manifest.get("configuration", {})
    if not isinstance(config, Mapping):
        raise SystemExit("matrix manifest has no configuration object")
    gauge = Path(config["gauge"]).expanduser().resolve()
    source = Path(config["source"]).expanduser().resolve()
    lattice = [int(value) for value in config["lattice_xyzt"]]
    dtype = str(config["dtype"])
    atol = float(config["tolerance"])
    environment = _run_environment()[0]
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["ROCR_VISIBLE_DEVICES"] = ""
    started = time.monotonic()
    deadline = started + float(args.wall_limit)
    summary_path = manifest_path.parent / "summary.json"

    for pair in manifest.get("runs", []):
        if not isinstance(pair, dict) or not isinstance(
                pair.get("quda"), Mapping):
            continue
        record = pair["quda"]
        independent = record.get("independent_residual")
        if not isinstance(independent, dict) or not independent.get("required"):
            continue
        if independent.get("status") == "PASS":
            continue
        if record.get("status") != "PASS":
            continue
        ranks = int(pair["ranks"])
        grid = [int(value) for value in pair["grid"]]
        solution = Path(record["solution"]).expanduser().resolve()
        report_path = Path(independent["report"]).expanduser().resolve()
        stdout_path = Path(independent.get(
            "stdout", report_path.with_name(
                "independent-residual.stdout.txt"))).expanduser().resolve()
        stderr_path = Path(independent.get(
            "stderr", report_path.with_name(
                "independent-residual.stderr.txt"))).expanduser().resolve()
        quda_report = json.loads(
            Path(record["report"]).read_text(encoding="utf-8"))
        quda_config = quda_report.get("config", {})
        command = _quda_residual_command(
            mpirun=args.mpirun, ranks=ranks, lattice=lattice, grid=grid,
            gauge=gauge, source=source, solution=solution, dtype=dtype,
            mass=float(quda_config["mass"]), csw=float(quda_config["csw"]),
            atol=atol, report=report_path)
        remaining = deadline - time.monotonic()
        timeout_seconds = min(float(args.child_timeout), remaining - 5.0)
        independent.update({
            "command": command,
            "execution_node": "CPU partition",
            "timeout_seconds": max(0.0, timeout_seconds),
        })
        if timeout_seconds < 10.0:
            independent["status"] = "SKIPPED-BUDGET"
            record["status"] = "FAIL"
            continue
        returncode, timed_out, elapsed = _run_captured_command(
            command, timeout_seconds=timeout_seconds, deadline=deadline,
            stdout_path=stdout_path, stderr_path=stderr_path,
            environment=environment)
        report_status = None
        if report_path.is_file():
            try:
                report_status = json.loads(
                    report_path.read_text(encoding="utf-8")).get("status")
            except (OSError, ValueError, TypeError):
                report_status = None
        passed = (
            not timed_out and returncode == 0 and report_status == "PASS" and
            solution.is_file())
        independent.update({
            "returncode": 124 if timed_out else returncode,
            "elapsed_seconds": elapsed,
            "report_status": report_status,
            "solution_published": solution.is_file(),
            "status": "PASS" if passed else (
                "TIMEOUT" if timed_out else "FAIL"),
        })
        record["deferred_residual_completed"] = passed
        if not passed:
            record["status"] = "FAIL"
        _update_status(manifest)
        _atomic_json(manifest_path, manifest)

    manifest["residual_phase_finished_utc"] = _utc_now()
    manifest["residual_phase_elapsed_seconds"] = time.monotonic() - started
    manifest["residual_phase_limit_seconds"] = float(args.wall_limit)
    manifest["residual_phase_within_wall_limit"] = (
        time.monotonic() <= deadline)
    _update_status(manifest)
    _atomic_json(manifest_path, manifest)
    summary = summarize_manifest(manifest_path)
    _atomic_json(summary_path, summary)
    print(json.dumps({
        "event": "deferred-residual-summary",
        "status": summary["status"],
        "complete_matrix": summary["complete_matrix"],
        "all_gates_pass": summary["all_gates_pass"],
        "manifest": str(manifest_path),
        "summary": str(summary_path),
    }, ensure_ascii=False), flush=True)
    return 0 if summary["status"] == "PASS" else 1


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gauge", default=str(DEFAULT_LIME))
    parser.add_argument("--source", default=None)
    parser.add_argument("--lat", nargs=4, type=int, default=(24, 24, 24, 64))
    parser.add_argument("--mass", type=float, default=None)
    parser.add_argument("--csw", type=float, default=None)
    parser.add_argument("--dtype", choices=("c64", "c128"), default="c64")
    parser.add_argument("--atol", type=float, default=1.0e-6)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backends", default="qcu,quda")
    parser.add_argument(
        "--rank-case", action="append", default=[],
        help="rank:gx,gy,gz,gt; defaults to all of 1/2/4")
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--mpirun", default="mpirun")
    parser.add_argument("--wall-limit", type=float, default=DEFAULT_WALL_LIMIT)
    parser.add_argument("--slurm-time-limit", type=float, default=600.0)
    parser.add_argument("--child-timeout", type=float, default=DEFAULT_CHILD_LIMIT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--defer-independent-residual", action="store_true",
        help="publish QUDA solutions without running CPU residuals on DCU")
    parser.add_argument(
        "--residual-only-manifest", type=Path, default=None,
        help="finalize deferred residuals from this manifest on a CPU partition")
    args = parser.parse_args(argv)
    cases = args.rank_case or [
        "1:1,1,1,1", "2:2,1,1,1", "4:2,2,1,1"]
    args.rank_cases = _parse_rank_cases(parser, cases)
    args.backends_selected = _parse_backends(parser, args.backends)
    args.lat = [int(value) for value in args.lat]
    if any(value <= 0 or value % 2 for value in args.lat):
        parser.error("--lat must contain four positive even extents")
    for _, grid in args.rank_cases:
        if any(args.lat[axis] % grid[axis] for axis in range(4)):
            parser.error(f"grid {grid} does not divide lattice {args.lat}")
    if args.wall_limit <= 0 or args.slurm_time_limit <= 0 or \
            args.child_timeout <= 0:
        parser.error("matrix and child time limits must be positive")
    if args.wall_limit >= args.slurm_time_limit:
        parser.error("--wall-limit must be below --slurm-time-limit")
    if args.warmups < 0 or args.repeats < 1 or args.max_iter < 1:
        parser.error("warmups must be nonnegative; repeats/max-iter positive")
    if args.atol <= 0 or not math.isfinite(args.atol):
        parser.error("--atol must be finite and positive")
    if args.defer_independent_residual and "quda" not in args.backends_selected:
        parser.error("--defer-independent-residual requires the quda backend")
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    args.result_root = (
        args.result_root or
        REPO / "data" / "cuda_quda_matrix" / f"v{stamp}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.residual_only_manifest is not None:
        return _run_deferred_residuals(args, args.residual_only_manifest)
    if (os.environ.get("SLURM_JOB_PARTITION") == "kshdnormal" and
            not args.defer_independent_residual and not args.dry_run):
        raise SystemExit(
            "DCU matrix requires --defer-independent-residual so CPU-only "
            "independent residuals run on a CPU partition")
    result_root = args.result_root.expanduser().resolve()
    args.matrix_source = result_root / "source.h5"
    manifest_path = result_root / "run-manifest.json"
    summary_path = result_root / "summary.json"
    audit_path = result_root / "five-minute-audit.json"
    manifest = _manifest_payload(
        args, args.rank_cases, args.backends_selected, result_root)
    if args.dry_run:
        print(json.dumps({
            "status": "DRY-RUN",
            "manifest": manifest,
            "commands": {
                f"r{ranks}-{backend}": _backend_command(
                    backend, ranks, grid, args,
                    result_root / f"r{ranks}" / backend / "report.json")
                for ranks, grid in args.rank_cases
                for backend in args.backends_selected
            },
        }, ensure_ascii=False))
        return 0

    result_root.mkdir(parents=True, exist_ok=True)
    matrix_source = _materialise_source(args, result_root)
    manifest["configuration"]["source"] = str(matrix_source)
    manifest["configuration"]["source_sha256"] = _sha256_file(matrix_source)
    environment, trace_requested = _run_environment()
    device_count = _visible_device_count(environment)
    if device_count is None:
        devices, error = _query_devices()
        device_count = len(devices) if devices else None
        if error:
            manifest.setdefault("warnings", []).append(error)
    maximum_rank = max(ranks for ranks, _ in args.rank_cases)
    if device_count is not None and maximum_rank > device_count:
        raise SystemExit(
            f"requested r{maximum_rank} but only {device_count} DCU are visible")
    started = time.monotonic()
    deadline = started + float(args.wall_limit)
    manifest["started_utc"] = _utc_now()
    manifest["visible_device_count"] = device_count
    manifest["deadline_monotonic_seconds"] = deadline
    _atomic_json(manifest_path, manifest)

    for pair in manifest["runs"]:
        ranks = int(pair["ranks"])
        for backend in args.backends_selected:
            print(json.dumps({
                "event": "start",
                "ranks": ranks,
                "grid": pair["grid"],
                "backend": backend,
                "elapsed_seconds": time.monotonic() - started,
            }, ensure_ascii=False), flush=True)
            record = _execute_one(
                backend=backend,
                ranks=ranks,
                grid=pair["grid"],
                args=args,
                result_root=result_root,
                environment=environment,
                trace_requested=trace_requested,
                deadline=deadline,
                started_monotonic=started,
                audit_path=audit_path,
            )
            pair[backend] = record
            _update_status(manifest)
            _atomic_json(manifest_path, manifest)
            print(json.dumps({
                "event": "finish",
                "ranks": ranks,
                "backend": backend,
                "status": record["status"],
                "returncode": record.get("returncode"),
                "elapsed_seconds": record.get("elapsed_seconds"),
            }, ensure_ascii=False), flush=True)
            if record["status"] == "TIMEOUT":
                break
        if manifest["status"] == "TIMEOUT":
            break

    elapsed = time.monotonic() - started
    if not audit_path.is_file() and elapsed >= AUDIT_SECONDS:
        _atomic_json(audit_path, {
            "status": "AUDIT",
            "elapsed_seconds": elapsed,
            "message": "five-minute audit emitted at matrix completion",
        })
    manifest["finished_utc"] = _utc_now()
    manifest["elapsed_seconds"] = elapsed
    manifest["within_wall_limit"] = elapsed < float(args.wall_limit)
    _update_status(manifest)
    _atomic_json(manifest_path, manifest)
    summary = summarize_manifest(manifest_path)
    _atomic_json(summary_path, summary)
    print(json.dumps({
        "event": "summary",
        "status": summary["status"],
        "complete_matrix": summary["complete_matrix"],
        "all_gates_pass": summary["all_gates_pass"],
        "manifest": str(manifest_path),
        "summary": str(summary_path),
    }, ensure_ascii=False), flush=True)
    deferred_complete = bool(
        args.defer_independent_residual and
        all(
            isinstance(pair.get("qcu"), Mapping) and
            pair["qcu"].get("status") == "PASS" and
            pair["qcu"].get("within_wall_limit") is True and
            isinstance(pair.get("quda"), Mapping) and
            pair["quda"].get("status") == "PASS" and
            pair["quda"].get("within_wall_limit") is True and
            pair["quda"].get("independent_residual", {}).get("status")
            == "DEFERRED"
            for pair in manifest["runs"]))
    return 0 if summary["status"] == "PASS" or deferred_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
