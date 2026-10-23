#!/usr/bin/env python3
"""Orchestrate the Torch/CUDA/KUPL Legacy/Strict six-way benchmark.

The benchmark uses one real LIME-derived gauge/source pair and a shared
canonical null-vector bundle.  Each side runs in an isolated process because
libqcu and libquda cannot be safely loaded into one process.  QUDA uses the
same lattice and MG configuration but builds its own null vectors because the
selected sm_70 installation was built without QIO.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
BENCH = Path(__file__).resolve().parent
WORKER = BENCH / "qcu_six_way_worker.py"
HARNESS = REPO / "pyqcu/testing/qcu/multigrid/benchmarks/legacy_vs_strict.py"

DEFAULT_GAUGE = REPO / "data/benchmark_legacy_vs_strict_20261022/cases/8x8x8x16/gauge_8x8x8x16_c64.h5"
DEFAULT_SOURCE = REPO / "data/benchmark_legacy_vs_strict_20261022/cases/8x8x8x16/source_8x8x8x16_c64.h5"
DEFAULT_NULL = REPO / "data/benchmark_legacy_vs_strict_20261022/nullvec/8x8x8x16_nvec12_full_c64_csw1p160920226.h5"
DEFAULT_CUDA_LEGACY_CACHE = REPO / "data/benchmark_legacy_vs_strict_20261022/legacy_cache/8x8x8x16_E12_c64.h5"
DEFAULT_KUPL_LEGACY_CACHE = REPO / "data/logs/qcu_six_way_20261023/kupl_legacy_coarse_c64.h5"
DEFAULT_KUPL_STRICT_ASSETS = REPO / "data/qcu_six_way_20261023/kupl_strict_assets_c64.h5"
DEFAULT_OUT = REPO / "logs/qcu_six_way_20261023/formal"
DEFAULT_SOLUTION_DIR = REPO / "data/qcu_six_way_20261023/solutions"

MAIN_VARIANTS = (
    "torch-legacy",
    "torch-strict",
    "cuda-legacy",
    "cuda-strict",
    "kupl-legacy",
    "kupl-strict",
    "quda-strict",
)
DIAGNOSTIC_VARIANTS = ("cuda-legacy-fine-only",)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _mad(values: list[float]) -> float | None:
    if not values:
        return None
    center = statistics.median(values)
    return float(statistics.median(abs(value - center) for value in values))


def _json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=float) + "\n",
        encoding="utf-8",
    )


def _run_command(command: list[str], env: dict[str, str], stdout_path: Path,
                 timeout: float, resume: bool = False) -> int:
    if resume and stdout_path.is_file():
        report = stdout_path.with_suffix(".json")
        existing = _json(report)
        if existing is not None and existing.get("status") == "PASS":
            return 0
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    print("+", " ".join(command), flush=True)
    with stdout_path.open("w", encoding="utf-8") as handle:
        try:
            completed = subprocess.run(
                command, env=env, stdout=handle, stderr=subprocess.STDOUT,
                timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            handle.write(f"\nTIMEOUT after {timeout} seconds\n")
            return 124
    return int(completed.returncode)


def _base_env(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONUNBUFFERED"] = "1"
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        env["MPI_ALLOW_RUN_AS_ROOT"] = "1"
        env["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
        env["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"
    env["OMP_NUM_THREADS"] = str(args.threads)
    env["KUPL_NUMA_BIND"] = "0"
    env["QUDA_ENABLE_TUNING"] = "0"
    env["PYTORCH_CUDA_ALLOC_CONF"] = env.get(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.pop("PYQCU_STRICT_TRACE_FILE", None)
    env.pop("QUDA_MG_TRACE_FILE", None)
    quda = REPO / "data/quda-sm70-install"
    if (quda / "lib/libquda.so").is_file():
        env.setdefault("QUDA_INSTALL", str(quda))
        env["LD_LIBRARY_PATH"] = str(quda / "lib") + (
            ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    return env


def _warm_gpu(env: dict[str, str], seconds: float, out_dir: Path) -> None:
    if seconds <= 0:
        return
    code = (
        "import time, torch\n"
        "device=torch.device('cuda:0')\n"
        "a=torch.randn((2048,2048),device=device,dtype=torch.float32)\n"
        "start=time.perf_counter()\n"
        "n=0\n"
        "while time.perf_counter()-start < " + repr(float(seconds)) + ":\n"
        " a=a@a\n"
        " a=a/a.norm()\n"
        " n+=1\n"
        "torch.cuda.synchronize()\n"
        "print({'iterations':n,'seconds':time.perf_counter()-start})\n"
    )
    path = out_dir / "gpu_clock_warmup.stdout"
    _run_command([sys.executable, "-c", code], env, path, timeout=120)


def _cuda_options(args: argparse.Namespace) -> list[str]:
    return [
        "--gauge", str(args.gauge), "--source", str(args.source),
        "--lat", *map(str, args.lat), "--mass", str(args.mass),
        "--csw", str(args.csw), "--tol", str(args.tolerance),
        "--max-iter", str(args.max_iter), "--warmups", str(args.warmups),
        "--repeats", str(args.repeats), "--device", args.device,
    ]


def _run_cuda_legacy(args: argparse.Namespace, env: dict[str, str],
                     out_dir: Path, variant: str, deflate: bool) -> None:
    command = [
        sys.executable, str(HARNESS), "legacy", *_cuda_options(args),
        "--cache", str(args.cuda_legacy_cache),
        "--log-dir", str(out_dir / f"{variant}-log"),
        "--coarse-dof", str(args.nvec),
        "--coarse-max-iter", str(args.coarse_max_iter),
        "--coarse-restart", str(
            args.correction_interval if deflate else 0),
        "--coarse-tol-factor", str(args.coarse_tolerance / args.tolerance),
        "--solution-output", str(args.solution_dir / f"{variant}-solution.h5"),
        "--output", str(out_dir / f"{variant}.json"),
    ]
    if deflate:
        command.append("--deflate")
    _run_command(
        command, env, out_dir / f"{variant}.stdout", args.timeout, args.resume)


def _run_cuda_strict(args: argparse.Namespace, env: dict[str, str],
                     out_dir: Path) -> None:
    command = [
        sys.executable, str(HARNESS), "strict", *_cuda_options(args),
        "--nullvec", str(args.nullvec),
        "--null-dataset", args.null_dataset, "--nvec", str(args.nvec),
        "--restart", str(args.restart),
        "--coarse-max-iter", str(args.coarse_max_iter),
        "--coarse-tol-factor", str(args.coarse_tolerance / args.tolerance),
        "--max-krylov-bytes", str(args.max_workspace_bytes),
        "--solution-output", str(args.solution_dir / "cuda-strict-solution.h5"),
        "--output", str(out_dir / "cuda-strict.json"),
    ]
    _run_command(
        command, env, out_dir / "cuda-strict.stdout", args.timeout, args.resume)


def _run_cuda_trace(args: argparse.Namespace, env: dict[str, str],
                    out_dir: Path) -> None:
    trace = out_dir / f"cuda-strict-trace-{int(time.time())}.tsv"
    report = out_dir / "cuda-strict-trace.json"
    stdout = out_dir / "cuda-strict-trace.stdout"
    if args.resume and report.is_file() and _json(report):
        return
    command = [
        sys.executable, str(HARNESS), "strict", *_cuda_options(args),
        "--nullvec", str(args.nullvec),
        "--null-dataset", args.null_dataset, "--nvec", str(args.nvec),
        "--restart", str(args.restart),
        "--coarse-max-iter", str(args.coarse_max_iter),
        "--coarse-tol-factor", str(args.coarse_tolerance / args.tolerance),
        "--max-krylov-bytes", str(args.max_workspace_bytes),
        "--warmups", "0", "--repeats", "1", "--output", str(report),
    ]
    trace_env = dict(env)
    trace_env["PYQCU_STRICT_TRACE_FILE"] = str(trace)
    _run_command(command, trace_env, stdout, args.timeout, resume=False)


def _worker_command(args: argparse.Namespace, variant: str,
                    out_dir: Path) -> list[str]:
    command = [
        sys.executable, str(WORKER), "--variant", variant,
        "--gauge", str(args.gauge), "--source", str(args.source),
        "--nullvec", str(args.nullvec), "--null-dataset", args.null_dataset,
        "--lat", *map(str, args.lat), "--mass", str(args.mass),
        "--csw", str(args.csw), "--tolerance", str(args.tolerance),
        "--coarse-tolerance", str(args.coarse_tolerance),
        "--max-iter", str(args.max_iter),
        "--coarse-max-iter", str(args.coarse_max_iter),
        "--levels", str(args.levels), "--block", *map(str, args.block),
        "--nvec", str(args.nvec), "--nu", str(args.nu),
        "--restart", str(args.restart),
        "--correction-interval", str(args.correction_interval),
        "--warmups", str(args.warmups), "--repeats", str(args.repeats),
        "--seed", str(args.seed), "--device", args.device,
        "--max-workspace-bytes", str(args.max_workspace_bytes),
        "--output", str(out_dir / f"{variant}.json"),
    ]
    if variant == "torch-legacy":
        command.extend(["--legacy-cache", str(args.cuda_legacy_cache)])
    if variant in ("kupl-legacy", "kupl-strict"):
        command.extend(["--legacy-cache", str(args.kupl_legacy_cache)])
    if variant == "kupl-strict":
        command.extend([
            "--strict-assets", str(args.kupl_strict_assets),
            "--kupl-strict-abi", args.kupl_strict_abi,
        ])
    if variant in ("torch-legacy", "torch-strict", "kupl-legacy", "kupl-strict"):
        command.extend([
            "--solution-output", str(args.solution_dir / f"{variant}-solution.h5")])
    if variant == "quda-strict":
        command.extend([
            "--solution-output", str(args.solution_dir / "quda-strict-solution.h5")])
    return command


def _run_worker_variant(args: argparse.Namespace, env: dict[str, str],
                        out_dir: Path, variant: str) -> None:
    _run_command(
        _worker_command(args, variant, out_dir), env,
        out_dir / f"{variant}.stdout", args.timeout, args.resume)


def _run_quda_residual(args: argparse.Namespace, env: dict[str, str],
                       out_dir: Path) -> None:
    command = [
        sys.executable, str(WORKER), "--variant", "residual",
        "--gauge", str(args.gauge), "--source", str(args.source),
        "--solution", str(args.solution_dir / "quda-strict-solution.h5"),
        "--lat", *map(str, args.lat), "--mass", str(args.mass),
        "--csw", str(args.csw), "--tolerance", str(args.tolerance),
        "--warmups", "0", "--repeats", "1",
        "--output", str(out_dir / "quda-strict-residual.json"),
    ]
    _run_command(
        command, env, out_dir / "quda-strict-residual.stdout",
        args.timeout, args.resume)


def _run_solution_residual(args: argparse.Namespace, env: dict[str, str],
                           out_dir: Path, variant: str,
                           solution_name: str) -> None:
    command = [
        sys.executable, str(WORKER), "--variant", "residual",
        "--gauge", str(args.gauge), "--source", str(args.source),
        "--solution", str(args.solution_dir / solution_name),
        "--lat", *map(str, args.lat), "--mass", str(args.mass),
        "--csw", str(args.csw), "--tolerance", str(args.tolerance),
        "--warmups", "0", "--repeats", "1",
        "--output", str(out_dir / f"{variant}-residual.json"),
    ]
    _run_command(
        command, env, out_dir / f"{variant}-residual.stdout",
        args.timeout, args.resume)


def _parse_cuda_trace(path: Path) -> dict[str, Any]:
    result = {
        "path": str(path),
        "exists": path.is_file(),
        "solve_end_count": 0,
        "coarse_vcycle_stages": 0,
        "outer_iteration_stages": 0,
        "restriction_calls": 0,
        "prolongation_calls": 0,
    }
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split("\t")
        if not parts:
            continue
        if parts[0] == "solve_end":
            result["solve_end_count"] += 1
        elif parts[0] == "stage" and len(parts) >= 4:
            if parts[3] == "coarse_vcycle":
                result["coarse_vcycle_stages"] += 1
            elif parts[3] == "outer_iteration":
                result["outer_iteration_stages"] += 1
        elif parts[0] == "iteration_count" and len(parts) >= 5:
            if parts[3] == "restriction_calls":
                result["restriction_calls"] += int(parts[4])
            elif parts[3] == "prolongation_calls":
                result["prolongation_calls"] += int(parts[4])
    return result


def _parse_kupl_legacy(text: str) -> dict[str, Any]:
    iterations = [int(value) for value in re.findall(
        r"final iterations=(\d+)", text)]
    deflation = re.findall(
        r"deflation residual=.*?corrections=(\d+)", text, re.S)
    periodic = re.findall(
        r"coarse-correction iterations=.*?corrections=(\d+)", text, re.S)
    states = [int(value) for value in deflation + periodic]
    rejected = sum(1 for value in states if value == 0)
    attempts = len(states)
    return {
        "iterations": iterations,
        "median": _median([float(value) for value in iterations]),
        "attempted_vcycles": attempts,
        "rejected_attempts": rejected,
        "executed": attempts > 0,
        "effective_mode": (
            "fine_only_after_rejected_correction"
            if attempts > 0 and rejected == attempts else
            "periodic_coarse_correction" if attempts > 0 else "fine_only"),
    }


def _normalise_side(variant: str, report_path: Path, stdout_path: Path,
                    out_dir: Path) -> dict[str, Any]:
    raw = _json(report_path)
    if raw is None:
        return {
            "variant": variant,
            "status": "FAIL",
            "error": "missing or invalid side JSON",
            "raw_json": str(report_path),
            "raw_stdout": str(stdout_path),
        }
    text = (stdout_path.read_text(encoding="utf-8", errors="replace")
            if stdout_path.is_file() else "")

    if variant.startswith("cuda-"):
        if variant in ("cuda-legacy", "cuda-legacy-fine-only"):
            strategy = "legacy"
        elif variant == "cuda-strict":
            strategy = "strict"
        else:
            raise ValueError(variant)
        timing = raw.get("timing", {})
        setup = raw.get("mg_init_seconds")
        cache = (raw.get("cache_load_seconds") if strategy == "legacy"
                 else raw.get("null_vector_load_seconds"))
        warmups = raw.get("warmup_seconds", [])
        steady = raw.get("steady_seconds", [])
        iterations = raw.get("iterations", [])
        residual = raw.get("independent_true_residual",
                           raw.get("final_true_residual"))
        memory_resources = raw.get("resources", {}).get("device", {})
        coarse = raw.get("coarse_execution", {})
        if strategy == "legacy" and "executed" not in coarse:
            coarse = {
                **coarse,
                "executed": bool(int(coarse.get("attempted_vcycles", 0)) > 0),
            }
        if strategy == "legacy" and raw.get("deflate"):
            attempts = int(coarse.get("attempted_vcycles", 0))
            coarse = {
                **coarse,
                "effective_mode": (
                    "initial_deflation_only" if attempts <= 1 else
                    "initial_deflation_plus_periodic_correction"),
            }
        elif variant == "cuda-legacy-fine-only":
            attempts = int(coarse.get("attempted_vcycles", 0))
            coarse = {
                **coarse,
                "effective_mode": (
                    "periodic_coarse_correction_without_initial_deflation"
                    if attempts > 0 else "fine_only"),
            }
        independent = _json(out_dir / f"{variant}-residual.json")
        if independent is not None:
            residual = independent.get("independent_relative")
        if strategy == "strict":
            trace_files = sorted(out_dir.glob("cuda-strict-trace-*.tsv"))
            trace = _parse_cuda_trace(trace_files[-1]) if trace_files else {}
            coarse = {
                "kind": "strict_recursive_vcycle",
                "executed": bool(
                    trace.get("coarse_vcycle_stages", 0) > 0 or
                    any(int(value) > 0 for value in iterations)),
                "coarse_vcycle_stages_diagnostic": trace.get(
                    "coarse_vcycle_stages"),
                "restriction_calls": trace.get("restriction_calls"),
                "prolongation_calls": trace.get("prolongation_calls"),
                "evidence": "separate trace-on diagnostic run; not used for timing",
                "trace": trace,
            }
        return {
            "schema": "pyqcu.qcu-six-way.side.v1",
            "variant": variant,
            "backend": "cuda",
            "strategy": strategy,
            "status": (
                "PASS" if raw.get("status") == "PASS" and
                residual is not None and
                float(residual) <= float(raw.get("tol", 0.0)) else "FAIL"),
            "lattice": raw.get("lattice"),
            "mass": raw.get("mass"),
            "csw": raw.get("csw"),
            "precision": raw.get("precision", "c64"),
            "configuration": {
                "tolerance": raw.get("tol"),
                "max_iter": raw.get("max_iter"),
                "levels": 2,
                "block": [2, 2, 2, 2],
                "nvec": raw.get("nvec", 12 if strategy == "strict" else 12),
                "coarse_dof": raw.get("coarse_dof"),
                "coarse_spin": 2 if strategy == "strict" else None,
                "coarse_tolerance": raw.get("tol", 1e-6) * (
                    1e2 if strategy == "strict" else 1e2),
                "coarse_max_iter": raw.get("coarse_max_iter"),
                "restart": raw.get("restart"),
                "legacy_correction_interval": raw.get("coarse_restart"),
                "warmups": len(warmups),
                "repeats": len(steady),
                "deflate": raw.get("deflate"),
            },
            "inputs": raw,
            "timing": {
                "setup_seconds": setup,
                "setup_scope": raw.get("mg_init_scope"),
                "cache_load_seconds": cache,
                "cache_scope": (
                    "33-tensor Legacy cache HDF5 load"
                    if strategy == "legacy" else
                    "canonical full null-vector HDF5 load"),
                "runtime_reinit_seconds": raw.get("runtime_reinit_seconds"),
                "warmup_seconds": warmups,
                "first_solve_seconds": warmups[0] if warmups else (
                    steady[0] if steady else None),
                "steady_seconds": steady,
                "steady_median_seconds": raw.get("steady_median_seconds"),
                "steady_mad_seconds": raw.get("steady_mad_seconds"),
                "steady_scope": raw.get("steady_scope"),
            },
            "iterations": {
                "samples": iterations,
                "median": _median([float(value) for value in iterations]),
                "semantics": (
                    "Legacy outer fine BiCGStab iterations from C++ log"
                    if strategy == "legacy" else
                    "strict outer right-preconditioned FGMRES iterations"),
            },
            "residual": {
                "independent_relative": residual,
                "native_relative": raw.get("native_true_residual"),
                "gate": raw.get("tol"),
                "pass": bool(residual is not None and residual <= raw.get("tol", 0)),
                "source": (
                    independent.get("source") if independent is not None
                    else raw.get("residual_source")),
                "independent_json": (
                    str(out_dir / f"{variant}-residual.json")
                    if independent is not None else None),
                "scope": "full",
            },
            "coarse_execution": coarse,
            "memory": {
                "device_memory_applicable": True,
                "device_wide": memory_resources,
                "host_peak_rss_bytes": raw.get("resources", {}).get(
                    "process_peak_rss_bytes"),
                "allocator": raw.get("memory"),
            },
            "raw_json": str(report_path),
            "raw_stdout": str(stdout_path),
        }

    side = dict(raw)
    side["raw_json"] = str(report_path)
    side["raw_stdout"] = str(stdout_path)
    if variant == "kupl-legacy":
        parsed = _parse_kupl_legacy(text)
        side.setdefault("iterations", {})["samples"] = parsed["iterations"]
        side["iterations"]["median"] = parsed["median"]
        side.setdefault("coarse_execution", {}).update({
            "executed": parsed["executed"],
            "attempted_vcycles": parsed["attempted_vcycles"],
            "rejected_attempts": parsed["rejected_attempts"],
            "effective_mode": parsed["effective_mode"],
            "source": "KUPL C++ debug stdout",
        })
        _write_json(out_dir / "kupl-legacy-diagnostics.json", parsed)
    if variant == "quda-strict":
        independent = _json(out_dir / "quda-strict-residual.json")
        if independent is not None:
            side.setdefault("residual", {})["independent_relative"] = (
                independent.get("independent_relative"))
            side["residual"]["independent_source"] = independent.get("source")
            side["residual"]["independent_pass"] = independent.get("pass")
            side["independent_residual_json"] = str(
                out_dir / "quda-strict-residual.json")
        residual = side.get("residual", {})
        side["status"] = (
            "PASS" if raw.get("status") == "PASS" and
            residual.get("backend_pass", False) and
            residual.get("independent_pass", False) else "FAIL")
    side["timing"].setdefault(
        "steady_mad_seconds",
        _mad([float(value) for value in side["timing"].get(
            "steady_seconds", [])]))
    side["iterations"].setdefault(
        "median", _median([float(value) for value in side["iterations"].get(
            "samples", []) if value is not None]))
    return side


def _environment_payload(args: argparse.Namespace) -> dict[str, Any]:
    def command_text(command: list[str]) -> str:
        try:
            return subprocess.check_output(
                command, text=True, stderr=subprocess.STDOUT).strip()
        except Exception as error:
            return f"unavailable: {error}"

    libraries = {
        "libqcu_sm70": REPO / "data/libqcu-sm70/libqcu.so",
        "libqcu_kup": REPO / "cpp/kupl/qcu/build/libqcu.so",
        "libquda_sm70": Path(os.environ.get(
            "QUDA_INSTALL", REPO / "data/quda-sm70-install")) / "lib/libquda.so",
    }
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_head": command_text(["git", "rev-parse", "HEAD"]),
        "git_status": command_text(["git", "status", "--short"]),
        "python": sys.version,
        "python_executable": sys.executable,
        "nvidia_smi": command_text([
            "nvidia-smi", "--query-gpu=index,pci.bus_id,name,uuid,compute_cap,"
            "memory.total,driver_version", "--format=csv,noheader"]),
        "lscpu": command_text(["lscpu"]),
        "quda_install": os.environ.get("QUDA_INSTALL"),
        "quda_build_dir": os.environ.get("QUDA_BUILD_DIR"),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
        "omp_num_threads": str(args.threads),
        "kupl_ranks": 1,
        "kupl_numa_bind": "0",
        "gpu_device": args.device,
        "libraries": {
            name: {
                "path": str(path),
                "exists": path.is_file(),
                "sha256": _sha256(path) if path.is_file() else None,
            }
            for name, path in libraries.items()
        },
        "packages": command_text([
            sys.executable, "-c",
            "import json, torch, pyquda, numpy, mpi4py; "
            "print(json.dumps({'torch':torch.__version__,"
            "'torch_cuda':torch.version.cuda,'pyquda':pyquda.__version__,"
            "'numpy':numpy.__version__,'mpi4py':mpi4py.__version__}))"]),
    }


def _input_payload(args: argparse.Namespace) -> dict[str, Any]:
    case_manifest = Path(args.gauge).expanduser().resolve().with_name(
        "case_manifest.json")
    full_lime = (
        Path(args.full_lime).expanduser().resolve()
        if args.full_lime is not None else
        Path.home() / "data" /
        "beta6.20_mu-0.2770_ms-0.2310_L24x64_cfg_9950.lime"
    )
    full_lime_payload: dict[str, Any] = {
        "path": str(full_lime),
        "available": full_lime.is_file(),
        "sha256": _sha256(full_lime) if full_lime.is_file() else None,
    }
    return {
        "selected_case": {
            "lattice": list(args.lat),
            "mass": args.mass,
            "csw": args.csw,
            "precision": "c64",
            "source_rule": "first-prefix slice of the full PCG64 source",
            "provenance": "real LIME prefix subvolume; seam boundaries are nonperiodic",
            "gauge": {"path": str(args.gauge), "sha256": _sha256(args.gauge)},
            "source": {"path": str(args.source), "sha256": _sha256(args.source)},
            "canonical_null_vectors": {
                "path": str(args.nullvec), "sha256": _sha256(args.nullvec),
                "dataset": args.null_dataset,
            },
            "case_manifest": _json(case_manifest),
        },
        "full_lime": full_lime_payload,
        "caches": {
            "cuda_legacy": {
                "path": str(args.cuda_legacy_cache),
                "sha256": _sha256(args.cuda_legacy_cache),
                "size_bytes": args.cuda_legacy_cache.stat().st_size,
            },
            "kupl_legacy": {
                "path": str(args.kupl_legacy_cache),
                "sha256": _sha256(args.kupl_legacy_cache),
                "size_bytes": args.kupl_legacy_cache.stat().st_size,
            },
            "kupl_strict_assets": {
                "path": str(args.kupl_strict_assets),
                "sha256": _sha256(args.kupl_strict_assets),
                "size_bytes": args.kupl_strict_assets.stat().st_size,
                "provenance": "Torch strict hierarchy export from the same gauge and canonical null vectors",
            },
        },
    }


def _residual_value(side: dict[str, Any]) -> float | None:
    value = side.get("residual", {}).get("independent_relative")
    return float(value) if value is not None else None


def _peak_memory(side: dict[str, Any]) -> int | None:
    memory = side.get("memory", {})
    if not memory.get("device_memory_applicable"):
        return None
    for key in ("device_wide", "first_solve_device_wide"):
        value = memory.get(key, {}).get("peak_delta_over_baseline_bytes")
        if value is not None:
            return int(value)
    device = memory.get("device_wide", {})
    if "peak_delta_over_baseline_bytes" in device:
        return int(device["peak_delta_over_baseline_bytes"])
    return None


def _workspace_bytes(side: dict[str, Any]) -> int | None:
    memory = side.get("memory", {})
    allocator = memory.get("allocator")
    if isinstance(allocator, dict):
        for key in ("fused_workspace_planned_bytes", "planned_accounted_owned_bytes"):
            value = allocator.get(key)
            if value is not None:
                return int(value)
    samples = memory.get("solve_workspace_samples_bytes")
    if samples:
        return int(samples[0])
    value = memory.get("first_solve_allocator_delta_bytes")
    return int(value) if value is not None else None


def _summary_rows(sides: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for variant, side in sides.items():
        timing = side.get("timing", {})
        coarse = side.get("coarse_execution", {})
        rows.append({
            "variant": variant,
            "backend": side.get("backend"),
            "strategy": side.get("strategy"),
            "status": side.get("status"),
            "coarse_executed": coarse.get("executed"),
            "effective_mode": coarse.get("effective_mode", coarse.get("kind")),
            "setup_seconds": timing.get("setup_seconds"),
            "cache_load_seconds": timing.get("cache_load_seconds"),
            "first_solve_seconds": timing.get("first_solve_seconds"),
            "first_workspace_allocator_delta_bytes": side.get(
                "memory", {}).get("first_solve_allocator_delta_bytes"),
            "workspace_bytes": _workspace_bytes(side),
            "steady_median_seconds": timing.get("steady_median_seconds"),
            "steady_mad_seconds": timing.get("steady_mad_seconds"),
            "iterations_median": side.get("iterations", {}).get("median"),
            "true_residual": _residual_value(side),
            "peak_device_bytes": _peak_memory(side),
            "host_peak_rss_bytes": side.get("memory", {}).get("host_peak_rss_bytes"),
            "raw_json": side.get("raw_json"),
            "raw_stdout": side.get("raw_stdout"),
        })
    return rows


def _comparison_payload(sides: dict[str, dict[str, Any]]) -> dict[str, Any]:
    quda = sides.get("quda-strict", {})
    quda_time = quda.get("timing", {}).get("steady_median_seconds")
    gpu_strict = {}
    for name in ("torch-strict", "cuda-strict"):
        value = sides.get(name, {}).get("timing", {}).get("steady_median_seconds")
        gpu_strict[name] = {
            "steady_median_seconds": value,
            "quda_over_variant": (
                float(quda_time) / float(value)
                if quda_time and value else None),
            "variant_over_quda": (
                float(value) / float(quda_time)
                if quda_time and value else None),
        }
    legacy = {
        name: {
            "steady_median_seconds": sides.get(name, {}).get(
                "timing", {}).get("steady_median_seconds"),
            "coarse_execution": sides.get(name, {}).get("coarse_execution"),
        }
        for name in ("torch-legacy", "cuda-legacy", "kupl-legacy")
    }
    cuda_legacy = sides.get("cuda-legacy", {}).get(
        "timing", {}).get("steady_median_seconds")
    cuda_fine = sides.get("cuda-legacy-fine-only", {}).get(
        "timing", {}).get("steady_median_seconds")
    return {
        "fair_gpu_strict_config": {
            "variants": ["torch-strict", "cuda-strict", "quda-strict"],
            "same_gauge_source": True,
            "same_lattice": True,
            "same_precision": True,
            "same_tolerance_max_iter_levels_block_nvec": True,
            "quda_uses_same_canonical_null_basis": False,
            "note": "QUDA generated null vectors on the shared gauge because the sm_70 install has QUDA_QIO=OFF",
            "rows": gpu_strict,
        },
        "legacy_modes": legacy,
        "cuda_legacy_algorithm_ablation": {
            "coarse_enabled_seconds": cuda_legacy,
            "fine_only_seconds": cuda_fine,
            "fine_only_over_coarse_enabled": (
                float(cuda_fine) / float(cuda_legacy)
                if cuda_fine and cuda_legacy else None),
            "note": "same GPU/input/config; only the initial coarse deflation switch differs",
        },
        "cpu_backend_control": {
            "variant": "kupl-strict",
            "steady_median_seconds": sides.get("kupl-strict", {}).get(
                "timing", {}).get("steady_median_seconds"),
            "note": "CPU MPI/OpenMP result; not an accelerator speedup",
        },
    }


def _markdown_summary(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# QCU six-way raw summary",
        "",
        "| variant | status | coarse executed | mode | setup s | cache s | first s | steady median s | iterations | true res | peak device B |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def value(key: str, digits: int = 6) -> str:
            item = row.get(key)
            return "-" if item is None else f"{float(item):.{digits}f}"

        residual = row.get("true_residual")
        residual_text = "-" if residual is None else f"{float(residual):.3e}"
        lines.append(
            f"| {row['variant']} | {row['status']} | {row['coarse_executed']} | "
            f"{row['effective_mode']} | {value('setup_seconds')} | "
            f"{value('cache_load_seconds')} | {value('first_solve_seconds')} | "
            f"{value('steady_median_seconds')} | {row['iterations_median']} | "
            f"{residual_text} | {row['peak_device_bytes']} |")
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gauge", type=Path, default=DEFAULT_GAUGE)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--full-lime", type=Path, default=None,
        help="optional source LIME provenance; defaults to the reference LIME under ~/data")
    parser.add_argument("--nullvec", type=Path, default=DEFAULT_NULL)
    parser.add_argument("--null-dataset", default="null")
    parser.add_argument("--cuda-legacy-cache", type=Path,
                        default=DEFAULT_CUDA_LEGACY_CACHE)
    parser.add_argument("--kupl-legacy-cache", type=Path,
                        default=DEFAULT_KUPL_LEGACY_CACHE)
    parser.add_argument("--kupl-strict-assets", type=Path,
                        default=DEFAULT_KUPL_STRICT_ASSETS)
    parser.add_argument("--lat", nargs=4, type=int, default=(8, 8, 8, 16))
    parser.add_argument("--mass", type=float, default=-0.277)
    parser.add_argument("--csw", type=float, default=1.160920226)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--coarse-tolerance", type=float, default=1e-4)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--coarse-max-iter", type=int, default=200)
    parser.add_argument("--levels", type=int, default=2)
    parser.add_argument("--block", nargs=4, type=int, default=(2, 2, 2, 2))
    parser.add_argument("--nvec", type=int, default=12)
    parser.add_argument("--nu", type=int, default=1)
    parser.add_argument("--restart", type=int, default=20)
    parser.add_argument("--correction-interval", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--max-workspace-bytes", type=int, default=512 << 20)
    parser.add_argument("--gpu-warmup-seconds", type=float, default=3.0)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--solution-dir", type=Path,
                        default=DEFAULT_SOLUTION_DIR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-trace", action="store_true",
        help="run the separate trace-on diagnostic; formal timings remain trace-off")
    parser.add_argument(
        "--kupl-strict-abi", choices=("public", "raw", "both"),
        default="both", help="KUPL Strict public/raw ABI validation mode")
    args = parser.parse_args(argv)
    args.lat = tuple(int(value) for value in args.lat)
    args.block = tuple(int(value) for value in args.block)
    for name in ("gauge", "source", "nullvec", "cuda_legacy_cache",
                 "kupl_legacy_cache", "kupl_strict_assets"):
        path = getattr(args, name).expanduser().resolve()
        if not path.is_file():
            parser.error(f"--{name.replace('_', '-')} does not exist: {path}")
        setattr(args, name, path)
    if args.levels != 2:
        parser.error("this formal driver currently fixes --levels 2")
    if tuple(args.block) != (2, 2, 2, 2):
        parser.error("this formal driver currently fixes --block 2 2 2 2")
    if args.repeats < 3:
        parser.error("formal comparison requires at least 3 steady repeats")
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    args.solution_dir = args.solution_dir.expanduser().resolve()
    args.solution_dir.mkdir(parents=True, exist_ok=True)
    env = _base_env(args)

    _write_json(out_dir / "inputs.json", _input_payload(args))
    _write_json(out_dir / "environment.json", _environment_payload(args))
    _warm_gpu(env, args.gpu_warmup_seconds, out_dir)

    _run_cuda_legacy(args, env, out_dir, "cuda-legacy", deflate=True)
    _run_cuda_legacy(
        args, env, out_dir, "cuda-legacy-fine-only", deflate=False)
    _run_cuda_strict(args, env, out_dir)
    if args.allow_trace:
        _run_cuda_trace(args, env, out_dir)
    _run_solution_residual(
        args, env, out_dir, "cuda-legacy", "cuda-legacy-solution.h5")
    _run_solution_residual(
        args, env, out_dir, "cuda-legacy-fine-only",
        "cuda-legacy-fine-only-solution.h5")
    _run_solution_residual(
        args, env, out_dir, "cuda-strict", "cuda-strict-solution.h5")
    for variant in ("torch-legacy", "torch-strict", "kupl-legacy", "kupl-strict"):
        _run_worker_variant(args, env, out_dir, variant)
    _run_worker_variant(args, env, out_dir, "quda-strict")
    _run_quda_residual(args, env, out_dir)

    sides: dict[str, dict[str, Any]] = {}
    for variant in (*DIAGNOSTIC_VARIANTS[:1], *MAIN_VARIANTS):
        if variant == "cuda-legacy-fine-only":
            report = out_dir / f"{variant}.json"
            raw = _json(report)
            if raw is not None:
                raw["variant"] = variant
                raw["coarse_execution"] = {
                    "kind": "legacy_periodic_correction",
                    "executed": bool(raw.get("coarse_execution", {}).get(
                        "attempted_vcycles", 0)),
                    "attempted_vcycles": raw.get("coarse_execution", {}).get(
                        "attempted_vcycles"),
                    "effective_mode": raw.get("coarse_execution", {}).get(
                        "effective_mode"),
                    "source": raw.get("coarse_execution", {}).get("source"),
                }
                _write_json(report, raw)
            sides[variant] = _normalise_side(
                variant, report,
                out_dir / f"{variant}.stdout", out_dir)
        else:
            sides[variant] = _normalise_side(
                variant, out_dir / f"{variant}.json",
                out_dir / f"{variant}.stdout", out_dir)

    rows = _summary_rows(sides)
    comparison = _comparison_payload(sides)
    summary = {
        "schema": "pyqcu.qcu-six-way.summary.v1",
        "status": "PASS" if all(
            side.get("status") == "PASS" for side in sides.values()) else "FAIL",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol": {
            "lattice": list(args.lat),
            "mass": args.mass,
            "csw": args.csw,
            "precision": "c64",
            "tolerance": args.tolerance,
            "coarse_tolerance": args.coarse_tolerance,
            "max_iter": args.max_iter,
            "coarse_max_iter": args.coarse_max_iter,
            "levels": args.levels,
            "block": list(args.block),
            "nvec": args.nvec,
            "coarse_spin": 2,
            "nu_pre": args.nu,
            "nu_post": args.nu,
            "restart": args.restart,
            "legacy_correction_interval": args.correction_interval,
            "warmups": args.warmups,
            "steady_repeats": args.repeats,
            "trace": False,
            "quda_enable_tuning": 0,
        },
        "comparison": comparison,
        "rows": rows,
        "sides": sides,
        "blockers": [
            "QUDA sm_70 installation has QUDA_QIO=OFF, so QUDA generated its own null vectors from the same gauge instead of loading the canonical bundle.",
            "KUPL Legacy rejects a non-improving coarse correction and falls back to the validated parity-Schur BiCGStab; the report retains the attempted V-cycle and fallback reason.",
        ],
    }
    _write_json(out_dir / "summary.json", summary)
    _write_csv(out_dir / "summary.csv", rows)
    (out_dir / "summary.md").write_text(
        _markdown_summary(rows), encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "summary": str(out_dir / "summary.json"),
        "rows": rows,
    }, indent=2, ensure_ascii=False, default=float))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
