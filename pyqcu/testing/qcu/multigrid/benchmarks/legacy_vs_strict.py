#!/usr/bin/env python3
"""Run one isolated CUDA QCU Legacy or Strict MG benchmark side.

The case files are canonical QCU HDF5 tensors.  Each invocation owns one
Python process, one MPI rank, and one CUDA device.  Legacy asset construction
and solve can be run as separate subcommands so MG initialization time is not
hidden in steady-solve timing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[5]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _json_mad(values: list[float]) -> float | None:
    if not values:
        return None
    center = statistics.median(values)
    return float(statistics.median(abs(value - center) for value in values))


def _load_case(gauge_path: Path, source_path: Path, lat: tuple[int, ...],
               device: Any):
    import torch
    from pyqcu import tools
    from pyqcu.testing.simple._common import read_h5

    gauge_full = torch.from_numpy(read_h5(gauge_path)).to(
        device=device, dtype=torch.complex64).contiguous()
    source_full = torch.from_numpy(read_h5(source_path)).to(
        device=device, dtype=torch.complex64).contiguous()
    if tuple(int(value) for value in gauge_full.shape[-4:]) != lat:
        raise ValueError(f"gauge shape {gauge_full.shape[-4:]} != {lat}")
    if tuple(int(value) for value in source_full.shape[-4:]) != lat:
        raise ValueError(f"source shape {source_full.shape[-4:]} != {lat}")
    gauge = tools.oooxyzt2poooxyzt(gauge_full).contiguous()
    rhs = tools.oooxyzt2poooxyzt(source_full).contiguous()
    return gauge_full, source_full, gauge, rhs


def _fine_operator(gauge_full: Any, lat: tuple[int, ...], mass: float,
                   csw: float, device: Any):
    import torch
    from pyqcu.torch import dslash

    kappa = 1.0 / (2.0 * mass + 8.0)
    kappa_t = torch.tensor([kappa], dtype=torch.float32, device=device)
    u0 = torch.ones(1, dtype=torch.float32, device=device)
    clover = dslash.make_clover(
        gauge_full, kappa=kappa_t, u_0=u0, csw=csw)
    operator = dslash.operator(
        U=gauge_full, clover_term=clover, kappa=kappa_t, u_0=u0,
        support_parity=True, verbose=False)
    return operator, clover, kappa, kappa_t, u0


def _true_residual(output: Any, rhs: Any, gauge: Any, gauge_full: Any,
                   clover: Any, mass: float,
                   operator: Any | None = None) -> float:
    import torch
    from pyqcu import tools
    from pyqcu.torch import dslash

    del gauge
    kappa = 1.0 / (2.0 * mass + 8.0)
    solution_full = tools.poooxyzt2oooxyzt(output)
    rhs_full = tools.poooxyzt2oooxyzt(rhs)
    del operator
    u0 = torch.ones(1, dtype=torch.float32, device=solution_full.device)
    residual = dslash.give_wilson(
        solution_full, gauge_full, kappa, u0, with_I=True)
    residual = residual + dslash.give_clover(solution_full, clover) - rhs_full
    denominator = max(float(torch.linalg.vector_norm(rhs_full)), 1.0e-30)
    return float(torch.linalg.vector_norm(residual) / denominator)


def _memory_snapshot(device: Any) -> dict[str, Any]:
    import torch

    torch.cuda.synchronize(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "device": str(device),
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
    }


def _process_peak_rss_bytes() -> int:
    import resource

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


class _DeviceMemorySampler:
    """Sample device-wide usage so native QCU allocations are included."""

    def __init__(self, device: Any, interval_seconds: float = 0.05):
        self.device = device
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._baseline = self._used_bytes()
        self._peak = self._baseline
        self._samples = 0
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run, name="qcu-device-memory-sampler", daemon=True)

    def _used_bytes(self) -> int:
        import torch

        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        return int(total_bytes) - int(free_bytes)

    def _sample(self) -> None:
        used_bytes = self._used_bytes()
        with self._lock:
            self._baseline = min(self._baseline, used_bytes)
            self._peak = max(self._peak, used_bytes)
            self._samples += 1

    def _run(self) -> None:
        try:
            while not self._stop.wait(self.interval_seconds):
                self._sample()
        except BaseException as exc:
            self._error = exc

    def start(self) -> "_DeviceMemorySampler":
        self._sample()
        self._thread.start()
        return self

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join()
        self._sample()
        if self._error is not None:
            raise RuntimeError("device memory sampler failed") from self._error
        with self._lock:
            return {
                "scope": "device-wide cudaMemGetInfo; includes CUDA context and native QCU allocations",
                "sample_interval_seconds": self.interval_seconds,
                "sample_count": self._samples,
                "baseline_used_bytes": self._baseline,
                "peak_used_bytes": self._peak,
                "peak_delta_over_baseline_bytes": max(
                    0, self._peak - self._baseline),
            }


def _resource_payload(memory: Mapping[str, Any], *,
                      cache_bytes: int | None = None,
                      null_vector_bytes: int | None = None,
                      strict_runtime_memory: Mapping[str, Any] | None = None,
                      device_peak: Mapping[str, Any] | None = None,
                      ) -> dict[str, Any]:
    strict_memory = {} if strict_runtime_memory is None else {
        "asset_resident_bytes": int(
            strict_runtime_memory.get("asset_resident_bytes", 0)),
        "fused_workspace_resident_bytes": int(
            strict_runtime_memory.get(
                "fused_workspace_resident_bytes", 0)),
        "fused_workspace_planned_bytes": int(
            strict_runtime_memory.get(
                "fused_workspace_planned_bytes", 0)),
        "coarse_workspace_bytes": int(
            strict_runtime_memory.get("coarse_workspace_bytes", 0)),
        "fine_transfer_bytes": int(
            strict_runtime_memory.get("fine_transfer_bytes", 0)),
        "borrowed_gauge_clover_bytes": int(
            strict_runtime_memory.get(
                "borrowed_gauge_clover_bytes", 0)),
        "lattice_known_device_bytes": int(
            strict_runtime_memory.get("lattice_known_device_bytes", 0)),
        "runtime_owned_bytes": int(
            strict_runtime_memory.get("accounted_owned_bytes", 0)),
        "known_live_device_bytes": int(
            strict_runtime_memory.get("known_live_device_bytes", 0)),
        "setup_detached_storage_bytes": int(
            strict_runtime_memory.get(
                "setup_detached_storage_bytes", 0)),
        "setup_allocator_released_bytes": int(
            strict_runtime_memory.get(
                "setup_allocator_released_bytes", 0)),
    }
    return {
        "schema": "pyqcu.mg_resources.v1",
        "process_peak_rss_bytes": _process_peak_rss_bytes(),
        "torch_peak_allocated_bytes": int(
            memory.get("peak_allocated_bytes", 0)),
        "torch_peak_reserved_bytes": int(
            memory.get("peak_reserved_bytes", 0)),
        "legacy_cache_bytes": cache_bytes,
        "strict_null_vector_bytes": null_vector_bytes,
        "strict_runtime": strict_memory,
        "device": {} if device_peak is None else dict(device_peak),
        "memory_detail": dict(memory),
    }


def _common_payload(args: argparse.Namespace) -> dict[str, Any]:
    from pyqcu.testing.simple._common import read_h5_metadata

    gauge = Path(args.gauge).expanduser().resolve()
    source = Path(args.source).expanduser().resolve()
    return {
        "side": args.side,
        "lattice": list(args.lat),
        "mass": float(args.mass),
        "csw": float(args.csw),
        "precision": "c64",
        "gauge": {"path": str(gauge), "sha256": _sha256(gauge),
                  "metadata": read_h5_metadata(gauge)},
        "source": {"path": str(source), "sha256": _sha256(source),
                   "metadata": read_h5_metadata(source)},
    }


def _build_legacy(args: argparse.Namespace) -> dict[str, Any]:
    import h5py
    import torch
    from pyqcu.cuda._multi_gpu import build_schur_levels

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    device_sampler = _DeviceMemorySampler(device).start()
    started = time.perf_counter()
    gauge_full, _, gauge, _ = _load_case(
        Path(args.gauge), Path(args.source), tuple(args.lat), device)
    operator, _clover, kappa, _kappa_t, _u0 = _fine_operator(
        gauge_full, tuple(args.lat), args.mass, args.csw, device)
    fine_prepare = time.perf_counter() - started
    started = time.perf_counter()
    assets = build_schur_levels(
        operator, operator.matvec_parity, 2, [12, args.coarse_dof],
        [2, 2, 2, 2], list(args.lat), args.coarse_dof, torch.complex64,
        device, nv_iters=args.nv_iters, use_cache=False, verbose=args.verbose,
        mass=args.mass, csw=args.csw)
    torch.cuda.synchronize(device)
    asset_build = time.perf_counter() - started
    lonv, hnn, hdg, sit = (item[0] for item in assets)
    output = Path(args.cache_output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as handle:
        for name, tensor in (("lonv", lonv), ("hnn", hnn),
                             ("hdg", hdg), ("sit", sit)):
            handle.create_dataset(name, data=tensor.detach().cpu().numpy())
    payload = _common_payload(args)
    memory = _memory_snapshot(device)
    device_peak = device_sampler.stop()
    payload.update({
        "operation": "legacy_asset_build",
        "status": "PASS",
        "fine_prepare_seconds": fine_prepare,
        "mg_asset_build_seconds": asset_build,
        "mg_init_seconds": asset_build,
        "mg_init_scope": "null vectors + 33-tensor Galerkin stencil build",
        "coarse_dof": args.coarse_dof,
        "mg_grid": [2, 2, 2, 2],
        "nv_iters": args.nv_iters,
        "cache": {"path": str(output), "sha256": _sha256(output),
                  "size_bytes": output.stat().st_size},
        "asset_shapes": {
            "lonv": list(lonv.shape), "hnn": list(hnn.shape),
            "hdg": list(hdg.shape), "sit": list(sit.shape)},
        "memory": memory,
        "resources": _resource_payload(
            memory, cache_bytes=output.stat().st_size,
            device_peak=device_peak),
    })
    return payload


def _legacy_run_once(qcu: Any, define: Any, set_ptrs: Any, params: Any,
                     argv: Any, output: Any, fields: tuple[Any, ...],
                     device: Any) -> tuple[float, dict[str, Any]]:
    import torch

    output.zero_()
    params[define._MG_USE_INIT_GUESS_] = 0
    source = fields[1].clone()
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    qcu.applyCloverMultigridQcu(
        output, source, fields[0], fields[3], fields[4], fields[5],
        fields[6], set_ptrs, params)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return elapsed, _memory_snapshot(device)


def _last_mg_iterations(log_path: Path) -> int:
    if not log_path.is_file():
        return 0
    matches = re.findall(r"Total iterations:\s*(\d+)",
                         log_path.read_text(errors="ignore"))
    return int(matches[-1]) if matches else 0


def _last_mg_true_residual(log_path: Path) -> float | None:
    if not log_path.is_file():
        return None
    matches = re.findall(
        r"FINAL TRUE residual \(full-op\).*?relative\s*=\s*([0-9.eE+-]+)",
        log_path.read_text(errors="ignore"), re.S)
    return float(matches[-1]) if matches else None


def _last_prof_sections(log_path: Path) -> dict[str, float | int]:
    """Return the final ``PROF_SECTIONS`` counters from a Legacy MG run."""
    if not log_path.is_file():
        return {}
    matches = re.findall(r"PROF_SECTIONS:\s*([^\n]+)",
                         log_path.read_text(errors="ignore"))
    if not matches:
        return {}
    values: dict[str, float | int] = {}
    for token in matches[-1].split():
        if "=" not in token:
            continue
        key, raw = token.split("=", 1)
        if key.startswith("n_") or key.endswith("_iters") or key.endswith("_checks"):
            try:
                values[key] = int(raw.removesuffix("ms"))
            except ValueError:
                pass
        else:
            try:
                values[key] = float(raw.removesuffix("ms"))
            except ValueError:
                pass
    return values


def _run_legacy(args: argparse.Namespace) -> dict[str, Any]:
    import h5py
    import torch
    from pyqcu import tools
    from pyqcu.cuda import define, qcu
    from pyqcu.testing.qcu.strict.quda_comparison.common import make_clover_tensors

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    device_sampler = _DeviceMemorySampler(device).start()
    log_dir = Path(args.log_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    os.environ["QCU_LOG_DIR"] = str(log_dir)
    started = time.perf_counter()
    gauge_full, _, gauge, rhs = _load_case(
        Path(args.gauge), Path(args.source), tuple(args.lat), device)
    _operator, clover, kappa, kappa_t, _u0 = _fine_operator(
        gauge_full, tuple(args.lat), args.mass, args.csw, device)
    fine_prepare = time.perf_counter() - started
    cache_path = Path(args.cache).expanduser().resolve()
    started = time.perf_counter()
    with h5py.File(cache_path, "r") as handle:
        assets = [
            torch.from_numpy(handle[name][...]).to(
                device=device, dtype=torch.complex64).contiguous()
            for name in ("lonv", "hnn", "hdg", "sit")
        ]
    torch.cuda.synchronize(device)
    cache_load = time.perf_counter() - started
    started = time.perf_counter()
    define.params[define._VERBOSE_] = 0
    ce, cei, coo, coi, set_ptrs, params, argv = make_clover_tensors(
        gauge, tuple(args.lat), args.mass, dtype=torch.complex64,
        data_type=define._LAT_C64_, csw=args.csw)
    torch.cuda.synchronize(device)
    clover_tensor_prepare = time.perf_counter() - started
    argv[define._ATOL_] = float(args.tol)
    params[define._SET_PLAN_] = define._SET_PLAN1_
    params[define._PARITY_] = 0
    params[define._MAX_ITER_] = int(args.max_iter)
    params[define._VERBOSE_] = 1
    params[define._MG_USE_DEFLATE_] = int(args.deflate)
    params[define._MG_NUM_LEVEL_] = 2
    params[define._MG_LEVEL1_E_] = int(args.coarse_dof)
    params[define._MG_LEVEL1_X_] = int(args.lat[0] // 2)
    params[define._MG_LEVEL1_Y_] = int(args.lat[1] // 2)
    params[define._MG_LEVEL1_Z_] = int(args.lat[2] // 2)
    params[define._MG_LEVEL1_T_] = int(args.lat[3] // 4)
    params[define._MG_LEVEL1_MAX_ITER_] = int(args.coarse_max_iter)
    params[define._MG_LEVEL1_DATA_TYPE_] = define._LAT_C64_
    params[define._MG_LEVEL1_NUM_RESTART_] = int(args.coarse_restart)
    argv[define._MG_LEVEL1_ATOL_] = float(args.tol * args.coarse_tol_factor)
    set_ptrs[30:34] = 0
    for index, tensor in enumerate(assets):
        set_ptrs[30 + index] = tensor.data_ptr()
    for tensor in assets:
        tensor.record_stream(torch.cuda.current_stream(device))
    # make_clover_tensors leaves SET_INDEX at the next reusable slot.
    torch.cuda.synchronize(device)
    init_started = time.perf_counter()
    qcu.applyInitQcu(set_ptrs, params, argv)
    torch.cuda.synchronize(device)
    runtime_init = time.perf_counter() - init_started
    output = torch.empty_like(rhs)
    warmups = []
    steady = []
    iterations = []
    runtime_reinit_seconds = []
    log_path = log_dir / "clover_multigrid.log"
    final_result = None
    native_residual = None
    python_residual = None
    try:
        for index in range(args.warmups + args.repeats):
            if index > 0:
                reinit_started = time.perf_counter()
                qcu.applyEndQcu(set_ptrs, params)
                qcu.applyInitQcu(set_ptrs, params, argv)
                torch.cuda.synchronize(device)
                runtime_reinit_seconds.append(
                    time.perf_counter() - reinit_started)
            elapsed, _mem = _legacy_run_once(
                qcu, define, set_ptrs, params, argv, output,
                (gauge, rhs, output, ce, coo, cei, coi), device)
            (warmups if index < args.warmups else steady).append(elapsed)
            iterations.append(_last_mg_iterations(log_path))
        native_residual = _last_mg_true_residual(log_path)
        python_residual = float(_true_residual(
            output, rhs, gauge, gauge_full, clover, args.mass, _operator))
        final_result = (native_residual if native_residual is not None
                        else python_residual)
    finally:
        qcu.applyEndQcu(set_ptrs, params)
    payload = _common_payload(args)
    memory = _memory_snapshot(device)
    device_peak = device_sampler.stop()
    prof = _last_prof_sections(log_path)
    if args.solution_output:
        solution_path = Path(args.solution_output).expanduser().resolve()
        solution_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(solution_path, "w") as handle:
            handle.create_dataset(
                "data",
                data=tools.poooxyzt2oooxyzt(output).detach().cpu().numpy())
    payload.update({
        "operation": "legacy_solve",
        "status": "PASS" if final_result is not None and final_result <= args.tol else "FAIL",
        "fine_prepare_seconds": fine_prepare,
        "cache_load_seconds": cache_load,
        "clover_tensor_prepare_seconds": clover_tensor_prepare,
        "mg_init_seconds": runtime_init,
        "mg_init_scope": "applyInitQcu LatticeSet/MG runtime allocation; excludes asset build",
        "runtime_reinit_seconds": runtime_reinit_seconds,
        "steady_scope": "solve only; LatticeSet runtime is rebuilt between samples outside the timer to prevent cross-solve state reuse",
        "warmup_seconds": warmups,
        "steady_seconds": steady,
        "steady_median_seconds": _json_median(steady),
        "steady_mad_seconds": _json_mad(steady),
        "iterations": iterations,
        "final_true_residual": final_result,
        "native_true_residual": native_residual,
        "independent_true_residual": python_residual,
        "residual_source": (
            "C++ full-operator true residual from Legacy MG log"
            if native_residual is not None else
            "Python full-operator residual fallback"),
        "coarse_dof": args.coarse_dof,
        "coarse_max_iter": args.coarse_max_iter,
        "coarse_restart": args.coarse_restart,
        "deflate": bool(args.deflate),
        "coarse_execution": {
            "attempted_vcycles": int(prof.get("n_vcycles", 0)),
            "coarse_iterations": int(prof.get("coarse_iters", 0)),
            "vcycle_seconds": float(prof.get("vcycle", 0.0)) / 1000.0,
            "effective_mode": (
                "periodic_coarse_correction"
                if int(prof.get("n_vcycles", 0)) > 0 else "fine_only"),
            "source": str(log_path),
        },
        "tol": args.tol,
        "max_iter": args.max_iter,
        "memory": memory,
        "solution_output": (
            str(Path(args.solution_output).expanduser().resolve())
            if args.solution_output else None),
        "resources": _resource_payload(
            memory, cache_bytes=cache_path.stat().st_size,
            device_peak=device_peak),
    })
    return payload


def _run_strict(args: argparse.Namespace) -> dict[str, Any]:
    import h5py
    import torch
    from pyqcu import tools
    from pyqcu.torch import dslash
    from pyqcu.cuda import define
    from pyqcu.cuda import StrictMultigrid as CudaStrictMultigrid
    from pyqcu.torch.solver import StrictMultigrid

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    device_sampler = _DeviceMemorySampler(device).start()
    started = time.perf_counter()
    gauge_full, _, gauge, rhs = _load_case(
        Path(args.gauge), Path(args.source), tuple(args.lat), device)
    _operator, clover, kappa, kappa_t, u0 = _fine_operator(
        gauge_full, tuple(args.lat), args.mass, args.csw, device)
    fine_prepare = time.perf_counter() - started
    null_path = Path(args.nullvec).expanduser().resolve()
    null_load_started = time.perf_counter()
    with h5py.File(null_path, "r") as handle:
        null = torch.from_numpy(handle[args.null_dataset][...]).to(
            device=device, dtype=torch.complex64).contiguous()
    torch.cuda.synchronize(device)
    null_vector_load_seconds = time.perf_counter() - null_load_started
    available_nvec = int(null.shape[0])
    nvec = available_nvec if args.nvec is None else int(args.nvec)
    if nvec <= 0 or nvec > available_nvec:
        raise ValueError(
            f"selected nvec={nvec} is outside available range 1..{available_nvec}")
    null = null[:nvec].contiguous()
    hierarchy = StrictMultigrid(
        U=gauge_full, clover_term=clover, kappa=kappa_t, u_0=u0,
        lat_size=tuple(args.lat), null_vectors=[null], nvec_list=[nvec],
        dof_list=[12, 2 * nvec], block_size=(2, 2, 2, 2), max_level=2,
        materialize_coarse=True, use_parity=True, target_parity=1,
        nu_pre=1, nu_post=1, coarse_max_iter=args.coarse_max_iter,
        coarse_tol=args.tol * args.coarse_tol_factor,
        restart=args.restart, max_iter=args.max_iter, tol=args.tol,
        setup_method="random", setup_iters=0, strict_galerkin_mode="colored",
        verbose=False)
    hierarchy_started = time.perf_counter()
    hierarchy.setup()
    torch.cuda.synchronize(device)
    hierarchy_setup = time.perf_counter() - hierarchy_started
    from pyqcu.testing.qcu.strict.quda_comparison.common import make_clover_tensors
    clover_started = time.perf_counter()
    define.params[define._VERBOSE_] = 0
    ce, cei, coo, coi, set_ptrs, params, argv = make_clover_tensors(
        gauge, tuple(args.lat), args.mass, dtype=torch.complex64,
        data_type=define._LAT_C64_, csw=args.csw)
    torch.cuda.synchronize(device)
    clover_tensor_prepare = time.perf_counter() - clover_started
    argv[define._ATOL_] = float(args.tol)
    params[define._MAX_ITER_] = int(args.max_iter)
    solver_started = time.perf_counter()
    solver = CudaStrictMultigrid(
        hierarchy, argv, gauge, ce, coo, cei, coi, params,
        restart=args.restart, max_krylov_bytes=args.max_krylov_bytes,
        release_setup_assets=True, verbose=False)
    torch.cuda.synchronize(device)
    runtime_init = time.perf_counter() - solver_started
    warmups = []
    steady = []
    iterations = []
    residuals = []
    final_result = None
    output = None
    try:
        for index in range(args.warmups + args.repeats):
            rhs_work = rhs.clone()
            started = time.perf_counter()
            output = solver.solve(rhs_work)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            (warmups if index < args.warmups else steady).append(elapsed)
            iterations.append(int(solver.iterations))
            residuals.append(float(solver.final_residual))
        final_result = float(_true_residual(
            output, rhs, gauge, gauge_full, clover, args.mass, _operator))
        runtime_memory = solver.memory_report()
        # Capture allocator high-water marks while the benchmark-owned objects
        # are still alive and before close() releases the Strict runtime.
        allocator_memory = _memory_snapshot(device)
        device_peak = device_sampler.stop()
    finally:
        solver.close()
    payload = _common_payload(args)
    memory = {**runtime_memory, "torch_allocator": allocator_memory}
    if args.solution_output:
        solution_path = Path(args.solution_output).expanduser().resolve()
        solution_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(solution_path, "w") as handle:
            handle.create_dataset(
                "data",
                data=tools.poooxyzt2oooxyzt(output).detach().cpu().numpy())
    payload.update({
        "operation": "strict_solve",
        "status": "PASS" if final_result is not None and final_result <= args.tol else "FAIL",
        "fine_prepare_seconds": fine_prepare,
        "null_vector_load_seconds": null_vector_load_seconds,
        "clover_tensor_prepare_seconds": clover_tensor_prepare,
        "hierarchy_setup_seconds": hierarchy_setup,
        "runtime_init_seconds": runtime_init,
        "mg_init_seconds": hierarchy_setup + runtime_init,
        "mg_init_scope": "strict Galerkin hierarchy/asset setup plus runtime binding; excludes null-vector file creation",
        "null_vector_path": str(null_path),
        "null_vector_sha256": _sha256(null_path),
        "nvec": nvec,
        "null_vector_source_nvec": available_nvec,
        "null_vector_indices": list(range(nvec)),
        "coarse_dof": 2 * nvec,
        "warmup_seconds": warmups,
        "steady_seconds": steady,
        "steady_median_seconds": _json_median(steady),
        "steady_mad_seconds": _json_mad(steady),
        "iterations": iterations,
        "backend_final_residuals": residuals,
        "final_true_residual": final_result,
        "residual_source": "Python full Wilson/Clover reconstruction",
        "tol": args.tol,
        "max_iter": args.max_iter,
        "restart": args.restart,
        "coarse_max_iter": args.coarse_max_iter,
        "memory": memory,
        "solution_output": (
            str(Path(args.solution_output).expanduser().resolve())
            if args.solution_output else None),
        "resources": _resource_payload(
            allocator_memory, null_vector_bytes=null_path.stat().st_size,
            strict_runtime_memory=runtime_memory, device_peak=device_peak),
    })
    payload["status"] = "PASS" if final_result is not None and final_result <= args.tol else "FAIL"
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--gauge", required=True)
    common.add_argument("--source", required=True)
    common.add_argument("--lat", nargs=4, type=int, required=True)
    common.add_argument("--mass", type=float, required=True)
    common.add_argument("--csw", type=float, required=True)
    common.add_argument("--device", default="cuda:0")
    common.add_argument("--tol", type=float, default=1e-6)
    common.add_argument("--max-iter", type=int, default=200)
    common.add_argument("--warmups", type=int, default=2)
    common.add_argument("--repeats", type=int, default=5)
    common.add_argument("--output", required=True)
    common.add_argument("--solution-output")

    build = sub.add_parser("legacy-build", parents=[common])
    build.add_argument("--cache-output", required=True)
    build.add_argument("--coarse-dof", type=int, default=12)
    build.add_argument("--nv-iters", type=int, default=1)
    build.add_argument("--verbose", action="store_true")

    legacy = sub.add_parser("legacy", parents=[common])
    legacy.add_argument("--cache", required=True)
    legacy.add_argument("--log-dir", required=True)
    legacy.add_argument("--coarse-dof", type=int, default=12)
    legacy.add_argument("--coarse-max-iter", type=int, default=200)
    legacy.add_argument("--coarse-restart", type=int, default=5)
    legacy.add_argument("--coarse-tol-factor", type=float, default=1e4)
    legacy.add_argument("--deflate", action="store_true",
                        help="attempt the initial Legacy coarse correction")

    strict = sub.add_parser("strict", parents=[common])
    strict.add_argument("--nullvec", required=True)
    strict.add_argument("--null-dataset", default="null")
    strict.add_argument("--nvec", type=int, default=None,
                        help="use the first N vectors from the canonical input")
    strict.add_argument("--restart", type=int, default=20)
    strict.add_argument("--coarse-max-iter", type=int, default=200)
    strict.add_argument("--coarse-tol-factor", type=float, default=1e2)
    strict.add_argument("--max-krylov-bytes", type=int, default=512 << 20)

    args = parser.parse_args(argv)
    args.side = "legacy" if args.command.startswith("legacy") else "strict"
    if args.command == "legacy-build":
        payload = _build_legacy(args)
    elif args.command == "legacy":
        payload = _run_legacy(args)
    else:
        payload = _run_strict(args)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
