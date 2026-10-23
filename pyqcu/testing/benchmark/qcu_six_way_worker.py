#!/usr/bin/env python3
"""Run one isolated solver side for the QCU six-way benchmark."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import io
import json
import os
import re
import statistics
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[3]
SIMPLE = REPO / "pyqcu" / "testing" / "simple"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(SIMPLE) not in sys.path:
    sys.path.insert(0, str(SIMPLE))


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


def _peak_rss_bytes() -> int:
    import resource

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _current_rss_bytes() -> int | None:
    status = Path("/proc/self/status")
    if not status.is_file():
        return None
    match = re.search(r"^VmRSS:\s+(\d+)\s+kB$", status.read_text(), re.M)
    return int(match.group(1)) * 1024 if match else None


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=float) + "\n",
        encoding="utf-8",
    )


def _load_h5(path: Path, dataset: str = "data"):
    import h5py

    with h5py.File(path, "r") as handle:
        key = dataset if dataset in handle else next(iter(handle))
        return handle[key][...]


def _save_h5(path: Path, value: Any, **metadata: Any) -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=value)
        for key, item in metadata.items():
            handle.attrs[key] = item


def _common_payload(args: argparse.Namespace, *, backend: str,
                    strategy: str) -> dict[str, Any]:
    gauge = Path(args.gauge).expanduser().resolve()
    source = Path(args.source).expanduser().resolve()
    nullvec = (Path(args.nullvec).expanduser().resolve()
               if args.nullvec else None)
    return {
        "schema": "pyqcu.qcu-six-way.side.v1",
        "variant": args.variant,
        "backend": backend,
        "strategy": strategy,
        "status": "RUNNING",
        "lattice": list(args.lat),
        "mass": float(args.mass),
        "csw": float(args.csw),
        "precision": "c64",
        "configuration": {
            "tolerance": float(args.tolerance),
            "max_iter": int(args.max_iter),
            "levels": int(args.levels),
            "block": list(args.block),
            "nvec": int(args.nvec),
            "coarse_dof": 2 * int(args.nvec) if strategy == "strict" else int(args.nvec),
            "coarse_spin": 2 if strategy == "strict" else None,
            "coarse_tolerance": float(args.coarse_tolerance),
            "coarse_max_iter": int(args.coarse_max_iter),
            "restart": int(args.restart),
            "legacy_correction_interval": int(args.correction_interval),
            "nu_pre": int(args.nu),
            "nu_post": int(args.nu),
            "warmups": int(args.warmups),
            "repeats": int(args.repeats),
        },
        "inputs": {
            "gauge": {"path": str(gauge), "sha256": _sha256(gauge)},
            "source": {"path": str(source), "sha256": _sha256(source)},
            "null_vectors": (
                {"path": str(nullvec), "sha256": _sha256(nullvec)}
                if nullvec is not None else None
            ),
        },
        "timing": {},
        "iterations": {},
        "residual": {},
        "coarse_execution": {},
        "memory": {},
        "provenance": {
            "python": sys.version,
            "pid": os.getpid(),
            "hostname": os.uname().nodename,
            "argv": list(sys.argv),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }


class _CudaDeviceSampler:
    """Sample device-wide usage so native allocations are included."""

    def __init__(self, torch: Any, device: Any, interval: float = 0.02):
        self.torch = torch
        self.device = device
        self.interval = float(interval)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._baseline = self._used()
        self._peak = self._baseline
        self._samples = 0
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run, name="qcu-six-way-memory", daemon=True)

    def _used(self) -> int:
        free_bytes, total_bytes = self.torch.cuda.mem_get_info(self.device)
        return int(total_bytes) - int(free_bytes)

    def _sample(self) -> None:
        used = self._used()
        with self._lock:
            self._baseline = min(self._baseline, used)
            self._peak = max(self._peak, used)
            self._samples += 1

    def _run(self) -> None:
        try:
            while not self._stop.wait(self.interval):
                self._sample()
        except BaseException as error:  # pragma: no cover - diagnostic thread
            self._error = error

    def start(self) -> "_CudaDeviceSampler":
        self._sample()
        self._thread.start()
        return self

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join()
        self._sample()
        if self._error is not None:
            raise RuntimeError("CUDA device memory sampler failed") from self._error
        with self._lock:
            return {
                "scope": "device-wide cudaMemGetInfo; includes native allocations and other processes",
                "sample_interval_seconds": self.interval,
                "sample_count": self._samples,
                "baseline_used_bytes": self._baseline,
                "peak_used_bytes": self._peak,
                "peak_delta_over_baseline_bytes": max(
                    0, self._peak - self._baseline),
            }


class _CpuDeviceSampler:
    """No-op device sampler with the same lifecycle as the CUDA sampler."""

    def start(self) -> "_CpuDeviceSampler":
        return self

    def stop(self) -> dict[str, Any]:
        return {
            "device_memory_applicable": False,
            "scope": "CPU run; host RSS is recorded separately",
            "sample_count": 0,
        }


def _torch_memory_snapshot(torch: Any, device: Any) -> dict[str, Any]:
    if device.type != "cuda":
        return {
            "device_memory_applicable": False,
            "device": str(device),
            "host_rss_bytes": _current_rss_bytes(),
        }
    torch.cuda.synchronize(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
    }


def _true_residual_torch(torch: Any, dslash: Any, output: Any, source: Any,
                         gauge: Any, mass: float, csw: float) -> float:
    if tuple(output.shape[-4:]) != tuple(source.shape[-4:]):
        raise ValueError("solution/source lattice mismatch")
    device = source.device
    kappa = torch.tensor(
        [1.0 / (2.0 * float(mass) + 8.0)],
        dtype=torch.float32 if source.dtype == torch.complex64 else torch.float64,
        device=device,
    )
    u0 = torch.ones(1, dtype=kappa.dtype, device=device)
    clover = dslash.make_clover(gauge, kappa=kappa, u_0=u0, csw=csw)
    residual = dslash.give_wilson(
        output, gauge, kappa, u0, with_I=True) + dslash.give_clover(
            output, clover) - source
    return float(
        torch.linalg.vector_norm(residual)
        / torch.linalg.vector_norm(source)
    )


def _capture_call(function: Callable[[], Any], torch: Any, device: Any,
                  ) -> tuple[float, Any, str]:
    stream = io.StringIO()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with contextlib.redirect_stdout(stream):
        result = function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    output = stream.getvalue()
    if output:
        print(output, end="", flush=True)
    return elapsed, result, output


def _last_total_iterations(text: str) -> int | None:
    matches = re.findall(r"Total iterations:\s*(\d+)", text)
    return int(matches[-1]) if matches else None


def _worker_torch(args: argparse.Namespace) -> dict[str, Any]:
    import h5py  # noqa: F401
    import torch
    from pyqcu import tools
    from pyqcu.torch import dslash
    from pyqcu.torch.solver import LegacyMultigrid, StrictMultigrid

    strategy = "legacy" if args.variant.endswith("legacy") else "strict"
    payload = _common_payload(args, backend="torch", strategy=strategy)
    device = torch.device(args.device)
    is_cuda = device.type == "cuda"
    if is_cuda:
        torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    if is_cuda:
        torch.cuda.manual_seed_all(args.seed)
    overall_sampler = (
        _CudaDeviceSampler(torch, device) if is_cuda else _CpuDeviceSampler()
    ).start()
    mg = None

    load_started = time.perf_counter()
    gauge = torch.from_numpy(_load_h5(Path(args.gauge))).to(
        device=device, dtype=torch.complex64).contiguous()
    source = torch.from_numpy(_load_h5(Path(args.source))).to(
        device=device, dtype=torch.complex64).contiguous()
    if strategy == "strict":
        null = torch.from_numpy(
            _load_h5(Path(args.nullvec), args.null_dataset)).to(
                device=device, dtype=torch.complex64).contiguous()[:args.nvec]
    else:
        null = None
    if is_cuda:
        torch.cuda.synchronize(device)
    payload["timing"]["input_load_seconds"] = time.perf_counter() - load_started
    payload["timing"]["cache_load_seconds"] = payload["timing"]["input_load_seconds"] if strategy == "strict" else 0.0

    kappa = torch.tensor(
        [1.0 / (2.0 * args.mass + 8.0)],
        dtype=torch.float32, device=device)
    u0 = torch.ones(1, dtype=torch.float32, device=device)
    prep_started = time.perf_counter()
    clover = dslash.make_clover(gauge, kappa=kappa, u_0=u0, csw=args.csw)
    if is_cuda:
        torch.cuda.synchronize(device)
    payload["timing"]["operator_prepare_seconds"] = time.perf_counter() - prep_started

    setup_started = time.perf_counter()
    if strategy == "legacy":
        mg = LegacyMultigrid(
            dtype_list=[torch.complex64, torch.complex64],
            device_list=[device, device],
            U=gauge,
            clover_term=clover,
            kappa=kappa,
            u_0=u0,
            min_size=4,
            max_level=args.levels,
            mg_grid_size=list(args.block),
            dof_list=[args.nvec] * args.levels,
            tol=args.tolerance,
            max_iter=args.max_iter,
            num_restart=args.correction_interval,
            support_parity=True,
            verbose=False,
            seed=args.seed,
            coarse_assets=[args.legacy_cache] if args.legacy_cache else None,
            coarse_tol=args.coarse_tolerance,
            coarse_max_iter=args.coarse_max_iter,
            deflate=bool(args.legacy_deflate),
            adaptive_correction=bool(args.legacy_adaptive_correction),
        )
    else:
        if int(args.levels) != 2:
            raise ValueError("this worker currently fixes strict levels=2")
        mg = StrictMultigrid(
            U=gauge,
            clover_term=clover,
            kappa=kappa,
            u_0=u0,
            lat_size=tuple(args.lat),
            null_vectors=[null],
            nvec_list=[args.nvec],
            dof_list=[12, 2 * args.nvec],
            block_size=tuple(args.block),
            max_level=args.levels,
            materialize_coarse=True,
            use_parity=True,
            target_parity=1,
            nu_pre=args.nu,
            nu_post=args.nu,
            coarse_max_iter=args.coarse_max_iter,
            coarse_tol=args.coarse_tolerance,
            restart=args.restart,
            max_iter=args.max_iter,
            tol=args.tolerance,
            setup_method="random",
            setup_iters=0,
            strict_galerkin_mode="colored",
            verbose=False,
        )
    mg.setup()
    if is_cuda:
        torch.cuda.synchronize(device)
    payload["timing"]["setup_seconds"] = time.perf_counter() - setup_started

    cycle_levels: list[int] = []
    observed_levels: set[int] = set()
    if strategy == "legacy":
        original_cycle = mg.cycle

        def cycle_wrapper(level: int = 0):
            cycle_levels.append(int(level))
            observed_levels.add(int(level))
            return original_cycle(level=level)

        mg.cycle = cycle_wrapper
    else:
        original_vcycle = mg._strict_v_cycle

        def vcycle_wrapper(rhs: Any, level: int = 0):
            cycle_levels.append(int(level))
            observed_levels.add(int(level))
            return original_vcycle(rhs, level=level)

        mg._strict_v_cycle = vcycle_wrapper

    if is_cuda:
        torch.cuda.synchronize(device)
    first_before = _torch_memory_snapshot(torch, device)
    phase_sampler = (
        _CudaDeviceSampler(torch, device) if is_cuda else _CpuDeviceSampler()
    ).start()
    first_elapsed, output, first_text = _capture_call(
        lambda: mg.solve(source), torch, device)
    first_device = phase_sampler.stop()
    first_after = _torch_memory_snapshot(torch, device)
    warmups = [first_elapsed]
    iteration_samples = [mg.last_solve_report.get("iterations")]
    stop_reasons = [mg.last_solve_report.get("stop_reason")]
    native_residuals = [mg.last_solve_report.get(
        "native_relative_residual",
        mg.last_solve_report.get("final_residual"))]
    solve_reports = [dict(mg.last_solve_report)]
    cycle_counts = [len(cycle_levels)]
    cycle_levels.clear()

    for _ in range(max(0, args.warmups - 1)):
        elapsed, output, text = _capture_call(
            lambda: mg.solve(source), torch, device)
        warmups.append(elapsed)
        iteration_samples.append(mg.last_solve_report.get("iterations"))
        stop_reasons.append(mg.last_solve_report.get("stop_reason"))
        native_residuals.append(mg.last_solve_report.get(
            "native_relative_residual",
            mg.last_solve_report.get("final_residual")))
        solve_reports.append(dict(mg.last_solve_report))
        cycle_counts.append(len(cycle_levels))
        cycle_levels.clear()

    steady: list[float] = []
    steady_sampler = (
        _CudaDeviceSampler(torch, device) if is_cuda else _CpuDeviceSampler()
    ).start()
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(args.repeats):
        elapsed, output, text = _capture_call(
            lambda: mg.solve(source), torch, device)
        steady.append(elapsed)
        iteration_samples.append(mg.last_solve_report.get("iterations"))
        stop_reasons.append(mg.last_solve_report.get("stop_reason"))
        native_residuals.append(mg.last_solve_report.get(
            "native_relative_residual",
            mg.last_solve_report.get("final_residual")))
        solve_reports.append(dict(mg.last_solve_report))
        cycle_counts.append(len(cycle_levels))
        cycle_levels.clear()
    steady_device = steady_sampler.stop()
    steady_allocator = _torch_memory_snapshot(torch, device)

    independent = _true_residual_torch(
        torch, dslash, output, source, gauge, args.mass, args.csw)
    overall_device = overall_sampler.stop()
    payload["timing"].update({
        "warmup_seconds": warmups,
        "first_solve_seconds": (warmups or steady)[0],
        "steady_seconds": steady,
        "steady_median_seconds": _median(steady),
        "steady_mad_seconds": _mad(steady),
    })
    payload["iterations"] = {
        "samples": iteration_samples,
        "median": _median([float(value) for value in iteration_samples if value is not None]),
        "semantics": (
            "Legacy outer fine BiCGStab iterations from last_solve_report"
            if strategy == "legacy" else
            "strict outer right-preconditioned FGMRES Arnoldi iterations"
        ),
        "stop_reasons": stop_reasons,
    }
    payload["residual"] = {
        "native_relative": native_residuals[-1],
        "native_samples": native_residuals,
        "independent_relative": independent,
        "gate": args.tolerance,
        "pass": bool(independent <= args.tolerance),
        "source": "independent PyTorch full Wilson/Clover reconstruction",
        "scope": "full",
    }
    payload["coarse_execution"] = {
        "kind": (
            "legacy_periodic_correction"
            if strategy == "legacy" else "strict_recursive_vcycle"),
        "executed": bool(any(value > 0 for value in cycle_counts)),
        "cycle_calls_per_solve": cycle_counts,
        "cycle_levels_observed": sorted(observed_levels),
        "last_solve_report": solve_reports[-1],
        "source": "benchmark-only wrapper around the public Python cycle method",
    }
    payload["memory"] = {
        "device_memory_applicable": is_cuda,
        "device_wide": overall_device,
        "first_solve_device_wide": first_device,
        "steady_device_wide": steady_device,
        "first_solve_allocator_before": first_before,
        "first_solve_allocator_after": first_after,
        "first_solve_allocator_delta_bytes": max(
            0,
            int(first_after.get("allocated_bytes", 0)) -
            int(first_before.get("allocated_bytes", 0))),
        "steady_allocator": steady_allocator,
        "host_peak_rss_bytes": _peak_rss_bytes(),
        "threads": torch.get_num_threads(),
    }
    payload["device"] = {
        "type": device.type,
        "name": (
            torch.cuda.get_device_name(device) if is_cuda else
            f"CPU ({torch.get_num_threads()} torch threads)"),
        "index": device.index,
    }
    if args.legacy_cache:
        cache_path = Path(args.legacy_cache).expanduser().resolve()
        payload["inputs"]["legacy_cache"] = {
            "path": str(cache_path),
            "sha256": _sha256(cache_path),
            "datasets": list(getattr(mg, "asset_identities", [])),
        }
    payload["status"] = "PASS" if payload["residual"]["pass"] else "FAIL"
    if args.solution_output:
        _save_h5(
            Path(args.solution_output), output.detach().cpu().numpy(),
            backend="torch", strategy=strategy, lattice=json.dumps(args.lat))
    mg.close()
    return payload


def _worker_torch_bicgstab(args: argparse.Namespace) -> dict[str, Any]:
    """Run the Torch parity-Schur BiCGStab control on the shared input."""
    import torch
    from pyqcu import tools
    from pyqcu.torch import dslash
    from pyqcu.torch.solver import bistabcg

    payload = _common_payload(args, backend="torch", strategy="bicgstab")
    device = torch.device(args.device)
    is_cuda = device.type == "cuda"
    if is_cuda:
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)

    load_started = time.perf_counter()
    gauge = torch.from_numpy(_load_h5(Path(args.gauge))).to(
        device=device, dtype=torch.complex64).contiguous()
    source = torch.from_numpy(_load_h5(Path(args.source))).to(
        device=device, dtype=torch.complex64).contiguous()
    if is_cuda:
        torch.cuda.synchronize(device)
    payload["timing"]["input_load_seconds"] = time.perf_counter() - load_started

    kappa = torch.tensor(
        [1.0 / (2.0 * args.mass + 8.0)], dtype=torch.float32, device=device)
    u0 = torch.ones(1, dtype=torch.float32, device=device)
    prepare_started = time.perf_counter()
    clover = dslash.make_clover(
        gauge, kappa=kappa, u_0=u0, csw=args.csw)
    operator = dslash.operator(
        U=gauge, clover_term=clover, kappa=kappa, u_0=u0,
        support_parity=True, verbose=False)
    source_eo = tools.oooxyzt2poooxyzt(
        source.reshape(12, *args.lat))
    rhs = operator.give_b_parity(
        b_e=source_eo[0], b_o=source_eo[1]).contiguous()
    if is_cuda:
        torch.cuda.synchronize(device)
    payload["timing"]["operator_prepare_seconds"] = (
        time.perf_counter() - prepare_started)

    elapsed_samples: list[float] = []
    iteration_samples: list[int] = []
    residual_samples: list[float] = []
    stop_reasons: list[str] = []
    rhs_norm = float(tools.norm(rhs))
    solution = None
    sampler = (
        _CudaDeviceSampler(torch, device) if is_cuda else _CpuDeviceSampler()
    ).start()
    try:
        for index in range(args.warmups + args.repeats):
            history: list[float] = []
            started = time.perf_counter()
            solution_odd = bistabcg(
                rhs, operator.matvec_parity, tol=args.tolerance,
                max_iter=args.max_iter, x0=torch.zeros_like(rhs),
                if_rtol=True, verbose=False, history=history)
            if is_cuda:
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            elapsed_samples.append(elapsed)
            iteration_samples.append(max(0, len(history) - 1))
            residual_samples.append(float(history[-1]))
            stop_reasons.append(
                "relative_schur_tolerance"
                if residual_samples[-1] <= args.tolerance * rhs_norm else
                "iteration_limit")
            solution = solution_odd
    finally:
        device_memory = sampler.stop()

    if solution is None:
        raise RuntimeError("Torch BiCGStab produced no solution")
    solution_e = operator.give_x_e(b_e=source_eo[0], x_o=solution)
    solution_full = tools.poooxyzt2oooxyzt(
        torch.stack([solution_e, solution], dim=0)).reshape(
            4, 3, *args.lat)
    independent = _true_residual_torch(
        torch, dslash, solution_full, source, gauge, args.mass, args.csw)
    warmups = elapsed_samples[:args.warmups]
    steady = elapsed_samples[args.warmups:]
    payload["timing"].update({
        "warmup_seconds": warmups,
        "first_solve_seconds": (warmups or steady)[0],
        "steady_seconds": steady,
        "steady_median_seconds": _median(steady),
        "steady_mad_seconds": _mad(steady),
    })
    payload["iterations"] = {
        "samples": iteration_samples,
        "median": _median([float(value) for value in iteration_samples]),
        "semantics": "Torch odd-Schur BiCGStab iterations",
        "stop_reasons": stop_reasons,
    }
    payload["residual"] = {
        "native_relative": residual_samples[-1] / max(
            rhs_norm, 1.0e-30),
        "independent_relative": independent,
        "gate": args.tolerance,
        "pass": bool(independent <= args.tolerance),
        "source": "independent PyTorch full Wilson/Clover reconstruction",
        "scope": "full",
    }
    payload["coarse_execution"] = {
        "kind": "not_applicable",
        "executed": False,
        "source": "BiCGStab boundary control, not included in MG speedup",
    }
    payload["memory"] = {
        "device_memory_applicable": is_cuda,
        "device_wide": device_memory,
        "host_peak_rss_bytes": _peak_rss_bytes(),
        "threads": torch.get_num_threads(),
    }
    payload["device"] = {
        "type": device.type,
        "name": (
            torch.cuda.get_device_name(device) if is_cuda else
            f"CPU ({torch.get_num_threads()} torch threads)"),
        "index": device.index,
    }
    payload["status"] = "PASS" if payload["residual"]["pass"] else "FAIL"
    if args.solution_output:
        _save_h5(
            Path(args.solution_output), solution_full.detach().cpu().numpy(),
            backend="torch", strategy="bicgstab",
            lattice=json.dumps(args.lat))
    return payload


def _configure_kupl_common(runtime: Any, args: argparse.Namespace,
                           coarse_dims: tuple[int, int, int, int],
                           coarse_dof: int) -> None:
    import torch
    from pyqcu.cuda import define

    runtime.params[define._DATA_TYPE_] = define._LAT_C64_
    runtime.params[define._PARITY_] = 1
    runtime.params[define._MAX_ITER_] = int(args.max_iter)
    runtime.params[define._MG_NUM_LEVEL_] = int(args.levels)
    runtime.params[define._MG_MU_PRE_] = int(args.nu)
    runtime.argv[define._ATOL_] = float(args.tolerance)
    if int(args.levels) == 2:
        base = define._MG_LEVEL1_E_
        runtime.params[base:base + define._MG_PARAMS_SIZE_] = torch.tensor(
                [coarse_dof, *coarse_dims, args.coarse_max_iter,
                 define._LAT_C64_, args.restart],
                dtype=torch.int32)
        runtime.argv[define._MG_LEVEL1_ATOL_] = float(args.coarse_tolerance)


def _kupl_expected_coarse_dims(
        lattice: tuple[int, ...], block: tuple[int, ...]
        ) -> tuple[int, int, int, int]:
    if len(lattice) != 4 or len(block) != 4:
        raise ValueError("KUPL lattice and block must both contain four extents")
    if any(int(extent) <= 0 for extent in lattice) or any(
            int(width) <= 0 for width in block):
        raise ValueError("KUPL lattice and block extents must be positive")
    if any(int(extent) % int(width) for extent, width in zip(
            lattice, block, strict=True)):
        raise ValueError(
            f"KUPL lattice {lattice} is not divisible by block {block}")
    return tuple(
        int(extent) // int(width)
        for extent, width in zip(lattice, block, strict=True)
    )  # type: ignore[return-value]


def _kupl_legacy_cache_layout(
        lonv: Any, hop_nn: Any, hop_diag: Any, sitting: Any, *,
        lattice: tuple[int, ...], block: tuple[int, ...], nvec: int
        ) -> tuple[str, tuple[int, int, int, int]]:
    """Validate a KUPL 33-stencil cache and return layout plus geometry."""
    full_coarse = _kupl_expected_coarse_dims(lattice, block)
    if int(lattice[3]) % (2 * int(block[3])):
        raise ValueError("odd-Schur T extent is not divisible by 2*block_t")
    odd_coarse = (*full_coarse[:3], int(lattice[3]) // (2 * int(block[3])))
    if lonv.ndim != 10:
        raise ValueError(
            "KUPL Legacy cache must use the 10-D blocked layout; "
            f"got {tuple(lonv.shape)}")
    if int(lonv.shape[0]) != int(nvec) or int(lonv.shape[1]) != 12:
        raise ValueError(
            f"KUPL Legacy lonv shape {tuple(lonv.shape)} does not match "
            f"nvec={nvec}, fine_dof=12")
    observed_coarse = tuple(int(lonv.shape[index]) for index in (2, 4, 6, 8))
    observed_block = tuple(int(lonv.shape[index]) for index in (3, 5, 7, 9))
    if observed_block != tuple(block):
        raise ValueError(
            f"KUPL Legacy cache block {observed_block} != {tuple(block)}")
    if observed_coarse == full_coarse:
        layout = "full-site-wilson-clover-33-stencil"
    elif observed_coarse == odd_coarse:
        layout = "odd-schur-33-stencil"
    else:
        raise ValueError(
            "KUPL Legacy cache geometry does not match one supported layout: "
            f"cache coarse={observed_coarse}; full-site={full_coarse}; "
            f"odd-Schur={odd_coarse}")
    expected_nn = (2, 4, int(nvec), 12, *observed_coarse)
    expected_diag = (2, 2, 6, int(nvec), 12, *observed_coarse)
    expected_sit = (int(nvec), 12, *observed_coarse)
    if tuple(hop_nn.shape) != expected_nn:
        raise ValueError(
            f"KUPL Legacy hnn shape {tuple(hop_nn.shape)} != {expected_nn}")
    if tuple(hop_diag.shape) != expected_diag:
        raise ValueError(
            f"KUPL Legacy hdg shape {tuple(hop_diag.shape)} != {expected_diag}")
    if tuple(sitting.shape) != expected_sit:
        raise ValueError(
            f"KUPL Legacy sit shape {tuple(sitting.shape)} != {expected_sit}")
    return layout, observed_coarse  # type: ignore[return-value]


def _kupl_strict_asset_layout(
        fine_null: Any, links: Any, onsite: Any, *,
        lattice: tuple[int, ...], block: tuple[int, ...], nvec: int
        ) -> tuple[int, int, int, int]:
    """Validate parity-compressed KUPL Strict assets and return coarse geometry."""
    coarse = _kupl_expected_coarse_dims(lattice, block)
    coarse_dof = 2 * int(nvec)
    expected_null = tuple(
        value for axis, extent in enumerate(lattice)
        for value in (coarse[axis], int(block[axis]))
    )
    expected_null = (coarse_dof, 12, *expected_null)
    expected_links = (2, 4, coarse_dof, coarse_dof, *coarse)
    expected_onsite = (2, coarse_dof, coarse_dof, *coarse)
    if tuple(fine_null.shape) != expected_null:
        raise ValueError(
            f"KUPL Strict fine_null shape {tuple(fine_null.shape)} != {expected_null}")
    if tuple(links.shape) != expected_links:
        raise ValueError(
            f"KUPL Strict links shape {tuple(links.shape)} != {expected_links}")
    if tuple(onsite.shape) != expected_onsite:
        raise ValueError(
            f"KUPL Strict onsite shape {tuple(onsite.shape)} != {expected_onsite}")
    return coarse


def _kupl_manifest(path: Path, *, lattice: tuple[int, ...],
                   block: tuple[int, ...], nvec: int,
                   coarse_dof: int) -> dict[str, Any] | None:
    manifest_path = path.with_suffix(".json")
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid KUPL cache manifest {manifest_path}: {error}") from error
    checks = {
        "lattice": [int(value) for value in lattice],
        "block": [int(value) for value in block],
        "nvec": int(nvec),
        "coarse_dof": int(coarse_dof),
    }
    for key, expected in checks.items():
        if key in manifest and manifest[key] != expected:
            raise ValueError(
                f"KUPL cache manifest {manifest_path} has {key}="
                f"{manifest[key]!r}, expected {expected!r}")
    if manifest.get("status", "PASS") != "PASS":
        raise ValueError(f"KUPL cache manifest {manifest_path} is not PASS")
    return manifest


def _worker_kupl_legacy(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from pyqcu import tools
    from pyqcu.cuda import QcuRuntime, define
    from pyqcu.kupl import qcu
    from pyqcu.kupl.multigrid import LegacyMultigrid

    payload = _common_payload(args, backend="kupl", strategy="legacy")
    cache = Path(args.legacy_cache).expanduser().resolve()
    payload["inputs"]["legacy_cache"] = {
        "path": str(cache), "sha256": _sha256(cache),
        "size_bytes": cache.stat().st_size,
    }
    host_rss_start = _current_rss_bytes()
    load_started = time.perf_counter()
    gauge_full = torch.from_numpy(_load_h5(Path(args.gauge))).to(
        dtype=torch.complex64).contiguous()
    source_full = torch.from_numpy(_load_h5(Path(args.source))).to(
        dtype=torch.complex64).contiguous()
    gauge = tools.oooxyzt2poooxyzt(gauge_full).contiguous()
    payload["timing"]["input_load_seconds"] = time.perf_counter() - load_started

    cache_started = time.perf_counter()
    with __import__("h5py").File(cache, "r") as handle:
        assets = [
            torch.from_numpy(handle[name][...]).to(
                dtype=torch.complex64).contiguous()
            for name in ("lonv", "hnn", "hdg", "sit")
        ]
    payload["timing"]["cache_load_seconds"] = time.perf_counter() - cache_started
    cache_layout, coarse_dims = _kupl_legacy_cache_layout(
        *assets, lattice=args.lat, block=args.block, nvec=args.nvec)
    manifest = _kupl_manifest(
        cache, lattice=args.lat, block=args.block, nvec=args.nvec,
        coarse_dof=args.nvec)
    payload["inputs"]["legacy_cache"].update({
        "backend": "kupl",
        "layout": cache_layout,
        "coarse_geometry": list(coarse_dims),
        "block": list(args.block),
        "manifest": manifest,
    })

    runtime = QcuRuntime.from_lattice(
        args.lat, mass=args.mass, csw=args.csw, atol=args.tolerance,
        dtype=torch.complex64, device="cpu", plan=2, parity=0,
        max_iter=args.max_iter, grid=(1, 1, 1, 1), qcu_module=qcu)
    ce, cei = runtime.make_clover(gauge, with_inverse=True, parity=0)
    coo, coi = runtime.make_clover(gauge, with_inverse=True, parity=1)
    _configure_kupl_common(runtime, args, coarse_dims, args.nvec)
    runtime.params[define._SET_PLAN_] = define._SET_PLAN1_
    runtime.params[define._PARITY_] = (
        1 if cache_layout == "odd-schur-33-stencil" else 0)
    runtime.params[define._MG_LEVEL1_NUM_RESTART_] = args.correction_interval
    runtime.params[define._MG_USE_DEFLATE_] = int(args.legacy_deflate)
    runtime.params[define._VERBOSE_] = 1
    runtime.set_ptrs[30:34] = 0
    for index, tensor in enumerate(assets):
        runtime.set_ptrs[30 + index] = tensor.data_ptr()
    rhs = tools.oooxyzt2poooxyzt(source_full).reshape(
        2, 4, 3, args.lat[0], args.lat[1], args.lat[2], args.lat[3] // 2
    ).contiguous()
    mg = LegacyMultigrid(
        runtime.params, runtime.argv, runtime.set_ptrs, gauge, ce, coo,
        cei, coi, cache_layout=cache_layout)
    setup_started = time.perf_counter()
    mg.setup()
    payload["timing"]["setup_seconds"] = time.perf_counter() - setup_started
    verify_status = mg.verify(rhs)
    if verify_status != 0:
        mg.close()
        raise RuntimeError(
            f"KUPL Legacy hierarchy verification failed: {verify_status}")

    warmups: list[float] = []
    steady: list[float] = []
    for index in range(args.warmups + args.repeats):
        print(f"QCU_SIX_WAY_SOLVE_BEGIN\t{index}", flush=True)
        started = time.perf_counter()
        output = mg.solve(rhs)
        elapsed = time.perf_counter() - started
        print(f"QCU_SIX_WAY_SOLVE_END\t{index}\t{elapsed:.17g}", flush=True)
        (warmups if index < args.warmups else steady).append(elapsed)
    full_output = tools.poooxyzt2oooxyzt(output).contiguous()
    independent = _true_residual_torch(
        torch, __import__("pyqcu.torch.dslash", fromlist=["dslash"]),
        full_output, source_full, gauge_full, args.mass, args.csw)
    host_rss_end = _current_rss_bytes()
    mg.close()
    payload["timing"].update({
        "warmup_seconds": warmups,
        "first_solve_seconds": (warmups or steady)[0],
        "steady_seconds": steady,
        "steady_median_seconds": _median(steady),
        "steady_mad_seconds": _mad(steady),
    })
    payload["iterations"] = {
        "samples": [],
        "median": None,
        "semantics": "parsed by the orchestrator from KUPL C++ debug stdout",
    }
    payload["residual"] = {
        "independent_relative": independent,
        "gate": args.tolerance,
        "pass": bool(independent <= args.tolerance),
        "source": "independent PyTorch full Wilson/Clover reconstruction",
        "scope": "full",
    }
    payload["coarse_execution"] = {
        "kind": "legacy_periodic_correction",
        "executed": None,
        "attempted_vcycles": None,
        "native_verify": int(verify_status),
        "source": "KUPL C++ debug stdout parsed by the orchestrator",
    }
    payload["memory"] = {
        "device_memory_applicable": False,
        "host_peak_rss_bytes": _peak_rss_bytes(),
        "host_rss_start_bytes": host_rss_start,
        "host_rss_end_bytes": host_rss_end,
        "workspace_scope": "CPU vectors allocated by the C++ Legacy solver",
    }
    payload["status"] = "PASS" if payload["residual"]["pass"] else "FAIL"
    if args.solution_output:
        _save_h5(Path(args.solution_output), full_output.numpy(),
                 backend="kupl", strategy="legacy", lattice=json.dumps(args.lat))
    return payload


def _worker_kupl_strict(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from pyqcu import tools
    from pyqcu.cuda import QcuRuntime, define
    from pyqcu.kupl import qcu
    from pyqcu.kupl.multigrid import StrictMultigrid

    payload = _common_payload(args, backend="kupl", strategy="strict")
    assets_path = Path(args.strict_assets).expanduser().resolve()
    payload["inputs"]["strict_assets"] = {
        "path": str(assets_path), "sha256": _sha256(assets_path),
        "size_bytes": assets_path.stat().st_size,
        "source": "exported from the Torch strict hierarchy on the same gauge/null vectors",
    }
    host_rss_start = _current_rss_bytes()
    load_started = time.perf_counter()
    gauge_full = torch.from_numpy(_load_h5(Path(args.gauge))).to(
        dtype=torch.complex64).contiguous()
    source_full = torch.from_numpy(_load_h5(Path(args.source))).to(
        dtype=torch.complex64).contiguous()
    gauge = tools.oooxyzt2poooxyzt(gauge_full).contiguous()
    payload["timing"]["input_load_seconds"] = time.perf_counter() - load_started

    cache_started = time.perf_counter()
    with __import__("h5py").File(assets_path, "r") as handle:
        fine_null = torch.from_numpy(handle["fine_null"][...]).contiguous()
        links = torch.from_numpy(handle["links"][...]).contiguous()
        onsite = torch.from_numpy(handle["onsite"][...]).contiguous()
    payload["timing"]["cache_load_seconds"] = time.perf_counter() - cache_started
    coarse_dims = _kupl_strict_asset_layout(
        fine_null, links, onsite, lattice=args.lat, block=args.block,
        nvec=args.nvec)
    manifest = _kupl_manifest(
        assets_path, lattice=args.lat, block=args.block, nvec=args.nvec,
        coarse_dof=2 * args.nvec)
    payload["inputs"]["strict_assets"].update({
        "backend": "kupl",
        "layout": "parity-compressed-matpc",
        "coarse_geometry": list(coarse_dims),
        "block": list(args.block),
        "manifest": manifest,
    })
    rhs_compact = tools.oooxyzt2poooxyzt(source_full).reshape(
        2, 4, 3, args.lat[0], args.lat[1], args.lat[2], args.lat[3] // 2
    ).contiguous()
    rhs = rhs_compact.reshape(12, *args.lat).contiguous()

    def make_runtime():
        runtime = QcuRuntime.from_lattice(
            args.lat, mass=args.mass, csw=args.csw, atol=args.tolerance,
            dtype=torch.complex64, device="cpu", plan=2, parity=0,
            max_iter=args.max_iter, grid=(1, 1, 1, 1), qcu_module=qcu)
        ce, cei = runtime.make_clover(gauge, with_inverse=True, parity=0)
        coo, coi = runtime.make_clover(gauge, with_inverse=True, parity=1)
        _configure_kupl_common(runtime, args, coarse_dims, 2 * args.nvec)
        runtime.set_ptrs[60] = fine_null.data_ptr()
        runtime.set_ptrs[62] = links.data_ptr()
        runtime.set_ptrs[63] = onsite.data_ptr()
        return runtime, ce, cei, coo, coi

    def split_runs(warmups, steady):
        return warmups, steady

    def run_public():
        runtime, ce, cei, coo, coi = make_runtime()
        manager = StrictMultigrid(
            runtime.params, runtime.argv, runtime.set_ptrs, gauge, ce, coo,
            cei, coi, fine_null, args.lat, coarse_dims,
            restart=args.restart, max_iter=args.max_iter,
            tolerance=args.tolerance, nu_pre=args.nu, nu_post=args.nu,
            max_workspace_bytes=args.max_workspace_bytes, start_level=1)
        setup_started = time.perf_counter()
        manager.setup()
        setup_seconds = time.perf_counter() - setup_started
        warmups: list[float] = []
        steady: list[float] = []
        iterations: list[int] = []
        backend_residuals: list[float] = []
        workspace_samples: list[int] = []
        try:
            for index in range(args.warmups + args.repeats):
                print(
                    f"QCU_SIX_WAY_SOLVE_BEGIN\tpublic\t{index}", flush=True)
                started = time.perf_counter()
                output = manager.solve(rhs)
                elapsed = time.perf_counter() - started
                print(
                    f"QCU_SIX_WAY_SOLVE_END\tpublic\t{index}\t"
                    f"{elapsed:.17g}", flush=True)
                warmups, steady = split_runs(warmups, steady)
                (warmups if index < args.warmups else steady).append(elapsed)
                iterations.append(int(manager.iterations.item()))
                backend_residuals.append(float(manager.final_residual.item()))
                workspace_samples.append(int(manager.allocated_bytes.item()))
            result = output.detach().clone()
        finally:
            manager.close()
        return {
            "output": result,
            "setup_seconds": setup_seconds,
            "warmups": warmups,
            "steady": steady,
            "iterations": iterations,
            "backend_residuals": backend_residuals,
            "workspace_samples": workspace_samples,
            "persistent_bytes": int(manager.allocated_bytes.item()),
        }

    def run_raw():
        runtime, ce, cei, coo, coi = make_runtime()
        output = torch.empty_like(rhs_compact)
        library = qcu._LIBRARY
        setup_started = time.perf_counter()
        qcu.applyInitQcu(runtime.set_ptrs, runtime.params, runtime.argv)
        setup_workspace = torch.zeros(1, dtype=torch.int64)
        status = qcu.applyMultigridStrictInitQcu(
            runtime.set_ptrs, runtime.params, 1, setup_workspace)
        setup_seconds = time.perf_counter() - setup_started
        if status != 0:
            qcu.applyEndQcu(runtime.set_ptrs, runtime.params)
            raise RuntimeError(f"KUPL strict hierarchy init failed: {status}")
        warmups: list[float] = []
        steady: list[float] = []
        iterations: list[int] = []
        backend_residuals: list[float] = []
        workspace_samples: list[int] = []
        try:
            for index in range(args.warmups + args.repeats):
                output.zero_()
                runtime.params[define._MG_USE_INIT_GUESS_] = 0
                result_iterations = ctypes.c_int(0)
                converged = ctypes.c_int(0)
                backend_residual = ctypes.c_double(0.0)
                workspace = ctypes.c_ulonglong(0)
                print(f"QCU_SIX_WAY_SOLVE_BEGIN\traw\t{index}", flush=True)
                started = time.perf_counter()
                status = library.applyMultigridStrictFgmresQcu(
                    int(output.data_ptr()), int(rhs_compact.data_ptr()),
                    int(gauge.data_ptr()), int(ce.data_ptr()),
                    int(coo.data_ptr()), int(cei.data_ptr()),
                    int(coi.data_ptr()), int(fine_null.data_ptr()),
                    int(runtime.set_ptrs.data_ptr()),
                    int(runtime.params.data_ptr()),
                    args.nvec, *args.lat, 2 * args.nvec, *coarse_dims,
                    int(gauge.element_size()), args.restart, args.max_iter,
                    args.tolerance, args.nu, args.nu,
                    args.max_workspace_bytes,
                    ctypes.byref(result_iterations), ctypes.byref(converged),
                    ctypes.byref(backend_residual), ctypes.byref(workspace))
                elapsed = time.perf_counter() - started
                print(
                    f"QCU_SIX_WAY_SOLVE_END\traw\t{index}\t"
                    f"{elapsed:.17g}", flush=True)
                if status != 0:
                    raise RuntimeError(f"KUPL strict FGMRES failed: {status}")
                if not bool(converged.value):
                    raise RuntimeError("KUPL strict FGMRES did not converge")
                warmups, steady = split_runs(warmups, steady)
                (warmups if index < args.warmups else steady).append(elapsed)
                iterations.append(int(result_iterations.value))
                backend_residuals.append(float(backend_residual.value))
                workspace_samples.append(int(workspace.value))
            result = output.detach().clone()
        finally:
            qcu.applyMultigridStrictEndQcu(runtime.set_ptrs, runtime.params)
            qcu.applyEndQcu(runtime.set_ptrs, runtime.params)
        return {
            "output": result,
            "setup_seconds": setup_seconds,
            "warmups": warmups,
            "steady": steady,
            "iterations": iterations,
            "backend_residuals": backend_residuals,
            "workspace_samples": workspace_samples,
            "persistent_bytes": int(setup_workspace.item()),
        }

    requested_abi = getattr(args, "kupl_strict_abi", "public")
    runs: dict[str, dict[str, Any]] = {}
    if requested_abi in ("public", "both"):
        runs["public"] = run_public()
    if requested_abi in ("raw", "both"):
        runs["raw"] = run_raw()
    selected_abi = "public" if "public" in runs else "raw"
    selected = runs[selected_abi]
    output = selected["output"]
    full_output = tools.poooxyzt2oooxyzt(output)
    independent = _true_residual_torch(
        torch, __import__("pyqcu.torch.dslash", fromlist=["dslash"]),
        full_output.contiguous(), source_full, gauge_full, args.mass, args.csw)
    host_rss_end = _current_rss_bytes()
    if "public" in runs and "raw" in runs:
        public_output = runs["public"]["output"]
        raw_output = runs["raw"]["output"]
        denominator = max(float(torch.linalg.vector_norm(public_output)), 1.0e-30)
        abi_difference = float(
            torch.linalg.vector_norm(public_output - raw_output) / denominator)
        abi_gate = max(5.0e-6, 20.0 * float(args.tolerance))
        if abi_difference > abi_gate:
            raise RuntimeError(
                f"KUPL public/raw ABI solution mismatch {abi_difference:.3e} "
                f"> {abi_gate:.3e}")
    else:
        abi_difference = None
        abi_gate = None
    payload["timing"].update({
        "warmup_seconds": selected["warmups"],
        "first_solve_seconds": (
            selected["warmups"] or selected["steady"])[0],
        "steady_seconds": selected["steady"],
        "steady_median_seconds": _median(selected["steady"]),
        "steady_mad_seconds": _mad(selected["steady"]),
    })
    payload["iterations"] = {
        "samples": selected["iterations"],
        "median": _median([float(value) for value in selected["iterations"]]),
        "semantics": "strict outer right-preconditioned FGMRES iterations",
    }
    payload["residual"] = {
        "independent_relative": independent,
        "backend_matpc_samples": selected["backend_residuals"],
        "gate": args.tolerance,
        "pass": bool(independent <= args.tolerance),
        "source": "independent PyTorch full Wilson/Clover reconstruction",
        "backend_scope": "fine compact MATPC residual",
        "scope": "full",
    }
    payload["coarse_execution"] = {
        "kind": "strict_recursive_vcycle",
        "executed": bool(
            selected["iterations"] and min(selected["iterations"]) > 0),
        "cycle_calls_per_solve_lower_bound": selected["iterations"],
        "source": "the KUPL strict FGMRES preconditioner invokes one recursive V-cycle per outer iteration",
    }
    payload["memory"] = {
        "device_memory_applicable": False,
        "host_peak_rss_bytes": _peak_rss_bytes(),
        "host_rss_start_bytes": host_rss_start,
        "host_rss_end_bytes": host_rss_end,
        "persistent_hierarchy_bytes": selected["persistent_bytes"],
        "solve_workspace_samples_bytes": selected["workspace_samples"],
        "workspace_formula": "(2m+5)B_f+2B_c",
    }
    payload["abi_validation"] = {
        "requested": requested_abi,
        "selected": selected_abi,
        "available": sorted(runs),
        "solution_relative_difference": abi_difference,
        "gate": abi_gate,
    }
    payload["provenance"]["execution_abi"] = {
        "public": "pyqcu.kupl.multigrid.StrictMultigrid",
        "raw": "ctypes C ABI",
        "selected": selected_abi,
    }
    payload["provenance"].pop("public_wrapper_blocker", None)
    payload["status"] = "PASS" if payload["residual"]["pass"] else "FAIL"
    if args.solution_output:
        _save_h5(Path(args.solution_output), full_output.numpy(),
                 backend="kupl", strategy="strict", lattice=json.dumps(args.lat))
    return payload


def _scalar(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.reshape(-1)[0] if value.size else default
    except Exception:
        pass
    return value


def _worker_quda(args: argparse.Namespace) -> dict[str, Any]:
    from _common import configure_quda_runtime, qdp_fermion_to_qcu

    configure_quda_runtime(preload=True)
    os.environ["QUDA_ENABLE_TUNING"] = "0"
    import torch
    import pyquda
    import pyquda_utils.core as core
    from pyquda.enum_quda import (
        QudaBoolean,
        QudaInverterType,
        QudaMassNormalization,
        QudaPreserveSource,
        QudaUseInitGuess,
        QudaVerbosity,
    )
    from pyquda.field import LatticeFermion, LatticeGauge
    from quda_bistabcg import _load_gauge, _load_source, make_dirac

    payload = _common_payload(args, backend="quda", strategy="strict")
    sampler = None
    dirac = None
    try:
        runtime_started = time.perf_counter()
        pyquda.init(
            grid_size=[1, 1, 1, 1], latt_size=list(args.lat),
            backend="torch", backend_target="cuda", enable_nvshmem=False,
            enable_tuning=False, enable_device_memory_pool=False,
            enable_pinned_memory_pool=False)
        payload["timing"]["runtime_init_seconds"] = time.perf_counter() - runtime_started
        info = core.LatticeInfo(list(args.lat), 1, 1.0)

        load_started = time.perf_counter()
        class _Args:
            pass

        loader_args = _Args()
        loader_args.lat = list(args.lat)
        loader_args.mass = args.mass
        loader_args.csw = args.csw
        loader_args.gauge = args.gauge
        loader_args.source = args.source
        loader_args.seed = args.seed
        qdp_gauge, resolved_csw, _ = _load_gauge(loader_args)
        canonical_source, qdp_source = _load_source(loader_args, list(args.lat))
        gauge = LatticeGauge(
            info, 4,
            torch.from_numpy(info.evenodd(qdp_gauge.astype(complex), True)).to("cuda"))
        source = LatticeFermion(
            info,
            torch.from_numpy(info.evenodd(qdp_source.astype(complex), False)).to("cuda"))
        payload["timing"]["input_load_seconds"] = time.perf_counter() - load_started

        dirac = make_dirac(
            info, args.mass, args.tolerance, args.max_iter, resolved_csw,
            multigrid=[list(args.block) for _ in range(args.levels - 1)])
        invert = dirac.invert_param
        mg = dirac.multigrid.param
        invert.mass_normalization = QudaMassNormalization.QUDA_KAPPA_NORMALIZATION
        invert.use_init_guess = QudaUseInitGuess.QUDA_USE_INIT_GUESS_NO
        invert.use_resident_solution = 0
        invert.preserve_source = QudaPreserveSource.QUDA_PRESERVE_SOURCE_YES
        invert.inv_type = QudaInverterType.QUDA_GCR_INVERTER
        invert.gcrNkrylov = args.restart
        invert.maxiter = args.max_iter
        invert.tol = args.tolerance
        invert.tol_precondition = args.coarse_tolerance
        invert.maxiter_precondition = 1
        invert.verbosity = QudaVerbosity.QUDA_SUMMARIZE
        invert.verbosity_precondition = QudaVerbosity.QUDA_SILENT

        array_length = len(mg.n_vec)
        mg.n_vec = [args.nvec] * array_length
        mg.n_block_ortho = [2] * len(mg.n_block_ortho)
        mg.nu_pre = [args.nu] * len(mg.nu_pre)
        mg.nu_post = [args.nu] * len(mg.nu_post)
        mg.smoother = [QudaInverterType.QUDA_MR_INVERTER] * len(mg.smoother)
        mg.smoother_tol = [0.0] * len(mg.smoother_tol)
        mg.coarse_solver = (
            [QudaInverterType.QUDA_GCR_INVERTER] +
            [QudaInverterType.QUDA_CA_GCR_INVERTER] * (len(mg.coarse_solver) - 1))
        mg.coarse_solver_maxiter = (
            [1, args.coarse_max_iter] +
            [16] * max(0, len(mg.coarse_solver_maxiter) - 2))
        mg.coarse_solver_tol = [args.coarse_tolerance] * len(mg.coarse_solver_tol)
        mg.setup_use_mma = [QudaBoolean.QUDA_BOOLEAN_FALSE] * len(mg.setup_use_mma)
        mg.dslash_use_mma = [QudaBoolean.QUDA_BOOLEAN_FALSE] * len(mg.dslash_use_mma)
        mg.num_setup_iter = [1] + [0] * (len(mg.num_setup_iter) - 1)
        mg.generate_all_levels = QudaBoolean.QUDA_BOOLEAN_FALSE
        mg.run_verify = QudaBoolean.QUDA_BOOLEAN_FALSE

        sampler = _CudaDeviceSampler(torch, torch.device("cuda:0")).start()
        setup_started = time.perf_counter()
        dirac.loadGauge(gauge)
        torch.cuda.synchronize()
        payload["timing"]["setup_seconds"] = time.perf_counter() - setup_started

        warmups: list[float] = []
        steady: list[float] = []
        iterations: list[int] = []
        backend_residuals: list[float] = []
        output = None
        for index in range(args.warmups + args.repeats):
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = dirac.invert(source)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            (warmups if index < args.warmups else steady).append(elapsed)
            iterations.append(int(_scalar(invert.iter, 0)))
            backend_residuals.append(float(_scalar(invert.true_res, float("nan"))))
        device_memory = sampler.stop()
        sampler = None
        allocator_memory = _torch_memory_snapshot(torch, torch.device("cuda:0"))
        if output is None:
            raise RuntimeError("QUDA produced no solution")
        solution_qcu = qdp_fermion_to_qcu(
            output.data.detach().cpu().numpy().astype(complex))
        if args.solution_output:
            _save_h5(Path(args.solution_output), solution_qcu,
                     backend="quda", strategy="aligned", lattice=json.dumps(args.lat))

        backend_residual = backend_residuals[-1]
        payload["timing"].update({
            "cache_load_seconds": None,
            "cache_note": "QUDA generated its own null vectors from the same gauge",
            "warmup_seconds": warmups,
            "first_solve_seconds": warmups[0],
            "steady_seconds": steady,
            "steady_median_seconds": _median(steady),
            "steady_mad_seconds": _mad(steady),
        })
        payload["iterations"] = {
            "samples": iterations,
            "median": _median([float(value) for value in iterations]),
            "semantics": "outer right-preconditioned GCR iterations",
        }
        payload["residual"] = {
            "backend_relative": backend_residual,
            "backend_samples": backend_residuals,
            "gate": args.tolerance,
            "backend_pass": bool(backend_residual <= args.tolerance),
            "source": "QUDA InvertParam.true_res; independent check stored separately",
            "scope": "full",
        }
        payload["coarse_execution"] = {
            "kind": "quda_aligned_multigrid",
            "executed": bool(iterations and min(iterations) > 0),
            "cycle_calls_per_solve_lower_bound": iterations,
            "source": "right-preconditioned GCR invokes the configured MG preconditioner each outer iteration",
        }
        payload["memory"] = {
            "device_memory_applicable": True,
            "device_wide": device_memory,
            "allocator": allocator_memory,
            "host_peak_rss_bytes": _peak_rss_bytes(),
        }
        payload["quda_configuration"] = {
            "version": getattr(pyquda, "__version__", None),
            "strategy": "aligned",
            "n_vec_readback": [_scalar(value) for value in mg.n_vec],
            "nu_pre_readback": [_scalar(value) for value in mg.nu_pre],
            "nu_post_readback": [_scalar(value) for value in mg.nu_post],
            "smoother_readback": [str(_scalar(value)) for value in mg.smoother],
            "generate_all_levels": str(mg.generate_all_levels),
            "null_vector_source": "QUDA null-vector setup on the shared gauge",
            "same_canonical_null_vectors": False,
            "tuning": "QUDA_ENABLE_TUNING=0",
        }
        payload["provenance"].update({
            "quda_install": os.environ.get("QUDA_INSTALL"),
            "quda_library": str(Path(os.environ.get("QUDA_INSTALL", "")) / "lib" / "libquda.so"),
            "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
        })
        payload["status"] = "PASS" if payload["residual"]["backend_pass"] else "FAIL"
        return payload
    finally:
        if sampler is not None:
            try:
                sampler.stop()
            except Exception:
                pass
        if dirac is not None:
            for name in ("freeGauge", "destroy", "close"):
                method = getattr(dirac, name, None)
                if callable(method):
                    try:
                        method()
                    except Exception:
                        pass
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


def _worker_residual(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from pyqcu.torch import dslash

    gauge = torch.from_numpy(_load_h5(Path(args.gauge))).to(
        dtype=torch.complex64).contiguous()
    source = torch.from_numpy(_load_h5(Path(args.source))).to(
        dtype=torch.complex64).contiguous()
    output = torch.from_numpy(_load_h5(Path(args.solution))).to(
        dtype=torch.complex64).contiguous()
    relative = _true_residual_torch(
        torch, dslash, output, source, gauge, args.mass, args.csw)
    return {
        "schema": "pyqcu.qcu-six-way.residual.v1",
        "variant": args.variant,
        "solution": str(Path(args.solution).expanduser().resolve()),
        "solution_sha256": _sha256(Path(args.solution).expanduser().resolve()),
        "independent_relative": relative,
        "gate": args.tolerance,
        "pass": bool(relative <= args.tolerance),
        "source": "independent PyTorch full Wilson/Clover reconstruction",
        "scope": "full",
    }


def _failure_payload(args: argparse.Namespace, error: BaseException) -> dict[str, Any]:
    strategy = ("bicgstab" if "bicgstab" in args.variant else
                "strict" if "strict" in args.variant else
                "legacy" if "legacy" in args.variant else "aligned")
    backend = args.variant.split("-", 1)[0]
    payload = _common_payload(args, backend=backend, strategy=strategy)
    payload.update({
        "status": "FAIL",
        "error": f"{type(error).__name__}: {error}",
        "traceback": traceback.format_exc(),
    })
    return payload


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--variant", required=True)
    parser.add_argument("--gauge", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--nullvec")
    parser.add_argument("--null-dataset", default="null")
    parser.add_argument("--legacy-cache")
    parser.add_argument("--strict-assets")
    parser.add_argument("--solution", help="input solution for residual mode")
    parser.add_argument("--solution-output")
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
    parser.add_argument("--legacy-deflate", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--legacy-adaptive-correction",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-workspace-bytes", type=int, default=512 << 20)
    parser.add_argument(
        "--kupl-strict-abi", choices=("public", "raw", "both"),
        default="public",
        help="KUPL Strict execution path; 'both' cross-checks public/raw outputs")
    parser.add_argument("--output", required=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_common(parser)
    args = parser.parse_args(argv)
    args.lat = tuple(int(value) for value in args.lat)
    args.block = tuple(int(value) for value in args.block)
    try:
        if args.variant in ("torch-legacy", "torch-strict"):
            payload = _worker_torch(args)
        elif args.variant == "torch-bicgstab":
            payload = _worker_torch_bicgstab(args)
        elif args.variant == "kupl-legacy":
            payload = _worker_kupl_legacy(args)
        elif args.variant == "kupl-strict":
            payload = _worker_kupl_strict(args)
        elif args.variant == "quda-strict":
            payload = _worker_quda(args)
        elif args.variant == "residual":
            if not args.solution:
                parser.error("--solution is required for residual mode")
            payload = _worker_residual(args)
        else:
            parser.error(f"unknown variant: {args.variant}")
        code = 0 if payload.get("status", "PASS") == "PASS" and payload.get("pass", True) else 1
    except BaseException as error:
        payload = _failure_payload(args, error)
        payload["traceback"] = traceback.format_exc()
        code = 1
    _json_write(Path(args.output), payload)
    print(json.dumps(payload, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
