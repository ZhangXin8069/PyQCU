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


def _torch_memory_snapshot(torch: Any, device: Any) -> dict[str, Any]:
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
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with contextlib.redirect_stdout(stream):
        result = function()
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
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    overall_sampler = _CudaDeviceSampler(torch, device).start()
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
    torch.cuda.synchronize(device)
    payload["timing"]["input_load_seconds"] = time.perf_counter() - load_started
    payload["timing"]["cache_load_seconds"] = payload["timing"]["input_load_seconds"] if strategy == "strict" else 0.0

    kappa = torch.tensor(
        [1.0 / (2.0 * args.mass + 8.0)],
        dtype=torch.float32, device=device)
    u0 = torch.ones(1, dtype=torch.float32, device=device)
    prep_started = time.perf_counter()
    clover = dslash.make_clover(gauge, kappa=kappa, u_0=u0, csw=args.csw)
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

    torch.cuda.synchronize(device)
    first_before = _torch_memory_snapshot(torch, device)
    phase_sampler = _CudaDeviceSampler(torch, device).start()
    first_elapsed, output, first_text = _capture_call(
        lambda: mg.solve(source), torch, device)
    first_device = phase_sampler.stop()
    first_after = _torch_memory_snapshot(torch, device)
    warmups = [first_elapsed]
    iteration_samples = [_last_total_iterations(first_text)]
    if strategy == "strict":
        iteration_samples[-1] = len(mg.convergence_history)
    cycle_counts = [len(cycle_levels)]
    cycle_levels.clear()

    for _ in range(max(0, args.warmups - 1)):
        elapsed, output, text = _capture_call(
            lambda: mg.solve(source), torch, device)
        warmups.append(elapsed)
        iterations = (_last_total_iterations(text) if strategy == "legacy"
                      else len(mg.convergence_history))
        iteration_samples.append(iterations)
        cycle_counts.append(len(cycle_levels))
        cycle_levels.clear()

    steady: list[float] = []
    steady_sampler = _CudaDeviceSampler(torch, device).start()
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(args.repeats):
        elapsed, output, text = _capture_call(
            lambda: mg.solve(source), torch, device)
        steady.append(elapsed)
        iterations = (_last_total_iterations(text) if strategy == "legacy"
                      else len(mg.convergence_history))
        iteration_samples.append(iterations)
        cycle_counts.append(len(cycle_levels))
        cycle_levels.clear()
    steady_device = steady_sampler.stop()
    steady_allocator = _torch_memory_snapshot(torch, device)

    independent = _true_residual_torch(
        torch, dslash, output, source, gauge, args.mass, args.csw)
    overall_device = overall_sampler.stop()
    payload["timing"].update({
        "warmup_seconds": warmups,
        "first_solve_seconds": warmups[0],
        "steady_seconds": steady,
        "steady_median_seconds": _median(steady),
        "steady_mad_seconds": _mad(steady),
    })
    payload["iterations"] = {
        "samples": iteration_samples,
        "median": _median([float(value) for value in iteration_samples if value is not None]),
        "semantics": (
            "Legacy outer fine BiCGStab iterations parsed from the backend log"
            if strategy == "legacy" else
            "strict outer right-preconditioned FGMRES iterations"
        ),
    }
    payload["residual"] = {
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
        "source": "benchmark-only wrapper around the public Python cycle method",
    }
    payload["memory"] = {
        "device_memory_applicable": True,
        "device_wide": overall_device,
        "first_solve_device_wide": first_device,
        "steady_device_wide": steady_device,
        "first_solve_allocator_before": first_before,
        "first_solve_allocator_after": first_after,
        "first_solve_allocator_delta_bytes": max(
            0, int(first_after["allocated_bytes"]) - int(first_before["allocated_bytes"])),
        "steady_allocator": steady_allocator,
        "host_peak_rss_bytes": _peak_rss_bytes(),
    }
    payload["status"] = "PASS" if payload["residual"]["pass"] else "FAIL"
    if args.solution_output:
        _save_h5(
            Path(args.solution_output), output.detach().cpu().numpy(),
            backend="torch", strategy=strategy, lattice=json.dumps(args.lat))
    mg.close()
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

    runtime = QcuRuntime.from_lattice(
        args.lat, mass=args.mass, csw=args.csw, atol=args.tolerance,
        dtype=torch.complex64, device="cpu", plan=2, parity=0,
        max_iter=args.max_iter, grid=(1, 1, 1, 1), qcu_module=qcu)
    ce, cei = runtime.make_clover(gauge, with_inverse=True, parity=0)
    coo, coi = runtime.make_clover(gauge, with_inverse=True, parity=1)
    _configure_kupl_common(runtime, args, (4, 4, 4, 8), args.nvec)
    runtime.params[define._SET_PLAN_] = define._SET_PLAN1_
    runtime.params[define._PARITY_] = 0
    runtime.params[define._MG_LEVEL1_NUM_RESTART_] = args.correction_interval
    runtime.params[define._MG_USE_DEFLATE_] = 1
    runtime.params[define._VERBOSE_] = 1
    runtime.set_ptrs[30:34] = 0
    for index, tensor in enumerate(assets):
        runtime.set_ptrs[30 + index] = tensor.data_ptr()
    rhs = tools.oooxyzt2poooxyzt(source_full).reshape(
        2, 4, 3, args.lat[0], args.lat[1], args.lat[2], args.lat[3] // 2
    ).contiguous()
    mg = LegacyMultigrid(
        runtime.params, runtime.argv, runtime.set_ptrs, gauge, ce, coo,
        cei, coi)
    setup_started = time.perf_counter()
    mg.setup()
    payload["timing"]["setup_seconds"] = time.perf_counter() - setup_started

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
        "first_solve_seconds": warmups[0],
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

    runtime = QcuRuntime.from_lattice(
        args.lat, mass=args.mass, csw=args.csw, atol=args.tolerance,
        dtype=torch.complex64, device="cpu", plan=2, parity=0,
        max_iter=args.max_iter, grid=(1, 1, 1, 1), qcu_module=qcu)
    ce, cei = runtime.make_clover(gauge, with_inverse=True, parity=0)
    coo, coi = runtime.make_clover(gauge, with_inverse=True, parity=1)
    _configure_kupl_common(runtime, args, (4, 4, 4, 8), 2 * args.nvec)
    runtime.set_ptrs[60] = fine_null.data_ptr()
    runtime.set_ptrs[62] = links.data_ptr()
    runtime.set_ptrs[63] = onsite.data_ptr()
    rhs = tools.oooxyzt2poooxyzt(source_full).reshape(
        2, 4, 3, args.lat[0], args.lat[1], args.lat[2], args.lat[3] // 2
    ).contiguous()
    output = torch.empty_like(rhs)

    setup_started = time.perf_counter()
    qcu.applyInitQcu(runtime.set_ptrs, runtime.params, runtime.argv)
    setup_workspace = ctypes.c_ulonglong(0)
    status = qcu.applyMultigridStrictInitQcu(
        runtime.set_ptrs, runtime.params, 1, setup_workspace)
    payload["timing"]["setup_seconds"] = time.perf_counter() - setup_started
    if status != 0:
        qcu.applyEndQcu(runtime.set_ptrs, runtime.params)
        raise RuntimeError(f"KUPL strict hierarchy init failed: {status}")

    library = qcu._LIBRARY
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
            print(f"QCU_SIX_WAY_SOLVE_BEGIN\t{index}", flush=True)
            started = time.perf_counter()
            status = library.applyMultigridStrictFgmresQcu(
                int(output.data_ptr()), int(rhs.data_ptr()),
                int(gauge.data_ptr()), int(ce.data_ptr()), int(coo.data_ptr()),
                int(cei.data_ptr()), int(coi.data_ptr()),
                int(fine_null.data_ptr()), int(runtime.set_ptrs.data_ptr()),
                int(runtime.params.data_ptr()),
                args.nvec, *args.lat, 2 * args.nvec, 4, 4, 4, 8,
                int(gauge.element_size()), args.restart, args.max_iter,
                args.tolerance, args.nu, args.nu, args.max_workspace_bytes,
                ctypes.byref(result_iterations), ctypes.byref(converged),
                ctypes.byref(backend_residual), ctypes.byref(workspace))
            elapsed = time.perf_counter() - started
            print(f"QCU_SIX_WAY_SOLVE_END\t{index}\t{elapsed:.17g}", flush=True)
            if status != 0:
                raise RuntimeError(f"KUPL strict FGMRES failed: {status}")
            if not bool(converged.value):
                raise RuntimeError("KUPL strict FGMRES did not converge")
            (warmups if index < args.warmups else steady).append(elapsed)
            iterations.append(int(result_iterations.value))
            backend_residuals.append(float(backend_residual.value))
            workspace_samples.append(int(workspace.value))
    finally:
        qcu.applyMultigridStrictEndQcu(runtime.set_ptrs, runtime.params)
        qcu.applyEndQcu(runtime.set_ptrs, runtime.params)

    full_output = tools.pooxyzt2oooxyzt2 if False else tools.poooxyzt2oooxyzt(output)
    independent = _true_residual_torch(
        torch, __import__("pyqcu.torch.dslash", fromlist=["dslash"]),
        full_output.contiguous(), source_full, gauge_full, args.mass, args.csw)
    host_rss_end = _current_rss_bytes()
    payload["timing"].update({
        "warmup_seconds": warmups,
        "first_solve_seconds": warmups[0],
        "steady_seconds": steady,
        "steady_median_seconds": _median(steady),
        "steady_mad_seconds": _mad(steady),
        "workspace_seconds": warmups[0],
    })
    payload["iterations"] = {
        "samples": iterations,
        "median": _median([float(value) for value in iterations]),
        "semantics": "strict outer right-preconditioned FGMRES iterations",
    }
    payload["residual"] = {
        "independent_relative": independent,
        "backend_matpc_samples": backend_residuals,
        "gate": args.tolerance,
        "pass": bool(independent <= args.tolerance),
        "source": "independent PyTorch full Wilson/Clover reconstruction",
        "backend_scope": "fine compact MATPC residual",
        "scope": "full",
    }
    payload["coarse_execution"] = {
        "kind": "strict_recursive_vcycle",
        "executed": bool(iterations and min(iterations) > 0),
        "cycle_calls_per_solve_lower_bound": iterations,
        "source": "the KUPL strict FGMRES preconditioner invokes one recursive V-cycle per outer iteration",
    }
    payload["memory"] = {
        "device_memory_applicable": False,
        "host_peak_rss_bytes": _peak_rss_bytes(),
        "host_rss_start_bytes": host_rss_start,
        "host_rss_end_bytes": host_rss_end,
        "persistent_hierarchy_bytes": int(setup_workspace.value),
        "solve_workspace_samples_bytes": workspace_samples,
        "workspace_formula": "(2m+5)B_f+2B_c",
    }
    payload["provenance"]["execution_abi"] = "raw ctypes C ABI"
    payload["provenance"]["public_wrapper_blocker"] = (
        "pyqcu.kupl.multigrid.StrictMultigrid passes a tensor for the ULL pointer "
        "to strict init and omits fine/coarse E in the FGMRES binding")
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
    strategy = ("strict" if "strict" in args.variant else
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
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-workspace-bytes", type=int, default=512 << 20)
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
