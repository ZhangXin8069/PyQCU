#!/usr/bin/env python3
"""MPI worker for the simple QCU Clover-MG examples.

KUPL is delegated to the established ``simple/kupl_multigrid.py`` path, which
already implements distributed halos and an independent distributed residual.
The CUDA path below follows the production topology: one Python process, one
MPI rank, one GPU/DCU/GPGPU. Multi-device solves use MPI domain decomposition;
the test-only ``MultiGpuMultigrid`` thread-per-GPU driver is intentionally not
used here. The same 33-tensor coarse cache layout is retained, with a NumPy
true-residual check independent of libqcu.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Sequence

import numpy as np
from mpi4py import MPI

from pyqcu.testing.simple._common import convergence_status, write_h5, write_json
from pyqcu.testing.simple.mpi._common import (
    canonical_fermion,
    canonical_gauge,
    load_coarse_assets,
    local_geometry,
    read_gauge,
    read_source,
    slice_last4,
    validate_lattice,
)
from pyqcu.testing.simple.mpi.mpi_ranks_power_of_two import validate_rank_count


def _affinity() -> list[int]:
    try:
        return sorted(int(cpu) for cpu in os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return []


def _run_kupl(args, comm: MPI.Comm) -> int:
    from pyqcu.testing.simple.kupl_multigrid import main as kupl_multigrid_main

    forwarded = [
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
        "--report", args.report,
        "--legacy-layout", args.kupl_legacy_layout,
    ]
    if args.coarse_cache:
        forwarded.extend(["--coarse-cache", args.coarse_cache])
    if args.output:
        forwarded.extend(["--output", args.output])
    if args.skip_independent_residual:
        forwarded.append("--skip-independent-residual")
    if args.verbose:
        forwarded.append("--verbose")
    exit_code = int(kupl_multigrid_main(forwarded))
    local_lattice, starts, coordinate = local_geometry(
        args.lat, args.grid, int(comm.Get_rank()))
    record = {
        "rank": int(comm.Get_rank()),
        "coordinate": list(coordinate),
        "local_lattice": list(local_lattice),
        "local_starts": list(starts),
        "affinity_cpus": _affinity(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "kupl_omp_threads": os.environ.get("KUPL_OMP_THREADS"),
        "kupl_numa_bind": os.environ.get("KUPL_NUMA_BIND"),
        "kupl_memory_policy": os.environ.get("KUPL_MEMORY_POLICY"),
        "kupl_memory_node": os.environ.get("KUPL_MEMORY_NODE"),
        "kupl_require_memory_bind": os.environ.get(
            "KUPL_REQUIRE_MEMORY_BIND"),
        "kupl_disable_clover_cache": os.environ.get(
            "KUPL_DISABLE_CLOVER_CACHE"),
        "kupl_fused_bicg_dots": os.environ.get("KUPL_FUSED_BICG_DOTS"),
        "kupl_hugepage": os.environ.get("KUPL_HUGEPAGE"),
        "pyqcu_kupl_profile": os.environ.get("PYQCU_KUPL_PROFILE"),
    }
    records = comm.gather(record, root=0)
    if int(comm.Get_rank()) == 0 and Path(args.report).is_file():
        payload = json.loads(Path(args.report).read_text(encoding="utf-8"))
        payload["ranks"] = records
        write_json(args.report, payload)
    return exit_code


def _cuda_local_rank(comm: MPI.Comm) -> int:
    local = comm.Split_type(MPI.COMM_TYPE_SHARED, 0)
    try:
        return int(local.Get_rank())
    finally:
        local.Free()


def _run_cuda(args, comm: MPI.Comm) -> int:
    rank = int(comm.Get_rank())
    size = int(comm.Get_size())
    local_rank = _cuda_local_rank(comm)
    # Production topology: this Python process owns exactly one MPI rank and
    # one GPGPU. The native backend reads this variable before pyqcu.cuda.qcu
    # is imported, so the process-to-device binding must be established here.
    os.environ["PYQCU_MPI_DEVICE_ID"] = str(local_rank)

    import torch
    from pyqcu.cuda import QcuRuntime, define
    from pyqcu.testing.simple.kupl_residual import (
        distributed_relative_residual,
        distributed_relative_residual_streaming,
        gather_solution,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA backend requested, but torch.cuda is unavailable")
    visible = int(torch.cuda.device_count())
    if visible < 1:
        raise RuntimeError("CUDA backend requested, but no visible GPU exists")
    if local_rank >= visible:
        raise RuntimeError(
            f"MPI local rank {local_rank} has no private CUDA device "
            f"(visible={visible}); one rank must map to one device")
    device_index = local_rank
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)
    tensor_dtype = torch.complex64 if args.dtype == "c64" else torch.complex128
    numpy_dtype = np.dtype(np.complex64 if args.dtype == "c64" else np.complex128)

    local_lattice, starts, coordinate = local_geometry(
        args.lat, args.grid, rank)
    gauge_path = Path(args.gauge).expanduser()
    source_path = Path(args.source).expanduser()
    if gauge_path.suffix.lower() in (".h5", ".hdf5"):
        from pyqcu.testing.simple.mpi._distributed_mg_cache import (
            _drop_file_cache,
            _hdf5_gauge_slab,
        )

        gauge_local = _hdf5_gauge_slab(
            gauge_path, starts, local_lattice, numpy_dtype)
        _drop_file_cache(gauge_path)
    else:
        gauge_full, _, _ = read_gauge(args.gauge, args.lat, numpy_dtype)
        gauge_local = slice_last4(gauge_full, starts, local_lattice)
    if source_path.suffix.lower() in (".h5", ".hdf5"):
        import h5py

        spatial = tuple(
            slice(int(starts[axis]), int(starts[axis] + local_lattice[axis]))
            for axis in range(4))
        with h5py.File(source_path, "r") as handle:
            source_local = np.ascontiguousarray(
                handle["data"][(slice(None), slice(None), *spatial)],
                dtype=numpy_dtype)
        _drop_file_cache(source_path)
    else:
        source_full, _, _ = read_source(args.source, args.lat, numpy_dtype)
        source_local = slice_last4(source_full, starts, local_lattice)
    gauge_np = canonical_gauge(gauge_local)
    source_np = canonical_fermion(source_local)

    runtime = QcuRuntime.from_lattice(
        local_lattice, mass=args.mass, csw=args.csw, atol=args.atol,
        dtype=tensor_dtype, device=device, plan=define._SET_PLAN2_,
        max_iter=args.max_iter, seed=args.seed,
        verbose=1 if args.verbose else 0, grid=args.grid)
    gauge = torch.from_numpy(gauge_np).to(device=device, dtype=tensor_dtype)
    source = torch.from_numpy(source_np).to(device=device, dtype=tensor_dtype)
    clover_even, clover_even_inv = runtime.make_clover(
        gauge, with_inverse=True, parity=0)
    clover_odd, clover_odd_inv = runtime.make_clover(
        gauge, with_inverse=True, parity=1)

    coarse_refs: list[torch.Tensor] = []
    use_coarse = args.levels > 1 and args.restart != 0
    runtime.params[define._MG_USE_DEFLATE_] = int(use_coarse)
    if use_coarse:
        cache_paths = [args.coarse_cache, *args.coarse_cache_level]
        if len(cache_paths) < args.levels - 1:
            raise ValueError(
                f"levels={args.levels} requires {args.levels - 1} coarse "
                f"caches, got {len(cache_paths)}")
        runtime.params[define._MG_NUM_LEVEL_] = args.levels
        # Runtime vectors are parity-compressed in time.  Slice caches in that
        # same geometry instead of relying on h5py to clip a full-T request.
        coarse_local = [
            int(local_lattice[0]), int(local_lattice[1]),
            int(local_lattice[2]), int(local_lattice[3]) // 2]
        coarse_starts = [
            int(starts[0]), int(starts[1]), int(starts[2]),
            int(starts[3]) // 2]
        for level_index, cache_path in enumerate(
                cache_paths[:args.levels - 1], start=1):
            if not cache_path:
                raise ValueError(
                    f"CUDA multigrid level {level_index} requires a cache")
            coarse_local = [
                coarse_local[axis] // args.mg_grid[axis] for axis in range(4)
            ]
            coarse_starts = [
                coarse_starts[axis] // args.mg_grid[axis] for axis in range(4)
            ]
            assets = load_coarse_assets(
                cache_path, numpy_dtype,
                starts=tuple(coarse_starts), local_shape=tuple(coarse_local))
            lonv, hnn, hdg, sit = [
                torch.from_numpy(value).to(
                    device=device, dtype=tensor_dtype).contiguous()
                for value in assets
            ]
            coarse_refs.extend((lonv, hnn, hdg, sit))
            _drop_file_cache(Path(cache_path))
            base = define._SET_PTRS_COARSE_BASE_ + 4 * (level_index - 1)
            runtime.set_ptrs[base + 0] = lonv.data_ptr()
            runtime.set_ptrs[base + 1] = hnn.data_ptr()
            runtime.set_ptrs[base + 2] = hdg.data_ptr()
            runtime.set_ptrs[base + 3] = sit.data_ptr()
            level_base = define._MG_LEVEL1_E_ + 8 * (level_index - 1)
            runtime.params[level_base + 0] = int(lonv.shape[0])
            runtime.params[level_base + 1] = int(lonv.shape[2])
            runtime.params[level_base + 2] = int(lonv.shape[4])
            runtime.params[level_base + 3] = int(lonv.shape[6])
            runtime.params[level_base + 4] = int(lonv.shape[8])
            runtime.params[level_base + 5] = int(args.coarse_max_iter)
            runtime.params[level_base + 6] = int(
                define.epytd(tensor_dtype).item())
            runtime.params[level_base + 7] = int(args.restart)
            runtime.argv[define._MG_LEVEL1_ATOL_ + level_index - 1] = float(
                args.atol * args.coarse_tol_factor *
                (3 ** (level_index - 1)))
    else:
        runtime.params[define._MG_NUM_LEVEL_] = 1

    runtime.configure(plan=define._SET_PLAN1_, parity=0,
                      mass=args.mass, csw=args.csw, atol=args.atol,
                      max_iter=args.max_iter)
    solution = torch.empty_like(source)
    comm.Barrier()
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with runtime.operation("applyCloverMultigridQcu") as operation:
        operation(solution, source, gauge, clover_even, clover_odd,
                  clover_even_inv, clover_odd_inv)
    torch.cuda.synchronize(device)
    comm.Barrier()
    solve_seconds = time.perf_counter() - started
    local_solve_seconds = float(solve_seconds)
    solve_seconds = float(comm.allreduce(solve_seconds, op=MPI.MAX))

    parity_shape = (*local_lattice[:3], local_lattice[3] // 2)
    solution_np = solution.detach().cpu().numpy().reshape(
        2, 12, *parity_shape)
    kappa = 1.0 / (2.0 * args.mass + 8.0)

    residual_started = time.perf_counter()
    if args.skip_independent_residual:
        residual = None
        converged = True
        status = "PASS"
        acceptance = float(args.atol)
        residual_source = "native solve only; independent check skipped"
    else:
        source_np_result = source.detach().cpu().numpy().reshape(
            2, 12, *parity_shape)
        gauge_np_result = gauge.detach().cpu().numpy()
        residual = distributed_relative_residual_streaming(
            solution_np, source_np_result, gauge_np_result,
            lambda target: (
                clover_even if target == 0 else clover_odd
            ).detach().cpu().numpy(), kappa,
            comm, coordinate, args.grid)
        converged, status, acceptance = convergence_status(
            residual, args.atol, dtype=numpy_dtype)
        residual_source = (
            "independent distributed NumPy Wilson-Clover reconstruction "
            "with MPI halo exchange")
    residual_seconds = time.perf_counter() - residual_started
    residual_seconds = float(comm.allreduce(residual_seconds, op=MPI.MAX))

    solution_full = gather_solution(
        solution_np, args.lat, args.grid, starts, comm)
    if rank == 0 and args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_name(output.name + ".partial")
        write_h5(partial, solution_full, metadata={
            "backend": "cuda",
            "solver": "applyCloverMultigridQcu",
            "mass": args.mass,
            "kappa": kappa,
            "csw": args.csw,
            "atol": args.atol,
            "lattice": list(args.lat),
            "grid": list(args.grid),
            "mpi_size": size,
            "layout": "[spin,color,x,y,z,t]",
        })
        os.replace(partial, output)

    local_record = {
        "rank": rank,
        "coordinate": list(coordinate),
        "gpu": device_index,
        "gpu_name": torch.cuda.get_device_name(device_index),
        "local_lattice": list(local_lattice),
        "local_starts": list(starts),
        "solve_seconds": local_solve_seconds,
        "affinity_cpus": _affinity(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }
    records = comm.gather(local_record, root=0)
    failed = int(comm.allreduce(int(status == "FAIL"), op=MPI.MAX))
    final_status = "FAIL" if failed else status
    report = {
        "status": final_status,
        "backend": "cuda",
        "solver": "applyCloverMultigridQcu",
        "lattice": list(args.lat),
        "grid": list(args.grid),
        "mpi_size": size,
        "dtype": args.dtype,
        "mass": args.mass,
        "kappa": kappa,
        "csw": args.csw,
        "atol": args.atol,
        "levels": args.levels if use_coarse else 1,
        "dof": list(args.dof),
        "mg_grid": list(args.mg_grid),
        "restart": args.restart,
        "coarse_cache": args.coarse_cache,
        "wall_seconds": solve_seconds,
        "residual_seconds": residual_seconds,
        "final_true_residual": residual,
        "residual_acceptance": acceptance,
        "residual_source": residual_source,
        "converged": converged and final_status != "FAIL",
        "ranks": records,
    }
    if rank == 0:
        write_json(args.report, report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if final_status != "FAIL" else 1


def _torch_device(args, comm: MPI.Comm):
    import pyqcu.cann as _torch

    requested = str(args.torch_device)
    on_dcu_partition = os.environ.get("SLURM_JOB_PARTITION") == "kshdnormal"
    if requested == "cpu" and on_dcu_partition:
        raise RuntimeError(
            "refusing a CPU-only Torch solve on the kshdnormal DCU partition")
    if requested == "cpu":
        return _torch.device("cpu"), None
    if not _torch.cuda.is_available():
        if requested == "cuda":
            raise RuntimeError("Torch CUDA requested, but CUDA is unavailable")
        if on_dcu_partition:
            raise RuntimeError(
                "refusing CPU fallback for a Torch solve on the DCU partition")
        return _torch.device("cpu"), None
    local_rank = _cuda_local_rank(comm)
    visible = int(_torch.cuda.device_count())
    if local_rank >= visible:
        raise RuntimeError(
            f"MPI local rank {local_rank} has no private CUDA device "
            f"(visible={visible})")
    _torch.cuda.set_device(local_rank)
    return _torch.device(f"cuda:{local_rank}"), local_rank


def _collective_call(comm: MPI.Comm, label: str, function):
    """Run one rank-local phase, then fail uniformly if any rank failed.

    Callers must complete all failure-prone local I/O, validation and tensor
    preparation before invoking any MPI operation.  If ``function`` enters
    MPI, that must happen only after a previous ``_collective_call`` has
    established that every rank is ready; otherwise one rank can still block
    in a mismatched send/receive or reduction.
    """
    rank = int(comm.Get_rank())
    try:
        result = function()
        local_failure = None
    except Exception as exc:
        result = None
        local_failure = (
            rank, type(exc).__name__, str(exc), traceback.format_exc())
    failures = [
        failure for failure in comm.allgather(local_failure)
        if failure is not None
    ]
    if failures:
        failed_rank, error_type, message, detail = failures[0]
        raise RuntimeError(
            f"{label} failed on rank {failed_rank} ({error_type}): "
            f"{message}\n{detail}") from None
    return result


def _torch_publish_solution(
        args, rank: int, local_lattice: Sequence[int], *,
        mpi_size: int, torch_num_threads: int, kappa: float, parts) -> None:
    """Assemble and atomically publish rank-zero Torch output."""
    if rank != 0 or not args.output:
        return
    if parts is None:
        raise RuntimeError("rank 0 did not receive gathered Torch solution parts")
    numpy_dtype = np.dtype(
        np.complex64 if args.dtype == "c64" else np.complex128)
    solution_full = np.zeros((4, 3, *args.lat), dtype=numpy_dtype)
    for part_coordinate, part in parts:
        part_starts = tuple(
            int(part_coordinate[axis]) * local_lattice[axis]
            for axis in range(4))
        selection = (slice(None), slice(None), *(
            slice(part_starts[axis],
                  part_starts[axis] + local_lattice[axis])
            for axis in range(4)))
        solution_full[selection] = part
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    try:
        write_h5(partial, solution_full, metadata={
            "backend": "torch",
            "solver": "LegacyMultigrid",
            "mass": args.mass,
            "kappa": kappa,
            "csw": args.csw,
            "atol": args.atol,
            "lattice": list(args.lat),
            "grid": list(args.grid),
            "mpi_size": mpi_size,
            "torch_num_threads": torch_num_threads,
            "warmup_runs": args.warmup,
            "timed_samples": args.repeat,
            "layout": "[spin,color,x,y,z,t]",
        })
        os.replace(partial, output)
    except BaseException:
        try:
            partial.unlink()
        except OSError:
            pass
        raise


def _torch_hierarchy_extents(
        local_lattice: Sequence[int], max_level: int, min_size: int,
        mg_grid: Sequence[int]) -> tuple[int, list[list[int]]]:
    """Mirror Legacy's level construction before loading coarse assets."""
    local = [int(value) for value in local_lattice]
    block = [int(value) for value in mg_grid]
    if len(local) != 4 or len(block) != 4 or any(value <= 0 for value in block):
        raise ValueError("local_lattice and mg_grid must contain four positive integers")
    effective_min_size = min(max(2, int(min_size)), min(local))
    current = list(local)
    extents: list[list[int]] = []
    while (len(extents) < int(max_level) and
           all(value >= effective_min_size for value in current)):
        extents.append(list(current))
        current = [current[axis] // block[axis] for axis in range(4)]
    if not extents:
        raise ValueError(
            f"local lattice {tuple(local)} cannot construct a Torch level")
    return effective_min_size, extents


def _median_mad(values: Sequence[float]) -> tuple[float, float]:
    samples = np.asarray(values, dtype=np.float64)
    median = float(np.median(samples))
    mad = float(np.median(np.abs(samples - median)))
    return median, mad


def _torch_solution_parity(
        solution, local_lattice: Sequence[int]) -> np.ndarray:
    """Extract the canonical parity field without entering MPI."""
    solution_local = solution.reshape(
        4, 3, *local_lattice).detach().cpu().numpy()
    parity_shape = (*local_lattice[:3], local_lattice[3] // 2)
    return canonical_fermion(solution_local).reshape(
        2, 12, *parity_shape)


def _torch_true_residual(
        solution, source_parity: np.ndarray, gauge_parity: np.ndarray,
        clover_parity: np.ndarray, kappa: float, comm: MPI.Comm,
        coordinate: Sequence[int], process_grid: Sequence[int],
        local_lattice: Sequence[int],
        solution_parity: np.ndarray | None = None) -> float:
    """Evaluate the rank-local solution with an independent NumPy operator."""
    from pyqcu.testing.simple.kupl_residual import (
        distributed_relative_residual,
    )

    if solution_parity is None:
        solution_parity = _torch_solution_parity(solution, local_lattice)
    return distributed_relative_residual(
        solution_parity, source_parity, gauge_parity,
        clover_parity[0], clover_parity[1], kappa, comm,
        tuple(coordinate), tuple(process_grid))


def _run_torch(args, comm: MPI.Comm) -> int:
    import pyqcu.cann as _torch
    from pyqcu import tools
    from pyqcu.torch import dslash
    from pyqcu.torch.solver._legacy_multigrid import LegacyMultigrid

    rank = int(comm.Get_rank())
    size = int(comm.Get_size())
    selected_grid = tuple(int(value) for value in args.grid)
    os.environ["PYQCU_PROCESS_GRID"] = ",".join(
        str(value) for value in selected_grid)
    setup_started = time.perf_counter()

    def initialize_runtime():
        expected_grid = tuple(int(value) for value in tools.give_grid_size())
        if selected_grid != expected_grid:
            raise ValueError(
                "Torch MPI process-grid environment does not match the worker "
                f"grid: expected={expected_grid}, selected={selected_grid}")
        local_device, local_device_index = _torch_device(args, comm)
        if local_device_index is None:
            _torch.set_num_threads(args.omp_threads)
        local_lattice, starts, coordinate = local_geometry(
            args.lat, args.grid, rank)
        return (
            local_device, local_device_index, local_lattice, starts,
            coordinate, _torch.get_num_threads())

    (device, device_index, local_lattice, starts, coordinate,
     torch_num_threads) = _collective_call(
         comm, "Torch runtime initialization", initialize_runtime)
    tensor_dtype = _torch.complex64 if args.dtype == "c64" else _torch.complex128
    real_dtype = _torch.float32 if args.dtype == "c64" else _torch.float64
    numpy_dtype = np.dtype(np.complex64 if args.dtype == "c64" else np.complex128)

    def prepare_fields():
        gauge_full, _, _ = read_gauge(args.gauge, args.lat, numpy_dtype)
        source_full, _, _ = read_source(args.source, args.lat, numpy_dtype)
        gauge_local = slice_last4(gauge_full, starts, local_lattice)
        source_local = slice_last4(source_full, starts, local_lattice)
        gauge_tensor = _torch.as_tensor(
            gauge_local, dtype=tensor_dtype, device=device)
        source_tensor = _torch.as_tensor(
            source_local, dtype=tensor_dtype, device=device)
        kappa_tensor = _torch.tensor(
            [1.0 / (2.0 * args.mass + 8.0)], dtype=real_dtype,
            device=device)
        u_0 = _torch.tensor([1.0], dtype=real_dtype, device=device)
        kappa = float(kappa_tensor.item())
        source_parity = None
        gauge_parity = None
        if not args.skip_independent_residual:
            parity_shape = (*local_lattice[:3], local_lattice[3] // 2)
            source_parity = canonical_fermion(source_local).reshape(
                2, 12, *parity_shape)
            gauge_parity = canonical_gauge(gauge_local)
        return (gauge_tensor, source_tensor, kappa_tensor, u_0, kappa,
                source_parity, gauge_parity)

    (gauge, source, kappa_tensor, u_0, kappa, source_parity,
     gauge_parity) = _collective_call(
         comm, "Torch local field setup", prepare_fields)
    clover = dslash.make_clover(
        gauge, kappa=kappa_tensor, u_0=u_0,
        support_parallel=True, verbose=False, csw=args.csw)

    def prepare_clover_parity():
        if args.skip_independent_residual:
            return None
        clover_onsite = dslash.add_I(clover)
        return tools.oooxyzt2poooxyzt(
            clover_onsite.reshape(12, 12, *local_lattice)
        ).detach().cpu().numpy()

    clover_parity = _collective_call(
        comm, "Torch local Clover parity setup", prepare_clover_parity)

    def hierarchy_layout():
        effective_min_size, level_extents = _torch_hierarchy_extents(
            local_lattice, args.levels, args.min_size, args.mg_grid)
        actual_levels = len(level_extents)
        return (
            effective_min_size, level_extents, actual_levels,
            actual_levels > 1 and args.restart != 0)

    (effective_min_size, level_extents, actual_levels,
     use_coarse) = _collective_call(
         comm, "Torch hierarchy validation", hierarchy_layout)

    solver = None

    def construct_solver():
        nonlocal solver
        coarse_assets = []
        if use_coarse:
            cache_paths = [args.coarse_cache, *args.coarse_cache_level]
            if len(cache_paths) < actual_levels - 1:
                raise ValueError(
                    f"levels={actual_levels} requires {actual_levels - 1} "
                    f"coarse caches, got {len(cache_paths)}")
            coarse_local = [
                int(local_lattice[0]), int(local_lattice[1]),
                int(local_lattice[2]), int(local_lattice[3]) // 2]
            coarse_starts = [
                int(starts[0]), int(starts[1]), int(starts[2]),
                int(starts[3]) // 2]
            for level_index, cache_path in enumerate(
                    cache_paths[:actual_levels - 1], start=1):
                if not cache_path:
                    raise ValueError(
                        f"Torch multigrid level {level_index} requires a cache")
                coarse_local = [
                    coarse_local[axis] // args.mg_grid[axis]
                    for axis in range(4)]
                coarse_starts = [
                    coarse_starts[axis] // args.mg_grid[axis]
                    for axis in range(4)]
                values = load_coarse_assets(
                    cache_path, numpy_dtype, starts=tuple(coarse_starts),
                    local_shape=tuple(coarse_local))
                coarse_assets.append({
                    name: _torch.as_tensor(
                        value, dtype=tensor_dtype, device=device)
                    for name, value in zip(
                        ("lonv", "hnn", "hdg", "sit"), values)
                })
        solver = LegacyMultigrid(
            dtype_list=[tensor_dtype] * actual_levels,
            device_list=[device] * actual_levels,
            U=gauge,
            clover_term=clover,
            kappa=kappa_tensor,
            u_0=u_0,
            max_level=actual_levels,
            min_size=effective_min_size,
            mg_grid_size=list(args.mg_grid),
            dof_list=list(args.dof[:actual_levels]),
            tol=args.atol,
            max_iter=args.max_iter,
            num_restart=args.restart,
            root=0,
            support_parity=True,
            verbose=bool(args.verbose),
            seed=args.seed,
            coarse_assets=coarse_assets if use_coarse else None,
            coarse_tol=args.atol * args.coarse_tol_factor,
            coarse_max_iter=args.coarse_max_iter,
            deflate=use_coarse,
            process_grid=selected_grid,
        )
        return solver

    try:
        _collective_call(comm, "Torch solver construction", construct_solver)
        assert solver is not None
        solver.setup()

        def synchronize_setup():
            if device_index is not None:
                _torch.cuda.synchronize(device)

        _collective_call(comm, "Torch hierarchy device sync", synchronize_setup)
        comm.Barrier()
        setup_seconds = float(comm.allreduce(
            time.perf_counter() - setup_started, op=MPI.MAX))

        def solve_once(label: str):
            def synchronize_before_solve():
                if device_index is not None:
                    _torch.cuda.synchronize(device)

            _collective_call(
                comm, f"{label} device preflight", synchronize_before_solve)
            comm.Barrier()
            started = time.perf_counter()

            def execute():
                result = solver.solve(source)
                if device_index is not None:
                    _torch.cuda.synchronize(device)
                return result

            result = _collective_call(comm, label, execute)
            local_seconds = time.perf_counter() - started
            finite_local = bool(_torch.isfinite(result).all().item())
            finite = bool(comm.allreduce(int(finite_local), op=MPI.MIN))
            stop_reasons = comm.allgather(
                solver.last_solve_report.get("stop_reason"))
            iterations = comm.allgather(
                int(solver.last_solve_report.get("iterations", 0)))
            if len(set(stop_reasons)) != 1 or len(set(iterations)) != 1:
                raise RuntimeError(
                    "Torch ranks diverged in solve control flow: "
                    f"stop_reasons={stop_reasons}, iterations={iterations}")
            stop_reason = stop_reasons[0]
            native_ok = finite and stop_reason in (
                "full_residual", "native_tolerance")
            if not native_ok:
                raise RuntimeError(
                    f"{label} did not converge: finite={finite}, "
                    f"stop_reason={stop_reason}, iterations={iterations[0]}")
            utility = float(comm.allreduce(local_seconds, op=MPI.MAX))
            return result, utility, stop_reason, iterations[0]

        for warmup_index in range(args.warmup):
            solve_once(f"Torch warmup {warmup_index + 1}")

        timed_samples = []
        solution = None
        for sample_index in range(args.repeat):
            solution, solve_seconds, stop_reason, iterations = solve_once(
                f"Torch timed solve {sample_index + 1}")
            if args.skip_independent_residual:
                residual = None
                converged = True
                acceptance = float(args.atol)
                residual_seconds = None
            else:
                def prepare_residual_solution():
                    return _torch_solution_parity(solution, local_lattice)

                prepared_solution = _collective_call(
                    comm,
                    f"Torch residual solution preparation {sample_index + 1}",
                    prepare_residual_solution)

                def evaluate_residual():
                    started = time.perf_counter()
                    value = _torch_true_residual(
                        solution, source_parity, gauge_parity, clover_parity,
                        kappa, comm, coordinate, selected_grid, local_lattice,
                        solution_parity=prepared_solution)
                    return value, time.perf_counter() - started

                residual, local_residual_seconds = _collective_call(
                    comm, f"Torch independent residual {sample_index + 1}",
                    evaluate_residual)
                residual_seconds = float(comm.allreduce(
                    local_residual_seconds, op=MPI.MAX))
                converged, residual_status, acceptance = convergence_status(
                    residual, args.atol, dtype=numpy_dtype)
                if residual_status == "FAIL":
                    raise RuntimeError(
                        f"independent residual failed: {residual:.6e} > "
                        f"{acceptance:.6e}")
            timed_samples.append({
                "sample": sample_index + 1,
                "iterations": iterations,
                "stop_reason": stop_reason,
                "solve_seconds": solve_seconds,
                "residual_seconds": residual_seconds,
                "true_residual": residual,
                "residual_acceptance": acceptance,
                "converged": converged,
            })

        solve_median, solve_mad = _median_mad(
            [sample["solve_seconds"] for sample in timed_samples])
        residual_values = [
            sample["true_residual"] for sample in timed_samples
            if sample["true_residual"] is not None]
        residual_times = [
            sample["residual_seconds"] for sample in timed_samples
            if sample["residual_seconds"] is not None]
        residual_median = (
            _median_mad(residual_values)[0] if residual_values else None)
        residual_time_median, residual_time_mad = (
            _median_mad(residual_times) if residual_times else (None, None))
        iterations = [sample["iterations"] for sample in timed_samples]

        assert solution is not None
        def prepare_local_solution():
            return solution.reshape(
                4, 3, *local_lattice).detach().cpu().numpy()

        local_solution = _collective_call(
            comm, "Torch local solution extraction", prepare_local_solution)
        parts = comm.gather((tuple(coordinate), local_solution), root=0)
        _collective_call(
            comm, "Torch rank-0 HDF5 output",
            lambda: _torch_publish_solution(
                args, rank, local_lattice, mpi_size=size,
                torch_num_threads=torch_num_threads, kappa=kappa, parts=parts))

        converged = all(sample["converged"] for sample in timed_samples)
        failed = int(comm.allreduce(int(not converged), op=MPI.MAX))
        status = "FAIL" if failed else "PASS"
        local_sample_times = [
            sample["solve_seconds"] for sample in timed_samples]
        local_median, local_mad = _median_mad(local_sample_times)
        local_record = {
            "rank": rank,
            "coordinate": list(coordinate),
            "device": str(device),
            "device_index": device_index,
            "local_lattice": list(local_lattice),
            "local_starts": list(starts),
            "solve_seconds_median": local_median,
            "solve_seconds_mad": local_mad,
            "affinity_cpus": _affinity(),
            "torch_num_threads": torch_num_threads,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        }
        records = comm.gather(local_record, root=0)
        solve_report = dict(solver.last_solve_report)
        report = {
            "status": status,
            "backend": "torch",
            "solver": "LegacyMultigrid",
            "lattice": list(args.lat),
            "grid": list(args.grid),
            "mpi_size": size,
            "topology": {
                "policy": "one Python process per MPI rank",
                "device_type": device.type,
                "torch_num_threads": torch_num_threads,
                "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            },
            "dtype": args.dtype,
            "mass": args.mass,
            "kappa": kappa,
            "csw": args.csw,
            "atol": args.atol,
            "requested_levels": args.levels,
            "levels": actual_levels,
            "level_extents": level_extents,
            "requested_min_size": args.min_size,
            "effective_min_size": effective_min_size,
            "dof": list(args.dof[:actual_levels]),
            "mg_grid": list(args.mg_grid),
            "restart": args.restart,
            "coarse_cache": args.coarse_cache,
            "warmup_runs": args.warmup,
            "timed_samples": timed_samples,
            "setup_seconds": setup_seconds,
            "wall_seconds": solve_median,
            "timing_summary": {
                "solve_seconds": {
                    "samples": [sample["solve_seconds"]
                                for sample in timed_samples],
                    "median": solve_median,
                    "mad": solve_mad,
                },
                "residual_seconds": {
                    "samples": residual_times,
                    "median": residual_time_median,
                    "mad": residual_time_mad,
                },
            },
            "final_true_residual": (
                max(residual_values) if residual_values else None),
            "true_residual_median": residual_median,
            "residual_acceptance": max(
                sample["residual_acceptance"] for sample in timed_samples),
            "residual_source": (
                "independent distributed NumPy Wilson-Clover reconstruction "
                "with MPI halo exchange" if residual_values else
                "native solve only; independent check skipped"),
            "iterations": iterations[-1],
            "iterations_per_sample": iterations,
            "converged": converged and status == "PASS",
            "stop_reason": timed_samples[-1]["stop_reason"],
            "solver_report": solve_report,
            "ranks": records,
        }
        if rank == 0:
            write_json(args.report, report)
            print(json.dumps(report, ensure_ascii=False), flush=True)
        return 0 if status == "PASS" else 1
    finally:
        if solver is not None:
            solver.close()


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("kupl", "cuda", "torch"),
                        required=True)
    parser.add_argument("--torch-device", choices=("auto", "cpu", "cuda"),
                        default="auto")
    parser.add_argument("--kupl-legacy-layout",
                        choices=("auto", "odd-schur", "full-site"),
                        default="auto")
    parser.add_argument("--gauge", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--lat", nargs=4, type=int, required=True)
    parser.add_argument("--grid", nargs=4, type=int, required=True)
    parser.add_argument("--mass", type=float, required=True)
    parser.add_argument("--csw", type=float, required=True)
    parser.add_argument("--atol", type=float, default=1.0e-6)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument(
        "--omp-threads", type=int,
        default=int(os.environ.get("OMP_NUM_THREADS", "1")))
    parser.add_argument("--dtype", choices=("c64", "c128"), default="c64")
    parser.add_argument("--levels", type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument("--min-size", type=int, default=4)
    parser.add_argument("--dof", nargs="+", type=int, required=True)
    parser.add_argument("--mg-grid", nargs=4, type=int, required=True)
    parser.add_argument("--restart", type=int, default=5)
    parser.add_argument("--coarse-max-iter", type=int, default=200)
    parser.add_argument("--coarse-tol-factor", type=float, default=1.0e4)
    parser.add_argument("--nv-iters", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--coarse-cache", default=None)
    parser.add_argument("--coarse-cache-level", action="append", default=[])
    parser.add_argument("--skip-independent-residual", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv)
    args.lat = list(validate_lattice(args.lat))
    if len(args.dof) < args.levels:
        parser.error("--dof must provide one value per requested level")
    if args.min_size < 2:
        parser.error("--min-size must be at least 2")
    if args.omp_threads < 1:
        parser.error("--omp-threads must be positive")
    if args.warmup < 0 or args.repeat < 1:
        parser.error("--warmup must be nonnegative and --repeat must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.backend == "kupl" and args.levels > 2:
        raise ValueError("the current KUPL Legacy benchmark supports at most 2 levels")
    comm = MPI.COMM_WORLD
    try:
        validate_rank_count(int(comm.Get_size()))
    except ValueError as exc:
        if int(comm.Get_rank()) == 0:
            print(f"error: {exc}", file=sys.stderr)
        return 2
    if int(np.prod(args.grid)) != int(comm.Get_size()):
        raise ValueError(
            f"grid {tuple(args.grid)} contains {int(np.prod(args.grid))} ranks, "
            f"MPI world has {int(comm.Get_size())}")
    local_geometry(args.lat, args.grid, int(comm.Get_rank()))
    if args.backend == "kupl":
        return _run_kupl(args, comm)
    if args.backend == "cuda":
        return _run_cuda(args, comm)
    return _run_torch(args, comm)


if __name__ == "__main__":
    exit_code = main()
    if not MPI.Is_finalized():
        MPI.Finalize()
    raise SystemExit(exit_code)
