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
    gauge_full, _, _ = read_gauge(args.gauge, args.lat, numpy_dtype)
    source_full, _, _ = read_source(args.source, args.lat, numpy_dtype)
    gauge_local = slice_last4(gauge_full, starts, local_lattice)
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
        coarse_local = list(local_lattice)
        coarse_starts = list(starts)
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
    source_np_result = source.detach().cpu().numpy().reshape(
        2, 12, *parity_shape)
    gauge_np_result = gauge.detach().cpu().numpy()
    clover_even_np = clover_even.detach().cpu().numpy()
    clover_odd_np = clover_odd.detach().cpu().numpy()
    kappa = 1.0 / (2.0 * args.mass + 8.0)

    residual_started = time.perf_counter()
    if args.skip_independent_residual:
        residual = None
        converged = True
        status = "PASS"
        acceptance = float(args.atol)
        residual_source = "native solve only; independent check skipped"
    else:
        residual = distributed_relative_residual(
            solution_np, source_np_result, gauge_np_result,
            clover_even_np, clover_odd_np, kappa,
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


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("kupl", "cuda"), required=True)
    parser.add_argument("--gauge", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--lat", nargs=4, type=int, required=True)
    parser.add_argument("--grid", nargs=4, type=int, required=True)
    parser.add_argument("--mass", type=float, required=True)
    parser.add_argument("--csw", type=float, required=True)
    parser.add_argument("--atol", type=float, default=1.0e-6)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--dtype", choices=("c64", "c128"), default="c64")
    parser.add_argument("--levels", type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument("--dof", nargs="+", type=int, required=True)
    parser.add_argument("--mg-grid", nargs=4, type=int, required=True)
    parser.add_argument("--restart", type=int, default=5)
    parser.add_argument("--coarse-max-iter", type=int, default=200)
    parser.add_argument("--coarse-tol-factor", type=float, default=1.0e4)
    parser.add_argument("--nv-iters", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
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
    return _run_cuda(args, comm)


if __name__ == "__main__":
    exit_code = main()
    if not MPI.Is_finalized():
        MPI.Finalize()
    raise SystemExit(exit_code)
