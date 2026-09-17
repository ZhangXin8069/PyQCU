#!/usr/bin/env python3
"""Small multi-rank strict-MG solve probe with an independent true residual.

The fine operator is a distributed Wilson/Clover operator.  Setup uses the
column-wise strict Galerkin builder so every call to the fine operator keeps
the existing MPI halo protocol.  The final residual is recomputed through the
Python dslash operator and global norm, independently of the C++ FGMRES
reported residual.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from mpi4py import MPI

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import grid_index_for_rank, make_clover_tensors, process_grid
from pyqcu import dslash, lattice, tools
from pyqcu.cuda import CudaStrictMultigridSolver, define, qcu
from pyqcu.solver import Checkerboard as QudaCheckerboard
from pyqcu.solver import QudaStrictMultigrid


def _slice_local(tensor: torch.Tensor, starts, local_shape):
    slices = [slice(None)] * (tensor.ndim - 4)
    slices.extend(
        slice(int(starts[axis]), int(starts[axis] + local_shape[axis]))
        for axis in range(4)
    )
    return tensor[tuple(slices)].contiguous()


def _parity_gauge(full_gauge: torch.Tensor) -> torch.Tensor:
    X, Y, Z, T = (int(value) for value in full_gauge.shape[-4:])
    result = torch.empty(2, 3, 3, 4, X, Y, Z, T // 2,
                         dtype=full_gauge.dtype)
    for parity in range(2):
        for x in range(X):
            for y in range(Y):
                for z in range(Z):
                    for tc in range(T // 2):
                        t = 2 * tc + ((parity - x - y - z) & 1)
                        result[parity, ..., x, y, z, tc] = (
                            full_gauge[..., x, y, z, t])
    return result.contiguous()


def _localise_compact(
        value: torch.Tensor, starts, local_shape, parity: int) -> torch.Tensor:
    x, y, z = local_shape[:3]
    th = local_shape[3] // 2
    local = torch.empty(
        (*value.shape[:-4], x, y, z, th), dtype=value.dtype,
        device=value.device)
    for xl in range(x):
        for yl in range(y):
            for zl in range(z):
                t_parity = (parity - xl - yl - zl) & 1
                for tc in range(th):
                    global_t = starts[3] + 2 * tc + t_parity
                    local[(..., xl, yl, zl, tc)] = value[
                        (..., starts[0] + xl, starts[1] + yl,
                         starts[2] + zl, global_t)]
    return local.contiguous()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=int, nargs=4, default=[4, 4, 4, 4])
    parser.add_argument("--grid", type=int, nargs=4, default=[2, 1, 1, 1])
    parser.add_argument("--mass", type=float, default=0.1)
    parser.add_argument("--tol", type=float, default=1.0e-6)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--restart", type=int, default=4)
    parser.add_argument("--probe-vcycle", action="store_true")
    parser.add_argument("--probe-fine-matpc", action="store_true")
    parser.add_argument("--probe-wilson", action="store_true")
    parser.add_argument(
        "--wilson-direction", choices=("x", "y", "z", "t", "all"),
        default="all")
    parser.add_argument("--delta-source", action="store_true")
    parser.add_argument("--delta-coordinate", type=int, nargs=4)
    parser.add_argument("--random-gauge", action="store_true")
    parser.add_argument("--dtype", choices=("c64", "c128"), default="c64")
    parser.add_argument(
        "--galerkin-mode", default="column",
        choices=("column", "site-batch", "colored", "auto"),
        help="strict Galerkin setup mode; distributed runs should avoid the "
             "per-column probe mode, which is O(volume) MPI exchanges")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    grid = process_grid(args.grid)
    if int(np.prod(grid)) != comm.Get_size():
        raise ValueError(f"grid={grid} does not match MPI size={comm.Get_size()}")
    if any(args.shape[axis] % grid[axis] for axis in range(4)):
        raise ValueError(f"shape={args.shape} is not divisible by grid={grid}")
    local_shape = tuple(
        args.shape[axis] // grid[axis] for axis in range(4))
    if any(extent % 2 for extent in local_shape):
        raise ValueError(f"local shape={local_shape} must be even")
    coordinate = grid_index_for_rank(grid, rank)
    starts = [coordinate[axis] * local_shape[axis] for axis in range(4)]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for strict MPI solve probe")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    if args.dtype == "c128":
        dtype = torch.complex128
        data_type = define._LAT_C128_
    else:
        dtype = torch.complex64
        data_type = define._LAT_C64_

    identity = torch.zeros(3, 3, 4, *args.shape, dtype=dtype)
    for color in range(3):
        identity[color, color] = 1.0
    if args.wilson_direction != "all":
        keep = "xyzt".index(args.wilson_direction)
        mask = torch.zeros(4, dtype=torch.bool)
        mask[keep] = True
        identity[:, :, ~mask] = 0.0
    local_full_gauge = _slice_local(
        identity, starts, local_shape).to(device).contiguous()
    local_parity_gauge = _parity_gauge(local_full_gauge).to(device)

    ce, cei, coo, coi, _unused_s, params, argv = make_clover_tensors(
        local_parity_gauge, local_shape, mass=args.mass, grid=grid,
        rank=rank, dtype=dtype, data_type=data_type)
    params[define._MAX_ITER_] = int(args.max_iter)
    argv[define._ATOL_] = float(args.tol)

    torch.manual_seed(20260917 + args.shape[0])
    global_null = torch.randn(
        2, 4, 3, *args.shape, dtype=dtype)
    local_null = _slice_local(
        global_null, starts, local_shape).to(device).contiguous()
    scalar_dtype = torch.float64 if args.dtype == "c128" else torch.float32
    kappa = torch.tensor(
        [1.0 / (2.0 * float(args.mass) + 8.0)], dtype=scalar_dtype,
        device=device)
    hierarchy = QudaStrictMultigrid(
        U=local_full_gauge,
        clover_term=None,
        kappa=kappa,
        u_0=torch.ones(1, dtype=scalar_dtype, device=device),
        lat_size=local_shape,
        null_vectors=[local_null],
        nvec_list=[2],
        dof_list=[12, 4],
        block_size=(2, 2, 2, 2),
        max_level=2,
        materialize_coarse=True,
        use_parity=True,
        target_parity=1,
        nu_pre=1,
        nu_post=1,
        coarse_max_iter=int(args.max_iter),
        coarse_tol=float(args.tol),
        restart=int(args.restart),
        max_iter=int(args.max_iter),
        tol=float(args.tol),
        setup_iters=0,
        strict_galerkin_mode=args.galerkin_mode,
        process_grid=(grid if comm.Get_size() > 1 else None),
        comm=(comm if comm.Get_size() > 1 else None),
        verbose=False,
    )
    hierarchy.setup()
    diagnostics = hierarchy.diagnostics()
    print(
        {"rank": rank, "strict_diagnostics": diagnostics},
        flush=True)

    rhs = torch.randn(
        2, 4, 3, *local_shape[:3], local_shape[3] // 2,
        dtype=dtype, device=device)
    solver = CudaStrictMultigridSolver(
        hierarchy, argv, local_parity_gauge, ce, coo, cei, coi, params,
        restart=int(args.restart), max_krylov_bytes=256 << 20,
        release_setup_assets=False, verbose=False)
    try:
        if args.probe_wilson:
            torch.manual_seed(20260920)
            global_compact = torch.randn(
                12, *args.shape[:3], args.shape[3] // 2,
                dtype=dtype)
            if args.delta_source:
                global_compact.zero_()
                coordinate = (
                    list(args.delta_coordinate)
                    if args.delta_coordinate is not None else [0, 0, 0, 0])
                if (args.delta_coordinate is None and
                        args.wilson_direction in "xyzt"):
                    coordinate["xyzt".index(args.wilson_direction)] = 1
                global_compact[
                    0, coordinate[0], coordinate[1],
                    coordinate[2], coordinate[3]] = 1.0
            source_parity = 1 - hierarchy.target_parity
            global_parity = torch.zeros(
                2, 12, *args.shape[:3], args.shape[3] // 2,
                dtype=dtype)
            global_parity[source_parity] = global_compact
            global_full_input = tools.poooxyzt2oooxyzt(global_parity)
            compact = _localise_compact(
                global_full_input, starts, local_shape, source_parity)
            compact = compact.to(device)
            compact_parity = torch.zeros(
                2, 12, *local_shape[:3], local_shape[3] // 2,
                dtype=dtype, device=device)
            compact_parity[source_parity] = compact
            compact_full = tools.poooxyzt2oooxyzt(compact_parity)
            expected_local_full = global_full_input[
                ..., starts[0]:starts[0] + local_shape[0],
                starts[1]:starts[1] + local_shape[1],
                starts[2]:starts[2] + local_shape[2],
                starts[3]:starts[3] + local_shape[3]].to(device)
            print(
                {"rank": rank,
                 "input_slice_max_abs": float(
                     (compact_full - expected_local_full).abs().max())},
                flush=True)
            global_reference_gauge = torch.zeros(
                3, 3, 4, *args.shape, dtype=dtype)
            for color in range(3):
                global_reference_gauge[color, color] = 1.0
            if args.random_gauge:
                global_reference_gauge = lattice.generate_gauge_field(
                    global_reference_gauge, sigma=0.1, seed=20260921,
                    verbose=False)
                local_reference_gauge = _slice_local(
                    global_reference_gauge, starts, local_shape).to(device)
                wilson_gauge = _parity_gauge(local_reference_gauge)
            else:
                wilson_gauge = local_parity_gauge.clone()
            if args.wilson_direction != "all":
                keep = "xyzt".index(args.wilson_direction)
                mask = torch.zeros(4, dtype=torch.bool)
                mask[keep] = True
                wilson_gauge[:, :, :, ~mask] = 0.0
            cpp_out = torch.empty_like(compact)
            wilson_params = solver.params.clone()
            wilson_params[define._SET_PLAN_] = define._SET_PLAN0_
            wilson_params[define._PARITY_] = hierarchy.target_parity
            qcu.applyWilsonDslashQcu(
                cpp_out, compact, wilson_gauge,
                solver.set_ptrs, wilson_params)
            if args.wilson_direction != "all":
                keep = "xyzt".index(args.wilson_direction)
                mask = torch.zeros(4, dtype=torch.bool)
                mask[keep] = True
                global_reference_gauge[:, :, ~mask] = 0.0
                wilson_gauge[:, :, :, ~mask] = 0.0
            hopping_full = dslash.give_wilson(
                global_full_input.reshape(4, 3, *args.shape),
                global_reference_gauge,
                kappa=kappa.cpu(),
                u_0=torch.ones(1, dtype=torch.float32),
                with_I=False, verbose=False)
            hopping_parity = tools.oooxyzt2poooxyzt(
                (-hopping_full / kappa.reshape(-1)[0].cpu()).reshape(
                    12, *args.shape))
            target_full_parity = torch.zeros_like(hopping_parity)
            target_full_parity[hierarchy.target_parity] = (
                hopping_parity[hierarchy.target_parity])
            target_full = tools.poooxyzt2oooxyzt(target_full_parity)
            hopping_compact = _localise_compact(
                target_full, starts, local_shape,
                hierarchy.target_parity).to(device)
            torch.cuda.synchronize()
            relative = float(tools.norm(cpp_out - hopping_compact)) / max(
                float(tools.norm(hopping_compact)), 1.0e-30)
            difference = (cpp_out - hopping_compact).abs()
            component_max = difference.reshape(
                difference.shape[0], -1).max(dim=1).values
            core = difference
            for axis in range(4):
                extent = core.shape[axis + 1]
                if extent > 2:
                    core = core.narrow(axis + 1, 1, extent - 2)
            print(
                {"rank": rank,
                 "component_max": [float(value) for value in component_max],
                 "core_max": float(core.max()) if core.numel() else 0.0},
                flush=True)
            if args.delta_source:
                actual_nonzero = torch.nonzero(
                    cpp_out.abs() > 1.0e-6, as_tuple=False).cpu()
                expected_nonzero = torch.nonzero(
                    hopping_compact.abs() > 1.0e-6,
                    as_tuple=False).cpu()
                print(
                    {"rank": rank,
                     "actual_nonzero": actual_nonzero.tolist(),
                     "expected_nonzero": expected_nonzero.tolist(),
                     "missing_value": complex(cpp_out[9, 1, 3, 0, 0]),
                     "expected_missing_value": complex(
                         hopping_compact[9, 1, 3, 0, 0])},
                    flush=True)
            for axis, name in enumerate("xyzt"):
                extent = difference.shape[axis + 1]
                interior = difference.narrow(
                    axis + 1, 1, extent - 2).max().item() if extent > 2 else 0.0
                print(
                    f"wilson rank={rank} axis={name} "
                    f"low={difference.select(axis + 1, 0).max().item():.6e} "
                    f"high={difference.select(axis + 1, -1).max().item():.6e} "
                    f"interior={interior:.6e}",
                    flush=True)
            payload = {"rank": rank, "wilson_relative": relative}
            print(payload, flush=True)
            gathered = comm.gather(payload, root=0)
            if rank == 0:
                print({"rank": 0, "wilson": gathered}, flush=True)
            return 0
        if args.probe_fine_matpc:
            torch.manual_seed(20260919)
            global_compact = torch.randn(
                12, *args.shape[:3], args.shape[3] // 2,
                dtype=dtype)
            global_parity = torch.zeros(
                2, 12, *args.shape[:3], args.shape[3] // 2,
                dtype=dtype)
            global_parity[hierarchy.target_parity] = global_compact
            global_full_input = tools.poooxyzt2oooxyzt(global_parity)
            compact = _localise_compact(
                global_full_input, starts, local_shape,
                hierarchy.target_parity).to(device)
            cpp_out = torch.empty_like(compact)
            qcu.applyMultigridStrictFineMatPCQcu(
                cpp_out, compact, local_parity_gauge,
                ce, coo, cei, coi, solver.set_ptrs, solver.params, 1)
            global_identity = torch.zeros(
                3, 3, 4, *args.shape, dtype=dtype)
            for color in range(3):
                global_identity[color, color] = 1.0
            layout = QudaCheckerboard(tuple(args.shape))
            target = global_compact.reshape(12, -1)
            target_full = layout.embed(
                target, hierarchy.target_parity, 12).reshape(
                    12, *args.shape)
            first = dslash.give_wilson(
                target_full.reshape(4, 3, *args.shape),
                global_identity, kappa=kappa.cpu(),
                u_0=torch.ones(1, dtype=torch.float32),
                with_I=False, verbose=False).reshape(12, *args.shape)
            first = (-first / kappa.reshape(-1)[0].cpu())
            other = layout.extract(
                first, 1 - hierarchy.target_parity)
            second_full = layout.embed(
                other, 1 - hierarchy.target_parity, 12).reshape(
                    12, *args.shape)
            second = dslash.give_wilson(
                second_full.reshape(4, 3, *args.shape),
                global_identity, kappa=kappa.cpu(),
                u_0=torch.ones(1, dtype=torch.float32),
                with_I=False, verbose=False).reshape(12, *args.shape)
            second = (-second / kappa.reshape(-1)[0].cpu())
            expected_global = (
                target - kappa.reshape(-1)[0].cpu()
                * kappa.reshape(-1)[0].cpu()
                * layout.extract(second, hierarchy.target_parity).reshape_as(target))
            expected_full_parity = torch.zeros_like(global_parity)
            expected_full_parity[hierarchy.target_parity] = (
                expected_global.reshape(
                    12, *args.shape[:3], args.shape[3] // 2))
            expected_full = tools.poooxyzt2oooxyzt(expected_full_parity)
            python_out = _localise_compact(
                expected_full, starts, local_shape,
                hierarchy.target_parity).to(device)
            torch.cuda.synchronize()
            relative = float(tools.norm(cpp_out - python_out)) / max(
                float(tools.norm(python_out)), 1.0e-30)
            payload = {
                "rank": rank,
                "fine_matpc_relative": relative,
            }
            print(payload, flush=True)
            gathered = comm.gather(payload, root=0)
            if rank == 0:
                print({"rank": 0, "fine_matpc": gathered}, flush=True)
            return 0
        if args.probe_vcycle:
            torch.manual_seed(20260918)
            coarse = hierarchy.operators[1]
            rhs_vcycle = torch.randn(
                coarse.dof, *coarse.shape, dtype=dtype, device=device)
            cpp_out = torch.empty_like(rhs_vcycle)
            qcu.applyMultigridStrictVCycleQcu(
                cpp_out, rhs_vcycle, solver.set_ptrs, solver.params, 1)
            python_out = hierarchy.v_cycle(rhs_vcycle, level=1)
            torch.cuda.synchronize()
            difference = cpp_out - python_out
            relative = float(tools.norm(difference)) / max(
                float(tools.norm(python_out)), 1.0e-30)
            payload = {
                "rank": rank,
                "vcycle_relative": relative,
            }
            print(payload, flush=True)
            gathered = comm.gather(payload, root=0)
            if rank == 0:
                print({"rank": 0, "vcycle": gathered}, flush=True)
            return 0
        solution = solver.solve(rhs)
        torch.cuda.synchronize()
        full_solution = tools.poooxyzt2oooxyzt(solution).contiguous()
        full_rhs = tools.poooxyzt2oooxyzt(rhs).contiguous()
        fine = dslash.operator(
            U=local_full_gauge.cpu().to(dtype=dtype),
            clover_term=None, kappa=kappa.cpu(), support_parity=False,
            verbose=False)
        residual = fine.matvec(full_solution) - full_rhs
        residual_norm = float(tools.norm(residual))
        rhs_norm = float(tools.norm(full_rhs))
        true_relative = residual_norm / max(rhs_norm, 1.0e-30)
        payload = {
            "rank": rank,
            "coordinate": coordinate,
            "converged": bool(solver.converged),
            "outer_iterations": int(solver.iterations),
            "reported_residual": float(solver.final_residual),
            "python_true_relative": true_relative,
        }
        print(payload, flush=True)
        gathered = comm.gather(payload, root=0)
        if rank == 0:
            worst = max(
                item["python_true_relative"] for item in gathered)
            print({"rank": 0, "worst_true_relative": worst}, flush=True)
            if worst >= 5.0e-5:
                raise AssertionError(
                    f"distributed strict solve true residual failed: {gathered}")
    finally:
        solver.close()
    comm.Barrier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
