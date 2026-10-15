#!/usr/bin/env python3
"""Minimal QCU Clover multigrid example with explicit legacy MG parameters."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

try:
    from ._common import (DATA_DIR, DEFAULT_ATOL, DEFAULT_MASS,
                          cuda_memory_snapshot, estimated_clover_gflops,
                          resolve_path, write_h5, write_json)
    from .qcu_bistabcg import prepare_case, qcu_available, true_residual
except ImportError:
    from _common import (DATA_DIR, DEFAULT_ATOL, DEFAULT_MASS,
                         cuda_memory_snapshot, estimated_clover_gflops,
                         resolve_path, write_h5, write_json)
    from qcu_bistabcg import prepare_case, qcu_available, true_residual

if str(Path(__file__).resolve().parents[3]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def configure_mg(params, argv, define, lat, levels, dof_list, mg_grid,
                 atol, restart, coarse_max_iter, coarse_tol_factor):
    params[define._MG_NUM_LEVEL_] = int(levels)
    if levels >= 2:
        params[define._MG_LEVEL1_E_] = int(dof_list[1])
        params[define._MG_LEVEL1_X_] = int(lat[0] // mg_grid[0])
        params[define._MG_LEVEL1_Y_] = int(lat[1] // mg_grid[1])
        params[define._MG_LEVEL1_Z_] = int(lat[2] // mg_grid[2])
        params[define._MG_LEVEL1_T_] = int(lat[3] // (2 * mg_grid[3]))
        params[define._MG_LEVEL1_MAX_ITER_] = int(coarse_max_iter)
        params[define._MG_LEVEL1_DATA_TYPE_] = int(params[define._DATA_TYPE_])
        params[define._MG_LEVEL1_NUM_RESTART_] = int(restart)
        argv[define._MG_LEVEL1_ATOL_] = float(atol * coarse_tol_factor)
    if levels >= 3:
        params[define._MG_LEVEL2_E_] = int(dof_list[2])
        params[define._MG_LEVEL2_X_] = int(lat[0] // (mg_grid[0] ** 2))
        params[define._MG_LEVEL2_Y_] = int(lat[1] // (mg_grid[1] ** 2))
        params[define._MG_LEVEL2_Z_] = int(lat[2] // (mg_grid[2] ** 2))
        params[define._MG_LEVEL2_T_] = int(lat[3] // (4 * mg_grid[3]))
        params[define._MG_LEVEL2_MAX_ITER_] = int(coarse_max_iter)
        params[define._MG_LEVEL2_DATA_TYPE_] = int(params[define._DATA_TYPE_])
        params[define._MG_LEVEL2_NUM_RESTART_] = int(max(2, restart // 2))
        argv[define._MG_LEVEL2_ATOL_] = float(atol * coarse_tol_factor * 3.0)


def parse_native_mg_log(
        text: str) -> tuple[list[float], list[dict[str, float]], list[float],
                            int | None, dict[str, float]]:
    history: list[float] = []
    outer_iterations: list[dict[str, float]] = []
    vcycles: list[float] = []
    iterations = None
    profile: dict[str, float] = {}
    number = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    for line in text.splitlines():
        match = re.search(r"CONVERGENCE_HISTORY:\s*\[([^\]]*)\]", line)
        if match:
            history = [float(value) for value in match.group(1).split(",") if value.strip()]
        match = re.search(r"\bB-\d+-[^:]+-Iteration\s+(\d+):\s*Residual\s*=\s*" + number, line)
        if match:
            outer_iterations.append({"iteration": int(match.group(1)),
                                     "residual_norm": float(match.group(2))})
        match = re.search(r"Residual\(norm2\):\(([^,]+),", line)
        if match:
            vcycles.append(float(match.group(1)))
        match = re.search(r"Total iterations:\s*(\d+)", line)
        if match:
            iterations = int(match.group(1))
        match = re.search(r"PROF_SECTIONS:\s*(.*)", line)
        if match:
            for token in match.group(1).split():
                if "=" in token:
                    key, value = token.split("=", 1)
                    try:
                        profile[key] = float(value.rstrip("ms"))
                    except ValueError:
                        continue
    return history, outer_iterations, vcycles, iterations, profile


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lat", nargs=4, type=int, default=(8, 8, 8, 16))
    parser.add_argument("--mass", type=float, default=DEFAULT_MASS)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--levels", type=int, default=2)
    parser.add_argument("--dof", nargs="+", type=int, default=(12, 48, 48))
    parser.add_argument("--mg-grid", nargs=4, type=int, default=(2, 2, 2, 2))
    parser.add_argument("--restart", type=int, default=5)
    parser.add_argument("--coarse-max-iter", type=int, default=200)
    parser.add_argument("--coarse-tol-factor", type=float, default=1.0e4)
    parser.add_argument("--nv-iters", type=int, default=2)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gauge", default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--report", default="performance_qcu_multigrid.json")
    args = parser.parse_args(argv)

    if not qcu_available():
        print(json.dumps({"status": "SKIP", "reason": "QCU extension/CUDA unavailable"}))
        return 0
    levels = int(args.levels)
    if levels < 1 or len(args.dof) < levels:
        parser.error("--dof must contain one value per requested level")

    log_dir = DATA_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    os.environ["QCU_LOG_DIR"] = str(log_dir)
    log_path = log_dir / "clover_multigrid.log"
    log_offset = log_path.stat().st_size if log_path.exists() else 0

    ctx, fields, operator, tools, kappa_value = prepare_case(
        tuple(args.lat), args.mass, args.atol, args.max_iter, args.device,
        args.gauge, args.source)
    import torch
    from pyqcu import dslash
    from pyqcu.cuda import define
    from pyqcu.cuda import qcu
    from pyqcu.cuda._multi_gpu import build_schur_levels

    g, src, out, ce, co, cei, coi = fields
    configure_mg(ctx.params, ctx.argv, define, tuple(args.lat), levels,
                 list(args.dof), list(args.mg_grid), args.atol,
                 args.restart, args.coarse_max_iter, args.coarse_tol_factor)
    if levels >= 2:
        u_full = tools.poooxyzt2oooxyzt(g)
        clover_full = dslash.make_clover(
            u_full, kappa=torch.tensor([kappa_value], dtype=torch.float32,
                                      device=u_full.device),
            u_0=torch.ones(1, dtype=torch.float32, device=u_full.device))
        fine = dslash.operator(U=u_full, clover_term=clover_full,
                              kappa=torch.tensor([kappa_value], device=u_full.device),
                              u_0=torch.ones(1, dtype=torch.float32, device=u_full.device),
                              support_parity=True, verbose=False)
        schur = getattr(fine, "matvec_parity", fine.matvec)
        lonvs, hop_nn, hop_diag, sitting = build_schur_levels(
            fine, schur, levels, list(args.dof), list(args.mg_grid),
            list(args.lat), list(args.dof)[1], ctx.dtype, ctx.device,
            nv_iters=args.nv_iters, use_cache=not args.no_cache,
            cache_dir=DATA_DIR / "logs" / "nullvec_cache", verbose=False)
        for level in range(len(lonvs)):
            for slot, tensor in enumerate((lonvs[level], hop_nn[level],
                                           hop_diag[level], sitting[level])):
                ctx.set_ptrs[30 + 4 * level + slot] = tensor.contiguous().data_ptr()

    ctx.params[define._SET_PLAN_] = define._SET_PLAN1_
    ctx.params[define._VERBOSE_] = 1
    memory = None
    qcu.applyInitQcu(ctx.set_ptrs, ctx.params, ctx.argv)
    try:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        qcu.applyCloverMultigridQcu(out, src, g, ce, co, cei, coi,
                                   ctx.set_ptrs, ctx.params)
        end.record()
        torch.cuda.synchronize(ctx.device)
        wall_seconds = start.elapsed_time(end) / 1000.0
        memory = cuda_memory_snapshot(ctx.device)
    finally:
        ctx.params[define._SET_INDEX_] += 1
        qcu.applyEndQcu(ctx.set_ptrs, ctx.params)
    residual = true_residual(fields, operator, tools, kappa_value)
    if log_path.exists():
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(log_offset)
            log_text = handle.read()
    else:
        log_text = ""
    residual_history, outer_iteration_history, vcycle_history, native_iterations, profile = \
        parse_native_mg_log(log_text)
    iterations = native_iterations if native_iterations is not None else len(residual_history)
    if args.output:
        output = resolve_path(args.output, resolve_path(None, args.output))
        write_h5(output, tools.poooxyzt2oooxyzt(out).detach().cpu().numpy(),
                 metadata={"solver": "applyCloverMultigridQcu", "mass": args.mass,
                           "atol": args.atol, "levels": levels,
                           "dof": list(args.dof[:levels]),
                           "mg_grid": list(args.mg_grid)})
    report = {
        "backend": "qcu", "solver": "applyCloverMultigridQcu",
        "lattice": list(args.lat), "mass": args.mass, "kappa": kappa_value,
        "levels": levels, "dof": list(args.dof[:levels]),
        "mg_grid": list(args.mg_grid), "iterations": iterations,
        "wall_seconds": wall_seconds,
        "seconds_per_iteration": wall_seconds / iterations if iterations > 0 else None,
        "final_true_residual": residual,
        "residual_history": residual_history,
        "outer_iteration_residual_history": outer_iteration_history,
        "vcycle_residual_history": vcycle_history,
        "profile_ms": profile,
        "residual_history_source": str(log_path),
        "gflops": estimated_clover_gflops(tuple(args.lat), iterations, wall_seconds),
        "gflops_estimate_note": "lower bound; coarse-grid applications are excluded",
        "memory": memory,
        "status": "PASS" if residual < 1e-2 else "WARN",
    }
    report_path = resolve_path(args.report, DATA_DIR / args.report)
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
