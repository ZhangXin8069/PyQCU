#!/usr/bin/env python3
"""Minimal QCU Clover BiStabCG example, with explicit ABI lifecycle."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

try:
    from ._common import (DATA_DIR, DEFAULT_ATOL, DEFAULT_MASS,
                          cuda_memory_snapshot, estimated_clover_gflops,
                          lime_metadata, qdp_gauge_to_qcu, read_h5, read_qdp_gauge,
                          resolve_path, write_h5, write_json)
except ImportError:
    from _common import (DATA_DIR, DEFAULT_ATOL, DEFAULT_MASS,
                         cuda_memory_snapshot, estimated_clover_gflops,
                         lime_metadata, qdp_gauge_to_qcu, read_h5, read_qdp_gauge,
                         resolve_path, write_h5, write_json)

if str(Path(__file__).resolve().parents[3]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def qcu_available() -> bool:
    try:
        import torch
        from pyqcu.cuda import qcu
        return torch.cuda.is_available() and hasattr(qcu, "applyCloverBistabCgQcu")
    except Exception:
        return False


def prepare_case(lat, mass, atol, max_iter, device, gauge_path=None, source_path=None, seed=42):
    """Create a loaded QCU case and return (ctx, fields, tensors, kappa)."""
    import torch
    from pyqcu import dslash, tools
    from pyqcu.cuda import define
    from pyqcu.testing.qcu.single_function_common import (lifecycle_call,
                                                          make_context)

    ctx = make_context(lat, mass=mass, plan=1, max_iter=max_iter, device=device)
    compact = tuple(define.lat_shape(ctx.params))
    g = torch.empty((2, 3, 3, 4, *compact), dtype=ctx.dtype, device=ctx.device)
    src = torch.empty((2, 4, 3, *compact), dtype=ctx.dtype, device=ctx.device)
    out = torch.empty_like(src)
    clover = (torch.empty((4, 3, 4, 3, *compact), dtype=ctx.dtype, device=ctx.device)
              for _ in range(4))
    ce, co, cei, coi = clover

    if gauge_path is None:
        ctx.params[define._SET_PLAN_] = define._SET_PLAN_N_1_
        ctx.params[define._SEED_] = int(seed)
        ctx.argv[define._SIGMA_] = 0.1
        lifecycle_call(ctx, "applyGaussGaugeQcu", g)
    else:
        path = resolve_path(gauge_path, resolve_path(None, gauge_path))
        if path.suffix == ".lime":
            qcu_gauge = qdp_gauge_to_qcu(read_qdp_gauge(path))
        else:
            qcu_gauge = read_h5(path)
        if tuple(qcu_gauge.shape[-4:]) != tuple(ctx.lattice):
            raise ValueError(
                f"gauge lattice {tuple(qcu_gauge.shape[-4:])} does not match "
                f"--lat {tuple(ctx.lattice)}")
        g.copy_(torch.from_numpy(qcu_gauge).to(device=ctx.device, dtype=ctx.dtype))

    if source_path is None:
        gen = torch.Generator(device="cpu").manual_seed(int(seed) + 1)
        real = torch.randn((2, 4, 3, *compact), generator=gen, dtype=torch.float32)
        imag = torch.randn((2, 4, 3, *compact), generator=gen, dtype=torch.float32)
        src.copy_(torch.complex(real, imag).to(ctx.device))
    else:
        source = read_h5(resolve_path(source_path, resolve_path(None, source_path)))
        src.copy_(tools.oooxyzt2poooxyzt(torch.from_numpy(source).to(
            device=ctx.device, dtype=ctx.dtype)))

    ctx.params[define._SET_PLAN_] = define._SET_PLAN2_
    ctx.params[define._PARITY_] = 0
    lifecycle_call(ctx, "applyCloversQcu", ce, cei, g)
    ctx.params[define._PARITY_] = 1
    lifecycle_call(ctx, "applyCloversQcu", co, coi, g)
    return ctx, (g, src, out, ce, co, cei, coi), dslash, tools, kappa(mass)


def kappa(mass: float) -> float:
    return 1.0 / (2.0 * mass + 8.0)


def true_residual(fields, operator, tools, kappa_value: float) -> float:
    import torch
    g, src, out, _, _, _, _ = fields
    u = tools.poooxyzt2oooxyzt(g)
    x = tools.poooxyzt2oooxyzt(out)
    b = tools.poooxyzt2oooxyzt(src)
    u0 = torch.ones(1, dtype=torch.float32, device=u.device)
    clover = operator.make_clover(
        u, kappa=torch.tensor([kappa_value], dtype=torch.float32, device=u.device),
        u_0=u0)
    residual = (operator.give_wilson(x, u, torch.tensor([kappa_value], device=u.device),
                                     u_0=u0, with_I=True) +
                operator.give_clover(x, clover) - b)
    return float(torch.linalg.vector_norm(residual) / torch.linalg.vector_norm(b))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lat", nargs=4, type=int, default=(8, 8, 16, 16))
    parser.add_argument("--mass", type=float, default=None,
                        help="QCU mass; default 0.05 for the generated example")
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gauge", default=None, help="QCU HDF5 or QUDA .lime")
    parser.add_argument("--source", default=None, help="QCU full fermion HDF5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="optional QCU solution HDF5")
    parser.add_argument("--report", default="performance_qcu_bistabcg.json")
    args = parser.parse_args(argv)

    if not qcu_available():
        print(json.dumps({"status": "SKIP", "reason": "QCU extension/CUDA unavailable"}))
        return 0
    mass = float(args.mass) if args.mass is not None else DEFAULT_MASS
    mass_source = "cli" if args.mass is not None else "safe-default"
    gauge_csw = None
    reference_mass = None
    comparison_valid = True
    warnings = []
    if mass < 0.0 and args.gauge is None:
        comparison_valid = False
        warnings.append(
            "negative mass with the generated csw=1 gauge is not a supported "
            "QCU reference case; use mass >= 0 or a matching backend configuration")
    if args.gauge is not None:
        gauge_path = resolve_path(args.gauge, resolve_path(None, args.gauge))
        if gauge_path.suffix == ".lime":
            meta = lime_metadata(gauge_path)
            gauge_csw = meta.get("csw")
            reference_mass = meta.get("reference_mass")
            if gauge_csw is not None and abs(float(gauge_csw) - 1.0) > 1e-12:
                comparison_valid = False
                warnings.append(
                    f"QCU uses csw=1 but input .lime requests csw={gauge_csw}; "
                    "the solve is gauge-only, not a Chroma-config equivalent")
    ctx, fields, operator, tools, kappa_value = prepare_case(
        tuple(args.lat), mass, args.atol, args.max_iter, args.device,
        args.gauge, args.source, args.seed)
    import torch
    from pyqcu.cuda import define, qcu
    g, src, out, ce, co, cei, coi = fields
    ctx.params[define._SET_PLAN_] = define._SET_PLAN1_
    memory = None
    qcu.applyInitQcu(ctx.set_ptrs, ctx.params, ctx.argv)
    try:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        qcu.applyCloverBistabCgQcu(out, src, g, ce, co, cei, coi,
                                  ctx.set_ptrs, ctx.params)
        end.record()
        torch.cuda.synchronize(ctx.device)
        wall_seconds = start.elapsed_time(end) / 1000.0
        iterations = int(qcu.getCloverBistabCgIterationsQcu(ctx.set_ptrs, ctx.params))
        memory = cuda_memory_snapshot(ctx.device)
    finally:
        ctx.params[define._SET_INDEX_] += 1
        qcu.applyEndQcu(ctx.set_ptrs, ctx.params)
    residual = true_residual(fields, operator, tools, kappa_value)
    if args.output:
        output = resolve_path(args.output, resolve_path(None, args.output))
        write_h5(output, tools.poooxyzt2oooxyzt(out).detach().cpu().numpy(), metadata={
            "solver": "applyCloverBistabCgQcu", "mass": mass,
            "kappa": kappa_value, "csw": 1.0, "atol": args.atol,
        })
    report = {
        "backend": "qcu", "solver": "applyCloverBistabCgQcu",
        "lattice": list(args.lat), "mass": mass, "mass_source": mass_source,
        "kappa": kappa_value, "csw": 1.0, "gauge_csw": gauge_csw,
        "reference_mass": reference_mass, "comparison_valid": comparison_valid,
        "warnings": warnings, "iterations": iterations, "wall_seconds": wall_seconds,
        "seconds_per_iteration": wall_seconds / iterations if iterations > 0 else None,
        "final_true_residual": residual,
        "residual_history": [],
        "residual_history_source": "not exposed by QCU BiStabCG native interface",
        "gflops": estimated_clover_gflops(tuple(args.lat), iterations, wall_seconds),
        "memory": memory,
        "converged": bool(math.isfinite(residual) and iterations < args.max_iter),
        "status": ("PASS" if math.isfinite(residual) and residual < 1e-2
                   and iterations < args.max_iter and comparison_valid
                   else "FAIL" if not math.isfinite(residual) or iterations >= args.max_iter
                   else "WARN"),
    }
    report_path = resolve_path(args.report, DATA_DIR / args.report)
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
