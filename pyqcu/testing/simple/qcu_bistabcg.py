#!/usr/bin/env python3
"""Minimal QCU Clover BiStabCG example, with explicit ABI lifecycle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import median

try:
    from ._common import (DATA_DIR, DEFAULT_ATOL, DEFAULT_CSW, DEFAULT_MASS,
                          convergence_status, cuda_memory_snapshot,
                          estimated_clover_gflops, csw_matches, lime_metadata,
                          qdp_gauge_to_qcu, read_h5, read_h5_metadata,
                          read_qdp_gauge, resolve_csw, resolve_path, write_h5,
                          write_json)
except ImportError:
    from _common import (DATA_DIR, DEFAULT_ATOL, DEFAULT_CSW, DEFAULT_MASS,
                         convergence_status, cuda_memory_snapshot,
                         estimated_clover_gflops, csw_matches, lime_metadata,
                         qdp_gauge_to_qcu, read_h5, read_h5_metadata,
                         read_qdp_gauge, resolve_csw, resolve_path, write_h5,
                         write_json)

if str(Path(__file__).resolve().parents[3]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def qcu_available() -> bool:
    try:
        import torch
        from pyqcu.cuda import qcu
        return torch.cuda.is_available() and hasattr(qcu, "applyCloverBistabCgQcu")
    except Exception:
        return False


def prepare_case(lat, mass, csw, atol, max_iter, device, gauge_path=None,
                 source_path=None, seed=42):
    """Create a loaded QCU case and return (runtime, fields, references, kappa)."""
    import torch
    from pyqcu import tools
    from pyqcu.torch import dslash
    from pyqcu.cuda import QcuRuntime, define

    runtime = QcuRuntime.from_lattice(
        lat, mass=mass, csw=csw, atol=atol, device=device, plan=1,
        max_iter=max_iter, seed=seed)
    compact = tuple(define.lat_shape(runtime.params))
    g = torch.empty((2, 3, 3, 4, *compact), dtype=torch.complex64,
                    device=runtime.device)
    src = torch.empty((2, 4, 3, *compact), dtype=torch.complex64,
                      device=runtime.device)
    out = torch.empty_like(src)

    if gauge_path is None:
        runtime.configure(plan=define._SET_PLAN_N_1_)
        runtime.call("applyGaussGaugeQcu", g)
    else:
        path = resolve_path(gauge_path, resolve_path(None, gauge_path))
        if path.suffix == ".lime":
            qcu_gauge = qdp_gauge_to_qcu(read_qdp_gauge(path))
        else:
            qcu_gauge = read_h5(path)
        if tuple(qcu_gauge.shape[-4:]) != tuple(runtime.lattice):
            raise ValueError(
                f"gauge lattice {tuple(qcu_gauge.shape[-4:])} does not match "
                f"--lat {tuple(runtime.lattice)}")
        full_gauge = torch.from_numpy(qcu_gauge).to(dtype=torch.complex64)
        g.copy_(tools.oooxyzt2poooxyzt(full_gauge).to(device=runtime.device))

    if source_path is None:
        gen = torch.Generator(device="cpu").manual_seed(int(seed) + 1)
        real = torch.randn((2, 4, 3, *compact), generator=gen, dtype=torch.float32)
        imag = torch.randn((2, 4, 3, *compact), generator=gen, dtype=torch.float32)
        src.copy_(torch.complex(real, imag).to(runtime.device))
    else:
        source = read_h5(resolve_path(source_path, resolve_path(None, source_path)))
        src.copy_(tools.oooxyzt2poooxyzt(torch.from_numpy(source).to(
            device=runtime.device, dtype=torch.complex64)))

    runtime.configure(plan=define._SET_PLAN2_)
    ce, cei = runtime.make_clover(g, with_inverse=True, parity=0)
    co, coi = runtime.make_clover(g, with_inverse=True, parity=1)
    return runtime, (g, src, out, ce, co, cei, coi), dslash, tools, kappa(mass)


def kappa(mass: float) -> float:
    return 1.0 / (2.0 * mass + 8.0)


def true_residual(fields, operator, tools, kappa_value: float,
                  csw: float) -> float:
    import torch
    g, src, out, _, _, _, _ = fields
    u = tools.poooxyzt2oooxyzt(g)
    x = tools.poooxyzt2oooxyzt(out)
    b = tools.poooxyzt2oooxyzt(src)
    u0 = torch.ones(1, dtype=torch.float32, device=u.device)
    clover = operator.make_clover(
        u, kappa=torch.tensor([kappa_value], dtype=torch.float32, device=u.device),
        u_0=u0, csw=csw)
    residual = (operator.give_wilson(x, u, torch.tensor([kappa_value], device=u.device),
                                     u_0=u0, with_I=True) +
                operator.give_clover(x, clover) - b)
    return float(torch.linalg.vector_norm(residual) / torch.linalg.vector_norm(b))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lat", nargs=4, type=int, default=(8, 8, 16, 16))
    parser.add_argument("--mass", type=float, default=None,
                        help="QCU mass; defaults to LIME Mass or 0.05")
    parser.add_argument("--csw", type=float, default=None,
                        help="Clover coefficient; defaults to LIME metadata or "
                             f"{DEFAULT_CSW:g}")
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gauge", default=None, help="QCU HDF5 or QUDA .lime")
    parser.add_argument("--source", default=None, help="QCU full fermion HDF5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1,
                        help="untimed solve repetitions before sampling")
    parser.add_argument("--repeats", type=int, default=3,
                        help="timed solve repetitions; report the median")
    parser.add_argument("--output", default=None, help="optional QCU solution HDF5")
    parser.add_argument("--report", default="performance_qcu_bistabcg.json")
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeats < 1:
        parser.error("--warmup must be >= 0 and --repeats >= 1")

    if not qcu_available():
        print(json.dumps({"status": "SKIP", "reason": "QCU extension/CUDA unavailable"}))
        return 0
    gauge_csw = None
    gauge_metadata = {}
    reference_mass = None
    comparison_valid = True
    warnings = []
    if args.gauge is not None:
        gauge_path = resolve_path(args.gauge, resolve_path(None, args.gauge))
        if gauge_path.suffix == ".lime":
            meta = lime_metadata(gauge_path)
            gauge_metadata = meta
            gauge_csw = meta.get("csw")
            reference_mass = meta.get("reference_mass")
        elif gauge_path.suffix in (".h5", ".hdf5"):
            meta = read_h5_metadata(gauge_path)
            gauge_metadata = meta
            gauge_csw = meta.get("csw")
            reference_mass = meta.get("reference_mass", meta.get("mass"))
    if args.mass is not None:
        mass = float(args.mass)
        mass_source = "cli"
    elif reference_mass is not None:
        mass = float(reference_mass)
        mass_source = "gauge-metadata"
    else:
        mass = DEFAULT_MASS
        mass_source = "safe-default"
    if mass < 0.0 and args.gauge is None:
        comparison_valid = False
        warnings.append(
            "negative mass with the generated gauge is not a reference Chroma "
            "configuration; provide a matching LIME gauge instead")
    csw = resolve_csw(args.csw, gauge_metadata)
    if not csw_matches(csw, gauge_csw):
        comparison_valid = False
        warnings.append(
            f"requested csw={csw:g} differs from LIME metadata csw={gauge_csw:g}")
    runtime, fields, operator, tools, kappa_value = prepare_case(
        tuple(args.lat), mass, csw, args.atol, args.max_iter, args.device,
        args.gauge, args.source, args.seed)
    import torch
    from pyqcu.cuda import define
    g, src, out, ce, co, cei, coi = fields
    runtime.configure(plan=define._SET_PLAN1_)
    memory = None
    samples = []
    iterations = 0
    for repeat in range(args.warmup + args.repeats):
        with runtime.operation("applyCloverBistabCgQcu") as operation:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            operation(out, src, g, ce, co, cei, coi)
            end.record()
            torch.cuda.synchronize(runtime.device)
            wall = start.elapsed_time(end) / 1000.0
            iterations = int(operation.read(
                "getCloverBistabCgIterationsQcu"))
        if repeat >= args.warmup:
            samples.append(wall)
        memory = cuda_memory_snapshot(runtime.device)
    wall_seconds = median(samples)
    residual = true_residual(fields, operator, tools, kappa_value, csw)
    converged, status, residual_acceptance = convergence_status(
        residual, args.atol, comparison_valid=comparison_valid)
    if args.output:
        output = resolve_path(args.output, resolve_path(None, args.output))
        write_h5(output, tools.poooxyzt2oooxyzt(out).detach().cpu().numpy(), metadata={
            "solver": "applyCloverBistabCgQcu", "mass": mass,
            "kappa": kappa_value, "csw": csw, "atol": args.atol,
        })
    report = {
        "backend": "qcu", "solver": "applyCloverBistabCgQcu",
        "lattice": list(args.lat), "mass": mass, "mass_source": mass_source,
        "kappa": kappa_value, "mass_normalization": "kappa",
        "csw": csw, "gauge_csw": gauge_csw,
        "reference_mass": reference_mass, "comparison_valid": comparison_valid,
        "warnings": warnings, "iterations": iterations, "wall_seconds": wall_seconds,
        "wall_samples": samples, "warmup": args.warmup,
        "seconds_per_iteration": wall_seconds / iterations if iterations > 0 else None,
        "final_true_residual": residual,
        "residual_history": [],
        "residual_history_source": "not exposed by QCU BiStabCG native interface",
        "gflops": estimated_clover_gflops(tuple(args.lat), iterations, wall_seconds),
        "memory": memory,
        "residual_acceptance": residual_acceptance,
        "converged": converged,
        "status": status,
    }
    report_path = resolve_path(args.report, DATA_DIR / args.report)
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
