#!/usr/bin/env python3
"""Build a full Wilson-Clover 33-tensor MG cache on CPU or CUDA."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pyqcu.tools._legacy_mg_cache import (
    file_sha256,
    legacy_mg_cache_identity,
    logical_array_sha256,
    save_legacy_mg_cache,
    write_legacy_mg_cache_manifest,
)

try:
    from ._common import (
        lime_metadata, qdp_gauge_to_qcu, read_h5, read_h5_metadata,
        read_qdp_gauge, resolve_path,
    )
    from .kupl_multigrid import build_coarse_assets
except ImportError:
    from _common import (
        lime_metadata, qdp_gauge_to_qcu, read_h5, read_h5_metadata,
        read_qdp_gauge, resolve_path,
    )
    from kupl_multigrid import build_coarse_assets


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gauge", required=True)
    parser.add_argument("--lat", nargs=4, type=int, required=True)
    parser.add_argument("--mass", type=float, required=True)
    parser.add_argument("--csw", type=float, required=True)
    parser.add_argument("--dtype", choices=("c64", "c128"), default="c64")
    parser.add_argument("--dof", nargs="+", type=int, default=(12, 12))
    parser.add_argument("--mg-grid", nargs=4, type=int, default=(2, 2, 2, 2))
    parser.add_argument("--nv-iters", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--build-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    lattice = tuple(int(value) for value in args.lat)
    gauge_path = resolve_path(args.gauge, Path(args.gauge).expanduser())
    if gauge_path.suffix.lower() == ".lime":
        metadata = lime_metadata(gauge_path)
        gauge = qdp_gauge_to_qcu(read_qdp_gauge(gauge_path))
    else:
        metadata = read_h5_metadata(gauge_path)
        gauge = read_h5(gauge_path)
    if tuple(gauge.shape[-4:]) != lattice:
        raise ValueError(
            f"gauge lattice {tuple(gauge.shape[-4:])} does not match {lattice}")

    dtype = np.dtype(np.complex64 if args.dtype == "c64" else np.complex128)
    gauge = np.ascontiguousarray(gauge, dtype=dtype)
    kappa = 1.0 / (2.0 * float(args.mass) + 8.0)
    started = time.perf_counter()
    setup_stats: dict[str, object] = {}
    assets = build_coarse_assets(
        gauge, lattice, args.mass, kappa, args.csw, dtype, 2,
        list(args.dof), list(args.mg_grid), args.nv_iters, args.seed,
        verbose=True, local_stencil=True,
        local_stencil_batch_size=args.batch_size,
        device_name=args.build_device, setup_stats=setup_stats)
    elapsed = time.perf_counter() - started
    output = resolve_path(args.output, Path(args.output).expanduser())
    datasets = {
        name: value[0] for name, value in zip(
            ("lonv", "hnn", "hdg", "sit"), assets, strict=True)
    }
    identity = legacy_mg_cache_identity(
        gauge_canonical_sha256=logical_array_sha256(gauge),
        gauge_file_sha256=file_sha256(gauge_path),
        lattice=lattice,
        dtype=dtype,
        mass=args.mass,
        csw=args.csw,
        dof=args.dof,
        mg_grid=args.mg_grid,
        nv_iters=args.nv_iters,
        seed=args.seed,
    )
    report = save_legacy_mg_cache(
        output, datasets, identity=identity,
        setup_method="cxx-null-vector+local-33-stencil",
        setup_backend=(
            "cuda-qcu-cxx" if args.build_device == "cuda"
            else "kupl-qcu-cxx"),
        setup_validation=setup_stats,
        source_metadata=metadata,
        setup_seconds=elapsed)
    report.update({
        "gauge": str(gauge_path),
        "build_device": args.build_device,
        "local_stencil": True,
        "local_stencil_batch_size": int(args.batch_size),
        "output": str(output),
    })
    if args.report is not None:
        report_path = resolve_path(args.report, Path(args.report).expanduser())
        write_legacy_mg_cache_manifest(output, report)
        if report_path.resolve() != output.with_suffix(".json").resolve():
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
