#!/usr/bin/env python3
"""Generate a Gaussian SU(3) gauge field through QCU or the PyTorch fallback."""

from __future__ import annotations

import argparse
import json

import numpy as np

try:
    from ._common import DATA_DIR, resolve_path, write_h5
except ImportError:
    from _common import DATA_DIR, resolve_path, write_h5


def _torch_generate(lat, seed: int, sigma: float, device: str):
    import torch
    from pyqcu import lattice

    dev = torch.device(device)
    gauge = torch.empty((3, 3, 4, *lat), dtype=torch.complex64, device=dev)
    lattice.generate_gauge_field(gauge, sigma=sigma, seed=seed, verbose=False)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    return gauge.detach().cpu().numpy(), "torch"


def _qcu_generate(lat, seed: int, sigma: float, device: str, mass: float):
    import torch
    from pyqcu import tools
    from pyqcu.cuda import QcuRuntime, define

    runtime = QcuRuntime.from_lattice(
        lat, mass=mass, csw=1.0, sigma=sigma, device=device, plan=-1,
        seed=seed)
    compact = define.lat_shape(runtime.params)
    shape = (2, 3, 3, 4, *compact)
    gauge_eo = torch.empty(shape, dtype=torch.complex64, device=runtime.device)
    runtime.call("applyGaussGaugeQcu", gauge_eo)
    gauge = tools.poooxyzt2oooxyzt(gauge_eo).detach().cpu().numpy()
    return gauge, "qcu"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lat", nargs=4, type=int, default=(8, 8, 16, 16))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--mass", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", choices=("auto", "qcu", "torch"), default="auto")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    lat = tuple(args.lat)
    if any(v <= 0 or v % 2 for v in lat):
        parser.error("lattice dimensions must be positive and even")

    backend = args.backend
    if backend == "auto":
        try:
            import torch
            from pyqcu.cuda import qcu
            backend = "qcu" if torch.cuda.is_available() and hasattr(qcu, "applyGaussGaugeQcu") else "torch"
        except Exception:
            backend = "torch"
    if backend == "qcu":
        gauge, actual = _qcu_generate(lat, args.seed, args.sigma, args.device, args.mass)
    else:
        gauge, actual = _torch_generate(lat, args.seed, args.sigma, args.device)

    output = resolve_path(args.output,
                          DATA_DIR / f"gauge_gauss_{'x'.join(map(str, lat))}_seed{args.seed}_c64.h5")
    write_h5(output, gauge, metadata={
        "layout": "[3,3,4,x,y,z,t]", "backend": actual,
        "seed": args.seed, "sigma": args.sigma, "mass": args.mass,
        "kappa": 1.0 / (2.0 * args.mass + 8.0), "csw": 1.0,
    })
    print(json.dumps({
        "backend": actual, "output": str(output), "shape": list(gauge.shape),
        "dtype": str(gauge.dtype), "seed": args.seed, "sigma": args.sigma,
        "nyquist_lattice": all(v > 1 for v in lat), "status": "PASS",
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
