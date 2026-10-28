#!/usr/bin/env python3
"""Single-function test for CUDA C++ multigrid null-vector CGS2."""

from __future__ import annotations

import json

import torch

from pyqcu.cuda import define, qcu
from pyqcu.testing.qcu.single_function_common import run_setup


def main(argv=None) -> int:
    args, report = run_setup(
        __doc__ or "MultiGrid orthogonalize 单测", __doc__, argv=argv, total=2)
    E = 6
    X, Y, Z, T = args.lat
    fine = (X, Y, Z, T // 2)
    coarse = (X // 2, Y // 2, Z // 2, T // 4)
    if not torch.cuda.is_available() or args.pure_only:
        report.skip("CUDA unavailable or --pure-only")
        report.finish("PURE_ONLY" if args.pure_only else "SKIP")
        return 0

    flat = torch.randn((E, 12, *fine), dtype=torch.complex64, device="cuda")
    blocked = torch.empty_like(flat)
    data_type = int(define.epytd(flat.dtype).item())
    report.run(
        "C++ CUDA CGS2",
        lambda: qcu.applyMultigridLocalOrthogonalizeQcu(
            flat, blocked, E, 12, *fine, *coarse, data_type))

    def gram_error() -> float:
        Xc, Yc, Zc, Tc = coarse
        columns = blocked.reshape(
            E, 12, Xc, 2, Yc, 2, Zc, 2, Tc, 2).permute(
                2, 4, 6, 8, 0, 1, 3, 5, 7, 9).reshape(
                    Xc * Yc * Zc * Tc, E, 12 * 16).transpose(-2, -1)
        gram = columns.conj().transpose(-2, -1) @ columns
        identity = torch.eye(E, dtype=gram.dtype, device=gram.device)
        return float((gram - identity).abs().max().item())

    error = report.run("块局部 Gram", gram_error)
    if error > 1.0e-4:
        raise AssertionError(f"Gram error {error:.3e} exceeds 1e-4")
    print(json.dumps({
        "functions": ["applyMultigridLocalOrthogonalizeQcu"],
        "gram_max_abs_error": error,
        "lattice": list(fine),
        "coarse": list(coarse),
        "status": "PASS",
    }))
    report.finish("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
