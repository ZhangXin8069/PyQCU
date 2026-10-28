#!/usr/bin/env python3
"""Single-function test for CUDA C++ multigrid near-null-vector setup."""

from __future__ import annotations

import json
import os

import torch

from pyqcu.cuda import define, qcu
from pyqcu.testing.qcu.single_function_common import run_setup


def main(argv=None) -> int:
    args, report = run_setup(
        __doc__ or "MultiGrid near-null-vector C++ setup 单测",
        __doc__, argv=argv, total=3)
    E, e = 3, 12
    X, Y, Z, T = args.lat
    if not torch.cuda.is_available() or args.pure_only:
        if args.pure_only:
            report.skip("CUDA unavailable or --pure-only")
            report.finish("PURE_ONLY")
            return 0
        if os.environ.get("PYQCU_REQUIRE_CUDA") == "1":
            raise RuntimeError("PYQCU_REQUIRE_CUDA=1 but CUDA is unavailable")
        report.skip("CUDA unavailable or --pure-only")
        report.finish("SKIP")
        return 0
    device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    dtype = torch.complex64
    parity_shape = (X, Y, Z, T // 2)
    # Production parity compression normally keeps time uncoarsened
    # (bt == 1), so the regression must cover that geometry explicitly.
    coarse = (X // 2, Y // 2, Z // 2, T // 2)
    gauge = torch.zeros(
        (2, 3, 3, 4, *parity_shape), dtype=dtype, device=device)
    eye = torch.eye(3, dtype=dtype, device=device)
    gauge[...] = eye.view(1, 3, 3, 1, 1, 1, 1, 1)
    onsite_shape = (4, 3, 4, 3, *parity_shape)
    clover_even = torch.empty(onsite_shape, dtype=dtype, device=device)
    clover_odd = torch.empty_like(clover_even)
    clover_even_inv = torch.empty_like(clover_even)
    clover_odd_inv = torch.empty_like(clover_even)
    bx, by, bz, bt = (parity_shape[i] // coarse[i] for i in range(4))
    blocked = torch.empty(
        (E, e, coarse[0], bx, coarse[1], by, coarse[2], bz, coarse[3], bt),
        dtype=dtype, device=device)
    sitting = torch.zeros((E, E, *coarse), dtype=dtype, device=device)
    hop_nn = torch.zeros((2, 4, E, E, *coarse), dtype=dtype, device=device)
    hop_diag = torch.zeros(
        (2, 2, 6, E, E, *coarse), dtype=dtype, device=device)

    params = torch.zeros(define._PARAMS_SIZE_, dtype=torch.int32)
    params[define._LAT_X_], params[define._LAT_Y_] = X, Y
    params[define._LAT_Z_], params[define._LAT_T_] = Z, T
    params[define._LAT_XYZT_] = X * Y * Z * T
    params[define._GRID_X_], params[define._GRID_Y_] = 1, 1
    params[define._GRID_Z_], params[define._GRID_T_] = 1, 1
    params[define._NODE_RANK_] = 0
    params[define._NODE_SIZE_] = 1
    params[define._MAX_ITER_] = 200
    params[define._DATA_TYPE_] = define._LAT_C64_
    params[define._SET_INDEX_] = 0
    params[define._SET_PLAN_] = define._SET_PLAN2_
    params[define._MG_LEVEL1_E_] = E
    params[define._MG_LEVEL1_X_] = coarse[0]
    params[define._MG_LEVEL1_Y_] = coarse[1]
    params[define._MG_LEVEL1_Z_] = coarse[2]
    params[define._MG_LEVEL1_T_] = coarse[3]
    params[define._SEED_] = 42
    argv_t = define.make_argv(args.mass, csw=args.csw, atol=1.0e-3)
    set_ptrs = torch.zeros(define._SET_PTRS_SIZE_, dtype=torch.int64)

    def run_builder() -> None:
        qcu.applyInitQcu(set_ptrs, params, argv_t)
        try:
            params[define._PARITY_] = 0
            qcu.applyCloversQcu(
                clover_even, clover_even_inv, gauge, set_ptrs, params)
            params[define._PARITY_] = 1
            qcu.applyCloversQcu(
                clover_odd, clover_odd_inv, gauge, set_ptrs, params)
            params[define._SET_PLAN_] = define._SET_PLAN1_
            qcu.applyMultigridBuildNullVectorsQcu(
                blocked, gauge, clover_even, clover_odd, clover_even_inv,
                clover_odd_inv, set_ptrs, params, E, e, *parity_shape,
                *coarse, 1, 42, 5.0e-2)
            qcu.applyMultigridBuildStencilLocalQcu(
                sitting, hop_nn, hop_diag, blocked, gauge, clover_even,
                clover_odd, clover_even_inv, clover_odd_inv, set_ptrs,
                params, E, e, *parity_shape, *coarse, 8, 8)
        finally:
            qcu.applyEndQcu(set_ptrs, params)

    report.run("C++ CUDA near-null inverse iteration + CGS2 + stencil",
               run_builder)

    def gram_error() -> float:
        Xc, Yc, Zc, Tc = coarse
        columns = blocked.reshape(
            E, e, Xc, bx, Yc, by, Zc, bz, Tc, bt).permute(
                2, 4, 6, 8, 0, 1, 3, 5, 7, 9).reshape(
                    Xc * Yc * Zc * Tc, E, e * bx * by * bz * bt).transpose(-2, -1)
        gram = columns.conj().transpose(-2, -1) @ columns
        identity = torch.eye(E, dtype=gram.dtype, device=gram.device)
        return float((gram - identity).abs().max().item())

    error = report.run("near-null 块局部 Gram", gram_error)
    if error > 1.0e-4:
        raise AssertionError(f"Gram error {error:.3e} exceeds 1e-4")

    def galerkin_error() -> float:
        generator = torch.Generator(device=device).manual_seed(941)
        coarse_vector = torch.randn(
            (E, *coarse), dtype=dtype, device=device, generator=generator)
        prolonged = torch.empty((e, *parity_shape), dtype=dtype, device=device)
        schur_vector = torch.empty_like(prolonged)
        restricted = torch.empty_like(coarse_vector)
        stencil_vector = torch.empty_like(coarse_vector)
        def checkpoint(label: str) -> None:
            torch.cuda.synchronize(device)
            print(f"[GALERKIN] {label}", flush=True)

        checkpoint("init")
        qcu.applyInitQcu(set_ptrs, params, argv_t)
        try:
            checkpoint("prolong")
            qcu.applyMultigridProLongQcu(
                prolonged, coarse_vector, blocked, set_ptrs, params)
            checkpoint("fine-schur")
            qcu.applyCloverBistabCgDslashQcu(
                schur_vector, prolonged, gauge, clover_even, clover_odd,
                clover_even_inv, clover_odd_inv, set_ptrs, params)
            checkpoint("restrict")
            qcu.applyMultigridRestrictQcu(
                restricted, schur_vector, blocked, set_ptrs, params)
            checkpoint("coarse-dslash")
            qcu.applyMultigridCoarseDslashWideQcu(
                stencil_vector, coarse_vector, sitting, hop_nn, hop_diag,
                set_ptrs, params)
            checkpoint("random-compare")
            denominator = torch.linalg.vector_norm(restricted).clamp_min(1.0)
            worst_error = float(
                (torch.linalg.vector_norm(stencil_vector - restricted) /
                 denominator).item())
            basis_centers = (
                (0, 0, 0, 0),
                (1, 1, 1, coarse[3] - 3),
                (coarse[0] - 1, coarse[1] - 1, coarse[2] - 1,
                 coarse[3] - 1),
            )
            for center in basis_centers:
                for probe in range(E):
                    coarse_vector.zero_()
                    coarse_vector[(probe, *center)] = 1
                    qcu.applyMultigridProLongQcu(
                        prolonged, coarse_vector, blocked, set_ptrs, params)
                    qcu.applyCloverBistabCgDslashQcu(
                        schur_vector, prolonged, gauge, clover_even,
                        clover_odd, clover_even_inv, clover_odd_inv,
                        set_ptrs, params)
                    qcu.applyMultigridRestrictQcu(
                        restricted, schur_vector, blocked, set_ptrs, params)
                    qcu.applyMultigridCoarseDslashWideQcu(
                        stencil_vector, coarse_vector, sitting, hop_nn,
                        hop_diag, set_ptrs, params)
                    basis_error = float((
                        torch.linalg.vector_norm(
                            stencil_vector - restricted) /
                        torch.linalg.vector_norm(restricted).clamp_min(1.0)
                    ).item())
                    worst_error = max(worst_error, basis_error)
        finally:
            qcu.applyEndQcu(set_ptrs, params)
        return worst_error

    galerkin = report.run("C++ CUDA 33-tensor Galerkin 一致性",
                          galerkin_error)
    if galerkin > 2.0e-5:
        raise AssertionError(f"Galerkin error {galerkin:.3e} exceeds 2e-5")
    print(json.dumps({
        "functions": [
            "applyCloversQcu", "applyMultigridBuildNullVectorsQcu",
            "applyMultigridBuildStencilLocalQcu"],
        "gram_max_abs_error": error,
        "galerkin_relative_error": galerkin,
        "lattice": [X, Y, Z, T],
        "parity_lattice": list(parity_shape),
        "coarse": list(coarse),
        "E": E,
        "status": "PASS",
    }))
    report.finish("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
