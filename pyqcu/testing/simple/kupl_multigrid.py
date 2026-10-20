#!/usr/bin/env python3
"""Minimal KUPL Clover multigrid example with Galerkin verification."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from ._common import (DATA_DIR, DEFAULT_ATOL, DEFAULT_BLOCK, DEFAULT_MASS,
                          convergence_status, kupl_fermion_to_qcu,
                          qcu_fermion_to_kupl, qcu_gauge_to_kupl,
                          resolve_csw, resolve_path, write_h5, write_json)
    from .kupl_bistabcg import (NumpyTensor, PLAN_CLOVER, PLAN_SOLVER,
                                PARITY, SET_PLAN, cpu_memory_snapshot,
                                kupl_backend, load_gauge, load_source,
                                make_params, relative_full_residual)
except ImportError:
    from _common import (DATA_DIR, DEFAULT_ATOL, DEFAULT_BLOCK, DEFAULT_MASS,
                         convergence_status, kupl_fermion_to_qcu,
                         qcu_fermion_to_kupl, qcu_gauge_to_kupl,
                         resolve_csw, resolve_path, write_h5, write_json)
    from kupl_bistabcg import (NumpyTensor, PLAN_CLOVER, PLAN_SOLVER,
                               PARITY, SET_PLAN, cpu_memory_snapshot,
                               kupl_backend, load_gauge, load_source,
                               make_params, relative_full_residual)

if str(Path(__file__).resolve().parents[3]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


MG_NUM_LEVEL, MG_LEVEL_INDEX = 17, 18
MG_LEVEL_BASE, MG_LEVEL_STRIDE = 19, 8
MG_LEVEL_MAX_ITER = 5
MG_LEVEL_DATA_TYPE = 6
MG_LEVEL_NUM_RESTART = 7
MG_USE_DEFLATE = 55
SET_PTRS_COARSE_BASE = 30
ARGV_MG_LEVEL_BASE = 3
DATA_TYPE = 14


def configure_mg(params: np.ndarray, argv: np.ndarray, set_ptrs: np.ndarray,
                 levels: int, atol: float, restart: int,
                 coarse_max_iter: int,
                 coarse_tol_factor: float, deflate: bool,
                 null_vectors: list[Any], hop_nn: list[Any],
                 hop_diag: list[Any], sitting: list[Any]) -> None:
    params[MG_NUM_LEVEL] = levels
    params[MG_USE_DEFLATE] = int(deflate)
    for level in range(1, levels):
        block = null_vectors[level - 1].shape
        base = MG_LEVEL_BASE + (level - 1) * MG_LEVEL_STRIDE
        params[base] = int(block[0])
        params[base + 1] = int(block[2])
        params[base + 2] = int(block[4])
        params[base + 3] = int(block[6])
        params[base + 4] = int(block[8])
        params[base + MG_LEVEL_MAX_ITER] = int(coarse_max_iter)
        params[base + MG_LEVEL_DATA_TYPE] = int(params[DATA_TYPE])
        params[base + MG_LEVEL_NUM_RESTART] = int(restart)
        argv[ARGV_MG_LEVEL_BASE + level - 1] = float(
            atol * coarse_tol_factor * (3 ** (level - 1)))
        slot = SET_PTRS_COARSE_BASE + 4 * (level - 1)
        set_ptrs[slot] = int(np.ascontiguousarray(null_vectors[level - 1]).ctypes.data)
        set_ptrs[slot + 1] = int(np.ascontiguousarray(hop_nn[level - 1]).ctypes.data)
        set_ptrs[slot + 2] = int(np.ascontiguousarray(hop_diag[level - 1]).ctypes.data)
        set_ptrs[slot + 3] = int(np.ascontiguousarray(sitting[level - 1]).ctypes.data)


def build_coarse_assets(gauge: np.ndarray, lattice: tuple[int, int, int, int],
                        mass: float, kappa: float, csw: float,
                        dtype: np.dtype, levels: int, dof: list[int],
                        mg_grid: list[int], nv_iters: int, seed: int,
                        verbose: bool = False, local_stencil: bool = False,
                        local_stencil_batch_size: int = 4,
                        device_name: str = "cpu"):
    try:
        import torch
        from pyqcu import dslash, solver, tools
    except ImportError as exc:
        raise RuntimeError(f"full-site Galerkin builder unavailable: {exc}") from exc
    if levels != 2:
        raise ValueError("the independent simple MG fixture supports two levels")
    torch_dtype = torch.complex64 if dtype == np.dtype(np.complex64) else \
        torch.complex128
    real_dtype = torch.float32 if dtype == np.dtype(np.complex64) else \
        torch.float64
    device = torch.device(device_name)
    gauge_tensor = torch.from_numpy(np.ascontiguousarray(gauge)).to(device)
    kappa_tensor = torch.tensor([kappa], dtype=real_dtype, device=device)
    u0 = torch.ones(1, dtype=real_dtype, device=device)
    clover = dslash.make_clover(
        gauge_tensor, kappa=kappa_tensor, u_0=u0, csw=csw)
    fine_operator = dslash.operator(
        U=gauge_tensor, clover_term=clover, kappa=kappa_tensor,
        u_0=u0,
        support_parity=False, verbose=False)

    def matvec(vector):
        return fine_operator.matvec(vector)

    if verbose:
        print("KUPL::MG_BUILD: null-space start", flush=True)
    torch.manual_seed(seed + 9001)
    from pyqcu.tools._multigrid import give_null_vecs_mt

    null_vectors = give_null_vecs_mt(
        None, dof[1], 12, list(lattice), torch_dtype, device,
        nv_iters=max(1, nv_iters), nthreads=1, seed=seed + 9001,
        verbose=False, nv_tol=1.0e-2,
        batch_matvec=fine_operator.matvec_batch, batch_chunk=4)
    coarse_lattice = [
        lattice[index] // mg_grid[index] for index in range(4)]
    if verbose:
        print("KUPL::MG_BUILD: orthogonalize start", flush=True)
    null_vectors = tools.local_orthogonalize(
        null_vectors, coarse_lat_size=coarse_lattice, verbose=False)
    if verbose:
        print(
            f"KUPL::MG_BUILD: stencil start coarse={coarse_lattice}", flush=True)
    if local_stencil:
        from pyqcu.tools._multigrid import (
            BatchedLocalWilsonClover, build_stencil_local,
        )

        local_operator = BatchedLocalWilsonClover(
            fine_operator, *lattice, W=6)
        hop_nn, hop_diag, sitting = build_stencil_local(
            local_operator, null_vectors, dof[1], list(lattice),
            coarse_lattice, torch_dtype, device, verbose=verbose,
            batch_size=local_stencil_batch_size)
    else:
        hop_nn, hop_diag, sitting = tools.build_stencil(
            matvec, null_vectors, dof[1], 12, list(lattice),
            coarse_lattice, torch_dtype, device, verbose=verbose)
    if verbose:
        print("KUPL::MG_BUILD: stencil done", flush=True)
    return tuple(
        [tensor.detach().cpu().contiguous().numpy()]
        for tensor in (null_vectors, hop_nn, hop_diag, sitting))


def load_coarse_assets(path: Path, dtype: np.dtype):
    """Load the backend-neutral ``lonv/hnn/hdg/sit`` 33-tensor cache."""
    import h5py

    with h5py.File(path, "r") as handle:
        required = {"lonv", "hnn", "hdg", "sit"}
        missing = sorted(required.difference(handle.keys()))
        if missing:
            raise ValueError(f"coarse cache {path} is missing datasets {missing}")
        values = {
            name: np.ascontiguousarray(handle[name][...], dtype=dtype)
            for name in ("lonv", "hnn", "hdg", "sit")
        }
    return values["lonv"], values["hnn"], values["hdg"], values["sit"]


def write_coarse_assets(path: Path, null_vectors, hop_nn, hop_diag, sitting):
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        for name, value in (
                ("lonv", null_vectors), ("hnn", hop_nn),
                ("hdg", hop_diag), ("sit", sitting)):
            handle.create_dataset(name, data=np.asarray(value))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lat", nargs=4, type=int, default=(4, 4, 4, 4))
    parser.add_argument("--mass", type=float, default=None)
    parser.add_argument("--csw", type=float, default=None)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--dtype", choices=("c64", "c128"), default="c64")
    parser.add_argument("--levels", type=int, default=2)
    parser.add_argument("--dof", nargs="+", type=int, default=(12, 2, 2, 2, 2))
    parser.add_argument("--mg-grid", nargs=4, type=int, default=DEFAULT_BLOCK)
    parser.add_argument("--restart", type=int, default=5)
    parser.add_argument("--coarse-max-iter", type=int, default=200)
    parser.add_argument("--coarse-tol-factor", type=float, default=1.0e4)
    parser.add_argument("--nv-iters", type=int, default=2)
    parser.add_argument("--coarse-cache", type=Path, default=None,
                        help="load lonv/hnn/hdg/sit instead of rebuilding setup")
    parser.add_argument("--write-coarse-cache", type=Path, default=None)
    parser.add_argument("--build-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--no-deflate", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument("--local-stencil", action="store_true")
    parser.add_argument("--local-stencil-batch", type=int, default=4)
    parser.add_argument("--gauge", default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    parser.add_argument("--report", default="performance_kupl_multigrid.json")
    args = parser.parse_args(argv)

    backend = kupl_backend()
    if isinstance(backend, Exception):
        print(json.dumps({"status": "SKIP", "reason": str(backend)}))
        return 0
    qcu = backend
    if args.levels not in (1, 2) or len(args.dof) < args.levels:
        parser.error("--levels must be 1 or 2 and --dof needs one entry per level")
    lattice = tuple(args.lat)
    if lattice[3] % 2:
        parser.error("KUPL requires an even time extent")
    dtype = np.dtype(np.complex64 if args.dtype == "c64" else np.complex128)
    gauge_full, metadata = load_gauge(args.gauge, lattice, dtype)
    gauge = qcu_gauge_to_kupl(gauge_full)
    source_full = load_source(args.source, lattice, args.seed, dtype)
    source = qcu_fermion_to_kupl(source_full)
    solution = np.zeros_like(source)
    clover_shape = (4, 3, 4, 3, *lattice[:3], lattice[3] // 2)
    clover_even = np.empty(clover_shape, dtype=dtype)
    clover_odd = np.empty_like(clover_even)
    clover_even_inv = np.empty_like(clover_even)
    clover_odd_inv = np.empty_like(clover_even)
    mass = float(args.mass if args.mass is not None else metadata.get(
        "reference_mass", metadata.get("mass", DEFAULT_MASS)))
    csw = resolve_csw(args.csw, metadata)
    kappa = 1.0 / (2.0 * mass + 8.0)
    params, argv, set_ptrs = make_params(
        lattice, mass, args.atol, csw, dtype, args.max_iter)
    params[MG_NUM_LEVEL] = args.levels
    if args.verbose:
        params[51] = 1
    wrapped = {
        name: NumpyTensor(array) for name, array in {
            "gauge": gauge, "source": source, "solution": solution,
            "even": clover_even, "odd": clover_odd,
            "even_inv": clover_even_inv, "odd_inv": clover_odd_inv,
            "params": params, "argv": argv, "set_ptrs": set_ptrs,
        }.items()
    }

    null_vectors, hop_nn, hop_diag, sitting = ((), (), (), ())
    builder_seconds = 0.0
    coarse_source = "built"
    if args.levels > 1:
        started = time.perf_counter()
        if args.coarse_cache is not None:
            cache_path = resolve_path(
                args.coarse_cache, Path(args.coarse_cache).expanduser())
            if not cache_path.is_file():
                raise FileNotFoundError(cache_path)
            null_vectors, hop_nn, hop_diag, sitting = load_coarse_assets(
                cache_path, dtype)
            coarse_source = str(cache_path)
        else:
            null_vectors, hop_nn, hop_diag, sitting = build_coarse_assets(
                gauge_full, lattice, mass, kappa, csw, dtype, args.levels,
                list(args.dof), list(args.mg_grid), args.nv_iters,
                args.seed, verbose=args.verbose or args.verbose_build,
                local_stencil=args.local_stencil,
                local_stencil_batch_size=args.local_stencil_batch,
                device_name=args.build_device)
        if int(null_vectors[0].shape[0]) != int(args.dof[1]):
            raise ValueError(
                f"coarse cache E={null_vectors[0].shape[0]} does not match "
                f"--dof[1]={args.dof[1]}")
        configure_mg(params, argv, set_ptrs, args.levels, args.atol,
                     args.restart, args.coarse_max_iter,
                     args.coarse_tol_factor,
                     not args.no_deflate, list(null_vectors), list(hop_nn),
                     list(hop_diag), list(sitting))
        builder_seconds = time.perf_counter() - started
        if args.write_coarse_cache is not None:
            cache_output = resolve_path(
                args.write_coarse_cache,
                Path(args.write_coarse_cache).expanduser())
            write_coarse_assets(
                cache_output, null_vectors[0], hop_nn[0], hop_diag[0],
                sitting[0])

    verify_status = None
    qcu.applyInitQcu(wrapped["set_ptrs"], wrapped["params"], wrapped["argv"])
    try:
        params[SET_PLAN] = PLAN_CLOVER
        params[PARITY] = 0
        qcu.applyCloversQcu(
            wrapped["even"], wrapped["even_inv"], wrapped["gauge"],
            wrapped["set_ptrs"], wrapped["params"])
        params[PARITY] = 1
        qcu.applyCloversQcu(
            wrapped["odd"], wrapped["odd_inv"], wrapped["gauge"],
            wrapped["set_ptrs"], wrapped["params"])
        params[SET_PLAN] = PLAN_SOLVER
        if args.levels > 1:
            verify_status = qcu.verifyCloverMultigridQcu(
                wrapped["solution"], wrapped["source"], wrapped["gauge"],
                wrapped["even"], wrapped["odd"], wrapped["even_inv"],
                wrapped["odd_inv"], wrapped["set_ptrs"], wrapped["params"])
        started = time.perf_counter()
        qcu.applyCloverMultigridQcu(
            wrapped["solution"], wrapped["source"], wrapped["gauge"],
            wrapped["even"], wrapped["odd"], wrapped["even_inv"],
            wrapped["odd_inv"], wrapped["set_ptrs"], wrapped["params"])
        wall_seconds = time.perf_counter() - started
    finally:
        qcu.applyEndQcu(wrapped["set_ptrs"], wrapped["params"])

    residual = relative_full_residual(
        source, solution, gauge, clover_even, clover_odd, kappa)
    converged, status, acceptance = convergence_status(
        residual, args.atol, dtype=dtype)
    if verify_status not in (None, 0):
        status = "FAIL"
        converged = False
    if args.output:
        output = resolve_path(args.output, resolve_path(None, args.output))
        write_h5(output, kupl_fermion_to_qcu(solution), metadata={
            "backend": "kupl", "solver": "applyCloverMultigridQcu",
            "mass": mass, "kappa": kappa, "csw": csw, "atol": args.atol,
            "levels": args.levels, "dof": list(args.dof[:args.levels]),
            "mg_grid": list(args.mg_grid),
        })
    report = {
        "backend": "kupl", "solver": "applyCloverMultigridQcu",
        "lattice": list(lattice), "dtype": str(dtype), "mass": mass,
        "kappa": kappa, "mass_normalization": "kappa", "csw": csw,
        "atol": args.atol, "max_iter": args.max_iter, "levels": args.levels,
        "dof": list(args.dof[:args.levels]), "mg_grid": list(args.mg_grid),
        "restart": args.restart, "coarse_max_iter": args.coarse_max_iter,
        "coarse_tol_factor": args.coarse_tol_factor,
        "deflate": not args.no_deflate, "builder_seconds": builder_seconds,
        "coarse_source": coarse_source,
        "wall_seconds": wall_seconds,
        "verify_status": verify_status,
        "final_true_residual": residual, "residual_acceptance": acceptance,
        "residual_source": "independent NumPy full Wilson-Clover reconstruction",
        "converged": converged, "memory": cpu_memory_snapshot(),
        "status": status,
    }
    write_json(resolve_path(args.report, DATA_DIR / args.report), report)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if status != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
