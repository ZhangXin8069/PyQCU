#!/usr/bin/env python3
"""Minimal KUPL Clover BiStabCG example with an independent NumPy residual."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from ._common import (DATA_DIR, DEFAULT_ATOL, DEFAULT_MASS,
                          convergence_status, estimated_clover_gflops,
                          lime_metadata, qcu_fermion_to_kupl,
                          qcu_gauge_to_kupl, qdp_gauge_to_qcu,
                          read_h5, read_h5_metadata, read_qdp_gauge,
                          kupl_fermion_to_qcu, resolve_csw, resolve_path,
                          write_h5, write_json)
except ImportError:
    from _common import (DATA_DIR, DEFAULT_ATOL, DEFAULT_MASS,
                         convergence_status, estimated_clover_gflops,
                         lime_metadata, qcu_fermion_to_kupl,
                         qcu_gauge_to_kupl, qdp_gauge_to_qcu, read_h5,
                         read_h5_metadata, read_qdp_gauge,
                         kupl_fermion_to_qcu, resolve_csw, resolve_path,
                         write_h5, write_json)

if str(Path(__file__).resolve().parents[3]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


PARAMS_SIZE = 58
ARGV_SIZE = 8
SET_PTRS_SIZE = 100
LAT_X, LAT_Y, LAT_Z, LAT_T = range(4)
LAT_XYZT, GRID_X, GRID_Y, GRID_Z, GRID_T = range(4, 9)
PARITY, NODE_RANK, NODE_SIZE = 9, 10, 11
MAX_ITER, DATA_TYPE, SET_INDEX, SET_PLAN = 13, 14, 15, 16
LAT_C64, LAT_C128 = 2, 3
PLAN_CLOVER, PLAN_SOLVER = 2, 1
VERBOSE = 51
MASS, ATOL, CSW = 0, 1, 7


class NumpyTensor:
    """Small tensor protocol adapter for the shared KUPL ctypes binding."""

    def __init__(self, array: np.ndarray):
        self.array = np.ascontiguousarray(array)

    def is_contiguous(self) -> bool:
        return bool(self.array.flags.c_contiguous)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.array.shape)

    def data_ptr(self) -> int:
        return int(self.array.ctypes.data)


def kupl_backend():
    try:
        from pyqcu.kupl import qcu
        return qcu
    except (FileNotFoundError, OSError) as exc:
        return exc


def cpu_memory_snapshot() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    resident_kib = None
    try:
        for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                resident_kib = int(line.split()[1])
                break
    except OSError:
        pass
    return {
        "available": True,
        "max_rss_kib": int(usage.ru_maxrss),
        "resident_kib": resident_kib,
        "omp_threads": os.environ.get("KUPL_OMP_THREADS"),
    }


def gamma_matrix(direction: int) -> np.ndarray:
    i = 1j
    if direction == 0:
        return np.array([[0, 0, 0, i], [0, 0, i, 0],
                         [0, -i, 0, 0], [-i, 0, 0, 0]], dtype=np.complex128)
    if direction == 1:
        return np.array([[0, 0, 0, -1], [0, 0, 1, 0],
                         [0, 1, 0, 0], [-1, 0, 0, 0]], dtype=np.complex128)
    if direction == 2:
        return np.array([[0, 0, i, 0], [0, 0, 0, -i],
                         [-i, 0, 0, 0], [0, i, 0, 0]], dtype=np.complex128)
    return np.array([[0, 0, 1, 0], [0, 0, 0, 1],
                     [1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.complex128)


def hop_gamma(direction: int, sign: int) -> np.ndarray:
    return np.eye(4, dtype=np.complex128) - sign * gamma_matrix(direction)


def bare_dslash_reference(field: np.ndarray, gauge: np.ndarray,
                          target_parity: int) -> np.ndarray:
    """Apply KUPL's D_bare to one parity using only NumPy."""
    _, _, x_dim, y_dim, z_dim, ht = field.shape
    source_parity = 1 - target_parity
    output = np.zeros((12, x_dim, y_dim, z_dim, ht), dtype=np.complex128)
    for x in range(x_dim):
        for y in range(y_dim):
            for z in range(z_dim):
                for t2 in range(ht):
                    time_coord = 2 * t2 + (((x + y + z) & 1) ^ target_parity)
                    coordinate = np.array([x, y, z, time_coord], dtype=np.int64)
                    extent = np.array([x_dim, y_dim, z_dim, 2 * ht],
                                      dtype=np.int64)
                    for direction in range(4):
                        forward = coordinate.copy()
                        forward[direction] = (forward[direction] + 1) % extent[direction]
                        forward_field = field[source_parity, :, forward[0],
                                              forward[1], forward[2],
                                              forward[3] // 2].reshape(4, 3)
                        link = gauge[target_parity, :, :, direction,
                                     x, y, z, t2]
                        transported = np.einsum(
                            "ij,sj->si", link, forward_field)
                        output[:, x, y, z, t2] += np.einsum(
                            "st,tc->sc", hop_gamma(direction, 1),
                            transported).reshape(12)

                        backward = coordinate.copy()
                        backward[direction] = (backward[direction] - 1) % extent[direction]
                        backward_field = field[source_parity, :, backward[0],
                                               backward[1], backward[2],
                                               backward[3] // 2].reshape(4, 3)
                        backward_link = gauge[source_parity, :, :, direction,
                                              backward[0], backward[1],
                                              backward[2], backward[3] // 2]
                        transported = np.einsum(
                            "ji,sj->si", np.conj(backward_link),
                            backward_field)
                        output[:, x, y, z, t2] += np.einsum(
                            "st,tc->sc", hop_gamma(direction, -1),
                            transported).reshape(12)
    return output


def relative_full_residual(source: np.ndarray, solution: np.ndarray,
                           gauge: np.ndarray, clover_even: np.ndarray,
                           clover_odd: np.ndarray, kappa: float) -> float:
    clover = (clover_even, clover_odd)
    numerator = 0.0
    denominator = 0.0
    for parity in range(2):
        onsite = clover[parity].reshape(12, 12, -1)
        parity_source = source[parity].reshape(12, -1)
        parity_solution = solution[parity].reshape(12, -1)
        matrix_solution = np.einsum(
            "rsn,sn->rn", onsite.astype(np.complex128), parity_solution)
        residual = (matrix_solution - kappa *
                    bare_dslash_reference(
                        solution.astype(np.complex128),
                        gauge.astype(np.complex128), parity).reshape(12, -1) -
                    parity_source)
        numerator += float(np.linalg.norm(residual) ** 2)
        denominator += float(np.linalg.norm(parity_source) ** 2)
    return float(np.sqrt(numerator) / max(np.sqrt(denominator), np.finfo(float).tiny))


def load_gauge(path_value: str | None, lattice: tuple[int, int, int, int],
               dtype: np.dtype) -> tuple[np.ndarray, dict[str, Any]]:
    if path_value is None:
        eye = np.eye(3, dtype=dtype)
        gauge = np.broadcast_to(
            eye[:, :, None, None, None, None, None],
            (3, 3, 4, *lattice)).copy()
        return gauge, {}
    path = resolve_path(path_value, resolve_path(None, path_value))
    if path.suffix == ".lime":
        metadata = lime_metadata(path)
        gauge = qdp_gauge_to_qcu(read_qdp_gauge(path))
    elif path.suffix in (".h5", ".hdf5"):
        metadata = read_h5_metadata(path)
        gauge = read_h5(path)
    else:
        raw = np.load(path, allow_pickle=False)
        metadata = {}
        gauge = raw if raw.ndim == 7 and raw.shape[:3] == (3, 3, 4) \
            else qdp_gauge_to_qcu(raw)
    if tuple(gauge.shape[-4:]) != tuple(lattice):
        raise ValueError(f"gauge lattice {tuple(gauge.shape[-4:])} does not "
                         f"match --lat {tuple(lattice)}")
    return np.ascontiguousarray(gauge.astype(dtype, copy=False)), metadata


def load_source(path_value: str | None,
                lattice: tuple[int, int, int, int], seed: int,
                dtype: np.dtype) -> np.ndarray:
    if path_value is None:
        rng = np.random.default_rng(seed)
        return (rng.normal(size=(4, 3, *lattice)) +
                1j * rng.normal(size=(4, 3, *lattice))).astype(dtype)
    path = resolve_path(path_value, resolve_path(None, path_value))
    source = read_h5(path)
    if tuple(source.shape) != (4, 3, *lattice):
        raise ValueError(f"source shape {source.shape} does not match "
                         f"(4,3,{','.join(map(str, lattice))})")
    return np.ascontiguousarray(source.astype(dtype, copy=False))


def make_params(lattice: tuple[int, int, int, int], mass: float,
                atol: float, csw: float, dtype: np.dtype,
                max_iter: int,
                grid: tuple[int, int, int, int] = (1, 1, 1, 1),
                rank: int = 0,
                size: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    params = np.zeros(PARAMS_SIZE, dtype=np.int32)
    params[LAT_X:LAT_T + 1] = lattice
    params[LAT_XYZT] = int(np.prod(lattice))
    params[GRID_X:GRID_T + 1] = grid
    params[PARITY] = 0
    params[NODE_RANK] = int(rank)
    params[NODE_SIZE] = int(size)
    params[MAX_ITER] = int(max_iter)
    params[DATA_TYPE] = LAT_C64 if dtype == np.dtype(np.complex64) else LAT_C128
    params[SET_INDEX] = 0
    params[SET_PLAN] = PLAN_CLOVER
    params[VERBOSE] = 0
    real_dtype = np.float32 if dtype == np.dtype(np.complex64) else np.float64
    argv = np.zeros(ARGV_SIZE, dtype=real_dtype)
    argv[MASS] = mass
    argv[ATOL] = atol
    argv[CSW] = csw
    return params, argv, np.zeros(SET_PTRS_SIZE, dtype=np.int64)


def prebind_kupl_inputs(qcu: Any, lattice: tuple[int, int, int, int],
                        mass: float, atol: float, csw: float,
                        dtype: np.dtype, max_iter: int,
                        grid: tuple[int, int, int, int], rank: int,
                        size: int) -> None:
    """Establish process placement before the caller first-touches inputs."""
    params, argv, set_ptrs = make_params(
        lattice, mass, atol, csw, dtype, max_iter, grid=grid, rank=rank,
        size=size)
    params_tensor = NumpyTensor(params)
    argv_tensor = NumpyTensor(argv)
    set_ptrs_tensor = NumpyTensor(set_ptrs)
    qcu.applyInitQcu(set_ptrs_tensor, params_tensor, argv_tensor)
    try:
        return
    finally:
        try:
            qcu.applyEndQcu(set_ptrs_tensor, params_tensor)
        finally:
            set_ptrs[params[SET_INDEX]] = 0
            params[SET_INDEX] += 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lat", nargs=4, type=int, default=(8, 8, 16, 16))
    parser.add_argument("--mass", type=float, default=None)
    parser.add_argument("--csw", type=float, default=None)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--dtype", choices=("c64", "c128"), default="c64")
    parser.add_argument("--gauge", default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    parser.add_argument("--report", default="performance_kupl_bistabcg.json")
    args = parser.parse_args(argv)

    backend = kupl_backend()
    if isinstance(backend, Exception):
        print(json.dumps({"status": "SKIP", "reason": str(backend)}))
        return 0
    qcu = backend
    dtype = np.dtype(np.complex64 if args.dtype == "c64" else np.complex128)
    lattice = tuple(args.lat)
    if lattice[3] % 2:
        parser.error("KUPL requires an even time extent")

    gauge_full, metadata = load_gauge(args.gauge, lattice, dtype)
    gauge = qcu_gauge_to_kupl(gauge_full)
    source_full = load_source(args.source, lattice, args.seed, dtype)
    source = qcu_fermion_to_kupl(source_full)
    solution = np.zeros_like(source)
    clover_even = np.empty((4, 3, 4, 3, *lattice[:3], lattice[3] // 2),
                           dtype=dtype)
    clover_odd = np.empty_like(clover_even)
    clover_even_inv = np.empty_like(clover_even)
    clover_odd_inv = np.empty_like(clover_even)

    mass = float(args.mass if args.mass is not None else metadata.get(
        "reference_mass", metadata.get("mass", DEFAULT_MASS)))
    csw = resolve_csw(args.csw, metadata)
    kappa = 1.0 / (2.0 * mass + 8.0)
    params, argv, set_ptrs = make_params(
        lattice, mass, args.atol, csw, dtype, args.max_iter)
    wrapped = {
        name: NumpyTensor(array) for name, array in {
            "gauge": gauge, "source": source, "solution": solution,
            "even": clover_even, "odd": clover_odd,
            "even_inv": clover_even_inv, "odd_inv": clover_odd_inv,
            "params": params, "argv": argv, "set_ptrs": set_ptrs,
        }.items()
    }

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
        started = time.perf_counter()
        qcu.applyCloverBistabCgQcu(
            wrapped["solution"], wrapped["source"], wrapped["gauge"],
            wrapped["even"], wrapped["odd"], wrapped["even_inv"],
            wrapped["odd_inv"], wrapped["set_ptrs"], wrapped["params"])
        wall_seconds = time.perf_counter() - started
        iterations = qcu.getCloverBistabCgIterationsQcu(
            wrapped["set_ptrs"], wrapped["params"])
    finally:
        qcu.applyEndQcu(wrapped["set_ptrs"], wrapped["params"])

    residual = relative_full_residual(
        source, solution, gauge, clover_even, clover_odd, kappa)
    converged, status, acceptance = convergence_status(
        residual, args.atol, dtype=dtype)
    solution_full = None
    if args.output:
        solution_full = kupl_fermion_to_qcu(solution)
        output = resolve_path(args.output, resolve_path(None, args.output))
        write_h5(output, solution_full, metadata={
            "backend": "kupl", "solver": "applyCloverBistabCgQcu",
            "mass": mass, "kappa": kappa, "csw": csw, "atol": args.atol,
            "layout": "[spin,color,x,y,z,t]",
        })
    gflops = estimated_clover_gflops(
        lattice, iterations, wall_seconds, operator="schur")
    report = {
        "backend": "kupl", "solver": "applyCloverBistabCgQcu",
        "lattice": list(lattice), "dtype": str(dtype), "mass": mass,
        "kappa": kappa, "mass_normalization": "kappa", "csw": csw,
        "atol": args.atol, "max_iter": args.max_iter,
        "iterations": iterations, "wall_seconds": wall_seconds,
        "seconds_per_iteration": wall_seconds / iterations if iterations else None,
        "final_true_residual": residual, "residual_acceptance": acceptance,
        "residual_source": "independent NumPy full Wilson-Clover reconstruction",
        "converged": converged, "gflops": gflops,
        "memory": cpu_memory_snapshot(), "status": status,
    }
    report_path = resolve_path(args.report, DATA_DIR / args.report)
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if status != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
