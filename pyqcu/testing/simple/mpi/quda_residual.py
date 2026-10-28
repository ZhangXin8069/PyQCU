#!/usr/bin/env python3
"""Independently validate a saved QUDA solution with a NumPy Wilson-Clover op."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np

if __package__:
    from pyqcu.testing.simple._common import (
        convergence_status,
        lime_metadata,
        qdp_gauge_to_qcu,
        read_h5,
        read_lime_gauge,
        resolve_path,
        write_json,
    )
    from pyqcu.testing.simple._layouts import (
        kupl_fermion_to_qcu,
        kupl_gauge_to_qcu,
        qcu_fermion_to_kupl,
        qcu_gauge_to_kupl,
    )
    from pyqcu.testing.simple.kupl_residual import (
        _apply_hop,
        distributed_relative_residual,
    )
    from pyqcu.testing.simple.numpy_clover import clover_to_kupl, make_clover
else:
    SIMPLE_ROOT = Path(__file__).resolve().parents[1]
    if str(SIMPLE_ROOT) not in sys.path:
        sys.path.insert(0, str(SIMPLE_ROOT))
    from _common import (  # type: ignore[no-redef]
        convergence_status,
        lime_metadata,
        qdp_gauge_to_qcu,
        read_h5,
        read_lime_gauge,
        resolve_path,
        write_json,
    )
    from _layouts import (  # type: ignore[no-redef]
        kupl_fermion_to_qcu,
        kupl_gauge_to_qcu,
        qcu_fermion_to_kupl,
        qcu_gauge_to_kupl,
    )
    from kupl_residual import (  # type: ignore[no-redef]
        _apply_hop,
        distributed_relative_residual,
    )
    from numpy_clover import clover_to_kupl, make_clover  # type: ignore[no-redef]


def _sha256_array(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return digest.hexdigest()


def _validate_rank_count(value: int) -> int:
    value = int(value)
    if value <= 0 or value & (value - 1):
        raise ValueError(f"rank count must be a positive power of two, got {value}")
    return value


def _resolve_grid(
    lattice: Sequence[int], ranks: int, grid: Sequence[int] | None,
) -> tuple[int, int, int, int]:
    if grid is None:
        raise ValueError("explicit --grid is required for this residual check")
    selected = tuple(int(value) for value in grid)
    if len(selected) != 4 or any(value <= 0 for value in selected):
        raise ValueError(f"grid must contain four positive integers: {selected}")
    if math.prod(selected) != int(ranks):
        raise ValueError(f"grid {selected} does not contain {ranks} ranks")
    return selected  # type: ignore[return-value]


def _local_geometry(
    lattice: Sequence[int], grid: Sequence[int], rank: int,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], tuple[int, int, int, int]]:
    if any(int(lattice[axis]) % int(grid[axis]) for axis in range(4)):
        raise ValueError(f"lattice {lattice} is not divisible by grid {grid}")
    local = tuple(int(lattice[axis]) // int(grid[axis]) for axis in range(4))
    if any(value % 2 for value in local):
        raise ValueError(f"local lattice must be even, got {local}")
    coordinate = tuple(int(value) for value in np.unravel_index(rank, grid, order="C"))
    starts = tuple(coordinate[axis] * local[axis] for axis in range(4))
    return local, starts, coordinate  # type: ignore[return-value]


def _slice_last4(
    array: np.ndarray, starts: Sequence[int], local: Sequence[int],
) -> np.ndarray:
    selection = [slice(None)] * (array.ndim - 4)
    selection.extend(
        slice(int(starts[axis]), int(starts[axis] + local[axis]))
        for axis in range(4))
    return np.ascontiguousarray(array[tuple(selection)])


def _read_qcu_gauge(
    path_value: str, lattice: Sequence[int], dtype: np.dtype,
) -> tuple[np.ndarray, str]:
    path = resolve_path(path_value, Path(path_value).expanduser())
    if path.suffix.lower() == ".lime":
        lime_metadata(path)
        gauge = qdp_gauge_to_qcu(read_lime_gauge(path))
        gauge_format = "LIME/SciDAC"
    else:
        with h5py.File(path, "r") as handle:
            if "data" in handle:
                raw = np.asarray(handle["data"])
            elif "gauge" in handle:
                raw = np.asarray(handle["gauge"])
            else:
                raise ValueError(f"{path} has no gauge or data dataset")
        if raw.ndim == 7 and raw.shape[:3] == (3, 3, 4):
            gauge = raw
            gauge_format = "QCU HDF5"
        elif raw.ndim == 7 and raw.shape[0] == 4 and raw.shape[-2:] == (3, 3):
            gauge = qdp_gauge_to_qcu(raw)
            gauge_format = "QDP HDF5"
        else:
            raise ValueError(f"unsupported gauge shape {raw.shape}")
    gauge = _slice_last4(gauge, (0, 0, 0, 0), lattice)
    return np.ascontiguousarray(gauge.astype(dtype, copy=False)), gauge_format


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gauge", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--solution", required=True)
    parser.add_argument("--lat", nargs=4, type=int, required=True)
    parser.add_argument("--grid", nargs=4, type=int, default=None)
    parser.add_argument("--ranks", type=int, default=None)
    parser.add_argument("--dtype", choices=("c64", "c128"), default="c64")
    parser.add_argument("--mass", type=float, required=True)
    parser.add_argument("--csw", type=float, required=True)
    parser.add_argument("--clover", default=None)
    parser.add_argument("--atol", type=float, default=1.0e-6)
    parser.add_argument("--full-site-debug", action="store_true")
    parser.add_argument("--report", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    from mpi4py import MPI

    args = _parse_args(argv)
    comm = MPI.COMM_WORLD
    rank = int(comm.Get_rank())
    started = time.perf_counter()
    try:
        size = _validate_rank_count(int(comm.Get_size()))
        if args.ranks is not None and int(args.ranks) != size:
            raise ValueError(
                f"--ranks={args.ranks} does not match MPI size {size}")
        lattice = tuple(int(value) for value in args.lat)
        grid = _resolve_grid(lattice, size, args.grid)
        local, starts, coordinate = _local_geometry(lattice, grid, rank)
        dtype = np.dtype(np.complex64 if args.dtype == "c64" else np.complex128)

        gauge_full, gauge_format = _read_qcu_gauge(args.gauge, lattice, dtype)
        source_full, _, source_format = read_source_compatible(args.source, lattice, dtype)
        solution_full, _, solution_format = read_source_compatible(
            args.solution, lattice, dtype)
        gauge_qcu = _slice_last4(gauge_full, starts, local)
        gauge_local = np.ascontiguousarray(qcu_gauge_to_kupl(gauge_qcu))
        source_spin = np.ascontiguousarray(
            qcu_fermion_to_kupl(_slice_last4(source_full, starts, local)).reshape(
                2, 4, 3, *local[:3], local[3] // 2))
        solution_spin = np.ascontiguousarray(
            qcu_fermion_to_kupl(_slice_last4(solution_full, starts, local)).reshape(
                2, 4, 3, *local[:3], local[3] // 2))
        parity_shape = (*local[:3], local[3] // 2)
        source_local = np.ascontiguousarray(
            source_spin.reshape(2, 12, *parity_shape))
        solution_local = np.ascontiguousarray(
            solution_spin.reshape(2, 12, *parity_shape))
        del source_spin, solution_spin
        del source_full, solution_full

        kappa = 1.0 / (2.0 * float(args.mass) + 8.0)
        clover_source = "numpy-PyQCU-torch-convention-I-plus-T"
        clover_parity = None
        clover_local = None
        if args.clover:
            clover_parity = _load_clover_parity(
                args.clover, starts, local)
            clover_source = "cuda-applyCloversQcu"
        else:
            # Clover is a four-plaquette operator.  It must be constructed on
            # the global periodic gauge and only then sliced to the rank block;
            # reconstructing it from a rank-local gauge imposes wrong periodic
            # boundaries and fails every multi-rank true-residual check.
            clover_full = make_clover(
                gauge_full, kappa, float(args.csw), convention="torch")
            clover_full = _add_clover_identity(clover_full)
            clover_local = _slice_last4(clover_full, starts, local)
            clover_parity = clover_to_kupl(clover_local)
            del clover_full
        del gauge_full
        full_site_residual = None
        if args.full_site_debug and clover_local is not None:
            full_site_residual = _full_site_relative_residual(
                solution_local, source_local, gauge_qcu, clover_local,
                kappa, comm, coordinate, grid)
        del clover_local
        residual = distributed_relative_residual(
            solution_local, source_local, gauge_local,
            clover_parity[0], clover_parity[1], kappa,
            comm, coordinate, grid)
        converged, status, acceptance = convergence_status(
            residual, float(args.atol), dtype=dtype)
        local_hashes = {
            "gauge": _sha256_array(gauge_local),
            "source": _sha256_array(source_local),
            "solution": _sha256_array(solution_local),
            "clover": _sha256_array(clover_parity),
        }
        failures = int(comm.allreduce(int(not converged), op=MPI.MAX))
        if failures:
            status = "FAIL"
        report = {
            "status": status,
            "residual_source": (
                "independent periodic NumPy Wilson-Clover operator with MPI halo exchange"),
            "relative_true_residual": float(residual),
            "full_site_relative_true_residual": full_site_residual,
            "atol": float(args.atol),
            "acceptance": float(acceptance),
            "converged": bool(converged),
            "ranks": size,
            "grid": list(grid),
            "lattice": list(lattice),
            "local_lattice": list(local),
            "local_starts": list(starts),
            "dtype": str(dtype),
            "mass": float(args.mass),
            "csw": float(args.csw),
            "kappa": float(kappa),
            "clover_source": clover_source,
            "inputs": {
                "gauge": str(resolve_path(args.gauge, Path(args.gauge))),
                "gauge_format": gauge_format,
                "source": str(resolve_path(args.source, Path(args.source))),
                "source_format": source_format,
                "solution": str(resolve_path(args.solution, Path(args.solution))),
                "solution_format": solution_format,
            },
            "rank_hashes": local_hashes,
            "elapsed_seconds": 0.0,
        }
    except Exception as exc:
        report = {
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
            "rank": rank,
        }
        failures = 1

    failures = int(comm.allreduce(failures, op=MPI.MAX))
    if rank == 0:
        report["elapsed_seconds"] = time.perf_counter() - started
        report["status"] = "PASS" if failures == 0 else "FAIL"
        output = Path(args.report).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
        write_json(temporary, report)
        os.replace(temporary, output)
        print(json.dumps(report, ensure_ascii=False))
    return 0 if failures == 0 else 1


def read_source_compatible(path_value: str, lattice: Sequence[int], dtype: np.dtype):
    path = resolve_path(path_value, Path(path_value).expanduser())
    array = np.asarray(read_h5(path))
    if array.shape != (4, 3, *(int(value) for value in lattice)):
        raise ValueError(f"field shape {array.shape} does not match lattice {lattice}")
    return np.ascontiguousarray(array.astype(dtype, copy=False)), {}, "HDF5"


def _full_site_relative_residual(
    solution: np.ndarray,
    source: np.ndarray,
    gauge: np.ndarray,
    clover: np.ndarray,
    kappa: float,
    comm,
    coordinate: tuple[int, int, int, int],
    grid: tuple[int, int, int, int],
) -> float:
    psi_parity = kupl_fermion_to_qcu(np.asarray(solution))
    rhs_parity = kupl_fermion_to_qcu(np.asarray(source))
    psi = np.ascontiguousarray(psi_parity.reshape(12, *psi_parity.shape[2:]))
    rhs = np.ascontiguousarray(rhs_parity.reshape(12, *rhs_parity.shape[2:]))
    links = np.ascontiguousarray(np.asarray(gauge))
    dslash = _apply_hop(psi, links, coordinate, grid, comm)
    dims = psi.shape[1:]
    xs, ys, zs, ts = np.indices(dims, dtype=np.int16)
    parity = ((xs + ys + zs + ts) & 1).reshape(-1)
    flat_psi = psi.reshape(12, -1)
    flat_rhs = rhs.reshape(12, -1)
    flat_dslash = dslash.reshape(12, -1)
    numerator = 0.0
    denominator = 0.0
    matrix_all = clover.reshape(2, 12, 12, -1)
    for target in range(2):
        mask = parity == target
        matrix = np.asarray(matrix_all[target], dtype=np.complex128)
        solution_block = np.asarray(flat_psi[:, mask], dtype=np.complex128)
        residual_block = (
            np.einsum("rsn,sn->rn", matrix, solution_block) -
            kappa * flat_dslash[:, mask].astype(np.complex128) -
            flat_rhs[:, mask].astype(np.complex128))
        numerator += float(np.linalg.norm(residual_block) ** 2)
        denominator += float(np.linalg.norm(flat_rhs[:, mask]) ** 2)
    numerator = float(comm.allreduce(numerator))
    denominator = float(comm.allreduce(denominator))
    return float(np.sqrt(numerator) /
                 max(np.sqrt(denominator), np.finfo(float).tiny))


def _load_clover_parity(
    path_value: str, starts: Sequence[int], local: Sequence[int],
) -> np.ndarray:
    path = resolve_path(path_value, Path(path_value).expanduser())
    parity_starts = (*starts[:3], int(starts[3]) // 2)
    parity_local = (*local[:3], int(local[3]) // 2)
    output = np.empty((2, 12, 12, *parity_local), dtype=np.complex64)
    with h5py.File(path, "r") as handle:
        for parity, name in enumerate(("even", "odd")):
            dataset = handle[name]
            selection = [slice(None)] * dataset.ndim
            for axis in range(4):
                selection[4 + axis] = (
                slice(int(parity_starts[axis]),
                      int(parity_starts[axis] + parity_local[axis]))
                )
            selected = np.asarray(dataset[tuple(selection)],
                                  dtype=np.complex64)
            output[parity] = selected.reshape(12, 12, *parity_local)
    return np.ascontiguousarray(output)


def _add_clover_identity(clover: np.ndarray) -> np.ndarray:
    """Convert the additive Clover term T into the onsite matrix A=I+T."""
    array = np.asarray(clover)
    if array.ndim != 8 or array.shape[:4] != (4, 3, 4, 3):
        raise ValueError(f"Clover must be [4,3,4,3,x,y,z,t], got {array.shape}")
    matrix = np.array(array.reshape(12, 12, *array.shape[-4:]), copy=True)
    diagonal = np.arange(12)
    matrix[diagonal, diagonal] += 1.0
    return np.ascontiguousarray(matrix.reshape(array.shape))


if __name__ == "__main__":
    raise SystemExit(main())
