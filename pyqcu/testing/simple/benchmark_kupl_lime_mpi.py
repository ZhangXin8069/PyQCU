#!/usr/bin/env python3
"""Benchmark KUPL Clover setup and BiStabCG on an MPI-split LIME gauge."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any, Callable, Iterator, Sequence

import h5py
import numpy as np
from mpi4py import MPI

try:
    from ._common import (
        DEFAULT_ATOL, DEFAULT_CSW, DEFAULT_MASS, lime_metadata,
        kupl_fermion_to_qcu, qcu_fermion_to_kupl, qcu_gauge_to_kupl,
        qdp_gauge_to_qcu, read_h5, read_h5_metadata, read_qdp_gauge,
        resolve_path, solver_acceptance, write_json)
except ImportError:
    from _common import (
        DEFAULT_ATOL, DEFAULT_CSW, DEFAULT_MASS, lime_metadata,
        kupl_fermion_to_qcu, qcu_fermion_to_kupl, qcu_gauge_to_kupl,
        qdp_gauge_to_qcu, read_h5, read_h5_metadata, read_qdp_gauge,
        resolve_path, solver_acceptance, write_json)


PARAMS_SIZE = 58
ARGV_SIZE = 8
SET_PTRS_SIZE = 100

LAT_X, LAT_Y, LAT_Z, LAT_T = range(4)
LAT_XYZT, GRID_X, GRID_Y, GRID_Z, GRID_T = range(4, 9)
PARITY, NODE_RANK, NODE_SIZE = 9, 10, 11
MAX_ITER, DATA_TYPE, SET_INDEX, SET_PLAN = 13, 14, 15, 16
VERBOSE = 51
SEED = 52
USE_INITIAL_GUESS = 57

MASS, ATOL, CSW = 0, 1, 7
LAT_C64, LAT_C128 = 2, 3
PLAN_CLOVER, PLAN_SOLVER = 2, 1

TARGET_PARITY = 0
DEFAULT_LATTICE = (24, 24, 24, 64)
DEFAULT_GRID = (1, 1, 1, 1)
DEFAULT_SEED = 42


class NumpyTensor:
    """Tensor protocol adapter required by ``pyqcu.kupl.qcu``."""

    def __init__(self, array: np.ndarray):
        self.array = np.ascontiguousarray(array)

    def is_contiguous(self) -> bool:
        return bool(self.array.flags.c_contiguous)

    def data_ptr(self) -> int:
        return int(self.array.ctypes.data)


@dataclass
class ControlState:
    """One private flat-ABI control state for a benchmark lifecycle."""

    params: np.ndarray
    argv: np.ndarray
    set_ptrs: np.ndarray


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _resolved_input(value: str) -> Path:
    return resolve_path(value, Path(value).expanduser())


def _file_sha256_on_root(path: Path, comm: MPI.Comm) -> str:
    digest = _sha256_file(path) if comm.Get_rank() == 0 else None
    return str(comm.bcast(digest, root=0))


def _metadata_float(metadata: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = metadata.get(key)
        if value is None:
            continue
        try:
            result = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(result):
            return result
    return None


def _metadata_lattice(metadata: dict[str, Any]) -> tuple[int, int, int, int] | None:
    value = metadata.get("lattice_xyzt", metadata.get("dims"))
    if value is None:
        return None
    if isinstance(value, str):
        parts = value.replace(",", " ").split()
    else:
        parts = list(value)
    if len(parts) != 4:
        return None
    return tuple(int(item) for item in parts)  # type: ignore[return-value]


def _load_gauge(path: Path, lattice: tuple[int, int, int, int],
                dtype: np.dtype, comm: MPI.Comm
                ) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".lime":
        metadata = lime_metadata(path)
        gauge = qdp_gauge_to_qcu(read_qdp_gauge(path))
        input_format = "LIME/SciDAC"
    elif suffix in (".h5", ".hdf5"):
        metadata = read_h5_metadata(path)
        raw = read_h5(path)
        if raw.shape == (4, lattice[3], lattice[2], lattice[1], lattice[0],
                         3, 3):
            gauge = qdp_gauge_to_qcu(raw)
        else:
            gauge = np.asarray(raw)
        input_format = "HDF5"
    else:
        raise ValueError("--gauge must be a .lime, .h5, or .hdf5 file")

    metadata_lattice = _metadata_lattice(metadata)
    if metadata_lattice is not None and metadata_lattice != lattice:
        raise ValueError(
            f"gauge metadata lattice {metadata_lattice} does not match --lat "
            f"{lattice}")
    expected = (3, 3, 4, *lattice)
    if gauge.shape != expected:
        raise ValueError(f"gauge shape {gauge.shape} does not match {expected}")

    file_sha256 = _file_sha256_on_root(path, comm)
    info = {
        "format": input_format,
        "path": str(path),
        "sha256": file_sha256,
        "global_qcu_shape": [3, 3, 4, *lattice],
        "metadata": _jsonable(metadata),
    }
    return np.ascontiguousarray(gauge.astype(dtype, copy=False)), metadata, info


def _load_source_slab(path: Path, slices: tuple[slice, ...],
                      lattice: tuple[int, int, int, int], dtype: np.dtype,
                      comm: MPI.Comm
                      ) -> tuple[np.ndarray, dict[str, Any]]:
    with h5py.File(path, "r") as handle:
        if "data" not in handle:
            raise ValueError(f"source HDF5 file has no 'data' dataset: {path}")
        source = np.asarray(handle["data"][...])
    expected = (4, 3, *lattice)
    if source.shape != expected:
        raise ValueError(f"source shape {source.shape} does not match {expected}")
    source = np.ascontiguousarray(source.astype(dtype, copy=False))
    local = np.ascontiguousarray(source[(slice(None), slice(None), *slices)])
    info = {
        "kind": "h5",
        "path": str(path),
        "dataset": "data",
        "sha256": _file_sha256_on_root(path, comm),
        "global_shape": [4, 3, *lattice],
        "global_data_sha256": _sha256_array(source),
    }
    return local, info


def _make_source_slab(slices: tuple[slice, ...],
                      lattice: tuple[int, int, int, int], dtype: np.dtype,
                      seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    shape = (4, 3, *lattice)
    # Match the QUDA and CUDA-QCU benchmarks bit-for-bit: one PCG64
    # float64 stream is drawn globally and then cast to the requested dtype.
    source = (rng.standard_normal(shape) +
              1j * rng.standard_normal(shape)).astype(dtype)
    source = np.ascontiguousarray(source)
    local = np.ascontiguousarray(source[(slice(None), slice(None), *slices)])
    source_sha256 = _sha256_array(source)
    info = {
        "kind": "seeded",
        "path": None,
        "seed": int(seed),
        "sha256": source_sha256,
        "global_shape": [4, 3, *lattice],
        "global_data_sha256": source_sha256,
    }
    return local, info


def _gather_global_blocks(
        local: np.ndarray, lattice: tuple[int, int, int, int],
        grid: tuple[int, int, int, int], comm: MPI.Comm) -> np.ndarray | None:
    """Gather rank-local ``...xyzt`` blocks into one global array on root."""
    blocks = comm.gather(np.ascontiguousarray(local), root=0)
    comm.Barrier()
    if comm.Get_rank() != 0:
        return None
    prefix = local.shape[:-4]
    whole = np.empty((*prefix, *lattice), dtype=local.dtype)
    local_shape = local.shape[-4:]
    for block_rank, block in enumerate(blocks or []):
        coordinate = np.unravel_index(block_rank, grid, order="C")
        selection = [slice(None)] * len(prefix)
        selection.extend(
            slice(int(coordinate[axis]) * local_shape[axis],
                  (int(coordinate[axis]) + 1) * local_shape[axis])
            for axis in range(4))
        whole[tuple(selection)] = block
    return whole


def _independent_true_residual(
        gauge_full: np.ndarray, source: np.ndarray, solution: np.ndarray,
        lattice: tuple[int, int, int, int],
        grid: tuple[int, int, int, int], mass: float, csw: float,
        threads: int, comm: MPI.Comm) -> float:
    """Compute a global full Wilson-Clover residual independently of KUPL."""
    local_solution = kupl_fermion_to_qcu(solution)
    local_source = kupl_fermion_to_qcu(source)
    global_solution = _gather_global_blocks(
        local_solution, lattice, grid, comm)
    global_source = _gather_global_blocks(local_source, lattice, grid, comm)
    value = None
    if comm.Get_rank() == 0:
        import torch
        from pyqcu import dslash

        torch.set_num_threads(max(1, int(threads) * int(comm.Get_size())))
        complex_dtype = (
            torch.complex64 if gauge_full.dtype == np.dtype(np.complex64)
            else torch.complex128)
        real_dtype = (
            torch.float32 if complex_dtype == torch.complex64
            else torch.float64)
        gauge = torch.from_numpy(np.ascontiguousarray(gauge_full))
        src = torch.from_numpy(global_solution).to(dtype=complex_dtype)
        rhs = torch.from_numpy(global_source).to(dtype=complex_dtype)
        kappa = 1.0 / (2.0 * float(mass) + 8.0)
        kappa_tensor = torch.tensor([kappa], dtype=real_dtype)
        u0 = torch.ones([1], dtype=real_dtype)
        clover = dslash.make_clover(
            gauge, kappa=kappa_tensor, u_0=u0, csw=float(csw))
        residual = (
            dslash.give_wilson(src, gauge, kappa_tensor, u0, True) +
            dslash.give_clover(src, clover) - rhs)
        numerator = float(torch.linalg.vector_norm(residual))
        denominator = max(float(torch.linalg.vector_norm(rhs)), 1.0e-30)
        value = numerator / denominator
        if not math.isfinite(value):
            raise RuntimeError(f"nonfinite independent true residual: {value!r}")
    return float(comm.bcast(value, root=0))


def _local_slices(coordinate: Sequence[int], local: Sequence[int]
                  ) -> tuple[slice, slice, slice, slice]:
    return tuple(
        slice(int(coord) * int(extent), (int(coord) + 1) * int(extent))
        for coord, extent in zip(coordinate, local, strict=True)
    )  # type: ignore[return-value]


def _make_control(local: tuple[int, int, int, int],
                  grid: tuple[int, int, int, int], rank: int, size: int,
                  mass: float, atol: float, csw: float, dtype: np.dtype,
                  max_iter: int, seed: int, plan: int,
                  parity: int) -> ControlState:
    params = np.zeros(PARAMS_SIZE, dtype=np.int32)
    params[LAT_X:LAT_T + 1] = local
    params[LAT_XYZT] = math.prod(local)
    params[GRID_X:GRID_T + 1] = grid
    params[PARITY] = int(parity)
    params[NODE_RANK] = int(rank)
    params[NODE_SIZE] = int(size)
    params[MAX_ITER] = int(max_iter)
    params[DATA_TYPE] = LAT_C64 if dtype == np.dtype(np.complex64) else LAT_C128
    params[SET_INDEX] = 0
    params[SET_PLAN] = int(plan)
    params[VERBOSE] = 0
    params[SEED] = int(seed)
    params[USE_INITIAL_GUESS] = 0

    real_dtype = np.float32 if dtype == np.dtype(np.complex64) else np.float64
    argv = np.zeros(ARGV_SIZE, dtype=real_dtype)
    argv[MASS] = mass
    argv[ATOL] = atol
    argv[CSW] = csw
    set_ptrs = np.zeros(SET_PTRS_SIZE, dtype=np.int64)
    return ControlState(
        params=np.ascontiguousarray(params),
        argv=np.ascontiguousarray(argv),
        set_ptrs=np.ascontiguousarray(set_ptrs),
    )


@contextmanager
def _backend_slot(qcu: Any, state: ControlState) -> Iterator[None]:
    """Run one complete QCU lifecycle and advance its private slot on exit."""
    params = NumpyTensor(state.params)
    argv = NumpyTensor(state.argv)
    set_ptrs = NumpyTensor(state.set_ptrs)
    qcu.applyInitQcu(set_ptrs, params, argv)
    try:
        yield
    finally:
        try:
            qcu.applyEndQcu(set_ptrs, params)
        finally:
            set_ptrs.array[params.array[SET_INDEX]] = 0
            params.array[SET_INDEX] += 1


def _timed_operation(comm: MPI.Comm,
                     operation: Callable[[], Any]) -> tuple[float, Any]:
    comm.Barrier()
    started = time.perf_counter()
    value = operation()
    elapsed = time.perf_counter() - started
    comm.Barrier()
    return float(elapsed), value


def _summary(values: Sequence[float]) -> dict[str, Any]:
    samples = [float(value) for value in values]
    median = statistics.median(samples)
    mad = statistics.median(abs(value - median) for value in samples)
    return {
        "seconds": samples,
        "samples": len(samples),
        "median_seconds": float(median),
        "mad_seconds": float(mad),
        "min_seconds": float(min(samples)),
        "max_seconds": float(max(samples)),
    }


def _rank_environment(rank: int, coordinate: Sequence[int],
                      local: tuple[int, int, int, int], threads: int
                      ) -> dict[str, Any]:
    try:
        affinity = sorted(int(cpu) for cpu in os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = []
    return {
        "rank": int(rank),
        "coordinate": [int(value) for value in coordinate],
        "host": os.uname().nodename,
        "local_shape": list(local),
        "threads_requested": int(threads),
        "affinity_cpus": affinity,
        "affinity_count": len(affinity),
        "omp": {
            "KUPL_OMP_THREADS": os.environ.get("KUPL_OMP_THREADS"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OMP_PROC_BIND": os.environ.get("OMP_PROC_BIND"),
            "OMP_PLACES": os.environ.get("OMP_PLACES"),
            "KUPL_NUMA_BIND": os.environ.get("KUPL_NUMA_BIND"),
        },
    }


def _run_setup_benchmark(
        qcu: Any, comm: MPI.Comm, gauge: np.ndarray, parity: int,
        clover: np.ndarray, clover_inv: np.ndarray, lattice: tuple[int, int,
         int, int], local: tuple[int, int, int, int],
        grid: tuple[int, int, int, int], rank: int, size: int,
        coordinate: Sequence[int], mass: float, atol: float, csw: float,
        dtype: np.dtype, max_iter: int, seed: int, threads: int,
        warmups: int, repeats: int) -> dict[str, Any]:
    state = _make_control(
        local, grid, rank, size, mass, atol, csw, dtype, max_iter,
        seed, PLAN_CLOVER, parity)
    gauge_tensor = NumpyTensor(gauge)
    clover_tensor = NumpyTensor(clover)
    inverse_tensor = NumpyTensor(clover_inv)
    set_tensor = NumpyTensor(state.set_ptrs)
    params_tensor = NumpyTensor(state.params)

    def operation() -> None:
        qcu.applyCloversQcu(
            clover_tensor, inverse_tensor, gauge_tensor, set_tensor,
            params_tensor)

    with _backend_slot(qcu, state):
        cold, _ = _timed_operation(comm, operation)
        warmup_seconds = [
            _timed_operation(comm, operation)[0] for _ in range(warmups)
        ]
        steady_seconds = [
            _timed_operation(comm, operation)[0] for _ in range(repeats)
        ]

    return {
        **_rank_environment(rank, coordinate, local, threads),
        "parity": int(parity),
        "cold": _summary([cold]),
        "warmup": _summary(warmup_seconds) if warmup_seconds else None,
        "steady": _summary(steady_seconds),
    }


def _run_solver_benchmark(
        qcu: Any, comm: MPI.Comm, gauge: np.ndarray, source: np.ndarray,
        solution: np.ndarray, clover_even: np.ndarray, clover_odd: np.ndarray,
        clover_even_inv: np.ndarray, clover_odd_inv: np.ndarray,
        lattice: tuple[int, int, int, int], local: tuple[int, int, int, int],
        grid: tuple[int, int, int, int], rank: int, size: int,
        coordinate: Sequence[int], mass: float, atol: float, csw: float,
        dtype: np.dtype, max_iter: int, seed: int, threads: int,
        warmups: int, repeats: int) -> dict[str, Any]:
    state = _make_control(
        local, grid, rank, size, mass, atol, csw, dtype, max_iter,
        seed, PLAN_SOLVER, TARGET_PARITY)
    tensors = {
        "solution": NumpyTensor(solution),
        "source": NumpyTensor(source),
        "gauge": NumpyTensor(gauge),
        "clover_even": NumpyTensor(clover_even),
        "clover_odd": NumpyTensor(clover_odd),
        "clover_even_inv": NumpyTensor(clover_even_inv),
        "clover_odd_inv": NumpyTensor(clover_odd_inv),
        "params": NumpyTensor(state.params),
        "set_ptrs": NumpyTensor(state.set_ptrs),
    }

    def operation() -> int:
        state.params[USE_INITIAL_GUESS] = 0
        qcu.applyCloverBistabCgQcu(
            tensors["solution"], tensors["source"], tensors["gauge"],
            tensors["clover_even"], tensors["clover_odd"],
            tensors["clover_even_inv"], tensors["clover_odd_inv"],
            tensors["set_ptrs"], tensors["params"])
        return int(qcu.getCloverBistabCgIterationsQcu(
            tensors["set_ptrs"], tensors["params"]))

    with _backend_slot(qcu, state):
        cold_seconds, cold_iterations = _timed_operation(comm, operation)
        warmup = [_timed_operation(comm, operation) for _ in range(warmups)]
        steady = [_timed_operation(comm, operation) for _ in range(repeats)]

    return {
        **_rank_environment(rank, coordinate, local, threads),
        "target_parity": TARGET_PARITY,
        "cold": {
            **_summary([cold_seconds]),
            "iterations": [cold_iterations],
        },
        "warmup": ({
            **_summary([seconds for seconds, _ in warmup]),
            "iterations": [iterations for _, iterations in warmup],
        } if warmup else None),
        "steady": {
            **_summary([seconds for seconds, _ in steady]),
            "iterations": [iterations for _, iterations in steady],
        },
    }


def _aggregate_block(per_rank: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "aggregation": "maximum rank-local wall time for each sample",
    }
    for period in ("cold", "warmup", "steady"):
        blocks = [rank[period] for rank in per_rank if rank[period] is not None]
        if not blocks:
            result[period] = None
            continue
        count = len(blocks[0]["seconds"])
        seconds = [max(block["seconds"][index] for block in blocks)
                   for index in range(count)]
        aggregate = _summary(seconds)
        if "iterations" in blocks[0]:
            by_rank = [block["iterations"] for block in blocks]
            aggregate["iterations"] = list(by_rank[0])
            aggregate["iterations_per_rank"] = by_rank
            aggregate["iterations_consistent"] = all(
                values == by_rank[0] for values in by_rank[1:])
        result[period] = aggregate
    return result


def _load_backend() -> tuple[Any | None, Exception | None]:
    try:
        from pyqcu.kupl import qcu
    except (FileNotFoundError, OSError, ImportError) as error:
        return None, error
    return qcu, None


def _validate_args(parser: argparse.ArgumentParser,
                   args: argparse.Namespace, size: int) -> None:
    if any(value <= 0 for value in args.lat):
        parser.error("--lat values must be positive")
    if any(value <= 0 for value in args.grid):
        parser.error("--grid values must be positive")
    if math.prod(args.grid) != size:
        parser.error(
            f"--grid product {math.prod(args.grid)} must equal MPI size {size}")
    if any(global_extent % process_extent for global_extent, process_extent
           in zip(args.lat, args.grid, strict=True)):
        parser.error("--grid must divide every --lat dimension")
    local = tuple(global_extent // process_extent for global_extent,
                  process_extent in zip(args.lat, args.grid, strict=True))
    if any(extent % 2 for extent in local):
        parser.error(f"KUPL local dimensions must all be even: {local}")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.warmups < 0:
        parser.error("--warmups must be nonnegative")
    if args.max_iter <= 0:
        parser.error("--max-iter must be positive")
    if args.threads <= 0:
        parser.error("--threads must be positive")
    if not math.isfinite(args.atol) or args.atol <= 0.0:
        parser.error("--atol must be finite and positive")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gauge", required=True,
                        help="global LIME/SciDAC or HDF5 gauge field")
    parser.add_argument("--lat", nargs=4, type=int, default=DEFAULT_LATTICE,
                        metavar=("X", "Y", "Z", "T"))
    parser.add_argument("--grid", nargs=4, type=int, default=DEFAULT_GRID,
                        metavar=("X", "Y", "Z", "T"))
    parser.add_argument("--mass", type=float, default=None,
                        help="default: LIME reference_mass, then common default")
    parser.add_argument("--csw", type=float, default=None,
                        help="default: LIME csw/clovCoeff, then common default")
    parser.add_argument("--dtype", choices=("c64", "c128"), default="c64")
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--source", default=None,
                        help="optional global HDF5 [spin,color,x,y,z,t] source")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--report", default="performance_kupl_lime_mpi.json")
    args = parser.parse_args(argv)

    comm = MPI.COMM_WORLD
    rank = int(comm.Get_rank())
    size = int(comm.Get_size())
    _validate_args(parser, args, size)

    lattice = tuple(int(value) for value in args.lat)
    grid = tuple(int(value) for value in args.grid)
    local = tuple(global_extent // process_extent for global_extent,
                  process_extent in zip(lattice, grid, strict=True))
    coordinate = tuple(int(value) for value in np.unravel_index(
        rank, grid, order="C"))
    slices = _local_slices(coordinate, local)

    os.environ["KUPL_OMP_THREADS"] = str(args.threads)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    qcu, backend_error = _load_backend()
    if backend_error is not None:
        if rank == 0:
            print(json.dumps({
                "status": "SKIP",
                "reason": str(backend_error),
            }, ensure_ascii=False))
        return 0

    gauge_path = _resolved_input(args.gauge)
    if not gauge_path.is_file():
        raise FileNotFoundError(f"gauge input not found: {gauge_path}")
    dtype = np.dtype(np.complex64 if args.dtype == "c64" else np.complex128)
    gauge_full, metadata, gauge_info = _load_gauge(
        gauge_path, lattice, dtype, comm)

    mass_metadata = _metadata_float(metadata, "reference_mass", "Mass", "mass")
    if args.mass is not None:
        mass = float(args.mass)
        mass_source = "command_line"
    elif mass_metadata is not None:
        mass = mass_metadata
        mass_source = "gauge_metadata"
    else:
        mass = float(DEFAULT_MASS)
        mass_source = "common_default"

    csw_metadata = _metadata_float(metadata, "csw", "clovCoeff")
    if args.csw is not None:
        csw = float(args.csw)
        csw_source = "command_line"
    elif csw_metadata is not None:
        csw = csw_metadata
        csw_source = "gauge_metadata"
    else:
        csw = float(DEFAULT_CSW)
        csw_source = "common_default"
    if not math.isfinite(mass) or 2.0 * mass + 8.0 == 0.0:
        parser.error("--mass must be finite and produce a finite kappa")
    if not math.isfinite(csw):
        parser.error("--csw must be finite")
    kappa = 1.0 / (2.0 * mass + 8.0)

    gauge_local = np.ascontiguousarray(
        gauge_full[(slice(None), slice(None), slice(None), *slices)])
    gauge = qcu_gauge_to_kupl(gauge_local)

    if args.source is None:
        source_local, source_info = _make_source_slab(
            slices, lattice, dtype, args.seed)
    else:
        source_path = _resolved_input(args.source)
        if not source_path.is_file():
            raise FileNotFoundError(f"source input not found: {source_path}")
        source_local, source_info = _load_source_slab(
            source_path, slices, lattice, dtype, comm)
    source = qcu_fermion_to_kupl(source_local)
    solution = np.zeros_like(source)

    clover_shape = (4, 3, 4, 3, *local[:3], local[3] // 2)
    clover_even = np.empty(clover_shape, dtype=dtype)
    clover_odd = np.empty(clover_shape, dtype=dtype)
    clover_even_inv = np.empty(clover_shape, dtype=dtype)
    clover_odd_inv = np.empty(clover_shape, dtype=dtype)

    common = dict(
        qcu=qcu, comm=comm, gauge=gauge, lattice=lattice, local=local,
        grid=grid, rank=rank, size=size, coordinate=coordinate, mass=mass,
        atol=args.atol, csw=csw, dtype=dtype, max_iter=args.max_iter,
        seed=args.seed, threads=args.threads, warmups=args.warmups,
        repeats=args.repeats)
    setup_even_rank = _run_setup_benchmark(
        parity=0, clover=clover_even, clover_inv=clover_even_inv, **common)
    setup_odd_rank = _run_setup_benchmark(
        parity=1, clover=clover_odd, clover_inv=clover_odd_inv, **common)
    solver_rank = _run_solver_benchmark(
        source=source, solution=solution, clover_even=clover_even,
        clover_odd=clover_odd, clover_even_inv=clover_even_inv,
        clover_odd_inv=clover_odd_inv, **common)
    floor = solver_acceptance(args.atol, dtype=dtype)
    true_residual = _independent_true_residual(
        gauge_full=gauge_full, source=source, solution=solution,
        lattice=lattice, grid=grid, mass=mass, csw=csw,
        threads=args.threads, comm=comm)
    independent_pass = bool(
        math.isfinite(true_residual) and true_residual <= floor)

    local_hashes = {
        "gauge_canonical_sha256": _sha256_array(gauge),
        "rhs_canonical_sha256": _sha256_array(source),
    }
    payloads = comm.gather({
        "setup_even": setup_even_rank,
        "setup_odd": setup_odd_rank,
        "solver": solver_rank,
        "local_hashes": local_hashes,
        "environment": _rank_environment(
            rank, coordinate, local, args.threads),
    }, root=0)

    report_path = resolve_path(args.report, Path(args.report).expanduser())
    if rank == 0:
        assert payloads is not None
        setup_even_per_rank = [payload["setup_even"] for payload in payloads]
        setup_odd_per_rank = [payload["setup_odd"] for payload in payloads]
        solver_per_rank = [payload["solver"] for payload in payloads]
        report_status = "PASS" if independent_pass else "FAIL"
        report = {
            "status": report_status,
            "backend": "kupl",
            "benchmark": "clover_setup_and_bistabcg",
            "scope": "real LIME/HDF5 gauge; MPI-split local slabs",
            "mpi": {
                "ranks": size,
                "grid": list(grid),
                "global_shape": list(lattice),
                "local_shape": list(local),
                "canonical_gauge_shape": [2, 3, 3, 4, *local],
                "canonical_fermion_shape": [
                    2, 4, 3, *local[:3], local[3] // 2],
                "canonical_clover_shape": [
                    4, 3, 4, 3, *local[:3], local[3] // 2],
                "global_volume": math.prod(lattice),
                "local_volume": math.prod(local),
                "strong_scaling_capable": True,
            },
            "parameters": {
                "dtype": str(dtype),
                "mass": mass,
                "mass_source": mass_source,
                "kappa": kappa,
                "csw": csw,
                "csw_source": csw_source,
                "atol": args.atol,
                "max_iter": args.max_iter,
                "target_parity": TARGET_PARITY,
                "threads_requested": args.threads,
                "warmups": args.warmups,
                "repeats": args.repeats,
                "rhs_seed": args.seed,
            },
            "inputs": {
                "gauge": gauge_info,
                "rhs": source_info,
                "per_rank_local_sha256": [
                    {"rank": int(payload["environment"]["rank"]),
                     **payload["local_hashes"]}
                    for payload in payloads
                ],
            },
            "rank_environment": [
                payload["environment"] for payload in payloads
            ],
            "timing": {
                "clock": "time.perf_counter",
                "rank_aggregation": "max rank-local wall time per sample",
                "clover_setup_even": {
                    "aggregate": _aggregate_block(setup_even_per_rank),
                    "per_rank": setup_even_per_rank,
                },
                "clover_setup_odd": {
                    "aggregate": _aggregate_block(setup_odd_per_rank),
                    "per_rank": setup_odd_per_rank,
                },
                "clover_bistabcg": {
                    "aggregate": _aggregate_block(solver_per_rank),
                    "per_rank": solver_per_rank,
                },
            },
            "acceptance": {
                "status": "NATIVE_PASS",
                "requested_relative_atol": args.atol,
                "residual_floor": floor,
                "native_inner_relative_tolerance": 0.1 * args.atol,
                "native_full_true_residual_checked": True,
                "native_full_true_residual_exported": False,
                "native_acceptance_formula": (
                    "max(atol, residual_floor) * max(global_rhs_norm, 1)"),
                "native_fail_closed": True,
                "native_failure_mode": (
                    "KUPL aborts on non-finite or out-of-tolerance native "
                    "true residual"),
                "independent_true_residual": float(true_residual),
                "independent_true_residual_verified": True,
                "independent_true_residual_pass": independent_pass,
                "independent_source": (
                    "root-gathered full solution and RHS evaluated by a "
                    "PyQCU CPU Wilson-Clover operator"),
            },
        }
        write_json(report_path, _jsonable(report))
        summary = {
            "status": report["status"],
            "report": str(report_path),
            "ranks": size,
            "grid": list(grid),
            "local_shape": list(local),
            "clover_setup_even_steady_median_seconds": report["timing"][
                "clover_setup_even"]["aggregate"]["steady"][
                    "median_seconds"],
            "clover_setup_odd_steady_median_seconds": report["timing"][
                "clover_setup_odd"]["aggregate"]["steady"][
                    "median_seconds"],
            "clover_bistabcg_steady_median_seconds": report["timing"][
                "clover_bistabcg"]["aggregate"]["steady"][
                    "median_seconds"],
            "iterations": report["timing"]["clover_bistabcg"]["aggregate"][
                "steady"]["iterations"],
            "native_fail_closed": True,
            "independent_true_residual": float(true_residual),
            "independent_true_residual_pass": independent_pass,
        }
        print(json.dumps(summary, ensure_ascii=False))
    final_pass = bool(comm.bcast(independent_pass, root=0))
    return 0 if final_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
