#!/usr/bin/env python3
"""Benchmark CUDA QCU Clover setup and Schur BiStabCG from one LIME gauge.

The script keeps setup and solve timing separate, supports one GPU per MPI
rank, and writes one combined JSON report from rank 0. It does not initialize
CUDA until the MPI local rank has been selected in
``PYQCU_MPI_DEVICE_ID``.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[3]
SIMPLE_DIR = REPO / "pyqcu" / "testing" / "simple"
DEFAULT_LAT = (24, 24, 24, 64)
DEFAULT_ATOL = 1.0e-6
DEFAULT_MAX_ITER = 1000
DEFAULT_WARMUPS = 1
DEFAULT_REPEATS = 5
DEFAULT_SEED = 42


def _load_common():
    """Load the NumPy-only LIME/layout helpers without importing pyqcu."""
    module_name = "_pyqcu_lime_benchmark_common"
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    # ``_common`` was split into sibling ``_lime``/``_layouts`` modules.  A
    # file-based import does not establish their package context, so make the
    # helpers discoverable through their direct-execution import path.
    simple_path = str(SIMPLE_DIR)
    if simple_path not in sys.path:
        sys.path.insert(0, simple_path)
    spec = importlib.util.spec_from_file_location(
        module_name, SIMPLE_DIR / "_common.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {SIMPLE_DIR / '_common.py'}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _tuple4(values: Sequence[int], name: str) -> tuple[int, int, int, int]:
    result = tuple(int(value) for value in values)
    if len(result) != 4:
        raise ValueError(f"{name} must contain four integers")
    return result  # type: ignore[return-value]


def _validate_lattice(lattice: Sequence[int]) -> tuple[int, int, int, int]:
    result = _tuple4(lattice, "lattice")
    if any(value <= 0 or value % 2 for value in result):
        raise ValueError(f"lattice must be four positive even extents: {result}")
    return result


def _default_grid(size: int) -> tuple[int, int, int, int]:
    """Match pyqcu.tools.give_grid_size() without importing pyqcu."""
    size = int(size)
    if size <= 0:
        raise ValueError("MPI size must be positive")
    factors: list[int] = []
    divisor = 2
    remaining = size
    while divisor * divisor <= remaining:
        while remaining % divisor == 0:
            factors.append(divisor)
            remaining //= divisor
        divisor += 1
    if remaining > 1:
        factors.append(remaining)
    groups = [1, 1, 1, 1]
    for factor in sorted(factors, reverse=True):
        axis = min(range(4), key=lambda index: groups[index])
        groups[axis] *= factor
    return tuple(sorted(groups))  # type: ignore[return-value]


def _grid_coordinate(grid: Sequence[int], rank: int) -> tuple[int, int, int, int]:
    values = _tuple4(grid, "grid")
    if any(value <= 0 for value in values):
        raise ValueError(f"grid must contain positive values: {values}")
    coordinate = tuple(int(value) for value in np.unravel_index(
        int(rank), values))
    return coordinate  # type: ignore[return-value]


def _local_geometry(
    lattice: Sequence[int], grid: Sequence[int], rank: int,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int],
           tuple[int, int, int, int]]:
    global_shape = _validate_lattice(lattice)
    process_grid = _tuple4(grid, "grid")
    if any(value <= 0 for value in process_grid):
        raise ValueError(f"grid must contain positive values: {process_grid}")
    if int(np.prod(process_grid)) <= int(rank):
        raise ValueError(f"rank {rank} is outside grid {process_grid}")
    if any(global_shape[axis] % process_grid[axis]
           for axis in range(4)):
        raise ValueError(
            f"lattice {global_shape} is not divisible by grid {process_grid}")
    local_shape = tuple(
        global_shape[axis] // process_grid[axis] for axis in range(4))
    if any(value % 2 for value in local_shape):
        raise ValueError(
            "checkerboard MPI layout requires every local extent to be even: "
            f"lattice={global_shape}, grid={process_grid}, local={local_shape}")
    coordinate = _grid_coordinate(process_grid, rank)
    starts = tuple(
        coordinate[axis] * local_shape[axis] for axis in range(4))
    return local_shape, starts, coordinate  # type: ignore[return-value]


def _slice_last4(
    array: np.ndarray, starts: Sequence[int], local_shape: Sequence[int],
) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim < 4:
        raise ValueError(f"expected at least four lattice axes, got {values.shape}")
    slices = [slice(None)] * (values.ndim - 4)
    slices.extend(
        slice(int(starts[axis]), int(starts[axis] + local_shape[axis]))
        for axis in range(4))
    return np.ascontiguousarray(values[tuple(slices)])


def _slice_qdp_gauge(
    gauge: np.ndarray, starts: Sequence[int], local_shape: Sequence[int],
) -> np.ndarray:
    """Slice QDP [dir,t,z,y,x,row,col] by an XY ZT process coordinate."""
    values = np.asarray(gauge)
    if values.ndim != 7 or values.shape[0] != 4 or values.shape[-2:] != (3, 3):
        raise ValueError(
            "QDP gauge must be [dir,t,z,y,x,color_out,color_in]")
    local_x, local_y, local_z, local_t = (int(value) for value in local_shape)
    start_x, start_y, start_z, start_t = (int(value) for value in starts)
    return np.ascontiguousarray(values[
        :,
        start_t:start_t + local_t,
        start_z:start_z + local_z,
        start_y:start_y + local_y,
        start_x:start_x + local_x,
        :,
        :,
    ])


def _parity_mask(shape: Sequence[int]) -> np.ndarray:
    x_dim, y_dim, z_dim, t_dim = (int(value) for value in shape)
    if any(value <= 0 for value in (x_dim, y_dim, z_dim, t_dim)):
        raise ValueError(f"invalid lattice shape: {tuple(shape)}")
    x = np.arange(x_dim, dtype=np.int16)[:, None, None, None]
    y = np.arange(y_dim, dtype=np.int16)[None, :, None, None]
    z = np.arange(z_dim, dtype=np.int16)[None, None, :, None]
    t = np.arange(t_dim, dtype=np.int16)[None, None, None, :]
    return ((x + y + z + t) & 1).astype(bool)


def _parity_compress(array: np.ndarray) -> np.ndarray:
    """Convert canonical [...,x,y,z,t] to QCU [parity,...,x,y,z,t/2]."""
    values = np.asarray(array)
    if values.ndim < 4:
        raise ValueError(f"expected at least four lattice axes, got {values.shape}")
    shape = tuple(int(value) for value in values.shape[-4:])
    if shape[3] % 2:
        raise ValueError(f"time extent must be even for parity packing: {shape}")
    prefix = values.shape[:-4]
    parity = _parity_mask(shape)
    output = np.empty(
        (2, *prefix, *shape[:-1], shape[3] // 2), dtype=values.dtype)
    output[0] = values[..., ~parity].reshape(
        *prefix, *shape[:-1], shape[3] // 2)
    output[1] = values[..., parity].reshape(
        *prefix, *shape[:-1], shape[3] // 2)
    return np.ascontiguousarray(output)


def _parity_expand(array: np.ndarray) -> np.ndarray:
    """Convert QCU [parity,...,x,y,z,t/2] to canonical [...,x,y,z,t]."""
    values = np.asarray(array)
    if values.ndim < 5 or values.shape[0] != 2:
        raise ValueError(
            "QCU parity field must have shape [2,...,x,y,z,t/2]")
    prefix = values.shape[1:-4]
    shape = (*values.shape[-4:-1], values.shape[-1] * 2)
    parity = _parity_mask(shape)
    output = np.empty((*prefix, *shape), dtype=values.dtype)
    output[..., ~parity] = values[0].reshape(*prefix, -1)
    output[..., parity] = values[1].reshape(*prefix, -1)
    return np.ascontiguousarray(output)


def _h5_dataset(
    path: Path, ndim: int, preferred: Sequence[str],
) -> tuple[np.ndarray, str]:
    import h5py

    with h5py.File(path, "r") as handle:
        names: list[str] = []

        def collect(name: str, item: Any) -> None:
            if isinstance(item, h5py.Dataset) and item.ndim == ndim:
                names.append(name)

        handle.visititems(collect)
        if not names:
            raise ValueError(f"{path} has no {ndim}-dimensional dataset")
        selected = None
        for name in preferred:
            if name in names:
                selected = name
                break
        if selected is None:
            selected = sorted(names)[0]
        return np.asarray(handle[selected][...]), selected


def _read_array(
    path: Path, ndim: int, preferred: Sequence[str],
) -> tuple[np.ndarray, str, str]:
    suffix = path.suffix.lower()
    if suffix in (".h5", ".hdf5"):
        array, dataset = _h5_dataset(path, ndim, preferred)
        return array, dataset, "hdf5"
    if suffix == ".npy":
        array = np.load(path, allow_pickle=False)
        if array.ndim != ndim:
            raise ValueError(f"{path} has shape {array.shape}, expected ndim={ndim}")
        return np.asarray(array), "<array>", "npy"
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as handle:
            names = [name for name in handle.files if handle[name].ndim == ndim]
            if not names:
                raise ValueError(f"{path} has no {ndim}-dimensional array")
            selected = next(
                (name for name in preferred if name in names), sorted(names)[0])
            return np.asarray(handle[selected]), selected, "npz"
    try:
        array, dataset = _h5_dataset(path, ndim, preferred)
    except Exception as exc:
        raise ValueError(f"unsupported input format: {path}") from exc
    return array, dataset, "hdf5"


def _read_gauge(path: Path) -> tuple[np.ndarray, str, str]:
    common = _load_common()
    if path.suffix.lower() == ".lime":
        return np.asarray(common.read_lime_gauge(path)), "<lime>", "lime"
    array, dataset, file_format = _read_array(
        path, 7, ("data", "g", "gauge", "U"))
    if array.shape[0] == 4 and array.shape[-2:] == (3, 3):
        return array, dataset, file_format
    if array.shape[:3] == (3, 3, 4):
        return np.ascontiguousarray(common.qcu_gauge_to_qdp(array)), dataset, file_format
    if array.shape[:4] == (2, 3, 3, 4):
        canonical = _parity_expand(array)
        return np.ascontiguousarray(
            common.qcu_gauge_to_qdp(canonical)), dataset, file_format
    raise ValueError(
        "gauge must be QDP [dir,t,z,y,x,3,3], canonical "
        "[3,3,4,x,y,z,t], or QCU parity [2,3,3,4,x,y,z,t/2]")


def _read_source(
    path: Path, lattice: Sequence[int],
) -> tuple[np.ndarray, str, str]:
    common = _load_common()
    desired_shape = (4, 3, *tuple(int(value) for value in lattice))
    candidates: list[tuple[np.ndarray, str, str]] = []
    errors: list[str] = []
    for ndim in (6, 7):
        try:
            array, dataset, file_format = _read_array(
                path, ndim, ("data", "source", "src", "b"))
            candidates.append((array, dataset, file_format))
        except Exception as exc:
            errors.append(str(exc))
    for array, dataset, file_format in candidates:
        if array.shape == desired_shape:
            return np.asarray(array), dataset, file_format
        qdp_shape = (
            2, desired_shape[5], desired_shape[4], desired_shape[3],
            desired_shape[2] // 2, 4, 3)
        if array.shape == qdp_shape:
            return np.ascontiguousarray(
                common.qdp_fermion_to_qcu(array)), dataset, file_format
        parity_shape = (
            2, 4, 3, desired_shape[2], desired_shape[3],
            desired_shape[4], desired_shape[5] // 2)
        if array.shape == parity_shape:
            return _parity_expand(array), dataset, file_format
    raise ValueError(
        f"source {path} does not match global lattice {tuple(lattice)}; "
        f"tried canonical, QDP-compressed and QCU-parity layouts ({errors})")


def _gauge_metadata(path: Path) -> dict[str, Any]:
    common = _load_common()
    if path.suffix.lower() == ".lime":
        return dict(common.lime_metadata(path))
    if path.suffix.lower() not in (".h5", ".hdf5"):
        return {}
    import h5py

    metadata: dict[str, Any] = {}
    with h5py.File(path, "r") as handle:
        for key, value in handle.attrs.items():
            array = np.asarray(value)
            metadata[str(key)] = array.item() if array.shape == () else array.tolist()
    return metadata


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    contiguous = np.ascontiguousarray(array)
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _generate_global_rhs(
    lattice: Sequence[int], dtype: np.dtype, seed: int,
) -> np.ndarray:
    shape = (4, 3, *tuple(int(value) for value in lattice))
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    # Keep one cross-backend source stream: PCG64 float64 normals are
    # generated first and only then cast to the requested field precision.
    real = generator.standard_normal(shape)
    imag = generator.standard_normal(shape)
    return np.ascontiguousarray(real + 1j * imag, dtype=dtype)


def _gather_local_blocks(value: Any, process_grid: Sequence[int],
                         root: int = 0, comm: Any | None = None) -> Any:
    """Gather a rank-local ``...xyzt`` tensor into the global lattice."""
    import torch
    if comm is None:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
    rank = int(comm.Get_rank())
    array = np.ascontiguousarray(value.detach().cpu().numpy())
    grid = _tuple4(process_grid, "process_grid")
    local = tuple(int(item) for item in array.shape[-4:])
    global_shape = (
        *array.shape[:-4],
        *(local[axis] * grid[axis] for axis in range(4)),
    )
    blocks = comm.gather(array, root=root)
    comm.Barrier()
    if rank != root:
        return None
    whole = np.empty(global_shape, dtype=array.dtype)
    for block_rank, block in enumerate(blocks or []):
        coordinate = _grid_coordinate(grid, block_rank)
        selection = [slice(None)] * (array.ndim - 4)
        selection.extend(
            slice(coordinate[axis] * local[axis],
                  (coordinate[axis] + 1) * local[axis])
            for axis in range(4))
        whole[tuple(selection)] = block
    return torch.from_numpy(whole).to(device=value.device)


def _independent_true_residual(
    *, torch: Any, solution_eo: Any, rhs_eo: Any, gauge_eo: Any,
    clover_even: Any, clover_odd: Any, mass: float,
    process_grid: Sequence[int], comm: Any | None = None,
) -> float:
    """Independent PyQCU full residual from globally gathered rank fields."""
    from pyqcu import tools
    from pyqcu.torch import dslash

    if comm is None:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD

    local_solution = tools.poooxyzt2oooxyzt(solution_eo)
    local_rhs = tools.poooxyzt2oooxyzt(rhs_eo)
    local_gauge = tools.poooxyzt2oooxyzt(gauge_eo)
    local_clover = tools.poooxyzt2oooxyzt(
        torch.stack((clover_even, clover_odd), dim=0))
    global_solution = _gather_local_blocks(
        local_solution, process_grid, comm=comm)
    global_rhs = _gather_local_blocks(local_rhs, process_grid, comm=comm)
    global_gauge = _gather_local_blocks(local_gauge, process_grid, comm=comm)
    global_clover = _gather_local_blocks(local_clover, process_grid, comm=comm)
    if global_solution is not None:
        # applyCloversQcu returns A=I+T; give_clover expects T alone.
        global_clover = dslash.cut_I(global_clover)
        kappa = torch.tensor(
            [1.0 / (2.0 * float(mass) + 8.0)], dtype=gauge_eo.real.dtype,
            device=gauge_eo.device)
        u0 = torch.ones([1], dtype=gauge_eo.real.dtype, device=gauge_eo.device)
        residual = (
            dslash.give_wilson(
                global_solution, global_gauge, kappa, u0, True) +
            dslash.give_clover(global_solution, global_clover) - global_rhs)
        numerator_local = residual
        denominator_local = global_rhs
    else:
        numerator_local = torch.zeros(1, dtype=solution_eo.dtype,
                                      device=solution_eo.device)
        denominator_local = torch.zeros(1, dtype=solution_eo.dtype,
                                        device=solution_eo.device)
    numerator = float(tools.norm(numerator_local))
    denominator = max(float(tools.norm(denominator_local)), 1.0e-30)
    value = numerator / denominator
    if not math.isfinite(value):
        raise RuntimeError(f"nonfinite independent true residual: {value!r}")
    value = float(comm.bcast(value, root=0))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return float(value)


def _summary(values: Sequence[Any]) -> dict[str, Any]:
    samples = [float(value) for value in values if value is not None]
    if not samples:
        return {"count": 0, "median": None, "mad": None}
    array = np.asarray(samples, dtype=np.float64)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    return {"count": len(samples), "median": median, "mad": mad}


def _bucket(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "samples": list(samples),
        "operation_seconds": _summary(
            [sample.get("operation_seconds") for sample in samples]),
        "lifecycle_seconds": _summary(
            [sample.get("lifecycle_seconds") for sample in samples]),
        "iterations": _summary(
            [sample.get("iterations") for sample in samples]),
    }


def _stage(cold: Mapping[str, Any],
           warmups: Sequence[Mapping[str, Any]],
           steady: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "cold": _bucket([cold]),
        "warmup": _bucket(warmups),
        "steady": _bucket(steady),
    }


def _run_lifecycle(
    *,
    comm: Any,
    torch: Any,
    qcu: Any,
    define: Any,
    params: Any,
    argv: Any,
    set_ptrs: Any,
    device: Any,
    plan: int,
    parity: int,
    operation: Callable[[], Any],
    getter: Callable[[], Any] | None = None,
    max_iter: int,
) -> dict[str, Any]:
    """Run init, operation, end, then advance and recycle the ordinary slot."""
    slot_limit = int(define._SET_PTRS_COARSE_BASE_)
    index = int(params[define._SET_INDEX_].item())
    if not 0 <= index < slot_limit:
        raise RuntimeError(
            f"QCU ordinary slot {index} is outside [0,{slot_limit})")
    if int(set_ptrs[index].item()) != 0:
        raise RuntimeError(f"QCU slot {index} is already occupied")

    params[define._SET_PLAN_] = int(plan)
    params[define._PARITY_] = int(parity)
    torch.cuda.synchronize(device)
    lifecycle_started = time.perf_counter()
    qcu.applyInitQcu(set_ptrs, params, argv)
    try:
        # Exclude rank-to-rank init skew and start distributed timing from a
        # common point.  The post-sync barrier includes the slowest rank in the
        # measured wall time.
        comm.Barrier()
        operation_started = time.perf_counter()
        operation()
        torch.cuda.synchronize(device)
        operation_seconds = time.perf_counter() - operation_started
        raw_iterations = getter() if getter is not None else None
        iterations = None if raw_iterations is None else int(raw_iterations)
        native_converged = (
            None if iterations is None
            else bool(0 < iterations < int(max_iter)))
    finally:
        qcu.applyEndQcu(set_ptrs, params)
        params[define._SET_INDEX_] += 1
        if int(params[define._SET_INDEX_].item()) >= slot_limit:
            params[define._SET_INDEX_] = 0
        lifecycle_seconds = time.perf_counter() - lifecycle_started

    return {
        "operation_seconds": float(operation_seconds),
        "lifecycle_seconds": float(lifecycle_seconds),
        "iterations": iterations,
        "native_converged": native_converged,
        "native_convergence_criterion": (
            None if iterations is None
            else "iterations < max_iter under native rnorm2 < atol^2"
        ),
    }


def _measure_stage(
    *,
    comm: Any,
    torch: Any,
    qcu: Any,
    define: Any,
    params: Any,
    argv: Any,
    set_ptrs: Any,
    device: Any,
    plan: int,
    parity: int,
    operation: Callable[[], Any],
    getter: Callable[[], Any] | None,
    max_iter: int,
    warmups: int,
    repeats: int,
) -> dict[str, Any]:
    def measure() -> dict[str, Any]:
        return _run_lifecycle(
            comm=comm, torch=torch, qcu=qcu, define=define, params=params, argv=argv,
            set_ptrs=set_ptrs, device=device, plan=plan, parity=parity,
            operation=operation, getter=getter, max_iter=max_iter)

    comm.Barrier()
    cold = measure()
    warmup_samples = [measure() for _ in range(int(warmups))]
    comm.Barrier()
    steady_samples = [measure() for _ in range(int(repeats))]
    return _stage(cold, warmup_samples, steady_samples)


def _measure_reused_stage(
    *,
    comm: Any,
    torch: Any,
    qcu: Any,
    define: Any,
    params: Any,
    argv: Any,
    set_ptrs: Any,
    device: Any,
    plan: int,
    parity: int,
    operation: Callable[[], Any],
    getter: Callable[[], Any] | None,
    max_iter: int,
    warmups: int,
    repeats: int,
) -> dict[str, Any]:
    """Measure repeated solves sharing one initialized ordinary QCU slot."""
    slot_limit = int(define._SET_PTRS_COARSE_BASE_)
    index = int(params[define._SET_INDEX_].item())
    if not 0 <= index < slot_limit:
        raise RuntimeError(
            f"QCU ordinary slot {index} is outside [0,{slot_limit})")
    if int(set_ptrs[index].item()) != 0:
        raise RuntimeError(f"QCU slot {index} is already occupied")

    params[define._SET_PLAN_] = int(plan)
    params[define._PARITY_] = int(parity)
    torch.cuda.synchronize(device)
    comm.Barrier()
    setup_started = time.perf_counter()
    try:
        qcu.applyInitQcu(set_ptrs, params, argv)
    except BaseException:
        # Initialization may have allocated part of the slot before failing.
        # Attempt native teardown, but preserve the original init exception.
        try:
            params[define._SET_INDEX_] = index
            try:
                qcu.applyEndQcu(set_ptrs, params)
            except BaseException:
                pass
        finally:
            try:
                set_ptrs[index] = 0
            finally:
                params[define._SET_INDEX_] = index
        raise
    comm.Barrier()
    setup_seconds = time.perf_counter() - setup_started

    samples: list[dict[str, Any]] = []
    try:
        for _ in range(1 + int(warmups) + int(repeats)):
            torch.cuda.synchronize(device)
            comm.Barrier()
            operation_started = time.perf_counter()
            operation()
            torch.cuda.synchronize(device)
            operation_seconds = time.perf_counter() - operation_started
            raw_iterations = getter() if getter is not None else None
            iterations = (
                None if raw_iterations is None else int(raw_iterations))
            samples.append({
                "operation_seconds": float(operation_seconds),
                "lifecycle_seconds": float(operation_seconds),
                "iterations": iterations,
                "native_converged": (
                    None if iterations is None
                    else bool(0 < iterations < int(max_iter))),
                "native_convergence_criterion": (
                    None if iterations is None
                    else "iterations < max_iter under native rnorm2 < atol^2"),
            })
    finally:
        close_started = time.perf_counter()
        try:
            # End must release the slot selected for init even if an operation
            # changed the control array.
            params[define._SET_INDEX_] = index
            qcu.applyEndQcu(set_ptrs, params)
        finally:
            try:
                set_ptrs[index] = 0
            finally:
                params[define._SET_INDEX_] = (index + 1) % slot_limit
                close_seconds = time.perf_counter() - close_started

    cold = samples[0]
    warmup_samples = samples[1:1 + int(warmups)]
    steady_samples = samples[1 + int(warmups):]
    result = _stage(cold, warmup_samples, steady_samples)
    operation_total = sum(
        float(sample["operation_seconds"]) for sample in samples)
    reuse_count = len(samples)
    result["reuse"] = {
        "setup_seconds": float(setup_seconds),
        "close_seconds": float(close_seconds),
        "operation_seconds_total": float(operation_total),
        "reuse_count": int(reuse_count),
        "amortized_lifecycle_seconds": float(
            (setup_seconds + operation_total + close_seconds) / reuse_count),
        "scope": (
            "one applyInitQcu/applyEndQcu lifecycle shared by cold, warmup, "
            "and steady solves; per-call lifecycle equals operation time"),
    }
    return result


def _build_params(
    torch: Any, define: Any, local_shape: Sequence[int],
    grid: Sequence[int], rank: int, size: int, data_type: int,
    max_iter: int, seed: int,
) -> Any:
    params = torch.zeros(define._PARAMS_SIZE_, dtype=torch.int32)
    params[define._LAT_X_] = int(local_shape[0])
    params[define._LAT_Y_] = int(local_shape[1])
    params[define._LAT_Z_] = int(local_shape[2])
    params[define._LAT_T_] = int(local_shape[3])
    params[define._LAT_XYZT_] = int(np.prod(local_shape, dtype=np.int64))
    params[define._GRID_X_] = int(grid[0])
    params[define._GRID_Y_] = int(grid[1])
    params[define._GRID_Z_] = int(grid[2])
    params[define._GRID_T_] = int(grid[3])
    params[define._PARITY_] = 0
    params[define._NODE_RANK_] = int(rank)
    params[define._NODE_SIZE_] = int(size)
    params[define._DAGGER_] = 0
    params[define._MAX_ITER_] = int(max_iter)
    params[define._DATA_TYPE_] = int(data_type)
    params[define._SET_INDEX_] = 0
    params[define._SET_PLAN_] = define._SET_PLAN0_
    params[define._VERBOSE_] = 0
    params[define._SEED_] = int(seed)
    params[define._TEST_IN_CPU_] = 0
    params[define._MG_USE_INIT_GUESS_] = 0
    return params.contiguous()


def _build_argv(
    torch: Any, define: Any, mass: float, csw: float, atol: float,
    data_type: int,
) -> Any:
    real_dtype = torch.float32 if data_type == define._LAT_C64_ else torch.float64
    argv = torch.zeros(define._ARGV_SIZE_, dtype=real_dtype)
    argv[define._MASS_] = float(mass)
    argv[define._ATOL_] = float(atol)
    argv[define._SIGMA_] = 0.1
    argv[define._CSW_] = float(csw)
    return argv.contiguous()


def _libqcu_path() -> Path | None:
    configured = os.environ.get("PYQCU_LIBQCU")
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())
    maps = Path("/proc/self/maps")
    if maps.is_file():
        for line in maps.read_text(encoding="utf-8", errors="ignore").splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) == 6 and "libqcu.so" in fields[5]:
                candidates.append(Path(fields[5].replace(" (deleted)", "")))
    candidates.append(REPO / "cpp" / "cuda" / "qcu" / "libqcu.so")
    for entry in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        if entry:
            candidates.append(Path(entry) / "libqcu.so")
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    return None


def _hip_device_uuid(device_index: int) -> str | None:
    """Return a HIP device UUID when PyTorch omits it on DCU systems."""
    candidates = []
    for variable in ("ROCM_PATH", "HIP_PATH"):
        root = os.environ.get(variable)
        if root:
            candidates.append(Path(root).expanduser() / "lib" / "libamdhip64.so")
    library_name = ctypes.util.find_library("amdhip64")
    if library_name:
        candidates.append(Path(library_name))

    class HipUuid(ctypes.Structure):
        _fields_ = [("bytes", ctypes.c_char * 16)]

    seen: set[str] = set()
    for candidate in candidates:
        load_name = str(candidate)
        key = str(candidate.resolve()) if candidate.exists() else load_name
        if key in seen:
            continue
        seen.add(key)
        try:
            runtime = ctypes.CDLL(load_name)
            get_uuid = runtime.hipDeviceGetUuid
        except (AttributeError, OSError):
            continue
        value = HipUuid()
        if int(get_uuid(ctypes.byref(value), ctypes.c_int(device_index))) != 0:
            continue
        raw = bytes(value.bytes)
        printable = raw.rstrip(b"\0")
        if printable and all(32 <= byte < 127 for byte in printable):
            return f"hip:{printable.decode('ascii')}"
        if any(raw):
            return f"hip:{raw.hex()}"
    return None


def _device_record(torch: Any, device: Any) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    torch_uuid = getattr(properties, "uuid", None)
    uuid = None if torch_uuid is None else str(torch_uuid).strip()
    if uuid in ("", "00000000-0000-0000-0000-000000000000"):
        uuid = None
    uuid_source = "torch"
    if uuid is None:
        uuid = _hip_device_uuid(int(device.index))
        uuid_source = "hip"
    if uuid is None:
        raise RuntimeError(
            "neither PyTorch nor HIP exposed a device UUID; refusing to omit "
            "required GPU provenance")
    return {
        "index": int(device.index),
        "string": str(device),
        "uuid": uuid,
        "uuid_source": uuid_source,
        "name": str(properties.name),
        "total_memory_bytes": int(properties.total_memory),
        "compute_capability": [
            int(properties.major), int(properties.minor)],
    }


def _solver_native_summary(stage: Mapping[str, Any]) -> dict[str, Any]:
    samples: list[Mapping[str, Any]] = []
    for phase in ("cold", "warmup", "steady"):
        samples.extend(stage[phase]["samples"])
    iterations = [int(sample["iterations"]) for sample in samples
                  if sample.get("iterations") is not None]
    converged = [bool(sample["native_converged"]) for sample in samples
                 if sample.get("native_converged") is not None]
    return {
        "criterion": "native iterations < max_iter with C++ rnorm2 < atol^2",
        "all_samples_converged": bool(converged) and all(converged),
        "converged_samples": sum(converged),
        "sample_count": len(converged),
        "iterations": iterations,
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gauge", required=True,
                        help="global LIME or HDF5 gauge configuration")
    parser.add_argument("--lat", nargs=4, type=int, default=DEFAULT_LAT,
                        metavar=("X", "Y", "Z", "T"),
                        help="global lattice (default: 24 24 24 64)")
    parser.add_argument("--grid", nargs=4, type=int, default=None,
                        metavar=("X", "Y", "Z", "T"),
                        help="MPI process grid; default: balanced MPI size")
    parser.add_argument("--mass", type=float, default=None,
                        help="quark mass; default: LIME Mass, else 0.05")
    parser.add_argument("--csw", type=float, default=None,
                        help="Clover coefficient; default: LIME clovCoeff")
    parser.add_argument("--dtype", choices=("c64", "c128"), default="c64")
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument(
        "--reuse-runtime", action="store_true",
        help="also measure one persistent LatticeSet across repeated solves")
    parser.add_argument("--source", default=None,
                        help="optional global canonical/QDP-compressed source")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="fixed seed for the generated global RHS")
    parser.add_argument("--report", default=None,
                        help="JSON report path; default under logs/qcu/")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    lattice = _validate_lattice(args.lat)
    if not math.isfinite(float(args.atol)) or args.atol <= 0.0:
        raise ValueError("--atol must be finite and positive")
    if int(args.max_iter) <= 0:
        raise ValueError("--max-iter must be positive")
    if int(args.warmups) < 0:
        raise ValueError("--warmups must be nonnegative")
    if int(args.repeats) <= 0:
        raise ValueError("--repeats must be positive")

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = int(comm.Get_rank())
    size = int(comm.Get_size())
    local_comm = comm.Split_type(MPI.COMM_TYPE_SHARED, 0)
    try:
        local_rank = int(local_comm.Get_rank())
    finally:
        local_comm.Free()

    # This must precede importing torch or any QCU module.
    os.environ["PYQCU_MPI_DEVICE_ID"] = str(local_rank)

    grid = _tuple4(args.grid, "grid") if args.grid is not None else _default_grid(size)
    if int(np.prod(grid)) != size:
        raise ValueError(
            f"grid {grid} contains {int(np.prod(grid))} ranks, MPI_COMM_WORLD "
            f"contains {size}")
    local_shape, starts, coordinate = _local_geometry(lattice, grid, rank)

    common = _load_common()
    gauge_path = common.resolve_path(args.gauge, common.DEFAULT_LIME)
    if not gauge_path.is_file():
        raise FileNotFoundError(gauge_path)
    metadata = _gauge_metadata(gauge_path)
    metadata_lattice = metadata.get("lattice_xyzt")
    if metadata_lattice is not None and _tuple4(metadata_lattice, "LIME lattice") != lattice:
        raise ValueError(
            f"LIME lattice {tuple(metadata_lattice)} does not match --lat {lattice}")

    reference_mass = metadata.get("reference_mass", metadata.get("mass"))
    if args.mass is not None:
        mass = float(args.mass)
        mass_source = "cli"
    elif reference_mass is not None:
        mass = float(reference_mass)
        mass_source = "gauge-metadata"
    else:
        mass = float(common.DEFAULT_MASS)
        mass_source = "default"
    if not math.isfinite(mass) or 2.0 * mass + 8.0 == 0.0:
        raise ValueError("--mass must be finite and produce a finite kappa")
    csw = float(common.resolve_csw(args.csw, metadata))
    csw_source = ("cli" if args.csw is not None
                  else "gauge-metadata" if metadata.get("csw") is not None
                  else "default")
    gauge_csw = metadata.get("csw")
    warnings: list[str] = []
    if gauge_csw is not None and not common.csw_matches(csw, float(gauge_csw)):
        warnings.append(
            f"requested csw={csw:g} differs from gauge metadata csw={gauge_csw:g}")
    if mass < 0.0:
        warnings.append("negative mass; confirm it matches the gauge ensemble")

    dtype = np.dtype(np.complex64 if args.dtype == "c64" else np.complex128)
    gauge_qdp, gauge_dataset, gauge_format = _read_gauge(gauge_path)
    gauge_global_shape = tuple(int(value) for value in gauge_qdp.shape[1:5])
    if gauge_global_shape != (lattice[3], lattice[2], lattice[1], lattice[0]):
        raise ValueError(
            f"QDP gauge lattice [t,z,y,x]={gauge_global_shape} does not match "
            f"--lat {lattice}")
    local_qdp = _slice_qdp_gauge(gauge_qdp, starts, local_shape)
    local_canonical = common.qdp_gauge_to_qcu(local_qdp)
    local_gauge = _parity_compress(local_canonical).astype(dtype, copy=False)
    del gauge_qdp, local_qdp, local_canonical

    if args.source is not None:
        source_path = common.resolve_path(args.source, common.DATA_DIR / args.source)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        global_rhs, source_dataset, source_format = _read_source(
            source_path, lattice)
        source_origin = "file"
        source_sha256 = _sha256_file(source_path)
    else:
        source_path = None
        global_rhs = _generate_global_rhs(lattice, dtype, args.seed)
        source_dataset = "<generated>"
        source_format = "pcg64-normal"
        source_origin = "generated-fixed-seed"
        source_sha256 = _sha256_array(global_rhs)
    local_rhs = _parity_compress(
        _slice_last4(global_rhs, starts, local_shape)).astype(dtype, copy=False)
    del global_rhs

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this benchmark on GPU nodes")
    visible_devices = int(torch.cuda.device_count())
    if local_rank >= visible_devices:
        raise RuntimeError(
            f"MPI local rank {local_rank} has no private CUDA device "
            f"(visible={visible_devices})")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    from pyqcu.cuda import define, qcu

    libqcu = _libqcu_path()
    if libqcu is None:
        raise RuntimeError("cannot locate libqcu.so for the required SHA256 record")

    data_type = define._LAT_C64_ if args.dtype == "c64" else define._LAT_C128_
    params = _build_params(
        torch, define, local_shape, grid, rank, size, data_type,
        args.max_iter, args.seed)
    argv = _build_argv(torch, define, mass, csw, args.atol, data_type)
    set_ptrs = torch.zeros(define._SET_PTRS_SIZE_, dtype=torch.int64)
    tensor_dtype = torch.complex64 if args.dtype == "c64" else torch.complex128
    g = torch.from_numpy(local_gauge).to(
        device=device, dtype=tensor_dtype).contiguous()
    rhs = torch.from_numpy(local_rhs).to(device=device, dtype=g.dtype).contiguous()
    out = torch.empty_like(rhs)
    clover_shape = (4, 3, 4, 3, *local_shape[:3], local_shape[3] // 2)
    ce = torch.empty(clover_shape, dtype=g.dtype, device=device)
    cei = torch.empty_like(ce)
    co = torch.empty_like(ce)
    coi = torch.empty_like(ce)
    del local_gauge, local_rhs

    timings: dict[str, Any] = {}
    print(json.dumps({"rank": rank, "stage": "clover_setup_even"}),
          flush=True)
    timings["clover_setup_even"] = _measure_stage(
        comm=comm, torch=torch, qcu=qcu, define=define, params=params,
        argv=argv, set_ptrs=set_ptrs, device=device,
        plan=define._SET_PLAN2_, parity=0,
        operation=lambda: qcu.applyCloversQcu(ce, cei, g, set_ptrs, params),
        getter=None, max_iter=args.max_iter, warmups=args.warmups,
        repeats=args.repeats)
    print(json.dumps({"rank": rank, "stage": "clover_setup_odd"}),
          flush=True)
    timings["clover_setup_odd"] = _measure_stage(
        comm=comm, torch=torch, qcu=qcu, define=define, params=params,
        argv=argv, set_ptrs=set_ptrs, device=device,
        plan=define._SET_PLAN2_, parity=1,
        operation=lambda: qcu.applyCloversQcu(co, coi, g, set_ptrs, params),
        getter=None, max_iter=args.max_iter, warmups=args.warmups,
        repeats=args.repeats)
    print(json.dumps({"rank": rank, "stage": "clover_bistabcg"}),
          flush=True)
    timings["clover_schur_bistabcg_solve"] = _measure_stage(
        comm=comm, torch=torch, qcu=qcu, define=define, params=params,
        argv=argv, set_ptrs=set_ptrs, device=device,
        plan=define._SET_PLAN1_, parity=0,
        operation=lambda: qcu.applyCloverBistabCgQcu(
            out, rhs, g, ce, co, cei, coi, set_ptrs, params),
        getter=lambda: qcu.getCloverBistabCgIterationsQcu(set_ptrs, params),
        max_iter=args.max_iter, warmups=args.warmups, repeats=args.repeats)
    if args.reuse_runtime:
        print(json.dumps({"rank": rank, "stage": "clover_bistabcg_reused"}),
              flush=True)
        timings["clover_schur_bistabcg_solve_reused"] = _measure_reused_stage(
            comm=comm, torch=torch, qcu=qcu, define=define, params=params,
            argv=argv, set_ptrs=set_ptrs, device=device,
            plan=define._SET_PLAN1_, parity=0,
            operation=lambda: qcu.applyCloverBistabCgQcu(
                out, rhs, g, ce, co, cei, coi, set_ptrs, params),
            getter=lambda: qcu.getCloverBistabCgIterationsQcu(
                set_ptrs, params),
            max_iter=args.max_iter, warmups=args.warmups,
            repeats=args.repeats)

    torch.cuda.synchronize(device)
    print(json.dumps({"rank": rank, "stage": "true_residual_start"}),
          flush=True)
    true_residual = _independent_true_residual(
        torch=torch, solution_eo=out, rhs_eo=rhs, gauge_eo=g,
        clover_even=ce, clover_odd=co, mass=mass, process_grid=grid,
        comm=comm)
    print(json.dumps({"rank": rank, "stage": "true_residual_done",
                      "value": true_residual}), flush=True)
    residual_floor = common.solver_acceptance(args.atol, dtype=dtype)
    independent_pass = bool(
        math.isfinite(true_residual) and true_residual <= residual_floor)
    solver_native = _solver_native_summary(
        timings["clover_schur_bistabcg_solve"])
    solver_native_reused = (
        _solver_native_summary(
            timings["clover_schur_bistabcg_solve_reused"])
        if args.reuse_runtime else None)
    native_pass = bool(solver_native["all_samples_converged"])
    reused_pass = (
        bool(solver_native_reused["all_samples_converged"])
        if solver_native_reused is not None else True)
    rank_pass = bool(native_pass and reused_pass and independent_pass)
    rank_payload = {
        "rank": rank,
        "local_rank": local_rank,
        "grid": list(grid),
        "grid_coordinate": list(coordinate),
        "global_shape": list(lattice),
        "local_shape": list(local_shape),
        "local_starts": list(starts),
        "device": _device_record(torch, device),
        "params": {
            "shape": list(params.shape),
            "dtype": str(params.dtype).replace("torch.", ""),
            "grid": [int(params[index].item()) for index in
                     range(define._GRID_X_, define._GRID_T_ + 1)],
            "node_rank": int(params[define._NODE_RANK_].item()),
            "node_size": int(params[define._NODE_SIZE_].item()),
            "set_index": int(params[define._SET_INDEX_].item()),
        },
        "timings": timings,
        "solver_native": solver_native,
        "solver_native_reused": solver_native_reused,
        "independent_true_residual": {
            "verified": True,
            "value": float(true_residual),
            "acceptance": float(residual_floor),
            "pass": independent_pass,
            "source": (
                "root-gathered independent PyQCU Wilson-Clover operator "
                "using the existing even/odd Clover terms with I stripped; "
                "value broadcast to every MPI rank"),
            "phase": "after final steady solve; excluded from timing",
        },
        "status": "PASS" if rank_pass else "FAIL",
    }
    print(json.dumps({
        "rank": rank,
        "local_rank": local_rank,
        "grid": list(grid),
        "local_shape": list(local_shape),
        "gpu_uuid": rank_payload["device"]["uuid"],
        "gpu_name": rank_payload["device"]["name"],
        "solver_steady_median_seconds": timings[
            "clover_schur_bistabcg_solve"]["steady"][
                "operation_seconds"]["median"],
        "solver_iterations": solver_native["iterations"],
        "native_converged": solver_native["all_samples_converged"],
        "independent_true_residual": true_residual,
        "independent_pass": independent_pass,
    }, ensure_ascii=False), flush=True)

    gathered = comm.gather(rank_payload, root=0)
    report_status = None
    if rank == 0:
        rank_payloads = sorted(gathered or [], key=lambda item: item["rank"])
        report_status = (
            "PASS" if rank_payloads and all(
                item["status"] == "PASS"
                for item in rank_payloads) else "FAIL")
        solver_medians = [
            item["timings"]["clover_schur_bistabcg_solve"]["steady"][
                "operation_seconds"]["median"]
            for item in rank_payloads
            if item["timings"]["clover_schur_bistabcg_solve"]["steady"][
                "operation_seconds"]["median"] is not None
        ]
        solver_wall = max(solver_medians) if solver_medians else None
        reused_operation_medians = [
            item["timings"]["clover_schur_bistabcg_solve_reused"]["steady"][
                "operation_seconds"]["median"]
            for item in rank_payloads
            if args.reuse_runtime and item["timings"][
                "clover_schur_bistabcg_solve_reused"]["steady"][
                    "operation_seconds"]["median"] is not None
        ]
        reused_amortized_values = [
            item["timings"]["clover_schur_bistabcg_solve_reused"]["reuse"][
                "amortized_lifecycle_seconds"]
            for item in rank_payloads if args.reuse_runtime
        ]
        reused_setup_values = [
            item["timings"]["clover_schur_bistabcg_solve_reused"]["reuse"][
                "setup_seconds"]
            for item in rank_payloads if args.reuse_runtime
        ]
        reused_close_values = [
            item["timings"]["clover_schur_bistabcg_solve_reused"]["reuse"][
                "close_seconds"]
            for item in rank_payloads if args.reuse_runtime
        ]
        reused_operation_wall = (
            max(reused_operation_medians)
            if reused_operation_medians else None)
        reused_amortized_wall = (
            max(reused_amortized_values)
            if reused_amortized_values else None)
        reused_setup_wall = (
            max(reused_setup_values) if reused_setup_values else None)
        reused_close_wall = (
            max(reused_close_values) if reused_close_values else None)
        iteration_values = [
            int(value) for item in rank_payloads
            for value in item["solver_native"]["iterations"]]
        representative_iterations = (
            int(np.median(iteration_values)) if iteration_values else 0)
        gflops = (
            common.estimated_clover_gflops(
                lattice, representative_iterations, solver_wall)
            if solver_wall and representative_iterations > 0 else None)
        libqcu_record = {
            "path": str(libqcu),
            "sha256": _sha256_file(libqcu),
        }
        report = {
            "schema": "pyqcu.qcu.lime_mpi_benchmark.v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": report_status,
            "command": list(sys.argv),
            "inputs": {
                "gauge": {
                    "path": str(gauge_path),
                    "sha256": _sha256_file(gauge_path),
                    "format": gauge_format,
                    "dataset": gauge_dataset,
                    "metadata": metadata,
                },
                "source": {
                    "path": None if source_path is None else str(source_path),
                    "sha256": source_sha256,
                    "origin": source_origin,
                    "format": source_format,
                    "dataset": source_dataset,
                    "seed": int(args.seed),
                },
            },
            "physics": {
                "lattice": list(lattice),
                "mass": mass,
                "mass_source": mass_source,
                "kappa": 1.0 / (2.0 * mass + 8.0),
                "csw": csw,
                "csw_source": csw_source,
                "gauge_csw": gauge_csw,
                "atol": float(args.atol),
                "max_iter": int(args.max_iter),
                "dtype": args.dtype,
            },
            "execution": {
                "mpi_size": size,
                "grid": list(grid),
                "warmups": int(args.warmups),
                "repeats": int(args.repeats),
                "device_policy": "PYQCU_MPI_DEVICE_ID=<MPI shared-memory local rank>",
                "mpi_device_aware": os.environ.get(
                    "PYQCU_MPI_DEVICE_AWARE", "auto"),
                "libqcu": libqcu_record,
            },
            "strong_scaling": {
                "global_sites": int(np.prod(lattice, dtype=np.int64)),
                "solver_steady_operation_seconds_max_rank": solver_wall,
                "solver_reused_steady_operation_seconds_max_rank": (
                    reused_operation_wall),
                "solver_reused_amortized_lifecycle_seconds_max_rank": (
                    reused_amortized_wall),
                "solver_iterations_median": representative_iterations,
                "estimated_gflops": gflops,
                "baseline": None,
                "note": (
                    "Compare separate --grid/MPI-size runs with identical gauge, "
                    "RHS seed, mass, csw, dtype and solver controls"),
            },
            "reuse_runtime": {
                "enabled": bool(args.reuse_runtime),
                "setup_seconds_max_rank": reused_setup_wall,
                "close_seconds_max_rank": reused_close_wall,
                "steady_operation_seconds_max_rank": reused_operation_wall,
                "amortized_lifecycle_seconds_max_rank": reused_amortized_wall,
                "amortization_scope": (
                    "setup + cold + warmups + steady operations + close, "
                    "divided by all solves in the persistent session"),
            },
            "ranks": rank_payloads,
            "warnings": warnings,
            "limitations": [
                "Native convergence is conservatively inferred from iterations < max_iter.",
                "Cold is the first call within each stage; only even Clover setup is process-cold.",
                "estimated_gflops is a documented operation-count model, not a hardware counter.",
            ],
        }
        report_path = (
            REPO / "logs" / "qcu" /
            f"benchmark_qcu_lime_mpi_{size}rank.json"
            if args.report is None else Path(args.report).expanduser())
        common.write_json(report_path, report)
        print(json.dumps({
            "status": report_status,
            "report": str(report_path),
            "mpi_size": size,
            "solver_steady_operation_seconds_max_rank": solver_wall,
            "solver_reused_steady_operation_seconds_max_rank": (
                reused_operation_wall),
            "solver_reused_amortized_lifecycle_seconds_max_rank": (
                reused_amortized_wall),
            "estimated_gflops": None if gflops is None else gflops["estimated_gflops"],
            "independent_true_residual_verified": all(
                item["independent_true_residual"]["verified"]
                for item in rank_payloads),
            "independent_true_residual_max": max(
                item["independent_true_residual"]["value"]
                for item in rank_payloads),
            "independent_true_residual_pass": all(
                item["independent_true_residual"]["pass"]
                for item in rank_payloads),
        }, ensure_ascii=False), flush=True)

    final_status = comm.bcast(report_status, root=0)
    return 0 if final_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
