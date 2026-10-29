"""Distributed Legacy-MG cache construction for the simple MPI entry point."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from pyqcu.testing.simple._common import (
    qcu_gauge_to_kupl,
    qdp_gauge_to_qcu,
    read_h5_metadata,
    resolve_path,
    write_json,
)
from pyqcu.testing.simple.kupl_bistabcg import NumpyTensor
from pyqcu.testing.simple.kupl_multigrid import build_coarse_assets
from pyqcu.testing.simple.mpi._common import (
    canonical_gauge,
    is_lime_file,
    lime_metadata,
    local_geometry,
    read_gauge,
    resolve_grid,
    validate_lattice,
)
from pyqcu.testing.simple.mpi.mpi_ranks_power_of_two import validate_rank_count
from pyqcu.tools._legacy_mg_cache import (
    LEGACY_MG_ASSET_NAMES,
    file_sha256,
    legacy_mg_cache_identity,
    legacy_mg_cache_shapes,
    load_legacy_mg_cache_slab,
    logical_array_sha256,
    save_legacy_mg_cache,
    validate_legacy_mg_cache,
    write_legacy_mg_cache_manifest,
)


ASSET_LATTICE_AXES = {
    "lonv": (2, 4, 6, 8),
    "hnn": (4, 5, 6, 7),
    "hdg": (5, 6, 7, 8),
    "sit": (2, 3, 4, 5),
}


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
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
    parser.add_argument("--build-device", choices=("cpu", "cuda"),
                        default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grid", nargs=4, type=int, default=None)
    parser.add_argument("--ranks", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _hdf5_gauge_slab(
    path: Path,
    starts: tuple[int, int, int, int],
    local: tuple[int, int, int, int],
    dtype: np.dtype,
) -> np.ndarray:
    import h5py

    lattice_slices = tuple(
        slice(int(starts[axis]), int(starts[axis] + local[axis]))
        for axis in range(4)
    )
    with h5py.File(path, "r") as handle:
        dataset = handle["data"]
        if dataset.ndim != 7:
            raise ValueError(f"unsupported gauge HDF5 shape {dataset.shape}")
        if tuple(dataset.shape[:3]) == (3, 3, 4):
            slab = dataset[(slice(None), slice(None), slice(None),
                            *lattice_slices)]
        elif int(dataset.shape[0]) == 4 and tuple(dataset.shape[-2:]) == (3, 3):
            raw = dataset[(slice(None), *lattice_slices,
                           slice(None), slice(None))]
            slab = qdp_gauge_to_qcu(raw)
        else:
            raise ValueError(f"unsupported gauge HDF5 shape {dataset.shape}")
    return np.ascontiguousarray(slab, dtype=dtype)


def _load_local_gauge(
    args: argparse.Namespace,
    comm: Any,
    lattice: tuple[int, int, int, int],
    local: tuple[int, int, int, int],
    starts: tuple[int, int, int, int],
    dtype: np.dtype,
) -> tuple[np.ndarray, dict[str, Any], str]:
    path = resolve_path(args.gauge, Path(args.gauge).expanduser())
    rank = int(comm.Get_rank())
    if is_lime_file(path):
        payload: tuple[dict[str, Any], str, Exception | None] | None = None
        local_slabs: list[np.ndarray] | None = None
        if rank == 0:
            try:
                full, metadata, input_format = read_gauge(
                    path, lattice, dtype)
                local_slabs = []
                grid = tuple(args.grid)
                for source_rank in range(int(comm.Get_size())):
                    source_local, source_starts, _ = local_geometry(
                        lattice, grid, source_rank)
                    selection = [slice(None)] * (full.ndim - 4)
                    selection.extend(
                        slice(source_starts[axis],
                              source_starts[axis] + source_local[axis])
                        for axis in range(4)
                    )
                    local_slabs.append(np.ascontiguousarray(
                        full[tuple(selection)]))
                payload = (metadata, input_format, None)
            except Exception as exc:  # propagated symmetrically below
                payload = ({}, "LIME/SciDAC", exc)
        metadata, input_format, error = comm.bcast(payload, root=0)
        if error is not None:
            raise RuntimeError(f"rank 0 failed to read {path}: {error}")
        gauge = comm.scatter(local_slabs, root=0)
        return np.ascontiguousarray(gauge, dtype=dtype), metadata, input_format

    metadata = read_h5_metadata(path) if rank == 0 else None
    metadata = comm.bcast(metadata, root=0)
    if path.suffix.lower() not in (".h5", ".hdf5"):
        raise ValueError(f"gauge input must be LIME or HDF5: {path}")
    gauge = _hdf5_gauge_slab(path, starts, local, dtype)
    return gauge, dict(metadata or {}), "HDF5"


def _cuda_local_rank(comm: Any) -> int:
    from mpi4py import MPI

    shared = comm.Split_type(MPI.COMM_TYPE_SHARED, 0)
    try:
        return int(shared.Get_rank())
    finally:
        shared.Free()


def _build_cuda_null_vectors(
    gauge: np.ndarray,
    local_lattice: tuple[int, int, int, int],
    grid: tuple[int, int, int, int],
    comm: Any,
    args: argparse.Namespace,
    dtype: np.dtype,
) -> tuple[np.ndarray, dict[str, float]]:
    local_rank = _cuda_local_rank(comm)
    os.environ["PYQCU_MPI_DEVICE_ID"] = str(local_rank)
    import torch
    from pyqcu.cuda import define, qcu

    device = torch.device(f"cuda:{local_rank}")
    torch_dtype = (
        torch.complex64 if dtype == np.dtype(np.complex64)
        else torch.complex128)
    fine_odd = (*local_lattice[:3], local_lattice[3] // 2)
    coarse = tuple(fine_odd[axis] // args.mg_grid[axis]
                   for axis in range(4))
    block = tuple(args.mg_grid)
    null_shape = (
        int(args.dof[1]), 12, coarse[0], block[0], coarse[1], block[1],
        coarse[2], block[2], coarse[3], block[3])
    onsite_shape = (4, 3, 4, 3, *fine_odd)

    gauge_device = torch.from_numpy(
        canonical_gauge(gauge)).to(device=device, dtype=torch_dtype).contiguous()
    clover_even = torch.empty(
        onsite_shape, device=device, dtype=torch_dtype)
    clover_odd = torch.empty_like(clover_even)
    clover_even_inv = torch.empty_like(clover_even)
    clover_odd_inv = torch.empty_like(clover_even)
    null_vectors = torch.empty(
        null_shape, device=device, dtype=torch_dtype)
    params = torch.zeros(define._PARAMS_SIZE_, dtype=torch.int32)
    params[define._LAT_X_], params[define._LAT_Y_] = local_lattice[:2]
    params[define._LAT_Z_], params[define._LAT_T_] = local_lattice[2:]
    params[define._LAT_XYZT_] = int(np.prod(local_lattice))
    params[define._GRID_X_], params[define._GRID_Y_] = grid[:2]
    params[define._GRID_Z_], params[define._GRID_T_] = grid[2:]
    params[define._NODE_RANK_] = int(comm.Get_rank())
    params[define._NODE_SIZE_] = int(comm.Get_size())
    params[define._MAX_ITER_] = 1000
    params[define._DATA_TYPE_] = int(define.epytd(torch_dtype).item())
    params[define._SET_INDEX_] = 0
    params[define._SET_PLAN_] = define._SET_PLAN2_
    params[define._SEED_] = int(args.seed)
    argv = define.make_argv(
        float(args.mass), csw=float(args.csw), atol=1.0e-7,
        data_type=int(params[define._DATA_TYPE_]))
    set_ptrs = torch.zeros(define._SET_PTRS_SIZE_, dtype=torch.int64)
    timings = {"clover_seconds": 0.0, "null_vector_seconds": 0.0,
               "host_transfer_seconds": 0.0}
    qcu.applyInitQcu(set_ptrs, params, argv)
    try:
        stage = time.perf_counter()
        params[define._PARITY_] = 0
        qcu.applyCloversQcu(
            clover_even, clover_even_inv, gauge_device, set_ptrs, params)
        params[define._PARITY_] = 1
        qcu.applyCloversQcu(
            clover_odd, clover_odd_inv, gauge_device, set_ptrs, params)
        timings["clover_seconds"] = time.perf_counter() - stage
        params[define._SET_PLAN_] = define._SET_PLAN1_
        stage = time.perf_counter()
        status = qcu.applyMultigridBuildNullVectorsQcu(
            null_vectors, gauge_device, clover_even, clover_odd,
            clover_even_inv, clover_odd_inv, set_ptrs, params,
            int(args.dof[1]), 12, *fine_odd, *coarse, int(args.nv_iters),
            int(args.seed), 1.0e-2)
        if int(status) != 0:
            raise RuntimeError(f"CUDA distributed null-vector setup failed: {status}")
        torch.cuda.synchronize(device)
        timings["null_vector_seconds"] = time.perf_counter() - stage
    finally:
        qcu.applyEndQcu(set_ptrs, params)
    stage = time.perf_counter()
    host = null_vectors.detach().cpu().contiguous().numpy()
    timings["host_transfer_seconds"] = time.perf_counter() - stage
    return np.ascontiguousarray(host, dtype=dtype), timings


def _build_kupl_null_vectors(
    gauge: np.ndarray,
    local_lattice: tuple[int, int, int, int],
    grid: tuple[int, int, int, int],
    comm: Any,
    args: argparse.Namespace,
    dtype: np.dtype,
) -> tuple[np.ndarray, dict[str, float]]:
    from pyqcu.kupl import define, qcu

    fine_odd = (*local_lattice[:3], local_lattice[3] // 2)
    coarse = tuple(fine_odd[axis] // args.mg_grid[axis]
                   for axis in range(4))
    block = tuple(args.mg_grid)
    null_shape = (
        int(args.dof[1]), 12, coarse[0], block[0], coarse[1], block[1],
        coarse[2], block[2], coarse[3], block[3])
    onsite_shape = (4, 3, 4, 3, *fine_odd)
    gauge_parity = canonical_gauge(gauge)
    clover_even = np.empty(onsite_shape, dtype=dtype)
    clover_odd = np.empty(onsite_shape, dtype=dtype)
    clover_even_inv = np.empty(onsite_shape, dtype=dtype)
    clover_odd_inv = np.empty(onsite_shape, dtype=dtype)
    null_vectors = np.empty(null_shape, dtype=dtype)
    params = np.zeros(58, dtype=np.int32)
    params[0:4] = local_lattice
    params[4] = int(np.prod(local_lattice))
    params[5:9] = grid
    params[10] = int(comm.Get_rank())
    params[11] = int(comm.Get_size())
    params[13] = 1000
    params[14] = (
        define.LAT_C64 if dtype == np.dtype(np.complex64)
        else define.LAT_C128)
    params[15] = 0
    params[16] = 2
    params[52] = int(args.seed)
    argv = np.zeros(
        8, dtype=np.float32 if dtype == np.dtype(np.complex64)
        else np.float64)
    argv[0], argv[1], argv[2], argv[7] = (
        args.mass, 1.0e-7, 0.1, args.csw)
    set_ptrs = np.zeros(100, dtype=np.int64)
    args_gauge = NumpyTensor(gauge_parity)
    args_even = NumpyTensor(clover_even)
    args_odd = NumpyTensor(clover_odd)
    args_even_inv = NumpyTensor(clover_even_inv)
    args_odd_inv = NumpyTensor(clover_odd_inv)
    args_null = NumpyTensor(null_vectors)
    args_params = NumpyTensor(params)
    args_argv = NumpyTensor(argv)
    args_set_ptrs = NumpyTensor(set_ptrs)
    timings = {"clover_seconds": 0.0, "null_vector_seconds": 0.0,
               "host_transfer_seconds": 0.0}
    qcu.applyInitQcu(args_set_ptrs, args_params, args_argv)
    try:
        stage = time.perf_counter()
        params[9] = 0
        qcu.applyCloversQcu(
            args_even, args_even_inv, args_gauge, args_set_ptrs, args_params)
        params[9] = 1
        qcu.applyCloversQcu(
            args_odd, args_odd_inv, args_gauge, args_set_ptrs, args_params)
        timings["clover_seconds"] = time.perf_counter() - stage
        params[16] = 1
        stage = time.perf_counter()
        status = qcu.applyMultigridBuildNullVectorsQcu(
            args_null, args_gauge, args_even, args_odd, args_even_inv,
            args_odd_inv, args_set_ptrs, args_params, int(args.dof[1]), 12,
            *fine_odd, *coarse, int(args.nv_iters), int(args.seed), 1.0e-2)
        if int(status) != 0:
            raise RuntimeError(f"KUPL distributed null-vector setup failed: {status}")
        timings["null_vector_seconds"] = time.perf_counter() - stage
    finally:
        qcu.applyEndQcu(args_set_ptrs, args_params)
    if args.verbose and int(comm.Get_rank()) == 0:
        print(
            f"PYQCU::MPI::MG_CACHE:\n local={local_lattice} "
            f"fine_odd={fine_odd} coarse={coarse} block={block} "
            f"null_shape={null_vectors.shape}",
            flush=True)
    return null_vectors, timings


def _fill_last4(
    destination: np.ndarray,
    source: np.ndarray,
    starts: Sequence[int],
) -> None:
    selection = [slice(None)] * (destination.ndim - 4)
    selection.extend(
        slice(int(starts[axis]), int(starts[axis] + source.shape[-4 + axis]))
        for axis in range(4)
    )
    destination[tuple(selection)] = source


def _fill_asset(
    destination: np.ndarray,
    source: np.ndarray,
    name: str,
    starts: Sequence[int],
) -> None:
    selection = [slice(None)] * destination.ndim
    for axis, storage_axis in enumerate(ASSET_LATTICE_AXES[name]):
        selection[storage_axis] = slice(
            int(starts[axis]), int(starts[axis] + source.shape[storage_axis]))
    destination[tuple(selection)] = source


def _asset_patch(
    source: np.ndarray,
    name: str,
    starts: Sequence[int],
    local: Sequence[int],
) -> np.ndarray:
    selection = [slice(None)] * source.ndim
    for axis, storage_axis in enumerate(ASSET_LATTICE_AXES[name]):
        selection[storage_axis] = slice(
            int(starts[axis]), int(starts[axis] + local[axis]))
    return np.ascontiguousarray(source[tuple(selection)])


def _patch_axis(
    global_extent: int,
    local_extent: int,
    coordinate: int,
    block: int,
) -> tuple[int, int, int]:
    halo = 2 * block
    desired = local_extent + 2 * halo
    if desired <= global_extent:
        start = (coordinate * local_extent - halo) % global_extent
        center = halo
        return start, desired, center
    if global_extent % block == 0:
        center = coordinate * local_extent
        return 0, global_extent, center
    raise ValueError(
        f"cannot build aligned overlap patch for global={global_extent} "
        f"local={local_extent} block={block}")


def _periodic_patch(
    source: np.ndarray,
    start: Sequence[int],
    extent: Sequence[int],
) -> np.ndarray:
    value = source
    last4 = value.ndim - 4
    for axis in range(4):
        size = int(value.shape[last4 + axis])
        indices = (int(start[axis]) + np.arange(int(extent[axis]))) % size
        value = np.take(value, indices, axis=last4 + axis)
    return np.ascontiguousarray(value)


def _periodic_asset_patch(
    source: np.ndarray,
    name: str,
    start: Sequence[int],
    extent: Sequence[int],
) -> np.ndarray:
    value = source
    for axis, storage_axis in enumerate(ASSET_LATTICE_AXES[name]):
        size = int(value.shape[storage_axis])
        indices = (int(start[axis]) + np.arange(int(extent[axis]))) % size
        value = np.take(value, indices, axis=storage_axis)
    return np.ascontiguousarray(value)


def _global_gauge(
    local_gauge: np.ndarray,
    lattice: tuple[int, int, int, int],
    grid: tuple[int, int, int, int],
    comm: Any,
) -> np.ndarray | None:
    gathered = comm.gather(local_gauge, root=0)
    if int(comm.Get_rank()) != 0:
        return None
    global_gauge = np.empty((3, 3, 4, *lattice), dtype=local_gauge.dtype)
    for source_rank, value in enumerate(gathered or []):
        local, starts, _ = local_geometry(lattice, grid, source_rank)
        _fill_last4(global_gauge, value, starts)
    return global_gauge


def _global_lonv(
    local_lonv: np.ndarray,
    lattice: tuple[int, int, int, int],
    grid: tuple[int, int, int, int],
    mg_grid: Sequence[int],
    shape: tuple[int, ...],
    comm: Any,
) -> np.ndarray | None:
    gathered = comm.gather(local_lonv, root=0)
    if int(comm.Get_rank()) != 0:
        return None
    result = np.empty(shape, dtype=local_lonv.dtype)
    for source_rank, value in enumerate(gathered or []):
        local_fine, _, coordinate = local_geometry(lattice, grid, source_rank)
        local_coarse = tuple(
            (local_fine[axis] // 2 if axis == 3 else local_fine[axis])
            // int(mg_grid[axis]) for axis in range(4))
        coarse_starts = tuple(
            coordinate[axis] * local_coarse[axis] for axis in range(4))
        _fill_asset(result, value, "lonv", coarse_starts)
    return result


def _global_assets(
    local_assets: dict[str, np.ndarray],
    shapes: dict[str, tuple[int, ...]],
    lattice: tuple[int, int, int, int],
    grid: tuple[int, int, int, int],
    mg_grid: Sequence[int],
    comm: Any,
) -> dict[str, np.ndarray] | None:
    gathered = {
        name: comm.gather(local_assets[name], root=0)
        for name in LEGACY_MG_ASSET_NAMES
    }
    if int(comm.Get_rank()) != 0:
        return None
    result = {
        name: np.empty(shapes[name], dtype=local_assets[name].dtype)
        for name in LEGACY_MG_ASSET_NAMES
    }
    for source_rank in range(int(comm.Get_size())):
        local_fine, _, coordinate = local_geometry(lattice, grid, source_rank)
        local_coarse = tuple(
            (local_fine[axis] // 2 if axis == 3 else local_fine[axis])
            // int(mg_grid[axis]) for axis in range(4))
        coarse_starts = tuple(
            coordinate[axis] * local_coarse[axis] for axis in range(4))
        for name in LEGACY_MG_ASSET_NAMES:
            _fill_asset(
                result[name], gathered[name][source_rank], name, coarse_starts)
    return result


def _main_native(argv: Sequence[str] | None = None) -> int:
    from mpi4py import MPI

    args = _parse_args(argv)
    comm = MPI.COMM_WORLD
    rank = int(comm.Get_rank())
    try:
        size = validate_rank_count(int(comm.Get_size()))
        if args.ranks is not None and int(args.ranks) != size:
            raise ValueError(
                f"--ranks={args.ranks} does not match MPI_COMM_WORLD size {size}")
        lattice = validate_lattice(args.lat)
        grid = resolve_grid(lattice, size, args.grid)
        local, starts, coordinate = local_geometry(lattice, grid, rank)
        args.grid = grid
        if len(args.dof) < 2 or int(args.dof[0]) != 12:
            raise ValueError("Legacy MG cache requires --dof 12 E [E ...]")
        if int(args.dof[1]) <= 0 or args.nv_iters <= 0 or args.batch_size <= 0:
            raise ValueError("dof[1], nv-iters and batch-size must be positive")
        fine_odd = (*local[:3], local[3] // 2)
        if any(fine_odd[axis] % args.mg_grid[axis] for axis in range(4)):
            raise ValueError(
                f"local odd lattice {fine_odd} is not divisible by "
                f"mg-grid {args.mg_grid}")
    except ValueError as exc:
        if rank == 0:
            print(f"error: {exc}", file=os.sys.stderr)
        return 2

    dtype = np.dtype(
        np.complex64 if args.dtype == "c64" else np.complex128)
    stage = time.perf_counter()
    try:
        local_gauge, metadata, input_format = _load_local_gauge(
            args, comm, lattice, local, starts, dtype)
        loading_seconds = time.perf_counter() - stage
        stage = time.perf_counter()
        if args.build_device == "cuda":
            local_lonv, null_timings = _build_cuda_null_vectors(
                local_gauge, local, grid, comm, args, dtype)
        else:
            local_lonv, null_timings = _build_kupl_null_vectors(
                local_gauge, local, grid, comm, args, dtype)
        setup_seconds = time.perf_counter() - stage

        shapes = legacy_mg_cache_shapes(lattice, args.mg_grid, args.dof[:2])
        global_gauge = _global_gauge(local_gauge, lattice, grid, comm)
        global_lonv = _global_lonv(
            local_lonv, lattice, grid, args.mg_grid, shapes["lonv"], comm)

        patch_spec = tuple(
            _patch_axis(lattice[axis], local[axis], coordinate[axis],
                        int(args.mg_grid[axis]))
            for axis in range(4))
        patch_starts = tuple(item[0] for item in patch_spec)
        patch_extent = tuple(item[1] for item in patch_spec)
        patch_centers = tuple(item[2] for item in patch_spec)
        coarse_centers = tuple(
            patch_centers[axis] // int(args.mg_grid[axis])
            for axis in range(4))

        gauge_patches = None
        lonv_patches = None
        if rank == 0:
            assert global_gauge is not None and global_lonv is not None
            gauge_patches = []
            lonv_patches = []
            for source_rank in range(size):
                source_local, _, source_coordinate = local_geometry(
                    lattice, grid, source_rank)
                source_spec = tuple(
                    _patch_axis(
                        lattice[axis], source_local[axis],
                        source_coordinate[axis], int(args.mg_grid[axis]))
                    for axis in range(4))
                gauge_patches.append(_periodic_patch(
                    global_gauge,
                    tuple(item[0] for item in source_spec),
                    tuple(item[1] for item in source_spec)))
                lonv_starts = tuple(
                    item[0] // (
                        int(args.mg_grid[axis]) *
                        (2 if axis == 3 else 1))
                    for axis, item in enumerate(source_spec))
                lonv_extent = tuple(
                    item[1] // (
                        int(args.mg_grid[axis]) *
                        (2 if axis == 3 else 1))
                    for axis, item in enumerate(source_spec))
                lonv_patches.append(_periodic_asset_patch(
                    global_lonv, "lonv", lonv_starts, lonv_extent))
        gauge_patch = comm.scatter(gauge_patches, root=0)
        lonv_patch = comm.scatter(lonv_patches, root=0)
        patch_lattice = tuple(int(value) for value in patch_extent)
        stencil_stats: dict[str, Any] = {}
        stage = time.perf_counter()
        # A multi-rank CUDA LatticeSet always uses MPI_COMM_WORLD for its
        # Clover/dslash collectives, even when the overlap patch grid is
        # 1x1x1x1. Different ranks can therefore enter incompatible CUDA halo
        # and Python collectives. Keep distributed null-vector generation on
        # the GPUs, but execute the independent patch stencil on CPU unless
        # the whole job is one rank.
        stencil_device = (
            f"cuda:{_cuda_local_rank(comm)}"
            if args.build_device == "cuda" and size == 1 else "cpu")
        patch_assets = build_coarse_assets(
            gauge_patch, patch_lattice, args.mass,
            1.0 / (2.0 * args.mass + 8.0), args.csw, dtype, 2,
            args.dof[:2], list(args.mg_grid), args.nv_iters, args.seed,
            verbose=bool(args.verbose), local_stencil=True,
            local_stencil_batch_size=args.batch_size,
            device_name=stencil_device,
            setup_stats=stencil_stats, supplied_null_vectors=lonv_patch,
            validate_galerkin=False)
        stencil_seconds = time.perf_counter() - stage
        patch_coarse = tuple(
            patch_extent[axis] // int(args.mg_grid[axis])
            for axis in range(4))
        local_coarse = tuple(
            (fine_odd[axis]) // int(args.mg_grid[axis])
            for axis in range(4))
        local_assets = {
            name: _asset_patch(
                patch_assets[index][0], name, coarse_centers, local_coarse)
            for index, name in enumerate(LEGACY_MG_ASSET_NAMES)
        }
        global_assets = _global_assets(
            local_assets, shapes, lattice, grid, args.mg_grid, comm)

        status = 0
        output = Path(comm.bcast(
            str(resolve_path(args.output, args.output.expanduser()))
            if rank == 0 else None, root=0))
        report_path = (
            resolve_path(args.report, args.report.expanduser())
            if args.report is not None else output.with_suffix(".json"))
        if rank == 0:
            assert global_gauge is not None and global_assets is not None
            gauge_canonical_sha = logical_array_sha256(global_gauge)
            identity = legacy_mg_cache_identity(
                gauge_canonical_sha256=gauge_canonical_sha,
                gauge_file_sha256=file_sha256(
                    resolve_path(args.gauge, Path(args.gauge).expanduser())),
                lattice=lattice, dtype=dtype, mass=args.mass, csw=args.csw,
                dof=args.dof[:2], mg_grid=args.mg_grid,
                nv_iters=args.nv_iters, seed=args.seed)
            rank_records = comm.gather({
                "rank": rank,
                "coordinate": list(coordinate),
                "local": list(local),
                "local_coarse": list(local_coarse),
                "patch_start": list(patch_starts),
                "patch_extent": list(patch_extent),
                "patch_coarse": list(patch_coarse),
                "null_timings": null_timings,
                "stencil_seconds": stencil_seconds,
            }, root=0)
            report = save_legacy_mg_cache(
                output, global_assets, identity=identity,
                setup_method="mpi-distributed-null-vector+overlap-local-stencil",
                setup_backend=(
                    "cuda-qcu-mpi-cxx" if args.build_device == "cuda"
                    else "kupl-qcu-mpi-cxx"),
                setup_validation={
                    "ranks": size,
                    "grid": list(grid),
                    "local": list(local),
                    "patch_extent": list(patch_extent),
                    "patch_coarse": list(patch_coarse),
                    "rank_records": rank_records or [],
                    "gauge_input_format": input_format,
                    "stencil_device": stencil_device,
                    "stencil": stencil_stats,
                    "loading_seconds": loading_seconds,
                    "distributed_null_vector_seconds": setup_seconds,
                    "stencil_seconds": stencil_seconds,
                },
                source_metadata=metadata, setup_seconds=loading_seconds + setup_seconds + stencil_seconds)
            validate_legacy_mg_cache(
                output, identity, verify_file_sha256=True,
                verify_dataset_sha256=True)
            report.update({
                "gauge": str(resolve_path(
                    args.gauge, Path(args.gauge).expanduser())),
                "build_device": args.build_device,
                "output": str(output),
                "distributed": True,
                "global_asset_sha256": {
                    name: logical_array_sha256(global_assets[name])
                    for name in LEGACY_MG_ASSET_NAMES},
            })
        else:
            comm.gather({
                "rank": rank,
                "coordinate": list(coordinate),
                "local": list(local),
                "local_coarse": list(local_coarse),
                "patch_start": list(patch_starts),
                "patch_extent": list(patch_extent),
                "patch_coarse": list(patch_coarse),
                "null_timings": null_timings,
                "stencil_seconds": stencil_seconds,
            }, root=0)
        status = int(comm.bcast(status, root=0))
        comm.Barrier()

        reconstructed = True
        try:
            cache_slabs = load_legacy_mg_cache_slab(
                output, dtype, starts=tuple(
                    coordinate[axis] * local_coarse[axis]
                    for axis in range(4)),
                local_shape=local_coarse)
            reconstructed = all(
                logical_array_sha256(cache_slabs[name]) ==
                logical_array_sha256(local_assets[name])
                for name in LEGACY_MG_ASSET_NAMES)
        except BaseException as exc:
            reconstructed = False
            if rank == 0:
                print(f"rank-0 cache slab validation failed: {exc}",
                      file=os.sys.stderr)
        reconstruction_checks = comm.allgather({
            "rank": rank,
            "pass": reconstructed,
        })
        if not all(item["pass"] for item in reconstruction_checks):
            if rank == 0:
                print("error: published cache does not match rank slabs",
                      file=os.sys.stderr)
            return 1
        if rank == 0:
            report["reconstruction_verified"] = True
            report["rank_slab_reconstruction"] = reconstruction_checks
            write_legacy_mg_cache_manifest(output, report)
            if report_path != output.with_suffix(".json"):
                write_json(report_path, report)
            print(json.dumps(report, ensure_ascii=False), flush=True)
        return status
    except BaseException:
        import traceback

        error = traceback.format_exc()
        errors = comm.allgather(error)
        if rank == 0:
            for source_rank, value in enumerate(errors):
                if value:
                    print(f"rank {source_rank} failed:\n{value}",
                          file=os.sys.stderr, end="")
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run the native MPI builder on every rank."""
    return _main_native(argv)


__all__ = ["main"]
