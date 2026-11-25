#!/usr/bin/env python3
"""Minimal KUPL Clover multigrid example with Galerkin verification."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from pyqcu.tools._legacy_mg_cache import (
    LEGACY_MG_ASSET_COARSE_AXES,
    LEGACY_MG_ASSET_NAMES,
    build_legacy_mg_ranked_cache_manifest,
    file_sha256,
    is_legacy_mg_ranked_cache,
    legacy_mg_cache_identity,
    legacy_mg_cache_shapes,
    legacy_mg_local_shapes,
    load_legacy_mg_cache_patch,
    load_legacy_mg_cache_slab,
    logical_array_sha256,
    null_vector_builder,
    read_legacy_mg_ranked_manifest,
    save_legacy_mg_cache,
    validate_legacy_mg_cache,
    write_legacy_mg_assets,
    write_legacy_mg_rank_slice,
    write_legacy_mg_ranked_cache,
)

try:
    from mpi4py import MPI
except ImportError:  # pragma: no cover - KUPL itself requires MPI.
    MPI = None

try:
    from ._common import (DATA_DIR, DEFAULT_ATOL, DEFAULT_BLOCK, DEFAULT_MASS,
                          convergence_status, json_safe, kupl_fermion_to_qcu,
                          qcu_fermion_to_kupl, qcu_gauge_to_kupl,
                          read_h5_metadata, resolve_csw, resolve_path,
                          solver_acceptance, write_h5, write_json)
    from ._layouts import qcu_gauge_to_kupl as gauge_to_parity
    from ._layouts import qdp_gauge_to_qcu
    from .kupl_bistabcg import (NumpyTensor, PLAN_CLOVER, PLAN_SOLVER,
                                PARITY, SET_PLAN, cpu_memory_snapshot,
                                kupl_backend, load_gauge, load_source,
                                make_params, prebind_kupl_inputs,
                                relative_full_residual)
    from .kupl_residual import (distributed_relative_residual,
                                gather_solution)
except ImportError:
    from _common import (DATA_DIR, DEFAULT_ATOL, DEFAULT_BLOCK, DEFAULT_MASS,
                         convergence_status, json_safe, kupl_fermion_to_qcu,
                         qcu_fermion_to_kupl, qcu_gauge_to_kupl,
                         read_h5_metadata, resolve_csw, resolve_path,
                         solver_acceptance, write_h5, write_json)
    from _layouts import qcu_gauge_to_kupl as gauge_to_parity
    from _layouts import qdp_gauge_to_qcu
    from kupl_bistabcg import (NumpyTensor, PLAN_CLOVER, PLAN_SOLVER,
                               PARITY, SET_PLAN, cpu_memory_snapshot,
                               kupl_backend, load_gauge, load_source,
                               make_params, prebind_kupl_inputs,
                               relative_full_residual)
    from kupl_residual import (distributed_relative_residual,
                               gather_solution)

if str(Path(__file__).resolve().parents[3]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


MG_NUM_LEVEL, MG_LEVEL_INDEX = 17, 18
MG_LEVEL_BASE, MG_LEVEL_STRIDE = 19, 8
MG_LEVEL_MAX_ITER = 5
MG_LEVEL_DATA_TYPE = 6
MG_LEVEL_NUM_RESTART = 7
MG_USE_DEFLATE = 55
SET_PTRS_COARSE_BASE = 30
SET_PTRS_LEGACY_LAYOUT = 99
LEGACY_LAYOUT_ODD_SCHUR = 1
ARGV_MG_LEVEL_BASE = 3
DATA_TYPE = 14


def _environment_enabled(name: str) -> bool:
    value = os.environ.get(name)
    return value is not None and value != "" and value.lower() not in {
        "0", "false", "no", "off"}


def _hdf5_mpio_read_enabled(comm) -> bool:
    """Return a uniform opt-in decision for collective HDF5 input reads."""
    requested = int(_environment_enabled("KUPL_HDF5_MPIO_READ"))
    if comm is None:
        return bool(requested)
    votes = comm.allgather(requested)
    if len(set(votes)) != 1:
        raise RuntimeError(
            "KUPL_HDF5_MPIO_READ must be identical on every MPI rank")
    if not requested:
        return False
    import h5py

    supported = int(bool(h5py.get_config().mpi))
    support = comm.allgather(supported)
    if not all(support):
        raise RuntimeError(
            "KUPL_HDF5_MPIO_READ=1 requires h5py built with MPI support")
    return True


def resolve_legacy_layout(requested: str, use_coarse_assets: bool,
                          null_vectors: tuple[np.ndarray, ...],
                          local_lattice: tuple[int, int, int, int]) -> int:
    """Resolve the KUPL Legacy fine layout from the loaded cache geometry."""
    if not use_coarse_assets or not null_vectors:
        if requested == "odd-schur":
            raise ValueError(
                "odd-schur layout requires a compatible coarse cache")
        return 0
    shape = null_vectors[0].shape
    cached_time = int(shape[-2]) * int(shape[-1])
    inferred = "full-site" if cached_time == int(local_lattice[3]) else (
        "odd-schur" if cached_time == int(local_lattice[3]) // 2 else None)
    if inferred is None:
        raise ValueError(
            f"coarse cache time span {cached_time} matches neither full-site "
            f"{local_lattice[3]} nor odd-Schur {local_lattice[3] // 2}")
    if requested != "auto" and requested != inferred:
        raise ValueError(
            f"coarse cache layout is {inferred}, but --legacy-layout="
            f"{requested}")
    return LEGACY_LAYOUT_ODD_SCHUR if inferred == "odd-schur" else 0


def inspect_cache_layout(path: Path, lattice: tuple[int, int, int, int],
                         mg_grid: list[int], process_grid: list[int],
                         local_starts: tuple[int, int, int, int],
                         requested: str) -> tuple[str, tuple[int, int, int, int],
                                                   tuple[int, int, int, int]]:
    """Return inferred cache layout and its rank-local coarse slab geometry."""
    import h5py

    if any(int(value) < 1 for value in mg_grid):
        raise ValueError(f"--mg-grid {tuple(mg_grid)} must be positive")
    if is_legacy_mg_ranked_cache(path):
        manifest = read_legacy_mg_ranked_manifest(path)
        declared = manifest["global_shapes"].get("lonv")
        if declared is None or len(declared) != 10:
            raise ValueError(
                f"multi-rank coarse cache {path} has no valid lonv geometry")
        shape = tuple(int(value) for value in declared)
    else:
        with h5py.File(path, "r") as handle:
            if "lonv" not in handle or handle["lonv"].ndim != 10:
                raise ValueError(f"coarse cache {path} has no valid lonv dataset")
            shape = tuple(int(value) for value in handle["lonv"].shape)
    global_time = shape[-2] * shape[-1]
    inferred = (
        "full-site" if global_time == lattice[3] else
        "odd-schur" if global_time == lattice[3] // 2 else None)
    if inferred is None:
        raise ValueError(
            f"coarse cache time span {global_time} matches neither full-site "
            f"{lattice[3]} nor odd-Schur {lattice[3] // 2}")
    if requested != "auto" and requested != inferred:
        raise ValueError(
            f"coarse cache layout is {inferred}, but --legacy-layout={requested}")
    global_coarse_time = global_time // int(mg_grid[3])
    if global_coarse_time % int(process_grid[3]):
        raise ValueError("coarse cache time extent is not divisible by grid_t")
    local_coarse = (
        lattice[0] // process_grid[0] // mg_grid[0],
        lattice[1] // process_grid[1] // mg_grid[1],
        lattice[2] // process_grid[2] // mg_grid[2],
        global_coarse_time // process_grid[3],
    )
    if any(int(extent) < 1 for extent in local_coarse):
        raise ValueError(
            f"coarse lattice {local_coarse} has a zero extent; reduce "
            f"--mg-grid {tuple(mg_grid)} for the {tuple(process_grid)} grid")
    coarse_starts = (
        local_starts[0] // mg_grid[0],
        local_starts[1] // mg_grid[1],
        local_starts[2] // mg_grid[2],
        (local_starts[3] // (2 if inferred == "odd-schur" else 1)) // mg_grid[3],
    )
    return inferred, local_coarse, coarse_starts


def configure_mg(params: np.ndarray, argv: np.ndarray, set_ptrs: np.ndarray,
                 levels: int, atol: float, restart: int,
                 coarse_max_iter: int,
                 coarse_tol_factor: float, deflate: bool,
                 null_vectors: list[Any], hop_nn: list[Any],
                 hop_diag: list[Any], sitting: list[Any]) -> None:
    # OPEN (2026-11-24): the coarse correction is additive and does not reduce
    # the fine iteration count.  L48 c64 32 ranks NUMA-bound: plain Clover
    # BiCGStab 18 iterations / 2.350 s (job 2237987) vs mg-2 18 fine iterations
    # + 4 accepted corrections / 3.041 s (job 2236599, level0=18 level1=4,
    # accepted_corrections=4, fallback_fine=0).  A sweep over the knobs below
    # only means something once they are proven to reach params/argv.
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


def mg_coarse_geometry(
        local_lattice: tuple[int, int, int, int],
        mg_grid: list[int] | tuple[int, ...]) -> tuple[int, int, int, int]:
    """Validate the odd-Schur MG block grid and return the coarse lattice."""
    if any(int(value) < 1 for value in mg_grid):
        raise ValueError(f"--mg-grid {tuple(mg_grid)} must be positive")
    fine_odd = (
        int(local_lattice[0]), int(local_lattice[1]), int(local_lattice[2]),
        int(local_lattice[3]) // 2)
    coarse = tuple(
        fine_odd[axis] // int(mg_grid[axis]) for axis in range(4))
    if (any(fine_odd[axis] % int(mg_grid[axis]) for axis in range(4)) or
            any(extent < 1 for extent in coarse)):
        raise ValueError(
            f"local lattice {tuple(local_lattice)} has odd-Schur fine extent "
            f"{fine_odd} (coarse {coarse}), which is not compatible with "
            f"--mg-grid {tuple(mg_grid)}; every coarse extent must be at "
            "least 1 and divide evenly. Reduce --mg-grid (use 1 on t for "
            "t-only rank grids) or use a different --grid.")
    return coarse


def _coarse_step(mg_grid, axis: int) -> int:
    """Fine-to-cache coarse stride of one axis (odd-Schur time halves)."""
    return int(mg_grid[axis]) * (2 if axis == 3 else 1)


def _patch_axis(global_extent: int, local_extent: int, coordinate: int,
                block: int) -> tuple[int, int, int]:
    """Return ``(start, extent, center)`` of a rank-local overlap patch.

    The halo is two coarse blocks per side, which is the minimum window the
    local 33-stencil build needs to reproduce the global operator at every
    interior coarse center.  When the halo does not fit, the patch falls back
    to the whole global axis (only possible when the block grid divides it).
    """
    halo = 2 * int(block)
    desired = int(local_extent) + 2 * halo
    if desired <= int(global_extent):
        start = (int(coordinate) * int(local_extent) - halo) % int(global_extent)
        return int(start), int(desired), int(halo)
    if int(global_extent) % int(block) == 0:
        return 0, int(global_extent), int(coordinate) * int(local_extent)
    raise ValueError(
        f"cannot build an aligned overlap patch for global={global_extent} "
        f"local={local_extent} block={block}")


def _periodic_window(array: np.ndarray, starts, extents,
                     axes) -> np.ndarray:
    """Cut a periodically wrapped window over the listed axes."""
    value = np.asarray(array)
    for axis, extent, start in zip(axes, extents, starts):
        size = int(value.shape[axis])
        indices = (int(start) + np.arange(int(extent))) % size
        value = np.take(value, indices, axis=axis)
    return np.ascontiguousarray(value)


def _slice_asset(asset: np.ndarray, name: str, starts, extents) -> np.ndarray:
    """Cut one rank-local coarse window out of a coarse-block tensor."""
    value = np.asarray(asset)
    selection = [slice(None)] * value.ndim
    for axis, storage_axis in enumerate(LEGACY_MG_ASSET_COARSE_AXES[name]):
        selection[storage_axis] = slice(
            int(starts[axis]), int(starts[axis] + extents[axis]))
    return np.ascontiguousarray(value[tuple(selection)])


def ranked_coarse_geometry(
        lattice: tuple[int, int, int, int],
        local_lattice: tuple[int, int, int, int],
        coordinate: tuple[int, int, int, int],
        grid: tuple[int, int, int, int],
        mg_grid: list[int] | tuple[int, ...],
        dof: list[int] | tuple[int, ...],
        ) -> dict[str, Any]:
    """Plan one rank's local coarse slice and its overlap patch.

    The returned geometry is the single source of truth for the multi-rank
    writer: ``local_coarse``/``coarse_starts`` describe the hyperslab this rank
    publishes, while ``patch_*`` describes the periodic gauge/lonv window the
    local stencil build consumes before the slice is cut.
    """
    mg = [int(value) for value in mg_grid]
    if any(value < 1 for value in mg):
        raise ValueError(f"--mg-grid {tuple(mg)} must be positive")
    fine_odd = (
        int(lattice[0]), int(lattice[1]), int(lattice[2]), int(lattice[3]) // 2)
    local_odd = (
        int(local_lattice[0]), int(local_lattice[1]), int(local_lattice[2]),
        int(local_lattice[3]) // 2)
    for axis in range(4):
        if local_odd[axis] % mg[axis]:
            raise ValueError(
                f"local odd extent {local_odd[axis]} is not divisible by "
                f"--mg-grid {mg[axis]} on axis {axis}")
    coarse = tuple(fine_odd[axis] // mg[axis] for axis in range(4))
    local_coarse = tuple(local_odd[axis] // mg[axis] for axis in range(4))
    if any(local_coarse[axis] * int(grid[axis]) != coarse[axis]
           for axis in range(4)):
        raise ValueError(
            f"local coarse slab {local_coarse} with process grid "
            f"{tuple(grid)} does not tile {coarse}")
    coarse_starts = tuple(
        int(coordinate[axis]) * local_coarse[axis] for axis in range(4))
    spec = [
        _patch_axis(int(lattice[axis]), int(local_lattice[axis]),
                    int(coordinate[axis]), mg[axis])
        for axis in range(4)
    ]
    patch_starts = tuple(item[0] for item in spec)
    patch_extent = tuple(item[1] for item in spec)
    patch_center = tuple(item[2] for item in spec)
    steps = tuple(_coarse_step(mg, axis) for axis in range(4))
    for axis in range(4):
        for value, label in ((patch_starts[axis], "start"),
                             (patch_extent[axis], "extent"),
                             (patch_center[axis], "center")):
            if value % steps[axis]:
                raise ValueError(
                    f"overlap patch {label} {value} on axis {axis} is not a "
                    f"multiple of the coarse step {steps[axis]}")
    patch_coarse_starts = tuple(
        patch_starts[axis] // steps[axis] for axis in range(4))
    patch_coarse_extent = tuple(
        patch_extent[axis] // steps[axis] for axis in range(4))
    patch_coarse_center = tuple(
        patch_center[axis] // steps[axis] for axis in range(4))
    for axis in range(4):
        if patch_coarse_center[axis] + local_coarse[axis] > \
                patch_coarse_extent[axis]:
            raise ValueError(
                f"local coarse slab {local_coarse} does not fit the patch "
                f"{patch_coarse_extent} at {patch_coarse_center}")
        if coarse_starts[axis] + local_coarse[axis] > coarse[axis]:
            raise ValueError(
                f"local coarse slab {coarse_starts}+{local_coarse} exceeds "
                f"the global coarse box {coarse}")
    shapes = legacy_mg_cache_shapes(lattice, mg, tuple(int(v) for v in dof))
    return {
        "fine_odd": fine_odd,
        "local_odd": local_odd,
        "coarse": coarse,
        "local_coarse": local_coarse,
        "coarse_starts": coarse_starts,
        "patch_starts": patch_starts,
        "patch_extent": patch_extent,
        "patch_center": patch_center,
        "patch_coarse_starts": patch_coarse_starts,
        "patch_coarse_extent": patch_coarse_extent,
        "patch_coarse_center": patch_coarse_center,
        "global_shapes": shapes,
        "local_shapes": legacy_mg_local_shapes(shapes, local_coarse),
    }


def _periodic_indices(start: int, extent: int, size: int) -> np.ndarray:
    """Index vector of a cyclically wrapped window of one axis."""
    if extent <= 0 or size <= 0 or extent > size:
        raise ValueError(
            f"invalid periodic window start={start} extent={extent} size={size}")
    return (int(start) + np.arange(int(extent))) % int(size)


def _hdf5_gauge_patch(
        path: Path,
        starts: tuple[int, int, int, int],
        extent: tuple[int, int, int, int],
        dtype: np.dtype) -> np.ndarray:
    """Read one periodic QCU gauge patch straight from the global HDF5 file.

    Only the patch window is touched: every wrapped axis is read through a
    sorted-unique selection (a wrapped window is one or two contiguous runs) and
    re-ordered back to the requested index vector, so the read never gathers the
    whole global gauge.
    """
    import h5py

    def sorted_axis(source: Any, values: np.ndarray, axis: int) -> np.ndarray:
        ordered = np.unique(values)
        selection = [slice(None)] * source.ndim
        selection[axis] = ordered
        selected = source[tuple(selection)]
        return np.take(selected, np.searchsorted(ordered, values), axis=axis)

    with h5py.File(path, "r") as handle:
        dataset = handle["data"]
        if dataset.ndim != 7:
            raise ValueError(f"unsupported gauge HDF5 shape {dataset.shape}")
        if tuple(dataset.shape[:3]) == (3, 3, 4):
            storage_axes = (3, 4, 5, 6)
            qdp_layout = False
            sizes = tuple(int(value) for value in dataset.shape[3:])
        elif int(dataset.shape[0]) == 4 and tuple(dataset.shape[-2:]) == (3, 3):
            # QDP ``[dir, t, z, y, x, color, color]``; lattice order is x,y,z,t.
            storage_axes = (4, 3, 2, 1)
            qdp_layout = True
            sizes = tuple(
                int(value) for value in
                (dataset.shape[4], dataset.shape[3], dataset.shape[2],
                 dataset.shape[1]))
        else:
            raise ValueError(f"unsupported gauge HDF5 shape {dataset.shape}")
        value = dataset
        for axis, window_start, window_extent, size in zip(
                storage_axes, starts, extent, sizes):
            indices = _periodic_indices(window_start, window_extent, size)
            value = sorted_axis(value, indices, axis)
    if qdp_layout:
        value = qdp_gauge_to_qcu(value)
    return np.ascontiguousarray(value, dtype=dtype)


def _gathered_gauge_patch(
        gauge_local: np.ndarray,
        lattice: tuple[int, int, int, int],
        grid: tuple[int, int, int, int],
        starts: tuple[int, int, int, int],
        extent: tuple[int, int, int, int],
        dtype: np.dtype,
        comm) -> np.ndarray:
    """Assemble the global gauge from the rank slabs, then cut the patch.

    This is the fallback for input that is neither a single global HDF5 file nor
    already resident (e.g. ``--gauge-slab-dir`` without ``--gauge``).  It costs
    one global gauge per rank, so the caller is told to pass ``--gauge`` when the
    box is large.
    """
    local = np.ascontiguousarray(gauge_local)
    if comm is None:
        global_gauge = local
    else:
        pieces = comm.allgather(local)
        global_gauge = np.empty((local.shape[0], local.shape[1], local.shape[2],
                                 *lattice), dtype=local.dtype)
        for source, value in enumerate(pieces):
            source_coordinate = np.unravel_index(
                source, tuple(int(item) for item in grid), order="C")
            selection = [slice(None)] * 3 + [
                slice(int(source_coordinate[axis]) * int(lattice[axis])
                      // int(grid[axis]),
                      (int(source_coordinate[axis]) + 1) * int(lattice[axis])
                      // int(grid[axis]))
                for axis in range(4)
            ]
            global_gauge[tuple(selection)] = value
        del pieces
    return _periodic_window(global_gauge, starts, extent, axes=(3, 4, 5, 6))


def read_gauge_patch(
        path: Path | None,
        gauge_full: np.ndarray | None,
        gauge_local: np.ndarray,
        lattice: tuple[int, int, int, int],
        grid: tuple[int, int, int, int],
        starts: tuple[int, int, int, int],
        extent: tuple[int, int, int, int],
        dtype: np.dtype,
        comm) -> np.ndarray:
    """Return the periodic QCU gauge patch ``[3, 3, 4, *extent]`` of one rank.

    The patch must be cut from the *global* gauge: wrapping a rank-local slab on
    its own extent would tile the local data instead of continuing into the
    neighbouring ranks, which is exactly the class of error the overlap-patch
    builder replaces.  A single global HDF5 file is read in place, an
    already-loaded global array is windowed in memory, and the slab-directory
    case falls back to a global gather.
    """
    if gauge_full is not None:
        return _periodic_window(gauge_full, starts, extent, axes=(3, 4, 5, 6))
    if path is not None and Path(path).suffix.lower() in (".h5", ".hdf5"):
        return _hdf5_gauge_patch(Path(path), starts, extent, dtype)
    return _gathered_gauge_patch(
        gauge_local, lattice, grid, starts, extent, dtype, comm)


def read_periodic_coarse_asset(
        cache_path: Path, name: str, dtype: np.dtype,
        starts: tuple[int, int, int, int],
        extents: tuple[int, int, int, int],
        global_coarse: tuple[int, int, int, int],
        global_shapes, *,
        allow_incomplete: bool = True) -> np.ndarray:
    """Read a periodically wrapped coarse window from a ranked cache.

    Thin wrapper over :func:`pyqcu.tools._legacy_mg_cache.load_legacy_mg_cache_patch`
    kept for the synthetic ranked-cache regression, which pins the wrapped read
    independently of the C++ builder.
    """
    return load_legacy_mg_cache_patch(
        cache_path, name, dtype, starts=starts, extents=extents,
        global_coarse=global_coarse, global_shapes=global_shapes,
        allow_incomplete=allow_incomplete)


def build_local_null_vectors(wrapped, params: np.ndarray, argv: np.ndarray,
                             set_ptrs: np.ndarray, local_lattice, mg_grid,
                             dof: list[int], nv_iters: int, seed: int,
                             dtype: np.dtype) -> tuple[np.ndarray, float]:
    """Build this rank's null-vector slab with the collective KUPL builder.

    ``params`` must already carry the process grid/rank/size, so the C++
    inverse iteration performs its halo exchange and global reductions exactly
    like a serial global build of the same geometry.
    """
    from pyqcu.kupl import qcu

    fine_odd = (
        int(local_lattice[0]), int(local_lattice[1]), int(local_lattice[2]),
        int(local_lattice[3]) // 2)
    mg = [int(value) for value in mg_grid]
    coarse = tuple(fine_odd[axis] // mg[axis] for axis in range(4))
    E, e = int(dof[1]), 12
    null_shape = (
        E, e, coarse[0], mg[0], coarse[1], mg[1], coarse[2], mg[2],
        coarse[3], mg[3])
    null_vectors = np.empty(null_shape, dtype=dtype)
    null_arg = NumpyTensor(null_vectors)
    params_arg = NumpyTensor(params)
    argv_arg = NumpyTensor(argv)
    set_ptrs_arg = NumpyTensor(set_ptrs)
    started = time.perf_counter()
    qcu.applyInitQcu(set_ptrs_arg, params_arg, argv_arg)
    try:
        params[PARITY] = 1
        params[SET_PLAN] = PLAN_SOLVER
        status = qcu.applyMultigridBuildNullVectorsQcu(
            null_arg, wrapped["gauge"], wrapped["even"], wrapped["odd"],
            wrapped["even_inv"], wrapped["odd_inv"], set_ptrs_arg, params_arg,
            E, e, *fine_odd, *coarse, max(1, int(nv_iters)), int(seed), 1.0e-2)
    finally:
        qcu.applyEndQcu(set_ptrs_arg, params_arg)
    seconds = time.perf_counter() - started
    if int(status) != 0:
        raise RuntimeError(f"KUPL null-vector setup failed: {status}")
    return null_vectors, seconds


def export_ranked_coarse_cache(
        cache_path: Path, *, wrapped, params: np.ndarray, argv: np.ndarray,
        set_ptrs: np.ndarray, gauge_local: np.ndarray, dtype: np.dtype,
        lattice: tuple[int, int, int, int],
        local_lattice: tuple[int, int, int, int],
        coordinate: tuple[int, int, int, int],
        grid: tuple[int, int, int, int], rank: int, size: int, comm,
        mg_grid: list[int], dof: list[int], nv_iters: int, seed: int,
        mass: float, kappa: float, csw: float, identity: dict[str, Any],
        mg_config: dict[str, Any], metadata: dict[str, Any],
        batch_size: int, source_digest: str | None = None,
        verbose: bool = False, gauge_path: Path | None = None,
        gauge_full: np.ndarray | None = None,
        reuse_lonv: Path | None = None,
        patch_galerkin: bool = True
) -> tuple[tuple[Any, ...], dict[str, Any], dict]:
    """Build and publish one rank's Legacy-MG coarse slice from a patch.

    This is the CUDA-QCU ``_distributed_mg_cache`` algorithm on the KUPL
    backend.  Every rank cuts a **periodic overlap patch** -- its own fine
    extent plus a ``2 * mg_grid`` halo per axis, degenerating to the whole
    global axis when the halo does not fit -- runs the verified *single-rank*
    C++ 33-stencil builder on that patch through a serial-local ``1x1x1x1``
    LatticeSet (grid ``1,1,1,1``, node rank 0, node size 1), keeps only the
    **center** coarse block of the patch result, and publishes its own
    hyperslab.  No distributed stencil halo/ownership path is involved: the
    patch is an independent serial geometry, so the operator it produces is the
    already validated single-rank operator.

    The patch is cut from the *global* gauge (``gauge_path``/``gauge_full``) and
    from the published global null vectors; ``reuse_lonv`` reuses an existing
    cache's null vectors (the CUDA ``--reuse-lonv`` option) so a reference
    stencil can be reproduced bit for bit.  Rank zero commits the manifest, so a
    half-written cache is never reported as ``PASS``.
    """
    geometry = ranked_coarse_geometry(
        lattice, local_lattice, coordinate, grid, mg_grid, dof)
    shapes = geometry["global_shapes"]
    local_coarse = geometry["local_coarse"]
    timings: dict[str, float] = {
        "null_vector_seconds": 0.0,
        "lonv_publish_seconds": 0.0,
        "lonv_patch_seconds": 0.0,
        "gauge_patch_seconds": 0.0,
        "stencil_seconds": 0.0,
        "asset_write_seconds": 0.0,
        "manifest_seconds": 0.0,
    }
    started = time.perf_counter()
    if reuse_lonv is None:
        null_vectors, timings["null_vector_seconds"] = build_local_null_vectors(
            wrapped, params, argv, set_ptrs, local_lattice, mg_grid, dof,
            nv_iters, seed, dtype)
        lonv_source = "distributed-null-vector"
    else:
        null_vectors = np.ascontiguousarray(
            load_legacy_mg_cache_slab(
                Path(reuse_lonv), dtype, starts=geometry["coarse_starts"],
                local_shape=local_coarse, names=("lonv",))["lonv"], dtype=dtype)
        expected = tuple(
            int(value) for value in geometry["local_shapes"]["lonv"])
        if tuple(int(value) for value in null_vectors.shape) != expected:
            raise ValueError(
                f"reused lonv slab {tuple(null_vectors.shape)} != {expected}")
        lonv_source = str(reuse_lonv)
        if verbose:
            print(
                f"PYQCU::CXX::MG_CACHE:\n rank={rank} reuse-lonv={reuse_lonv} "
                f"slab={list(null_vectors.shape)}", flush=True)

    def rank_record(datasets, mode) -> dict[str, Any]:
        return write_legacy_mg_rank_slice(
            cache_path, datasets, rank=rank, coordinate=coordinate,
            process_grid=grid, starts=geometry["coarse_starts"],
            local_coarse=local_coarse, global_shapes=shapes,
            identity=identity, mode=mode)

    stage = time.perf_counter()
    lonv_record = rank_record({"lonv": null_vectors}, "w")
    timings["lonv_publish_seconds"] = time.perf_counter() - stage
    if comm is not None:
        comm.Barrier()

    # Rank zero publishes the interim manifest so every rank can read the
    # neighbours' null vectors through the regular ranked-cache reader.
    stage = time.perf_counter()
    if comm is not None:
        stage_records = comm.gather(lonv_record, root=0)
    else:
        stage_records = [lonv_record]
    if rank == 0:
        interim = build_legacy_mg_ranked_cache_manifest(
            cache_path, identity, process_grid=grid, ranks=stage_records,
            global_shapes=shapes, local_shapes=geometry["local_shapes"],
            global_coarse=geometry["coarse"], local_coarse=local_coarse,
            setup_method="cxx-null-vector+overlap-local-33-stencil",
            setup_backend="kupl-qcu-cxx-mpi",
            mg_config=mg_config, source_metadata=metadata,
            extra_metadata={
                "mpi_size": int(size), "grid": list(grid),
                "local_lattice": list(local_lattice),
                "stage": "null-vectors",
            },
            assets=("lonv",), status="BUILDING")
        write_legacy_mg_ranked_cache(cache_path, interim)
    if comm is not None:
        comm.Barrier()
    timings["manifest_seconds"] += time.perf_counter() - stage

    stage = time.perf_counter()
    lonv_patch = read_periodic_coarse_asset(
        cache_path, "lonv", dtype, geometry["patch_coarse_starts"],
        geometry["patch_coarse_extent"], geometry["coarse"], shapes,
        allow_incomplete=True)
    del null_vectors
    timings["lonv_patch_seconds"] = time.perf_counter() - stage

    stage = time.perf_counter()
    gauge_patch = read_gauge_patch(
        gauge_path, gauge_full, gauge_local, lattice, grid,
        geometry["patch_starts"], geometry["patch_extent"], dtype, comm)
    timings["gauge_patch_seconds"] = time.perf_counter() - stage

    stage = time.perf_counter()
    stencil_stats: dict[str, Any] = {}
    patch_assets = build_coarse_assets(
        gauge_patch, geometry["patch_extent"], mass, kappa, csw, dtype, 2,
        list(dof), list(mg_grid), nv_iters, seed,
        verbose=verbose, local_stencil=True,
        local_stencil_batch_size=batch_size, device_name="cpu",
        setup_stats=stencil_stats, supplied_null_vectors=lonv_patch,
        validate_galerkin=patch_galerkin)
    del gauge_patch, lonv_patch
    timings["stencil_seconds"] = time.perf_counter() - stage
    patch_galerkin_error = float(stencil_stats.get("galerkin_relative_error", 0.0))
    local_assets = tuple(
        _slice_asset(patch_assets[index][0], name,
                     geometry["patch_coarse_center"], local_coarse)
        for index, name in enumerate(LEGACY_MG_ASSET_NAMES))
    del patch_assets

    stage = time.perf_counter()
    final_record = rank_record(
        {name: value for name, value in zip(
            ("hnn", "hdg", "sit"), local_assets[1:])}, "r+")
    final_record = dict(final_record, patch_galerkin_error=patch_galerkin_error,
                        patch_extent=list(geometry["patch_extent"]),
                        lonv_source=lonv_source)
    timings["asset_write_seconds"] = time.perf_counter() - stage
    if comm is not None:
        comm.Barrier()

    stage = time.perf_counter()
    final_records = (comm.gather(final_record, root=0) if comm is not None
                     else [final_record])
    manifest: dict[str, Any] = {}
    if rank == 0:
        manifest = build_legacy_mg_ranked_cache_manifest(
            cache_path, identity, process_grid=grid, ranks=final_records,
            global_shapes=shapes, local_shapes=geometry["local_shapes"],
            global_coarse=geometry["coarse"], local_coarse=local_coarse,
            setup_method="cxx-null-vector+periodic-overlap-patch-serial-stencil",
            setup_backend="kupl-qcu-cxx-mpi",
            mg_config=mg_config,
            setup_validation={
                "schema": "pyqcu.legacy-mg-ranked.v1",
                "mpi_size": int(size),
                "grid": list(grid),
                "local_lattice": list(local_lattice),
                "local_starts": list(
                    int(coordinate[axis]) * int(local_lattice[axis])
                    for axis in range(4)),
                "local_coarse": list(local_coarse),
                "coarse_starts": list(geometry["coarse_starts"]),
                "patch_starts": list(geometry["patch_starts"]),
                "patch_extent": list(geometry["patch_extent"]),
                "patch_coarse_starts": list(geometry["patch_coarse_starts"]),
                "patch_coarse_extent": list(geometry["patch_coarse_extent"]),
                "patch_coarse_center": list(geometry["patch_coarse_center"]),
                "patch_halo": [
                    int(geometry["patch_center"][axis]) for axis in range(4)],
                "builder": "kupl-qcu-cxx-local-stencil-patch",
                "builder_geometry": "serial-local 1x1x1x1 LatticeSet",
                "patch_galerkin_error": max(
                    (float(record.get("patch_galerkin_error", 0.0))
                     for record in (final_records or [])), default=0.0),
                "lonv_source": lonv_source,
                "timings": timings,
            },
            source_metadata=metadata,
            extra_metadata={
                "mpi_size": int(size),
                "grid": list(grid),
                "build_device": "cpu",
                "source": f"ranked-cache-export rank {rank} of {size}",
                "source_file_sha256": source_digest,
                "stage": "complete",
            },
            setup_seconds=time.perf_counter() - started)
        write_legacy_mg_ranked_cache(cache_path, manifest)
    if comm is not None:
        comm.Barrier()
    timings["manifest_seconds"] += time.perf_counter() - stage
    timings["seconds"] = time.perf_counter() - started

    # Fail closed on the artifact we just published: every rank verifies its
    # own slab digests and rank zero re-checks the manifest tiling.
    validate_legacy_mg_cache(
        cache_path, identity, verify_file_sha256=True,
        verify_dataset_sha256=True, ranks=(rank,))
    if rank == 0:
        validate_legacy_mg_cache(
            cache_path, identity, verify_file_sha256=True,
            verify_dataset_sha256=False, ranks=())
    if comm is not None:
        comm.Barrier()
    if verbose:
        print(
            f"PYQCU::CXX::MG_CACHE:\n rank={rank} local_coarse={local_coarse} "
            f"patch_coarse={geometry['patch_coarse_extent']} "
            f"seconds={timings['seconds']:.3f}", flush=True)
    return (tuple([local_assets[0]]),
            tuple([local_assets[1]]),
            tuple([local_assets[2]]),
            tuple([local_assets[3]])), timings, manifest


def build_coarse_assets(gauge: np.ndarray, lattice: tuple[int, int, int, int],
                        mass: float, kappa: float, csw: float,
                        dtype: np.dtype, levels: int, dof: list[int],
                        mg_grid: list[int], nv_iters: int, seed: int,
                        verbose: bool = False, local_stencil: bool = False,
                        local_stencil_batch_size: int = 4,
                        device_name: str = "cpu",
                        setup_stats: dict[str, Any] | None = None,
                        supplied_null_vectors: np.ndarray | None = None,
                        validate_galerkin: bool = True,
                        gauge_parity: np.ndarray | None = None,
                        clover_assets: (
                            tuple[np.ndarray, np.ndarray, np.ndarray,
                                  np.ndarray] | None) = None):
    if levels != 2:
        raise ValueError("the independent simple MG fixture supports two levels")
    del kappa
    coarse_lattice = mg_coarse_geometry(lattice, mg_grid)
    if gauge_parity is None:
        gauge_parity = gauge_to_parity(np.ascontiguousarray(gauge))
    else:
        gauge_parity = np.ascontiguousarray(gauge_parity)
    fine_odd = (lattice[0], lattice[1], lattice[2], lattice[3] // 2)
    block = tuple(
        fine_odd[index] // coarse_lattice[index] for index in range(4))
    local_stencil_window = max(3 * extent + 2 for extent in block)
    E, e = int(dof[1]), 12
    null_shape = (
        E, e, coarse_lattice[0], block[0], coarse_lattice[1], block[1],
        coarse_lattice[2], block[2], coarse_lattice[3], block[3])
    onsite_shape = (4, 3, 4, 3, *fine_odd)
    sit_shape = (E, E, *coarse_lattice)
    hnn_shape = (2, 4, E, E, *coarse_lattice)
    hdg_shape = (2, 2, 6, E, E, *coarse_lattice)
    if supplied_null_vectors is not None:
        supplied_null_vectors = np.ascontiguousarray(
            supplied_null_vectors, dtype=dtype)
        if supplied_null_vectors.shape != null_shape:
            raise ValueError(
                f"supplied null-vector shape {supplied_null_vectors.shape} "
                f"!= {null_shape}")

    stage_started = time.perf_counter()
    galerkin_error = 0.0
    timings = {
        "null_vector_seconds": 0.0,
        "stencil_seconds": 0.0,
        "galerkin_validation_seconds": 0.0,
        "host_transfer_seconds": 0.0,
    }
    if device_name.startswith("cuda"):
        try:
            import torch
            from pyqcu.cuda import define, qcu
        except ImportError as exc:
            raise RuntimeError(f"CUDA QCU setup unavailable: {exc}") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA QCU setup requested without a visible DCU")
        device = torch.device(device_name)
        torch_dtype = torch.complex64 if dtype == np.dtype(np.complex64) else \
            torch.complex128
        gauge_device = torch.from_numpy(gauge_parity).to(
            device=device, dtype=torch_dtype, non_blocking=False).contiguous()
        clover_even = torch.empty(onsite_shape, dtype=torch_dtype, device=device)
        clover_odd = torch.empty_like(clover_even)
        clover_even_inv = torch.empty_like(clover_even)
        clover_odd_inv = torch.empty_like(clover_even)
        if supplied_null_vectors is None:
            null_vectors = torch.empty(
                null_shape, dtype=torch_dtype, device=device)
        else:
            null_vectors = torch.from_numpy(supplied_null_vectors).to(
                device=device, dtype=torch_dtype, non_blocking=False
            ).contiguous()
        sitting = torch.zeros(sit_shape, dtype=torch_dtype, device=device)
        hop_nn = torch.zeros(hnn_shape, dtype=torch_dtype, device=device)
        hop_diag = torch.zeros(hdg_shape, dtype=torch_dtype, device=device)
        params = torch.zeros(define._PARAMS_SIZE_, dtype=torch.int32)
        params[define._LAT_X_], params[define._LAT_Y_] = lattice[:2]
        params[define._LAT_Z_], params[define._LAT_T_] = lattice[2:]
        params[define._LAT_XYZT_] = int(np.prod(lattice))
        params[define._GRID_X_], params[define._GRID_Y_] = 1, 1
        params[define._GRID_Z_], params[define._GRID_T_] = 1, 1
        params[define._NODE_RANK_], params[define._NODE_SIZE_] = 0, 1
        params[define._MAX_ITER_] = 1000
        params[define._DATA_TYPE_] = int(define.epytd(torch_dtype).item())
        params[define._SET_INDEX_] = 0
        params[define._SET_PLAN_] = define._SET_PLAN2_
        params[define._MG_LEVEL1_E_] = E
        params[define._MG_LEVEL1_X_] = coarse_lattice[0]
        params[define._MG_LEVEL1_Y_] = coarse_lattice[1]
        params[define._MG_LEVEL1_Z_] = coarse_lattice[2]
        params[define._MG_LEVEL1_T_] = coarse_lattice[3]
        params[define._SEED_] = int(seed)
        argv = define.make_argv(
            float(mass), csw=float(csw), atol=1.0e-7,
            data_type=int(params[DATA_TYPE]))
        set_ptrs = torch.zeros(define._SET_PTRS_SIZE_, dtype=torch.int64)
        qcu.applyInitQcu(set_ptrs, params, argv)
        try:
            params[define._PARITY_] = 0
            qcu.applyCloversQcu(
                clover_even, clover_even_inv, gauge_device, set_ptrs, params)
            params[define._PARITY_] = 1
            qcu.applyCloversQcu(
                clover_odd, clover_odd_inv, gauge_device, set_ptrs, params)
            params[define._SET_PLAN_] = define._SET_PLAN1_
            if verbose and supplied_null_vectors is None:
                print(
                    "PYQCU::CUDA::MG_SETUP:\n C++ null-space start",
                    flush=True)
            stage = time.perf_counter()
            if supplied_null_vectors is None:
                qcu.applyMultigridBuildNullVectorsQcu(
                    null_vectors, gauge_device, clover_even, clover_odd,
                    clover_even_inv, clover_odd_inv, set_ptrs, params, E, e,
                    *fine_odd, *coarse_lattice, max(1, nv_iters), seed,
                    1.0e-2)
            timings["null_vector_seconds"] = time.perf_counter() - stage
            if verbose:
                print(
                    "PYQCU::CUDA::MG_SETUP:\n C++ stencil start",
                    flush=True)
            stage = time.perf_counter()
            qcu.applyMultigridBuildStencilLocalQcu(
                sitting, hop_nn, hop_diag, null_vectors, gauge_device,
                clover_even, clover_odd, clover_even_inv, clover_odd_inv,
                set_ptrs, params, E, e, *fine_odd, *coarse_lattice, 8,
                local_stencil_batch_size)
            timings["stencil_seconds"] = time.perf_counter() - stage
            if validate_galerkin:
                stage = time.perf_counter()
                generator = torch.Generator(device="cpu").manual_seed(941)
                coarse_vector = torch.randn(
                    (E, *coarse_lattice), generator=generator,
                    dtype=torch_dtype).to(device)
                prolonged = torch.empty(
                    (e, *fine_odd), dtype=torch_dtype, device=device)
                schur_vector = torch.empty_like(prolonged)
                restricted = torch.empty_like(coarse_vector)
                stencil_vector = torch.empty_like(coarse_vector)
                qcu.applyMultigridProLongQcu(
                    prolonged, coarse_vector, null_vectors, set_ptrs, params)
                qcu.applyCloverBistabCgDslashQcu(
                    schur_vector, prolonged, gauge_device, clover_even,
                    clover_odd, clover_even_inv, clover_odd_inv, set_ptrs,
                    params)
                qcu.applyMultigridRestrictQcu(
                    restricted, schur_vector, null_vectors, set_ptrs, params)
                qcu.applyMultigridCoarseDslashWideQcu(
                    stencil_vector, coarse_vector, sitting, hop_nn, hop_diag,
                    set_ptrs, params)
                denominator = torch.linalg.vector_norm(
                    restricted).clamp_min(1.0)
                galerkin_error = float((
                    torch.linalg.vector_norm(stencil_vector - restricted) /
                    denominator).item())
                if (not np.isfinite(galerkin_error) or
                        galerkin_error > 2.0e-5):
                    raise RuntimeError(
                        f"CUDA C++ stencil Galerkin error {galerkin_error:.3e}")
                timings["galerkin_validation_seconds"] = (
                    time.perf_counter() - stage)
                del prolonged, schur_vector, restricted, stencil_vector
        finally:
            qcu.applyEndQcu(set_ptrs, params)
        stage = time.perf_counter()
        values = tuple(
            tensor.detach().cpu().contiguous().numpy()
            for tensor in (null_vectors, hop_nn, hop_diag, sitting))
        timings["host_transfer_seconds"] = time.perf_counter() - stage
    elif device_name == "cpu":
        from pyqcu.kupl import qcu
        try:
            from pyqcu.kupl import define
        except ImportError as exc:
            raise RuntimeError(f"KUPL C++ setup unavailable: {exc}") from exc
        if clover_assets is None:
            clover_even = np.empty(onsite_shape, dtype=dtype)
            clover_odd = np.empty(onsite_shape, dtype=dtype)
            clover_even_inv = np.empty(onsite_shape, dtype=dtype)
            clover_odd_inv = np.empty(onsite_shape, dtype=dtype)
            reuse_clovers = False
        else:
            clover_even, clover_odd, clover_even_inv, clover_odd_inv = (
                np.ascontiguousarray(array) for array in clover_assets)
            for array in (clover_even, clover_odd, clover_even_inv,
                          clover_odd_inv):
                if array.shape != onsite_shape or array.dtype != dtype:
                    raise ValueError(
                        f"supplied Clover asset {array.shape}/{array.dtype} "
                        f"does not match {onsite_shape}/{dtype}")
            reuse_clovers = True
        if supplied_null_vectors is None:
            null_vectors = np.empty(null_shape, dtype=dtype)
        else:
            null_vectors = supplied_null_vectors.copy()
        sitting = np.zeros(sit_shape, dtype=dtype)
        hop_nn = np.zeros(hnn_shape, dtype=dtype)
        hop_diag = np.zeros(hdg_shape, dtype=dtype)
        params = np.zeros(58, dtype=np.int32)
        params[0:4] = lattice
        params[4] = int(np.prod(lattice))
        params[5:9] = 1
        params[10], params[11] = 0, 1
        params[13] = 1000
        params[DATA_TYPE] = define.LAT_C64 if dtype == np.dtype(np.complex64) else \
            define.LAT_C128
        params[15] = 0
        params[SET_PLAN] = PLAN_CLOVER
        params[19] = E
        params[20:24] = coarse_lattice
        params[52] = int(seed)
        argv = np.zeros(8, dtype=np.float32 if dtype == np.dtype(np.complex64)
                        else np.float64)
        argv[0], argv[1], argv[2], argv[7] = mass, 1.0e-7, 0.1, csw
        set_ptrs = np.zeros(100, dtype=np.int64)
        gauge_arg = NumpyTensor(gauge_parity)
        clover_even_arg = NumpyTensor(clover_even)
        clover_odd_arg = NumpyTensor(clover_odd)
        clover_even_inv_arg = NumpyTensor(clover_even_inv)
        clover_odd_inv_arg = NumpyTensor(clover_odd_inv)
        null_arg = NumpyTensor(null_vectors)
        sitting_arg = NumpyTensor(sitting)
        hop_nn_arg = NumpyTensor(hop_nn)
        hop_diag_arg = NumpyTensor(hop_diag)
        params_arg = NumpyTensor(params)
        argv_arg = NumpyTensor(argv)
        set_ptrs_arg = NumpyTensor(set_ptrs)
        qcu.applyInitQcu(set_ptrs_arg, params_arg, argv_arg)
        try:
            if not reuse_clovers:
                params[PARITY] = 0
                qcu.applyCloversQcu(
                    clover_even_arg, clover_even_inv_arg, gauge_arg,
                    set_ptrs_arg, params_arg)
                params[PARITY] = 1
                qcu.applyCloversQcu(
                    clover_odd_arg, clover_odd_inv_arg, gauge_arg,
                    set_ptrs_arg, params_arg)
            params[PARITY] = 1
            params[SET_PLAN] = PLAN_SOLVER
            stage = time.perf_counter()
            if supplied_null_vectors is None:
                status = qcu.applyMultigridBuildNullVectorsQcu(
                    null_arg, gauge_arg, clover_even_arg, clover_odd_arg,
                    clover_even_inv_arg, clover_odd_inv_arg, set_ptrs_arg,
                    params_arg, E, e,
                    *fine_odd, *coarse_lattice, max(1, nv_iters), seed,
                    1.0e-2)
                if status != 0:
                    raise RuntimeError(
                        f"KUPL null-vector setup failed: {status}")
            timings["null_vector_seconds"] = time.perf_counter() - stage
            stage = time.perf_counter()
            status = qcu.applyMultigridBuildStencilLocalQcu(
                sitting_arg, hop_nn_arg, hop_diag_arg, null_arg, gauge_arg,
                clover_even_arg, clover_odd_arg, clover_even_inv_arg,
                clover_odd_inv_arg, set_ptrs_arg, params_arg, E, e,
                *fine_odd, *coarse_lattice, local_stencil_window,
                local_stencil_batch_size)
            if status != 0:
                raise RuntimeError(f"KUPL stencil setup failed: {status}")
            timings["stencil_seconds"] = time.perf_counter() - stage
            if validate_galerkin:
                stage = time.perf_counter()
                rng = np.random.default_rng(941)
                coarse_vector = (
                    rng.normal(size=(E, *coarse_lattice)) +
                    1j * rng.normal(size=(E, *coarse_lattice))).astype(dtype)
                prolonged = np.empty((e, *fine_odd), dtype=dtype)
                schur_vector = np.empty_like(prolonged)
                restricted = np.empty_like(coarse_vector)
                stencil_vector = np.empty_like(coarse_vector)
                qcu.applyMultigridProLongQcu(
                    NumpyTensor(prolonged), NumpyTensor(coarse_vector),
                    null_arg, set_ptrs_arg, params_arg)
                qcu.applyCloverBistabCgDslashQcu(
                    NumpyTensor(schur_vector), NumpyTensor(prolonged),
                    gauge_arg, clover_even_arg, clover_odd_arg,
                    clover_even_inv_arg, clover_odd_inv_arg, set_ptrs_arg,
                    params_arg)
                qcu.applyMultigridRestrictQcu(
                    NumpyTensor(restricted), NumpyTensor(schur_vector),
                    null_arg, set_ptrs_arg, params_arg)
                qcu.applyMultigridCoarseDslashWideQcu(
                    NumpyTensor(stencil_vector), NumpyTensor(coarse_vector),
                    sitting_arg, hop_nn_arg, hop_diag_arg, set_ptrs_arg,
                    params_arg)
                denominator = max(float(np.linalg.norm(restricted)), 1.0)
                galerkin_error = float(
                    np.linalg.norm(stencil_vector - restricted) / denominator)
                if (not np.isfinite(galerkin_error) or
                        galerkin_error > 2.0e-5):
                    difference = stencil_vector - restricted
                    site_difference = np.linalg.norm(
                        difference.reshape(E, -1), axis=0)
                    site_reference = np.linalg.norm(
                        restricted.reshape(E, -1), axis=0)
                    worst_site = int(np.argmax(site_difference))
                    worst_coordinate = np.unravel_index(
                        worst_site, coarse_lattice)
                    raise RuntimeError(
                        f"KUPL C++ stencil Galerkin error "
                        f"{galerkin_error:.3e} worst={worst_coordinate} "
                        f"diff_norm={site_difference[worst_site]:.3e} "
                        f"reference_norm={site_reference[worst_site]:.3e}")
                timings["galerkin_validation_seconds"] = (
                    time.perf_counter() - stage)
        finally:
            qcu.applyEndQcu(set_ptrs_arg, params_arg)
        values = (null_vectors, hop_nn, hop_diag, sitting)
    else:
        raise ValueError(f"unsupported build device {device_name!r}")

    null_vectors, hop_nn, hop_diag, sitting = values
    gram_error = 0.0
    if supplied_null_vectors is None:
        columns = null_vectors.reshape(
            E, e, coarse_lattice[0], block[0], coarse_lattice[1], block[1],
            coarse_lattice[2], block[2], coarse_lattice[3], block[3]
        ).transpose(2, 4, 6, 8, 0, 1, 3, 5, 7, 9).reshape(
            int(np.prod(coarse_lattice)), E,
            e * int(np.prod(block)))
        gram = columns.conj() @ np.swapaxes(columns, 1, 2)
        gram_error = float(np.max(np.abs(
            gram - np.eye(E, dtype=complex))))
        if not np.isfinite(gram_error) or gram_error > 1.0e-4:
            raise RuntimeError(f"MG null-vector Gram error {gram_error:.3e}")
    if verbose:
        print(
            "PYQCU::CXX::MG_SETUP:\n "
            f"device={device_name} method=CXX_NULL_STENCIL "
            f"fine_odd={fine_odd} coarse={coarse_lattice} "
            f"gram={gram_error:.3e} galerkin={galerkin_error:.3e} "
            f"seconds={time.perf_counter()-stage_started:.3f}",
            flush=True)
    if setup_stats is not None:
        setup_stats.update({
            "method": "cxx-null-vector+local-33-stencil",
            "backend": (
                "cuda-qcu-cxx" if device_name.startswith("cuda")
                else "kupl-qcu-cxx"),
            "fine_odd": list(fine_odd),
            "coarse": list(coarse_lattice),
            "local_stencil_window": 8,
            "local_stencil_batch_size": int(local_stencil_batch_size),
            "gram_max_abs_error": gram_error,
            "galerkin_relative_error": galerkin_error,
            "null_vectors_supplied": supplied_null_vectors is not None,
            "galerkin_validated": bool(validate_galerkin),
            **timings,
            "seconds": time.perf_counter() - stage_started,
        })
    return tuple([value] for value in values)


def load_coarse_assets(path: Path, dtype: np.dtype,
                       starts: tuple[int, int, int, int] | None = None,
                       local_shape: tuple[int, int, int, int] | None = None):
    """Load the backend-neutral ``lonv/hnn/hdg/sit`` 33-tensor cache."""
    values = load_legacy_mg_cache_slab(
        path, dtype, starts=starts, local_shape=local_shape)
    # Match build_coarse_assets(): each return value is a per-level sequence.
    # Returning the raw 10-D array makes callers interpret its E axis as levels.
    return (values["lonv"],), (values["hnn"],), (values["hdg"],), (values["sit"],)


def write_coarse_assets(path: Path, null_vectors, hop_nn, hop_diag, sitting):
    write_legacy_mg_assets(path, {
        "lonv": null_vectors, "hnn": hop_nn,
        "hdg": hop_diag, "sit": sitting,
    })


def _slice_last4(array: np.ndarray, starts: tuple[int, int, int, int],
                 local_shape: tuple[int, int, int, int]) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim < 4:
        raise ValueError(f"expected at least four lattice axes, got {values.shape}")
    slices = [slice(None)] * (values.ndim - 4)
    slices.extend(
        slice(int(starts[axis]), int(starts[axis] + local_shape[axis]))
        for axis in range(4))
    return np.ascontiguousarray(values[tuple(slices)])


def _read_local_slab(path_value, starts: tuple[int, int, int, int],
                     local_shape: tuple[int, int, int, int], *,
                     hdf5_mpio: bool = False, comm=None,
                     slab_dir: Path | None = None, rank: int = 0):
    """Read only this rank's HDF5 hyperslab (dataset ``data``, last four axes).

    Observed (remote L48, 2026-11-17): this is a rank-local hyperslab, never a
    full-file read, but the canonical file is one *contiguous* dataset with
    ``t`` fastest, so splitting ``t`` leaves 3,981,312 elementary 384 B runs
    per rank (10,380 ``pread`` calls, +3.56x bytes read).  The shared ``dtfs``
    client also stalls intermittently (same file and node: 43.8 s vs 276.8 s),
    which is why ``slab_dir`` (contiguous ``rank_XXXX.h5``) is the preferred
    input path.  ``hdf5_mpio`` cannot fix this here: the remote h5py is built
    without MPI support.
    """
    if slab_dir is not None:
        slab_path = Path(slab_dir) / f"rank_{int(rank):04d}.h5"
        if not slab_path.is_file():
            raise FileNotFoundError(slab_path)
        import h5py

        with h5py.File(slab_path, "r") as handle:
            value = np.ascontiguousarray(handle["data"][...])
        if tuple(value.shape[-4:]) != tuple(local_shape):
            raise ValueError(
                f"slab {slab_path} trailing shape {value.shape[-4:]} != "
                f"{tuple(local_shape)}")
        return value
    if path_value is None:
        return None
    path = resolve_path(path_value, resolve_path(None, path_value))
    if path.suffix not in (".h5", ".hdf5"):
        return None
    import h5py

    selection = (Ellipsis,) + tuple(
        slice(int(start), int(start) + int(count))
        for start, count in zip(starts, local_shape))
    open_kwargs = {}
    if hdf5_mpio:
        if comm is None:
            raise ValueError("collective HDF5 read requires an MPI communicator")
        open_kwargs = {"driver": "mpio", "comm": comm}
    with h5py.File(path, "r", **open_kwargs) as handle:
        return np.ascontiguousarray(handle["data"][selection])


def _rank_geometry(lattice: tuple[int, int, int, int],
                   grid: tuple[int, int, int, int], rank: int):
    if int(np.prod(grid)) < rank + 1:
        raise ValueError(f"rank {rank} is outside process grid {grid}")
    if any(lattice[axis] % grid[axis] for axis in range(4)):
        raise ValueError(f"lattice {lattice} is not divisible by grid {grid}")
    local = tuple(lattice[axis] // grid[axis] for axis in range(4))
    if any(extent % 2 for extent in local):
        raise ValueError(f"KUPL CHROMA layout requires even local dims: {local}")
    coordinate = tuple(
        int(value) for value in np.unravel_index(rank, grid, order="C"))
    starts = tuple(coordinate[axis] * local[axis] for axis in range(4))
    return coordinate, local, starts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lat", nargs=4, type=int, default=(8, 8, 8, 16))
    parser.add_argument("--grid", nargs=4, type=int, default=None,
                        help="MPI process grid; default: one rank in x")
    parser.add_argument("--mass", type=float, default=None)
    parser.add_argument("--csw", type=float, default=None)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument(
        "--strict-acceptance", action="store_true",
        help="gate the reported PASS/FAIL on atol alone (production policy)")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--dtype", choices=("c64", "c128"), default="c64")
    # Before interpreting any sweep over the four knobs below (--levels,
    # --restart, --mg-grid, --coarse-tol-factor), assert that the recorded
    # configuration equals the requested one: a row that still reports the
    # default value means the knob never reached the solver.  Learned on
    # 2026-11-24 - job 2238157 ran six named points as one identical point
    # because pyqcu.test's Case dropped restart/mg_grid/coarse_tol_factor.
    parser.add_argument("--levels", type=int, default=2)
    parser.add_argument("--dof", nargs="+", type=int, default=(12, 2, 2, 2, 2))
    parser.add_argument("--mg-grid", nargs=4, type=int, default=DEFAULT_BLOCK)
    parser.add_argument(
        "--restart", type=int, default=5,
        help="coarse-correction interval; 0 selects optimized parity-Schur BiCGStab")
    parser.add_argument("--coarse-max-iter", type=int, default=200)
    parser.add_argument("--coarse-tol-factor", type=float, default=1.0e4)
    parser.add_argument("--nv-iters", type=int, default=2)
    # Observed (L48, 2026-11-17): reuse the cache instead of rebuilding it.  A
    # 16-rank E=2 (`--dof 12 2`) build measured 1.276 GiB and 1741.9 s of
    # builder, there is no matching 32r/two-node cache yet, and break-even
    # against the no-cache baseline is ~24 solves.
    parser.add_argument("--coarse-cache", type=Path, default=None,
                        help="load lonv/hnn/hdg/sit instead of rebuilding setup")
    parser.add_argument("--write-coarse-cache", type=Path, default=None)
    parser.add_argument(
        "--reuse-lonv", type=Path, default=None,
        help="multi-rank export reuses the null vectors of this cache instead "
             "of rebuilding them (reproduces a reference stencil bit for bit)")
    parser.add_argument(
        "--skip-export-galerkin", action="store_true",
        help="skip the in-band Galerkin self-check of each rank's overlap patch")
    parser.add_argument("--build-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--no-deflate", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument("--local-stencil", action="store_true")
    parser.add_argument("--local-stencil-batch", type=int, default=4)
    parser.add_argument(
        "--legacy-layout", choices=("auto", "odd-schur", "full-site"),
        default="auto",
        help="canonical input layout used by the KUPL Legacy MG ABI")
    parser.add_argument("--gauge", default=None)
    parser.add_argument("--source", default=None)
    # Preferred over the global --gauge/--source files for large lattices: a
    # slab directory skips the 384 B-request hyperslab pattern described in
    # _read_local_slab.
    parser.add_argument("--gauge-slab-dir", type=Path, default=None,
                        help="directory of contiguous rank_XXXX.h5 gauge slabs")
    parser.add_argument("--source-slab-dir", type=Path, default=None,
                        help="directory of contiguous rank_XXXX.h5 source slabs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-independent-residual", action="store_true",
        help="skip the expensive Python full-operator residual for timing scans")
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
    comm = MPI.COMM_WORLD if MPI is not None else None
    rank = int(comm.Get_rank()) if comm is not None else 0
    size = int(comm.Get_size()) if comm is not None else 1
    grid = tuple(args.grid) if args.grid is not None else (size, 1, 1, 1)
    if int(np.prod(grid)) != size:
        parser.error(f"--grid {grid} must contain {size} ranks")
    coordinate, local_lattice, local_starts = _rank_geometry(
        lattice, grid, rank)
    if (args.levels > 1 and args.restart != 0
            and args.coarse_cache is None):
        try:
            mg_coarse_geometry(local_lattice, args.mg_grid)
        except ValueError as exc:
            parser.error(str(exc))
    if (size > 1 and args.restart != 0 and args.coarse_cache is None
            and args.write_coarse_cache is None):
        parser.error(
            "multi-rank coarse-correction mode requires a global --coarse-cache "
            "or --write-coarse-cache (rank-local export)")
    if size > 1 and args.write_coarse_cache is not None:
        if args.build_device != "cpu":
            parser.error(
                "multi-rank coarse-cache export currently requires "
                "--build-device cpu (the KUPL C++ local 33-stencil builder)")
        if args.levels < 2 or args.restart == 0:
            parser.error(
                "multi-rank coarse-cache export requires --levels 2 and "
                "--restart != 0")
    if args.reuse_lonv is not None and not (
            size > 1 and args.write_coarse_cache is not None):
        parser.error(
            "--reuse-lonv applies to the multi-rank coarse-cache export "
            "(--write-coarse-cache with more than one rank)")
    dtype = np.dtype(np.complex64 if args.dtype == "c64" else np.complex128)
    hdf5_mpio_read = _hdf5_mpio_read_enabled(comm)
    # Observed (L48, 2026-11-17): `input_prebind_seconds` is KUPL
    # applyInitQcu/register work (MPI topology plus scratch allocation, then
    # torn down again), not a second read of the inputs and not overlapped
    # with them.
    input_prebind_seconds = 0.0
    if (_environment_enabled("KUPL_NUMA_BIND")
            and not _environment_enabled("KUPL_SKIP_INPUT_PREBIND")):
        started = time.perf_counter()
        prebind_kupl_inputs(
            qcu, local_lattice,
            float(args.mass if args.mass is not None else DEFAULT_MASS),
            args.atol, float(args.csw if args.csw is not None else 1.0), dtype,
            args.max_iter, grid, rank, size)
        input_prebind_seconds = time.perf_counter() - started
    input_read_started = time.perf_counter()
    gauge_full = None
    gauge_slab_dir = (
        resolve_path(args.gauge_slab_dir, args.gauge_slab_dir)
        if args.gauge_slab_dir is not None else None)
    gauge_local = _read_local_slab(
        args.gauge, local_starts, local_lattice,
        hdf5_mpio=hdf5_mpio_read, comm=comm,
        slab_dir=gauge_slab_dir, rank=rank)
    if gauge_local is None:
        gauge_full, metadata = load_gauge(args.gauge, lattice, dtype)
        gauge_local = _slice_last4(gauge_full, local_starts, local_lattice)
    else:
        gauge_local = gauge_local.astype(dtype, copy=False)
        if args.gauge is not None:
            metadata = read_h5_metadata(
                resolve_path(args.gauge, resolve_path(None, args.gauge)))
        else:
            # Slab-only mode: optional attrs provide mass/csw defaults; the
            # lattice and dtype come from --lat/--dtype.
            slab_path = Path(gauge_slab_dir) / f"rank_{int(rank):04d}.h5"
            metadata = (read_h5_metadata(slab_path)
                        if slab_path.is_file() else {})
    gauge_read_seconds = time.perf_counter() - input_read_started
    gauge = qcu_gauge_to_kupl(gauge_local)
    source_read_started = time.perf_counter()
    source_local = _read_local_slab(
        args.source, local_starts, local_lattice,
        hdf5_mpio=hdf5_mpio_read, comm=comm,
        slab_dir=(resolve_path(args.source_slab_dir, args.source_slab_dir)
                  if args.source_slab_dir is not None else None),
        rank=rank)
    if source_local is None:
        source_full = load_source(args.source, lattice, args.seed, dtype)
        source_local = _slice_last4(source_full, local_starts, local_lattice)
    else:
        source_local = source_local.astype(dtype, copy=False)
    source_read_seconds = time.perf_counter() - source_read_started
    source = qcu_fermion_to_kupl(source_local)
    solution = np.zeros_like(source)
    clover_shape = (4, 3, 4, 3, *local_lattice[:3], local_lattice[3] // 2)
    clover_even = np.empty(clover_shape, dtype=dtype)
    clover_odd = np.empty_like(clover_even)
    clover_even_inv = np.empty_like(clover_even)
    clover_odd_inv = np.empty_like(clover_even)
    mass = float(args.mass if args.mass is not None else metadata.get(
        "reference_mass", metadata.get("mass", DEFAULT_MASS)))
    csw = resolve_csw(args.csw, metadata)
    kappa = 1.0 / (2.0 * mass + 8.0)
    params, argv, set_ptrs = make_params(
        local_lattice, mass, args.atol, csw, dtype, args.max_iter,
        grid=grid, rank=rank, size=size)
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

    clover_assets = None
    if args.build_device == "cpu":
        clover_assets = (
            clover_even, clover_odd, clover_even_inv, clover_odd_inv)
        # Build the Clover terms once; build_coarse_assets() reuses these
        # arrays for its null-vector/stencil setup on the CPU backend. The
        # CUDA builder cannot reuse host arrays, so it keeps applying them
        # itself and the solver builds the host terms later as before.
        qcu.applyInitQcu(
            wrapped["set_ptrs"], wrapped["params"], wrapped["argv"])
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
        finally:
            qcu.applyEndQcu(wrapped["set_ptrs"], wrapped["params"])

    use_coarse_assets = args.levels > 1 and args.restart != 0
    null_vectors, hop_nn, hop_diag, sitting = ((), (), (), ())
    builder_seconds = 0.0
    coarse_export = None
    coarse_source = "built" if use_coarse_assets else "unused: restart=0"
    cache_layout = args.legacy_layout
    coarse_starts = tuple(
        local_starts[axis] // args.mg_grid[axis] for axis in range(4))
    coarse_local = tuple(
        local_lattice[axis] // args.mg_grid[axis] for axis in range(4))
    if use_coarse_assets:
        started = time.perf_counter()
        gauge_file_digest = None
        gauge_canonical_digest = None
        if args.gauge is not None:
            gauge_path = resolve_path(
                args.gauge, Path(args.gauge).expanduser())
            if rank == 0:
                gauge_file_digest = file_sha256(gauge_path)
                if gauge_full is not None:
                    gauge_canonical_digest = logical_array_sha256(gauge_full)
                else:
                    hash_gauge, _ = load_gauge(args.gauge, lattice, dtype)
                    gauge_canonical_digest = logical_array_sha256(hash_gauge)
                    del hash_gauge
            if comm is not None:
                gauge_file_digest = comm.bcast(gauge_file_digest, root=0)
                gauge_canonical_digest = comm.bcast(
                    gauge_canonical_digest, root=0)
        cache_identity = legacy_mg_cache_identity(
            gauge_canonical_sha256=gauge_canonical_digest,
            **({
                "gauge_file_sha256": gauge_file_digest
            } if gauge_file_digest is not None else {}),
            lattice=lattice,
            dtype=dtype,
            mass=mass,
            csw=csw,
            dof=args.dof[:args.levels],
            mg_grid=args.mg_grid,
            nv_iters=args.nv_iters,
            seed=args.seed,
            # This builder uses the KUPL null-vector loop, which has been a real
            # inverse iteration since commit 97508c6a.  Declaring the revision
            # makes the artifact refuse a case that asks for a different builder;
            # an artifact that does not declare one cannot be told apart from the
            # old `v -= S^-1(S v)` (solver-noise) basis and is rejected by
            # pyqcu.test.  The CUDA builder deliberately does NOT declare it.
            # ``null_vector_builder()`` reads KUPL_NV_BUILDER so this manifest
            # names the loop the C++ actually ran (legacy-residual/1 when the
            # switch asks for the old one), in both the write and validate paths
            # below: the same `cache_identity` object is used for both.
            nv_builder=null_vector_builder(),
        )
        if args.coarse_cache is not None:
            cache_path = resolve_path(
                args.coarse_cache, Path(args.coarse_cache).expanduser())
            if not cache_path.is_file():
                raise FileNotFoundError(cache_path)
            ranked_cache = is_legacy_mg_ranked_cache(cache_path)
            validation_error = None
            if ranked_cache:
                # Every rank proves its own hyperslab first; rank zero then
                # re-checks the manifest tiling of the whole cache.
                try:
                    validate_legacy_mg_cache(
                        cache_path, cache_identity,
                        verify_file_sha256=True,
                        verify_dataset_sha256=True, ranks=(rank,))
                    if rank == 0:
                        validate_legacy_mg_cache(
                            cache_path, cache_identity,
                            verify_file_sha256=True,
                            verify_dataset_sha256=False, ranks=())
                except (OSError, TypeError, ValueError) as exc:
                    validation_error = f"{type(exc).__name__}: {exc}"
                if comm is not None:
                    errors = comm.allgather(validation_error)
                    validation_error = next(
                        (item for item in errors if item is not None), None)
            else:
                if rank == 0:
                    try:
                        validate_legacy_mg_cache(
                            cache_path, cache_identity,
                            verify_file_sha256=True,
                            verify_dataset_sha256=True)
                    except (OSError, TypeError, ValueError) as exc:
                        validation_error = f"{type(exc).__name__}: {exc}"
                if comm is not None:
                    validation_error = comm.bcast(validation_error, root=0)
            if validation_error is not None:
                raise ValueError(
                    f"coarse cache identity validation failed: "
                    f"{validation_error}")
            cache_layout, coarse_local, coarse_starts = inspect_cache_layout(
                cache_path, lattice, list(args.mg_grid), list(grid),
                local_starts, args.legacy_layout)
            null_vectors, hop_nn, hop_diag, sitting = load_coarse_assets(
                cache_path, dtype,
                starts=coarse_starts if size > 1 else None,
                local_shape=coarse_local if size > 1 else None)
            coarse_source = str(cache_path)
        elif size > 1:
            cache_output = resolve_path(
                args.write_coarse_cache,
                Path(args.write_coarse_cache).expanduser())
            if comm is not None:
                cache_output = Path(comm.bcast(str(cache_output), root=0))
            source_digest = None
            if args.source is not None and rank == 0:
                source_path = Path(args.source).expanduser()
                if source_path.is_file():
                    source_digest = file_sha256(source_path)
            if comm is not None:
                source_digest = comm.bcast(source_digest, root=0)
            reuse_lonv_path = None
            if args.reuse_lonv is not None:
                reuse_lonv_path = resolve_path(
                    args.reuse_lonv, Path(args.reuse_lonv).expanduser())
                if comm is not None:
                    reuse_lonv_path = Path(
                        comm.bcast(str(reuse_lonv_path), root=0))
                reuse_error = None
                if rank == 0:
                    try:
                        validate_legacy_mg_cache(
                            reuse_lonv_path, cache_identity,
                            verify_file_sha256=False,
                            verify_dataset_sha256=False, ranks=())
                    except (OSError, TypeError, ValueError) as exc:
                        reuse_error = f"{type(exc).__name__}: {exc}"
                if comm is not None:
                    reuse_error = comm.bcast(reuse_error, root=0)
                if reuse_error is not None:
                    raise ValueError(
                        f"--reuse-lonv identity validation failed: {reuse_error}")
            (null_vectors, hop_nn, hop_diag, sitting), export_timings, _ = \
                export_ranked_coarse_cache(
                    cache_output, wrapped=wrapped, params=params, argv=argv,
                    set_ptrs=set_ptrs, gauge_local=gauge_local, dtype=dtype,
                    lattice=lattice, local_lattice=local_lattice,
                    coordinate=coordinate, grid=grid, rank=rank, size=size,
                    comm=comm, mg_grid=list(args.mg_grid),
                    dof=list(args.dof), nv_iters=args.nv_iters,
                    seed=args.seed, mass=mass, kappa=kappa, csw=csw,
                    identity=cache_identity,
                    mg_config={
                        "levels": int(args.levels),
                        "restart": int(args.restart),
                        "mg_grid": [int(value) for value in args.mg_grid],
                        "nv_iters": int(args.nv_iters),
                        "dof": [int(value) for value in args.dof[:args.levels]],
                        "coarse_max_iter": int(args.coarse_max_iter),
                        "coarse_tol_factor": float(args.coarse_tol_factor),
                        "deflate": not args.no_deflate,
                    },
                    metadata=metadata, batch_size=args.local_stencil_batch,
                    source_digest=source_digest,
                    verbose=args.verbose or args.verbose_build,
                    gauge_path=(gauge_path if args.gauge is not None else None),
                    gauge_full=gauge_full,
                    reuse_lonv=reuse_lonv_path,
                    patch_galerkin=not args.skip_export_galerkin)
            export_seconds = export_timings
            coarse_source = f"{cache_output} (rank-local export)"
            coarse_export = export_seconds
        else:
            coarse_export = None
            null_vectors, hop_nn, hop_diag, sitting = build_coarse_assets(
                gauge_local, lattice, mass, kappa, csw, dtype, args.levels,
                list(args.dof), list(args.mg_grid), args.nv_iters,
                args.seed, verbose=args.verbose or args.verbose_build,
                local_stencil=args.local_stencil,
                local_stencil_batch_size=args.local_stencil_batch,
                device_name=args.build_device, gauge_parity=gauge,
                clover_assets=(
                    clover_assets if args.build_device == "cpu" else None))
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
        if args.write_coarse_cache is not None and size == 1:
            cache_output = resolve_path(
                args.write_coarse_cache,
                Path(args.write_coarse_cache).expanduser())
            save_legacy_mg_cache(
                cache_output,
                {
                    "lonv": null_vectors[0], "hnn": hop_nn[0],
                    "hdg": hop_diag[0], "sit": sitting[0],
                },
                identity=cache_identity,
                setup_method="cxx-null-vector+local-33-stencil",
                setup_backend="kupl-qcu-cxx",
                source_metadata=metadata,
                setup_seconds=builder_seconds)

    set_ptrs[SET_PTRS_LEGACY_LAYOUT] = resolve_legacy_layout(
        cache_layout, use_coarse_assets, tuple(null_vectors), local_lattice)

    # Observed (L48, 2026-11-17): `wall_seconds` is the solve call only, max
    # over ranks.  For the same 11 BiCGStab iterations it was 1.665 s at 32
    # ranks on one exclusive node and 183 s at 16 ranks on a shared node; the
    # 16-rank shared point is pathological (the node, not the rank count), so
    # 32-rank comparisons need an exclusive node.
    verify_status = None
    qcu.applyInitQcu(wrapped["set_ptrs"], wrapped["params"], wrapped["argv"])
    try:
        if args.build_device != "cpu":
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
        if use_coarse_assets:
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
        iterations = (None if use_coarse_assets else
                      qcu.getCloverBistabCgIterationsQcu(
                          wrapped["set_ptrs"], wrapped["params"]))
    finally:
        qcu.applyEndQcu(wrapped["set_ptrs"], wrapped["params"])

    if comm is not None:
        builder_seconds = float(comm.allreduce(builder_seconds, op=MPI.MAX))
        wall_seconds = float(comm.allreduce(wall_seconds, op=MPI.MAX))
        if verify_status is not None:
            verify_status = int(comm.allreduce(verify_status, op=MPI.MAX))

    # Observed (L48, 2026-11-17): `residual_seconds` below times the
    # independent NumPy reconstruction in kupl_residual, not the solver.  It can
    # dominate the row (139.8 s against `wall_seconds` 183.0 s at 16 ranks;
    # 161.7 s against 110.4 s with a coarse cache), so it is never added to the
    # solver wall.
    residual_seconds = 0.0
    residual_started = time.perf_counter()
    if args.skip_independent_residual:
        residual = None
        acceptance = solver_acceptance(args.atol, dtype=dtype)
        if verify_status == 0:
            converged, status = True, "PASS"
        elif verify_status is None:
            # No independent residual and no native verification: the solve
            # is unverified, so it must not be reported as PASS.
            converged, status = False, "WARN"
        else:
            converged, status = False, "FAIL"
        residual_source = "KUPL native fail-closed solve; independent check skipped"
    elif size > 1:
        residual = distributed_relative_residual(
            solution, source, gauge, clover_even, clover_odd, kappa,
            comm, coordinate, grid)
        converged, status, acceptance = convergence_status(
            residual, args.atol, dtype=dtype,
            strict=getattr(args, "strict_acceptance", False))
        residual_source = (
            "independent distributed NumPy full Wilson-Clover reconstruction "
            "with MPI halo exchange")
    else:
        residual = relative_full_residual(
            source, solution, gauge, clover_even, clover_odd, kappa)
        converged, status, acceptance = convergence_status(
            residual, args.atol, dtype=dtype,
            strict=getattr(args, "strict_acceptance", False))
        residual_source = "independent NumPy full Wilson-Clover reconstruction"
    residual_seconds = time.perf_counter() - residual_started
    if comm is not None:
        residual_seconds = float(comm.allreduce(residual_seconds, op=MPI.MAX))
    if verify_status not in (None, 0):
        status = "FAIL"
        converged = False
    if args.output:
        output = resolve_path(args.output, resolve_path(None, args.output))
        if size > 1:
            solution_full = gather_solution(
                solution, lattice, grid, local_starts, comm)
        else:
            solution_full = kupl_fermion_to_qcu(solution)
        if rank == 0:
            write_h5(output, solution_full, metadata={
                "backend": "kupl", "solver": "applyCloverMultigridQcu",
                "mass": mass, "kappa": kappa, "csw": csw, "atol": args.atol,
                "levels": args.levels, "dof": list(args.dof[:args.levels]),
                "mg_grid": list(args.mg_grid), "mpi_size": size,
                "grid": list(grid), "layout": "[spin,color,x,y,z,t]",
            })
    report = {
        "backend": "kupl", "solver": "applyCloverMultigridQcu",
        "lattice": list(lattice), "local_lattice": list(local_lattice),
        "local_starts": list(local_starts), "grid": list(grid),
        "grid_coordinate": list(coordinate), "mpi_rank": rank,
        "mpi_size": size, "dtype": str(dtype), "mass": mass,
        "kappa": kappa, "mass_normalization": "kappa", "csw": csw,
        "atol": args.atol, "max_iter": args.max_iter, "levels": args.levels,
        "dof": list(args.dof[:args.levels]), "mg_grid": list(args.mg_grid),
        "restart": args.restart, "coarse_max_iter": args.coarse_max_iter,
        "coarse_tol_factor": args.coarse_tol_factor,
        "deflate": not args.no_deflate, "builder_seconds": builder_seconds,
        "input_prebind_seconds": input_prebind_seconds,
        "hdf5_mpio_read": hdf5_mpio_read,
        "gauge_slab_dir": (str(args.gauge_slab_dir)
                           if args.gauge_slab_dir is not None else None),
        "source_slab_dir": (str(args.source_slab_dir)
                            if args.source_slab_dir is not None else None),
        "gauge_read_seconds": gauge_read_seconds,
        "source_read_seconds": source_read_seconds,
        "coarse_source": coarse_source,
        "coarse_export": coarse_export,
        "wall_seconds": wall_seconds, "iterations": iterations,
        "residual_seconds": residual_seconds,
        "verify_status": verify_status,
        "final_true_residual": residual, "residual_acceptance": acceptance,
        "residual_source": residual_source,
        "converged": converged, "memory": cpu_memory_snapshot(),
        "status": status,
    }
    if comm is not None:
        failed = int(comm.allreduce(int(status == "FAIL"), op=MPI.MAX))
        warned = int(comm.allreduce(int(status == "WARN"), op=MPI.MAX))
        status = "FAIL" if failed else ("WARN" if warned else "PASS")
        report["status"] = status
    if rank == 0:
        report = json_safe(report, omit_none=True)
        write_json(resolve_path(args.report, DATA_DIR / args.report), report,
                   omit_none=True)
        print(json.dumps(report, ensure_ascii=False))
    return 0 if status != "FAIL" else 1


if __name__ == "__main__":
    exit_code = main()
    if MPI is not None and not MPI.Is_finalized():
        MPI.Finalize()
    raise SystemExit(exit_code)
