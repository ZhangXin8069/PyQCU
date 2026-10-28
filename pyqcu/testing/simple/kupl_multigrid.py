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

try:
    from mpi4py import MPI
except ImportError:  # pragma: no cover - KUPL itself requires MPI.
    MPI = None

try:
    from ._common import (DATA_DIR, DEFAULT_ATOL, DEFAULT_BLOCK, DEFAULT_MASS,
                          convergence_status, kupl_fermion_to_qcu,
                          qcu_fermion_to_kupl, qcu_gauge_to_kupl,
                          resolve_csw, resolve_path, write_h5, write_json)
    from ._layouts import qcu_gauge_to_kupl as gauge_to_parity
    from .kupl_bistabcg import (NumpyTensor, PLAN_CLOVER, PLAN_SOLVER,
                                PARITY, SET_PLAN, cpu_memory_snapshot,
                                kupl_backend, load_gauge, load_source,
                                make_params, prebind_kupl_inputs,
                                relative_full_residual)
    from .kupl_residual import (distributed_relative_residual,
                                gather_solution)
except ImportError:
    from _common import (DATA_DIR, DEFAULT_ATOL, DEFAULT_BLOCK, DEFAULT_MASS,
                         convergence_status, kupl_fermion_to_qcu,
                         qcu_fermion_to_kupl, qcu_gauge_to_kupl,
                         resolve_csw, resolve_path, write_h5, write_json)
    from _layouts import qcu_gauge_to_kupl as gauge_to_parity
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
                        device_name: str = "cpu",
                        setup_stats: dict[str, Any] | None = None):
    if levels != 2:
        raise ValueError("the independent simple MG fixture supports two levels")
    del kappa
    gauge_parity = gauge_to_parity(np.ascontiguousarray(gauge))
    fine_odd = (lattice[0], lattice[1], lattice[2], lattice[3] // 2)
    coarse_lattice = tuple(
        fine_odd[index] // mg_grid[index] for index in range(4))
    block = tuple(
        fine_odd[index] // coarse_lattice[index] for index in range(4))
    E, e = int(dof[1]), 12
    null_shape = (
        E, e, coarse_lattice[0], block[0], coarse_lattice[1], block[1],
        coarse_lattice[2], block[2], coarse_lattice[3], block[3])
    onsite_shape = (4, 3, 4, 3, *fine_odd)
    sit_shape = (E, E, *coarse_lattice)
    hnn_shape = (2, 4, E, E, *coarse_lattice)
    hdg_shape = (2, 2, 6, E, E, *coarse_lattice)

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
        null_vectors = torch.empty(null_shape, dtype=torch_dtype, device=device)
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
            data_type=int(params[DATA_TYPE]), array_module="numpy")
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
            if verbose:
                print(
                    "PYQCU::CUDA::MG_SETUP:\n C++ null-space start",
                    flush=True)
            stage = time.perf_counter()
            qcu.applyMultigridBuildNullVectorsQcu(
                null_vectors, gauge_device, clover_even, clover_odd,
                clover_even_inv, clover_odd_inv, set_ptrs, params, E, e,
                *fine_odd, *coarse_lattice, max(1, nv_iters), seed, 1.0e-2)
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
                clover_odd, clover_even_inv, clover_odd_inv, set_ptrs, params)
            qcu.applyMultigridRestrictQcu(
                restricted, schur_vector, null_vectors, set_ptrs, params)
            qcu.applyMultigridCoarseDslashWideQcu(
                stencil_vector, coarse_vector, sitting, hop_nn, hop_diag,
                set_ptrs, params)
            denominator = torch.linalg.vector_norm(restricted).clamp_min(1.0)
            galerkin_error = float((
                torch.linalg.vector_norm(stencil_vector - restricted) /
                denominator).item())
            if not np.isfinite(galerkin_error) or galerkin_error > 2.0e-5:
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
        clover_even = np.empty(onsite_shape, dtype=dtype)
        clover_odd = np.empty(onsite_shape, dtype=dtype)
        clover_even_inv = np.empty(onsite_shape, dtype=dtype)
        clover_odd_inv = np.empty(onsite_shape, dtype=dtype)
        null_vectors = np.empty(null_shape, dtype=dtype)
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
            params[PARITY] = 0
            qcu.applyCloversQcu(
                clover_even_arg, clover_even_inv_arg, gauge_arg,
                set_ptrs_arg, params_arg)
            params[PARITY] = 1
            qcu.applyCloversQcu(
                clover_odd_arg, clover_odd_inv_arg, gauge_arg,
                set_ptrs_arg, params_arg)
            params[SET_PLAN] = PLAN_SOLVER
            stage = time.perf_counter()
            status = qcu.applyMultigridBuildNullVectorsQcu(
                null_arg, gauge_arg, clover_even_arg, clover_odd_arg,
                clover_even_inv_arg, clover_odd_inv_arg, set_ptrs_arg,
                params_arg, E, e,
                *fine_odd, *coarse_lattice, max(1, nv_iters), seed, 1.0e-2)
            if status != 0:
                raise RuntimeError(f"KUPL null-vector setup failed: {status}")
            timings["null_vector_seconds"] = time.perf_counter() - stage
            stage = time.perf_counter()
            status = qcu.applyMultigridBuildStencilLocalQcu(
                sitting_arg, hop_nn_arg, hop_diag_arg, null_arg, gauge_arg,
                clover_even_arg, clover_odd_arg, clover_even_inv_arg,
                clover_odd_inv_arg, set_ptrs_arg, params_arg, E, e,
                *fine_odd, *coarse_lattice, 8, local_stencil_batch_size)
            if status != 0:
                raise RuntimeError(f"KUPL stencil setup failed: {status}")
            timings["stencil_seconds"] = time.perf_counter() - stage
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
                NumpyTensor(prolonged), NumpyTensor(coarse_vector), null_arg,
                set_ptrs_arg, params_arg)
            qcu.applyCloverBistabCgDslashQcu(
                NumpyTensor(schur_vector), NumpyTensor(prolonged), gauge_arg,
                clover_even_arg, clover_odd_arg, clover_even_inv_arg,
                clover_odd_inv_arg, set_ptrs_arg, params_arg)
            qcu.applyMultigridRestrictQcu(
                NumpyTensor(restricted), NumpyTensor(schur_vector), null_arg,
                set_ptrs_arg, params_arg)
            qcu.applyMultigridCoarseDslashWideQcu(
                NumpyTensor(stencil_vector), NumpyTensor(coarse_vector),
                sitting_arg, hop_nn_arg, hop_diag_arg, set_ptrs_arg,
                params_arg)
            denominator = max(float(np.linalg.norm(restricted)), 1.0)
            galerkin_error = float(
                np.linalg.norm(stencil_vector - restricted) / denominator)
            if not np.isfinite(galerkin_error) or galerkin_error > 2.0e-5:
                raise RuntimeError(
                    f"KUPL C++ stencil Galerkin error {galerkin_error:.3e}")
            timings["galerkin_validation_seconds"] = (
                time.perf_counter() - stage)
        finally:
            qcu.applyEndQcu(set_ptrs_arg, params_arg)
        values = (null_vectors, hop_nn, hop_diag, sitting)
    else:
        raise ValueError(f"unsupported build device {device_name!r}")

    null_vectors, hop_nn, hop_diag, sitting = values
    columns = null_vectors.reshape(
        E, e, coarse_lattice[0], block[0], coarse_lattice[1], block[1],
        coarse_lattice[2], block[2], coarse_lattice[3], block[3]).transpose(
            2, 4, 6, 8, 0, 1, 3, 5, 7, 9).reshape(
                int(np.prod(coarse_lattice)), E, e * int(np.prod(block)))
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
            **timings,
            "seconds": time.perf_counter() - stage_started,
        })
    return tuple([value] for value in values)


def load_coarse_assets(path: Path, dtype: np.dtype,
                       starts: tuple[int, int, int, int] | None = None,
                       local_shape: tuple[int, int, int, int] | None = None):
    """Load the backend-neutral ``lonv/hnn/hdg/sit`` 33-tensor cache."""
    import h5py

    with h5py.File(path, "r") as handle:
        required = {"lonv", "hnn", "hdg", "sit"}
        missing = sorted(required.difference(handle.keys()))
        if missing:
            raise ValueError(f"coarse cache {path} is missing datasets {missing}")
        values = {}
        for name in ("lonv", "hnn", "hdg", "sit"):
            dataset = handle[name]
            if starts is None or local_shape is None:
                selection = dataset[...]
            else:
                if dataset.ndim < 4:
                    raise ValueError(
                        f"coarse cache {name} has invalid shape {dataset.shape}")
                slices = [slice(None)] * dataset.ndim
                if name == "lonv":
                    if dataset.ndim != 10:
                        raise ValueError(
                            f"coarse cache lonv has invalid shape {dataset.shape}")
                    dimensions = (2, 4, 6, 8)
                else:
                    dimensions = tuple(range(dataset.ndim - 4, dataset.ndim))
                for axis, dimension in enumerate(dimensions):
                    slices[dimension] = slice(
                        int(starts[axis]),
                        int(starts[axis] + local_shape[axis]))
                selection = dataset[tuple(slices)]
            values[name] = np.ascontiguousarray(selection, dtype=dtype)
    # Match build_coarse_assets(): each return value is a per-level sequence.
    # Returning the raw 10-D array makes callers interpret its E axis as levels.
    return (values["lonv"],), (values["hnn"],), (values["hdg"],), (values["sit"],)


def write_coarse_assets(path: Path, null_vectors, hop_nn, hop_diag, sitting):
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        for name, value in (
                ("lonv", null_vectors), ("hnn", hop_nn),
                ("hdg", hop_diag), ("sit", sitting)):
            handle.create_dataset(name, data=np.asarray(value))


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
    parser.add_argument("--lat", nargs=4, type=int, default=(4, 4, 4, 4))
    parser.add_argument("--grid", nargs=4, type=int, default=None,
                        help="MPI process grid; default: one rank in x")
    parser.add_argument("--mass", type=float, default=None)
    parser.add_argument("--csw", type=float, default=None)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--dtype", choices=("c64", "c128"), default="c64")
    parser.add_argument("--levels", type=int, default=2)
    parser.add_argument("--dof", nargs="+", type=int, default=(12, 2, 2, 2, 2))
    parser.add_argument("--mg-grid", nargs=4, type=int, default=DEFAULT_BLOCK)
    parser.add_argument(
        "--restart", type=int, default=5,
        help="coarse-correction interval; 0 selects optimized parity-Schur BiCGStab")
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
    parser.add_argument(
        "--legacy-layout", choices=("auto", "odd-schur", "full-site"),
        default="auto",
        help="canonical input layout used by the KUPL Legacy MG ABI")
    parser.add_argument("--gauge", default=None)
    parser.add_argument("--source", default=None)
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
    if args.levels > 1 and any(
            local_lattice[axis] % args.mg_grid[axis] for axis in range(4)):
        parser.error(
            f"local lattice {local_lattice} is not divisible by "
            f"--mg-grid {tuple(args.mg_grid)}")
    if size > 1 and args.restart != 0 and args.coarse_cache is None:
        parser.error(
            "multi-rank coarse-correction mode requires a global --coarse-cache")
    if size > 1 and args.write_coarse_cache is not None:
        parser.error("multi-rank coarse-cache export is not implemented")
    dtype = np.dtype(np.complex64 if args.dtype == "c64" else np.complex128)
    input_prebind_seconds = 0.0
    if _environment_enabled("KUPL_NUMA_BIND"):
        started = time.perf_counter()
        prebind_kupl_inputs(
            qcu, local_lattice,
            float(args.mass if args.mass is not None else DEFAULT_MASS),
            args.atol, float(args.csw if args.csw is not None else 1.0), dtype,
            args.max_iter, grid, rank, size)
        input_prebind_seconds = time.perf_counter() - started
    gauge_full, metadata = load_gauge(args.gauge, lattice, dtype)
    gauge_local = _slice_last4(gauge_full, local_starts, local_lattice)
    gauge = qcu_gauge_to_kupl(gauge_local)
    source_full = load_source(args.source, lattice, args.seed, dtype)
    source_local = _slice_last4(source_full, local_starts, local_lattice)
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

    use_coarse_assets = args.levels > 1 and args.restart != 0
    null_vectors, hop_nn, hop_diag, sitting = ((), (), (), ())
    builder_seconds = 0.0
    coarse_source = "built" if use_coarse_assets else "unused: restart=0"
    cache_layout = args.legacy_layout
    coarse_starts = tuple(
        local_starts[axis] // args.mg_grid[axis] for axis in range(4))
    coarse_local = tuple(
        local_lattice[axis] // args.mg_grid[axis] for axis in range(4))
    if use_coarse_assets:
        started = time.perf_counter()
        if args.coarse_cache is not None:
            cache_path = resolve_path(
                args.coarse_cache, Path(args.coarse_cache).expanduser())
            if not cache_path.is_file():
                raise FileNotFoundError(cache_path)
            cache_layout, coarse_local, coarse_starts = inspect_cache_layout(
                cache_path, lattice, list(args.mg_grid), list(grid),
                local_starts, args.legacy_layout)
            null_vectors, hop_nn, hop_diag, sitting = load_coarse_assets(
                cache_path, dtype,
                starts=coarse_starts if size > 1 else None,
                local_shape=coarse_local if size > 1 else None)
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

    set_ptrs[SET_PTRS_LEGACY_LAYOUT] = resolve_legacy_layout(
        cache_layout, use_coarse_assets, tuple(null_vectors), local_lattice)

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

    residual_seconds = 0.0
    residual_started = time.perf_counter()
    if args.skip_independent_residual:
        residual = None
        converged = verify_status in (None, 0)
        status = "PASS" if converged else "FAIL"
        acceptance = float(args.atol)
        residual_source = "KUPL native fail-closed solve; independent check skipped"
    elif size > 1:
        residual = distributed_relative_residual(
            solution, source, gauge, clover_even, clover_odd, kappa,
            comm, coordinate, grid)
        converged, status, acceptance = convergence_status(
            residual, args.atol, dtype=dtype)
        residual_source = (
            "independent distributed NumPy full Wilson-Clover reconstruction "
            "with MPI halo exchange")
    else:
        residual = relative_full_residual(
            source, solution, gauge, clover_even, clover_odd, kappa)
        converged, status, acceptance = convergence_status(
            residual, args.atol, dtype=dtype)
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
        "coarse_source": coarse_source,
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
        status = "FAIL" if failed else "PASS"
        report["status"] = status
    if rank == 0:
        write_json(resolve_path(args.report, DATA_DIR / args.report), report)
        print(json.dumps(report, ensure_ascii=False))
    return 0 if status != "FAIL" else 1


if __name__ == "__main__":
    exit_code = main()
    if MPI is not None and not MPI.Is_finalized():
        MPI.Finalize()
    raise SystemExit(exit_code)
