#!/usr/bin/env python3
"""Differential tests for KUPL strict MultiGrid primitives and V-cycle."""

from __future__ import annotations

import argparse
import ctypes
from pathlib import Path

import numpy as np
from mpi4py import MPI


LL = ctypes.c_longlong
ULL_P = ctypes.POINTER(ctypes.c_ulonglong)
I = ctypes.c_int
D = ctypes.c_double
I_P = ctypes.POINTER(ctypes.c_int)
D_P = ctypes.POINTER(ctypes.c_double)


def bind_library(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL)
    void_signatures = {
        "applyInitQcu": 3,
        "applyEndQcu": 2,
    }
    status_signatures = {
        "applyMultigridStrictCoarseQcu": (LL,) * 6 + (I,) * 6,
        "applyMultigridStrictMatPCQcu": (LL,) * 6 + (I,) * 6,
        "applyMultigridStrictFineMatPCQcu": (LL,) * 9 + (I,),
        "applyMultigridStrictPrepareQcu": (LL,) * 7 + (I,) * 6,
        "applyMultigridStrictReconstructQcu": (LL,) * 8 + (I,) * 6,
        "applyMultigridStrictRestrictQcu": (LL,) * 5 + (I,) * 11,
        "applyMultigridStrictProLongQcu": (LL,) * 5 + (I,) * 11,
        "applyMultigridStrictVCycleQcu": (LL,) * 4 + (I, ULL_P),
        "applyMultigridStrictInitQcu": (LL,) * 2 + (I, ULL_P),
        "applyMultigridStrictEndQcu": (LL,) * 2,
    }
    for name, count in void_signatures.items():
        function = getattr(library, name)
        function.argtypes = (LL,) * count
        function.restype = None
    for name, argtypes in status_signatures.items():
        function = getattr(library, name)
        function.argtypes = argtypes
        function.restype = ctypes.c_int
    library.applyMultigridStrictFgmresQcu.argtypes = (
        (LL,) * 10 +
        (I,) * 13 + (D,) + (I,) * 2 +
        (ctypes.c_ulonglong, I_P, I_P, D_P, ULL_P)
    )
    library.applyMultigridStrictFgmresQcu.restype = ctypes.c_int
    return library


def pointer(array: np.ndarray) -> int:
    return int(array.ctypes.data)


def local_slice(global_array: np.ndarray, components: int,
                global_dims: tuple[int, int, int, int],
                rank: int) -> np.ndarray:
    shaped = global_array.reshape(components, *global_dims)
    local_x = global_dims[0] // MPI.COMM_WORLD.size
    begin = rank * local_x
    return np.ascontiguousarray(
        shaped[(slice(None), slice(begin, begin + local_x), ...,)]
    ).reshape(-1)


def globalize_local(local_array: np.ndarray, components: int,
                    local_dims: tuple[int, int, int, int],
                    global_dims: tuple[int, int, int, int],
                    rank: int) -> np.ndarray:
    local = local_array.reshape(components, *local_dims)
    global_shaped = np.zeros(
        (components, *global_dims), dtype=local_array.dtype)
    begin = rank * local_dims[0]
    global_shaped[:, begin:begin + local_dims[0]] = local
    result = np.empty_like(global_shaped)
    MPI.COMM_WORLD.Allreduce(global_shaped, result, op=MPI.SUM)
    return result.reshape(-1)


def compact_indices(global_dims: tuple[int, int, int, int],
                    parity: int) -> np.ndarray:
    x_extent, y_extent, z_extent, t_extent = global_dims
    result = []
    for x in range(x_extent):
        for y in range(y_extent):
            for z in range(z_extent):
                for t2 in range(t_extent // 2):
                    t = 2 * t2 + (((x + y + z) & 1) ^ parity)
                    result.append(((x * y_extent + y) * z_extent + z) *
                                  t_extent + t)
    return np.asarray(result, dtype=np.int64)


def compact_half_index(global_dims: tuple[int, int, int, int],
                       x: int, y: int, z: int, t: int) -> int:
    _, y_extent, z_extent, t_extent = global_dims
    return ((x * y_extent + y) * z_extent + z) * (t_extent // 2) + t // 2


def local_compact_slice(global_compact: np.ndarray, components: int,
                        global_dims: tuple[int, int, int, int],
                        local_dims: tuple[int, int, int, int], rank: int,
                        parity: int) -> np.ndarray:
    shaped = global_compact.reshape(components, -1)
    local = np.empty(
        components * local_dims[0] * local_dims[1] * local_dims[2] *
        (local_dims[3] // 2), dtype=global_compact.dtype)
    cursor = 0
    for component in range(components):
        for x in range(local_dims[0]):
            global_x = rank * local_dims[0] + x
            for y in range(local_dims[1]):
                for z in range(local_dims[2]):
                    for t2 in range(local_dims[3] // 2):
                        t = 2 * t2 + (((global_x + y + z) & 1) ^ parity)
                        global_site = (
                            (global_x * global_dims[1] + y) *
                            global_dims[2] + z) * global_dims[3] + t
                        source = compact_half_index(
                            global_dims, global_x, y, z, t)
                        local[cursor] = shaped[component, source]
                        cursor += 1
    return local


def coarse_reference(inp: np.ndarray, links: np.ndarray,
                     onsite: np.ndarray, dims: tuple[int, int, int, int],
                     E: int, onsite_index: int) -> np.ndarray:
    x_extent, y_extent, z_extent, t_extent = dims
    volume = int(np.prod(dims))
    shaped_in = inp.reshape(E, volume)
    shaped_links = links.reshape(2, 4, E, E, volume)
    shaped_onsite = onsite.reshape(2, E, E, volume)
    output = np.zeros_like(shaped_in)
    for row in range(E):
        for x in range(x_extent):
            for y in range(y_extent):
                for z in range(z_extent):
                    for t in range(t_extent):
                        site = ((x * y_extent + y) * z_extent + z) * t_extent + t
                        value = 0.0j
                        if onsite_index >= 0:
                            value += np.einsum(
                                "c,rc->r",
                                shaped_in[:, site],
                                shaped_onsite[onsite_index, :, :, site],
                            )[row]
                        coordinate = [x, y, z, t]
                        extent = [x_extent, y_extent, z_extent, t_extent]
                        for dim in range(4):
                            forward = coordinate.copy()
                            backward = coordinate.copy()
                            forward[dim] = (forward[dim] + 1) % extent[dim]
                            backward[dim] = (backward[dim] - 1) % extent[dim]
                            forward_site = (
                                (forward[0] * y_extent + forward[1]) *
                                z_extent + forward[2]) * t_extent + forward[3]
                            backward_site = (
                                (backward[0] * y_extent + backward[1]) *
                                z_extent + backward[2]) * t_extent + backward[3]
                            value += np.dot(
                                shaped_links[0, dim, row, :, site],
                                shaped_in[:, forward_site])
                            value += np.dot(
                                np.conj(
                                    shaped_links[1, dim, :, row,
                                                 backward_site]),
                                shaped_in[:, backward_site])
                        output[row, site] = value
    return output.reshape(-1)


def compact_holding_reference(
        inp: np.ndarray, links: np.ndarray, dims: tuple[int, int, int, int],
        E: int, source_parity: int, target_parity: int,
        base: np.ndarray | None = None) -> np.ndarray:
    x_extent, y_extent, z_extent, t_extent = dims
    volume = int(np.prod(dims))
    compact_volume = volume // 2
    shaped_in = inp.reshape(E, compact_volume)
    shaped_links = links.reshape(2, 4, E, E, volume)
    output = np.zeros((E, compact_volume), dtype=inp.dtype)
    for row in range(E):
        for x in range(x_extent):
            for y in range(y_extent):
                for z in range(z_extent):
                    for t2 in range(t_extent // 2):
                        target_t = 2 * t2 + (((x + y + z) & 1) ^ target_parity)
                        target_site = (
                            (x * y_extent + y) * z_extent + z) * t_extent + target_t
                        target_half = compact_half_index(
                            dims, x, y, z, target_t)
                        coordinate = [x, y, z, target_t]
                        extent = [x_extent, y_extent, z_extent, t_extent]
                        value = 0.0j
                        for dim in range(4):
                            forward = coordinate.copy()
                            backward = coordinate.copy()
                            forward[dim] = (forward[dim] + 1) % extent[dim]
                            backward[dim] = (backward[dim] - 1) % extent[dim]
                            forward_half = compact_half_index(
                                dims, *forward)
                            backward_half = compact_half_index(
                                dims, *backward)
                            backward_site = (
                                (backward[0] * y_extent + backward[1]) *
                                z_extent + backward[2]) * t_extent + backward[3]
                            value += np.dot(
                                shaped_links[0, dim, row, :, target_site],
                                shaped_in[:, forward_half])
                            value += np.dot(
                                np.conj(
                                    shaped_links[1, dim, :, row,
                                                 backward_site]),
                                shaped_in[:, backward_half])
                        output[row, target_half] = (
                            value if base is None else base.reshape(
                                E, compact_volume)[row, target_half] - value)
    return output.reshape(-1)


_GAMMA = np.asarray([
    [[0, 0, 0, 1j], [0, 0, 1j, 0],
     [0, -1j, 0, 0], [-1j, 0, 0, 0]],
    [[0, 0, 0, -1], [0, 0, 1, 0],
     [0, 1, 0, 0], [-1, 0, 0, 0]],
    [[0, 0, 1j, 0], [0, 0, 0, -1j],
     [-1j, 0, 0, 0], [0, 1j, 0, 0]],
    [[0, 0, 1, 0], [0, 0, 0, 1],
     [1, 0, 0, 0], [0, 1, 0, 0]],
], dtype=np.complex128)


def one_minus_gamma(direction: int, field: np.ndarray,
                    sign: int) -> np.ndarray:
    output = field.copy()
    for color in range(3):
        output[:, color] -= sign * (_GAMMA[direction] @ field[:, color])
    return output


def bare_dslash_reference(
        source: np.ndarray, target: int, gauge: np.ndarray,
        dims: tuple[int, int, int, int]) -> np.ndarray:
    x_extent, y_extent, z_extent, t_extent = dims
    half_volume = int(np.prod(dims)) // 2
    shaped_source = source.reshape(12, half_volume)
    output = np.zeros_like(shaped_source)
    for x in range(x_extent):
        for y in range(y_extent):
            for z in range(z_extent):
                for t2 in range(t_extent // 2):
                    t = 2 * t2 + (((x + y + z) & 1) ^ target)
                    target_half = compact_half_index(dims, x, y, z, t)
                    coordinate = [x, y, z, t]
                    extent = [x_extent, y_extent, z_extent, t_extent]
                    value = np.zeros((4, 3), dtype=source.dtype)
                    for direction in range(4):
                        forward = coordinate.copy()
                        backward = coordinate.copy()
                        forward[direction] = (
                            forward[direction] + 1) % extent[direction]
                        backward[direction] = (
                            backward[direction] - 1) % extent[direction]
                        forward_half = compact_half_index(dims, *forward)
                        backward_half = compact_half_index(dims, *backward)
                        forward_matrix = gauge[
                            target, :, :, direction, x, y, z, t2]
                        backward_matrix = gauge[
                            (sum(backward) & 1), :, :, direction,
                            backward[0], backward[1], backward[2],
                            backward[3] // 2]
                        forward_field = shaped_source[
                            :, forward_half].reshape(4, 3)
                        backward_field = shaped_source[
                            :, backward_half].reshape(4, 3)
                        transported_forward = np.einsum(
                            "ij,sj->si", forward_matrix, forward_field)
                        transported_backward = np.einsum(
                            "ji,sj->si", np.conj(backward_matrix),
                            backward_field)
                        value += one_minus_gamma(
                            direction, transported_forward, 1)
                        value += one_minus_gamma(
                            direction, transported_backward, -1)
                    output[:, target_half] = value.reshape(-1)
    return output.reshape(-1)


def onsite_inverse_reference(
        source: np.ndarray, inverse: np.ndarray,
        dims: tuple[int, int, int, int]) -> np.ndarray:
    half_volume = int(np.prod(dims)) // 2
    matrices = inverse.reshape(12, 12, half_volume)
    return np.einsum(
        "rsc,sc->rc", matrices, source.reshape(12, half_volume)).reshape(-1)


def fine_matpc_reference(
        source: np.ndarray, clover_inverse: tuple[np.ndarray, np.ndarray],
        gauge: np.ndarray, dims: tuple[int, int, int, int],
        kappa: float, parity: int) -> np.ndarray:
    other = 1 - parity
    first = bare_dslash_reference(source, other, gauge, dims)
    first = onsite_inverse_reference(first, clover_inverse[other], dims)
    second = bare_dslash_reference(first, parity, gauge, dims)
    second = onsite_inverse_reference(second, clover_inverse[parity], dims)
    return source - kappa * kappa * second


def onsite_parity_reference(full: np.ndarray, onsite: np.ndarray,
                            dims: tuple[int, int, int, int], E: int,
                            parity: int, inverse: bool) -> np.ndarray:
    x_extent, y_extent, z_extent, t_extent = dims
    volume = int(np.prod(dims))
    shaped_full = full.reshape(E, volume)
    shaped_onsite = onsite.reshape(2, E, E, volume)
    output = np.empty(E * volume // 2, dtype=full.dtype)
    cursor = 0
    for row in range(E):
        for x in range(x_extent):
            for y in range(y_extent):
                for z in range(z_extent):
                    for t2 in range(t_extent // 2):
                        t = 2 * t2 + (((x + y + z) & 1) ^ parity)
                        site = ((x * y_extent + y) * z_extent + z) * t_extent + t
                        output[cursor] = np.dot(
                            shaped_onsite[int(inverse), row, :, site],
                            shaped_full[:, site])
                        cursor += 1
    return output


def join_reference(target: np.ndarray, other: np.ndarray,
                   dims: tuple[int, int, int, int], E: int,
                   parity: int) -> np.ndarray:
    x_extent, y_extent, z_extent, t_extent = dims
    volume = int(np.prod(dims))
    full = np.empty(E * volume, dtype=target.dtype)
    for component in range(E):
        for x in range(x_extent):
            for y in range(y_extent):
                for z in range(z_extent):
                    for t in range(t_extent):
                        site = ((x * y_extent + y) * z_extent + z) * t_extent + t
                        half = compact_half_index(dims, x, y, z, t)
                        source = target if ((x + y + z + t) & 1) == parity else other
                        full[component * volume + site] = source[
                            component * (volume // 2) + half]
    return full


def params_for(local_dims: tuple[int, int, int, int],
               grid: tuple[int, int, int, int], dtype: np.dtype,
               level_geometry: tuple[int, int, int, int, int] | None = None,
               num_levels: int = 1) -> tuple[np.ndarray, np.ndarray]:
    params = np.zeros(58, dtype=np.int32)
    params[0:4] = local_dims
    params[4] = int(np.prod(local_dims))
    params[5:9] = grid
    params[9] = 1
    params[10] = MPI.COMM_WORLD.rank
    params[11] = MPI.COMM_WORLD.size
    params[13] = 300
    params[14] = 2 if dtype == np.dtype(np.complex64) else 3
    params[15] = 0
    params[16] = 1
    params[17] = num_levels
    params[56] = 2
    argv = np.zeros(8, dtype=np.float32 if dtype == np.dtype(np.complex64)
                    else np.float64)
    argv[0] = 0.17
    argv[1] = 1.0e-7
    argv[3] = 1.0e-7
    if level_geometry is not None:
        params[19:24] = level_geometry[:5]
        params[24] = 150
        params[25] = params[14]
        params[26] = 3
    return params, argv


def relative(actual: np.ndarray, expected: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(expected)), 1.0)
    return float(np.linalg.norm(actual - expected) / denominator)


def run_primitive_test(library: ctypes.CDLL, dtype: np.dtype,
                       tolerance: float) -> tuple[float, float, float, float]:
    comm = MPI.COMM_WORLD
    rank = comm.rank
    local_dims = (2, 2, 2, 2)
    global_dims = (2 * comm.size, 2, 2, 2)
    E = 2
    volume = int(np.prod(global_dims))
    rng = np.random.default_rng(20261020 + comm.size)
    raw_links = (
        rng.normal(size=2 * 4 * E * E * volume) +
        1j * rng.normal(size=2 * 4 * E * E * volume)
    ).astype(dtype)
    precond_links = (
        rng.normal(size=2 * 4 * E * E * volume) +
        1j * rng.normal(size=2 * 4 * E * E * volume)
    ).astype(dtype)
    onsite = (
        rng.normal(size=2 * E * E * volume) +
        1j * rng.normal(size=2 * E * E * volume)
    ).astype(dtype)
    onsite[1 * E * E * volume:(1 + 1) * E * E * volume] = 0
    onsite[1 * E * E * volume:
           (1 + 1) * E * E * volume].reshape(E, E, volume)[
               np.arange(E), np.arange(E), :] = 1

    local_raw = local_slice(raw_links, 2 * 4 * E * E, global_dims, rank)
    local_pre = local_slice(precond_links, 2 * 4 * E * E, global_dims, rank)
    local_onsite = local_slice(onsite, 2 * E * E, global_dims, rank)

    full_global = (
        rng.normal(size=E * volume) + 1j * rng.normal(size=E * volume)
    ).astype(dtype)
    local_full = local_slice(full_global, E, global_dims, rank)
    local_out = np.empty_like(local_full)
    compact_size = E * int(np.prod(local_dims)) // 2
    compact_out = np.empty(compact_size, dtype=dtype)
    compact_scratch = np.empty(compact_size, dtype=dtype)

    params, argv = params_for(local_dims, (comm.size, 1, 1, 1), dtype)
    set_ptrs = np.zeros(100, dtype=np.int64)
    library.applyInitQcu(pointer(set_ptrs), pointer(params), pointer(argv))
    errors = []
    try:
        for onsite_index in (0, -1):
            status = library.applyMultigridStrictCoarseQcu(
                pointer(local_out), pointer(local_full), pointer(local_raw),
                pointer(local_onsite), pointer(set_ptrs), pointer(params),
                E, *local_dims, onsite_index)
            if status != 0:
                raise AssertionError("strict coarse returned nonzero")
            expected = coarse_reference(
                full_global, raw_links, onsite, global_dims, E, onsite_index)
            errors.append(relative(
                local_out, local_slice(expected, E, global_dims, rank)))

        for parity in (0, 1):
            compact_global = full_global.reshape(E, -1)[
                :, compact_indices(global_dims, parity)].reshape(-1)
            local_compact = local_compact_slice(
                compact_global, E, global_dims, local_dims, rank, parity)
            status = library.applyMultigridStrictMatPCQcu(
                pointer(compact_out), pointer(local_compact),
                pointer(local_pre), pointer(compact_scratch),
                pointer(set_ptrs), pointer(params), E, *local_dims, parity)
            if status != 0:
                raise AssertionError("strict MATPC returned nonzero")
            first = compact_holding_reference(
                compact_global, precond_links, global_dims, E, parity, 1 - parity)
            expected = compact_holding_reference(
                first, precond_links, global_dims, E, 1 - parity, parity,
                base=compact_global)
            errors.append(relative(
                compact_out,
                local_compact_slice(expected, E, global_dims, local_dims,
                                    rank, parity)))

            status = library.applyMultigridStrictPrepareQcu(
                pointer(compact_out), pointer(local_full), pointer(local_pre),
                pointer(local_onsite), pointer(compact_scratch),
                pointer(set_ptrs), pointer(params), E, *local_dims, parity)
            if status != 0:
                raise AssertionError("strict prepare returned nonzero")
            target = onsite_parity_reference(
                full_global, onsite, global_dims, E, parity, True)
            other = onsite_parity_reference(
                full_global, onsite, global_dims, E, 1 - parity, True)
            expected = target - compact_holding_reference(
                other, precond_links, global_dims, E, 1 - parity, parity)
            errors.append(relative(
                compact_out,
                local_compact_slice(expected, E, global_dims, local_dims,
                                    rank, parity)))

            reconstructed = np.empty_like(local_full)
            status = library.applyMultigridStrictReconstructQcu(
                pointer(reconstructed), pointer(local_full),
                pointer(local_compact), pointer(local_pre),
                pointer(local_onsite), pointer(compact_scratch),
                pointer(set_ptrs), pointer(params), E, *local_dims, parity)
            if status != 0:
                raise AssertionError("strict reconstruct returned nonzero")
            other_solution = compact_holding_reference(
                compact_global, precond_links, global_dims, E, parity,
                1 - parity, base=other)
            expected = join_reference(
                compact_global, other_solution, global_dims, E, parity)
            errors.append(relative(
                reconstructed,
                local_slice(expected, E, global_dims, rank)))
    finally:
        library.applyEndQcu(pointer(set_ptrs), pointer(params))

    if comm.rank == 0 and any(error >= tolerance for error in errors):
        raise AssertionError(
            f"strict primitive mismatch dtype={dtype}: "
            + ", ".join(f"{error:.3e}" for error in errors))
    return tuple(errors)  # type: ignore[return-value]


def run_fine_matpc_test(library: ctypes.CDLL, dtype: np.dtype,
                        tolerance: float) -> tuple[float, float]:
    comm = MPI.COMM_WORLD
    rank = comm.rank
    local_dims = (2, 2, 2, 2)
    global_dims = (2 * comm.size, 2, 2, 2)
    half_volume = int(np.prod(global_dims)) // 2
    rng = np.random.default_rng(20261021 + comm.size)
    gauge = (
        rng.normal(size=2 * 3 * 3 * 4 * half_volume) +
        1j * rng.normal(size=2 * 3 * 3 * 4 * half_volume)
    ).astype(dtype) * np.asarray(0.04, dtype=dtype)
    gauge = gauge.reshape(2, 3, 3, 4, *global_dims[:3],
                          global_dims[3] // 2)
    clover_inverse = np.empty(
        (2, 12, 12, *global_dims[:3], global_dims[3] // 2),
        dtype=dtype)
    identity = np.eye(12, dtype=dtype)
    for parity in range(2):
        for x in range(global_dims[0]):
            for y in range(global_dims[1]):
                for z in range(global_dims[2]):
                    for t2 in range(global_dims[3] // 2):
                        matrix = identity + np.asarray(0.03, dtype=dtype) * (
                            rng.normal(size=(12, 12)) +
                            1j * rng.normal(size=(12, 12))).astype(dtype)
                        clover_inverse[parity, :, :, x, y, z, t2] = (
                            np.linalg.inv(matrix))
    source = (
        rng.normal(size=12 * half_volume) +
        1j * rng.normal(size=12 * half_volume)
    ).astype(dtype).reshape(12, *global_dims[:3],
                            global_dims[3] // 2)

    gauge_dims = (global_dims[0], global_dims[1], global_dims[2],
                  global_dims[3] // 2)
    local_gauge = local_slice(gauge.reshape(-1), 2 * 3 * 3 * 4,
                              gauge_dims, rank)
    local_inverse = [
        local_slice(clover_inverse[parity].reshape(-1), 12 * 12,
                    gauge_dims, rank)
        for parity in range(2)
    ]
    params, argv = params_for(
        local_dims, (comm.size, 1, 1, 1), dtype)
    kappa = 1.0 / (2.0 * float(argv[0]) + 8.0)
    set_ptrs = np.zeros(100, dtype=np.int64)
    library.applyInitQcu(pointer(set_ptrs), pointer(params), pointer(argv))
    errors = []
    try:
        for parity in range(2):
            local_source = local_compact_slice(
                source.reshape(-1), 12, global_dims, local_dims, rank, parity)
            local_output = np.empty_like(local_source)
            status = library.applyMultigridStrictFineMatPCQcu(
                pointer(local_output), pointer(local_source),
                pointer(local_gauge), pointer(local_inverse[0]),
                pointer(local_inverse[1]), pointer(local_inverse[0]),
                pointer(local_inverse[1]), pointer(set_ptrs), pointer(params),
                parity)
            if status != 0:
                raise AssertionError("strict fine MATPC returned nonzero")
            expected = fine_matpc_reference(
                source.reshape(-1),
                (clover_inverse[0].reshape(-1).reshape(12, 12, -1),
                 clover_inverse[1].reshape(-1).reshape(12, 12, -1)),
                gauge, global_dims, kappa, parity)
            errors.append(relative(
                local_output,
                local_compact_slice(expected, 12, global_dims, local_dims,
                                    rank, parity)))

    finally:
        library.applyEndQcu(pointer(set_ptrs), pointer(params))
    if comm.rank == 0 and any(error >= tolerance for error in errors):
        raise AssertionError(
            f"strict fine MATPC mismatch dtype={dtype}: "
            + ", ".join(f"{error:.3e}" for error in errors))
    return tuple(errors)  # type: ignore[return-value]


def run_vcycle_test(library: ctypes.CDLL, dtype: np.dtype,
                    tolerance: float) -> float:
    comm = MPI.COMM_WORLD
    local_dims = (2, 2, 2, 2)
    global_dims = (2 * comm.size, 2, 2, 2)
    E = 2
    volume = int(np.prod(global_dims))
    rng = np.random.default_rng(4242 + comm.size)
    links_global = (
        rng.normal(size=2 * 4 * E * E * volume) +
        1j * rng.normal(size=2 * 4 * E * E * volume)
    ).astype(dtype) * np.asarray(0.02, dtype=dtype)
    onsite_global = np.zeros(2 * E * E * volume, dtype=dtype)
    for inverse in range(2):
        matrix = onsite_global[inverse * E * E * volume:
                               (inverse + 1) * E * E * volume].reshape(
                                   E, E, volume)
        matrix[np.arange(E), np.arange(E), :] = 1
    rhs_global = (
        rng.normal(size=E * volume) + 1j * rng.normal(size=E * volume)
    ).astype(dtype)
    local_links = local_slice(
        links_global, 2 * 4 * E * E, global_dims, comm.rank)
    local_onsite = local_slice(
        onsite_global, 2 * E * E, global_dims, comm.rank)
    local_rhs = local_slice(rhs_global, E, global_dims, comm.rank)
    local_out = np.zeros_like(local_rhs)
    params, argv = params_for(
        local_dims, (comm.size, 1, 1, 1), dtype, level_geometry=(E, *local_dims),
        num_levels=2)
    argv[3] = (1.0e-6 if dtype == np.dtype(np.complex64)
               else 1.0e-12)
    set_ptrs = np.zeros(100, dtype=np.int64)
    set_ptrs[60 + 2] = pointer(local_links)
    set_ptrs[60 + 3] = pointer(local_onsite)
    library.applyInitQcu(pointer(set_ptrs), pointer(params), pointer(argv))
    allocated = ctypes.c_ulonglong(0)
    try:
        status = library.applyMultigridStrictInitQcu(
            pointer(set_ptrs), pointer(params), 1, ctypes.byref(allocated))
        if status != 0 or allocated.value == 0:
            raise AssertionError("strict hierarchy init failed")
        status = library.applyMultigridStrictVCycleQcu(
            pointer(local_out), pointer(local_rhs), pointer(set_ptrs),
            pointer(params), 1, ctypes.byref(allocated))
        if status != 0:
            raise AssertionError("strict V-cycle returned nonzero")
        repeated = np.zeros_like(local_out)
        status = library.applyMultigridStrictVCycleQcu(
            pointer(repeated), pointer(local_rhs), pointer(set_ptrs),
            pointer(params), 1, ctypes.byref(allocated))
        if status != 0 or relative(local_out, repeated) >= tolerance:
            raise AssertionError("persistent strict V-cycle is not deterministic")
        status = library.applyMultigridStrictEndQcu(
            pointer(set_ptrs), pointer(params))
        if status != 0 or set_ptrs[80] != 0:
            raise AssertionError("strict hierarchy end failed")
    finally:
        library.applyEndQcu(pointer(set_ptrs), pointer(params))

    global_out = globalize_local(
        local_out, E, local_dims, global_dims, comm.rank)
    applied = coarse_reference(
        global_out, links_global, onsite_global, global_dims, E, 0)
    residual = relative(applied, rhs_global)
    if comm.rank == 0 and residual >= tolerance:
        raise AssertionError(f"strict V-cycle residual {residual:.3e}")
    return residual


def blocked_index(E: int, e: int, coarse_component: int,
                  fine_component: int, x: int, y: int, z: int, t: int,
                  fine_dims: tuple[int, int, int, int],
                  coarse_dims: tuple[int, int, int, int]) -> int:
    xf, yf, zf, tf = fine_dims
    xc, yc, zc, tc = coarse_dims
    bx, by, bz, bt = xf // xc, yf // yc, zf // zc, tf // tc
    index = coarse_component
    for extent, value in (
        (e, fine_component), (xc, x // bx), (bx, x % bx),
        (yc, y // by), (by, y % by), (zc, z // bz), (bz, z % bz),
        (tc, t // bt), (bt, t % bt),
    ):
        index = index * extent + value
    return index


def restrict_reference(fine: np.ndarray, null: np.ndarray, E: int, e: int,
                       fine_dims: tuple[int, int, int, int],
                       coarse_dims: tuple[int, int, int, int],
                       parity: int) -> np.ndarray:
    xf, yf, zf, tf = fine_dims
    xc, yc, zc, tc = coarse_dims
    bx, by, bz, bt = xf // xc, yf // yc, zf // zc, tf // tc
    fine_half = xf * yf * zf * (tf // 2)
    shaped_fine = fine.reshape(e, fine_half)
    output = np.zeros(E * xc * yc * zc * tc, dtype=fine.dtype)
    for coarse_component in range(E):
        for xc_idx in range(xc):
            for yc_idx in range(yc):
                for zc_idx in range(zc):
                    for tc_idx in range(tc):
                        value = 0.0j
                        for dx in range(bx):
                            x = xc_idx * bx + dx
                            for dy in range(by):
                                y = yc_idx * by + dy
                                for dz in range(bz):
                                    z = zc_idx * bz + dz
                                    for dt in range(bt):
                                        t = tc_idx * bt + dt
                                        if ((x + y + z + t) & 1) != parity:
                                            continue
                                        for fine_component in range(e):
                                            value += np.conj(null[
                                                blocked_index(
                                                    E, e, coarse_component,
                                                    fine_component, x, y, z, t,
                                                    fine_dims, coarse_dims)]) * (
                                                        shaped_fine[
                                                            fine_component,
                                                            ((x * yf + y) * zf + z) *
                                                            (tf // 2) + t // 2])
                        coarse_site = (
                            (coarse_component * xc + xc_idx) * yc + yc_idx
                        ) * zc + zc_idx
                        coarse_site = coarse_site * tc + tc_idx
                        output[coarse_site] = value
    return output


def prolong_reference(coarse: np.ndarray, null: np.ndarray, E: int, e: int,
                      fine_dims: tuple[int, int, int, int],
                      coarse_dims: tuple[int, int, int, int],
                      parity: int) -> np.ndarray:
    xf, yf, zf, tf = fine_dims
    xc, yc, zc, tc = coarse_dims
    output = np.empty(e * xf * yf * zf * (tf // 2), dtype=coarse.dtype)
    cursor = 0
    for fine_component in range(e):
        for x in range(xf):
            for y in range(yf):
                for z in range(zf):
                    for t2 in range(tf // 2):
                        t = 2 * t2 + (((x + y + z) & 1) ^ parity)
                        value = 0.0j
                        for coarse_component in range(E):
                            value += null[
                                blocked_index(
                                    E, e, coarse_component, fine_component,
                                    x, y, z, t, fine_dims, coarse_dims)] * (
                                        coarse.reshape(E, -1)[
                                            coarse_component,
                                            (((x * xc // xf) * yc +
                                              y * yc // yf) * zc +
                                             z * zc // zf) * tc +
                                            t * tc // tf])
                        output[cursor] = value
                        cursor += 1
    return output


def run_transfer_test(library: ctypes.CDLL, dtype: np.dtype,
                      tolerance: float) -> tuple[float, float]:
    comm = MPI.COMM_WORLD
    fine_dims = (4, 2, 2, 2)
    coarse_dims = (2, 2, 2, 2)
    E, e = 2, 4
    rng = np.random.default_rng(731 + comm.rank)
    null = (
        rng.normal(size=E * e * int(np.prod(fine_dims))) +
        1j * rng.normal(size=E * e * int(np.prod(fine_dims)))
    ).astype(dtype)
    coarse = (
        rng.normal(size=E * int(np.prod(coarse_dims))) +
        1j * rng.normal(size=E * int(np.prod(coarse_dims)))
    ).astype(dtype)
    params, argv = params_for(fine_dims, (comm.size, 1, 1, 1), dtype)
    set_ptrs = np.zeros(100, dtype=np.int64)
    library.applyInitQcu(pointer(set_ptrs), pointer(params), pointer(argv))
    errors = []
    try:
        for parity in (0, 1):
            fine = (
                rng.normal(size=e * int(np.prod(fine_dims)) // 2) +
                1j * rng.normal(size=e * int(np.prod(fine_dims)) // 2)
            ).astype(dtype)
            coarse_out = np.empty_like(coarse)
            status = library.applyMultigridStrictRestrictQcu(
                pointer(coarse_out), pointer(fine), pointer(null),
                pointer(set_ptrs), pointer(params), E, e, *fine_dims,
                *coarse_dims, parity)
            if status != 0:
                raise AssertionError("strict restrict returned nonzero")
            expected_coarse = restrict_reference(
                fine, null, E, e, fine_dims, coarse_dims, parity)
            errors.append(relative(coarse_out, expected_coarse))

            fine_out = np.empty_like(fine)
            status = library.applyMultigridStrictProLongQcu(
                pointer(fine_out), pointer(coarse), pointer(null),
                pointer(set_ptrs), pointer(params), E, e, *fine_dims,
                *coarse_dims, parity)
            if status != 0:
                raise AssertionError("strict prolong returned nonzero")
            expected_fine = prolong_reference(
                coarse, null, E, e, fine_dims, coarse_dims, parity)
            errors.append(relative(fine_out, expected_fine))
    finally:
        library.applyEndQcu(pointer(set_ptrs), pointer(params))
    if comm.rank == 0 and any(error >= tolerance for error in errors):
        raise AssertionError(
            f"strict transfer mismatch dtype={dtype}: "
            + ", ".join(f"{error:.3e}" for error in errors))
    return errors[0], errors[1]


def run_fgmres_test(library: ctypes.CDLL, dtype: np.dtype) -> tuple[int, float]:
    comm = MPI.COMM_WORLD
    fine_dims = (4, 2, 2, 4)
    coarse_dims = (2, 2, 2, 2)
    fine_E, coarse_E = 12, 2
    fine_half_volume = int(np.prod(fine_dims)) // 2
    coarse_volume = int(np.prod(coarse_dims))

    gauge = np.zeros(
        (2, 3, 3, 4, *fine_dims[:3], fine_dims[3] // 2), dtype=dtype)
    for parity in range(2):
        for direction in range(4):
            for site in range(
                    fine_dims[0] * fine_dims[1] * fine_dims[2] *
                    (fine_dims[3] // 2)):
                link = np.eye(3, dtype=dtype)
                coordinates = []
                remaining = site
                t2 = remaining % (fine_dims[3] // 2)
                remaining //= fine_dims[3] // 2
                z = remaining % fine_dims[2]
                remaining //= fine_dims[2]
                y = remaining % fine_dims[1]
                x = remaining // fine_dims[1]
                coordinates = (x, y, z, t2)
                gauge[(parity, slice(None), slice(None), direction,
                       *coordinates)] = link

    clover = np.zeros(2 * fine_E * fine_E * fine_half_volume, dtype=dtype)
    clover_view = clover.reshape(2, fine_E, fine_E, fine_half_volume)
    clover_view[:, np.arange(fine_E), np.arange(fine_E), :] = 1
    clover_inv = clover.copy()
    clover_even = np.ascontiguousarray(clover_view[0].reshape(-1))
    clover_odd = np.ascontiguousarray(clover_view[1].reshape(-1))
    clover_even_inv = np.ascontiguousarray(clover_view[0].reshape(-1))
    clover_odd_inv = np.ascontiguousarray(clover_view[1].reshape(-1))

    rng = np.random.default_rng(991 + comm.rank)
    fine_null = (
        rng.normal(size=coarse_E * fine_E * int(np.prod(fine_dims))) +
        1j * rng.normal(size=coarse_E * fine_E * int(np.prod(fine_dims)))
    ).astype(dtype)
    coarse_links = np.zeros(
        2 * 4 * coarse_E * coarse_E * coarse_volume, dtype=dtype)
    coarse_onsite = np.zeros(
        2 * coarse_E * coarse_E * coarse_volume, dtype=dtype)
    coarse_onsite_view = coarse_onsite.reshape(
        2, coarse_E, coarse_E, coarse_volume)
    coarse_onsite_view[:, np.arange(coarse_E), np.arange(coarse_E), :] = 1
    full_rhs = (
        rng.normal(size=2 * fine_E * fine_half_volume) +
        1j * rng.normal(size=2 * fine_E * fine_half_volume)
    ).astype(dtype)
    full_out = np.zeros_like(full_rhs)

    params, argv = params_for(
        fine_dims, (comm.size, 1, 1, 1), dtype,
        level_geometry=(coarse_E, *coarse_dims), num_levels=2)
    params[57] = 0
    argv[0] = 1.0e6
    argv[1] = 1.0e-6
    argv[3] = 1.0e-6
    set_ptrs = np.zeros(100, dtype=np.int64)
    set_ptrs[60] = pointer(fine_null)
    set_ptrs[62] = pointer(coarse_links)
    set_ptrs[63] = pointer(coarse_onsite)
    library.applyInitQcu(pointer(set_ptrs), pointer(params), pointer(argv))
    allocated = ctypes.c_ulonglong(0)
    iterations = ctypes.c_int(0)
    converged = ctypes.c_int(0)
    residual = ctypes.c_double(0)
    restart = 6
    workspace_bytes = (
        (2 * restart + 5) * fine_E * fine_half_volume +
        2 * coarse_E * coarse_volume) * np.dtype(dtype).itemsize

    def call_fgmres(budget: int) -> int:
        return library.applyMultigridStrictFgmresQcu(
            pointer(full_out), pointer(full_rhs), pointer(gauge),
            pointer(clover_even), pointer(clover_odd),
            pointer(clover_even_inv), pointer(clover_odd_inv),
            pointer(fine_null), pointer(set_ptrs), pointer(params),
            fine_E, *fine_dims, coarse_E, *coarse_dims,
            np.dtype(dtype).itemsize, restart, 40, 1.0e-6, 1, 1,
            budget, ctypes.byref(iterations), ctypes.byref(converged),
            ctypes.byref(residual), ctypes.byref(allocated))
    try:
        if library.applyMultigridStrictInitQcu(
                pointer(set_ptrs), pointer(params), 1,
                ctypes.byref(allocated)) != 0:
            raise AssertionError("strict FGMRES hierarchy init failed")
        if call_fgmres(workspace_bytes - 1) == 0:
            raise AssertionError("strict FGMRES accepted an undersized budget")
        status = call_fgmres(workspace_bytes)
        if status != 0:
            raise AssertionError("strict FGMRES returned nonzero")
        if allocated.value != workspace_bytes:
            raise AssertionError(
                "strict FGMRES workspace ledger mismatch: "
                f"{allocated.value} != {workspace_bytes}")
        cold_output = full_out.copy()
        cold_iterations = iterations.value
        cold_residual = residual.value
        params[57] = 1
        status = call_fgmres(workspace_bytes)
        if status != 0 or converged.value != 1:
            raise AssertionError("strict FGMRES warm start failed")
        if relative(full_out, cold_output) >= 1.0e-5:
            raise AssertionError("strict FGMRES warm start changed the solution")
        if library.applyMultigridStrictEndQcu(
                pointer(set_ptrs), pointer(params)) != 0:
            raise AssertionError("strict FGMRES hierarchy end failed")
    finally:
        library.applyEndQcu(pointer(set_ptrs), pointer(params))
    if comm.rank == 0 and (
            converged.value != 1 or not np.isfinite(residual.value)):
        raise AssertionError(
            f"strict FGMRES did not converge: iterations={iterations.value} "
            f"residual={residual.value:.3e}")
    return cold_iterations, cold_residual


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    args = parser.parse_args()
    library = bind_library(args.library)
    c64_errors = run_primitive_test(
        library, np.dtype(np.complex64), 3.0e-4)
    c128_errors = run_primitive_test(
        library, np.dtype(np.complex128), 2.0e-11)
    c64_fine = run_fine_matpc_test(
        library, np.dtype(np.complex64), 3.0e-4)
    c128_fine = run_fine_matpc_test(
        library, np.dtype(np.complex128), 2.0e-11)
    c64_transfer = run_transfer_test(
        library, np.dtype(np.complex64), 3.0e-4)
    c128_transfer = run_transfer_test(
        library, np.dtype(np.complex128), 2.0e-11)
    c64_fgmres = run_fgmres_test(library, np.dtype(np.complex64))
    c128_fgmres = run_fgmres_test(library, np.dtype(np.complex128))
    c64_vcycle = run_vcycle_test(
        library, np.dtype(np.complex64), 3.0e-4)
    c128_vcycle = run_vcycle_test(
        library, np.dtype(np.complex128), 2.0e-11)
    if MPI.COMM_WORLD.rank == 0:
        print(
            f"status=PASS ranks={MPI.COMM_WORLD.size} "
            f"c64_max={max(c64_errors):.3e} "
            f"c128_max={max(c128_errors):.3e} "
            f"fine_c64={max(c64_fine):.3e} "
            f"fine_c128={max(c128_fine):.3e} "
            f"transfer_c64={max(c64_transfer):.3e} "
            f"transfer_c128={max(c128_transfer):.3e} "
            f"fgmres_c64={c64_fgmres[0]}/{c64_fgmres[1]:.3e} "
            f"fgmres_c128={c128_fgmres[0]}/{c128_fgmres[1]:.3e} "
            f"vcycle_c64={c64_vcycle:.3e} vcycle_c128={c128_vcycle:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
