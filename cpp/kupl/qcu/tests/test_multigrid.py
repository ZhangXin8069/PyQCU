#!/usr/bin/env python3
"""Differential MPI/OpenMP smoke tests for KUPL multigrid primitives."""

from __future__ import annotations

import argparse
import ctypes
from pathlib import Path

import numpy as np
from mpi4py import MPI


LL = ctypes.c_longlong


def bind_library(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL)
    signatures = {
        "applyInitQcu": 3,
        "applyEndQcu": 2,
        "applyGaussGaugeQcu": 3,
        "applyMultigridRestrictQcu": 5,
        "applyMultigridProLongQcu": 5,
        "applyMultigridCoarseDslashQcu": 6,
        "applyMultigridCoarseDslashWideQcu": 7,
        "applyCloversQcu": 5,
        "applyCloverBistabCgDslashQcu": 9,
    }
    for name, count in signatures.items():
        function = getattr(library, name)
        function.argtypes = (LL,) * count
        function.restype = None
    orthogonalize = library.applyMultigridLocalOrthogonalizeQcu
    orthogonalize.argtypes = (LL, LL) + (ctypes.c_int,) * 11
    orthogonalize.restype = ctypes.c_int
    build_null = library.applyMultigridBuildNullVectorsQcu
    build_null.argtypes = (
        (LL,) * 8 + (ctypes.c_int,) * 11 +
        (ctypes.c_ulonglong, ctypes.c_double))
    build_null.restype = ctypes.c_int
    build_stencil = library.applyMultigridBuildStencilLocalQcu
    build_stencil.argtypes = (LL,) * 11 + (ctypes.c_int,) * 12
    build_stencil.restype = ctypes.c_int
    return library


def pointer(array: np.ndarray) -> int:
    return int(array.ctypes.data)


def null_index(E_idx: int, e_idx: int, x: int, y: int, z: int, t: int,
               E: int, e: int, fine: tuple[int, int, int, int],
               coarse: tuple[int, int, int, int]) -> int:
    Xf, Yf, Zf, Tf = fine
    Xc, Yc, Zc, Tc = coarse
    bx, by, bz, bt = (Xf // Xc, Yf // Yc, Zf // Zc, Tf // Tc)
    xc, yc, zc, tc = x // bx, y // by, z // bz, t // bt
    dx, dy, dz, dt = x % bx, y % by, z % bz, t % bt
    index = E_idx
    for extent, value in (
        (e, e_idx), (Xc, xc), (bx, dx), (Yc, yc), (by, dy),
        (Zc, zc), (bz, dz), (Tc, tc), (bt, dt),
    ):
        index = index * extent + value
    return index


def expected_restrict(fine: np.ndarray, null: np.ndarray, E: int, e: int,
                      fine_dims: tuple[int, int, int, int],
                      coarse_dims: tuple[int, int, int, int]) -> np.ndarray:
    Xf, Yf, Zf, Tf = fine_dims
    Xc, Yc, Zc, Tc = coarse_dims
    bx, by, bz, bt = (Xf // Xc, Yf // Yc, Zf // Zc, Tf // Tc)
    output = np.zeros(E * Xc * Yc * Zc * Tc, dtype=fine.dtype)
    fine_volume = Xf * Yf * Zf * Tf
    coarse_volume = Xc * Yc * Zc * Tc
    for E_idx in range(E):
        for xc in range(Xc):
            for yc in range(Yc):
                for zc in range(Zc):
                    for tc in range(Tc):
                        value = 0.0j
                        for dx in range(bx):
                            for dy in range(by):
                                for dz in range(bz):
                                    for dt in range(bt):
                                        x = xc * bx + dx
                                        y = yc * by + dy
                                        z = zc * bz + dz
                                        t = tc * bt + dt
                                        site = (
                                            ((x * Yf + y) * Zf + z) * Tf + t)
                                        for e_idx in range(e):
                                            value += np.conj(null[
                                                null_index(
                                                    E_idx, e_idx, x, y, z, t,
                                                    E, e, fine_dims,
                                                    coarse_dims)]) * fine[
                                                        e_idx * fine_volume +
                                                        site]
                        output[
                            E_idx * coarse_volume +
                            ((xc * Yc + yc) * Zc + zc) * Tc + tc] = value
    return output


def expected_prolong(coarse: np.ndarray, null: np.ndarray, E: int, e: int,
                     fine_dims: tuple[int, int, int, int],
                     coarse_dims: tuple[int, int, int, int]) -> np.ndarray:
    Xf, Yf, Zf, Tf = fine_dims
    Xc, Yc, Zc, Tc = coarse_dims
    bx, by, bz, bt = (Xf // Xc, Yf // Yc, Zf // Zc, Tf // Tc)
    fine_volume = Xf * Yf * Zf * Tf
    coarse_volume = Xc * Yc * Zc * Tc
    output = np.zeros(e * fine_volume, dtype=coarse.dtype)
    for e_idx in range(e):
        for x in range(Xf):
            for y in range(Yf):
                for z in range(Zf):
                    for t in range(Tf):
                        site = ((x * Yf + y) * Zf + z) * Tf + t
                        coarse_site = (
                            (((x // bx) * Yc + y // by) * Zc + z // bz) *
                            Tc + t // bt)
                        value = 0.0j
                        for E_idx in range(E):
                            value += null[
                                null_index(
                                    E_idx, e_idx, x, y, z, t, E, e,
                                    fine_dims, coarse_dims)] * coarse[
                                        E_idx * coarse_volume + coarse_site]
                        output[e_idx * fine_volume + site] = value
    return output


def run_gaussian_test(library: ctypes.CDLL, dtype: np.dtype,
                      tolerance: float) -> float:
    comm = MPI.COMM_WORLD
    grid = (comm.size, 1, 1, 1)
    local = (4, 4, 4, 4)
    params = np.zeros(58, dtype=np.int32)
    params[0:4] = local
    params[4] = int(np.prod(local))
    params[5:9] = grid
    params[9] = 0
    params[10] = comm.rank
    params[11] = comm.size
    params[13] = 20
    params[14] = 2 if dtype == np.complex64 else 3
    params[15] = 0
    params[16] = -1
    params[52] = 71
    argv = np.zeros(8, dtype=np.float32 if dtype == np.complex64 else np.float64)
    argv[0] = 0.17
    argv[1] = 1.0e-7
    argv[2] = 0.12
    argv[7] = 0.8
    set_ptrs = np.zeros(100, dtype=np.int64)
    gauge = np.empty((2, 3, 3, 4, *local[:3], local[3] // 2), dtype=dtype)
    library.applyInitQcu(pointer(set_ptrs), pointer(params), pointer(argv))
    try:
        library.applyGaussGaugeQcu(pointer(gauge), pointer(set_ptrs),
                                   pointer(params))
        repeated = np.empty_like(gauge)
        library.applyGaussGaugeQcu(pointer(repeated), pointer(set_ptrs),
                                   pointer(params))
        if not np.array_equal(gauge, repeated):
            raise AssertionError("Gaussian gauge is not deterministic")
    finally:
        library.applyEndQcu(pointer(set_ptrs), pointer(params))

    global_shape = (2, 3, 3, 4, grid[0] * local[0], local[1], local[2],
                    local[3])
    local_global = np.zeros(global_shape, dtype=dtype)
    local_full = np.empty(
        (2, 3, 3, 4, local[0], local[1], local[2], local[3]),
        dtype=dtype)
    for parity in range(2):
        for t2 in range(local[3] // 2):
            for x in range(local[0]):
                for y in range(local[1]):
                    for z in range(local[2]):
                        t = 2 * t2 + (((x + y + z) & 1) ^ parity)
                        local_full[:, :, :, :, x, y, z, t] = gauge[
                            :, :, :, :, x, y, z, t2]
    x0 = comm.rank * local[0]
    local_global[..., x0:x0 + local[0], :, :, :] = local_full
    global_gauge = np.zeros_like(local_global)
    comm.Allreduce(local_global, global_gauge, op=MPI.SUM)
    matrices = global_gauge.transpose(0, 3, 4, 5, 6, 7, 1, 2).reshape(
        -1, 3, 3)
    identity = np.eye(3, dtype=dtype)
    error = np.linalg.norm(
        matrices.conj().transpose(0, 2, 1) @ matrices - identity) / np.linalg.norm(
            identity)
    determinant_error = np.max(np.abs(np.linalg.det(matrices) - 1.0))
    if comm.rank == 0 and (error >= tolerance or determinant_error >= tolerance * 20):
        raise AssertionError(
            f"Gaussian gauge SU(3) failure: unitarity={error:.3e} "
            f"det={determinant_error:.3e}")
    return float(error)


def run_transfer_test(library: ctypes.CDLL) -> tuple[float, float]:
    comm = MPI.COMM_WORLD
    grid = (comm.size, 1, 1, 1)
    physical_dims = (4, 2, 2, 8)
    fine_dims = (4, 2, 2, 4)
    coarse_dims = (2, 2, 2, 2)
    E, e = 3, 12
    fine_volume = int(np.prod(fine_dims))
    coarse_volume = int(np.prod(coarse_dims))
    fine = (
        np.arange(e * fine_volume, dtype=np.float64).reshape(e, fine_volume)
        * (0.03 + 0.02j) + (comm.rank + 1) * (0.01 - 0.04j)
    ).astype(np.complex64).reshape(-1)
    null = (
        np.arange(E * e * fine_volume, dtype=np.float64).reshape(
            E * e * fine_volume) * (0.002 - 0.001j) + 0.05j
    ).astype(np.complex64)
    coarse = np.empty(E * coarse_volume, dtype=np.complex64)
    prolonged = np.empty(e * fine_volume, dtype=np.complex64)

    params = np.zeros(58, dtype=np.int32)
    params[0:4] = physical_dims
    params[4] = int(np.prod(physical_dims))
    params[5:9] = grid
    params[10] = comm.rank
    params[11] = comm.size
    params[13] = 200
    params[14] = 2
    params[15] = 0
    params[16] = 0
    params[19] = E
    params[20:24] = coarse_dims
    argv = np.zeros(8, dtype=np.float32)
    argv[0], argv[1], argv[2], argv[7] = 0.17, 1.0e-7, 0.1, 0.8
    set_ptrs = np.zeros(100, dtype=np.int64)
    library.applyInitQcu(pointer(set_ptrs), pointer(params), pointer(argv))
    try:
        library.applyMultigridRestrictQcu(
            pointer(coarse), pointer(fine), pointer(null), pointer(set_ptrs),
            pointer(params))
        library.applyMultigridProLongQcu(
            pointer(prolonged), pointer(coarse), pointer(null),
            pointer(set_ptrs), pointer(params))
    finally:
        library.applyEndQcu(pointer(set_ptrs), pointer(params))

    expected_coarse = expected_restrict(
        fine, null, E, e, fine_dims, coarse_dims)
    expected_fine = expected_prolong(
        expected_coarse, null, E, e, fine_dims, coarse_dims)
    restrict_error = float(
        np.linalg.norm(coarse - expected_coarse) /
        max(np.linalg.norm(expected_coarse), 1.0))
    prolong_error = float(
        np.linalg.norm(prolonged - expected_fine) /
        max(np.linalg.norm(expected_fine), 1.0))
    if restrict_error >= 2.0e-5 or prolong_error >= 2.0e-5:
        raise AssertionError(
            f"MG transfer mismatch: restrict={restrict_error:.3e} "
            f"prolong={prolong_error:.3e}")
    return restrict_error, prolong_error


def globalize(local: np.ndarray, local_dims: tuple[int, int, int, int],
              global_dims: tuple[int, int, int, int], components: int,
              comm: MPI.Comm) -> np.ndarray:
    local4 = local.reshape(components, *local_dims)
    x0 = comm.rank * local_dims[0]
    padded = np.zeros((components, *global_dims), dtype=local.dtype)
    padded[:, x0:x0 + local_dims[0], :, :, :] = local4
    result = np.zeros_like(padded)
    comm.Allreduce(padded, result, op=MPI.SUM)
    return result


def run_coarse_dslash_test(library: ctypes.CDLL) -> tuple[float, float]:
    comm = MPI.COMM_WORLD
    grid = (comm.size, 1, 1, 1)
    local_dims = (2, 4, 2, 4)
    global_dims = (
        local_dims[0] * grid[0], local_dims[1], local_dims[2], local_dims[3])
    E = 2
    volume = int(np.prod(local_dims))
    global_volume = int(np.prod(global_dims))
    generator = np.random.default_rng(91 + comm.rank)
    source = (
        generator.normal(size=E * volume) +
        1j * generator.normal(size=E * volume)
    ).astype(np.complex64)
    sitting = (
        generator.normal(size=E * E * volume) +
        1j * generator.normal(size=E * E * volume)
    ).astype(np.complex64)
    hopping = (
        generator.normal(size=2 * 4 * E * E * volume) +
        1j * generator.normal(size=2 * 4 * E * E * volume)
    ).astype(np.complex64)
    hop_diag = (
        generator.normal(size=2 * 2 * 6 * E * E * volume) +
        1j * generator.normal(size=2 * 2 * 6 * E * E * volume)
    ).astype(np.complex64)

    output = np.empty_like(source)
    output_wide = np.empty_like(source)
    params = np.zeros(58, dtype=np.int32)
    params[0:4] = local_dims
    params[4] = volume
    params[5:9] = grid
    params[10] = comm.rank
    params[11] = comm.size
    params[13] = 200
    params[14] = 2
    params[15] = 0
    params[16] = 0
    params[19] = E
    params[20:24] = local_dims
    argv = np.zeros(8, dtype=np.float32)
    argv[0], argv[1], argv[2], argv[7] = 0.17, 1.0e-7, 0.1, 0.8
    set_ptrs = np.zeros(100, dtype=np.int64)
    library.applyInitQcu(pointer(set_ptrs), pointer(params), pointer(argv))
    try:
        library.applyMultigridCoarseDslashQcu(
            pointer(output), pointer(source), pointer(hopping),
            pointer(sitting), pointer(set_ptrs), pointer(params))
        library.applyMultigridCoarseDslashWideQcu(
            pointer(output_wide), pointer(source), pointer(sitting),
            pointer(hopping), pointer(hop_diag), pointer(set_ptrs),
            pointer(params))
    finally:
        library.applyEndQcu(pointer(set_ptrs), pointer(params))

    global_source = globalize(source, local_dims, global_dims, E, comm)
    global_output = globalize(output, local_dims, global_dims, E, comm)
    global_output_wide = globalize(
        output_wide, local_dims, global_dims, E, comm)
    expected = np.zeros_like(global_source)
    expected_wide = np.zeros_like(global_source)
    local_x0 = comm.rank * local_dims[0]
    pair_first = (0, 0, 0, 1, 1, 2)
    pair_second = (1, 2, 3, 2, 3, 3)
    for E_out in range(E):
        for x in range(local_x0, local_x0 + local_dims[0]):
            for y in range(local_dims[1]):
                for z in range(local_dims[2]):
                    for t in range(local_dims[3]):
                        site = (
                            ((x * local_dims[1] + y) * local_dims[2] + z) *
                            local_dims[3] + t)
                        local_site = (
                            (((x - local_x0) * local_dims[1] + y) *
                             local_dims[2] + z) * local_dims[3] + t)
                        narrow = 0.0j
                        wide = 0.0j
                        for e_idx in range(E):
                            coefficient = sitting[
                                (E_out * E + e_idx) * volume + local_site]
                            value = global_source[
                                e_idx, x, y, z, t]
                            narrow += coefficient * value
                            wide += coefficient * value
                        for dim in range(4):
                            for sign_idx, sign in enumerate((1, -1)):
                                neighbor = [x, y, z, t]
                                neighbor[dim] = (
                                    neighbor[dim] + sign) % global_dims[dim]
                                for e_idx in range(E):
                                    coefficient = hopping[
                                        ((sign_idx * 4 + dim) * E * E +
                                         E_out * E + e_idx) * volume +
                                        local_site]
                                    value = global_source[
                                        (e_idx, *neighbor)]
                                    narrow += coefficient * value
                                    wide += coefficient * value
                        for pair, (first, second) in enumerate(
                                zip(pair_first, pair_second)):
                            for first_sign_idx, first_sign in enumerate((1, -1)):
                                for second_sign_idx, second_sign in enumerate(
                                        (1, -1)):
                                    neighbor = [x, y, z, t]
                                    neighbor[first] = (
                                        neighbor[first] + first_sign
                                    ) % global_dims[first]
                                    neighbor[second] = (
                                        neighbor[second] + second_sign
                                    ) % global_dims[second]
                                    for e_idx in range(E):
                                        coefficient = hop_diag[
                                            ((first_sign_idx * 2 +
                                              second_sign_idx) * 6 + pair) *
                                            E * E * volume +
                                            (E_out * E + e_idx) * volume +
                                            local_site]
                                        wide += coefficient * global_source[
                                            (e_idx, *neighbor)]
                        expected[E_out, x, y, z, t] = narrow
                        expected_wide[E_out, x, y, z, t] = wide
    expected_all = np.zeros_like(expected)
    expected_wide_all = np.zeros_like(expected_wide)
    comm.Allreduce(expected, expected_all, op=MPI.SUM)
    comm.Allreduce(expected_wide, expected_wide_all, op=MPI.SUM)
    narrow_error = float(
        np.linalg.norm(global_output - expected_all) /
        max(np.linalg.norm(expected_all), 1.0))
    wide_error = float(
        np.linalg.norm(global_output_wide - expected_wide_all) /
        max(np.linalg.norm(expected_wide_all), 1.0))
    if narrow_error >= 2.0e-5 or wide_error >= 2.0e-5:
        narrow_difference = np.abs(global_output - expected_all)
        wide_difference = np.abs(global_output_wide - expected_wide_all)
        narrow_worst = np.unravel_index(
            int(np.argmax(narrow_difference)), narrow_difference.shape)
        wide_worst = np.unravel_index(
            int(np.argmax(wide_difference)), wide_difference.shape)
        raise AssertionError(
            f"coarse dslash mismatch: narrow={narrow_error:.3e} "
            f"wide={wide_error:.3e} "
            f"narrow_worst={narrow_worst} "
            f"narrow_actual={global_output[narrow_worst]:.6g} "
            f"narrow_expected={expected_all[narrow_worst]:.6g} "
            f"wide_worst={wide_worst} "
            f"wide_actual={global_output_wide[wide_worst]:.6g} "
            f"wide_expected={expected_wide_all[wide_worst]:.6g}")
    return narrow_error, wide_error


def run_orthogonalize_test(library: ctypes.CDLL) -> float:
    E, e = 6, 12
    fine = (4, 4, 4, 8)
    coarse = (2, 2, 2, 4)
    rng = np.random.default_rng(17 + MPI.COMM_WORLD.rank)
    flat = (
        rng.normal(size=(E, e, *fine)) +
        1j * rng.normal(size=(E, e, *fine))).astype(np.complex64)
    blocked = np.empty_like(flat)
    status = library.applyMultigridLocalOrthogonalizeQcu(
        pointer(flat), pointer(blocked), E, e, *fine, *coarse, 2)
    if status != 0:
        raise AssertionError(f"KUPL CGS2 failed with status {status}")
    Xc, Yc, Zc, Tc = coarse
    bx, by, bz, bt = (fine[index] // coarse[index] for index in range(4))
    columns = blocked.reshape(
        E, e, Xc, bx, Yc, by, Zc, bz, Tc, bt).transpose(
            2, 4, 6, 8, 0, 1, 3, 5, 7, 9).reshape(
                Xc * Yc * Zc * Tc, E, e * bx * by * bz * bt)
    gram = columns.conj() @ columns.transpose(0, 2, 1)
    worst = float(np.max(np.abs(
        gram - np.eye(E, dtype=np.complex64))))
    if worst >= 1.0e-4:
        raise AssertionError(f"KUPL CGS2 Gram error {worst:.3e}")
    return worst


def run_null_vector_test(
        library: ctypes.CDLL) -> tuple[float, float, float]:
    comm = MPI.COMM_WORLD
    grid = (comm.size, 1, 1, 1)
    local_physical = (8, 8, 8, 16)
    fine = (8, 8, 8, 8)
    # Match production parity-compressed time while keeping the padded
    # stencil window smaller than the periodic time extent.  Spatial coarse
    # extents are greater than two so +1/-1 cannot collapse to the same site.
    coarse = (4, 4, 4, 8)
    E, e = 3, 12
    parity_shape = fine
    gauge = np.empty((2, 3, 3, 4, *parity_shape), dtype=np.complex64)
    clover_even = np.empty((4, 3, 4, 3, *parity_shape), dtype=np.complex64)
    clover_odd = np.empty_like(clover_even)
    clover_even_inv = np.empty_like(clover_even)
    clover_odd_inv = np.empty_like(clover_even)
    blocked_shape = (
        E, e, coarse[0], fine[0] // coarse[0],
        coarse[1], fine[1] // coarse[1], coarse[2], fine[2] // coarse[2],
        coarse[3], fine[3] // coarse[3])
    blocked = np.empty(blocked_shape, dtype=np.complex64)
    repeated = np.empty_like(blocked)
    run_stencil = comm.size == 1
    sitting = np.zeros((E, E, *coarse), dtype=np.complex64)
    hop_nn = np.zeros((2, 4, E, E, *coarse), dtype=np.complex64)
    hop_diag = np.zeros((2, 2, 6, E, E, *coarse), dtype=np.complex64)

    params = np.zeros(58, dtype=np.int32)
    params[0:4] = local_physical
    params[4] = int(np.prod(local_physical))
    params[5:9] = grid
    params[10] = comm.rank
    params[11] = comm.size
    params[13] = 300
    params[14] = 2
    params[15] = 0
    params[16] = -1
    params[19] = E
    params[20:24] = coarse
    params[52] = 117
    argv = np.zeros(8, dtype=np.float32)
    argv[0], argv[1], argv[2], argv[7] = 0.17, 1.0e-7, 0.1, 0.8
    set_ptrs = np.zeros(100, dtype=np.int64)

    library.applyInitQcu(pointer(set_ptrs), pointer(params), pointer(argv))
    try:
        library.applyGaussGaugeQcu(pointer(gauge), pointer(set_ptrs),
                                   pointer(params))
    finally:
        library.applyEndQcu(pointer(set_ptrs), pointer(params))

    params[16] = 1
    library.applyInitQcu(pointer(set_ptrs), pointer(params), pointer(argv))
    try:
        params[9] = 0
        library.applyCloversQcu(
            pointer(clover_even), pointer(clover_even_inv), pointer(gauge),
            pointer(set_ptrs), pointer(params))
        params[9] = 1
        library.applyCloversQcu(
            pointer(clover_odd), pointer(clover_odd_inv), pointer(gauge),
            pointer(set_ptrs), pointer(params))
        status = library.applyMultigridBuildNullVectorsQcu(
            pointer(blocked), pointer(gauge), pointer(clover_even),
            pointer(clover_odd), pointer(clover_even_inv),
            pointer(clover_odd_inv), pointer(set_ptrs), pointer(params), E, e,
            *fine, *coarse, 1, 42, 5.0e-2)
        if status != 0:
            raise AssertionError(
                f"KUPL C++ null-vector builder failed with status {status}")
        repeated_status = library.applyMultigridBuildNullVectorsQcu(
            pointer(repeated), pointer(gauge), pointer(clover_even),
            pointer(clover_odd), pointer(clover_even_inv),
            pointer(clover_odd_inv), pointer(set_ptrs), pointer(params), E, e,
            *fine, *coarse, 1, 42, 5.0e-2)
        if repeated_status != 0:
            raise AssertionError(
                f"KUPL C++ null-vector repeat failed with status "
                f"{repeated_status}")
        if run_stencil:
            stencil_status = library.applyMultigridBuildStencilLocalQcu(
                pointer(sitting), pointer(hop_nn), pointer(hop_diag),
                pointer(blocked), pointer(gauge), pointer(clover_even),
                pointer(clover_odd), pointer(clover_even_inv),
                pointer(clover_odd_inv), pointer(set_ptrs), pointer(params),
                E, e, *fine, *coarse, 8, 8)
            if stencil_status != 0:
                raise AssertionError(
                    f"KUPL C++ local stencil builder failed with status "
                    f"{stencil_status}")
            rng = np.random.default_rng(941)
            coarse_vector = (
                rng.normal(size=(E, *coarse)) +
                1j * rng.normal(size=(E, *coarse))).astype(np.complex64)
            prolonged = np.zeros((e, *fine), dtype=np.complex64)
            schur_vector = np.zeros((e, *fine), dtype=np.complex64)
            restricted = np.zeros((E, *coarse), dtype=np.complex64)
            stencil_vector = np.zeros((E, *coarse), dtype=np.complex64)
            library.applyMultigridProLongQcu(
                pointer(prolonged), pointer(coarse_vector), pointer(blocked),
                pointer(set_ptrs), pointer(params))
            library.applyCloverBistabCgDslashQcu(
                pointer(schur_vector), pointer(prolonged), pointer(gauge),
                pointer(clover_even), pointer(clover_odd),
                pointer(clover_even_inv), pointer(clover_odd_inv),
                pointer(set_ptrs), pointer(params))
            library.applyMultigridRestrictQcu(
                pointer(restricted), pointer(schur_vector), pointer(blocked),
                pointer(set_ptrs), pointer(params))
            library.applyMultigridCoarseDslashWideQcu(
                pointer(stencil_vector), pointer(coarse_vector),
                pointer(sitting), pointer(hop_nn), pointer(hop_diag),
                pointer(set_ptrs), pointer(params))
            worst_basis = (0.0, None, -1)
            basis_centers = (
                (0, 0, 0, 0),
                (1, 1, 1, 5),
                (2, 1, 3, 3),
                (3, 3, 3, 7),
            )
            for center in basis_centers:
                for probe in range(E):
                    coarse_vector.fill(0)
                    coarse_vector[(probe, *center)] = 1
                    library.applyMultigridProLongQcu(
                        pointer(prolonged), pointer(coarse_vector),
                        pointer(blocked), pointer(set_ptrs), pointer(params))
                    library.applyCloverBistabCgDslashQcu(
                        pointer(schur_vector), pointer(prolonged),
                        pointer(gauge), pointer(clover_even),
                        pointer(clover_odd), pointer(clover_even_inv),
                        pointer(clover_odd_inv), pointer(set_ptrs),
                        pointer(params))
                    library.applyMultigridRestrictQcu(
                        pointer(restricted), pointer(schur_vector),
                        pointer(blocked), pointer(set_ptrs), pointer(params))
                    library.applyMultigridCoarseDslashWideQcu(
                        pointer(stencil_vector), pointer(coarse_vector),
                        pointer(sitting), pointer(hop_nn), pointer(hop_diag),
                        pointer(set_ptrs), pointer(params))
                    basis_error = float(
                        np.linalg.norm(stencil_vector - restricted) /
                        max(np.linalg.norm(restricted), 1.0))
                    if basis_error > worst_basis[0]:
                        worst_basis = (basis_error, center, probe)
            if worst_basis[0] >= 2.0e-5:
                raise AssertionError(
                    f"KUPL local stencil basis error {worst_basis[0]:.3e} "
                    f"at center={worst_basis[1]} probe={worst_basis[2]}")
    finally:
        library.applyEndQcu(pointer(set_ptrs), pointer(params))

    Xc, Yc, Zc, Tc = coarse
    bx, by, bz, bt = (fine[index] // coarse[index] for index in range(4))
    columns = blocked.reshape(
        E, e, Xc, bx, Yc, by, Zc, bz, Tc, bt).transpose(
            2, 4, 6, 8, 0, 1, 3, 5, 7, 9).reshape(
                Xc * Yc * Zc * Tc, E, e * bx * by * bz * bt)
    gram = columns.conj() @ columns.transpose(0, 2, 1)
    gram_error = float(np.max(np.abs(
        gram - np.eye(E, dtype=np.complex64))))
    deterministic_error = float(
        np.linalg.norm(blocked - repeated) /
        max(np.linalg.norm(repeated), 1.0))
    if not np.any(blocked):
        raise AssertionError("KUPL C++ null-vector builder returned zeros")
    if gram_error >= 1.0e-4:
        raise AssertionError(f"KUPL null-vector Gram error {gram_error:.3e}")
    if deterministic_error >= 1.0e-5:
        raise AssertionError(
            f"KUPL null-vector deterministic error {deterministic_error:.3e}")
    if not run_stencil:
        return gram_error, deterministic_error, 0.0
    galerkin_error = float(
        np.linalg.norm(stencil_vector - restricted) /
        max(np.linalg.norm(restricted), 1.0))
    if galerkin_error >= 2.0e-5:
        raise AssertionError(
            f"KUPL local stencil Galerkin error {galerkin_error:.3e}")
    return gram_error, deterministic_error, galerkin_error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("all", "gaussian", "transfer", "coarse",
                            "orthogonalize", "null"),
        default="all")
    args = parser.parse_args()
    library = bind_library(args.library)
    total_volume = 4 * MPI.COMM_WORLD.size * 4 * 4 * 4
    if int(np.prod((MPI.COMM_WORLD.size, 1, 1, 1))) != MPI.COMM_WORLD.size:
        raise RuntimeError("invalid test process grid")
    gaussian_c64 = gaussian_c128 = 0.0
    restrict_error = prolong_error = 0.0
    narrow_error = wide_error = 0.0
    orthogonalize_error = 0.0
    null_gram_error = null_deterministic_error = 0.0
    null_galerkin_error = 0.0
    if args.stage in ("all", "gaussian"):
        gaussian_c64 = run_gaussian_test(
            library, np.dtype(np.complex64), 2.0e-5)
        gaussian_c128 = run_gaussian_test(
            library, np.dtype(np.complex128), 2.0e-12)
    if args.stage in ("all", "transfer"):
        restrict_error, prolong_error = run_transfer_test(library)
    if args.stage in ("all", "coarse"):
        narrow_error, wide_error = run_coarse_dslash_test(library)
    if args.stage in ("all", "orthogonalize"):
        orthogonalize_error = run_orthogonalize_test(library)
    if args.stage in ("all", "null"):
        (null_gram_error, null_deterministic_error,
         null_galerkin_error) = run_null_vector_test(library)
    if MPI.COMM_WORLD.rank == 0:
        print(
            f"status=PASS ranks={MPI.COMM_WORLD.size} "
            f"global_sites={total_volume} "
            f"gaussian_c64={gaussian_c64:.3e} "
            f"gaussian_c128={gaussian_c128:.3e} "
            f"restrict={restrict_error:.3e} prolong={prolong_error:.3e} "
            f"coarse={narrow_error:.3e} coarse_wide={wide_error:.3e} "
            f"orthogonalize={orthogonalize_error:.3e} "
            f"null_gram={null_gram_error:.3e} "
            f"null_deterministic={null_deterministic_error:.3e} "
            f"null_galerkin={null_galerkin_error:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
