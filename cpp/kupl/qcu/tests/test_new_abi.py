#!/usr/bin/env python3
"""Independent NumPy differential tests for the expanded KUPL C ABI."""

from __future__ import annotations

import argparse
import ctypes
import math
import os
from dataclasses import dataclass
import numpy as np
from mpi4py import MPI


PARAMS_SIZE = 58
ARGV_SIZE = 8
SET_PTRS_SIZE = 100


def gamma_matrix(direction: int) -> np.ndarray:
    i = 1j
    if direction == 0:
        return np.array(
            [[0, 0, 0, i], [0, 0, i, 0], [0, -i, 0, 0], [-i, 0, 0, 0]],
            dtype=np.complex128,
        )
    if direction == 1:
        return np.array(
            [[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]],
            dtype=np.complex128,
        )
    if direction == 2:
        return np.array(
            [[0, 0, i, 0], [0, 0, 0, -i], [-i, 0, 0, 0], [0, i, 0, 0]],
            dtype=np.complex128,
        )
    return np.array(
        [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]],
        dtype=np.complex128,
    )


@dataclass
class Grid:
    local: tuple[int, int, int, int]
    process: tuple[int, int, int, int]
    coord: tuple[int, int, int, int]
    rank: int
    size: int

    @property
    def global_shape(self) -> tuple[int, int, int, int]:
        return tuple(
            self.local[index] * self.process[index] for index in range(4)
        )

    @property
    def half_sites(self) -> int:
        return (
            self.local[0]
            * self.local[1]
            * self.local[2]
            * (self.local[3] // 2)
        )

    def local_slices(self) -> tuple[slice, slice, slice, slice]:
        return tuple(
            slice(
                self.coord[index] * self.local[index],
                (self.coord[index] + 1) * self.local[index],
            )
            for index in range(4)
        )


def bind_library(path: str) -> ctypes.CDLL:
    library = ctypes.CDLL(path)
    ll = ctypes.c_longlong
    library.applyInitQcu.argtypes = [ll, ll, ll]
    library.applyEndQcu.argtypes = [ll, ll]
    for name in (
        "applyWilsonDslashQcu",
        "applyWilsonBistabCgQcu",
        "applyWilsonBistabCgDslashQcu",
        "applyWilsonCgQcu",
        "applyWilsonCgDslashQcu",
        "applyCloverBistabCgQcu",
        "applyCloverBistabCgDslashQcu",
        "applyLaplacianQcu",
    ):
        getattr(library, name).argtypes = [ll, ll, ll, ll, ll]
    library.applyCloversQcu.argtypes = [ll, ll, ll, ll, ll]
    library.applyCloverBistabCgPrepareQcu.argtypes = [
        ll,
        ll,
        ll,
        ll,
        ll,
        ll,
        ll,
        ll,
        ll,
    ]
    library.applyCloverBistabCgReconstructQcu.argtypes = [
        ll,
        ll,
        ll,
        ll,
        ll,
        ll,
        ll,
        ll,
        ll,
        ll,
    ]
    library.applyCloverBistabCgPrepareQcu.restype = ctypes.c_int
    library.applyCloverBistabCgReconstructQcu.restype = ctypes.c_int
    return library


def pointer(array: np.ndarray) -> int:
    return int(array.ctypes.data)


def gauge_matrix(x: int, y: int, z: int, t: int, direction: int) -> np.ndarray:
    theta = 0.31 * math.sin(0.17 * (x + 2 * y + 3 * z + 5 * t + 7 * direction)) + 0.11
    phi = 0.43 * (x - y + 2 * z - 3 * t + direction)
    psi = 0.29 * (2 * x + y - z + 4 * t - direction)
    return np.array(
        [
            [math.cos(theta), -math.sin(theta) * np.exp(1j * phi), 0],
            [math.sin(theta) * np.exp(1j * phi), math.cos(theta), 0],
            [0, 0, np.exp(1j * psi)],
        ],
        dtype=np.complex128,
    )


def make_global_state(grid: Grid, dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    gx, gy, gz, gt = grid.global_shape
    gauge = np.empty((3, 3, 4, gx, gy, gz, gt), dtype=dtype)
    source = np.empty((gx, gy, gz, gt, 4, 3), dtype=dtype)
    for x in range(gx):
        for y in range(gy):
            for z in range(gz):
                for t in range(gt):
                    for direction in range(4):
                        gauge[:, :, direction, x, y, z, t] = gauge_matrix(
                            x, y, z, t, direction
                        )
                    phase = 0.071 * (x + 3 * y + 5 * z + 7 * t)
                    for spin in range(4):
                        for color in range(3):
                            local_phase = phase + 0.037 * (spin * color + 1)
                            source[x, y, z, t, spin, color] = (
                                math.cos(local_phase)
                                + 1j * math.sin(1.7 * local_phase)
                            )
    return gauge.astype(dtype, copy=False), source.astype(dtype, copy=False)


def gauge_to_local(global_gauge: np.ndarray, grid: Grid) -> np.ndarray:
    local = global_gauge[(slice(None), slice(None), slice(None), *grid.local_slices())]
    lx, ly, lz, lt = grid.local
    blocked = np.empty(
        (2, 3, 3, 4, lx, ly, lz, lt // 2), dtype=global_gauge.dtype
    )
    for parity in range(2):
        for t2 in range(lt // 2):
            for z in range(lz):
                for y in range(ly):
                    for x in range(lx):
                        t = 2 * t2 + (((x + y + z) & 1) ^ parity)
                        blocked[parity, :, :, :, x, y, z, t2] = local[
                            :, :, :, x, y, z, t
                        ]
    return np.ascontiguousarray(blocked)


def fermion_to_local(global_field: np.ndarray, grid: Grid) -> np.ndarray:
    local = global_field[grid.local_slices()]
    lx, ly, lz, lt = grid.local
    blocked = np.empty((2, 12, lx, ly, lz, lt // 2), dtype=global_field.dtype)
    for parity in range(2):
        for t2 in range(lt // 2):
            for z in range(lz):
                for y in range(ly):
                    for x in range(lx):
                        t = 2 * t2 + (((x + y + z) & 1) ^ parity)
                        blocked[parity, :, x, y, z, t2] = local[
                            x, y, z, t, :, :
                        ].reshape(12)
    return np.ascontiguousarray(blocked)


def local_compact_to_global(local: np.ndarray, grid: Grid, parity: int) -> np.ndarray:
    gx, gy, gz, gt = grid.global_shape
    global_field = np.zeros((gx, gy, gz, gt, 12), dtype=local.dtype)
    lx, ly, lz, _ = grid.local
    blocked = local.reshape(12, lx, ly, lz, grid.local[3] // 2)
    for t2 in range(grid.local[3] // 2):
        for z in range(lz):
            for y in range(ly):
                for x in range(lx):
                    gx_index = grid.coord[0] * lx + x
                    gy_index = grid.coord[1] * ly + y
                    gz_index = grid.coord[2] * lz + z
                    t_local = 2 * t2 + (((x + y + z) & 1) ^ parity)
                    gt_index = grid.coord[3] * grid.local[3] + t_local
                    global_field[gx_index, gy_index, gz_index, gt_index] = blocked[
                        :, x, y, z, t2
                    ]
    MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, global_field, op=MPI.SUM)
    return global_field


def local_full_to_global(local: np.ndarray, grid: Grid) -> np.ndarray:
    gx, gy, gz, gt = grid.global_shape
    global_field = np.zeros((gx, gy, gz, gt, 12), dtype=local.dtype)
    lx, ly, lz, lt = grid.local
    blocked = local.reshape(2, 12, lx, ly, lz, lt // 2)
    for parity in range(2):
        for t2 in range(lt // 2):
            for z in range(lz):
                for y in range(ly):
                    for x in range(lx):
                        gx_index = grid.coord[0] * lx + x
                        gy_index = grid.coord[1] * ly + y
                        gz_index = grid.coord[2] * lz + z
                        t_local = 2 * t2 + (((x + y + z) & 1) ^ parity)
                        gt_index = grid.coord[3] * lt + t_local
                        global_field[gx_index, gy_index, gz_index, gt_index] = blocked[
                            parity, :, x, y, z, t2
                        ]
    MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, global_field, op=MPI.SUM)
    return global_field


def apply_gamma(field: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    return np.einsum("ab,...bc->...ac", gamma, field)


def bare_wilson(
    field: np.ndarray, gauge: np.ndarray, target_parity: int
) -> np.ndarray:
    output = np.zeros_like(field)
    for direction in range(4):
        gamma = gamma_matrix(direction)
        link = np.transpose(gauge[:, :, direction], (2, 3, 4, 5, 0, 1))
        forward_source = np.roll(field, -1, axis=direction)
        forward_link = np.einsum(
            "abcdsij,abcdsj->abcdsi",
            link[..., None, :, :],
            forward_source,
        )
        output -= apply_gamma(forward_link, gamma)
        output += forward_link

        backward_source = np.roll(field, 1, axis=direction)
        previous_link = np.roll(link, 1, axis=direction)
        dagger_link = np.swapaxes(previous_link.conj(), -1, -2)
        backward_link = np.einsum(
            "abcdsij,abcdsj->abcdsi",
            dagger_link[..., None, :, :],
            backward_source,
        )
        output += apply_gamma(backward_link, gamma)
        output += backward_link
    mask = np.zeros(field.shape[:4], dtype=bool)
    parity = (
        np.indices(mask.shape).sum(axis=0) & 1
    )
    mask[parity == target_parity] = True
    output[~mask] = 0
    return output


def bare_wilson_dagger(
    field: np.ndarray, gauge: np.ndarray, output_parity: int
) -> np.ndarray:
    output = np.zeros_like(field)
    for direction in range(4):
        gamma = gamma_matrix(direction).conj().T
        link = np.transpose(gauge[:, :, direction], (2, 3, 4, 5, 0, 1))

        previous_source = np.roll(field, 1, axis=direction)
        previous_link = np.roll(link, 1, axis=direction)
        dagger_link = np.swapaxes(previous_link.conj(), -1, -2)
        previous_link_field = np.einsum(
            "abcdsij,abcdsj->abcdsi",
            dagger_link[..., None, :, :],
            previous_source,
        )
        output += previous_link_field - apply_gamma(
            previous_link_field, gamma
        )

        next_source = np.roll(field, -1, axis=direction)
        next_link_field = np.einsum(
            "abcdsij,abcdsj->abcdsi",
            link[..., None, :, :],
            next_source,
        )
        output += next_link_field + apply_gamma(next_link_field, gamma)
    mask = np.zeros(field.shape[:4], dtype=bool)
    parity = np.indices(mask.shape).sum(axis=0) & 1
    mask[parity == output_parity] = True
    output[~mask] = 0
    return output


def wilson_schur_global(
    field: np.ndarray, gauge: np.ndarray, target_parity: int, mass: float
) -> np.ndarray:
    kappa = 1.0 / (2.0 * mass + 8.0)
    first = bare_wilson(field, gauge, 1 - target_parity)
    second = bare_wilson(first, gauge, target_parity)
    return field - kappa * kappa * second


def wilson_schur_dagger_global(
    field: np.ndarray, gauge: np.ndarray, target_parity: int, mass: float
) -> np.ndarray:
    kappa = 1.0 / (2.0 * mass + 8.0)
    first = bare_wilson_dagger(field, gauge, 1 - target_parity)
    second = bare_wilson_dagger(first, gauge, target_parity)
    return field - kappa * kappa * second


def full_wilson(
    field: np.ndarray, gauge: np.ndarray, mass: float
) -> np.ndarray:
    kappa = 1.0 / (2.0 * mass + 8.0)
    hopping = (
        bare_wilson(field, gauge, 0) + bare_wilson(field, gauge, 1)
    )
    return field - kappa * hopping


def relative_norm(actual: np.ndarray, expected: np.ndarray) -> float:
    local_difference = float(np.vdot(actual - expected, actual - expected).real)
    local_norm = float(np.vdot(expected, expected).real)
    difference = MPI.COMM_WORLD.allreduce(local_difference, op=MPI.SUM)
    norm = MPI.COMM_WORLD.allreduce(local_norm, op=MPI.SUM)
    return math.sqrt(difference / max(norm, 1.0e-30))


def params_for(
    grid: Grid,
    data_type: int,
    mass: float,
    csw: float,
    plan: int,
    target_parity: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    params = np.zeros(PARAMS_SIZE, dtype=np.int32)
    params[[0, 1, 2, 3]] = grid.local
    params[4] = int(np.prod(grid.local))
    params[[5, 6, 7, 8]] = grid.process
    params[9] = target_parity
    params[10] = grid.rank
    params[11] = grid.size
    params[13] = 300
    params[14] = data_type
    params[16] = plan
    params[51] = 1 if os.environ.get("KUPL_TEST_VERBOSE") else 0
    params[57] = 0
    argv = np.zeros(ARGV_SIZE, dtype=np.float32 if data_type == 2 else np.float64)
    argv[0] = mass
    argv[1] = 1.0e-7 if data_type == 2 else 1.0e-12
    argv[7] = csw
    return params, argv, np.zeros(SET_PTRS_SIZE, dtype=np.int64)


def run_wilson_precision(
    library: ctypes.CDLL,
    grid: Grid,
    global_gauge: np.ndarray,
    global_source: np.ndarray,
    dtype: np.dtype,
    data_type: int,
) -> dict[str, float]:
    mass = 0.10
    target_parity = 1
    kappa = 1.0 / (2.0 * mass + 8.0)
    gauge = gauge_to_local(global_gauge, grid)
    source = fermion_to_local(global_source, grid)
    params, argv, set_ptrs = params_for(
        grid, data_type, mass, 0.0, 1, target_parity
    )
    library.applyInitQcu(pointer(set_ptrs), pointer(params), pointer(argv))

    dslash_input = np.ascontiguousarray(source[1 - target_parity])
    dslash_output = np.zeros_like(dslash_input)
    params[16] = 0
    library.applyWilsonDslashQcu(
        pointer(dslash_output),
        pointer(dslash_input),
        pointer(gauge),
        pointer(set_ptrs),
        pointer(params),
    )
    dslash_global = local_compact_to_global(
        dslash_output, grid, target_parity
    )
    dslash_expected = bare_wilson(
        global_source, global_gauge, target_parity
    ).reshape((-1, 12)).astype(dtype)
    parity_mask = (
        np.indices(grid.global_shape).sum(axis=0) & 1
    ).reshape(-1) == target_parity
    dslash_expected[~parity_mask] = 0
    dslash_difference = (
        dslash_global.reshape((-1, 12)).astype(dtype) - dslash_expected
    )
    dslash_num = MPI.COMM_WORLD.allreduce(
        float(np.vdot(dslash_difference, dslash_difference).real), op=MPI.SUM
    )
    dslash_den = MPI.COMM_WORLD.allreduce(
        float(np.vdot(dslash_expected, dslash_expected).real), op=MPI.SUM
    )
    dslash_error = math.sqrt(dslash_num / max(dslash_den, 1.0e-30))
    params[16] = 1

    schur_input = np.ascontiguousarray(source[target_parity])
    schur_output = np.zeros_like(schur_input)
    library.applyWilsonBistabCgDslashQcu(
        pointer(schur_output),
        pointer(schur_input),
        pointer(gauge),
        pointer(set_ptrs),
        pointer(params),
    )
    input_global = local_compact_to_global(
        schur_input, grid, target_parity
    ).reshape((*grid.global_shape, 4, 3))
    expected_global = wilson_schur_global(
        input_global, global_gauge, target_parity, mass
    )
    expected_local = fermion_to_local(
        expected_global.reshape((*grid.global_shape, 4, 3)).astype(dtype),
        grid,
    )[target_parity]
    schur_error = relative_norm(schur_output, expected_local)

    normal_output = np.zeros_like(schur_input)
    library.applyWilsonCgDslashQcu(
        pointer(normal_output),
        pointer(schur_input),
        pointer(gauge),
        pointer(set_ptrs),
        pointer(params),
    )
    first_global = wilson_schur_global(
        input_global, global_gauge, target_parity, mass
    )
    normal_expected_global = wilson_schur_dagger_global(
        first_global, global_gauge, target_parity, mass
    )
    normal_expected_local = fermion_to_local(
        normal_expected_global.reshape((*grid.global_shape, 4, 3)).astype(dtype),
        grid,
    )[target_parity]
    normal_error = relative_norm(normal_output, normal_expected_local)

    solutions: dict[str, np.ndarray] = {}
    for name, function in (
        ("bistab", library.applyWilsonBistabCgQcu),
        ("cg", library.applyWilsonCgQcu),
    ):
        solution = np.zeros(source.size, dtype=dtype)
        function(
            pointer(solution),
            pointer(source),
            pointer(gauge),
            pointer(set_ptrs),
            pointer(params),
        )
        solutions[name] = solution

    source_global_flat = global_source.reshape((-1, 12)).astype(dtype)
    residuals: dict[str, float] = {}
    for name, solution in solutions.items():
        solution_global = local_full_to_global(solution, grid)
        solution_global = solution_global.reshape((*grid.global_shape, 4, 3))
        applied = full_wilson(solution_global, global_gauge, mass)
        residual = applied.reshape((-1, 12)).astype(dtype) - source_global_flat
        denominator = MPI.COMM_WORLD.allreduce(
            float(np.vdot(source_global_flat, source_global_flat).real),
            op=MPI.SUM,
        )
        numerator = MPI.COMM_WORLD.allreduce(
            float(np.vdot(residual, residual).real), op=MPI.SUM
        )
        residuals[name] = math.sqrt(numerator / max(denominator, 1.0e-30))

    solution_full = global_source.reshape((-1, 12)).astype(dtype).copy()
    rhs_global = full_wilson(
        solution_full.reshape((*grid.global_shape, 4, 3)),
        global_gauge,
        mass,
    )
    rhs = fermion_to_local(
        rhs_global.reshape((*grid.global_shape, 4, 3)).astype(dtype), grid
    )
    target_solution = np.ascontiguousarray(source[target_parity])
    compact_rhs = np.zeros_like(target_solution)
    # Clover fields are generated below with a real lifecycle to test the
    # prepare/reconstruct status path at csw=0 (A=I).
    clover_even = np.zeros(grid.half_sites * 144, dtype=dtype)
    clover_odd = np.zeros_like(clover_even)
    inverse_even = np.zeros_like(clover_even)
    inverse_odd = np.zeros_like(clover_even)
    params[16] = 2
    params[9] = 0
    library.applyCloversQcu(
        pointer(clover_even),
        pointer(inverse_even),
        pointer(gauge),
        pointer(set_ptrs),
        pointer(params),
    )
    params[9] = 1
    library.applyCloversQcu(
        pointer(clover_odd),
        pointer(inverse_odd),
        pointer(gauge),
        pointer(set_ptrs),
        pointer(params),
    )
    params[16] = 1
    params[9] = target_parity
    prepare_status = library.applyCloverBistabCgPrepareQcu(
        pointer(compact_rhs),
        pointer(rhs),
        pointer(gauge),
        pointer(clover_even),
        pointer(clover_odd),
        pointer(inverse_even),
        pointer(inverse_odd),
        pointer(set_ptrs),
        pointer(params),
    )
    reconstructed = np.zeros_like(rhs)
    reconstruct_status = library.applyCloverBistabCgReconstructQcu(
        pointer(reconstructed),
        pointer(rhs),
        pointer(target_solution),
        pointer(gauge),
        pointer(clover_even),
        pointer(clover_odd),
        pointer(inverse_even),
        pointer(inverse_odd),
        pointer(set_ptrs),
        pointer(params),
    )
    library.applyEndQcu(pointer(set_ptrs), pointer(params))
    reconstruct_error = relative_norm(reconstructed, source)
    if prepare_status != 0 or reconstruct_status != 0:
        reconstruct_error = float("inf")
    return {
        "dslash": dslash_error,
        "schur": schur_error,
        "normal": normal_error,
        "bistab_residual": residuals["bistab"],
        "cg_residual": residuals["cg"],
        "reconstruct": reconstruct_error,
    }


def make_global_laplacian_state(
    grid: Grid, dtype: np.dtype
) -> tuple[np.ndarray, np.ndarray]:
    gx, gy, gz = grid.global_shape[:3]
    gauge = np.empty((3, 3, 3, gx, gy, gz), dtype=dtype)
    source = np.empty((3, gx, gy, gz), dtype=dtype)
    for x in range(gx):
        for y in range(gy):
            for z in range(gz):
                for direction in range(3):
                    gauge[:, :, direction, x, y, z] = gauge_matrix(
                        x, y, z, 0, direction
                    )
                phase = 0.071 * (x + 3 * y + 5 * z)
                for color in range(3):
                    local_phase = phase + 0.037 * (color + 1)
                    source[color, x, y, z] = (
                        math.cos(local_phase)
                        + 1j * math.sin(1.7 * local_phase)
                    )
    return gauge.astype(dtype, copy=False), source.astype(dtype, copy=False)


def laplacian_reference(
    source: np.ndarray, gauge: np.ndarray
) -> np.ndarray:
    output = 6.0 * source
    for direction in range(3):
        source_axis = direction + 1
        link_axis = direction
        link = np.transpose(gauge[:, :, direction], (2, 3, 4, 0, 1))
        forward = np.einsum(
            "...ij,j...->i...", link, np.roll(source, -1, axis=source_axis)
        )
        previous = np.roll(link, 1, axis=link_axis)
        dagger_link = np.swapaxes(previous.conj(), -1, -2)
        backward = np.einsum(
            "...ij,j...->i...",
            dagger_link,
            np.roll(source, 1, axis=source_axis),
        )
        output -= forward + backward
    return output


def run_laplacian_precision(
    library: ctypes.CDLL,
    grid: Grid,
    global_gauge: np.ndarray,
    global_source: np.ndarray,
    dtype: np.dtype,
    data_type: int,
) -> float:
    slices = grid.local_slices()[:3]
    gauge = np.ascontiguousarray(global_gauge[(slice(None), slice(None), slice(None), *slices)])
    source = np.ascontiguousarray(global_source[(slice(None), *slices)])
    output = np.zeros_like(source)
    params, argv, set_ptrs = params_for(grid, data_type, 0.1, 0.0, -2, 0)
    params[3] = 1
    params[4] = int(np.prod(grid.local[:3]))
    params[8] = 1
    library.applyInitQcu(pointer(set_ptrs), pointer(params), pointer(argv))
    library.applyLaplacianQcu(
        pointer(output),
        pointer(source),
        pointer(gauge),
        pointer(set_ptrs),
        pointer(params),
    )
    library.applyEndQcu(pointer(set_ptrs), pointer(params))
    gx, gy, gz, _ = grid.global_shape
    global_output = np.zeros((3, gx, gy, gz), dtype=dtype)
    global_output[(slice(None), *slices)] = output
    MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, global_output, op=MPI.SUM)
    expected = laplacian_reference(global_source, global_gauge)
    expected_local = np.ascontiguousarray(expected[(slice(None), *slices)])
    return relative_norm(output, expected_local)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--lattice", nargs=4, type=int, default=(2, 2, 2, 4))
    parser.add_argument("--grid", nargs=4, type=int, default=(1, 1, 1, 1))
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("--threads must be positive")
    os.environ["KUPL_OMP_THREADS"] = str(args.threads)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["OMP_DYNAMIC"] = "FALSE"

    if int(np.prod(args.grid)) != MPI.COMM_WORLD.Get_size():
        raise RuntimeError("process grid product must equal MPI world size")
    rank = MPI.COMM_WORLD.Get_rank()
    coordinate = tuple(
        int(value) for value in np.unravel_index(rank, tuple(args.grid), order="C")
    )
    grid = Grid(
        tuple(args.lattice),
        tuple(args.grid),
        coordinate,
        rank,
        MPI.COMM_WORLD.Get_size(),
    )
    library = bind_library(args.library)
    global_gauge64, global_source64 = make_global_state(grid, np.dtype(np.complex128))
    report: dict[str, object] = {
        "lattice": list(args.lattice),
        "grid": list(args.grid),
        "ranks": grid.size,
        "threads": args.threads,
    }
    passed = True
    for name, dtype, data_type, tolerance in (
        ("c64", np.dtype(np.complex64), 2, 5.0e-4),
        ("c128", np.dtype(np.complex128), 3, 2.0e-9),
    ):
        metrics = run_wilson_precision(
            library,
            grid,
            global_gauge64.astype(dtype),
            global_source64.astype(dtype),
            dtype,
            data_type,
        )
        lap_gauge64, lap_source64 = make_global_laplacian_state(
            grid, np.dtype(np.complex128)
        )
        metrics["laplacian"] = run_laplacian_precision(
            library,
            grid,
            lap_gauge64.astype(dtype),
            lap_source64.astype(dtype),
            dtype,
            data_type,
        )
        precision_pass = (
            metrics["dslash"] < tolerance
            and metrics["schur"] < tolerance
            and metrics["normal"] < tolerance
            and metrics["bistab_residual"] < tolerance
            and metrics["cg_residual"] < tolerance
            and metrics["reconstruct"] < tolerance
            and metrics["laplacian"] < tolerance
        )
        metrics["status"] = "PASS" if precision_pass else "FAIL"
        report[name] = metrics
        passed = passed and precision_pass
    report["status"] = "PASS" if passed else "FAIL"
    if grid.rank == 0:
        import json

        print(json.dumps(report, sort_keys=True))
    local_status = 1 if passed else 0
    global_status = MPI.COMM_WORLD.allreduce(local_status, op=MPI.MIN)
    return 0 if global_status else 1


if __name__ == "__main__":
    raise SystemExit(main())
