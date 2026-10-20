#!/usr/bin/env python3
"""Validate the KUPL Clover ABI through Python C-order arrays and ctypes."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import sys

import numpy as np
from mpi4py import MPI

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pyqcu.cann as _torch
from pyqcu import dslash, tools


def gauge_matrix(x: int, y: int, z: int, t: int, direction: int) -> np.ndarray:
    theta = 0.31 * np.sin(0.17 * (x + 2 * y + 3 * z + 5 * t + 7 * direction)) + 0.11
    phi = 0.43 * (x - y + 2 * z - 3 * t + direction)
    psi = 0.29 * (2 * x + y - z + 4 * t - direction)
    cosine = np.cos(theta)
    sine = np.sin(theta)
    return np.array(
        [
            [cosine, -sine * np.exp(1j * phi), 0],
            [sine * np.exp(1j * phi), cosine, 0],
            [0, 0, np.exp(1j * psi)],
        ],
        dtype=np.complex128,
    )


def bind_library(path: str) -> ctypes.CDLL:
    library = ctypes.CDLL(path)
    long_long = ctypes.c_longlong
    library.applyInitQcu.argtypes = [long_long, long_long, long_long]
    library.applyEndQcu.argtypes = [long_long, long_long]
    library.applyCloverQcu.argtypes = [
        long_long,
        long_long,
        long_long,
        long_long,
    ]
    library.applyCloversQcu.argtypes = [
        long_long,
        long_long,
        long_long,
        long_long,
        long_long,
    ]
    return library


def pointer(array: np.ndarray) -> int:
    return int(array.ctypes.data)


def run_precision(
    library: ctypes.CDLL,
    full_gauge: np.ndarray,
    dims: tuple[int, int, int, int],
    dtype: np.dtype,
    data_type: int,
    tolerance: float,
) -> dict[str, object]:
    x_extent, y_extent, z_extent, t_extent = dims
    half_time = t_extent // 2
    half_sites = x_extent * y_extent * z_extent * half_time
    vector_size = 12

    full_gauge = np.asarray(full_gauge, dtype=dtype, order="C")
    full_tensor = _torch.as_tensor(full_gauge)
    blocked_tensor = tools.oooxyzt2poooxyzt(full_tensor).contiguous()
    blocked_gauge = blocked_tensor.detach().cpu().numpy()

    params = np.zeros(58, dtype=np.int32)
    params[[0, 1, 2, 3]] = dims
    params[4] = int(np.prod(dims))
    params[[5, 6, 7, 8]] = 1
    params[9] = 0
    params[10] = 0
    params[11] = 1
    params[13] = 200
    params[14] = data_type
    params[15] = 0
    params[16] = 2
    params[51] = 0
    set_ptrs = np.zeros(100, dtype=np.int64)

    clover_shape = (4, 3, 4, 3, x_extent, y_extent, z_extent, half_time)
    clover = np.empty(clover_shape, dtype=dtype, order="C")
    inverse = np.empty(clover_shape, dtype=dtype, order="C")
    single = np.empty(clover_shape, dtype=dtype, order="C")
    identity = np.eye(vector_size, dtype=dtype)
    identity_field = identity.reshape(4, 3, 4, 3, 1, 1, 1, 1)
    identity_flat = identity.reshape(vector_size, vector_size, 1)

    mass = 0.10
    kappa = 1.0 / (2.0 * mass + 8.0)
    csw = 1.0
    argv = np.zeros(8, dtype=np.float32 if dtype == np.dtype(np.complex64) else np.float64)
    argv[0] = mass
    argv[1] = 1e-7 if dtype == np.dtype(np.complex64) else 1e-12
    argv[7] = csw

    library.applyInitQcu(pointer(set_ptrs), pointer(params), pointer(argv))
    library.applyCloverQcu(
        pointer(single),
        pointer(blocked_gauge),
        pointer(set_ptrs),
        pointer(params),
    )
    library.applyCloversQcu(
        pointer(clover),
        pointer(inverse),
        pointer(blocked_gauge),
        pointer(set_ptrs),
        pointer(params),
    )
    library.applyEndQcu(pointer(set_ptrs), pointer(params))
    params[15] += 1

    reference_term_full = dslash.make_clover(
        full_tensor,
        kappa=_torch.as_tensor(
            [kappa],
            dtype=_torch.float64 if dtype == np.dtype(np.complex128) else _torch.float32,
        ),
        u_0=_torch.as_tensor(
            [1.0],
            dtype=_torch.float64 if dtype == np.dtype(np.complex128) else _torch.float32,
        ),
        csw=csw,
    )
    reference_term = (
        tools.oooxyzt2poooxyzt(reference_term_full)[0]
        .detach()
        .cpu()
        .numpy()
        .astype(dtype, copy=False)
    )
    kupl_term = clover - identity_field
    expected_onsite = reference_term + identity_field
    term_error = float(
        np.linalg.norm(kupl_term - reference_term)
        / max(np.linalg.norm(reference_term), 1e-30)
    )
    onsite_error = float(
        np.linalg.norm(clover - expected_onsite)
        / max(np.linalg.norm(expected_onsite), 1e-30)
    )
    single_error = float(
        np.linalg.norm(single - clover) / max(np.linalg.norm(clover), 1e-30)
    )
    clover_mats = clover.reshape(vector_size, vector_size, half_sites).transpose(2, 0, 1)
    inverse_mats = inverse.reshape(vector_size, vector_size, half_sites).transpose(2, 0, 1)
    inverse_error = float(
        np.linalg.norm(clover_mats @ inverse_mats - identity)
        / np.linalg.norm(identity)
    )

    argv_zero = argv.copy()
    argv_zero[7] = 0.0
    library.applyInitQcu(pointer(set_ptrs), pointer(params), pointer(argv_zero))
    library.applyCloversQcu(
        pointer(clover),
        pointer(inverse),
        pointer(blocked_gauge),
        pointer(set_ptrs),
        pointer(params),
    )
    library.applyEndQcu(pointer(set_ptrs), pointer(params))
    params[15] += 1
    csw0_clover_error = float(
        np.linalg.norm(clover.reshape(vector_size, vector_size, half_sites) - identity_flat)
        / np.linalg.norm(identity)
    )
    csw0_inverse_error = float(
        np.linalg.norm(inverse.reshape(vector_size, vector_size, half_sites) - identity_flat)
        / np.linalg.norm(identity)
    )

    passed = (
        term_error < tolerance
        and onsite_error < tolerance
        and single_error < tolerance
        and inverse_error < tolerance
        and csw0_clover_error < 1e-14
        and csw0_inverse_error < 1e-14
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "data_type": "c64" if data_type == 2 else "c128",
        "term_error": term_error,
        "onsite_error": onsite_error,
        "single_clover_error": single_error,
        "inverse_error": inverse_error,
        "csw0_clover_identity_error": csw0_clover_error,
        "csw0_inverse_identity_error": csw0_inverse_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    args = parser.parse_args()

    if MPI.COMM_WORLD.Get_size() != 1:
        raise RuntimeError("this ABI test expects exactly one MPI rank")

    dims = (2, 2, 2, 4)
    x_extent, y_extent, z_extent, t_extent = dims
    half_time = t_extent // 2
    half_sites = x_extent * y_extent * z_extent * half_time
    vector_size = 12
    onsite_size = vector_size * vector_size

    # Explicit canonical assertions: component axes precede the flattened site.
    fermion_shape = (2, 4, 3, x_extent, y_extent, z_extent, half_time)
    fermion = np.arange(np.prod(fermion_shape), dtype=np.float64).reshape(
        fermion_shape, order="C"
    )
    fermion_view = fermion.reshape(2, vector_size, half_sites)
    for parity in range(2):
        for component in range(vector_size):
            for site in range(half_sites):
                expected = (parity * vector_size + component) * half_sites + site
                assert fermion_view[parity, component, site] == expected

    clover_shape = (4, 3, 4, 3, x_extent, y_extent, z_extent, half_time)
    clover_layout = np.arange(onsite_size * half_sites, dtype=np.float64).reshape(
        clover_shape, order="C"
    )
    clover_view = clover_layout.reshape(vector_size, vector_size, half_sites)
    for row in range(vector_size):
        for column in range(vector_size):
            for site in range(half_sites):
                expected = (row * vector_size + column) * half_sites + site
                assert clover_view[row, column, site] == expected

    full_gauge = np.zeros(
        (3, 3, 4, x_extent, y_extent, z_extent, t_extent),
        dtype=np.complex128,
        order="C",
    )
    for x in range(x_extent):
        for y in range(y_extent):
            for z in range(z_extent):
                for t in range(t_extent):
                    for direction in range(4):
                        full_gauge[:, :, direction, x, y, z, t] = gauge_matrix(
                            x, y, z, t, direction
                        )

    library = bind_library(args.library)
    c64 = run_precision(
        library,
        full_gauge,
        dims,
        np.dtype(np.complex64),
        2,
        5e-4,
    )
    c128 = run_precision(
        library,
        full_gauge,
        dims,
        np.dtype(np.complex128),
        3,
        1e-10,
    )
    passed = c64["status"] == "PASS" and c128["status"] == "PASS"
    print(
        json.dumps(
            {
                "status": "PASS" if passed else "FAIL",
                "fermion_layout": "[2,4,3,X,Y,Z,T/2]",
                "clover_layout": "[4,3,4,3,X,Y,Z,T/2]",
                "c64": c64,
                "c128": c128,
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
