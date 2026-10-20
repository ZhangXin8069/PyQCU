#!/usr/bin/env python3
"""Differential tests for KUPL legacy Clover multigrid and verifier."""

from __future__ import annotations

import argparse
import ctypes
from pathlib import Path

import numpy as np
from mpi4py import MPI

if __package__:
    from . import test_strict as strict
else:
    import test_strict as strict


LL = ctypes.c_longlong


def bind_library(path: Path) -> ctypes.CDLL:
    library = strict.bind_library(path)
    for name in ("applyCloverBistabCgQcu", "applyCloverMultigridQcu"):
        function = getattr(library, name)
        function.argtypes = (LL,) * 9
        function.restype = None
    function = library.verifyCloverMultigridQcu
    function.argtypes = (LL,) * 9
    function.restype = ctypes.c_int
    return library


def make_identity_state(
        dtype: np.dtype, dims: tuple[int, int, int, int],
        grid: tuple[int, int, int, int], num_levels: int,
        deflate: bool, restart: int,
        coarse_dims: tuple[int, int, int, int] | None = None,
        coarse_E: int = 2) -> tuple[np.ndarray, ...]:
    half_sites = int(np.prod(dims)) // 2
    gauge = np.zeros((2, 3, 3, 4, *dims[:3], dims[3] // 2), dtype=dtype)
    for parity in range(2):
        for direction in range(4):
            for t2 in range(dims[3] // 2):
                for x in range(dims[0]):
                    for y in range(dims[1]):
                        for z in range(dims[2]):
                            gauge[(parity, slice(None), slice(None),
                                   direction, x, y, z, t2)] = np.eye(
                                       3, dtype=dtype)
    onsite = np.zeros(2 * 12 * 12 * half_sites, dtype=dtype)
    onsite_view = onsite.reshape(2, 12, 12, half_sites)
    onsite_view[:, np.arange(12), np.arange(12), :] = 1
    even = np.ascontiguousarray(onsite_view[0].reshape(-1))
    odd = np.ascontiguousarray(onsite_view[1].reshape(-1))
    params, argv = strict.params_for(dims, grid, dtype, num_levels=num_levels)
    params[16] = 1
    params[9] = 1
    params[13] = 400
    params[26] = restart
    params[55] = int(deflate)
    argv[0] = 1.0e12
    argv[1] = 1.0e-6 if dtype == np.dtype(np.complex64) else 1.0e-11
    argv[3] = argv[1]
    set_ptrs = np.zeros(100, dtype=np.int64)
    coarse_assets: tuple[np.ndarray, ...] = ()
    if num_levels > 1:
        assert coarse_dims is not None
        params[19] = coarse_E
        params[20:24] = coarse_dims
        params[24] = 100
        params[25] = params[14]
        params[26] = 3
        block = tuple(dims[i] // coarse_dims[i] for i in range(4))
        null = np.zeros(
            coarse_E * 12 * int(np.prod(dims)), dtype=dtype)
        null_view = null.reshape(
            coarse_E, 12, coarse_dims[0], block[0], coarse_dims[1],
            block[1], coarse_dims[2], block[2], coarse_dims[3], block[3])
        for component in range(coarse_E):
            null_view[component, component, ...] = (
                1.0 / np.sqrt(block[0] * block[1] * block[2] * block[3]))
        coarse_vol = int(np.prod(coarse_dims))
        hop_nn = np.zeros(2 * 4 * coarse_E * coarse_E * coarse_vol,
                          dtype=dtype)
        hop_diag = np.zeros(
            2 * 2 * 6 * coarse_E * coarse_E * coarse_vol, dtype=dtype)
        sitting = np.zeros(coarse_E * coarse_E * coarse_vol, dtype=dtype)
        sitting.reshape(coarse_E, coarse_E, coarse_vol)[
            np.arange(coarse_E), np.arange(coarse_E), :] = 1
        set_ptrs[30:34] = [
            strict.pointer(null), strict.pointer(hop_nn),
            strict.pointer(hop_diag), strict.pointer(sitting),
        ]
        coarse_assets = (null, hop_nn, hop_diag, sitting)
    return params, argv, set_ptrs, gauge, even, odd, coarse_assets


def run_case(library: ctypes.CDLL, dtype: np.dtype,
             num_levels: int, deflate: bool,
             restart: int) -> tuple[float, int]:
    comm = MPI.COMM_WORLD
    dims = (4, 2, 2, 4)
    coarse_dims = (2, 2, 2, 2)
    (params, argv, set_ptrs, gauge, even, odd,
     coarse_assets) = make_identity_state(
        dtype, dims, (comm.size, 1, 1, 1), num_levels, deflate, restart,
        coarse_dims)
    _ = coarse_assets
    rng = np.random.default_rng(8181 + comm.rank + num_levels)
    source = (
        rng.normal(size=2 * 12 * int(np.prod(dims)) // 2) +
        1j * rng.normal(size=2 * 12 * int(np.prod(dims)) // 2)
    ).astype(dtype)
    reference = np.empty_like(source)
    multigrid = np.empty_like(source)
    library.applyInitQcu(
        strict.pointer(set_ptrs), strict.pointer(params),
        strict.pointer(argv))
    try:
        library.applyCloverBistabCgQcu(
            strict.pointer(reference), strict.pointer(source),
            strict.pointer(gauge), strict.pointer(even), strict.pointer(odd),
            strict.pointer(even), strict.pointer(odd),
            strict.pointer(set_ptrs), strict.pointer(params))
        before_source = source.copy()
        verify = library.verifyCloverMultigridQcu(
            strict.pointer(multigrid), strict.pointer(source),
            strict.pointer(gauge), strict.pointer(even), strict.pointer(odd),
            strict.pointer(even), strict.pointer(odd),
            strict.pointer(set_ptrs), strict.pointer(params))
        after_source = source.copy()
        if not np.array_equal(before_source, after_source):
            raise AssertionError("legacy MG verifier modified source")
        params[57] = 0
        library.applyCloverMultigridQcu(
            strict.pointer(multigrid), strict.pointer(source),
            strict.pointer(gauge), strict.pointer(even), strict.pointer(odd),
            strict.pointer(even), strict.pointer(odd),
            strict.pointer(set_ptrs), strict.pointer(params))
        warm = multigrid.copy()
        params[57] = 1
        library.applyCloverMultigridQcu(
            strict.pointer(warm), strict.pointer(source),
            strict.pointer(gauge), strict.pointer(even), strict.pointer(odd),
            strict.pointer(even), strict.pointer(odd),
            strict.pointer(set_ptrs), strict.pointer(params))
    finally:
        library.applyEndQcu(
            strict.pointer(set_ptrs), strict.pointer(params))
    error = max(strict.relative(multigrid, reference),
                strict.relative(warm, reference))
    if comm.rank == 0 and (verify != 0 or error >= 2.0e-4):
        raise AssertionError(
            f"legacy MG mismatch levels={num_levels} dtype={dtype} "
            f"verify={verify} error={error:.3e}")
    return error, verify


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    args = parser.parse_args()
    library = bind_library(args.library)
    results = []
    for dtype, levels, deflate, restart in (
            (np.dtype(np.complex64), 1, False, 0),
            (np.dtype(np.complex64), 2, False, 0),
            (np.dtype(np.complex64), 2, True, 3),
            (np.dtype(np.complex128), 1, False, 0),
            (np.dtype(np.complex128), 2, False, 0),
            (np.dtype(np.complex128), 2, True, 3)):
        results.append((
            dtype, levels, deflate, restart,
            *run_case(library, dtype, levels, deflate, restart)))
    if MPI.COMM_WORLD.rank == 0:
        print(
            "status=PASS ranks="
            f"{MPI.COMM_WORLD.size} "
            + " ".join(
                f"{dtype}/{levels}/deflate={int(deflate)}/restart={restart}:"
                f"err={error:.3e},verify={verify}"
                for dtype, levels, deflate, restart, error, verify in results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
