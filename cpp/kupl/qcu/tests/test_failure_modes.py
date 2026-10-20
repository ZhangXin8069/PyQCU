#!/usr/bin/env python3
"""Check KUPL non-convergence and unimplemented-entry failure behavior."""

from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


LL = ctypes.c_longlong


def bind_library(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL)
    void_signatures = {
        "applyInitQcu": 3,
        "applyEndQcu": 2,
        "applyCloverBistabCgQcu": 9,
        "applyCloverMultigridQcu": 9,
    }
    for name, count in void_signatures.items():
        function = getattr(library, name)
        function.argtypes = (LL,) * count
        function.restype = None
    status = library.applyMultigridStrictInitQcu
    status.argtypes = (LL, LL, ctypes.c_int, ctypes.POINTER(ctypes.c_ulonglong))
    status.restype = ctypes.c_int
    return library


def pointer(array: np.ndarray) -> int:
    return int(array.ctypes.data)


def make_state():
    local = (2, 2, 2, 4)
    half_sites = int(np.prod(local[:3])) * (local[3] // 2)
    params = np.zeros(58, dtype=np.int32)
    params[0:4] = local
    params[4] = int(np.prod(local))
    params[5:9] = 1
    params[13] = 1
    params[14] = 2
    params[15] = 0
    params[16] = 1
    argv = np.zeros(8, dtype=np.float32)
    argv[0] = 0.1
    argv[1] = 1.0e-7
    argv[2] = 0.1
    argv[7] = 0.0
    set_ptrs = np.zeros(100, dtype=np.int64)
    return params, argv, set_ptrs, half_sites


def child_clover_nonconvergence(library: ctypes.CDLL) -> None:
    params, argv, set_ptrs, half_sites = make_state()
    onsite_size = 12 * 12 * half_sites
    onsite = np.zeros(onsite_size, dtype=np.complex64)
    onsite.reshape(12, 12, half_sites)[
        np.arange(12), np.arange(12), :] = 1
    source = (
        np.arange(24 * half_sites, dtype=np.float32) / 1000.0
    ).astype(np.complex64)
    output = np.empty_like(source)
    library.applyInitQcu(pointer(set_ptrs), pointer(params), pointer(argv))
    library.applyCloverBistabCgQcu(
        pointer(output), pointer(source), pointer(np.zeros(1, np.complex64)),
        pointer(onsite), pointer(onsite), pointer(onsite), pointer(onsite),
        pointer(set_ptrs), pointer(params))


def child_void_stub(library: ctypes.CDLL) -> None:
    params, argv, set_ptrs, _ = make_state()
    library.applyInitQcu(pointer(set_ptrs), pointer(params), pointer(argv))
    library.applyCloverMultigridQcu(
        0, 0, 0, 0, 0, 0, 0, pointer(set_ptrs), pointer(params))


def run_child(library_path: Path, mode: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update({
        "OMPI_ALLOW_RUN_AS_ROOT": "1",
        "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
        "KUPL_OMP_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    })
    return subprocess.run(
        [sys.executable, "-B", __file__, "--child", mode,
         "--library", str(library_path)],
        text=True, capture_output=True, timeout=60, check=False,
        env=environment)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--child", choices=("clover", "void-stub"))
    args = parser.parse_args()
    library = bind_library(args.library)
    if args.child == "clover":
        child_clover_nonconvergence(library)
        return 0
    if args.child == "void-stub":
        child_void_stub(library)
        return 0

    status = library.applyMultigridStrictInitQcu(0, 0, 0, None)
    if status == 0:
        raise AssertionError("strict-MG integer stub unexpectedly succeeded")

    clover = run_child(args.library, "clover")
    if clover.returncode == 0 or "true residual" not in clover.stderr:
        raise AssertionError(
            "Clover non-convergence did not fail closed: "
            f"returncode={clover.returncode} stderr={clover.stderr[-2000:]}")
    stub = run_child(args.library, "void-stub")
    if stub.returncode == 0 or "not implemented" not in stub.stderr:
        raise AssertionError(
            "Void stub did not fail explicitly: "
            f"returncode={stub.returncode} stderr={stub.stderr[-2000:]}")
    print("status=PASS clover_nonconvergence=abort void_stub=abort status_stub=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
