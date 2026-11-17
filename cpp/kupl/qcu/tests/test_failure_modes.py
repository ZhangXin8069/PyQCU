#!/usr/bin/env python3
"""Check KUPL non-convergence, invalid controls and unsupported-mode gates."""

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
    verify = library.verifyCloverMultigridQcu
    verify.argtypes = (LL,) * 9
    verify.restype = ctypes.c_int
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


def child_multigrid_null(library: ctypes.CDLL) -> None:
    params, argv, set_ptrs, _ = make_state()
    library.applyInitQcu(pointer(set_ptrs), pointer(params), pointer(argv))
    library.applyCloverMultigridQcu(
        0, 0, 0, 0, 0, 0, 0, pointer(set_ptrs), pointer(params))


def child_multigrid_unsupported_mode(library: ctypes.CDLL) -> None:
    params, argv, set_ptrs, _ = make_state()
    params[17] = 1
    params[54] = 1
    dummy = np.zeros(1, dtype=np.complex64)
    library.applyInitQcu(pointer(set_ptrs), pointer(params), pointer(argv))
    library.applyCloverMultigridQcu(
        pointer(dummy), pointer(dummy), pointer(dummy), pointer(dummy),
        pointer(dummy), pointer(dummy), pointer(dummy), pointer(set_ptrs),
        pointer(params))


def child_multigrid_require_correction(library: ctypes.CDLL) -> None:
    params, argv, set_ptrs, _ = make_state()
    dummy = np.zeros(1, dtype=np.complex64)
    library.applyInitQcu(pointer(set_ptrs), pointer(params), pointer(argv))
    library.applyCloverMultigridQcu(
        pointer(dummy), pointer(dummy), pointer(dummy), pointer(dummy),
        pointer(dummy), pointer(dummy), pointer(dummy), pointer(set_ptrs),
        pointer(params))


def child_ordinary_slot(index: int, library: ctypes.CDLL) -> None:
    """Prove ordinary INIT/END refuse the reserved multigrid slots."""
    params, argv, set_ptrs, _ = make_state()
    params[15] = index
    library.applyInitQcu(pointer(set_ptrs), pointer(params), pointer(argv))
    print(f"UNEXPECTED: applyInitQcu accepted reserved slot {index}")


def child_ordinary_slot_end(index: int, library: ctypes.CDLL) -> None:
    params, argv, set_ptrs, _ = make_state()
    params[15] = index
    library.applyEndQcu(pointer(set_ptrs), pointer(params))
    print(f"UNEXPECTED: applyEndQcu accepted reserved slot {index}")


def child_ordinary_last_slot(library: ctypes.CDLL) -> None:
    """The last ordinary slot (_SET_PTRS_COARSE_BASE_ - 1) must still work."""
    params, argv, set_ptrs, _ = make_state()
    params[15] = 29
    library.applyInitQcu(pointer(set_ptrs), pointer(params), pointer(argv))
    print(f"init_ok={int(set_ptrs[29]) != 0}")
    library.applyEndQcu(pointer(set_ptrs), pointer(params))
    print(f"end_ok={int(set_ptrs[29]) == 0}")


def child_multigrid_verify_levels(library: ctypes.CDLL) -> None:
    """An absurd _MG_NUM_LEVEL_ must not index set_ptrs past its 100 entries."""
    params, argv, set_ptrs, _ = make_state()
    library.applyInitQcu(pointer(set_ptrs), pointer(params), pointer(argv))
    dummy = np.zeros(1, dtype=np.complex64)
    for levels in (5, 18, 19, 21):
        params[17] = levels
        status = library.verifyCloverMultigridQcu(
            pointer(dummy), pointer(dummy), pointer(dummy), pointer(dummy),
            pointer(dummy), pointer(dummy), pointer(dummy), pointer(set_ptrs),
            pointer(params))
        print(f"levels={levels} status={status}")


def run_child(library_path: Path, mode: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update({
        "OMPI_ALLOW_RUN_AS_ROOT": "1",
        "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
        "KUPL_OMP_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    })
    if mode == "multigrid-require-correction":
        environment["KUPL_REQUIRE_MG_CORRECTION"] = "1"
    return subprocess.run(
        [sys.executable, "-B", __file__, "--child", mode,
         "--library", str(library_path)],
        text=True, capture_output=True, timeout=60, check=False,
        env=environment)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument(
        "--child",
        choices=("clover", "multigrid-null", "multigrid-mode",
                 "multigrid-require-correction", "init-reserved-slot",
                 "init-legacy-layout-slot", "end-reserved-slot",
                 "ordinary-last-slot", "multigrid-verify-levels"))
    args = parser.parse_args()
    library = bind_library(args.library)
    if args.child == "clover":
        child_clover_nonconvergence(library)
        return 0
    if args.child == "multigrid-null":
        child_multigrid_null(library)
        return 0
    if args.child == "multigrid-mode":
        child_multigrid_unsupported_mode(library)
        return 0
    if args.child == "multigrid-require-correction":
        child_multigrid_require_correction(library)
        return 0
    if args.child == "init-reserved-slot":
        child_ordinary_slot(30, library)
        return 0
    if args.child == "init-legacy-layout-slot":
        child_ordinary_slot(99, library)
        return 0
    if args.child == "end-reserved-slot":
        child_ordinary_slot_end(30, library)
        return 0
    if args.child == "ordinary-last-slot":
        child_ordinary_last_slot(library)
        return 0
    if args.child == "multigrid-verify-levels":
        child_multigrid_verify_levels(library)
        return 0

    status = library.applyMultigridStrictInitQcu(0, 0, 0, None)
    if status == 0:
        raise AssertionError("strict-MG null result pointer unexpectedly succeeded")
    allocated = ctypes.c_ulonglong(0)
    status = library.applyMultigridStrictInitQcu(
        0, 0, 0, ctypes.byref(allocated))
    if status == 0 or allocated.value != 0:
        raise AssertionError("strict-MG null controls did not fail closed")

    clover = run_child(args.library, "clover")
    if clover.returncode == 0 or "true residual" not in clover.stderr:
        raise AssertionError(
            "Clover non-convergence did not fail closed: "
            f"returncode={clover.returncode} stderr={clover.stderr[-2000:]}")
    null_mg = run_child(args.library, "multigrid-null")
    if null_mg.returncode == 0 or "null pointer" not in null_mg.stderr:
        raise AssertionError(
            "Clover-MG null input did not fail explicitly: "
            f"returncode={null_mg.returncode} "
            f"stderr={null_mg.stderr[-2000:]}")
    mode_mg = run_child(args.library, "multigrid-mode")
    if mode_mg.returncode == 0 or "CUDA-only" not in mode_mg.stderr:
        raise AssertionError(
            "Clover-MG unsupported CUDA mode did not fail explicitly: "
            f"returncode={mode_mg.returncode} "
            f"stderr={mode_mg.stderr[-2000:]}")
    correction_mg = run_child(args.library, "multigrid-require-correction")
    if (correction_mg.returncode == 0 or
            "requires levels>1 and restart>0" not in correction_mg.stderr):
        raise AssertionError(
            "Clover-MG correction requirement did not fail explicitly: "
            f"returncode={correction_mg.returncode} "
            f"stderr={correction_mg.stderr[-2000:]}")

    # N3: ordinary INIT/END own set_ptrs[0, _SET_PTRS_COARSE_BASE_) only; the
    # coarse/strict/layout slots must be refused instead of overwritten.
    for mode, label in (
            ("init-reserved-slot", "INIT at _SET_INDEX_=30"),
            ("init-legacy-layout-slot", "INIT at _SET_INDEX_=99"),
            ("end-reserved-slot", "END at _SET_INDEX_=30")):
        reserved = run_child(args.library, mode)
        if (reserved.returncode == 0 or
                "reserved for multigrid assets" not in reserved.stderr):
            raise AssertionError(
                f"{label} was not rejected: "
                f"returncode={reserved.returncode} "
                f"stderr={reserved.stderr[-2000:]}")
        if "UNEXPECTED" in reserved.stdout:
            raise AssertionError(f"{label} was accepted by the backend")
    last_slot = run_child(args.library, "ordinary-last-slot")
    if (last_slot.returncode != 0 or "init_ok=True" not in last_slot.stdout or
            "end_ok=True" not in last_slot.stdout):
        raise AssertionError(
            "ordinary slot 29 no longer initializes: "
            f"returncode={last_slot.returncode} stdout={last_slot.stdout!r} "
            f"stderr={last_slot.stderr[-2000:]}")

    # N5: _MG_NUM_LEVEL_ beyond the 17 coarse-asset transitions the 100-entry
    # set_ptrs table can describe must be rejected before table[] is read.
    verify_levels = run_child(args.library, "multigrid-verify-levels")
    if verify_levels.returncode != 0:
        raise AssertionError(
            "verifyCloverMultigridQcu crashed on an out-of-range level count: "
            f"returncode={verify_levels.returncode} "
            f"stderr={verify_levels.stderr[-2000:]}")
    for expected in ("levels=5 status=1", "levels=18 status=1",
                     "levels=19 status=2", "levels=21 status=2"):
        if expected not in verify_levels.stdout:
            raise AssertionError(
                f"verifyCloverMultigridQcu clamp lost {expected!r}: "
                f"stdout={verify_levels.stdout!r} "
                f"stderr={verify_levels.stderr[-2000:]}")

    print(
        "status=PASS clover_nonconvergence=abort "
        "multigrid_null=abort multigrid_mode=abort "
        "multigrid_require_correction=abort status_error=1 "
        "ordinary_slot_reserved=abort ordinary_slot_29=ok "
        "multigrid_verify_levels=clamped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
