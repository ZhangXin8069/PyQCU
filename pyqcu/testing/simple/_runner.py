"""Shared process/device helpers for the simple production and benchmark CLIs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[3]


def run_report(argv, env, report_path):
    """Run a backend subprocess and return its JSON report.

    A stale report from a previous invocation must never mask a crashed run,
    so the target is removed first and its absence is treated as failure.
    """
    report_path = Path(report_path)
    if report_path.exists():
        report_path.unlink()
    completed = subprocess.run([str(item) for item in argv], cwd=REPO,
                               env=env, capture_output=True, text=True)
    if completed.returncode != 0 or not report_path.exists():
        tail = (completed.stderr or completed.stdout or "")[-1200:]
        return {"status": "FAIL", "error": tail}
    with report_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def cuda_environment(device: str) -> tuple[dict, str]:
    """Map an explicit physical GPU index onto an isolated logical device.

    Production CUDA programs use one subprocess/MPI rank per device; this helper
    must not be used to construct a single-process multi-device production
    worker. ``MultiGpuMultigrid`` remains an explicit test-only exception.

    ``--device cuda:2`` means PCI bus index 2 unless the caller already set
    ``CUDA_VISIBLE_DEVICES``; in that case the caller's mapping is respected.
    """
    env = dict(os.environ)
    if ":" in device and not env.get("CUDA_VISIBLE_DEVICES"):
        index = device.rsplit(":", 1)[1]
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = index
        device = "cuda:0"
    return env, device


ARCH_LIBRARY_DIRS = {
    "6.0": REPO / "data" / "libqcu-sm60",
    "7.0": REPO / "data" / "libqcu-sm70",
}
QUDA_INSTALL_DIRS = {
    "6.0": REPO / "data" / "quda-sm60-install",
    "7.0": REPO / "data" / "quda-sm70-install",
}
KUPL_LIBRARIES = {
    "default": REPO / "cpp" / "kupl" / "qcu" / "build" / "libqcu.so",
    "lto": REPO / "cpp" / "kupl" / "qcu" / "build-lto" / "libqcu.so",
}


def gpu_compute_capability(physical_index: str = "0") -> str | None:
    """Return ``major.minor`` for a physical GPU, or None if unavailable."""
    try:
        completed = subprocess.run(
            ["nvidia-smi", f"--id={physical_index}",
             "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    lines = [line.strip() for line in completed.stdout.splitlines()
             if line.strip()]
    return lines[0] if lines and lines[0][0].isdigit() else None


def _physical_gpu_index(env: dict, device: str) -> str | None:
    """Translate a (possibly logical) device ordinal to the nvidia-smi index."""
    visible = [item.strip() for item in
               env.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    if ":" in device:
        ordinal = int(device.rsplit(":", 1)[1])
        return visible[ordinal] if 0 <= ordinal < len(visible) else None
    return visible[0] if visible else None


def device_compute_capability(env: dict, device: str) -> str | None:
    return gpu_compute_capability(_physical_gpu_index(env, device) or "0")


def quda_environment(env: dict, device: str, variant: str = "auto") -> dict:
    """Select the QUDA installation matching the physical GPU architecture.

    ``QUDA_INSTALL`` is honored when the caller sets it explicitly; otherwise
    ``data/quda-sm60-install`` / ``data/quda-sm70-install`` are chosen by
    compute capability.  ``default`` leaves the caller's stack alone.
    """
    capability = device_compute_capability(env, device)
    env["QCU_GPU_COMPUTE_CAPABILITY"] = capability or "unknown"
    if variant == "default" or env.get("QUDA_INSTALL"):
        return env
    directory = (QUDA_INSTALL_DIRS.get(capability or "")
                 if variant == "auto" else
                 REPO / "data" / f"quda-{variant}-install")
    if directory is not None and (directory / "lib" / "libquda.so").is_file():
        env["QUDA_INSTALL"] = str(directory)
        env["QCU_QUDA_VARIANT"] = directory.name
    return env


def libqcu_environment(env: dict, device: str,
                       variant: str = "auto") -> dict:
    """Prepend the libqcu build matching the physical GPU architecture.

    The Cython bridge has no RPATH, so the first ``libqcu.so`` on
    ``LD_LIBRARY_PATH`` wins.  ``data/libqcu-sm60`` and ``data/libqcu-sm70``
    contain native SASS builds; ``auto`` selects by compute capability,
    ``sm60``/``sm70`` force a variant for A/B work, and ``default`` keeps the
    caller's stack (usually the sm_60+PTX default) untouched.
    """
    capability = device_compute_capability(env, device)
    env["QCU_GPU_COMPUTE_CAPABILITY"] = capability or "unknown"
    if variant == "default":
        return env
    directory = (ARCH_LIBRARY_DIRS.get(capability or "")
                 if variant == "auto" else
                 REPO / "data" / f"libqcu-{variant}")
    if directory is not None and (directory / "libqcu.so").is_file():
        env["LD_LIBRARY_PATH"] = (str(directory) + ":" +
                                  env.get("LD_LIBRARY_PATH", ""))
        env["QCU_LIBCU_VARIANT"] = directory.name
    return env


def kupl_environment(env: dict, variant: str = "default") -> dict:
    """Select the KUPL ``libqcu.so`` through ``PYQCU_KUPL_LIB``.

    ``default`` keeps the caller's environment (so ``PYQCU_KUPL_LIB`` set by
    the user still wins).  ``lto`` selects the link-time-optimized build in
    ``cpp/kupl/qcu/build-lto``.  An explicit path is accepted as well.  The
    child process passes the variable to every MPI rank.
    """
    if variant in (None, "", "default"):
        return env
    if "/" in variant or variant.endswith(".so"):
        path = Path(variant).expanduser()
    else:
        path = KUPL_LIBRARIES.get(variant)
        if path is None:
            raise ValueError(f"unknown KUPL library variant: {variant}")
    if not path.is_file():
        raise FileNotFoundError(
            f"KUPL library {path} not found; build cpp/kupl/qcu"
            + (" (LTO: build-lto)" if variant == "lto" else ""))
    env["PYQCU_KUPL_LIB"] = str(path)
    env["QCU_KUPL_VARIANT"] = variant if variant in KUPL_LIBRARIES else str(path)
    return env


def gpu_clock_warmup(seconds: float, device: str) -> None:
    """Ramp the GPU clocks; short solves otherwise run at idle clocks."""
    if seconds <= 0:
        return
    import time

    import torch

    torch.cuda.init()
    target = torch.device(device)
    left = torch.randn(1024, 1024, device=target, dtype=torch.float32)
    right = torch.randn(1024, 1024, device=target, dtype=torch.float32)
    started = time.perf_counter()
    while time.perf_counter() - started < seconds:
        for _ in range(8):
            left = left @ right
        norm = left.norm()
        if norm > 0 and torch.isfinite(norm):
            left = left / norm
    torch.cuda.synchronize(target)
    del left, right
    torch.cuda.empty_cache()


def default_kupl_grid(lat, ranks: int) -> tuple[int, int, int, int] | None:
    """Choose a 4D KUPL grid with even local extents.

    KUPL's CHROMA layout needs every local dimension to be even.  Fill the
    spatial y, z, t axes first and leave x unsplit (x is the slowest-varying
    site axis in the compressed layout, so splitting it costs the most in
    packing).  For the reference ``24³×64`` this gives ``1×4×1×1`` for 4
    ranks, ``1×4×2×1`` for 8, and ``1×4×4×1`` for 16; ``--kupl-grid`` still
    overrides the choice for measurements.
    """
    if ranks == 1:
        return (1, 1, 1, 1)
    if any(dim % 2 for dim in lat):
        return None
    remaining = ranks
    grid = [1, 1, 1, 1]
    for axis in (1, 2, 3, 0):
        chosen = 1
        for factor in range(remaining, 1, -1):
            if remaining % factor:
                continue
            if lat[axis] % factor or (lat[axis] // factor) % 2:
                continue
            chosen = factor
            break
        grid[axis] = chosen
        remaining //= chosen
    return tuple(grid) if remaining == 1 else None


def kupl_command(args, grid, lattice, mass: float, csw: float,
                 gauge: Path, source: Path, output, report: Path) -> list:
    script = Path(__file__).resolve().parent / "kupl_multigrid.py"
    return ["mpirun", "-np", int(np.prod(grid)), "--bind-to", "none",
            "--allow-run-as-root", sys.executable, str(script),
            "--lat", *lattice, "--grid", *grid,
            "--mass", mass, "--csw", csw, "--atol", args.atol,
            "--dtype", "c64", "--levels", "1", "--restart", "0",
            "--gauge", gauge, "--source", source, "--seed", args.seed,
            "--report", report] + (["--output", output] if output else [])
