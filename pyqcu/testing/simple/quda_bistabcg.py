#!/usr/bin/env python3
"""Minimal QUDA/PyQUDA Clover BiStabCG example using QDP host layouts."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import re
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

try:
    from ._common import (DEFAULT_ATOL, DEFAULT_MASS,
                          DATA_DIR, configure_quda_runtime, convergence_status,
                          cuda_memory_snapshot, lime_metadata, qcu_gauge_to_qdp,
                          qcu_fermion_to_qdp, qdp_fermion_to_qcu, read_h5,
                          read_h5_metadata, read_qdp_gauge, resolve_csw,
                          resolve_path, write_h5, write_json)
except ImportError:
    from _common import (DATA_DIR, DEFAULT_ATOL, DEFAULT_MASS,
                         configure_quda_runtime, convergence_status,
                         cuda_memory_snapshot, lime_metadata, qcu_gauge_to_qdp,
                         qcu_fermion_to_qdp, qdp_fermion_to_qcu, read_h5,
                         read_h5_metadata, read_qdp_gauge, resolve_csw,
                         resolve_path, write_h5, write_json)

if str(Path(__file__).resolve().parents[3]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def pyquda_available() -> bool:
    return _pyquda_api() is not None


def _pyquda_api() -> str | None:
    configure_quda_runtime(preload=True)
    try:
        import torch
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            import pyquda_utils.core  # noqa: F401
            from pyquda.field import LatticeFermion, LatticeGauge  # noqa: F401
        return "pyquda_utils" if torch.cuda.is_available() else None
    except Exception:
        pass
    try:
        import torch
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            import pyquda.core  # noqa: F401
            from pyquda.field import LatticeFermion, LatticeGauge  # noqa: F401
        return "pyquda" if torch.cuda.is_available() else None
    except Exception:
        return None


def _numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "get"):
        return np.asarray(value.get())
    return np.asarray(value)


def _scalar(value, default=None):
    if value is None:
        return default
    if isinstance(value, (list, tuple, np.ndarray)):
        return value[0] if len(value) else default
    return value


def _cupy_memory_snapshot() -> dict[str, Any]:
    try:
        import cupy as cp
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        pool = cp.get_default_memory_pool()
        return {
            "available": True,
            "backend": "cupy",
            "pool_used_bytes": int(pool.used_bytes()),
            "pool_total_bytes": int(pool.total_bytes()),
            "device_free_bytes": int(free_bytes),
            "device_total_bytes": int(total_bytes),
        }
    except Exception:
        return {"available": False, "backend": "cupy"}


def _residual_history(text: str) -> list[dict[str, float]]:
    """Parse QUDA's native verbose solver lines without inventing values."""
    number = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    patterns = (
        re.compile(r"(?i)BiCGstab[^\n]*?(\d+)\s+iterations?[^\n]*?"
                   r"(?:residual|true_res|\|r\|)[^0-9+-]*" + number),
        re.compile(r"(?i)GCR[^\n]*?(\d+)\s+iterations?[^\n]*?"
                   r"(?:residual|true_res|\|r\|)[^0-9+-]*" + number),
        re.compile(r"(?i)iter(?:ation)?s?\s*[=:]\s*(\d+)[^\n]*?"
                   r"(?:residual|true_res|\|r\|)[^0-9+-]*" + number),
    )
    rows: dict[int, float] = {}
    for line in text.splitlines():
        for pattern in patterns:
            match = pattern.search(line)
            if match:
                rows[int(match.group(1))] = float(match.group(2))
                break
    return [{"iteration": iteration, "residual_norm": rows[iteration]}
            for iteration in sorted(rows)]


def _identity_qdp(lat):
    x, y, z, t = lat
    eye = np.eye(3, dtype=np.complex128)
    return np.broadcast_to(eye, (4, t, z, y, x, 3, 3)).copy()


def _load_gauge(args):
    if args.gauge is None:
        return _identity_qdp(tuple(args.lat)), resolve_csw(args.csw), {}
    path = resolve_path(args.gauge, resolve_path(None, args.gauge))
    if path.suffix == ".lime":
        meta = lime_metadata(path)
        return (read_qdp_gauge(path),
                resolve_csw(args.csw, meta),
                meta)
    if path.suffix in (".h5", ".hdf5"):
        canonical = read_h5(path)
        meta = read_h5_metadata(path)
        return (qcu_gauge_to_qdp(canonical),
                resolve_csw(args.csw, meta), meta)
    array = np.load(path, allow_pickle=False)
    if array.ndim == 7 and array.shape[0] == 4 and array.shape[-2:] == (3, 3):
        return array, resolve_csw(args.csw), {}
    raise ValueError("--gauge must be .lime, QDP .npy, or QCU .h5")


def _load_source(args, lat):
    if args.source is None:
        rng = np.random.default_rng(args.seed)
        canonical = (rng.normal(size=(4, 3, *lat)) +
                     1j * rng.normal(size=(4, 3, *lat))).astype(np.complex64)
    else:
        canonical = read_h5(resolve_path(args.source, resolve_path(None, args.source)))
    return canonical, np.ascontiguousarray(np.transpose(canonical, (5, 4, 3, 2, 0, 1)))


def make_dirac(info, mass, tol, max_iter, csw, multigrid=None):
    from pyquda_utils import core

    if math.isclose(float(csw), 0.0, rel_tol=0.0, abs_tol=1.0e-15):
        return core.getWilson(info, mass, tol, max_iter, multigrid=multigrid)
    kwargs = {"clover_csw_t": csw}
    if multigrid is not None:
        kwargs["multigrid"] = multigrid
    try:
        return core.getClover(info, mass, tol, max_iter, **kwargs)
    except TypeError:
        kwargs.pop("clover_csw_t", None)
        kwargs["clover_coeff_t"] = csw
        return core.getClover(info, mass, tol, max_iter, **kwargs)


def _solve_case_old(args, multigrid=None, nvec=None):
    """PyQuda 0.3.x compatibility path (cupy + pyquda.core)."""
    import cupy as cp
    configure_quda_runtime(preload=True)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import pyquda
        import pyquda.core as core
        from pyquda.enum_quda import QudaInverterType, QudaMassNormalization
        from pyquda.field import LatticeFermion, LatticeGauge, cb2

    lat = tuple(args.lat)
    pyquda.init([1, 1, 1, 1])
    qdp_gauge, csw, gauge_meta = _load_gauge(args)
    canonical_source, _ = _load_source(args, lat)
    mass = (float(args.mass) if args.mass is not None
            else float(gauge_meta.get(
                "reference_mass", gauge_meta.get("mass", DEFAULT_MASS))))
    gauge_data = cb2(qdp_gauge.astype(np.complex128), [1, 2, 3, 4])
    gauge = LatticeGauge(list(lat), cp.asarray(gauge_data))
    source = LatticeFermion(list(lat), cp.asarray(qcu_fermion_to_qdp(canonical_source)))
    old_multigrid = None
    if multigrid is not None:
        old_multigrid = [list(block) for block in multigrid]
        old_multigrid.append([2 * value for value in old_multigrid[-1]])
    if math.isclose(float(csw), 0.0, rel_tol=0.0, abs_tol=1.0e-15):
        dirac = core.getWilson(
            list(lat), mass, args.tol, args.max_iter, multigrid=old_multigrid)
    else:
        dirac = core.getDslash(
            list(lat), mass, args.tol, args.max_iter, clover_coeff_t=csw,
            anti_periodic_t=False, multigrid=old_multigrid)
    dirac.invert_param.mass_normalization = (
        QudaMassNormalization.QUDA_KAPPA_NORMALIZATION)
    try:
        dirac.invert_param.use_init_guess = 0
    except Exception:
        pass
    try:
        dirac.invert_param.use_resident_solution = 0
    except Exception:
        pass
    if multigrid is None:
        dirac.invert_param.inv_type = QudaInverterType.QUDA_BICGSTAB_INVERTER
    nvec_applied = False
    if multigrid is not None and nvec is not None and hasattr(dirac, "mg_param"):
        try:
            dirac.mg_param.n_vec = [int(nvec)] * len(dirac.mg_param.n_vec)
            nvec_applied = True
        except Exception:
            pass
    try:
        dirac.loadGauge(gauge)
        stdout = io.StringIO()
        stderr = io.StringIO()
        wall_samples = []
        solver_samples = []
        iteration_samples = []
        solution = None
        for repeat in range(args.warmup + args.repeats):
            started = time.perf_counter()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                solution = dirac.invert(source)
            cp.cuda.runtime.deviceSynchronize()
            wall = time.perf_counter() - started
            invert = dirac.invert_param
            iteration_count = int(_scalar(getattr(invert, "iter", 0), 0))
            if repeat >= args.warmup:
                wall_samples.append(wall)
                solver_samples.append(
                    float(_scalar(getattr(invert, "secs", 0.0), 0.0)))
                iteration_samples.append(iteration_count)
        wall_seconds = median(wall_samples)
        solver_seconds = median(solver_samples)
        iterations = iteration_samples[-1]
        memory = _cupy_memory_snapshot()
        invert = dirac.invert_param
        metrics = {
            "api": "pyquda-0.3",
            "nvec_applied": nvec_applied if multigrid is not None else None,
            "nvec_note": ("old PyQuda API uses its library multigrid defaults"
                          if multigrid is not None and not nvec_applied else None),
            "mass": mass,
            "mass_normalization": "kappa",
            "iterations": iterations,
            "wall_seconds": wall_seconds,
            "solver_seconds": solver_seconds,
            "wall_samples": wall_samples,
            "solver_samples": solver_samples,
            "iteration_samples": iteration_samples,
            "warmup": args.warmup,
            "seconds_per_iteration": (solver_seconds / iterations
                                      if iterations > 0 else None),
            "gflops": float(_scalar(getattr(invert, "gflops", 0.0), 0.0)),
            "gflops_source": "QudaInvertParam.gflops",
            "true_residual": float(_scalar(getattr(invert, "true_res", np.nan), np.nan)),
            "residual_history": [],
            "residual_history_source": "old PyQuda API does not expose captured per-iteration history",
            "memory": memory,
        }
        solution_qcu = qdp_fermion_to_qcu(cp.asnumpy(solution.data).astype(np.complex128))
        if args.output:
            output = resolve_path(args.output, resolve_path(None, args.output))
            write_h5(output, solution_qcu, metadata={
                "backend": "quda", "mass": mass, "csw": csw,
                "layout": "[spin,color,x,y,z,t]",
            })
        return solution_qcu, csw, canonical_source, metrics
    finally:
        try:
            dirac.destroy()
        except Exception:
            pass


def solve_case(args, multigrid=None, nvec=None):
    if _pyquda_api() == "pyquda":
        return _solve_case_old(args, multigrid=multigrid, nvec=nvec)
    configure_quda_runtime(preload=True)
    import torch
    import pyquda
    import pyquda_utils.core as core
    from pyquda.enum_quda import QudaMassNormalization
    from pyquda.field import LatticeFermion, LatticeGauge

    lat = tuple(args.lat)
    pyquda.init(grid_size=[1, 1, 1, 1], latt_size=list(lat), backend="torch",
                backend_target="cuda", enable_nvshmem=False, enable_tuning=False,
                enable_device_memory_pool=False, enable_pinned_memory_pool=False)
    try:
        info = core.LatticeInfo(list(lat), 1, 1.0)
    except TypeError:
        info = core.LatticeInfo(list(lat))
    qdp_gauge, csw, gauge_meta = _load_gauge(args)
    mass = (float(args.mass) if args.mass is not None
            else float(gauge_meta.get(
                "reference_mass", gauge_meta.get("mass", DEFAULT_MASS))))
    canonical_source, qdp_source_full = _load_source(args, lat)
    gauge = LatticeGauge(
        info, 4,
        torch.from_numpy(info.evenodd(qdp_gauge.astype(np.complex128), True)).to("cuda"))
    source = LatticeFermion(
        info,
        torch.from_numpy(info.evenodd(qdp_source_full.astype(np.complex128), False)).to("cuda"))
    dirac = make_dirac(info, mass, args.tol, args.max_iter, csw, multigrid)
    try:
        dirac.invert_param.mass_normalization = (
            QudaMassNormalization.QUDA_KAPPA_NORMALIZATION)
        # A repeated solve must not inherit the previous solution from the
        # caching allocator; force a cold start and disable residency.
        try:
            from pyquda.enum_quda import QudaUseInitGuess
            dirac.invert_param.use_init_guess = (
                QudaUseInitGuess.QUDA_USE_INIT_GUESS_NO)
        except Exception:
            pass
        dirac.invert_param.use_resident_solution = 0
        mg_inv_param = getattr(
            getattr(dirac, "multigrid", None), "inv_param", None)
        if mg_inv_param is not None:
            mg_inv_param.mass_normalization = (
                QudaMassNormalization.QUDA_KAPPA_NORMALIZATION)
        nvec_applied = False
        if nvec is not None:
            mg_obj = getattr(dirac, "multigrid", None)
            mg_param = getattr(mg_obj, "param", None)
            if mg_param is not None:
                # PyQUDA array getters return copies; assign the whole column.
                mg_param.n_vec = [int(nvec)] * len(mg_param.n_vec)
                nvec_applied = True
        try:
            from pyquda.enum_quda import QudaVerbosity
            dirac.setVerbosity(QudaVerbosity.QUDA_VERBOSE)
        except Exception:
            pass
        dirac.loadGauge(gauge)
        stdout = io.StringIO()
        stderr = io.StringIO()
        wall_samples = []
        solver_samples = []
        iteration_samples = []
        solution = None
        for repeat in range(args.warmup + args.repeats):
            started = time.perf_counter()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                solution = dirac.invert(source)
            torch.cuda.synchronize()
            wall = time.perf_counter() - started
            invert = dirac.invert_param
            iterations = int(_scalar(getattr(invert, "iter", 0), 0))
            solver_seconds = float(_scalar(getattr(invert, "secs", 0.0), 0.0))
            if repeat >= args.warmup:
                wall_samples.append(wall)
                solver_samples.append(solver_seconds)
                iteration_samples.append(iterations)
        wall_seconds = median(wall_samples)
        solver_seconds = median(solver_samples)
        iterations = iteration_samples[-1]
        memory = cuda_memory_snapshot(torch.cuda.current_device())
        invert = dirac.invert_param
        metrics = {
            "api": "pyquda_utils",
            "nvec_applied": nvec_applied if multigrid is not None else None,
            "mass": mass,
            "mass_normalization": "kappa",
            "iterations": iterations,
            "wall_seconds": wall_seconds,
            "solver_seconds": solver_seconds,
            "wall_samples": wall_samples,
            "solver_samples": solver_samples,
            "iteration_samples": iteration_samples,
            "warmup": args.warmup,
            "seconds_per_iteration": solver_seconds / iterations if iterations > 0 else None,
            "gflops": float(_scalar(getattr(invert, "gflops", 0.0), 0.0)),
            "gflops_source": "QudaInvertParam.gflops",
            "true_residual": float(_scalar(getattr(invert, "true_res", np.nan), np.nan)),
            "residual_history": _residual_history(stdout.getvalue() + stderr.getvalue()),
            "residual_history_source": "QUDA native verbose stdout/stderr",
            "memory": memory,
        }
        solution_qdp = _numpy(solution.data).astype(np.complex128)
        solution_qcu = qdp_fermion_to_qcu(solution_qdp)
        if args.output:
            output = resolve_path(args.output, resolve_path(None, args.output))
            write_h5(output, solution_qcu, metadata={
                "backend": "quda", "mass": mass, "csw": csw,
                "layout": "[spin,color,x,y,z,t]",
            })
        return solution_qcu, csw, canonical_source, metrics
    finally:
        for name in ("freeGauge", "destroy", "close", "end"):
            method = getattr(dirac, name, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lat", nargs=4, type=int, default=(8, 8, 16, 16))
    parser.add_argument("--mass", type=float, default=None,
                        help="quark mass; defaults to LIME Mass or 0.05")
    parser.add_argument("--tol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--csw", type=float, default=None)
    parser.add_argument("--gauge", default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1,
                        help="untimed solves before sampling")
    parser.add_argument("--repeats", type=int, default=3,
                        help="timed solves; report the median")
    parser.add_argument("--output", default=None)
    parser.add_argument("--report", default="performance_quda_bistabcg.json")
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeats < 1:
        parser.error("--warmup must be >= 0 and --repeats >= 1")
    if not pyquda_available():
        print(json.dumps({"status": "SKIP", "reason": "PyQUDA/CUDA unavailable"}))
        return 0
    solution, csw, _, metrics = solve_case(args)
    converged, status, residual_acceptance = convergence_status(
        float(metrics.get("true_residual", math.nan)), args.tol,
        dtype=np.complex128)
    report = {
        "backend": "quda", "solver": "Dirac.invert",
        "lattice": list(args.lat), "csw": csw,
        "solution_shape": list(solution.shape),
        "residual_acceptance": residual_acceptance,
        "converged": converged, "status": status, **metrics,
    }
    report_path = resolve_path(args.report, DATA_DIR / args.report)
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
