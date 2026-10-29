#!/usr/bin/env python3
"""Launch a small MPI QCU Clover-MG solve from a PyQUDA LIME gauge.

The public entry point prepares one small gauge/source pair, optionally builds
the shared 33-tensor coarse hierarchy, then launches the MPI worker. KUPL uses
4D MPI domain decomposition plus OpenMP; CUDA uses one MPI rank per visible
GPU. This is distinct from ``MultiGpuMultigrid``, which replicates a complete
solve in one process and is intended for throughput/consistency studies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from pyqcu.tools._legacy_mg_cache import (
    file_sha256,
    legacy_mg_cache_identity,
    logical_array_sha256,
    validate_legacy_mg_cache,
)

from pyqcu.testing.simple._common import (
    DATA_DIR,
    read_h5_metadata,
    resolve_path,
    write_h5,
    write_json,
)
from pyqcu.testing.simple._runner import libqcu_environment
from pyqcu.testing.simple.mpi._common import (
    DEFAULT_LIME,
    is_lime_file,
    lime_metadata,
    mass_and_csw,
    read_gauge,
    read_source,
    resolve_grid,
    validate_lattice,
)
from pyqcu.testing.simple.mpi._entrypoint import (
    benchmark_environment,
    require_launcher_singleton,
)
from pyqcu.testing.simple.mpi.mpi_ranks_power_of_two import validate_rank_count


REPO = Path(__file__).resolve().parents[4]
BUILD_CACHE = REPO / "pyqcu" / "testing" / "simple" / "build_full_mg_cache.py"


def _input_metadata(path: Path) -> dict[str, Any]:
    if is_lime_file(path):
        return lime_metadata(path)
    if path.suffix.lower() in (".h5", ".hdf5"):
        return read_h5_metadata(path)
    return {}


def _local_numa_count() -> int | None:
    """Return local online NUMA domains without requiring numactl."""
    try:
        text = Path("/sys/devices/system/node/online").read_text(
            encoding="ascii").strip()
    except OSError:
        return None
    count = 0
    try:
        for part in text.split(","):
            if "-" in part:
                start, stop = (int(value) for value in part.split("-", 1))
                count += stop - start + 1
            elif part:
                count += 1
    except ValueError:
        return None
    return count or None


def _case_tag(path: Path, lattice: Sequence[int], mass: float, csw: float,
              seed: int, dtype: str) -> str:
    try:
        stat = path.stat()
        identity_path = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    except OSError:
        identity_path = f"{path.absolute()}:missing-dry-run"
    identity = (
        f"{identity_path}:"
        f"{tuple(lattice)}:{mass:.12g}:{csw:.12g}:{seed}:{dtype}")
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)
    lat = "x".join(str(value) for value in lattice)
    mass_tag = f"{mass:.6g}".replace("-", "m").replace(".", "p")
    return f"{stem}_{lat}_m{mass_tag}_s{seed}_{digest}"


def _materialize_case(args, dtype: np.dtype) -> dict[str, Any]:
    source_input = resolve_path(
        args.gauge or args.lime, Path(args.gauge or args.lime).expanduser())
    if not source_input.is_file():
        raise FileNotFoundError(source_input)
    lattice = validate_lattice(args.lat)

    gauge_full, metadata, gauge_format = read_gauge(source_input, lattice, dtype)
    mass, csw = mass_and_csw(metadata, args.mass, args.csw)
    work_dir = (
        resolve_path(args.work_dir, Path(args.work_dir).expanduser())
        if args.work_dir else
        DATA_DIR / "mpi_simple" / _case_tag(
            source_input, lattice, mass, csw, args.seed, args.dtype))
    work_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = work_dir / "case_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "input": str(source_input),
            "lattice": list(lattice),
            "mass": mass,
            "csw": csw,
            "seed": int(args.seed),
            "dtype": args.dtype,
        }
        if all(existing.get(key) == value for key, value in expected.items()):
            gauge_record = existing.get("gauge", {})
            source_record = existing.get("source", {})
            gauge_existing = Path(str(gauge_record.get("path", "")))
            source_existing = Path(str(source_record.get("path", "")))
            if (gauge_existing.is_file() and source_existing.is_file() and
                    file_sha256(gauge_existing) == gauge_record.get("sha256") and
                    file_sha256(source_existing) == source_record.get("sha256")):
                return {
                    **existing,
                    "work_dir": str(work_dir),
                    "manifest": str(manifest_path),
                    "gauge_path": str(gauge_existing),
                    "source_path": str(source_existing),
                }

    gauge_path = work_dir / "gauge_global_c64.h5" if args.dtype == "c64" else \
        work_dir / "gauge_global_c128.h5"
    if args.dtype == "c128":
        gauge_path = work_dir / "gauge_global_c128.h5"
    write_h5(gauge_path, gauge_full, metadata={
        "backend_input": gauge_format,
        "source_path": str(source_input),
        "lattice": list(lattice),
        "mass": mass,
        "csw": csw,
        "layout": "[color,color,dir,x,y,z,t]",
        "note": "small prefix sub-volume; seams are not periodic",
    })

    if args.source:
        source_input_path = resolve_path(
            args.source, Path(args.source).expanduser())
        if not source_input_path.is_file():
            raise FileNotFoundError(source_input_path)
        source_full, source_metadata, source_format = read_source(
            source_input_path, lattice, dtype)
        source_provenance = str(source_input_path)
    else:
        rng = np.random.default_rng(args.seed)
        source_full = (
            rng.normal(size=(4, 3, *lattice)) +
            1j * rng.normal(size=(4, 3, *lattice))).astype(dtype)
        source_metadata = {"seed": int(args.seed), "generator": "PCG64"}
        source_format = "generated"
        source_provenance = "generated"
    source_path = work_dir / f"source_global_{args.dtype}.h5"
    write_h5(source_path, np.ascontiguousarray(source_full), metadata={
        "source_format": source_format,
        "source_path": source_provenance,
        "seed": int(args.seed),
        "lattice": list(lattice),
        "layout": "[spin,color,x,y,z,t]",
    })

    manifest = {
        "input": str(source_input),
        "input_format": gauge_format,
        "input_metadata": metadata,
        "source_input_format": source_format,
        "source_metadata": source_metadata,
        "lattice": list(lattice),
        "mass": mass,
        "csw": csw,
        "seed": int(args.seed),
        "dtype": args.dtype,
        "gauge": {
            "path": str(gauge_path),
            "sha256": file_sha256(gauge_path),
            "canonical_sha256": logical_array_sha256(gauge_full),
        },
        "source": {"path": str(source_path), "sha256": file_sha256(source_path)},
        "sub_volume_note": (
            "The requested lattice is a prefix sub-volume of the input gauge; "
            "its boundary seams are not periodic."),
    }
    write_json(manifest_path, manifest)
    return {
        **manifest,
        "work_dir": str(work_dir),
        "manifest": str(manifest_path),
        "gauge_path": str(gauge_path),
        "source_path": str(source_path),
    }


def _coarse_cache_path(args, case: dict[str, Any]) -> Path:
    dof = "_".join(str(value) for value in args.dof[:args.levels])
    block = "x".join(str(value) for value in args.mg_grid)
    return Path(case["work_dir"]) / (
        f"coarse_{args.dtype}_lv{args.levels}_E{dof}_mg{block}"
        f"_nvi{args.nv_iters}_s{args.seed}.h5")


def _coarse_cache_identity(args, case: dict[str, Any]) -> dict[str, Any]:
    return legacy_mg_cache_identity(
        gauge_canonical_sha256=case["gauge"]["canonical_sha256"],
        gauge_file_sha256=case["gauge"]["sha256"],
        lattice=args.lat,
        dtype=args.dtype,
        mass=case["mass"],
        csw=case["csw"],
        dof=args.dof[:args.levels],
        mg_grid=args.mg_grid,
        nv_iters=args.nv_iters,
        seed=args.seed,
    )


def _coarse_cache_valid(
        path: Path, identity: dict[str, Any]) -> bool:
    try:
        validate_legacy_mg_cache(path, identity, verify_file_sha256=True)
    except (OSError, ValueError, TypeError):
        return False
    return True


def _build_coarse_cache(args, case: dict[str, Any], cache_path: Path) -> None:
    report = cache_path.with_suffix(".json")
    builder_arguments = [
        sys.executable, str(BUILD_CACHE),
        "--gauge", case["gauge_path"],
        "--lat", *(str(value) for value in args.lat),
        "--mass", str(case["mass"]),
        "--csw", str(case["csw"]),
        "--dtype", args.dtype,
        "--dof", *(str(value) for value in args.dof[:args.levels]),
        "--mg-grid", *(str(value) for value in args.mg_grid),
        "--nv-iters", str(args.nv_iters),
        "--seed", str(args.seed),
        "--build-device", args.build_device,
        "--batch-size", str(args.build_batch_size),
        "--grid", *(str(value) for value in args.grid),
        "--output", str(cache_path),
        "--report", str(report),
    ]
    command = build_mpirun_command(
        mpirun=args.mpirun, ranks=args.ranks, bind_to=args.bind_to,
        map_by=args.map_by, oversubscribe=args.oversubscribe,
        allow_run_as_root=args.allow_run_as_root,
        worker_arguments=builder_arguments)
    completed = subprocess.run(command, cwd=REPO, check=False)
    if completed.returncode != 0 or not cache_path.is_file():
        raise RuntimeError(
            f"coarse-cache build failed with exit={completed.returncode}: "
            f"{' '.join(command)}")


def _worker_arguments(args, case: dict[str, Any], cache_path: Path | None,
                      grid: Sequence[int], output: Path, report: Path) -> list[str]:
    values = [
        sys.executable, "-m", "pyqcu.testing.simple.mpi.worker",
        "--backend", args.backend,
        "--torch-device", args.torch_device,
        "--kupl-legacy-layout", args.kupl_legacy_layout,
        "--gauge", case["gauge_path"],
        "--source", case["source_path"],
        "--lat", *(str(value) for value in args.lat),
        "--grid", *(str(value) for value in grid),
        "--mass", str(case["mass"]),
        "--csw", str(case["csw"]),
        "--atol", str(args.atol),
        "--max-iter", str(args.max_iter),
        "--omp-threads", str(args.omp_threads),
        "--dtype", args.dtype,
        "--levels", str(args.levels),
        "--min-size", str(args.min_size),
        "--dof", *(str(value) for value in args.dof[:args.levels]),
        "--mg-grid", *(str(value) for value in args.mg_grid),
        "--restart", str(args.restart),
        "--coarse-max-iter", str(args.coarse_max_iter),
        "--coarse-tol-factor", str(args.coarse_tol_factor),
        "--nv-iters", str(args.nv_iters),
        "--seed", str(args.seed),
        "--warmup", str(args.warmup),
        "--repeat", str(args.repeat),
        "--output", str(output),
        "--report", str(report),
    ]
    if cache_path is not None:
        values.extend(["--coarse-cache", str(cache_path)])
    for level_cache in args.coarse_cache_level:
        values.extend(["--coarse-cache-level", str(level_cache)])
    if args.skip_independent_residual:
        values.append("--skip-independent-residual")
    return values


def build_mpirun_command(*, mpirun: str, ranks: int, bind_to: str,
                         map_by: str | None, oversubscribe: bool,
                         allow_run_as_root: bool,
                         worker_arguments: Sequence[str]) -> list[str]:
    command = [mpirun]
    if allow_run_as_root:
        command.append("--allow-run-as-root")
    if oversubscribe:
        command.append("--oversubscribe")
    command.extend(["--bind-to", bind_to])
    if map_by:
        command.extend(["--map-by", map_by])
    command.extend(["-np", str(int(ranks)), *worker_arguments])
    return command


def _launch_environment(args) -> dict[str, str]:
    env = benchmark_environment()
    python_path = env.get("PYTHONPATH", "")
    entries = [str(REPO), *[item for item in python_path.split(":") if item]]
    env["PYTHONPATH"] = ":".join(dict.fromkeys(entries))
    env["OMP_NUM_THREADS"] = str(args.omp_threads)
    env["KUPL_OMP_THREADS"] = str(args.omp_threads)
    if args.backend == "kupl":
        for name in (
            "OMP_DYNAMIC", "OMP_PROC_BIND", "OMP_PLACES",
            "GOMP_CPU_AFFINITY", "KMP_AFFINITY", "KUPL_NUMA_BIND",
            "KUPL_CPU_NODE", "KUPL_REQUIRE_TOPOLOGY",
            "KUPL_ALLOW_OVERSUBSCRIBE", "KUPL_SYSFS_NODE_ROOT",
            "KUPL_FUSED_BICG_DOTS", "KUPL_DISABLE_CLOVER_CACHE",
            "KUPL_MEMORY_POLICY", "KUPL_MEMORY_NODE",
            "KUPL_REQUIRE_MEMORY_BIND", "KUPL_HUGEPAGE",
            "KUPL_TOPOLOGY_REPORT", "PYQCU_KUPL_PROFILE",
        ):
            env.pop(name, None)
        env["KUPL_NUMA_BIND"] = "1" if args.kupl_numa_bind == "library" else "0"
        env["OMP_DYNAMIC"] = "FALSE"
        env["KUPL_FUSED_BICG_DOTS"] = "1" if args.kupl_fused_bicg_dots else "0"
        env["KUPL_DISABLE_CLOVER_CACHE"] = (
            "1" if args.kupl_disable_clover_cache else "0")
        if args.kupl_cpu_node is not None:
            env["KUPL_CPU_NODE"] = str(args.kupl_cpu_node)
        if args.kupl_allow_oversubscribe:
            env["KUPL_ALLOW_OVERSUBSCRIBE"] = "1"
        if args.kupl_profile != "generic":
            env["PYQCU_KUPL_PROFILE"] = args.kupl_profile
        if args.kupl_memory_policy is not None:
            env["KUPL_MEMORY_POLICY"] = args.kupl_memory_policy
            env["KUPL_REQUIRE_MEMORY_BIND"] = "1"
        if args.kupl_memory_node is not None:
            env["KUPL_MEMORY_NODE"] = args.kupl_memory_node
        if args.kupl_hugepage:
            env["KUPL_HUGEPAGE"] = "1"
        if args.kupl_topology_report:
            env["KUPL_TOPOLOGY_REPORT"] = "1"
    elif args.backend == "torch":
        for name in (
            "OMP_PROC_BIND", "OMP_PLACES", "GOMP_CPU_AFFINITY",
            "KMP_AFFINITY", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS",
        ):
            env.pop(name, None)
        threads = str(args.omp_threads)
        env["OMP_DYNAMIC"] = "FALSE"
        env["MKL_NUM_THREADS"] = threads
        env["OPENBLAS_NUM_THREADS"] = threads
        env["NUMEXPR_NUM_THREADS"] = threads
        env["TORCH_NUM_THREADS"] = threads
    elif args.backend == "cuda":
        env = libqcu_environment(env, "cuda:0", args.libqcu)
    return env


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=("kupl", "cuda", "torch"),
                        default="kupl")
    parser.add_argument("--lime", default=str(DEFAULT_LIME),
                        help="PyQUDA/QIO LIME gauge; default: ~/data/<reference>.lime")
    parser.add_argument("--gauge", default=None,
                        help="LIME or HDF5 gauge; overrides --lime")
    parser.add_argument("--source", default=None,
                        help="optional HDF5 source [spin,color,x,y,z,t]")
    parser.add_argument("--lat", nargs=4, type=int, default=(8, 8, 8, 16),
                        help="small validation lattice; sub-volume of the input")
    parser.add_argument("--mass", type=float, default=None)
    parser.add_argument("--csw", type=float, default=None)
    parser.add_argument("--atol", type=float, default=1.0e-6)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--dtype", choices=("c64", "c128"), default="c64")
    parser.add_argument("--levels", type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument("--min-size", type=int, default=4)
    parser.add_argument("--dof", nargs="+", type=int, default=(12, 12))
    parser.add_argument("--mg-grid", nargs=4, type=int, default=(2, 2, 2, 2))
    parser.add_argument("--restart", type=int, default=5)
    parser.add_argument("--coarse-max-iter", type=int, default=200)
    parser.add_argument("--coarse-tol-factor", type=float, default=1.0e4)
    parser.add_argument("--nv-iters", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--grid", nargs=4, type=int, default=None)
    parser.add_argument("--omp-threads", type=int, default=8)
    parser.add_argument("--bind-to", choices=("core", "numa", "none"),
                        default=None)
    parser.add_argument("--map-by", default=None)
    parser.add_argument("--kupl-numa-bind", choices=("scheduler", "library"),
                        default="scheduler")
    parser.add_argument("--kupl-cpu-node", type=int, default=None,
                        help="requires --kupl-numa-bind library")
    parser.add_argument("--kupl-fused-bicg-dots",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--kupl-disable-clover-cache",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--kupl-legacy-layout",
                        choices=("auto", "odd-schur", "full-site"),
                        default="auto")
    parser.add_argument("--kupl-allow-oversubscribe", action="store_true")
    parser.add_argument("--kupl-memory-policy",
                        choices=("preferred", "bind"), default=None)
    parser.add_argument("--kupl-memory-node", default=None,
                        help="local, N, or local+N; requires --kupl-numa-bind library")
    parser.add_argument("--kupl-profile",
                        choices=("generic", "kunpeng", "kunpeng-2numa"),
                        default="generic")
    parser.add_argument("--kupl-hugepage", action="store_true")
    parser.add_argument("--kupl-topology-report", action="store_true")
    parser.add_argument("--oversubscribe", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--allow-run-as-root", action=argparse.BooleanOptionalAction,
                        default=None)
    parser.add_argument("--mpirun", default="mpirun")
    parser.add_argument("--libqcu", choices=("auto", "sm60", "sm70", "default"),
                        default="auto")
    parser.add_argument("--torch-device", choices=("auto", "cpu", "cuda"),
                        default="auto")
    parser.add_argument("--build-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--build-batch-size", type=int, default=4)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--coarse-cache", default=None)
    parser.add_argument("--coarse-cache-level", action="append", default=[])
    parser.add_argument("--rebuild-coarse-cache", action="store_true")
    parser.add_argument("--skip-independent-residual", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    require_launcher_singleton(parser)

    args.lat = list(validate_lattice(args.lat))
    if len(args.dof) < args.levels:
        parser.error("--dof must provide one value per requested level")
    if args.backend == "kupl" and args.levels > 2:
        parser.error("the current KUPL Legacy benchmark supports at most 2 levels")
    if args.levels > 2 and args.coarse_cache is None:
        parser.error("levels > 2 require --coarse-cache")
    if len(args.coarse_cache_level) < max(0, args.levels - 2):
        parser.error(
            "levels > 2 require one repeated --coarse-cache-level per "
            "additional coarse transition")
    if args.levels < 1 or args.omp_threads < 1:
        parser.error("--levels and --omp-threads must be positive")
    if args.min_size < 2:
        parser.error("--min-size must be at least 2")
    if args.build_batch_size < 1:
        parser.error("--build-batch-size must be positive")
    if args.backend == "kupl" and args.kupl_numa_bind == "library" and \
            args.bind_to not in (None, "none"):
        parser.error("--kupl-numa-bind library requires --bind-to none")
    if args.backend == "kupl" and args.kupl_numa_bind != "library" and (
            args.kupl_memory_policy is not None or args.kupl_memory_node is not None):
        parser.error("KUPL memory policy/node requires --kupl-numa-bind library")
    if args.backend == "kupl" and args.kupl_cpu_node is not None and \
            args.kupl_numa_bind != "library":
        parser.error("--kupl-cpu-node requires --kupl-numa-bind library")
    if args.backend != "kupl" and (
            args.kupl_numa_bind != "scheduler" or args.kupl_profile != "generic" or
            args.kupl_memory_policy is not None or args.kupl_memory_node is not None or
            args.kupl_cpu_node is not None or not args.kupl_fused_bicg_dots or
            args.kupl_disable_clover_cache or args.kupl_allow_oversubscribe or
            args.kupl_hugepage or args.kupl_topology_report):
        parser.error("KUPL runtime options require --backend kupl")
    if args.backend != "torch" and args.torch_device != "auto":
        parser.error("--torch-device requires --backend torch")
    if (args.backend != "kupl" and
            args.kupl_legacy_layout != "auto"):
        parser.error("--kupl-legacy-layout requires --backend kupl")
    try:
        args.ranks = validate_rank_count(args.ranks)
    except ValueError as exc:
        parser.error(str(exc))
    if args.restart < 0 or args.max_iter < 1 or args.nv_iters < 1:
        parser.error("--restart must be nonnegative and iteration counts positive")
    if args.warmup < 0 or args.repeat < 1:
        parser.error("--warmup must be nonnegative and --repeat must be positive")
    if not np.isfinite(args.atol) or args.atol <= 0.0:
        parser.error("--atol must be finite and positive")
    args.grid = list(resolve_grid(args.lat, args.ranks, args.grid))
    if any(args.lat[axis] % args.grid[axis] for axis in range(4)):
        parser.error("--grid must divide every lattice axis")
    if args.levels > 1 and args.restart != 0:
        local = [args.lat[axis] // args.grid[axis] for axis in range(4)]
        if any(local[axis] % args.mg_grid[axis] for axis in range(4)):
            parser.error(
                f"local lattice {tuple(local)} is not divisible by "
                f"--mg-grid {tuple(args.mg_grid)}")
    numa_count = _local_numa_count() if args.backend == "kupl" else None
    args.binding_fallback = False
    if args.bind_to is None:
        if args.backend == "kupl" and args.kupl_numa_bind == "library":
            args.bind_to = "none"
        elif args.backend == "kupl" and numa_count is not None and \
                args.ranks <= numa_count:
            args.bind_to = "numa"
        elif (args.backend == "torch" and args.torch_device == "cpu" and
              args.omp_threads == 1):
            args.bind_to = "core"
        else:
            args.bind_to = "none"
            args.binding_fallback = args.backend == "kupl"
    if args.map_by is None:
        if args.backend == "kupl" and args.bind_to == "numa":
            args.map_by = "ppr:1:numa"
        elif args.backend == "kupl" and args.kupl_numa_bind == "library":
            args.map_by = "ppr:1:numa"
        elif (args.backend == "torch" and args.torch_device == "cpu" and
              args.bind_to == "core" and args.omp_threads == 1):
            args.map_by = "slot:PE=1"
        else:
            args.map_by = None
    if (args.backend == "torch" and args.torch_device == "cpu" and
            args.bind_to == "core" and args.omp_threads != 1):
        parser.error(
            "--bind-to core currently requires --omp-threads 1 for Torch CPU")
    if (args.backend == "kupl" and args.bind_to == "numa" and
            args.map_by and "pe=" in args.map_by):
        parser.error(
            "OpenMPI rejects --bind-to numa with map-by pe=N; use "
            "--bind-to core for pe=N, or --map-by ppr:1:numa with NUMA binding")
    map_ppr = re.fullmatch(
        r"ppr:(\d+):(?:numa|node)(?::.*)?", args.map_by or "")
    if (args.backend == "kupl" and args.bind_to == "numa" and map_ppr and
            int(map_ppr.group(1)) > 1):
        parser.error(
            "multiple ranks per NUMA with --bind-to numa overlap cpusets; "
            "use --kupl-numa-bind library --bind-to none or an explicit rankfile")
    if (args.backend == "kupl" and args.bind_to == "numa" and
            numa_count is not None and args.ranks > numa_count):
        parser.error(
            "more MPI ranks than local NUMA domains cannot all use "
            "--bind-to numa; use --kupl-numa-bind library --bind-to none")
    if args.allow_run_as_root is None:
        args.allow_run_as_root = hasattr(os, "geteuid") and os.geteuid() == 0
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    dtype = np.dtype(np.complex64 if args.dtype == "c64" else np.complex128)

    if args.dry_run:
        input_path = resolve_path(
            args.gauge or args.lime, Path(args.gauge or args.lime).expanduser())
        metadata = _input_metadata(input_path) if input_path.is_file() else {}
        mass, csw = mass_and_csw(metadata, args.mass, args.csw)
        work_dir = resolve_path(
            args.work_dir, Path(args.work_dir).expanduser()
        ) if args.work_dir else (
            DATA_DIR / "mpi_simple" / _case_tag(
                input_path, args.lat, mass, csw, args.seed, args.dtype))
        output_default = work_dir / f"solution_mpi_{args.dtype}.h5"
        report_default = work_dir / "report_mpi.json"
        output = resolve_path(args.output, output_default) if args.output else \
            output_default
        report = resolve_path(args.report, report_default) if args.report else \
            report_default
        placeholder_case = {"work_dir": str(work_dir)}
        cache = (
            resolve_path(args.coarse_cache, Path(args.coarse_cache).expanduser())
            if args.coarse_cache else
            _coarse_cache_path(args, placeholder_case)
            if args.levels > 1 and args.restart else None)
        placeholder = {
            "gauge_path": str(work_dir / f"gauge_global_{args.dtype}.h5"),
            "source_path": str(work_dir / f"source_global_{args.dtype}.h5"),
            "mass": mass,
            "csw": csw,
        }
        worker = _worker_arguments(
            args, placeholder, cache, args.grid, output, report)
        command = build_mpirun_command(
            mpirun=args.mpirun, ranks=args.ranks, bind_to=args.bind_to,
            map_by=args.map_by, oversubscribe=args.oversubscribe,
            allow_run_as_root=args.allow_run_as_root,
            worker_arguments=worker)
        launch_environment = _launch_environment(args)
        print(json.dumps({
            "status": "DRY-RUN", "backend": args.backend,
            "lime": str(input_path), "lattice": args.lat, "grid": args.grid,
            "ranks": args.ranks, "omp_threads": args.omp_threads,
            "bind_to": args.bind_to, "map_by": args.map_by,
            "binding_fallback": args.binding_fallback,
            "trace_enabled": False,
            "quda_enable_tuning": "0",
            "command": command,
            "environment": {
                key: launch_environment.get(key) for key in (
                    "KUPL_OMP_THREADS", "OMP_NUM_THREADS", "OMP_DYNAMIC",
                    "KUPL_NUMA_BIND", "KUPL_MEMORY_POLICY", "KUPL_MEMORY_NODE",
                    "KUPL_REQUIRE_MEMORY_BIND", "KUPL_HUGEPAGE",
                    "KUPL_TOPOLOGY_REPORT", "KUPL_CPU_NODE",
                    "KUPL_ALLOW_OVERSUBSCRIBE", "KUPL_FUSED_BICG_DOTS",
                    "KUPL_DISABLE_CLOVER_CACHE",
                    "PYQCU_KUPL_PROFILE", "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                    "TORCH_NUM_THREADS")
            },
        }, indent=2, ensure_ascii=False))
        return 0

    case = _materialize_case(args, dtype)
    cache_path: Path | None = None
    if args.levels > 1 and args.restart != 0:
        expected_cache_identity = _coarse_cache_identity(args, case)
        cache_path = (
            resolve_path(args.coarse_cache, Path(args.coarse_cache).expanduser())
            if args.coarse_cache else _coarse_cache_path(args, case))
        if args.rebuild_coarse_cache or not _coarse_cache_valid(
                cache_path, expected_cache_identity):
            _build_coarse_cache(args, case, cache_path)

    output = (
        resolve_path(args.output, Path(args.output).expanduser())
        if args.output else Path(case["work_dir"]) / f"solution_mpi_{args.dtype}.h5")
    report = (
        resolve_path(args.report, Path(args.report).expanduser())
        if args.report else Path(case["work_dir"]) / "report_mpi.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    run_tag = f"{os.getpid()}-{time.time_ns()}"
    staged_output = output.with_name(f".{output.name}.{run_tag}.partial")
    staged_report = report.with_name(f".{report.name}.{run_tag}.partial")

    worker = _worker_arguments(
        args, case, cache_path, args.grid, staged_output, staged_report)
    command = build_mpirun_command(
        mpirun=args.mpirun, ranks=args.ranks, bind_to=args.bind_to,
        map_by=args.map_by, oversubscribe=args.oversubscribe,
        allow_run_as_root=args.allow_run_as_root,
        worker_arguments=worker)
    env = _launch_environment(args)
    print(json.dumps({
        "stage": "launch", "backend": args.backend,
        "command": command, "lattice": args.lat, "grid": args.grid,
        "omp_threads": args.omp_threads, "bind_to": args.bind_to,
        "map_by": args.map_by, "binding_fallback": args.binding_fallback,
    }, ensure_ascii=False), flush=True)
    completed = subprocess.run(command, cwd=REPO, env=env, check=False)
    if not staged_report.is_file():
        try:
            staged_output.unlink()
        except FileNotFoundError:
            pass
        failure = {
            "status": "FAIL",
            "backend": args.backend,
            "exit_code": completed.returncode,
            "error": "worker did not publish a JSON report",
        }
        write_json(staged_report, failure)
        os.replace(staged_report, report)
        print(json.dumps({
            "status": "FAIL", "backend": args.backend,
            "exit_code": completed.returncode,
            "error": "worker did not publish a JSON report",
            "report": str(report),
        }, ensure_ascii=False))
        return 1 if completed.returncode == 0 else completed.returncode

    payload = json.loads(staged_report.read_text(encoding="utf-8"))
    publish_output = bool(
        completed.returncode == 0 and payload.get("status") not in ("FAIL", "SKIP")
        and staged_output.is_file())
    if not publish_output:
        try:
            staged_output.unlink()
        except FileNotFoundError:
            pass
    payload["launcher"] = {
        "exit_code": completed.returncode,
        "command": command,
        "lime": case["input"],
        "input_format": case["input_format"],
        "case_manifest": case["manifest"],
        "small_lattice": args.lat,
        "sub_volume_note": case["sub_volume_note"],
        "output_published": publish_output,
        "output": str(output) if publish_output else None,
    }
    if completed.returncode != 0 and payload.get("status") not in ("FAIL", "SKIP"):
        payload["status"] = "FAIL"
        payload["converged"] = False
    write_json(staged_report, payload)
    if publish_output:
        os.replace(staged_output, output)
    os.replace(staged_report, report)
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if completed.returncode == 0 and payload.get("status") != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
