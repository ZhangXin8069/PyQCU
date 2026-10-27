#!/usr/bin/env python3
"""Launch the standalone KUPL C++ operator benchmark as one MPI job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
from collections.abc import Sequence
from pathlib import Path

from pyqcu.testing.simple._common import REPO, write_json
from pyqcu.testing.simple.benchmark_kupl import parse_output, validate_records
from pyqcu.testing.kupl.kunpeng_runtime import read_topology
from pyqcu.testing.simple.mpi._entrypoint import (
    add_mpi_job_options,
    benchmark_environment,
    finalize_mpi_job,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _ranks_per_numa(map_by: str | None) -> int | None:
    if not map_by:
        return None
    match = re.fullmatch(r"ppr:(\d+):(?:numa|node)(?::.*)?", map_by)
    return int(match.group(1)) if match else None


def _local_numa_count() -> int | None:
    count = len(read_topology().nodes)
    return count or None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default=None)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--numa-bind", choices=("off", "library"), default="off")
    parser.add_argument("--cpu-node", type=int, default=None)
    parser.add_argument("--memory-policy", choices=("preferred", "bind"), default=None)
    parser.add_argument("--memory-node", default=None,
                        help="local, N, or local+N; requires --numa-bind library")
    parser.add_argument("--profile", choices=("generic", "kunpeng", "kunpeng-2numa"),
                        default="generic")
    parser.add_argument("--hugepage", action="store_true")
    parser.add_argument("--topology-report", action="store_true")
    parser.add_argument("--allow-oversubscribe", action="store_true")
    parser.add_argument("--fused-bicg-dots", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--disable-clover-cache",
                        action=argparse.BooleanOptionalAction, default=False)
    add_mpi_job_options(
        parser, lattice=(16, 16, 16, 32), ranks=2,
        report="benchmark_kupl_mpi.json")
    args = finalize_mpi_job(parser, parser.parse_args(argv))
    if args.repeats <= 0 or args.iterations <= 0 or args.threads <= 0:
        parser.error("--repeats, --iterations and --threads must be positive")
    if args.numa_bind == "library" and args.bind_to != "none":
        parser.error("--numa-bind=library requires --bind-to none")
    if args.memory_node is not None and args.numa_bind != "library":
        parser.error("--memory-node requires --numa-bind=library")
    if args.memory_policy is not None and args.numa_bind != "library":
        parser.error("--memory-policy requires --numa-bind=library")
    if args.cpu_node is not None and args.numa_bind != "library":
        parser.error("--cpu-node requires --numa-bind=library")
    ranks_per_numa = _ranks_per_numa(args.map_by)
    if args.bind_to == "numa" and ranks_per_numa is not None and ranks_per_numa > 1:
        parser.error(
            "multiple ranks per NUMA with --bind-to numa overlap cpusets; "
            "use --rankfile externally or --numa-bind library --bind-to none")
    numa_count = _local_numa_count() if args.bind_to == "numa" else None
    if args.bind_to == "numa" and numa_count is not None and args.ranks > numa_count:
        parser.error(
            "more MPI ranks than local NUMA domains cannot all use "
            "--bind-to numa; use --numa-bind library --bind-to none")
    binary = (Path(args.binary).expanduser() if args.binary else
              REPO / "cpp/kupl/qcu/build/kupl_qcu_bench")
    module_arguments = [
        "--lattice", *(str(value) for value in args.lat),
        "--grid", *(str(value) for value in args.grid),
        "--iterations", str(args.iterations),
    ]
    command = [str(args.mpirun)]
    if args.allow_run_as_root:
        command.append("--allow-run-as-root")
    if args.oversubscribe:
        command.append("--oversubscribe")
    command.extend(("--bind-to", str(args.bind_to)))
    if args.map_by:
        command.extend(("--map-by", str(args.map_by)))
    command.extend(("-np", str(args.ranks), str(binary), *module_arguments))
    environment = benchmark_environment(updates={
        "KUPL_OMP_THREADS": str(args.threads),
        "OMP_NUM_THREADS": str(args.threads),
        "OMP_DYNAMIC": "FALSE",
        "KUPL_NUMA_BIND": "1" if args.numa_bind == "library" else "0",
    })
    for name in (
        "OMP_PROC_BIND", "OMP_PLACES", "GOMP_CPU_AFFINITY", "KMP_AFFINITY",
        "KUPL_CPU_NODE", "KUPL_REQUIRE_TOPOLOGY", "KUPL_ALLOW_OVERSUBSCRIBE",
        "KUPL_SYSFS_NODE_ROOT", "KUPL_FUSED_BICG_DOTS",
        "KUPL_DISABLE_CLOVER_CACHE",
        "KUPL_MEMORY_POLICY", "KUPL_MEMORY_NODE", "KUPL_REQUIRE_MEMORY_BIND",
        "KUPL_HUGEPAGE", "KUPL_TOPOLOGY_REPORT", "PYQCU_KUPL_PROFILE",
    ):
        environment.pop(name, None)
    if args.profile != "generic":
        environment["PYQCU_KUPL_PROFILE"] = args.profile
    environment["KUPL_FUSED_BICG_DOTS"] = (
        "1" if args.fused_bicg_dots else "0")
    environment["KUPL_DISABLE_CLOVER_CACHE"] = (
        "1" if args.disable_clover_cache else "0")
    if args.cpu_node is not None:
        environment["KUPL_CPU_NODE"] = str(args.cpu_node)
    if args.allow_oversubscribe:
        environment["KUPL_ALLOW_OVERSUBSCRIBE"] = "1"
    if args.memory_policy is not None:
        environment["KUPL_MEMORY_POLICY"] = args.memory_policy
        environment["KUPL_REQUIRE_MEMORY_BIND"] = "1"
    if args.memory_node is not None:
        environment["KUPL_MEMORY_NODE"] = args.memory_node
    if args.hugepage:
        environment["KUPL_HUGEPAGE"] = "1"
    if args.topology_report:
        environment["KUPL_TOPOLOGY_REPORT"] = "1"
    if args.dry_run:
        print(json.dumps({
            "status": "DRY-RUN",
            "mode": "delegated-launcher",
            "ranks": args.ranks,
            "grid": args.grid,
            "command": command,
            "environment": {
                key: environment.get(key) for key in (
                    "KUPL_OMP_THREADS", "OMP_NUM_THREADS", "OMP_DYNAMIC",
                    "KUPL_NUMA_BIND", "KUPL_MEMORY_POLICY", "KUPL_MEMORY_NODE",
                    "KUPL_REQUIRE_MEMORY_BIND", "KUPL_HUGEPAGE",
                    "KUPL_TOPOLOGY_REPORT", "KUPL_CPU_NODE",
                    "KUPL_ALLOW_OVERSUBSCRIBE", "KUPL_FUSED_BICG_DOTS",
                    "KUPL_DISABLE_CLOVER_CACHE",
                    "PYQCU_KUPL_PROFILE")
            },
            "trace_enabled": False,
            "quda_enable_tuning": "0",
            "report": str(args.report),
        }, ensure_ascii=False))
        return 0
    if not binary.is_file():
        output = Path(args.report).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
        write_json(temporary, {
            "status": "SKIP",
            "backend": "kupl",
            "reason": f"missing {binary}",
            "report": str(output),
        })
        os.replace(temporary, output)
        print(json.dumps({
            "status": "SKIP", "reason": f"missing {binary}",
            "report": str(output),
        }))
        return 0

    runs = []
    for repeat in range(args.repeats):
        try:
            completed = subprocess.run(
                command, cwd=REPO, env=environment, text=True,
                capture_output=True, timeout=args.timeout, check=False)
            run = {
                "repeat": repeat + 1,
                "returncode": int(completed.returncode),
                "parsed": parse_output(completed.stdout),
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
            missing, incomplete = validate_records(run["parsed"])
            run["missing_records"] = [list(key) for key in missing]
            run["incomplete_records"] = [list(key) for key in incomplete]
        except subprocess.TimeoutExpired as exc:
            timeout_stdout = exc.stdout or ""
            if isinstance(timeout_stdout, bytes):
                timeout_stdout = timeout_stdout.decode(errors="replace")
            run = {
                "repeat": repeat + 1,
                "returncode": 124,
                "parsed": {},
                "stdout": timeout_stdout,
                "stderr": f"timeout after {args.timeout:g} seconds",
                "missing_records": [["timeout"]],
                "incomplete_records": [],
            }
        runs.append(run)
        if run["returncode"] != 0 or run["missing_records"] or \
                run["incomplete_records"]:
            break
    status = "PASS" if len(runs) == args.repeats and all(
        item["returncode"] == 0 and not item["missing_records"]
        and not item["incomplete_records"] for item in runs) else "FAIL"
    results = {}
    if status == "PASS":
        samples = {}
        for run in runs:
            for key, record in run["parsed"].items():
                samples.setdefault(key, []).append(record)
        for (precision, operator), values in sorted(samples.items()):
            seconds = [float(value["seconds"]) for value in values]
            median = statistics.median(seconds)
            mad = statistics.median(abs(value - median) for value in seconds)
            results[f"{precision}/{operator}"] = {
                "samples": len(seconds),
                "seconds_median": median,
                "seconds_mad": mad,
                "seconds_min": min(seconds),
                "seconds_max": max(seconds),
            }
    report = {
        "status": status,
        "backend": "kupl",
        "binary": str(binary),
        "binary_sha256": _sha256_file(binary),
        "command": command,
        "ranks": args.ranks,
        "grid": args.grid,
        "threads": args.threads,
        "placement": {
            key: environment.get(key) for key in (
                "KUPL_NUMA_BIND", "KUPL_MEMORY_POLICY", "KUPL_MEMORY_NODE",
                "KUPL_REQUIRE_MEMORY_BIND", "KUPL_HUGEPAGE",
                "KUPL_TOPOLOGY_REPORT", "KUPL_CPU_NODE",
                "KUPL_ALLOW_OVERSUBSCRIBE", "KUPL_FUSED_BICG_DOTS",
                "KUPL_DISABLE_CLOVER_CACHE", "PYQCU_KUPL_PROFILE",
                "OMP_NUM_THREADS", "OMP_DYNAMIC", "OMP_PROC_BIND", "OMP_PLACES")
        },
        "trace_enabled": False,
        "quda_enable_tuning": "0",
        "results": results,
        "runs": runs,
    }
    output = Path(args.report).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    write_json(temporary, report)
    os.replace(temporary, output)
    print(json.dumps({
        "status": status,
        "report": str(output),
        "ranks": args.ranks,
        "grid": args.grid,
        "results": results,
    }, ensure_ascii=False))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
