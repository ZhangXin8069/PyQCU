#!/usr/bin/env python3
"""Validate that KUPL's local-rank binding produces disjoint CPU sets."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


TOPOLOGY = re.compile(
    r"local_rank=(?P<rank>\d+) local_size=(?P<size>\d+) "
    r"numa_nodes=(?P<nodes>\d+) cpu_node=(?P<node>-?\d+) "
    r"cpuset=(?P<cpus>\S+) memory_node=(?P<memory_node>-?\d+) "
    r"memory_policy=(?P<policy>\S+) memory_bound=(?P<bound>true|false) "
    r"threads=(?P<threads>\d+) bound_threads=(?P<bound_threads>\d+) "
    r"source=(?P<source>\S+)"
)


def parse_cpu_set(text: str) -> set[int]:
    result: set[int] = set()
    for token in text.split(","):
        if not token or token == "none":
            continue
        if "-" in token:
            first, last = (int(value) for value in token.split("-", 1))
            result.update(range(first, last + 1))
        else:
            result.add(int(token))
    return result


def format_cpu_list(cpus: list[int]) -> str:
    return ",".join(str(cpu) for cpu in cpus)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--mpirun", default="mpirun")
    parser.add_argument("--ranks", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--synthetic-two-numa", action="store_true",
                        help="test the dual-NUMA branch with a temporary sysfs view")
    parser.add_argument("--cpu-node-map", default=None,
                        help="comma-separated KUPL_CPU_NODE for each local rank")
    args = parser.parse_args()
    if args.ranks <= 0:
        parser.error("--ranks must be positive")
    if args.threads <= 0:
        parser.error("--threads must be positive")
    if args.synthetic_two_numa and args.ranks < 2:
        parser.error("--synthetic-two-numa requires at least two ranks")
    node_map = None
    if args.cpu_node_map is not None:
        try:
            node_map = [int(value) for value in args.cpu_node_map.split(",")]
        except ValueError as error:
            parser.error(f"invalid --cpu-node-map: {error}")
        if len(node_map) != args.ranks:
            parser.error("--cpu-node-map must contain one node per rank")
    if not hasattr(os, "sched_getaffinity"):
        print("SKIPPED: sched_getaffinity is unavailable", flush=True)
        return 77
    try:
        affinity = set(os.sched_getaffinity(0))
    except OSError as error:
        print(f"SKIPPED: cannot query affinity: {error}", flush=True)
        return 77
    required_cpus = args.ranks * args.threads
    if len(affinity) < required_cpus:
        print(
            f"SKIPPED: affinity exposes {len(affinity)} CPUs for "
            f"{args.ranks} ranks x {args.threads} threads",
            flush=True,
        )
        return 77
    try:
        version = subprocess.run(
            [args.mpirun, "--version"], text=True, capture_output=True,
            timeout=10, check=False,
        )
        launcher = (version.stdout or version.stderr).lower()
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"SKIPPED: cannot query MPI launcher: {error}", flush=True)
        return 77
    if not any(marker in launcher for marker in
               ("open mpi", "openrte", "hyper mpi")):
        print("SKIPPED: test requires Open MPI-compatible launcher arguments",
              flush=True)
        return 77

    with tempfile.TemporaryDirectory(prefix="pyqcu_numa_") as temporary:
        environment = os.environ.copy()
        for name in (
            "KUPL_NUMA_BIND", "KUPL_CPU_NODE", "KUPL_MEMORY_POLICY",
            "KUPL_MEMORY_NODE", "KUPL_REQUIRE_MEMORY_BIND",
            "KUPL_REQUIRE_TOPOLOGY", "KUPL_TOPOLOGY_REPORT", "KUPL_HUGEPAGE",
            "KUPL_FUSED_BICG_DOTS", "PYQCU_KUPL_PROFILE", "KUPL_OMP_THREADS",
            "OMP_NUM_THREADS", "OMP_DYNAMIC", "OMP_PROC_BIND", "OMP_PLACES",
            "GOMP_CPU_AFFINITY", "KMP_AFFINITY",
        ):
            environment.pop(name, None)
        environment.update({
            "KUPL_NUMA_BIND": "1",
            "KUPL_MEMORY_POLICY": "default",
            "KUPL_OMP_THREADS": str(args.threads),
            "OMP_NUM_THREADS": str(args.threads),
            "OMP_DYNAMIC": "FALSE",
            "KUPL_TOPOLOGY_REPORT": "1",
        })
        if args.synthetic_two_numa:
            ordered = sorted(affinity)
            split = len(ordered) // 2
            root = Path(temporary)
            (root / "node0").mkdir()
            (root / "node1").mkdir()
            (root / "online").write_text("0-1\n", encoding="utf-8")
            (root / "node0" / "cpulist").write_text(
                format_cpu_list(ordered[:split]) + "\n", encoding="utf-8")
            (root / "node1" / "cpulist").write_text(
                format_cpu_list(ordered[split:]) + "\n", encoding="utf-8")
            environment["KUPL_SYSFS_NODE_ROOT"] = temporary
            environment["KUPL_REQUIRE_TOPOLOGY"] = "1"
        binary_command = [
            str(args.binary),
            "--lattice", str(max(4, 2 * args.ranks)), "2", "2", "4",
            "--grid", str(args.ranks), "1", "1", "1",
            "--iterations", "1",
        ]
        launcher_command = [
            args.mpirun,
            "--allow-run-as-root",
            "--oversubscribe",
            "--bind-to", "none",
            "-np", str(args.ranks),
        ]
        if node_map is None:
            command = [*launcher_command, *binary_command]
        else:
            wrapper = (
                "import os,sys;"
                f"nodes={node_map!r};"
                "rank=int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK',"
                "os.environ.get('PMI_LOCAL_RANK','0')));"
                "os.environ['KUPL_CPU_NODE']=str(nodes[rank]);"
                "os.execv(sys.argv[1],sys.argv[1:])"
            )
            command = [
                *launcher_command, sys.executable, "-B", "-c", wrapper,
                *binary_command,
            ]
        completed = subprocess.run(
            command, cwd=Path(__file__).resolve().parents[3], env=environment,
            text=True, capture_output=True, timeout=args.timeout, check=False,
        )
    if completed.returncode != 0:
        print(completed.stdout, end="")
        print(completed.stderr, end="", file=sys.stderr)
        return completed.returncode

    records = {}
    for match in TOPOLOGY.finditer(completed.stdout + "\n" + completed.stderr):
        record = match.groupdict()
        rank = int(record["rank"])
        records.setdefault(rank, record)
    if len(records) != args.ranks:
        print(f"FAIL: found topology for {len(records)} of {args.ranks} ranks")
        return 1

    cpusets = {}
    cpu_nodes = {}
    occupied: set[int] = set()
    for rank in range(args.ranks):
        record = records[rank]
        if int(record["size"]) != args.ranks:
            print(f"FAIL: rank {rank} local_size={record['size']}")
            return 1
        if int(record["threads"]) != args.threads:
            print(f"FAIL: rank {rank} resolved threads={record['threads']}")
            return 1
        if int(record["bound_threads"]) != args.threads:
            print(
                f"FAIL: rank {rank} bound_threads={record['bound_threads']} "
                f"expected={args.threads}"
            )
            return 1
        cpus = parse_cpu_set(record["cpus"])
        if not cpus:
            print(f"FAIL: rank {rank} has an empty cpu set")
            return 1
        if len(cpus) != args.threads:
            print(
                f"FAIL: rank {rank} cpuset={sorted(cpus)} does not contain "
                f"one CPU per OpenMP thread ({args.threads})"
            )
            return 1
        if occupied & cpus:
            print(f"FAIL: rank {rank} overlaps CPUs {sorted(occupied & cpus)}")
            return 1
        occupied.update(cpus)
        cpusets[rank] = sorted(cpus)
        cpu_nodes[rank] = int(record["node"])

    if args.synthetic_two_numa:
        if set(cpu_nodes.values()) != {0, 1}:
            print(f"FAIL: expected synthetic NUMA nodes 0 and 1, got {cpu_nodes}")
            return 1
        if any(int(records[rank]["nodes"]) != 2 for rank in range(args.ranks)):
            print("FAIL: synthetic topology did not resolve exactly two NUMA nodes")
            return 1
    if node_map is not None and [
            cpu_nodes[rank] for rank in range(args.ranks)] != node_map:
        print(f"FAIL: cpu-node map mismatch expected={node_map} got={cpu_nodes}")
        return 1

    print(
        f"status=PASS ranks={args.ranks} cpusets={cpusets} "
        f"cpu_nodes={cpu_nodes} nodes={records[0]['nodes']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
