#!/usr/bin/env python3
"""Inspect CPU topology and emit auditable Kunpeng MPI/OpenMP launch plans."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from pyqcu.tools._argparse import BooleanOptionalAction


DEFAULT_NODE_ROOT = Path("/sys/devices/system/node")
DEFAULT_CPUINFO = Path("/proc/cpuinfo")
DEFAULT_MEMINFO = Path("/proc/meminfo")
DEFAULT_NUMA_BALANCING = Path("/proc/sys/kernel/numa_balancing")
DEFAULT_THP = Path("/sys/kernel/mm/transparent_hugepage/enabled")


@dataclass(frozen=True)
class CpuNode:
    node_id: int
    cpus: tuple[int, ...]


@dataclass(frozen=True)
class HostTopology:
    machine: str
    cpu_model: str
    affinity: tuple[int, ...]
    nodes: tuple[CpuNode, ...]
    numa_balancing: bool | None = None
    transparent_hugepage: str | None = None
    memory_total_bytes: int | None = None
    cgroup_memory_bytes: int | None = None

    @property
    def is_kunpeng(self) -> bool:
        model = self.cpu_model.lower()
        machine = self.machine.lower()
        return machine in {"aarch64", "arm64"} and (
            "kunpeng" in model or "huawei" in model
        )

    @property
    def profile(self) -> str:
        if not self.is_kunpeng:
            return "generic"
        if len(self.nodes) == 2:
            return "kunpeng-2numa"
        return "kunpeng"


@dataclass(frozen=True)
class RuntimePlan:
    profile: str
    binding: str
    mpi: str
    ranks: int
    threads_per_rank: int
    ranks_per_numa: int
    environment: dict[str, str]
    mpi_arguments: tuple[str, ...]
    rankfile: str | None = None
    mpi_executable: str = "mpirun"
    mpi_detected: str | None = None
    warnings: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "binding": self.binding,
            "mpi": self.mpi,
            "ranks": self.ranks,
            "threads_per_rank": self.threads_per_rank,
            "ranks_per_numa": self.ranks_per_numa,
            "environment": self.environment,
            "mpi_arguments": list(self.mpi_arguments),
            "rankfile": self.rankfile,
            "mpi_executable": self.mpi_executable,
            "mpi_detected": self.mpi_detected,
            "warnings": list(self.warnings),
            "notes": list(self.notes),
        }

    def shell(self, program: str = "<program>") -> str:
        lines = [
            "unset KUPL_NUMA_BIND KUPL_MEMORY_POLICY KUPL_MEMORY_NODE "
            "KUPL_REQUIRE_MEMORY_BIND KUPL_HUGEPAGE KUPL_TOPOLOGY_REPORT "
            "KUPL_REQUIRE_TOPOLOGY KUPL_ALLOW_OVERSUBSCRIBE KUPL_CPU_NODE "
            "KUPL_FUSED_BICG_DOTS KUPL_DISABLE_CLOVER_CACHE "
            "KUPL_SYSFS_NODE_ROOT "
            "PYQCU_KUPL_PROFILE OMP_NUM_THREADS OMP_DYNAMIC OMP_PROC_BIND "
            "OMP_PLACES GOMP_CPU_AFFINITY KMP_AFFINITY"
        ]
        lines.extend(
            f"export {key}={shlex.quote(value)}"
            for key, value in sorted(self.environment.items()))
        command = [
            self.mpi_executable,
            *self.mpi_arguments,
            "-np",
            str(self.ranks),
            program,
        ]
        lines.append(shlex.join(command))
        return "\n".join(lines)


def parse_cpu_list(text: str) -> tuple[int, ...]:
    cpus: set[int] = set()
    for token in text.strip().split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            first_text, last_text = token.split("-", 1)
            first = int(first_text)
            last = int(last_text)
            if first < 0 or last < first:
                raise ValueError(f"invalid CPU range: {token}")
            cpus.update(range(first, last + 1))
        else:
            cpu = int(token)
            if cpu < 0:
                raise ValueError(f"invalid CPU id: {token}")
            cpus.add(cpu)
    return tuple(sorted(cpus))


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _read_bytes(text: str | None) -> int | None:
    if not text:
        return None
    try:
        value = int(text)
    except ValueError:
        return None
    return None if value < 0 else value


def _read_meminfo(path: Path) -> int | None:
    text = _read_text(path)
    if text is None:
        return None
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            fields = line.split()
            if len(fields) >= 2:
                return int(fields[1]) * 1024 if fields[1].isdigit() else None
    return None


def _read_cgroup_memory() -> int | None:
    for path in (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ):
        text = _read_text(path)
        if text is None or text == "max":
            continue
        value = _read_bytes(text)
        if value is not None and value < (1 << 60):
            return value
    return None


def _read_cpu_model(path: Path) -> str:
    text = _read_text(path)
    if text:
        preferred: dict[str, str] = {}
        for line in text.splitlines():
            if ":" not in line:
                continue
            key, value = (part.strip() for part in line.split(":", 1))
            preferred.setdefault(key.lower(), value)
        for key in ("model name", "hardware", "processor", "cpu model"):
            if preferred.get(key):
                return preferred[key]
    return platform.processor() or "unknown"


def _read_online_nodes(node_root: Path) -> tuple[int, ...]:
    online = _read_text(node_root / "online")
    if online:
        return parse_cpu_list(online)
    result = []
    for path in sorted(node_root.glob("node[0-9]*")):
        if path.name[4:].isdigit() and (path / "cpulist").is_file():
            result.append(int(path.name[4:]))
    return tuple(result)


def read_topology(
    *,
    node_root: Path = DEFAULT_NODE_ROOT,
    cpuinfo_path: Path = DEFAULT_CPUINFO,
    meminfo_path: Path = DEFAULT_MEMINFO,
    numa_balancing_path: Path = DEFAULT_NUMA_BALANCING,
    thp_path: Path = DEFAULT_THP,
    affinity_override: Iterable[int] | None = None,
) -> HostTopology:
    if affinity_override is None:
        if hasattr(os, "sched_getaffinity"):
            affinity_values = os.sched_getaffinity(0)
        else:
            affinity_values = range(os.cpu_count() or 1)
    else:
        affinity_values = affinity_override
    affinity = tuple(sorted(set(affinity_values)))
    nodes = []
    for node_id in _read_online_nodes(node_root):
        text = _read_text(node_root / f"node{node_id}" / "cpulist")
        if not text:
            continue
        cpus = parse_cpu_list(text)
        local = tuple(cpu for cpu in cpus if cpu in set(affinity))
        if local:
            nodes.append(CpuNode(node_id, local))
    numa_text = _read_text(numa_balancing_path)
    numa_balancing = None if numa_text is None else numa_text == "1"
    thp_text = _read_text(thp_path)
    return HostTopology(
        machine=platform.machine(),
        cpu_model=_read_cpu_model(cpuinfo_path),
        affinity=affinity,
        nodes=tuple(nodes),
        numa_balancing=numa_balancing,
        transparent_hugepage=thp_text,
        memory_total_bytes=_read_meminfo(meminfo_path),
        cgroup_memory_bytes=_read_cgroup_memory(),
    )


def _select_profile(topology: HostTopology, requested: str) -> str:
    profile = topology.profile if requested == "auto" else requested
    if profile in {"kunpeng", "kunpeng-2numa"} and not topology.is_kunpeng:
        raise ValueError(f"{profile} profile requested on a non-Kunpeng host")
    if profile == "kunpeng-2numa":
        if len(topology.nodes) != 2:
            raise ValueError(
                f"kunpeng-2numa profile requires 2 NUMA nodes, found {len(topology.nodes)}"
            )
    return profile


def _node_capacity(nodes: tuple[CpuNode, ...], affinity: tuple[int, ...]) -> tuple[CpuNode, ...]:
    if nodes:
        return nodes
    return (CpuNode(0, affinity),) if affinity else ()


def build_plan(
    topology: HostTopology,
    *,
    ranks: int,
    threads_per_rank: int,
    binding: str = "scheduler",
    mpi: str = "auto",
    profile: str = "auto",
    memory_policy: str | None = None,
    memory_node_offset: int | None = None,
    rankfile: str | None = None,
    fused_bicg_dots: bool = True,
    disable_clover_cache: bool = False,
    allow_oversubscribe: bool = False,
    mpi_executable: str = "mpirun",
    mpi_detected: str | None = None,
) -> RuntimePlan:
    if ranks <= 0 or threads_per_rank <= 0:
        raise ValueError("ranks and threads_per_rank must be positive")
    if binding not in {"scheduler", "library", "none"}:
        raise ValueError(f"unsupported binding mode: {binding}")
    if memory_policy not in {None, "preferred", "bind"}:
        raise ValueError(f"unsupported memory policy: {memory_policy}")
    if memory_node_offset is not None and memory_node_offset < 0:
        raise ValueError("memory_node_offset must be non-negative")
    if memory_node_offset is not None and binding != "library":
        raise ValueError("memory_node_offset requires --binding library")
    if memory_policy is not None and binding != "library":
        raise ValueError("memory_policy requires --binding library")
    if rankfile is not None and not rankfile:
        raise ValueError("rankfile must not be empty")
    selected_profile = _select_profile(topology, profile)
    nodes = _node_capacity(topology.nodes, topology.affinity)
    if not nodes:
        raise ValueError("no usable CPU affinity or NUMA topology was detected")
    ranks_per_numa = math.ceil(ranks / len(nodes))
    minimum_cpus = min(len(node.cpus) for node in nodes)
    max_threads = minimum_cpus // ranks_per_numa
    if threads_per_rank > max_threads and not allow_oversubscribe:
        raise ValueError(
            f"{ranks} ranks with {threads_per_rank} threads/rank exceed the "
            f"smallest NUMA CPU pool ({minimum_cpus} CPUs, "
            f"{ranks_per_numa} ranks/node at most {max_threads} threads/rank)"
        )
    if binding == "scheduler" and ranks_per_numa > 1 and rankfile is None:
        raise ValueError(
            "scheduler binding with multiple ranks per NUMA requires --rankfile; "
            "--bind-to numa otherwise gives every rank the full NUMA cpuset"
        )

    mpi_kind = mpi
    if mpi_kind == "auto":
        mpi_kind = "openmpi"
    if mpi_kind not in {"openmpi", "hmpi"}:
        raise ValueError(f"unsupported MPI launcher: {mpi_kind}")

    environment = {
        "KUPL_OMP_THREADS": str(threads_per_rank),
        "OMP_NUM_THREADS": str(threads_per_rank),
        "OMP_DYNAMIC": "FALSE",
        "KUPL_FUSED_BICG_DOTS": "1" if fused_bicg_dots else "0",
        "KUPL_DISABLE_CLOVER_CACHE": "1" if disable_clover_cache else "0",
    }
    warnings: list[str] = []
    if selected_profile != "generic":
        environment["PYQCU_KUPL_PROFILE"] = selected_profile

    map_value = (f"ppr:{ranks_per_numa}:numa" if ranks_per_numa > 1 else "numa")
    if binding == "scheduler":
        environment["KUPL_NUMA_BIND"] = "0"
        mpi_arguments = (("--report-bindings", "--rankfile", rankfile)
                         if rankfile is not None else
                         ("--report-bindings", "--map-by", map_value,
                          "--bind-to", "numa"))
    elif binding == "library":
        environment["KUPL_NUMA_BIND"] = "1"
        mpi_arguments = (("--report-bindings", "--rankfile", rankfile,
                          "--bind-to", "none")
                         if rankfile is not None else
                         ("--report-bindings", "--map-by", map_value,
                          "--bind-to", "none"))
        if mpi_kind == "hmpi":
            warnings.append(
                "Hyper MPI may fall back to generic collectives with "
                "--bind-to none; scheduler binding is the preferred HMPI mode."
            )
        if memory_policy is not None:
            environment["KUPL_MEMORY_POLICY"] = memory_policy
            environment["KUPL_REQUIRE_MEMORY_BIND"] = "1"
        if memory_node_offset is not None:
            environment["KUPL_MEMORY_NODE"] = f"local+{memory_node_offset}"
        notes = (
            "All Python/NumPy tensors must be allocated after applyInitQcu, "
            "or before it with an equivalent NUMA-local cpuset.",
        )
    else:
        environment["KUPL_NUMA_BIND"] = "0"
        mpi_arguments = (("--report-bindings", "--rankfile", rankfile,
                          "--bind-to", "none")
                         if rankfile is not None else
                         ("--report-bindings", "--bind-to", "none"))
        notes = ()

    if allow_oversubscribe:
        environment["KUPL_ALLOW_OVERSUBSCRIBE"] = "1"
        mpi_arguments = ("--oversubscribe", *mpi_arguments)

    if topology.numa_balancing:
        warnings.append(
            "numa_balancing=1; benchmark again with it disabled on a dedicated node."
        )
    if topology.transparent_hugepage is not None and "[never]" in topology.transparent_hugepage:
        warnings.append(
            "Transparent huge pages are disabled; enable MADV_HUGEPAGE support "
            "or treat KUPL_HUGEPAGE=1 as ineffective."
        )
    if binding == "scheduler":
        notes = ("MPI must bind each rank before Python allocates input tensors.",)

    return RuntimePlan(
        profile=selected_profile,
        binding=binding,
        mpi=mpi_kind,
        ranks=ranks,
        threads_per_rank=threads_per_rank,
        ranks_per_numa=ranks_per_numa,
        environment=environment,
        mpi_arguments=mpi_arguments,
        rankfile=rankfile,
        mpi_executable=mpi_executable,
        mpi_detected=mpi_detected,
        warnings=tuple(warnings),
        notes=notes,
    )


def _mpirun_version(mpirun: str = "mpirun") -> str | None:
    try:
        completed = subprocess.run(
            [mpirun, "--version"], text=True, capture_output=True,
            check=False, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (completed.stdout or completed.stderr).strip().splitlines()
    return text[0] if text else None


def _probe_mpi_launcher(mpirun: str) -> tuple[str | None, str | None]:
    version = _mpirun_version(mpirun)
    if version is None:
        return None, None
    text = version.lower()
    if "hyper mpi" in text or "hmpi" in text:
        return "hmpi", version
    if "open mpi" in text or "openrte" in text:
        return "openmpi", version
    return None, version


def _topology_dict(topology: HostTopology,
                   mpirun: str = "mpirun") -> dict[str, object]:
    return {
        "machine": topology.machine,
        "cpu_model": topology.cpu_model,
        "is_kunpeng": topology.is_kunpeng,
        "profile": topology.profile,
        "affinity": list(topology.affinity),
        "nodes": [
            {"node_id": node.node_id, "cpus": list(node.cpus)}
            for node in topology.nodes
        ],
        "numa_balancing": topology.numa_balancing,
        "transparent_hugepage": topology.transparent_hugepage,
        "memory_total_bytes": topology.memory_total_bytes,
        "cgroup_memory_bytes": topology.cgroup_memory_bytes,
        "mpirun": _mpirun_version(mpirun),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-root", type=Path, default=DEFAULT_NODE_ROOT,
                        help=argparse.SUPPRESS)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect = subparsers.add_parser("inspect", help="show host CPU/NUMA topology")
    inspect.add_argument("--json", action="store_true")
    inspect.add_argument("--mpirun", default="mpirun")
    plan = subparsers.add_parser("plan", help="emit a KUPL MPI/OpenMP launch plan")
    plan.add_argument("--ranks", type=int, required=True)
    plan.add_argument("--threads-per-rank", type=int, required=True)
    plan.add_argument("--binding", choices=("scheduler", "library", "none"),
                      default="scheduler")
    plan.add_argument("--mpi", choices=("auto", "openmpi", "hmpi"), default="auto")
    plan.add_argument("--mpirun", default="mpirun")
    plan.add_argument("--allow-launcher-mismatch", action="store_true")
    plan.add_argument("--profile", choices=("auto", "generic", "kunpeng", "kunpeng-2numa"),
                      default="auto")
    plan.add_argument("--memory-policy", choices=("preferred", "bind"), default=None)
    plan.add_argument("--memory-node-offset", type=int, default=None,
                      help="use KUPL_MEMORY_NODE=local+OFFSET for on-chip memory")
    plan.add_argument("--rankfile", default=None,
                      help="use exact launcher slots instead of --map-by")
    plan.add_argument("--fused-bicg-dots", action=BooleanOptionalAction,
                      default=True)
    plan.add_argument("--disable-clover-cache", action=BooleanOptionalAction,
                      default=False)
    plan.add_argument("--allow-oversubscribe", action="store_true")
    plan.add_argument("--program", default="<program>")
    plan.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    topology = read_topology(node_root=args.node_root)
    if args.command == "inspect":
        payload = _topology_dict(topology, args.mpirun)
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"machine: {topology.machine}")
            print(f"cpu model: {topology.cpu_model}")
            print(f"affinity: {list(topology.affinity)}")
            for node in topology.nodes:
                print(f"NUMA {node.node_id}: {list(node.cpus)}")
            print(f"profile: {topology.profile}")
            print(f"numa_balancing: {topology.numa_balancing}")
            print(f"transparent_hugepage: {topology.transparent_hugepage}")
            print(f"mpirun: {_mpirun_version(args.mpirun)}")
        return 0

    try:
        detected_mpi, launcher_version = _probe_mpi_launcher(args.mpirun)
        if args.mpi == "auto":
            if detected_mpi is None:
                raise ValueError(
                    f"cannot identify MPI launcher {args.mpirun!r}: "
                    f"{launcher_version or 'version probe failed'}")
            selected_mpi = detected_mpi
        else:
            if detected_mpi is None:
                raise ValueError(
                    f"cannot identify requested MPI launcher {args.mpirun!r}")
            if detected_mpi != args.mpi and not args.allow_launcher_mismatch:
                raise ValueError(
                    f"requested MPI {args.mpi!r}, but {args.mpirun!r} "
                    f"reports {detected_mpi!r}; pass --allow-launcher-mismatch "
                    "only for an explicit dry-run")
            selected_mpi = args.mpi
        plan = build_plan(
            topology,
            ranks=args.ranks,
            threads_per_rank=args.threads_per_rank,
            binding=args.binding,
            mpi=selected_mpi,
            profile=args.profile,
            memory_policy=args.memory_policy,
            memory_node_offset=args.memory_node_offset,
            rankfile=args.rankfile,
            fused_bicg_dots=args.fused_bicg_dots,
            disable_clover_cache=args.disable_clover_cache,
            allow_oversubscribe=args.allow_oversubscribe,
            mpi_executable=args.mpirun,
            mpi_detected=detected_mpi,
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(plan.as_dict(), indent=2, sort_keys=True))
    else:
        print(plan.shell(args.program))
        for warning in plan.warnings:
            print(f"warning: {warning}", file=sys.stderr)
        for note in plan.notes:
            print(f"note: {note}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
