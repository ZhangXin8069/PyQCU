#!/usr/bin/env python3
"""Inspect a KUPL host and write a production-oriented environment report.

The probe is deliberately MPI-free: it never imports ``mpi4py`` and never
starts ``mpirun``.  It combines Linux sysfs/cgroup data with the existing
Kunpeng topology reader, identifies separately numbered memory-only NUMA
nodes, and emits conservative KUPL variables plus commands for a real
Legacy-MG solve and the optimized parity-Schur BiStabCG path.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

try:
    from .kunpeng_runtime import (
        DEFAULT_CPUINFO,
        DEFAULT_MEMINFO,
        DEFAULT_NODE_ROOT,
        DEFAULT_NUMA_BALANCING,
        DEFAULT_THP,
        CpuNode,
        HostTopology,
        parse_cpu_list,
        read_topology,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from kunpeng_runtime import (  # type: ignore[no-redef]
        DEFAULT_CPUINFO,
        DEFAULT_MEMINFO,
        DEFAULT_NODE_ROOT,
        DEFAULT_NUMA_BALANCING,
        DEFAULT_THP,
        CpuNode,
        HostTopology,
        parse_cpu_list,
        read_topology,
    )


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOG_DIR = REPO_ROOT / "logs"
DEFAULT_KUPL_LIBRARY = REPO_ROOT / "cpp/kupl/qcu/libqcu.so"
REPORT_PREFIX = "kupl_env-v"

KUPL_RUNTIME_KEYS = (
    "KUPL_OMP_THREADS",
    "KUPL_NUMA_BIND",
    "KUPL_CPU_NODE",
    "KUPL_MEMORY_NODE",
    "KUPL_MEMORY_POLICY",
    "KUPL_REQUIRE_MEMORY_BIND",
    "KUPL_REQUIRE_TOPOLOGY",
    "KUPL_ALLOW_OVERSUBSCRIBE",
    "KUPL_TOPOLOGY_REPORT",
    "KUPL_HUGEPAGE",
    "KUPL_FUSED_BICG_DOTS",
    "KUPL_DISABLE_CLOVER_CACHE",
    "PYQCU_KUPL_PROFILE",
    "PYQCU_KUPL_LIB",
)
OMP_PLACEMENT_KEYS = (
    "OMP_NUM_THREADS",
    "OMP_DYNAMIC",
    "OMP_PROC_BIND",
    "OMP_PLACES",
    "GOMP_CPU_AFFINITY",
    "KMP_AFFINITY",
)
UNSET_FOR_LIBRARY_BINDING = (
    "KUPL_CPU_NODE",
    "OMP_PROC_BIND",
    "OMP_PLACES",
    "GOMP_CPU_AFFINITY",
    "KMP_AFFINITY",
)


@dataclass(frozen=True)
class MemoryNode:
    node_id: int
    cpus: tuple[int, ...]
    mem_total_bytes: int | None
    mem_free_bytes: int | None
    distances: tuple[tuple[int, int], ...]

    def distance_to(self, node_id: int) -> int | None:
        return dict(self.distances).get(node_id)


@dataclass(frozen=True)
class HbmMapping:
    cpu_node: int
    memory_node: int
    distance: int
    offset: int


@dataclass(frozen=True)
class HbmAssessment:
    status: str
    candidate_nodes: tuple[int, ...]
    mappings: tuple[HbmMapping, ...]
    offset: int | None
    message: str

    @property
    def has_unique_offset(self) -> bool:
        return self.status == "detected_offset" and self.offset is not None


@dataclass(frozen=True)
class Recommendation:
    ranks: int
    threads_per_rank: int
    ranks_per_numa: int
    grid: tuple[int, int, int, int]
    local_lattice: tuple[int, int, int, int]
    environment: dict[str, str]
    mpi_arguments: tuple[str, ...]
    hbm_guaranteed: bool
    warnings: tuple[str, ...]


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _parse_kib_meminfo(text: str | None) -> dict[str, int]:
    result: dict[str, int] = {}
    if text is None:
        return result
    for line in text.splitlines():
        fields = line.replace(":", " ").split()
        if len(fields) >= 2 and fields[0].startswith("Mem") and fields[1].isdigit():
            result[fields[0]] = int(fields[1]) * 1024
    return result


def _online_node_ids(node_root: Path) -> tuple[int, ...]:
    online = _read_text(node_root / "online")
    if online:
        return parse_cpu_list(online)
    node_ids = []
    for path in sorted(node_root.glob("node[0-9]*")):
        suffix = path.name.removeprefix("node")
        if suffix.isdigit() and path.is_dir():
            node_ids.append(int(suffix))
    return tuple(node_ids)


def read_memory_nodes(
    node_root: Path = DEFAULT_NODE_ROOT,
    *,
    affinity: Iterable[int] | None = None,
) -> tuple[MemoryNode, ...]:
    """Read every online NUMA node, including nodes with an empty CPU list."""
    node_ids = _online_node_ids(node_root)
    allowed = set(affinity) if affinity is not None else None
    result = []
    for node_id in node_ids:
        directory = node_root / f"node{node_id}"
        cpu_text = _read_text(directory / "cpulist")
        try:
            cpus = parse_cpu_list(cpu_text or "")
        except ValueError:
            cpus = ()
        if allowed is not None:
            cpus = tuple(cpu for cpu in cpus if cpu in allowed)
        meminfo = _parse_kib_meminfo(_read_text(directory / "meminfo"))
        distance_text = _read_text(directory / "distance")
        distances: tuple[tuple[int, int], ...] = ()
        if distance_text:
            values = [int(value) for value in distance_text.split() if value.isdigit()]
            distances = tuple(zip(node_ids, values))
        result.append(MemoryNode(
            node_id=node_id,
            cpus=cpus,
            mem_total_bytes=meminfo.get("MemTotal"),
            mem_free_bytes=meminfo.get("MemFree"),
            distances=distances,
        ))
    return tuple(result)


def assess_onchip_memory(
    cpu_nodes: Sequence[CpuNode],
    memory_nodes: Sequence[MemoryNode],
) -> HbmAssessment:
    """Find a consistent CPU-NUMA to memory-only-NUMA mapping.

    A memory-only node is a candidate for Kunpeng on-chip/HBM memory, not
    proof of its hardware class.  The recommendation is emitted only when
    sysfs distances select one candidate per CPU node and the offsets are
    identical and positive (the form required by ``KUPL_MEMORY_NODE=local+N``).
    """
    candidates = tuple(
        node.node_id for node in memory_nodes
        if not node.cpus and (node.mem_total_bytes or 0) > 0)
    if not candidates:
        return HbmAssessment(
            status="no_candidate",
            candidate_nodes=(),
            mappings=(),
            offset=None,
            message="未发现无 CPU 且有内存的独立 NUMA 节点；没有可见 HBM/片上内存证据。",
        )

    records = {node.node_id: node for node in memory_nodes}
    mappings: list[HbmMapping] = []
    ambiguous: list[str] = []
    missing_distance: list[int] = []
    for cpu_node in cpu_nodes:
        record = records.get(cpu_node.node_id)
        if record is None:
            missing_distance.append(cpu_node.node_id)
            continue
        distances: dict[int, int] = {}
        for candidate in candidates:
            distance = record.distance_to(candidate)
            if distance is not None:
                distances[candidate] = distance
        if not distances:
            missing_distance.append(cpu_node.node_id)
            continue
        nearest = min(distances.values())
        selected = [node_id for node_id, distance in distances.items()
                    if distance == nearest]
        if len(selected) != 1:
            ambiguous.append(
                f"CPU node {cpu_node.node_id} 最近候选 {selected} 距离同为 {nearest}")
            continue
        memory_node = selected[0]
        mappings.append(HbmMapping(
            cpu_node=cpu_node.node_id,
            memory_node=memory_node,
            distance=nearest,
            offset=memory_node - cpu_node.node_id,
        ))

    if ambiguous:
        return HbmAssessment(
            status="ambiguous",
            candidate_nodes=candidates,
            mappings=tuple(mappings),
            offset=None,
            message="存在片上内存候选，但最近节点映射不唯一：" + "；".join(ambiguous),
        )
    if missing_distance or len(mappings) != len(cpu_nodes):
        return HbmAssessment(
            status="mapping_unavailable",
            candidate_nodes=candidates,
            mappings=tuple(mappings),
            offset=None,
            message=(
                "存在片上内存候选，但缺少完整的 distance 映射；"
                f"无法确认 CPU nodes {missing_distance or 'unknown'}。"
            ),
        )

    offsets = {mapping.offset for mapping in mappings}
    if len(offsets) == 1 and next(iter(offsets)) > 0:
        offset = next(iter(offsets))
        return HbmAssessment(
            status="detected_offset",
            candidate_nodes=candidates,
            mappings=tuple(mappings),
            offset=offset,
            message=(
                f"检测到一致的 memory-only NUMA 映射，可建议 "
                f"KUPL_MEMORY_NODE=local+{offset}；仍需用实际页分布复核。"
            ),
        )
    if len(cpu_nodes) == 1 and len(mappings) == 1:
        return HbmAssessment(
            status="detected_absolute",
            candidate_nodes=candidates,
            mappings=tuple(mappings),
            offset=None,
            message=(
                f"单 CPU NUMA 可映射到 memory-only node "
                f"{mappings[0].memory_node}；多 rank 可显式使用该绝对节点号。"
            ),
        )
    return HbmAssessment(
        status="ambiguous",
        candidate_nodes=candidates,
        mappings=tuple(mappings),
        offset=None,
        message=(
            "存在 memory-only NUMA 节点，但 CPU 到内存节点 offset 不一致；"
            "不能使用单一的 local+N 环境变量。"
        ),
    )


def _default_grid(
    lattice: tuple[int, int, int, int], ranks: int,
) -> tuple[int, int, int, int] | None:
    if ranks == 1:
        return (1, 1, 1, 1)
    if any(extent % 2 for extent in lattice):
        return None
    remaining = ranks
    grid = [1, 1, 1, 1]
    for axis in (1, 2, 3, 0):
        chosen = 1
        for factor in range(remaining, 1, -1):
            if remaining % factor:
                continue
            if lattice[axis] % factor or (lattice[axis] // factor) % 2:
                continue
            chosen = factor
            break
        grid[axis] = chosen
        remaining //= chosen
    return tuple(grid) if remaining == 1 else None


def _local_shape(
    lattice: tuple[int, int, int, int],
    grid: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    return tuple(lattice[axis] // grid[axis] for axis in range(4))


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1024.0
    return f"{value} B"


def _complex_bytes(dtype: str) -> int:
    return 8 if dtype == "c64" else 16


def workload_sizes(
    lattice: tuple[int, int, int, int],
    grid: tuple[int, int, int, int],
    dtype: str,
) -> dict[str, int]:
    local = _local_shape(lattice, grid)
    half_sites = math.prod(local) // 2
    element = _complex_bytes(dtype)
    fermion = 2 * 4 * 3 * half_sites * element
    gauge = 2 * 3 * 3 * 4 * half_sites * element
    clover_parity = 4 * 3 * 4 * 3 * half_sites * element
    return {
        "local_sites": math.prod(local),
        "local_half_sites": half_sites,
        "complex_bytes": element,
        "gauge_bytes": gauge,
        "fermion_vector_bytes": fermion,
        "clover_parity_bytes": clover_parity,
        # A clearly labelled static scale, not a measured process peak.
        "core_static_scale_bytes": gauge + 2 * clover_parity + 12 * fermion,
    }


def tuning_candidates(
    topology: HostTopology,
    lattice: tuple[int, int, int, int],
    requested_ranks: int | None = None,
) -> tuple[dict[str, object], ...]:
    """Return a small rank/thread/grid A/B shortlist, not an optimum claim."""
    nodes = topology.nodes or ((CpuNode(0, topology.affinity),)
                               if topology.affinity else ())
    if not nodes:
        return ()
    node_count = len(nodes)
    rank_values = {node_count, 2 * node_count, 4 * node_count}
    if requested_ranks:
        rank_values.add(requested_ranks)
    result = []
    for ranks in sorted(rank_values):
        ranks_per_numa = math.ceil(ranks / node_count)
        threads = min(len(node.cpus) for node in nodes) // ranks_per_numa
        grid = _default_grid(lattice, ranks)
        if threads < 1 or grid is None:
            continue
        result.append({
            "ranks": ranks,
            "threads_per_rank": threads,
            "ranks_per_numa": ranks_per_numa,
            "grid": list(grid),
        })
    return tuple(result)


def build_recommendation(
    topology: HostTopology,
    memory_nodes: Sequence[MemoryNode],
    hbm: HbmAssessment,
    *,
    ranks: int | None = None,
    threads_per_rank: int | None = None,
    lattice: tuple[int, int, int, int] = (24, 24, 24, 64),
    dtype: str = "c64",
    grid: tuple[int, int, int, int] | None = None,
    mpi: str | None = None,
    sysfs_available: bool | None = None,
) -> Recommendation:
    if sysfs_available is None:
        sysfs_available = bool(
            memory_nodes and any(node.cpus for node in memory_nodes))
    usable_nodes = topology.nodes or (
        (CpuNode(0, topology.affinity),) if topology.affinity else ())
    if not usable_nodes:
        raise ValueError("当前 affinity 和 sysfs 均未提供可用 CPU NUMA 域")

    selected_ranks = int(ranks or len(usable_nodes))
    if selected_ranks <= 0:
        raise ValueError("ranks 必须为正数")
    ranks_per_numa = math.ceil(selected_ranks / len(usable_nodes))
    max_threads = min(len(node.cpus) for node in usable_nodes) // ranks_per_numa
    if max_threads < 1:
        raise ValueError("rank 数超过可用 CPU 容量，无法给出非超订建议")
    default_threads = max_threads if sysfs_available else min(max_threads, 32)
    selected_threads = int(threads_per_rank or default_threads)
    if selected_threads <= 0:
        raise ValueError("threads-per-rank 必须为正数")

    selected_grid = grid or _default_grid(lattice, selected_ranks)
    if selected_grid is None:
        raise ValueError(f"{selected_ranks} ranks 无法形成偶数 local lattice")
    if math.prod(selected_grid) != selected_ranks:
        raise ValueError(
            f"grid {selected_grid} 的乘积不等于 ranks {selected_ranks}")
    if any(lattice[axis] % selected_grid[axis] for axis in range(4)):
        raise ValueError(f"lattice {lattice} 不能被 grid {selected_grid} 整除")
    local = _local_shape(lattice, selected_grid)
    if any(extent % 2 for extent in local):
        raise ValueError(f"KUPL 要求每一维 local extent 为偶数，得到 {local}")

    warnings: list[str] = []
    if selected_threads > max_threads:
        warnings.append(
            f"--threads-per-rank={selected_threads} 超过每 rank 可独占的 "
            f"{max_threads} CPUs；会产生超订，不建议生产使用。")
    if mpi == "hmpi":
        warnings.append(
            "HMPI 下 --bind-to none 可能令部分集合算法回退；"
            "请将本建议与 scheduler/rankfile 方案做同题 A/B。")
    if topology.numa_balancing:
        warnings.append("numa_balancing=1，会引入页迁移和噪声；性能验收应在专用节点关闭后复测。")

    environment = {
        "KUPL_OMP_THREADS": str(selected_threads),
        "OMP_NUM_THREADS": str(selected_threads),
        "OMP_DYNAMIC": "FALSE",
        "KUPL_TOPOLOGY_REPORT": "1",
        "KUPL_FUSED_BICG_DOTS": "1",
        "KUPL_DISABLE_CLOVER_CACHE": "0",
        "KUPL_ALLOW_OVERSUBSCRIBE": "0",
    }
    if topology.is_kunpeng and sysfs_available:
        environment["PYQCU_KUPL_PROFILE"] = topology.profile

    hbm_guaranteed = False
    if not sysfs_available:
        environment["KUPL_NUMA_BIND"] = "0"
        warnings.append(
            "未读取到可用 NUMA sysfs；不要设置 KUPL_MEMORY_NODE 或要求库内绑定，"
            "先修正容器挂载/调度器 cpuset。")
    elif hbm.status == "detected_offset" and hbm.offset is not None:
        environment["KUPL_NUMA_BIND"] = "1"
        environment["KUPL_REQUIRE_TOPOLOGY"] = "1"
        environment["KUPL_MEMORY_NODE"] = f"local+{hbm.offset}"
        environment["KUPL_MEMORY_POLICY"] = "bind"
        environment["KUPL_REQUIRE_MEMORY_BIND"] = "1"
        hbm_guaranteed = True
    elif hbm.status == "detected_absolute" and hbm.mappings:
        environment["KUPL_NUMA_BIND"] = "1"
        environment["KUPL_REQUIRE_TOPOLOGY"] = "1"
        environment["KUPL_MEMORY_NODE"] = str(hbm.mappings[0].memory_node)
        environment["KUPL_MEMORY_POLICY"] = "bind"
        environment["KUPL_REQUIRE_MEMORY_BIND"] = "1"
        hbm_guaranteed = True
    else:
        environment["KUPL_NUMA_BIND"] = "1"
        environment["KUPL_REQUIRE_TOPOLOGY"] = "1"
        environment["KUPL_MEMORY_NODE"] = "local"
        environment["KUPL_MEMORY_POLICY"] = "bind"
        environment["KUPL_REQUIRE_MEMORY_BIND"] = "1"
        warnings.append(
            "未形成唯一 HBM 映射；安全建议是绑定本地 DDR NUMA，"
            "不能把该配置表述为“已使用 HBM”。")

    if hbm_guaranteed and hbm.mappings:
        memory_ids = {mapping.memory_node for mapping in hbm.mappings}
        free_values = [
            node.mem_free_bytes for node in memory_nodes
            if node.node_id in memory_ids and node.mem_free_bytes is not None
        ]
        static_scale = workload_sizes(lattice, selected_grid, dtype)[
            "core_static_scale_bytes"]
        required = static_scale * ranks_per_numa
        if not free_values:
            warnings.append(
                "HBM 候选节点缺少 MemFree；严格 bind 前必须实测目标 rank 峰值。")
        elif min(free_values) < required:
            warnings.append(
                f"HBM 最小 MemFree {_format_bytes(min(free_values))} 小于静态尺度 "
                f"{_format_bytes(required)}；严格 bind 可能 OOM，需减少 rank 或改回本地 DDR。")

    mpi_arguments = ["--report-bindings"]
    if sysfs_available:
        mpi_arguments.extend(["--map-by", f"ppr:{ranks_per_numa}:numa"])
    mpi_arguments.extend(["--bind-to", "none", "-np", str(selected_ranks)])
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        mpi_arguments.insert(0, "--allow-run-as-root")
    return Recommendation(
        ranks=selected_ranks,
        threads_per_rank=selected_threads,
        ranks_per_numa=ranks_per_numa,
        grid=selected_grid,
        local_lattice=local,
        environment=environment,
        mpi_arguments=tuple(mpi_arguments),
        hbm_guaranteed=hbm_guaranteed,
        warnings=tuple(warnings),
    )


def _command_output(command: Sequence[str], timeout: float = 5.0) -> str | None:
    try:
        completed = subprocess.run(
            list(command), text=True, capture_output=True, check=False,
            timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (completed.stdout or completed.stderr).strip()
    return text or None


def _tool_versions(probe_tools: bool) -> dict[str, object]:
    names = ("mpirun", "mpicc", "mpicxx", "cmake", "g++", "numactl", "numastat")
    paths = {name: shutil.which(name) for name in names}
    result: dict[str, object] = {"paths": paths, "versions": {}}
    if not probe_tools:
        return result
    versions = result["versions"]
    assert isinstance(versions, dict)
    for name in names:
        path = paths[name]
        if not path:
            continue
        if name == "numastat":
            versions[name] = path
            continue
        versions[name] = _command_output([path, "--version"])
    return result


def _python_package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _read_os_release() -> dict[str, str]:
    text = _read_text(Path("/etc/os-release")) or ""
    result = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key] = value.strip().strip('"')
    return result


def _read_cgroup_cpu() -> dict[str, object]:
    result: dict[str, object] = {
        "v2_cpu_max": _read_text(Path("/sys/fs/cgroup/cpu.max")),
        "v2_cpuset_effective": _read_text(
            Path("/sys/fs/cgroup/cpuset.cpus.effective")),
        "v1_quota_us": _read_text(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")),
        "v1_period_us": _read_text(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")),
        "v1_cpuset_effective": _read_text(
            Path("/sys/fs/cgroup/cpuset/cpuset.effective_cpus")),
    }
    try:
        result["affinity"] = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        result["affinity"] = []
    return result


def _git_state() -> dict[str, object]:
    commit = _command_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])
    status = _command_output(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], timeout=10.0)
    return {
        "commit": commit,
        "dirty": bool(status),
        "dirty_entries": status.splitlines() if status else [],
    }


def _library_state() -> dict[str, object]:
    requested = os.environ.get("PYQCU_KUPL_LIB")
    path = Path(requested).expanduser() if requested else DEFAULT_KUPL_LIBRARY
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    result: dict[str, object] = {
        "requested": requested,
        "path": str(path),
        "inside_repository": path.is_relative_to(REPO_ROOT),
        "exists": path.is_file(),
    }
    if not path.is_file():
        return result
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    result.update({
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "file": _command_output(["file", "-b", str(path)]),
    })
    return result


def _current_environment() -> dict[str, str | None]:
    return {
        key: os.environ.get(key)
        for key in (*KUPL_RUNTIME_KEYS, *OMP_PLACEMENT_KEYS)
    }


def environment_issues(environment: dict[str, str | None]) -> tuple[str, ...]:
    issues = []
    bind = (environment.get("KUPL_NUMA_BIND") or "").lower()
    if bind not in {"", "0", "false", "no", "off"}:
        conflicts = [
            key for key in UNSET_FOR_LIBRARY_BINDING
            if environment.get(key) not in (None, "") and key != "KUPL_CPU_NODE"
        ]
        if conflicts:
            issues.append(
                "KUPL_NUMA_BIND 启用时仍设置了 OpenMP 放置变量："
                + ", ".join(conflicts))
    if environment.get("KUPL_REQUIRE_MEMORY_BIND") == "1" and not environment.get(
            "KUPL_MEMORY_POLICY"):
        issues.append(
            "KUPL_REQUIRE_MEMORY_BIND=1 但未显式设置 KUPL_MEMORY_POLICY；"
            "库将采用 preferred，可能回退 DDR。")
    if environment.get("KUPL_MEMORY_POLICY") == "preferred":
        issues.append(
            "KUPL_MEMORY_POLICY=preferred 在容量不足时会回退 DDR；"
            "若目标是确保使用 HBM，应使用 bind 并验证页分布。")
    return tuple(issues)


def _report_path(path: Path | None, now: datetime) -> Path:
    if path is not None:
        return path.expanduser().resolve()
    name = f"{REPORT_PREFIX}{now:%Y-%m-%d-%H-%M-%S}.txt"
    return (DEFAULT_LOG_DIR / name).resolve()


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 10000):
        candidate = path.with_name(f"{path.stem}-{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"cannot allocate a unique report path near {path}")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _shell_exports(environment: dict[str, str]) -> str:
    lines = ["unset " + " ".join(UNSET_FOR_LIBRARY_BINDING)]
    lines.extend(
        f"export {key}={shlex.quote(value)}"
        for key, value in environment.items())
    return "\n".join(lines)


def _solver_commands(
    recommendation: Recommendation,
    lattice: tuple[int, int, int, int],
    dtype: str,
) -> tuple[str, str]:
    common = [
        sys.executable, "-B",
        str(REPO_ROOT / "pyqcu/testing/simple/mpi/solve_case.py"),
        "--backend", "kupl",
        "--gauge", "<gauge.h5>",
        "--source", "<source.h5>",
        "--lat", *(str(value) for value in lattice),
        "--dtype", dtype,
        "--ranks", str(recommendation.ranks),
        "--grid", *(str(value) for value in recommendation.grid),
        "--omp-threads", str(recommendation.threads_per_rank),
        "--kupl-topology-report",
    ]
    if recommendation.environment.get("KUPL_NUMA_BIND") == "1":
        common.extend([
            "--kupl-numa-bind", "library",
            "--kupl-memory-policy", "bind",
            "--kupl-memory-node", recommendation.environment["KUPL_MEMORY_NODE"],
        ])
    else:
        common.extend(["--kupl-numa-bind", "scheduler"])
    if "PYQCU_KUPL_PROFILE" in recommendation.environment:
        common.extend([
            "--kupl-profile", recommendation.environment["PYQCU_KUPL_PROFILE"]])
    mg = [
        *common,
        "--levels", "2",
        "--dof", "12", "12",
        "--mg-grid", "2", "2", "2", "2",
        "--restart", "5",
        "--coarse-max-iter", "200",
        "--coarse-tol-factor", "1e4",
        "--nv-iters", "2",
        "--work-dir", "<work-dir>/legacy-mg",
    ]
    bicg = [
        *common,
        "--levels", "1",
        "--restart", "0",
        "--work-dir", "<work-dir>/bistabcg",
    ]
    return shlex.join(mg), shlex.join(bicg)


def _snapshot(
    *,
    now: datetime,
    report_path: Path,
    topology: HostTopology,
    memory_nodes: Sequence[MemoryNode],
    hbm: HbmAssessment,
    recommendation: Recommendation,
    lattice: tuple[int, int, int, int],
    dtype: str,
    tools: dict[str, object],
    library: dict[str, object],
    environment: dict[str, str | None],
    issues: Sequence[str],
) -> dict[str, object]:
    sizes = workload_sizes(lattice, recommendation.grid, dtype)
    packages = {
        name: _python_package_version(name)
        for name in ("numpy", "h5py", "mpi4py", "torch")
    }
    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "report_path": str(report_path),
        "host": {
            "hostname": socket.gethostname(),
            "user": os.environ.get("USER") or getpass.getuser(),
            "os_release": _read_os_release(),
            "kernel": platform.release(),
            "machine": topology.machine,
            "python": sys.version.splitlines()[0],
            "executable": sys.executable,
            "packages": packages,
            "load_average": _read_text(Path("/proc/loadavg")),
            "logged_in_users": len((_command_output(["who"]) or "").splitlines()),
        },
        "topology": topology,
        "memory_nodes": tuple(memory_nodes),
        "hbm": hbm,
        "recommendation": recommendation,
        "workload": {"lattice": lattice, "dtype": dtype, "sizes": sizes},
        "tuning_candidates": tuning_candidates(
            topology, lattice, recommendation.ranks),
        "tools": tools,
        "library": library,
        "git": _git_state(),
        "current_environment": environment,
        "environment_issues": tuple(issues),
    }


def _jsonable(value: object) -> object:
    if hasattr(value, "__dataclass_fields__"):
        return {
            key: _jsonable(getattr(value, key))
            for key in value.__dataclass_fields__  # type: ignore[attr-defined]
        }
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def render_report(snapshot: dict[str, object]) -> str:
    topology: HostTopology = snapshot["topology"]  # type: ignore[assignment]
    memory_nodes: Sequence[MemoryNode] = snapshot["memory_nodes"]  # type: ignore[assignment]
    hbm: HbmAssessment = snapshot["hbm"]  # type: ignore[assignment]
    recommendation: Recommendation = snapshot["recommendation"]  # type: ignore[assignment]
    workload = snapshot["workload"]
    assert isinstance(workload, dict)
    lattice = tuple(workload["lattice"])
    dtype = str(workload["dtype"])
    sizes = workload["sizes"]
    assert isinstance(sizes, dict)
    candidates = snapshot["tuning_candidates"]
    assert isinstance(candidates, tuple)
    host = snapshot["host"]
    assert isinstance(host, dict)
    tools = snapshot["tools"]
    assert isinstance(tools, dict)
    library = snapshot["library"]
    assert isinstance(library, dict)
    current = snapshot["current_environment"]
    assert isinstance(current, dict)
    issues = snapshot["environment_issues"]
    assert isinstance(issues, tuple)
    git = snapshot["git"]
    assert isinstance(git, dict)
    mg_command, bicg_command = _solver_commands(recommendation, lattice, dtype)

    lines = [
        "PyQCU KUPL 环境检测与生产设置建议",
        "=" * 64,
        f"生成时间: {snapshot['generated_at']}",
        f"报告文件: {snapshot['report_path']}",
        f"仓库: {REPO_ROOT}",
        f"Git commit: {git.get('commit') or 'unknown'}",
        f"Git dirty: {bool(git.get('dirty'))}",
        "",
        "[1] 主机与软件",
        "-" * 64,
        f"hostname: {host.get('hostname')}",
        f"user: {host.get('user')}",
        f"OS: {host.get('os_release')}",
        f"kernel: {host.get('kernel')}",
        f"machine: {topology.machine}",
        f"CPU model: {topology.cpu_model}",
        f"load average / logged-in users: {host.get('load_average')} / {host.get('logged_in_users')}",
        f"Python: {host.get('python')}",
        f"Python packages: {host.get('packages')}",
        f"工具路径: {tools.get('paths')}",
        f"工具版本: {tools.get('versions')}",
        "",
        "[2] CPU affinity 与 cgroup",
        "-" * 64,
        f"affinity count: {len(topology.affinity)}",
        f"affinity: {list(topology.affinity)}",
        f"cgroup: {_read_cgroup_cpu()}",
        f"物理内存: {_format_bytes(topology.memory_total_bytes)}",
        f"cgroup memory limit: {_format_bytes(topology.cgroup_memory_bytes)}",
        f"numa_balancing: {topology.numa_balancing}",
        f"transparent_hugepage: {topology.transparent_hugepage}",
        "",
        "[3] NUMA 节点",
        "-" * 64,
    ]
    if not memory_nodes:
        lines.append("未读取到 sysfs NUMA 数据；以下建议只具有 affinity 级别证据。")
    for node in memory_nodes:
        lines.append(
            f"node {node.node_id}: cpus={list(node.cpus) or '[]'} "
            f"MemTotal={_format_bytes(node.mem_total_bytes)} "
            f"MemFree={_format_bytes(node.mem_free_bytes)} "
            f"distance={list(node.distances)}")

    lines.extend([
        "",
        "[4] HBM / 片上内存判定",
        "-" * 64,
        f"status: {hbm.status}",
        f"candidate memory-only nodes: {list(hbm.candidate_nodes)}",
        f"mappings: {list(hbm.mappings)}",
        f"判定: {hbm.message}",
        "证据边界: sysfs 只能证明“无 CPU 且有内存”的节点和距离关系；",
        "          不能仅凭节点号宣称厂商定义的 HBM。必须结合机型/BIOS 文档、",
        "          numactl/numastat 与实际进程页分布。",
        "",
        "[5] KUPL 动态库",
        "-" * 64,
        f"path: {library.get('path')}",
        f"exists: {library.get('exists')}",
        f"inside_repository: {library.get('inside_repository')}",
        f"size: {_format_bytes(library.get('size_bytes'))}",
        f"sha256: {library.get('sha256')}",
        f"file: {library.get('file')}",
        "警告: 若 path 指向另一棵旧 PyQCU 工作树，必须显式确认它不是历史库。"
        if not library.get("inside_repository") else
        "动态库位于当前仓库树内。",
        "",
        "[6] 当前相关环境变量",
        "-" * 64,
    ])
    for key, value in current.items():
        lines.append(f"{key}={value if value is not None else '<unset>'}")
    lines.append("冲突/风险:")
    if issues:
        lines.extend(f"- {issue}" for issue in issues)
    else:
        lines.append("- 未发现显式冲突。")

    lines.extend([
        "",
        "[7] 适配本环境的推荐基线",
        "-" * 64,
        f"ranks: {recommendation.ranks}",
        f"threads/rank: {recommendation.threads_per_rank}",
        f"ranks/NUMA: {recommendation.ranks_per_numa}",
        f"grid: {list(recommendation.grid)}",
        f"local lattice: {list(recommendation.local_lattice)}",
        f"HBM strict-bind recommendation: {recommendation.hbm_guaranteed}",
        "说明: 上述是拓扑驱动基线，不是未经实测的“全局最优”；必须按第 9 节 A/B。",
        "同题短名单 (ranks / threads / grid):",
        *(
            f"- {item['ranks']} / {item['threads_per_rank']} / "
            f"{'x'.join(map(str, item['grid']))}"
            for item in candidates
        ),
        "",
        "环境变量:",
        _shell_exports(recommendation.environment),
        (
            "MPI 放置: KUPL 库接管 CPU 与内存策略，因此使用 --bind-to none；"
            if recommendation.environment.get("KUPL_NUMA_BIND") == "1" else
            "MPI 放置: 本机未建立库内 NUMA 绑定，必须由调度器/cgroup 提供精确 cpuset。"
        ),
        "           不要同时设置 OMP_PROC_BIND/OMP_PLACES/GOMP_CPU_AFFINITY/KMP_AFFINITY。",
        "",
        "[8] 生产命令",
        "-" * 64,
        "A. 真正执行 coarse correction 的 Legacy-MG（不是 restart=0）：",
        mg_command,
        "",
        "B. 高效 parity-Schur BiStabCG（不执行 coarse hierarchy）:",
        bicg_command,
        "",
        "注意: mpi/solve_case.py 会在内部启动 mpirun，外层不得再次嵌套 mpirun。",
        "      该生产入口是 Legacy additive correction，不等同于 Strict-MG V-cycle/FGMRES。",
        "      BiStabCG 核心由 applyCloverBistabCgQcu 直接调用；restart=0 的 MG 入口仅转发。",
        "      正式正确性验收不要传 --skip-independent-residual。",
        "      MG 报告必须出现 levels=2、restart!=0、coarse_source!=unused、",
        "      verify_status=0，并且 independent true residual < atol。",
        "",
        "[9] 性能 A/B 与验证",
        "-" * 64,
        "1. 用同一 gauge/source、dtype、atol、ranks/grid/threads 比较 MG 与 BiStabCG。",
        "2. 首次 MG 运行包含 cache build；稳态比较必须复用同一 cache。",
        "3. 扫描 threads 时优先尝试 1、4、8、16 和每 NUMA 满核；不要只看一轮。",
        "4. KUPL_FUSED_BICG_DOTS=1 保持默认；只有同题 A/B 明显收益时才改。",
        "5. KUPL_HUGEPAGE=1 仅作专用节点 A/B，并核实实际大页使用；不是通用默认。",
        "6. HBM 验收必须对运行中的每个 rank 执行:",
        "   numastat -p <rank_pid>",
        "   或检查 /proc/<rank_pid>/numa_maps 的 N<node>= 页数。",
        "   memory_bound=true 只证明 set_mempolicy 成功，不证明所有输入页已迁移。",
        "7. 记录 NUMA/cpuset、实际 memory node、cache identity、迭代、真残差和耗时。",
        "",
        "[10] 工作集静态尺度",
        "-" * 64,
        f"local sites: {sizes.get('local_sites')}",
        f"local half-sites: {sizes.get('local_half_sites')}",
        f"gauge: {_format_bytes(sizes.get('gauge_bytes'))}",
        f"one fermion vector: {_format_bytes(sizes.get('fermion_vector_bytes'))}",
        f"one Clover parity matrix: {_format_bytes(sizes.get('clover_parity_bytes'))}",
        f"core static scale (not measured peak): {_format_bytes(sizes.get('core_static_scale_bytes'))}",
        "HBM bind 前必须在目标格点记录实际峰值；静态尺度不含完整 coarse setup 临时量。",
        "",
        "[11] 警告与限制",
        "-" * 64,
    ])
    lines.extend(f"- {warning}" for warning in recommendation.warnings)
    if not recommendation.warnings:
        lines.append("- 无额外拓扑警告。")
    lines.extend([
        "- 本脚本不执行 solve、不改变 set_mempolicy，也不导入 mpi4py。",
        "- 共享登录节点上的负载和 NUMA free memory 会变化，正式结论必须在计算作业中复测。",
        "",
    ])
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ranks", type=int, default=None,
                        help="推荐 MPI rank 数；默认每个可见 CPU NUMA 一个 rank")
    parser.add_argument("--threads-per-rank", type=int, default=None)
    parser.add_argument("--lattice", nargs=4, type=int, default=(24, 24, 24, 64))
    parser.add_argument("--dtype", choices=("c64", "c128"), default="c64")
    parser.add_argument("--grid", nargs=4, type=int, default=None)
    parser.add_argument("--output", type=Path, default=None,
                        help="默认 logs/kupl_env-vYYYY-MM-DD-HH-MM-SS.txt")
    parser.add_argument("--stdout", action="store_true", help="同时打印完整报告")
    parser.add_argument("--json", action="store_true", help="打印机器可读摘要")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--no-tool-probes", action="store_true")
    parser.add_argument("--node-root", type=Path, default=DEFAULT_NODE_ROOT,
                        help=argparse.SUPPRESS)
    parser.add_argument("--cpuinfo", type=Path, default=DEFAULT_CPUINFO,
                        help=argparse.SUPPRESS)
    parser.add_argument("--meminfo", type=Path, default=DEFAULT_MEMINFO,
                        help=argparse.SUPPRESS)
    parser.add_argument("--numa-balancing", type=Path,
                        default=DEFAULT_NUMA_BALANCING, help=argparse.SUPPRESS)
    parser.add_argument("--thp", type=Path, default=DEFAULT_THP,
                        help=argparse.SUPPRESS)
    return parser


def _argv(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = _parser()
    return parser.parse_args(argv), parser


def main(argv: Sequence[str] | None = None) -> int:
    args, parser = _argv(argv)
    now = datetime.now().astimezone()
    requested_path = _report_path(args.output, now)
    report_path = unique_path(requested_path) if not args.no_write else requested_path
    topology = read_topology(
        node_root=args.node_root,
        cpuinfo_path=args.cpuinfo,
        meminfo_path=args.meminfo,
        numa_balancing_path=args.numa_balancing,
        thp_path=args.thp,
    )
    memory_nodes = read_memory_nodes(args.node_root, affinity=None)
    cpu_nodes = tuple(
        CpuNode(node.node_id, node.cpus)
        for node in memory_nodes if node.cpus)
    if not cpu_nodes:
        cpu_nodes = topology.nodes
    if memory_nodes:
        hbm = assess_onchip_memory(cpu_nodes, memory_nodes)
    else:
        hbm = HbmAssessment(
            status="topology_unavailable",
            candidate_nodes=(),
            mappings=(),
            offset=None,
            message="未读取到 sysfs NUMA 节点，无法判断 HBM/片上内存。",
        )
    tools = _tool_versions(not args.no_tool_probes)
    mpi_versions = tools.get("versions", {})
    mpi_text = (
        str(mpi_versions.get("mpirun", ""))
        if isinstance(mpi_versions, dict) else "")
    mpi_kind = "hmpi" if "hyper mpi" in mpi_text.lower() else (
        "openmpi" if "open mpi" in mpi_text.lower() else None)
    try:
        recommendation = build_recommendation(
            topology,
            memory_nodes,
            hbm,
            ranks=args.ranks,
            threads_per_rank=args.threads_per_rank,
            lattice=tuple(args.lattice),
            dtype=args.dtype,
            grid=tuple(args.grid) if args.grid else None,
            mpi=mpi_kind,
            sysfs_available=bool(memory_nodes and cpu_nodes),
        )
    except ValueError as error:
        parser.error(str(error))
    environment = _current_environment()
    issues = list(environment_issues(environment))
    current_memory = environment.get("KUPL_MEMORY_NODE") or ""
    if current_memory.startswith("local+") and not hbm.has_unique_offset:
        issues.append(
            f"当前 KUPL_MEMORY_NODE={current_memory}，但本机未检测到唯一 local+N HBM 映射。")
    library = _library_state()
    if not library.get("exists"):
        issues.append("KUPL 动态库不存在；先构建或修正 PYQCU_KUPL_LIB。")
    if not library.get("inside_repository"):
        issues.append("KUPL 动态库位于当前仓库之外；确认它不是另一棵 PyQCU 的旧构建。")
    for key in ("OMP_NUM_THREADS", "KUPL_OMP_THREADS"):
        value = environment.get(key)
        if value and value != str(recommendation.threads_per_rank):
            issues.append(
                f"当前 {key}={value} 与本机推荐 threads/rank="
                f"{recommendation.threads_per_rank} 不同。")

    snapshot = _snapshot(
        now=now,
        report_path=report_path,
        topology=topology,
        memory_nodes=memory_nodes,
        hbm=hbm,
        recommendation=recommendation,
        lattice=tuple(args.lattice),
        dtype=args.dtype,
        tools=tools,
        library=library,
        environment=environment,
        issues=tuple(issues),
    )
    report = render_report(snapshot)
    if not args.no_write:
        atomic_write_text(report_path, report)
    if args.json:
        print(json.dumps(_jsonable(snapshot), indent=2, ensure_ascii=False))
    elif args.stdout or args.no_write:
        print(report)
    else:
        print(f"report={report_path}")
        print(
            f"recommendation=ranks:{recommendation.ranks} "
            f"threads:{recommendation.threads_per_rank} "
            f"grid:{'x'.join(map(str, recommendation.grid))} "
            f"memory_node:{recommendation.environment.get('KUPL_MEMORY_NODE', 'unset')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
