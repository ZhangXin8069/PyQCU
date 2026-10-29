#!/usr/bin/env python3
"""Contract tests for the KUPL environment report and HBM detection."""

from __future__ import annotations

import os
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import kupl_env  # noqa: E402
from kunpeng_runtime import CpuNode, HostTopology  # noqa: E402


def _write_node(
    root: Path,
    node_id: int,
    *,
    cpus: str,
    mem_total_kib: int,
    mem_free_kib: int,
    distances: tuple[int, ...],
) -> None:
    directory = root / f"node{node_id}"
    directory.mkdir()
    (directory / "cpulist").write_text(cpus + "\n", encoding="ascii")
    (directory / "meminfo").write_text(
        f"Node {node_id} MemTotal: {mem_total_kib} kB\n"
        f"Node {node_id} MemFree: {mem_free_kib} kB\n",
        encoding="ascii")
    (directory / "distance").write_text(
        " ".join(map(str, distances)) + "\n", encoding="ascii")


def _topology(nodes: tuple[CpuNode, ...]) -> HostTopology:
    return HostTopology(
        machine="aarch64",
        cpu_model="Kunpeng-920",
        affinity=tuple(cpu for node in nodes for cpu in node.cpus),
        nodes=nodes,
        numa_balancing=False,
        transparent_hugepage="[always] madvise never",
        memory_total_bytes=128 << 30,
        cgroup_memory_bytes=96 << 30,
    )


class KuplEnvironmentTests(unittest.TestCase):
    def test_cpu_only_numa_recommends_local_ddr(self) -> None:
        nodes = tuple(
            CpuNode(node_id, tuple(range(node_id * 32, (node_id + 1) * 32)))
            for node_id in range(4))
        topology = _topology(nodes)
        memory_nodes = tuple(
            kupl_env.MemoryNode(
                node_id=node.node_id,
                cpus=node.cpus,
                mem_total_bytes=128 << 30,
                mem_free_bytes=64 << 30,
                distances=tuple((peer, 10 if peer == node.node_id else 20)
                                for peer in range(4)),
            )
            for node in nodes)
        assessment = kupl_env.assess_onchip_memory(nodes, memory_nodes)
        self.assertEqual(assessment.status, "no_candidate")
        recommendation = kupl_env.build_recommendation(
            topology, memory_nodes, assessment,
            ranks=4, threads_per_rank=32,
            grid=(1, 1, 2, 2))
        self.assertEqual(recommendation.environment["KUPL_MEMORY_NODE"], "local")
        self.assertEqual(recommendation.environment["KUPL_MEMORY_POLICY"], "bind")
        self.assertEqual(recommendation.environment["KUPL_REQUIRE_MEMORY_BIND"], "1")
        self.assertFalse(recommendation.hbm_guaranteed)
        candidates = kupl_env.tuning_candidates(
            topology, (24, 24, 24, 64), requested_ranks=4)
        self.assertEqual(
            [(item["ranks"], item["threads_per_rank"], item["grid"])
             for item in candidates],
            [
                (4, 32, [1, 4, 1, 1]),
                (8, 16, [1, 4, 2, 1]),
                (16, 8, [1, 4, 4, 1]),
            ])

    def test_unique_memory_only_offset_is_detected(self) -> None:
        cpu_nodes = (
            CpuNode(0, tuple(range(0, 32))),
            CpuNode(1, tuple(range(32, 64))),
        )
        memory_nodes = (
            kupl_env.MemoryNode(
                0, cpu_nodes[0].cpus, 128 << 30, 64 << 30,
                ((0, 10), (1, 12), (16, 10), (17, 12))),
            kupl_env.MemoryNode(
                1, cpu_nodes[1].cpus, 128 << 30, 64 << 30,
                ((0, 12), (1, 10), (16, 12), (17, 10))),
            kupl_env.MemoryNode(
                16, (), 32 << 30, 24 << 30,
                ((0, 10), (1, 12), (16, 10), (17, 12))),
            kupl_env.MemoryNode(
                17, (), 32 << 30, 24 << 30,
                ((0, 12), (1, 10), (16, 12), (17, 10))),
        )
        assessment = kupl_env.assess_onchip_memory(cpu_nodes, memory_nodes)
        self.assertEqual(assessment.status, "detected_offset")
        self.assertEqual(assessment.offset, 16)
        self.assertEqual(
            [(item.cpu_node, item.memory_node) for item in assessment.mappings],
            [(0, 16), (1, 17)])
        recommendation = kupl_env.build_recommendation(
            _topology(cpu_nodes), memory_nodes, assessment,
            ranks=2, threads_per_rank=32, lattice=(8, 8, 8, 16))
        self.assertEqual(
            recommendation.environment["KUPL_MEMORY_NODE"], "local+16")
        self.assertEqual(recommendation.environment["KUPL_MEMORY_POLICY"], "bind")
        self.assertTrue(recommendation.hbm_guaranteed)

    def test_ambiguous_distance_never_emits_hbm_offset(self) -> None:
        cpu_nodes = (CpuNode(0, tuple(range(32))),)
        memory_nodes = (
            kupl_env.MemoryNode(
                0, cpu_nodes[0].cpus, 128 << 30, 64 << 30,
                ((0, 10), (16, 10), (17, 10))),
            kupl_env.MemoryNode(
                16, (), 32 << 30, 24 << 30,
                ((0, 10), (16, 10), (17, 20))),
            kupl_env.MemoryNode(
                17, (), 32 << 30, 24 << 30,
                ((0, 10), (16, 20), (17, 10))),
        )
        assessment = kupl_env.assess_onchip_memory(cpu_nodes, memory_nodes)
        self.assertEqual(assessment.status, "ambiguous")
        self.assertIsNone(assessment.offset)

    def test_missing_sysfs_does_not_emit_unsafe_memory_binding(self) -> None:
        topology = HostTopology(
            machine="x86_64",
            cpu_model="Unknown CPU",
            affinity=tuple(range(8)),
            nodes=(),
            numa_balancing=None,
            transparent_hugepage=None,
            memory_total_bytes=16 << 30,
            cgroup_memory_bytes=None,
        )
        assessment = kupl_env.HbmAssessment(
            "topology_unavailable", (), (), None, "no sysfs")
        recommendation = kupl_env.build_recommendation(
            topology, (), assessment, ranks=1, lattice=(8, 8, 8, 16))
        self.assertEqual(recommendation.environment["KUPL_NUMA_BIND"], "0")
        self.assertNotIn("KUPL_MEMORY_NODE", recommendation.environment)
        self.assertNotIn("KUPL_REQUIRE_MEMORY_BIND", recommendation.environment)

    def test_default_name_and_collision_suffix(self) -> None:
        now = kupl_env.datetime.fromisoformat("2026-10-30T12:34:56+08:00")
        path = kupl_env._report_path(None, now)
        self.assertEqual(path.name, "kupl_env-v2026-10-30-12-34-56.txt")
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / path.name
            base.write_text("existing", encoding="utf-8")
            unique = kupl_env.unique_path(base)
            self.assertEqual(unique.name, "kupl_env-v2026-10-30-12-34-56-2.txt")
            kupl_env.atomic_write_text(unique, "report\n")
            self.assertEqual(unique.read_text(encoding="utf-8"), "report\n")

    def test_cli_is_mpi_free_and_prints_both_real_solver_routes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            blocker = Path(temporary) / "mpi4py.py"
            blocker.write_text(
                "raise RuntimeError('mpi4py must not be imported')\n",
                encoding="utf-8")
            env = os.environ.copy()
            env["PYTHONPATH"] = temporary
            completed = subprocess.run(
                [
                    sys.executable, "-B", str(HERE / "kupl_env.py"),
                    "--no-write", "--stdout", "--no-tool-probes",
                ],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
                timeout=60,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("真正执行 coarse correction 的 Legacy-MG", completed.stdout)
        self.assertIn("parity-Schur BiStabCG", completed.stdout)
        self.assertIn("applyCloverBistabCgQcu", completed.stdout)
        self.assertIn("外层不得再次嵌套 mpirun", completed.stdout)
        self.assertNotIn("mpi4py must not be imported", completed.stderr)

    def test_cli_json_is_a_single_json_document(self) -> None:
        completed = subprocess.run(
            [
                sys.executable, "-B", str(HERE / "kupl_env.py"),
                "--no-write", "--json", "--no-tool-probes",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertIn("recommendation", payload)
        self.assertIn("hbm", payload)

    def test_cli_normal_mode_writes_report_and_exits_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "kupl_env.txt"
            completed = subprocess.run(
                [
                    sys.executable, "-B", str(HERE / "kupl_env.py"),
                    "--output", str(output), "--no-tool-probes",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
                timeout=60,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(output.is_file())
            self.assertIn(f"report={output}", completed.stdout)
            self.assertIn("memory_node:", completed.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
