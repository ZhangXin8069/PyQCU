#!/usr/bin/env python3
"""Contract tests for Kunpeng topology discovery and launch planning."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from pyqcu.testing.kupl.kunpeng_runtime import (
    CpuNode,
    HostTopology,
    build_plan,
    parse_cpu_list,
    read_topology,
)


ROOT = Path(__file__).resolve().parents[3]


def _decode_json_stdout(text: str) -> dict:
    start = text.find("{")
    if start < 0:
        raise AssertionError(f"CLI stdout contains no JSON object: {text!r}")
    payload, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(payload, dict):
        raise AssertionError(f"CLI JSON payload is not an object: {payload!r}")
    return payload


def kunpeng_topology(
    nodes: tuple[CpuNode, ...] = (
        CpuNode(0, tuple(range(0, 24))),
        CpuNode(1, tuple(range(24, 48))),
    ),
    *,
    machine: str = "aarch64",
    model: str = "Kunpeng-920",
) -> HostTopology:
    return HostTopology(
        machine=machine,
        cpu_model=model,
        affinity=tuple(cpu for node in nodes for cpu in node.cpus),
        nodes=nodes,
        numa_balancing=False,
        transparent_hugepage="[always] never",
        memory_total_bytes=128 << 30,
        cgroup_memory_bytes=64 << 30,
    )


class KunpengRuntimeTests(unittest.TestCase):
    def test_parse_cpu_list_supports_ranges_and_union(self) -> None:
        self.assertEqual(parse_cpu_list("0-2,4,6-7"), (0, 1, 2, 4, 6, 7))

    def test_scheduler_plan_binds_one_rank_per_numa(self) -> None:
        plan = build_plan(
            kunpeng_topology(), ranks=2, threads_per_rank=24,
            binding="scheduler", mpi="openmpi",
        )
        self.assertEqual(plan.profile, "kunpeng-2numa")
        self.assertEqual(plan.ranks_per_numa, 1)
        self.assertEqual(plan.environment["KUPL_NUMA_BIND"], "0")
        self.assertEqual(plan.environment["KUPL_FUSED_BICG_DOTS"], "1")
        self.assertEqual(plan.environment["KUPL_DISABLE_CLOVER_CACHE"], "0")
        self.assertNotIn("OMP_PROC_BIND", plan.environment)
        self.assertNotIn("OMP_PLACES", plan.environment)
        self.assertEqual(
            plan.mpi_arguments,
            ("--report-bindings", "--map-by", "numa", "--bind-to", "numa"),
        )

    def test_library_plan_owns_cpu_and_memory_placement(self) -> None:
        plan = build_plan(
            kunpeng_topology(), ranks=2, threads_per_rank=24,
            binding="library", mpi="openmpi", memory_policy="preferred",
        )
        self.assertEqual(plan.environment["KUPL_NUMA_BIND"], "1")
        self.assertEqual(plan.environment["KUPL_MEMORY_POLICY"], "preferred")
        self.assertEqual(plan.environment["KUPL_REQUIRE_MEMORY_BIND"], "1")
        self.assertNotIn("OMP_PROC_BIND", plan.environment)
        self.assertNotIn("OMP_PLACES", plan.environment)
        self.assertEqual(plan.mpi_arguments[-1], "none")

    def test_onchip_memory_offset_is_explicit(self) -> None:
        plan = build_plan(
            kunpeng_topology(), ranks=2, threads_per_rank=24,
            binding="library", memory_policy="preferred",
            memory_node_offset=16,
        )
        self.assertEqual(plan.environment["KUPL_MEMORY_NODE"], "local+16")

    def test_hmpi_library_binding_warns_about_topology_fallback(self) -> None:
        plan = build_plan(
            kunpeng_topology(), ranks=2, threads_per_rank=24,
            binding="library", mpi="hmpi",
        )
        self.assertTrue(any("Hyper MPI" in warning for warning in plan.warnings))

    def test_cli_rejects_launcher_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fake = Path(temporary) / "fake-mpirun"
            fake.write_text(
                "#!/bin/sh\nprintf '%s\\n' 'Open MPI 4.1.2'\n",
                encoding="utf-8")
            fake.chmod(0o755)
            completed = subprocess.run(
                [
                    sys.executable, "-B",
                    str(ROOT / "pyqcu/testing/kupl/kunpeng_runtime.py"),
                    "plan", "--profile", "generic", "--ranks", "1",
                    "--threads-per-rank", "1", "--binding", "none",
                    "--mpi", "hmpi", "--mpirun", str(fake), "--json",
                ],
                cwd=ROOT, text=True, capture_output=True, check=False,
                timeout=60,
            )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("reports 'openmpi'", completed.stderr)

    def test_cli_records_explicit_launcher_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fake = Path(temporary) / "fake-mpirun"
            fake.write_text(
                "#!/bin/sh\nprintf '%s\\n' 'Open MPI 4.1.2'\n",
                encoding="utf-8")
            fake.chmod(0o755)
            completed = subprocess.run(
                [
                    sys.executable, "-B",
                    str(ROOT / "pyqcu/testing/kupl/kunpeng_runtime.py"),
                    "plan", "--profile", "generic", "--ranks", "1",
                    "--threads-per-rank", "1", "--binding", "none",
                    "--mpi", "hmpi", "--mpirun", str(fake),
                    "--allow-launcher-mismatch", "--json",
                ],
                cwd=ROOT, text=True, capture_output=True, check=False,
                timeout=60,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = _decode_json_stdout(completed.stdout)
        self.assertEqual(payload["mpi"], "hmpi")
        self.assertEqual(payload["mpi_detected"], "openmpi")
        self.assertEqual(payload["mpi_executable"], str(fake))

    def test_rankfile_replaces_generated_mapping(self) -> None:
        plan = build_plan(
            kunpeng_topology(), ranks=2, threads_per_rank=24,
            binding="scheduler", rankfile="data/kunpeng.rankfile",
        )
        self.assertEqual(
            plan.mpi_arguments,
            ("--report-bindings", "--rankfile", "data/kunpeng.rankfile"),
        )

    def test_shell_plan_clears_owned_environment(self) -> None:
        plan = build_plan(
            kunpeng_topology(), ranks=2, threads_per_rank=24,
            binding="scheduler",
        )
        shell = plan.shell("bench")
        self.assertTrue(shell.startswith("unset KUPL_NUMA_BIND"))
        self.assertIn("unset KUPL_NUMA_BIND", shell)
        self.assertIn("export KUPL_NUMA_BIND=0", shell)

    def test_generic_plan_does_not_force_short_region_openmp_binding(self) -> None:
        topology = kunpeng_topology(
            (CpuNode(0, tuple(range(18))),), machine="x86_64",
            model="Intel Xeon",
        )
        plan = build_plan(
            topology, ranks=1, threads_per_rank=8, binding="scheduler",
        )
        self.assertEqual(plan.profile, "generic")
        self.assertNotIn("OMP_PROC_BIND", plan.environment)
        self.assertNotIn("OMP_PLACES", plan.environment)

    def test_plan_rejects_thread_oversubscription_by_default(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceed"):
            build_plan(
                kunpeng_topology(), ranks=4, threads_per_rank=16,
                binding="scheduler",
            )

    def test_oversubscription_is_propagated_to_mpi_and_library(self) -> None:
        plan = build_plan(
            kunpeng_topology(), ranks=4, threads_per_rank=16,
            binding="library", allow_oversubscribe=True,
        )
        self.assertEqual(plan.environment["KUPL_ALLOW_OVERSUBSCRIBE"], "1")
        self.assertEqual(plan.mpi_arguments[0], "--oversubscribe")
        self.assertIn("--oversubscribe", plan.shell("bench"))

    def test_scheduler_multi_rank_per_numa_requires_rankfile(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires --rankfile"):
            build_plan(
                kunpeng_topology(), ranks=4, threads_per_rank=12,
                binding="scheduler",
            )

    def test_memory_policy_requires_library_binding(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires --binding library"):
            build_plan(
                kunpeng_topology(), ranks=2, threads_per_rank=12,
                binding="scheduler", memory_policy="preferred",
            )

    def test_fake_sysfs_discovery_uses_affinity_not_cpu_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "online").write_text("0-1\n", encoding="utf-8")
            (root / "node0").mkdir()
            (root / "node1").mkdir()
            (root / "node0" / "cpulist").write_text("0-23\n", encoding="utf-8")
            (root / "node1" / "cpulist").write_text("24-47\n", encoding="utf-8")
            cpuinfo = root / "cpuinfo"
            cpuinfo.write_text("model name : Kunpeng-920\n", encoding="utf-8")
            meminfo = root / "meminfo"
            meminfo.write_text("MemTotal: 1024 kB\n", encoding="utf-8")
            topology = read_topology(
                node_root=root,
                cpuinfo_path=cpuinfo,
                meminfo_path=meminfo,
                numa_balancing_path=root / "missing-numa",
                thp_path=root / "missing-thp",
                affinity_override=(0, 1, 24, 25),
            )
        self.assertEqual(topology.affinity, (0, 1, 24, 25))
        self.assertEqual([node.node_id for node in topology.nodes], [0, 1])
        self.assertEqual([node.cpus for node in topology.nodes], [(0, 1), (24, 25)])
        self.assertEqual(topology.memory_total_bytes, 1024 * 1024)

    def test_cli_json_plan_is_machine_readable(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(ROOT / "pyqcu/testing/kupl/kunpeng_runtime.py"),
                "inspect",
                "--json",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = _decode_json_stdout(completed.stdout)
        self.assertIn("affinity", payload)
        self.assertIn("nodes", payload)

    def test_cli_plan_json_is_machine_readable(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(ROOT / "pyqcu/testing/kupl/kunpeng_runtime.py"),
                "plan", "--profile", "generic", "--ranks", "1",
                "--threads-per-rank", "1", "--binding", "none", "--json",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = _decode_json_stdout(completed.stdout)
        self.assertEqual(payload["ranks"], 1)
        self.assertEqual(payload["environment"]["KUPL_NUMA_BIND"], "0")


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(KunpengRuntimeTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
