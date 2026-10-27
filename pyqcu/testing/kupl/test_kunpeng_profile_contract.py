#!/usr/bin/env python3
"""Verify strict Kunpeng profile identity and MPI-rank consistency."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path


def _host_is_kunpeng() -> bool:
    if platform.machine().lower() not in {"aarch64", "arm64"}:
        return False
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8").lower()
    except OSError:
        return False
    return any(marker in text for marker in
               ("kunpeng", "hisilicon", "cpu implementer\t: 0x48")) or \
        any(marker in text for marker in
            ("cpu part\t: 0xd01", "cpu part\t: 0xd02"))


def _clean_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "KUPL_NUMA_BIND", "KUPL_CPU_NODE", "KUPL_MEMORY_POLICY",
        "KUPL_MEMORY_NODE", "KUPL_REQUIRE_MEMORY_BIND",
        "KUPL_REQUIRE_TOPOLOGY", "KUPL_TOPOLOGY_REPORT", "KUPL_HUGEPAGE",
        "KUPL_FUSED_BICG_DOTS", "PYQCU_KUPL_PROFILE", "KUPL_OMP_THREADS",
        "OMP_NUM_THREADS", "OMP_DYNAMIC", "OMP_PROC_BIND", "OMP_PLACES",
        "GOMP_CPU_AFFINITY", "KMP_AFFINITY", "KUPL_ALLOW_OVERSUBSCRIBE",
    ):
        environment.pop(name, None)
    environment.update({
        "KUPL_NUMA_BIND": "0",
        "KUPL_OMP_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OMP_DYNAMIC": "FALSE",
    })
    return environment


def _binary_arguments(binary: Path, ranks: int) -> list[str]:
    return [
        str(binary), "--lattice", str(2 * ranks), "2", "2", "4",
        "--grid", str(ranks), "1", "1", "1", "--iterations", "1",
    ]


def _run(command: list[str], environment: dict[str, str], timeout: float,
         cwd: Path) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            command, cwd=cwd, env=environment, text=True, capture_output=True,
            timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        print(f"FAIL: profile contract timed out after {timeout:g}s",
              file=sys.stderr)
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--mpirun", default="mpirun")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    root = Path(__file__).resolve().parents[3]
    try:
        version = subprocess.run(
            [args.mpirun, "--version"], text=True, capture_output=True,
            timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"SKIPPED: cannot query MPI launcher: {error}")
        return 77
    marker = (version.stdout or version.stderr).lower()
    if not any(value in marker for value in
               ("open mpi", "openrte", "hyper mpi")):
        print("SKIPPED: test requires an Open MPI-compatible launcher")
        return 77

    environment = _clean_environment()
    environment["PYQCU_KUPL_PROFILE"] = "kunpeng"
    environment["KUPL_TOPOLOGY_REPORT"] = "1"
    single = _run(_binary_arguments(args.binary, 1), environment,
                  args.timeout, root)
    if single is None:
        return 1
    output = single.stdout + single.stderr
    if _host_is_kunpeng():
        if single.returncode != 0 or "PYQCU::KUPL::TOPOLOGY:" not in output:
            print("FAIL: valid Kunpeng profile did not produce topology evidence")
            print(output, end="")
            return 1
    elif single.returncode == 0 or "non-Kunpeng host" not in output:
        print("FAIL: non-Kunpeng host accepted a Kunpeng profile")
        print(output, end="")
        return 1

    environment = _clean_environment()
    wrapper = (
        "import os,sys;"
        "rank=int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK',"
        "os.environ.get('PMI_LOCAL_RANK','0')));"
        "os.environ['PYQCU_KUPL_PROFILE']="
        "'generic' if rank == 0 else 'kunpeng';"
        "os.execv(sys.argv[1],sys.argv[1:])"
    )
    command = [
        args.mpirun, "--allow-run-as-root", "--oversubscribe",
        "--bind-to", "none", "-np", "2", sys.executable, "-B", "-c", wrapper,
        *_binary_arguments(args.binary, 2),
    ]
    mismatched = _run(command, environment, args.timeout, root)
    if mismatched is None:
        return 1
    output = mismatched.stdout + mismatched.stderr
    if mismatched.returncode == 0 or "must be identical on every MPI rank" not in output:
        print("FAIL: rank-inconsistent profile was not rejected promptly")
        print(output, end="")
        return 1
    print("status=PASS non_kunpeng_rejected=true rank_mismatch_rejected=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
