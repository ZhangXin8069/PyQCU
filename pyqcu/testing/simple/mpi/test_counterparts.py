"""Contract tests for the flat simple-MPI entry-point specialization."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from pyqcu.testing.simple.mpi._entrypoint import (
    TRACE_ENV_VARS,
    _publish_staged,
    _stage_outputs,
    benchmark_environment,
)


DIR = Path(__file__).resolve().parent
PARENT_ENTRIES = (
    "benchmark_kupl.py",
    "benchmark_kupl_lime_mpi.py",
    "build_full_mg_cache.py",
    "check_clover.py",
    "check_fermion.py",
    "check_gauge.py",
    "compare_bistabcg.py",
    "compare_kupl_bistabcg.py",
    "compare_kupl_multigrid.py",
    "compare_three_way.py",
    "give_gauss_gauge.py",
    "kupl_bistabcg.py",
    "kupl_multigrid.py",
    "kupl_residual.py",
    "prepare_case.py",
    "qcu_bistabcg.py",
    "qcu_clover2quda_clover.py",
    "qcu_fermion2quda_fermion.py",
    "qcu_gauge2quda_gauge.py",
    "qcu_multigrid.py",
    "quda_bistabcg.py",
    "quda_clover2qcu_clover.py",
    "quda_fermion2qcu_fermion.py",
    "quda_gauge2qcu_gauge.py",
    "quda_multigrid.py",
    "slurm_lime_backend_matrix.sh",
    "solve_case.py",
    "test_coarse_cache.py",
)


def _run(command: list[str], *, env: dict[str, str] | None = None
         ) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, cwd=DIR.parents[3], env=env, text=True,
        capture_output=True, check=False, timeout=120)


def test_every_parent_entry_has_a_mapped_counterpart():
    for name in PARENT_ENTRIES:
        assert (DIR / name).is_file(), name
    mapping = (DIR / "MAPPING.md").read_text(encoding="utf-8")
    rows = set(re.findall(r"^\| `([^`]+)` \| `([^`]+)` \|", mapping, re.M))
    assert {left for left, _ in rows} == set(PARENT_ENTRIES)
    assert {right for _, right in rows} == set(PARENT_ENTRIES)


@pytest.mark.parametrize(
    "script",
    ("qcu_bistabcg.py", "quda_bistabcg.py", "kupl_bistabcg.py",
     "quda_multigrid.py", "qcu_multigrid.py", "kupl_multigrid.py"))
def test_delegated_dry_run_rejects_non_power_of_two(script: str):
    completed = _run([
        sys.executable, "-B", str(DIR / script), "--dry-run", "--ranks", "3",
        "--gauge", "fake.lime", "--lat", "4", "4", "4", "8",
    ])
    assert completed.returncode == 2
    assert "positive power of two" in completed.stderr


def test_dry_run_command_mapping_and_trace_contract(tmp_path: Path):
    completed = _run([
        sys.executable, "-B", str(DIR / "qcu_bistabcg.py"), "--dry-run",
        "--ranks", "2", "--grid", "1", "1", "1", "2",
        "--gauge", "fake.lime", "--lat", "4", "4", "4", "8",
        "--report", str(tmp_path / "qcu.json"),
    ])
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["mode"] == "delegated-launcher"
    assert payload["trace_enabled"] is False
    assert payload["quda_enable_tuning"] == "0"
    assert payload["grid"] == [1, 1, 1, 2]
    command = payload["command"]
    assert command[command.index("-np") + 1] == "2"
    assert command[-1] == str(tmp_path / "qcu.json")
    assert "--report" in command


def test_three_way_dry_run_uses_isolated_jobs():
    completed = _run([
        sys.executable, "-B", str(DIR / "compare_three_way.py"), "--dry-run",
        "--ranks", "2", "--gauge", "fake.lime", "--lat", "4", "4", "4", "8",
    ])
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["mixed_backend_mpi_job"] is False
    assert [job["label"] for job in payload["jobs"]] == [
        "cuda-qcu", "kupl-qcu", "quda"]
    for job in payload["jobs"]:
        assert job["command"][1] == "-B"
        assert "--ranks" in job["command"]


def test_three_way_rejects_nested_mpi_orchestration():
    environment = os.environ.copy()
    environment["OMPI_COMM_WORLD_SIZE"] = "2"
    completed = _run([
        sys.executable, "-B", str(DIR / "compare_three_way.py"), "--dry-run",
        "--ranks", "2", "--gauge", "fake.lime", "--lat", "4", "4", "4", "8",
    ], env=environment)
    assert completed.returncode == 2
    assert "must run outside MPI" in completed.stderr


def test_delegated_launcher_rejects_nested_mpi_execution():
    environment = os.environ.copy()
    environment["OMPI_COMM_WORLD_SIZE"] = "2"
    completed = _run([
        sys.executable, "-B", str(DIR / "qcu_bistabcg.py"), "--dry-run",
        "--ranks", "2", "--gauge", "fake.lime", "--lat", "4", "4", "4", "8",
    ], env=environment)
    assert completed.returncode == 2
    assert "must run as one process outside MPI" in completed.stderr


def test_benchmark_environment_drops_every_trace_variable():
    base = {name: "/tmp/trace.jsonl" for name in TRACE_ENV_VARS}
    base["PYQCU_FUTURE_TRACE_FILE"] = "/tmp/future.jsonl"
    base["QUDA_ENABLE_TUNING"] = "1"
    environment = benchmark_environment(base)
    assert all(name not in environment for name in TRACE_ENV_VARS)
    assert "PYQCU_FUTURE_TRACE_FILE" not in environment
    assert environment["QUDA_ENABLE_TUNING"] == "0"


def test_slurm_dry_run_parameterizes_rank_grid_and_power():
    environment = os.environ.copy()
    environment.update({
        "DRY_RUN": "1",
        "PYQCU_ROOT": str(DIR.parents[3]),
        "PYQCU_LIME_CASES": "2:1,1,1,2",
        "PYQCU_LIME_BACKENDS": "qcu",
        "PYQCU_LIME_PRECISIONS": "c64",
        "PYQCU_LIME_POWER_LIMIT_WATTS": "125",
    })
    completed = _run(
        ["bash", str(DIR / "slurm_lime_backend_matrix.sh")], env=environment)
    assert completed.returncode == 0, completed.stderr
    assert "--grid 1 1 1 2" in completed.stdout
    assert "lime backend MPI matrix: DRY-RUN" in completed.stdout
    assert "tuning=0" in completed.stdout

    environment["PYQCU_LIME_CASES"] = "3:1,1,1,3"
    rejected = _run(
        ["bash", str(DIR / "slurm_lime_backend_matrix.sh")], env=environment)
    assert rejected.returncode != 0
    assert "positive power of two" in rejected.stderr


def test_rank_zero_output_is_staged_then_atomically_published(tmp_path: Path):
    output = tmp_path / "converted.npy"
    output.write_bytes(b"old")
    arguments, staged = _stage_outputs(
        ["--output", str(output)], ("--output",))
    temporary = Path(arguments[arguments.index("--output") + 1])
    assert temporary != output
    assert temporary.parent == output.parent
    assert temporary.suffix == ".npy"
    temporary.write_bytes(b"new")
    _publish_staged(staged)
    assert output.read_bytes() == b"new"
    assert not temporary.exists()


def test_staged_json_report_rewrites_temporary_output_paths(tmp_path: Path):
    output = tmp_path / "cache.h5"
    report = tmp_path / "cache.json"
    arguments, staged = _stage_outputs(
        ["--output", str(output), "--report", str(report)],
        ("--output", "--report"))
    temporary_output = Path(arguments[arguments.index("--output") + 1])
    temporary_report = Path(arguments[arguments.index("--report") + 1])
    temporary_output.write_bytes(b"cache")
    temporary_report.write_text(
        json.dumps({"output": str(temporary_output)}), encoding="utf-8")
    _publish_staged(staged)
    assert json.loads(report.read_text(encoding="utf-8"))["output"] == str(output)
