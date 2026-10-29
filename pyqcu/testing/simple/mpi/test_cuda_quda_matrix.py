from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import pytest

from pyqcu.testing.simple.mpi import cuda_quda_matrix
from pyqcu.testing.simple.mpi.cuda_quda_matrix import (
    _materialise_source,
    _run_deferred_residuals,
)
from pyqcu.testing.simple.mpi.summarize_cuda_quda_matrix import (
    summarize_manifest,
)


DIR = Path(__file__).resolve().parent
REPO = DIR.parents[3]
LATTICE = [4, 4, 4, 8]
GRID_1 = [1, 1, 1, 1]
GRID_2 = [2, 1, 1, 1]
ACCEPTANCE = 2.44140625e-4


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _uuid(index: int) -> str:
    return f"hip:00000000-0000-0000-0000-{index + 1:012x}"


def _setup_stage() -> dict:
    return {
        "steady": {
            "operation_seconds": {"median": 0.05},
            "lifecycle_seconds": {"median": 0.06},
            "samples": [],
        }
    }


def _qcu_report(
    *, ranks: int, grid: list[int], source_path: Path, gauge_path: Path,
    uuids: list[str], independent: float = 4.0e-7,
) -> dict:
    solve_samples = [
        {
            "operation_seconds": value,
            "lifecycle_seconds": value + 0.1,
            "iterations": 18,
        }
        for value in (0.2, 0.21)
    ]
    rank_records = []
    for rank, uuid_value in enumerate(uuids):
        rank_records.append({
            "rank": rank,
            "local_rank": rank,
            "device": {
                "uuid": uuid_value,
                "total_memory_bytes": 1024,
            },
            "timings": {
                "clover_setup_even": _setup_stage(),
                "clover_setup_odd": _setup_stage(),
                "clover_schur_bistabcg_solve": {
                    "steady": {
                        "operation_seconds": {"median": 0.21},
                        "lifecycle_seconds": {"median": 0.31},
                        "samples": solve_samples,
                    }
                },
            },
            "solver_native": {"all_samples_converged": True},
            "independent_true_residual": {
                "verified": True,
                "value": independent,
                "acceptance": ACCEPTANCE,
                "pass": independent <= ACCEPTANCE,
                "source": (
                    "root-gathered independent PyQCU Wilson-Clover operator"),
                "phase": "after final steady solve; excluded from timing",
            },
        })
    return {
        "status": "PASS",
        "inputs": {
            "gauge": {"path": str(gauge_path), "sha256": "gauge-sha256"},
            "source": {"path": str(source_path), "sha256": "rhs-file-sha256"},
        },
        "physics": {
            "lattice": LATTICE,
            "mass": -0.277,
            "csw": 1.16,
            "dtype": "c64",
            "atol": 1.0e-6,
            "max_iter": 1000,
        },
        "execution": {
            "mpi_size": ranks,
            "grid": grid,
            "warmups": 1,
            "repeats": 2,
            "initial_guess": {
                "use_init_guess": False,
                "params_index": 57,
                "params_value": 0,
            },
            "libqcu": {"sha256": "qcu-library"},
        },
        "ranks": rank_records,
    }


def _quda_report(
    *, ranks: int, grid: list[int], source_path: Path, gauge_path: Path,
    solution_path: Path, uuids: list[str],
) -> dict:
    contract = {
        "environment_tuning_disabled": True,
        "tuning_enabled": False,
    }
    rank_records = []
    for rank, uuid_value in enumerate(uuids):
        rank_records.append({
            "rank": rank,
            "gpu": {
                "uuid": uuid_value,
                "total_global_memory_bytes": 1024,
            },
            "device_selection": {"device": rank},
            "parameter_contract": contract,
            "gauge_load_setup_seconds": 0.1,
            "steady": [{
                "wall_seconds": 0.45,
                "quda_solver_seconds": 0.3,
                "iterations": 19,
                "true_res": 5.0e-7,
            }],
        })
    samples = [{
        "sample": 0,
        "max_wall_seconds": 0.45,
        "wall_seconds_by_rank": [0.45] * ranks,
        "quda_solver_seconds_by_rank": [0.3] * ranks,
        "iterations_by_rank": [19] * ranks,
        "true_res_by_rank": [5.0e-7] * ranks,
        "true_res_max": 5.0e-7,
    }]
    return {
        "status": "PASS",
        "config": {
            "global_lattice_xyzt": LATTICE,
            "process_grid_xyzt": grid,
            "mpi_rank_count": ranks,
            "mass": -0.277,
            "csw": 1.16,
            "dtype": "c64",
            "tol": 1.0e-6,
            "max_iter": 1000,
            "warmups": 1,
            "repeats": 2,
            "use_init_guess": False,
        },
        "input": {
            "lime": {"path": str(gauge_path)},
            "lime_sha256": "gauge-sha256",
            "source": {
                "path": str(source_path),
                "global_sha256": "rhs-array-sha256",
                "file_sha256": "rhs-file-sha256",
            },
        },
        "solution": {"path": str(solution_path)},
        "libquda": {"sha256": "quda-library"},
        "mpi": {"ranks": rank_records},
        "phases": {"steady": {"samples": samples}},
        "solver": {"true_res_tolerance": 1.0e-6},
    }


def _independent_report(
    *, ranks: int, grid: list[int], source_path: Path, gauge_path: Path,
    solution_path: Path, residual: float = 4.0e-7,
) -> dict:
    return {
        "status": "PASS",
        "residual_source": (
            "independent periodic NumPy Wilson-Clover operator with MPI halo exchange"),
        "relative_true_residual": residual,
        "atol": 1.0e-6,
        "acceptance": ACCEPTANCE,
        "converged": residual <= ACCEPTANCE,
        "ranks": ranks,
        "grid": grid,
        "lattice": LATTICE,
        "dtype": "complex64",
        "mass": -0.277,
        "csw": 1.16,
        "inputs": {
            "gauge": str(gauge_path),
            "source": str(source_path),
            "solution": str(solution_path),
        },
    }


def _manifest(
    tmp_path: Path, *, ranks: int = 1,
    qcu_uuids: list[str] | None = None,
    quda_uuids: list[str] | None = None,
) -> Path:
    grid = GRID_1 if ranks == 1 else GRID_2
    qcu_uuids = qcu_uuids or [_uuid(index) for index in range(ranks)]
    quda_uuids = quda_uuids or [_uuid(index) for index in range(ranks)]
    source_path = tmp_path / "source.h5"
    gauge_path = tmp_path / "gauge.lime"
    qcu_path = tmp_path / f"r{ranks}/qcu/report.json"
    qcu_stdout = tmp_path / f"r{ranks}/qcu/stdout.txt"
    quda_path = tmp_path / f"r{ranks}/quda/report.json"
    quda_stdout = tmp_path / f"r{ranks}/quda/stdout.txt"
    solution_path = quda_path.with_name("solution.h5")
    residual_path = quda_path.with_name("independent-residual.json")
    _write(qcu_path, _qcu_report(
        ranks=ranks, grid=grid, source_path=source_path,
        gauge_path=gauge_path, uuids=qcu_uuids))
    _write(quda_path, _quda_report(
        ranks=ranks, grid=grid, source_path=source_path,
        gauge_path=gauge_path, solution_path=solution_path,
        uuids=quda_uuids))
    _write(residual_path, _independent_report(
        ranks=ranks, grid=grid, source_path=source_path,
        gauge_path=gauge_path, solution_path=solution_path))
    qcu_stdout.write_text(
        "##RANK:0##LOOP:18##Relative(norm2):1.0e-12\n", encoding="utf-8")
    quda_stdout.write_text(
        "BiCGstab: iterated = 5.0e-7, true = 5.0e-7\n", encoding="utf-8")
    devices = {
        str(rank): {
            "uuid": quda_uuids[rank],
            "total_bytes": 1024,
            "peak_used_bytes": 512,
        }
        for rank in range(ranks)
    }
    manifest = {
        "status": "COMPLETE",
        "requested_ranks": [ranks],
        "backends": ["qcu", "quda"],
        "limits": {"internal_wall_limit_seconds": 570.0},
        "elapsed_seconds": 10.0,
        "within_wall_limit": True,
        "configuration": {
            "algorithm": "clover-schur-bicgstab",
            "gauge_sha256": "gauge-sha256",
            "source_sha256": "rhs-file-sha256",
            "lattice_xyzt": LATTICE,
        },
        "runs": [{
            "ranks": ranks,
            "grid": grid,
            "qcu": {
                "status": "PASS",
                "within_wall_limit": True,
                "report": str(qcu_path),
                "stdout": str(qcu_stdout),
                "trace_environment": {},
                "report_published": True,
                "devices": devices,
            },
            "quda": {
                "status": "PASS",
                "within_wall_limit": True,
                "report": str(quda_path),
                "stdout": str(quda_stdout),
                "trace_environment": {},
                "report_published": True,
                "devices": devices,
                "independent_residual": {
                    "required": True,
                    "status": "PASS",
                    "report": str(residual_path),
                },
            },
        }],
    }
    path = tmp_path / "manifest.json"
    _write(path, manifest)
    return path


def test_matrix_summary_emits_speedup_only_after_all_gates(tmp_path: Path):
    summary = summarize_manifest(_manifest(tmp_path))
    failed_gates = {
        name: value for name, value in summary["rows"][0]["gates"].items()
        if not value}
    assert summary["status"] == "PASS", failed_gates
    assert all(summary["manifest_gates"].values())
    row = summary["rows"][0]
    assert all(row["gates"].values())
    assert row["speedup"]["qcu_over_quda_steady"] == pytest.approx(0.45 / 0.305)
    assert row["qcu"]["setup_seconds"] == 0.1
    assert row["qcu"]["lifecycle_seconds"] == 0.31
    assert row["qcu"]["device_uuids"] == [_uuid(0)]
    assert row["quda"]["native_true_residual"] == pytest.approx(5.0e-7)
    assert row["quda"]["independent_true_residual"] == pytest.approx(4.0e-7)
    assert "not independent" in row["quda"]["native_residual_semantics"]
    assert row["quda"]["residual_acceptance"] == pytest.approx(ACCEPTANCE)
    assert row["speedup"]["qcu_over_quda_solve_wall"] == pytest.approx(
        0.45 / 0.21)
    assert row["speedup"]["acceptance_metric"] == "qcu_over_quda_solve_wall"


def test_missing_fairness_identities_do_not_pass_as_none_equals_none(
    tmp_path: Path,
):
    manifest_path = _manifest(tmp_path)
    manifest = _load(manifest_path)
    qcu_path = Path(manifest["runs"][0]["qcu"]["report"])
    quda_path = Path(manifest["runs"][0]["quda"]["report"])
    qcu = _load(qcu_path)
    quda = _load(quda_path)
    qcu["inputs"]["gauge"]["sha256"] = None
    qcu["inputs"]["source"]["sha256"] = None
    quda["input"]["lime_sha256"] = None
    quda["input"]["source"]["file_sha256"] = None
    quda["input"]["source"]["global_sha256"] = None
    manifest["configuration"]["gauge_sha256"] = None
    manifest["configuration"]["source_sha256"] = None
    _write(qcu_path, qcu)
    _write(quda_path, quda)
    _write(manifest_path, manifest)

    summary = summarize_manifest(manifest_path)
    gates = summary["rows"][0]["gates"]
    assert summary["status"] == "FAIL"
    assert gates["same_gauge_sha256"] is False
    assert gates["same_rhs_sha256"] is False
    assert summary["rows"][0]["speedup"] is None


@pytest.mark.parametrize("backend", ["qcu", "quda"])
def test_duplicate_device_uuid_fails_each_backend_gate(
    tmp_path: Path, backend: str,
):
    duplicate = [_uuid(0), _uuid(0)]
    manifest_path = _manifest(
        tmp_path, ranks=2,
        qcu_uuids=duplicate if backend == "qcu" else None,
        quda_uuids=duplicate if backend == "quda" else None,
    )
    summary = summarize_manifest(manifest_path)
    gates = summary["rows"][0]["gates"]
    assert summary["status"] == "FAIL"
    assert gates[f"{backend}_device_uuid_count_nonempty_unique"] is False
    assert summary["rows"][0]["speedup"] is None


def test_visible_device_mapping_selects_physical_rocm_card(tmp_path: Path):
    manifest_path = _manifest(tmp_path)
    manifest = _load(manifest_path)
    run = manifest["runs"][0]
    qcu_path = Path(run["qcu"]["report"])
    quda_path = Path(run["quda"]["report"])
    qcu = _load(qcu_path)
    quda = _load(quda_path)
    qcu["ranks"][0]["device"]["uuid"] = None
    quda["mpi"]["ranks"][0]["gpu"]["uuid"] = None
    physical = {
        "1": {
            "uuid": _uuid(0),
            "total_bytes": 1024,
            "peak_used_bytes": 512,
        }
    }
    for backend_record in (run["qcu"], run["quda"]):
        backend_record["devices"] = physical
        backend_record["visible_device_tokens"] = ["1"]
    _write(qcu_path, qcu)
    _write(quda_path, quda)
    _write(manifest_path, manifest)

    summary = summarize_manifest(manifest_path)
    row = summary["rows"][0]
    assert summary["status"] == "PASS", {
        name: value for name, value in row["gates"].items() if not value}
    assert row["qcu"]["device_uuids"] == [_uuid(0)]
    assert row["quda"]["device_uuids"] == [_uuid(0)]
    assert row["gates"]["device_uuid_present_and_matched"] is True


def test_incomplete_manifest_blocks_release(tmp_path: Path):
    manifest_path = _manifest(tmp_path)
    manifest = _load(manifest_path)
    manifest["status"] = "PARTIAL"
    _write(manifest_path, manifest)

    summary = summarize_manifest(manifest_path)
    assert summary["status"] == "FAIL"
    assert summary["manifest_gates"]["manifest_complete"] is False
    assert summary["rows"][0]["speedup"] is None


def test_each_run_must_be_within_wall_limit(tmp_path: Path):
    manifest_path = _manifest(tmp_path)
    manifest = _load(manifest_path)
    manifest["runs"][0]["qcu"]["within_wall_limit"] = False
    _write(manifest_path, manifest)

    summary = summarize_manifest(manifest_path)
    assert summary["status"] == "FAIL"
    assert summary["manifest_gates"]["all_runs_within_wall_limit"] is False
    assert summary["rows"][0]["gates"]["run_records_pass"] is False
    assert summary["rows"][0]["speedup"] is None


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_requested_case_coverage_is_exact(tmp_path: Path, mutation: str):
    manifest_path = _manifest(tmp_path)
    manifest = _load(manifest_path)
    if mutation == "missing":
        manifest["requested_ranks"] = [1, 2]
    else:
        manifest["runs"][0]["kupl"] = {"status": "PASS"}
    _write(manifest_path, manifest)

    summary = summarize_manifest(manifest_path)
    assert summary["status"] != "PASS"
    assert summary["manifest_gates"][
        "requested_backends_and_ranks_covered"] is False
    assert summary["rows"][0]["speedup"] is None
    if mutation == "missing":
        assert summary["coverage"]["missing_ranks"] == [2]
    else:
        assert summary["coverage"]["extra_cases"] == [
            {"ranks": 1, "backend": "kupl"}]


def test_native_residual_does_not_substitute_for_independent_residual(
    tmp_path: Path,
):
    manifest_path = _manifest(tmp_path)
    manifest = _load(manifest_path)
    quda_path = Path(manifest["runs"][0]["quda"]["report"])
    residual_path = Path(
        manifest["runs"][0]["quda"]["independent_residual"]["report"])
    quda = _load(quda_path)
    residual = _load(residual_path)
    residual["relative_true_residual"] = 1.0e-3
    residual["converged"] = False
    _write(quda_path, quda)
    _write(residual_path, residual)

    summary = summarize_manifest(manifest_path)
    row = summary["rows"][0]
    assert row["quda"]["native_true_residual"] == pytest.approx(5.0e-7)
    assert row["quda"]["independent_true_residual"] == pytest.approx(1.0e-3)
    assert row["gates"]["native_residual_pass"] is True
    assert row["gates"]["independent_residual_pass"] is False
    assert summary["status"] == "FAIL"
    assert row["speedup"] is None


def test_missing_external_report_never_falls_back_to_native_true_res(
    tmp_path: Path,
):
    manifest_path = _manifest(tmp_path)
    manifest = _load(manifest_path)
    del manifest["runs"][0]["quda"]["independent_residual"]
    _write(manifest_path, manifest)

    summary = summarize_manifest(manifest_path)
    row = summary["rows"][0]
    assert row["quda"]["native_true_residual"] == pytest.approx(5.0e-7)
    assert row["quda"]["independent_true_residual"] is None
    assert row["gates"]["independent_residual_pass"] is False
    assert row["speedup"] is None


def test_summary_uses_persistent_qcu_runtime_for_both_metrics(
    tmp_path: Path,
):
    manifest_path = _manifest(tmp_path)
    manifest = _load(manifest_path)
    manifest["configuration"]["qcu"] = {"reuse_runtime": True}
    qcu_path = Path(manifest["runs"][0]["qcu"]["report"])
    qcu = _load(qcu_path)
    rank = qcu["ranks"][0]
    rank["timings"]["clover_schur_bistabcg_solve_reused"] = {
        "steady": {
            "operation_seconds": {"median": 0.15},
            "lifecycle_seconds": {"median": 0.15},
            "samples": [
                {"operation_seconds": 0.15, "lifecycle_seconds": 0.15,
                 "iterations": 18},
                {"operation_seconds": 0.16, "lifecycle_seconds": 0.16,
                 "iterations": 18},
            ],
        },
        "reuse": {
            "setup_seconds": 0.2,
            "close_seconds": 0.1,
            "amortized_lifecycle_seconds": 0.2,
        },
    }
    rank["solver_native_reused"] = {"all_samples_converged": True}
    _write(qcu_path, qcu)
    _write(manifest_path, manifest)

    summary = summarize_manifest(manifest_path)
    row = summary["rows"][0]
    assert summary["status"] == "PASS", {
        name: value for name, value in row["gates"].items() if not value}
    assert row["gates"]["qcu_persistent_runtime"] is True
    assert row["qcu"]["runtime_reuse"]["enabled"] is True
    assert row["qcu"]["operation_seconds"] == pytest.approx(0.15)
    assert row["qcu"]["lifecycle_seconds"] == pytest.approx(0.2)
    assert row["speedup"]["qcu_over_quda_operation"] == pytest.approx(
        0.3 / 0.15)
    assert row["speedup"]["qcu_over_quda_steady"] == pytest.approx(0.45 / 0.2)
    assert row["speedup"]["qcu_over_quda_solve_wall"] == pytest.approx(
        0.45 / 0.15)
    assert row["speedup"]["acceptance_metric"] == "qcu_over_quda_solve_wall"


def test_deferred_residual_is_never_run_on_dcu_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("SLURM_JOB_PARTITION", "kshdnormal")
    with pytest.raises(SystemExit, match="refusing CPU-only"):
        _run_deferred_residuals(
            SimpleNamespace(wall_limit=540.0),
            tmp_path / "run-manifest.json")


def test_dcu_matrix_requires_deferred_residual_phase(monkeypatch):
    from pyqcu.testing.simple.mpi.cuda_quda_matrix import main

    monkeypatch.setenv("SLURM_JOB_PARTITION", "kshdnormal")
    with pytest.raises(SystemExit, match="requires --defer-independent"):
        main([
            "--gauge", "fake.lime", "--lat", "4", "4", "4", "8",
            "--rank-case", "1:1,1,1,1", "--backends", "qcu,quda",
            "--result-root", "/tmp/unused-cuda-quda-matrix",
        ])


def test_deferred_residual_cpu_phase_finalizes_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    manifest_path = _manifest(tmp_path)
    manifest = _load(manifest_path)
    run = manifest["runs"][0]
    solution_path = Path(run["quda"]["report"]).with_name("solution.h5")
    solution_path.write_bytes(b"solution")
    run["quda"]["solution"] = str(solution_path)
    run["quda"]["independent_residual"]["status"] = "DEFERRED"
    manifest["status"] = "PARTIAL"
    manifest["configuration"].update({
        "gauge": str(tmp_path / "gauge.lime"),
        "source": str(tmp_path / "source.h5"),
        "dtype": "c64",
        "tolerance": 1.0e-6,
    })
    _write(manifest_path, manifest)
    monkeypatch.delenv("SLURM_JOB_PARTITION", raising=False)

    def fake_run(command, **kwargs):
        report = Path(command[command.index("--report") + 1])
        _write(report, _independent_report(
            ranks=1, grid=GRID_1,
            source_path=Path(manifest["configuration"]["source"]),
            gauge_path=Path(manifest["configuration"]["gauge"]),
            solution_path=solution_path))
        return 0, False, 0.1

    monkeypatch.setattr(cuda_quda_matrix, "_run_captured_command", fake_run)
    status = _run_deferred_residuals(
        SimpleNamespace(wall_limit=540.0, child_timeout=180.0,
                        mpirun="mpirun"),
        manifest_path)
    assert status == 0
    updated = _load(manifest_path)
    assert updated["status"] == "COMPLETE"
    independent = updated["runs"][0]["quda"]["independent_residual"]
    assert independent["status"] == "PASS"
    assert independent["execution_node"] == "CPU partition"
    summary = _load(tmp_path / "summary.json")
    assert summary["status"] == "PASS"


def test_matrix_materialises_one_canonical_rhs(tmp_path: Path):
    args = SimpleNamespace(
        lat=LATTICE, dtype="c64", source=None, seed=42)
    path = _materialise_source(args, tmp_path)
    with h5py.File(path, "r") as handle:
        assert handle["data"].shape == (4, 3, *LATTICE)
        assert handle["data"].dtype.name == "complex64"
        assert handle.attrs["layout"] == "[spin,color,x,y,z,t]"
        assert handle.attrs["origin"] == "seeded-global-pcg64"


def test_compare_matrix_dry_run_is_isolated_and_records_semantics():
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO) + os.pathsep + environment.get(
        "PYTHONPATH", "")
    completed = subprocess.run(
        [
            sys.executable, "-B", str(DIR / "compare_three_way.py"),
            "--matrix", "--dry-run", "--backends", "cuda,quda",
            "--gauge", "fake.lime", "--lat", "4", "4", "4", "8",
            "--rank-case", "1:1,1,1,1", "--rank-case", "2:1,1,1,2",
            "--defer-independent-residual",
        ], cwd=REPO, env=environment, text=True, capture_output=True,
        check=False, timeout=120)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    manifest = payload["manifest"]
    assert manifest["configuration"]["trace"].startswith("disabled")
    assert manifest["configuration"]["quda"]["tuning"].startswith("disabled")
    assert manifest["configuration"]["qcu"]["solver_plan"]["set_plan"] == 1
    assert manifest["configuration"]["qcu"]["reuse_runtime"] is True
    assert manifest["configuration"]["quda"][
        "independent_residual_policy"] == "deferred to a CPU partition"
    assert manifest["configuration"]["grid_semantics"]["time_axis"] == 3
    assert manifest["configuration"]["source"].endswith("source.h5")
    assert set(payload["commands"]) == {
        "r1-qcu", "r1-quda", "r2-qcu", "r2-quda"}
    assert "--solution-output" in payload["commands"]["r1-quda"]
    assert "--reuse-runtime" in payload["commands"]["r1-qcu"]
    assert "--source" in payload["commands"]["r1-qcu"]
