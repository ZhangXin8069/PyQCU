"""CPU-only tests for the full MultiGrid matrix orchestrator."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bench_mg_matrix_full as matrix


def _unit(
        *,
        side: str = "pyqcu",
        precision: str = "c64",
        lattice: tuple[int, int, int, int] = (8, 8, 8, 16),
        trace: str = "off",
        levels: int = 1,
        device: str = "v100",
) -> matrix.MatrixUnit:
    return matrix.MatrixUnit(
        side=side,
        precision=precision,
        lattice=lattice,
        trace=trace,
        levels=levels,
        device=device,
    )


def test_full_matrix_count_is_unique() -> None:
    units = matrix.build_units()
    assert len(units) == 144
    assert len({unit.unit_id for unit in units}) == 144
    assert sum(len(unit.phase_records) for unit in units) == 1152
    assert all(len(unit.phase_records) == 8 for unit in units)


def test_filters_are_cartesian_and_countable() -> None:
    selection = matrix.Selection(
        sides=("quda",),
        levels=(1,),
        precisions=("c128",),
        lattices=matrix.LATTICES,
        traces=("on", "off"),
        devices=("v100", "p100"),
    )
    units = matrix.select_units(matrix.build_units(), selection)
    assert len(units) == 12
    assert sum(len(unit.phase_records) for unit in units) == 96
    assert {unit.side for unit in units} == {"quda"}
    assert {unit.levels for unit in units} == {1}
    assert {unit.precision for unit in units} == {"c128"}


def test_unit_id_is_stable_and_encodes_dimensions() -> None:
    unit = _unit(
        side="quda",
        precision="c128",
        lattice=(16, 32, 32, 48),
        trace="on",
        levels=3,
        device="p100",
    )
    assert unit.unit_id == (
        "quda__p100__16x32x32x48__c128__l3__trace-on")
    assert _unit(
        side="quda",
        precision="c128",
        lattice=(16, 32, 32, 48),
        trace="on",
        levels=3,
        device="p100",
    ).unit_id == unit.unit_id


def test_collector_command_legacy_mapping(tmp_path: Path) -> None:
    unit = _unit(trace="on", levels=3, precision="c128")
    command = matrix._collector_command(
        unit,
        collector=matrix.DEFAULT_COLLECTOR,
        output_dir=tmp_path,
        collector_interface="legacy",
        extra_args=("--custom", "value"),
    )
    assert "--side" in command
    assert command[command.index("--levels") + 1] == "3"
    assert command[command.index("--precision") + 1] == "c128"
    assert command[command.index("--profile") + 1] == "smoke"
    assert command[command.index("--cache-expect") + 1] == "miss"
    assert command[command.index("--repeats") + 1] == "5"
    assert "--allow-trace" in command
    assert "--phases" not in command
    assert "--mpi-ranks" not in command
    assert command[-2:] == ["--custom", "value"]


def test_collector_command_frozen_and_p100_mapping(tmp_path: Path) -> None:
    unit = _unit(device="p100", trace="off", levels=2)
    frozen = matrix._collector_command(
        unit,
        collector=matrix.DEFAULT_COLLECTOR,
        output_dir=tmp_path,
        collector_interface="frozen",
    )
    assert frozen[frozen.index("--phases") + 1] == "cold,warmup,steady"
    assert frozen[frozen.index("--mpi-ranks") + 1] == "2"
    assert frozen.count("--mpi-ranks") == 1
    grid_start = frozen.index("--process-grid") + 1
    assert frozen[grid_start:grid_start + 4] == ["2", "1", "1", "1"]
    assert frozen.count("--process-grid") == 1
    assert "2,1,1,1" not in frozen
    assert frozen[frozen.index("--device") + 1] == "p100"
    assert "--resume" not in frozen
    assert frozen.count("mpirun") == 0

    legacy = matrix._collector_command(
        unit,
        collector=matrix.DEFAULT_COLLECTOR,
        output_dir=tmp_path,
        collector_interface="legacy",
    )
    assert legacy[:3] == ["mpirun", "-np", "2"]
    assert legacy.count("mpirun") == 1


def test_collector_interface_defaults_to_frozen() -> None:
    args = matrix._parser().parse_args([])
    assert args.collector_interface == "frozen"
    unit = _unit(device="p100")
    command = matrix._collector_command(
        unit,
        collector=matrix.DEFAULT_COLLECTOR,
        output_dir=Path("/tmp/mg-full-test"),
        collector_interface=args.collector_interface,
    )
    assert command.count("mpirun") == 0
    grid_start = command.index("--process-grid") + 1
    assert command[grid_start:grid_start + 4] == ["2", "1", "1", "1"]


def test_collector_environment_trace_paths(tmp_path: Path) -> None:
    unit = _unit(trace="on")
    env = matrix.collector_environment(
        unit, tmp_path, base_environment={"PATH": "/bin"})
    assert env["PYQCU_MG_MATRIX_MPI_RANKS"] == "1"
    assert env["PYQCU_STRICT_TRACE_FILE"].endswith("pyqcu.tsv")
    assert env["QUDA_MG_TRACE_FILE"].endswith("quda.tsv")
    assert env["PATH"] == "/bin"


def test_asset_paths_and_per_unit_cache_directories(tmp_path: Path) -> None:
    roots = matrix.AssetRoots(
        asset_root=tmp_path / "assets",
        cache_root=tmp_path / "cache",
    )
    c64 = _unit(side="quda", precision="c64", levels=2)
    c64_assets = matrix.resolve_unit_assets(c64, roots, tmp_path)
    assert Path(c64_assets["gauge_path"]).name == (
        "gauge_8x8x8x16_m0.05_seed42_c64.h5")
    assert Path(c64_assets["nullvec_path"]).name == (
        "L8x8x8x16_nvec12_full_c64.h5")
    assert Path(c64_assets["quda_nullvec_prefix"]).name == (
        "L8x8x8x16_nvec12_quda")
    assert Path(c64_assets["quda_nullvec_manifest"]).name == (
        "L8x8x8x16_nvec12_quda.v1.json")

    c128 = _unit(side="pyqcu", precision="c128")
    c128_assets = matrix.resolve_unit_assets(c128, roots, tmp_path)
    # Without a canonical c128 asset the resolver falls back to the canonical
    # c64 input and records the mixed-input protocol.
    assert Path(c128_assets["gauge_path"]).name.endswith("_c64.h5")
    assert Path(c128_assets["nullvec_path"]).name.endswith("_c64.h5")
    assert c128_assets["input_storage_precision"] == "c64"
    assert c128_assets["notes"]
    assert c64_assets["strict_cache_dir"] != c128_assets["strict_cache_dir"]
    assert Path(c64_assets["strict_cache_dir"]).parent == tmp_path / "cache"


def test_quda_asset_validation_uses_manifest_artifacts(tmp_path: Path) -> None:
    roots = matrix.AssetRoots(
        asset_root=tmp_path / "assets",
        qio_root=tmp_path / "qio",
    )
    unit = _unit(side="quda", precision="c64")
    assets = matrix.resolve_unit_assets(unit, roots, tmp_path)
    for key in ("gauge_path", "nullvec_path"):
        path = Path(assets[key])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"asset")
    manifest = Path(assets["quda_nullvec_manifest"])
    artifact = Path(assets["quda_nullvec_prefix"]).with_name(
        Path(assets["quda_nullvec_prefix"]).name + "_level_0_nvec_12")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"qio")
    manifest.write_text(json.dumps({
        "artifacts": [{"path": artifact.name}],
    }), encoding="utf-8")

    valid = matrix.validate_unit_assets(unit, assets)
    assert valid["ok"] is True
    assert valid["qio_artifacts"] == [str(artifact)]

    artifact.unlink()
    invalid = matrix.validate_unit_assets(unit, assets)
    assert invalid["ok"] is False
    assert invalid["missing"] == [{
        "kind": "qio_artifact",
        "path": str(artifact),
    }]


def test_c128_input_assets_block_without_calling_runner(tmp_path: Path) -> None:
    unit = _unit(side="pyqcu", precision="c128")
    called: list[list[str]] = []

    def runner(command, _cwd, _env, _timeout):
        called.append(list(command))
        raise AssertionError("runner must not start when assets are missing")

    records = matrix.run_matrix(
        [unit],
        output_dir=tmp_path,
        asset_roots=matrix.AssetRoots(asset_root=tmp_path / "missing"),
        runner=runner,
        source_version="test-revision",
    )
    assert called == []
    assert records[-1]["status"] == "blocked"
    assert records[-1]["error_code"] == "input_asset_missing"
    # c128 falls back to the canonical c64 input; the missing-asset report
    # must therefore name the fallback paths, not a non-existent c128 file.
    assert "gauge_8x8x8x16_m0.05_seed42_c64.h5" in records[-1]["error_reason"]
    assert "L8x8x8x16_nvec12_full_c64.h5" in records[-1]["error_reason"]

    quda_unit = _unit(side="quda", precision="c128")
    quda_records = matrix.run_matrix(
        [quda_unit],
        output_dir=tmp_path,
        asset_roots=matrix.AssetRoots(asset_root=tmp_path / "missing"),
        runner=runner,
        source_version="test-revision",
    )
    assert quda_records[-1]["status"] == "blocked"
    assert "L8x8x8x16_nvec12_quda" in quda_records[-1]["error_reason"]

    summary = matrix.summarize_matrix([unit], output_dir=tmp_path)
    assert summary["coverage"]["missing_unit_ids"] == [unit.unit_id]
    assert summary["coverage"]["blocked_unit_ids"] == [unit.unit_id]
    assert summary["units"][0]["error_code"] == "input_asset_missing"


def test_extra_args_merge_order_and_conflicts(tmp_path: Path) -> None:
    unit = _unit(side="pyqcu", precision="c64", levels=2)
    report = matrix._collector_command_report(
        unit,
        collector=matrix.DEFAULT_COLLECTOR,
        output_dir=tmp_path,
        collector_interface="frozen",
        extra_args=(
            "--levels", "99",
            "--custom", "value",
            "--gauge-path", "/explicit/override.h5",
        ),
    )
    command = report["argv"]
    assert command[command.index("--levels") + 1] == "2"
    assert command.count("--levels") == 1
    assert "/explicit/override.h5" not in command
    assert command[-2:] == ["--custom", "value"]
    assert [item["option"] for item in report["argument_conflicts"]] == [
        "--gauge-path", "--levels"]


def test_state_records_conflicts_and_isolated_cache_dirs(tmp_path: Path) -> None:
    first = _unit(trace="off")
    second = _unit(trace="on")

    def runner(command, _cwd, _env, _timeout):
        output = Path(command[command.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({
            "sides": {"pyqcu": {"status": "ok"}},
        }), encoding="utf-8")
        return matrix.RunResult(returncode=0)

    records = matrix.run_matrix(
        [first, second],
        output_dir=tmp_path,
        extra_args=("--levels", "99"),
        runner=runner,
        source_version="test-revision",
    )
    assert all(record["status"] == "ok" for record in records)
    assert all(
        record["argument_conflicts"][0]["option"] == "--levels"
        for record in records)
    cache_dirs = []
    for record in records:
        command = record["argv"]
        cache_dirs.append(command[command.index("--strict-cache-dir") + 1])
    assert len(set(cache_dirs)) == 2

    final_state = matrix._read_state(matrix._state_path(tmp_path))[-2:]
    assert all(
        record["argument_conflicts"][0]["resolution"] == "unit_specific"
        for record in final_state)


def test_resume_skips_completed_unit(tmp_path: Path) -> None:
    first = _unit(trace="off")
    second = _unit(trace="on")
    state_path = matrix._state_path(tmp_path)
    matrix._append_state(state_path, {
        "unit_id": first.unit_id,
        "status": "ok",
        "output_path": str(matrix.unit_output_path(tmp_path, first)),
    })
    called: list[list[str]] = []

    def runner(command, _cwd, _env, _timeout):
        called.append(list(command))
        output = matrix.unit_output_path(tmp_path, second)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({
            "sides": {"pyqcu": {"status": "ok"}},
        }), encoding="utf-8")
        return matrix.RunResult(returncode=0)

    records = matrix.run_matrix(
        [first, second],
        output_dir=tmp_path,
        resume=True,
        runner=runner,
        source_version="test-revision",
    )
    assert len(called) == 1
    assert [record["unit_id"] for record in records] == [
        first.unit_id, second.unit_id]
    assert records[0]["git_describe"] == "test-revision"


def test_blocked_propagation_is_not_success(tmp_path: Path) -> None:
    unit = _unit(device="p100")

    def runner(_command, _cwd, _env, _timeout):
        return matrix.RunResult(
            returncode=2,
            stderr="P100 sm_60 unavailable",
            blocked=True,
            reason="P100 unavailable",
        )

    records = matrix.run_matrix(
        [unit],
        output_dir=tmp_path,
        runner=runner,
        source_version="test-revision",
    )
    assert records[-1]["status"] == "blocked"
    state = matrix._read_state(matrix._state_path(tmp_path))
    assert state[-1]["status"] == "blocked"
    assert state[-1]["error_reason"] == "P100 unavailable"


def test_side_unsupported_is_blocked_and_missing(tmp_path: Path) -> None:
    unit = _unit(side="pyqcu", levels=1)

    def runner(command, _cwd, _env, _timeout):
        output_value = command[command.index("--output") + 1]
        output = Path(output_value)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({
            "sides": {
                "pyqcu": {
                    "status": "side_unsupported",
                    "reason": {
                        "code": "levels_1_unsupported",
                        "detail": "PyQCU strict fused FGMRES requires levels >= 2",
                    },
                },
            },
        }), encoding="utf-8")
        return matrix.RunResult(returncode=2)

    records = matrix.run_matrix(
        [unit],
        output_dir=tmp_path,
        runner=runner,
        source_version="test-revision",
    )
    assert records[-1]["status"] == "blocked"
    assert "requires levels >= 2" in records[-1]["error_reason"]

    summary = matrix.summarize_matrix([unit], output_dir=tmp_path)
    assert summary["coverage"]["completed_units"] == 0
    assert summary["coverage"]["missing_unit_ids"] == [unit.unit_id]
    assert summary["coverage"]["blocked_unit_ids"] == [unit.unit_id]


def test_summary_reports_metrics_and_missing_units(tmp_path: Path) -> None:
    complete = _unit(trace="on")
    missing = _unit(side="quda", trace="off")
    output = matrix.unit_output_path(tmp_path, complete)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "schema": {"name": "pyqcu.strict-vs-quda.iteration-trace", "version": 1},
        "sides": {
            "pyqcu": {
                "iterations": {"samples": [2, 2], "median": 2.0},
                "true_residual": {"max_rel": 1.0e-8},
                "steady": [{
                    "iterations": 2,
                    "residual_curve": [
                        {"iteration": 0, "relative": 1.0},
                        {"iteration": 2, "relative": 1.0e-8},
                    ],
                    "stages": [
                        {
                            "level": 0,
                            "name": "fine_pre_smoother",
                            "seconds": 0.1,
                        },
                        {
                            "level": 0,
                            "name": "fine_post_smoother",
                            "seconds": 0.2,
                        },
                        {
                            "level": 0,
                            "name": "fine_restriction",
                            "seconds": 0.3,
                        },
                        {
                            "level": 0,
                            "name": "fine_prolongation",
                            "seconds": 0.4,
                        },
                        {
                            "level": 0,
                            "name": "coarsest_bicgstab",
                            "seconds": 0.5,
                        },
                    ],
                }],
            },
        },
    }), encoding="utf-8")
    state_path = matrix._state_path(tmp_path)
    matrix._append_state(state_path, {
        "unit_id": complete.unit_id,
        "status": "ok",
        "output_path": str(output),
        "trace_files": matrix.trace_paths(tmp_path, complete),
    })
    matrix._append_state(state_path, {
        "unit_id": missing.unit_id,
        "status": "blocked",
        "output_path": str(matrix.unit_output_path(tmp_path, missing)),
        "error_reason": "QUDA MPI unavailable",
    })

    summary = matrix.summarize_matrix(
        [complete, missing],
        output_dir=tmp_path,
        full_expected_units=144,
    )
    assert summary["matrix_status"] == "incomplete"
    assert summary["coverage"]["expected_units"] == 2
    assert summary["coverage"]["completed_units"] == 1
    assert summary["coverage"]["missing_unit_ids"] == [missing.unit_id]
    assert summary["coverage"]["blocked_unit_ids"] == [missing.unit_id]
    row = summary["units"][0]
    assert row["status"] == "ok"
    assert row["finest_level_residual"]["value"] == pytest.approx(1.0e-8)
    assert row["outer_iterations"]["median"] == 2.0
    assert set(row["layers"]["0"]["phases"]) == set(matrix.PHASE_CATEGORIES)
    assert row["layers"]["0"]["phases"]["pre_smoother"]["seconds"] == (
        pytest.approx(0.1))
    assert row["layers"]["0"]["phases"]["other"]["seconds"] == pytest.approx(0.0)


def test_parser_filters_and_defaults() -> None:
    args = matrix._parser().parse_args([
        "--only-side", "quda,pyqcu",
        "--only-levels", "2",
        "--only-precision", "c64",
        "--only-lattice", "8x8x8x16",
        "--only-trace", "off",
        "--only-device", "v100",
    ])
    selection = matrix._selection_from_args(args)
    units = matrix.select_units(matrix.build_units(), selection)
    assert len(units) == 2
    assert {unit.side for unit in units} == {"pyqcu", "quda"}


def test_summary_can_be_written_to_default_name(tmp_path: Path) -> None:
    units = [_unit()]
    summary = matrix.summarize_matrix(units, output_dir=tmp_path)
    path = tmp_path / "matrix_summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["coverage"]["expected_units"] == 1
    assert value["coverage"]["missing_unit_ids"] == [units[0].unit_id]
