"""CPU-only tests for the MG report builder."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest


MODULE_PATH = Path(__file__).with_name("build_mg_report.py")
SPEC = importlib.util.spec_from_file_location("build_mg_report", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
report = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = report
SPEC.loader.exec_module(report)


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def _fixture(tmp_path: Path, *, missing: bool = False) -> Path:
    case_dir = tmp_path / "case-synth-c64-l3"
    case_dir.mkdir()
    benchmark_path = case_dir / "benchmark.json"
    trace_path = case_dir / "trace.json"
    matrix_path = case_dir / "matrix_summary.json"

    sides: Dict[str, Any] = {}
    for side in ("pyqcu", "quda"):
        side_doc: Dict[str, Any] = {
            "side": side,
            "timing": {
                "cold_seconds": 0.05,
                "warmups": [
                    {
                        "seconds": 0.02,
                        "iterations": 4,
                        "true_residual_rel": 2.0e-6,
                    },
                    {
                        "seconds": 0.03,
                        "iterations": 4,
                        "true_residual_rel": 2.0e-6,
                    },
                ],
                "steady": {
                    "samples_seconds": [0.01, 0.012],
                    "median_seconds": 0.011,
                    "mad_seconds": 0.001,
                },
            },
            "iterations": {"samples": [5, 5], "median": 5.0},
            "true_residual": {
                "samples_rel": [1.0e-6, 1.0e-6],
                "max_rel": 1.0e-6,
            },
        }
        if not missing:
            side_doc["device"] = "cuda:0 Tesla V100-SXM2-32GB"
        else:
            side_doc.pop("true_residual")
            for warmup in side_doc["timing"]["warmups"]:
                warmup.pop("true_residual_rel", None)
        sides[side] = side_doc

    benchmark = {
        "schema": {"name": "pyqcu.strict-vs-quda.benchmark", "version": 1},
        "protocol": {
            "case_id": "synth-c64-l3",
            "precision": {"name": "c64"},
            "lattice_xyzt": [8, 8, 8, 16],
            "levels": 3,
        },
        "sides": sides,
    }
    if not missing:
        benchmark["protocol"]["device"] = "cuda:0 Tesla V100-SXM2-32GB"

    pyqcu_stages = [
        {"level": 0, "name": "fine_pre_smoother", "seconds": 0.10},
        {"level": 0, "name": "fine_restriction", "seconds": 0.20},
        {"level": 0, "name": "fine_prolongation", "seconds": 0.20},
        {"level": 0, "name": "fine_post_smoother", "seconds": 0.10},
        {"level": 0, "name": "fine_correction_residual", "seconds": 0.05},
        {"level": 1, "name": "level_pre_smoother", "seconds": 0.30},
        {"level": 1, "name": "level_restriction", "seconds": 0.10},
        {"level": 1, "name": "level_prolongation", "seconds": 0.10},
        {"level": 1, "name": "level_post_smoother", "seconds": 0.20},
        {"level": 1, "name": "coarsest_bicgstab", "seconds": 0.50},
    ]
    quda_stages = [
        {
            "level": 0,
            "name": "pre_smoother_iterations",
            "seconds": 0.10,
            "pre_iterations": 2,
        },
        {"level": 0, "name": "restrict", "seconds": 0.20},
        {"level": 0, "name": "prolongate", "seconds": 0.20},
        {
            "level": 0,
            "name": "post_smoother_iterations",
            "seconds": 0.10,
            "post_iterations": 1,
        },
        {
            "level": 1,
            "name": "coarse_solver_iterations",
            "seconds": 0.30,
            "coarse_iterations": 7,
        },
        {
            "level": 1,
            "name": "coarsest_solve",
            "seconds": 0.50,
            "pre_iterations": 100,
            "post_iterations": 123,
        },
    ]
    curve = [] if missing else [
        {"iteration": 0, "relative": 1.0, "kind": "initial"},
        {"iteration": 1, "relative": 0.1, "kind": "estimate"},
        {
            "iteration": 1,
            "relative": 0.01,
            "true_relative": 0.01,
            "true_r2": 1.0e-4,
            "b2": 1.0,
            "kind": "outer_iteration",
        },
    ]
    trace = {
        "schema": {"name": "pyqcu.strict-vs-quda.iteration-trace", "version": 1},
        "sides": {
            "pyqcu": {
                "steady": [{
                    "solve_index": 2,
                    "iterations": 5,
                    "stages": pyqcu_stages,
                    "residual_curve": curve,
                }],
            },
            "quda": {
                "steady": [{
                    "solve_index": 2,
                    "iterations": 5,
                    "stages": quda_stages,
                    "residual_curve": curve,
                }],
            },
        },
    }
    matrix = {
        "schema": {"name": "pyqcu.mg-matrix", "version": 1},
        "cases": [{
            "id": "synth-c64-l3",
            "status": "ok",
            "case": {
                "id": "synth-c64-l3",
                "precision": "c64",
                "lattice": [8, 8, 8, 16],
                "levels": 3,
            },
            "runs": [{"output": str(benchmark_path)}],
            "trace": {"output": str(trace_path)},
        }],
    }
    _write_json(benchmark_path, benchmark)
    _write_json(trace_path, trace)
    _write_json(matrix_path, matrix)
    return matrix_path


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_report_csv_row_counts_and_finest_iteration_semantics(tmp_path):
    matrix = _fixture(tmp_path)
    outdir = tmp_path / "report"
    summary = report.build_report([matrix], outdir)

    matrix_rows = _csv_rows(outdir / "mg_matrix.csv")
    assert len(matrix_rows) == 6  # two sides x cold/warmup/steady
    assert {row["phase"] for row in matrix_rows} == {
        "cold", "warmup", "steady"}
    pyqcu_steady = next(
        row for row in matrix_rows
        if row["side"] == "pyqcu" and row["phase"] == "steady")
    assert pyqcu_steady["finest_iterations"] == "5"
    assert pyqcu_steady["trace"] == "on"
    assert pyqcu_steady["device"]
    assert float(pyqcu_steady["finest_residual"]) == pytest.approx(1.0e-6)

    level_rows = _csv_rows(outdir / "mg_levels.csv")
    assert len(level_rows) == 6  # two sides x three declared levels
    quda_level1 = next(
        row for row in level_rows
        if row["side"] == "quda" and row["level"] == "1")
    # 7 from coarse_solver_iterations + (123-100) from coarsest_solve.
    assert quda_level1["total_iterations"] == "30"
    missing_level2 = next(
        row for row in level_rows
        if row["side"] == "pyqcu" and row["level"] == "2")
    assert missing_level2["total_seconds"] == ""
    assert "level.2.total_seconds" in missing_level2["missing_fields"]
    assert summary["coverage"]["side_coverage"] == 1.0


def test_reliable_update_trace_rows_are_deduplicated(tmp_path):
    points = report._deduplicate_points([
        {"iteration": 1, "relative": 0.1, "kind": "estimate"},
        {
            "iteration": 1,
            "iterated_relative": 0.09,
            "true_r2": 1.0e-4,
            "b2": 1.0,
            "kind": "outer_iteration",
        },
    ])
    assert len(points) == 1
    assert points[0]["relative"] == pytest.approx(0.01)
    assert points[0]["kind"] == "true_residual"


def test_raw_quda_trace_uses_absolute_coarsest_counts(tmp_path):
    trace = tmp_path / "quda_mg.tsv"
    trace.write_text(
        "trace_version\t1\n"
        "cycle_begin\t1\t1\t2\t1\tcoarse\t0.1\n"
        "stage\t1\t1\tcoarsest_solve\t0.5\t100\t123\t-1\t1\t0.2\n"
        "outer_begin\t1\t0.1\t0.1\t1\t0.1\t0.3\n"
        "outer_iteration\t1\t1\t0.1\t0.1\t1\t0.1\t-1\t0\t0\t0.4\n"
        "outer_iteration\t1\t1\t0.09\t0.09\t1\t0.09\t0.0001\t0.01\t0.01\t0.5\n",
        encoding="utf-8")
    sections = report._parse_quda_mg_tsv(trace)
    assert len(sections) == 1
    assert sections[0]["iterations"] == 1
    assert sections[0]["residual_curve"][0]["relative"] == pytest.approx(0.01)
    pre, post, coarse = report._stage_iterations(sections[0]["stages"][0])
    assert (pre, post, coarse) == (0, 0, 23)


def test_full_matrix_units_schema_is_consumed(tmp_path):
    units = []
    for side in ("pyqcu", "quda"):
        units.append({
            "unit_id": (
                f"{side}__v100__8x8x8x16__c64__l2__trace-off"),
            "unit": {
                "unit_id": (
                    f"{side}__v100__8x8x8x16__c64__l2__trace-off"),
                "side": side,
                "device": "v100",
                "lattice": [8, 8, 8, 16],
                "precision": "c64",
                "levels": 2,
                "trace": "off",
                "phase_records": [
                    {"phase": "cold"},
                    {"phase": "warmup"},
                    {"phase": "warmup"},
                    {"phase": "steady"},
                    {"phase": "steady"},
                ],
            },
            "status": "ok",
            "outer_iterations": {"samples": [5], "median": 5},
            "finest_level_residual": {"level": 0, "value": 1.0e-6},
            "layers": {
                "0": {
                    "total_seconds": 0.5,
                    "iterations": 5,
                    "phases": {
                        "pre_smoother": {"seconds": 0.1},
                        "post_smoother": {"seconds": 0.1},
                        "restrict": {"seconds": 0.1},
                        "prolongate": {"seconds": 0.1},
                        "coarse_solver": {"seconds": 0.0},
                        "other": {"seconds": 0.1},
                    },
                },
                "1": {
                    "total_seconds": 0.3,
                    "iterations": 20,
                    "phases": {
                        "coarse_solver": {"seconds": 0.2},
                        "other": {"seconds": 0.1},
                    },
                },
            },
        })
    path = tmp_path / "matrix_summary.json"
    _write_json(path, {
        "schema": {"name": "pyqcu.mg-matrix-full.summary", "version": 1},
        "matrix_status": "complete",
        "units": units,
    })
    outdir = tmp_path / "full-report"
    summary = report.build_report([path], outdir)
    matrix_rows = _csv_rows(outdir / "mg_matrix.csv")
    assert len(matrix_rows) == 6
    assert {row["case_id"] for row in matrix_rows} == {
        "v100__8x8x8x16__c64__l2__trace-off"}
    level_rows = _csv_rows(outdir / "mg_levels.csv")
    pyqcu_level0 = next(
        row for row in level_rows
        if row["side"] == "pyqcu" and row["level"] == "0")
    assert pyqcu_level0["total_seconds"] == "0.5"
    assert pyqcu_level0["total_iterations"] == "5"
    assert pyqcu_level0["restriction"] == "0.1"
    assert pyqcu_level0["prolongation"] == "0.1"
    assert pyqcu_level0["other"] == "0.1"
    assert summary["coverage"]["level_cells_observed"] == 4


def test_missing_fields_degrade_without_invention(tmp_path):
    matrix = _fixture(tmp_path, missing=True)
    outdir = tmp_path / "report-missing"
    summary = report.build_report([matrix], outdir)
    rows = _csv_rows(outdir / "mg_matrix.csv")
    assert rows
    assert all(row["device"] == "" for row in rows)
    assert all(row["finest_residual"] == "" for row in rows)
    missing = summary["coverage"]["missing_field_counts"]
    assert missing["device"] >= 1
    assert missing["finest_residual"] >= 1


def test_negative_other_is_rejected(tmp_path):
    input_path = tmp_path / "negative-other.json"
    _write_json(input_path, {
        "protocol": {
            "case_id": "negative-other",
            "precision": {"name": "c64"},
            "lattice_xyzt": [8, 8, 8, 16],
            "levels": 1,
        },
        "sides": {
            "pyqcu": {
                "timing": {"steady": {"median_seconds": 1.0}},
                "iterations": 3,
                "level_profiles": {
                    "0": {
                        "total_seconds": 0.1,
                        "pre_smoother": 0.2,
                        "post_smoother": 0.0,
                        "restriction": 0.0,
                        "prolongation": 0.0,
                        "coarse_solver": 0.0,
                    },
                },
            },
        },
    })
    with pytest.raises(report.ReportError, match="other is negative"):
        report.build_report([input_path], tmp_path / "bad-report")


def test_plot_and_latex_artifacts_are_nonempty_and_consistent(tmp_path):
    matrix = _fixture(tmp_path)
    outdir = tmp_path / "report-artifacts"
    source_tex = tmp_path / "source.tex"
    source_tex.write_text(
        "\\documentclass{article}\n\\usepackage{booktabs}\n", encoding="utf-8")
    before = source_tex.read_text(encoding="utf-8")

    report.main([
        "--input", str(matrix),
        "--outdir", str(outdir),
        "--tex", str(source_tex),
    ])
    assert source_tex.read_text(encoding="utf-8") == before

    stems = (
        "mg_level_time_stack",
        "mg_level_iterations",
        "mg_finest_residual",
        "mg_speedup_device",
    )
    for stem in stems:
        for suffix in (".svg", ".pdf"):
            path = outdir / f"{stem}{suffix}"
            assert path.is_file()
            assert path.stat().st_size > 100

    tables = (outdir / "tables.tex").read_text(encoding="utf-8")
    assert r"\toprule" in tables
    assert r"\midrule" in tables
    assert r"\bottomrule" in tables
    report.validate_latex_tables(tables)

    generated = (outdir / "report_generated.tex").read_text(encoding="utf-8")
    assert r"\input{\detokenize{tables.tex}}" in generated
    assert generated.count(r"\includegraphics") == 4
