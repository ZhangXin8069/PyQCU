"""CPU-only tests for the final matrix report assembler."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import assemble_final_matrix as assembler


def test_index_sides_rejects_duplicate_side(tmp_path: Path) -> None:
    path = tmp_path / "pyqcu__v100__8x8x8x16__c64__l2__trace-off.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        assembler.index_sides(
            [tmp_path, tmp_path],
            "pyqcu",
        )


def test_parse_unit_id_and_stage_rows() -> None:
    suffix = "p100__16x32x32x48__c64__l3__trace-on.json"
    metadata = assembler.parse_unit_id(suffix)
    assert metadata == {
        "unit_id": suffix.removesuffix(".json"),
        "device": "p100",
        "lattice": "16x32x32x48",
        "precision": "c64",
        "levels": 3,
        "trace": "on",
    }
    rows = assembler.stage_rows(suffix, "pyqcu", {
        "mg_levels": [{
            "level": 0,
            "total_seconds": 2.0,
            "total_iterations": 5,
            "pre_smoother_seconds": 0.2,
            "pre_smoother_iterations": 5,
            "post_smoother_seconds": 0.3,
            "post_smoother_iterations": 5,
            "restriction_seconds": 0.1,
            "restriction_calls": 5,
            "prolongation_seconds": 0.1,
            "prolongation_calls": 5,
            "coarse_solver_seconds": 0.0,
            "coarse_solver_iterations": 0,
            "coarse_solver_calls": 0,
            "other_seconds": 1.3,
            "final_residual": 1.0e-7,
        }],
    })
    assert rows == [{
        **metadata,
        "side": "pyqcu",
        "level": 0,
        "total_seconds": 2.0,
        "total_iterations": 5.0,
        "pre_smoother_seconds": 0.2,
        "pre_smoother_iterations": 5.0,
        "post_smoother_seconds": 0.3,
        "post_smoother_iterations": 5.0,
        "restrict_seconds": 0.1,
        "restrict_calls": 5.0,
        "prolongate_seconds": 0.1,
        "prolongate_calls": 5.0,
        "coarse_solver_seconds": 0.0,
        "coarse_solver_iterations": 0.0,
        "coarse_solver_calls": 0.0,
        "other_seconds": 1.3,
        "final_residual": 1.0e-7,
    }]


def test_reference_rows_records_cold_warmups_and_steady() -> None:
    suffix = "v100__8x8x8x16__c128__l1__trace-off.json"
    rows = assembler.reference_rows(suffix, "quda", {
        "reference_solver": {
            "cold": {
                "seconds": 1.0,
                "iterations": 3,
                "true_residual_rel": 1.0e-8,
            },
            "warmups": [
                {"seconds": 0.5, "iterations": 3, "true_residual_rel": 1.0e-8},
                {"seconds": 0.4, "iterations": 3, "true_residual_rel": 1.0e-8},
            ],
            "samples": [
                {"seconds": 0.3, "iterations": 3, "true_residual_rel": 1.0e-8},
                {"seconds": 0.3, "iterations": 3, "true_residual_rel": 1.0e-8},
                {"seconds": 0.3, "iterations": 3, "true_residual_rel": 1.0e-8},
                {"seconds": 0.3, "iterations": 3, "true_residual_rel": 1.0e-8},
                {"seconds": 0.3, "iterations": 3, "true_residual_rel": 1.0e-8},
            ],
        },
    })
    assert [row["phase"] for row in rows] == [
        "cold", "warmup", "warmup", "steady", "steady", "steady",
        "steady", "steady"]
    assert len(rows) == 8


def test_write_csv_uses_lf_line_endings(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    assembler._write_csv(path, [{"unit": "p100", "seconds": 1.0}])
    assert b"\r\n" not in path.read_bytes()
