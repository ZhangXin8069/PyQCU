"""CPU-only checks for the MultiGrid comparison-matrix orchestrator."""

from __future__ import annotations

from pathlib import Path

import pytest

import bench_mg_matrix as matrix


def test_matrix_covers_levels_precision_and_volume():
    ids = {case["id"] for case in matrix.CASES}
    assert {"small-c64-l2", "small-c64-l3", "small-c128-l3"} <= ids
    assert {"medium-c64-l3", "formal-c64-l3", "formal-c128-l3"} <= ids
    assert {tuple(case["lattice"]) for case in matrix.CASES} >= {
        (8, 8, 8, 16), (16, 16, 16, 16), (16, 32, 32, 48)}
    assert {case["precision"] for case in matrix.CASES} == {"c64", "c128"}
    assert max(case["levels"] for case in matrix.CASES) >= 3


def test_case_command_pins_geometry_and_shared_assets(tmp_path):
    case = matrix.CASES[1]
    output = tmp_path / "out.json"
    command = matrix._case_command(
        case, profile="smoke", repeats=2, timeout=60.0, output=output,
        extra=("--cache-expect", "any"))
    assert "--levels" in command
    assert command[command.index("--levels") + 1] == "3"
    assert command[command.index("--lattice") + 1:command.index("--lattice") + 5] == [
        "8", "8", "8", "16"]
    assert command[command.index("--precision") + 1] == "c64"
    assert str(output) in command


def test_missing_asset_is_reported_as_skip(tmp_path):
    case = {
        "id": "missing",
        "lattice": [4, 4, 4, 4],
        "levels": 2,
        "precision": "c64",
    }
    result = matrix._run_case(
        case, profile="smoke", repeats=1, timeout=0.1,
        output_dir=tmp_path, trace=False)
    assert result["status"] == "skipped"
    assert result["reason"]["code"] == "missing_assets"


def test_cli_list_contains_three_level_cases(capsys):
    assert matrix.main(["--list"]) == 0
    output = capsys.readouterr().out
    assert '"levels": 3' in output


def test_c128_cases_are_not_downgraded_to_pyqcu_only():
    c128 = [case for case in matrix.CASES if case["precision"] == "c128"]
    assert c128
    assert all("quda_mg_supported" not in case for case in c128)
