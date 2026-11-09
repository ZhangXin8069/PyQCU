"""Contracts for the packaged ``pyqcu`` CLI and self-test."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from pyqcu import _selftest
from pyqcu.__main__ import _selftest_exit_code

ROOT = Path(__file__).resolve().parents[3]


def test_torch_layer_passes():
    report = _selftest.run_all(include=("torch",))
    assert report["status"] == "PASS"
    assert report["layers"]["torch"]["status"] == "PASS"
    assert report["layers"]["torch"]["true_residual"] < 1e-3


def test_cli_selftest_json_is_machine_readable():
    process = subprocess.run(
        [sys.executable, "-m", "pyqcu", "--selftest", "--json",
         "--lattice", "4", "4", "4", "4"],
        cwd=ROOT, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    assert report["status"] == "PASS"
    assert report["layers"]["torch"]["status"] == "PASS"
    assert report["lattice"] == [4, 4, 4, 4]


def test_invalid_lattice_is_rejected():
    with pytest.raises(ValueError):
        _selftest.run_all(lattice_shape=(4, 4, 4, 5), include=("torch",))


def test_too_small_lattice_is_rejected():
    with pytest.raises(ValueError, match="at least 4"):
        _selftest.run_all(lattice_shape=(2, 2, 2, 2), include=("torch",))


def test_include_string_and_empty_selection():
    report = _selftest.run_all(include="torch")
    assert report["status"] == "PASS"
    assert set(report["layers"]) == {"torch"}
    with pytest.raises(ValueError):
        _selftest.run_all(include=())


def test_torch_layer_is_deterministic():
    first = _selftest.run_all(include=("torch",))["layers"]["torch"]
    second = _selftest.run_all(include=("torch",))["layers"]["torch"]
    assert first["iterations"] == second["iterations"]
    assert abs(first["true_residual"] - second["true_residual"]) < 1e-12


def test_cli_rejects_invalid_lattice():
    process = subprocess.run(
        [sys.executable, "-m", "pyqcu", "--selftest",
         "--lattice", "4", "4", "4", "5"],
        cwd=ROOT, capture_output=True, text=True)
    assert process.returncode == 2
    assert "positive even" in process.stderr


def test_selftest_exit_code_matches_status():
    assert _selftest_exit_code({"status": "PASS"}) == 0
    assert _selftest_exit_code({"status": "FAIL"}) == 2


def test_main_nonzero_rank_failure_returns_2(monkeypatch, capsys):
    from pyqcu import __main__ as cli

    monkeypatch.setattr(
        _selftest, "run_all",
        lambda *args, **kwargs: {
            "status": "FAIL", "mpi_rank": 1,
            "layers": {"torch": {"status": "FAIL", "error": "boom"}}})
    assert cli.main(["--selftest"]) == 2
    captured = capsys.readouterr()
    assert "MPI rank 1" in captured.err
    assert "torch: boom" in captured.err


def test_cli_json_rejects_too_small_lattice():
    process = subprocess.run(
        [sys.executable, "-m", "pyqcu", "--selftest", "--json",
         "--lattice", "2", "2", "2", "2"],
        cwd=ROOT, capture_output=True, text=True)
    assert process.returncode == 2
    report = json.loads(process.stdout)
    assert report["status"] == "FAIL"
    assert "at least 4" in report["error"]


def test_cli_json_rejects_invalid_lattice():
    process = subprocess.run(
        [sys.executable, "-m", "pyqcu", "--selftest", "--json",
         "--lattice", "4", "4", "4", "5"],
        cwd=ROOT, capture_output=True, text=True)
    assert process.returncode == 2
    report = json.loads(process.stdout)
    assert report["status"] == "FAIL"
    assert "positive even" in report["error"]
