"""Contracts for the packaged ``pyqcu`` CLI and self-test."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from pyqcu import _selftest

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


def test_cli_json_rejects_invalid_lattice():
    process = subprocess.run(
        [sys.executable, "-m", "pyqcu", "--selftest", "--json",
         "--lattice", "4", "4", "4", "5"],
        cwd=ROOT, capture_output=True, text=True)
    assert process.returncode == 2
    report = json.loads(process.stdout)
    assert report["status"] == "FAIL"
    assert "positive even" in report["error"]
