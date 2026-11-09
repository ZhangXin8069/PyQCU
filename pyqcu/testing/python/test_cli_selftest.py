"""Contracts for the packaged ``pyqcu`` CLI and self-test."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types
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


def _run_cli(args, rank):
    env = os.environ.copy()
    env["OMPI_COMM_WORLD_RANK"] = str(rank)
    return subprocess.run(
        [sys.executable, "-m", "pyqcu", *args],
        cwd=ROOT, capture_output=True, text=True, env=env)


def test_cli_rank0_prints_version_and_help():
    version = _run_cli(["--version"], 0)
    assert version.returncode == 0
    assert version.stdout.startswith("pyqcu ")
    help_output = _run_cli(["--help"], 0)
    assert help_output.returncode == 0
    assert "usage: pyqcu" in help_output.stdout


@pytest.mark.parametrize("flag", ["--version", "--help", "-h"])
def test_cli_nonzero_rank_info_is_silent(flag):
    process = _run_cli([flag], 1)
    assert process.returncode == 0
    assert process.stdout == ""
    assert process.stderr == ""


def test_collect_does_not_import_heavy_dependencies():
    code = (
        "import sys; from pyqcu.__main__ import collect; collect(); "
        "print(int('torch' in sys.modules), int('mpi4py' in sys.modules))"
    )
    process = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert process.stdout.strip() == "0 0"


def test_mpi_info_uses_launcher_env_without_importing_mpi4py():
    code = (
        "import sys; from pyqcu._selftest import mpi_info; "
        "print(mpi_info()); print(int('mpi4py' in sys.modules))"
    )
    env = os.environ.copy()
    env["OMPI_COMM_WORLD_RANK"] = "1"
    env["OMPI_COMM_WORLD_SIZE"] = "2"
    process = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT, capture_output=True, text=True, env=env)
    assert process.returncode == 0, process.stderr
    assert process.stdout.splitlines() == ["(2, 1)", "0"]


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


def test_cli_json_rejects_non_integer_lattice():
    process = subprocess.run(
        [sys.executable, "-m", "pyqcu", "--selftest", "--json",
         "--lattice", "4", "4", "4", "x"],
        cwd=ROOT, capture_output=True, text=True)
    assert process.returncode == 2
    report = json.loads(process.stdout)
    assert report["status"] == "FAIL"
    assert "four integers" in report["error"]


def test_cuda_status_requires_companion_library(tmp_path, monkeypatch):
    from pyqcu import __main__ as cli

    bridge = tmp_path / "cuda" / "qcu.cpython-test.so"
    bridge.parent.mkdir()
    bridge.write_text("", encoding="utf-8")
    monkeypatch.delenv("PYQCU_CUDA_LIB", raising=False)
    monkeypatch.setattr(
        cli.importlib.util, "find_spec",
        lambda name: types.SimpleNamespace(origin=str(bridge)))
    missing = cli._cuda_status()
    assert missing["available"] is False
    assert "libqcu.so" in missing["error"]

    library = bridge.parent / "libqcu.so"
    library.write_text("", encoding="utf-8")
    found = cli._cuda_status()
    assert found["available"] is True
    assert found["library"] == str(library)


def test_selftest_import_failure_is_json(monkeypatch, capsys):
    from pyqcu import __main__ as cli
    import pyqcu

    monkeypatch.delattr(pyqcu, "_selftest", raising=False)
    monkeypatch.setitem(sys.modules, "pyqcu._selftest", None)
    assert cli.main(["--selftest", "--json"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "FAIL"
    assert "dependencies unavailable" in report["error"]


def test_selftest_runtime_failure_is_json(monkeypatch, capsys):
    from pyqcu import __main__ as cli

    def fail(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(_selftest, "run_all", fail)
    assert cli.main(["--selftest", "--json"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "FAIL"
    assert "RuntimeError: boom" in report["error"]


def test_cli_rejects_non_integer_lattice():
    process = subprocess.run(
        [sys.executable, "-m", "pyqcu", "--selftest",
         "--lattice", "4", "4", "4", "x"],
        cwd=ROOT, capture_output=True, text=True)
    assert process.returncode == 2
    assert "four integers" in process.stderr
