from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import types

import numpy as np
import torch

from pyqcu import tools
from pyqcu.testing.simple._common import (
    REPO, configure_quda_runtime, convergence_status, lime_metadata,
    qdp_gauge_to_qcu, read_qdp_gauge, resolve_path, write_lime_gauge)
from pyqcu.testing.simple.compare_bistabcg import (
    _relative_solution_difference)
from pyqcu.testing.simple import compare_bistabcg
from pyqcu.testing.simple.quda_bistabcg import make_dirac


def test_lime_gauge_is_split_into_canonical_parity_layout(tmp_path) -> None:
    lattice = (4, 4, 4, 4)
    eye = np.eye(3, dtype=np.complex128)
    gauge = np.broadcast_to(
        eye, (4, lattice[3], lattice[2], lattice[1], lattice[0], 3, 3)).copy()
    path = tmp_path / "tiny.lime"
    write_lime_gauge(
        path, gauge,
        {"csw": 1.160920226, "reference_mass": -0.277})

    metadata = lime_metadata(path)
    assert metadata["lattice_xyzt"] == lattice
    assert metadata["csw"] == 1.160920226
    assert metadata["reference_mass"] == -0.277

    canonical = qdp_gauge_to_qcu(read_qdp_gauge(path))
    parity = tools.oooxyzt2poooxyzt(torch.from_numpy(canonical).contiguous())
    assert tuple(parity.shape) == (2, 3, 3, 4, 4, 4, 4, 2)
    expected = np.zeros((2, 3, 3, 4, 4, 4, 4, 2), dtype=np.complex128)
    expected[:, :, :, :] = eye.reshape(1, 3, 3, 1, 1, 1, 1, 1)
    assert torch.allclose(
        parity, torch.from_numpy(expected))


def test_data_prefixed_output_path_stays_under_repository() -> None:
    assert resolve_path("data/new-output.json", REPO / "unused") == (
        REPO / "data/new-output.json").resolve()


def test_repository_quda_install_is_selected(monkeypatch) -> None:
    expected = REPO / "data/quda-sm70-install"
    if not (expected / "lib/libquda.so").is_file():
        import pytest
        pytest.skip("repository-local sm70 QUDA install is not present")
    monkeypatch.delenv("QUDA_INSTALL", raising=False)
    monkeypatch.delenv("QUDA_PATH", raising=False)
    selected = configure_quda_runtime(preload=False)
    assert selected == expected.resolve()
    assert os.environ["QUDA_INSTALL"] == str(expected.resolve())
    assert os.environ["QUDA_PATH"] == str(expected.resolve())


def test_solver_status_uses_requested_tolerance() -> None:
    converged, status, acceptance = convergence_status(9.0e-3, 1.0e-9)
    assert not converged
    assert status == "FAIL"
    assert acceptance >= 1.0e-9
    converged, status, _ = convergence_status(2.0e-7, 1.0e-6)
    assert converged
    assert status == "PASS"
    assert convergence_status(
        2.0e-7, 1.0e-6, comparison_valid=False)[1] == "WARN"


def test_relative_solution_difference_rejects_shape_mismatch() -> None:
    reference = np.ones((2, 2), dtype=np.complex64)
    assert _relative_solution_difference(reference, reference) == 0.0
    try:
        _relative_solution_difference(
            np.ones((3,), dtype=np.complex64), reference)
    except ValueError as error:
        assert "shape mismatch" in str(error)
    else:
        raise AssertionError("shape mismatch must be rejected")


def test_compare_does_not_accept_missing_or_stale_backend_reports(
        tmp_path, monkeypatch) -> None:
    stale = tmp_path / "stale.qcu.json"
    stale.write_text('{"backend": "qcu", "status": "PASS"}', encoding="utf-8")
    commands = []

    def fake_run(_script, command, _timeout):
        commands.append(command)
        return {
            "command": command, "returncode": 0,
            "stdout": '{"status":"SKIP"}', "stderr": ""}

    monkeypatch.setattr(compare_bistabcg, "_run_backend", fake_run)
    output = tmp_path / "pair.json"
    result = compare_bistabcg.main([
        "--gauge", str(tmp_path / "dummy.lime"),
        "--lat", "4", "4", "4", "4",
        "--output", str(output),
    ])
    assert result == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "FAIL"
    assert report["numerical_match"] is False
    assert any("report unavailable" in error for error in report["errors"])
    assert Path(report["reports"]["qcu"]) != stale
    source_arguments = []
    for command in commands:
        source_arguments.append(command[command.index("--source") + 1])
    assert len(set(source_arguments)) == 1


def test_quda_csw_zero_uses_wilson_dirac(monkeypatch) -> None:
    calls = []
    core = types.SimpleNamespace(
        getWilson=lambda *args, **kwargs: calls.append(("wilson", args, kwargs)),
        getClover=lambda *args, **kwargs: calls.append(("clover", args, kwargs)))
    module = types.ModuleType("pyquda_utils")
    module.core = core
    monkeypatch.setitem(sys.modules, "pyquda_utils", module)
    result = make_dirac(None, 0.05, 1.0e-6, 100, 0.0)
    assert result is None
    assert calls[0][0] == "wilson"
