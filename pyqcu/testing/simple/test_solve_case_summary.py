"""Contracts for production solve summaries."""

from __future__ import annotations

import json

from pyqcu.testing.simple.solve_case import _drop_none_fields


def test_solve_summary_omits_inapplicable_none_fields() -> None:
    summary = _drop_none_fields({
        "backend": "kupl",
        "status": "PASS",
        "iterations": 19,
        "solver_seconds": None,
        "libqcu_variant": None,
        "quda_variant": None,
        "kupl_variant": "/repo/cpp/kupl/qcu/libqcu.so",
        "error": None,
    })

    assert summary == {
        "backend": "kupl",
        "status": "PASS",
        "iterations": 19,
        "kupl_variant": "/repo/cpp/kupl/qcu/libqcu.so",
    }
    assert "null" not in json.dumps(summary)


def test_solve_summary_retains_failure_error() -> None:
    summary = _drop_none_fields({
        "backend": "kupl",
        "status": "FAIL",
        "error": "solver did not converge",
    })

    assert summary["error"] == "solver did not converge"
