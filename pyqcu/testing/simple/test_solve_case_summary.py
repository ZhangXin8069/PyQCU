"""Contracts for production solve summaries."""

from __future__ import annotations

import json

from pyqcu.testing.simple._common import json_safe


def test_solve_summary_omits_inapplicable_none_fields() -> None:
    summary = json_safe({
        "backend": "kupl",
        "status": "PASS",
        "iterations": 19,
        "solver_seconds": None,
        "libqcu_variant": None,
        "quda_variant": None,
        "kupl_variant": "/repo/cpp/kupl/qcu/libqcu.so",
        "error": None,
    }, omit_none=True)

    assert summary == {
        "backend": "kupl",
        "status": "PASS",
        "iterations": 19,
        "kupl_variant": "/repo/cpp/kupl/qcu/libqcu.so",
    }
    assert "null" not in json.dumps(summary)


def test_solve_summary_retains_failure_error() -> None:
    summary = json_safe({
        "backend": "kupl",
        "status": "FAIL",
        "error": "solver did not converge",
    }, omit_none=True)

    assert summary["error"] == "solver did not converge"


def test_json_safe_recursively_omits_none_fields_and_items() -> None:
    payload = json_safe({
        "status": "PASS",
        "optional": None,
        "rank": {"memory_policy": None, "rank": 0},
        "samples": [1.0, None, 2.0],
    }, omit_none=True)

    assert payload == {
        "status": "PASS",
        "rank": {"rank": 0},
        "samples": [1.0, 2.0],
    }
