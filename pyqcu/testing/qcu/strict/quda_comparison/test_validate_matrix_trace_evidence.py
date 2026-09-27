"""Tests for the matrix trace/reference evidence validator."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("validate_matrix_trace_evidence.py")
SPEC = importlib.util.spec_from_file_location(
    "validate_matrix_trace_evidence", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


def test_trace_evidence_and_reference_phases_fail_closed(tmp_path: Path) -> None:
    trace = tmp_path / "trace.tsv"
    trace.write_text("trace_version\t3\n", encoding="utf-8")
    document = {
        "derived_kind": "bicgstab-reference",
        "unit_levels": 1,
        "sides": {
            "pyqcu": {
                "status": "ok",
                "trace_paths": {"PYQCU_STRICT_TRACE_FILE": str(trace)},
                "reference_solver": {
                    "cold": {"seconds": 1.0},
                    "warmups": [{}, {}],
                    "samples": [{}, {}, {}, {}, {}],
                },
            },
        },
    }
    document_path = tmp_path / "unit__trace-on.json"
    assert validator.validate_document(document_path, document) == []

    document["sides"]["pyqcu"]["trace_paths"][
        "PYQCU_STRICT_TRACE_FILE"] = str(tmp_path / "missing.tsv")
    document["sides"]["pyqcu"]["reference_solver"]["cold"] = None
    errors = validator.validate_document(document_path, document)
    assert any("missing:" in error for error in errors)
    assert any("reference cold missing" in error for error in errors)
