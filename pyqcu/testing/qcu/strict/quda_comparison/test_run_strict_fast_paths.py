"""Validate that fast-gate pytest node paths still resolve to source tests."""

from __future__ import annotations

import ast
from pathlib import Path

from pyqcu.testing.qcu.strict.quda_comparison.run_strict_fast import COMMANDS


ROOT = Path(__file__).resolve().parents[5]


def test_fast_gate_pytest_nodes_resolve() -> None:
    checked = 0
    for command in COMMANDS:
        for argument in command.argv:
            if "::" not in argument or not argument.startswith("pyqcu/"):
                continue
            path_text, test_name = argument.split("::", 1)
            path = ROOT / path_text
            assert path.is_file(), f"missing fast-gate test file: {path_text}"
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            definitions = {
                node.name for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            assert test_name in definitions, (
                f"missing fast-gate test node: {argument}")
            checked += 1
    assert checked >= 20
