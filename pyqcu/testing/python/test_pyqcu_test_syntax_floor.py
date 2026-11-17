"""Regression: the ``pyqcu.test`` Python 3.10 syntax floor (``--check``).

``pyproject.toml`` declares ``requires-python >= 3.9`` and the release
validation installs the wheel on Python 3.10.9, while local development runs
3.12.  ``source[*option.source_site]`` (PEP 646, 3.11+) therefore shipped in
``pyqcu/test/kupl_qcu.py``/``cuda_qcu.py`` and only failed on the remote
3.10.9 install.  ``python -m pyqcu.test --check`` now owns a ``py310_syntax``
gate that re-parses every ``pyqcu/test/*.py`` with the 3.10 grammar
(``ast.parse(feature_version=(3, 10))``) plus a structural PEP 646 scan.

These tests drive that released check in a subprocess against a **temporary
directory**, so no 3.11-only syntax is ever committed under ``pyqcu/test``
(the gate would fail on it).  ``ast.parse`` is also patched to ignore
``feature_version`` in one case: the gate must then degrade to ``SKIP`` with a
reason instead of a fake ``PASS``, while a structural PEP 646 defect is still
reported as ``FAIL``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]

PROBE_TIMEOUT_SECONDS = 300

#: Runs the real check from the working tree against ``argv[1]``.
#: ``argv[2] == "ignore_feature_version"`` simulates an interpreter whose
#: ``ast.parse`` accepts but ignores the ``feature_version`` keyword, which is
#: the case the gate must not mistake for a proven PASS.
_PROBE = """
import ast, json, sys
from pathlib import Path

simulation = sys.argv[2]
if simulation == "ignore_feature_version":
    _real_parse = ast.parse

    def _parse_ignoring_feature_version(source, filename="<unknown>",
                                       mode="exec", **kwargs):
        kwargs.pop("feature_version", None)
        return _real_parse(source, filename=filename, mode=mode, **kwargs)

    ast.parse = _parse_ignoring_feature_version

from pyqcu.test.selfcheck import check_py310_syntax

check = check_py310_syntax(Path(sys.argv[1]))
print(json.dumps({"name": check.name, "status": check.status,
                  "detail": check.detail}))
"""

#: Legal on Python 3.9/3.10: the starred element lives in a *tuple display*,
#: which is why ``source[(0, spin, color, *site)]`` is fine and
#: ``source[0, spin, color, *site]`` is not.
COMPATIBLE_SOURCE = (
    "def build(source, site):\n"
    "    index = (0, 1, 2, *site)\n"
    "    source[index] = 1.0 + 0.0j\n"
    "    return source[1:2]\n"
)

#: The construct that broke the 3.10.9 run: a starred subscript item.
PEP646_SOURCE = (
    "def build(source, site):\n"
    "    source[0, 1, 2, *site] = 1.0 + 0.0j\n"
)

#: The other PEP 646 position Python 3.10 rejects: a starred ``*args``
#: annotation (``ast.parse`` accepts it here even under feature_version).
PEP646_ANNOTATION_SOURCE = (
    "def build(*args: *Ts):\n"
    "    return args\n"
)

#: Python 3.11-only grammar that ``feature_version`` does reject.
EXCEPT_STAR_SOURCE = (
    "def guard():\n"
    "    try:\n"
    "        return 1\n"
    "    except* ValueError:\n"
    "        return 0\n"
)


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    pythonpath = str(ROOT)
    if environment.get("PYTHONPATH"):
        pythonpath += os.pathsep + environment["PYTHONPATH"]
    environment["PYTHONPATH"] = pythonpath
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _run_check(directory: Path, simulation: str = "normal") -> dict[str, str]:
    """Run ``check_py310_syntax`` from the working tree over ``directory``."""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", _PROBE, str(directory), simulation],
        cwd=ROOT,
        env=_environment(),
        text=True,
        capture_output=True,
        check=False,
        timeout=PROBE_TIMEOUT_SECONDS,
    )
    assert completed.returncode == 0, (
        f"py310_syntax probe failed (rc={completed.returncode})\n"
        f"stdout:\n{completed.stdout[-2000:]}\n"
        f"stderr:\n{completed.stderr[-4000:]}"
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["name"] == "py310_syntax", payload
    return payload


@pytest.mark.parametrize("name, source, location", [
    ("starred.py", PEP646_SOURCE, "starred.py:2"),
    ("annotation.py", PEP646_ANNOTATION_SOURCE, "annotation.py:1"),
])
def test_pep646_syntax_is_reported_with_file_and_line(
        tmp_path: Path, name: str, source: str, location: str) -> None:
    """The 3.10.9 regression class fails locally, with file and line."""
    _write(tmp_path / "compatible.py", COMPATIBLE_SOURCE)
    _write(tmp_path / name, source)

    payload = _run_check(tmp_path)

    assert payload["status"] == "FAIL", payload
    assert location in payload["detail"], payload
    assert "PEP 646" in payload["detail"], payload
    assert "compatible.py" not in payload["detail"], payload


def test_feature_version_gated_syntax_is_reported(tmp_path: Path) -> None:
    """``except*`` (3.11-only grammar) is reported when the parser gates it."""
    _write(tmp_path / "guard.py", EXCEPT_STAR_SOURCE)

    payload = _run_check(tmp_path)

    if payload["status"] == "SKIP":
        # An interpreter that ignores feature_version cannot prove a PASS; the
        # reason must be recorded instead of a green check.
        assert payload["detail"].strip(), "SKIP must carry a reason"
        assert "feature_version" in payload["detail"], payload
        pytest.skip(f"interpreter does not gate feature_version: "
                    f"{payload['detail']}")
    assert payload["status"] == "FAIL", payload
    assert "guard.py:" in payload["detail"], payload
    assert "3.11" in payload["detail"], payload


def test_gate_degrades_to_skip_when_feature_version_is_ignored(
        tmp_path: Path) -> None:
    """No fake PASS: a parser without the flag yields SKIP plus a reason."""
    _write(tmp_path / "guard.py", EXCEPT_STAR_SOURCE)

    payload = _run_check(tmp_path, simulation="ignore_feature_version")

    assert payload["status"] == "SKIP", payload
    assert payload["detail"].strip(), "SKIP must carry a reason"
    assert "feature_version" in payload["detail"], payload


def test_structural_defect_fails_even_without_feature_version(
        tmp_path: Path) -> None:
    """The PEP 646 layer is parser-independent: the real bug still FAILs."""
    _write(tmp_path / "starred.py", PEP646_SOURCE)

    payload = _run_check(tmp_path, simulation="ignore_feature_version")

    assert payload["status"] == "FAIL", payload
    assert "starred.py:2" in payload["detail"], payload


def test_committed_pyqcu_test_tree_passes_the_floor() -> None:
    """The shipped tree is 3.10-clean and ``--check`` reports the gate."""
    completed = subprocess.run(
        [sys.executable, "-B", "-m", "pyqcu.test", "--check", "--json", "-"],
        cwd=ROOT,
        env=_environment(),
        text=True,
        capture_output=True,
        check=False,
        timeout=PROBE_TIMEOUT_SECONDS,
    )
    assert completed.returncode == 0, (
        f"pyqcu.test --check failed (rc={completed.returncode})\n"
        f"{completed.stdout[-4000:]}\n{completed.stderr[-4000:]}"
    )
    payload = json.loads(completed.stdout)
    checks = {item["name"]: item for item in payload["checks"]}
    check = checks.get("py310_syntax")
    assert check is not None, (
        f"py310_syntax missing from --check: {sorted(checks)}")
    assert check["status"] in ("PASS", "SKIP"), check
    if check["status"] == "SKIP":
        pytest.skip("the grammar layer is unavailable here: "
                    f"{check['detail']}")
