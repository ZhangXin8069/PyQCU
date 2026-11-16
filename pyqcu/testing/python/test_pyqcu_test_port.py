"""Informal pytest adapter for the formal ``pyqcu.test`` port.

The shipped entry point is ``python -m pyqcu.test``; this module only publishes
its report through pytest so the informal tree can run the same gate:

* ``PASS`` -> the test passes;
* ``SKIP`` -> ``pytest.skip(reason)`` with the port's own reason (never empty);
* ``FAIL``/``ERROR``/``TIMEOUT`` -> ``pytest.fail`` with the recorded details.

No operator, solver or backend probe is reimplemented here: the adapter runs the
formal CLI in a subprocess and validates its machine-readable report.  The
formal package keeps owning selection, thresholds, timing and exit codes.

Set ``PYQCU_TEST_PORT_ADAPTER=0`` to disable the adapter (for example in an
offline tree where the slow ``--quick`` solve is unwanted).
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]

# Tier-0 contract from the formal port: exactly these 19 tags at 8x8x8x16.
# The first 14 are the original matrix, the last 5 the gauge/Wuppertal/CG/clover
# additions (`pyqcu/test/units.py`).
EXPECTED_TAGS = (
    "torch/lattice",
    "torch/dslash/wilson",
    "torch/dslash/parity",
    "torch/dslash/clover",
    "torch/smear/stout",
    "torch/solve/bistabcg",
    "torch/mg/legacy",
    "torch/mg/strict",
    "kupl/abi/symbols",
    "kupl/dslash/wilson",
    "kupl/solve/bistabcg",
    "cuda/smoke/init_end",
    "cuda/dslash/wilson",
    "cuda/solve/bistabcg",
    "torch/gauge/random",
    "torch/smear/wuppertal",
    "torch/solve/cg",
    "kupl/dslash/clover",
    "kupl/solve/cg",
)

QUICK_LATTICE = (8, 8, 8, 16)

#: The CUDA cases of the tier-0 matrix, derived from the matrix itself so the
#: count cannot go stale: the smoke/dslash/solve trio (3 cases).
CUDA_TAGS = tuple(tag for tag in EXPECTED_TAGS if tag.startswith("cuda/"))

CLI_TIMEOUT_SECONDS = 1800


def _skip_whole_module(reason: str) -> None:
    pytest.skip(reason, allow_module_level=True)


if os.environ.get("PYQCU_TEST_PORT_ADAPTER", "1") == "0":
    _skip_whole_module("PYQCU_TEST_PORT_ADAPTER=0 disables the adapter")

def _formal_port_is_runnable() -> bool:
    """True only for a materialized port, not a bare ``pyqcu/test/`` directory.

    An empty or partly written ``pyqcu/test/`` is importable as a namespace
    package, so ``find_spec`` alone cannot decide: ``__main__`` must import too.
    """
    if importlib.util.find_spec("pyqcu.test") is None:
        return False
    try:
        import pyqcu.test  # noqa: F401
        import pyqcu.test.__main__  # noqa: F401
    except Exception:
        return False
    return True


if not _formal_port_is_runnable():
    _skip_whole_module(
        "formal pyqcu.test port is not runnable in this checkout "
        "(pyqcu/test/__init__.py or __main__.py missing or not importable)"
    )


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    pythonpath = str(ROOT)
    if os.environ.get("PYTHONPATH"):
        pythonpath += os.pathsep + os.environ["PYTHONPATH"]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = pythonpath
    environment.setdefault("PYTHONUNBUFFERED", "1")
    return subprocess.run(
        [sys.executable, "-B", "-m", "pyqcu.test", *arguments],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=CLI_TIMEOUT_SECONDS,
    )


def _decode_json_payload(text: str, what: str) -> object:
    """Return the first JSON document carrying a ``cases`` key.

    The human report and the JSON document may share stdout, so scan from the
    end and accept the last balanced document instead of assuming clean stdout.
    """
    decoder = json.JSONDecoder()
    stripped = text.strip()
    if stripped:
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and "cases" in payload:
            return payload
        if isinstance(payload, list) and all(
                isinstance(item, dict) and "tag" in item for item in payload):
            return payload
    attempts = 0
    for index in range(len(text) - 1, -1, -1):
        if text[index] != "{":
            continue
        attempts += 1
        if attempts > 512:
            break
        try:
            payload, _ = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "cases" in payload:
            return payload
    pytest.fail(f"{what}: no JSON report found on stdout\n{text[-4000:]}")


def _case_tags(payload: object) -> list[str]:
    if isinstance(payload, list):
        cases = payload
    else:
        cases = payload.get("cases") or []
    return [str(case.get("tag", "")) for case in cases]


def _cases(payload: dict) -> list[dict]:
    cases = payload.get("cases") or []
    return [case for case in cases if isinstance(case, dict)]


def _normalized_lattice(value: object) -> tuple[int, ...] | None:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            return tuple(int(item) for item in value)
        except (TypeError, ValueError):
            return None
    return None


@pytest.fixture(scope="module")
def quick_report() -> dict:
    completed = _run_cli("--quick", "--json", "-")
    if completed.returncode == 2:
        pytest.fail(
            "pyqcu.test --quick reported a usage/environment error (exit 2)\n"
            f"stdout:\n{completed.stdout[-4000:]}\n"
            f"stderr:\n{completed.stderr[-4000:]}"
        )
    payload = _decode_json_payload(completed.stdout, "pyqcu.test --quick --json -")
    if not isinstance(payload, dict):
        pytest.fail("pyqcu.test --quick --json - did not return a report object")
    payload.setdefault("_exit_code", completed.returncode)
    payload.setdefault("_stderr", completed.stderr)
    return payload


@pytest.fixture(scope="module")
def listed_tags() -> list[str]:
    completed = _run_cli("--list", "--json")
    assert completed.returncode == 0, (
        f"pyqcu.test --list --json failed: rc={completed.returncode}\n"
        f"{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}"
    )
    payload = _decode_json_payload(completed.stdout, "pyqcu.test --list --json")
    return _case_tags(payload)


def test_quick_report_lists_the_nineteen_tier0_tags(quick_report: dict) -> None:
    cases = _cases(quick_report)
    tags = [str(case.get("tag", "")) for case in cases]
    assert len(cases) == 19, f"expected 19 tier-0 cases, got {len(cases)}: {tags}"
    assert len(set(tags)) == len(tags), f"duplicate tags in report: {tags}"
    missing = sorted(set(EXPECTED_TAGS) - set(tags))
    extra = sorted(set(tags) - set(EXPECTED_TAGS))
    assert not missing and not extra, f"missing={missing} extra={extra}"


def test_quick_report_uses_the_correctness_lattice(quick_report: dict) -> None:
    deviations = []
    for case in _cases(quick_report):
        lattice = _normalized_lattice(case.get("lattice"))
        if lattice is not None and lattice != QUICK_LATTICE:
            deviations.append((case.get("tag"), lattice))
    assert not deviations, f"cases outside {QUICK_LATTICE}: {deviations}"


def test_quick_report_exit_code_matches_case_status(quick_report: dict) -> None:
    statuses = [str(case.get("status", "")).upper() for case in _cases(quick_report)]
    assert statuses, "empty report"
    blocking = {"FAIL", "ERROR", "TIMEOUT"}
    all_skipped = all(status == "SKIP" for status in statuses)
    expected = 1 if (blocking & set(statuses) or all_skipped) else 0
    assert quick_report["_exit_code"] == expected, (
        f"exit {quick_report['_exit_code']} does not match statuses {statuses}"
    )


@pytest.mark.parametrize("tag", EXPECTED_TAGS)
def test_quick_case_status(quick_report: dict, tag: str) -> None:
    cases = {str(case.get("tag", "")): case for case in _cases(quick_report)}
    case = cases.get(tag)
    assert case is not None, f"{tag}: case missing from the quick report"
    status = str(case.get("status", "")).upper()
    if status == "PASS":
        return
    if status == "SKIP":
        reason = str(case.get("reason") or "").strip()
        assert reason, f"{tag}: SKIP without a reason"
        pytest.skip(f"{tag}: {reason}")
    pytest.fail(
        f"{tag}: {status or '<no status>'} "
        f"failed={case.get('failed')} metrics={case.get('metrics')}"
    )


def test_cuda_cases_carry_a_reason_when_skipped(quick_report: dict) -> None:
    """A missing CUDA bridge is a SKIP with a reason, never a silent pass."""
    cuda_cases = [case for case in _cases(quick_report)
                  if str(case.get("tag", "")).startswith("cuda/")]
    tags = [str(case.get("tag", "")) for case in cuda_cases]
    assert len(cuda_cases) == len(CUDA_TAGS), (
        f"expected {len(CUDA_TAGS)} CUDA cases {list(CUDA_TAGS)}, "
        f"got {len(cuda_cases)}: {tags}"
    )
    assert sorted(tags) == sorted(CUDA_TAGS), (
        f"CUDA tags {sorted(tags)} != the tier-0 matrix {sorted(CUDA_TAGS)}"
    )
    for case in cuda_cases:
        status = str(case.get("status", "")).upper()
        if status == "PASS":
            continue
        if status == "SKIP":
            reason = str(case.get("reason") or "").strip()
            assert reason, f"{case.get('tag')}: CUDA SKIP without a reason"
            continue
        pytest.fail(f"{case.get('tag')}: {status} instead of PASS/SKIP")


def test_list_json_agrees_with_the_quick_report(
        quick_report: dict, listed_tags: list[str]) -> None:
    assert len(listed_tags) == 19, f"--list reported {len(listed_tags)} tags: {listed_tags}"
    quick_tags = [str(case.get("tag", "")) for case in _cases(quick_report)]
    assert sorted(listed_tags) == sorted(quick_tags), (
        f"--list tags {sorted(listed_tags)} != quick tags {sorted(quick_tags)}"
    )


# ---------------------------------------------------------------------------
# expanded CLI: the requested parameter form must parse, normalise and propagate
# ---------------------------------------------------------------------------
def _list_json(*arguments: str) -> tuple[int, dict, str]:
    """Run ``--list --json`` and return ``(rc, payload, stderr)``."""
    completed = _run_cli(*arguments, "--list", "--json")
    if completed.returncode != 0:
        return completed.returncode, {}, completed.stderr
    payload = json.loads(completed.stdout)
    return completed.returncode, payload, completed.stderr


#: The requested form: every expanded geometry/physics/selection flag at once.
REQUESTED_FORM = (
    "--lattice-size", "[8,8,8,16]",
    "--gauge-sigma", "0.1",
    "--mass", "-3.4",
    "--lattice-grid", "[1,2,2,1]",
    "--solver", "multigrid",
    "--backend", "torch",
    "--device", "cpu",
    "--dslash", "clover",
    "--mg-grid", "[2,2,2,2]",
    "--atol", "1e-6",
    "--dtype", "complex128",
)


def test_requested_form_listing_propagates_every_option() -> None:
    """The expanded form parses and every value reaches the listing payload."""
    rc, payload, stderr = _list_json(*REQUESTED_FORM)
    assert rc == 0, f"--list rejected the requested form: {stderr[-2000:]}"
    expected = {
        "lattice": [8, 8, 8, 16],
        "dtype": "c128",
        "lattice_grid": [1, 2, 2, 1],
        "mg_grid": [2, 2, 2, 2],
        "mass": -3.4,
        "gauge_sigma": 0.1,
        "solver": "multigrid",
        "dslash": "clover",
        "device": "cpu",
        "runner_device": "cpu",
        "backends": ["torch"],
        "atol": 1e-6,
    }
    mismatched = {key: (payload.get(key), value) for key, value in expected.items()
                  if payload.get(key) != value}
    assert not mismatched, f"listing payload lost options: {mismatched}"
    assert payload["count"] >= 1
    for case in _cases(payload):
        assert tuple(case["lattice"]) == (8, 8, 8, 16), case["tag"]
        assert case["dtype"] == "c128", case["tag"]
    tags = set(_case_tags(payload))
    assert "torch/mg/legacy" in tags and "torch/dslash/clover" in tags
    assert not any(tag.startswith("torch/solve/") for tag in tags)


@pytest.mark.parametrize("alias, backends", [
    ("cpu", ["torch", "kupl"]),
    ("dcu", ["cuda"]),
    ("npu", ["torch"]),
    ("torch,cuda,kupl", ["torch", "cuda", "kupl"]),
])
def test_backend_aliases_expand_to_canonical_names(alias: str, backends: list[str]) -> None:
    rc, payload, stderr = _list_json("--backend", alias)
    assert rc == 0, f"--backend {alias}: {stderr[-1000:]}"
    assert payload["backends"] == backends, payload["backends"]
    assert {case["backend"] for case in _cases(payload)} <= set(backends)


@pytest.mark.parametrize("choice, runner_device", [
    ("auto", None),
    ("cpu", "cpu"),
    ("gpu", "cuda:0"),
    ("dcu", "dcu:0"),
    ("npu", "npu:0"),
])
def test_device_choices_map_to_runner_devices(choice: str, runner_device: str | None) -> None:
    rc, payload, stderr = _list_json("--device", choice)
    assert rc == 0, f"--device {choice}: {stderr[-1000:]}"
    assert payload["device"] == choice
    assert payload["runner_device"] == runner_device


def test_dtype_accepts_numpy_spellings_but_reports_canonical_names() -> None:
    rc, payload, stderr = _list_json("--dtype", "complex128")
    assert rc == 0, stderr[-1000:]
    assert payload["dtype"] == "c128"
    assert {case["dtype"] for case in _cases(payload)} == {"c128"}
    rc, payload, stderr = _list_json("--dtype", "complex64")
    assert rc == 0 and payload["dtype"] == "c64", stderr[-1000:]
    failed = _run_cli("--dtype", "int32", "--list")
    assert failed.returncode == 2, "an unknown dtype must be a usage error"


def test_solver_and_dslash_filter_only_their_own_kinds() -> None:
    rc, payload, stderr = _list_json("--solver", "cg", "--dslash", "clover")
    assert rc == 0, stderr[-1000:]
    for case in _cases(payload):
        tag, kind = case["tag"], case["kind"]
        if kind in ("solve", "mg"):
            assert tag.endswith("/solve/cg") or tag.endswith("/mg/legacy") \
                or tag.endswith("/mg/strict"), f"solver=cg kept {tag}"
        if kind == "dslash":
            assert tag.endswith("/clover"), f"dslash=clover kept {tag}"
    tags = set(_case_tags(payload))
    # Non-solver, non-dslash cases are never filtered out.
    assert {"torch/lattice", "kupl/abi/symbols", "torch/gauge/random"} <= tags


def test_only_accepts_the_documented_comma_space_and_repeated_forms() -> None:
    """``--only a,b`` must select the same cases as ``--only a b``.

    The README documents a comma-separated ``--only`` list; the CLI used to
    hand the whole ``a,b`` word to ``Test.select`` as one literal tag and exit
    2.  All three spellings are part of the shipped contract, so all three run
    here through the real subprocess CLI.
    """
    forms = {
        "comma": ("--only", "torch/lattice,torch/gauge/random"),
        "space": ("--only", "torch/lattice", "torch/gauge/random"),
        "repeated": ("--only", "torch/lattice", "--only",
                     "torch/gauge/random"),
    }
    expected = ["torch/gauge/random", "torch/lattice"]
    for name, argv in forms.items():
        rc, payload, stderr = _list_json(*argv)
        assert rc == 0, f"--only {name} form exited {rc}: {stderr[-1000:]}"
        listed = sorted(_case_tags(payload))
        assert listed == expected, (
            f"--only {name} form listed {listed}, expected {expected}")
        assert payload.get("tags_filter") == ["torch/lattice",
                                              "torch/gauge/random"], (
            f"--only {name} form did not report the normalised filter: "
            f"{payload.get('tags_filter')!r}")


def _single_case(*arguments: str) -> dict:
    """Run one ``--only`` case and return its report entry."""
    completed = _run_cli(*arguments, "--json", "-")
    assert completed.returncode == 0, (
        f"{' '.join(arguments)}: rc={completed.returncode}\n"
        f"{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}"
    )
    payload = _decode_json_payload(completed.stdout, " ".join(arguments))
    cases = _cases(payload)
    assert len(cases) == 1, _case_tags(payload)
    return cases[0]


def test_native_lattice_grid_mismatch_skips_with_mpi_grid() -> None:
    """A grid mismatch is never a fake PASS; only grid-bound cases skip on it.

    ``kupl/abi/symbols`` is a *symbol* probe (``requires=("kupl_library",)``):
    it binds the C ABI and never builds a process grid, so a non-1x1x1x1
    ``--lattice-grid`` is not a reason to skip it - it keeps its own verdict
    (``PASS``, or ``SKIP`` on its ``kupl_library`` requirement) and must never
    consume the ``mpi_grid`` token.  The grid gate belongs to the numerical
    native runners, the cases that would compute a wrong slab: with the same
    mismatched grid ``kupl/dslash/wilson`` must skip, and when the KUPL library
    is importable (proved by the ABI case passing) the token must be
    ``mpi_grid``.
    """
    grid = ("--lattice-grid", "[1,2,2,1]", "--repeat", "1", "--warmup", "1",
            "--allow-all-skipped")

    abi = _single_case("--backend", "kupl", "--only", "kupl/abi/symbols", *grid)
    abi_status = str(abi.get("status", "")).upper()
    assert abi_status in ("PASS", "SKIP"), abi
    assert abi.get("skip_token") != "mpi_grid", (
        f"kupl/abi/symbols is grid-independent but skipped with mpi_grid: {abi}"
    )
    if abi_status == "SKIP":
        assert abi.get("skip_token") == "kupl_library", abi
        assert str(abi.get("reason") or "").strip(), "SKIP must carry a reason"

    numeric = _single_case("--backend", "kupl", "--only", "kupl/dslash/wilson",
                           *grid)
    numeric_status = str(numeric.get("status", "")).upper()
    assert numeric_status != "PASS", (
        f"a grid-bound native case PASSed on a mismatched grid: {numeric}"
    )
    assert numeric_status == "SKIP", numeric
    reason = str(numeric.get("reason") or "").strip()
    assert reason, "SKIP must carry a reason"
    if abi_status == "PASS":
        # The library loads here, so the grid gate is the only thing that can
        # have stopped the runner: the token has to be the documented one.
        assert numeric.get("skip_token") == "mpi_grid", numeric
        assert "lattice_grid" in reason, reason
    else:
        assert numeric.get("skip_token") in ("kupl_library", "mpi_single_rank",
                                             "mpi_grid"), numeric
