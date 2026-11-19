"""Focused tests for the multi-rank ledger/control plane of ``pyqcu.test``.

A 32-process ``mpirun``/``drun`` launch runs ``python -m pyqcu.test --mode
solve`` on 32 processes of the *same* row.  The port must behave as **one
logical row**, not as 32 independent ones:

* only rank 0 writes ``rows/<run_id>.json``, ``reports/<run_id>.json``, the
  append-only CSV, the solution copies and the report payload;
* every rank reports a compact record (``status``/``skip_token``/``reason``/
  ``iterations``/``relative_residual``/``converged``) that the run allgathers,
  and the **worst** rank wins - a single rank's ``ERROR`` must not be hidden by
  31 ``PASS`` rows;
* the ``SKIP(mpi_grid)`` decision is taken before any native call and is
  unanimous, and ``--fail-fast`` stops every rank at the same row (a rank that
  drops out of the collective while the others wait is the deadlock the
  2026-11 audit records as P0-5).

No MPI launch is needed here and none is started: the protocol is driven with
fake rank records / a fake communicator, and a non-root rank is simulated with
the launcher environment variables (``OMPI_COMM_WORLD_RANK``/``_SIZE``).  The
tests therefore run on a machine without mpi4py, KUPL or a scheduler.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
LATTICE = (8, 8, 8, 16)
CLI_TIMEOUT_SECONDS = 600

RANK_ENV = ("OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE")

#: Every launcher variable :func:`pyqcu.test.common.mpi_info` reads.
RANK_VARIABLES = RANK_ENV + ("PMI_RANK", "PMI_SIZE", "PMIX_RANK", "PMIX_SIZE",
                             "CCS_TASK_INDEX", "CCS_TASK_REPLICA",
                             "SLURM_PROCID", "SLURM_NTASKS", "I_MPI_RANK",
                             "I_MPI_SIZE", "MV2_COMM_WORLD_RANK",
                             "MV2_COMM_WORLD_SIZE")

try:
    from pyqcu.test import common, driver
except Exception:  # noqa: BLE001 - a partial tree is "not runnable"
    pytest.skip("formal pyqcu.test port is not runnable in this checkout",
                allow_module_level=True)


# ---------------------------------------------------------------------------
# fixtures and helpers
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _clean_mpi_state(monkeypatch: pytest.MonkeyPatch):
    """No launcher variables, cached communicator, or stale ABI rank state."""
    for name in RANK_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(common, "_RANK_COMMUNICATORS", {}, raising=False)
    # The native tests may import ``pyqcu.cuda.define``; its module-level
    # rank/size then wins over the fake OMPI_* environment.  Remove that stale
    # binding so the next test's launcher environment is authoritative.
    monkeypatch.delitem(sys.modules, "pyqcu.cuda.define", raising=False)
    yield


def _world(rank: int, size: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make this process report itself as ``rank``/``size`` of an MPI world."""
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", str(rank))
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", str(size))


def _record(rank: int, status: str = "PASS", *, reason: str | None = None,
            skip_token: str | None = None, iterations: int | None = None,
            residual: float | None = None,
            converged: bool | None = None) -> dict:
    """One hand-built rank record (the wire format of the allgather)."""
    return {
        "rank": rank,
        "status": status,
        "skip_token": skip_token,
        "reason": reason,
        "iterations": iterations,
        "relative_residual": residual,
        "converged": converged,
    }


def _world_records(size: int, *, status: str = "PASS", **fields) -> list[dict]:
    """``size`` identical records (the boring "all ranks agree" case)."""
    return [_record(rank, status, **fields) for rank in range(size)]


def _fake_gather(groups, calls: list[dict] | None = None):
    """A ``common.allgather_rank_records`` replacement (never touches mpi4py).

    ``groups`` is either one record list (the same answer on every call) or a
    list of record lists (one answer per successive collective).  When ``calls``
    is given, every local record the caller contributed is appended to it, so a
    test can assert *how many* collectives a rank entered - the deadlock proxy:
    every rank must enter the same collectives, in the same order.
    """
    if groups and isinstance(groups[0], list):
        plan = [[dict(item) for item in group] for group in groups]
    else:
        plan = None
        single = [dict(item) for item in groups]
    seen = [0]

    def allgather(record, *, size=None, rank=None, comm=None):  # noqa: ANN001
        if calls is not None:
            calls.append(dict(record))
        if plan is None:
            answer = single
        else:
            answer = plan[min(seen[0], len(plan) - 1)]
        seen[0] += 1
        return [dict(item) for item in answer], "allgather", None

    return allgather


class _FakeComm:
    """Minimal ``MPI.COMM_WORLD`` stand-in for :func:`common.gather_rank_outcome`."""

    def __init__(self, records: list[dict]) -> None:
        self.records = [dict(item) for item in records]

    def allgather(self, record):  # noqa: ANN001 - structural protocol
        return [dict(item) for item in self.records]


def _case(tag: str = "torch/lattice"):
    """A case whose runner is never imported by these tests."""
    backend, _, kind = tag.partition("/")
    return common.Case(
        tag=tag, backend=backend, kind=kind or "lattice",
        runner="torch_qcu:run_lattice", lattice=LATTICE, dtype="c64",
        expect={"relative_error": {"le": "compare"}})


def _test_and_options(tmp_path: Path, *, tag: str = "torch/lattice",
                      ranks: int = 1, grid: list[int] | None = None):
    """A one-row full-mode plan, exactly as the CLI would build it."""
    test = common.Test(cases=(_case(tag),), lattice=LATTICE, dtype="c64")
    options = common.FullModeOptions(
        mode="solve", paths=common.ModePaths.resolve(root=tmp_path / "run"),
        ranks=ranks, lattice_grid=tuple(grid) if grid else None)
    rows = common.build_rows(test, options)
    return test, options, rows


def _make_result(case, status: str = "PASS", *, reason: str | None = None,
                 residual: float = 1e-12, iterations: int = 7):
    """A :class:`common.Result` with the gates' numbers filled in."""
    return common.Result(
        tag=case.tag, backend=case.backend, kind=case.kind, status=status,
        seconds=0.01,
        metrics={"relative_error": 0.0, "relative_residual": residual,
                 "iterations": iterations, "converged": True},
        reason=reason, lattice=case.lattice, dtype=case.dtype, seed=42,
        mpi_size=1)


def _run_cli(*arguments: str, env: dict | None = None
             ) -> subprocess.CompletedProcess[str]:
    """Run ``python -m pyqcu.test`` with a clean launcher environment."""
    environment = os.environ.copy()
    for name in RANK_VARIABLES:
        environment.pop(name, None)
    environment["PYTHONPATH"] = str(ROOT) + (
        os.pathsep + environment["PYTHONPATH"]
        if environment.get("PYTHONPATH") else "")
    if env:
        environment.update(env)
    return subprocess.run(
        [sys.executable, "-B", "-m", "pyqcu.test", *arguments],
        cwd=ROOT, env=environment, text=True, capture_output=True,
        check=False, timeout=CLI_TIMEOUT_SECONDS)


# ---------------------------------------------------------------------------
# 1. the outcome aggregator: the worst rank wins
# ---------------------------------------------------------------------------
def test_status_precedence_is_error_fail_timeout_skip_pass() -> None:
    assert common.RANK_STATUS_PRECEDENCE == ("ERROR", "FAIL", "TIMEOUT", "SKIP",
                                             "PASS")
    keys = [common.rank_status_key(status)
            for status in common.RANK_STATUS_PRECEDENCE]
    assert keys == sorted(keys), "precedence must be worst-first"
    # An unparsable status sorts before ERROR and becomes ERROR when aggregated.
    assert common.rank_status_key("nonsense") < common.rank_status_key("ERROR")


def test_the_rank_record_is_the_documented_compact_wire_format() -> None:
    """One record per rank and row: 7 values, JSON-safe, no observation dump."""
    record = common.rank_record(
        3, {"status": "PASS", "iterations": 7, "relative_residual": 1e-12,
            "converged": True, "gauge_sha256": "a" * 64, "gauge_bytes": 99})
    assert tuple(record) == common.RANK_RECORD_FIELDS, record
    assert len(common.RANK_RECORD_FIELDS) == 7
    assert record["rank"] == 3 and record["status"] == "PASS"
    assert record["iterations"] == 7 and record["converged"] is True
    # The full observation mapping stays on the rank that produced it: shipping
    # input identity/hashes would make the collective the slowest part of a
    # 32-rank row.
    assert "gauge_sha256" not in record and "gauge_bytes" not in record
    assert json.loads(json.dumps(record)) == record

    # A record the aggregator cannot parse must fail the row, never pass it.
    broken = common.rank_record(1, {"status": "definitely-not-a-status"})
    assert broken["status"] == "ERROR", broken
    assert "definitely-not-a-status" in broken["reason"]
    # An explicit decision (the pre-flight) wins over the observations.
    forced = common.rank_record(2, {"status": "PASS"}, status="SKIP",
                                skip_token="mpi_grid", reason="cannot run here")
    assert forced["status"] == "SKIP"
    assert forced["skip_token"] == "mpi_grid"


def test_one_rank_error_is_not_hidden_by_31_passes() -> None:
    records = _world_records(32)
    records[17] = _record(17, "ERROR", reason="kupl solver returned -3")
    outcome = common.rank_outcome(records, size=32)
    assert outcome["status"] == "ERROR", outcome
    assert outcome["worst_rank"] == 17
    assert "rank 17/32" in outcome["reason"], outcome["reason"]
    assert "kupl solver returned -3" in outcome["reason"]
    assert outcome["disagreement"] is True


def test_skip_outranks_pass_and_keeps_its_token() -> None:
    outcome = common.rank_outcome(
        [_record(0), _record(1, "SKIP", reason="lattice_grid 2x2x2x8 != size",
                             skip_token="mpi_grid")], size=2)
    assert outcome["status"] == "SKIP"
    assert outcome["skip_token"] == "mpi_grid"
    assert outcome["reason"].startswith("rank 1/2: ")
    assert outcome["disagreement"] is True

    # ... but a real error still wins over another rank's skip.
    mixed = common.rank_outcome(
        [_record(0, "SKIP", reason="no kupl library", skip_token="kupl_library"),
         _record(1, "ERROR", reason="input rejected")], size=2)
    assert mixed["status"] == "ERROR", mixed
    assert mixed["skip_token"] is None


def test_a_skip_without_a_reason_cannot_pass_as_a_clean_skip() -> None:
    outcome = common.rank_outcome([_record(0, "SKIP")], size=4)
    assert outcome["status"] == "ERROR", outcome
    assert outcome["reason"], "a reasonless skip must say why it is an error"


def test_the_worst_numbers_are_the_rows_evidence() -> None:
    outcome = common.rank_outcome([
        _record(0, iterations=11, residual=1e-9, converged=True),
        _record(1, iterations=19, residual=3e-7, converged=False),
        _record(2, iterations=11, residual=1e-9, converged=True),
    ], size=3)
    assert outcome["status"] == "PASS"       # statuses agree ...
    assert outcome["iterations"] == 19       # ... the evidence does not
    assert outcome["relative_residual"] == 3e-7
    assert outcome["converged"] is False, "one unconverged rank fails the row"
    assert outcome["disagreement"] is True


def test_a_rounding_spread_is_not_a_disagreement() -> None:
    outcome = common.rank_outcome([
        _record(0, iterations=11, residual=1.0e-9, converged=True),
        _record(1, iterations=11, residual=1.0e-9 * (1 + 1e-13),
                converged=True),
    ], size=2)
    assert outcome["status"] == "PASS"
    assert outcome["disagreement"] is False, outcome


def test_no_record_is_an_error_never_a_pass() -> None:
    outcome = common.rank_outcome([], size=32)
    assert outcome["status"] == "ERROR"
    assert "no rank reported" in outcome["reason"]


# ---------------------------------------------------------------------------
# 2. the gathered evidence: disagreement / mpi_rank / mpi_size
# ---------------------------------------------------------------------------
def test_gather_records_disagreement_rank_and_size() -> None:
    records = [_record(rank, residual=1e-12, iterations=9, converged=True)
               for rank in range(4)]
    records[3] = _record(3, "ERROR", reason="native solve failed")
    evidence = common.gather_rank_outcome(
        {"relative_residual": 1e-12, "iterations": 9},
        size=4, rank=0, comm=_FakeComm(records))
    assert evidence["mpi_rank"] == 0
    assert evidence["mpi_size"] == 4
    assert evidence["rank_disagreement"] is True
    assert evidence["rank_gather"] == "allgather"
    assert evidence["rank_status"] == "ERROR"
    assert "rank 3/4" in evidence["rank_reason"]
    assert len(evidence["rank_outcomes"]) == 4
    assert evidence["rank_phase"] == "row"
    json.dumps(evidence)  # the evidence must be JSON-safe (it lands in rows/)


def test_gather_without_a_communicator_says_so_and_never_blocks(
        monkeypatch: pytest.MonkeyPatch) -> None:
    _world(1, 4, monkeypatch)
    monkeypatch.setattr(
        common, "_rank_communicator",
        lambda size, rank: (None, "mpi4py unavailable: ImportError: no MPI"))
    evidence = common.gather_rank_outcome({"relative_residual": 1e-12},
                                          status="PASS")
    assert evidence["rank_gather"] == "unavailable"
    assert "mpi4py" in evidence["rank_gather_reason"]
    assert evidence["rank_status"] == "PASS"   # the local verdict, uncompared
    assert evidence["mpi_rank"] == 1 and evidence["mpi_size"] == 4


def test_a_communicator_of_the_wrong_size_is_refused(
        monkeypatch: pytest.MonkeyPatch) -> None:
    _world(0, 4, monkeypatch)
    evidence = common.gather_rank_outcome({}, size=4, rank=0,
                                          comm=_FakeComm([_record(0),
                                                          _record(1)]))
    assert evidence["rank_gather"] == "unavailable"
    assert "2 record(s) for 4 rank(s)" in evidence["rank_gather_reason"]


def test_a_single_process_needs_no_collective() -> None:
    evidence = common.gather_rank_outcome({"iterations": 3}, phase="verdict")
    assert evidence["rank_gather"] == "single_rank"
    assert evidence["rank_disagreement"] is False
    assert evidence["mpi_size"] == 1 and evidence["mpi_rank"] == 0
    assert evidence["iterations"] == 3
    assert len(evidence["rank_outcomes"]) == 1


# ---------------------------------------------------------------------------
# 3. rank 0 alone writes the ledger
# ---------------------------------------------------------------------------
def test_write_row_does_no_io_on_a_replica_rank(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, options, rows = _test_and_options(tmp_path)
    row = dict(rows[0])
    row.update(status="PASS", metrics={"relative_residual": 1e-12})

    _world(3, 4, monkeypatch)          # a non-root rank of a 4-rank world
    path = common.write_row(options.paths, dict(row))
    assert path == options.paths.rows / f"{row['run_id']}.json"
    assert not path.exists(), "a replica rank wrote rows/<run_id>.json"
    assert not (options.paths.reports / f"{row['run_id']}.json").exists()
    assert not options.paths.csv.exists(), "a replica rank appended to the CSV"

    _world(0, 4, monkeypatch)          # rank 0 owns the ledger
    common.write_row(options.paths, dict(row))
    assert path.is_file()
    assert (options.paths.reports / f"{row['run_id']}.json").is_file()
    lines = options.paths.csv.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2, lines      # header + exactly one row
    assert common.ledger_owner() is True
    assert common.ledger_owner(rank=3) is False


def test_mode_solve_writes_one_row_and_keeps_the_worst_rank(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    test, options, rows = _test_and_options(tmp_path)
    records = _world_records(4, residual=1e-12, iterations=7, converged=True)
    records[2] = _record(2, "ERROR", reason="native solve failed on this rank")
    calls: list[dict] = []
    monkeypatch.setattr(common, "allgather_rank_records",
                        _fake_gather(records, calls))
    _world(0, 4, monkeypatch)

    payload, code = common.run_mode(
        "solve", test, options, rows=[dict(row) for row in rows],
        execute=lambda selected: _make_result(selected))

    assert calls, "rank 0 never entered the per-row allgather"
    assert payload["rows"]["count"] == 1
    entry = payload["rows"]["items"][0]
    assert entry["status"] == "ERROR", entry
    assert "rank 2/4" in entry["reason"]
    assert entry["metrics"]["rank_disagreement"] is True
    assert entry["metrics"]["mpi_size"] == 4
    assert len(entry["metrics"]["rank_outcomes"]) == 4
    assert code == 1
    # The disagreement is in the written row, not only in the payload.
    stored = json.loads(
        (options.paths.rows / f"{rows[0]['run_id']}.json").read_text("utf-8"))
    assert stored["status"] == "ERROR"
    assert stored["metrics"]["rank_disagreement"] is True


def test_a_replica_rank_computes_but_writes_nothing(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    test, options, rows = _test_and_options(tmp_path)
    executed: list[str] = []

    def execute(selected):
        executed.append(selected.tag)
        return _make_result(selected)

    monkeypatch.setattr(common, "allgather_rank_records",
                        _fake_gather(_world_records(4)))
    _world(3, 4, monkeypatch)
    payload, code = common.run_mode(
        "solve", test, options, rows=[dict(row) for row in rows],
        execute=execute)
    assert executed == ["torch/lattice"], "a replica rank must still compute"
    assert payload["rows"]["count"] == 1
    assert payload["ledger_writer"] is False
    assert not options.paths.rows.exists(), "a replica rank created rows/"
    assert not options.paths.csv.exists()
    assert code == 0


def test_fail_fast_stops_every_rank_at_the_same_row(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A replica and the owner enter the same collectives and stop alike."""
    test, _, rows = _test_and_options(tmp_path)
    plan = [dict(rows[0]), dict(rows[0])]
    plan[0]["run_id"] = "first"
    plan[1]["run_id"] = "second"
    options = common.FullModeOptions(
        mode="solve", paths=common.ModePaths.resolve(root=tmp_path / "ff"),
        ranks=4, fail_fast=True)
    records = _world_records(4)
    records[1] = _record(1, "ERROR", reason="boom on rank 1")

    outcomes = {}
    for rank in (0, 3):
        calls: list[dict] = []
        executed: list[str] = []
        monkeypatch.setattr(common, "allgather_rank_records",
                            _fake_gather(records, calls))
        _world(rank, 4, monkeypatch)

        def execute(selected, seen=executed):
            seen.append(selected.tag)
            return _make_result(selected)

        payload, code = common.run_mode(
            "solve", test, options, rows=[dict(row) for row in plan],
            execute=execute)
        outcomes[rank] = {
            "gathers": len(calls),
            "executed": len(executed),
            "stopped": payload["fail_fast_stopped"],
            "documented": payload["executed"],
            "code": code,
        }
    assert outcomes[0] == outcomes[3], outcomes
    # One collective per executed row on *every* rank: the rank that stops
    # first cannot leave the others inside a collective.
    assert outcomes[0]["gathers"] == outcomes[0]["executed"] == 1
    assert outcomes[0]["stopped"] is True
    assert outcomes[0]["code"] == 1


def test_merge_rewrites_the_csv_only_on_the_owning_rank(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, options, rows = _test_and_options(tmp_path)
    row = dict(rows[0])
    row.update(status="PASS", metrics={"relative_residual": 1e-12})
    _world(0, 2, monkeypatch)
    common.write_row(options.paths, row)
    options.paths.csv.unlink()

    _world(1, 2, monkeypatch)
    document, code = common.merge_ledger(options.paths)
    assert document["rows"]["count"] == 1
    assert code == 0
    assert not options.paths.csv.exists(), "a replica rank rewrote the CSV"
    assert not (options.paths.reports / "merged.json").exists()

    _world(0, 2, monkeypatch)
    common.merge_ledger(options.paths)
    assert options.paths.csv.is_file()
    assert (options.paths.reports / "merged.json").is_file()


# ---------------------------------------------------------------------------
# 3c. --resume is one decision for the whole world (no rank skips a collective)
# ---------------------------------------------------------------------------
def test_resume_is_re_run_when_a_rank_cannot_see_the_row(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A replica that cannot resume must not let the owner skip the collectives.

    ``--resume`` reads ``rows/<run_id>.json``.  If one rank resumed a row while
    another re-ran it, the resuming rank would skip the per-row allgather and
    leave the other inside it - the P0-5 hang.  The decision is therefore
    gathered: when the ranks disagree, **every** rank executes the row.
    """
    test, options, rows = _test_and_options(tmp_path)
    row = dict(rows[0])
    row.update(status="PASS", metrics={"relative_residual": 1e-12})
    _world(0, 2, monkeypatch)
    common.write_row(options.paths, row)
    options = common.FullModeOptions(
        mode="solve", paths=options.paths, ranks=1, resume=True)

    executed: list[str] = []

    def execute(selected):
        executed.append(selected.tag)
        return _make_result(selected)

    # rank 0 owns the row file and could resume it; rank 1 cannot see it.
    # Two collectives per row: the resume decision, then the row's verdict.
    monkeypatch.setattr(common, "allgather_rank_records", _fake_gather([
        [{"rank": 0, "resumable": True}, {"rank": 1, "resumable": False}],
        _world_records(2, residual=1e-12, iterations=7, converged=True),
    ]))
    payload, code = common.run_mode(
        "solve", test, options, rows=[dict(row)], execute=execute)
    entry = payload["rows"]["items"][0]
    assert executed == ["torch/lattice"], (
        "a row only one rank could resume was skipped by the owner")
    assert entry["resumed"] is False
    assert entry["metrics"]["resume_disagreement"] is True
    assert "re-run" in entry["metrics"]["resume_reason"]
    assert entry["status"] == "PASS"
    assert code == 0


def test_resume_is_reused_when_every_rank_can_resume_it(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A unanimous "yes" keeps ``--resume`` semantics (and its cheapness)."""
    test, options, rows = _test_and_options(tmp_path)
    row = dict(rows[0])
    row.update(status="PASS", metrics={"relative_residual": 1e-12})
    _world(0, 2, monkeypatch)
    common.write_row(options.paths, row)
    options = common.FullModeOptions(
        mode="solve", paths=options.paths, ranks=1, resume=True)

    def forbidden(selected):  # pragma: no cover - the row is resumed
        raise AssertionError(f"{selected.tag} was executed although all ranks "
                             "could resume it")

    monkeypatch.setattr(common, "allgather_rank_records", _fake_gather([
        [{"rank": 0, "resumable": True}, {"rank": 1, "resumable": True}],
        _world_records(2, residual=1e-12, iterations=7, converged=True),
    ]))
    payload, code = common.run_mode(
        "solve", test, options, rows=[dict(row)], execute=forbidden)
    entry = payload["rows"]["items"][0]
    assert entry["resumed"] is True
    assert entry["metrics"]["resume_disagreement"] is False
    assert entry["metrics"]["resume_gather"] == "allgather"
    assert code == 0


def test_a_single_process_resumes_without_a_collective(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The default single-rank path keeps reading its own row file."""
    test, options, rows = _test_and_options(tmp_path)
    row = dict(rows[0])
    row.update(status="PASS", metrics={"relative_residual": 1e-12})
    common.write_row(options.paths, row)
    options = common.FullModeOptions(
        mode="solve", paths=options.paths, ranks=1, resume=True)

    def forbidden(selected):  # pragma: no cover - the row is resumed
        raise AssertionError("the single-rank resume re-ran the row")

    payload, code = common.run_mode(
        "solve", test, options, rows=[dict(row)], execute=forbidden)
    entry = payload["rows"]["items"][0]
    assert entry["resumed"] is True
    assert entry["metrics"]["resume_gather"] == "single_rank"
    assert code == 0


# ---------------------------------------------------------------------------
# 4. the driver: unanimous launch, worst outcome
# ---------------------------------------------------------------------------
def test_driver_preflight_skip_stops_the_launch_on_every_rank(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A rank that must SKIP(mpi_grid) keeps every rank out of the native call."""
    _, options, rows = _test_and_options(
        tmp_path, tag="kupl/native/solve", ranks=4, grid=[1, 1, 2, 2])
    row = dict(rows[0])
    _world(0, 4, monkeypatch)

    records = _world_records(4)
    records[2] = _record(2, "SKIP", reason="lattice_grid does not tile here",
                         skip_token="mpi_grid")
    monkeypatch.setattr(common, "allgather_rank_records",
                        _fake_gather(records))

    def forbidden(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("the native call ran although a rank had to skip")

    monkeypatch.setattr(driver.native, "run_case", forbidden)
    result = driver.run_row(row=row, options=options.as_dict(),
                            paths=options.paths.as_dict(), tag=row["tag"])
    assert result is not None
    assert result["status"] == "SKIP", result
    assert result["skip_token"] == "mpi_grid"
    assert result["rank_phase"] == "preflight"
    assert result["rank_disagreement"] is True
    assert result["mpi_size"] == 4
    # This rank could run on its own: the skip is the *collective* decision,
    # which is exactly the point.
    assert result["grid_source"] == "row"
    assert result["lattice_grid"] == [1, 1, 2, 2]


def test_driver_skips_before_running_when_the_rank_count_is_wrong(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """``SKIP(mpi_grid)`` still wins before native execution (single rank)."""
    row = {"tag": "kupl/native/solve", "lattice": list(LATTICE),
           "dtype": "c64", "ranks": 4, "solver": "bicgstab",
           "dslash": "wilson", "seed": 42}

    def forbidden(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("the native call ran at the wrong world size")

    monkeypatch.setattr(driver.native, "run_case", forbidden)
    result = driver.run_row(row=row, options={"repeat": 1, "warmup": 1},
                            tag=row["tag"])
    assert result is not None
    assert result["status"] == "SKIP"
    assert result["skip_token"] == "mpi_grid"
    assert "mpirun" in result["reason"]
    assert result["rank_gather"] == "single_rank"
    assert result["rank_disagreement"] is False


def test_a_local_skip_does_not_hide_another_ranks_error(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The pre-flight verdict is the *worst* one, not this rank's own.

    Every rank must leave the launch with the same execution status, otherwise
    ``--fail-fast`` (and the written row) would depend on which process happens
    to be rank 0.  This rank's own decision is ``SKIP(mpi_grid)``; rank 1's is an
    ``ERROR`` - an error is never demoted to a skip.
    """
    row = {"tag": "kupl/native/solve", "backend": "kupl",
           "lattice": list(LATTICE), "dtype": "c64", "ranks": 3,
           "solver": "all", "dslash": "all", "seed": 42}
    _world(0, 3, monkeypatch)
    monkeypatch.setattr(common, "allgather_rank_records", _fake_gather([
        _record(0, "SKIP", reason="no even grid tiles this lattice",
                skip_token="mpi_grid"),
        _record(1, "ERROR", reason="the kupl library is missing on this node"),
        _record(2, "PASS"),
    ]))

    def forbidden(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("the native call ran although a rank ERROR-ed")

    monkeypatch.setattr(driver.native, "run_case", forbidden)
    result = driver.run_row(row=row, options={"repeat": 1, "warmup": 1},
                            tag=row["tag"])
    assert result["status"] == "ERROR", result
    assert "rank 1/3" in result["reason"]
    assert "kupl library" in result["reason"]
    assert result["rank_disagreement"] is True


def test_driver_outcome_gather_cannot_hide_a_rank_error(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Rank 0's good numbers do not outvote rank 3's error."""
    test, options, rows = _test_and_options(
        tmp_path, tag="kupl/native/solve", ranks=4, grid=[1, 1, 2, 2])
    row = dict(rows[0])
    _world(0, 4, monkeypatch)
    monkeypatch.setattr(
        driver.native, "run_case",
        lambda kind, **kwargs: {"relative_residual": 1e-12, "iterations": 9,
                                "converged": True, "seconds": 0.5,
                                "threads": 1, "shm": False})
    records = _world_records(4, residual=1e-12, iterations=9, converged=True)
    records[3] = _record(3, "ERROR", reason="rank 3 lost its GPU")
    monkeypatch.setattr(
        common, "allgather_rank_records",
        _fake_gather([_world_records(4), records]))

    result = driver.run_row(row=row, options=options.as_dict(),
                            paths=options.paths.as_dict(), tag=row["tag"])
    assert result["status"] == "ERROR", result
    assert "rank 3/4" in result["reason"]
    assert result["rank_disagreement"] is True
    assert result["rank_outcomes"][3]["status"] == "ERROR"

    judged = common._apply_driver(
        {"run_id": "unit", "tag": row["tag"], "metrics": {}}, result,
        test.cases[0])
    assert judged["status"] == "ERROR", judged


def test_driver_records_the_worst_numbers_without_declaring_a_verdict(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, options, rows = _test_and_options(
        tmp_path, tag="kupl/native/solve", ranks=4, grid=[1, 1, 2, 2])
    row = dict(rows[0])
    _world(0, 4, monkeypatch)
    monkeypatch.setattr(
        driver.native, "run_case",
        lambda kind, **kwargs: {"relative_residual": 1e-12, "iterations": 9,
                                "converged": True, "seconds": 0.5,
                                "threads": 1, "shm": False})
    monkeypatch.setattr(common, "allgather_rank_records", _fake_gather([
        _world_records(4),
        [_record(0, residual=1e-12, iterations=9),
         _record(1, residual=4e-7, iterations=12, converged=False),
         _record(2, residual=1e-12, iterations=9),
         _record(3, residual=1e-12, iterations=9)],
    ]))

    result = driver.run_row(row=row, options=options.as_dict(),
                            paths=options.paths.as_dict(), tag=row["tag"])
    assert "status" not in result, "a completed row must not declare a verdict"
    assert result["rank_disagreement"] is True
    assert result["iterations"] == 12
    assert result["relative_residual"] == 4e-7
    assert result["converged"] is False


# ---------------------------------------------------------------------------
# 5. the CLI: rank 0 writes, a replica rank does not
# ---------------------------------------------------------------------------
def test_cli_single_rank_still_writes_the_ledger(tmp_path: Path) -> None:
    outdir = tmp_path / "single"
    completed = _run_cli("--mode", "solve", "--quick", "--only",
                         "torch/lattice", "--repeat", "1", "--warmup", "1",
                         "--outdir", str(outdir),
                         "--json", str(tmp_path / "report.json"))
    assert completed.returncode in (0, 1), completed.stderr[-2000:]
    rows = sorted((outdir / "rows").glob("*.json"))
    assert len(rows) == 1, rows
    lines = (outdir / "reports" / "rows.csv").read_text("utf-8").splitlines()
    assert len(lines) == 2, lines
    assert (tmp_path / "report.json").is_file()
    stored = json.loads(rows[0].read_text("utf-8"))
    assert stored["metrics"]["mpi_size"] == 1
    assert stored["metrics"]["rank_gather"] == "single_rank"
    assert stored["metrics"]["rank_disagreement"] is False


def test_cli_replica_rank_prints_and_writes_nothing(tmp_path: Path) -> None:
    outdir = tmp_path / "replica"
    completed = _run_cli("--mode", "solve", "--quick", "--only",
                         "torch/lattice", "--repeat", "1", "--warmup", "1",
                         "--outdir", str(outdir), "--json", "-",
                         env={"OMPI_COMM_WORLD_RANK": "7",
                              "OMPI_COMM_WORLD_SIZE": "32"})
    assert completed.returncode in (0, 1), completed.stderr[-2000:]
    assert completed.stdout.strip() == "", (
        f"a replica rank wrote to stdout: {completed.stdout[:400]!r}")
    assert "rank 7/32" in completed.stderr
    assert not (outdir / "rows").exists(), "a replica rank wrote rows/"
    assert not (outdir / "reports").exists()
    assert not (outdir / "report.json").exists()


def test_cli_replica_rank_exit_code_matches_the_owner(tmp_path: Path) -> None:
    arguments = ("--mode", "solve", "--quick", "--only", "torch/lattice",
                 "--repeat", "1", "--warmup", "1")
    owner = _run_cli(*arguments, "--outdir", str(tmp_path / "o"),
                     env={"OMPI_COMM_WORLD_RANK": "0",
                          "OMPI_COMM_WORLD_SIZE": "2"})
    replica = _run_cli(*arguments, "--outdir", str(tmp_path / "r"),
                       env={"OMPI_COMM_WORLD_RANK": "1",
                            "OMPI_COMM_WORLD_SIZE": "2"})
    assert owner.returncode == replica.returncode, (
        owner.returncode, replica.returncode, owner.stderr[-500:],
        replica.stderr[-500:])
    assert (tmp_path / "o" / "rows").is_dir()
    assert not (tmp_path / "r" / "rows").exists()
