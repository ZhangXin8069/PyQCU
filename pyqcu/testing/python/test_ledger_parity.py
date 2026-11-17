"""Parity tests for the reference CSV/Excel ledger exporters.

These tests pin the two published contracts of the informal KUPL-Kunpeng
driver - the 100-column CSV (``CSV_COLUMNS``/``append_csv``/``merge_rows``/
``upgrade_csv_schema``) and the 21-sheet workbook
``kupl_kunpeng_performance_20261115.xlsx`` - against
:mod:`pyqcu.test.ledger_csv` and :mod:`pyqcu.test.ledger_excel`.

What is asserted, and why it is the contract rather than an implementation
detail:

* the frozen 100-name header still equals the driver's ``CSV_COLUMNS`` (read
  from the informal tree with :mod:`ast`, never imported);
* a reference CSV re-rendered from its own values is **byte-identical**
  (header, cell serialisation, CRLF, quoting);
* the legacy-header upgrade keeps legacy-only columns and every value;
* every record has all 100 keys, the columns the formal port cannot observe are
  ``None`` (rendered as an empty field), and the derivable ones carry the
  documented value;
* the workbook has the reference's 21 sheets in order, the 44-/27-column
  curated headers, row order, freeze pane, tab colours, table objects,
  conditional status formats, two summary charts, index hyperlinks, number
  formats and the reference's 16-significant-digit float rule.

The Excel assertions load the produced package with ``openpyxl`` when it is
installed (it is the library that generated the reference) and are skipped
otherwise: the exporter itself never imports it.

Run::

    python -m pytest pyqcu/testing/python/test_ledger_parity.py -q

The reference artifacts are read-only inputs; when the 2026-11-15 snapshot is
not present (an installed wheel, a fresh clone) the tests that compare against
it skip instead of failing.
"""

from __future__ import annotations

import ast
import csv
import io
import json
import zipfile
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import pytest

from pyqcu.test import ledger_csv, ledger_excel

ROOT = Path(__file__).resolve().parents[3]
DRIVER = ROOT / "pyqcu/testing/kupl/test_kunpeng_all.py"
REFERENCE_CSV = ROOT / "data/kunpeng_all_20261115/kupl_kunpeng_all.csv"
REFERENCE_XLSX = (
    ROOT / "data/kunpeng_all_20261115/kupl_kunpeng_performance_20261115.xlsx")

openpyxl = pytest.importorskip("openpyxl", reason="optional parity check")


# ---------------------------------------------------------------------------
# fixtures: rows shaped like pyqcu.test full-mode rows
# ---------------------------------------------------------------------------
def _row(run_id: str, tag: str, **overrides):
    """One ledger row with the fields the exporters read."""
    row = {
        "run_id": run_id,
        "status": "PASS",
        "tag": tag,
        "backend": tag.split("/")[0],
        "kind": tag.split("/")[1],
        "tier": 0,
        "lattice": [8, 8, 8, 16],
        "dtype": "c64",
        "seed": 42,
        "solver": "all",
        "dslash": "all",
        "device": "auto",
        "ranks": 1,
        "threads": None,
        "shm": False,
        "shared": False,
        "gauge": "random",
        "source": "point",
        "gauge_file": None,
        "source_file": None,
        "mass": None,
        "kappa": None,
        "csw": 1.0,
        "seconds": 0.25,
        "metrics": {"iterations": 18, "relative_residual": 7.12e-07},
        "failed": [],
        "reason": None,
        "skip_token": None,
        "resumed": False,
        "run_label": "",
        "report_path": None,
        "solution_path": None,
    }
    row.update(overrides)
    return row


def _run_options(**overrides):
    options = {
        "lattice": [8, 8, 8, 16],
        "lattice_grid": [1, 1, 1, 1],
        "mg_grid": [2, 2, 2, 2],
        "dtype": "c64",
        "seed": 42,
        "source": "point",
        "gauge": "random",
        "mass": None,
        "kappa": None,
        "csw": 1.0,
        "solver": "all",
        "dslash": "all",
        "device": "auto",
        "max_iter": None,
        "restart": 5,
        "mg_levels": 2,
        "threads": None,
        "atol": None,
    }
    options.update(overrides)
    return options


def _records(rows, *, mode="solve", options=None):
    payload = {"mode": mode, "options": {"mode": mode}}
    return ledger_csv.ledger_records(payload, rows=rows,
                                     run_options=options or _run_options())


@pytest.fixture()
def rows():
    return [
        _row("aaa111", "torch/solve/bistabcg"),
        _row("bbb222", "kupl/mg/legacy", dtype="c128", seconds=1.5,
             metrics={"iterations": 6, "relative_residual": 5.3e-11}),
        _row("ccc333", "torch/dslash/wilson", seconds=0.01,
             metrics={"relative_error": 1e-6}),
        _row("ddd444", "kupl/solve/cg", status="FAIL", seconds=None,
             metrics={}, reason="relative_residual 1e-2 > 1e-3",
             failed=["solve"]),
        _row("eee555", "cuda/smoke/init_end", status="SKIP",
             seconds=None, metrics={}, reason="cuda_bridge: bridge not built",
             skip_token="cuda_bridge"),
        _row("fff666", "torch/lattice", status="TIMEOUT", seconds=None,
             metrics={}, reason="case timeout after 5 s"),
        # The per-lattice sheets need a lattice dimension to sort on.
        _row("ggg777", "kupl/solve/bistabcg", lattice=[24, 24, 24, 64],
             dtype="c64", ranks=16, threads=18, seconds=0.2404732911381871),
        _row("hhh888", "kupl/solve/bistabcg", lattice=[24, 24, 24, 64],
             dtype="c128", ranks=16, threads=18, seconds=0.475555),
        _row("iii999", "kupl/solve/bistabcg", lattice=[32, 32, 32, 64],
             dtype="c64", ranks=16, threads=18, seconds=1.2),
        _row("jjj000", "kupl/mg/strict", lattice=[24, 24, 24, 64],
             dtype="c64", ranks=16, threads=18, seconds=0.9),
    ]


# ---------------------------------------------------------------------------
# the frozen header
# ---------------------------------------------------------------------------
def test_reference_header_is_the_driver_contract():
    columns = ledger_csv.REFERENCE_CSV_COLUMNS
    assert len(columns) == ledger_csv.REFERENCE_COLUMN_COUNT == 100
    assert len(set(columns)) == 100
    assert tuple(rule.name for rule in ledger_csv.COLUMN_RULES) == columns
    assert tuple(ledger_csv.REFERENCE_CSV_COLUMNS) == columns


@pytest.mark.skipif(not DRIVER.is_file(), reason="informal driver not present")
def test_frozen_header_matches_the_informal_driver():
    """The copy must not drift from ``CSV_COLUMNS`` (parsed, not imported)."""
    tree = ast.parse(DRIVER.read_text(encoding="utf-8"))
    declared = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "CSV_COLUMNS"
                for target in node.targets):
            declared = ast.literal_eval(node.value)
            break
    assert declared is not None, "CSV_COLUMNS not found in the driver"
    assert tuple(declared) == ledger_csv.REFERENCE_CSV_COLUMNS


def test_every_missing_column_documents_why():
    for name in ledger_csv.MISSING_COLUMNS:
        assert ledger_csv.column_note(name).strip(), f"{name} has no reason"
    assert set(ledger_csv.MISSING_COLUMNS) <= set(
        ledger_csv.REFERENCE_CSV_COLUMNS)
    assert set(ledger_csv.DERIVED_COLUMNS) <= set(
        ledger_csv.REFERENCE_CSV_COLUMNS)


# ---------------------------------------------------------------------------
# CSV: serialisation, round-trip, header upgrade
# ---------------------------------------------------------------------------
def test_csv_cell_semantics_match_the_driver():
    assert ledger_csv.csv_cell(None) == ""
    assert ledger_csv.csv_cell(True) == "True"
    assert ledger_csv.csv_cell(False) == "False"
    assert ledger_csv.csv_cell(0.0) == "0.0"
    assert ledger_csv.csv_cell(1e-06) == "1e-06"
    assert ledger_csv.csv_cell([24, 24, 24, 64]) == "[24, 24, 24, 64]"
    assert ledger_csv.csv_cell(["a", "b"]) == "['a', 'b']"


def test_csv_text_uses_the_reference_line_terminator():
    text = ledger_csv.reference_csv_text([{"run_id": "x"}])
    assert text.count("\r\n") == 2  # header + one record
    assert "\n" not in text.replace("\r\n", "")


@pytest.mark.skipif(not REFERENCE_CSV.is_file(), reason="snapshot not present")
def test_reference_csv_round_trips_byte_identically():
    raw = REFERENCE_CSV.read_bytes()
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"), newline="")))
    assert len(rows) == 142
    rendered = ledger_csv.reference_csv_text(rows).encode("utf-8")
    assert rendered == raw, "the CSV writer no longer reproduces the reference"


def test_every_record_has_all_columns(rows):
    for record in _records(rows):
        assert tuple(record) == ledger_csv.REFERENCE_CSV_COLUMNS


def test_missing_columns_are_empty_and_derived_columns_are_filled(rows):
    record = {r["run_id"]: r for r in _records(rows)}["aaa111"]
    for name in ledger_csv.MISSING_COLUMNS:
        assert record[name] is None, f"{name} must stay empty"
    assert record["lat_x"] == 8 and record["lat_t"] == 16
    assert record["rank_x_thread"] is None       # threads unset
    assert record["grid"] == [1, 1, 1, 1]
    assert record["grid_mode"] == "t-finest"
    assert record["shm"] == 0
    assert record["solver"] == "bistabcg"
    assert record["phase"] == "solve"
    assert record["iterations"] == 18
    assert record["wall_seconds"] == 0.25
    assert record["final_true_residual"] == 7.12e-07
    assert record["relative_full_residual"] == 7.12e-07
    assert record["distributed_relative_residual"] is None
    assert record["timeout"] is False
    assert record["notes"][:3] == ["backend=torch", "tag=torch/solve/bistabcg",
                                   "tier=0"]


def test_status_specific_columns(rows):
    by_id = {r["run_id"]: r for r in _records(rows)}
    failure = by_id["ddd444"]
    assert failure["error"] is None           # FAIL keeps the reason in notes
    assert any(n.startswith("reason=") for n in failure["notes"])
    assert any(n == "failed_gate=solve" for n in failure["notes"])
    skip = by_id["eee555"]
    assert skip["error"] is None
    assert "skip_token=cuda_bridge" in skip["notes"]
    assert by_id["fff666"]["timeout"] is True


def test_non_solver_rows_do_not_claim_a_solver(rows):
    by_id = {r["run_id"]: r for r in _records(rows)}
    assert by_id["ccc333"]["solver"] is None      # dslash case
    assert by_id["fff666"]["solver"] is None      # lattice case
    assert by_id["bbb222"]["solver"] == "mg"


#: Observations ``pyqcu.test.native._common_observations``/``_finish`` emit for
#: one native Legacy-MG row.  The harness merges them into ``row["metrics"]``
#: (``common._apply_driver``), so these are measurements, not defaults.
_NATIVE_METRICS = {
    "atol": 1e-10, "dof": [12, 2, 2, 2, 2], "nv_iters": 2,
    "coarse_max_iter": 40, "coarse_tol_factor": 4.0, "deflate": True,
    "verify_status": 0, "converged": True, "residual_source": "full-operator",
    "reference_seconds": 0.031, "cache_seconds": 1.75, "case_seconds": 12.5,
    "mg_levels": 2, "mg_grid": [2, 2, 2, 2], "restart": 5,
    "lattice_grid": [1, 1, 1, 1], "max_iter": 500, "iterations": 7,
    "relative_residual": 3.1e-11, "threads": 18, "shm": True, "check": "mg_legacy",
}


def _native_row(run_id="native01", tag="kupl/native/mg", **overrides):
    row = _row(run_id, tag, lattice=[24, 24, 24, 64], dtype="c128", ranks=1,
               threads=18, mass=-0.277, kappa=0.5617, seconds=0.9,
               metrics=dict(_NATIVE_METRICS))
    row.update(overrides)
    return row


def test_native_observations_fill_the_documented_columns():
    """Columns fed by native observations carry them - not ``None``."""
    record = _records([_native_row()], options=_run_options(
        lattice=[24, 24, 24, 64], lattice_grid=[1, 1, 1, 1], dtype="c128",
        mg_grid=[2, 2, 2, 2], mg_levels=2, restart=5, threads=18))[0]
    assert record["solver"] == "mg"
    assert record["atol"] == 1e-10
    assert record["mg_dof"] == [12, 2]        # dof[:mg_levels], like the driver
    assert record["nv_iters"] == 2
    assert record["coarse_max_iter"] == 40
    assert record["coarse_tol_factor"] == 4.0
    assert record["deflate"] is True
    assert record["verify_status"] == 0
    assert record["converged"] is True
    assert record["residual_source"] == "full-operator"
    assert record["residual_seconds"] == 0.031
    assert record["builder_seconds"] == 1.75
    assert record["elapsed_seconds"] == 12.5
    assert record["max_iter"] == 500
    assert record["mg_levels"] == 2 and record["mg_restart"] == 5
    assert record["grid"] == [1, 1, 1, 1] and record["grid_mode"] == "t-finest"
    assert record["shm"] == 1
    assert record["threads"] == 18 and record["rank_x_thread"] == "1x18"
    assert record["mass"] == -0.277 and record["kappa"] == 0.5617


def test_tier0_rows_do_not_invent_native_only_columns():
    """A torch row records none of them: every such column stays empty."""
    record = _records([_row("plain01", "torch/lattice", metrics={})])[0]
    for name in ("atol", "mg_dof", "nv_iters", "coarse_max_iter",
                 "coarse_tol_factor", "deflate", "verify_status", "converged",
                 "residual_source", "residual_seconds", "builder_seconds",
                 "elapsed_seconds"):
        assert record[name] is None, f"{name} must stay empty"


def test_native_tags_name_the_runner_algorithm():
    """``native/solve`` is the Bistab-CG runner, ``native/mg`` the Legacy-MG one."""
    solve = _records([_native_row("n1", "kupl/native/solve", metrics={})])[0]
    mg = _records([_native_row("n2", "kupl/native/mg", metrics={})])[0]
    dslash = _records([_native_row("n3", "kupl/native/dslash", metrics={})])[0]
    assert solve["solver"] == "bistabcg"
    assert mg["solver"] == "mg"
    assert dslash["solver"] is None


def test_merged_rows_keep_their_grid_without_run_options():
    """``--mode merge`` carries no ``lattice_grid``; the row's own copy is used."""
    row = _row("m1", "torch/solve/bistabcg", lattice=[24, 24, 24, 64], ranks=16,
               metrics={"lattice_grid": [1, 1, 4, 4]})
    record = ledger_csv.ledger_records(
        {"mode": "merge", "options": {"mode": "merge"}}, rows=[row],
        run_options={"dtype": "c64", "seed": 42})[0]
    assert record["grid"] == [1, 1, 4, 4]
    assert record["grid_mode"] == "balanced"


def test_phase_mapping():
    rows = [_row("x1", "torch/lattice")]
    assert _records(rows, mode="solve")[0]["phase"] == "solve"
    assert _records(rows, mode="run")[0]["phase"] == "solve"
    assert _records(rows, mode="merge")[0]["phase"] == "solve"
    assert _records(rows, mode="validate")[0]["phase"] == "validate"
    assert _records(rows, mode="list")[0]["phase"] is None


def test_records_are_sorted_by_run_id():
    rows = [_row("zzz", "torch/lattice"), _row("aaa", "torch/lattice")]
    assert [r["run_id"] for r in _records(rows)] == ["aaa", "zzz"]


def test_input_records_use_the_role_phase():
    """A cache entry is a prepare row; a gauge/source entry is a prewarm row.

    One ``--mode prepare`` document validates the gauge/source files *and*
    probes the cache, and the reference driver files those two kinds under
    different phases (``prepare``/``cache_build`` vs ``prewarm``/
    ``input_prewarm``).  Keying the phase on the role keeps a gauge validation
    row from claiming it built a cache.
    """
    payload = {
        "mode": "prepare",
        "options": {"mode": "prepare"},
        "inputs": [
            {"role": "gauge", "path": "/tmp/gauge.h5", "status": "PASS",
             "bytes": 1024},
            {"role": "cache", "path": "/tmp/cache.h5", "status": "PASS",
             "bytes": 2048},
        ],
    }
    records = ledger_csv.ledger_records(payload, run_options=_run_options())
    assert {r["phase"] for r in records} == {"prepare", "prewarm"}
    by_role = {r["cache_path"] or r["gauge_path"]: r for r in records}
    assert by_role["/tmp/cache.h5"]["solver"] == "cache_build"
    assert by_role["/tmp/gauge.h5"]["solver"] == "input_prewarm"
    assert by_role["/tmp/cache.h5"]["run_id"].startswith("cache-")
    assert by_role["/tmp/gauge.h5"]["run_id"].startswith("prewarm-")
    for record in records:
        assert record["dtype"] == "c64"
        assert record["lat_t"] == 16

    prewarm = {"mode": "prewarm", "options": {"mode": "prewarm"},
               "inputs": [payload["inputs"][0]]}
    warm = ledger_csv.ledger_records(prewarm, run_options=_run_options())
    assert {r["phase"] for r in warm} == {"prewarm"}
    assert all(r["solver"] == "input_prewarm" for r in warm)
    assert all(r["run_id"].startswith("prewarm-") for r in warm)


def test_input_run_ids_are_deterministic():
    payload = {"mode": "prewarm", "options": {"mode": "prewarm"},
               "inputs": [{"role": "gauge", "path": "/tmp/g.h5",
                           "status": "PASS"}]}
    first = ledger_csv.ledger_records(payload, run_options=_run_options())
    second = ledger_csv.ledger_records(payload, run_options=_run_options())
    assert first == second


def test_a_run_document_writes_one_row_per_step_input():
    """``--mode run`` observes an input twice: one row, the better observation.

    Both steps derive the same ``run_id`` (a gauge read is ``phase=prewarm`` in
    either step), so writing both would duplicate the id - a join on ``run_id``
    double-counts - and the reference table has one row per executed row.  The
    surviving row is a real observation taken whole: the full prewarm read
    (SHA256 + duration) wins over the prepare ``stat()``, and a failure is never
    hidden by the successful observation of the other step.
    """
    prepare = {"role": "gauge", "path": "/tmp/gauge.h5", "status": "PASS",
               "bytes": 1024, "seconds": 0.000004}
    prewarm = dict(prepare, sha256="ab" * 32, seconds=1.25)
    payload = {
        "mode": "run",
        "options": {"mode": "run"},
        "rows": {"items": [_row("solve01", "torch/lattice")]},
        "steps": {"prepare": {"inputs": [prepare]},
                  "prewarm": {"inputs": [prewarm]}},
    }
    records = ledger_csv.ledger_records(payload, run_options=_run_options())
    ids = [record["run_id"] for record in records]
    assert ids.count("solve01") == 1
    assert len(ids) == len(set(ids)) == 2       # the solve row + one input row
    gauge = [record for record in records
             if record["run_id"].startswith("prewarm-")][0]
    assert gauge["phase"] == "prewarm" and gauge["status"] == "PASS"
    assert gauge["gauge_path"] == "/tmp/gauge.h5"
    assert gauge["gauge_sha256"] == "ab" * 32          # the prewarm read ...
    assert gauge["gauge_prewarm_seconds"] == 1.25      # ... not the prepare stat()

    # A failure is never hidden behind the other step's PASS.
    payload["steps"]["prewarm"]["inputs"] = [
        dict(prepare, status="FAIL", reason="sha256 mismatch")]
    records = ledger_csv.ledger_records(payload, run_options=_run_options())
    warm = [record for record in records
            if record["run_id"].startswith("prewarm-")]
    assert len(warm) == 1
    assert warm[0]["status"] == "FAIL"
    assert "reason=sha256 mismatch" in warm[0]["notes"]
    ids = [record["run_id"] for record in records]
    assert len(ids) == len(set(ids)) == 2


def test_prewarm_sha_and_dtype_are_derived_from_inputs():
    payload = {
        "mode": "solve",
        "options": {"mode": "solve"},
        "inputs": [{"role": "gauge", "path": "/tmp/gauge.h5",
                    "status": "PASS", "sha256": "ab" * 32, "seconds": 1.25,
                    "io": {"dtype": "complex64"}}],
    }
    row = _row("x1", "torch/lattice", gauge_file="/tmp/gauge.h5")
    records = ledger_csv.ledger_records(
        payload, rows=[row], run_options=_run_options())
    record = records[0]
    assert record["gauge_sha256"] == "ab" * 32
    assert record["gauge_file_dtype"] == "complex64"
    assert record["gauge_prewarm_seconds"] == 1.25
    assert record["gauge_path"] == "/tmp/gauge.h5"


def test_write_ledger_csv_is_atomic_and_deterministic(tmp_path, rows):
    records = _records(rows)
    first = ledger_csv.write_ledger_csv(tmp_path / "a.csv", records)
    second = ledger_csv.write_ledger_csv(tmp_path / "b.csv", records)
    assert first.read_bytes() == second.read_bytes()
    assert list(tmp_path.glob("*.tmp")) == []


def test_legacy_header_is_upgraded_not_misaligned(tmp_path):
    """A 94-column table (the driver's pre-2026-11 header) keeps its values."""
    legacy = [name for name in ledger_csv.REFERENCE_CSV_COLUMNS
              if name not in ("gauge_generate_seconds", "source_generate_seconds",
                              "gauge_prewarm_seconds", "source_prewarm_seconds",
                              "gauge_read_mode", "source_read_mode")]
    legacy.append("legacy_only")
    target = tmp_path / "legacy.csv"
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=legacy)
        writer.writeheader()
        writer.writerow({"run_id": "old", "phase": "solve",
                         "legacy_only": "keep-me"})
    columns = ledger_csv.upgrade_columns(target)
    assert columns[:100] == list(ledger_csv.REFERENCE_CSV_COLUMNS)
    assert columns[100:] == ["legacy_only"]
    assert (tmp_path / "legacy.csv.schema.bak").is_file()
    rows = list(csv.DictReader(target.open(encoding="utf-8", newline="")))
    assert rows[0]["run_id"] == "old"
    assert rows[0]["legacy_only"] == "keep-me"


# ---------------------------------------------------------------------------
# Excel: value convention, workbook contract
# ---------------------------------------------------------------------------
def test_excel_value_uses_sixteen_significant_digits():
    assert ledger_excel.excel_value(20.859673652797937) == 20.85967365279794
    assert ledger_excel.excel_value(7.12178161948686e-07) == 7.1216191999999999e-07 \
        or ledger_excel.excel_value(7.12178161948686e-07) == 7.12178161948686e-07
    assert ledger_excel.excel_value(None) is None
    assert ledger_excel.excel_value(True) is True
    assert ledger_excel.excel_value([24, 24, 24, 64]) == "[24, 24, 24, 64]"


def test_sheet_names_are_the_reference_set():
    assert len(ledger_excel.SHEET_NAMES) == 21
    assert ledger_excel.SHEET_NAMES[0] == "00_Cover"
    assert ledger_excel.SHEET_NAMES[-1] == "20_Job_Ledger"
    assert ledger_excel.EDITORIAL_SHEETS  # documented gap list is non-empty


#: Sheets whose row 4 is a table header in the reference workbook.  The other
#: sheets open with a title/subtitle/key-value block (``00``-``03``, ``09``,
#: ``10``), so their row 4 is data, not a header.
REFERENCE_TABLE_SHEETS = (
    "04_Performance_All", "05_L24_c64", "06_L24_c128", "07_L32", "08_MG",
    "11_Cache", "12_Failures", "13_Abandoned", "14_Input_Integrity",
    "15_Historical", "16_Scripts", "17_Data_Dictionary", "18_CSV_Index",
    "19_Reports_Index", "20_Job_Ledger",
)


@pytest.mark.skipif(not REFERENCE_XLSX.is_file(), reason="snapshot not present")
def test_every_table_sheet_keeps_the_reference_header(tmp_path, rows):
    """Each table sheet's header is the reference's, column for column."""
    reference = openpyxl.load_workbook(REFERENCE_XLSX)
    _, book = _written_workbook(tmp_path, _records(rows))
    for name in REFERENCE_TABLE_SHEETS:
        wanted = [c.value for c in reference[name][4]]
        while wanted and wanted[-1] is None:      # trailing padding cells
            wanted.pop()
        produced = [c.value for c in book[name][4]]
        while produced and produced[-1] is None:
            produced.pop()
        assert produced == wanted, f"{name}: header drifted from the reference"


def _written_workbook(tmp_path, records, **kwargs):
    path = tmp_path / "book.xlsx"
    ledger_excel.write_ledger_excel(path, records, **kwargs)
    return path, openpyxl.load_workbook(path)


def test_workbook_has_the_reference_sheets_in_order(tmp_path, rows):
    _, book = _written_workbook(tmp_path, _records(rows))
    assert tuple(book.sheetnames) == ledger_excel.SHEET_NAMES


def test_workbook_data_sheet_headers_and_row_order(tmp_path, rows):
    records = _records(rows)
    _, book = _written_workbook(tmp_path, records)
    all_rows = book["04_Performance_All"]
    assert [c.value for c in all_rows[4]] == list(
        ledger_csv.PERFORMANCE_COLUMNS)
    assert [all_rows.cell(row=r, column=1).value
            for r in range(5, all_rows.max_row + 1)] == [
        record["run_id"] for record in records]
    for name in ("05_L24_c64", "06_L24_c128", "07_L32", "08_MG"):
        assert [c.value for c in book[name][4]] == list(
            ledger_excel.MG_SHEET_COLUMNS)


def test_workbook_sheet_filters_match_the_reference_split(tmp_path, rows):
    """05-08 are the reference's own split: MG rows repeat, inputs never appear."""
    _, book = _written_workbook(tmp_path, _records(rows))

    def ids(name):
        sheet = book[name]
        return [sheet.cell(row=r, column=1).value
                for r in range(5, sheet.max_row + 1)]

    # 08_MG is the cross-lattice cut: every MG row, whatever its lattice.
    assert ids("08_MG") == ["bbb222", "jjj000"]
    # 05/06/07 are lattice cuts that *include* their MG rows (the reference's
    # 74/31/25 row counts for its 130 solve rows only add up that way).
    assert ids("05_L24_c64") == ["ggg777", "jjj000"]
    assert ids("06_L24_c128") == ["hhh888"]
    assert ids("07_L32") == ["iii999"]
    assert len(ids("04_Performance_All")) == len(rows)


def test_prepare_rows_stay_out_of_the_lattice_sheets(tmp_path):
    """A ``prepare`` record has a lattice but no solve measurement to report."""
    payload = {
        "mode": "prepare",
        "options": {"mode": "prepare"},
        "inputs": [{"role": "cache", "path": "/tmp/cache.json", "status": "PASS"}],
    }
    records = ledger_csv.ledger_records(
        payload, run_options=_run_options(lattice=[24, 24, 24, 64]))
    assert [record["phase"] for record in records] == ["prepare"]
    path, book = _written_workbook(tmp_path, records)
    for name in ("05_L24_c64", "06_L24_c128", "07_L32", "08_MG"):
        assert book[name].max_row == 4, f"{name} must stay empty"


@pytest.mark.skipif(not (REFERENCE_CSV.is_file() and REFERENCE_XLSX.is_file()),
                    reason="snapshot not present")
def test_lattice_split_reproduces_the_reference_workbook_counts():
    """The 05-08 split is the reference's: 74 / 31 / 25 / 69 rows of its CSV."""
    raw = list(csv.DictReader(
        io.StringIO(REFERENCE_CSV.read_text(encoding="utf-8"), newline="")))
    records = []
    for row in raw:
        lattice = row.get("lattice") or None
        records.append({
            "run_id": row["run_id"], "phase": row["phase"],
            "solver": row["solver"], "dtype": row["dtype"],
            "lattice": ast.literal_eval(lattice) if lattice else None,
        })
    book = openpyxl.load_workbook(REFERENCE_XLSX)
    for name in ("05_L24_c64", "06_L24_c128", "07_L32", "08_MG"):
        expected = book[name].max_row - 4
        produced = ledger_excel.lattice_sheet_records(name, records)
        assert len(produced) == expected, f"{name}: {len(produced)} != {expected}"


def test_workbook_view_number_formats_and_tables(tmp_path, rows):
    _, book = _written_workbook(tmp_path, _records(rows))
    sheet = book["04_Performance_All"]
    assert sheet.freeze_panes == "A5"
    assert sheet.sheet_view.showGridLines is False
    assert sheet.sheet_view.zoomScale == 90
    assert list(sheet.tables) == ["tbl_perf_all"]
    assert sheet.tables["tbl_perf_all"].ref == f"A4:AR{4 + len(rows)}"
    column = {name: index for index, name in enumerate(
        ledger_csv.PERFORMANCE_COLUMNS, start=1)}
    assert sheet.cell(row=5, column=column["wall_seconds"]).number_format \
        == "0.000000"
    assert sheet.cell(row=5, column=column["final_true_residual"]).number_format \
        == "0.000E+00"
    assert sheet.cell(row=5, column=column["build_mg_cache_seconds"]).number_format \
        == "0.000"


def test_workbook_status_conditional_format(tmp_path, rows):
    _, book = _written_workbook(tmp_path, _records(rows))
    for name, column in (("04_Performance_All", "C"), ("05_L24_c64", "B"),
                         ("08_MG", "B")):
        rules = book[name].conditional_formatting
        ranges = list(rules)
        assert ranges, f"{name} has no conditional format"
        assert any(column in str(item.sqref) for item in ranges)
    assert "tbl_perf_all" in book["04_Performance_All"].tables


def test_workbook_index_hyperlinks_and_charts(tmp_path, rows):
    _, book = _written_workbook(tmp_path, _records(rows))
    index = book["01_Index"]
    linked = [index.cell(row=r, column=1).value
              for r in range(5, index.max_row + 1)
              if index.cell(row=r, column=1).hyperlink is not None]
    assert linked == list(ledger_excel.SHEET_NAMES)
    assert len(book["02_Summary"]._charts) == 2


def test_workbook_summary_counts_are_derived(tmp_path, rows):
    records = _records(rows)
    counts = ledger_excel.summary_counts(records)
    assert counts["rows"] == len(records)
    assert counts["status"] == {"FAIL": 1, "PASS": 7, "SKIP": 1,
                                "TIMEOUT": 1}
    assert counts["solver"] == {"bistabcg": 4, "cg": 1, "mg": 2}
    _, book = _written_workbook(tmp_path, records)
    summary = book["02_Summary"]
    values = {summary.cell(row=r, column=1).value:
              summary.cell(row=r, column=2).value for r in range(4, 16)}
    assert values["Rows"] == len(records)
    assert values["Status counts"] == \
        '{"FAIL": 1, "PASS": 7, "SKIP": 1, "TIMEOUT": 1}'


def test_workbook_editorial_sheets_say_na(tmp_path, rows):
    _, book = _written_workbook(tmp_path, _records(rows))
    for name in ledger_excel.EDITORIAL_SHEETS:
        sheet = book[name]
        text = " ".join(str(cell.value) for row in sheet.iter_rows()
                        for cell in row if cell.value is not None)
        assert "n/a" in text.lower(), f"{name} must state what is missing"


def test_workbook_failures_sheet_lists_every_failure(tmp_path, rows):
    """FAIL/ERROR/TIMEOUT are failures; SKIP is not (it is not a failure)."""
    _, book = _written_workbook(tmp_path, _records(rows))
    sheet = book["12_Failures"]
    ids = [sheet.cell(row=r, column=1).value
           for r in range(5, sheet.max_row + 1)]
    assert ids == ["ddd444", "fff666"]


def test_workbook_part_xml_is_well_formed_and_deterministic(tmp_path, rows):
    """Two exports of the same rows are byte-identical (same target path)."""
    records = _records(rows)
    target = tmp_path / "book.xlsx"
    ledger_excel.write_ledger_excel(target, records)
    first = target.read_bytes()
    ledger_excel.write_ledger_excel(target, records)
    assert target.read_bytes() == first
    assert sorted(p.name for p in tmp_path.iterdir()) == ["book.xlsx"]
    with zipfile.ZipFile(target) as archive:
        names = archive.namelist()
        for name in names:
            if name.endswith((".xml", ".rels")):
                ElementTree.fromstring(archive.read(name))
    assert "[Content_Types].xml" in names
    assert "xl/workbook.xml" in names
    assert len([name for name in names
                if name.startswith("xl/worksheets/sheet")]) == 21


def test_workbook_cover_records_the_file_name(tmp_path, rows):
    """The only legitimately path-dependent cell is the workbook's own name."""
    path, book = _written_workbook(tmp_path, _records(rows))
    assert book["00_Cover"]["B4"].value == path.name


def test_workbook_sheet_names_have_tab_colors(tmp_path, rows):
    _, book = _written_workbook(tmp_path, _records(rows))
    for name in ledger_excel.SHEET_NAMES:
        color = book[name].sheet_properties.tabColor
        assert color is not None, f"{name} has no tab colour"
        assert color.rgb.endswith(ledger_excel.TAB_COLORS[name])


@pytest.mark.skipif(not REFERENCE_XLSX.is_file(), reason="snapshot not present")
def test_reference_workbook_structure_matches_the_model():
    """The model in ``ledger_excel`` still describes the reference workbook."""
    book = openpyxl.load_workbook(REFERENCE_XLSX)
    assert tuple(book.sheetnames) == ledger_excel.SHEET_NAMES
    assert [c.value for c in book["04_Performance_All"][4]] == list(
        ledger_csv.PERFORMANCE_COLUMNS)
    for name in ("05_L24_c64", "06_L24_c128", "07_L32", "08_MG"):
        assert [c.value for c in book[name][4]] == list(
            ledger_excel.MG_SHEET_COLUMNS)
    assert book["04_Performance_All"].freeze_panes == "A5"
    assert list(book["04_Performance_All"].tables) == ["tbl_perf_all"]
    assert len(book["02_Summary"]._charts) == 2
    assert book["04_Performance_All"].cell(row=5, column=20).number_format \
        == "0.000000"      # wall_seconds
