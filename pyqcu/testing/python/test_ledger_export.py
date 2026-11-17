"""Round-trip tests for the reference CSV/Excel export integration.

:mod:`pyqcu.test.ledger_export` is the integration point between a full-mode
document (``--mode solve|run|merge|prepare|prewarm`` with ``--json``, or the
``rows/<run_id>.json`` items alone) and the two published reference artifacts
built by bg-10/bg-15: the 100-column ``CSV_COLUMNS`` table
(:mod:`pyqcu.test.ledger_csv`) and the 21-sheet
``kupl_kunpeng_performance_20261115.xlsx`` workbook
(:mod:`pyqcu.test.ledger_excel`).

What is asserted, and why each assertion is the contract rather than an
implementation detail:

* the CSV header is the reference's 100 names **in order**, one row per ledger
  row, sorted by ``run_id`` - a spreadsheet reads a column by position;
* every metric a :mod:`pyqcu.test` row/report records that *can* be mapped is
  mapped (native ``metrics['atol']``, the torch solver tolerance, gauge/source
  path+hash+dtype, the Legacy-MG cache path/hash/identity/manifest/setup time,
  ``mpi_size``, threads, shm, MG restart/dof/...);
* a column the formal port cannot observe stays an empty field, and the compare
  gate (``policy.RESIDUAL_GATES``/``metrics['tolerance']``) is never copied
  into the solver-tolerance ``atol`` column - a fabricated value is worse than
  a blank one;
* the workbook has the reference's 21 sheets in order, the 44-/27-column
  curated headers, the same row order as the CSV, and the reference's summary
  key order (verified against the 2026-11-15 snapshot when it is present);
* the export is atomic and deterministic (two exports of the same records are
  byte-identical and no ``*.tmp`` sibling is left behind), and it refuses to
  overwrite the port's own 30-column ``reports/rows.csv``;
* the CLI hook (``export_from_payload``) is inert unless
  ``PYQCU_TEST_LEDGER_EXPORT`` asks for it, so an ordinary run is unaffected;
* ``--ledger-export {none,csv,excel,both}`` is the published CLI hook: it is
  rejected outside a full mode and under ``--mode list``, it writes into
  ``<run root>/ledger/`` and never over the port's 30-column
  ``reports/rows.csv``, and ``--ledger-export none`` overrides the environment;
* one real ``python -m pyqcu.test --quick --only torch/lattice --mode solve``
  invocation is exported end to end and its row count/ids/sheets are checked.

Run::

    python -m pytest pyqcu/testing/python/test_ledger_export.py -q

The workbook assertions use ``openpyxl`` when it is installed (the library that
generated the reference) and are skipped otherwise: the exporter itself never
imports it.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pyqcu.test import ledger_csv, ledger_excel, ledger_export

ROOT = Path(__file__).resolve().parents[3]
REFERENCE_XLSX = (
    ROOT / "data/kunpeng_all_20261115/kupl_kunpeng_performance_20261115.xlsx")

#: The reference workbook's first summary block (``02_Summary`` A4:A15), copied
#: from the 2026-11-15 snapshot: a reader filters on these labels.
REFERENCE_SUMMARY_LABELS: tuple[str, ...] = (
    "Rows",
    "Solve rows",
    "Prepare/cache rows",
    "Reports",
    "CSV files",
    "Cache manifests",
    "Local h5",
    "Status counts",
    "Phase counts",
    "Solver counts (solve)",
    "Dtype counts (solve)",
    "Lattice counts (solve)",
)

#: Columns that must stay empty unless a native/cache row really recorded them.
NATIVE_ONLY_COLUMNS: tuple[str, ...] = (
    "atol", "mg_dof", "nv_iters", "coarse_max_iter", "coarse_tol_factor",
    "deflate", "verify_status", "converged", "residual_source",
    "residual_seconds", "builder_seconds", "cache_path", "cache_sha256",
    "cache_identity_sha256", "cache_manifest_path", "distributed_relative_residual",
    "out_path", "err_path", "returncode", "review", "node", "freq_khz",
    "freq_max_khz", "governor", "profile_rank", "gauge_read_seconds",
    "source_read_seconds", "gauge_read_mode", "source_read_mode",
    "hdf5_mpio_read",
)


# ---------------------------------------------------------------------------
# fixtures: rows shaped like pyqcu.test full-mode rows
# ---------------------------------------------------------------------------
def _row(run_id: str, tag: str, **overrides) -> dict:
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
        "shm": None,
        "shared": False,
        "gauge": "random",
        "source": "point",
        "gauge_file": None,
        "source_file": None,
        "mass": None,
        "kappa": None,
        "csw": 1.0,
        "seconds": 0.25,
        "metrics": {},
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


def _run_options(**overrides) -> dict:
    """``Test.config()``-shaped run configuration (the run, not the mode)."""
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


def _payload(rows, *, mode="solve", root=None, **extra) -> dict:
    """A full-mode document carrying ``rows`` (and the mode's options)."""
    document = {
        "mode": mode,
        "options": {"mode": mode},
        "rows": {"count": len(rows), "items": list(rows)},
        "counts": {"PASS": len(rows)},
    }
    if root is not None:
        document["paths"] = {"root": str(root), "rows": str(Path(root) / "rows"),
                             "reports": str(Path(root) / "reports"),
                             "csv": str(Path(root) / "reports" / "rows.csv")}
    document.update(extra)
    return document


#: Observations a native Legacy-MG row carries (``native._common_observations``
#: + ``_finish`` + ``cache.CacheEntry.as_dict``), merged into ``row["metrics"]``
#: by ``common._apply_driver``.  Every value here is a measurement.
_NATIVE_METRICS = {
    "atol": 1e-10, "dof": [12, 2, 2, 2, 2], "nv_iters": 2,
    "coarse_max_iter": 40, "coarse_tol_factor": 4.0, "deflate": True,
    "verify_status": 0, "converged": True, "residual_source": "full-operator",
    "reference_seconds": 0.031, "cache_seconds": 1.75, "case_seconds": 12.5,
    "mg_levels": 2, "mg_grid": [2, 2, 2, 2], "restart": 5,
    "lattice_grid": [1, 1, 1, 1], "max_iter": 500, "iterations": 7,
    "relative_residual": 3.1e-11, "threads": 18, "shm": True,
    "mpi_size": 4, "mpi_rank": 0,
    "gauge_path": "/data/gauge.h5", "gauge_sha256": "aa" * 32,
    "gauge_file_dtype": "complex64", "gauge_origin": "hdf5",
    "source_path": "/data/source.h5", "source_sha256": "bb" * 32,
    "source_file_dtype": "complex64", "source_origin": "hdf5",
    "cache_path": "/data/mg-cache.h5",
    "cache_manifest_path": "/data/mg-cache.json",
    "cache_identity_sha256": "cc" * 32, "cache_file_sha256": "dd" * 32,
    "cache_setup_seconds": 4.5, "cache_reused": True,
    "asset_source": "cache",
}


def _native_mg_row(run_id: str = "native01") -> dict:
    """A native Legacy-MG row: every mappable observation is present."""
    return _row(run_id, "kupl/native/mg", lattice=[24, 24, 24, 64],
                dtype="c128", ranks=4, threads=18, mass=-0.277,
                kappa=0.5617, seconds=0.9, metrics=dict(_NATIVE_METRICS),
                report_path=None, solution_path=None)


def _tier0_rows() -> list[dict]:
    """A small mixed ledger: torch, kupl solve and a skip/failure row."""
    return [
        _row("aaa111", "torch/lattice"),
        _row("bbb222", "torch/solve/cg", seconds=2.5,
             metrics={"iterations": 14, "relative_residual": 2.6e-07,
                      "converged": True, "solver_tolerance": 1e-06,
                      "tolerance": 1e-06, "elapsed_seconds": 34.5,
                      "lattice_grid": [1, 1, 1, 1], "restart": 30}),
        _row("ccc333", "kupl/solve/bistabcg", lattice=[24, 24, 24, 64],
             ranks=16, threads=18, seconds=0.24, shm=True,
             metrics={"iterations": 18, "relative_residual": 7.12e-07,
                      "shm": True, "threads": 18, "lattice_grid": [1, 1, 4, 4]}),
        _row("ddd444", "kupl/solve/cg", status="FAIL", seconds=None,
             reason="relative_residual 1e-2 > 1e-3", failed=["solve"]),
        _row("eee555", "cuda/smoke/init_end", status="SKIP", seconds=None,
             reason="cuda_bridge: bridge not built", skip_token="cuda_bridge"),
    ]


def _read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _export(tmp_path: Path, rows, **kwargs) -> ledger_export.LedgerExport:
    """Export ``rows`` into ``tmp_path`` (a payload without a run root)."""
    return ledger_export.export_ledger(
        _payload(rows), rows=rows, directory=tmp_path, **kwargs)


# ---------------------------------------------------------------------------
# 1. the CSV contract
# ---------------------------------------------------------------------------
def test_csv_header_is_the_reference_contract_in_order(tmp_path) -> None:
    """The written header is the reference's 100 names, in the same order."""
    result = _export(tmp_path, _tier0_rows(), run_options=_run_options())
    header, _ = _read_csv(result.csv_path)
    assert header == list(ledger_csv.REFERENCE_CSV_COLUMNS)
    assert len(header) == len(set(header)) == 100
    assert result.columns == tuple(header)
    # The record objects carry every column too (no KeyError-prone rows).
    assert all(set(record) == set(header) for record in _read_csv(result.csv_path)[1])


def test_row_count_and_run_id_order_follow_merge_rows(tmp_path) -> None:
    """One CSV row per ledger row, sorted by ``run_id`` (``merge_rows`` rule)."""
    rows = list(reversed(_tier0_rows()))
    result = _export(tmp_path, rows, run_options=_run_options())
    _, records = _read_csv(result.csv_path)
    assert len(records) == len(rows) == result.rows
    assert [record["run_id"] for record in records] == sorted(
        row["run_id"] for row in rows)
    assert result.run_ids == tuple(record["run_id"] for record in records)


def test_excel_row_count_matches_the_csv(tmp_path) -> None:
    openpyxl = pytest.importorskip("openpyxl", reason="optional workbook check")
    rows = _tier0_rows()
    result = _export(tmp_path, rows, run_options=_run_options())
    book = openpyxl.load_workbook(result.excel_path)
    sheet = book["04_Performance_All"]
    ids = [sheet.cell(row=row, column=1).value
           for row in range(5, sheet.max_row + 1)]
    _, records = _read_csv(result.csv_path)
    assert ids == [record["run_id"] for record in records]
    assert len(ids) == len(rows)


# ---------------------------------------------------------------------------
# 2. mapping: every observable metric is written, nothing is invented
# ---------------------------------------------------------------------------
def test_native_row_maps_every_observable_metric(tmp_path) -> None:
    """Native observations (including cache/input evidence) reach the table."""
    result = _export(tmp_path, [_native_mg_row()],
                     run_options=_run_options(lattice=[24, 24, 24, 64],
                                              dtype="c128", restart=5,
                                              mg_levels=2, threads=18))
    _, records = _read_csv(result.csv_path)
    record = records[0]
    assert record["solver"] == "mg"
    assert record["atol"] == "1e-10"
    assert record["mg_dof"] == "[12, 2]"          # dof[:mg_levels], like the driver
    assert record["nv_iters"] == "2"
    assert record["coarse_max_iter"] == "40"
    assert record["coarse_tol_factor"] == "4.0"
    assert record["deflate"] == "True"
    assert record["verify_status"] == "0"
    assert record["converged"] == "True"
    assert record["residual_source"] == "full-operator"
    assert record["residual_seconds"] == "0.031"
    assert record["builder_seconds"] == "1.75"
    assert record["build_mg_cache_seconds"] == "4.5"
    assert record["elapsed_seconds"] == "12.5"
    assert record["max_iter"] == "500"
    assert record["iterations"] == "7"
    assert record["mg_levels"] == "2" and record["mg_restart"] == "5"
    assert record["grid"] == "[1, 1, 1, 1]" and record["grid_mode"] == "balanced"
    assert record["shm"] == "1"
    assert record["threads"] == "18" and record["rank_x_thread"] == "4x18"
    assert record["ranks"] == "4"
    # Input evidence the row itself recorded (not the run's request).
    assert record["gauge_path"] == "/data/gauge.h5"
    assert record["gauge_sha256"] == "aa" * 32
    assert record["gauge_file_dtype"] == "complex64"
    assert record["source_path"] == "/data/source.h5"
    assert record["source_sha256"] == "bb" * 32
    # Legacy-MG cache evidence (cache.CacheEntry.as_dict).
    assert record["cache_path"] == "/data/mg-cache.h5"
    assert record["cache_sha256"] == "dd" * 32
    assert record["cache_identity_sha256"] == "cc" * 32
    assert record["cache_manifest_path"] == "/data/mg-cache.json"
    assert record["cache_manifest_sha256"] == ""      # opt-in hashing only
    assert "cache_reused=true" in record["notes"]
    assert "asset_source=cache" in record["notes"]


def test_torch_solver_tolerance_is_mapped_but_the_gate_is_not(tmp_path) -> None:
    """``metrics['solver_tolerance']`` is the solver tolerance; the gate is not."""
    gate_only = _row("g1", "torch/solve/cg", metrics={"tolerance": 1e-06})
    solved = _row("g2", "torch/solve/cg",
                  metrics={"tolerance": 1e-06, "solver_tolerance": 1e-08})
    result = _export(tmp_path, [gate_only, solved], run_options=_run_options())
    _, records = _read_csv(result.csv_path)
    by_id = {record["run_id"]: record for record in records}
    assert by_id["g1"]["atol"] == ""      # only the compare gate was recorded
    assert by_id["g2"]["atol"] == "1e-08"  # the tolerance the solver was asked for


def test_unobservable_columns_stay_blank_and_are_never_fabricated(tmp_path) -> None:
    """A tier-0 row records none of the native/KUPL-only columns."""
    row = _row("plain01", "torch/lattice", metrics={})
    result = _export(tmp_path, [row], run_options=_run_options())
    _, records = _read_csv(result.csv_path)
    record = records[0]
    for name in NATIVE_ONLY_COLUMNS:
        assert record[name] == "", f"{name} must stay an empty field"
    for name in ledger_csv.MISSING_COLUMNS:
        assert record[name] == "", f"{name} is not observable by pyqcu.test"
    assert record["atol"] == ""
    assert record["wall_seconds"] == "0.25"       # the timed region, not a guess
    assert record["status"] == "PASS"
    assert record["timeout"] == "False"


def test_the_missing_column_set_is_pinned() -> None:
    """Adding or silently dropping a mapping must update this list."""
    assert ledger_csv.MISSING_COLUMNS == (
        "cache_kind", "distributed_relative_residual", "node", "freq_khz",
        "freq_max_khz", "governor", "profile_rank", "dslash_seconds",
        "dslash_calls", "dslash_bytes_mb", "dslash_gbps_rank",
        "dslash_gbps_total", "clover_seconds", "clover_calls",
        "clover_bytes_mb", "clover_gbps_rank", "clover_gbps_total",
        "reduce_seconds", "reduce_calls", "halo_pack_seconds",
        "halo_sync_seconds", "halo_copy_seconds", "phase_prepare_seconds",
        "phase_solve_seconds", "phase_reconstruct_seconds",
        "input_prebind_seconds", "gauge_read_seconds", "source_read_seconds",
        "gauge_generate_seconds", "source_generate_seconds",
        "gauge_read_mode", "source_read_mode", "hdf5_mpio_read", "out_path",
        "err_path", "returncode", "review",
    )
    for name in ledger_csv.MISSING_COLUMNS:
        assert ledger_csv.column_note(name).strip(), f"{name} has no reason"


def test_prepare_document_exports_cache_and_input_rows(tmp_path) -> None:
    """A ``--mode prepare`` document yields one cache row and the input rows."""
    payload = {
        "mode": "prepare",
        "options": {"mode": "prepare"},
        "inputs": [
            {"role": "gauge", "path": "/tmp/gauge.h5", "status": "PASS",
             "bytes": 1024, "sha256": "ab" * 32,
             "seconds": 1.25, "io": {"dtype": "complex64"}},
            {"role": "source", "path": "/tmp/source.h5", "status": "SKIP",
             "reason": "no source file requested (--source-file unset)"},
            {"role": "cache", "path": "/tmp/caches/index.json",
             "status": "PASS", "bytes": 596},
        ],
    }
    result = ledger_export.export_ledger(
        payload, directory=tmp_path,
        run_options=_run_options(lattice=[24, 24, 24, 64]))
    _, records = _read_csv(result.csv_path)
    by_id = {record["run_id"][:7]: record for record in records}
    assert {record["phase"] for record in records} == {"prepare", "prewarm"}
    assert {record["solver"] for record in records} == {"cache_build",
                                                        "input_prewarm"}
    gauge = by_id["prewarm"]
    assert gauge["gauge_path"] == "/tmp/gauge.h5"
    assert gauge["gauge_sha256"] == "ab" * 32
    assert gauge["gauge_file_dtype"] == "complex64"
    assert gauge["gauge_prewarm_seconds"] == "1.25"
    cache = [record for record in records
             if record["phase"] == "prepare"][0]
    assert cache["cache_manifest_path"] == "/tmp/caches/index.json"
    # The probe index is not a cache artifact: no artifact is claimed.
    assert cache["cache_path"] == ""
    assert cache["lat_t"] == "64"


def _run_payload(rows, *, root=None) -> dict:
    """A ``--mode run`` document: both steps observe the same gauge/source.

    ``steps.prepare`` validates and ``stat()``-s the files (no SHA256, a
    microsecond duration) and probes the cache; ``steps.prewarm`` reads and
    hashes them, which is the measurement the reference files under
    ``phase=prewarm``.
    """
    gauge = {"role": "gauge", "path": "/data/gauge.h5", "status": "PASS",
             "bytes": 2365776, "seconds": 0.000004,
             "io": {"dtype": "complex64", "lattice": [8, 8, 8, 16]}}
    source = {"role": "source", "path": "/data/source.h5", "status": "PASS",
              "bytes": 792888, "seconds": 0.000003,
              "io": {"dtype": "complex64", "lattice": [8, 8, 8, 16]}}
    cache = {"role": "cache", "path": "/data/caches/index.json",
             "status": "PASS", "bytes": 594}
    return _payload(
        rows, mode="run", root=root,
        steps={"prepare": {"inputs": [gauge, source, cache]},
               "prewarm": {"inputs": [
                   dict(gauge, sha256="ab" * 32, seconds=0.003105),
                   dict(source, sha256="cd" * 32, seconds=0.000665)]},
               "solve": None, "merge": None})


def test_run_export_has_one_row_per_run_id(tmp_path) -> None:
    """F1: the prepare/prewarm double observation must not duplicate a run_id.

    ``--mode run`` lists the same gauge/source entry under ``steps.prepare`` and
    ``steps.prewarm``, and both derive the same ``run_id`` (a gauge/source read
    is ``phase=prewarm`` in either step).  Exporting both produced 7 rows with
    5 unique ids and pushed the duplicate into ``04_Performance_All`` and
    ``14_Input_Integrity``; a join on ``run_id`` double-counted.  The table now
    carries one row per id, and that row is the real prewarm observation -
    SHA256 plus the read duration, not the prepare ``stat()``.
    """
    openpyxl = pytest.importorskip("openpyxl", reason="optional workbook check")
    payload = _run_payload([_row("solve01", "torch/lattice")],
                           root=tmp_path / "run")
    result = ledger_export.export_ledger(payload, run_options=_run_options())
    header, records = _read_csv(result.csv_path)

    assert header == list(ledger_csv.REFERENCE_CSV_COLUMNS)
    ids = [record["run_id"] for record in records]
    assert len(ids) == len(set(ids)) == result.rows == 4
    assert {record["phase"] for record in records} == {"prepare", "prewarm",
                                                        "solve"}
    gauge = [record for record in records
             if record["gauge_path"] == "/data/gauge.h5"]
    source = [record for record in records
              if record["source_path"] == "/data/source.h5"]
    assert len(gauge) == 1 and len(source) == 1
    assert gauge[0]["phase"] == "prewarm" and gauge[0]["status"] == "PASS"
    assert gauge[0]["gauge_sha256"] == "ab" * 32
    # The read duration, not the prepare stat() microsecond.
    assert gauge[0]["gauge_prewarm_seconds"] == "0.003105"
    assert source[0]["source_sha256"] == "cd" * 32
    assert source[0]["source_prewarm_seconds"] == "0.000665"

    book = openpyxl.load_workbook(result.excel_path)
    performance = book["04_Performance_All"]
    assert [performance.cell(row=row, column=1).value
            for row in range(5, performance.max_row + 1)] == ids
    summary = book["02_Summary"]
    values = {summary.cell(row=row, column=1).value:
              summary.cell(row=row, column=2).value for row in range(4, 16)}
    # 2 inputs + cache + solve; the duplicated export had 6 rows.
    assert values["Rows"] == len(records) == 4
    assert values["Prepare/cache rows"] == 3
    integrity = book["14_Input_Integrity"]
    files = [integrity.cell(row=row, column=1).value
             for row in range(5, integrity.max_row + 1)]
    files = [value for value in files
             if isinstance(value, str) and value.startswith("/data/")]
    # One row per input file: the duplicate pushed this to 4.
    assert files == ["/data/gauge.h5", "/data/source.h5"]


# ---------------------------------------------------------------------------
# 3. the workbook contract
# ---------------------------------------------------------------------------
def test_workbook_has_the_reference_sheets_headers_and_row_order(tmp_path) -> None:
    openpyxl = pytest.importorskip("openpyxl", reason="optional workbook check")
    result = _export(tmp_path, _tier0_rows(), run_options=_run_options())
    book = openpyxl.load_workbook(result.excel_path)
    assert tuple(book.sheetnames) == ledger_excel.SHEET_NAMES
    assert len(book.sheetnames) == 21 and result.sheets == ledger_excel.SHEET_NAMES
    assert [cell.value for cell in book["04_Performance_All"][4]] == list(
        ledger_csv.PERFORMANCE_COLUMNS)
    for name in ("05_L24_c64", "06_L24_c128", "07_L32", "08_MG"):
        assert [cell.value for cell in book[name][4]] == list(
            ledger_excel.MG_SHEET_COLUMNS)
    # The per-lattice split stays the reference's: MG rows repeat, input rows
    # never appear (05 = L24 c64, 06 = L24 c128, 07 = L32, 08 = every MG row).
    def ids(name):
        sheet = book[name]
        return [sheet.cell(row=row, column=1).value
                for row in range(5, sheet.max_row + 1)]

    assert ids("05_L24_c64") == ["ccc333"]
    assert ids("06_L24_c128") == []
    assert ids("07_L32") == []
    assert ids("08_MG") == []
    assert book["04_Performance_All"].freeze_panes == "A5"


def test_workbook_summary_keeps_the_reference_key_order(tmp_path) -> None:
    openpyxl = pytest.importorskip("openpyxl", reason="optional workbook check")
    rows = _tier0_rows() + [_native_mg_row()]
    result = _export(tmp_path, rows, run_options=_run_options())
    book = openpyxl.load_workbook(result.excel_path)
    summary = book["02_Summary"]
    labels = [summary.cell(row=row, column=1).value for row in range(4, 16)]
    assert tuple(labels) == REFERENCE_SUMMARY_LABELS
    values = dict(zip(labels, (summary.cell(row=row, column=2).value
                               for row in range(4, 16))))
    assert values["Rows"] == len(rows)
    assert values["Solve rows"] == len(rows)
    assert values["Prepare/cache rows"] == 0
    assert values["CSV files"] == 1
    assert values["Cache manifests"] == 1
    assert values["Status counts"] == '{"FAIL": 1, "PASS": 4, "SKIP": 1}'


@pytest.mark.skipif(not REFERENCE_XLSX.is_file(), reason="snapshot not present")
def test_summary_labels_are_the_reference_snapshot_labels() -> None:
    """The frozen label tuple above is what the 2026-11-15 workbook contains."""
    openpyxl = pytest.importorskip("openpyxl", reason="optional workbook check")
    book = openpyxl.load_workbook(REFERENCE_XLSX, data_only=True)
    summary = book["02_Summary"]
    labels = tuple(summary.cell(row=row, column=1).value
                   for row in range(4, 16))
    assert labels == REFERENCE_SUMMARY_LABELS


# ---------------------------------------------------------------------------
# 4. write discipline: atomic, deterministic, never the port's ledger
# ---------------------------------------------------------------------------
def test_export_is_deterministic_and_leaves_no_temporary_files(tmp_path) -> None:
    rows = _tier0_rows()
    first = _export(tmp_path, rows, run_options=_run_options())
    csv_bytes = first.csv_path.read_bytes()
    xlsx_bytes = first.excel_path.read_bytes()
    second = _export(tmp_path, rows, run_options=_run_options())
    assert second.csv_path.read_bytes() == csv_bytes
    assert second.excel_path.read_bytes() == xlsx_bytes
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        ledger_export.DEFAULT_CSV_NAME, ledger_export.DEFAULT_EXCEL_NAME]


def test_csv_is_crlf_and_none_is_an_empty_field(tmp_path) -> None:
    """Cell serialisation is the reference's (CRLF, ``""`` for ``None``)."""
    result = _export(tmp_path, [_row("x1", "torch/lattice")],
                     run_options=_run_options())
    text = result.csv_path.read_bytes().decode("utf-8")
    assert text.count("\r\n") == 2            # header + one record
    assert "\n" not in text.replace("\r\n", "")


def test_export_refuses_to_overwrite_the_port_ledger(tmp_path) -> None:
    root = tmp_path / "run"
    payload = _payload(_tier0_rows(), root=root)
    target = Path(payload["paths"]["csv"])
    with pytest.raises(ValueError, match="ROW_FIELDS"):
        ledger_export.export_ledger(payload, csv_path=target, excel=False)
    with pytest.raises(ValueError, match="ROW_FIELDS"):
        ledger_export.export_ledger(
            payload, csv_path=root / "reports" / ".." / "reports" / "rows.csv",
            excel=False)


def test_default_targets_live_next_to_the_run_root(tmp_path) -> None:
    root = tmp_path / "run"
    payload = _payload(_tier0_rows(), root=root)
    csv_path, excel_path = ledger_export.export_targets(payload)
    assert csv_path == root / "ledger" / ledger_export.DEFAULT_CSV_NAME
    assert excel_path == root / "ledger" / ledger_export.DEFAULT_EXCEL_NAME
    result = ledger_export.export_ledger(payload, run_options=_run_options())
    assert result.csv_path == csv_path and result.excel_path == excel_path
    assert csv_path.is_file() and excel_path.is_file()
    # A payload without a run root must ask for a directory instead of writing
    # into the current one.
    with pytest.raises(ValueError, match="run root"):
        ledger_export.export_ledger({"mode": "solve", "rows": {"items": []}})


# ---------------------------------------------------------------------------
# 5. the CLI hook: inert unless the environment asks for it
# ---------------------------------------------------------------------------
def test_export_from_payload_is_inert_by_default(tmp_path, monkeypatch,
                                                 capsys) -> None:
    root = tmp_path / "run"
    payload = _payload(_tier0_rows(), root=root)
    monkeypatch.delenv(ledger_export.EXPORT_ENVIRONMENT, raising=False)
    assert ledger_export.export_from_payload(
        payload, run_options=_run_options()) is None
    assert not (root / "ledger").exists()
    assert capsys.readouterr().err == ""


def test_export_from_payload_writes_what_the_environment_asks_for(
        tmp_path, monkeypatch, capsys) -> None:
    root = tmp_path / "run"
    payload = _payload(_tier0_rows(), root=root)
    monkeypatch.setenv(ledger_export.EXPORT_ENVIRONMENT, "csv")
    csv_only = ledger_export.export_from_payload(
        payload, run_options=_run_options())
    assert csv_only.csv_path.is_file() and csv_only.excel_path is None
    assert not Path(str(csv_only.csv_path).replace(
        ledger_export.DEFAULT_CSV_NAME,
        ledger_export.DEFAULT_EXCEL_NAME)).exists()
    assert "ledger: wrote" in capsys.readouterr().err

    monkeypatch.setenv(ledger_export.EXPORT_ENVIRONMENT, "both")
    monkeypatch.setenv(ledger_export.EXPORT_DIRECTORY_ENVIRONMENT,
                       str(tmp_path / "elsewhere"))
    both = ledger_export.export_from_payload(payload,
                                             run_options=_run_options())
    assert both.csv_path.parent == tmp_path / "elsewhere"
    assert both.excel_path.is_file()
    assert both.rows == len(_tier0_rows())


def test_export_from_payload_reports_instead_of_raising(tmp_path, monkeypatch,
                                                        capsys) -> None:
    payload = _payload(_tier0_rows())
    monkeypatch.setenv(ledger_export.EXPORT_ENVIRONMENT, "both")
    # No run root and no directory: a usage error, reported not raised.
    result = ledger_export.export_from_payload(payload,
                                               run_options=_run_options())
    assert result.error and "run root" in result.error
    assert "ledger: export failed" in capsys.readouterr().err
    # A typo in the environment variable is a *configuration* error: reported on
    # the same status line (so the CLI can exit 2), never a traceback.
    monkeypatch.setenv(ledger_export.EXPORT_ENVIRONMENT, "nonsense")
    result = ledger_export.export_from_payload(payload)
    assert result.error and "nonsense" in result.error
    assert ledger_export.EXPORT_ENVIRONMENT in result.error
    assert "ledger: export failed" in capsys.readouterr().err
    with pytest.raises(ValueError, match=ledger_export.EXPORT_ENVIRONMENT):
        ledger_export.export_from_payload(payload, strict=True)


def test_a_plan_document_is_not_exported(tmp_path) -> None:
    """``--mode list`` rows carry no status: nothing is written unless asked."""
    payload = _payload([_row("p1", "torch/lattice", status=None)],
                       mode="list", root=tmp_path / "run")
    result = ledger_export.export_ledger(payload)
    assert result.rows == 0 and result.csv_path is None
    assert "plan" in result.note
    assert not (tmp_path / "run" / "ledger").exists()
    allowed = ledger_export.export_ledger(payload, allow_plan=True)
    header, records = _read_csv(allowed.csv_path)
    assert header == list(ledger_csv.REFERENCE_CSV_COLUMNS)
    assert records[0]["phase"] == "" and records[0]["status"] == ""


# ---------------------------------------------------------------------------
# 6. a real ``--mode solve`` round trip
# ---------------------------------------------------------------------------
def _run_cli(*arguments: str) -> subprocess.CompletedProcess:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (str(ROOT) if not existing
                                 else str(ROOT) + os.pathsep + existing)
    return subprocess.run(
        [sys.executable, "-B", "-m", "pyqcu.test", *arguments],
        cwd=ROOT, env=environment, text=True, capture_output=True,
        check=False, timeout=1800)


@pytest.mark.skipif(
    pytest.importorskip("importlib.util").find_spec("torch") is None,
    reason="the tier-0 torch case needs torch")
def test_real_full_mode_solve_round_trip(tmp_path) -> None:
    """Export the rows of a real ``--mode solve`` invocation end to end."""
    outdir = tmp_path / "run"
    document = tmp_path / "payload.json"
    completed = _run_cli("--quick", "--only", "torch/lattice", "--mode",
                         "solve", "--outdir", str(outdir),
                         "--json", str(document))
    assert completed.returncode == 0, completed.stderr[-2000:]
    payload = json.loads(document.read_text(encoding="utf-8"))
    items = payload["rows"]["items"]
    assert items, "the solve run produced no row"

    # The port's own ledger CSV must survive the export untouched.
    port_ledger = outdir / "reports" / "rows.csv"
    before = port_ledger.read_bytes()
    ledger_header = before.decode("utf-8").splitlines()[0].split(",")

    result = ledger_export.export_ledger(payload)
    assert result.csv_path == outdir / "ledger" / ledger_export.DEFAULT_CSV_NAME
    header, records = _read_csv(result.csv_path)
    assert tuple(header) == ledger_csv.REFERENCE_CSV_COLUMNS
    assert len(records) == len(items) == result.rows
    assert [record["run_id"] for record in records] == sorted(
        row["run_id"] for row in items)
    assert port_ledger.read_bytes() == before
    assert len(ledger_header) == 30        # ROW_FIELDS, untouched
    record = records[0]
    assert record["status"] == items[0]["status"]
    assert record["dtype"] == items[0]["dtype"]
    assert record["lattice"] == str(items[0]["lattice"])
    assert "tag" not in record        # a ROW_FIELDS-only key never leaks in
    assert "kind" not in record
    assert all(len(row) == 100 for row in records)

    openpyxl = pytest.importorskip("openpyxl", reason="optional workbook check")
    book = openpyxl.load_workbook(result.excel_path)
    assert tuple(book.sheetnames) == ledger_excel.SHEET_NAMES
    sheet = book["04_Performance_All"]
    ids = [sheet.cell(row=row, column=1).value
           for row in range(5, sheet.max_row + 1)]
    assert ids == [record["run_id"] for record in records]


# ---------------------------------------------------------------------------
# 7. ``--ledger-export``: the published CLI hook of the full modes
# ---------------------------------------------------------------------------
def _write_ledger(root: Path, rows) -> Path:
    """Create ``<root>/rows/<run_id>.json`` for ``rows`` (a merge input)."""
    directory = root / "rows"
    directory.mkdir(parents=True, exist_ok=True)
    for row in rows:
        (directory / f"{row['run_id']}.json").write_text(
            json.dumps(row, indent=2), encoding="utf-8")
    return root


def _main(*arguments: str) -> int:
    """Run the CLI in-process and return its exit code."""
    from pyqcu.test.__main__ import main

    return main(list(arguments))


def test_ledger_export_flag_is_documented_and_typed() -> None:
    """``--ledger-export`` takes exactly the exporter's four selections."""
    from pyqcu.test.__main__ import build_parser

    parser = build_parser()
    assert "--ledger-export {none,csv,excel,both}" in parser.format_help()
    assert parser.parse_args(
        ["--mode", "merge"]).ledger_export is None
    assert parser.parse_args(
        ["--mode", "merge", "--ledger-export", "both"]).ledger_export == "both"
    for choice in ledger_export.EXPORT_SELECTIONS:
        assert parser.parse_args(
            ["--mode", "merge", "--ledger-export", choice]
        ).ledger_export == choice
    with pytest.raises(SystemExit):
        parser.parse_args(["--mode", "merge", "--ledger-export", "nonsense"])


def test_cli_merge_writes_the_reference_ledger(tmp_path, capsys) -> None:
    """``--mode merge --ledger-export both`` writes CSV + workbook, keeps rows.csv."""
    from pyqcu.test.common import ROW_FIELDS

    rows = _tier0_rows()
    root = _write_ledger(tmp_path / "run", rows)
    document = tmp_path / "merge.json"
    code = _main("--mode", "merge", "--outdir", str(root),
                 "--ledger-export", "both", "--json", str(document))
    capsys.readouterr()
    assert code == 0

    # 1) The port's own 30-column ledger is unchanged: same header, same rows.
    port_header, port_rows = _read_csv(root / "reports" / "rows.csv")
    assert port_header == list(ROW_FIELDS)
    assert len(port_rows) == len(rows)

    # 2) The reference table is a *second* artifact, under <root>/ledger/.
    csv_path = root / "ledger" / ledger_export.DEFAULT_CSV_NAME
    excel_path = root / "ledger" / ledger_export.DEFAULT_EXCEL_NAME
    assert csv_path.is_file() and excel_path.is_file()
    header, records = _read_csv(csv_path)
    assert header == list(ledger_csv.REFERENCE_CSV_COLUMNS)
    assert len(records) == len(rows)
    by_id = {record["run_id"]: record for record in records}
    assert set(by_id) == {row["run_id"] for row in rows}
    assert by_id["ccc333"]["lattice"] == "[24, 24, 24, 64]"
    assert by_id["ccc333"]["shm"] == "1"
    assert by_id["ccc333"]["rank_x_thread"] == "16x18"
    assert by_id["ddd444"]["status"] == "FAIL"
    assert by_id["eee555"]["status"] == "SKIP"
    assert "skip_token=cuda_bridge" in by_id["eee555"]["notes"]
    # Nothing is fabricated: unobservable columns are empty fields, not "None".
    assert by_id["aaa111"]["node"] == "" and by_id["aaa111"]["atol"] == ""
    for record in records:
        assert all(value not in ("None", "nan", "NaN")
                   for value in record.values()), record["run_id"]
    assert all(len(record) == 100 for record in records)
    # Atomic write: no temporary sibling survives.
    assert sorted(path.name for path in (root / "ledger").iterdir()) == [
        ledger_export.DEFAULT_CSV_NAME, ledger_export.DEFAULT_EXCEL_NAME]

    # 3) The document records what was written (a --json-able block).
    payload = json.loads(document.read_text(encoding="utf-8"))
    block = payload["ledger_export"]
    assert block["csv_path"] == str(csv_path)
    assert block["excel_path"] == str(excel_path)
    assert block["rows"] == len(records)
    assert block["columns"] == list(ledger_csv.REFERENCE_CSV_COLUMNS)
    assert block["sheets"] == list(ledger_excel.SHEET_NAMES)
    assert block["error"] is None and block["note"] == ""

    openpyxl = pytest.importorskip("openpyxl", reason="optional workbook check")
    book = openpyxl.load_workbook(excel_path)
    assert tuple(book.sheetnames) == ledger_excel.SHEET_NAMES
    assert len(book.sheetnames) == 21
    sheet = book["04_Performance_All"]
    assert [sheet.cell(row=row, column=1).value
            for row in range(5, sheet.max_row + 1)] == list(by_id)


def test_cli_ledger_export_passes_the_run_configuration(tmp_path, capsys) -> None:
    """The CLI forwards ``Test.config()``: run-level grid fills a bare row.

    ``--lat``/``grid`` are run options (a merge document has no ``options``
    block), so the export can only know them through ``run_options``.  A row
    that recorded its own ``metrics['lattice_grid']`` keeps it, which is the
    documented precedence.
    """
    root = _write_ledger(tmp_path / "run", _tier0_rows())
    code = _main("--mode", "merge", "--outdir", str(root),
                 "--ledger-export", "csv", "--lattice-grid", "1", "1", "1", "2")
    capsys.readouterr()
    assert code == 0
    _, records = _read_csv(root / "ledger" / ledger_export.DEFAULT_CSV_NAME)
    by_id = {record["run_id"]: record for record in records}
    assert by_id["aaa111"]["grid"] == "[1, 1, 1, 2]"      # the run's grid
    assert by_id["aaa111"]["grid_mode"] == "balanced"
    assert by_id["ccc333"]["grid"] == "[1, 1, 4, 4]"      # the row's own grid


def test_cli_ledger_export_selects_one_artifact(tmp_path, capsys) -> None:
    """``csv``/``excel`` write exactly one file; nothing is written otherwise."""
    root = _write_ledger(tmp_path / "run", _tier0_rows())
    assert _main("--mode", "merge", "--outdir", str(root),
                 "--ledger-export", "csv") == 0
    capsys.readouterr()
    assert (root / "ledger" / ledger_export.DEFAULT_CSV_NAME).is_file()
    assert not (root / "ledger" / ledger_export.DEFAULT_EXCEL_NAME).exists()

    other = _write_ledger(tmp_path / "other", _tier0_rows())
    assert _main("--mode", "merge", "--outdir", str(other),
                 "--ledger-export", "excel") == 0
    capsys.readouterr()
    assert (other / "ledger" / ledger_export.DEFAULT_EXCEL_NAME).is_file()
    assert not (other / "ledger" / ledger_export.DEFAULT_CSV_NAME).exists()


def test_cli_ledger_export_is_off_by_default(tmp_path, capsys,
                                            monkeypatch) -> None:
    """No flag and no environment: an ordinary full mode writes no ledger dir."""
    monkeypatch.delenv(ledger_export.EXPORT_ENVIRONMENT, raising=False)
    root = _write_ledger(tmp_path / "run", _tier0_rows())
    assert _main("--mode", "merge", "--outdir", str(root)) == 0
    assert "ledger:" not in capsys.readouterr().err
    assert not (root / "ledger").exists()

    # The documented environment fallback still works ...
    monkeypatch.setenv(ledger_export.EXPORT_ENVIRONMENT, "csv")
    assert _main("--mode", "merge", "--outdir", str(root)) == 0
    capsys.readouterr()
    assert (root / "ledger" / ledger_export.DEFAULT_CSV_NAME).is_file()

    # ... and the explicit flag wins, including the explicit "none".
    other = _write_ledger(tmp_path / "other", _tier0_rows())
    assert _main("--mode", "merge", "--outdir", str(other),
                 "--ledger-export", "none") == 0
    capsys.readouterr()
    assert not (other / "ledger").exists()


def test_cli_ledger_export_is_rejected_outside_a_full_mode(capsys) -> None:
    """A flag that cannot produce an artifact is a usage error, never a no-op."""
    assert _main("--ledger-export", "csv", "--only", "torch/lattice") == 2
    assert "of a full mode" in capsys.readouterr().err

    assert _main("--mode", "list", "--ledger-export", "csv") == 2
    assert "plan without a status" in capsys.readouterr().err

    assert _main("--list", "--mode", "solve", "--ledger-export", "csv") == 2
    assert "does not execute a full mode" in capsys.readouterr().err

    assert _main("--check", "--ledger-export", "csv") == 2
    assert "--ledger-export" in capsys.readouterr().err


def test_cli_environment_typo_is_a_usage_error(tmp_path, capsys,
                                               monkeypatch) -> None:
    """F2: a typo in ``PYQCU_TEST_LEDGER_EXPORT`` exits 2, not as a traceback.

    The variable is the second gate of the export; an unknown value used to
    escape as an uncaught ``ValueError`` from inside the exporter (exit 1, no
    ``pyqcu.test: error:`` line).  It is validated before the mode runs, so the
    documented usage path applies and no run root is created.
    """
    monkeypatch.setenv(ledger_export.EXPORT_ENVIRONMENT, "bogus")
    root = tmp_path / "run"
    code = _main("--mode", "merge", "--outdir", str(root))
    captured = capsys.readouterr()
    assert code == 2
    assert "Traceback" not in captured.err
    assert captured.err.startswith("pyqcu.test: error: ")
    assert ledger_export.EXPORT_ENVIRONMENT in captured.err
    assert "not a known export selection" in captured.err
    assert not root.exists()          # rejected before the merge wrote anything

    # The flag still wins, including the documented "none" override.
    other = _write_ledger(tmp_path / "other", _tier0_rows())
    assert _main("--mode", "merge", "--outdir", str(other),
                 "--ledger-export", "none") == 0
    capsys.readouterr()
    assert not (other / "ledger").exists()


def test_cli_environment_typo_exits_two_in_a_subprocess(tmp_path,
                                                       monkeypatch) -> None:
    """The process exit code (not only ``main()``) is the documented 2."""
    monkeypatch.setenv(ledger_export.EXPORT_ENVIRONMENT, "bogus")
    completed = _run_cli("--mode", "merge", "--outdir", str(tmp_path / "run"))
    assert completed.returncode == 2, completed.stderr[-2000:]
    assert completed.stdout == ""
    assert "Traceback" not in completed.stderr
    assert completed.stderr.startswith("pyqcu.test: error: ")
    assert "not a known export selection" in completed.stderr
    assert not (tmp_path / "run").exists()


@pytest.mark.skipif(
    pytest.importorskip("importlib.util").find_spec("torch") is None,
    reason="the tier-0 torch case needs torch")
def test_real_full_mode_solve_with_the_cli_flag(tmp_path) -> None:
    """A real ``--mode solve`` writes the reference CSV when the flag asks."""
    outdir = tmp_path / "run"
    document = tmp_path / "payload.json"
    completed = _run_cli("--quick", "--only", "torch/lattice", "--mode",
                         "solve", "--outdir", str(outdir),
                         "--ledger-export", "csv", "--json", str(document))
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "ledger: wrote" in completed.stderr

    payload = json.loads(document.read_text(encoding="utf-8"))
    items = payload["rows"]["items"]
    block = payload["ledger_export"]
    assert block["csv_path"] == str(
        outdir / "ledger" / ledger_export.DEFAULT_CSV_NAME)
    assert block["excel_path"] is None
    header, records = _read_csv(Path(block["csv_path"]))
    assert tuple(header) == ledger_csv.REFERENCE_CSV_COLUMNS
    assert len(records) == len(items) == 1
    record = records[0]
    assert record["phase"] == "solve"
    assert record["status"] == items[0]["status"]
    assert record["lattice"] == str(items[0]["lattice"])
    assert record["grid"] == "[1, 1, 1, 1]"          # Test.config() default
    # The port's own ledger keeps its ROW_FIELDS header and row count.
    port_header, port_rows = _read_csv(outdir / "reports" / "rows.csv")
    assert len(port_header) == 30 and len(port_rows) == 1
