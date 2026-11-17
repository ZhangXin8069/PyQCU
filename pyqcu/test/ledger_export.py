"""One-call reference-parity export of a full-mode ledger (CSV + workbook).

The two contracts of the informal KUPL-Kunpeng driver are implemented by
:mod:`pyqcu.test.ledger_csv` (the 100-column ``CSV_COLUMNS`` table) and
:mod:`pyqcu.test.ledger_excel` (the 21-sheet
``kupl_kunpeng_performance_20261115.xlsx`` workbook).  This module is the
*integration point* between them and a real run: it takes what a full mode
already produces - the ``--json`` document (``payload``) and/or the
``rows/<run_id>.json`` items - and writes both artifacts, atomically, into one
directory.

Why a separate module
---------------------
The CLI has exactly one published ledger CSV: ``reports/rows.csv``
(:data:`pyqcu.test.common.ROW_FIELDS`, 29 columns).  The reference CSV has 100
columns in a different order, so it must never be written over that file -
``--mode merge``/``--mode validate`` and the user's own ``awk``/pandas scripts
read it.  The export therefore lives in its own directory
(``<run root>/ledger/``) under the reference artifact names
(:data:`DEFAULT_CSV_NAME`, :data:`DEFAULT_EXCEL_NAME`), and
:func:`export_ledger` refuses a target that is the run's own ``rows.csv``.

What the exporter passes through
--------------------------------
* ``run_options`` - the run configuration (``Test.config()``).  The payload's
  ``options`` block describes the *mode* (paths, ranks, shm, inputs); the
  reference columns ``lattice``/``grid``/``mg_grid``/``mg_levels``/
  ``mg_restart``/``max_iter`` describe the *run*, and a ``--mode merge``
  document does not repeat them.  Pass it whenever it is available - the CLI
  hook does (see :func:`export_from_payload`).
* ``rows`` - the ledger rows to export (default: ``payload["rows"]["items"]``),
  so an exporter can be driven from ``rows/*.json`` alone.
* ``cache_manifest`` - an already-read Legacy-MG manifest
  (:func:`pyqcu.test.cache.read_manifest`); it is never opened here.
* ``hash_files`` - opt-in SHA256 of the report/solution/manifest files.  It is
  off by default because it makes the export read the disk and because a
  content hash of a file the run just wrote is not a measurement.

Import rules: standard library only (``csv``, ``os``, ``sys``, ``pathlib``) plus
:mod:`pyqcu.test.ledger_csv` and :mod:`pyqcu.test.ledger_excel`.  No backend
import, no ``openpyxl`` at run time.

Programmatic use::

    from pyqcu.test import ledger_export

    result = ledger_export.export_ledger(payload, run_options=test.config())
    result.csv_path, result.excel_path, result.rows

    # rows only (a merged ledger, a saved document, a test fixture):
    ledger_export.export_ledger({"mode": "merge"}, rows=rows,
                                run_options=test.config(),
                                directory="out/ledger")

CLI integration
---------------
``python -m pyqcu.test --mode <mode> --ledger-export {none,csv,excel,both}``
is the published hook: ``pyqcu.test.__main__._run_mode`` calls
:func:`export_from_payload` with ``selection=`` the flag value after the mode
document is complete, so ``--mode solve|run|merge|prepare|prewarm`` can write
the reference artifacts without touching ``reports/rows.csv``.
``--mode list`` (a plan has no status) and ``--mode validate`` (its rows were
produced by solve) are rejected by the CLI; the Python API still exports both
when asked explicitly.  Without the flag the export stays off unless
``PYQCU_TEST_LEDGER_EXPORT`` asks for it (``csv``, ``excel``/``xlsx``,
``1``/``both``/``all``; unset or ``0``/``off`` disables), and
``PYQCU_TEST_LEDGER_EXPORT_DIR`` overrides the target directory.  The exporter
is also reachable from Python and from any
``python -m pyqcu.test ... --json`` document.
"""

from __future__ import annotations

import csv
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import ledger_csv, ledger_excel

__all__ = [
    "DEFAULT_CSV_NAME",
    "DEFAULT_EXCEL_NAME",
    "EXPORT_DIRECTORY",
    "EXPORT_DIRECTORY_ENVIRONMENT",
    "EXPORT_ENVIRONMENT",
    "EXPORT_SELECTIONS",
    "FORMATS",
    "LedgerExport",
    "export_formats",
    "export_from_payload",
    "export_ledger",
    "export_targets",
]

#: Subdirectory of the run root the export is written to.  A separate directory
#: keeps ``reports/rows.csv`` (the port's own 29-column ledger) untouched.
EXPORT_DIRECTORY = "ledger"

#: Reference artifact names.  ``kupl_kunpeng_all.csv`` is the informal driver's
#: ``csv_path()`` default and ``kupl_kunpeng_performance.xlsx`` is the workbook
#: family of the 2026-11-15 deliverable, so a downstream script that was pointed
#: at the reference names keeps working after a one-directory change.
DEFAULT_CSV_NAME = "kupl_kunpeng_all.csv"
DEFAULT_EXCEL_NAME = "kupl_kunpeng_performance.xlsx"

#: Environment variable that turns the CLI export on (see the module docstring).
EXPORT_ENVIRONMENT = "PYQCU_TEST_LEDGER_EXPORT"

#: Environment variable that overrides the export directory.
EXPORT_DIRECTORY_ENVIRONMENT = "PYQCU_TEST_LEDGER_EXPORT_DIR"

#: Accepted values of :data:`EXPORT_ENVIRONMENT` and the formats they select.
FORMATS: Mapping[str, tuple[bool, bool]] = {
    "": (False, False),
    "0": (False, False),
    "off": (False, False),
    "no": (False, False),
    "false": (False, False),
    "none": (False, False),
    "1": (True, True),
    "on": (True, True),
    "yes": (True, True),
    "true": (True, True),
    "all": (True, True),
    "both": (True, True),
    "csv": (True, False),
    "excel": (False, True),
    "xlsx": (False, True),
    "workbook": (False, True),
}

#: The four spellings of the ``--ledger-export`` CLI flag, in help order.  The
#: flag is deliberately narrower than :data:`FORMATS` (which also accepts the
#: on/off spellings of the environment variable): a published CLI option should
#: read as a selection, not as a boolean soup.  ``none`` is an explicit "off",
#: so ``--ledger-export none`` overrides a ``PYQCU_TEST_LEDGER_EXPORT`` that
#: asks for an artifact.
EXPORT_SELECTIONS: tuple[str, ...] = ("none", "csv", "excel", "both")


# ---------------------------------------------------------------------------
# result
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LedgerExport:
    """What one export wrote, and what it was written from.

    ``columns`` is the header of the CSV that is on disk (the reference 100
    names, plus any legacy-only column :func:`ledger_csv.upgrade_columns`
    preserved); ``sheets`` is the workbook's sheet order.  ``note`` carries the
    reason a document was *not* exported (a plan, an empty ledger) and ``error``
    the message of a failed one, so a caller never has to parse a log line.
    """

    csv_path: Path | None = None
    excel_path: Path | None = None
    rows: int = 0
    run_ids: tuple[str, ...] = ()
    columns: tuple[str, ...] = ()
    sheets: tuple[str, ...] = ()
    mode: str = ""
    note: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True when at least one artifact was written without an error."""
        return self.error is None and (self.csv_path is not None
                                       or self.excel_path is not None)

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe description (``--json``-able, no ``Path`` objects)."""
        return {
            "csv_path": None if self.csv_path is None else str(self.csv_path),
            "excel_path": (None if self.excel_path is None
                           else str(self.excel_path)),
            "rows": int(self.rows),
            "run_ids": list(self.run_ids),
            "columns": list(self.columns),
            "sheets": list(self.sheets),
            "mode": self.mode,
            "note": self.note,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# formats / targets
# ---------------------------------------------------------------------------
def export_formats(value: Any) -> tuple[bool, bool]:
    """``(csv, excel)`` selected by an :data:`EXPORT_ENVIRONMENT` value.

    Unset and the usual off-spellings disable the export; anything else that is
    not in :data:`FORMATS` raises :class:`ValueError` - a typo in an environment
    variable must not silently produce a partial export.
    """
    text = "" if value is None else str(value).strip().lower()
    if text in FORMATS:
        return FORMATS[text]
    spellings = ", ".join(sorted(name for name in FORMATS if name))
    raise ValueError(
        f"{EXPORT_ENVIRONMENT}={value!r} is not a known export selection; "
        f"use one of {spellings} (unset or 0/off disables the export)")


def _document_root(payload: Mapping[str, Any]) -> Path | None:
    """The run root a payload declares (``paths.root``, else ``rows.dir``)."""
    paths = payload.get("paths")
    if isinstance(paths, Mapping) and paths.get("root"):
        return Path(str(paths["root"]))
    rows = payload.get("rows")
    if isinstance(rows, Mapping) and rows.get("dir"):
        return Path(str(rows["dir"])).parent
    options = payload.get("options")
    if isinstance(options, Mapping):
        nested = options.get("paths")
        if isinstance(nested, Mapping) and nested.get("root"):
            return Path(str(nested["root"]))
    return None


def export_targets(payload: Mapping[str, Any], *,
                   directory: Any = None) -> tuple[Path, Path]:
    """Default ``(csv_path, excel_path)`` of one full-mode document.

    ``directory`` wins when given; otherwise ``<run root>/ledger`` is used.  A
    payload that carries no ``paths.root`` (a hand-built document, a bare
    ``{"mode": ...}``) raises :class:`ValueError` instead of writing into the
    current directory - an export is never a surprise file next to the caller.
    """
    if directory is not None:
        base = Path(str(directory))
    else:
        root = _document_root(payload)
        if root is None:
            raise ValueError(
                "the payload declares no run root (paths.root); pass "
                "directory= to choose where the CSV and workbook are written")
        base = root / EXPORT_DIRECTORY
    return base / DEFAULT_CSV_NAME, base / DEFAULT_EXCEL_NAME


def _port_ledger_paths(payload: Mapping[str, Any]) -> set[str]:
    """Absolute paths of the port's own ledger CSVs (never overwritten)."""
    found: set[str] = set()
    candidates: list[Any] = []
    paths = payload.get("paths")
    if isinstance(paths, Mapping):
        candidates.append(paths.get("csv"))
    rows = payload.get("rows")
    if isinstance(rows, Mapping):
        candidates.append(rows.get("csv"))
    options = payload.get("options")
    if isinstance(options, Mapping) and isinstance(options.get("paths"),
                                                    Mapping):
        candidates.append(options["paths"].get("csv"))
    for candidate in candidates:
        if candidate:
            found.add(str(Path(str(candidate)).resolve()))
    return found


def _csv_header(path: Path) -> tuple[str, ...]:
    """Header of a written CSV (the exact columns the file carries)."""
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return tuple(next(csv.reader(handle), ()))
    except (OSError, csv.Error, UnicodeDecodeError):
        return ()


# ---------------------------------------------------------------------------
# the export
# ---------------------------------------------------------------------------
def export_ledger(payload: Mapping[str, Any], *,
                  rows: Sequence[Mapping[str, Any]] | None = None,
                  run_options: Mapping[str, Any] | None = None,
                  directory: Any = None,
                  csv_path: Any = None,
                  excel_path: Any = None,
                  csv: bool = True,
                  excel: bool = True,
                  cache_manifest: Mapping[str, Any] | None = None,
                  hash_files: bool = False,
                  allow_plan: bool = False) -> LedgerExport:
    """Write the reference CSV and workbook for ``payload``; return the result.

    ``payload`` is the document a full mode returns (``--json``) - ``solve``,
    ``run``, ``merge``, ``validate``, ``prepare`` or ``prewarm``.  ``rows``
    overrides ``payload["rows"]["items"]``, which is how an export is built from
    ``rows/<run_id>.json`` files alone.  ``run_options`` should be
    ``Test.config()`` (see the module docstring).  ``csv``/``excel`` select the
    artifacts; ``csv_path``/``excel_path``/``directory`` override the targets.

    A ``list`` document is a *plan*: its rows carry no status and the reference
    has no "planned" phase, so nothing is written unless ``allow_plan=True``.
    Both writes are atomic (:func:`ledger_csv.write_ledger_csv`,
    :func:`ledger_excel.write_ledger_excel`): a reader sees the previous file or
    the complete new one, never a partial export.
    """
    mode = "" if payload.get("mode") is None else str(payload.get("mode"))
    document_rows = rows
    if document_rows is None:
        block = payload.get("rows")
        items = block.get("items") if isinstance(block, Mapping) else None
        document_rows = [item for item in (items or ())
                         if isinstance(item, Mapping)]
    if mode == "list" and not allow_plan:
        return LedgerExport(mode=mode,
                            note="--mode list is a plan (no status, no phase); "
                                 "pass allow_plan=True to export it anyway")
    default_csv: Path | None = None
    default_excel: Path | None = None
    if directory is not None or _document_root(payload) is not None:
        default_csv, default_excel = export_targets(payload, directory=directory)
    target_csv = (None if not csv else
                  Path(str(csv_path)) if csv_path is not None else default_csv)
    target_excel = (None if not excel else
                    Path(str(excel_path)) if excel_path is not None
                    else default_excel)
    if csv and target_csv is None:
        raise ValueError(
            "no CSV target: the payload declares no run root (paths.root); "
            "pass csv_path= or directory=")
    if excel and target_excel is None:
        raise ValueError(
            "no workbook target: the payload declares no run root "
            "(paths.root); pass excel_path= or directory=")
    protected = _port_ledger_paths(payload)
    if target_csv is not None and str(target_csv.resolve()) in protected:
        raise ValueError(
            f"{target_csv} is the port's own ledger CSV "
            "(pyqcu.test.common.ROW_FIELDS); the reference 100-column table "
            "needs its own path (default: <run root>/ledger/)")
    records = ledger_csv.ledger_records(payload, rows=document_rows,
                                       run_options=run_options,
                                       cache_manifest=cache_manifest,
                                       hash_files=hash_files)
    if not records:
        return LedgerExport(mode=mode, rows=0,
                            note="no ledger row to export")
    written_csv: Path | None = None
    if target_csv is not None:
        written_csv = ledger_csv.write_ledger_csv(target_csv, records)
    written_excel: Path | None = None
    if target_excel is not None:
        written_excel = ledger_excel.write_ledger_excel(
            target_excel, records, payload=payload, csv_path=written_csv)
    return LedgerExport(
        csv_path=written_csv,
        excel_path=written_excel,
        rows=len(records),
        run_ids=tuple(str(record.get("run_id")) for record in records),
        columns=_csv_header(written_csv) if written_csv is not None else (),
        sheets=tuple(ledger_excel.SHEET_NAMES) if written_excel else (),
        mode=mode,
    )


# ---------------------------------------------------------------------------
# the CLI hook
# ---------------------------------------------------------------------------
def export_from_payload(payload: Mapping[str, Any], *,
                        run_options: Mapping[str, Any] | None = None,
                        directory: Any = None,
                        environ: Mapping[str, str] | None = None,
                        stream: Any = None,
                        strict: bool = False,
                        selection: str | None = None) -> LedgerExport | None:
    """Environment-gated :func:`export_ledger` - the one-line CLI hook.

    Returns ``None`` when the export is disabled (the default), so a run that
    does not ask for the parity artifacts is byte-for-byte unaffected.  When
    enabled it writes the CSV and/or the workbook, prints one status line to
    ``stream`` (``stderr`` by default) and returns the :class:`LedgerExport`.
    A write failure is reported on that line and in ``result.error`` instead of
    raising - a finished run must not turn into a traceback because an export
    directory is read-only.  A typo in :data:`EXPORT_ENVIRONMENT` (or an unknown
    ``selection``) is reported the same way, so the CLI can turn it into the
    documented usage error (exit code 2) instead of a traceback.
    ``strict=True`` re-raises instead of reporting, for a caller that wants the
    exception.

    ``selection`` is the explicit form of the gate: the value of the
    ``--ledger-export {none,csv,excel,both}`` CLI flag.  When it is given it
    wins over :data:`EXPORT_ENVIRONMENT` (so ``--ledger-export none`` really
    disables an export the environment would otherwise turn on);
    :data:`EXPORT_DIRECTORY_ENVIRONMENT` still applies, because a directory
    override is not a format selection.
    """
    environment = os.environ if environ is None else environ
    stream = sys.stderr if stream is None else stream
    mode = "" if payload.get("mode") is None else str(payload.get("mode"))
    try:
        wanted = export_formats(
            environment.get(EXPORT_ENVIRONMENT, "")
            if selection is None else selection)
    except ValueError as error:
        # A typo in ``PYQCU_TEST_LEDGER_EXPORT`` (or in ``selection``) is a
        # configuration error, not a crash: report it on the same status line as
        # a failed write, so the CLI turns it into its documented usage error
        # (exit code 2) instead of a traceback from inside the exporter.
        if strict:
            raise
        result = LedgerExport(mode=mode,
                              error=f"{type(error).__name__}: {error}")
        print(_status_line(result), file=stream, flush=True)
        return result
    if wanted == (False, False):
        return None
    chosen = None if directory is not None else \
        environment.get(EXPORT_DIRECTORY_ENVIRONMENT) or None
    try:
        result = export_ledger(payload, run_options=run_options,
                               directory=chosen, csv=wanted[0],
                               excel=wanted[1])
    except Exception as error:  # noqa: BLE001 - reported, never a traceback
        if strict:
            raise
        result = LedgerExport(mode=mode,
                              error=f"{type(error).__name__}: {error}")
    print(_status_line(result), file=stream, flush=True)
    return result


def _status_line(result: LedgerExport | None) -> str:
    """One-line human status of an environment-gated export."""
    if result is None:
        return "ledger: export disabled"
    if result.error:
        return f"ledger: export failed ({result.error})"
    if result.csv_path is None and result.excel_path is None:
        return f"ledger: nothing written ({result.note or 'no row'})"
    paths = [str(path) for path in (result.csv_path, result.excel_path)
             if path is not None]
    return f"ledger: wrote {', '.join(paths)} ({result.rows} row(s))"
