"""Reference-compatible Excel workbook for the formal :mod:`pyqcu.test` port.

The 2026-11-15 deliverable ``kupl_kunpeng_performance_20261115.xlsx`` (21 sheets,
generated with openpyxl 3.1.5) is the human-facing view of the informal driver's
CSV ledger.  This module writes the same workbook **without a third-party
dependency**: a small SpreadsheetML (``.xlsx`` = ZIP + XML) writer reproduces the
reference's sheet set, headers, row order, column formats, table objects,
conditional status fills, hyperlinks and the two summary charts.

The reference workbook, as measured from its own XML
----------------------------------------------------
================  ==========================================================
sheets            21, in the fixed order of :data:`SHEET_NAMES`
layout            row 1 title (merged A1:L1), row 2 subtitle (merged A2:L2),
                  row 3 empty, row 4 header, data from row 5
view              ``showGridLines="0"``, ``zoomScale="90"``, frozen ``A5``
title             Arial 18 bold ``#1F4E78``, row height 28
subtitle          Arial 10 italic ``#595959``, wrapped, row height 32
table header      Arial 9 bold white on ``#1F4E78``, centred, wrapped, thin
                  ``#BFBFBF`` borders, row height 30
data cell         Arial 9, wrapped, top-aligned, thin borders
key/value block   column A = Arial 9 bold on ``#D9E2F3``; column B = plain
status colours    ``PASS`` ``#E2EFDA`` / ``FAIL`` ``#FCE4D6`` / ``SKIP``
                  ``#FFF2CC`` (a ``dxf`` conditional format on the status
                  column of every data sheet)
numbers           ``0.000000`` (``wall_seconds``), ``0.000`` (other seconds),
                  ``0.000E+00`` (residuals), ``#,##0`` (byte counts);
                  every float is stored at **16 significant digits**
                  (``float(f"{value:.16g}")``) - measured on all 229
                  non-round-trip cells of the reference
tables            a ListObject per data sheet (``tbl_perf_all`` ...),
                  ``TableStyleMedium2`` with row stripes, header row = row 4
columns of 04     44 curated columns (:data:`ledger_csv.PERFORMANCE_COLUMNS`)
columns of 05-08  27 curated columns (:data:`MG_SHEET_COLUMNS`)
charts            2 clustered bar charts on ``02_Summary`` reading the count
                  blocks directly below the chart
document props    ``dc:creator`` ``PyQCU / CommandCode``, a title/subject and
                  a description; ``Application`` "Openpyxl 3.1.5"
================  ==========================================================

What the exporter can and cannot derive
---------------------------------------
The reference workbook mixes four kinds of content:

* **data sheets** (``04``-``08``, ``11``, ``17``, ``18``, ``19``) - pure views of
  the ledger, reproduced exactly from :mod:`pyqcu.test.ledger_csv` records;
* **computed summaries** (``02`` counts, ``03`` coverage, ``12`` failures,
  ``14`` input integrity, ``16`` script identity) - derived from the same
  records and the run's own files;
* **editorial sheets** (``00`` narrative, ``09`` startup timings, ``10`` external
  bandwidth definitions, ``13`` abandoned axes, ``15`` historical numbers,
  ``20`` scheduler ledger) - they quote measurements and decisions that exist
  only in the informal HPCKit tree.  They are emitted with their reference
  header and an explicit ``n/a`` row naming what is missing: a blank would look
  like "no failures", and an invented value would be a lie;
* **the 100-column data dictionary** (``17``) - generated from
  :data:`pyqcu.test.ledger_csv.COLUMN_RULES`, so it documents *this* exporter's
  real provenance instead of the reference's prose.

Determinism and atomicity
-------------------------
Two exports of the same records are byte-identical: no timestamps, no
environment-derived values, rows sorted by ``run_id``, sheet order fixed, ZIP
entries written with a constant timestamp and stored (uncompressed) in a fixed
order.  The file is written to a sibling temporary and moved into place with
:func:`os.replace`, so a reader never sees a half-written workbook.

Programmatic use::

    from pyqcu.test import ledger_csv, ledger_excel

    records = ledger_csv.ledger_records(payload)
    ledger_excel.write_ledger_excel("kupl_kunpeng_performance.xlsx", records)

Import rules: standard library only (``hashlib``, ``os``, ``zipfile``,
``xml.sax.saxutils``) plus :mod:`pyqcu.test.ledger_csv` and
:mod:`pyqcu.test.policy`.  No ``openpyxl`` at run time and no backend import;
the tests use ``openpyxl`` when it happens to be installed to cross-check the
written package, and skip that check when it is not.
"""

from __future__ import annotations

import hashlib
import os
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from . import ledger_csv
from .ledger_csv import PERFORMANCE_COLUMNS, REFERENCE_CSV_COLUMNS, column_note

__all__ = [
    "CHART_SUMMARY",
    "COLUMN_NUMBER_FORMATS",
    "EDITORIAL_SHEETS",
    "MG_SHEET_COLUMNS",
    "NUMBER_FORMATS",
    "REFERENCE_STYLE",
    "SHEET_NAMES",
    "Block",
    "Cell",
    "ChartSpec",
    "Sheet",
    "build_sheets",
    "excel_value",
    "summary_counts",
    "write_ledger_excel",
    "write_workbook",
]

#: The reference workbook's 21 sheets, in order.  The names are the contract
#: (they are what ``01_Index`` links to and what a reader filters on).
SHEET_NAMES: tuple[str, ...] = (
    "00_Cover",
    "01_Index",
    "02_Summary",
    "03_Matrix",
    "04_Performance_All",
    "05_L24_c64",
    "06_L24_c128",
    "07_L32",
    "08_MG",
    "09_Startup_Loop",
    "10_Bandwidth",
    "11_Cache",
    "12_Failures",
    "13_Abandoned",
    "14_Input_Integrity",
    "15_Historical",
    "16_Scripts",
    "17_Data_Dictionary",
    "18_CSV_Index",
    "19_Reports_Index",
    "20_Job_Ledger",
)

#: The 27 curated columns of the reference's per-lattice sheets (``05``-``08``).
MG_SHEET_COLUMNS: tuple[str, ...] = (
    "run_id", "status", "dtype", "lattice", "solver", "grid_mode", "grid",
    "rank_x_thread", "shm", "source_kind", "gauge_kind", "seed", "mg_restart",
    "atol", "wall_seconds", "builder_seconds", "build_mg_cache_seconds",
    "iterations", "final_true_residual", "verify_status", "node", "freq_khz",
    "freq_max_khz", "gauge_read_seconds", "elapsed_seconds", "cache_kind",
    "notes",
)

#: ``solver`` values of an MG row: ``mg1``/``mg2`` are the reference driver's
#: 1-/2-level configurations, ``mg`` is the formal port's extension for its
#: native Legacy-MG runner.
MG_SOLVERS: tuple[str, ...] = ("mg", "mg1", "mg2")

#: Sheets whose content cannot be derived from a :mod:`pyqcu.test` ledger.  They
#: are still emitted (a missing sheet breaks ``01_Index`` and the reader's mental
#: map) but contain an explicit ``n/a`` row instead of invented values.
EDITORIAL_SHEETS: tuple[str, ...] = (
    "09_Startup_Loop", "10_Bandwidth", "13_Abandoned", "15_Historical",
    "20_Job_Ledger",
)

#: Sheet-tab colours of the reference workbook, by role.
TAB_COLORS: Mapping[str, str] = {
    "00_Cover": "1F4E78",
    "01_Index": "2F5597",
    "02_Summary": "548235",
    "03_Matrix": "BF8F00",
    "04_Performance_All": "4472C4",
    "05_L24_c64": "70AD47",
    "06_L24_c128": "70AD47",
    "07_L32": "70AD47",
    "08_MG": "70AD47",
    "09_Startup_Loop": "FFC000",
    "10_Bandwidth": "7030A0",
    "11_Cache": "C00000",
    "12_Failures": "A50021",
    "13_Abandoned": "7F6000",
    "14_Input_Integrity": "833C0B",
    "15_Historical": "595959",
    "16_Scripts": "2F5597",
    "17_Data_Dictionary": "2F5597",
    "18_CSV_Index": "2F5597",
    "19_Reports_Index": "2F5597",
    "20_Job_Ledger": "7030A0",
}

#: Custom number formats, keyed by the name the column map uses.
NUMBER_FORMATS: Mapping[str, str] = {
    "seconds6": "0.000000",
    "seconds3": "0.000",
    "sci": "0.000E+00",
    "int": "#,##0",
}

#: Column -> number-format name.  Measured from the reference's own styles:
#: ``wall_seconds`` uses six decimals, the other durations three, residuals
#: scientific notation, byte counts a thousands separator; everything else is
#: ``General``.
COLUMN_NUMBER_FORMATS: Mapping[str, str] = {
    "wall_seconds": "seconds6",
    "seconds": "seconds3",
    "builder_seconds": "seconds3",
    "build_mg_cache_seconds": "seconds3",
    "gauge_read_seconds": "seconds3",
    "source_read_seconds": "seconds3",
    "elapsed_seconds": "seconds3",
    "gauge_prewarm_seconds": "seconds3",
    "source_prewarm_seconds": "seconds3",
    "setup_seconds": "seconds3",
    "final_true_residual": "sci",
    "relative_full_residual": "sci",
    "distributed_relative_residual": "sci",
    "size_bytes": "int",
    "file_size_bytes": "int",
    "rows": "int",
    "bytes": "int",
}

#: Reference document properties.
REFERENCE_STYLE: Mapping[str, str] = {
    "creator": "PyQCU / CommandCode",
    "application": "PyQCU ledger_excel (stdlib SpreadsheetML writer)",
    "title": "KUPL-QCU Kunpeng performance data",
    "subject": "Matrix runs, annotations, failures, caches, scripts",
    "description": ("Generated from pyqcu.test full-mode rows and reports; "
                    "no data deleted."),
}

#: Booleans a data sheet uses for its status column conditional format.
STATUS_FILLS: Mapping[str, str] = {
    "PASS": "E2EFDA",
    "FAIL": "FCE4D6",
    "SKIP": "FFF2CC",
}


def excel_value(value: Any) -> Any:
    """One cell value in the reference workbook's numeric convention.

    Floats are stored at 16 significant digits - the reference's own rule,
    verified against all 229 cells that did not round-trip exactly - so a cell
    and its CSV text agree to the last digit a spreadsheet can hold.  Lists and
    dicts become their Python ``repr`` (``[24, 24, 24, 64]``), the same text the
    CSV column carries.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return str(value)
        return float(f"{value:.16g}")
    if isinstance(value, (list, tuple, dict)):
        return ledger_csv.csv_cell(value)
    return value


# ---------------------------------------------------------------------------
# sheet model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Cell:
    """One written cell: a value plus the style role it is rendered with."""

    value: Any = None
    kind: str = "data"
    numfmt: str = ""

    def with_numfmt(self, numfmt: str) -> "Cell":
        return Cell(self.value, self.kind, numfmt or self.numfmt)


@dataclass(frozen=True)
class ChartSpec:
    """One clustered bar chart referencing cells of its own sheet.

    ``cats``/``vals``/``series_name`` are absolute A1 references (including the
    quoted sheet name), so the layout code decides them after the blocks have
    been placed.  ``anchor_row``/``anchor_col`` are 0-based like the drawing
    anchor the reference uses.
    """

    title: str
    series_name: str
    cats: str
    vals: str
    anchor_col: int = 3
    anchor_row: int = 29
    cx: int = 5040000
    cy: int = 2520000


@dataclass
class Block:
    """One run of rows inside a sheet, separated from the previous by blanks."""

    rows: list[list[Cell | None]]
    gap_before: int = 1
    table: str | None = None
    header: bool = True
    links: bool = False


@dataclass
class Sheet:
    """One worksheet: title, subtitle, column widths and a list of blocks."""

    name: str
    title: str
    subtitle: str
    widths: Mapping[str, float] = field(default_factory=dict)
    blocks: list[Block] = field(default_factory=list)
    tab_color: str = "1F4E78"
    status_column: str | None = None
    charts: list[ChartSpec] = field(default_factory=list)
    merge_columns: int = 12

    @property
    def header_names(self) -> tuple[str, ...]:
        """Header row of the first table block (empty for a fact sheet)."""
        if not self.blocks or not self.blocks[0].header:
            return ()
        first = self.blocks[0].rows[0]
        return tuple(str(cell.value) for cell in first if cell is not None)


# ---------------------------------------------------------------------------
# styles
# ---------------------------------------------------------------------------
_BASE_FONT = ('<font><name val="Calibri"/><family val="2"/><color theme="1"/>'
              '<sz val="11"/><scheme val="minor"/></font>')
_FONTS: tuple[str, ...] = (
    _BASE_FONT,
    '<font><name val="Arial"/><b val="1"/><color rgb="001F4E78"/><sz val="18"/></font>',
    '<font><name val="Arial"/><i val="1"/><color rgb="00595959"/><sz val="10"/></font>',
    '<font><name val="Arial"/><b val="1"/><sz val="9"/></font>',
    '<font><name val="Arial"/><sz val="9"/></font>',
    '<font><name val="Arial"/><b val="1"/><color rgb="00C00000"/><sz val="9"/></font>',
    '<font><name val="Arial"/><b val="1"/><color rgb="00FFFFFF"/><sz val="9"/></font>',
    '<font><name val="Arial"/><color rgb="000563C1"/><sz val="9"/><u val="single"/></font>',
)

_FILLS: tuple[str, ...] = (
    '<fill><patternFill/></fill>',
    '<fill><patternFill patternType="gray125"/></fill>',
    '<fill><patternFill patternType="solid"><fgColor rgb="00D9E2F3"/></patternFill></fill>',
    '<fill><patternFill patternType="solid"><fgColor rgb="00FFF2CC"/></patternFill></fill>',
    '<fill><patternFill patternType="solid"><fgColor rgb="001F4E78"/></patternFill></fill>',
    '<fill><patternFill patternType="solid"><fgColor rgb="00FCE4D6"/></patternFill></fill>',
)

_BORDERS: tuple[str, ...] = (
    '<border><left/><right/><top/><bottom/><diagonal/></border>',
    '<border><left style="thin"><color rgb="00BFBFBF"/></left>'
    '<right style="thin"><color rgb="00BFBFBF"/></right>'
    '<top style="thin"><color rgb="00BFBFBF"/></top>'
    '<bottom style="thin"><color rgb="00BFBFBF"/></bottom><diagonal/></border>',
)

#: ``(font, fill, border, halign, valign, wrap)`` per style role.
_ROLES: Mapping[str, tuple[int, int, int, str | None, str | None, bool]] = {
    "title": (1, 0, 0, None, "center", False),
    "subtitle": (2, 0, 0, None, "top", True),
    "header": (6, 4, 1, "center", "center", True),
    "kv_label": (3, 2, 1, None, "center", False),
    "kv_value": (4, 0, 1, None, "top", True),
    "data": (4, 0, 1, None, "top", True),
    "data_nowrap": (4, 0, 1, None, "top", False),
    "data_bold": (3, 0, 1, None, "top", True),
    "warning": (5, 3, 1, None, "top", True),
    "fail": (4, 5, 1, None, "top", True),
    "link": (7, 0, 1, None, "top", True),
    "note": (2, 0, 0, None, "top", True),
}

#: Built-in number-format ids (no custom entry needed).
_BUILTIN_FORMATS = {"General": 0, "#,##0": 3}


class _StyleBook:
    """Builds ``xl/styles.xml`` and hands out style indexes.

    Indexes are content-addressed, so the same role/number-format pair always
    gets the same ``s=`` attribute and the generated file does not depend on the
    order cells were visited.
    """

    def __init__(self) -> None:
        self._xfs: list[tuple[int, int, int, int, str | None, str | None, bool]] = []
        self._index: dict[Any, int] = {}
        self._custom: dict[str, int] = {}
        # cellXfs index 0 must be the default style.
        self.style("data_nowrap")
        self.style("data")

    def _numfmt_id(self, code: str) -> int:
        if not code:
            return 0
        if code in _BUILTIN_FORMATS:
            return _BUILTIN_FORMATS[code]
        if code not in self._custom:
            self._custom[code] = 164 + len(self._custom)
        return self._custom[code]

    def style(self, kind: str, numfmt: str = "") -> int:
        """Style index for ``kind`` with an optional custom number format."""
        role = _ROLES.get(kind, _ROLES["data"])
        key = (kind, numfmt)
        if key in self._index:
            return self._index[key]
        if not self._xfs:
            self._xfs.append((0, 0, 0, 0, None, None, False))
            self._index[("default", "")] = 0
        font, fill, border, halign, valign, wrap = role
        self._xfs.append((self._numfmt_id(numfmt), font, fill, border,
                          halign, valign, wrap))
        index = len(self._xfs) - 1
        self._index[key] = index
        return index

    def xml(self) -> str:
        """The whole ``styles.xml`` document."""
        numfmts = "".join(
            f'<numFmt numFmtId="{index}" formatCode="{escape(code, {chr(34): "&quot;"})}"/>'
            for code, index in sorted(self._custom.items(), key=lambda pair: pair[1]))
        fonts = "".join(_FONTS)
        fills = "".join(_FILLS)
        borders = "".join(_BORDERS)
        parts: list[str] = []
        for numfmt, font, fill, border, halign, valign, wrap in self._xfs:
            attributes = [f'numFmtId="{numfmt}"', f'fontId="{font}"',
                          f'fillId="{fill}"', f'borderId="{border}"',
                          'xfId="0"']
            if numfmt:
                attributes.append('applyNumberFormat="1"')
            if font:
                attributes.append('applyFont="1"')
            if fill:
                attributes.append('applyFill="1"')
            if border:
                attributes.append('applyBorder="1"')
            alignment: list[str] = []
            if halign:
                alignment.append(f'horizontal="{halign}"')
            if valign:
                alignment.append(f'vertical="{valign}"')
            if wrap:
                alignment.append('wrapText="1"')
            if alignment:
                attributes.append('applyAlignment="1"')
                parts.append(f'<xf {" ".join(attributes)}><alignment '
                             f'{" ".join(alignment)}/></xf>')
            else:
                parts.append(f'<xf {" ".join(attributes)}/>')
        dxfs = "".join(
            f'<dxf><fill><patternFill patternType="solid">'
            f'<fgColor rgb="00{color}"/></patternFill></fill></dxf>'
            for color in STATUS_FILLS.values())
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<styleSheet xmlns="http://schemas.openxmlformats.org/'
            'spreadsheetml/2006/main">'
            f'<numFmts count="{len(self._custom)}">{numfmts}</numFmts>'
            f'<fonts count="{len(_FONTS)}">{fonts}</fonts>'
            f'<fills count="{len(_FILLS)}">{fills}</fills>'
            f'<borders count="{len(_BORDERS)}">{borders}</borders>'
            '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" '
            'borderId="0"/></cellStyleXfs>'
            f'<cellXfs count="{len(self._xfs)}">{"".join(parts)}</cellXfs>'
            '<cellStyles count="1"><cellStyle name="Normal" xfId="0" '
            'builtinId="0"/></cellStyles>'
            f'<dxfs count="{len(STATUS_FILLS)}">{dxfs}</dxfs>'
            '<tableStyles count="0" defaultTableStyle="TableStyleMedium2" '
            'defaultPivotStyle="PivotStyleLight16"/>'
            '</styleSheet>'
        )


# ---------------------------------------------------------------------------
# worksheet / package XML
# ---------------------------------------------------------------------------
_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
_XML_HEAD = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
#: A fixed ZIP timestamp (1980-01-01) so two exports are byte-identical.
_ZIP_DATE = (1980, 1, 1, 0, 0, 0)


def _column_letter(index: int) -> str:
    """1-based column index -> spreadsheet letters (1 -> A, 27 -> AA)."""
    letters = ""
    value = int(index)
    while value > 0:
        value, remainder = divmod(value - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def _column_index(letters: str) -> int:
    """Spreadsheet letters -> 1-based column index (``AA`` -> 27)."""
    total = 0
    for character in str(letters).upper():
        total = total * 26 + (ord(character) - 64)
    return total


def _cell_xml(reference: str, cell: Cell, style: int) -> str:
    """One ``<c>`` element; ``None`` renders as a styled empty cell."""
    value = excel_value(cell.value)
    if isinstance(value, bool):
        return f'<c r="{reference}" s="{style}" t="b"><v>{int(value)}</v></c>'
    if isinstance(value, (int, float)):
        return f'<c r="{reference}" s="{style}" t="n"><v>{value!r}</v></c>'
    text = "" if value is None else str(value)
    if not text:
        return f'<c r="{reference}" s="{style}" t="n"/>'
    return (f'<c r="{reference}" s="{style}" t="inlineStr"><is>'
            f'<t xml:space="preserve">{escape(text)}</t></is></c>')


def _hyperlinks_xml(links: Sequence[tuple[str, str]], ids: Sequence[str]) -> str:
    """``<hyperlinks>`` element for ``(reference, location)`` pairs.

    Excel's own internal links are relationship-based (``r:id`` plus a
    ``TargetMode="External"`` rel whose target is ``#<sheet>!A1``), and that is
    what the reference workbook contains, so the same shape is emitted here.
    """
    if not links:
        return ""
    items = "".join(
        f'<hyperlink ref="{reference}" r:id="{rid}"/>'
        for (reference, _location), rid in zip(links, ids))
    return f"<hyperlinks>{items}</hyperlinks>"


def _worksheet_xml(sheet: Sheet, styles: _StyleBook, *,
                   drawing: bool, table_index: int | None
                   ) -> tuple[str, list[tuple[str, str, str, str | None]]]:
    """``xl/worksheets/sheetN.xml`` and the relationships it needs."""
    rows_xml: list[str] = []
    links: list[tuple[str, str]] = []
    max_column = 1
    row_number = 1

    def emit_row(values: Sequence[Cell | None], height: float | None = None,
                 link: bool = False) -> None:
        nonlocal row_number, max_column
        cells: list[str] = []
        for offset, cell in enumerate(values):
            if cell is None:
                continue
            column = offset + 1
            max_column = max(max_column, column)
            reference = f"{_column_letter(column)}{row_number}"
            numfmt = NUMBER_FORMATS.get(cell.numfmt, cell.numfmt)
            style = styles.style(cell.kind, numfmt)
            if link and column == 1 and cell.value:
                links.append((reference, f"'{cell.value}'!A1"))
            cells.append(_cell_xml(reference, cell, style))
        if not cells:
            row_number += 1
            return
        attribute = f' ht="{height}" customHeight="1"' if height else ""
        rows_xml.append(f'<row r="{row_number}"{attribute}>{"".join(cells)}</row>')
        row_number += 1

    # row 1 title / row 2 subtitle / row 3 spacer, then the blocks
    emit_row([Cell(sheet.title, "title")], height=28)
    emit_row([Cell(sheet.subtitle, "subtitle")], height=32)
    row_number += 1
    for index, block in enumerate(sheet.blocks):
        if index == 0:
            row_number += max(block.gap_before - 1, 0)
        else:
            row_number += max(block.gap_before, 1)
        for position, values in enumerate(block.rows):
            height = 30 if (position == 0 and block.header) else None
            emit_row(values, height=height,
                     link=bool(block.links and position > 0))
    last_row = max(row_number - 1, 1)

    relationships: list[tuple[str, str, str, str | None]] = []
    link_ids: list[str] = []
    for reference, location in links:
        rid = f"rId{len(relationships) + 1}"
        link_ids.append(rid)
        relationships.append((rid, f"{_REL}/hyperlink", f"#{location}",
                              "External"))
    drawing_rid = None
    if drawing:
        drawing_rid = f"rId{len(relationships) + 1}"
        relationships.append((drawing_rid, f"{_REL}/drawing",
                              "/xl/drawings/drawing1.xml", None))
    table_rid = None
    if table_index is not None:
        table_rid = f"rId{len(relationships) + 1}"
        relationships.append((table_rid, f"{_REL}/table",
                              f"/xl/tables/table{table_index}.xml", None))

    merges = "".join(
        f'<mergeCell ref="A{r}:{_column_letter(max(sheet.merge_columns, 1))}{r}"/>'
        for r in (1, 2))
    conditional = ""
    if sheet.status_column:
        reference = f"{sheet.status_column}5:{sheet.status_column}{max(last_row, 5)}"
        rules = "".join(
            f'<cfRule type="cellIs" priority="{index}" operator="equal" '
            f'dxfId="{index - 1}"><formula>"{status}"</formula></cfRule>'
            for index, status in enumerate(STATUS_FILLS, start=1))
        conditional = f'<conditionalFormatting sqref="{reference}">{rules}' \
                      '</conditionalFormatting>'
    columns = "".join(
        f'<col min="{_column_index(name)}" max="{_column_index(name)}" '
        f'width="{width}" customWidth="1"/>'
        for name, width in sorted(sheet.widths.items(),
                                  key=lambda pair: _column_index(pair[0])))
    columns = f"<cols>{columns}</cols>" if columns else ""
    drawing_xml = f'<drawing r:id="{drawing_rid}"/>' if drawing_rid else ""
    tables = (f'<tableParts count="1"><tablePart r:id="{table_rid}"/>'
              '</tableParts>' if table_rid else "")
    document = (
        f'{_XML_HEAD}<worksheet xmlns="{_MAIN}" xmlns:r="{_REL}">'
        f'<sheetPr><tabColor rgb="00{sheet.tab_color}"/></sheetPr>'
        f'<dimension ref="A1:{_column_letter(max_column)}{last_row}"/>'
        '<sheetViews><sheetView showGridLines="0" zoomScale="90" '
        'workbookViewId="0"><pane ySplit="4" topLeftCell="A5" '
        'activePane="bottomLeft" state="frozen"/>'
        '<selection pane="bottomLeft" activeCell="A1" sqref="A1"/>'
        '</sheetView></sheetViews>'
        '<sheetFormatPr defaultRowHeight="15"/>'
        f'{columns}<sheetData>{"".join(rows_xml)}</sheetData>'
        f'<mergeCells count="2">{merges}</mergeCells>'
        f'{conditional}{_hyperlinks_xml(links, link_ids)}'
        '<pageMargins left="0.75" right="0.75" top="1" bottom="1" '
        'header="0.5" footer="0.5"/>'
        f'{drawing_xml}{tables}'
        '</worksheet>'
    )
    return document, relationships


def _table_xml(table_id: int, name: str, header: Sequence[str],
               last_row: int, last_column: int) -> str:
    """One ``xl/tables/tableN.xml`` ListObject definition."""
    reference = f"A4:{_column_letter(last_column)}{last_row}"
    columns = "".join(
        f'<tableColumn id="{index}" name="{escape(str(title), {chr(34): "&quot;"})}"/>'
        for index, title in enumerate(header, start=1))
    return (
        f'{_XML_HEAD}<table xmlns="{_MAIN}" id="{table_id}" name="{name}" '
        f'displayName="{name}" ref="{reference}" headerRowCount="1" '
        'totalsRowCount="0"><autoFilter '
        f'ref="{reference}"/><tableColumns count="{len(header)}">{columns}'
        '</tableColumns><tableStyleInfo name="TableStyleMedium2" '
        'showRowStripes="1" showColumnStripes="0"/></table>'
    )


def _chart_xml(spec: ChartSpec) -> str:
    """One ``xl/charts/chartN.xml`` clustered bar chart."""
    title = escape(spec.title)
    return (
        f'{_XML_HEAD}<chartSpace xmlns:a="http://schemas.openxmlformats.org/'
        'drawingml/2006/main" xmlns="http://schemas.openxmlformats.org/'
        'drawingml/2006/chart"><chart><title><tx><rich><a:bodyPr/><a:p>'
        f'<a:r><a:t>{title}</a:t></a:r></a:p></rich></tx></title><plotArea>'
        '<barChart><barDir val="col"/><grouping val="clustered"/>'
        '<ser><idx val="0"/><order val="0"/><tx><strRef>'
        f'<f>{escape(spec.series_name)}</f></strRef></tx>'
        '<spPr><a:ln><a:prstDash val="solid"/></a:ln></spPr>'
        f'<cat><numRef><f>{escape(spec.cats)}</f></numRef></cat>'
        f'<val><numRef><f>{escape(spec.vals)}</f></numRef></val>'
        '</ser><gapWidth val="150"/><axId val="10"/><axId val="100"/>'
        '</barChart><catAx><axId val="10"/><scaling><orientation val="minMax"/>'
        '</scaling><axPos val="l"/><majorTickMark val="none"/>'
        '<minorTickMark val="none"/><crossAx val="100"/><lblOffset val="100"/>'
        '</catAx><valAx><axId val="100"/><scaling><orientation val="minMax"/>'
        '</scaling><axPos val="l"/><majorGridlines/>'
        '<majorTickMark val="none"/><minorTickMark val="none"/>'
        '<crossAx val="10"/></valAx></plotArea><legend><legendPos val="r"/>'
        '</legend><plotVisOnly val="1"/><dispBlanksAs val="gap"/></chart>'
        '</chartSpace>'
    )


def _drawing_xml(specs: Sequence[ChartSpec]) -> str:
    """``xl/drawings/drawing1.xml`` with one one-cell anchor per chart."""
    anchors = "".join(
        f'<oneCellAnchor><from><col>{spec.anchor_col}</col><colOff>0</colOff>'
        f'<row>{spec.anchor_row}</row><rowOff>0</rowOff></from>'
        f'<ext cx="{spec.cx}" cy="{spec.cy}"/><graphicFrame>'
        f'<nvGraphicFramePr><cNvPr id="{index}" name="Chart {index}"/>'
        '<cNvGraphicFramePr/></nvGraphicFramePr>'
        f'<xfrm/><a:graphic><a:graphicData '
        'uri="http://schemas.openxmlformats.org/drawingml/2006/chart">'
        f'<c:chart r:id="rId{index}"/></a:graphicData></a:graphic>'
        '</graphicFrame></oneCellAnchor>'
        for index, spec in enumerate(specs, start=1))
    return (
        f'{_XML_HEAD}<xdr:wsDr xmlns:xdr="http://schemas.openxmlformats.org/'
        'drawingml/2006/spreadsheetDrawing" xmlns:a="http://schemas.'
        'openxmlformats.org/drawingml/2006/main" xmlns:c="http://schemas.'
        'openxmlformats.org/drawingml/2006/chart" xmlns:r="http://schemas.'
        f'openxmlformats.org/officeDocument/2006/relationships">{anchors}'
        '</xdr:wsDr>'
    )


# ---------------------------------------------------------------------------
# package assembly
# ---------------------------------------------------------------------------
def _relationships(entries: Sequence[Sequence[str | None]]) -> str:
    """One ``.rels`` document from ``(id, type, target[, targetMode])`` rows."""
    items = "".join(
        f'<Relationship Id="{row[0]}" Type="{row[1]}" Target="{row[2]}"'
        + (f' TargetMode="{row[3]}"' if len(row) > 3 and row[3] else "")
        + "/>"
        for row in entries)
    return f'{_XML_HEAD}<Relationships xmlns="{_PKG_REL}">{items}</Relationships>'


def _content_types(sheet_count: int, table_count: int, chart_count: int,
                   has_drawing: bool) -> str:
    """``[Content_Types].xml`` for the parts this writer emits."""
    overrides = [
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.'
        'openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.'
        'openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
        '<Override PartName="/xl/theme/theme1.xml" ContentType="application/'
        'vnd.openxmlformats-officedocument.theme+xml"/>',
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.'
        'openxmlformats-package.core-properties+xml"/>',
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.'
        'openxmlformats-officedocument.extended-properties+xml"/>',
    ]
    overrides.extend(
        f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.'
        'spreadsheetml.worksheet+xml"/>'
        for index in range(1, sheet_count + 1))
    overrides.extend(
        f'<Override PartName="/xl/tables/table{index}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.'
        'spreadsheetml.table+xml"/>'
        for index in range(1, table_count + 1))
    if has_drawing:
        overrides.append(
            '<Override PartName="/xl/drawings/drawing1.xml" ContentType='
            '"application/vnd.openxmlformats-officedocument.drawing+xml"/>')
        overrides.extend(
            f'<Override PartName="/xl/charts/chart{index}.xml" ContentType='
            '"application/vnd.openxmlformats-officedocument.drawingml.chart+xml"/>'
            for index in range(1, chart_count + 1))
    return (
        f'{_XML_HEAD}<Types xmlns="http://schemas.openxmlformats.org/package/'
        '2006/content-types"><Default Extension="rels" ContentType='
        '"application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        f'{"".join(overrides)}</Types>'
    )


_THEME = (
    f'{_XML_HEAD}<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/'
    '2006/main" name="Office Theme"><a:themeElements><a:clrScheme name='
    '"Office"><a:dk1><a:sysClr val="windowText" lastClr="000000"/></a:dk1>'
    '<a:lt1><a:sysClr val="window" lastClr="FFFFFF"/></a:lt1>'
    '<a:dk2><a:srgbClr val="1F497D"/></a:dk2><a:lt2><a:srgbClr val="EEECE1"/>'
    '</a:lt2><a:accent1><a:srgbClr val="4F81BD"/></a:accent1>'
    '<a:accent2><a:srgbClr val="C0504D"/></a:accent2><a:accent3>'
    '<a:srgbClr val="9BBB59"/></a:accent3><a:accent4><a:srgbClr val="8064A2"/>'
    '</a:accent4><a:accent5><a:srgbClr val="4BACC6"/></a:accent5>'
    '<a:accent6><a:srgbClr val="F79646"/></a:accent6><a:hlink>'
    '<a:srgbClr val="0000FF"/></a:hlink><a:folHlink><a:srgbClr val="800080"/>'
    '</a:folHlink></a:clrScheme><a:fontScheme name="Office"><a:majorFont>'
    '<a:latin typeface="Cambria"/><a:ea typeface=""/><a:cs typeface=""/>'
    '</a:majorFont><a:minorFont><a:latin typeface="Calibri"/>'
    '<a:ea typeface=""/><a:cs typeface=""/></a:minorFont></a:fontScheme>'
    '<a:fmtScheme name="Office"><a:fillStyleLst><a:solidFill>'
    '<a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst>'
    '<a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/>'
    '</a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle>'
    '<a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst>'
    '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst>'
    '</a:fmtScheme></a:themeElements></a:theme>'
)


def _core_xml(title: str, subject: str, creator: str, description: str,
              created: str | None) -> str:
    """``docProps/core.xml``; the timestamp is opt-in so exports are stable."""
    stamp = (f'<dcterms:created xsi:type="dcterms:W3CDTF">{created}'
             '</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">'
             f'{created}</dcterms:modified>' if created else "")
    return (
        f'{_XML_HEAD}<cp:coreProperties xmlns:cp="http://schemas.'
        'openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        f'<dc:creator>{escape(creator)}</dc:creator>'
        f'<dc:title>{escape(title)}</dc:title>'
        f'<dc:subject>{escape(subject)}</dc:subject>'
        f'<dc:description>{escape(description)}</dc:description>'
        f'{stamp}</cp:coreProperties>'
    )


def _app_xml(application: str) -> str:
    return (
        f'{_XML_HEAD}<Properties xmlns="http://schemas.openxmlformats.org/'
        'officeDocument/2006/extended-properties">'
        f'<Application>{escape(application)}</Application>'
        '<AppVersion>1.0</AppVersion></Properties>'
    )


def _write_zip(path: Path, parts: Sequence[tuple[str, str]]) -> None:
    """Write ``parts`` into a ZIP deterministically (fixed order and mtime)."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, text in parts:
            info = zipfile.ZipInfo(name, date_time=_ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, text.encode("utf-8"))


def write_workbook(path: Any, sheets: Sequence[Sheet], *,
                   properties: Mapping[str, str] | None = None,
                   created: str | None = None) -> Path:
    """Write ``sheets`` as an ``.xlsx`` package; return the written path.

    The file is assembled in memory and published atomically (sibling temporary
    + ``fsync`` + :func:`os.replace`), so a reader sees the previous workbook or
    the complete new one.  ``created`` is an ISO-8601 UTC stamp for
    ``docProps/core.xml``; it is *optional* and left out by default so two
    exports of the same rows are byte-identical.
    """
    target = Path(str(path))
    target.parent.mkdir(parents=True, exist_ok=True)
    stylebook = _StyleBook()
    props = dict(REFERENCE_STYLE)
    if properties:
        props.update({key: str(value) for key, value in properties.items()})

    table_specs: list[tuple[int, str, tuple[str, ...], int, int]] = []
    sheet_xml: list[str] = []
    extras: list[tuple[str, str]] = []
    chart_specs: list[ChartSpec] = []

    for sheet_index, sheet in enumerate(sheets, start=1):
        table_index = None
        block = sheet.blocks[0] if sheet.blocks else None
        data_rows = (len(block.rows) - 1) if block and block.header else 0
        if block is not None and block.table:
            header = tuple(str(cell.value) for cell in block.rows[0]
                           if cell is not None)
            last_row = 4 + data_rows
            table_specs.append((len(table_specs) + 1, block.table, header,
                                last_row, len(header)))
            table_index = table_specs[-1][0]
        document, relationships = _worksheet_xml(
            sheet, stylebook, drawing=bool(sheet.charts),
            table_index=table_index)
        sheet_xml.append(document)
        if relationships:
            extras.append((f"xl/worksheets/_rels/sheet{sheet_index}.xml.rels",
                           _relationships(relationships)))
        if sheet.charts:
            chart_specs.extend(sheet.charts)

    parts: list[tuple[str, str]] = []
    parts.append(("[Content_Types].xml",
                  _content_types(len(sheets), len(table_specs),
                                 len(chart_specs), bool(chart_specs))))
    parts.append(("_rels/.rels", _relationships((
        ("rId1", f"{_REL}/officeDocument", "xl/workbook.xml"),
        ("rId2", "http://schemas.openxmlformats.org/package/2006/"
                 "relationships/metadata/core-properties", "docProps/core.xml"),
        ("rId3", f"{_REL}/extended-properties", "docProps/app.xml")))))
    parts.append(("docProps/core.xml",
                  _core_xml(props["title"], props["subject"], props["creator"],
                            props["description"], created)))
    parts.append(("docProps/app.xml", _app_xml(props["application"])))
    workbook_relationships = [
        (f"rId{index}", f"{_REL}/worksheet",
         f"/xl/worksheets/sheet{index}.xml")
        for index in range(1, len(sheets) + 1)]
    next_id = len(sheets) + 1
    workbook_relationships.append(
        (f"rId{next_id}", f"{_REL}/theme", "/xl/theme/theme1.xml"))
    workbook_relationships.append(
        (f"rId{next_id + 1}", f"{_REL}/styles", "/xl/styles.xml"))
    parts.append(("xl/_rels/workbook.xml.rels",
                  _relationships(workbook_relationships)))
    parts.append(("xl/workbook.xml", (
        f'{_XML_HEAD}<workbook xmlns="{_MAIN}" xmlns:r="{_REL}"><workbookPr/>'
        '<bookViews><workbookView visibility="visible" minimized="0" '
        'showHorizontalScroll="1" showVerticalScroll="1" showSheetTabs="1" '
        'tabRatio="600" firstSheet="0" activeTab="0" '
        'autoFilterDateGrouping="1"/></bookViews><sheets>'
        + "".join(
            f'<sheet name="{escape(sheet.name, {chr(34): "&quot;"})}" '
            f'sheetId="{index}" state="visible" r:id="rId{index}"/>'
            for index, sheet in enumerate(sheets, start=1))
        + '</sheets><definedNames/><calcPr calcId="124519" '
          'fullCalcOnLoad="1"/></workbook>')))
    parts.append(("xl/styles.xml", stylebook.xml()))
    parts.append(("xl/theme/theme1.xml", _THEME))
    for index, text in enumerate(sheet_xml, start=1):
        parts.append((f"xl/worksheets/sheet{index}.xml", text))
    for number, name, header, last_row, last_column in table_specs:
        parts.append((f"xl/tables/table{number}.xml",
                      _table_xml(number, name, header, last_row, last_column)))
    if chart_specs:
        parts.append(("xl/drawings/drawing1.xml", _drawing_xml(chart_specs)))
        parts.append(("xl/drawings/_rels/drawing1.xml.rels", _relationships(
            [(f"rId{index}", f"{_REL}/chart", f"/xl/charts/chart{index}.xml")
             for index in range(1, len(chart_specs) + 1)])))
        for index, spec in enumerate(chart_specs, start=1):
            parts.append((f"xl/charts/chart{index}.xml", _chart_xml(spec)))
    parts.extend(extras)

    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        _write_zip(temporary, parts)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    _fsync_directory(target.parent)
    return target


def _fsync_directory(directory: Path) -> None:
    """Best-effort ``fsync`` of a directory entry (ignored when unsupported)."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(str(directory), flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


# ---------------------------------------------------------------------------
# content: helpers
# ---------------------------------------------------------------------------
def _block_start_rows(sheet: Sheet) -> list[int]:
    """First row of every block, following the layout rule of the writer.

    The title occupies row 1, the subtitle row 2 and row 3 is blank, so the
    first block starts at row 4; every later block starts ``gap_before`` rows
    after the previous one ends.  Kept next to the writer so a chart reference
    and the cell it points at cannot drift apart.
    """
    starts: list[int] = []
    row = 4
    for index, block in enumerate(sheet.blocks):
        if index:
            row += max(block.gap_before, 1)
        starts.append(row)
        row += len(block.rows)
    return starts


def _width(name: str) -> float:
    """Deterministic column width for a header name.

    The reference workbook's widths are hand-tuned per sheet (28/9/8/20/...);
    regenerating them from the header keeps a 100-column dictionary and a
    44-column data sheet readable without a per-column table to maintain.
    """
    fixed = {
        "run_id": 28.0, "notes": 60.0, "report_path": 60.0,
        "solution_path": 60.0, "path": 60.0, "purpose": 50.0,
        "meaning": 60.0, "column": 30.0, "value": 42.0,
        "comparability": 46.0, "note": 60.0, "what is covered": 70.0,
        "scope": 50.0, "retained data": 55.0, "decision": 30.0,
    }
    key = str(name).strip().lower()
    if key in fixed:
        return fixed[key]
    if key.endswith("sha256"):
        return 66.0
    return max(9.0, min(46.0, len(str(name)) * 1.15 + 4.0))


def _widths(header: Sequence[str]) -> dict[str, float]:
    return {_column_letter(index): _width(name)
            for index, name in enumerate(header, start=1)}


def _table_block(header: Sequence[str], rows: Sequence[Sequence[Any]], *,
                 table: str | None = None,
                 numfmts: Mapping[str, str] | None = None,
                 kinds: Mapping[int, str] | None = None,
                 gap_before: int = 1) -> Block:
    """A header row plus data rows, one cell per header column."""
    formats = dict(numfmts or {})
    cells: list[list[Cell | None]] = [[
        Cell(name, "header") for name in header]]
    for row in rows:
        line: list[Cell | None] = []
        for index, name in enumerate(header):
            value = row[index] if isinstance(row, (list, tuple)) else row.get(name)
            kind = "data"
            if kinds and index in kinds:
                kind = kinds[index]
            line.append(Cell(value, kind, formats.get(name, "")))
        cells.append(line)
    return Block(rows=cells, gap_before=gap_before, table=table, header=True)


def _record_block(header: Sequence[str], records: Sequence[Mapping[str, Any]],
                  *, table: str | None = None,
                  status_index: int | None = None,
                  gap_before: int = 1) -> Block:
    """A table block straight from reference records (number formats applied)."""
    cells: list[list[Cell | None]] = [[
        Cell(name, "header") for name in header]]
    for record in records:
        line: list[Cell | None] = []
        for index, name in enumerate(header):
            kind = "data"
            if status_index is not None and index == status_index:
                status = str(record.get(name) or "")
                kind = {"FAIL": "fail", "SKIP": "warning",
                        "ERROR": "fail", "TIMEOUT": "fail"}.get(status, "data")
            line.append(Cell(record.get(name), kind,
                             COLUMN_NUMBER_FORMATS.get(name, "")))
        cells.append(line)
    return Block(rows=cells, gap_before=gap_before, table=table, header=True)


def _kv_block(pairs: Sequence[Sequence[Any]], *, gap_before: int = 1,
              header: Sequence[str] | None = None) -> Block:
    """A label/value block (column A bold on ``#D9E2F3``, later columns plain).

    ``pairs`` may be 2-tuples (label, value) or longer rows, which is what the
    reference's three-column "Key result | Value | Comparability" block uses.
    """
    cells: list[list[Cell | None]] = []
    if header:
        cells.append([Cell(name, "header") for name in header])
    for row in pairs:
        values = list(row)
        line = [Cell(values[0], "kv_label")]
        line.extend(Cell(value, "kv_value") for value in values[1:])
        cells.append(line)
    return Block(rows=cells, gap_before=gap_before, header=bool(header))


def _note_block(lines: Sequence[str], *, gap_before: int = 1) -> Block:
    """One explanatory line per row (used where data cannot be derived)."""
    return Block(rows=[[Cell(line, "note")] for line in lines],
                 gap_before=gap_before, header=False)


def _text(value: Any) -> str:
    """Cell text of any value, matching the CSV rendering."""
    return ledger_csv.csv_cell(value)


# ---------------------------------------------------------------------------
# content: derived summaries
# ---------------------------------------------------------------------------
def _count(values: Iterable[Any]) -> dict[str, int]:
    """Count non-``None`` values by their CSV text (``None`` means "absent")."""
    counts: dict[str, int] = {}
    for value in values:
        if value is None:
            continue
        key = _text(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def lattice_sheet_records(name: str,
                          records: Sequence[Mapping[str, Any]]
                          ) -> list[Mapping[str, Any]]:
    """Rows of one reference per-lattice sheet (``05``-``08``).

    The split is the reference workbook's own, measured from its 142 rows:

    ``05_L24_c64``   lattice ``24x24x24x64``, dtype ``c64`` (74 rows)
    ``06_L24_c128``  lattice ``24x24x24x64``, dtype ``c128`` (31 rows)
    ``07_L32``       lattice ``32x32x32x64``, *either* dtype (25 rows)
    ``08_MG``        solver in :data:`MG_SOLVERS`, every lattice (69 rows)

    All four are restricted to ``phase == "solve"``: the reference's
    prepare/prewarm rows (``cache_build``/``input_prewarm``) never appear in
    them, while an MG row appears **twice** - in its lattice sheet and in
    ``08_MG``.  ``74 + 31 + 25`` therefore exceeds the 130 solve rows.
    """
    if name not in ("05_L24_c64", "06_L24_c128", "07_L32", "08_MG"):
        raise ValueError(f"{name!r} is not a reference per-lattice sheet")
    lattice = {"05_L24_c64": (24, 24, 24, 64),
               "06_L24_c128": (24, 24, 24, 64),
               "07_L32": (32, 32, 32, 64)}.get(name)
    dtype = {"05_L24_c64": "c64", "06_L24_c128": "c128"}.get(name)
    selected: list[Mapping[str, Any]] = []
    for record in records:
        if str(record.get("phase")) != "solve":
            continue
        if name == "08_MG":
            if str(record.get("solver")) in MG_SOLVERS:
                selected.append(record)
            continue
        value = record.get("lattice")
        if not isinstance(value, (list, tuple)):
            continue
        if tuple(int(item) for item in value) != lattice:
            continue
        if dtype is not None and str(record.get("dtype")) != dtype:
            continue
        selected.append(record)
    return selected


def summary_counts(records: Sequence[Mapping[str, Any]], *,
                   csv_path: Any = None) -> dict[str, Any]:
    """The aggregate figures of ``02_Summary`` (all derivable, no prose).

    The keys and their order mirror the reference workbook's own first summary
    block (``Rows`` ... ``Lattice counts (solve)``), plus the three counts this
    export can state exactly:

    ``csv_files``        the CSV this export wrote (``1`` when ``csv_path`` is
                         given, ``0`` otherwise - the reference counts the CSV
                         files of its run directory);
    ``cache_manifests``  distinct non-empty ``cache_manifest_path`` values the
                         rows declare - only a native MG row carries one;
    ``local_h5``         distinct non-empty ``solution_path`` values, i.e. the
                         solution files the rows wrote locally.

    ``solutions`` (the row count with a solution path) stays available for the
    cover sheet; ``local_h5`` is the de-duplicated form of the same evidence.
    """
    solve = [record for record in records
             if str(record.get("phase")) == "solve"]
    prepare = [record for record in records
               if str(record.get("phase")) == "prepare"]
    prewarm = [record for record in records
               if str(record.get("phase")) == "prewarm"]
    cache_manifests = {str(record["cache_manifest_path"]) for record in records
                       if record.get("cache_manifest_path")}
    local_h5 = {str(record["solution_path"]) for record in records
                if record.get("solution_path")}
    return {
        "rows": len(records),
        "solve": len(solve),
        "prepare": len(prepare),
        "prewarm": len(prewarm),
        "prepare_and_cache": len(prepare) + len(prewarm),
        "reports": sum(1 for record in records if record.get("report_path")),
        "solutions": sum(1 for record in records if record.get("solution_path")),
        "csv_files": 0 if csv_path is None else 1,
        "cache_manifests": len(cache_manifests),
        "local_h5": len(local_h5),
        "status": _count(record.get("status") for record in records),
        "phase": _count(record.get("phase") for record in records),
        "solver": _count(record.get("solver") for record in solve),
        "dtype": _count(record.get("dtype") for record in solve),
        "lattice": _count(record.get("lattice") for record in solve),
        "shm": _count(record.get("shm") for record in solve),
        "grid_mode": _count(record.get("grid_mode") for record in solve),
    }


def _json_counts(counts: Mapping[str, int]) -> str:
    """``{"PASS": 115, ...}`` exactly as the reference writes the count cells."""
    items = ", ".join(f'"{key}": {value}' for key, value in counts.items())
    return "{" + items + "}"


def _best_rows(records: Sequence[Mapping[str, Any]],
               limit: int = 12) -> list[tuple[str, str, str]]:
    """Fastest PASS row per (dtype, lattice, solver, rank_x_thread, grid, shm).

    This is the derivable part of the reference's hand-written "Key result"
    table: the same query, answered from the exported rows instead of from a
    human's notes.  The value is ``"<wall> s / <iterations> it / <residual>"``
    and the comparability cell states the selection rule, so the sheet never
    implies more than it measured.
    """
    best: dict[tuple[str, ...], Mapping[str, Any]] = {}
    for record in records:
        if str(record.get("status")) != "PASS":
            continue
        seconds = record.get("wall_seconds")
        if not isinstance(seconds, (int, float)) or isinstance(seconds, bool):
            continue
        key = (_text(record.get("dtype")), _text(record.get("lattice")),
               _text(record.get("solver")), _text(record.get("rank_x_thread")),
               _text(record.get("grid_mode")), _text(record.get("shm")))
        current = best.get(key)
        if current is None or float(seconds) < float(current["wall_seconds"]):
            best[key] = record
    ordered = sorted(best.items(), key=lambda item: (
        str(item[0][0]), str(item[0][1]), str(item[0][2]),
        str(item[0][3]), str(item[0][4]), str(item[0][5])))
    derived: list[tuple[str, str, str]] = []
    for key, record in ordered[:limit]:
        label = (f"{key[1].replace('[', '').replace(']', '').replace(' ', '')}"
                 f" {key[0]} {key[2]} {key[3]} {key[4]} shm{key[5]}")
        seconds = float(record["wall_seconds"])
        iterations = record.get("iterations")
        residual = record.get("final_true_residual")
        residual_text = ("" if residual is None
                         else f"{float(residual):.4e}"
                         if isinstance(residual, (int, float)) else _text(residual))
        value = f"{seconds:.6f} s"
        if iterations is not None:
            value += f" / {_text(iterations)} it"
        if residual_text:
            value += f" / {residual_text}"
        derived.append((label, value, "min wall_seconds of the PASS rows "
                                      "(single sample; not a benchmark claim)"))
    return derived


# ---------------------------------------------------------------------------
# content: the 21 sheets
# ---------------------------------------------------------------------------
def _data_sheet(name: str, title: str, subtitle: str, header: Sequence[str],
                records: Sequence[Mapping[str, Any]], *,
                table: str, status_column: str = "C",
                status_index: int | None = None) -> Sheet:
    """A reference data sheet: title/subtitle, row-4 header, row-5 data."""
    block = _record_block(header, records, table=table,
                          status_index=status_index)
    return Sheet(name=name, title=title, subtitle=subtitle,
                 widths=_widths(header), blocks=[block],
                 tab_color=TAB_COLORS.get(name, "1F4E78"),
                 status_column=status_column)


def _lattice_text(value: Any) -> str:
    """``24x24x24x64`` of a lattice list (``""`` when absent)."""
    if isinstance(value, (list, tuple)):
        return "x".join(str(item) for item in value)
    return _text(value)


def build_sheets(records: Sequence[Mapping[str, Any]], *,
                 payload: Mapping[str, Any] | None = None,
                 csv_path: Any = None,
                 workbook_name: str = "kupl_kunpeng_performance.xlsx",
                 generated: str | None = None,
                 source_dir: Any = None) -> list[Sheet]:
    """Build the 21 reference sheets from reference records.

    ``records`` are :mod:`pyqcu.test.ledger_csv` records (the same objects the
    CSV is written from), so the workbook and the CSV can never disagree about a
    value.  ``payload`` is the full-mode document (for provenance and paths);
    ``csv_path``/``source_dir`` are what ``18_CSV_Index`` reports.
    """
    records = sorted(records, key=lambda record: str(record.get("run_id")))
    counts = summary_counts(records, csv_path=csv_path)
    sheets: list[Sheet] = []

    # -- 00_Cover ----------------------------------------------------------
    cover_rows: list[tuple[str, str]] = [
        ("Workbook", str(workbook_name)),
        ("Generated", generated or "n/a (deterministic export; pass generated=)"),
        ("Provenance", "pyqcu.test full-mode ledger (--mode solve|run|merge)"),
        ("Rows", str(counts["rows"])),
        ("Solve rows", str(counts["solve"])),
        ("Prepare/cache rows", str(counts["prepare_and_cache"])),
        ("Status counts", _json_counts(counts["status"])),
        ("Phase counts", _json_counts(counts["phase"])),
        ("Solver counts (solve)", _json_counts(counts["solver"])),
        ("Dtype counts (solve)", _json_counts(counts["dtype"])),
        ("Lattice counts (solve)", _json_counts(counts["lattice"])),
        ("Reports", str(counts["reports"])),
        ("Solutions", str(counts["solutions"])),
        ("CSV files", str(counts["csv_files"])),
        ("Cache manifests", str(counts["cache_manifests"])),
        ("Local h5", str(counts["local_h5"])),
    ]
    if payload is not None:
        paths = payload.get("paths")
        if isinstance(paths, Mapping) and paths.get("root"):
            cover_rows.append(("Data root", str(paths["root"])))
        options = payload.get("options")
        if isinstance(options, Mapping) and options.get("mode"):
            cover_rows.append(("Mode", str(options["mode"])))
    sheets.append(Sheet(
        name="00_Cover", title="KUPL-QCU Kunpeng performance data",
        subtitle=("Ledger view of the formal pyqcu.test full-mode run. "
                  "Counts are derived from the exported rows; every column the "
                  "formal port cannot observe is left empty (see 17)."),
        widths={"A": 22.0, "B": 120.0},
        blocks=[
            _kv_block(cover_rows),
            _note_block([
                "READ FIRST: do not mix atol / source / gauge / grid / shm / "
                "rank / cache identities.",
                "The KUPL-Kunpeng driver's 1e-4 historical numbers and the "
                "1e-6/1e-9 matrix numbers are not comparable (see 17).",
            ]),
        ],
        tab_color=TAB_COLORS["00_Cover"], merge_columns=8))

    # -- 01_Index ----------------------------------------------------------
    index_header = ("Sheet", "Purpose", "Primary source", "Rows", "Notes")
    index_purposes = {
        "00_Cover": "Task cover, environment, key results, warnings",
        "01_Index": "Sheet map and definitions",
        "02_Summary": "KPI counts and key performance tables",
        "03_Matrix": "Matrix dimensions, coverage and missing axes",
        "04_Performance_All": "All rows, curated columns",
        "05_L24_c64": "L24 c64 solve rows (MG included)",
        "06_L24_c128": "L24 c128 solve rows (MG included)",
        "07_L32": "L32 solve rows, both dtypes",
        "08_MG": "MG rows: restart/cache/residual",
        "09_Startup_Loop": "Startup and loop prototype timings",
        "10_Bandwidth": "Effective bandwidth definitions",
        "11_Cache": "Cache artifacts, sizes, setup, identity",
        "12_Failures": "FAIL/ERROR/TIMEOUT rows",
        "13_Abandoned": "Abandoned axes and decisions",
        "14_Input_Integrity": "Input files, hashes, atomic write rules",
        "15_Historical": "Historical numbers (not controlled)",
        "16_Scripts": "Script identity",
        "17_Data_Dictionary": "100 CSV columns, provenance and units",
        "18_CSV_Index": "Exported CSV files",
        "19_Reports_Index": "Per-row reports",
        "20_Job_Ledger": "Scheduler jobs and outcomes",
    }
    index_sources = {
        "00_Cover": "records", "01_Index": "this workbook",
        "02_Summary": "records", "03_Matrix": "records",
        "04_Performance_All": "records", "05_L24_c64": "records",
        "06_L24_c128": "records", "07_L32": "records", "08_MG": "records",
        "09_Startup_Loop": "not in pyqcu.test", "10_Bandwidth": "reference docs",
        "11_Cache": "cache manifests", "12_Failures": "records",
        "13_Abandoned": "not in pyqcu.test", "14_Input_Integrity": "records",
        "15_Historical": "not in pyqcu.test", "16_Scripts": "package files",
        "17_Data_Dictionary": "ledger_csv.COLUMN_RULES",
        "18_CSV_Index": "the exported CSV", "19_Reports_Index": "records",
        "20_Job_Ledger": "not in pyqcu.test",
    }
    index_rows: list[list[Any]] = []
    for name in SHEET_NAMES:
        rows_for_sheet = _sheet_row_count(name, records, counts)
        index_rows.append([name, index_purposes[name], index_sources[name],
                           rows_for_sheet, ""])
    sheets.append(Sheet(
        name="01_Index", title="Index / 子表导航",
        subtitle=("Every sheet is derived from this ledger. The Sheet cells are "
                  "internal links."),
        widths={"A": 24.0, "B": 62.0, "C": 28.0, "D": 8.0, "E": 38.0},
        blocks=[Block(rows=[[Cell(name, "header") for name in index_header]]
                      + [[Cell(value, "link" if column == 0 else "data")
                          for column, value in enumerate(row)]
                         for row in index_rows],
                      table="tbl_index", header=True, links=True)],
        tab_color=TAB_COLORS["01_Index"], merge_columns=5))

    # -- 02_Summary --------------------------------------------------------
    # The first block is the reference workbook's own key order (measured from
    # its A4:B15 range); every value is derived from the exported records, so a
    # "0" means "no row carries it", never "not measured".  The two extra rows
    # after the reference twelve are this export's own (a node name and a
    # grid-mode breakdown are meaningful here and cost nothing to state).
    status_counts = counts["status"]
    summary_block = _kv_block([
        ("Rows", counts["rows"]),
        ("Solve rows", counts["solve"]),
        ("Prepare/cache rows", counts["prepare_and_cache"]),
        ("Reports", counts["reports"]),
        ("CSV files", counts["csv_files"]),
        ("Cache manifests", counts["cache_manifests"]),
        ("Local h5", counts["local_h5"]),
        ("Status counts", _json_counts(status_counts)),
        ("Phase counts", _json_counts(counts["phase"])),
        ("Solver counts (solve)", _json_counts(counts["solver"])),
        ("Dtype counts (solve)", _json_counts(counts["dtype"])),
        ("Lattice counts (solve)", _json_counts(counts["lattice"])),
        ("Distinct nodes", "n/a (not recorded per row)"),
        ("Grid-mode counts (solve)", _json_counts(counts["grid_mode"])),
    ])
    key_block = _kv_block(
        _best_rows(records),
        header=("Key result", "Value", "Comparability / note"))
    grid_pairs = sorted(counts["grid_mode"].items())
    summary_sheet = Sheet(
        name="02_Summary", title="Summary / 总览",
        subtitle=("Counts are derived from the exported rows; the two value "
                  "blocks feed the charts below. CSV files / Cache manifests / "
                  "Local h5 count this export's CSV, the distinct cache "
                  "manifests and the distinct solution files the rows name."),
        widths={"A": 38.0, "B": 42.0, "C": 52.0},
        blocks=[
            summary_block,
            key_block,
            _kv_block(grid_pairs, header=("Config", "row count")),
            _kv_block(sorted(status_counts.items()), header=("Status", "count")),
        ],
        tab_color=TAB_COLORS["02_Summary"], merge_columns=3)
    # Charts read the two value blocks; the row numbers are resolved from the
    # layout rule so the references cannot drift out of step with the blocks.
    starts = _block_start_rows(summary_sheet)
    grid_start, status_start = starts[2], starts[3]
    summary_sheet.charts = [
        ChartSpec(
            title="row count per grid_mode",
            series_name=f"'02_Summary'!$B${grid_start}",
            cats=f"'02_Summary'!$A${grid_start + 1}:$A${grid_start + len(grid_pairs)}",
            vals=f"'02_Summary'!$B${grid_start + 1}:$B${grid_start + len(grid_pairs)}",
            anchor_col=3, anchor_row=29, cx=5040000, cy=2520000),
        ChartSpec(
            title="PASS/FAIL/SKIP/TIMEOUT/ERROR",
            series_name=f"'02_Summary'!$B${status_start}",
            cats=(f"'02_Summary'!$A${status_start + 1}:$A$"
                  f"{status_start + len(status_counts)}"),
            vals=(f"'02_Summary'!$B${status_start + 1}:$B$"
                  f"{status_start + len(status_counts)}"),
            anchor_col=3, anchor_row=47, cx=3600000, cy=2520000),
    ]
    sheets.append(summary_sheet)

    # -- 03_Matrix ---------------------------------------------------------
    solve = [record for record in records
             if str(record.get("phase")) == "solve"]
    matrix_pairs: list[tuple[str, Any]] = [
        ("Rows", counts["rows"]),
        ("Solve rows", counts["solve"]),
        ("Distinct lattices", _json_counts(counts["lattice"])),
        ("Distinct dtypes", _json_counts(counts["dtype"])),
        ("Distinct solvers", _json_counts(counts["solver"])),
        ("Distinct grid modes", _json_counts(counts["grid_mode"])),
        ("Distinct shm values", _json_counts(counts["shm"])),
        ("Distinct seeds", _json_counts(_count(r.get("seed") for r in solve))),
        ("Distinct ranks x threads",
         _json_counts(_count(r.get("rank_x_thread") for r in solve))),
        ("Coverage vs the 100-column contract",
         f"{len(REFERENCE_CSV_COLUMNS) - len(ledger_csv.MISSING_COLUMNS)} of "
         f"{len(REFERENCE_CSV_COLUMNS)} columns populated by this export"),
        ("Missing / not observable",
         f"{len(ledger_csv.MISSING_COLUMNS)} columns; see 17_Data_Dictionary"),
    ]
    coverage_header = ("Axis", "What is covered", "Status")
    coverage_rows = [
        ["dtype", _json_counts(counts["dtype"]), "derived"],
        ["solver", _json_counts(counts["solver"]), "derived"],
        ["lattice", _json_counts(counts["lattice"]), "derived"],
        ["grid_mode", _json_counts(counts["grid_mode"]), "derived"],
        ["shm", _json_counts(counts["shm"]), "derived"],
        ["rank_x_thread",
         _json_counts(_count(r.get("rank_x_thread") for r in solve)),
         "derived"],
        ["profiler counters", "not emitted by pyqcu.test", "missing"],
        ["node / cpufreq", "not emitted by pyqcu.test", "missing"],
        ["scheduler jobs", "not emitted by pyqcu.test", "missing"],
    ]
    sheets.append(Sheet(
        name="03_Matrix", title="Matrix / 矩阵口径与覆盖",
        subtitle=("The matrix is enumerable from the ledger; coverage below is "
                  "derived from the exported rows."),
        widths={"A": 28.0, "B": 90.0, "C": 24.0},
        blocks=[_kv_block(matrix_pairs),
                _table_block(coverage_header, coverage_rows)],
        tab_color=TAB_COLORS["03_Matrix"], merge_columns=3))

    # -- 04_Performance_All ------------------------------------------------
    sheets.append(_data_sheet(
        "04_Performance_All", "04 Performance All",
        (f"All {len(records)} rows; filter by phase/status/dtype/lattice/"
         "solver. Empty cells are columns the formal port cannot observe."),
        PERFORMANCE_COLUMNS, records, table="tbl_perf_all",
        status_column="C"))

    # -- 05 / 06 / 07 / 08 -------------------------------------------------
    # The split is measured from the reference workbook, not guessed: 05 = every
    # L24 c64 *solve* row, 06 = every L24 c128 solve row, 07 = every L32 solve
    # row of either dtype, 08 = every MG solve row across lattices.  MG rows
    # therefore appear both in their lattice sheet and in 08_MG (the reference's
    # 74/31/25/69 counts for 142 rows say exactly that), and prepare/prewarm rows
    # appear in none of them.
    for name, title, subtitle in (
        ("05_L24_c64", "05 L24 c64",
         "L24 c64 solve rows, MG included (MG rows repeat in 08_MG)."),
        ("06_L24_c128", "06 L24 c128",
         "L24 c128 solve rows; c128/c64 ratios are not per-operation cost."),
        ("07_L32", "07 L32",
         "L32 solve rows, both dtypes."),
        ("08_MG", "08 MG",
         "MG rows of every lattice: restart/cache/residual. MG iterations may "
         "be empty; use the residual and status."),
    ):
        sheets.append(_data_sheet(
            name, title, subtitle, MG_SHEET_COLUMNS,
            lattice_sheet_records(name, records),
            table=f"tbl_{name}", status_column="B"))

    # -- 09_Startup_Loop ---------------------------------------------------
    sheets.append(Sheet(
        name="09_Startup_Loop", title="Startup / Loop prototype",
        subtitle=("Startup and loop-prototype timings are measurements of the "
                  "informal HPCKit tree; the formal port records one timing per "
                  "case (wall_seconds) only."),
        widths={"A": 24.0, "B": 90.0, "C": 30.0},
        blocks=[_note_block([
            "n/a: pyqcu.test does not measure mpirun boot / import / loop "
            "amortisation.",
            "Reproduce with the informal driver: "
            "pyqcu/testing/kupl/test_kunpeng_all.py --mode validate and the "
            "loop prototype under data/kupl_loop_proto_*.",
        ])],
        tab_color=TAB_COLORS["09_Startup_Loop"], merge_columns=3))

    # -- 10_Bandwidth ------------------------------------------------------
    bandwidth_rows = [
        ["Dslash bytes",
         "2 * [ (2*4*9 + 2*12) * half_sites * sizeof(C) + 4 * half_sites / "
         "t_slices * 12 * sizeof(C) ]"],
        ["Clover bytes", "2 * [ (2*6*6 + 2*12) * half_sites * sizeof(C) ]"],
        ["t_slices", "max(local[3] / 2, 1)"],
        ["sizeof(C)", "c64 = 8, c128 = 16"],
        ["gbps_rank", "bytes / seconds / 1e9"],
        ["gbps_total", "gbps_rank * ranks"],
        ["Source", "KUPL OPT_PROFILE counters (dslash_*/clover_*/reduce_*)"],
        ["Availability", "n/a: not emitted by pyqcu.test - the formal port "
                         "records no operator profiler counters"],
    ]
    sheets.append(Sheet(
        name="10_Bandwidth", title="Bandwidth / 有效带宽口径",
        subtitle=("Model bytes divided by measured seconds; no PMU/uncore "
                  "counters. The four metrics are not interchangeable."),
        widths={"A": 24.0, "B": 100.0},
        blocks=[_table_block(("Quantity", "Definition"), bandwidth_rows)],
        tab_color=TAB_COLORS["10_Bandwidth"], merge_columns=2))

    # -- 11_Cache ----------------------------------------------------------
    # The header is the reference's, verbatim: the workbook's sheets are read by
    # position, and adding a twelfth column here would break a reader that maps
    # ``cache_sha256`` to column K.  The manifest identity hash the reference
    # never carried stays in the CSV (``cache_identity_sha256``) and is named in
    # the sheet note instead.
    cache_header = ("file", "status", "setup_seconds", "size_bytes", "dtype",
                    "lattice", "mg_grid", "nv_iters", "gauge_file_sha256",
                    "gauge_canonical_sha256", "cache_sha256")
    cache_rows: list[list[Any]] = []
    seen_caches: set[str] = set()
    for record in records:
        path = record.get("cache_path")
        if not path or str(path) in seen_caches:
            continue
        seen_caches.add(str(path))
        cache_rows.append([
            path, record.get("status"), record.get("build_mg_cache_seconds"),
            None, record.get("dtype"), record.get("lattice"),
            record.get("mg_grid"), record.get("nv_iters"),
            record.get("gauge_sha256"),
            None,                      # canonicalized gauge hash: not produced
            record.get("cache_sha256")])
    cache_blocks = [_table_block(cache_header, cache_rows,
                                 table="tbl_cache" if cache_rows else None)]
    cache_blocks.append(_note_block([
        "n/a: no cache artifact was recorded by this run (pyqcu.test --mode "
        "prepare only probes an existing artifact; it never rebuilds one).",
    ]) if not cache_rows else _note_block([
        "Rows come from the exported ledger. gauge_canonical_sha256 is the "
        "reference driver's canonicalized gauge hash and is empty here: the "
        "formal port records the file hash (gauge_file_sha256) and the cache "
        "manifest identity, which the reference schema has no column for - it "
        "is the CSV's cache_identity_sha256 column and the prepare document's "
        "identity_sha256.",
    ]))
    sheets.append(Sheet(
        name="11_Cache", title="11 Cache",
        subtitle=("Cache artifacts recorded by the ledger; the manifest is the "
                  "source of identity and sizes."),
        widths=_widths(cache_header), blocks=cache_blocks,
        tab_color=TAB_COLORS["11_Cache"], merge_columns=11))

    # -- 12_Failures -------------------------------------------------------
    failures = [record for record in records
                if str(record.get("status")) in ("FAIL", "ERROR", "TIMEOUT")]
    failure_rows = []
    for record in failures:
        note = _text(record.get("error") or "")
        notes = record.get("notes")
        if isinstance(notes, (list, tuple)):
            note = "; ".join(str(item) for item in notes)
        name = _text(record.get("tag") or record.get("solver"))
        failure_rows.append([
            _text(record.get("run_id")), name,
            _text(record.get("status")), note])
    sheets.append(Sheet(
        name="12_Failures", title="Failures / 失败与隔离",
        subtitle=(f"{len(failures)} FAIL/ERROR/TIMEOUT row(s); no data "
                  "deleted."),
        widths={"A": 30.0, "B": 40.0, "C": 12.0, "D": 95.0},
        blocks=[_table_block(("ID/run", "Name", "Status", "Note"),
                             failure_rows, kinds={2: "fail"})],
        tab_color=TAB_COLORS["12_Failures"], merge_columns=4))

    # -- 13_Abandoned ------------------------------------------------------
    sheets.append(Sheet(
        name="13_Abandoned", title="Abandoned / 放弃项",
        subtitle=("Abandoned axes are decisions about the informal matrix; the "
                  "formal port has no notion of an abandoned axis."),
        widths={"A": 20.0, "B": 50.0, "C": 55.0, "D": 30.0},
        blocks=[_table_block(("Axis", "Scope", "Retained data", "Decision"), []),
                _note_block(["n/a: not recorded by pyqcu.test."])],
        tab_color=TAB_COLORS["13_Abandoned"], merge_columns=4))

    # -- 14_Input_Integrity ------------------------------------------------
    input_header = ("File", "State", "Size / hash", "Note")
    input_rows: list[list[Any]] = []
    for record in records:
        for role in ("gauge", "source"):
            path = record.get(f"{role}_path")
            if not path:
                continue
            digest = record.get(f"{role}_sha256")
            input_rows.append([
                str(path), _text(record.get("status")),
                ("" if not digest else f"sha256 {digest}"),
                f"{role} input; dtype {_text(record.get(f'{role}_file_dtype'))}"
                if record.get(f"{role}_file_dtype") else f"{role} input"])
    blocks = [_table_block(input_header, input_rows)]
    blocks.append(_note_block([
        "Rule: inputs are read-only; derived writes use temp + os.replace; an "
        "invalid input is rejected (prepare FAIL) instead of reused.",
        "pyqcu.test validates with pyqcu.test.io (dataset/shape/dtype/attrs) and "
        "prewarm hashes the whole file; SHA256 is only present when prewarm ran.",
    ]))
    sheets.append(Sheet(
        name="14_Input_Integrity", title="Input integrity / 输入不可变",
        subtitle=("Input files the ledger declares, their recorded hash and the "
                  "atomic-write rule."),
        widths={"A": 70.0, "B": 12.0, "C": 78.0, "D": 60.0},
        blocks=blocks, tab_color=TAB_COLORS["14_Input_Integrity"],
        merge_columns=4))

    # -- 15_Historical -----------------------------------------------------
    sheets.append(Sheet(
        name="15_Historical", title="Historical / 历史对照（不可控）",
        subtitle=("Historical numbers come from a different driver and a "
                  "different tolerance; they are not part of this ledger."),
        widths={"A": 34.0, "B": 28.0, "C": 70.0, "D": 28.0},
        blocks=[_table_block(("Metric", "Value", "Context", "Comparability"),
                             []),
                _note_block(["n/a: not recorded by pyqcu.test; never divide "
                             "across atol."])],
        tab_color=TAB_COLORS["15_Historical"], merge_columns=4))

    # -- 16_Scripts --------------------------------------------------------
    script_rows = _script_rows(source_dir)
    sheets.append(Sheet(
        name="16_Scripts", title="16 Scripts",
        subtitle=("Identity of the exporter modules that produced this "
                  "workbook (path, size, sha256)."),
        widths={"A": 46.0, "B": 10.0, "C": 12.0, "D": 70.0, "E": 60.0},
        blocks=[_table_block(("path", "exists", "size_bytes", "sha256",
                              "purpose"), script_rows, table="tbl_scripts")],
        tab_color=TAB_COLORS["16_Scripts"], merge_columns=5))

    # -- 17_Data_Dictionary ------------------------------------------------
    dictionary_header = ("column", "meaning", "unit/type", "note")
    dictionary_rows = [
        [name, _column_meaning(name), _column_type(name),
         _dictionary_note(name)]
        for name in REFERENCE_CSV_COLUMNS]
    sheets.append(Sheet(
        name="17_Data_Dictionary", title="17 Data Dictionary",
        subtitle=(f"{len(REFERENCE_CSV_COLUMNS)} CSV columns: meaning, unit "
                  "and the provenance/empty reason of each."),
        widths={"A": 32.0, "B": 60.0, "C": 16.0, "D": 110.0},
        blocks=[_table_block(dictionary_header, dictionary_rows,
                             table="tbl_dict")],
        tab_color=TAB_COLORS["17_Data_Dictionary"], merge_columns=4))

    # -- 18_CSV_Index ------------------------------------------------------
    csv_header = ("file", "rows", "size_bytes", "purpose")
    csv_rows: list[list[Any]] = []
    if csv_path is not None:
        target = Path(str(csv_path))
        size = target.stat().st_size if target.is_file() else None
        csv_rows.append([str(target), len(records), size,
                         "merged all rows (reference 100-column contract)"])
    sheets.append(Sheet(
        name="18_CSV_Index", title="18 CSV Index",
        subtitle=("CSV files this export wrote, with row and byte counts."),
        widths={"A": 70.0, "B": 10.0, "C": 14.0, "D": 55.0},
        blocks=[_table_block(csv_header, csv_rows,
                             table="tbl_csv" if csv_rows else None)],
        tab_color=TAB_COLORS["18_CSV_Index"], merge_columns=4))

    # -- 19_Reports_Index --------------------------------------------------
    reports_header = ("run_id", "status", "solver", "dtype", "lattice",
                      "wall_seconds", "iterations", "final_true_residual",
                      "path")
    report_rows = [
        [record.get("run_id"), record.get("status"), record.get("solver"),
         record.get("dtype"), record.get("lattice"), record.get("wall_seconds"),
         record.get("iterations"), record.get("final_true_residual"),
         record.get("report_path")]
        for record in records if record.get("report_path")]
    report_block = _record_block(reports_header, [
        dict(zip(reports_header, row)) for row in report_rows],
        table="tbl_reports", status_index=1)
    sheets.append(Sheet(
        name="19_Reports_Index", title="19 Reports Index",
        subtitle=(f"{len(report_rows)} per-row reports; row-level status is "
                  "the source of truth."),
        widths=_widths(reports_header), blocks=[report_block],
        tab_color=TAB_COLORS["19_Reports_Index"], status_column="B",
        merge_columns=9))

    # -- 20_Job_Ledger -----------------------------------------------------
    sheets.append(Sheet(
        name="20_Job_Ledger", title="20 Job Ledger",
        subtitle=("Scheduler jobs belong to the informal dsub workflow; the "
                  "formal port runs in-process and has no job id."),
        widths={"A": 12.0, "B": 38.0, "C": 12.0, "D": 20.0, "E": 20.0,
                "F": 10.0, "G": 44.0, "H": 40.0},
        blocks=[_table_block(("job_id", "name", "state", "start", "end", "node",
                              "purpose", "notes"), [], table="tbl_jobs"),
                _note_block(["n/a: not recorded by pyqcu.test."])],
        tab_color=TAB_COLORS["20_Job_Ledger"], merge_columns=8))

    return sheets


def _sheet_row_count(name: str, records: Sequence[Mapping[str, Any]],
                     counts: Mapping[str, Any]) -> Any:
    """``Rows`` cell of ``01_Index``: the data-row count of one sheet."""
    if name == "04_Performance_All":
        return counts["rows"]
    if name in ("05_L24_c64", "06_L24_c128", "07_L32", "08_MG"):
        return len(lattice_sheet_records(name, records))
    if name == "12_Failures":
        return sum(1 for r in records
                   if str(r.get("status")) in ("FAIL", "ERROR", "TIMEOUT"))
    if name == "19_Reports_Index":
        return sum(1 for r in records if r.get("report_path"))
    if name == "17_Data_Dictionary":
        return len(REFERENCE_CSV_COLUMNS)
    if name == "01_Index":
        return len(SHEET_NAMES)
    return None


def _column_meaning(name: str) -> str:
    """Short human meaning of one reference column (from the data dictionary)."""
    meanings = {
        "run_id": "unique run identifier / row id",
        "phase": "solve / prepare / prewarm / validate",
        "solver": "algorithm of the case (or cache_build / input_prewarm)",
        "dtype": "c64 / c128", "atol": "solver tolerance",
        "lattice": "Lx,Ly,Lz,Lt", "mass": "fermion mass",
        "csw": "clover coefficient", "kappa": "hopping parameter",
        "ranks": "MPI ranks", "threads": "OMP threads",
        "rank_x_thread": "rank x thread label", "grid": "4D rank grid",
        "grid_mode": "balanced / t-finest",
        "shm": "shared-halo flag 0/1", "source_kind": "source generator",
        "source_path": "source artifact path",
        "source_file_dtype": "source dataset dtype",
        "source_sha256": "source file hash",
        "gauge_kind": "gauge generator", "gauge_path": "gauge artifact path",
        "gauge_file_dtype": "gauge dataset dtype",
        "gauge_sha256": "gauge file hash",
        "cache_kind": "global / ranked", "cache_path": "cache artifact path",
        "cache_sha256": "cache file hash",
        "cache_identity_sha256": "cache identity hash",
        "cache_manifest_path": "cache manifest path",
        "cache_manifest_sha256": "cache manifest hash",
        "mg_levels": "multigrid levels", "mg_restart": "MG restart",
        "mg_grid": "MG block grid", "mg_dof": "MG dof vector",
        "nv_iters": "null-vector iterations",
        "coarse_max_iter": "coarse solver iteration cap",
        "coarse_tol_factor": "coarse tolerance factor",
        "deflate": "deflation flag", "max_iter": "iteration cap",
        "verify_status": "cache verification status", "iterations": "solver iterations",
        "wall_seconds": "solver wall time", "builder_seconds": "cache load/build time",
        "build_mg_cache_seconds": "cache build time",
        "final_true_residual": "true residual",
        "relative_full_residual": "relative full residual",
        "distributed_relative_residual": "relative distributed residual",
        "residual_source": "residual provenance", "converged": "solver converged flag",
        "status": "verdict", "node": "compute node", "freq_khz": "current CPU frequency",
        "freq_max_khz": "max CPU frequency", "governor": "cpufreq governor",
        "profile_rank": "rank the profile row was taken from",
        "gauge_read_seconds": "gauge HDF5 read (in solver)",
        "source_read_seconds": "source HDF5 read (in solver)",
        "gauge_generate_seconds": "gauge generation time",
        "source_generate_seconds": "source generation time",
        "gauge_prewarm_seconds": "gauge prewarm read", "source_prewarm_seconds": "source prewarm read",
        "gauge_read_mode": "rank-block HDF5 reader: mpio / slab",
        "source_read_mode": "rank-block HDF5 reader: mpio / slab",
        "hdf5_mpio_read": "MPI HDF5 read enabled", "residual_seconds": "residual time",
        "report_path": "driver report path", "report_sha256": "report hash",
        "solution_path": "solution file path", "solution_sha256": "solution hash",
        "out_path": "stdout log path", "err_path": "stderr log path",
        "returncode": "driver return code", "elapsed_seconds": "whole row elapsed",
        "review": "review threshold exceeded", "timeout": "timeout flag",
        "error": "error text", "notes": "row notes",
    }
    if name in meanings:
        return meanings[name]
    if name.endswith("_gbps_rank"):
        return f"{name.split('_gbps')[0]} effective bandwidth per rank"
    if name.endswith("_gbps_total"):
        return f"{name.split('_gbps')[0]} effective bandwidth total"
    if name.startswith("halo_"):
        return f"halo {name.split('_', 1)[1]} time"
    if name.startswith("phase_") and name.endswith("_seconds"):
        return f"phase {name[len('phase_'):-len('_seconds')]} time"
    return name.replace("_", " ")


def _column_type(name: str) -> str:
    """Unit/type of one reference column."""
    if name in ("atol", "mass", "csw", "kappa", "wall_seconds",
                "builder_seconds", "build_mg_cache_seconds", "elapsed_seconds",
                "gauge_read_seconds", "source_read_seconds",
                "gauge_generate_seconds", "source_generate_seconds",
                "gauge_prewarm_seconds", "source_prewarm_seconds",
                "input_prebind_seconds", "residual_seconds",
                "phase_prepare_seconds", "phase_solve_seconds",
                "phase_reconstruct_seconds", "halo_pack_seconds",
                "halo_sync_seconds", "halo_copy_seconds", "dslash_seconds",
                "clover_seconds", "reduce_seconds"):
        return "float s" if name.endswith("_seconds") and name != "atol" else "float"
    if name in ("final_true_residual", "relative_full_residual",
                "distributed_relative_residual", "dslash_gbps_rank",
                "dslash_gbps_total", "clover_gbps_rank", "clover_gbps_total",
                "dslash_bytes_mb", "clover_bytes_mb"):
        return "float"
    if name in ("ranks", "threads", "shm", "seed", "mg_levels", "mg_restart",
                "nv_iters", "coarse_max_iter", "max_iter", "iterations",
                "verify_status", "profile_rank", "freq_khz",
                "freq_max_khz", "returncode", "lat_x", "lat_y", "lat_z",
                "lat_t", "dslash_calls", "clover_calls", "reduce_calls"):
        return "int"
    if name in ("deflate", "converged", "hdf5_mpio_read", "review", "timeout"):
        return "bool"
    if name in ("lattice", "grid", "mg_grid", "mg_dof", "notes"):
        return "list"
    return "string"


def _column_source(name: str) -> str:
    """``row``/``option``/``derived``/``missing`` of one column."""
    for rule in ledger_csv.COLUMN_RULES:
        if rule.name == name:
            return rule.source
    return "missing"


def _dictionary_note(name: str) -> str:
    """Provenance tag plus the per-column explanation, in one note cell.

    The reference's data-dictionary sheet has four columns (column / meaning /
    unit-type / note), so the provenance lives at the head of the note
    (``[derived] ...``) instead of in a fifth column - that keeps the sheet's
    header exactly the reference's while still saying where a value comes from.
    """
    source = _column_source(name)
    note = column_note(name).strip()
    return f"[{source}] {note}" if note else f"[{source}]"


def _script_rows(source_dir: Any) -> list[list[Any]]:
    """``path``/``exists``/``size``/``sha256``/``purpose`` of the exporter."""
    package = Path(__file__).resolve().parent
    purposes = {
        "ledger_csv.py": "reference 100-column CSV exporter",
        "ledger_excel.py": "reference workbook exporter",
        "common.py": "full-mode rows and ledger",
        "__main__.py": "CLI (integration point)",
    }
    rows: list[list[Any]] = []
    for name, purpose in purposes.items():
        path = package / name
        if not path.is_file():
            rows.append([str(path), False, None, None, purpose])
            continue
        try:
            data = path.read_bytes()
        except OSError:
            rows.append([str(path), True, None, None, purpose])
            continue
        rows.append([str(path), True, len(data),
                     hashlib.sha256(data).hexdigest(), purpose])
    return rows


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------
def write_ledger_excel(path: Any, records: Sequence[Mapping[str, Any]], *,
                       payload: Mapping[str, Any] | None = None,
                       csv_path: Any = None,
                       generated: str | None = None,
                       properties: Mapping[str, str] | None = None,
                       created: str | None = None) -> Path:
    """Write the reference workbook for ``records`` and return the path.

    ``records`` are :mod:`pyqcu.test.ledger_csv` records; ``payload`` is the
    full-mode document (optional, used for the cover's provenance line);
    ``csv_path`` is reported by ``18_CSV_Index``.  ``generated``/``created`` are
    opt-in stamps: leaving them out keeps the export byte-identical across
    machines, which is what a re-run comparison needs.
    """
    target = Path(str(path))
    sheets = build_sheets(records, payload=payload, csv_path=csv_path,
                          workbook_name=target.name, generated=generated)
    return write_workbook(target, sheets, properties=properties, created=created)
