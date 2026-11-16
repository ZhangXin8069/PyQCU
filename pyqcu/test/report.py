"""Report exporters for the formal :mod:`pyqcu.test` port (stdlib only).

The harness produces one :class:`pyqcu.test.common.TestReport`; this module
turns it (or the equivalent ``TestReport.as_dict()`` mapping) into the formats
a CI system or a code review consumes.  It is a *pure formatter*: it never
runs a case, never re-decides a verdict, never re-samples a timing and never
changes a number.

Input contract
--------------
``report`` may be a :class:`~pyqcu.test.common.TestReport`, the mapping from
its ``as_dict()``, or any object with an ``as_dict()`` method returning such a
mapping.  ``report["cases"]`` may hold ``Result`` objects, mappings, or objects
with ``as_dict()``.  Missing *header* fields are derived: the aggregate
``status`` and ``counts`` come from :func:`pyqcu.test.common.summarize`, the
``exit_code`` from :func:`pyqcu.test.policy.exit_code`, and a lattice/dtype/
seed left out of the header is taken from the first case that carries one.  A
missing *case* status is a ``ValueError``: a case without a verdict is not a
result, and inventing one would turn a malformed report into a green one.

Formats
-------
============  =============================================================
``json``      the report's own payload (``TestReport.as_dict()``, sorted)
``csv``       one row per case, the stable :data:`CSV_COLUMNS` header
``junit``     JUnit XML: ``testsuites/testsuite``, one ``testcase`` per case
``markdown``  human summary plus a per-case table
============  =============================================================

Guarantees
----------
* **Deterministic**: no timestamp, no absolute path, no ordering by timing;
  identical input renders byte-identical output.
* **Text-safe**: XML attributes are quoted and text is escaped, and characters
  XML 1.0 forbids (most C0 controls, stray surrogates) are dropped, so a
  native log line cannot corrupt a JUnit file.  Markdown table cells escape
  ``|`` and CSV cells never contain a raw newline: one physical line per
  record, one record per case.
* **Newlines**: every format ends with exactly one ``\\n``, and the CSV writer
  is pinned to ``lineterminator="\\n"`` with ``StringIO(newline="")`` (RFC 4180
  CRLF is *not* emitted) so a report diff is stable across platforms.
* **Atomic writes**: :func:`write_report` writes a sibling temporary file,
  ``fsync`` s it, renames it onto the target with :func:`os.replace` and then
  best-effort ``fsync`` s the directory, so a reader sees the previous report or
  the complete new report - never a half-written file.  A failed write leaves
  no ``.tmp`` behind.
* **No performance claims**: ``seconds`` is a wall-clock diagnostic of this
  run.  The markdown output says so explicitly, and no exporter ranks cases by
  time, computes speed-ups or combines measurements into a conclusion.

Wiring for the CLI (``--junit`` / ``--csv`` / ``--markdown``)::

    from .report import render, write_report

    write_report(path, report, "junit")     # explicit format, atomic
    write_report(path, report)              # format inferred from the suffix
    sys.stdout.write(render(report, "csv"))  # a '-' target is the CLI's job
    sys.stdout.write(render(report, "json"))  # the pure TestReport payload

Import rules: standard library (``csv``, ``io``, ``json``, ``os``, ``re``,
``xml.sax.saxutils``) plus :mod:`pyqcu.test.policy` and the two pure helpers
``json_safe``/``summarize`` of :mod:`pyqcu.test.common`.  No ``torch``,
``numpy``, ``mpi4py``, ``pytest`` and no ``pyqcu.testing``.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import xml.sax.saxutils as _saxutils
from collections.abc import Mapping, Sequence
from itertools import count
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .. import __version__ as _PACKAGE_VERSION
from . import policy
from .common import json_safe, summarize

__all__ = [
    "CSV_COLUMNS",
    "FORMATS",
    "FORMAT_ALIASES",
    "SUFFIX_FORMATS",
    "infer_format",
    "normalize_format",
    "render",
    "to_csv",
    "to_json",
    "to_junit",
    "to_markdown",
    "write_report",
]

#: Canonical format names accepted by :func:`render` and :func:`write_report`.
FORMATS: tuple[str, ...] = ("json", "csv", "junit", "markdown")

#: Accepted spellings (case-insensitive, a leading dot is ignored) -> format.
FORMAT_ALIASES: dict[str, str] = {
    "json": "json",
    "csv": "csv",
    "junit": "junit",
    "junitxml": "junit",
    "xml": "junit",
    "markdown": "markdown",
    "md": "markdown",
}

#: File suffix -> format, used when :func:`write_report` gets no ``format``.
SUFFIX_FORMATS: dict[str, str] = {
    ".json": "json",
    ".csv": "csv",
    ".xml": "junit",
    ".junit": "junit",
    ".md": "markdown",
    ".markdown": "markdown",
}

#: CSV header, in this order.  Renaming a column is a breaking change for a
#: downstream spreadsheet or ``awk``, so the list is part of the API.
CSV_COLUMNS: tuple[str, ...] = (
    "tag",
    "backend",
    "kind",
    "status",
    "seconds",
    "wall_seconds",
    "prepare_seconds",
    "solve_seconds",
    "samples",
    "seconds_min",
    "seconds_max",
    "lattice",
    "dtype",
    "seed",
    "mpi_size",
    "iterations",
    "relative_error",
    "relative_residual",
    "reason",
    "skip_token",
)

#: Alternative spellings of the three numeric metrics, as the backend runners
#: and the legacy ``pyqcu/testing`` reports write them.  ``relative_error`` is
#: deliberately **not** an alias of ``relative_residual`` here: for a dslash
#: case it is the operator difference, and copying it into the residual column
#: would claim a solver convergence that was never measured.  (The gate
#: evaluator accepts that alias because it compares a named gate's own value;
#: a report column is a different question.)
_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "iterations": ("iters", "iteration"),
    "relative_error": ("relative",),
    "relative_residual": ("rel_res", "true_residual"),
}

#: Characters XML 1.0 allows: tab, LF, CR, #x20-#xD7FF, #xE000-#xFFFD and
#: #x10000-#x10FFFF.  Everything else is dropped, never escaped: there is no
#: entity for U+0001 and a consumer would reject the document.
_INVALID_XML = re.compile(
    "[^\u0009\u000a\u000d\u0020-\ud7ff\ue000-\ufffd"
    "\U00010000-\U0010ffff]")

#: Process-local counter making the temporary file name unique per write.
_TEMPORARY = count()


# ---------------------------------------------------------------------------
# text and number formatting
# ---------------------------------------------------------------------------
def _clean_text(value: Any) -> str:
    """Return ``value`` as text with XML-illegal characters removed.

    ``None`` becomes an empty string, ``bytes`` are decoded with replacement
    and CRLF/CR line ends are normalised to LF.  The same cleaner feeds all
    three text formats so a value is never legal in one exporter and fatal in
    another.
    """
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray)):
        text = bytes(value).decode("utf-8", "replace")
    else:
        text = str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return _INVALID_XML.sub("", text)


def _collapsed(text: str, limit: int | None = None) -> str:
    """Collapse whitespace into single spaces and truncate to ``limit``."""
    collapsed = " ".join(text.split())
    if limit is not None and len(collapsed) > limit:
        collapsed = collapsed[: max(limit - 3, 0)] + "..."
    return collapsed


def _plain_text(value: Any, limit: int | None = None) -> str:
    """One-line, XML-safe rendering of any value."""
    return _collapsed(_clean_text(value), limit)


def _as_float(value: Any) -> float | None:
    """Return ``value`` as a float, or ``None`` when it is not a number.

    Booleans are not numbers here (``True`` must not become ``1.0``), and an
    unparsable string (``"n/a"``) is reported as absent instead of raising:
    a report exporter must never fail because a diagnostic field is odd.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    """Return ``value`` as an int when it is an integral number, else ``None``."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    number = _as_float(value)
    if number is None or not float(number).is_integer():
        return None
    return int(number)


def _format_seconds(value: Any) -> str:
    """Machine-readable seconds (6 decimals; empty when absent)."""
    number = _as_float(value)
    return "" if number is None else f"{number:.6f}"


def _format_seconds_human(value: Any) -> str:
    """Human-readable seconds for markdown (3 decimals, no false precision)."""
    number = _as_float(value)
    return "" if number is None else f"{number:.3f}"


def _format_float(value: Any) -> str:
    """Scientific notation for a relative error/residual (empty when absent)."""
    number = _as_float(value)
    return "" if number is None else f"{number:.6e}"


def _format_int(value: Any) -> str:
    """Integer text (empty when absent, never ``3.0``)."""
    number = _as_int(value)
    return "" if number is None else str(number)


def _lattice_text(value: Any) -> str:
    """Render a lattice as ``8x8x8x16`` (a string passes through)."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "x".join(_plain_text(item) for item in value)
    return _plain_text(value)


def _status(value: Any) -> str:
    """Normalise and validate one status word against :data:`policy.STATUSES`."""
    text = "" if value is None else str(value).strip().upper()
    if text not in policy.STATUSES:
        raise ValueError(
            f"unknown status {value!r}; choose from "
            f"{', '.join(policy.STATUSES)}")
    return text


def _tag_segment(tag: str, index: int) -> str:
    """Return segment ``index`` of a ``backend/area/variant`` tag (or ``""``)."""
    parts = tag.split("/")
    return parts[index].strip() if len(parts) > index else ""


# ---------------------------------------------------------------------------
# normalisation: TestReport / mapping -> one plain dict
# ---------------------------------------------------------------------------
def _report_mapping(report: Any) -> Mapping[str, Any]:
    """Return the report payload as a mapping (never mutating the input)."""
    if isinstance(report, Mapping):
        return report
    method = getattr(report, "as_dict", None)
    if callable(method):
        payload = method()
        if isinstance(payload, Mapping):
            return payload
        raise ValueError(
            f"as_dict() returned {type(payload).__name__}, expected a mapping")
    raise ValueError(
        "report must be a TestReport, a mapping, or an object with as_dict(); "
        f"got {type(report).__name__}")


def _report_cases(report: Mapping[str, Any]) -> Sequence[Any]:
    """Return ``report["cases"]`` as a sequence (empty when absent)."""
    cases = report.get("cases")
    if cases is None:
        return ()
    if isinstance(cases, (str, bytes, bytearray, Mapping)):
        raise ValueError(
            "report['cases'] must be a sequence of case records, got "
            f"{type(cases).__name__}")
    if not isinstance(cases, Sequence):
        raise ValueError(
            "report['cases'] must be a sequence of case records, got "
            f"{type(cases).__name__}")
    return cases


def _case_mapping(case: Any) -> Mapping[str, Any]:
    """Return one case record as a mapping."""
    if isinstance(case, Mapping):
        return case
    method = getattr(case, "as_dict", None)
    if callable(method):
        payload = method()
        if isinstance(payload, Mapping):
            return payload
        raise ValueError(
            f"case as_dict() returned {type(payload).__name__}, expected a "
            "mapping")
    raise ValueError(
        "case record must be a mapping or an object with as_dict(); got "
        f"{type(case).__name__}")


def _metric(record: Mapping[str, Any], metrics: Mapping[str, Any],
            name: str) -> Any:
    """Read ``name`` from ``metrics``, honouring aliases and flat records.

    Flat records (a row that carries ``iterations`` at the top level instead
    of under ``metrics``) are accepted because the same writer is pointed at
    hand-built mappings in tests and at ``Result.as_dict()`` in production.
    """
    keys = (name, *_METRIC_ALIASES.get(name, ()))
    for key in keys:
        if key in metrics:
            return metrics[key]
    for key in keys:
        if key in record:
            return record[key]
    return None


def _failed(value: Any) -> tuple[str, ...]:
    """Normalise the ``failed`` gate list into a tuple of one-line strings."""
    if value is None:
        return ()
    if isinstance(value, str):
        text = _plain_text(value)
        return (text,) if text else ()
    if isinstance(value, (list, tuple)):
        items = (_plain_text(item) for item in value)
        return tuple(item for item in items if item)
    text = _plain_text(value)
    return (text,) if text else ()


def _reason_and_token(record: Mapping[str, Any],
                      status: str) -> tuple[str, str]:
    """Return ``(reason, skip_token)``, splitting ``"token: reason"`` on SKIP.

    :meth:`Result.__post_init__` already stores the token separately, but a
    report that travelled through JSON or was built by hand may only carry the
    ``"cuda_bridge: ..."`` prefix.  Splitting it here keeps the CSV ``reason``
    and ``skip_token`` columns disjoint without inventing a token.
    """
    reason = _plain_text(record.get("reason"))
    token = _plain_text(record.get("skip_token"))
    if status == "SKIP" and not token:
        head, separator, tail = reason.partition(":")
        if separator and policy.is_skip_token(head.strip()) and tail.strip():
            return tail.strip(), head.strip()
    return reason, token


def _case_record(case: Any) -> dict[str, Any]:
    """Normalise one case (``Result``, mapping or ``as_dict()`` object)."""
    raw = _case_mapping(case)
    tag = _plain_text(raw.get("tag"))
    status = _status(raw.get("status"))
    metrics = raw.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    reason, skip_token = _reason_and_token(raw, status)
    return {
        "tag": tag,
        "backend": _plain_text(raw.get("backend")) or _tag_segment(tag, 0),
        "kind": _plain_text(raw.get("kind")) or _tag_segment(tag, 1),
        "status": status,
        "seconds": raw.get("seconds"),
        "wall_seconds": (_metric(raw, metrics, "wall_seconds")
                         if _metric(raw, metrics, "wall_seconds") is not None
                         else _metric(raw, metrics, "elapsed_seconds")),
        "prepare_seconds": _metric(raw, metrics, "prepare_seconds"),
        "solve_seconds": _metric(raw, metrics, "solve_seconds"),
        "samples": (_metric(raw, metrics, "samples")
                    if _metric(raw, metrics, "samples") is not None
                    else _metric(raw, metrics, "repeat")),
        "seconds_min": _metric(raw, metrics, "seconds_min"),
        "seconds_max": _metric(raw, metrics, "seconds_max"),
        "lattice": raw.get("lattice"),
        "dtype": _plain_text(raw.get("dtype")),
        "seed": raw.get("seed"),
        "mpi_size": raw.get("mpi_size"),
        "iterations": _metric(raw, metrics, "iterations"),
        "relative_error": _metric(raw, metrics, "relative_error"),
        "relative_residual": _metric(raw, metrics, "relative_residual"),
        "reason": reason,
        "skip_token": skip_token,
        "failed": _failed(raw.get("failed")),
        "runner": _plain_text(raw.get("runner")),
    }


def _first_case_value(cases: Sequence[Mapping[str, Any]],
                      key: str) -> Any:
    """First non-empty ``cases[*][key]`` (``None`` when no case carries one)."""
    for case in cases:
        value = case.get(key)
        if value is not None and value != "":
            return value
    return None


def _counts_and_status(raw_counts: Any, raw_status: Any,
                       cases: Sequence[Mapping[str, Any]]
                       ) -> tuple[dict[str, int], str]:
    """Return the aggregate counts and status.

    ``common.summarize`` owns the precedence (``ERROR`` > ``FAIL`` > ``TIMEOUT``
    > all-``SKIP`` > ``PASS``); it is called with status-only shims so the
    report cannot drift from the runner's own aggregation.  An explicit
    ``counts``/``status`` in the payload wins, with missing count keys filled
    from the per-case records.
    """
    derived_counts, derived_status = summarize(
        tuple(SimpleNamespace(status=case["status"]) for case in cases))
    counts = dict(derived_counts)
    if isinstance(raw_counts, Mapping):
        for name in policy.STATUSES:
            value = _as_int(raw_counts.get(name))
            if value is not None:
                counts[name] = value
    status = derived_status if raw_status is None else _status(raw_status)
    return counts, status


def _report_dict(report: Any) -> dict[str, Any]:
    """Normalise a report into the single dict every text exporter reads.

    This is the projection the CSV/JUnit/markdown writers share: the header
    fields they display (with derived fallbacks), the aggregate counts and one
    flat record per case built by :func:`_case_record`.  Every text value is
    already cleaned here, so a renderer never touches a raw input object, and
    two renderers can never disagree about a status or a count.
    """
    raw = _report_mapping(report)
    cases = tuple(_case_record(case) for case in _report_cases(raw))
    counts, status = _counts_and_status(raw.get("counts"), raw.get("status"),
                                        cases)

    all_skipped = raw.get("all_skipped")
    if all_skipped is None:
        all_skipped = bool(cases) and counts["SKIP"] == len(cases)
    allow_all_skipped = bool(raw.get("allow_all_skipped", False))

    exit_code = _as_int(raw.get("exit_code"))
    if exit_code is None:
        exit_code = policy.exit_code(
            status, all_skipped=bool(all_skipped),
            allow_all_skipped=allow_all_skipped)

    duration = _as_float(raw.get("duration_seconds"))
    lattice = raw.get("lattice")
    if lattice is None:
        lattice = _first_case_value(cases, "lattice")
    seed = raw.get("seed")
    if seed is None:
        seed = _first_case_value(cases, "seed")
    mpi_size = raw.get("mpi_size")
    if mpi_size is None:
        mpi_size = _first_case_value(cases, "mpi_size")
    dtype = _plain_text(raw.get("dtype")) or _plain_text(
        _first_case_value(cases, "dtype"))

    return {
        "package": _plain_text(raw.get("package")) or policy.PACKAGE,
        "version": _plain_text(raw.get("version")) or _PACKAGE_VERSION,
        "status": status,
        "exit_code": exit_code,
        "lattice": lattice,
        "dtype": dtype,
        "seed": seed,
        "mpi_size": mpi_size,
        "mpi_rank": _as_int(raw.get("mpi_rank")) or 0,
        "counts": counts,
        "all_skipped": bool(all_skipped),
        "allow_all_skipped": allow_all_skipped,
        "duration_seconds": None if duration is None else round(duration, 6),
        "cases": cases,
    }


# ---------------------------------------------------------------------------
# shared one-line renderings
# ---------------------------------------------------------------------------
def _skip_detail(case: Mapping[str, Any]) -> str:
    """``token: reason`` for a skipped case, never an empty string."""
    token = case["skip_token"]
    reason = case["reason"]
    if token and reason:
        return f"{token}: {reason}"
    return token or reason or "(no reason recorded)"


def _case_detail(case: Mapping[str, Any]) -> str:
    """One-line detail: skip token/reason, failure text, or PASS metrics."""
    status = case["status"]
    if status == "SKIP":
        return _skip_detail(case)
    if status in ("FAIL", "ERROR", "TIMEOUT"):
        return case["reason"] or "; ".join(case["failed"])
    parts: list[str] = []
    iterations = _format_int(case["iterations"])
    if iterations:
        parts.append(f"iterations={iterations}")
    relative_error = _format_float(case["relative_error"])
    if relative_error:
        parts.append(f"relative_error={relative_error}")
    relative_residual = _format_float(case["relative_residual"])
    if relative_residual:
        parts.append(f"relative_residual={relative_residual}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------
def _json_text(report: Any, *, indent: int | None = 2) -> str:
    """Serialise the report payload itself (not the normalised projection).

    JSON is the machine-readable contract: ``to_json(TestReport)`` must equal
    the ``--json`` payload byte for byte, including the ``options``/``env``
    blocks and the CLI-added keys.  Only case *objects* are normalised (a
    mapping may hold ``Result`` instances); everything else is passed through
    :func:`pyqcu.test.common.json_safe` unchanged, so this exporter can never
    silently drop a field a reader depends on.
    """
    payload = _report_mapping(report)
    if "cases" in payload:
        cases = [_case_mapping(case) for case in _report_cases(payload)]
        payload = {**payload, "cases": cases}
    return json.dumps(json_safe(payload), indent=indent, sort_keys=True,
                      ensure_ascii=False, allow_nan=False) + "\n"


def to_json(report: Any, *, indent: int | None = 2) -> str:
    """Render ``report``'s own payload as JSON (``TestReport.as_dict()``)."""
    return _json_text(report, indent=indent)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
def _csv_cell(case: Mapping[str, Any], column: str) -> str:
    """Render one CSV cell; every value is a single physical line."""
    value = case.get(column)
    if column == "lattice":
        return _lattice_text(value)
    if column in ("seconds", "wall_seconds", "prepare_seconds",
                  "solve_seconds", "seconds_min", "seconds_max"):
        return _format_seconds(value)
    if column in ("seed", "mpi_size", "iterations", "samples"):
        return _format_int(value)
    if column in ("relative_error", "relative_residual"):
        return _format_float(value)
    return _plain_text(value)


def _csv_text(data: Mapping[str, Any]) -> str:
    """Render the CSV table (header plus one row per case)."""
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for case in data["cases"]:
        writer.writerow([_csv_cell(case, column) for column in CSV_COLUMNS])
    return buffer.getvalue()


def to_csv(report: Any) -> str:
    """Render ``report`` as CSV with the stable :data:`CSV_COLUMNS` header.

    One row per case, ``\\n`` line endings, ``csv`` quoting rules.  The
    ``lattice`` column is ``8x8x8x16`` (not ``[8, 8, 8, 16]``), numeric columns
    use fixed precision (``%.6f`` seconds, ``%.6e`` relative values) and an
    absent value is the empty string, so ``csv.DictReader`` gets a rectangular
    table.  A SKIP's reason and token are separate columns; the reason never
    repeats the ``"token: "`` prefix.
    """
    return _csv_text(_report_dict(report))


# ---------------------------------------------------------------------------
# JUnit XML
# ---------------------------------------------------------------------------
def _xml_attributes(pairs: Sequence[tuple[str, Any]]) -> str:
    """Render ``name="value"`` pairs with :func:`xml.sax.saxutils.quoteattr`."""
    return "".join(f" {name}={_saxutils.quoteattr(_plain_text(value, 200))}"
                   for name, value in pairs)


def _xml_body(value: Any) -> str:
    """Escape element text and keep it on one physical line."""
    return _saxutils.escape(_clean_text(value)).replace("\n", "&#10;")


def _junit_properties(data: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Suite-level metadata that explains the numbers in the XML."""
    candidates = (
        ("package", data["package"]),
        ("version", data["version"]),
        ("status", data["status"]),
        ("exit_code", str(data["exit_code"])),
        ("lattice", _lattice_text(data["lattice"])),
        ("dtype", data["dtype"]),
        ("seed", _format_int(data["seed"])),
        ("mpi_size", _format_int(data["mpi_size"])),
        ("mpi_rank", str(data["mpi_rank"])),
        ("duration_seconds", _format_seconds(data["duration_seconds"])),
        ("all_skipped", "true" if data["all_skipped"] else "false"),
        ("allow_all_skipped",
         "true" if data["allow_all_skipped"] else "false"),
    )
    return [(name, value) for name, value in candidates if value]


def _junit_message(case: Mapping[str, Any]) -> str:
    """The ``message=`` attribute of a failure/error/skipped element."""
    status = case["status"]
    if status == "SKIP":
        return _plain_text(_skip_detail(case), 200)
    if status in ("FAIL", "ERROR", "TIMEOUT"):
        return _plain_text(case["reason"] or "; ".join(case["failed"])
                           or status.lower(), 200)
    return ""


def _junit_detail(case: Mapping[str, Any]) -> str:
    """The text content of a failure/error element (multi-line, escaped)."""
    lines = [f"tag: {case['tag']}", f"status: {case['status']}"]
    if case["backend"]:
        lines.append(f"backend: {case['backend']}")
    if case["kind"]:
        lines.append(f"kind: {case['kind']}")
    if case["runner"]:
        lines.append(f"runner: {case['runner']}")
    if case["reason"]:
        lines.append(f"reason: {case['reason']}")
    for item in case["failed"]:
        lines.append(f"failed gate: {item}")
    for name, value in (("iterations", _format_int(case["iterations"])),
                        ("relative_error",
                         _format_float(case["relative_error"])),
                        ("relative_residual",
                         _format_float(case["relative_residual"]))):
        if value:
            lines.append(f"{name}: {value}")
    return "\n".join(lines)


def _junit_case(case: Mapping[str, Any]) -> list[str]:
    """Render one ``testcase`` element as indented XML lines.

    ``PASS`` is an empty element; ``FAIL`` is ``<failure>``; ``ERROR`` and
    ``TIMEOUT`` are ``<error>`` (JUnit has no timeout element, and a timeout is
    a harness-level error, not a numerical failure); ``SKIP`` is ``<skipped>``.
    """
    classname = f"pyqcu.test.{case['backend'] or 'unknown'}"
    attributes = _xml_attributes((
        ("classname", classname),
        ("name", case["tag"] or "(untagged)"),
        ("time", _format_seconds(case["seconds"]) or "0.000000"),
    ))
    status = case["status"]
    if status == "PASS":
        return [f"    <testcase{attributes}/>"]
    if status == "SKIP":
        message = _xml_attributes((("message", _junit_message(case)),))
        return [f"    <testcase{attributes}>",
                f"      <skipped{message}/>",
                "    </testcase>"]
    element = "failure" if status == "FAIL" else "error"
    details = _xml_attributes((("message", _junit_message(case)),
                               ("type", status)))
    body = _xml_body(_junit_detail(case))
    return [f"    <testcase{attributes}>",
            f"      <{element}{details}>{body}</{element}>",
            "    </testcase>"]


def _junit_text(data: Mapping[str, Any]) -> str:
    """Render the whole JUnit document."""
    cases = data["cases"]
    counts = data["counts"]
    errors = int(counts["ERROR"]) + int(counts["TIMEOUT"])
    duration = data["duration_seconds"]
    if duration is None:
        duration = sum(_as_float(case["seconds"]) or 0.0 for case in cases)
    suite = _xml_attributes((
        ("name", "pyqcu.test"),
        ("tests", str(len(cases))),
        ("failures", str(counts["FAIL"])),
        ("errors", str(errors)),
        ("skipped", str(counts["SKIP"])),
        ("time", f"{float(duration):.6f}"),
    ))
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             f"<testsuites{suite}>",
             f"  <testsuite{suite}>"]
    properties = _junit_properties(data)
    if properties:
        lines.append("    <properties>")
        for name, value in properties:
            pair = _xml_attributes((("name", name), ("value", value)))
            lines.append(f"      <property{pair}/>")
        lines.append("    </properties>")
    for case in cases:
        lines.extend(_junit_case(case))
    lines.extend(("  </testsuite>", "</testsuites>"))
    return "\n".join(lines) + "\n"


def to_junit(report: Any) -> str:
    """Render ``report`` as a self-contained JUnit XML document.

    The root is ``<testsuites>`` with a single ``<testsuite name="pyqcu.test">``
    so both one-suite and multi-suite consumers read it; ``tests``/``failures``/
    ``errors``/``skipped``/``time`` are carried on both elements.  Each case is
    one ``<testcase classname="pyqcu.test.<backend>" name="<tag>" time="...">``,
    with ``<failure>`` for ``FAIL``, ``<error type="ERROR|TIMEOUT">`` for
    ``ERROR``/``TIMEOUT`` and ``<skipped message="token: reason">`` for ``SKIP``.
    Text and attributes are escaped and XML-illegal characters are dropped; no
    external entity, DTD or dependency is involved.
    """
    return _junit_text(_report_dict(report))


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def _md_cell(value: Any, limit: int | None = None) -> str:
    """Escape one markdown table cell (``|`` would end the cell)."""
    return _plain_text(value, limit).replace("|", "\\|")


def _markdown_fields(data: Mapping[str, Any]) -> list[tuple[str, str]]:
    """The summary table rows (only fields the report actually carries)."""
    fields: list[tuple[str, str]] = []
    package = data["package"]
    if package:
        version = f" {data['version']}" if data["version"] else ""
        fields.append(("package", f"`{package}{version}`"))
    lattice = _lattice_text(data["lattice"])
    if lattice:
        fields.append(("lattice", f"`{lattice}`"))
    if data["dtype"]:
        fields.append(("dtype", f"`{_md_cell(data['dtype'])}`"))
    seed = _format_int(data["seed"])
    if seed:
        fields.append(("seed", f"`{seed}`"))
    size = _format_int(data["mpi_size"])
    if size:
        fields.append(("MPI size / rank", f"`{size}` / `{data['mpi_rank']}`"))
    fields.append(("exit code", f"`{data['exit_code']}`"))
    fields.append(("all skipped",
                   f"`{'true' if data['all_skipped'] else 'false'}`"))
    return fields


def _markdown_text(data: Mapping[str, Any]) -> str:
    """Render the markdown report (summary, case table, skips, notes)."""
    cases = data["cases"]
    counts = data["counts"]
    total = len(cases)
    lines = ["# pyqcu.test report", ""]
    headline = (
        f"**Status `{data['status']}`** - {total} case(s): "
        f"{counts['PASS']} PASS, {counts['FAIL']} FAIL, "
        f"{counts['ERROR']} ERROR, {counts['TIMEOUT']} TIMEOUT, "
        f"{counts['SKIP']} SKIP")
    duration = _format_seconds_human(data["duration_seconds"])
    if duration:
        headline += f", {duration} s wall clock"
    headline += f", exit code {data['exit_code']}."
    lines.extend((headline, ""))

    fields = _markdown_fields(data)
    if fields:
        lines.extend(("| field | value |", "|---|---|"))
        lines.extend(f"| {name} | {value} |" for name, value in fields)
        lines.append("")

    lines.extend(("## Cases", ""))
    if cases:
        lines.extend((
            "| # | tag | backend | kind | status | core | wall | prep | solve | detail |",
            "|---:|---|---|---|---|---:|---:|---:|---:|---|",
        ))
        for index, case in enumerate(cases, 1):
            cells = (
                str(index),
                f"`{_md_cell(case['tag'])}`",
                _md_cell(case["backend"]) or "-",
                _md_cell(case["kind"]) or "-",
                _md_cell(case["status"]),
                _format_seconds_human(case["seconds"]) or "-",
                _format_seconds_human(case["wall_seconds"]) or "-",
                _format_seconds_human(case["prepare_seconds"]) or "-",
                _format_seconds_human(case["solve_seconds"]) or "-",
                _md_cell(_case_detail(case), 160) or "-",
            )
            lines.append("| " + " | ".join(cells) + " |")
    else:
        lines.append("_No case was selected._")

    skipped = [case for case in cases if case["status"] == "SKIP"]
    if skipped:
        lines.extend(("", "## Skipped", ""))
        for case in skipped:
            lines.append(f"- `{_md_cell(case['tag'])}` - "
                         f"{_md_cell(_skip_detail(case), 240)}")

    lines.extend((
        "",
        "## Notes",
        "",
        "- `seconds` is a wall-clock diagnostic of this run, not a benchmark:",
        "  this report makes no performance claim. Each value is a single",
        "  sample that includes per-case setup, a median is only meaningful",
        "  with `--repeat >= 5` after at least one warmup, and the numbers are",
        "  not comparable across machines or commits.",
        "- Status vocabulary is `PASS`/`FAIL`/`SKIP`/`TIMEOUT`/`ERROR`. A `SKIP`",
        "  is not a failure, while a run in which every selected case skipped",
        "  exits non-zero unless `--allow-all-skipped` was given.",
        "- `relative_error` is normalised by `max|expected|` and",
        "  `relative_residual` is the independently reconstructed",
        "  `||D x - b|| / ||b||`, never a solver's internal recurrence.",
    ))
    return "\n".join(lines) + "\n"


def to_markdown(report: Any) -> str:
    """Render ``report`` as markdown: summary, per-case table, skips, notes.

    The table carries one row per case with its status and a detail column:
    the ``token: reason`` text for a ``SKIP``, the failure text for
    ``FAIL``/``ERROR``/``TIMEOUT`` and the reported metrics for a ``PASS``.
    Skipped cases are listed again under ``## Skipped`` so a reason is never
    lost in a narrow table.  The ``Notes`` section states explicitly that the
    timings are diagnostics and that the report makes no performance claim.
    """
    return _markdown_text(_report_dict(report))


# ---------------------------------------------------------------------------
# dispatch and atomic writes
# ---------------------------------------------------------------------------
#: Text renderers, all consuming the dict produced by :func:`_report_dict`.
_TEXT_RENDERERS = {
    "csv": _csv_text,
    "junit": _junit_text,
    "markdown": _markdown_text,
}


def normalize_format(value: Any) -> str:
    """Return the canonical format name for ``value``.

    Accepts case-insensitive spellings, a leading dot (``".csv"``) and the
    aliases ``xml``/``junitxml`` (JUnit XML) and ``md`` (markdown).  Anything
    else - including a non-string - raises :class:`ValueError` listing the
    supported formats, because silently falling back to JSON would hand a CI
    job a file its consumer cannot parse.
    """
    if not isinstance(value, str):
        raise ValueError(
            f"report format must be a string, got {type(value).__name__}; "
            f"choose from {', '.join(FORMATS)}")
    text = value.strip().lower()
    if text.startswith("."):
        text = text[1:]
    canonical = FORMAT_ALIASES.get(text)
    if canonical is None:
        raise ValueError(
            f"unknown report format {value!r}; choose from "
            f"{', '.join(FORMATS)} (aliases: xml/junitxml -> junit, "
            "md -> markdown)")
    return canonical


def infer_format(path: Any) -> str:
    """Return the format implied by ``path``'s suffix.

    ``.json``/``.csv``/``.xml``/``.junit``/``.md``/``.markdown`` are known
    (case-insensitive).  A missing or unknown suffix raises
    :class:`ValueError` naming the known suffixes and telling the caller to
    pass ``format=`` - guessing JSON for ``report.out`` would be a coin flip
    presented as a decision.
    """
    suffix = Path(str(path)).suffix.lower()
    canonical = SUFFIX_FORMATS.get(suffix)
    if canonical is None:
        shown = suffix or "<none>"
        raise ValueError(
            f"cannot infer the report format from {str(path)!r} (suffix "
            f"{shown}); pass format= one of {', '.join(FORMATS)} "
            f"(known suffixes: {', '.join(sorted(SUFFIX_FORMATS))})")
    return canonical


def render(report: Any, format: str = "json", *,
           indent: int | None = 2) -> str:
    """Render ``report`` in ``format`` and return the text.

    The single entry point for a ``-`` (stdout) target: the CLI writes the
    returned string to ``sys.stdout`` while :func:`write_report` puts the very
    same bytes in a file, so ``--csv -`` and ``--csv PATH`` cannot diverge.
    ``json`` serialises the payload as given (``TestReport.as_dict()`` for a
    report object, CLI-added keys included); the other formats read the
    normalised projection of that same payload.
    """
    name = normalize_format(format)
    if name == "json":
        return _json_text(report, indent=indent)
    return _TEXT_RENDERERS[name](_report_dict(report))


def _fsync_directory(directory: Path) -> None:
    """Best-effort ``fsync`` of a directory entry after :func:`os.replace`.

    Durability of the rename needs the directory flush, not just the file
    flush.  It is not available on every platform (Windows cannot open a
    directory descriptor), so a failure here is ignored rather than turning a
    written report into an error.
    """
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(str(directory), flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _atomic_write(target: Path, text: str) -> None:
    """Write ``text`` to ``target`` atomically (tmp + fsync + replace).

    The temporary file lives in the destination directory, so the final
    :func:`os.replace` is a rename on one filesystem; a concurrent reader sees
    the previous report or the complete new one.  The text is written with
    ``newline=""`` so the ``\\n`` line endings of the renderers reach the file
    unchanged on every platform.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{next(_TEMPORARY)}.tmp")
    try:
        handle = open(temporary, "w", encoding="utf-8", newline="")
        try:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            handle.close()
        os.replace(temporary, target)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    _fsync_directory(target.parent)


def write_report(path: Any, report: Any, format: str | None = None, *,
                 indent: int | None = 2) -> Path:
    """Atomically write ``report`` to ``path``; return the written path.

    ``format`` is ``"json"``, ``"csv"``, ``"junit"`` or ``"markdown"`` (plus
    the ``xml``/``junitxml``/``md`` aliases).  When it is omitted the suffix of
    ``path`` decides (:func:`infer_format`); an unknown suffix or an unknown
    format raises :class:`ValueError` *before* anything is written.  A ``"-"``
    (stdout) or empty path is rejected rather than turned into a file named
    ``-``: the CLI writes :func:`render` to ``sys.stdout`` for that case.

    A write failure (permissions, full disk, a directory target) propagates as
    :class:`OSError`; the temporary file is removed, so a failed write never
    leaves a stale ``.tmp`` next to the report.
    """
    shown = str(path)
    if shown in ("", "-"):
        raise ValueError(
            f"write_report needs a file path, got {shown!r}; '-' means "
            "stdout, so write render(report, format) to sys.stdout instead")
    target = Path(shown).expanduser()
    name = normalize_format(format) if format is not None else infer_format(
        target)
    text = render(report, name, indent=indent)
    _atomic_write(target, text)
    return target
