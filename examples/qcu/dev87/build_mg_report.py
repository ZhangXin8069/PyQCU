#!/usr/bin/env python3
"""Build CSV, plots, LaTeX tables, and analysis JSON from MG matrix data.

The input schema is deliberately tolerant:

* ``pyqcu.mg-matrix`` summaries with ``cases[].runs[].output``;
* direct ``bench_strict_vs_quda.py`` benchmark JSON documents;
* direct iteration trace JSON documents;
* native QUDA ``QUDA_MG_TRACE_FILE`` TSV traces.

Missing fields are represented by empty CSV cells and are also listed in
``summary_analysis.json``.  They are never replaced by invented measurements.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping
from typing import Optional, Sequence, Tuple


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

PHASES = (
    "pre_smoother",
    "post_smoother",
    "restriction",
    "prolongation",
    "coarse_solver",
)
PHASE_COLORS = {
    "pre_smoother": "#245ba3",
    "post_smoother": "#4f8fcb",
    "restriction": "#be4137",
    "prolongation": "#df7a52",
    "coarse_solver": "#5a6973",
    "other": "#b8bec3",
}


class ReportError(RuntimeError):
    """Raised when a report invariant is violated."""


@dataclass
class CaseInput:
    case_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    benchmark: Dict[str, Any] = field(default_factory=dict)
    trace: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    missing: List[str] = field(default_factory=list)


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, Mapping)):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def _deep_merge(left: Mapping[str, Any], right: Mapping[str, Any]) -> Dict[str, Any]:
    result = deepcopy(dict(left))
    for key, value in right.items():
        if isinstance(result.get(key), Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _first(value: Any, *paths: str, default: Any = None) -> Any:
    for path in paths:
        current = value
        found = True
        for part in path.split("."):
            if not isinstance(current, Mapping) or part not in current:
                found = False
                break
            current = current[part]
        if found and current is not None:
            return current
    return default


def _float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _median(values: Iterable[Optional[float]]) -> Optional[float]:
    present = [float(value) for value in values if value is not None]
    return statistics.median(present) if present else None


def _format_number(value: Any, digits: int = 10) -> str:
    number = _float(value)
    if number is None:
        return ""
    if number == 0.0:
        return "0"
    return f"{number:.{digits}g}"


def _normalise_side(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"pyqcu", "qcu", "pyqcu_strict", "strict"}:
        return "pyqcu"
    if text in {"quda", "pyquda"}:
        return "quda"
    return text or None


def _normalise_lattice(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.lower().replace("x", " ").replace(",", " ")
        parts = [part for part in text.split() if part]
        try:
            return "x".join(str(int(part)) for part in parts) if parts else None
        except ValueError:
            return value
    try:
        parts = [int(item) for item in value]
    except (TypeError, ValueError):
        return str(value)
    return "x".join(str(item) for item in parts) if parts else None


def _normalise_precision(value: Any) -> Optional[str]:
    if isinstance(value, Mapping):
        value = value.get("name", value.get("dtype"))
    if value is None:
        return None
    text = str(value).lower()
    if "complex128" in text or text in {"c128", "double"}:
        return "c128"
    if "complex64" in text or text in {"c64", "single"}:
        return "c64"
    return str(value)


def _infer_case_id(source: Path, document: Mapping[str, Any],
                    metadata: Mapping[str, Any]) -> str:
    unit_id = _first(metadata, "unit_id", default=_first(
        document, "unit_id"))
    if unit_id:
        text = str(unit_id)
        for prefix in ("pyqcu__", "quda__"):
            if text.startswith(prefix):
                text = text[len(prefix):]
                break
        return text
    explicit = _first(
        metadata, "case_id", "id", "name",
        default=_first(document, "case_id", "id", "name"))
    if explicit:
        return str(explicit)
    protocol = _mapping(document.get("protocol"))
    explicit = _first(protocol, "case_id", "id", "name")
    if explicit:
        return str(explicit)
    generic_names = {
        "benchmark", "matrix", "matrix_summary", "summary", "trace"}
    stem = source.stem
    if stem.lower() not in generic_names and not stem.lower().startswith(
            "matrix_"):
        for prefix in ("trace-", "trace_", "benchmark-", "benchmark_"):
            if stem.lower().startswith(prefix):
                stem = stem[len(prefix):]
                break
        return stem
    if source.parent.name:
        return source.parent.name
    return source.stem


def _resolve_reference(reference: Any, source: Path) -> Optional[Path]:
    if not reference:
        return None
    candidate = Path(str(reference))
    choices = (
        candidate if candidate.is_absolute() else source.parent / candidate,
        candidate if candidate.is_absolute() else REPO / candidate,
        candidate if candidate.is_absolute() else Path.cwd() / candidate,
    )
    for choice in choices:
        if choice.is_file():
            return choice.resolve()
    return None


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportError(f"cannot read JSON input {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ReportError(f"JSON input {path} must contain an object")
    return dict(value)


def _deduplicate_points(points: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate residual points by outer iteration.

    Reliable-update rows replace an Arnoldi/GCR estimate with the true residual
    for the same iteration.  The true value therefore wins.
    """

    selected: Dict[int, Dict[str, Any]] = {}
    priorities: Dict[int, int] = {}
    for raw in points:
        iteration = _int(_first(raw, "iteration", "outer_iteration"))
        true_r2 = _float(_first(raw, "true_r2"))
        b2 = _float(_first(raw, "b2"))
        true_relative = _float(raw.get("true_relative"))
        if true_r2 is not None and true_r2 >= 0.0:
            if b2 is not None and b2 > 0.0:
                relative = math.sqrt(true_r2 / b2)
            else:
                relative = math.sqrt(true_r2)
            if true_relative is not None:
                relative = true_relative
        else:
            relative = _float(_first(
                raw, "relative", "iterated_relative"))
        if iteration is None or relative is None or relative < 0.0:
            continue
        true_based = true_r2 is not None and true_r2 >= 0.0
        kind = str(raw.get("kind", ""))
        priority = 3 if true_based else (2 if "restart" in kind else 1)
        item = {
            "iteration": int(iteration),
            "relative": float(relative),
            "kind": (
                "true_residual" if true_based else
                kind or "iterated_residual"),
        }
        if iteration not in selected or priority >= priorities[int(iteration)]:
            selected[int(iteration)] = item
            priorities[int(iteration)] = priority
    return [selected[key] for key in sorted(selected)]


def _points_from_solve(solve: Mapping[str, Any]) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    curve = solve.get("residual_curve")
    if isinstance(curve, Sequence) and not isinstance(curve, (str, bytes)):
        points.extend(item for item in curve if isinstance(item, Mapping))

    outer = solve.get("outer_iterations")
    if isinstance(outer, Sequence) and not isinstance(outer, (str, bytes)):
        points.extend(item for item in outer if isinstance(item, Mapping))

    events = solve.get("events")
    if isinstance(events, Sequence) and not isinstance(events, (str, bytes)):
        for item in events:
            if not isinstance(item, Mapping):
                continue
            kind = str(item.get("kind", ""))
            if kind in {"iteration", "restart_residual"}:
                points.append(item)

    residuals = solve.get("residuals")
    if isinstance(residuals, Sequence) and not isinstance(residuals, (str, bytes)):
        for item in residuals:
            if not isinstance(item, Mapping):
                continue
            level = _int(item.get("level"))
            if level in (None, 0):
                points.append(item)
    return _deduplicate_points(points)


def _curves_from_side(side_doc: Mapping[str, Any]) -> List[List[Dict[str, Any]]]:
    curves: List[List[Dict[str, Any]]] = []
    steady = side_doc.get("steady")
    if isinstance(steady, Sequence) and not isinstance(steady, (str, bytes)):
        for solve in steady:
            if isinstance(solve, Mapping):
                points = _points_from_solve(solve)
                if points:
                    curves.append(points)
    points = _points_from_solve(side_doc)
    if points:
        curves.append(points)
    return curves


def _parse_quda_mg_tsv(path: Path) -> List[Dict[str, Any]]:
    """Parse native QUDA MG trace TSV into steady-solve dictionaries."""

    sections: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise ReportError(f"cannot read trace {path}: {exc}") from exc

    for raw in lines:
        if not raw or raw.startswith("#"):
            continue
        fields = raw.split("\t")
        kind = fields[0]
        try:
            if kind == "trace_version":
                continue
            if kind == "python_solve_begin":
                current = {
                    "solve_index": int(fields[1]),
                    "stages": [],
                    "residuals": [],
                    "outer_iterations": [],
                    "cycles": [],
                }
                sections.append(current)
                continue
            if kind == "python_solve_end":
                current = None
                continue
            if current is None and kind in {
                    "cycle_begin", "cycle_end", "stage", "residual",
                    "outer_begin", "outer_iteration"}:
                current = {
                    "solve_index": len(sections),
                    "stages": [],
                    "residuals": [],
                    "outer_iterations": [],
                    "cycles": [],
                }
                sections.append(current)
            if current is None:
                continue
            if kind == "cycle_begin":
                current["cycles"].append({
                    "cycle": int(fields[1]),
                    "level": int(fields[2]),
                    "levels": int(fields[3]),
                    "outer_iteration": int(fields[4]),
                    "location": fields[5],
                    "timestamp_seconds": float(fields[6]),
                })
            elif kind == "cycle_end":
                current["cycles"].append({
                    "kind": "cycle_end",
                    "cycle": int(fields[1]),
                    "level": int(fields[2]),
                    "r2": float(fields[3]),
                    "rn": float(fields[4]),
                    "b2": float(fields[5]),
                    "relative": float(fields[6]),
                    "timestamp_seconds": float(fields[7]),
                })
            elif kind == "stage":
                current["stages"].append({
                    "kind": kind,
                    "cycle": int(fields[1]),
                    "level": int(fields[2]),
                    "phase": fields[3],
                    "name": fields[3],
                    "seconds": float(fields[4]),
                    "pre_iterations": int(fields[5]),
                    "post_iterations": int(fields[6]),
                    "coarse_iterations": int(fields[7]),
                    "outer_iteration": int(fields[8]),
                    "timestamp_seconds": float(fields[9]),
                })
            elif kind == "residual":
                current["residuals"].append({
                    "kind": kind,
                    "cycle": int(fields[1]),
                    "level": int(fields[2]),
                    "phase": fields[3],
                    "name": fields[3],
                    "r2": float(fields[4]),
                    "rn": float(fields[5]),
                    "b2": float(fields[6]),
                    "relative": float(fields[7]),
                    "outer_iteration": int(fields[8]),
                    "timestamp_seconds": float(fields[9]),
                })
            elif kind == "outer_begin":
                current["outer_iterations"].append({
                    "kind": kind,
                    "iteration": int(fields[1]),
                    "r2": float(fields[2]),
                    "rn": float(fields[3]),
                    "b2": float(fields[4]),
                    "relative": float(fields[5]),
                    "timestamp_seconds": float(fields[6]),
                })
            elif kind == "outer_iteration":
                current["outer_iterations"].append({
                    "kind": kind,
                    "iteration": int(fields[1]),
                    "cycle_iteration": int(fields[2]),
                    "iterated_r2": float(fields[3]),
                    "iterated_relative": float(fields[6]),
                    "b2": float(fields[5]),
                    "true_r2": float(fields[7]),
                    "true_relative": float(fields[9]),
                    "timestamp_seconds": float(fields[10]),
                })
        except (IndexError, TypeError, ValueError) as exc:
            raise ReportError(
                f"malformed QUDA MG trace line in {path}: {raw!r}") from exc

    for section in sections:
        section["residual_curve"] = _points_from_solve(section)
        section["iterations"] = (
            max((point["iteration"] for point in section["residual_curve"]),
                default=None))
    return sections


def _trace_document_from_tsv(path: Path) -> Dict[str, Any]:
    lower = path.name.lower()
    side = "quda" if "quda" in lower else "pyqcu"
    sections = _parse_quda_mg_tsv(path)
    return {
        "schema": {"name": "pyqcu.native-trace-tsv", "version": 1},
        "sides": {side: {"steady": sections}},
    }


def _expand_input_file(path: Path) -> List[CaseInput]:
    if path.suffix.lower() in {".tsv", ".trace"}:
        document = _trace_document_from_tsv(path)
    else:
        document = _load_json(path)

    case_collection = document.get(
        "cases", document.get("results", document.get("units")))
    if isinstance(case_collection, Sequence) and not isinstance(
            case_collection, (str, bytes)):
        cases: List[CaseInput] = []
        for raw_case in case_collection:
            if not isinstance(raw_case, Mapping):
                continue
            item = dict(raw_case)
            metadata = _deep_merge(
                _mapping(item.get("case")), _mapping(item.get("metadata")))
            metadata = _deep_merge(metadata, _mapping(item.get("unit")))
            for key in ("id", "lattice", "levels", "precision", "device",
                        "trace", "status"):
                if key in item and key not in metadata:
                    metadata[key] = deepcopy(item[key])

            benchmark: Dict[str, Any] = {}
            trace: Dict[str, Any] = {}
            runs = item.get("runs")
            if isinstance(runs, Sequence) and not isinstance(runs, (str, bytes)):
                for run in reversed(list(runs)):
                    if not isinstance(run, Mapping):
                        continue
                    output = _resolve_reference(
                        run.get(
                            "output",
                            run.get("result", run.get("benchmark"))),
                        path)
                    if output is not None:
                        benchmark = _deep_merge(benchmark, _load_json(output))
                        break
                for run in reversed(list(runs)):
                    if not isinstance(run, Mapping):
                        continue
                    output = _resolve_reference(
                        run.get("trace_output", run.get("trace")), path)
                    if output is not None:
                        if output.suffix.lower() in {".tsv", ".trace"}:
                            trace = _deep_merge(
                                trace, _trace_document_from_tsv(output))
                        else:
                            trace = _deep_merge(trace, _load_json(output))
                        break

            if not benchmark:
                for key in (
                        "benchmark_output", "output_path", "output", "result"):
                    output = _resolve_reference(item.get(key), path)
                    if output is not None:
                        benchmark = _load_json(output)
                        break
            if not trace:
                for key in ("trace_output", "trace_file", "trace_json"):
                    output = _resolve_reference(item.get(key), path)
                    if output is not None:
                        trace = (
                            _trace_document_from_tsv(output)
                            if output.suffix.lower() in {".tsv", ".trace"}
                            else _load_json(output))
                        break

            metric_side = _normalise_side(metadata.get("side"))
            metric_keys = (
                "layers", "outer_iterations", "finest_level_residual",
                "phase_records", "metric_warnings", "setup_seconds")
            if metric_side is not None and any(
                    item.get(key) is not None for key in metric_keys):
                metrics = {
                    "side": metric_side,
                    "trace": metadata.get("trace", "off"),
                    **{
                        key: deepcopy(
                            item.get(key, metadata.get(key)))
                        for key in metric_keys
                    },
                }
                trace = _deep_merge(
                    trace, {"sides": {metric_side: metrics}})

            if not benchmark and isinstance(item.get("sides"), Mapping):
                benchmark = item
            trace_info = item.get("trace")
            if not trace and isinstance(trace_info, Mapping):
                if isinstance(trace_info.get("sides"), Mapping):
                    trace = dict(trace_info)
                else:
                    output = _resolve_reference(
                        trace_info.get("output"), path)
                    if output is not None:
                        if output.suffix.lower() in {".tsv", ".trace"}:
                            trace = _trace_document_from_tsv(output)
                        else:
                            trace = _load_json(output)

            case_id = _infer_case_id(path, item, metadata)
            missing: List[str] = []
            if not benchmark and not trace:
                missing.append("case.output")
            case = CaseInput(
                case_id=case_id,
                metadata=metadata,
                benchmark=benchmark,
                trace=trace,
                source=str(path),
                missing=missing,
            )
            cases.append(case)
        return cases

    metadata = _deep_merge(
        _mapping(document.get("case")),
        _mapping(document.get("protocol")))
    case_id = _infer_case_id(path, document, metadata)
    benchmark = document if isinstance(document.get("sides"), Mapping) else {}
    trace = (
        document if isinstance(document.get("sides"), Mapping) and
        _has_trace_payload(document) else {})
    return [CaseInput(
        case_id=case_id,
        metadata=metadata,
        benchmark=benchmark,
        trace=trace,
        source=str(path),
    )]


def _has_trace_payload(document: Mapping[str, Any]) -> bool:
    sides = document.get("sides")
    if not isinstance(sides, Mapping):
        return False
    for side_doc in sides.values():
        if not isinstance(side_doc, Mapping):
            continue
        if side_doc.get("stages") or side_doc.get("residuals"):
            return True
        steady = side_doc.get("steady")
        if not isinstance(steady, Sequence) or isinstance(steady, (str, bytes)):
            continue
        for solve in steady:
            if not isinstance(solve, Mapping):
                continue
            if solve.get("stages") or solve.get("residuals") or \
                    solve.get("residual_curve") or solve.get("events"):
                return True
    return False


def _discover_inputs(paths: Sequence[Path]) -> List[Path]:
    discovered: List[Path] = []
    for path in paths:
        if path.is_file():
            discovered.append(path)
            continue
        if not path.is_dir():
            raise ReportError(f"input does not exist: {path}")
        patterns = (
            "matrix_summary.json", "matrix_*.json",
            "benchmark*.json", "trace*.json", "*.tsv")
        for pattern in patterns:
            discovered.extend(sorted(path.glob(pattern)))
    unique: List[Path] = []
    seen = set()
    for path in discovered:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    if not unique:
        raise ReportError("no JSON/TSV inputs were discovered")
    return unique


def collect_case_inputs(inputs: Sequence[Path]) -> List[CaseInput]:
    merged: Dict[str, CaseInput] = {}
    for path in _discover_inputs(inputs):
        for case in _expand_input_file(path):
            key = case.case_id
            if key not in merged:
                merged[key] = case
                continue
            current = merged[key]
            current.metadata = _deep_merge(current.metadata, case.metadata)
            current.benchmark = _deep_merge(
                current.benchmark, case.benchmark)
            current.trace = _deep_merge(current.trace, case.trace)
            current.missing.extend(case.missing)
            current.source = f"{current.source}; {case.source}"
    return list(merged.values())


def _side_documents(case: CaseInput) -> Dict[str, Dict[str, Any]]:
    benchmark_sides = _mapping(case.benchmark.get("sides"))
    trace_sides = _mapping(case.trace.get("sides"))
    result: Dict[str, Dict[str, Any]] = {}
    for raw_side in set(benchmark_sides) | set(trace_sides):
        side = _normalise_side(raw_side)
        if side is None:
            continue
        result[side] = _deep_merge(
            _mapping(benchmark_sides.get(raw_side)),
            _mapping(trace_sides.get(raw_side)))
    return result


def _case_dimensions(case: CaseInput, side_doc: Mapping[str, Any]
                     ) -> Dict[str, Any]:
    protocol = _mapping(case.benchmark.get("protocol"))
    metadata = _deep_merge(protocol, case.metadata)
    precision = _normalise_precision(_first(
        metadata, "precision.name", "precision", default=_first(
            side_doc, "precision.name", "precision")))
    lattice = _normalise_lattice(_first(
        metadata, "lattice_xyzt", "lattice", default=_first(
            side_doc, "lattice_xyzt", "lattice")))
    levels = _int(_first(metadata, "levels", "n_level", default=_first(
        side_doc, "levels", "n_level")))
    device = _first(
        side_doc, "device.name", "device_name", "device",
        "provenance.runtime.device_name",
        "provenance.runtime.device",
        "provenance.pyqcu_runtime.device_name",
        "provenance.quda_runtime.device_name",
        "provenance.pyqcu_runtime.device",
        "provenance.quda_runtime.device",
        default=_first(
            case.benchmark,
            "provenance.runtime.device_name",
            "provenance.pyqcu_runtime.device_name",
            "provenance.quda_runtime.device_name",
            "device.name", "device_name", "device",
            default=_first(
                metadata, "device.name", "device_name", "device")))
    if isinstance(device, Mapping):
        device = device.get("name", device.get("device"))
    return {
        "precision": precision,
        "lattice": lattice,
        "levels": levels,
        "device": None if device is None else str(device),
    }


def _iterations_from_side(side_doc: Mapping[str, Any]) -> Optional[int]:
    value = side_doc.get(
        "iterations", side_doc.get("outer_iterations"))
    if isinstance(value, Mapping):
        value = value.get("median", value.get("samples"))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parsed = [_int(item) for item in value]
        median = _median(parsed)
        return None if median is None else int(round(median))
    parsed = _int(value)
    if parsed is not None:
        return parsed
    trace_values: List[int] = []
    steady = side_doc.get("steady")
    if isinstance(steady, Sequence) and not isinstance(steady, (str, bytes)):
        for solve in steady:
            if isinstance(solve, Mapping):
                parsed = _int(solve.get("iterations"))
                if parsed is not None:
                    trace_values.append(parsed)
    median = _median(trace_values)
    return None if median is None else int(round(median))


def _residual_from_side(side_doc: Mapping[str, Any]) -> Optional[float]:
    finest = side_doc.get("finest_level_residual")
    if isinstance(finest, Mapping):
        parsed = _float(finest.get("value", finest.get("relative")))
        if parsed is not None:
            return parsed
    value = side_doc.get("true_residual")
    if isinstance(value, Mapping):
        for key in ("max_rel", "median_rel", "samples_rel", "untimed_probe_rel"):
            if key in value:
                if isinstance(value[key], Sequence) and not isinstance(
                        value[key], (str, bytes)):
                    parsed = _median(_float(item) for item in value[key])
                else:
                    parsed = _float(value[key])
                if parsed is not None:
                    return parsed
    parsed = _float(value)
    if parsed is not None:
        return parsed
    for key in ("final_true_residual", "final_residual", "residual"):
        parsed = _float(side_doc.get(key))
        if parsed is not None:
            return parsed
    curves = _curves_from_side(side_doc)
    if curves:
        return curves[-1][-1]["relative"]
    return None


def _phase_records(case: CaseInput, side_doc: Mapping[str, Any]
                   ) -> Dict[str, Dict[str, Any]]:
    timing = _mapping(_first(
        side_doc, "timing", "timing_reference", default={}))
    iterations = _iterations_from_side(side_doc)
    residual = _residual_from_side(side_doc)
    result: Dict[str, Dict[str, Any]] = {}

    cold = _first(timing, "cold", "cold_seconds", "cold_solve")
    if cold is not None:
        if isinstance(cold, Mapping):
            seconds = _median(_float(item) for item in _sequence(
                cold.get("samples_seconds", cold.get("samples", []))))
            if seconds is None:
                seconds = _first(
                    cold, "median_seconds", "seconds", "total_seconds")
            result["cold"] = {
                "total_seconds": _float(seconds),
                "samples": len(_sequence(cold.get("samples_seconds", []))) or 1,
                "finest_iterations": _int(cold.get("iterations")) or iterations,
                "finest_residual": _first(
                    cold, "true_residual_rel", "residual",
                    default=residual),
            }
        else:
            result["cold"] = {
                "total_seconds": _float(cold),
                "samples": 1,
                "finest_iterations": iterations,
                "finest_residual": residual,
            }

    warmups = timing.get("warmups")
    if isinstance(warmups, Sequence) and not isinstance(warmups, (str, bytes)):
        clean = [item for item in warmups if isinstance(item, Mapping)]
        if clean:
            result["warmup"] = {
                "total_seconds": _median(
                    _float(item.get("seconds")) for item in clean),
                "samples": len(clean),
                "finest_iterations": (
                    _median(_int(item.get("iterations")) for item in clean)),
                "finest_residual": _median(
                    _float(item.get("true_residual_rel")) for item in clean),
            }

    steady = _first(timing, "steady", "steady_solve")
    if steady is not None:
        if isinstance(steady, Mapping):
            seconds = _first(
                steady, "median_seconds", "total_seconds", "seconds")
            if seconds is None:
                seconds = _median(
                    _float(item) for item in _sequence(
                        steady.get("samples_seconds", [])))
            samples = _sequence(steady.get("samples_seconds", []))
            result["steady"] = {
                "total_seconds": _float(seconds),
                "samples": len(samples) or 1,
                "finest_iterations": iterations,
                "finest_residual": residual,
            }
        else:
            result["steady"] = {
                "total_seconds": _float(steady),
                "samples": 1,
                "finest_iterations": iterations,
                "finest_residual": residual,
            }

    trace_steady = side_doc.get("steady")
    if isinstance(trace_steady, Sequence) and not isinstance(
            trace_steady, (str, bytes)):
        trace_items = [
            item for item in trace_steady if isinstance(item, Mapping)]
        if trace_items and "steady" not in result:
            result["steady"] = {
                "total_seconds": None,
                "samples": len(trace_items),
                "finest_iterations": iterations,
                "finest_residual": residual,
            }

    flat_seconds = _first(
        timing, "median_seconds", "total_seconds", "seconds")
    flat_samples = _sequence(timing.get("samples_seconds", []))
    if flat_seconds is not None or flat_samples:
        flat_total = (
            _float(flat_seconds)
            if flat_seconds is not None
            else _median(_float(item) for item in flat_samples))
        if "steady" not in result:
            result["steady"] = {
                "total_seconds": flat_total,
                "samples": len(flat_samples) or 1,
                "finest_iterations": (
                    _int(timing.get("median_iterations")) or iterations),
                "finest_residual": residual,
            }
        elif result["steady"].get("total_seconds") is None:
            result["steady"]["total_seconds"] = flat_total
            if result["steady"].get("samples") is None:
                result["steady"]["samples"] = len(flat_samples) or 1

    if not result:
        phase_records = side_doc.get("phase_records")
        if isinstance(phase_records, Sequence) and not isinstance(
                phase_records, (str, bytes)):
            grouped: Dict[str, int] = defaultdict(int)
            for record in phase_records:
                if isinstance(record, Mapping) and record.get("phase") in {
                        "cold", "warmup", "steady"}:
                    grouped[str(record["phase"])] += 1
            for phase, count in grouped.items():
                result[phase] = {
                    "total_seconds": None,
                    "samples": count,
                    "finest_iterations": iterations,
                    "finest_residual": residual,
                }
        if not result:
            result["steady"] = {
                "total_seconds": None,
                "samples": None,
                "finest_iterations": iterations,
                "finest_residual": residual,
            }
    return result


def _normalise_level(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str) and value.lower().startswith("l"):
        value = value[1:]
    return _int(value)


def _canonical_phase(name: Any) -> str:
    text = str(name or "").lower()
    if "pre_smooth" in text or "presmooth" in text:
        return "pre_smoother"
    if "post_smooth" in text or "postsmooth" in text:
        return "post_smoother"
    if "restrict" in text:
        return "restriction"
    if "prolong" in text:
        return "prolongation"
    if "coarse" in text or "coarsest" in text or "recursive_solve" in text:
        return "coarse_solver"
    return "other"


def _stage_iterations(stage: Mapping[str, Any]) -> Tuple[int, int, int]:
    name = str(_first(stage, "name", "phase", default="")).lower()
    pre = _int(stage.get("pre_iterations")) or 0
    post = _int(stage.get("post_iterations")) or 0
    coarse = _int(stage.get("coarse_iterations")) or 0

    # QUDA quirk: coarsest_solve stores absolute before/after counts in the
    # pre_iterations/post_iterations columns, unlike the *_iterations stages.
    if name == "coarsest_solve" and pre >= 0 and post >= 0:
        return 0, 0, max(0, post - pre)
    if "pre_smoother_iterations" in name:
        return max(0, pre), 0, 0
    if "post_smoother_iterations" in name:
        return 0, max(0, post), 0
    if "coarse_solver_iterations" in name:
        return 0, 0, max(0, coarse)
    return 0, 0, 0


def _flatten_stages(side_doc: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    direct = side_doc.get("stages")
    if isinstance(direct, Sequence) and not isinstance(direct, (str, bytes)):
        yield from (item for item in direct if isinstance(item, Mapping))

    steady = side_doc.get("steady")
    if isinstance(steady, Sequence) and not isinstance(steady, (str, bytes)):
        for solve in steady:
            if not isinstance(solve, Mapping):
                continue
            stages = solve.get("stages")
            if isinstance(stages, Sequence) and not isinstance(
                    stages, (str, bytes)):
                yield from (
                    item for item in stages if isinstance(item, Mapping))
            mg_trace = solve.get("mg_trace")
            if isinstance(mg_trace, Mapping):
                cycles = mg_trace.get("cycles")
                if isinstance(cycles, Sequence) and not isinstance(
                        cycles, (str, bytes)):
                    for cycle in cycles:
                        if not isinstance(cycle, Mapping):
                            continue
                        begin = _mapping(cycle.get("begin"))
                        level = begin.get("level")
                        stages = cycle.get("stages")
                        if isinstance(stages, Sequence) and not isinstance(
                                stages, (str, bytes)):
                            for stage in stages:
                                if isinstance(stage, Mapping):
                                    item = dict(stage)
                                    item.setdefault("level", level)
                                    yield item


def _explicit_level_profiles(side_doc: Mapping[str, Any]
                             ) -> Dict[int, Dict[str, Any]]:
    candidates = [
        side_doc.get("level_profiles"),
        side_doc.get("mg_levels"),
        side_doc.get("level_stats"),
        side_doc.get("per_level"),
        side_doc.get("layers"),
        side_doc.get("levels"),
        _first(side_doc, "profile.levels"),
    ]
    result: Dict[int, Dict[str, Any]] = {}
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            iterable = candidate.items()
        elif isinstance(candidate, Sequence) and not isinstance(
                candidate, (str, bytes)):
            iterable = enumerate(candidate)
        else:
            continue
        for key, value in iterable:
            if not isinstance(value, Mapping):
                continue
            value = dict(value)
            phases = _mapping(value.get("phases"))
            if phases:
                aliases = {
                    "pre_smoother": ("pre_smoother", "pre-smoother"),
                    "post_smoother": ("post_smoother", "post-smoother"),
                    "restriction": ("restriction", "restrict"),
                    "prolongation": ("prolongation", "prolongate"),
                    "coarse_solver": ("coarse_solver", "coarse-solver"),
                    "other": ("other",),
                }
                for canonical, names in aliases.items():
                    phase_data = next(
                        (phases.get(name) for name in names
                         if phases.get(name) is not None), None)
                    if isinstance(phase_data, Mapping):
                        value.setdefault(
                            canonical, phase_data.get("seconds"))
                    elif phase_data is not None:
                        value.setdefault(canonical, phase_data)
            if "total_iterations" not in value and "iterations" in value:
                value["total_iterations"] = value["iterations"]
            level = _normalise_level(value.get("level", key))
            if level is None:
                continue
            result[level] = _deep_merge(result.get(level, {}), value)
    return result


def _profile_number(profile: Mapping[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        value = _float(profile.get(key))
        if value is None:
            continue
        if key.endswith("_ms"):
            return value / 1000.0
        return value
    return None


def _aggregate_levels(case: CaseInput, side_doc: Mapping[str, Any],
                      dimensions: Mapping[str, Any],
                      finest_iterations: Optional[int]
                      ) -> List[Dict[str, Any]]:
    duration: Dict[int, Dict[str, float]] = defaultdict(
        lambda: defaultdict(float))
    present: Dict[int, set] = defaultdict(set)
    iteration_count: Dict[int, int] = defaultdict(int)
    stage_total: Dict[int, float] = defaultdict(float)
    wrapper_total: Dict[int, float] = defaultdict(float)
    observed_levels = set()

    for stage in _flatten_stages(side_doc):
        level = _normalise_level(stage.get("level"))
        if level is None:
            continue
        name = str(_first(stage, "name", "phase", default="unknown"))
        seconds = _float(stage.get("seconds"))
        if seconds is not None and seconds >= 0.0:
            observed_levels.add(level)
            if name == "outer_iteration":
                wrapper_total[level] += seconds
            else:
                stage_total[level] += seconds
                phase = _canonical_phase(name)
                duration[level][phase] += seconds
                present[level].add(phase)
        pre, post, coarse = _stage_iterations(stage)
        iteration_count[level] += pre + post + coarse

    explicit = _explicit_level_profiles(side_doc)
    observed_levels.update(explicit)
    declared_levels = _int(dimensions.get("levels"))
    if declared_levels is not None and declared_levels > 0:
        observed_levels.update(range(declared_levels))
    if not observed_levels:
        observed_levels.add(0)

    rows: List[Dict[str, Any]] = []
    for level in sorted(observed_levels):
        profile = _mapping(explicit.get(level))
        level_observed = level in present or level in stage_total or \
            level in wrapper_total or bool(profile)
        missing: List[str] = []
        values: Dict[str, Optional[float]] = {}
        for phase in PHASES:
            value = _profile_number(
                profile, phase, f"{phase}_seconds", f"{phase}_ms")
            if value is None and phase in present[level]:
                value = duration[level][phase]
            if value is None:
                value = 0.0 if level_observed else None
                missing.append(f"level.{level}.{phase}")
            values[phase] = value

        total = _profile_number(
            profile, "total_seconds", "total_time_seconds", "total_ms",
            "time_seconds")
        if total is None and level_observed:
            total = stage_total[level] + wrapper_total[level]
        if total is not None and total < 0.0:
            raise ReportError(
                f"{case.case_id}: level {level} has negative total time")

        if all(value is not None for value in values.values()) and \
                total is not None:
            other = float(total) - sum(float(value) for value in values.values())
            if other < -1e-12:
                raise ReportError(
                    f"{case.case_id}: level {level} other is negative "
                    f"({other:.12g})")
            if other < 0.0:
                other = 0.0
        else:
            other = None
            missing.append(f"level.{level}.other")
        if total is None:
            missing.append(f"level.{level}.total_seconds")

        total_iterations = _int(_first(
            profile, "total_iterations", "iterations"))
        if level == 0 and finest_iterations is not None:
            total_iterations = int(finest_iterations)
        elif total_iterations is None and iteration_count[level] > 0:
            total_iterations = int(iteration_count[level])
        if total_iterations is None:
            missing.append(f"level.{level}.total_iterations")

        rows.append({
            "case_id": case.case_id,
            "side": _normalise_side(side_doc.get("side")) or "",
            "precision": dimensions.get("precision"),
            "lattice": dimensions.get("lattice"),
            "trace": "on" if _side_has_trace(side_doc) else "off",
            "levels": dimensions.get("levels"),
            "device": dimensions.get("device"),
            "level": level,
            "total_seconds": total,
            "total_iterations": total_iterations,
            **values,
            "other": other,
            "missing_fields": sorted(set(missing)),
        })
    return rows


def _side_has_trace(side_doc: Mapping[str, Any]) -> bool:
    if str(side_doc.get("trace", "")).lower() == "on":
        return True
    if side_doc.get("stage_trace_available") is True:
        return True
    if _curves_from_side(side_doc):
        return True
    return any(True for _ in _flatten_stages(side_doc))


def _collect_records(cases: Sequence[CaseInput]) -> Tuple[
        List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    records: List[Dict[str, Any]] = []
    level_rows: List[Dict[str, Any]] = []
    curves: Dict[str, List[Dict[str, Any]]] = {}

    for case in cases:
        for side, side_doc in sorted(_side_documents(case).items()):
            side_doc.setdefault("side", side)
            dimensions = _case_dimensions(case, side_doc)
            finest_iterations = _iterations_from_side(side_doc)
            finest_residual = _residual_from_side(side_doc)
            trace_on = _side_has_trace(side_doc)
            phases = _phase_records(case, side_doc)
            for phase, phase_data in phases.items():
                missing = list(case.missing)
                for key in ("precision", "lattice", "levels"):
                    if dimensions.get(key) is None:
                        missing.append(key)
                if dimensions.get("device") is None:
                    missing.append("device")
                total_seconds = _float(phase_data.get("total_seconds"))
                if total_seconds is None:
                    missing.append(f"{phase}.total_seconds")
                iterations = _int(phase_data.get("finest_iterations"))
                if iterations is None:
                    iterations = finest_iterations
                if iterations is None:
                    missing.append("finest_iterations")
                residual = _float(phase_data.get("finest_residual"))
                if residual is None:
                    residual = finest_residual
                if residual is None:
                    missing.append("finest_residual")
                records.append({
                    "case_id": case.case_id,
                    "side": side,
                    "precision": dimensions.get("precision"),
                    "lattice": dimensions.get("lattice"),
                    "trace": "on" if trace_on else "off",
                    "levels": dimensions.get("levels"),
                    "device": dimensions.get("device"),
                    "phase": phase,
                    "total_seconds": total_seconds,
                    "finest_iterations": iterations,
                    "finest_residual": residual,
                    "samples": phase_data.get("samples"),
                    "missing_fields": sorted(set(missing)),
                })

            level_rows.extend(
                _aggregate_levels(
                    case, side_doc, dimensions, finest_iterations))
            side_curves = _curves_from_side(side_doc)
            if side_curves:
                curves[f"{case.case_id}|{side}"] = side_curves[-1]
    return records, level_rows, curves


def _steady_records(records: Sequence[Mapping[str, Any]]
                    ) -> List[Mapping[str, Any]]:
    return [record for record in records if record.get("phase") == "steady"]


def _coverage(records: Sequence[Mapping[str, Any]],
              level_rows: Sequence[Mapping[str, Any]],
              curves: Mapping[str, Sequence[Mapping[str, Any]]]
              ) -> Dict[str, Any]:
    case_ids = sorted({str(record["case_id"]) for record in records})
    side_cells = {
        (str(record["case_id"]), str(record["side"])) for record in records}
    expected_side_cells = 2 * len(case_ids)
    phase_cells = {
        (str(record["case_id"]), str(record["side"]), str(record["phase"]))
        for record in records}
    expected_phase_cells = 3 * expected_side_cells
    level_cells = {
        (str(row["case_id"]), str(row["side"]), int(row["level"]))
        for row in level_rows}
    case_side_levels: Dict[Tuple[str, str], int] = {}
    for record in records:
        levels = _int(record.get("levels"))
        if levels is None or levels <= 0:
            continue
        key = (str(record["case_id"]), str(record["side"]))
        case_side_levels[key] = max(levels, case_side_levels.get(key, 0))
    expected_level_cells = (
        sum(case_side_levels.values()) if case_side_levels else None)
    expected_curves = 2 * len(case_ids)

    missing_cells: List[Dict[str, Any]] = []
    for case_id in case_ids:
        sides = {side for item, side in side_cells if item == case_id}
        for side in ("pyqcu", "quda"):
            if side not in sides:
                missing_cells.append({
                    "kind": "side", "case_id": case_id, "side": side})
        if f"{case_id}|pyqcu" not in curves:
            missing_cells.append({
                "kind": "finest_residual_curve",
                "case_id": case_id, "side": "pyqcu"})
        if f"{case_id}|quda" not in curves:
            missing_cells.append({
                "kind": "finest_residual_curve",
                "case_id": case_id, "side": "quda"})

    missing_counts = Counter(
        field for record in records
        for field in _sequence(record.get("missing_fields")))
    missing_counts.update(
        field for row in level_rows
        for field in _sequence(row.get("missing_fields")))
    return {
        "cases": len(case_ids),
        "case_ids": case_ids,
        "side_cells_observed": len(side_cells),
        "side_cells_expected_from_observed_cases": expected_side_cells,
        "side_coverage": (
            len(side_cells) / expected_side_cells
            if expected_side_cells else None),
        "phase_cells_observed": len(phase_cells),
        "phase_cells_expected_from_observed_cases": expected_phase_cells,
        "phase_coverage": (
            len(phase_cells) / expected_phase_cells
            if expected_phase_cells else None),
        "level_cells_observed": len(level_cells),
        "level_cells_expected_from_declared_levels": expected_level_cells,
        "level_coverage": (
            len(level_cells) / expected_level_cells
            if expected_level_cells else None),
        "trace_curves": sorted(curves),
        "trace_curves_expected": expected_curves,
        "trace_curve_coverage": (
            len(curves) / expected_curves if expected_curves else None),
        "missing_cells": missing_cells,
        "missing_field_counts": dict(sorted(missing_counts.items())),
    }


def _summarise(records: Sequence[Mapping[str, Any]],
               level_rows: Sequence[Mapping[str, Any]],
               curves: Mapping[str, Sequence[Mapping[str, Any]]],
               inputs: Sequence[Path]) -> Dict[str, Any]:
    steady = _steady_records(records)
    fastest: Dict[str, Any] = {}
    for side in ("pyqcu", "quda"):
        samples = [
            record for record in steady
            if record["side"] == side and record.get("total_seconds") is not None]
        if samples:
            best = min(samples, key=lambda item: float(item["total_seconds"]))
            fastest[side] = {
                "case_id": best["case_id"],
                "total_seconds": best["total_seconds"],
                "precision": best["precision"],
                "lattice": best["lattice"],
                "levels": best["levels"],
                "device": best["device"],
            }

    by_case: Dict[str, Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for record in steady:
        by_case[str(record["case_id"])][str(record["side"])] = record
    comparisons: List[Dict[str, Any]] = []
    for case_id, pair in sorted(by_case.items()):
        left = pair.get("pyqcu")
        right = pair.get("quda")
        if left is None or right is None:
            comparisons.append({
                "case_id": case_id,
                "pyqcu_seconds": (
                    None if left is None else left.get("total_seconds")),
                "quda_seconds": (
                    None if right is None else right.get("total_seconds")),
                "pyqcu_over_quda": None,
                "speedup_quda_over_pyqcu": None,
                "status": "missing_side",
            })
            continue
        pyqcu_seconds = _float(left.get("total_seconds"))
        quda_seconds = _float(right.get("total_seconds"))
        ratio = (
            pyqcu_seconds / quda_seconds
            if pyqcu_seconds is not None and quda_seconds not in (None, 0.0)
            else None)
        comparisons.append({
            "case_id": case_id,
            "pyqcu_seconds": pyqcu_seconds,
            "quda_seconds": quda_seconds,
            "pyqcu_over_quda": ratio,
            "speedup_quda_over_pyqcu": (
                None if ratio in (None, 0.0) else 1.0 / ratio),
            "device_match": left.get("device") == right.get("device"),
            "status": "ok" if ratio is not None else "missing_time",
        })

    valid_ratios = [
        float(item["pyqcu_over_quda"])
        for item in comparisons
        if item.get("pyqcu_over_quda") is not None]
    median_ratio = _median(valid_ratios)
    best_by_dimension: Dict[str, Dict[str, Any]] = {}
    for dimension in ("precision", "lattice", "levels", "device"):
        per_side: Dict[str, Any] = {}
        for side in ("pyqcu", "quda"):
            candidates = [
                record for record in steady
                if record["side"] == side and
                record.get("total_seconds") is not None and
                record.get(dimension) is not None]
            if candidates:
                best = min(
                    candidates, key=lambda item: float(item["total_seconds"]))
                per_side[side] = {
                    "case_id": best["case_id"],
                    "value": best[dimension],
                    "total_seconds": best["total_seconds"],
                }
        if per_side:
            best_by_dimension[dimension] = per_side

    comparable = [
        item for item in comparisons
        if item.get("speedup_quda_over_pyqcu") is not None]
    fastest_comparison = (
        max(
            comparable,
            key=lambda item: float(item["speedup_quda_over_pyqcu"]))
        if comparable else None)

    anomalies: List[Dict[str, Any]] = []
    for record in records:
        seconds = _float(record.get("total_seconds"))
        if seconds is not None and seconds < 0.0:
            anomalies.append({
                "kind": "negative_total_seconds",
                "case_id": record["case_id"],
                "side": record["side"],
                "value": seconds,
            })
        iterations = _int(record.get("finest_iterations"))
        if iterations is not None and iterations <= 0:
            anomalies.append({
                "kind": "nonpositive_finest_iterations",
                "case_id": record["case_id"],
                "side": record["side"],
                "value": iterations,
            })
    for item in comparisons:
        ratio = item.get("pyqcu_over_quda")
        if ratio is not None and not 0.5 <= float(ratio) <= 2.0:
            anomalies.append({
                "kind": "large_time_ratio",
                "case_id": item["case_id"],
                "value": ratio,
            })
        if item.get("device_match") is False:
            anomalies.append({
                "kind": "device_mismatch",
                "case_id": item["case_id"],
            })

    return {
        "schema": {"name": "pyqcu.mg-report-analysis", "version": 1},
        "inputs": [str(path) for path in inputs],
        "records": len(records),
        "level_rows": len(level_rows),
        "coverage": _coverage(records, level_rows, curves),
        "fastest_steady_case_by_side": fastest,
        "best_steady_case_by_dimension": best_by_dimension,
        "comparisons": comparisons,
        "relative_time": {
            "definition": "PyQCU steady seconds divided by QUDA steady seconds",
            "median_pyqcu_over_quda": median_ratio,
            "median_speedup_quda_over_pyqcu": (
                None if median_ratio in (None, 0.0) else 1.0 / median_ratio),
            "fastest_speedup_case": fastest_comparison,
            "comparable_cases": len(valid_ratios),
        },
        "trace_curve_count": len(curves),
        "anomalies": anomalies,
        "schema_assumptions": [
            "Inputs may be matrix_summary JSON, collector benchmark JSON, "
            "iteration trace JSON, or QUDA MG trace TSV.",
            "The finest-level total iteration count is outer FGMRES/GCR "
            "iterations, never the sum of smoother or coarse solver iterations.",
            "Missing measurements remain blank and are listed in missing_fields "
            "or coverage.missing_cells.",
            "QUDA coarsest_solve pre_iterations/post_iterations are absolute "
            "before/after counts; the effective delta is post-minus-pre.",
            "Reliable-update residual rows are deduplicated by outer iteration "
            "with true_r2 preferred over the iterated estimate.",
        ],
    }


def _csv_rows(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        row = {
            "side": record["side"],
            "precision": record.get("precision") or "",
            "lattice": record.get("lattice") or "",
            "trace": record.get("trace") or "",
            "levels": "" if record.get("levels") is None else record["levels"],
            "device": record.get("device") or "",
            "phase": record.get("phase") or "",
            "total_seconds": _format_number(record.get("total_seconds")),
            "finest_iterations": (
                "" if record.get("finest_iterations") is None
                else record["finest_iterations"]),
            "finest_residual": _format_number(
                record.get("finest_residual")),
            "missing_fields": ";".join(record.get("missing_fields", [])),
            "samples": "" if record.get("samples") is None else record["samples"],
            "case_id": record["case_id"],
        }
        rows.append(row)
    return rows


def _level_csv_rows(level_rows: Sequence[Mapping[str, Any]]
                    ) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in level_rows:
        output: Dict[str, Any] = {}
        for key in (
                "case_id", "side", "precision", "lattice", "trace", "levels",
                "device", "level", "total_seconds", "total_iterations",
                *PHASES, "other"):
            value = row.get(key)
            if key in {"case_id", "side", "precision", "lattice", "trace",
                       "device", "level"}:
                output[key] = "" if value is None else value
            elif key == "levels":
                output[key] = "" if value is None else value
            elif key == "total_iterations":
                output[key] = "" if value is None else value
            else:
                output[key] = _format_number(value)
        output["missing_fields"] = ";".join(row.get("missing_fields", []))
        rows.append(output)
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if rows:
        fieldnames = list(rows[0])
    else:
        fieldnames = []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _case_key(row: Mapping[str, Any]) -> str:
    return "|".join((
        str(row.get("case_id", "")),
        str(row.get("side", "")),
        str(row.get("precision", "")),
        str(row.get("lattice", "")),
        f"L{row.get('levels', '?')}",
    ))


def _case_label(row: Mapping[str, Any]) -> str:
    return (
        f"{row.get('case_id', '?')}\n"
        f"{row.get('side', '?')} | {row.get('precision', '?')} | "
        f"{row.get('lattice', '?')} | L{row.get('levels', '?')}\n"
        f"{row.get('device') or 'device:missing'}")


def _save_figure(fig: Any, svg: Path, pdf: Path) -> None:
    fig.tight_layout()
    fig.savefig(svg, format="svg", bbox_inches="tight")
    fig.savefig(pdf, format="pdf", bbox_inches="tight")


def _plot_level_times(level_rows: Sequence[Mapping[str, Any]],
                      svg: Path, pdf: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(9.0, 1.2 * len(level_rows)), 6.0))
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in level_rows:
        grouped[_case_key(row)].append(row)
    keys = sorted(grouped)
    if not keys:
        ax.text(0.5, 0.5, "No level timing rows", ha="center", va="center")
        ax.set_axis_off()
        _save_figure(fig, svg, pdf)
        plt.close(fig)
        return

    x_base = list(range(len(keys)))
    levels = sorted({
        int(row["level"]) for row in level_rows
        if row.get("level") is not None})
    total_width = 0.82
    width = total_width / max(1, len(levels))
    for index, level in enumerate(levels):
        x_values = [x - total_width / 2 + (index + 0.5) * width
                    for x in x_base]
        bottoms = [0.0] * len(keys)
        totals: List[Optional[float]] = []
        for key in keys:
            row = next(
                (item for item in grouped[key]
                 if int(item["level"]) == level), None)
            totals.append(None if row is None else _float(
                row.get("total_seconds")))
        for phase in (*PHASES, "other"):
            heights: List[float] = []
            for key, total in zip(keys, totals):
                row = next(
                    (item for item in grouped[key]
                     if int(item["level"]) == level), None)
                value = None if row is None else _float(row.get(phase))
                share = (
                    None if value is None or total in (None, 0.0)
                    else 100.0 * max(0.0, value) / total)
                heights.append(max(0.0, float(share or 0.0)))
            ax.bar(
                x_values, heights, width=width * 0.92, bottom=bottoms,
                color=PHASE_COLORS[phase], edgecolor="white", linewidth=0.4,
                label=phase if index == 0 else None)
            bottoms = [left + right for left, right in zip(bottoms, heights)]

        for x_value, key, total in zip(x_values, keys, totals):
            row = next(
                (item for item in grouped[key]
                 if int(item["level"]) == level), None)
            if row is None or total is None or total <= 0.0:
                ax.text(
                    x_value, 0.0, "NA", rotation=90, ha="center",
                    va="bottom", fontsize=7, color="#8b1a1a")
                continue
            ax.text(
                x_value, 100.0, f"{total:.4g}s", rotation=90,
                ha="center", va="bottom", fontsize=6)
            missing = [
                phase for phase in PHASES
                if f"level.{level}.{phase}" in row.get("missing_fields", [])]
            if missing:
                ax.text(
                    x_value, 98.0,
                    "gap:" + ",".join(phase[:4] for phase in missing),
                    rotation=90, ha="center", va="bottom",
                    fontsize=6, color="#8b1a1a")

    ax.set_xticks(x_base)
    ax.set_xticklabels(
        [_case_label(grouped[key][0]) for key in keys],
        rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0.0, 106.0)
    ax.set_ylabel("share of level total (%)")
    ax.set_title(
        "Per-level phase share (total seconds labelled; gaps never filled)")
    ax.grid(axis="y", alpha=0.22)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles, labels, ncol=min(6, len(labels)),
            fontsize=8, loc="upper left")
    _save_figure(fig, svg, pdf)
    plt.close(fig)


def _plot_level_iterations(level_rows: Sequence[Mapping[str, Any]],
                           svg: Path, pdf: Path) -> None:
    import matplotlib.pyplot as plt

    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in level_rows:
        grouped[_case_key(row)].append(row)
    keys = sorted(grouped)
    fig, ax = plt.subplots(figsize=(max(9.0, 1.0 * len(keys)), 5.5))
    if not keys:
        ax.text(0.5, 0.5, "No level iteration rows", ha="center", va="center")
        ax.set_axis_off()
        _save_figure(fig, svg, pdf)
        plt.close(fig)
        return

    levels = sorted({
        int(row["level"]) for row in level_rows
        if row.get("level") is not None})
    x_base = list(range(len(keys)))
    width = 0.78 / max(1, len(levels))
    for index, level in enumerate(levels):
        values: List[float] = []
        labels: List[str] = []
        x_values: List[float] = []
        missing_x: List[float] = []
        for x_value, key in zip(x_base, keys):
            row = next(
                (item for item in grouped[key]
                 if int(item["level"]) == level), None)
            offset = x_value - 0.39 + (index + 0.5) * width
            value = None if row is None else _int(row.get("total_iterations"))
            if value is None:
                missing_x.append(offset)
                continue
            x_values.append(offset)
            values.append(float(value))
            labels.append(str(value))
        ax.bar(
            x_values, values, width=width * 0.9,
            color="#4f8fcb" if level != 0 else "#245ba3",
            label=f"level {level}")
        for x_value, value, label in zip(x_values, values, labels):
            ax.text(
                x_value, value, label, ha="center", va="bottom", fontsize=7)
        for x_value in missing_x:
            ax.text(
                x_value, 0.0, "NA", rotation=90, ha="center", va="bottom",
                fontsize=7, color="#8b1a1a")

    ax.set_xticks(x_base)
    ax.set_xticklabels(
        [_case_label(grouped[key][0]) for key in keys],
        rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("iterations")
    ax.set_title("Per-level iteration count (fine level uses outer solver total)")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(fontsize=8)
    _save_figure(fig, svg, pdf)
    plt.close(fig)


def _plot_residuals(curves: Mapping[str, Sequence[Mapping[str, Any]]],
                    records: Sequence[Mapping[str, Any]],
                    svg: Path, pdf: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    if not curves:
        ax.text(
            0.5, 0.5,
            "No trace-on finest-level residual curves\n"
            "Required: sides.<side>.steady[].residual_curve or native trace",
            ha="center", va="center")
        ax.set_axis_off()
        _save_figure(fig, svg, pdf)
        plt.close(fig)
        return

    for label, points in sorted(curves.items()):
        clean = _deduplicate_points(points)
        if not clean:
            continue
        ax.plot(
            [point["iteration"] for point in clean],
            [max(point["relative"], 1e-30) for point in clean],
            marker="o", markersize=2.5, linewidth=1.4, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("outer iteration")
    ax.set_ylabel("relative residual")
    ax.set_title("Finest-level convergence (reliable-update rows deduplicated)")
    ax.grid(which="both", alpha=0.22)
    ax.legend(fontsize=7)
    expected = sorted({
        f"{record['case_id']}|{record['side']}"
        for record in _steady_records(records)})
    missing = [label for label in expected if label not in curves]
    if missing:
        shown = ", ".join(missing[:6])
        if len(missing) > 6:
            shown += f", ... (+{len(missing) - 6})"
        ax.text(
            0.01, 0.01, "missing curves: " + shown,
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7, color="#8b1a1a")
    _save_figure(fig, svg, pdf)
    plt.close(fig)


def _plot_speedup(records: Sequence[Mapping[str, Any]],
                  svg: Path, pdf: Path) -> None:
    import matplotlib.pyplot as plt

    steady = _steady_records(records)
    by_case: Dict[str, Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for record in steady:
        by_case[str(record["case_id"])][str(record["side"])] = record
    case_ids = sorted(by_case)
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(max(9.0, 1.1 * len(case_ids)), 8.0),
        sharex=True, gridspec_kw={"height_ratios": [1.25, 1.0]})
    if not case_ids:
        top.text(0.5, 0.5, "No steady timing records", ha="center", va="center")
        top.set_axis_off()
        bottom.set_axis_off()
        _save_figure(fig, svg, pdf)
        plt.close(fig)
        return

    x_base = list(range(len(case_ids)))
    width = 0.36
    pyqcu_values: List[float] = []
    quda_values: List[float] = []
    pyqcu_x: List[float] = []
    quda_x: List[float] = []
    pyqcu_labels: List[str] = []
    quda_labels: List[str] = []
    for x_value, case_id in zip(x_base, case_ids):
        pair = by_case[case_id]
        for side, x_offset, x_list, y_list, label_list in (
                ("pyqcu", -width / 2, pyqcu_x, pyqcu_values, pyqcu_labels),
                ("quda", width / 2, quda_x, quda_values, quda_labels)):
            value = (
                None if side not in pair
                else _float(pair[side].get("total_seconds")))
            if value is None:
                top.text(
                    x_value + x_offset, 0.0, "NA", rotation=90,
                    ha="center", va="bottom", fontsize=7, color="#8b1a1a")
                continue
            x_list.append(x_value + x_offset)
            y_list.append(value)
            label_list.append(f"{value:.4g}")
    if pyqcu_values:
        top.bar(
            pyqcu_x, pyqcu_values, width=width * 0.9,
            color="#245ba3", label="PyQCU")
    if quda_values:
        top.bar(
            quda_x, quda_values, width=width * 0.9,
            color="#be4137", label="QUDA")
    for x_values, values, labels in (
            (pyqcu_x, pyqcu_values, pyqcu_labels),
            (quda_x, quda_values, quda_labels)):
        for x_value, value, label in zip(x_values, values, labels):
            top.text(
                x_value, value, label, ha="center", va="bottom", fontsize=7)

    ratios: List[float] = []
    ratio_x: List[float] = []
    ratio_labels: List[str] = []
    for x_value, case_id in zip(x_base, case_ids):
        pair = by_case[case_id]
        pyqcu = (
            None if "pyqcu" not in pair
            else _float(pair["pyqcu"].get("total_seconds")))
        quda = (
            None if "quda" not in pair
            else _float(pair["quda"].get("total_seconds")))
        ratio = (
            pyqcu / quda
            if pyqcu is not None and quda not in (None, 0.0) else None)
        if ratio is None:
            bottom.text(
                x_value, 1.0, "NA", ha="center", va="center",
                fontsize=7, color="#8b1a1a")
        else:
            ratio_x.append(x_value)
            ratios.append(ratio)
            ratio_labels.append(f"{ratio:.3f}")
    if ratios:
        bottom.bar(
            ratio_x, ratios, width=0.55, color="#5a6973", label="PyQCU/QUDA")
        for x_value, ratio, label in zip(ratio_x, ratios, ratio_labels):
            bottom.text(
                x_value, ratio, label, ha="center", va="bottom", fontsize=7)
    bottom.axhline(1.0, color="#be4137", linestyle="--", linewidth=1.0)
    bottom.set_ylabel("relative time\nPyQCU/QUDA")
    top.set_ylabel("steady seconds")
    top.set_title("PyQCU/QUDA steady time and device-labelled comparison")
    top.grid(axis="y", alpha=0.22)
    bottom.grid(axis="y", alpha=0.22)
    if top.get_legend_handles_labels()[0]:
        top.legend(fontsize=8)
    if bottom.get_legend_handles_labels()[0]:
        bottom.legend(fontsize=8)

    bottom.set_xticks(x_base)
    labels = []
    for case_id in case_ids:
        pair = by_case[case_id]
        sample = next(iter(pair.values()))
        labels.append(
            _case_label(sample) + "\n"
            f"P:{pair.get('pyqcu', {}).get('device') or 'missing'} | "
            f"Q:{pair.get('quda', {}).get('device') or 'missing'}")
    bottom.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    _save_figure(fig, svg, pdf)
    plt.close(fig)


def _latex_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    replacements = (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    )
    for source, target in replacements:
        text = text.replace(source, target)
    return text


def _latex_table(caption: str, headers: Sequence[str],
                 rows: Sequence[Sequence[Any]], spec: str) -> str:
    if not rows:
        rows = [["---"] * len(headers)]
    if len(spec) != len(headers):
        raise ReportError("LaTeX table specification/header mismatch")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{_latex_escape(caption)}}}",
        r"\small",
        rf"\begin{{tabular}}{{@{{}}{spec}@{{}}}}",
        r"\toprule",
        " & ".join(_latex_escape(header) for header in headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        if len(row) != len(headers):
            raise ReportError("LaTeX table row/column mismatch")
        lines.append(
            " & ".join(_latex_escape(value) for value in row) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}", r"\end{table}"))
    return "\n".join(lines)


def _write_tables(path: Path, records: Sequence[Mapping[str, Any]],
                  level_rows: Sequence[Mapping[str, Any]],
                  analysis: Mapping[str, Any]) -> None:
    timing_rows = []
    for row in level_rows:
        timing_rows.append([
            row["case_id"], row["side"], row["level"],
            _format_number(row.get("pre_smoother")),
            _format_number(row.get("post_smoother")),
            _format_number(row.get("restriction")),
            _format_number(row.get("prolongation")),
            _format_number(row.get("coarse_solver")),
            _format_number(row.get("other")),
            _format_number(row.get("total_seconds")),
        ])
    timing = _latex_table(
        "MG level time decomposition (seconds)",
        ("Case", "Side", "L", "Pre", "Post", "Restrict", "Prolong",
         "Coarse", "Other", "Total"),
        timing_rows, "llrrrrrrrr")

    iteration_rows = [[
        row["case_id"], row["side"], row["level"],
        "" if row.get("total_iterations") is None
        else row["total_iterations"],
    ] for row in level_rows]
    iterations = _latex_table(
        "MG level iteration counts; level 0 is outer FGMRES/GCR total",
        ("Case", "Side", "L", "Iterations"),
        iteration_rows, "llrr")

    residual_rows = [[
        record["case_id"], record["side"], record["phase"],
        "" if record.get("finest_iterations") is None
        else record["finest_iterations"],
        _format_number(record.get("finest_residual")),
    ] for record in records]
    residuals = _latex_table(
        "Finest-level outer iteration and true residual",
        ("Case", "Side", "Phase", "Outer iterations", "Relative residual"),
        residual_rows, "llrrr")

    coverage = _mapping(analysis.get("coverage"))
    coverage_rows = [
        ["Side cells", coverage.get("side_cells_observed", ""),
         coverage.get("side_cells_expected_from_observed_cases", ""),
         _format_number(coverage.get("side_coverage"))],
        ["Phase cells", coverage.get("phase_cells_observed", ""),
         coverage.get("phase_cells_expected_from_observed_cases", ""),
         _format_number(coverage.get("phase_coverage"))],
        ["Level cells", coverage.get("level_cells_observed", ""),
         coverage.get("level_cells_expected_from_declared_levels", ""),
         _format_number(coverage.get("level_coverage"))],
        ["Trace curves", len(coverage.get("trace_curves", [])),
         coverage.get("trace_curves_expected", ""),
         _format_number(coverage.get("trace_curve_coverage"))],
    ]
    coverage_table = _latex_table(
        "Coverage and missing-cell audit",
        ("Quantity", "Observed", "Expected", "Coverage"),
        coverage_rows, "lrrr")

    content = "\n\n".join((
        "% Generated by examples/qcu/dev87/build_mg_report.py",
        timing,
        iterations,
        residuals,
        coverage_table,
    )) + "\n"
    path.write_text(content, encoding="utf-8")


def validate_latex_tables(text: str) -> List[Dict[str, int]]:
    """Validate booktabs macros and tabular column counts."""

    if r"\toprule" not in text or r"\midrule" not in text or \
            r"\bottomrule" not in text:
        raise ReportError("LaTeX fragment lacks booktabs macros")
    results: List[Dict[str, int]] = []
    cursor = 0
    while True:
        start = text.find(r"\begin{tabular}", cursor)
        if start < 0:
            break
        spec_start = text.find(
            "{", start + len(r"\begin{tabular}"))
        if spec_start < 0:
            raise ReportError("malformed tabular specification")
        depth = 0
        spec_end = -1
        for index in range(spec_start, len(text)):
            token = text[index]
            if token == "{":
                depth += 1
            elif token == "}":
                depth -= 1
                if depth == 0:
                    spec_end = index
                    break
        if spec_end < 0:
            raise ReportError("unterminated tabular specification")
        spec = text[spec_start + 1:spec_end]
        columns = sum(spec.count(letter) for letter in ("l", "r", "c"))
        body_start = spec_end + 1
        body_end = text.find(r"\end{tabular}", body_start)
        if body_end < 0:
            raise ReportError("unterminated tabular environment")
        body = text[body_start:body_end]
        rows = 0
        for line in body.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("\\") or \
                    stripped.startswith("%"):
                continue
            if r"\\" not in stripped:
                continue
            payload = stripped.replace(r"\\", "").strip()
            amp_count = payload.count("&")
            if amp_count != columns - 1:
                raise ReportError(
                    f"tabular row has {amp_count + 1} columns, expected {columns}")
            rows += 1
        results.append({"columns": columns, "rows": rows})
        cursor = body_end + len(r"\end{tabular}")
    if not results:
        raise ReportError("no tabular environments found")
    return results


def _write_generated_tex(path: Path, tex_source: Optional[Path],
                         analysis: Mapping[str, Any]) -> None:
    source_note = (
        "% Optional read-only source: "
        f"{tex_source}\n" if tex_source is not None else "")
    content = (
        "% Generated include fragment; the source report is never modified.\n"
        f"{source_note}"
        r"\section*{Automated MultiGrid performance report}" + "\n"
        r"\input{\detokenize{tables.tex}}" + "\n\n"
        r"\begin{figure}[htbp]" + "\n"
        r"\centering" + "\n"
        r"\includegraphics[width=\linewidth]{\detokenize{mg_level_time_stack.pdf}}" + "\n"
        r"\caption{Per-level MG time decomposition.}"
        "\n" r"\end{figure}" + "\n\n"
        r"\begin{figure}[htbp]" + "\n"
        r"\centering" + "\n"
        r"\includegraphics[width=\linewidth]{\detokenize{mg_level_iterations.pdf}}" + "\n"
        r"\caption{Per-level iteration counts.}"
        "\n" r"\end{figure}" + "\n\n"
        r"\begin{figure}[htbp]" + "\n"
        r"\centering" + "\n"
        r"\includegraphics[width=\linewidth]{\detokenize{mg_finest_residual.pdf}}" + "\n"
        r"\caption{Finest-level residual convergence.}"
        "\n" r"\end{figure}" + "\n\n"
        r"\begin{figure}[htbp]" + "\n"
        r"\centering" + "\n"
        r"\includegraphics[width=\linewidth]{\detokenize{mg_speedup_device.pdf}}" + "\n"
        r"\caption{PyQCU/QUDA steady time ratio and devices.}"
        "\n" r"\end{figure}" + "\n\n"
        "% Analysis JSON: summary_analysis.json\n"
    )
    path.write_text(content, encoding="utf-8")


def build_report(inputs: Sequence[Path], outdir: Path,
                 tex_source: Optional[Path] = None) -> Dict[str, Any]:
    """Build all report artifacts and return the machine-readable summary."""

    input_paths = _discover_inputs([Path(path) for path in inputs])
    cases = collect_case_inputs(input_paths)
    if not cases:
        raise ReportError("no matrix/trace cases found")

    records, level_rows, curves = _collect_records(cases)
    if not records:
        raise ReportError("no side records could be produced from inputs")
    analysis = _summarise(records, level_rows, curves, input_paths)
    outdir.mkdir(parents=True, exist_ok=True)

    _write_csv(outdir / "mg_matrix.csv", _csv_rows(records))
    _write_csv(outdir / "mg_levels.csv", _level_csv_rows(level_rows))
    (outdir / "summary_analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    _plot_level_times(
        level_rows,
        outdir / "mg_level_time_stack.svg",
        outdir / "mg_level_time_stack.pdf")
    _plot_level_iterations(
        level_rows,
        outdir / "mg_level_iterations.svg",
        outdir / "mg_level_iterations.pdf")
    _plot_residuals(
        curves,
        records,
        outdir / "mg_finest_residual.svg",
        outdir / "mg_finest_residual.pdf")
    _plot_speedup(
        records,
        outdir / "mg_speedup_device.svg",
        outdir / "mg_speedup_device.pdf")

    _write_tables(outdir / "tables.tex", records, level_rows, analysis)
    validate_latex_tables((outdir / "tables.tex").read_text(encoding="utf-8"))
    _write_generated_tex(
        outdir / "report_generated.tex", tex_source, analysis)
    return analysis


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build CSV/SVG/PDF/LaTeX reports from MG matrix JSON/trace")
    parser.add_argument(
        "--input", type=Path, nargs="+", required=True,
        help="One or more matrix JSON, collector JSON, trace JSON, or TSV files")
    parser.add_argument(
        "--outdir", type=Path, required=True,
        help="Directory for generated report artifacts")
    parser.add_argument(
        "--tex", type=Path, default=None,
        help="Optional source TeX report read for provenance; never overwritten")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.tex is not None and not args.tex.is_file():
        raise ReportError(f"--tex file does not exist: {args.tex}")
    analysis = build_report(args.input, args.outdir, args.tex)
    print(json.dumps({
        "outdir": str(args.outdir.resolve()),
        "records": analysis["records"],
        "level_rows": analysis["level_rows"],
        "trace_curve_count": analysis["trace_curve_count"],
        "missing_field_counts": analysis["coverage"]["missing_field_counts"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ReportError, ValueError) as exc:
        print(f"build_mg_report: {exc}", file=sys.stderr)
        raise SystemExit(1)
