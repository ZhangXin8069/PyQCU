#!/usr/bin/env python3
"""Assemble final PyQCU/QUDA side records into audited report artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(HERE))

from pyqcu.testing.qcu.strict.quda_comparison import bench_strict_vs_quda as collector


PHASE_FIELDS = (
    "pre_smoother",
    "post_smoother",
    "restrict",
    "prolongate",
    "coarse_solver",
    "other",
)


def index_sides(roots: Iterable[Path], side: str) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    prefix = f"{side}__"
    for root in roots:
        for path in sorted(root.glob(f"{prefix}*.json")):
            key = path.name.removeprefix(prefix)
            if key in indexed:
                raise ValueError(f"duplicate {side} side: {key}")
            indexed[key] = path
    return indexed


def parse_unit_id(unit_id: str) -> dict[str, Any]:
    name = unit_id.removesuffix(".json")
    parts = name.split("__")
    if len(parts) != 5:
        raise ValueError(f"invalid matrix unit id: {unit_id}")
    device, lattice, precision, level_tag, trace_tag = parts
    return {
        "unit_id": name,
        "device": device,
        "lattice": lattice,
        "precision": precision,
        "levels": int(level_tag.removeprefix("l")),
        "trace": trace_tag.removeprefix("trace-"),
    }


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def stage_rows(
        suffix: str, side: str, record: Mapping[str, Any],
) -> list[dict[str, Any]]:
    metadata = parse_unit_id(suffix)
    rows: list[dict[str, Any]] = []
    for raw in record.get("mg_levels") or []:
        if not isinstance(raw, Mapping):
            continue
        rows.append({
            **metadata,
            "side": side,
            "level": int(raw["level"]),
            "total_seconds": _number(raw.get("total_seconds")),
            "total_iterations": _number(raw.get("total_iterations")),
            "pre_smoother_seconds": _number(
                raw.get("pre_smoother_seconds")),
            "pre_smoother_iterations": _number(
                raw.get("pre_smoother_iterations")),
            "post_smoother_seconds": _number(
                raw.get("post_smoother_seconds")),
            "post_smoother_iterations": _number(
                raw.get("post_smoother_iterations")),
            "restrict_seconds": _number(raw.get("restriction_seconds")),
            "restrict_calls": _number(raw.get("restriction_calls")),
            "prolongate_seconds": _number(raw.get("prolongation_seconds")),
            "prolongate_calls": _number(raw.get("prolongation_calls")),
            "coarse_solver_seconds": _number(
                raw.get("coarse_solver_seconds")),
            "coarse_solver_iterations": _number(
                raw.get("coarse_solver_iterations")),
            "coarse_solver_calls": _number(
                raw.get("coarse_solver_calls")),
            "other_seconds": _number(raw.get("other_seconds")),
            "final_residual": _number(raw.get("final_residual")),
        })
    return rows


def reference_rows(
        suffix: str, side: str, record: Mapping[str, Any],
) -> list[dict[str, Any]]:
    metadata = parse_unit_id(suffix)
    reference = record.get("reference_solver")
    if not isinstance(reference, Mapping):
        return []
    rows: list[dict[str, Any]] = []
    for phase, samples in (
            ("cold", [reference.get("cold")]),
            ("warmup", list(reference.get("warmups") or [])),
            ("steady", list(reference.get("samples") or []))):
        for index, sample in enumerate(samples):
            if not isinstance(sample, Mapping):
                continue
            rows.append({
                **metadata,
                "side": side,
                "phase": phase,
                "sample": index,
                "seconds": _number(sample.get("seconds")),
                "iterations": _number(sample.get("iterations")),
                "true_residual": _number(sample.get("true_residual_rel")),
                "converged": sample.get("converged"),
            })
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: document root is not an object")
    return value


def assemble(
        pyqcu_roots: Sequence[Path],
        quda_roots: Sequence[Path],
        outdir: Path,
) -> dict[str, Any]:
    pyqcu = index_sides(pyqcu_roots, "pyqcu")
    quda = index_sides(quda_roots, "quda")
    expected = set(pyqcu)
    missing_quda = sorted(expected - set(quda))
    missing_pyqcu = sorted(set(quda) - expected)
    if missing_quda or missing_pyqcu:
        raise ValueError(
            f"incomplete pairs: missing_quda={missing_quda}, "
            f"missing_pyqcu={missing_pyqcu}")

    unit_rows: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    combined_dir = outdir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    errors: list[dict[str, Any]] = []
    for suffix in sorted(expected):
        py_path = pyqcu[suffix]
        qu_path = quda[suffix]
        py_document = _load(py_path)
        qu_document = _load(qu_path)
        py_record = py_document["sides"]["pyqcu"]
        qu_record = qu_document["sides"]["quda"]
        metadata = parse_unit_id(suffix)
        py_hash = py_record.get("config_hash")
        qu_hash = qu_record.get("config_hash")
        py_input = (py_document.get("input_fingerprints") or {}).get(
            "bundle_hash")
        qu_input = (qu_document.get("input_fingerprints") or {}).get(
            "bundle_hash")
        py_reference = list((py_record.get("reference_solver") or {}).get(
            "warmups") or [])
        qu_reference = list((qu_record.get("reference_solver") or {}).get(
            "warmups") or [])
        checks = {
            "status_ok": (
                py_record.get("status") == "ok" and
                qu_record.get("status") == "ok"),
            "config_hash": py_hash == qu_hash,
            "input_bundle_hash": (
                isinstance(py_input, str) and py_input == qu_input),
            "residual_pass": (
                bool((py_record.get("true_residual") or {}).get("pass")) and
                bool((qu_record.get("true_residual") or {}).get("pass"))),
            "reference_warmups_two": (
                len(py_reference) == 2 and len(qu_reference) == 2),
        }
        if not all(checks.values()):
            errors.append({**metadata, **checks})

        combined = collector.merge_documents([str(py_path), str(qu_path)])
        combined_path = combined_dir / f"{metadata['unit_id']}.json"
        combined_path.write_text(
            json.dumps(combined, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        comparison = combined.get("comparison") or {}
        if comparison.get("fair") is not True:
            errors.append({
                **metadata,
                "combined_fair": False,
                "reasons": comparison.get("reasons"),
            })

        py_seconds = _number(
            ((py_record.get("timing") or {}).get("steady") or {}).get(
                "median_seconds"))
        qu_seconds = _number(
            ((qu_record.get("timing") or {}).get("steady") or {}).get(
                "median_seconds"))
        py_iterations = _number(
            (py_record.get("iterations") or {}).get("median"))
        qu_iterations = _number(
            (qu_record.get("iterations") or {}).get("median"))
        unit_rows.append({
            **metadata,
            "solver": (
                "bicgstab-reference"
                if metadata["levels"] == 1 else
                f"mg-{metadata['levels']}"),
            "fair": comparison.get("fair"),
            "pyqcu_seconds": py_seconds,
            "quda_seconds": qu_seconds,
            "ratio_quda_over_pyqcu": (
                qu_seconds / py_seconds
                if py_seconds is not None and qu_seconds is not None
                and py_seconds > 0.0 else None),
            "pyqcu_iterations": py_iterations,
            "quda_iterations": qu_iterations,
            "pyqcu_true_residual": (
                py_record.get("true_residual") or {}).get("max_rel"),
            "quda_true_residual": (
                qu_record.get("true_residual") or {}).get("max_rel"),
            "config_hash": py_hash,
            "input_bundle_hash": py_input,
            "config_hash_match": checks["config_hash"],
            "input_bundle_match": checks["input_bundle_hash"],
            "residual_pass": checks["residual_pass"],
            "reference_warmups_two": checks["reference_warmups_two"],
            "combined_path": str(combined_path.resolve()),
        })
        stages.extend(stage_rows(suffix, "pyqcu", py_record))
        stages.extend(stage_rows(suffix, "quda", qu_record))
        references.extend(reference_rows(suffix, "pyqcu", py_record))
        references.extend(reference_rows(suffix, "quda", qu_record))

    mg_ratios = [
        float(row["ratio_quda_over_pyqcu"])
        for row in unit_rows
        if row["levels"] > 1 and row["ratio_quda_over_pyqcu"] is not None
    ]
    audit = {
        "status": "pass" if not errors else "fail",
        "expected_units": len(expected),
        "combined_units": len(unit_rows),
        "fair_units": sum(row["fair"] is True for row in unit_rows),
        "config_hash_matches": sum(
            row["config_hash_match"] is True for row in unit_rows),
        "input_bundle_matches": sum(
            row["input_bundle_match"] is True for row in unit_rows),
        "residual_passes": sum(
            row["residual_pass"] is True for row in unit_rows),
        "reference_warmup_contract_matches": sum(
            row["reference_warmups_two"] is True for row in unit_rows),
        "mg_units": len(mg_ratios),
        "mg_wins": sum(value > 1.0 for value in mg_ratios),
        "mg_losses": sum(value < 1.0 for value in mg_ratios),
        "mg_ratio_median": (
            statistics.median(mg_ratios) if mg_ratios else None),
        "errors": errors,
    }
    _write_csv(outdir / "units.csv", unit_rows)
    _write_csv(outdir / "stages.csv", stages)
    _write_csv(outdir / "references.csv", references)
    (outdir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return audit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pyqcu-root", type=Path, action="append", required=True)
    parser.add_argument("--quda-root", type=Path, action="append", required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    audit = assemble(args.pyqcu_root, args.quda_root, args.outdir)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if audit["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
