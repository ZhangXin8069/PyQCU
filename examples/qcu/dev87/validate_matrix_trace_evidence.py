#!/usr/bin/env python3
"""Fail-closed trace and derived-reference checks for matrix result documents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: document root is not an object")
    return value


def validate_document(path: Path, document: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    sides = document.get("sides")
    if not isinstance(sides, Mapping):
        return [f"{path}: sides is not an object"]

    for side, raw in sides.items():
        if not isinstance(raw, Mapping):
            continue
        if raw.get("status") != "ok" or "trace-on" not in path.name:
            continue
        trace_paths = raw.get("trace_paths")
        if not isinstance(trace_paths, Mapping):
            errors.append(f"{path}: {side}.trace_paths missing")
            continue
        if isinstance(trace_paths, Mapping):
            required_names = (
                ("PYQCU_STRICT_TRACE_FILE",)
                if side == "pyqcu" else
                ("QUDA_MG_TRACE_FILE", "PYQCU_QUDA_TRACE_FILE"))
            for name in required_names:
                value = trace_paths.get(name)
                if not isinstance(value, str) or not value:
                    errors.append(f"{path}: {side}.{name} is empty")
                    continue
                trace = Path(value)
                if not trace.is_file():
                    errors.append(f"{path}: {side}.{name} missing: {trace}")
                elif trace.stat().st_size == 0:
                    errors.append(f"{path}: {side}.{name} is empty: {trace}")

    if document.get("derived_kind") != "bicgstab-reference":
        return errors
    if document.get("unit_levels") != 1:
        errors.append(f"{path}: derived unit_levels must be 1")
    for side, raw in sides.items():
        if not isinstance(raw, Mapping):
            continue
        if raw.get("status") != "ok":
            continue
        reference = raw.get("reference_solver")
        if not isinstance(reference, Mapping):
            errors.append(f"{path}: {side}.reference_solver missing")
            continue
        if not isinstance(reference.get("cold"), Mapping):
            errors.append(f"{path}: {side} reference cold missing")
        warmups = reference.get("warmups")
        if not isinstance(warmups, list) or len(warmups) != 2:
            errors.append(f"{path}: {side} reference warmups != 2")
        samples = reference.get("samples")
        if not isinstance(samples, list) or len(samples) != 5:
            errors.append(f"{path}: {side} reference steady samples != 5")
    return errors


def validate_paths(paths: Iterable[Path]) -> list[str]:
    errors: list[str] = []
    for path in sorted({Path(item).resolve() for item in paths}):
        errors.extend(validate_document(path, _load(path)))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    files: list[Path] = []
    for input_path in args.inputs:
        if input_path.is_dir():
            files.extend(sorted(input_path.glob("*.json")))
        else:
            files.append(input_path)
    errors = validate_paths(files)
    report = {
        "status": "pass" if not errors else "fail",
        "document_count": len(files),
        "error_count": len(errors),
        "errors": errors,
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
