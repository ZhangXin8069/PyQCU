#!/usr/bin/env python3
"""Check that KUPL matches and exports the CUDA QCU C ABI surface."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


_DECLARATION = re.compile(
    r"\b(?P<return_type>void|int)\s+"
    r"(?P<name>(?:apply|test|get|verify)[A-Za-z0-9]+Qcu)\s*"
    r"\((?P<arguments>.*?)\)\s*;",
    re.DOTALL,
)
_EXPORTED_SYMBOL = re.compile(
    r"(?:apply|test|get|verify)[A-Za-z0-9]+Qcu"
)


def _declarations(path: Path) -> dict[str, tuple[str, tuple[str, ...]]]:
    declarations: dict[str, tuple[str, tuple[str, ...]]] = {}
    for match in _DECLARATION.finditer(path.read_text(encoding="utf-8")):
        arguments = []
        for argument in match.group("arguments").split(","):
            normalized = " ".join(argument.split())
            normalized = re.sub(r"\s+[A-Za-z_]\w*$", "", normalized)
            normalized = re.sub(r"\*[A-Za-z_]\w*$", "*", normalized)
            normalized = re.sub(r"\s*\*\s*", "*", normalized)
            arguments.append(normalized)
        declarations[match.group("name")] = (
            match.group("return_type"),
            tuple(arguments),
        )
    return declarations


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument("--cuda-header", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    declared = _declarations(args.header)
    cuda_declared = _declarations(args.cuda_header)
    expected = set(declared)
    output = subprocess.run(
        ["nm", "-D", "--defined-only", str(args.library)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    exported = set()
    for line in output.splitlines():
        fields = line.split()
        if not fields:
            continue
        symbol = fields[-1].split("@", 1)[0]
        if _EXPORTED_SYMBOL.fullmatch(symbol):
            exported.add(symbol)

    signature_mismatches = sorted(
        name
        for name in expected & set(cuda_declared)
        if declared[name] != cuda_declared[name]
    )
    missing = sorted(expected - exported)
    unexpected = sorted(exported - expected)
    if len(expected) != 37:
        print(f"expected header declares 37 QCU symbols, found {len(expected)}")
    if missing:
        print("missing exports:", ", ".join(missing))
    if unexpected:
        print("unexpected exports:", ", ".join(unexpected))
    if set(declared) != set(cuda_declared):
        print(
            "declaration set mismatch:",
            f"kupl_only={sorted(expected - set(cuda_declared))}",
            f"cuda_only={sorted(set(cuda_declared) - expected)}",
        )
    for name in signature_mismatches:
        print(f"signature mismatch {name}: kupl={declared[name]} cuda={cuda_declared[name]}")
    if (len(expected) != 37 or set(declared) != set(cuda_declared) or
            signature_mismatches or missing or unexpected):
        return 1
    print(
        f"PASS: {len(expected)} CUDA-compatible declarations exported")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
