#!/usr/bin/env python3
"""Check that KUPL exports exactly the declared QCU C ABI surface."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


_SYMBOL = re.compile(r"\b((?:apply|test|get|verify)[A-Za-z0-9]+Qcu)\s*\(")
_EXPORTED_SYMBOL = re.compile(
    r"(?:apply|test|get|verify)[A-Za-z0-9]+Qcu"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--header", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    expected = set(_SYMBOL.findall(args.header.read_text(encoding="utf-8")))
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

    missing = sorted(expected - exported)
    unexpected = sorted(exported - expected)
    if len(expected) != 37:
        print(f"expected header declares 37 QCU symbols, found {len(expected)}")
    if missing:
        print("missing exports:", ", ".join(missing))
    if unexpected:
        print("unexpected exports:", ", ".join(unexpected))
    if len(expected) != 37 or missing or unexpected:
        return 1
    print(f"PASS: {len(expected)} declared QCU symbols exported")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
