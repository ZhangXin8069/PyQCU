#!/usr/bin/env python3
"""Validate an MPI world size before launching a distributed example."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def is_power_of_two(ranks: int) -> bool:
    """Return whether ``ranks`` is a positive power of two."""
    value = int(ranks)
    return value > 0 and (value & (value - 1)) == 0


def validate_rank_count(ranks: int) -> int:
    """Return a validated rank count or raise a clear ``ValueError``."""
    value = int(ranks)
    if not is_power_of_two(value):
        raise ValueError(
            f"MPI rank count must be a positive power of two; got {value}")
    return value


def _rank_argument(value: str) -> int:
    try:
        return int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"rank count must be a base-10 integer; got {value!r}") from exc


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reject MPI rank counts that are not positive powers of two.")
    parser.add_argument(
        "--quiet", action="store_true",
        help="produce no output when validation succeeds")
    parser.add_argument("ranks", type=_rank_argument, help="MPI world size")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        ranks = validate_rank_count(args.ranks)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not args.quiet:
        print(f"MPI rank count {ranks} is a positive power of two.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
