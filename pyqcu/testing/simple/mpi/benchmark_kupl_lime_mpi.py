#!/usr/bin/env python3
"""Native-MPI wrapper for the existing KUPL LIME benchmark module."""

from __future__ import annotations

from collections.abc import Sequence

from pyqcu.testing.simple.mpi._entrypoint import run_native_module


def main(argv: Sequence[str] | None = None) -> int:
    return run_native_module(
        "pyqcu.testing.simple.benchmark_kupl_lime_mpi", argv,
        description=__doc__, atomic_output_flags=("--report",))


if __name__ == "__main__":
    raise SystemExit(main())
