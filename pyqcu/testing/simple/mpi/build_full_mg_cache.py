#!/usr/bin/env python3
"""Build a Legacy-MG cache cooperatively across MPI ranks."""

from __future__ import annotations

from collections.abc import Sequence

from pyqcu.testing.simple.mpi._entrypoint import run_native_module


def main(argv: Sequence[str] | None = None) -> int:
    return run_native_module(
        "pyqcu.testing.simple.mpi._distributed_mg_cache", argv,
        description=__doc__, atomic_output_flags=("--output", "--report"))


if __name__ == "__main__":
    raise SystemExit(main())
