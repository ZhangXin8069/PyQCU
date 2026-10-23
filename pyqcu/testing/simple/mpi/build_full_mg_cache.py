#!/usr/bin/env python3
"""MPI rank-zero entry for the shared full multigrid cache builder."""

from __future__ import annotations

from collections.abc import Sequence

from pyqcu.testing.simple.mpi._entrypoint import run_rank0_module


def main(argv: Sequence[str] | None = None) -> int:
    return run_rank0_module(
        "pyqcu.testing.simple.build_full_mg_cache", argv,
        description=__doc__, atomic_output_flags=("--output", "--report"))


if __name__ == "__main__":
    raise SystemExit(main())
