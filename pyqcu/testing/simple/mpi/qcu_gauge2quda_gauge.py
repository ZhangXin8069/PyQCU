#!/usr/bin/env python3
"""MPI rank-zero entry for QCU-to-QUDA gauge conversion."""

from __future__ import annotations

from collections.abc import Sequence

from pyqcu.testing.simple.mpi._entrypoint import run_rank0_module


def main(argv: Sequence[str] | None = None) -> int:
    return run_rank0_module(
        "pyqcu.testing.simple.qcu_gauge2quda_gauge", argv,
        description=__doc__, atomic_output_flags=("--output",))


if __name__ == "__main__":
    raise SystemExit(main())
