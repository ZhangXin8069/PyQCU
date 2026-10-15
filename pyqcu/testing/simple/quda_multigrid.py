#!/usr/bin/env python3
"""Minimal QUDA/PyQUDA Clover multigrid example."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from ._common import (DATA_DIR, DEFAULT_ATOL, DEFAULT_BLOCK, DEFAULT_MASS,
                          DEFAULT_NVEC, resolve_path, write_json)
    from .quda_bistabcg import pyquda_available, solve_case
except ImportError:
    from _common import (DATA_DIR, DEFAULT_ATOL, DEFAULT_BLOCK, DEFAULT_MASS,
                         DEFAULT_NVEC, resolve_path, write_json)
    from quda_bistabcg import pyquda_available, solve_case

if str(Path(__file__).resolve().parents[3]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lat", nargs=4, type=int, default=(8, 8, 16, 16))
    parser.add_argument("--mass", type=float, default=None)
    parser.add_argument("--tol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--csw", type=float, default=None)
    parser.add_argument("--gauge", default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    parser.add_argument("--levels", type=int, default=2)
    parser.add_argument("--nvec", type=int, default=DEFAULT_NVEC)
    parser.add_argument("--block", nargs=4, type=int, default=DEFAULT_BLOCK)
    parser.add_argument("--report", default="performance_quda_multigrid.json")
    args = parser.parse_args(argv)
    if args.levels < 2:
        parser.error("QUDA multigrid needs at least two levels")
    if not pyquda_available():
        print(json.dumps({"status": "SKIP", "reason": "PyQUDA/CUDA unavailable"}))
        return 0

    # PyQUDA appends the terminal level itself: pass one block per transition.
    transitions = [list(args.block) for _ in range(args.levels - 1)]
    solution, csw, _, metrics = solve_case(args, multigrid=transitions, nvec=args.nvec)
    nvec_applied = bool(metrics.get("nvec_applied", True))

    # n_vec and other QudaMultigridParam array getters return copies; the
    # worker above receives the geometry, while this snapshot records the
    # requested QUDA-side parameter convention.
    report = {
        "backend": "quda", "function": "getClover+invert",
        "lattice": list(args.lat), "csw": csw,
        "levels": args.levels, "n_vec": args.nvec if nvec_applied else None,
        "n_vec_requested": args.nvec, "block": list(args.block),
        "coarse_spin": 2,
        "coarse_color": args.nvec * 2 if nvec_applied else None,
        "solution_shape": list(solution.shape), "status": "PASS", **metrics,
    }
    report_path = resolve_path(args.report, DATA_DIR / args.report)
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
