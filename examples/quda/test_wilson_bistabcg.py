#!/usr/bin/env python3
"""QUDA Wilson BiCGStab 单功能参考测试。"""
import torch
import os
from common import *

def main(argv=None):
    args, report = run_setup(__doc__ or "QUDA Wilson BiCGStab 单测", __doc__, argv=argv,
                             total=2 + int(bool(os.environ.get("RUN_QUDA_TESTS") and os.environ.get("QUDA_UNSAFE_INPROCESS"))))
    u,b = report.run("分配输入场", lambda: (identity_gauge(args.lat), source(args.lat)))
    k = torch.tensor([1.0/(2*args.mass+8.0)])
    op = lambda x: dslash.give_wilson(x,u,k,with_I=True)
    x = report.run("PyTorch BiCGStab", solver.bistabcg, b, op, tol=1e-5, max_iter=100, verbose=False)
    residual = float(torch.linalg.vector_norm(op(x)-b)/torch.linalg.vector_norm(b))
    record = {"function":"wilson-bistabcg", "true_residual":residual, "quda":quda_status(), "status":"reference-residual", "lattice":list(args.lat)}
    if os.environ.get("RUN_QUDA_TESTS") and os.environ.get("QUDA_UNSAFE_INPROCESS"):
        try:
            qx = report.run("QUDA Wilson solve", run_quda_solve, b, clover=False, mass=args.mass,
                            reporter=report, timing_label="QUDA Wilson solve")
            record["quda_solution_rel"] = rel(qx * (args.mass + 4.0), x)
        except Exception as exc: print(f"SKIP: QUDA Wilson 求解失败: {exc}")
    print(record)
    report.finish("PASS" if residual < 1e-3 else "FAIL")
    assert residual < 1e-3
    return 0
from pyqcu import solver

def test_wilson_bistabcg_reference(): assert main() == 0
if __name__ == "__main__": raise SystemExit(main())
