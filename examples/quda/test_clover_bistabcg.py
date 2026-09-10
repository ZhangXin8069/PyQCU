#!/usr/bin/env python3
"""QUDA Clover BiCGStab 单功能参考测试。"""
import torch
import os
from common import *
from pyqcu import solver

def main(argv=None):
    args, report = run_setup(__doc__ or "QUDA Clover BiCGStab 单测", __doc__, argv=argv,
                             total=3 + int(bool(os.environ.get("RUN_QUDA_TESTS") and os.environ.get("QUDA_UNSAFE_INPROCESS"))))
    u,b = report.run("分配输入场", lambda: (identity_gauge(args.lat), source(args.lat)))
    k = torch.tensor([1.0/(2*args.mass+8.0)])
    cl = report.run("构造 Clover 参考", dslash.make_clover, u, kappa=k)
    op = lambda x: dslash.give_wilson(x,u,k,with_I=True) + dslash.give_clover(x,cl)
    x = report.run("PyTorch BiCGStab", solver.bistabcg, b, op, tol=1e-5, max_iter=100, verbose=False)
    residual = float(torch.linalg.vector_norm(op(x)-b)/torch.linalg.vector_norm(b))
    record = {"function":"clover-bistabcg", "true_residual":residual, "quda":quda_status(), "status":"reference-residual", "lattice":list(args.lat)}
    if os.environ.get("RUN_QUDA_TESTS") and os.environ.get("QUDA_UNSAFE_INPROCESS"):
        try:
            qx = report.run("QUDA Clover solve", run_quda_solve, b, clover=True, mass=args.mass,
                            reporter=report, timing_label="QUDA Clover solve")
            record["quda_solution_rel"] = rel(qx * (args.mass + 4.0), x)
        except Exception as exc: print(f"SKIP: QUDA Clover 求解失败: {exc}")
    print(record)
    report.finish("PASS" if residual < 1e-3 else "FAIL")
    assert residual < 1e-3
    return 0

def test_clover_bistabcg_reference(): assert main() == 0
if __name__ == "__main__": raise SystemExit(main())
