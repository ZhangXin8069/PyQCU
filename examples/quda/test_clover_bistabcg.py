#!/usr/bin/env python3
"""QUDA Clover BiCGStab 单功能参考测试。"""
import torch
import os
from common import *
from pyqcu import solver

def main():
    u,b = identity_gauge(), source(); k = torch.tensor([1.0/(2*MASS+8.0)])
    cl = dslash.make_clover(u,kappa=k)
    op = lambda x: dslash.give_wilson(x,u,k,with_I=True) + dslash.give_clover(x,cl)
    x = solver.bistabcg(b, op, tol=1e-5, max_iter=100, verbose=False)
    residual = float(torch.linalg.vector_norm(op(x)-b)/torch.linalg.vector_norm(b))
    record = {"function":"clover-bistabcg", "true_residual":residual, "quda":quda_status(), "status":"reference-residual"}
    if os.environ.get("RUN_QUDA_TESTS") and os.environ.get("QUDA_UNSAFE_INPROCESS"):
        try:
            qx = run_quda_solve(b, clover=True)
            record["quda_solution_rel"] = rel(qx * (MASS + 4.0), x)
        except Exception as exc: print(f"SKIP: QUDA Clover 求解失败: {exc}")
    print(record)
    assert residual < 1e-3
    return 0

def test_clover_bistabcg_reference(): assert main() == 0
if __name__ == "__main__": raise SystemExit(main())
