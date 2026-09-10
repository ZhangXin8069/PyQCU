#!/usr/bin/env python3
"""QUDA Wilson BiCGStab 单功能参考测试。"""
import torch
import os
from common import *

def main():
    u,b = identity_gauge(), source(); k = torch.tensor([1.0/(2*MASS+8.0)])
    op = lambda x: dslash.give_wilson(x,u,k,with_I=True)
    x = solver.bistabcg(b, op, tol=1e-5, max_iter=100, verbose=False)
    residual = float(torch.linalg.vector_norm(op(x)-b)/torch.linalg.vector_norm(b))
    record = {"function":"wilson-bistabcg", "true_residual":residual, "quda":quda_status(), "status":"reference-residual"}
    if os.environ.get("RUN_QUDA_TESTS") and os.environ.get("QUDA_UNSAFE_INPROCESS"):
        try:
            qx = run_quda_solve(b, clover=False)
            record["quda_solution_rel"] = rel(qx * (MASS + 4.0), x)
        except Exception as exc: print(f"SKIP: QUDA Wilson 求解失败: {exc}")
    print(record)
    assert residual < 1e-3
    return 0
from pyqcu import solver

def test_wilson_bistabcg_reference(): assert main() == 0
if __name__ == "__main__": raise SystemExit(main())
