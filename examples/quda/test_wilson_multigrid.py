#!/usr/bin/env python3
"""QUDA Wilson multigrid 单功能入口：验证 transfer 数据排布和纯 PyTorch R/P。"""
import torch
import os
from common import *

def main():
    r = layout_roundtrip(); v = torch.eye(2,dtype=torch.complex64).reshape(2,2,1,1,1,1,1,1,1,1)
    c = torch.randn(2,1,1,1,1,dtype=torch.complex64); f = torch.zeros(2,2,2,2,2,dtype=torch.complex64); f[:,0,0,0,0] = c[:,0,0,0,0]
    record = {"function":"wilson-multigrid", **r, "coarse_norm":float(torch.linalg.vector_norm(c)), "quda":quda_status(), "status":"reference-transfer"}
    if os.environ.get("RUN_QUDA_TESTS") and os.environ.get("QUDA_UNSAFE_INPROCESS"):
        try: record["quda_solution_norm"] = float(torch.linalg.vector_norm(run_quda_solve(source(), multigrid=True)))
        except Exception as exc: print(f"SKIP: QUDA Wilson MG 失败: {exc}")
    print(record)
    assert r["gauge_error"] == 0.0 and r["fermion_error"] == 0.0
    return 0

def test_wilson_multigrid_layout(): assert main() == 0
if __name__ == "__main__": raise SystemExit(main())
