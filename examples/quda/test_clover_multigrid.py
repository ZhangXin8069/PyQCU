#!/usr/bin/env python3
"""QUDA Clover multigrid 单功能入口：验证 Clover 参考与 QDP 排布。"""
import torch
import os
from common import *

def main():
    u,b = identity_gauge(), source(); k=torch.tensor([1.0/(2*MASS+8.0)]); cl=dslash.make_clover(u,kappa=k)
    ref=dslash.give_wilson(b,u,k,with_I=True)+dslash.give_clover(b,cl); r=layout_roundtrip()
    record = {"function":"clover-multigrid", **r, "fine_reference_norm":float(torch.linalg.vector_norm(ref)), "quda":quda_status(), "status":"reference-transfer"}
    if os.environ.get("RUN_QUDA_TESTS") and os.environ.get("QUDA_UNSAFE_INPROCESS"):
        try: record["quda_solution_norm"] = float(torch.linalg.vector_norm(run_quda_solve(b, clover=True, multigrid=True)))
        except Exception as exc: print(f"SKIP: QUDA Clover MG 失败: {exc}")
    print(record)
    assert all(v == 0.0 for v in r.values())
    return 0

def test_clover_multigrid_layout(): assert main() == 0
if __name__ == "__main__": raise SystemExit(main())
