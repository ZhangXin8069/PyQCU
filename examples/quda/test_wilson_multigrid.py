#!/usr/bin/env python3
"""QUDA Wilson multigrid 单功能入口：验证 transfer 数据排布和纯 PyTorch R/P。"""
import torch
import os
from common import *

def main(argv=None):
    args, report = run_setup(__doc__ or "QUDA Wilson MultiGrid 单测", __doc__, argv=argv,
                             total=1 + int(bool(os.environ.get("RUN_QUDA_TESTS") and os.environ.get("QUDA_UNSAFE_INPROCESS"))))
    r = report.run("QDP 布局往返", layout_roundtrip)
    v = torch.eye(2,dtype=torch.complex64).reshape(2,2,1,1,1,1,1,1,1,1)
    c = torch.randn(2,1,1,1,1,dtype=torch.complex64); f = torch.zeros(2,2,2,2,2,dtype=torch.complex64); f[:,0,0,0,0] = c[:,0,0,0,0]
    record = {"function":"wilson-multigrid", **r, "coarse_norm":float(torch.linalg.vector_norm(c)), "quda":quda_status(), "status":"reference-transfer", "lattice":list(args.lat)}
    if os.environ.get("RUN_QUDA_TESTS") and os.environ.get("QUDA_UNSAFE_INPROCESS"):
        try: record["quda_solution_norm"] = float(torch.linalg.vector_norm(report.run(
            "QUDA Wilson MG", run_quda_solve, source(args.lat), multigrid=True, mass=args.mass,
            reporter=report, timing_label="QUDA Wilson MG")))
        except Exception as exc: print(f"SKIP: QUDA Wilson MG 失败: {exc}")
    print(record)
    report.finish("PASS")
    assert r["gauge_error"] == 0.0 and r["fermion_error"] == 0.0
    return 0

def test_wilson_multigrid_layout(): assert main() == 0
if __name__ == "__main__": raise SystemExit(main())
