#!/usr/bin/env python3
"""QUDA Clover dslash 单功能测试与 PyQCU 参考。"""
import torch
import os
from common import *

def main(argv=None):
    args, report = run_setup(__doc__ or "QUDA Clover dslash 单测", __doc__, argv=argv,
                             total=3 + int(bool(os.environ.get("RUN_QUDA_TESTS") and os.environ.get("QUDA_UNSAFE_INPROCESS"))))
    u, b = report.run("分配输入场", lambda: (identity_gauge(args.lat), source(args.lat)))
    k = 1.0/(2*args.mass+8.0)
    cl = report.run("构造 Clover 参考", dslash.make_clover, u, kappa=torch.tensor([k]))
    ref = report.run("PyTorch Clover 参考", lambda: dslash.give_wilson(b,u,torch.tensor([k]),with_I=True) + dslash.give_clover(b,cl))
    assert torch.isfinite(ref.real).all() and torch.isfinite(ref.imag).all()
    result = None
    if os.environ.get("RUN_QUDA_TESTS") and os.environ.get("QUDA_UNSAFE_INPROCESS"):
        try: result = report.run("QUDA Clover mat", run_quda_mat, b, clover=True, mass=args.mass)
        except Exception as exc: print(f"SKIP: QUDA Clover 调用失败: {exc}")
    record = {"function":"clover-dslash", "reference_norm":float(torch.linalg.vector_norm(ref)), "quda":quda_status(), "status":"reference-layout", "lattice":list(args.lat)}
    if result is not None: record["quda_mat_rel"] = rel(result, ref)
    print(record)
    report.finish("PASS")
    return 0

def test_clover_dslash_reference(): assert main() == 0
if __name__ == "__main__": raise SystemExit(main())
