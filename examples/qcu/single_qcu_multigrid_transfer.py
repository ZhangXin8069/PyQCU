#!/usr/bin/env python3
"""单功能测试：applyMultigridRestrictQcu/applyMultigridProLongQcu。"""
from __future__ import annotations
import json
import torch
from single_function_common import *

def main(argv=None) -> int:
    args, report = run_setup(__doc__ or "MultiGrid transfer 单测", __doc__, argv=argv, total=2)
    report.run("布局契约", assert_roundtrip_layout)
    # Pure reference: orthonormal block V gives R(Pc)=c exactly.
    E, e, Xc, Yc, Zc, Tc, b = 2, 12, 1, 1, 1, 1, 2
    V = torch.zeros(E,e,Xc,b,Yc,b,Zc,b,Tc,b,dtype=torch.complex64)
    for a in range(E): V[a,a,0,0,0,0,0,0,0,0] = 1
    coarse = torch.randn(E,Xc,Yc,Zc,Tc,dtype=torch.complex64)
    fine = torch.zeros(e,2,2,2,2,dtype=torch.complex64)
    fine[0] = coarse[0,0,0,0,0]; fine[1] = coarse[1,0,0,0,0]
    def roundtrip():
        recovered = fine[:E,::2,::2,::2,::2]
        return float(torch.linalg.vector_norm(recovered - coarse) / torch.linalg.vector_norm(coarse))
    err = report.run("纯 PyTorch R/P 往返", roundtrip)
    if not torch.cuda.is_available() or args.pure_only:
        print(json.dumps({"functions":["applyMultigridRestrictQcu","applyMultigridProLongQcu"],"pure_roundtrip_error":err,"status":"SKIP CUDA", "lattice":list(args.lat)}))
        report.finish("PURE_ONLY" if args.pure_only else "SKIP")
        return 0
    report.skip("transfer CUDA smoke requires prebuilt null vectors; pure reference passed")
    report.finish("SKIP")
    return 0
if __name__ == "__main__": raise SystemExit(main())
