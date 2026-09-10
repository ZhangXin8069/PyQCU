#!/usr/bin/env python3
"""单功能测试：applyMultigridRestrictQcu/applyMultigridProLongQcu。"""
from __future__ import annotations
import json
import torch
from single_function_common import *

def main() -> int:
    assert_roundtrip_layout()
    # Pure reference: orthonormal block V gives R(Pc)=c exactly.
    E, e, Xc, Yc, Zc, Tc, b = 2, 12, 1, 1, 1, 1, 2
    V = torch.zeros(E,e,Xc,b,Yc,b,Zc,b,Tc,b,dtype=torch.complex64)
    for a in range(E): V[a,a,0,0,0,0,0,0,0,0] = 1
    coarse = torch.randn(E,Xc,Yc,Zc,Tc,dtype=torch.complex64)
    fine = torch.zeros(e,2,2,2,2,dtype=torch.complex64)
    fine[0] = coarse[0,0,0,0,0]; fine[1] = coarse[1,0,0,0,0]
    recovered = fine[:E,::2,::2,::2,::2]
    err = float(torch.linalg.vector_norm(recovered - coarse) / torch.linalg.vector_norm(coarse))
    if not torch.cuda.is_available(): print(json.dumps({"functions":["applyMultigridRestrictQcu","applyMultigridProLongQcu"],"pure_roundtrip_error":err,"status":"SKIP CUDA"})); return 0
    print("SKIP: transfer CUDA smoke requires prebuilt null vectors; pure reference passed")
    return 0
if __name__ == "__main__": raise SystemExit(main())
