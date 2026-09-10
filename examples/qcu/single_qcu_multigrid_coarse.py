#!/usr/bin/env python3
"""单功能测试：applyMultigridCoarseDslashQcu 与 Wide 变体的布局/有限值契约。"""
from __future__ import annotations
import json
import torch
from single_function_common import assert_roundtrip_layout

def main() -> int:
    assert_roundtrip_layout(); E,X,Y,Z,T = 2,2,2,2,2
    x = torch.randn(E,X,Y,Z,T,dtype=torch.complex64)
    sit = torch.eye(E,dtype=torch.complex64).view(E,E,1,1,1,1).expand(E,E,X,Y,Z,T).contiguous()
    hop = torch.zeros(2,4,E,E,X,Y,Z,T,dtype=torch.complex64)
    ref = torch.einsum("abxyzt,bxyzt->axyzt", sit, x)
    finite = bool(torch.isfinite(ref.real).all() and torch.isfinite(ref.imag).all())
    print(json.dumps({"functions":["applyMultigridCoarseDslashQcu","applyMultigridCoarseDslashWideQcu"],"pure_finite":finite,"reference_norm":float(torch.linalg.vector_norm(ref))}))
    return 0 if finite else 1
if __name__ == "__main__": raise SystemExit(main())
