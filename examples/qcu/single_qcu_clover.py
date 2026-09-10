#!/usr/bin/env python3
"""单功能测试：applyCloverQcu/applyCloversQcu 的构造与逆一致性。"""
from __future__ import annotations
import json
import torch
from single_function_common import *

def main() -> int:
    assert_roundtrip_layout(); cpu_reference_smoke("clover")
    if not torch.cuda.is_available(): print("SKIP: CUDA 不可用；已完成 Clover 纯 PyTorch 参考"); return 0
    ctx = make_context(plan=2); gauge, _, _, ce, co, cei, coi = allocate_state(ctx, clover=True)
    # Single-parity Clover construction and the paired convenience wrapper.
    lifecycle_call(ctx, "applyCloverQcu", ce, gauge)
    lifecycle_call(ctx, "applyCloversQcu", ce, cei, gauge)
    # The operation is repeated for odd parity with a fresh slot.
    ctx.params[define._PARITY_] = 1
    lifecycle_call(ctx, "applyCloversQcu", co, coi, gauge)
    ident = torch.eye(12, dtype=ce.dtype, device=ce.device)
    mats = ce.permute(4,5,6,7,0,1,2,3).reshape(-1,12,12)
    invs = cei.permute(4,5,6,7,0,1,2,3).reshape(-1,12,12)
    err = float(torch.linalg.vector_norm(mats @ invs - ident) / torch.linalg.vector_norm(ident))
    print(json.dumps({"function":"applyCloversQcu", "inverse_error":err}))
    return 0 if err < 2e-2 else 1
if __name__ == "__main__": raise SystemExit(main())
