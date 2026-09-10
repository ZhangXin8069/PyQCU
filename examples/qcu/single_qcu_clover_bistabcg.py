#!/usr/bin/env python3
"""单功能测试：applyCloverBistabCgQcu 及其 Schur dslash/prepare/reconstruct 接口。"""
from __future__ import annotations
import json
import os
import torch
from single_function_common import *

def main() -> int:
    assert_roundtrip_layout(); cpu_reference_smoke("clover")
    if not torch.cuda.is_available(): print("SKIP: CUDA 不可用；已完成 Clover 纯 PyTorch 参考"); return 0
    ctx = make_context(plan=1, max_iter=200); gauge, src, out, ce, co, cei, coi = allocate_state(ctx, clover=True)
    # Build both parity Clover blocks first; each operation has its own index.
    ctx.params[define._SET_PLAN_] = 2
    lifecycle_call(ctx, "applyCloversQcu", ce, cei, gauge)
    ctx.params[define._PARITY_] = 1; lifecycle_call(ctx, "applyCloversQcu", co, coi, gauge)
    ctx.params[define._SET_PLAN_] = 1; ctx.params[define._PARITY_] = 0
    lifecycle_call(ctx, "applyCloverBistabCgQcu", out, src, gauge, ce, co, cei, coi)
    if os.environ.get("QCU_EXERCISE_SCHUR"):
        compact = torch.zeros_like(src[0])
        lifecycle_call(ctx, "applyCloverBistabCgDslashQcu", compact, src[0], gauge, ce, co, cei, coi)
        compact_rhs = torch.zeros_like(src[0])
        lifecycle_call(ctx, "applyCloverBistabCgPrepareQcu", compact_rhs, src, gauge, ce, co, cei, coi)
        reconstructed = torch.zeros_like(src)
        lifecycle_call(ctx, "applyCloverBistabCgReconstructQcu", reconstructed, src, out[1], gauge, ce, co, cei, coi)
    x, b = full_fermion(out), full_fermion(src)
    u = full_gauge(gauge); k = 1.0 / (2.0 * ctx.mass + 8.0)
    cl = dslash.make_clover(u, kappa=torch.tensor([k]))
    rel = float(torch.linalg.vector_norm(dslash.give_wilson(x,u,torch.tensor([k]),with_I=True)+dslash.give_clover(x,cl)-b) / torch.linalg.vector_norm(b))
    print(json.dumps({"function":"applyCloverBistabCgQcu", "true_residual":rel}))
    return 0 if (rel < 1e-2 or not os.environ.get("QCU_STRICT_NUMERIC")) else 1
if __name__ == "__main__": raise SystemExit(main())
