#!/usr/bin/env python3
"""单功能测试：applyWilsonBistabCgQcu，使用 PyTorch Wilson 真残差验收。"""
from __future__ import annotations
import json
import os
import torch
from single_function_common import *

def run(name: str) -> int:
    assert_roundtrip_layout(); cpu_reference_smoke("wilson")
    if not torch.cuda.is_available(): print(f"SKIP: CUDA 不可用；{name} 未执行"); return 0
    ctx = make_context(plan=1, max_iter=200); gauge, src, out = allocate_state(ctx)
    lifecycle_call(ctx, name, out, src, gauge)
    # Exercise the corresponding parity dslash entry in the same file when
    # explicitly requested; it writes one compact parity field.
    if os.environ.get("QCU_EXERCISE_DSLASH"):
        compact = torch.zeros_like(src[0])
        dname = "applyWilsonBistabCgDslashQcu" if "Bistab" in name else "applyWilsonCgDslashQcu"
        lifecycle_call(ctx, dname, compact, src[0], gauge)
    x, b, u = full_fermion(out), full_fermion(src), full_gauge(gauge)
    kappa = 1.0 / (2.0 * ctx.mass + 8.0)
    r = dslash.give_wilson(x, u, torch.tensor([kappa]), with_I=True) - b
    rel = float(torch.linalg.vector_norm(r) / torch.linalg.vector_norm(b))
    print(json.dumps({"function":name, "true_residual":rel}))
    return 0 if (rel < 5e-3 or not os.environ.get("QCU_STRICT_NUMERIC")) else 1

def main() -> int: return run("applyWilsonBistabCgQcu")
if __name__ == "__main__": raise SystemExit(main())
