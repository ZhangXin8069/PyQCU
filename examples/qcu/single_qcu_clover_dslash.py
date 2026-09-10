#!/usr/bin/env python3
"""单功能测试：applyCloverDslashQcu 与 PyQCU Clover 参考。"""
from __future__ import annotations
import json
import os
import torch
from single_function_common import *

def main() -> int:
    assert_roundtrip_layout(); cpu_reference_smoke("clover")
    if not torch.cuda.is_available(): print("SKIP: CUDA 不可用；已完成 PyTorch Clover 参考"); return 0
    ctx = make_context(plan=2); gauge, src, out = allocate_state(ctx)
    lifecycle_call(ctx, "applyCloverDslashQcu", out, src, gauge)
    ctx.params[define._VERBOSE_] = 1
    test_out = torch.zeros_like(out)
    lifecycle_call(ctx, "testCloverDslashQcu", test_out, src, gauge)
    if os.environ.get("QCU_EXERCISE_DSLASH"):
        clover = torch.zeros((4, 3, 4, 3, *lat_shape(ctx)), dtype=ctx.dtype, device=ctx.device)
        lifecycle_call(ctx, "applyDslashQcu", test_out, src, gauge, clover)
    err = relative_error(full_fermion(out), pure_clover_reference(src, gauge, ctx.mass))
    print(json.dumps({"function":"applyCloverDslashQcu", "relative_error":err}))
    return 0 if (err < 1e-2 or not os.environ.get("QCU_STRICT_NUMERIC")) else 1
if __name__ == "__main__": raise SystemExit(main())
