#!/usr/bin/env python3
"""单功能测试：applyWilsonDslashQcu 与 PyQCU Wilson 参考。"""
from __future__ import annotations
import json
import os
import torch
from single_function_common import *

def main() -> int:
    assert_roundtrip_layout(); cpu_reference_smoke("wilson")
    if not torch.cuda.is_available(): print("SKIP: CUDA 不可用；已完成 PyTorch Wilson 参考"); return 0
    ctx = make_context(plan=0); gauge, src, out = allocate_state(ctx)
    lifecycle_call(ctx, "applyWilsonDslashQcu", out, src, gauge)
    # The test variant uses the same kernel with timing diagnostics enabled.
    ctx.params[define._VERBOSE_] = 1
    test_out = torch.zeros_like(out)
    lifecycle_call(ctx, "testWilsonDslashQcu", test_out, src, gauge)
    actual = full_fermion(out); expected = pure_wilson_reference(src, gauge, ctx.mass, with_I=False)
    err = relative_error(actual, expected)
    print(json.dumps({"function":"applyWilsonDslashQcu", "relative_error":err, "layout":list(out.shape)}))
    return 0 if (err < 5e-3 or not os.environ.get("QCU_STRICT_NUMERIC")) else 1
if __name__ == "__main__": raise SystemExit(main())
