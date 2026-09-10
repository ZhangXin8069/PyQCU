#!/usr/bin/env python3
"""单功能测试：applyGaussGaugeQcu（plan=-1）。"""
from __future__ import annotations
import json
import torch
from single_function_common import assert_roundtrip_layout, make_context, allocate_state, lifecycle_call, cpu_reference_smoke


def main() -> int:
    assert_roundtrip_layout(); cpu_reference_smoke("roundtrip")
    if not torch.cuda.is_available():
        print("SKIP: CUDA 不可用；已完成布局契约检查"); return 0
    ctx = make_context(plan=-1)
    gauge, _, _ = allocate_state(ctx)
    lifecycle_call(ctx, "applyGaussGaugeQcu", gauge)
    u = gauge.permute(0, 3, 4, 5, 6, 7, 1, 2).reshape(-1, 3, 3)
    eye = torch.eye(3, dtype=gauge.dtype, device=gauge.device)
    err = float(torch.linalg.vector_norm(u @ u.conj().transpose(-1, -2) - eye) / torch.linalg.vector_norm(eye * u.shape[0] ** 0.5))
    print(json.dumps({"function": "applyGaussGaugeQcu", "unitarity_error": err}))
    return 0 if err < 5e-3 else 1

if __name__ == "__main__": raise SystemExit(main())
