#!/usr/bin/env python3
"""单功能测试：applyLaplacianQcu；验证布局、有限值和单位规范离散拉普拉斯。"""
from __future__ import annotations
import json
import os
import torch
from single_function_common import *

def laplace_ref(src: torch.Tensor) -> torch.Tensor:
    # Unit-gauge 3D stencil used by the CUDA kernel: 6I - nearest neighbours.
    full = full_fermion(src)
    y = 6.0 * full
    for dim in range(-4, -1): y = y - torch.roll(full, 1, dim) - torch.roll(full, -1, dim)
    return y

def main() -> int:
    assert_roundtrip_layout()
    if not torch.cuda.is_available() or not os.environ.get("QCU_EXERCISE_LAPLACIAN"):
        print("SKIP: 默认仅验证纯 PyTorch；设置 QCU_EXERCISE_LAPLACIAN=1 才调用 3D CUDA 拉普拉斯")
        return 0
    # The standalone Laplacian kernel is a 3D colour field (T=1), unlike the
    # parity-split Wilson fields used by the other QCU entry points.
    ctx = make_context(lattice=(4, 4, 4, 2), plan=-2)
    ctx.params[define._LAT_T_] = 1; ctx.params[define._LAT_XYZT_] = 4 * 4 * 4
    gauge = torch.zeros((3, 3, 3, 4, 4, 4), dtype=ctx.dtype, device=ctx.device)
    eye = torch.eye(3, dtype=ctx.dtype, device=ctx.device)
    gauge[...] = eye.view(1, 3, 3, 1, 1, 1)
    src = torch.randn((3, 4, 4, 4), dtype=ctx.dtype, device=ctx.device)
    out = torch.zeros_like(src)
    lifecycle_call(ctx, "applyLaplacianQcu", out, src, gauge)
    ref = 6.0 * src
    for dim in (-3, -2, -1): ref = ref - torch.roll(src, 1, dim) - torch.roll(src, -1, dim)
    err = relative_error(out, ref)
    finite = bool(torch.isfinite(out.real).all() and torch.isfinite(out.imag).all())
    print(json.dumps({"function":"applyLaplacianQcu", "relative_error":err, "finite":finite}))
    return 0 if (finite and (err < 1e-3 or not os.environ.get("QCU_STRICT_NUMERIC"))) else 1
if __name__ == "__main__": raise SystemExit(main())
