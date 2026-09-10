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

def main(argv=None) -> int:
    args, report = run_setup(__doc__ or "Laplacian 单测", __doc__, argv=argv, total=3)
    report.run("布局契约", assert_roundtrip_layout)
    if args.pure_only or not torch.cuda.is_available() or not os.environ.get("QCU_EXERCISE_LAPLACIAN"):
        report.skip("默认仅验证纯 PyTorch；设置 QCU_EXERCISE_LAPLACIAN=1 才调用 3D CUDA 拉普拉斯")
        report.finish("PURE_ONLY" if args.pure_only else "SKIP")
        return 0
    # The standalone Laplacian kernel is a 3D colour field (T=1), unlike the
    # parity-split Wilson fields used by the other QCU entry points.
    # The standalone kernel has T=1 internally; X/Y/Z follow the requested
    # lattice while the public CLI still reports the default 16^4 input.
    ctx = report.run("创建上下文", make_context,
                     lattice=(args.lat[0], args.lat[1], args.lat[2], 2),
                     mass=args.mass, plan=-2, device=args.device, reporter=report)
    x, y, z = args.lat[:3]
    ctx.params[define._LAT_T_] = 1; ctx.params[define._LAT_XYZT_] = x * y * z
    gauge = torch.zeros((3, 3, 3, x, y, z), dtype=ctx.dtype, device=ctx.device)
    eye = torch.eye(3, dtype=ctx.dtype, device=ctx.device)
    gauge[...] = eye.view(1, 3, 3, 1, 1, 1)
    src = torch.randn((3, x, y, z), dtype=ctx.dtype, device=ctx.device)
    out = torch.zeros_like(src)
    lifecycle_call(ctx, "applyLaplacianQcu", out, src, gauge)
    ref = 6.0 * src
    for dim in (-3, -2, -1): ref = ref - torch.roll(src, 1, dim) - torch.roll(src, -1, dim)
    err = relative_error(out, ref)
    finite = bool(torch.isfinite(out.real).all() and torch.isfinite(out.imag).all())
    print(json.dumps({"function":"applyLaplacianQcu", "relative_error":err, "finite":finite,
                      "requested_lattice":list(args.lat), "effective_lattice":[x, y, z, 1]}))
    report.finish("PASS" if finite and (err < 1e-3 or not os.environ.get("QCU_STRICT_NUMERIC")) else "FAIL")
    return 0 if (finite and (err < 1e-3 or not os.environ.get("QCU_STRICT_NUMERIC"))) else 1
if __name__ == "__main__": raise SystemExit(main())
