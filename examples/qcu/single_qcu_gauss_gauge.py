#!/usr/bin/env python3
"""单功能测试：applyGaussGaugeQcu（plan=-1）。"""
from __future__ import annotations
import json
import torch
from single_function_common import (assert_roundtrip_layout, make_context, allocate_state,
                                    lifecycle_call, cpu_reference_smoke, run_setup)


def main(argv=None) -> int:
    args, report = run_setup(__doc__ or "Gauss gauge 单测", __doc__, argv=argv, total=5)
    report.run("布局契约", assert_roundtrip_layout)
    report.run("纯布局参考", cpu_reference_smoke, "roundtrip")
    if args.pure_only:
        report.finish("PURE_ONLY")
        return 0
    if not torch.cuda.is_available():
        report.skip("CUDA 不可用；已完成布局契约检查")
        report.finish("SKIP")
        return 0
    ctx = report.run("创建上下文", make_context, lattice=args.lat, mass=args.mass,
                     plan=-1, device=args.device, reporter=report)
    gauge, _, _ = report.run("分配规范场", allocate_state, ctx)
    lifecycle_call(ctx, "applyGaussGaugeQcu", gauge)
    u = gauge.permute(0, 3, 4, 5, 6, 7, 1, 2).reshape(-1, 3, 3)
    eye = torch.eye(3, dtype=gauge.dtype, device=gauge.device)
    err = float(torch.linalg.vector_norm(u @ u.conj().transpose(-1, -2) - eye) / torch.linalg.vector_norm(eye * u.shape[0] ** 0.5))
    print(json.dumps({"function": "applyGaussGaugeQcu", "unitarity_error": err,
                      "lattice": list(ctx.lattice)}))
    report.finish("PASS" if err < 5e-3 else "FAIL")
    return 0 if err < 5e-3 else 1

if __name__ == "__main__": raise SystemExit(main())
