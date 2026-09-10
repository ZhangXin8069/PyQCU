#!/usr/bin/env python3
"""单功能测试：applyCloverDslashQcu 与 PyQCU Clover 参考。"""
from __future__ import annotations
import json
import os
import torch
from single_function_common import *

def main(argv=None) -> int:
    args, report = run_setup(__doc__ or "Clover dslash 单测", __doc__, argv=argv,
                             total=6 + int(bool(os.environ.get("QCU_EXERCISE_DSLASH"))))
    report.run("布局契约", assert_roundtrip_layout)
    report.run("PyTorch Clover 参考", cpu_reference_smoke, "clover", args.mass)
    if args.pure_only:
        report.finish("PURE_ONLY")
        return 0
    if not torch.cuda.is_available():
        report.skip("CUDA 不可用；已完成 PyTorch Clover 参考")
        report.finish("SKIP")
        return 0
    ctx = report.run("创建上下文", make_context, lattice=args.lat, mass=args.mass,
                     plan=2, device=args.device, reporter=report)
    gauge, src, out = report.run("分配测试场", allocate_state, ctx)
    lifecycle_call(ctx, "applyCloverDslashQcu", out, src, gauge)
    ctx.params[define._VERBOSE_] = 1
    test_out = torch.zeros_like(out)
    lifecycle_call(ctx, "testCloverDslashQcu", test_out, src, gauge)
    if os.environ.get("QCU_EXERCISE_DSLASH"):
        clover = torch.zeros((4, 3, 4, 3, *lat_shape(ctx)), dtype=ctx.dtype, device=ctx.device)
        lifecycle_call(ctx, "applyDslashQcu", test_out, src, gauge, clover)
    err = relative_error(full_fermion(out), pure_clover_reference(src, gauge, ctx.mass))
    print(json.dumps({"function":"applyCloverDslashQcu", "relative_error":err,
                      "lattice":list(ctx.lattice)}))
    report.finish("PASS" if err < 1e-2 or not os.environ.get("QCU_STRICT_NUMERIC") else "FAIL")
    return 0 if (err < 1e-2 or not os.environ.get("QCU_STRICT_NUMERIC")) else 1
if __name__ == "__main__": raise SystemExit(main())
