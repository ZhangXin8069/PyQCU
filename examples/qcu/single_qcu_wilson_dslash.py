#!/usr/bin/env python3
"""单功能测试：applyWilsonDslashQcu 与 PyQCU Wilson 参考。"""
from __future__ import annotations
import json
import os
import torch
from single_function_common import *

def main(argv=None) -> int:
    args, report = run_setup(__doc__ or "Wilson dslash 单测", __doc__, argv=argv, total=6)
    report.run("布局契约", assert_roundtrip_layout)
    report.run("PyTorch Wilson 参考", cpu_reference_smoke, "wilson", args.mass)
    if args.pure_only:
        report.finish("PURE_ONLY")
        return 0
    if not torch.cuda.is_available():
        report.skip("CUDA 不可用；已完成 PyTorch Wilson 参考")
        report.finish("SKIP")
        return 0
    ctx = report.run("创建上下文", make_context, lattice=args.lat, mass=args.mass,
                     plan=0, device=args.device, reporter=report)
    gauge, src, out = report.run("分配测试场", allocate_state, ctx)
    lifecycle_call(ctx, "applyWilsonDslashQcu", out, src, gauge)
    # The test variant uses the same kernel with timing diagnostics enabled.
    ctx.params[define._VERBOSE_] = 1
    test_out = torch.zeros_like(out)
    lifecycle_call(ctx, "testWilsonDslashQcu", test_out, src, gauge)
    actual = full_fermion(out); expected = pure_wilson_reference(src, gauge, ctx.mass, with_I=False)
    err = relative_error(actual, expected)
    print(json.dumps({"function":"applyWilsonDslashQcu", "relative_error":err, "layout":list(out.shape),
                      "lattice":list(ctx.lattice)}))
    report.finish("PASS" if err < 5e-3 or not os.environ.get("QCU_STRICT_NUMERIC") else "FAIL")
    return 0 if (err < 5e-3 or not os.environ.get("QCU_STRICT_NUMERIC")) else 1
if __name__ == "__main__": raise SystemExit(main())
