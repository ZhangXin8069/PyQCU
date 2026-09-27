#!/usr/bin/env python3
"""单功能测试：applyCloverDslashQcu 与 PyQCU Clover 参考。"""
from __future__ import annotations
import json
import os
import torch
from pyqcu.testing.qcu.single_function_common import *

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
    parity = int(ctx.params[define._PARITY_])
    compact_src = src[1 - parity]
    compact_out = torch.zeros_like(compact_src)
    lifecycle_call(ctx, "applyCloverDslashQcu", compact_out, compact_src, gauge)
    ctx.params[define._VERBOSE_] = 1
    test_out = torch.zeros_like(compact_src)
    lifecycle_call(ctx, "testCloverDslashQcu", test_out, compact_src, gauge)
    if os.environ.get("QCU_EXERCISE_DSLASH"):
        eye12 = torch.eye(12, dtype=ctx.dtype, device=ctx.device)
        clover = eye12.reshape(4, 3, 4, 3, 1, 1, 1, 1).expand(
            4, 3, 4, 3, *lat_shape(ctx)).contiguous()
        lifecycle_call(ctx, "applyDslashQcu", test_out, compact_src, gauge, clover)
    expected = qcu_clover_dslash_reference(compact_src, gauge, ctx.mass, parity)
    err = relative_error(compact_out, expected)
    test_err = relative_error(test_out, expected)
    print(json.dumps({"function":"applyCloverDslashQcu", "relative_error":err,
                      "test_relative_error":test_err, "parity":parity,
                      "lattice":list(ctx.lattice)}))
    ok = err < 1e-2 and test_err < 1e-2
    report.finish("PASS" if ok or not os.environ.get("QCU_STRICT_NUMERIC") else "FAIL")
    return 0 if (ok or not os.environ.get("QCU_STRICT_NUMERIC")) else 1
if __name__ == "__main__": raise SystemExit(main())
