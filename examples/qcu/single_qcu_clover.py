#!/usr/bin/env python3
"""单功能测试：applyCloverQcu/applyCloversQcu 的构造与逆一致性。"""
from __future__ import annotations
import json
import torch
from single_function_common import *

def main(argv=None) -> int:
    args, report = run_setup(__doc__ or "Clover 构造单测", __doc__, argv=argv, total=7)
    report.run("布局契约", assert_roundtrip_layout)
    report.run("PyTorch Clover 参考", cpu_reference_smoke, "clover", args.mass)
    if args.pure_only:
        report.finish("PURE_ONLY")
        return 0
    if not torch.cuda.is_available():
        report.skip("CUDA 不可用；已完成 Clover 纯 PyTorch 参考")
        report.finish("SKIP")
        return 0
    ctx = report.run("创建上下文", make_context, lattice=args.lat, mass=args.mass,
                     plan=2, device=args.device, reporter=report)
    gauge, _, _, ce, co, cei, coi = report.run("分配测试场", allocate_state, ctx, clover=True)
    # Single-parity Clover construction and the paired convenience wrapper.
    lifecycle_call(ctx, "applyCloverQcu", ce, gauge)
    lifecycle_call(ctx, "applyCloversQcu", ce, cei, gauge)
    # The operation is repeated for odd parity with a fresh slot.
    ctx.params[define._PARITY_] = 1
    lifecycle_call(ctx, "applyCloversQcu", co, coi, gauge)
    ident = torch.eye(12, dtype=ce.dtype, device=ce.device)
    mats = ce.permute(4,5,6,7,0,1,2,3).reshape(-1,12,12)
    invs = cei.permute(4,5,6,7,0,1,2,3).reshape(-1,12,12)
    err = float(torch.linalg.vector_norm(mats @ invs - ident) / torch.linalg.vector_norm(ident))
    print(json.dumps({"function":"applyCloversQcu", "inverse_error":err,
                      "lattice":list(ctx.lattice)}))
    report.finish("PASS" if err < 2e-2 else "FAIL")
    return 0 if err < 2e-2 else 1
if __name__ == "__main__": raise SystemExit(main())
