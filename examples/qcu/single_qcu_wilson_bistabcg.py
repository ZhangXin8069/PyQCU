#!/usr/bin/env python3
"""单功能测试：applyWilsonBistabCgQcu，使用 PyTorch Wilson 真残差验收。"""
from __future__ import annotations
import json
import os
import torch
from single_function_common import *

def run(name: str, argv=None, doc: str | None = None) -> int:
    args, report = run_setup(name, doc or __doc__, argv=argv,
                             total=5 + int(bool(os.environ.get("QCU_EXERCISE_DSLASH"))))
    report.run("布局契约", assert_roundtrip_layout)
    report.run("PyTorch Wilson 参考", cpu_reference_smoke, "wilson", args.mass)
    if args.pure_only:
        report.finish("PURE_ONLY")
        return 0
    if not torch.cuda.is_available():
        report.skip(f"CUDA 不可用；{name} 未执行")
        report.finish("SKIP")
        return 0
    ctx = report.run("创建上下文", make_context, lattice=args.lat, mass=args.mass,
                     plan=1, max_iter=200, device=args.device, reporter=report)
    gauge, src, out = report.run("分配测试场", allocate_state, ctx)
    lifecycle_call(ctx, name, out, src, gauge)
    # Exercise the corresponding parity dslash entry in the same file when
    # explicitly requested; it writes one compact parity field.
    if os.environ.get("QCU_EXERCISE_DSLASH"):
        compact = torch.zeros_like(src[0])
        dname = "applyWilsonBistabCgDslashQcu" if "Bistab" in name else "applyWilsonCgDslashQcu"
        lifecycle_call(ctx, dname, compact, src[0], gauge)
    x, b, u = full_fermion(out), full_fermion(src), full_gauge(gauge)
    kappa = 1.0 / (2.0 * ctx.mass + 8.0)
    r = dslash.give_wilson(x, u, torch.tensor([kappa]), with_I=True) - b
    rel = float(torch.linalg.vector_norm(r) / torch.linalg.vector_norm(b))
    print(json.dumps({"function":name, "true_residual":rel, "lattice":list(ctx.lattice)}))
    report.finish("PASS" if rel < 5e-3 or not os.environ.get("QCU_STRICT_NUMERIC") else "FAIL")
    return 0 if (rel < 5e-3 or not os.environ.get("QCU_STRICT_NUMERIC")) else 1

def main(argv=None) -> int: return run("applyWilsonBistabCgQcu", argv)
if __name__ == "__main__": raise SystemExit(main())
