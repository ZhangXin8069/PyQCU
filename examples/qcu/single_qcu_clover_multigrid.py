#!/usr/bin/env python3
"""单功能测试：applyCloverMultigridQcu/verifyCloverMultigridQcu。"""
from __future__ import annotations
import json
import torch
from single_function_common import *

def main(argv=None) -> int:
    args, report = run_setup(__doc__ or "Clover MultiGrid 单测", __doc__, argv=argv, total=2)
    report.run("布局契约", assert_roundtrip_layout)
    report.run("PyTorch Clover 参考", cpu_reference_smoke, "clover", args.mass)
    if args.pure_only or not torch.cuda.is_available():
        report.skip("仅完成 Clover 纯 PyTorch 参考" if args.pure_only else "CUDA 不可用；已完成 Clover 纯 PyTorch 参考")
        report.finish("PURE_ONLY" if args.pure_only else "SKIP")
        return 0
    print(json.dumps({"functions":["applyCloverMultigridQcu","verifyCloverMultigridQcu"],"status":"SKIP: requires generated null vectors; use conftest.clover.multigrid.py", "lattice":list(args.lat)}))
    report.finish("SKIP")
    return 0
if __name__ == "__main__": raise SystemExit(main())
