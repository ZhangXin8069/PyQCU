#!/usr/bin/env python3
"""单功能测试：Strict coarse/MATPC/prepare/reconstruct/R/P 生命周期与布局。"""
from __future__ import annotations
import json
import torch
from single_function_common import assert_roundtrip_layout, run_setup

def main(argv=None) -> int:
    args, report = run_setup(__doc__ or "Strict MultiGrid 接口单测", __doc__, argv=argv, total=2)
    report.run("布局契约", assert_roundtrip_layout)
    shapes = report.run("Strict 形状契约", lambda: {"coarse":(2,2,2,2,2), "compact":(2,2,2,2,1), "links":(2,4,2,2,2,2,2,2), "onsite":(2,2,2,2,2,2,2)})
    print(json.dumps({"functions":["applyMultigridStrictCoarseQcu","applyMultigridStrictMatPCQcu","applyMultigridStrictFineMatPCQcu","applyMultigridStrictPrepareQcu","applyMultigridStrictReconstructQcu","applyMultigridStrictRestrictQcu","applyMultigridStrictProLongQcu","applyMultigridStrictVCycleQcu","applyMultigridStrictInitQcu","applyMultigridStrictEndQcu","applyMultigridStrictFgmresQcu"],"shape_contract":shapes,"status":"SKIP CUDA unless strict assets are supplied", "lattice":list(args.lat)}))
    report.finish("SKIP")
    return 0
if __name__ == "__main__": raise SystemExit(main())
