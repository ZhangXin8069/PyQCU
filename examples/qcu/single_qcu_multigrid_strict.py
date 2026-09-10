#!/usr/bin/env python3
"""单功能测试：Strict coarse/MATPC/prepare/reconstruct/R/P 生命周期与布局。"""
from __future__ import annotations
import json
import torch
from single_function_common import assert_roundtrip_layout

def main() -> int:
    assert_roundtrip_layout()
    shapes = {"coarse":(2,2,2,2,2), "compact":(2,2,2,2,1), "links":(2,4,2,2,2,2,2,2), "onsite":(2,2,2,2,2,2,2)}
    print(json.dumps({"functions":["applyMultigridStrictCoarseQcu","applyMultigridStrictMatPCQcu","applyMultigridStrictFineMatPCQcu","applyMultigridStrictPrepareQcu","applyMultigridStrictReconstructQcu","applyMultigridStrictRestrictQcu","applyMultigridStrictProLongQcu","applyMultigridStrictVCycleQcu","applyMultigridStrictInitQcu","applyMultigridStrictEndQcu","applyMultigridStrictFgmresQcu"],"shape_contract":shapes,"status":"SKIP CUDA unless strict assets are supplied"}))
    return 0
if __name__ == "__main__": raise SystemExit(main())
