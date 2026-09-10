#!/usr/bin/env python3
"""单功能测试：applyCloverMultigridQcu/verifyCloverMultigridQcu。"""
from __future__ import annotations
import json
import torch
from single_function_common import *

def main() -> int:
    assert_roundtrip_layout(); cpu_reference_smoke("clover")
    if not torch.cuda.is_available(): print("SKIP: CUDA 不可用；已完成 Clover 纯 PyTorch 参考"); return 0
    print(json.dumps({"functions":["applyCloverMultigridQcu","verifyCloverMultigridQcu"],"status":"SKIP: requires generated null vectors; use conftest.clover.multigrid.py"}))
    return 0
if __name__ == "__main__": raise SystemExit(main())
