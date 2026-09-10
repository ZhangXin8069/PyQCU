#!/usr/bin/env python3
"""QCU Cython API 单功能冒烟检查。

默认只检查桥接模块是否导出 ``pyqcu.h`` 中声明的接口，因此可在无 CUDA
设备的 CI 中运行；设置 ``QCU_API_REQUIRE_CUDA=1`` 时再执行最小 init/end
生命周期检查。
"""
from __future__ import annotations

import os
from pathlib import Path
import re

HEADER = Path(__file__).resolve().parents[2] / "cpp/cuda/qcu/python/pyqcu.h"


def declared_api() -> list[str]:
    text = HEADER.read_text(encoding="utf-8")
    return sorted(set(re.findall(r"\b(?:apply|test)[A-Za-z0-9]+Qcu\b", text)))


def test_all_header_symbols_exported() -> None:
    from pyqcu.cuda import qcu
    missing = [name for name in declared_api() if not hasattr(qcu, name)]
    assert not missing, f"Cython bridge missing symbols: {missing}"


def test_init_end_lifecycle_when_requested() -> None:
    if os.environ.get("QCU_API_REQUIRE_CUDA") != "1":
        return
    import torch
    from pyqcu.cuda import qcu
    import pyqcu.cuda.define as define
    params = define.params.clone(); params[define._SET_INDEX_] = 0
    params[define._SET_PLAN_] = 0
    set_ptrs = define.set_ptrs.clone(); argv = define.argv.clone()
    qcu.applyInitQcu(set_ptrs, params, argv)
    qcu.applyEndQcu(set_ptrs, params)
    assert torch.isfinite(params.to(torch.float32)).all()


if __name__ == "__main__":
    print("QCU API symbols:", len(declared_api()))
    for name in declared_api():
        print(name)
