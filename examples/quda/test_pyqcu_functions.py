#!/usr/bin/env python3
"""逐函数验证 ``pyqcu.h`` 的 QCU CUDA 实现。

默认覆盖所有导出符号的可调用性；设置 ``QCU_FUNCTION_LIFECYCLE=1`` 时，
额外执行真实的 ``applyInitQcu/applyEndQcu`` 生命周期，作为其它函数调用
测试的共同前置检查。
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

HEADER = Path(__file__).resolve().parents[2] / "cpp/cuda/qcu/python/pyqcu.h"


def header_functions() -> list[str]:
    return sorted(set(re.findall(r"\b(?:apply|test)[A-Za-z0-9]+Qcu\b", HEADER.read_text())))


@pytest.mark.parametrize("name", header_functions())
def test_pyqcu_h_function_is_exported(name: str) -> None:
    """每个 pyqcu.h 函数都必须由 Cython 桥导出为可调用对象。"""
    from pyqcu.cuda import qcu
    fn = getattr(qcu, name, None)
    assert callable(fn), f"{name} 未导出或不可调用"


def test_apply_init_end_lifecycle() -> None:
    if __import__("os").environ.get("QCU_FUNCTION_LIFECYCLE") != "1":
        pytest.skip("设置 QCU_FUNCTION_LIFECYCLE=1 才执行 CUDA 生命周期")
    from pyqcu.cuda import qcu
    from pyqcu.cuda.define import params, argv, set_ptrs
    import pyqcu.cuda.define as define
    params[define._SET_INDEX_] = 0
    qcu.applyInitQcu(set_ptrs, params, argv)
    qcu.applyEndQcu(set_ptrs, params)
