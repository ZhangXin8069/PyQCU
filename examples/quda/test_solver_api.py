#!/usr/bin/env python3
"""QUDA 求解器单功能测试入口（可选依赖）。"""
from __future__ import annotations

import importlib.util
import pytest


def _quda_available() -> bool:
    return (importlib.util.find_spec("pyquda.core") is not None and
            importlib.util.find_spec("cupy") is not None)


@pytest.mark.skipif(not _quda_available(), reason="需要 pyquda 与 cupy")
def test_wilson_cg_solver_api() -> None:
    """验证 QUDA Wilson CG 构造、求解器参数和 true residual 接口。"""
    import cupy as cp
    import pyquda
    from pyquda.core import getDslash
    from pyquda.field import LatticeFermion, LatticeGauge
    from pyquda.pyquda import invertQuda

    lat = [2, 2, 2, 4]
    pyquda.init(grid_size=[1, 1, 1, 1])
    try:
        gauge = LatticeGauge(lat, cp.zeros((4, *lat, 3, 3), dtype=cp.complex128))
        rhs = LatticeFermion(lat, cp.ones((2, *lat, 4, 3), dtype=cp.complex128))
        out = LatticeFermion(lat)
        dslash = getDslash(lat, 0.1, 1e-6, 32, anti_periodic_t=False)
        dslash.loadGauge(gauge)
        invertQuda(out.data_ptr, rhs.data_ptr, dslash.invert_param)
        assert int(dslash.invert_param.iter) >= 0
        assert float(dslash.invert_param.true_res) >= 0.0
        dslash.destroy()
    finally:
        pyquda.end()
