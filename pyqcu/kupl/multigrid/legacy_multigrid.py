"""KUPL legacy multigrid lifecycle wrapper."""

from __future__ import annotations

from typing import Any, Optional

import pyqcu.cann as _torch

from pyqcu.kupl import qcu
from pyqcu.kupl import define
from pyqcu._multigrid_warnings import warn_excessive_multigrid_levels


class LegacyMultigrid:
    """Thin CPU lifecycle wrapper around the KUPL legacy Clover-MG ABI.

    Coarse stencils are expected to be present in ``set_ptrs`` according to
    the canonical ``params/argv/set_ptrs`` protocol.  This class standardizes
    setup/solve/close without duplicating the backend's hierarchy builder.
    """

    variant = "legacy"
    backend = "kupl"

    def __init__(self, params: Any, argv: Any, set_ptrs: Any, gauge: Any,
                 clover_even: Any, clover_odd: Any,
                 clover_even_inv: Any, clover_odd_inv: Any,
                 output: Optional[Any] = None):
        self.params = _cpu_tensor(params, "params")
        self.argv = _cpu_tensor(argv, "argv")
        self.set_ptrs = _cpu_tensor(set_ptrs, "set_ptrs")
        self.gauge = _cpu_tensor(gauge, "gauge")
        self.clover_even = _cpu_tensor(clover_even, "clover_even")
        self.clover_odd = _cpu_tensor(clover_odd, "clover_odd")
        self.clover_even_inv = _cpu_tensor(clover_even_inv, "clover_even_inv")
        self.clover_odd_inv = _cpu_tensor(clover_odd_inv, "clover_odd_inv")
        self.output = output
        levels = int(params[define._MG_NUM_LEVEL_])
        lattice = tuple(int(params[index]) for index in range(
            define._LAT_X_, define._LAT_T_ + 1))
        process_grid = tuple(int(params[index]) for index in range(
            define._GRID_X_, define._GRID_T_ + 1))
        level_one = define._MG_LEVEL1_E_
        block = (2, 2, 2, 2)
        if levels > 1 and int(params[level_one + 1]) > 0:
            block = (
                lattice[0] // int(params[level_one + 1]),
                lattice[1] // int(params[level_one + 2]),
                lattice[2] // int(params[level_one + 3]),
                (lattice[3] // 2) // int(params[level_one + 4]),
            )
        self.multigrid_depth_warning = warn_excessive_multigrid_levels(
            levels, variant="legacy", backend="kupl",
            context="KUPL LegacyMultigrid", lattice=lattice,
            block=block, process_grid=process_grid)
        self._setup_done = False
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def setup(self) -> "LegacyMultigrid":
        if self._closed:
            raise RuntimeError("KUPL LegacyMultigrid 已关闭")
        if not self._setup_done:
            qcu.applyInitQcu(self.set_ptrs, self.params, self.argv)
            self._setup_done = True
        return self

    def solve(self, rhs: Any, x0: Any = None) -> Any:
        if not self._setup_done:
            raise RuntimeError("调用 solve 前必须先 setup")
        source = _cpu_tensor(rhs, "rhs")
        if self.output is None or tuple(self.output.shape) != tuple(source.shape):
            self.output = source.new_empty(source.shape)
        elif self.output.dtype != source.dtype:
            raise TypeError("LegacyMultigrid output 与 rhs dtype 不一致")
        if x0 is None:
            self.params[57] = 0
            self.output.zero_()
        else:
            initial = _cpu_tensor(x0, "x0")
            if tuple(initial.shape) != tuple(source.shape):
                raise ValueError("LegacyMultigrid x0 与 rhs 形状不一致")
            self.params[57] = 1
            self.output.copy_(initial)
        qcu.applyCloverMultigridQcu(
            self.output, source, self.gauge, self.clover_even,
            self.clover_odd, self.clover_even_inv, self.clover_odd_inv,
            self.set_ptrs, self.params)
        return self.output

    def verify(self, rhs: Any) -> int:
        if not self._setup_done:
            raise RuntimeError("调用 verify 前必须先 setup")
        source = _cpu_tensor(rhs, "rhs")
        if self.output is None or tuple(self.output.shape) != tuple(source.shape):
            self.output = source.new_empty(source.shape)
        return qcu.verifyCloverMultigridQcu(
            self.output, source, self.gauge, self.clover_even,
            self.clover_odd, self.clover_even_inv, self.clover_odd_inv,
            self.set_ptrs, self.params)

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._setup_done:
                qcu.applyEndQcu(self.set_ptrs, self.params)
                self._setup_done = False
        finally:
            self._closed = True

    def __enter__(self) -> "LegacyMultigrid":
        return self.setup()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def _cpu_tensor(value: Any, name: str) -> Any:
    if not hasattr(value, "device") or value.device.type != "cpu":
        raise ValueError(f"KUPL {name} 必须是 CPU 张量")
    if not value.is_contiguous():
        raise ValueError(f"KUPL {name} 必须 contiguous")
    return value


__all__ = ["LegacyMultigrid"]
