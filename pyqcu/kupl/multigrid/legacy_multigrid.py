"""KUPL legacy multigrid lifecycle wrapper."""

from __future__ import annotations

from typing import Any, Optional

import pyqcu.cann as _torch

from pyqcu.kupl import define, qcu
from pyqcu._multigrid_warnings import warn_excessive_multigrid_levels


_SET_PTRS_LEGACY_LAYOUT = 99
_LEGACY_LAYOUTS = {
    "full-site-wilson-clover-33-stencil": 0,
    "odd-schur-33-stencil": 1,
}


class LegacyMultigrid:
    """Thin CPU lifecycle wrapper around the KUPL legacy Clover-MG ABI.

    The backend runs one outer solver, right-preconditioned FGMRES(m=10) with a
    hierarchy application as ``M^-1`` (see :meth:`set_nu_pre`).  ``params`` owns
    every knob: ``_MG_USE_DEFLATE_ == 0`` or one level means there is no coarse
    space at all and the call is forwarded to the parity-Schur BiCGStab, and
    ``_MG_LEVEL1_NUM_RESTART_ == 0`` disables the hierarchy the same way.

    Coarse stencils are expected to be present in ``set_ptrs`` according to
    the canonical ``params/argv/set_ptrs`` protocol.  This class standardizes
    setup/solve/close without duplicating the backend's hierarchy builder.
    """

    variant = "legacy"
    backend = "kupl"

    def set_nu_pre(self, steps: int) -> "LegacyMultigrid":
        """Set ``nu_pre``: the MR smoothing steps inside ``M^-1``.

        The KUPL Legacy-MG has ONE outer solver, right-preconditioned
        FGMRES(m=10) with ``M^-1 v`` = ``nu_pre`` steps of MR smoothing on ``v``,
        then one coarse cycle on the residual that remains, then the
        prolongation of that correction added with weight 1 (no post-smoothing).
        ``nu_pre`` is ``params[_MG_MU_PRE_]`` (=56), the same slot and the same
        default as CUDA: the backend uses 4 when the slot is 0 and clamps the
        value to 8.

        Zero therefore means "use the backend default (4)", not "no smoothing",
        and not - as the previous docstring of this method claimed - "the number
        of fine iterations before the first coarse correction".  That in-loop
        corrector was deleted with every knob that fed it, so no solve has a
        correction interval any more.
        """
        value = int(steps)
        if value < 0:
            raise ValueError(f"nu_pre must be >= 0, got {value}")
        self.params[define._MG_MU_PRE_] = value
        return self

    def __init__(self, params: Any, argv: Any, set_ptrs: Any, gauge: Any,
                 clover_even: Any, clover_odd: Any,
                 clover_even_inv: Any, clover_odd_inv: Any,
                 output: Optional[Any] = None,
                 cache_layout: str = "full-site-wilson-clover-33-stencil"):
        self.params = _cpu_tensor(
            params, "params", dtype=_torch.torch.int32, numel=define._PARAMS_SIZE_)
        self.set_ptrs = _cpu_tensor(
            set_ptrs, "set_ptrs", dtype=_torch.torch.int64,
            numel=define._SET_PTRS_SIZE_)
        if cache_layout not in _LEGACY_LAYOUTS:
            raise ValueError(
                f"unsupported KUPL Legacy cache layout: {cache_layout}")
        self.cache_layout = cache_layout
        self.set_ptrs[_SET_PTRS_LEGACY_LAYOUT] = _LEGACY_LAYOUTS[cache_layout]
        self.gauge = _cpu_tensor(gauge, "gauge")
        self.data_dtype = self.gauge.dtype
        expected_real_dtype = (
            _torch.torch.float32 if self.data_dtype == _torch.torch.complex64
            else _torch.torch.float64 if self.data_dtype == _torch.torch.complex128
            else None)
        if expected_real_dtype is None:
            raise TypeError("KUPL LegacyMultigrid gauge must be complex64/complex128")
        self.argv = _cpu_tensor(
            argv, "argv", dtype=expected_real_dtype, numel=define._ARGV_SIZE_)
        self.clover_even = _cpu_tensor(
            clover_even, "clover_even", dtype=self.data_dtype)
        self.clover_odd = _cpu_tensor(
            clover_odd, "clover_odd", dtype=self.data_dtype)
        self.clover_even_inv = _cpu_tensor(
            clover_even_inv, "clover_even_inv", dtype=self.data_dtype)
        self.clover_odd_inv = _cpu_tensor(
            clover_odd_inv, "clover_odd_inv", dtype=self.data_dtype)
        self.output = output
        # Caller contract, measured on Kunpeng (jobs 2236451 vs 2236485): the
        # fine operator streams the gauge slab plus four Clover terms, so the
        # same L48 c64 16-rank solve took 196.00 s unbound and 2.06 s with
        # KUPL_NUMA_BIND=1 KUPL_MEMORY_NODE=local+16 KUPL_HUGEPAGE=1 (95x).
        # This wrapper cannot bind: the environment must be set before setup().
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
            try:
                qcu.applyInitQcu(self.set_ptrs, self.params, self.argv)
            except BaseException:
                try:
                    qcu.applyEndQcu(self.set_ptrs, self.params)
                except BaseException:
                    pass
                raise
            self._setup_done = True
        return self

    def solve(self, rhs: Any, x0: Any = None) -> Any:
        if not self._setup_done:
            raise RuntimeError("调用 solve 前必须先 setup")
        source = _cpu_tensor(rhs, "rhs", dtype=self.data_dtype)
        if self.output is None:
            self.output = source.new_empty(source.shape)
        else:
            self.output = _cpu_tensor(
                self.output, "output", dtype=source.dtype)
            if tuple(self.output.shape) != tuple(source.shape):
                raise ValueError("KUPL LegacyMultigrid output 与 rhs 形状不一致")
        if x0 is None:
            self.params[define._MG_USE_INIT_GUESS_] = 0
            self.output.zero_()
        else:
            initial = _cpu_tensor(x0, "x0", dtype=source.dtype)
            if tuple(initial.shape) != tuple(source.shape):
                raise ValueError("LegacyMultigrid x0 与 rhs 形状不一致")
            self.params[define._MG_USE_INIT_GUESS_] = 1
            self.output.copy_(initial)
        qcu.applyCloverMultigridQcu(
            self.output, source, self.gauge, self.clover_even,
            self.clover_odd, self.clover_even_inv, self.clover_odd_inv,
            self.set_ptrs, self.params)
        if not bool(_torch.torch.isfinite(self.output).all().item()):
            raise FloatingPointError("KUPL LegacyMultigrid 返回非有限解")
        return self.output

    def verify(self, rhs: Any) -> int:
        if not self._setup_done:
            raise RuntimeError("调用 verify 前必须先 setup")
        source = _cpu_tensor(rhs, "rhs", dtype=self.data_dtype)
        if self.output is None:
            self.output = source.new_empty(source.shape)
        else:
            self.output = _cpu_tensor(
                self.output, "output", dtype=source.dtype)
            if tuple(self.output.shape) != tuple(source.shape):
                raise ValueError("KUPL LegacyMultigrid output 与 rhs 形状不一致")
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


def _cpu_tensor(
        value: Any, name: str, *, dtype: Any = None,
        numel: Optional[int] = None) -> Any:
    if not hasattr(value, "device") or value.device.type != "cpu":
        raise ValueError(f"KUPL {name} 必须是 CPU 张量")
    if not value.is_contiguous():
        raise ValueError(f"KUPL {name} 必须 contiguous")
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"KUPL {name} dtype 应为 {dtype}，得到 {value.dtype}")
    if numel is not None and int(value.numel()) != int(numel):
        raise ValueError(f"KUPL {name} 必须包含 {numel} 个元素")
    return value


__all__ = ["LegacyMultigrid"]
