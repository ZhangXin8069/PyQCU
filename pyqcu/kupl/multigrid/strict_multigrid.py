"""KUPL strict multigrid lifecycle wrapper."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import pyqcu.cann as _torch

from pyqcu.kupl import qcu
from pyqcu.kupl import define
from pyqcu._multigrid_warnings import warn_excessive_multigrid_levels


class StrictMultigrid:
    """Thin CPU lifecycle wrapper around the KUPL strict FGMRES ABI.

    Runtime hierarchy assets must already be bound in ``set_ptrs``.  The
    wrapper owns the LatticeSet and persistent strict workspace lifecycle and
    exposes the common setup/solve/close contract.
    """

    variant = "strict"
    backend = "kupl"
    residual_scope = "matpc"

    def __init__(
            self, params: Any, argv: Any, set_ptrs: Any, gauge: Any,
            clover_even: Any, clover_odd: Any, clover_even_inv: Any,
            clover_odd_inv: Any, fine_null_vectors: Any,
            fine_dims: Sequence[int], coarse_dims: Sequence[int],
            restart: int = 20, max_iter: int = 100,
            tolerance: float = 1e-8, nu_pre: int = 2, nu_post: int = 2,
            max_workspace_bytes: int = 512 << 20, start_level: int = 1,
            output: Optional[Any] = None):
        self.params = _cpu_tensor(params, "params")
        self.argv = _cpu_tensor(argv, "argv")
        self.set_ptrs = _cpu_tensor(set_ptrs, "set_ptrs")
        self.gauge = _cpu_tensor(gauge, "gauge")
        self.clover_even = _cpu_tensor(clover_even, "clover_even")
        self.clover_odd = _cpu_tensor(clover_odd, "clover_odd")
        self.clover_even_inv = _cpu_tensor(clover_even_inv, "clover_even_inv")
        self.clover_odd_inv = _cpu_tensor(clover_odd_inv, "clover_odd_inv")
        self.fine_null_vectors = _cpu_tensor(
            fine_null_vectors, "fine_null_vectors")
        self.fine_dims = _dims4(fine_dims, "fine_dims")
        self.coarse_dims = _dims4(coarse_dims, "coarse_dims")
        self.restart = int(restart)
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)
        self.nu_pre = int(nu_pre)
        self.nu_post = int(nu_post)
        self.max_workspace_bytes = int(max_workspace_bytes)
        self.start_level = int(start_level)
        levels = int(params[define._MG_NUM_LEVEL_])
        lattice = tuple(int(params[index]) for index in range(
            define._LAT_X_, define._LAT_T_ + 1))
        process_grid = tuple(int(params[index]) for index in range(
            define._GRID_X_, define._GRID_T_ + 1))
        block = tuple(
            max(1, (lattice[axis] // 2 if axis == 3 else lattice[axis]) //
                self.coarse_dims[axis])
            for axis in range(4))
        self.multigrid_depth_warning = warn_excessive_multigrid_levels(
            levels, variant="strict", backend="kupl",
            context="KUPL StrictMultigrid", lattice=lattice,
            block=block, process_grid=process_grid)
        if min(self.restart, self.max_iter, self.max_workspace_bytes) <= 0:
            raise ValueError("StrictMultigrid restart/max_iter/预算必须为正数")
        if self.nu_pre < 0 or self.nu_post < 0:
            raise ValueError("StrictMultigrid smoother steps 必须非负")
        if self.tolerance <= 0:
            raise ValueError("StrictMultigrid tolerance 必须为正数")
        self.output = output
        self.iterations = _torch.torch.zeros(1, dtype=_torch.torch.int32)
        self.converged = _torch.torch.zeros(1, dtype=_torch.torch.int32)
        self.final_residual = _torch.torch.zeros(1, dtype=_torch.float64)
        self.allocated_bytes = _torch.torch.zeros(1, dtype=_torch.torch.int64)
        self._setup_done = False
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def setup(self) -> "StrictMultigrid":
        if self._closed:
            raise RuntimeError("KUPL StrictMultigrid 已关闭")
        if self._setup_done:
            return self
        qcu.applyInitQcu(self.set_ptrs, self.params, self.argv)
        try:
            status = qcu.applyMultigridStrictInitQcu(
                self.set_ptrs, self.params, self.start_level,
                self.allocated_bytes)
        except Exception:
            qcu.applyEndQcu(self.set_ptrs, self.params)
            raise
        if status != 0:
            qcu.applyEndQcu(self.set_ptrs, self.params)
            raise RuntimeError(f"KUPL strict hierarchy init 失败: {status}")
        self._setup_done = True
        return self

    def solve(self, rhs: Any, x0: Any = None) -> Any:
        if not self._setup_done:
            raise RuntimeError("调用 solve 前必须先 setup")
        source = _cpu_tensor(rhs, "rhs")
        expected_elements = 12 * _volume(self.fine_dims)
        if int(source.numel()) != expected_elements:
            raise ValueError(
                f"StrictMultigrid rhs 元素数应为 {expected_elements}，"
                f"得到 {int(source.numel())}")
        shape = (*self.fine_dims[:3], self.fine_dims[3] // 2)
        output_shape = (2, 4, 3, *shape)
        if self.output is None or tuple(self.output.shape) != output_shape:
            self.output = _torch.torch.empty(
                output_shape, dtype=source.dtype, device=source.device)
        elif self.output.dtype != source.dtype:
            raise TypeError("StrictMultigrid output 与 rhs dtype 不一致")
        if x0 is None:
            self.params[57] = 0
            self.output.zero_()
        else:
            initial = _cpu_tensor(x0, "x0")
            if int(initial.numel()) != expected_elements:
                raise ValueError("StrictMultigrid x0 元素数与 rhs 不一致")
            self.params[57] = 1
            self.output.copy_(initial.reshape(output_shape))
        status = qcu.applyMultigridStrictFgmresQcu(
            self.output, source, self.gauge, self.clover_even,
            self.clover_odd, self.clover_even_inv, self.clover_odd_inv,
            self.fine_null_vectors, self.set_ptrs, self.params,
            self.fine_dims, self.coarse_dims, int(self.gauge.element_size()),
            self.restart, self.max_iter, self.tolerance, self.nu_pre,
            self.nu_post, self.max_workspace_bytes, self.iterations,
            self.converged, self.final_residual, self.allocated_bytes)
        if status != 0:
            raise RuntimeError(f"KUPL strict FGMRES 失败: {status}")
        return self.output

    def close(self) -> None:
        if self._closed:
            return
        if self._setup_done:
            try:
                qcu.applyMultigridStrictEndQcu(self.set_ptrs, self.params)
            finally:
                qcu.applyEndQcu(self.set_ptrs, self.params)
                self._setup_done = False
        self._closed = True

    def __enter__(self) -> "StrictMultigrid":
        return self.setup()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def _cpu_tensor(value: Any, name: str) -> Any:
    if not hasattr(value, "device") or value.device.type != "cpu":
        raise ValueError(f"KUPL {name} 必须是 CPU 张量")
    if not value.is_contiguous():
        raise ValueError(f"KUPL {name} 必须 contiguous")
    return value


def _dims4(value: Sequence[int], name: str) -> tuple[int, int, int, int]:
    dims = tuple(int(item) for item in value)
    if len(dims) != 4 or any(item <= 0 for item in dims):
        raise ValueError(f"{name} 必须是四个正整数")
    return dims  # type: ignore[return-value]


def _volume(dims: Sequence[int]) -> int:
    result = 1
    for extent in dims:
        result *= int(extent)
    return result


__all__ = ["StrictMultigrid"]
