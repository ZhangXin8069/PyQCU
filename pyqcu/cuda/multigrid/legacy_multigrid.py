"""CUDA legacy multigrid lifecycle wrapper."""

from __future__ import annotations

from typing import Any, Optional

import pyqcu.cann as _torch

from pyqcu.cuda import qcu


class LegacyMultigrid:
    """Lifecycle wrapper around the CUDA legacy Clover-MG C ABI.

    Coarse stencils are expected to be present in ``set_ptrs`` according to the
    canonical legacy layout. The wrapper owns ``applyInitQcu/applyEndQcu`` and
    deliberately does not reuse the pure-Python adaptive hierarchy.
    """

    variant = "legacy"
    backend = "cuda"

    def __init__(self, params: Any, argv: Any, set_ptrs: Any, gauge: Any,
                 clover_even: Any, clover_odd: Any,
                 clover_even_inv: Any, clover_odd_inv: Any,
                 output: Optional[Any] = None):
        tensors = {
            "params": params,
            "argv": argv,
            "set_ptrs": set_ptrs,
            "gauge": gauge,
            "clover_even": clover_even,
            "clover_odd": clover_odd,
            "clover_even_inv": clover_even_inv,
            "clover_odd_inv": clover_odd_inv,
        }
        for name, tensor in tensors.items():
            _validate_tensor(tensor, name)
        devices = {tensor.device for name, tensor in tensors.items()
                   if name not in {"params", "argv", "set_ptrs"}}
        if len(devices) != 1:
            raise ValueError("CUDA LegacyMultigrid 资产必须位于同一设备")
        for name, tensor in tensors.items():
            if name in {"params", "argv", "set_ptrs"}:
                if tensor.device.type != "cpu":
                    raise ValueError(f"CUDA {name} 必须是 CPU 张量")
            elif tensor.device.type != "cuda":
                raise ValueError(f"CUDA {name} 必须是 CUDA 张量")
        self.params = params
        self.argv = argv
        self.set_ptrs = set_ptrs
        self.gauge = gauge
        self.clover_even = clover_even
        self.clover_odd = clover_odd
        self.clover_even_inv = clover_even_inv
        self.clover_odd_inv = clover_odd_inv
        self.output = output
        self._setup_done = False
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def setup(self) -> "LegacyMultigrid":
        if self._closed:
            raise RuntimeError("CUDA LegacyMultigrid 已关闭")
        if not self._setup_done:
            qcu.applyInitQcu(self.set_ptrs, self.params, self.argv)
            self._setup_done = True
        return self

    def solve(self, rhs: Any, x0: Any = None) -> Any:
        if not self._setup_done:
            raise RuntimeError("调用 solve 前必须先 setup")
        source = _validate_operand(rhs, "rhs", self.gauge.device)
        if self.output is None or tuple(self.output.shape) != tuple(source.shape):
            self.output = _torch.torch.empty_like(source)
        elif self.output.dtype != source.dtype or self.output.device != source.device:
            raise TypeError("CUDA LegacyMultigrid output 与 rhs dtype/device 不一致")
        if x0 is None:
            self.params[57] = 0
            self.output.zero_()
        else:
            initial = _validate_operand(x0, "x0", source.device)
            if tuple(initial.shape) != tuple(source.shape):
                raise ValueError("CUDA LegacyMultigrid x0 与 rhs 形状不一致")
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
        source = _validate_operand(rhs, "rhs", self.gauge.device)
        if self.output is None or tuple(self.output.shape) != tuple(source.shape):
            self.output = _torch.torch.empty_like(source)
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


def _validate_tensor(value: Any, name: str) -> None:
    if not hasattr(value, "is_contiguous") or not value.is_contiguous():
        raise ValueError(f"CUDA {name} 必须是 contiguous 张量")
    if not hasattr(value, "device"):
        raise ValueError(f"CUDA {name} 必须是设备张量")


def _validate_operand(value: Any, name: str, device: Any) -> Any:
    _validate_tensor(value, name)
    if value.device != device:
        raise ValueError(f"CUDA {name} 必须位于 {device}")
    return value


__all__ = ["LegacyMultigrid"]
