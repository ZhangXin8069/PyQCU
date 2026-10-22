"""CUDA whole-solve Krylov drivers."""

from __future__ import annotations

from typing import Any

from pyqcu.api import BiCGStabControl, CGControl, WholeSolveKrylovDriver
from pyqcu.cuda.linear_operator import CudaLinearOperator


class CudaBiCGStabDriver(WholeSolveKrylovDriver):
    def __init__(self, operator: Any, control: BiCGStabControl):
        if not isinstance(operator, CudaLinearOperator):
            raise TypeError("CUDA Krylov drivers require CudaLinearOperator")
        super().__init__(
            backend="cuda", method="bicgstab", operator=operator,
            control=control)


class CudaCGDriver(WholeSolveKrylovDriver):
    def __init__(self, operator: Any, control: CGControl):
        if not isinstance(operator, CudaLinearOperator):
            raise TypeError("CUDA Krylov drivers require CudaLinearOperator")
        if not bool(getattr(operator, "capabilities", {}).get("cg", False)):
            raise ValueError("CUDA operator must declare cg=True")
        super().__init__(
            backend="cuda", method="cg", operator=operator, control=control)


__all__ = ["CudaBiCGStabDriver", "CudaCGDriver"]
