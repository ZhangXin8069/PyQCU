"""KUPL whole-solve Krylov drivers."""

from __future__ import annotations

from typing import Any

from pyqcu.api import BiCGStabControl, CGControl, WholeSolveKrylovDriver
from pyqcu.kupl.linear_operator import KuplLinearOperator


class KuplBiCGStabDriver(WholeSolveKrylovDriver):
    def __init__(self, operator: Any, control: BiCGStabControl):
        if not isinstance(operator, KuplLinearOperator):
            raise TypeError("KUPL Krylov drivers require KuplLinearOperator")
        super().__init__(
            backend="kupl", method="bicgstab", operator=operator,
            control=control)


class KuplCGDriver(WholeSolveKrylovDriver):
    def __init__(self, operator: Any, control: CGControl):
        if not isinstance(operator, KuplLinearOperator):
            raise TypeError("KUPL Krylov drivers require KuplLinearOperator")
        if not bool(getattr(operator, "capabilities", {}).get("cg", False)):
            raise ValueError("KUPL operator must declare cg=True")
        super().__init__(
            backend="kupl", method="cg", operator=operator, control=control)


__all__ = ["KuplBiCGStabDriver", "KuplCGDriver"]
