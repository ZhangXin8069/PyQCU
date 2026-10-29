"""Backend-neutral BiCGStab/CG selector."""

from __future__ import annotations

from typing import Any

from pyqcu.api import (
    BackendCapabilities,
    BiCGStabControl,
    CGControl,
    KrylovDriver,
    SolveResult,
    SolverControl,
)


class Krylov:
    """Select a Torch iterative or native whole-solve Krylov backend."""

    _BACKENDS = {"torch", "cuda", "kupl"}
    _METHODS = {"bicgstab": "bicgstab", "bistabcg": "bicgstab", "cg": "cg"}

    def __init__(
            self, *, backend: str, method: str, operator: Any,
            control: SolverControl):
        backend_name = str(backend).strip().lower()
        if backend_name not in self._BACKENDS:
            raise ValueError(
                "backend must be torch, cuda, or kupl, "
                f"got {backend!r}")
        method_name = str(method).strip().lower().replace("-", "_")
        if method_name not in self._METHODS:
            raise ValueError(
                f"method must be bicgstab, bistabcg, or cg, got {method!r}")
        method_name = self._METHODS[method_name]
        expected_control = (
            BiCGStabControl if method_name == "bicgstab" else CGControl)
        if not isinstance(control, expected_control):
            raise TypeError(
                f"method={method_name} requires {expected_control.__name__}")
        self.backend = backend_name
        self.method = method_name
        self.operator = operator
        self._implementation = _implementation(
            backend_name, method_name, operator, control)

    @property
    def implementation(self) -> Any:
        return self._implementation

    @property
    def control(self) -> SolverControl:
        return self._implementation.control

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._implementation.capabilities

    def solve_result(self, rhs: Any, x0: Any = None) -> SolveResult:
        return self._implementation.solve_result(rhs, x0=x0)

    def solve(self, rhs: Any, x0: Any = None) -> Any:
        return self.solve_result(rhs, x0=x0).x


class Solver(Krylov):
    """Canonical high-level name for the backend-neutral Krylov selector."""


def solve_bicgstab(
        *, backend: str, operator: Any, rhs: Any, control: SolverControl,
        x0: Any = None) -> SolveResult:
    """Run BiCGStab with the common API on torch, cuda, or kupl."""
    return Solver(
        backend=backend, method="bicgstab", operator=operator,
        control=control).solve_result(rhs, x0=x0)


def solve_cg(
        *, backend: str, operator: Any, rhs: Any, control: SolverControl,
        x0: Any = None) -> SolveResult:
    """Run CG with the common API on torch, cuda, or kupl."""
    return Solver(
        backend=backend, method="cg", operator=operator,
        control=control).solve_result(rhs, x0=x0)


def _implementation(
        backend: str, method: str, operator: Any,
        control: SolverControl) -> KrylovDriver:
    module_name = (
        "pyqcu.torch.solver" if backend == "torch"
        else f"pyqcu.{backend}.krylov")
    prefix = {"torch": "Torch", "cuda": "Cuda", "kupl": "Kupl"}[backend]
    suffix = "BiCGStabDriver" if method == "bicgstab" else "CGDriver"
    from importlib import import_module
    module = import_module(module_name)
    driver_type = getattr(module, f"{prefix}{suffix}")
    return driver_type(operator, control)


__all__ = ["Krylov", "Solver", "solve_bicgstab", "solve_cg"]
