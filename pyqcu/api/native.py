"""Whole-solve adapters for backends with native Krylov implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional

from .config import SolverControl
from .models import (
    BackendCapabilities,
    ResidualScope,
    SolveResult,
    SolveStatus,
    _optional_nonnegative_float,
    _optional_nonnegative_int,
    _token,
)


@dataclass(frozen=True)
class NativeSolve:
    """Evidence returned by one whole native Krylov solve call."""

    x: Any
    status: SolveStatus = SolveStatus.UNKNOWN
    converged: Optional[bool] = None
    iterations: Optional[int] = None
    final_residual: Optional[float] = None
    relative_true_residual: Optional[float] = None
    residual_scope: Optional[ResidualScope] = None
    workspace_bytes: Optional[int] = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = (self.status if isinstance(self.status, SolveStatus)
                  else SolveStatus(_token(self.status, "status")))
        converged = self.converged
        if converged is not None and not isinstance(converged, bool):
            raise TypeError("converged must be bool or None")
        if status is SolveStatus.UNKNOWN:
            if converged is True:
                status = SolveStatus.CONVERGED
            elif converged is False:
                status = SolveStatus.NOT_CONVERGED
        elif status is SolveStatus.CONVERGED:
            if converged is False:
                raise ValueError("status and converged disagree")
            converged = True
        elif status in {
                SolveStatus.MAX_ITERATIONS, SolveStatus.BREAKDOWN,
                SolveStatus.NOT_CONVERGED, SolveStatus.FAILED}:
            if converged is True:
                raise ValueError("status and converged disagree")
            converged = False
        scope = self.residual_scope
        if scope is not None and not isinstance(scope, ResidualScope):
            scope = ResidualScope(_token(scope, "residual_scope"))
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "converged", converged)
        object.__setattr__(self, "residual_scope", scope)
        object.__setattr__(
            self, "iterations", _optional_nonnegative_int(
                self.iterations, "iterations"))
        object.__setattr__(
            self, "workspace_bytes", _optional_nonnegative_int(
                self.workspace_bytes, "workspace_bytes"))
        for name in ("final_residual", "relative_true_residual"):
            object.__setattr__(
                self, name, _optional_nonnegative_float(
                    getattr(self, name), name))
        if (scope is None and any(getattr(self, name) is not None for name in (
                "final_residual", "relative_true_residual"))):
            raise ValueError("residual values require residual_scope")
        object.__setattr__(
            self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


NativeSolveCallable = Callable[
    [str, Any, Any, SolverControl], NativeSolve]


class WholeSolveKrylovDriver:
    """Run one native solve and expose the common Krylov result contract."""

    def __init__(
            self, *, backend: str, method: str, operator: Any,
            control: SolverControl, native_solve: Optional[NativeSolveCallable] = None):
        if not isinstance(control, SolverControl):
            raise TypeError("control must be a SolverControl")
        native = native_solve
        if native is None:
            native = getattr(operator, "solve_native", None)
        if not callable(native):
            raise TypeError("native backend must provide solve_native()")
        self.backend = _token(backend, "backend")
        self.method = _token(method, "method")
        self.operator = operator
        self._control = control
        self._native_solve = native
        operator_caps = dict(getattr(operator, "capabilities", {}))
        self.capabilities = BackendCapabilities(
            backend=self.backend,
            adjoint=bool(operator_caps.get("adjoint", False)),
            matpc=bool(operator_caps.get("matpc", False)),
            hpd_cg=bool(operator_caps.get("hpd_cg", False)),
            normal_cg=bool(operator_caps.get("normal_cg", False)),
            schur_cg=bool(operator_caps.get("schur_cg", False)),
            native_krylov=True,
            python_krylov=False,
        )

    @property
    def control(self) -> SolverControl:
        return self._control

    def solve_result(self, rhs: Any, x0: Any = None) -> SolveResult:
        start = perf_counter()
        outcome = self._native_solve(
            self.method, rhs, x0, self.control)
        if not isinstance(outcome, NativeSolve):
            raise TypeError("solve_native() must return NativeSolve")
        elapsed = perf_counter() - start
        scope = outcome.residual_scope or self.control.residual_scope
        return SolveResult(
            x=outcome.x,
            backend=self.backend,
            variant="krylov",
            method=self.method,
            status=outcome.status,
            converged=outcome.converged,
            krylov_iterations=outcome.iterations,
            final_residual=outcome.final_residual,
            residual_scope=scope,
            relative_true_residual=outcome.relative_true_residual,
            setup_seconds=None,
            solve_seconds=elapsed,
            workspace_bytes=outcome.workspace_bytes,
            diagnostics={
                "native_call_count": 1,
                **dict(outcome.diagnostics),
            },
        )

    def solve(self, rhs: Any, x0: Any = None) -> Any:
        return self.solve_result(rhs, x0=x0).x


__all__ = ["NativeSolve", "WholeSolveKrylovDriver"]
