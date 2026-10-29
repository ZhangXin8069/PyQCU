"""Common-result BiCGStab and CG drivers for Torch LinearOperator objects."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Optional

import pyqcu.cann as _torch
from pyqcu.torch import tools as torch_tools
from pyqcu.api import (
    BackendCapabilities,
    BiCGStabControl,
    CGControl,
    CGOperator,
    ResidualScope,
    SolveResult,
    SolveStatus,
    SolverControl,
)


class TorchBiCGStabDriver:
    """BiCGStab control plane over ``operator.apply``."""

    backend = "torch"
    method = "bicgstab"

    def __init__(self, operator: Any, control: SolverControl):
        if not isinstance(control, BiCGStabControl):
            raise TypeError("TorchBiCGStabDriver requires BiCGStabControl")
        _require_operator(operator)
        _require_scope_match(operator, control)
        self.operator = operator
        self._control = control
        self.capabilities = _driver_capabilities(operator)

    @property
    def control(self) -> SolverControl:
        return self._control

    def solve_result(self, rhs: Any, x0: Any = None) -> SolveResult:
        start = perf_counter()
        solution, iterations, status, history, restart_count = _bicgstab(
            self.operator, rhs, x0, self._control)
        residual, relative, rhs_norm = _true_residual(
            self.operator, rhs, solution, self._control.residual_scope)
        converged = residual <= self._control.convergence_threshold(
            rhs_norm)
        if converged:
            status = SolveStatus.CONVERGED
        elif status is SolveStatus.CONVERGED:
            status = SolveStatus.NOT_CONVERGED
        return SolveResult(
            x=solution,
            backend=self.backend,
            variant="krylov",
            method=self.method,
            status=status,
            converged=converged,
            krylov_iterations=iterations,
            restart_count=restart_count,
            final_residual=residual,
            residual_scope=self._residual_scope(),
            relative_true_residual=relative,
            setup_seconds=0.0,
            solve_seconds=perf_counter() - start,
            diagnostics={
                "recurrence_history": tuple(history),
                "rhs_norm": rhs_norm,
            },
        )

    def solve(self, rhs: Any, x0: Any = None) -> Any:
        return self.solve_result(rhs, x0=x0).x

    def _residual_scope(self) -> ResidualScope:
        scope = getattr(self.operator, "residual_scope", None)
        return scope or self.control.residual_scope


class TorchCGDriver:
    """CG control plane restricted to explicitly declared HPD operators."""

    backend = "torch"
    method = "cg"

    def __init__(self, operator: Any, control: SolverControl):
        if not isinstance(control, CGControl):
            raise TypeError("TorchCGDriver requires CGControl")
        _require_operator(operator)
        _require_scope_match(operator, control)
        required = {
            CGOperator.HPD: "hpd_cg",
            CGOperator.NORMAL: "normal_cg",
            CGOperator.SCHUR: "schur_cg",
        }[control.operator_kind]
        capabilities = dict(getattr(operator, "capabilities", {}))
        if not bool(capabilities.get(required, False)):
            raise ValueError(f"CG operator must declare {required}=True")
        self.operator = operator
        self._control = control
        self.capabilities = _driver_capabilities(operator)

    @property
    def control(self) -> SolverControl:
        return self._control

    def solve_result(self, rhs: Any, x0: Any = None) -> SolveResult:
        start = perf_counter()
        solution, iterations, status, history = _cg(
            self.operator, rhs, x0, self._control)
        residual, relative, rhs_norm = _true_residual(
            self.operator, rhs, solution, self._control.residual_scope)
        converged = residual <= self._control.convergence_threshold(
            rhs_norm)
        if converged:
            status = SolveStatus.CONVERGED
        elif status is SolveStatus.CONVERGED:
            status = SolveStatus.NOT_CONVERGED
        scope = getattr(self.operator, "residual_scope", None)
        scope = scope or self._control.residual_scope
        return SolveResult(
            x=solution,
            backend=self.backend,
            variant="krylov",
            method=self.method,
            status=status,
            converged=converged,
            krylov_iterations=iterations,
            final_residual=residual,
            residual_scope=scope,
            relative_true_residual=relative,
            setup_seconds=0.0,
            solve_seconds=perf_counter() - start,
            diagnostics={
                "recurrence_history": tuple(history),
                "rhs_norm": rhs_norm,
                "operator_kind": self.control.operator_kind.value,
            },
        )

    def solve(self, rhs: Any, x0: Any = None) -> Any:
        return self.solve_result(rhs, x0=x0).x


def _bicgstab(
        operator: Any, rhs: Any, x0: Any, control: BiCGStabControl
        ) -> tuple[Any, int, SolveStatus, list[float], int]:
    x = _initial_guess(rhs, x0)
    r = rhs - operator.apply(x)
    r_norm, rhs_norm = torch_tools.norm_many((r, rhs))
    threshold = control.convergence_threshold(rhs_norm)
    history = [r_norm]
    if r_norm <= threshold:
        return x, 0, SolveStatus.CONVERGED, history, 0

    shadow = r.clone()
    shadow_norm = r_norm
    p = _torch.zeros_like(rhs)
    v = _torch.zeros_like(rhs)
    s = _torch.zeros_like(rhs)
    t = _torch.zeros_like(rhs)
    rho_prev = rhs.new_tensor(1.0)
    alpha = rhs.new_tensor(1.0)
    omega = rhs.new_tensor(1.0)
    restart_count = 0
    status = SolveStatus.MAX_ITERATIONS
    iterations = 0

    def restart_after_breakdown() -> bool:
        nonlocal shadow, shadow_norm, p, v, s, t
        nonlocal rho_prev, alpha, omega, restart_count
        if restart_count >= control.breakdown_restarts:
            return False
        shadow = r.clone()
        shadow_norm = r_norm
        p = _torch.zeros_like(rhs)
        v = _torch.zeros_like(rhs)
        s = _torch.zeros_like(rhs)
        t = _torch.zeros_like(rhs)
        rho_prev = rhs.new_tensor(1.0)
        alpha = rhs.new_tensor(1.0)
        omega = rhs.new_tensor(1.0)
        restart_count += 1
        return True

    for iteration in range(control.max_iter):
        iterations = iteration + 1
        rho = torch_tools.vdot(shadow, r)
        if _breakdown(rho, shadow_norm * r_norm, r):
            if not restart_after_breakdown():
                status = SolveStatus.BREAKDOWN
                break
            continue
        beta = (rho / rho_prev) * (alpha / omega)
        rho_prev = rho
        p = r + beta * (p - omega * v)
        v = operator.apply(p)
        denominator, v_norm_sq = torch_tools.vdot_many((
            (shadow, v), (v, v)))
        v_norm = _norm_from_squared(v_norm_sq)
        if _breakdown(denominator, shadow_norm * v_norm, v):
            if not restart_after_breakdown():
                status = SolveStatus.BREAKDOWN
                break
            continue
        alpha = rho / denominator
        if _near_zero(alpha):
            if not restart_after_breakdown():
                status = SolveStatus.BREAKDOWN
                break
            continue
        s = r - alpha * v
        t = operator.apply(s)
        t_dot_s, t_norm_sq = torch_tools.vdot_many(((t, s), (t, t)))
        if _near_zero(t_norm_sq):
            if not restart_after_breakdown():
                status = SolveStatus.BREAKDOWN
                break
            continue
        omega = t_dot_s / t_norm_sq
        if not bool(_torch.torch.isfinite(alpha)) or not bool(
                _torch.torch.isfinite(omega)) or _near_zero(omega):
            if not restart_after_breakdown():
                status = SolveStatus.BREAKDOWN
                break
            continue
        x = x + alpha * p + omega * s
        r = s - omega * t
        r_norm = _field_norm(r)
        history.append(r_norm)
        if iterations >= control.min_iter and r_norm <= threshold:
            status = SolveStatus.CONVERGED
            break
        if _monitor_due(control, iterations):
            monitored, _, _ = _true_residual(
                operator, rhs, x, control.residual_scope)
            if iterations >= control.min_iter and monitored <= threshold:
                status = SolveStatus.CONVERGED
                break
    return x, iterations, status, history, restart_count


def _cg(
        operator: Any, rhs: Any, x0: Any, control: CGControl
        ) -> tuple[Any, int, SolveStatus, list[float]]:
    x = _initial_guess(rhs, x0)
    r = rhs - operator.apply(x)
    p = r.clone()
    rs_old, rhs_norm_sq = torch_tools.vdot_many((
        (r, r), (rhs, rhs)))
    r_norm = _norm_from_squared(rs_old)
    rhs_norm = _norm_from_squared(rhs_norm_sq)
    threshold = control.convergence_threshold(rhs_norm)
    history = [r_norm]
    if history[0] <= threshold:
        return x, 0, SolveStatus.CONVERGED, history
    status = SolveStatus.MAX_ITERATIONS
    iterations = 0
    for iteration in range(control.max_iter):
        iterations = iteration + 1
        image = operator.apply(p)
        denominator, p_norm_sq, image_norm_sq = torch_tools.vdot_many((
            (p, image), (p, p), (image, image)))
        scale = _norm_from_squared(p_norm_sq) * _norm_from_squared(
            image_norm_sq)
        if _breakdown(denominator, scale, image):
            status = SolveStatus.BREAKDOWN
            break
        alpha = rs_old / denominator
        x = x + alpha * p
        r = r - alpha * image
        rs_new = torch_tools.vdot(r, r)
        r_norm = _norm_from_squared(rs_new)
        history.append(r_norm)
        if iterations >= control.min_iter and r_norm <= threshold:
            status = SolveStatus.CONVERGED
            break
        if _monitor_due(control, iterations):
            monitored, _, _ = _true_residual(
                operator, rhs, x, control.residual_scope)
            if iterations >= control.min_iter and monitored <= threshold:
                status = SolveStatus.CONVERGED
                break
        if _near_zero(rs_old):
            status = SolveStatus.BREAKDOWN
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x, iterations, status, history


def _initial_guess(rhs: Any, x0: Any) -> Any:
    if x0 is None:
        return _torch.zeros_like(rhs)
    if tuple(x0.shape) != tuple(rhs.shape):
        raise ValueError("x0 and rhs shapes must match")
    if x0.dtype != rhs.dtype or x0.device != rhs.device:
        raise ValueError("x0 and rhs dtype/device must match")
    return x0.clone()


def _field_norm(value: Any) -> float:
    return float(torch_tools.norm(value))


def _norm_from_squared(value: Any) -> float:
    return float(_torch.torch.sqrt(_torch.torch.abs(value)).item())


def _true_residual(
        operator: Any, rhs: Any, x: Any, scope: ResidualScope
        ) -> tuple[float, float, float]:
    del scope
    residual = rhs - operator.apply(x)
    absolute, rhs_norm = torch_tools.norm_many((residual, rhs))
    relative = 0.0 if rhs_norm == 0 else absolute / rhs_norm
    return absolute, relative, rhs_norm


def _monitor_due(control: SolverControl, iteration: int) -> bool:
    return bool(
        control.require_true_residual or
        (control.true_residual_every is not None and
         iteration % control.true_residual_every == 0))


def _near_zero(value: Any) -> bool:
    return abs(value) <= 1e-30


def _breakdown(value: Any, scale: float, reference: Any) -> bool:
    eps = float(_torch.torch.finfo(reference.real.dtype).eps)
    return (
        not bool(_torch.isfinite(value)) or scale == 0 or
        abs(value) <= eps * scale
    )


def _require_operator(operator: Any) -> None:
    if not callable(getattr(operator, "apply", None)):
        raise TypeError("operator must provide apply()")


def _require_scope_match(operator: Any, control: SolverControl) -> None:
    scope = getattr(operator, "residual_scope", None)
    if scope is not None and control.residual_scope is not scope:
        raise ValueError(
            "control.residual_scope must match operator.residual_scope")


def _driver_capabilities(operator: Any) -> BackendCapabilities:
    source = dict(getattr(operator, "capabilities", {}))
    return BackendCapabilities(
        backend="torch",
        parity=bool(source.get("parity", False)),
        matpc=bool(source.get("matpc", False)),
        adjoint=bool(source.get("adjoint", False)),
        hpd_cg=bool(source.get("hpd_cg", False)),
        normal_cg=bool(source.get("normal_cg", False)),
        schur_cg=bool(source.get("schur_cg", False)),
        python_krylov=True,
        supported_precisions=("c64", "c128"),
    )


__all__ = ["TorchBiCGStabDriver", "TorchCGDriver"]
