"""Adapters for exposing existing backend objects through the public API."""

from __future__ import annotations

from typing import Any, Optional

from .models import ResidualScope, SolveResult, SolveStatus


def scalar(value: Any) -> Any:
    """Return a scalar without importing a backend tensor implementation."""

    if value is None:
        return None
    numel = getattr(value, "numel", None)
    if callable(numel):
        try:
            if int(numel()) != 1:
                return None
        except (TypeError, ValueError):
            return None
        detach = getattr(value, "detach", None)
        if callable(detach):
            value = detach()
        item = getattr(value, "item", None)
        if callable(item):
            try:
                return item()
            except (RuntimeError, TypeError, ValueError):
                return None
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (RuntimeError, TypeError, ValueError):
            return None
    if isinstance(value, (bool, int, float)):
        return value
    return None


def optional_int(value: Any) -> Optional[int]:
    converted = scalar(value)
    if converted is None or isinstance(converted, bool):
        return None
    try:
        integer = int(converted)
    except (TypeError, ValueError):
        return None
    return integer if integer >= 0 else None


def optional_float(value: Any) -> Optional[float]:
    converted = scalar(value)
    if converted is None or isinstance(converted, bool):
        return None
    try:
        number = float(converted)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def optional_bool(value: Any) -> Optional[bool]:
    converted = scalar(value)
    if isinstance(converted, bool):
        return converted
    if converted in (0, 1):
        return bool(converted)
    return None


def _first_attr(implementation: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        if hasattr(implementation, name):
            return getattr(implementation, name)
    return None


def _residual_scope(value: Any) -> Optional[ResidualScope]:
    if value is None:
        return None
    if isinstance(value, ResidualScope):
        return value
    try:
        return ResidualScope(str(value).strip().lower())
    except ValueError:
        return ResidualScope.UNKNOWN


def _solve_status(value: Any, converged: Optional[bool]) -> SolveStatus:
    if value is not None:
        if isinstance(value, SolveStatus):
            return value
        try:
            return SolveStatus(str(value).strip().lower())
        except ValueError:
            return SolveStatus.UNKNOWN
    if converged is True:
        return SolveStatus.CONVERGED
    if converged is False:
        return SolveStatus.NOT_CONVERGED
    return SolveStatus.UNKNOWN


def compatibility_result(
        *, implementation: Any, x: Any, backend: str, variant: str,
        setup_seconds: float, solve_seconds: float) -> SolveResult:
    """Build a result using only evidence exposed by an existing backend."""

    converged = optional_bool(
        _first_attr(implementation, ("converged", "is_converged")))
    status_value = _first_attr(
        implementation, ("solve_status", "status"))
    status = _solve_status(status_value, converged)
    scope = _residual_scope(
        _first_attr(implementation, ("residual_scope",)))
    final_residual = optional_float(
        _first_attr(implementation, ("final_residual",)))
    if final_residual is None:
        history = _first_attr(
            implementation, ("convergence_history", "residual_history"))
        if history is not None:
            try:
                if len(history):
                    final_residual = optional_float(history[-1])
            except TypeError:
                pass
    if scope is None:
        final_residual = None
    matpc_residual = optional_float(
        _first_attr(implementation, ("matpc_true_residual",)))
    if matpc_residual is None and scope is ResidualScope.MATPC:
        matpc_residual = final_residual
    full_residual = optional_float(
        _first_attr(implementation, ("full_true_residual",)))
    relative_residual = optional_float(
        _first_attr(implementation, ("relative_true_residual",)))
    iterations = optional_int(_first_attr(
        implementation, ("outer_iterations", "iterations")))
    workspace_bytes = optional_int(_first_attr(
        implementation,
        ("workspace_bytes", "allocated_bytes",
         "_fused_workspace_resident_bytes",
         "fused_workspace_planned_bytes")))
    reported_solve_seconds = optional_float(
        _first_attr(implementation, ("last_solve_seconds",)))
    cache_status = _first_attr(implementation, ("cache_status",))
    diagnostics = {
        "compatibility_adapter": True,
        "implementation_class": (
            f"{type(implementation).__module__}.{type(implementation).__name__}"),
    }
    if reported_solve_seconds is not None:
        diagnostics["reported_solve_seconds"] = reported_solve_seconds

    return SolveResult(
        x=x,
        backend=backend,
        variant=variant,
        method="multigrid",
        status=status,
        converged=converged,
        krylov_iterations=iterations,
        restart_count=optional_int(_first_attr(
            implementation, ("restart_count",))),
        correction_count=optional_int(_first_attr(
            implementation, ("correction_count",))),
        proposed_correction_count=optional_int(_first_attr(
            implementation, ("proposed_correction_count",))),
        rejected_correction_count=optional_int(_first_attr(
            implementation, ("rejected_correction_count",))),
        cycle_count=optional_int(_first_attr(
            implementation, ("cycle_count",))),
        coarse_iterations=optional_int(_first_attr(
            implementation, ("coarse_iterations",))),
        final_residual=final_residual,
        residual_scope=scope,
        matpc_true_residual=matpc_residual,
        full_true_residual=full_residual,
        relative_true_residual=relative_residual,
        setup_seconds=setup_seconds,
        solve_seconds=solve_seconds,
        workspace_bytes=workspace_bytes,
        cache_status=(None if cache_status is None else str(cache_status)),
        diagnostics=diagnostics,
    )
