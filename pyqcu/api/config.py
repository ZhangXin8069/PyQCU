"""Validated, backend-neutral solver and multigrid configurations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

from .models import ResidualScope, _shape4, _token


class CGOperator(str, Enum):
    """Operator contract explicitly selected for a CG solve."""

    HPD = "hpd"
    NORMAL = "normal"
    SCHUR = "schur"


@dataclass(frozen=True)
class SolverControl:
    """Common convergence and true-residual policy."""

    rtol: float = 1e-8
    atol: float = 0.0
    max_iter: int = 1000
    min_iter: int = 0
    true_residual_every: Optional[int] = None
    require_true_residual: bool = False
    residual_scope: ResidualScope = ResidualScope.FULL

    def __post_init__(self) -> None:
        rtol = float(self.rtol)
        atol = float(self.atol)
        if not math.isfinite(rtol) or rtol < 0:
            raise ValueError("rtol must be finite and non-negative")
        if not math.isfinite(atol) or atol < 0:
            raise ValueError("atol must be finite and non-negative")
        if rtol == 0 and atol == 0:
            raise ValueError("rtol and atol must not both be zero")
        max_iter = int(self.max_iter)
        min_iter = int(self.min_iter)
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if min_iter < 0 or min_iter > max_iter:
            raise ValueError("min_iter must be between 0 and max_iter")
        check_every = self.true_residual_every
        if check_every is not None:
            check_every = int(check_every)
            if check_every <= 0:
                raise ValueError("true_residual_every must be positive or None")
        scope = self.residual_scope
        if not isinstance(scope, ResidualScope):
            scope = ResidualScope(_token(scope, "residual_scope"))
        object.__setattr__(self, "rtol", rtol)
        object.__setattr__(self, "atol", atol)
        object.__setattr__(self, "max_iter", max_iter)
        object.__setattr__(self, "min_iter", min_iter)
        object.__setattr__(self, "true_residual_every", check_every)
        object.__setattr__(self, "residual_scope", scope)

    def convergence_threshold(self, rhs_norm: float) -> float:
        norm = float(rhs_norm)
        if not math.isfinite(norm) or norm < 0:
            raise ValueError("rhs_norm must be finite and non-negative")
        return max(self.rtol * norm, self.atol)


@dataclass(frozen=True)
class BiCGStabControl(SolverControl):
    """BiCGStab controls for a general non-Hermitian operator."""

    breakdown_restarts: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if int(self.breakdown_restarts) < 0:
            raise ValueError("breakdown_restarts must be non-negative")
        object.__setattr__(
            self, "breakdown_restarts", int(self.breakdown_restarts))


@dataclass(frozen=True)
class FGMRESControl(SolverControl):
    """Right-preconditioned FGMRES controls."""

    restart: int = 20

    def __post_init__(self) -> None:
        super().__post_init__()
        restart = int(self.restart)
        if restart <= 0 or restart > self.max_iter:
            raise ValueError("restart must satisfy 1 <= restart <= max_iter")
        object.__setattr__(self, "restart", restart)


@dataclass(frozen=True)
class CGControl(SolverControl):
    """CG controls with an explicit operator assumption."""

    operator_kind: CGOperator = CGOperator.HPD

    def __post_init__(self) -> None:
        super().__post_init__()
        kind = self.operator_kind
        if not isinstance(kind, CGOperator):
            kind = CGOperator(_token(kind, "operator_kind"))
        object.__setattr__(self, "operator_kind", kind)


@dataclass(frozen=True)
class CoarsenConfig:
    """Shared block-coarsening geometry."""

    nvec: int
    block_size: Sequence[int]
    levels: int

    def __post_init__(self) -> None:
        nvec = int(self.nvec)
        block_size = _shape4(self.block_size, "block_size")
        levels = int(self.levels)
        if nvec <= 0:
            raise ValueError("nvec must be positive")
        if levels < 2:
            raise ValueError("levels must be at least 2")
        object.__setattr__(self, "nvec", nvec)
        object.__setattr__(self, "block_size", block_size)
        object.__setattr__(self, "levels", levels)


@dataclass(frozen=True)
class MRConfig:
    """Fixed-step MR smoother configuration."""

    steps: int = 2

    def __post_init__(self) -> None:
        steps = int(self.steps)
        if steps < 0:
            raise ValueError("steps must be non-negative")
        object.__setattr__(self, "steps", steps)


@dataclass(frozen=True)
class CycleConfig:
    """Recursive cycle shape and smoothing counts."""

    kind: str = "V"
    pre: int = 2
    post: int = 2

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().upper()
        if kind not in {"V", "W", "F", "K"}:
            raise ValueError("kind must be one of V, W, F, K")
        pre = int(self.pre)
        post = int(self.post)
        if pre < 0 or post < 0:
            raise ValueError("pre/post smoother counts must be non-negative")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "pre", pre)
        object.__setattr__(self, "post", post)


@dataclass(frozen=True)
class PeriodicCorrectionConfig:
    """Legacy periodic additive coarse-correction policy."""

    interval: int = 5
    gate: bool = True
    true_residual_rollback: bool = True

    def __post_init__(self) -> None:
        interval = int(self.interval)
        if interval <= 0:
            raise ValueError("interval must be positive")
        object.__setattr__(self, "interval", interval)


@dataclass(frozen=True)
class StrictMultigridConfig:
    """Canonical configuration for a strict recursive hierarchy."""

    coarsen: CoarsenConfig
    smoother: MRConfig = field(default_factory=MRConfig)
    cycle: CycleConfig = field(default_factory=CycleConfig)
    outer: FGMRESControl = field(default_factory=FGMRESControl)


@dataclass(frozen=True)
class LegacyMultigridConfig:
    """Canonical configuration for periodic legacy corrections."""

    coarsen: CoarsenConfig
    corrector: PeriodicCorrectionConfig = field(
        default_factory=PeriodicCorrectionConfig)
    outer: SolverControl = field(default_factory=BiCGStabControl)
