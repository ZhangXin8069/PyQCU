"""Backend-neutral adapters for the LinearOperator protocol."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional

from .models import LatticeGeometry, ResidualScope, _token


ApplyCallable = Callable[[Any, Optional[int]], Any]
DiagonalCallable = Callable[[Optional[int]], Any]


class CallableLinearOperator:
    """Expose explicit callables through the common LinearOperator protocol."""

    def __init__(
            self, *, backend: str, geometry: LatticeGeometry,
            apply_fn: ApplyCallable,
            capabilities: Optional[Mapping[str, Any]] = None,
            residual_scope: Optional[ResidualScope] = None,
            adjoint_fn: Optional[ApplyCallable] = None,
            diagonal_fn: Optional[DiagonalCallable] = None,
            normal_fn: Optional[ApplyCallable] = None,
            matpc_fn: Optional[ApplyCallable] = None):
        if not isinstance(geometry, LatticeGeometry):
            raise TypeError("geometry must be a LatticeGeometry")
        if not callable(apply_fn):
            raise TypeError("apply_fn must be callable")
        scope = residual_scope
        if scope is not None and not isinstance(scope, ResidualScope):
            scope = ResidualScope(_token(scope, "residual_scope"))
        self.backend = _token(backend, "backend")
        self.geometry = geometry
        self.residual_scope = scope
        self.capabilities = MappingProxyType(dict(capabilities or {}))
        self._apply_fn = apply_fn
        self._adjoint_fn = adjoint_fn
        self._diagonal_fn = diagonal_fn
        self._normal_fn = normal_fn
        self._matpc_fn = matpc_fn

    @staticmethod
    def _parity(parity: Optional[int]) -> Optional[int]:
        if parity is None:
            return None
        value = int(parity)
        if value not in (0, 1):
            raise ValueError("parity must be None, 0, or 1")
        return value

    def apply(self, x: Any, *, parity: Optional[int] = None,
              adjoint: bool = False) -> Any:
        parity = self._parity(parity)
        if adjoint:
            return self.apply_adjoint(x, parity=parity)
        return self._apply_fn(x, parity)

    def apply_adjoint(self, x: Any, *, parity: Optional[int] = None) -> Any:
        parity = self._parity(parity)
        if self._adjoint_fn is None:
            raise NotImplementedError("operator does not provide an adjoint")
        return self._adjoint_fn(x, parity)

    def diagonal(self, parity: Optional[int] = None) -> Any:
        parity = self._parity(parity)
        if self._diagonal_fn is None:
            raise NotImplementedError("operator does not provide a diagonal")
        return self._diagonal_fn(parity)

    def apply_normal(self, x: Any, *, parity: Optional[int] = None) -> Any:
        parity = self._parity(parity)
        if self._normal_fn is not None:
            return self._normal_fn(x, parity)
        return self.apply_adjoint(self.apply(x, parity=parity), parity=parity)

    def apply_matpc(self, x: Any, *, parity: Optional[int] = None) -> Any:
        parity = self._parity(parity)
        if self._matpc_fn is None:
            raise NotImplementedError("operator does not provide MATPC apply")
        return self._matpc_fn(x, parity)


class NormalOperator(CallableLinearOperator):
    """Explicit HPD normal operator ``A†A`` for CG."""

    def __init__(self, operator: Any):
        if not callable(getattr(operator, "apply", None)):
            raise TypeError("operator must provide apply()")
        if not callable(getattr(operator, "apply_adjoint", None)):
            raise TypeError("normal operator requires an adjoint")
        capabilities = dict(getattr(operator, "capabilities", {}))
        capabilities.update({
            "adjoint": True,
            "hpd_cg": True,
            "normal_cg": True,
        })
        self.operator = operator

        def apply_normal(value: Any, parity: Optional[int]) -> Any:
            image = operator.apply(value, parity=parity)
            return operator.apply_adjoint(image, parity=parity)

        super().__init__(
            backend=getattr(operator, "backend", "custom"),
            geometry=operator.geometry,
            residual_scope=ResidualScope.NORMAL,
            capabilities=capabilities,
            apply_fn=apply_normal,
            adjoint_fn=apply_normal,
            normal_fn=apply_normal,
        )


__all__ = ["CallableLinearOperator", "NormalOperator"]
