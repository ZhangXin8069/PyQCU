"""Structural protocols for the unified PyQCU control plane."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, runtime_checkable

from .config import SolverControl
from .models import (
    AssetManifest,
    BackendCapabilities,
    LatticeGeometry,
    SolveResult,
)


@runtime_checkable
class LinearOperator(Protocol):
    """A Dirac, normal, Schur, or coarse linear operator."""

    backend: str

    @property
    def residual_scope(self) -> Optional[Any]:
        ...

    @property
    def geometry(self) -> LatticeGeometry:
        ...

    @property
    def capabilities(self) -> Mapping[str, Any]:
        ...

    def apply(self, x: Any, *, parity: Optional[int] = None,
              adjoint: bool = False) -> Any:
        ...

    def apply_adjoint(self, x: Any, *, parity: Optional[int] = None) -> Any:
        ...

    def diagonal(self, parity: Optional[int] = None) -> Any:
        ...

    def apply_normal(self, x: Any, *, parity: Optional[int] = None) -> Any:
        ...

    def apply_matpc(self, x: Any, *, parity: Optional[int] = None) -> Any:
        ...


@runtime_checkable
class TransferOperator(Protocol):
    """Restriction/prolongation pair between two geometries."""

    @property
    def fine_geometry(self) -> LatticeGeometry:
        ...

    @property
    def coarse_geometry(self) -> LatticeGeometry:
        ...

    def build(self, manifest: Optional[AssetManifest] = None) -> Any:
        ...

    def restrict(self, fine: Any) -> Any:
        ...

    def prolong(self, coarse: Any) -> Any:
        ...

    def restrict_adjoint(self, coarse: Any) -> Any:
        ...


@runtime_checkable
class CoarseOperator(LinearOperator, Protocol):
    """Linear operator acting on a coarse-grid field."""


@runtime_checkable
class Smoother(Protocol):
    """A fixed-step smoother used inside a multigrid cycle."""

    def step(self, rhs: Any, solution: Any, steps: int = 1) -> Any:
        ...


@runtime_checkable
class KrylovDriver(Protocol):
    """A configured Krylov solver over a LinearOperator."""

    @property
    def control(self) -> SolverControl:
        ...

    @property
    def capabilities(self) -> BackendCapabilities:
        ...

    def solve_result(self, rhs: Any, x0: Any = None) -> SolveResult:
        ...


@runtime_checkable
class MultigridHierarchy(Protocol):
    """Uniform lifecycle for Legacy and Strict multigrid hierarchies."""

    backend: str
    variant: str

    def capabilities(self) -> BackendCapabilities:
        ...

    def setup(self) -> Any:
        ...

    def solve_result(self, rhs: Any, x0: Any = None) -> SolveResult:
        ...

    def close(self) -> None:
        ...


@runtime_checkable
class PyQCUBackend(Protocol):
    """Long-term canonical backend factory for operators and solvers."""

    name: str

    def capabilities(self, operator: Optional[LinearOperator] = None,
                     variant: Optional[str] = None) -> BackendCapabilities:
        ...

    def dslash(self, **options: Any) -> LinearOperator:
        ...

    def solver(self, method: str, *, operator: LinearOperator,
               control: SolverControl) -> KrylovDriver:
        ...

    def multigrid(self, *, variant: str, operator: LinearOperator,
                  **options: Any) -> MultigridHierarchy:
        ...
