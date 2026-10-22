"""Stable, backend-neutral PyQCU contracts.

This package defines validated configuration, result, capability, asset, and
protocol objects. It does not contain backend kernels or a fourth solver
implementation.
"""

from .config import (
    BiCGStabControl,
    CGControl,
    CGOperator,
    CoarsenConfig,
    CycleConfig,
    FGMRESControl,
    LegacyMultigridConfig,
    MRConfig,
    PeriodicCorrectionConfig,
    SolverControl,
    StrictMultigridConfig,
)
from .models import (
    AssetManifest,
    BackendCapabilities,
    LatticeGeometry,
    ResidualScope,
    SolveResult,
    SolveStatus,
)
from .native import NativeSolve, WholeSolveKrylovDriver
from .operator import CallableLinearOperator, NormalOperator
from .protocols import (
    CoarseOperator,
    KrylovDriver,
    LinearOperator,
    MultigridHierarchy,
    PyQCUBackend,
    Smoother,
    TransferOperator,
)

__all__ = [
    "AssetManifest",
    "BackendCapabilities",
    "BiCGStabControl",
    "CallableLinearOperator",
    "CGControl",
    "CGOperator",
    "CoarseOperator",
    "CoarsenConfig",
    "CycleConfig",
    "FGMRESControl",
    "KrylovDriver",
    "LatticeGeometry",
    "LegacyMultigridConfig",
    "LinearOperator",
    "MRConfig",
    "MultigridHierarchy",
    "NativeSolve",
    "NormalOperator",
    "PeriodicCorrectionConfig",
    "PyQCUBackend",
    "ResidualScope",
    "Smoother",
    "SolveResult",
    "SolveStatus",
    "SolverControl",
    "StrictMultigridConfig",
    "TransferOperator",
    "WholeSolveKrylovDriver",
]
