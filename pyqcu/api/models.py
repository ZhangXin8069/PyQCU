"""Backend-neutral data models shared by PyQCU operators and solvers."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple


_CAPABILITY_FLAGS = frozenset({
    "full_coarse",
    "compact_coarse",
    "parity",
    "matpc",
    "adjoint",
    "persistent_hierarchy",
    "persistent_workspace",
    "overlap_halo",
    "mixed_precision",
    "warm_start",
    "cuda_graph",
    "nvshmem",
    "hpd_cg",
    "normal_cg",
    "schur_cg",
    "native_krylov",
    "python_krylov",
})


class ResidualScope(str, Enum):
    """Equation whose residual is reported by a solver result."""

    FULL = "full"
    MATPC = "matpc"
    SCHUR = "schur"
    NORMAL = "normal"
    PRECONDITIONED = "preconditioned"
    UNKNOWN = "unknown"


class SolveStatus(str, Enum):
    """Terminal status of a solve operation."""

    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    BREAKDOWN = "breakdown"
    NOT_CONVERGED = "not_converged"
    FAILED = "failed"
    UNKNOWN = "unknown"


def _token(value: Any, field_name: str) -> str:
    token = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if not token:
        raise ValueError(f"{field_name} must not be empty")
    return token


def _shape4(value: Sequence[int], field_name: str) -> Tuple[int, int, int, int]:
    shape = tuple(int(item) for item in value)
    if len(shape) != 4 or any(item <= 0 for item in shape):
        raise ValueError(f"{field_name} must contain four positive integers")
    return shape  # type: ignore[return-value]


def _optional_nonnegative_float(value: Optional[float],
                                field_name: str) -> Optional[float]:
    if value is None:
        return None
    converted = float(value)
    if converted < 0 or not math.isfinite(converted):
        raise ValueError(f"{field_name} must be finite and non-negative")
    return converted


def _optional_nonnegative_int(value: Optional[int],
                              field_name: str) -> Optional[int]:
    if value is None:
        return None
    converted = int(value)
    if converted < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return converted


@dataclass(frozen=True)
class LatticeGeometry:
    """Global process geometry and rank-local field shape."""

    local_shape: Sequence[int]
    dtype: str
    process_grid: Sequence[int] = (1, 1, 1, 1)
    parity: Optional[int] = None
    layout: str = "xyzt"

    def __post_init__(self) -> None:
        shape = _shape4(self.local_shape, "local_shape")
        grid = _shape4(self.process_grid, "process_grid")
        parity = None if self.parity is None else int(self.parity)
        if parity not in (None, 0, 1):
            raise ValueError("parity must be None, 0, or 1")
        if parity is not None and shape[3] % 2:
            raise ValueError("parity geometry requires an even local T extent")
        dtype = _token(self.dtype, "dtype")
        layout = _token(self.layout, "layout")
        object.__setattr__(self, "local_shape", shape)
        object.__setattr__(self, "process_grid", grid)
        object.__setattr__(self, "parity", parity)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "layout", layout)

    @property
    def full_shape(self) -> Tuple[int, int, int, int]:
        return tuple(self.local_shape)  # type: ignore[return-value]

    @property
    def volume(self) -> int:
        result = 1
        for extent in self.local_shape:
            result *= extent
        return result

    @property
    def local_sites(self) -> int:
        return self.volume if self.parity is None else self.volume // 2


@dataclass(frozen=True)
class SolveResult:
    """Uniform solve result without changing backend array ownership."""

    x: Any
    backend: str
    variant: str
    method: str
    status: SolveStatus = SolveStatus.UNKNOWN
    converged: Optional[bool] = None
    krylov_iterations: Optional[int] = None
    restart_count: Optional[int] = None
    correction_count: Optional[int] = None
    proposed_correction_count: Optional[int] = None
    rejected_correction_count: Optional[int] = None
    cycle_count: Optional[int] = None
    coarse_iterations: Optional[int] = None
    final_residual: Optional[float] = None
    residual_scope: Optional[ResidualScope] = None
    matpc_true_residual: Optional[float] = None
    full_true_residual: Optional[float] = None
    relative_true_residual: Optional[float] = None
    setup_seconds: Optional[float] = None
    solve_seconds: Optional[float] = None
    workspace_bytes: Optional[int] = None
    cache_status: Optional[str] = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        backend = _token(self.backend, "backend")
        variant = _token(self.variant, "variant")
        method = _token(self.method, "method")
        status = (self.status if isinstance(self.status, SolveStatus)
                  else SolveStatus(_token(self.status, "status")))
        scope = self.residual_scope
        if scope is not None and not isinstance(scope, ResidualScope):
            scope = ResidualScope(_token(scope, "residual_scope"))
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

        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "variant", variant)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "converged", converged)
        object.__setattr__(self, "residual_scope", scope)
        for name in (
                "krylov_iterations", "restart_count", "correction_count",
                "proposed_correction_count", "rejected_correction_count",
                "cycle_count", "coarse_iterations", "workspace_bytes"):
            object.__setattr__(
                self, name,
                _optional_nonnegative_int(getattr(self, name), name))
        for name in (
                "final_residual", "matpc_true_residual", "full_true_residual",
                "relative_true_residual", "setup_seconds", "solve_seconds"):
            object.__setattr__(
                self, name,
                _optional_nonnegative_float(getattr(self, name), name))
        if (self.residual_scope is None and any(
                getattr(self, name) is not None for name in (
                    "final_residual", "matpc_true_residual",
                    "full_true_residual", "relative_true_residual"))):
            raise ValueError("residual values require residual_scope")
        if self.cache_status is not None:
            object.__setattr__(
                self, "cache_status", _token(self.cache_status, "cache_status"))
        object.__setattr__(
            self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


@dataclass(frozen=True)
class BackendCapabilities:
    """Declared capabilities used for fail-closed planning."""

    backend: str
    variant: Optional[str] = None
    full_coarse: bool = False
    compact_coarse: bool = False
    parity: bool = False
    matpc: bool = False
    adjoint: bool = False
    persistent_hierarchy: bool = False
    persistent_workspace: bool = False
    overlap_halo: bool = False
    mixed_precision: bool = False
    warm_start: bool = False
    cuda_graph: bool = False
    nvshmem: bool = False
    hpd_cg: bool = False
    normal_cg: bool = False
    schur_cg: bool = False
    native_krylov: bool = False
    python_krylov: bool = False
    supported_precisions: Sequence[str] = ()
    max_levels: Optional[int] = None
    cache_schema: Optional[str] = None

    def __post_init__(self) -> None:
        backend = _token(self.backend, "backend")
        variant = None if self.variant is None else _token(self.variant, "variant")
        precisions = tuple(
            _token(value, "supported_precisions")
            for value in self.supported_precisions)
        if len(set(precisions)) != len(precisions):
            raise ValueError("supported_precisions must not contain duplicates")
        max_levels = self.max_levels
        if max_levels is not None:
            max_levels = int(max_levels)
            if max_levels <= 0:
                raise ValueError("max_levels must be positive or None")
        cache_schema = self.cache_schema
        if cache_schema is not None:
            cache_schema = _token(cache_schema, "cache_schema")
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "variant", variant)
        object.__setattr__(self, "supported_precisions", precisions)
        object.__setattr__(self, "max_levels", max_levels)
        object.__setattr__(self, "cache_schema", cache_schema)

    def supports_precision(self, precision: str) -> bool:
        return _token(precision, "precision") in self.supported_precisions

    def require(self, capability: str) -> "BackendCapabilities":
        name = _token(capability, "capability")
        if name not in _CAPABILITY_FLAGS:
            raise ValueError(f"unknown capability {name!r}")
        if not bool(getattr(self, name, False)):
            raise ValueError(
                f"backend={self.backend!r}, variant={self.variant!r} "
                f"does not support {name}")
        return self


@dataclass(frozen=True)
class AssetManifest:
    """Typed provenance record for a backend-owned hierarchy asset."""

    kind: str
    backend: str
    variant: str
    precision: str
    lattice_shape: Sequence[int]
    process_grid: Sequence[int] = (1, 1, 1, 1)
    block_size: Optional[Sequence[int]] = None
    level_count: int = 1
    algorithm_version: Optional[str] = None
    logical_bytes: Optional[int] = None
    tensor_sha256: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        kind = _token(self.kind, "kind")
        backend = _token(self.backend, "backend")
        variant = _token(self.variant, "variant")
        precision = _token(self.precision, "precision")
        lattice_shape = _shape4(self.lattice_shape, "lattice_shape")
        process_grid = _shape4(self.process_grid, "process_grid")
        block_size = None
        if self.block_size is not None:
            block_size = _shape4(self.block_size, "block_size")
        level_count = int(self.level_count)
        if level_count <= 0:
            raise ValueError("level_count must be positive")
        logical_bytes = _optional_nonnegative_int(
            self.logical_bytes, "logical_bytes")
        schema_version = int(self.schema_version)
        if schema_version <= 0:
            raise ValueError("schema_version must be positive")
        digests = {}
        for name, digest in dict(self.tensor_sha256).items():
            key = _token(name, "tensor_sha256 name")
            value = str(digest).strip().lower()
            if len(value) != 64 or any(ch not in "0123456789abcdef"
                                       for ch in value):
                raise ValueError(
                    f"tensor_sha256[{key!r}] must be a 64-character hex digest")
            digests[key] = value
        metadata = MappingProxyType(dict(self.metadata))
        try:
            json.dumps(dict(metadata), ensure_ascii=True, sort_keys=True)
        except (TypeError, ValueError) as error:
            raise TypeError("metadata must be JSON serializable") from error
        algorithm_version = self.algorithm_version
        if algorithm_version is not None:
            algorithm_version = _token(algorithm_version, "algorithm_version")

        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "variant", variant)
        object.__setattr__(self, "precision", precision)
        object.__setattr__(self, "lattice_shape", lattice_shape)
        object.__setattr__(self, "process_grid", process_grid)
        object.__setattr__(self, "block_size", block_size)
        object.__setattr__(self, "level_count", level_count)
        object.__setattr__(self, "algorithm_version", algorithm_version)
        object.__setattr__(self, "logical_bytes", logical_bytes)
        object.__setattr__(self, "tensor_sha256", MappingProxyType(digests))
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "schema_version", schema_version)

    @property
    def identity(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "backend": self.backend,
            "variant": self.variant,
            "precision": self.precision,
            "lattice_shape": list(self.lattice_shape),
            "process_grid": list(self.process_grid),
            "block_size": (None if self.block_size is None
                           else list(self.block_size)),
            "level_count": self.level_count,
            "algorithm_version": self.algorithm_version,
            "logical_bytes": self.logical_bytes,
            "tensor_sha256": dict(self.tensor_sha256),
            "metadata": dict(self.metadata),
        }
        encoded = json.dumps(
            payload, ensure_ascii=True, sort_keys=True,
            separators=(",", ":")).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()
