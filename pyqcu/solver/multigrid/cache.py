"""Unified backend-neutral API for Legacy and Strict MultiGrid caches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pyqcu.tools._legacy_mg_cache import (
    LEGACY_MG_ASSET_NAMES,
    canonical_dtype,
    legacy_mg_cache_path,
    load_legacy_mg_cache_slab,
    save_legacy_mg_cache,
    validate_legacy_mg_cache,
)
from pyqcu.tools._strict_mg_cache import (
    inspect_strict_mg_cache,
    load_strict_mg_cache,
    save_strict_mg_cache,
    strict_mg_backend_assets,
    strict_mg_cache_path,
)


@dataclass(frozen=True)
class MultigridCacheLoadResult:
    """Uniform cache load evidence for every QCU backend."""

    variant: str
    backend: str
    path: str
    assets: Mapping[str, Any]
    manifest: Mapping[str, Any]
    evidence: Mapping[str, Any]


def _backend_name(backend: str) -> str:
    name = str(backend).strip().lower()
    if name not in {"torch", "cuda", "kupl"}:
        raise ValueError(f"backend must be torch, cuda, or kupl, got {backend!r}")
    return name


def _variant_name(variant: str) -> str:
    name = str(variant).strip().lower().replace("-", "_")
    if name not in {"legacy", "strict"}:
        raise ValueError(f"variant must be legacy or strict, got {variant!r}")
    return name


class LegacyMultigridCache:
    """Canonical odd-Schur 33-tensor cache API."""

    variant = "legacy"

    def __init__(self, backend: str = "torch"):
        self.backend = _backend_name(backend)

    @staticmethod
    def path(identity: Mapping[str, Any], cache_dir: str | Path) -> Path:
        return legacy_mg_cache_path(identity, cache_dir)

    def validate(
            self, path: str | Path, *, identity: Mapping[str, Any],
            verify_file_sha256: bool = False) -> dict[str, Any]:
        return validate_legacy_mg_cache(
            path, identity, verify_file_sha256=verify_file_sha256)

    def load(
            self, path: str | Path, *, identity: Mapping[str, Any],
            dtype: Any = None, starts: Sequence[int] | None = None,
            local_shape: Sequence[int] | None = None,
            verify_file_sha256: bool = False) -> MultigridCacheLoadResult:
        manifest = self.validate(
            path, identity=identity, verify_file_sha256=verify_file_sha256)
        selected_dtype = canonical_dtype(
            identity["dtype"] if dtype is None else dtype)
        assets = load_legacy_mg_cache_slab(
            path, selected_dtype, starts=starts, local_shape=local_shape)
        evidence = {
            "schema_version": manifest.get("schema_version"),
            "identity_sha256": manifest.get("identity_sha256"),
            "file_sha256": manifest.get("file_sha256"),
            "datasets": dict(manifest.get("assets") or {}),
        }
        return MultigridCacheLoadResult(
            variant=self.variant, backend=self.backend,
            path=str(Path(path).resolve()), assets=assets,
            manifest=manifest, evidence=evidence)

    def save(
            self, path: str | Path, *, identity: Mapping[str, Any],
            assets: Mapping[str, np.ndarray], metadata: Any = None,
            stats: Any = None) -> dict[str, Any]:
        missing = sorted(set(LEGACY_MG_ASSET_NAMES).difference(assets))
        if missing:
            raise ValueError(f"Legacy MG cache assets are missing {missing}")
        setup_backend = str((metadata or {}).get(
            "setup_backend", f"{self.backend}-qcu"))
        setup_method = str((metadata or {}).get(
            "setup_method", "cxx-null-vector+local-33-stencil"))
        return save_legacy_mg_cache(
            path, assets, identity=identity, setup_method=setup_method,
            setup_backend=setup_backend, setup_validation=stats,
            source_metadata=metadata)


class StrictMultigridCache:
    """Canonical ``fine V + Yhat/(X,X^-1)`` runtime cache API."""

    variant = "strict"

    def __init__(self, backend: str = "torch"):
        self.backend = _backend_name(backend)

    @staticmethod
    def path(identity: Mapping[str, Any], cache_dir: str | Path) -> Path:
        return strict_mg_cache_path(identity, cache_dir)

    def validate(
            self, path: str | Path, *, identity: Mapping[str, Any],
            expected_manifest: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        result = inspect_strict_mg_cache(
            path, identity=identity, expected_manifest=expected_manifest)
        if not result.hit:
            raise ValueError(
                f"Strict MG cache miss: {result.reason}: {result.detail}")
        return dict(result.manifest or {})

    def load(
            self, path: str | Path, *, identity: Mapping[str, Any],
            expected_manifest: Mapping[str, Any] | None = None,
            device: Any = "cpu") -> MultigridCacheLoadResult:
        result = load_strict_mg_cache(
            path, identity=identity, device=device,
            expected_manifest=expected_manifest)
        mapped = strict_mg_backend_assets(result)
        assets = {
            "fine_null_vectors": _to_numpy(mapped["fine_null_vectors"]),
            "levels": tuple({
                **level,
                "preconditioned_links": _to_numpy(
                    level["preconditioned_links"]),
                "onsite_pair": _to_numpy(level["onsite_pair"]),
                "null_vectors": (
                    None if level["null_vectors"] is None else
                    _to_numpy(level["null_vectors"])),
            } for level in mapped["levels"]),
        }
        return MultigridCacheLoadResult(
            variant=self.variant, backend=self.backend,
            path=str(Path(path).resolve()), assets=assets,
            manifest=dict(mapped["manifest"]),
            evidence=dict(mapped["evidence"]))

    def save(
            self, path: str | Path, *, identity: Mapping[str, Any],
            assets: Sequence[Mapping[str, Any]] | Mapping[str, Any],
            metadata: Any = None, stats: Any = None) -> Mapping[str, Any]:
        transitions = _strict_transitions(assets)
        write_result = save_strict_mg_cache(
            path, identity=identity, transitions=transitions,
            metadata=metadata, stats=stats)
        return {
            "written": bool(write_result.written),
            "reason": str(write_result.reason),
            "path": str(write_result.path),
            "identity_sha256": str(write_result.identity_sha256),
            "tensor_count": int(write_result.tensor_count),
            "logical_bytes": int(write_result.logical_bytes),
        }


class MultigridCache:
    """Select one uniform cache API for any QCU backend."""

    def __init__(self, *, variant: str = "legacy", backend: str = "torch"):
        self.variant = _variant_name(variant)
        self.backend = _backend_name(backend)
        cache_type = (
            StrictMultigridCache if self.variant == "strict"
            else LegacyMultigridCache)
        self._implementation = cache_type(self.backend)

    def path(self, identity: Mapping[str, Any], cache_dir: str | Path) -> Path:
        return self._implementation.path(identity, cache_dir)

    def validate(self, path: str | Path, **kwargs: Any) -> dict[str, Any]:
        return self._implementation.validate(path, **kwargs)

    def load(self, path: str | Path, **kwargs: Any
             ) -> MultigridCacheLoadResult:
        return self._implementation.load(path, **kwargs)

    def save(self, path: str | Path, **kwargs: Any) -> Mapping[str, Any]:
        return self._implementation.save(path, **kwargs)


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value)
    if hasattr(value, "detach"):
        value = value.detach().cpu().contiguous()
    return np.ascontiguousarray(value)


def _strict_transitions(
        assets: Sequence[Mapping[str, Any]] | Mapping[str, Any]
) -> tuple[Mapping[str, Any], ...]:
    if isinstance(assets, Mapping):
        fine = assets.get("fine_null_vectors", assets.get("fine_blocked_v"))
        levels = assets.get("levels")
        if fine is None or not isinstance(levels, Sequence):
            raise ValueError(
                "Strict cache mapping requires fine_null_vectors and levels")
        transitions = []
        for index, level in enumerate(levels):
            transitions.append({
                "null_vectors": fine if index == 0 else level.get(
                    "null_vectors"),
                "preconditioned_links": level.get("preconditioned_links"),
                "onsite_pair": level.get("onsite_pair"),
            })
        return tuple(transitions)
    values = tuple(assets)
    if not values:
        raise ValueError("Strict cache requires at least one transition")
    return values


__all__ = [
    "LegacyMultigridCache",
    "MultigridCache",
    "MultigridCacheLoadResult",
    "StrictMultigridCache",
]
