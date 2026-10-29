"""Backend-neutral access to Strict-MultiGrid runtime caches.

The on-disk protocol remains ``pyqcu.strict-runtime-cache`` v2.  It already
contains the complete reusable hierarchy:

* the fine blocked null-space ``V``;
* each transition's preconditioned links ``Yhat`` and onsite pair ``X/X^-1``;
* recursive coarse ``V`` tensors for transitions beyond the first.

CUDA, Torch, and KUPL differ only in how these tensors are bound.  This module
centralizes identity construction, path selection, transition normalization,
and backend-facing unpacking without duplicating the validated v2 HDF5 code.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


STRICT_MG_CACHE_IDENTITY_SCHEMA = "pyqcu.strict-mg-cache-identity/v1"
STRICT_MG_CACHE_FORMAT = "pyqcu.strict-runtime-cache"
STRICT_MG_CACHE_FORMAT_VERSION = 2


def _require_sha256(value: Any, name: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return text


def _shape4(values: Sequence[int], name: str) -> tuple[int, int, int, int]:
    result = tuple(int(value) for value in values)
    if len(result) != 4 or any(value <= 0 for value in result):
        raise ValueError(f"{name} must contain four positive integers: {result}")
    return result  # type: ignore[return-value]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False)


def make_strict_mg_cache_identity(
        *,
        gauge_sha256: str,
        null_vectors_sha256: str,
        lattice_xyzt: Sequence[int],
        dtype: str,
        mass: float,
        kappa: float,
        levels: int,
        block_xyzt_per_level: Sequence[Sequence[int]],
        nvec: int,
        coarse_spin: int,
        target_parity: int,
        gamma_basis: str,
        setup_semantics: str,
        mpi_context: Mapping[str, Any] | None = None,
        coarse_block_setup_precision: str | None = None) -> dict[str, Any]:
    """Build the physical identity shared by all Strict-MG backends."""
    lattice = _shape4(lattice_xyzt, "lattice_xyzt")
    level_count = int(levels)
    blocks = tuple(
        _shape4(block, f"block_xyzt_per_level[{index}]")
        for index, block in enumerate(block_xyzt_per_level))
    if level_count < 2 or len(blocks) != level_count - 1:
        raise ValueError(
            "Strict MG requires levels >= 2 and one block per transition")
    mass_value = float(mass)
    kappa_value = float(kappa)
    if not math.isfinite(mass_value) or not math.isfinite(kappa_value):
        raise ValueError("Strict MG mass and kappa must be finite")
    precision = {
        "c64": "c64", "complex64": "c64",
        "c128": "c128", "complex128": "c128",
    }.get(str(dtype).lower())
    if precision is None:
        raise ValueError(f"unsupported Strict MG dtype {dtype!r}")
    parity = int(target_parity)
    if parity not in (0, 1):
        raise ValueError("target_parity must be 0 or 1")
    nvec_value = int(nvec)
    spin = int(coarse_spin)
    if nvec_value <= 0 or spin <= 0:
        raise ValueError("nvec and coarse_spin must be positive")
    identity: dict[str, Any] = {
        "schema": STRICT_MG_CACHE_IDENTITY_SCHEMA,
        "runtime_cache_format": STRICT_MG_CACHE_FORMAT,
        "runtime_cache_format_version": STRICT_MG_CACHE_FORMAT_VERSION,
        "gauge_sha256": _require_sha256(gauge_sha256, "gauge_sha256"),
        "null_vectors_sha256": _require_sha256(
            null_vectors_sha256, "null_vectors_sha256"),
        "lattice_xyzt": list(lattice),
        "precision": precision,
        "mass": mass_value,
        "kappa": kappa_value,
        "levels": level_count,
        "block_xyzt_per_level": [list(block) for block in blocks],
        "nvec": nvec_value,
        "coarse_spin": spin,
        "coarse_dof": spin * nvec_value,
        "target_parity": parity,
        "gamma_basis": str(gamma_basis),
        "setup_semantics": str(setup_semantics),
        "runtime_assets": "fine blocked V; per-transition Yhat/(X,X^-1)",
    }
    if coarse_block_setup_precision is not None:
        identity["coarse_block_setup_precision"] = str(
            coarse_block_setup_precision)
    if mpi_context is not None:
        rank = int(mpi_context.get("rank", 0))
        size = int(mpi_context.get("size", 1))
        process_grid = _shape4(
            mpi_context.get("process_grid", (1, 1, 1, 1)),
            "mpi_context.process_grid")
        rank_coordinate = tuple(
            int(value) for value in mpi_context.get(
                "rank_coordinate", (0, 0, 0, 0)))
        if len(rank_coordinate) != 4 or any(value < 0 for value in rank_coordinate):
            raise ValueError("mpi_context.rank_coordinate must be four nonnegative ints")
        product = 1
        for value in process_grid:
            product *= value
        if product != size or rank < 0 or rank >= size:
            raise ValueError("MPI rank/grid context is inconsistent")
        identity["mpi_shard"] = {
            "ranks": size,
            "process_grid": list(process_grid),
            "rank": rank,
            "rank_coordinate": list(rank_coordinate),
        }
    return identity


def strict_mg_cache_identity_sha256(identity: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(dict(identity)).encode("utf-8")).hexdigest()


def strict_mg_cache_path(
        identity: Mapping[str, Any],
        cache_dir: str | Path) -> Path:
    digest = strict_mg_cache_identity_sha256(identity)[:24]
    return Path(cache_dir) / f"strict_runtime_{digest}.h5"


def _normalize_transition_assets(
        transitions: Sequence[Mapping[str, Any]],
) -> tuple[Any, tuple[dict[str, Any], ...]]:
    values = tuple(transitions)
    if not values:
        raise ValueError("Strict MG cache needs at least one transition")
    normalized: list[dict[str, Any]] = []
    fine_blocked_v = None
    for level, transition in enumerate(values):
        missing = [
            name for name in ("null_vectors", "preconditioned_links", "onsite_pair")
            if transition.get(name) is None
        ]
        if missing:
            raise ValueError(f"transition {level} is missing {missing}")
        if level == 0:
            fine_blocked_v = transition["null_vectors"]
        normalized.append({
            "level": level,
            "preconditioned_links": transition["preconditioned_links"],
            "onsite_pair": transition["onsite_pair"],
            "null_vectors": (
                None if level == 0 else transition["null_vectors"]),
        })
    return fine_blocked_v, tuple(normalized)


def save_strict_mg_cache(
        path: str | Path, *, identity: Mapping[str, Any],
        transitions: Sequence[Mapping[str, Any]], metadata: Any = None,
        stats: Any = None) -> Any:
    """Save backend-neutral transition assets to the common v2 protocol."""
    from pyqcu.cuda._strict_cache import save_strict_runtime_cache

    fine_blocked_v, levels = _normalize_transition_assets(transitions)
    return save_strict_runtime_cache(
        path, identity=identity, fine_blocked_v=fine_blocked_v,
        levels=levels, metadata=metadata, stats=stats)


def load_strict_mg_cache(
        path: str | Path, *, identity: Mapping[str, Any],
        device: Any = "cpu", expected_manifest: Mapping[str, Any] | None = None
) -> Any:
    """Load a common v2 cache after full identity/content verification."""
    from pyqcu.cuda._strict_cache import load_strict_runtime_cache

    return load_strict_runtime_cache(
        path, identity=identity, device=device,
        expected_manifest=expected_manifest)


def inspect_strict_mg_cache(
        path: str | Path, *, identity: Mapping[str, Any],
        expected_manifest: Mapping[str, Any] | None = None) -> Any:
    """Validate a common v2 cache without transferring tensors to a device."""
    from pyqcu.cuda._strict_cache import inspect_strict_runtime_cache

    return inspect_strict_runtime_cache(
        path, identity=identity, expected_manifest=expected_manifest)


def strict_mg_backend_assets(result: Any) -> dict[str, Any]:
    """Map a verified cache result to CUDA/Torch/KUPL bindings."""
    if not getattr(result, "hit", False):
        reason = getattr(result, "reason", "unknown")
        detail = getattr(result, "detail", "")
        raise ValueError(f"Strict MG cache miss: {reason}: {detail}")
    assets = getattr(result, "assets", None)
    if assets is None:
        raise ValueError("Strict MG cache hit did not return assets")
    levels = tuple(assets.levels)
    return {
        "fine_null_vectors": assets.fine_blocked_v,
        "levels": tuple({
            "level": int(level.level),
            "preconditioned_links": level.preconditioned_links,
            "onsite_pair": level.onsite_pair,
            "null_vectors": level.null_vectors,
        } for level in levels),
        "manifest": dict(result.manifest or {}),
        "evidence": dict(result.evidence or {}),
    }


__all__ = [
    "STRICT_MG_CACHE_FORMAT",
    "STRICT_MG_CACHE_FORMAT_VERSION",
    "STRICT_MG_CACHE_IDENTITY_SCHEMA",
    "inspect_strict_mg_cache",
    "load_strict_mg_cache",
    "make_strict_mg_cache_identity",
    "save_strict_mg_cache",
    "strict_mg_backend_assets",
    "strict_mg_cache_identity_sha256",
    "strict_mg_cache_path",
]
