"""Shared Legacy-MultiGrid cache format for QCU backends.

The cache stores the blocked odd-Schur 33-tensor operator used by the
CUDA/Torch Legacy multigrid path.  KUPL consumes the same tensors through
``kLegacyLayoutOddSchur``; keeping geometry, slicing, and provenance checks in
one place prevents backend-local interpretations from drifting apart.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np


LEGACY_MG_CACHE_SCHEMA = "pyqcu.legacy-mg-cache.v1"
LEGACY_MG_CACHE_LAYOUT = "odd-schur"
LEGACY_MG_ASSET_NAMES = ("lonv", "hnn", "hdg", "sit")
LEGACY_MG_IDENTITY_FIELDS = (
    "lattice",
    "dtype",
    "mass",
    "csw",
    "dof",
    "mg_grid",
    "nv_iters",
    "seed",
)


def _tuple4(values: Sequence[int], name: str) -> tuple[int, int, int, int]:
    result = tuple(int(value) for value in values)
    if len(result) != 4:
        raise ValueError(f"{name} must contain four integers: {result}")
    return result  # type: ignore[return-value]


def canonical_dtype(dtype: Any) -> np.dtype[Any]:
    """Normalize backend dtype spellings to the two cache dtypes."""
    if isinstance(dtype, str):
        aliases = {
            "c64": np.complex64,
            "complex64": np.complex64,
            "float32": np.complex64,
            "c128": np.complex128,
            "complex128": np.complex128,
            "float64": np.complex128,
        }
        try:
            dtype = aliases[dtype.lower()]
        except KeyError as exc:
            raise ValueError(f"unsupported Legacy MG cache dtype {dtype!r}") from exc
    normalized = np.dtype(dtype)
    if normalized not in (np.dtype(np.complex64), np.dtype(np.complex128)):
        raise ValueError(
            f"Legacy MG cache dtype must be complex64 or complex128, got "
            f"{normalized}")
    return normalized


def legacy_mg_geometry(
        lattice: Sequence[int], mg_grid: Sequence[int]
        ) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Return ``(fine_odd, coarse)`` for the canonical cache layout."""
    lattice = _tuple4(lattice, "lattice")
    mg_grid = _tuple4(mg_grid, "mg_grid")
    if any(value <= 0 for value in lattice):
        raise ValueError(f"lattice extents must be positive: {lattice}")
    if lattice[3] % 2:
        raise ValueError(f"odd-Schur cache requires even time extent: {lattice}")
    if any(value <= 0 for value in mg_grid):
        raise ValueError(f"mg_grid extents must be positive: {mg_grid}")
    fine_odd = (*lattice[:3], lattice[3] // 2)
    if any(fine_odd[axis] % mg_grid[axis] for axis in range(4)):
        raise ValueError(
            f"fine odd geometry {fine_odd} is not divisible by mg_grid "
            f"{mg_grid}")
    coarse = tuple(fine_odd[axis] // mg_grid[axis] for axis in range(4))
    return fine_odd, coarse  # type: ignore[return-value]


def legacy_mg_cache_shapes(
        lattice: Sequence[int], mg_grid: Sequence[int],
        dof: Sequence[int]) -> dict[str, tuple[int, ...]]:
    """Return the exact per-transition HDF5 dataset shapes."""
    values = tuple(int(value) for value in dof)
    if len(values) < 2 or any(value <= 0 for value in values):
        raise ValueError(f"dof must contain at least two positive levels: {values}")
    _, coarse = legacy_mg_geometry(lattice, mg_grid)
    # A transition cache always starts from the canonical 12-component fine
    # odd field, while level 0 in the dof list describes that field.  The
    # reusable coarse basis dimension is therefore dof[1].
    if values[0] != 12:
        raise ValueError(
            f"Legacy MG transition cache requires fine dof 12, got {values[0]}")
    out_dof = values[1]
    fine_dof = 12
    return {
        "lonv": (
            out_dof, fine_dof, coarse[0], int(mg_grid[0]),
            coarse[1], int(mg_grid[1]), coarse[2], int(mg_grid[2]),
            coarse[3], int(mg_grid[3]),
        ),
        "hnn": (2, 4, out_dof, out_dof, *coarse),
        "hdg": (2, 2, 6, out_dof, out_dof, *coarse),
        "sit": (out_dof, out_dof, *coarse),
    }


def logical_array_sha256(array: np.ndarray) -> str:
    """Hash values independently of HDF5 storage details."""
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    metadata = {
        "shape": tuple(int(item) for item in value.shape),
        "dtype": str(np.dtype(value.dtype)),
    }
    digest.update(json.dumps(
        metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def logical_array_sha256_chunked(
        array: np.ndarray, *, max_chunk_bytes: int = 64 << 20) -> str:
    """Hash an array view in C order without contiguous full-array copies."""
    if max_chunk_bytes <= 0:
        raise ValueError("max_chunk_bytes must be positive")
    value = np.asarray(array)
    digest = hashlib.sha256()
    metadata = {
        "shape": tuple(int(item) for item in value.shape),
        "dtype": str(np.dtype(value.dtype)),
    }
    digest.update(json.dumps(
        metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    if value.ndim == 0:
        digest.update(np.ascontiguousarray(value[()]).tobytes())
        return digest.hexdigest()
    shape = tuple(int(item) for item in value.shape)
    chunk = list(shape)
    target_elements = max(
        1, int(max_chunk_bytes) // int(value.dtype.itemsize))
    while math.prod(chunk) > target_elements:
        oversized = math.ceil(math.prod(chunk) / target_elements)
        changed = False
        for axis, extent in enumerate(chunk):
            if extent <= 1:
                continue
            chunk[axis] = max(1, math.ceil(extent / oversized))
            changed = True
            if math.prod(chunk) <= target_elements:
                break
        if not changed:
            break
    starts = [range(0, shape[axis], chunk[axis]) for axis in range(len(shape))]
    for origin in product(*starts):
        selection = tuple(
            slice(origin[axis], min(
                origin[axis] + chunk[axis], shape[axis]))
            for axis in range(len(shape)))
        digest.update(np.ascontiguousarray(value[selection]).tobytes())
    return digest.hexdigest()


def logical_dataset_sha256(
        dataset: Any, *, max_chunk_bytes: int = 64 << 20) -> str:
    """Stream a dataset hash in bounded multidimensional chunks."""
    if max_chunk_bytes <= 0:
        raise ValueError("max_chunk_bytes must be positive")
    digest = hashlib.sha256()
    metadata = {
        "shape": tuple(int(value) for value in dataset.shape),
        "dtype": str(np.dtype(dataset.dtype)),
    }
    digest.update(json.dumps(
        metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    if dataset.ndim == 0:
        digest.update(np.ascontiguousarray(dataset[()]).tobytes())
        return digest.hexdigest()

    shape = tuple(int(value) for value in dataset.shape)
    chunk = list(shape)
    element_bytes = int(np.dtype(dataset.dtype).itemsize)
    target_elements = max(1, int(max_chunk_bytes) // element_bytes)
    while math.prod(chunk) > target_elements:
        changed = False
        oversized = math.ceil(math.prod(chunk) / target_elements)
        for axis, extent in enumerate(chunk):
            if extent <= 1:
                continue
            chunk[axis] = max(1, math.ceil(extent / oversized))
            changed = True
            if math.prod(chunk) <= target_elements:
                break
        if not changed:
            break
    starts = [range(0, shape[axis], chunk[axis]) for axis in range(len(shape))]
    for origin in product(*starts):
        selection = tuple(
            slice(origin[axis], min(
                origin[axis] + chunk[axis], shape[axis]))
            for axis in range(len(shape)))
        digest.update(np.ascontiguousarray(dataset[selection]).tobytes())
    return digest.hexdigest()


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_manifest_path(path: str | os.PathLike[str]) -> Path:
    return Path(path).with_suffix(".json")


def legacy_mg_cache_path(
        identity: Mapping[str, Any], cache_dir: str | os.PathLike[str]) -> Path:
    digest = identity_sha256(_normalize_identity(identity))[:24]
    return Path(cache_dir) / f"legacy_mg_{digest}.h5"


def _normalize_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    missing = [name for name in LEGACY_MG_IDENTITY_FIELDS if name not in identity]
    if missing:
        raise ValueError(f"Legacy MG cache identity is missing {missing}")
    lattice = _tuple4(identity["lattice"], "lattice")
    mg_grid = _tuple4(identity["mg_grid"], "mg_grid")
    dof = tuple(int(value) for value in identity["dof"])
    legacy_mg_geometry(lattice, mg_grid)
    mass = float(identity["mass"])
    csw = float(identity["csw"])
    if not math.isfinite(mass) or not math.isfinite(csw):
        raise ValueError("Legacy MG cache mass and csw must be finite")
    result: dict[str, Any] = {
        "lattice": list(lattice),
        "dtype": str(canonical_dtype(identity["dtype"])),
        "mass": mass,
        "csw": csw,
        "dof": list(dof),
        "mg_grid": list(mg_grid),
        "nv_iters": int(identity["nv_iters"]),
        "seed": int(identity["seed"]),
    }
    if int(result["nv_iters"]) <= 0:
        raise ValueError("Legacy MG cache nv_iters must be positive")
    for key in ("gauge_canonical_sha256", "gauge_file_sha256", "gauge_sha256"):
        value = identity.get(key)
        if value is None:
            continue
        text = str(value).lower()
        if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
            raise ValueError(f"{key} must be a lowercase SHA256 digest")
        result[key] = text
    if not any(key in result for key in (
            "gauge_canonical_sha256", "gauge_file_sha256", "gauge_sha256")):
        raise ValueError("Legacy MG cache identity requires a gauge SHA256")
    return result


def legacy_mg_cache_identity(**values: Any) -> dict[str, Any]:
    """Build a normalized cross-backend cache identity."""
    return _normalize_identity(values)


def identity_sha256(identity: Mapping[str, Any]) -> str:
    normalized = _normalize_identity(identity)
    payload = json.dumps(
        normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_legacy_mg_cache_manifest(
        cache_path: str | os.PathLike[str], identity: Mapping[str, Any], *,
        datasets: Mapping[str, np.ndarray] | None, setup_method: str,
        setup_backend: str, setup_validation: Mapping[str, Any] | None = None,
        source_metadata: Mapping[str, Any] | None = None,
        setup_seconds: float | None = None) -> dict[str, Any]:
    """Build the sidecar consumed by every QCU Legacy backend.

    ``datasets=None`` hashes the already-written HDF5 datasets in streaming
    order.  This lets a distributed MPI builder publish ``lonv`` directly
    without first gathering the full tensor on rank zero.
    """
    path = Path(cache_path)
    normalized = _normalize_identity(identity)
    expected_shapes = legacy_mg_cache_shapes(
        normalized["lattice"], normalized["mg_grid"], normalized["dof"])
    records: dict[str, dict[str, Any]] = {}
    if datasets is None:
        import h5py

        with h5py.File(path, "r") as handle:
            missing = sorted(set(LEGACY_MG_ASSET_NAMES).difference(handle.keys()))
            if missing:
                raise ValueError(
                    f"Legacy MG cache {path} is missing datasets {missing}")
            for name in LEGACY_MG_ASSET_NAMES:
                dataset = handle[name]
                if tuple(dataset.shape) != expected_shapes[name]:
                    raise ValueError(
                        f"{name} shape {tuple(dataset.shape)} != "
                        f"{expected_shapes[name]}")
                if np.dtype(dataset.dtype) != np.dtype(normalized["dtype"]):
                    raise ValueError(
                        f"{name} dtype {dataset.dtype} != {normalized['dtype']}")
                records[name] = {
                    "shape": list(dataset.shape),
                    "dtype": str(dataset.dtype),
                    "sha256": logical_dataset_sha256(dataset),
                }
    else:
        for name in LEGACY_MG_ASSET_NAMES:
            value = np.asarray(datasets[name])
            if tuple(value.shape) != expected_shapes[name]:
                raise ValueError(
                    f"{name} shape {tuple(value.shape)} != {expected_shapes[name]}")
            if np.dtype(value.dtype) != np.dtype(normalized["dtype"]):
                raise ValueError(
                    f"{name} dtype {value.dtype} != {normalized['dtype']}")
            records[name] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": logical_array_sha256(value),
            }
    report: dict[str, Any] = {
        "status": "PASS",
        "schema_version": LEGACY_MG_CACHE_SCHEMA,
        "layout": LEGACY_MG_CACHE_LAYOUT,
        "identity": normalized,
        "identity_sha256": identity_sha256(normalized),
        "assets": records,
        "setup_method": str(setup_method),
        "setup_backend": str(setup_backend),
    }
    # Flat aliases keep reports consumed by older tooling readable.  The
    # canonical source of truth remains ``identity`` above.
    for name in LEGACY_MG_IDENTITY_FIELDS:
        report[name] = normalized[name]
    if "gauge_file_sha256" in normalized:
        report["gauge_sha256"] = normalized["gauge_file_sha256"]
    if setup_validation is not None:
        report["setup_validation"] = dict(setup_validation)
    if source_metadata is not None:
        report["source_metadata"] = dict(source_metadata)
    if setup_seconds is not None:
        report["setup_seconds"] = float(setup_seconds)
    if path.is_file():
        digest = file_sha256(path)
        size = path.stat().st_size
        report["file_sha256"] = digest
        report["file_size_bytes"] = size
        report["output_sha256"] = digest
        report["output_size_bytes"] = size
    return report


def write_legacy_mg_cache_manifest(
        cache_path: str | os.PathLike[str], manifest: Mapping[str, Any]) -> Path:
    path = cache_manifest_path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)
    return path


def save_legacy_mg_cache(
        cache_path: str | os.PathLike[str],
        datasets: Mapping[str, np.ndarray], *, identity: Mapping[str, Any],
        setup_method: str, setup_backend: str,
        setup_validation: Mapping[str, Any] | None = None,
        source_metadata: Mapping[str, Any] | None = None,
        setup_seconds: float | None = None) -> dict[str, Any]:
    """Write canonical tensors and the unified sidecar manifest."""
    path = Path(cache_path)
    write_legacy_mg_assets(path, datasets, identity=identity)
    manifest = build_legacy_mg_cache_manifest(
        path, identity, datasets=datasets, setup_method=setup_method,
        setup_backend=setup_backend, setup_validation=setup_validation,
        source_metadata=source_metadata, setup_seconds=setup_seconds)
    write_legacy_mg_cache_manifest(path, manifest)
    return manifest


def write_legacy_mg_assets(
        cache_path: str | os.PathLike[str],
        datasets: Mapping[str, np.ndarray], *,
        identity: Mapping[str, Any] | None = None) -> Path:
    """Atomically write the single canonical HDF5 tensor layout."""
    import h5py

    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    attrs: dict[str, Any] = {
        "pyqcu_cache_schema": LEGACY_MG_CACHE_SCHEMA,
        "pyqcu_cache_layout": LEGACY_MG_CACHE_LAYOUT,
    }
    if identity is not None:
        normalized = _normalize_identity(identity)
        attrs["pyqcu_cache_identity_sha256"] = identity_sha256(normalized)
    try:
        with h5py.File(temporary, "w") as handle:
            for key, value in attrs.items():
                handle.attrs[key] = value
            for name in LEGACY_MG_ASSET_NAMES:
                handle.create_dataset(name, data=np.asarray(datasets[name]))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def write_legacy_mg_asset_subset(
        cache_path: str | os.PathLike[str],
        datasets: Mapping[str, np.ndarray], *,
        identity: Mapping[str, Any]) -> Path:
    """Append selected tensors to an existing staged cache.

    The MPI builder creates ``lonv`` collectively first, then rank zero appends
    the smaller ``hnn/hdg/sit`` tensors.  Existing datasets are rejected so a
    partially initialized file cannot silently change its logical contents.
    """
    import h5py

    path = Path(cache_path)
    if not path.is_file():
        raise FileNotFoundError(f"staged Legacy MG cache not found: {path}")
    normalized = _normalize_identity(identity)
    expected_shapes = legacy_mg_cache_shapes(
        normalized["lattice"], normalized["mg_grid"], normalized["dof"])
    unknown = sorted(set(datasets).difference(LEGACY_MG_ASSET_NAMES))
    if unknown:
        raise ValueError(f"unknown Legacy MG datasets {unknown}")
    with h5py.File(path, "r+") as handle:
        handle.attrs["pyqcu_cache_schema"] = LEGACY_MG_CACHE_SCHEMA
        handle.attrs["pyqcu_cache_layout"] = LEGACY_MG_CACHE_LAYOUT
        handle.attrs["pyqcu_cache_identity_sha256"] = identity_sha256(normalized)
        for name in LEGACY_MG_ASSET_NAMES:
            if name not in datasets:
                continue
            value = np.asarray(datasets[name])
            if tuple(value.shape) != expected_shapes[name]:
                raise ValueError(
                    f"{name} shape {tuple(value.shape)} != {expected_shapes[name]}")
            if np.dtype(value.dtype) != np.dtype(normalized["dtype"]):
                raise ValueError(
                    f"{name} dtype {value.dtype} != {normalized['dtype']}")
            if name in handle:
                raise ValueError(f"staged Legacy MG cache already has {name}")
            handle.create_dataset(name, data=value)
    return path


def load_legacy_mg_cache_slab(
        cache_path: str | os.PathLike[str], dtype: Any, *,
        starts: Sequence[int] | None = None,
        local_shape: Sequence[int] | None = None,
        names: Sequence[str] = LEGACY_MG_ASSET_NAMES,
        ) -> dict[str, np.ndarray]:
    """Load a global cache or one rank-local coarse hyperslab."""
    import h5py

    path = Path(cache_path)
    selected_dtype = canonical_dtype(dtype)
    if (starts is None) != (local_shape is None):
        raise ValueError("starts and local_shape must be provided together")
    start_values = None if starts is None else _tuple4(starts, "starts")
    shape_values = None if local_shape is None else _tuple4(
        local_shape, "local_shape")
    selected_names = tuple(str(name) for name in names)
    unknown = sorted(set(selected_names).difference(LEGACY_MG_ASSET_NAMES))
    if unknown:
        raise ValueError(f"unknown Legacy MG datasets {unknown}")
    values: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as handle:
        missing = sorted(set(selected_names).difference(handle.keys()))
        if missing:
            raise ValueError(
                f"Legacy MG cache {path} is missing datasets {missing}")
        for name in selected_names:
            dataset = handle[name]
            if start_values is None:
                selected = dataset[...]
            else:
                if name == "lonv":
                    axes = (2, 4, 6, 8)
                elif name == "hnn":
                    axes = (4, 5, 6, 7)
                elif name == "hdg":
                    axes = (5, 6, 7, 8)
                else:
                    axes = (2, 3, 4, 5)
                selection = [slice(None)] * dataset.ndim
                for axis, h5_axis in enumerate(axes):
                    selection[h5_axis] = slice(
                        start_values[axis], start_values[axis] + shape_values[axis])
                selected = dataset[tuple(selection)]
            values[name] = np.ascontiguousarray(selected, dtype=selected_dtype)
    return values


def _manifest_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    nested = manifest.get("identity")
    if isinstance(nested, Mapping):
        return dict(nested)
    legacy = {name: manifest[name] for name in LEGACY_MG_IDENTITY_FIELDS
              if name in manifest}
    gauge = manifest.get("gauge_canonical_sha256")
    if gauge is None:
        gauge = manifest.get("gauge_file_sha256", manifest.get("gauge_sha256"))
    if gauge is not None:
        legacy["gauge_sha256"] = gauge
    return legacy


def validate_legacy_mg_cache(
        cache_path: str | os.PathLike[str],
        expected_identity: Mapping[str, Any], *,
        verify_file_sha256: bool = False,
        verify_dataset_sha256: bool = True) -> dict[str, Any]:
    """Fail closed when a cache is not the requested physical/geometric asset."""
    import h5py

    path = Path(cache_path)
    if not path.is_file():
        raise FileNotFoundError(f"Legacy MG cache not found: {path}")
    manifest_path = cache_manifest_path(path)
    if not manifest_path.is_file():
        raise ValueError(f"Legacy MG cache manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS":
        raise ValueError(f"Legacy MG cache manifest status is not PASS: {manifest_path}")
    schema = manifest.get("schema_version")
    if schema not in (None, LEGACY_MG_CACHE_SCHEMA):
        raise ValueError(f"unsupported Legacy MG cache schema {schema!r}")
    layout = manifest.get("layout")
    if layout not in (None, LEGACY_MG_CACHE_LAYOUT):
        raise ValueError(f"unsupported Legacy MG cache layout {layout!r}")

    expected = _normalize_identity(expected_identity)
    actual = _normalize_identity(_manifest_identity(manifest))
    expected_canonical = expected.get("gauge_canonical_sha256")
    actual_canonical = actual.get("gauge_canonical_sha256")
    if expected_canonical is not None and actual_canonical is not None:
        if actual_canonical != expected_canonical:
            raise ValueError(
                f"Legacy MG cache gauge_canonical_sha256 {actual_canonical} "
                f"!= expected {expected_canonical}")
    else:
        expected_file = expected.get(
            "gauge_file_sha256", expected.get("gauge_sha256"))
        actual_file = actual.get(
            "gauge_file_sha256", actual.get("gauge_sha256"))
        if expected_file is None or actual_file is None:
            raise ValueError("Legacy MG cache gauge digest is incomplete")
        if actual_file != expected_file:
            raise ValueError(
                f"Legacy MG cache gauge_file_sha256 {actual_file} != "
                f"expected {expected_file}")
    for key in LEGACY_MG_IDENTITY_FIELDS:
        if actual[key] != expected[key]:
            raise ValueError(
                f"Legacy MG cache {key} {actual[key]!r} != expected "
                f"{expected[key]!r}")

    expected_shapes = legacy_mg_cache_shapes(
        expected["lattice"], expected["mg_grid"], expected["dof"])
    records = manifest.get("assets")
    with h5py.File(path, "r") as handle:
        root_schema = handle.attrs.get("pyqcu_cache_schema")
        root_layout = handle.attrs.get("pyqcu_cache_layout")
        if root_schema is not None and str(root_schema) != LEGACY_MG_CACHE_SCHEMA:
            raise ValueError(f"unsupported HDF5 cache schema {root_schema!r}")
        if root_layout is not None and str(root_layout) != LEGACY_MG_CACHE_LAYOUT:
            raise ValueError(f"unsupported HDF5 cache layout {root_layout!r}")
        for name in LEGACY_MG_ASSET_NAMES:
            if name not in handle:
                raise ValueError(f"Legacy MG cache is missing dataset {name}")
            dataset = handle[name]
            if tuple(dataset.shape) != expected_shapes[name]:
                raise ValueError(
                    f"Legacy MG cache {name} shape {tuple(dataset.shape)} != "
                    f"{expected_shapes[name]}")
            if np.dtype(dataset.dtype) != np.dtype(expected["dtype"]):
                raise ValueError(
                    f"Legacy MG cache {name} dtype {dataset.dtype} != "
                    f"{expected['dtype']}")
            if (verify_dataset_sha256 and isinstance(records, Mapping) and
                    name in records):
                expected_digest = records[name].get("sha256")
                if expected_digest and logical_dataset_sha256(dataset) != expected_digest:
                    raise ValueError(f"Legacy MG cache {name} digest mismatch")

    if verify_file_sha256:
        expected_file_digest = manifest.get(
            "file_sha256", manifest.get("output_sha256"))
        if expected_file_digest and file_sha256(path) != expected_file_digest:
            raise ValueError("Legacy MG cache file digest mismatch")
    return manifest


__all__ = [
    "LEGACY_MG_ASSET_NAMES",
    "LEGACY_MG_CACHE_LAYOUT",
    "LEGACY_MG_CACHE_SCHEMA",
    "build_legacy_mg_cache_manifest",
    "cache_manifest_path",
    "canonical_dtype",
    "file_sha256",
    "identity_sha256",
    "legacy_mg_cache_identity",
    "legacy_mg_cache_path",
    "legacy_mg_cache_shapes",
    "legacy_mg_geometry",
    "load_legacy_mg_cache_slab",
    "logical_array_sha256",
    "logical_array_sha256_chunked",
    "logical_dataset_sha256",
    "save_legacy_mg_cache",
    "validate_legacy_mg_cache",
    "write_legacy_mg_asset_subset",
    "write_legacy_mg_assets",
    "write_legacy_mg_cache_manifest",
]
