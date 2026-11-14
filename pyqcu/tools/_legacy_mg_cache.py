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
from collections.abc import Iterator, Mapping, Sequence
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np


LEGACY_MG_CACHE_SCHEMA = "pyqcu.legacy-mg-cache.v1"
LEGACY_MG_CACHE_LAYOUT = "odd-schur"
# Multi-rank writers publish the same odd-Schur tensors as one HDF5 file per
# rank plus a root manifest.  The layout name keeps every backend on one
# interpretation: ``global_shapes`` describes the single-file cache that the
# rank files tile, and each rank file stores that rank's coarse hyperslab.
LEGACY_MG_RANKED_LAYOUT = "odd-schur-ranked"
LEGACY_MG_RANK_DIR_SUFFIX = ".d"
LEGACY_MG_RANK_FILE = "rank_{rank:05d}.h5"
LEGACY_MG_ASSET_NAMES = ("lonv", "hnn", "hdg", "sit")
#: Coarse axes of every cached tensor in the canonical HDF5 layout.
LEGACY_MG_ASSET_COARSE_AXES = {
    "lonv": (2, 4, 6, 8),
    "hnn": (4, 5, 6, 7),
    "hdg": (5, 6, 7, 8),
    "sit": (2, 3, 4, 5),
}
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


def _logical_chunk_selections(
        shape: Sequence[int], target_elements: int) -> Iterator[tuple[Any, ...]]:
    """Yield bounded selections in strict C-order without reordering blocks.

    Splitting multiple leading axes into a Cartesian grid interleaves blocks
    that are not adjacent in C order.  Keep every prefix coordinate fixed and
    split only one axis, leaving all trailing axes complete.
    """
    extents = tuple(int(extent) for extent in shape)
    if any(extent == 0 for extent in extents):
        return

    trailing = 1
    split_axis = len(extents) - 1
    split_trailing = 1
    for axis in range(len(extents) - 1, -1, -1):
        if trailing > target_elements:
            break
        split_axis = axis
        split_trailing = trailing
        trailing *= extents[axis]

    chunk_extent = min(
        extents[split_axis], max(1, target_elements // split_trailing))
    prefix_ranges = tuple(range(extent) for extent in extents[:split_axis])
    for prefix in product(*prefix_ranges):
        for start in range(0, extents[split_axis], chunk_extent):
            selection: list[Any] = [int(index) for index in prefix]
            selection.append(slice(
                start, min(start + chunk_extent, extents[split_axis])))
            selection.extend(slice(None) for _ in extents[split_axis + 1:])
            yield tuple(selection)


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
    target_elements = max(
        1, int(max_chunk_bytes) // int(value.dtype.itemsize))
    for selection in _logical_chunk_selections(shape, target_elements):
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
    element_bytes = int(np.dtype(dataset.dtype).itemsize)
    target_elements = max(1, int(max_chunk_bytes) // element_bytes)
    for selection in _logical_chunk_selections(shape, target_elements):
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


def legacy_mg_cache_stem(cache_path: str | os.PathLike[str]) -> Path:
    """Return the cache path without a trailing ``.h5``/``.hdf5`` suffix."""
    path = Path(cache_path)
    if path.suffix.lower() in (".h5", ".hdf5"):
        return path.with_suffix("")
    return path


def legacy_mg_rank_dir(cache_path: str | os.PathLike[str]) -> Path:
    """Return the per-rank directory of a multi-rank cache."""
    stem = legacy_mg_cache_stem(cache_path)
    return stem.with_name(stem.name + LEGACY_MG_RANK_DIR_SUFFIX)


def legacy_mg_rank_cache_path(
        cache_path: str | os.PathLike[str], rank: int) -> Path:
    """Return the rank-local HDF5 file inside the ranked cache directory."""
    if int(rank) < 0:
        raise ValueError(f"rank must be non-negative: {rank}")
    return legacy_mg_rank_dir(cache_path) / LEGACY_MG_RANK_FILE.format(
        rank=int(rank))


def legacy_mg_local_shapes(
        shapes: Mapping[str, Sequence[int]],
        local_coarse: Sequence[int]) -> dict[str, tuple[int, ...]]:
    """Replace the coarse extents of the global shapes by a local slab."""
    coarse = _tuple4(local_coarse, "local_coarse")
    if any(extent <= 0 for extent in coarse):
        raise ValueError(f"local coarse extents must be positive: {coarse}")
    local: dict[str, tuple[int, ...]] = {}
    for name in LEGACY_MG_ASSET_NAMES:
        if name not in shapes:
            continue
        global_shape = tuple(int(value) for value in shapes[name])
        axes = LEGACY_MG_ASSET_COARSE_AXES[name]
        if any(global_shape[axis] < coarse[index]
               for index, axis in enumerate(axes)):
            raise ValueError(
                f"{name} coarse extents {tuple(global_shape[axis] for axis in axes)} "
                f"are smaller than the local slice {coarse}")
        selected = list(global_shape)
        for index, axis in enumerate(axes):
            selected[axis] = int(coarse[index])
        local[name] = tuple(selected)
    return local


def _asset_coarse_slice(
        name: str, starts: Sequence[int], extent: Sequence[int],
        ndim: int) -> tuple[slice, ...]:
    """Full dataset selection (all other axes complete, coarse axes cut)."""
    axes = LEGACY_MG_ASSET_COARSE_AXES[name]
    if ndim <= max(axes):
        raise ValueError(f"{name} rank slab has too few axes: {ndim}")
    selection = [slice(None)] * int(ndim)
    for index, axis in enumerate(axes):
        start = int(starts[index])
        selection[axis] = slice(start, start + int(extent[index]))
    return tuple(selection)


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


def _dataset_records(
        handle: Any, local_shapes: Mapping[str, Sequence[int]],
        dtype: np.dtype[Any], *, complete: bool = True) -> dict[str, dict[str, Any]]:
    """Describe every dataset an already open rank file must contain."""
    names = tuple(name for name in LEGACY_MG_ASSET_NAMES if name in handle)
    if complete:
        missing = sorted(set(LEGACY_MG_ASSET_NAMES).difference(handle.keys()))
        if missing:
            raise ValueError(f"rank cache is missing datasets {missing}")
        names = LEGACY_MG_ASSET_NAMES
    records: dict[str, dict[str, Any]] = {}
    for name in names:
        dataset = handle[name]
        expected = tuple(int(value) for value in local_shapes[name])
        if tuple(int(value) for value in dataset.shape) != expected:
            raise ValueError(
                f"rank cache {name} shape {tuple(dataset.shape)} != {expected}")
        if np.dtype(dataset.dtype) != dtype:
            raise ValueError(
                f"rank cache {name} dtype {dataset.dtype} != {dtype}")
        records[name] = {
            "shape": list(expected),
            "dtype": str(np.dtype(dataset.dtype)),
            "sha256": logical_dataset_sha256(dataset),
        }
    return records


def write_legacy_mg_rank_slice(
        cache_path: str | os.PathLike[str],
        datasets: Mapping[str, np.ndarray], *,
        rank: int, coordinate: Sequence[int], process_grid: Sequence[int],
        starts: Sequence[int], local_coarse: Sequence[int],
        global_shapes: Mapping[str, Sequence[int]],
        identity: Mapping[str, Any], mode: str = "w") -> dict[str, Any]:
    """Write one rank's coarse hyperslab into ``<cache>.d/rank_%05d.h5``.

    ``mode="w"`` creates the file atomically (used for the null vectors that
    neighbouring ranks must read before the stencil stage) and ``mode="r+"``
    appends the remaining tensors.  Every dataset is shape/dtype checked
    against the global cache geometry, so a rank file can never silently
    describe a different slab.
    """
    import h5py

    normalized = _normalize_identity(identity)
    dtype = np.dtype(normalized["dtype"])
    coordinates = _tuple4(coordinate, "coordinate")
    grid = _tuple4(process_grid, "process_grid")
    coarse_starts = _tuple4(starts, "starts")
    coarse = _tuple4(local_coarse, "local_coarse")
    if any(value < 0 for value in coordinates):
        raise ValueError(f"rank coordinate must be non-negative: {coordinates}")
    if any(value <= 0 for value in grid):
        raise ValueError(f"process grid must be positive: {grid}")
    if any(value < 0 for value in coarse_starts):
        raise ValueError(f"coarse starts must be non-negative: {coarse_starts}")
    if int(np.prod(grid)) <= int(rank):
        raise ValueError(f"rank {rank} is outside process grid {grid}")
    expected_local = legacy_mg_local_shapes(global_shapes, coarse)
    unknown = sorted(set(datasets).difference(LEGACY_MG_ASSET_NAMES))
    if unknown:
        raise ValueError(f"unknown Legacy MG datasets {unknown}")
    if not datasets:
        raise ValueError("no Legacy MG datasets to write")

    path = legacy_mg_rank_cache_path(cache_path, rank)
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode not in ("w", "r+"):
        raise ValueError(f"unsupported rank cache write mode {mode!r}")
    if mode == "r+" and not path.is_file():
        raise FileNotFoundError(f"rank cache {path} must exist for append")
    attributes: dict[str, Any] = {
        "pyqcu_cache_schema": LEGACY_MG_CACHE_SCHEMA,
        "pyqcu_cache_layout": LEGACY_MG_RANKED_LAYOUT,
        "pyqcu_cache_identity_sha256": identity_sha256(normalized),
        "pyqcu_cache_rank": int(rank),
        "pyqcu_cache_process_grid": list(grid),
        "pyqcu_cache_coordinate": list(coordinates),
        "pyqcu_cache_coarse_starts": list(coarse_starts),
        "pyqcu_cache_local_coarse": list(coarse),
        "pyqcu_cache_global_shapes": json.dumps(
            {name: list(global_shapes[name]) for name in LEGACY_MG_ASSET_NAMES
             if name in global_shapes}, sort_keys=True),
    }

    def populate(handle: Any) -> None:
        for key, value in attributes.items():
            handle.attrs[key] = value
        for name in LEGACY_MG_ASSET_NAMES:
            if name not in datasets:
                continue
            value = np.asarray(datasets[name])
            if tuple(int(item) for item in value.shape) != expected_local[name]:
                raise ValueError(
                    f"{name} local shape {tuple(value.shape)} != "
                    f"{expected_local[name]} for coarse slab {coarse}")
            if np.dtype(value.dtype) != dtype:
                raise ValueError(
                    f"{name} dtype {value.dtype} != {dtype}")
            if name in handle:
                raise ValueError(f"rank cache {path} already has {name}")
            handle.create_dataset(name, data=np.ascontiguousarray(value))

    if mode == "w":
        temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
        try:
            with h5py.File(temporary, "w") as handle:
                populate(handle)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
    else:
        with h5py.File(path, "r+") as handle:
            populate(handle)

    with h5py.File(path, "r") as handle:
        records = _dataset_records(
            handle, expected_local, dtype,
            complete=set(LEGACY_MG_ASSET_NAMES).issubset(datasets))
    return {
        "rank": int(rank),
        "coordinate": list(coordinates),
        "process_grid": list(grid),
        "starts": list(coarse_starts),
        "local_coarse": list(coarse),
        "path": path.name,
        "directory": str(path.parent),
        "datasets": records,
        "file_size_bytes": path.stat().st_size,
        "file_sha256": file_sha256(path),
    }


def build_legacy_mg_ranked_cache_manifest(
        cache_path: str | os.PathLike[str], identity: Mapping[str, Any], *,
        process_grid: Sequence[int], ranks: Sequence[Mapping[str, Any]],
        global_shapes: Mapping[str, Sequence[int]],
        local_shapes: Mapping[str, Sequence[int]],
        global_coarse: Sequence[int], local_coarse: Sequence[int],
        setup_method: str, setup_backend: str,
        mg_config: Mapping[str, Any] | None = None,
        setup_validation: Mapping[str, Any] | None = None,
        source_metadata: Mapping[str, Any] | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
        setup_seconds: float | None = None,
        created_utc: str | None = None,
        assets: Sequence[str] = LEGACY_MG_ASSET_NAMES,
        status: str = "PASS") -> dict[str, Any]:
    """Build the root manifest of a multi-rank Legacy MG cache.

    ``assets``/``status`` exist for the two-phase writer: ``lonv`` is published
    first (``status="BUILDING"``) because every rank needs its neighbours' null
    vectors before it can build the local stencil, and the manifest is rewritten
    with ``status="PASS"`` once all four tensors are complete.  Readers refuse
    a manifest that is not ``PASS``.
    """
    normalized = _normalize_identity(identity)
    grid = _tuple4(process_grid, "process_grid")
    coarse = _tuple4(global_coarse, "global_coarse")
    local = _tuple4(local_coarse, "local_coarse")
    declared = tuple(str(name) for name in assets)
    unknown = sorted(set(declared).difference(LEGACY_MG_ASSET_NAMES))
    if unknown:
        raise ValueError(f"unknown Legacy MG datasets {unknown}")
    if not declared:
        raise ValueError("a ranked cache manifest needs at least one asset")
    if status not in ("PASS", "BUILDING"):
        raise ValueError(f"unsupported ranked cache status {status!r}")
    if len(ranks) != int(np.prod(grid)):
        raise ValueError(
            f"ranked cache has {len(ranks)} rank records for grid {grid}")
    shapes = {
        name: tuple(int(value) for value in global_shapes[name])
        for name in LEGACY_MG_ASSET_NAMES
    }
    expected = legacy_mg_cache_shapes(
        normalized["lattice"], normalized["mg_grid"], normalized["dof"])
    for name in declared:
        if shapes[name] != expected[name]:
            raise ValueError(
                f"{name} global shape {shapes[name]} != canonical "
                f"{expected[name]}")
    local_expected = legacy_mg_local_shapes(shapes, local)
    for name in declared:
        if tuple(int(value) for value in local_shapes[name]) != \
                local_expected[name]:
            raise ValueError(
                f"{name} local shape {tuple(local_shapes[name])} != "
                f"{local_expected[name]}")
    ordered = sorted(
        (dict(record) for record in ranks), key=lambda item: int(item["rank"]))
    for index, record in enumerate(ordered):
        if int(record["rank"]) != index:
            raise ValueError(
                f"ranked cache records must be contiguous: got rank "
                f"{record['rank']} at position {index}")
        if tuple(int(value) for value in record["local_coarse"]) != local:
            raise ValueError(
                f"rank {index} local coarse slab {record['local_coarse']} != "
                f"{local}")
        for name in declared:
            if tuple(int(value) for value in record["datasets"][name]["shape"]) \
                    != local_expected[name]:
                raise ValueError(
                    f"rank {index} {name} slab shape "
                    f"{record['datasets'][name]['shape']} != "
                    f"{local_expected[name]}")
    _check_rank_tiling(ordered, coarse)

    report: dict[str, Any] = {
        "status": status,
        "schema_version": LEGACY_MG_CACHE_SCHEMA,
        "layout": LEGACY_MG_RANKED_LAYOUT,
        "identity": normalized,
        "identity_sha256": identity_sha256(normalized),
        "rank_count": len(ordered),
        "assets": list(declared),
        "process_grid": list(grid),
        "coarse": {"global": list(coarse), "local": list(local)},
        "global_coarse": list(coarse),
        "local_coarse": list(local),
        "global_shapes": {name: list(shapes[name])
                          for name in LEGACY_MG_ASSET_NAMES},
        "local_shapes": {name: list(local_expected[name])
                         for name in LEGACY_MG_ASSET_NAMES},
        "rank_directory": legacy_mg_rank_dir(cache_path).name,
        "ranks": ordered,
        "setup_method": str(setup_method),
        "setup_backend": str(setup_backend),
        "created_utc": (created_utc if created_utc is not None else
                        _utc_timestamp()),
    }
    for name in LEGACY_MG_IDENTITY_FIELDS:
        report[name] = normalized[name]
    if "gauge_file_sha256" in normalized:
        report["gauge_sha256"] = normalized["gauge_file_sha256"]
    if mg_config is not None:
        # ``levels``/``restart`` are solve parameters rather than cache
        # identity fields, but a reuse run must be able to prove it asked for
        # the same multigrid setup that produced the cache.
        report["mg"] = dict(mg_config)
    if setup_validation is not None:
        report["setup_validation"] = dict(setup_validation)
    if source_metadata is not None:
        report["source_metadata"] = dict(source_metadata)
    if extra_metadata is not None:
        report["metadata"] = dict(extra_metadata)
    if setup_seconds is not None:
        report["setup_seconds"] = float(setup_seconds)
    total_bytes = sum(int(record.get("file_size_bytes", 0)) for record in ordered)
    report["rank_file_bytes"] = total_bytes
    # The digest of the root container is *not* recorded here: a manifest that
    # is built before (or as part of) ``write_legacy_mg_ranked_cache`` would
    # otherwise carry the digest of the previous revision of the file it is
    # about to overwrite, and every ``verify_file_sha256=True`` read would fail
    # on a freshly exported cache.  ``write_legacy_mg_ranked_cache`` stamps the
    # digest of the file it actually publishes into the JSON sidecar, which is
    # the authoritative copy read by ``read_legacy_mg_ranked_manifest``.
    return report


def _check_rank_tiling(
        records: Sequence[Mapping[str, Any]],
        global_coarse: Sequence[int]) -> None:
    """Fail closed unless the rank slabs tile the global coarse box exactly."""
    coarse = tuple(int(value) for value in global_coarse)
    covered = np.zeros(coarse, dtype=np.int8)
    for record in records:
        starts = tuple(int(value) for value in record["starts"])
        shape = tuple(int(value) for value in record["local_coarse"])
        if any(starts[axis] + shape[axis] > coarse[axis] for axis in range(4)):
            raise ValueError(
                f"rank {record['rank']} slab {starts}+{shape} exceeds the "
                f"global coarse box {coarse}")
        window = covered[
            starts[0]:starts[0] + shape[0], starts[1]:starts[1] + shape[1],
            starts[2]:starts[2] + shape[2], starts[3]:starts[3] + shape[3]]
        if np.any(window):
            raise ValueError(
                f"rank {record['rank']} slab {starts}+{shape} overlaps another "
                f"rank")
        window[...] = 1
    missing = int(covered.size - int(covered.sum()))
    if missing:
        raise ValueError(
            f"ranked cache leaves {missing} of {covered.size} coarse sites "
            f"uncovered for {coarse}")


def write_legacy_mg_ranked_cache(
        cache_path: str | os.PathLike[str],
        manifest: Mapping[str, Any]) -> tuple[Path, Path]:
    """Publish the root manifest (HDF5 attributes + JSON sidecar).

    The embedded copy carries every field except the digest of the container
    itself (a file cannot contain its own SHA256).  The sidecar is written
    last, so it always describes the exact bytes that were just published and
    ``validate_legacy_mg_cache(..., verify_file_sha256=True)`` succeeds on a
    fresh export.
    """
    import h5py

    root = Path(cache_path)
    root.parent.mkdir(parents=True, exist_ok=True)
    published = dict(manifest)
    for key in ("file_sha256", "file_size_bytes", "output_sha256",
                "output_size_bytes"):
        published.pop(key, None)
    payload = json.dumps(
        published, indent=2, sort_keys=True, ensure_ascii=False,
        allow_nan=False)
    attributes = {
        "pyqcu_cache_schema": LEGACY_MG_CACHE_SCHEMA,
        "pyqcu_cache_layout": LEGACY_MG_RANKED_LAYOUT,
        "pyqcu_cache_status": str(published.get("status", "")),
        "pyqcu_cache_identity_sha256": str(manifest["identity_sha256"]),
        "pyqcu_cache_rank_count": int(manifest["rank_count"]),
        "pyqcu_cache_process_grid": list(manifest["process_grid"]),
        "pyqcu_cache_global_coarse": list(manifest["global_coarse"]),
        "pyqcu_cache_local_coarse": list(manifest["local_coarse"]),
        "pyqcu_cache_global_shapes": json.dumps(
            manifest["global_shapes"], sort_keys=True),
    }
    temporary = root.with_name(f".{root.name}.{os.getpid()}.partial")
    try:
        with h5py.File(temporary, "w") as handle:
            for key, value in attributes.items():
                handle.attrs[key] = value
            handle.create_dataset("manifest", data=payload)
        os.replace(temporary, root)
    finally:
        if temporary.exists():
            temporary.unlink()
    digest = file_sha256(root)
    size = root.stat().st_size
    published["file_sha256"] = digest
    published["file_size_bytes"] = size
    published["output_sha256"] = digest
    published["output_size_bytes"] = size
    sidecar = write_legacy_mg_cache_manifest(root, published)
    return root, sidecar


def _utc_timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def is_legacy_mg_ranked_cache(cache_path: str | os.PathLike[str]) -> bool:
    """Return ``True`` when ``cache_path`` is a multi-rank cache root."""
    import h5py

    path = Path(cache_path)
    if not path.is_file():
        return False
    try:
        with h5py.File(path, "r") as handle:
            layout = handle.attrs.get("pyqcu_cache_layout")
            if layout is not None:
                return str(layout) == LEGACY_MG_RANKED_LAYOUT
            if "lonv" in handle or "sit" in handle:
                return False
            if "manifest" in handle:
                return True
    except OSError:
        return False
    sidecar = cache_manifest_path(path)
    if sidecar.is_file():
        try:
            manifest = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return manifest.get("layout") == LEGACY_MG_RANKED_LAYOUT
    return False


def read_legacy_mg_ranked_manifest(
        cache_path: str | os.PathLike[str], *,
        allow_incomplete: bool = False) -> dict[str, Any]:
    """Read and sanity check the manifest of a multi-rank cache."""
    import h5py

    path = Path(cache_path)
    if not path.is_file():
        raise FileNotFoundError(f"Legacy MG cache not found: {path}")
    manifest: dict[str, Any] | None = None
    sidecar = cache_manifest_path(path)
    if sidecar.is_file():
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    else:
        with h5py.File(path, "r") as handle:
            if "manifest" not in handle:
                raise ValueError(
                    f"ranked Legacy MG cache {path} has no manifest dataset")
            manifest = json.loads(handle["manifest"][()].decode("utf-8")
                                  if isinstance(handle["manifest"][()], bytes)
                                  else str(handle["manifest"][()]))
    accepted = ("PASS", "BUILDING") if allow_incomplete else ("PASS",)
    if manifest.get("status") not in accepted:
        raise ValueError(
            f"ranked Legacy MG cache status is {manifest.get('status')!r}; "
            f"expected one of {list(accepted)}")
    if manifest.get("schema_version") != LEGACY_MG_CACHE_SCHEMA:
        raise ValueError(
            f"unsupported ranked cache schema {manifest.get('schema_version')!r}")
    if manifest.get("layout") != LEGACY_MG_RANKED_LAYOUT:
        raise ValueError(
            f"unsupported ranked cache layout {manifest.get('layout')!r}")
    for key in ("identity", "global_shapes", "local_shapes", "global_coarse",
                "local_coarse", "process_grid", "ranks"):
        if key not in manifest:
            raise ValueError(f"ranked Legacy MG cache manifest lacks {key}")
    return manifest


def _load_ranked_asset(
        manifest: Mapping[str, Any], root: Path, name: str, dtype: np.dtype[Any],
        starts: Sequence[int], extent: Sequence[int]) -> np.ndarray:
    """Assemble one rank-local slab from the per-rank files."""
    import h5py

    local_shapes = legacy_mg_local_shapes(manifest["global_shapes"], extent)
    expected = tuple(int(value) for value in local_shapes[name])
    result = np.empty(expected, dtype=dtype)
    covered = np.zeros(tuple(int(value) for value in extent), dtype=np.int8)
    requested = tuple(int(value) for value in starts)
    extent_values = tuple(int(value) for value in extent)
    directory = root.parent / str(manifest.get(
        "rank_directory", legacy_mg_rank_dir(root).name))
    for record in manifest["ranks"]:
        rank_starts = tuple(int(value) for value in record["starts"])
        rank_shape = tuple(int(value) for value in record["local_coarse"])
        low = [max(requested[axis], rank_starts[axis]) for axis in range(4)]
        high = [min(requested[axis] + extent_values[axis],
                    rank_starts[axis] + rank_shape[axis]) for axis in range(4)]
        if any(high[axis] <= low[axis] for axis in range(4)):
            continue
        source = [low[axis] - rank_starts[axis] for axis in range(4)]
        width = [high[axis] - low[axis] for axis in range(4)]
        destination = [low[axis] - requested[axis] for axis in range(4)]
        window = covered[
            destination[0]:destination[0] + width[0],
            destination[1]:destination[1] + width[1],
            destination[2]:destination[2] + width[2],
            destination[3]:destination[3] + width[3]]
        if np.any(window):
            raise ValueError(
                f"ranked cache slabs overlap inside {name} at {tuple(low)}")
        path = directory / str(record["path"])
        if not path.is_file():
            raise FileNotFoundError(
                f"ranked Legacy MG cache file is missing: {path}")
        record_datasets = record.get("datasets")
        if not isinstance(record_datasets, Mapping) or name not in record_datasets:
            # A two-phase (BUILDING) manifest only describes the tensors that
            # have been published yet; asking for a later stage tensor must
            # fail closed instead of surfacing a bare KeyError.
            published = sorted(record_datasets) if isinstance(
                record_datasets, Mapping) else []
            raise ValueError(
                f"rank {record.get('rank')} of ranked Legacy MG cache {root} "
                f"does not publish {name} yet (published: {published})")
        declared = tuple(int(value) for value in
                         record_datasets[name]["shape"])
        rank_expected = legacy_mg_local_shapes(
            manifest["global_shapes"], rank_shape)[name]
        if declared != rank_expected:
            raise ValueError(
                f"rank {record['rank']} {name} manifest shape {declared} != "
                f"{rank_expected}")
        with h5py.File(path, "r") as handle:
            if name not in handle:
                raise ValueError(f"rank cache {path} is missing {name}")
            dataset = handle[name]
            if tuple(int(value) for value in dataset.shape) != declared:
                raise ValueError(
                    f"rank cache {path} {name} shape "
                    f"{tuple(dataset.shape)} != manifest {declared}")
            if np.dtype(dataset.dtype) != dtype:
                raise ValueError(
                    f"rank cache {path} {name} dtype {dataset.dtype} != "
                    f"{dtype}")
            selection = _asset_coarse_slice(name, source, width, len(expected))
            value = np.ascontiguousarray(dataset[selection], dtype=dtype)
        result[_asset_coarse_slice(name, destination, width, len(expected))] = \
            value
        window[...] = 1
    missing = int(covered.size - int(covered.sum()))
    if missing:
        raise ValueError(
            f"ranked cache does not cover {missing} of {covered.size} coarse "
            f"sites requested for {name} at {requested}+{extent_values}")
    return result


def assemble_legacy_mg_cache(
        cache_path: str | os.PathLike[str], dtype: Any, *,
        names: Sequence[str] = LEGACY_MG_ASSET_NAMES) -> dict[str, np.ndarray]:
    """Assemble the global tensors of a multi-rank cache on one rank."""
    manifest = read_legacy_mg_ranked_manifest(cache_path)
    selected = tuple(str(name) for name in names)
    unknown = sorted(set(selected).difference(LEGACY_MG_ASSET_NAMES))
    if unknown:
        raise ValueError(f"unknown Legacy MG datasets {unknown}")
    canonical = canonical_dtype(dtype)
    if np.dtype(manifest["identity"]["dtype"]) != canonical:
        raise ValueError(
            f"ranked cache dtype {manifest['identity']['dtype']} != {canonical}")
    coarse = tuple(int(value) for value in manifest["global_coarse"])
    return {
        name: _load_ranked_asset(
            manifest, Path(cache_path), name, canonical, (0, 0, 0, 0), coarse)
        for name in selected
    }


def _load_legacy_mg_ranked_slab(
        path: Path, dtype: np.dtype[Any],
        starts: tuple[int, int, int, int] | None,
        extent: tuple[int, int, int, int] | None,
        names: Sequence[str], *, allow_incomplete: bool = False
        ) -> dict[str, np.ndarray]:
    """Read the rank-local slabs of a multi-rank cache, fail closed."""
    manifest = read_legacy_mg_ranked_manifest(
        path, allow_incomplete=allow_incomplete)
    if np.dtype(manifest["identity"]["dtype"]) != dtype:
        raise ValueError(
            f"ranked cache dtype {manifest['identity']['dtype']} != {dtype}")
    coarse = tuple(int(value) for value in manifest["global_coarse"])
    if starts is None:
        requested = (0, 0, 0, 0)
        window = tuple(coarse)
    else:
        assert extent is not None
        requested = starts
        window = extent
        for axis in range(4):
            if requested[axis] < 0 or requested[axis] + window[axis] > coarse[axis]:
                raise ValueError(
                    f"requested slab {requested}+{window} exceeds the global "
                    f"coarse box {coarse}")
    return {
        name: _load_ranked_asset(
            manifest, path, name, dtype, requested, window)
        for name in names
    }


def load_legacy_mg_cache_slab(
        cache_path: str | os.PathLike[str], dtype: Any, *,
        starts: Sequence[int] | None = None,
        local_shape: Sequence[int] | None = None,
        names: Sequence[str] = LEGACY_MG_ASSET_NAMES,
        allow_incomplete: bool = False,
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
    if is_legacy_mg_ranked_cache(path):
        return _load_legacy_mg_ranked_slab(
            path, selected_dtype, start_values, shape_values, selected_names,
            allow_incomplete=allow_incomplete)
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


def periodic_coarse_ranges(
        start: int, extent: int, size: int) -> list[tuple[int, int, slice]]:
    """Split a cyclically wrapped coarse interval into contiguous reads.

    The overlap patch of the ranked builder wraps across the global boundary, so
    a window that leaves the box on one side re-enters on the other.  A wrapped
    interval touches at most two contiguous ranges per axis, hence at most
    sixteen hyperslab reads for a four-dimensional window.
    """
    if extent <= 0 or size <= 0:
        raise ValueError(f"invalid periodic window extent={extent} size={size}")
    if extent > size:
        raise ValueError(
            f"periodic window extent {extent} exceeds the global extent {size}")
    start %= size
    if start + extent <= size:
        return [(start, extent, slice(0, extent))]
    first = size - start
    return [(start, first, slice(0, first)),
            (0, extent - first, slice(first, extent))]


def load_legacy_mg_cache_patch(
        cache_path: str | os.PathLike[str], name: str, dtype: Any, *,
        starts: Sequence[int], extents: Sequence[int],
        global_coarse: Sequence[int],
        global_shapes: Mapping[str, Sequence[int]],
        allow_incomplete: bool = False) -> np.ndarray:
    """Load a periodically wrapped coarse window of one Legacy MG asset.

    This is the read half of the overlap-patch builder: every MPI rank needs a
    window that is larger than its own hyperslab and that may straddle the
    global boundary.  The pieces are scattered onto the asset's *coarse* storage
    axes (``lonv`` keeps two leading dof axes, ``hnn/hdg`` keep their slot axes),
    never onto the leading axes of the tensor.
    """
    from itertools import product

    if name not in LEGACY_MG_ASSET_NAMES:
        raise ValueError(f"unknown Legacy MG dataset {name!r}")
    canonical = canonical_dtype(dtype)
    coarse = _tuple4(global_coarse, "global_coarse")
    window_starts = _tuple4(starts, "starts")
    window_extents = _tuple4(extents, "extents")
    if name not in global_shapes:
        raise ValueError(f"global shapes do not describe {name}")
    shape = legacy_mg_local_shapes(global_shapes, window_extents)[name]
    result = np.empty(shape, dtype=canonical)
    coarse_axes = LEGACY_MG_ASSET_COARSE_AXES[name]
    pieces = product(*[
        periodic_coarse_ranges(window_starts[axis], window_extents[axis],
                              coarse[axis])
        for axis in range(4)
    ])
    for piece in pieces:
        read_starts = tuple(int(item[0]) for item in piece)
        read_extents = tuple(int(item[1]) for item in piece)
        value = load_legacy_mg_cache_slab(
            cache_path, canonical, starts=read_starts,
            local_shape=read_extents, names=(name,),
            allow_incomplete=allow_incomplete)[name]
        selection = [slice(None)] * result.ndim
        for index, axis in enumerate(coarse_axes):
            selection[axis] = piece[index][2]
        selection = tuple(selection)
        expected = result[selection].shape
        if tuple(value.shape) != expected:
            raise ValueError(
                f"periodic {name} piece {read_starts}+{read_extents} read back "
                f"with shape {tuple(value.shape)} != {expected}")
        result[selection] = value
    return result


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


def _check_identity(
        manifest: Mapping[str, Any],
        expected_identity: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless the manifest identity matches the requested one."""
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
    return expected


def validate_legacy_mg_cache(
        cache_path: str | os.PathLike[str],
        expected_identity: Mapping[str, Any], *,
        verify_file_sha256: bool = False,
        verify_dataset_sha256: bool = True,
        ranks: Sequence[int] | None = None) -> dict[str, Any]:
    """Fail closed when a cache is not the requested physical/geometric asset.

    ``ranks`` restricts the per-rank file checks of a multi-rank cache to the
    listed ranks (each MPI rank can validate its own slab without touching the
    whole cache); ``None`` validates every rank file.
    """
    import h5py

    path = Path(cache_path)
    if not path.is_file():
        raise FileNotFoundError(f"Legacy MG cache not found: {path}")
    if is_legacy_mg_ranked_cache(path):
        return _validate_legacy_mg_ranked_cache(
            path, expected_identity, verify_file_sha256=verify_file_sha256,
            verify_dataset_sha256=verify_dataset_sha256, ranks=ranks)
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

    expected = _check_identity(manifest, expected_identity)

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


def _validate_legacy_mg_ranked_cache(
        path: Path, expected_identity: Mapping[str, Any], *,
        verify_file_sha256: bool, verify_dataset_sha256: bool,
        ranks: Sequence[int] | None) -> dict[str, Any]:
    """Validate a multi-rank cache root, its manifest, and the rank files."""
    import h5py

    manifest = read_legacy_mg_ranked_manifest(path)
    expected = _check_identity(manifest, expected_identity)
    # The root container must describe the same publication stage as the
    # manifest that is being validated; a stale sidecar next to a half-written
    # root (or the reverse) is a torn artifact and must not be consumed.
    with h5py.File(path, "r") as root_handle:
        root_status = root_handle.attrs.get("pyqcu_cache_status")
        root_layout = root_handle.attrs.get("pyqcu_cache_layout")
    if root_layout is not None and str(root_layout) != LEGACY_MG_RANKED_LAYOUT:
        raise ValueError(
            f"ranked Legacy MG cache root {path} declares layout "
            f"{root_layout!r}, expected {LEGACY_MG_RANKED_LAYOUT}")
    if root_status is not None and str(root_status) != str(manifest["status"]):
        raise ValueError(
            f"ranked Legacy MG cache root {path} status {root_status!r} != "
            f"manifest status {manifest['status']!r}")
    expected_shapes = legacy_mg_cache_shapes(
        expected["lattice"], expected["mg_grid"], expected["dof"])
    for name in LEGACY_MG_ASSET_NAMES:
        declared = tuple(int(value) for value in
                         manifest["global_shapes"][name])
        if declared != expected_shapes[name]:
            raise ValueError(
                f"ranked cache {name} global shape {declared} != "
                f"{expected_shapes[name]}")
    grid = tuple(int(value) for value in manifest["process_grid"])
    coarse = tuple(int(value) for value in manifest["global_coarse"])
    local = tuple(int(value) for value in manifest["local_coarse"])
    for axis in range(4):
        if local[axis] * grid[axis] != coarse[axis]:
            raise ValueError(
                f"ranked cache local coarse slab {local} with process grid "
                f"{grid} does not tile {coarse}")
    records = list(manifest["ranks"])
    if len(records) != int(np.prod(grid)):
        raise ValueError(
            f"ranked cache holds {len(records)} rank records for grid {grid}")
    _check_rank_tiling(records, coarse)
    for record in records:
        coordinate = tuple(int(value) for value in record["coordinate"])
        expected_coordinate = tuple(int(value) for value in np.unravel_index(
            int(record["rank"]), grid, order="C"))
        if coordinate != expected_coordinate:
            raise ValueError(
                f"rank {record['rank']} coordinate {coordinate} != "
                f"{expected_coordinate} for grid {grid}")
    local_shapes = legacy_mg_local_shapes(expected_shapes, local)
    selected = (None if ranks is None
                else {int(value) for value in ranks})
    directory = path.parent / str(manifest["rank_directory"])
    for record in records:
        rank = int(record["rank"])
        if selected is not None and rank not in selected:
            continue
        rank_path = directory / str(record["path"])
        if not rank_path.is_file():
            raise FileNotFoundError(
                f"ranked Legacy MG cache file is missing: {rank_path}")
        with h5py.File(rank_path, "r") as handle:
            if str(handle.attrs.get("pyqcu_cache_layout")) != \
                    LEGACY_MG_RANKED_LAYOUT:
                raise ValueError(
                    f"rank cache {rank_path} does not declare the ranked layout")
            for key, value in (
                    ("pyqcu_cache_rank", rank),
                    ("pyqcu_cache_coordinate", list(record["coordinate"])),
                    ("pyqcu_cache_coarse_starts", list(record["starts"])),
                    ("pyqcu_cache_local_coarse", list(local))):
                actual = handle.attrs.get(key)
                if actual is None or list(np.atleast_1d(actual)) != list(
                        np.atleast_1d(value)):
                    raise ValueError(
                        f"rank cache {rank_path} attribute {key} {actual!r} != "
                        f"{value!r}")
            if str(handle.attrs.get("pyqcu_cache_identity_sha256")) != \
                    str(manifest["identity_sha256"]):
                raise ValueError(
                    f"rank cache {rank_path} identity digest does not match "
                    f"the manifest")
            for name in LEGACY_MG_ASSET_NAMES:
                if name not in handle:
                    raise ValueError(f"rank cache {rank_path} lacks {name}")
                dataset = handle[name]
                declared = tuple(int(value) for value in
                                 record["datasets"][name]["shape"])
                if declared != local_shapes[name]:
                    raise ValueError(
                        f"rank {rank} {name} manifest shape {declared} != "
                        f"{local_shapes[name]}")
                if tuple(int(value) for value in dataset.shape) != declared:
                    raise ValueError(
                        f"rank cache {rank_path} {name} shape "
                        f"{tuple(dataset.shape)} != {declared}")
                if np.dtype(dataset.dtype) != np.dtype(expected["dtype"]):
                    raise ValueError(
                        f"rank cache {rank_path} {name} dtype "
                        f"{dataset.dtype} != {expected['dtype']}")
                if verify_dataset_sha256:
                    digest = record["datasets"][name].get("sha256")
                    if digest and logical_dataset_sha256(dataset) != digest:
                        raise ValueError(
                            f"rank cache {rank_path} {name} digest mismatch")
        if verify_file_sha256:
            digest = record.get("file_sha256")
            if digest and file_sha256(rank_path) != digest:
                raise ValueError(
                    f"rank cache {rank_path} file digest mismatch")
    if verify_file_sha256:
        root_digest = manifest.get("file_sha256")
        if root_digest and file_sha256(path) != root_digest:
            raise ValueError("ranked Legacy MG cache root digest mismatch")
    return manifest


__all__ = [
    "LEGACY_MG_ASSET_COARSE_AXES",
    "LEGACY_MG_ASSET_NAMES",
    "LEGACY_MG_CACHE_LAYOUT",
    "LEGACY_MG_CACHE_SCHEMA",
    "LEGACY_MG_RANKED_LAYOUT",
    "assemble_legacy_mg_cache",
    "build_legacy_mg_cache_manifest",
    "build_legacy_mg_ranked_cache_manifest",
    "cache_manifest_path",
    "canonical_dtype",
    "file_sha256",
    "identity_sha256",
    "is_legacy_mg_ranked_cache",
    "legacy_mg_cache_identity",
    "legacy_mg_cache_path",
    "legacy_mg_cache_shapes",
    "legacy_mg_cache_stem",
    "legacy_mg_geometry",
    "legacy_mg_local_shapes",
    "legacy_mg_rank_cache_path",
    "legacy_mg_rank_dir",
    "load_legacy_mg_cache_patch",
    "load_legacy_mg_cache_slab",
    "logical_array_sha256",
    "logical_array_sha256_chunked",
    "logical_dataset_sha256",
    "read_legacy_mg_ranked_manifest",
    "save_legacy_mg_cache",
    "validate_legacy_mg_cache",
    "write_legacy_mg_asset_subset",
    "write_legacy_mg_assets",
    "write_legacy_mg_cache_manifest",
    "write_legacy_mg_rank_slice",
    "write_legacy_mg_ranked_cache",
]
