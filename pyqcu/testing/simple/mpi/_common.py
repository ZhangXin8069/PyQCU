"""Shared MPI geometry and LIME/HDF5 helpers for the simple MPI examples."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from pyqcu.tools._legacy_mg_cache import load_legacy_mg_cache_slab

from pyqcu.testing.simple._common import (
    DEFAULT_CSW,
    DEFAULT_MASS,
    lime_metadata,
    qcu_gauge_to_kupl,
    qdp_gauge_to_qcu,
    qcu_fermion_to_kupl,
    read_h5,
    read_h5_metadata,
    read_lime_gauge,
    resolve_path,
)


DEFAULT_LIME = Path.home() / "data" / (
    "beta6.20_mu-0.2770_ms-0.2310_L24x64_cfg_9950.lime")
LIME_MAGIC = b"\x45\x67\x89\xab"


def tuple4(values: Sequence[int], name: str) -> tuple[int, int, int, int]:
    result = tuple(int(value) for value in values)
    if len(result) != 4:
        raise ValueError(f"{name} must contain four integers: {result}")
    return result  # type: ignore[return-value]


def validate_lattice(values: Sequence[int]) -> tuple[int, int, int, int]:
    lattice = tuple4(values, "lattice")
    if any(value <= 0 for value in lattice):
        raise ValueError(f"lattice extents must be positive: {lattice}")
    if lattice[3] % 2:
        raise ValueError(f"time extent must be even: {lattice}")
    return lattice


def default_grid(lattice: Sequence[int], ranks: int) -> tuple[int, int, int, int]:
    """Choose a 4D grid with even local extents, preferring y/z/t splits."""
    lattice = validate_lattice(lattice)
    ranks = int(ranks)
    if ranks <= 0:
        raise ValueError(f"ranks must be positive, got {ranks}")
    remaining = ranks
    grid = [1, 1, 1, 1]
    # KUPL packs x most expensively, so split y, z, t before x.
    for axis in (1, 2, 3, 0):
        chosen = 1
        for factor in range(remaining, 1, -1):
            if remaining % factor:
                continue
            candidate_local = lattice[axis] // (grid[axis] * factor)
            if lattice[axis] % (grid[axis] * factor):
                continue
            if candidate_local <= 0 or candidate_local % 2:
                continue
            chosen = factor
            break
        grid[axis] *= chosen
        remaining //= chosen
    if remaining != 1:
        raise ValueError(
            f"{ranks} ranks cannot be mapped to {lattice} with even local "
            "extents; pass --grid explicitly")
    return tuple(grid)  # type: ignore[return-value]


def resolve_grid(lattice: Sequence[int], ranks: int,
                 grid: Sequence[int] | None) -> tuple[int, int, int, int]:
    selected = default_grid(lattice, ranks) if grid is None else tuple4(
        grid, "grid")
    if any(value <= 0 for value in selected):
        raise ValueError(f"grid extents must be positive: {selected}")
    if math.prod(selected) != int(ranks):
        raise ValueError(
            f"grid {selected} contains {math.prod(selected)} ranks, "
            f"requested {ranks}")
    local_geometry(lattice, selected, 0)
    return selected


def local_geometry(lattice: Sequence[int], grid: Sequence[int], rank: int
                   ) -> tuple[tuple[int, int, int, int],
                              tuple[int, int, int, int],
                              tuple[int, int, int, int]]:
    lattice = validate_lattice(lattice)
    grid = tuple4(grid, "grid")
    rank = int(rank)
    if rank < 0 or rank >= math.prod(grid):
        raise ValueError(f"rank {rank} is outside process grid {grid}")
    if any(lattice[axis] % grid[axis] for axis in range(4)):
        raise ValueError(f"lattice {lattice} is not divisible by grid {grid}")
    local = tuple(lattice[axis] // grid[axis] for axis in range(4))
    if any(extent % 2 for extent in local):
        raise ValueError(
            f"local lattice must be even in every axis: {local}")
    coordinate = tuple(
        int(value) for value in np.unravel_index(rank, grid, order="C"))
    starts = tuple(coordinate[axis] * local[axis] for axis in range(4))
    return local, starts, coordinate  # type: ignore[return-value]


def slice_last4(array: np.ndarray, starts: Sequence[int],
                local_shape: Sequence[int]) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim < 4:
        raise ValueError(f"expected at least four lattice axes: {values.shape}")
    selection = [slice(None)] * (values.ndim - 4)
    selection.extend(
        slice(int(starts[axis]), int(starts[axis] + local_shape[axis]))
        for axis in range(4))
    return np.ascontiguousarray(values[tuple(selection)])


def is_lime_file(path: str | Path) -> bool:
    """Recognize QIO/LIME by magic, independent of suffix case."""
    candidate = Path(path).expanduser()
    if not candidate.is_file():
        return False
    try:
        with candidate.open("rb") as handle:
            return handle.read(4) == LIME_MAGIC
    except OSError:
        return False


def read_gauge(path_value: str | Path, lattice: Sequence[int],
               dtype: np.dtype) -> tuple[np.ndarray, dict[str, Any], str]:
    """Read a QCU/QDP HDF5 gauge or a QIO/LIME gauge and slice a prefix block."""
    path = resolve_path(path_value, Path(path_value).expanduser())
    target = validate_lattice(lattice)
    if is_lime_file(path):
        metadata = lime_metadata(path)
        full = tuple(int(value) for value in metadata["lattice_xyzt"])
        if any(target[axis] > full[axis] for axis in range(4)):
            raise ValueError(
                f"requested lattice {target} exceeds LIME lattice {full}")
        gauge = qdp_gauge_to_qcu(read_lime_gauge(path))
        input_format = "LIME/SciDAC"
    elif path.suffix.lower() in (".h5", ".hdf5"):
        metadata = read_h5_metadata(path)
        raw = np.asarray(read_h5(path))
        if raw.ndim == 7 and raw.shape[:3] == (3, 3, 4):
            gauge = raw
            input_format = "QCU HDF5"
        elif raw.ndim == 7 and raw.shape[0] == 4 and raw.shape[-2:] == (3, 3):
            gauge = qdp_gauge_to_qcu(raw)
            input_format = "QDP HDF5"
        else:
            raise ValueError(f"unsupported gauge HDF5 shape: {raw.shape}")
    else:
        raise ValueError(f"gauge input must be LIME or HDF5: {path}")

    if gauge.ndim != 7 or gauge.shape[:3] != (3, 3, 4):
        raise ValueError(f"gauge must be [3,3,4,x,y,z,t], got {gauge.shape}")
    full = tuple(int(value) for value in gauge.shape[-4:])
    if any(target[axis] > full[axis] for axis in range(4)):
        raise ValueError(f"requested lattice {target} exceeds gauge {full}")
    gauge = slice_last4(gauge, (0, 0, 0, 0), target)
    return np.ascontiguousarray(gauge.astype(dtype, copy=False)), metadata, input_format


def read_source(path_value: str | Path, lattice: Sequence[int],
                dtype: np.dtype) -> tuple[np.ndarray, dict[str, Any], str]:
    path = resolve_path(path_value, Path(path_value).expanduser())
    target = validate_lattice(lattice)
    if path.suffix.lower() in (".h5", ".hdf5"):
        source = np.asarray(read_h5(path))
        metadata = read_h5_metadata(path)
        input_format = "HDF5"
    elif path.suffix.lower() == ".npy":
        source = np.load(path, allow_pickle=False)
        metadata = {}
        input_format = "NumPy"
    else:
        raise ValueError(f"source input must be HDF5 or .npy: {path}")
    if source.ndim != 6 or source.shape[:2] != (4, 3):
        raise ValueError(f"source must be [spin,color,x,y,z,t], got {source.shape}")
    full = tuple(int(value) for value in source.shape[-4:])
    if any(target[axis] > full[axis] for axis in range(4)):
        raise ValueError(f"requested lattice {target} exceeds source {full}")
    source = slice_last4(source, (0, 0, 0, 0), target)
    return np.ascontiguousarray(source.astype(dtype, copy=False)), metadata, input_format


def canonical_gauge(gauge_full: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(qcu_gauge_to_kupl(gauge_full))


def canonical_fermion(fermion_full: np.ndarray) -> np.ndarray:
    compact = qcu_fermion_to_kupl(fermion_full)
    return np.ascontiguousarray(
        compact.reshape((2, 4, 3, *compact.shape[2:])))


def load_coarse_assets(path_value: str | Path, dtype: np.dtype,
                       starts: Sequence[int] | None = None,
                       local_shape: Sequence[int] | None = None
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load global 33-tensor assets or one rank-local hyperslab."""
    path = resolve_path(path_value, Path(path_value).expanduser())
    values = load_legacy_mg_cache_slab(
        path, dtype, starts=starts, local_shape=local_shape)
    return values["lonv"], values["hnn"], values["hdg"], values["sit"]


def mass_and_csw(metadata: dict[str, Any], mass: float | None,
                 csw: float | None) -> tuple[float, float]:
    selected_mass = (
        float(mass) if mass is not None else
        float(metadata.get("reference_mass", metadata.get("mass", DEFAULT_MASS))))
    selected_csw = (
        float(csw) if csw is not None else
        float(metadata.get("csw", DEFAULT_CSW)))
    if not math.isfinite(selected_mass) or 2.0 * selected_mass + 8.0 == 0.0:
        raise ValueError(
            f"mass must be finite and produce a finite kappa: {selected_mass}")
    if not math.isfinite(selected_csw):
        raise ValueError(f"csw must be finite: {selected_csw}")
    return selected_mass, selected_csw
