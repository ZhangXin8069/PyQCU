"""Synthetic-stencil regressions for the multi-rank Legacy-MG cache.

Every asset here is a synthetic array of arbitrary known values, cut into the
per-rank coarse hyperslabs exactly the way the MPI exporter publishes them
(two phases: ``lonv`` first, then ``hnn/hdg/sit``).  The tests never touch
KUPL/C++, so they pin the on-disk contract -- file names, manifest fields,
local slab shapes, dtypes, identity and SHA256 bookkeeping -- independently of
the distributed stencil builder:

* the global tensors rebuild bit for bit from the per-rank files;
* every rank reads back exactly its own hyperslab;
* the periodically wrapped ``lonv`` patch the local 33-stencil consumes equals
  the wrapped global window, and the slice cut from it is the rank's own slab;
* missing files, wrong shapes, tampered bytes and identity drift fail closed.
"""

from pathlib import Path

import numpy as np
import pytest

from pyqcu.tools._legacy_mg_cache import (
    LEGACY_MG_ASSET_COARSE_AXES,
    LEGACY_MG_ASSET_NAMES,
    assemble_legacy_mg_cache,
    build_legacy_mg_ranked_cache_manifest,
    is_legacy_mg_ranked_cache,
    legacy_mg_cache_identity,
    legacy_mg_cache_shapes,
    legacy_mg_local_shapes,
    legacy_mg_rank_dir,
    load_legacy_mg_cache_slab,
    read_legacy_mg_ranked_manifest,
    validate_legacy_mg_cache,
    write_legacy_mg_rank_slice,
    write_legacy_mg_ranked_cache,
)
from pyqcu.testing.simple.kupl_multigrid import (
    _periodic_window,
    _slice_asset,
    inspect_cache_layout,
    ranked_coarse_geometry,
    read_periodic_coarse_asset,
)

# (lattice, grid, mg_grid, dof, dtype); the last two fit the two-block halo and
# therefore exercise wrapped periodic patch reads.
CASES = [
    ((4, 4, 4, 8), (2, 1, 1, 1), (2, 2, 2, 2), (12, 2), np.complex64),
    ((4, 4, 4, 8), (1, 1, 1, 2), (2, 2, 2, 2), (12, 2), np.complex64),
    ((4, 4, 4, 8), (2, 2, 1, 1), (2, 2, 2, 2), (12, 2), np.complex64),
    ((4, 4, 4, 8), (1, 1, 1, 4), (2, 2, 2, 1), (12, 2), np.complex64),
    ((4, 4, 4, 8), (2, 1, 1, 2), (2, 2, 2, 1), (12, 2), np.complex128),
    ((16, 8, 8, 8), (2, 1, 1, 1), (2, 2, 2, 2), (12, 2), np.complex64),
    ((16, 8, 8, 16), (1, 1, 1, 2), (2, 2, 2, 2), (12, 2), np.complex64),
]


def _synth(shape, dtype, seed: int) -> np.ndarray:
    """Arbitrary known values, including the two signed zeros."""
    rng = np.random.default_rng(seed)
    real_dtype = np.float32 if np.dtype(dtype) == np.dtype(np.complex64) \
        else np.float64
    real = rng.standard_normal(shape).astype(real_dtype)
    imag = rng.standard_normal(shape).astype(real_dtype)
    value = np.ascontiguousarray((real + 1j * imag).astype(dtype))
    flat = value.reshape(-1)
    flat[0] = 0.0
    flat[1] = complex(-0.0, -0.0)
    return value


def _rank_geometry(lattice, grid, mg, rank):
    coordinate = tuple(int(value) for value in
                       np.unravel_index(rank, grid, order="C"))
    local_lattice = tuple(lattice[axis] // grid[axis] for axis in range(4))
    local_odd = (*local_lattice[:3], local_lattice[3] // 2)
    fine_odd = (*lattice[:3], lattice[3] // 2)
    coarse = tuple(fine_odd[axis] // mg[axis] for axis in range(4))
    local_coarse = tuple(local_odd[axis] // mg[axis] for axis in range(4))
    coarse_starts = tuple(coordinate[axis] * local_coarse[axis]
                          for axis in range(4))
    return coordinate, local_lattice, coarse, local_coarse, coarse_starts


def _rank_slice(assets, name, starts, extent):
    selection = [slice(None)] * assets[name].ndim
    for index, axis in enumerate(LEGACY_MG_ASSET_COARSE_AXES[name]):
        begin = int(starts[index])
        selection[axis] = slice(begin, begin + int(extent[index]))
    return np.ascontiguousarray(assets[name][tuple(selection)])


def _bits(array) -> bytes:
    return np.ascontiguousarray(array).tobytes()


def _publish(tmp_path: Path, lattice, grid, mg, dof, dtype):
    """Two-phase synthetic ranked-cache export; returns the fixture state."""
    shapes = legacy_mg_cache_shapes(lattice, mg, dof)
    assets = {name: _synth(shapes[name], dtype, seed=index + 11)
              for index, name in enumerate(LEGACY_MG_ASSET_NAMES)}
    identity = legacy_mg_cache_identity(
        gauge_canonical_sha256="c" * 64, lattice=lattice, dtype=dtype,
        mass=-0.277, csw=1.160920226, dof=list(dof), mg_grid=list(mg),
        nv_iters=2, seed=42)
    cache = tmp_path / "ranked.h5"
    geometry = {rank: _rank_geometry(lattice, grid, mg, rank)
                for rank in range(int(np.prod(grid)))}
    local_coarse = geometry[0][3]
    local_shapes = legacy_mg_local_shapes(shapes, local_coarse)
    coarse = geometry[0][2]
    first = []
    for rank, (coordinate, _, _, local, starts) in geometry.items():
        first.append(write_legacy_mg_rank_slice(
            cache, {"lonv": _rank_slice(assets, "lonv", starts, local)},
            rank=rank, coordinate=coordinate, process_grid=grid, starts=starts,
            local_coarse=local, global_shapes=shapes, identity=identity,
            mode="w"))
    write_legacy_mg_ranked_cache(cache, build_legacy_mg_ranked_cache_manifest(
        cache, identity, process_grid=grid, ranks=first, global_shapes=shapes,
        local_shapes=local_shapes, global_coarse=coarse,
        local_coarse=local_coarse, setup_method="synthetic",
        setup_backend="synthetic", assets=("lonv",), status="BUILDING"))
    final = []
    for rank, (coordinate, _, _, local, starts) in geometry.items():
        final.append(write_legacy_mg_rank_slice(
            cache, {name: _rank_slice(assets, name, starts, local)
                    for name in ("hnn", "hdg", "sit")},
            rank=rank, coordinate=coordinate, process_grid=grid, starts=starts,
            local_coarse=local, global_shapes=shapes, identity=identity,
            mode="r+"))
    manifest = build_legacy_mg_ranked_cache_manifest(
        cache, identity, process_grid=grid, ranks=final, global_shapes=shapes,
        local_shapes=local_shapes, global_coarse=coarse,
        local_coarse=local_coarse, setup_method="synthetic",
        setup_backend="synthetic", setup_seconds=1.0)
    write_legacy_mg_ranked_cache(cache, manifest)
    return {
        "cache": cache, "assets": assets, "identity": identity,
        "shapes": shapes, "geometry": geometry, "manifest": manifest,
        "coarse": coarse, "local_coarse": local_coarse, "grid": grid,
        "lattice": lattice, "mg": mg, "dof": dof, "dtype": np.dtype(dtype),
    }


@pytest.mark.parametrize("lattice,grid,mg,dof,dtype", CASES)
def test_ranked_cache_export_readback_is_bitwise_exact(
        tmp_path: Path, lattice, grid, mg, dof, dtype) -> None:
    state = _publish(tmp_path, lattice, grid, mg, dof, dtype)
    cache, assets, identity = state["cache"], state["assets"], state["identity"]
    ranks = int(np.prod(grid))

    assert is_legacy_mg_ranked_cache(cache) is True
    manifest = validate_legacy_mg_cache(
        cache, identity, verify_file_sha256=True, verify_dataset_sha256=True)
    assert manifest["status"] == "PASS"
    assert manifest["layout"] == "odd-schur-ranked"
    assert manifest["rank_count"] == ranks
    assert set(manifest["assets"]) == set(LEGACY_MG_ASSET_NAMES)
    assert manifest["rank_directory"] == legacy_mg_rank_dir(cache).name

    # Global rebuild from the per-rank files.
    rebuilt = assemble_legacy_mg_cache(cache, state["dtype"])
    for name in LEGACY_MG_ASSET_NAMES:
        assert _bits(rebuilt[name]) == _bits(assets[name]), name

    # Public whole-cache read.
    whole = load_legacy_mg_cache_slab(cache, state["dtype"])
    for name in LEGACY_MG_ASSET_NAMES:
        assert _bits(whole[name]) == _bits(assets[name]), name

    # Per-rank hyperslab read-back, including the per-rank validation path.
    for rank, (coordinate, local_lattice, _, local, starts) in \
            state["geometry"].items():
        validate_legacy_mg_cache(
            cache, identity, verify_file_sha256=True,
            verify_dataset_sha256=True, ranks=(rank,))
        slab = load_legacy_mg_cache_slab(
            cache, state["dtype"], starts=starts, local_shape=local)
        for name in LEGACY_MG_ASSET_NAMES:
            expected = _rank_slice(assets, name, starts, local)
            assert _bits(slab[name]) == _bits(expected), (rank, name)
        layout, inspected_local, inspected_starts = inspect_cache_layout(
            cache, tuple(lattice), list(mg), list(grid),
            tuple(coordinate[axis] * local_lattice[axis] for axis in range(4)),
            "auto")
        assert layout == "odd-schur"
        assert inspected_local == local
        assert inspected_starts == starts


@pytest.mark.parametrize("lattice,grid,mg,dof,dtype", CASES[-2:])
def test_periodic_lonv_patch_equals_wrapped_global_window(
        tmp_path: Path, lattice, grid, mg, dof, dtype) -> None:
    state = _publish(tmp_path, lattice, grid, mg, dof, dtype)
    wrapped_axes = set()
    for rank, (coordinate, local_lattice, coarse, local, starts) in \
            state["geometry"].items():
        geometry = ranked_coarse_geometry(
            tuple(lattice), tuple(local_lattice), coordinate, tuple(grid),
            list(mg), list(dof))
        assert geometry["coarse_starts"] == starts
        for axis in range(4):
            if int(geometry["patch_coarse_starts"][axis]) + int(
                    geometry["patch_coarse_extent"][axis]) > int(coarse[axis]):
                wrapped_axes.add(axis)
        patch = read_periodic_coarse_asset(
            state["cache"], "lonv", state["dtype"],
            geometry["patch_coarse_starts"], geometry["patch_coarse_extent"],
            geometry["coarse"], state["shapes"], allow_incomplete=False)
        expected = _periodic_window(
            state["assets"]["lonv"], geometry["patch_coarse_starts"],
            geometry["patch_coarse_extent"],
            axes=LEGACY_MG_ASSET_COARSE_AXES["lonv"])
        assert _bits(patch) == _bits(expected), rank
        cut = _slice_asset(patch, "lonv", geometry["patch_coarse_center"],
                           local)
        assert _bits(cut) == _bits(
            _rank_slice(state["assets"], "lonv", starts, local)), rank
    # The halo must actually wrap for these geometries, otherwise the wrapped
    # branch of the reader would silently stay untested.
    assert wrapped_axes


@pytest.mark.parametrize("lattice,grid,mg,dof,dtype", [CASES[2], CASES[5]])
def test_ranked_cache_fails_closed(tmp_path: Path, lattice, grid, mg, dof,
                                   dtype) -> None:
    import h5py
    import shutil

    state = _publish(tmp_path, lattice, grid, mg, dof, dtype)
    cache, identity = state["cache"], state["identity"]
    dtype = state["dtype"]
    starts = state["geometry"][0][4]
    local = state["local_coarse"]

    # A half-written cache is not consumed as if it were complete.
    building = tmp_path / "building.h5"
    shutil.copytree(legacy_mg_rank_dir(cache),
                    legacy_mg_rank_dir(building))
    interim = dict(state["manifest"])
    interim["status"] = "BUILDING"
    write_legacy_mg_ranked_cache(building, interim)
    with pytest.raises(ValueError, match="status"):
        read_legacy_mg_ranked_manifest(building)
    assert read_legacy_mg_ranked_manifest(
        building, allow_incomplete=True)["status"] == "BUILDING"

    with pytest.raises(ValueError, match="exceeds the global coarse box"):
        load_legacy_mg_cache_slab(
            cache, dtype, starts=(int(starts[0]), 0, 0, 0),
            local_shape=(int(state["coarse"][0]) + 1, 1, 1, 1))

    other = np.complex128 if dtype == np.complex64 else np.complex64
    with pytest.raises(ValueError, match="dtype"):
        load_legacy_mg_cache_slab(cache, other)

    changed = dict(identity)
    changed["mass"] = float(identity["mass"]) + 1.0e-3
    with pytest.raises(ValueError, match="mass"):
        validate_legacy_mg_cache(cache, changed)
    changed = dict(identity)
    changed["gauge_canonical_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="gauge"):
        validate_legacy_mg_cache(cache, changed)

    # Missing rank file.
    victim = legacy_mg_rank_dir(cache) / "rank_00000.h5"
    parked = victim.with_suffix(".parked")
    shutil.move(victim, parked)
    try:
        with pytest.raises(FileNotFoundError):
            assemble_legacy_mg_cache(cache, dtype)
        with pytest.raises(FileNotFoundError):
            load_legacy_mg_cache_slab(
                cache, dtype, starts=starts, local_shape=local)
    finally:
        shutil.move(parked, victim)

    with h5py.File(victim, "r") as handle:
        original = handle["sit"][...]

    # Wrong slab shape inside one rank file.
    with h5py.File(victim, "r+") as handle:
        del handle["sit"]
        handle.create_dataset("sit", data=original.reshape(-1))
    try:
        with pytest.raises(ValueError):
            validate_legacy_mg_cache(cache, identity, ranks=(0,))
        with pytest.raises(ValueError):
            load_legacy_mg_cache_slab(
                cache, dtype, starts=starts, local_shape=local)
    finally:
        with h5py.File(victim, "r+") as handle:
            del handle["sit"]
            handle.create_dataset("sit", data=original)

    # Tampered bytes with an intact shape.
    corner = (0,) * original.ndim
    with h5py.File(victim, "r+") as handle:
        handle["sit"][corner] = handle["sit"][corner] + 1
    try:
        with pytest.raises(ValueError, match="digest mismatch"):
            validate_legacy_mg_cache(cache, identity, ranks=(0,))
    finally:
        with h5py.File(victim, "r+") as handle:
            handle["sit"][corner] = original[corner]

    # Overlapping rank tiling is rejected when the manifest is built.
    if int(np.prod(grid)) > 1:
        overlapping = [dict(record) for record in state["manifest"]["ranks"]]
        overlapping[1] = dict(overlapping[1], starts=overlapping[0]["starts"])
        with pytest.raises(ValueError, match="overlap"):
            build_legacy_mg_ranked_cache_manifest(
                cache, identity, process_grid=grid, ranks=overlapping,
                global_shapes=state["shapes"],
                local_shapes=legacy_mg_local_shapes(
                    state["shapes"], local),
                global_coarse=state["coarse"], local_coarse=local,
                setup_method="synthetic", setup_backend="synthetic")
