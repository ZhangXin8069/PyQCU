"""Cross-backend Legacy and Strict MG cache protocol regressions."""

from __future__ import annotations

import numpy as np
import pytest

from pyqcu.tools._legacy_mg_cache import (
    LEGACY_MG_ASSET_NAMES,
    build_legacy_mg_cache_manifest,
    legacy_mg_cache_identity,
    legacy_mg_cache_shapes,
    load_legacy_mg_cache_slab,
    validate_legacy_mg_cache,
    write_legacy_mg_assets,
    write_legacy_mg_cache_manifest,
)
from pyqcu.tools._strict_mg_cache import (
    load_strict_mg_cache,
    make_strict_mg_cache_identity,
    save_strict_mg_cache,
    strict_mg_backend_assets,
)
from pyqcu.solver.multigrid import MultigridCache


def _legacy_fixture(lattice=(4, 4, 4, 4), mg_grid=(2, 2, 2, 2)):
    shapes = legacy_mg_cache_shapes(lattice, mg_grid, (12, 2))
    assets = {}
    for index, name in enumerate(LEGACY_MG_ASSET_NAMES, start=1):
        count = int(np.prod(shapes[name]))
        base = np.arange(count, dtype=np.float32).reshape(shapes[name])
        assets[name] = np.ascontiguousarray(base + 1j * index, dtype=np.complex64)
    identity = legacy_mg_cache_identity(
        gauge_sha256="a" * 64,
        lattice=lattice,
        dtype="c64",
        mass=-0.277,
        csw=1.160920226,
        dof=[12, 2],
        mg_grid=mg_grid,
        nv_iters=1,
        seed=42,
    )
    return assets, identity


def _write_legacy_cache(tmp_path, assets, identity):
    path = tmp_path / "legacy.h5"
    write_legacy_mg_assets(path, assets, identity=identity)
    manifest = build_legacy_mg_cache_manifest(
        path, identity, datasets=assets,
        setup_method="test-cxx-null-vector+local-33-stencil",
        setup_backend="test-common")
    write_legacy_mg_cache_manifest(path, manifest)
    return path


def test_legacy_cache_is_validated_and_sliced_once_for_all_backends(tmp_path):
    assets, identity = _legacy_fixture()
    path = _write_legacy_cache(tmp_path, assets, identity)

    manifest = validate_legacy_mg_cache(
        path, identity, verify_file_sha256=True)
    assert manifest["layout"] == "odd-schur"
    assert set(manifest["assets"]) == set(LEGACY_MG_ASSET_NAMES)

    slab = load_legacy_mg_cache_slab(
        path, "c64", starts=(1, 0, 1, 0), local_shape=(1, 2, 1, 1))
    assert slab["lonv"].shape == (2, 12, 1, 2, 2, 2, 1, 2, 1, 2)
    assert slab["hnn"].shape == (2, 4, 2, 2, 1, 2, 1, 1)
    assert slab["hdg"].shape == (2, 2, 6, 2, 2, 1, 2, 1, 1)
    assert slab["sit"].shape == (2, 2, 1, 2, 1, 1)
    np.testing.assert_array_equal(
        slab["lonv"],
        assets["lonv"][:, :, 1:2, :, 0:2, :, 1:2, :, 0:1, :])


@pytest.mark.parametrize("field,value", [
    ("gauge_sha256", "b" * 64),
    ("mass", -0.276),
    ("mg_grid", [2, 2, 2, 4]),
])
def test_legacy_cache_rejects_identity_mismatch(tmp_path, field, value):
    assets, identity = _legacy_fixture()
    path = _write_legacy_cache(tmp_path, assets, identity)
    changed = dict(identity)
    changed[field] = value
    with pytest.raises(ValueError):
        validate_legacy_mg_cache(path, changed)


def test_legacy_cache_detects_dataset_content_tamper(tmp_path):
    assets, identity = _legacy_fixture()
    path = _write_legacy_cache(tmp_path, assets, identity)
    import h5py

    with h5py.File(path, "r+") as handle:
        handle["sit"][0, 0, 0, 0, 0, 0] += 1
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_legacy_mg_cache(path, identity)


def test_strict_cache_roundtrip_maps_to_kupl_and_torch_assets(tmp_path):
    identity = make_strict_mg_cache_identity(
        gauge_sha256="c" * 64,
        null_vectors_sha256="d" * 64,
        lattice_xyzt=(4, 4, 4, 4),
        dtype="c64",
        mass=-0.277,
        kappa=0.13430029546065003,
        levels=2,
        block_xyzt_per_level=((2, 2, 2, 2),),
        nvec=2,
        coarse_spin=2,
        target_parity=1,
        gamma_basis="QUDA_DEGRAND_ROSSI_GAMMA_BASIS",
        setup_semantics="R(X^-1 D)P")
    transitions = (
        {
            "null_vectors": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
            "preconditioned_links": np.arange(
                64, dtype=np.float32).reshape(2, 4, 2, 2, 2),
            "onsite_pair": np.arange(
                32, dtype=np.float32).reshape(2, 2, 2, 4),
        },
    )
    path = tmp_path / "strict.h5"
    saved = save_strict_mg_cache(
        path, identity=identity, transitions=transitions)
    assert saved.written is True

    result = load_strict_mg_cache(path, identity=identity, device="cpu")
    mapped = strict_mg_backend_assets(result)
    assert tuple(mapped["fine_null_vectors"].shape) == (2, 3, 4)
    assert len(mapped["levels"]) == 1
    assert mapped["levels"][0]["null_vectors"] is None
    assert tuple(mapped["levels"][0]["preconditioned_links"].shape) == (
        2, 4, 2, 2, 2)
    assert result.manifest["dtype"] == "float32"
    assert result.evidence["tensor_count"] == 3


def test_strict_cache_rejects_different_identity(tmp_path):
    base = dict(
        gauge_sha256="c" * 64,
        null_vectors_sha256="d" * 64,
        lattice_xyzt=(4, 4, 4, 4),
        dtype="c64",
        mass=-0.277,
        kappa=0.13430029546065003,
        levels=2,
        block_xyzt_per_level=((2, 2, 2, 2),),
        nvec=2,
        coarse_spin=2,
        target_parity=1,
        gamma_basis="QUDA_DEGRAND_ROSSI_GAMMA_BASIS",
        setup_semantics="R(X^-1 D)P")
    identity = make_strict_mg_cache_identity(**base)
    transitions = ({
        "null_vectors": np.ones((2, 3, 4), dtype=np.float32),
        "preconditioned_links": np.ones((2, 4, 2, 2, 2), dtype=np.float32),
        "onsite_pair": np.ones((2, 2, 2, 4), dtype=np.float32),
    },)
    path = tmp_path / "strict.h5"
    save_strict_mg_cache(path, identity=identity, transitions=transitions)
    changed = dict(base)
    changed["target_parity"] = 0
    result = load_strict_mg_cache(
        path, identity=make_strict_mg_cache_identity(**changed), device="cpu")
    assert result.hit is False
    assert result.reason == "identity_mismatch"


@pytest.mark.parametrize("backend", ["torch", "cuda", "kupl"])
def test_unified_legacy_cache_api_is_backend_independent(tmp_path, backend):
    assets, identity = _legacy_fixture()
    cache = MultigridCache(variant="legacy", backend=backend)
    path = tmp_path / f"legacy-{backend}.h5"
    cache.save(
        path, identity=identity, assets=assets,
        metadata={"setup_method": "test", "setup_backend": backend})
    loaded = cache.load(path, identity=identity)
    assert loaded.variant == "legacy"
    assert loaded.backend == backend
    assert set(loaded.assets) == set(LEGACY_MG_ASSET_NAMES)


@pytest.mark.parametrize("backend", ["torch", "cuda", "kupl"])
def test_unified_strict_cache_api_is_backend_independent(tmp_path, backend):
    identity = make_strict_mg_cache_identity(
        gauge_sha256="e" * 64,
        null_vectors_sha256="f" * 64,
        lattice_xyzt=(4, 4, 4, 4),
        dtype="c64",
        mass=-0.277,
        kappa=0.13430029546065003,
        levels=2,
        block_xyzt_per_level=((2, 2, 2, 2),),
        nvec=2,
        coarse_spin=2,
        target_parity=1,
        gamma_basis="QUDA_DEGRAND_ROSSI_GAMMA_BASIS",
        setup_semantics="R(X^-1 D)P")
    transitions = ({
        "null_vectors": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        "preconditioned_links": np.arange(
            64, dtype=np.float32).reshape(2, 4, 2, 2, 2),
        "onsite_pair": np.arange(32, dtype=np.float32).reshape(2, 2, 2, 4),
    },)
    cache = MultigridCache(variant="strict", backend=backend)
    path = tmp_path / f"strict-{backend}.h5"
    cache.save(path, identity=identity, assets=transitions)
    loaded = cache.load(path, identity=identity)
    assert loaded.variant == "strict"
    assert loaded.backend == backend
    np.testing.assert_array_equal(
        loaded.assets["fine_null_vectors"], transitions[0]["null_vectors"])
    np.testing.assert_array_equal(
        loaded.assets["levels"][0]["preconditioned_links"],
        transitions[0]["preconditioned_links"])
