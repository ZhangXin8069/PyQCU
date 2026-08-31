"""CPU-only, small-array tests for strict runtime HDF5 cache."""

from __future__ import annotations

from hashlib import sha256
import json
from math import prod

import h5py
import numpy as np
import pytest
import torch

import pyqcu.cuda._strict_cache as strict_cache

from pyqcu.cuda._strict_cache import (
    STRICT_RUNTIME_CACHE_SCHEMA,
    STRICT_RUNTIME_CACHE_VERSION,
    StrictRuntimeCacheConflictError,
    inspect_strict_runtime_cache,
    load_strict_runtime_cache,
    make_strict_runtime_cache_manifest,
    save_strict_runtime_cache,
)


def _complex_values(shape, start=0, dtype=torch.complex64):
    count = 1
    for extent in shape:
        count *= extent
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    real = torch.arange(
        start, start + count, dtype=real_dtype).reshape(shape)
    return (real + 1j * (real + 0.25)).to(dtype)


def _fixture(dtype=torch.complex64):
    fine = _complex_values((4, 3, 2), 0, dtype)
    levels = [
        {
            "level": 0,
            "preconditioned_links": _complex_values(
                (2, 4, 2, 2), 100, dtype),
            "onsite_pair": _complex_values((2, 2, 2), 200, dtype),
            "null_vectors": None,
        },
        {
            "level": 1,
            "preconditioned_links": _complex_values(
                (2, 4, 1, 1), 300, dtype),
            "onsite_pair": _complex_values((2, 1, 1), 400, dtype),
            "null_vectors": _complex_values((1, 2, 2), 500, dtype),
        },
    ]
    identity = {
        "gauge_sha256": "sha256:gauge-small",
        "operator_sha256": "sha256:operator-small",
        "dtype": "complex64" if dtype == torch.complex64 else "complex128",
        "local_shape": [4, 4, 4, 4],
        "rank": 0,
    }
    metadata = {
        "target_parity": 1,
        "block_sizes": [[2, 2, 2, 2], [2, 2, 2, 2]],
        "note": "小数组；不启动 CUDA",
    }
    stats = [
        {"level": 0, "setup_seconds": 0.01, "peak_bytes": 1234},
        {"level": 1, "setup_seconds": 0.02, "peak_bytes": 456},
    ]
    return fine, levels, identity, metadata, stats


def _canonical_json(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False)


def _replace_manifest(handle, manifest):
    text = _canonical_json(manifest)
    handle.attrs.modify("manifest_json", text)
    handle.attrs.modify("manifest_sha256", sha256(text.encode()).hexdigest())


def _assert_assets_equal(hit, fine, levels):
    assert hit.assets is not None
    assert torch.equal(hit.assets.fine_blocked_v, fine)
    assert len(hit.assets.levels) == len(levels)
    for index, (actual, expected) in enumerate(zip(hit.assets.levels, levels)):
        assert actual.level == index
        assert torch.equal(
            actual.preconditioned_links, expected["preconditioned_links"])
        assert torch.equal(actual.onsite_pair, expected["onsite_pair"])
        if index == 0:
            assert actual.null_vectors is None
        else:
            assert torch.equal(actual.null_vectors, expected["null_vectors"])


def _assert_structural_contract(full_manifest, expected_manifest):
    assert full_manifest is not None
    for key in ("layout", "level_count", "tensor_count", "dtype", "total_bytes"):
        assert full_manifest[key] == expected_manifest[key]
    assert set(full_manifest["tensors"]) == set(expected_manifest["tensors"])
    for path, expected in expected_manifest["tensors"].items():
        assert {
            key: full_manifest["tensors"][path][key]
            for key in ("shape", "dtype", "nbytes")
        } == expected


def test_roundtrip_is_single_handle_per_operation_and_streams_small_tensors(
        tmp_path, monkeypatch):
    fine, levels, identity, metadata, stats = _fixture()
    path = tmp_path / "strict-runtime.h5"
    expected_manifest = make_strict_runtime_cache_manifest(
        fine_blocked_v=fine, levels=levels)

    original_file = h5py.File
    calls = []

    def counted_file(*args, **kwargs):
        calls.append((str(args[0]), args[1] if len(args) > 1 else kwargs.get("mode")))
        return original_file(*args, **kwargs)

    monkeypatch.setattr("pyqcu.cuda._strict_cache.h5py.File", counted_file)
    written = save_strict_runtime_cache(
        path,
        identity=identity,
        fine_blocked_v=fine,
        levels=levels,
        metadata=metadata,
        stats=stats,
    )
    assert written.written and written.reason == "published"
    assert len(calls) == 1
    assert calls[0][1] == "w"
    assert not list(tmp_path.glob(".strict-runtime.h5.*.tmp"))

    calls.clear()
    loaded = load_strict_runtime_cache(
        path, identity=identity, device="cpu",
        expected_manifest=expected_manifest)
    assert loaded.hit and loaded.reason == "hit" and loaded.miss_reason is None
    assert len(calls) == 1
    assert calls[0] == (str(path), "r")
    assert loaded.metadata == metadata
    assert loaded.stats == stats
    assert loaded.evidence is not None
    assert loaded.evidence["identity_sha256"] == sha256(
        _canonical_json(identity).encode()).hexdigest()
    assert loaded.evidence["metadata_sha256"] == sha256(
        _canonical_json(metadata).encode()).hexdigest()
    assert loaded.evidence["stats_sha256"] == sha256(
        _canonical_json(stats).encode()).hexdigest()
    assert loaded.evidence["manifest_sha256"] == sha256(
        _canonical_json(loaded.manifest).encode()).hexdigest()
    assert loaded.evidence["logical_bytes"] == expected_manifest["total_bytes"]
    assert set(loaded.evidence["tensor_digests"]) == set(
        expected_manifest["tensors"])
    _assert_structural_contract(loaded.manifest, expected_manifest)
    _assert_assets_equal(loaded, fine, levels)

    with original_file(path, "r") as handle:
        assert handle.attrs["schema"] == STRICT_RUNTIME_CACHE_SCHEMA
        assert int(handle.attrs["schema_version"]) == \
            STRICT_RUNTIME_CACHE_VERSION == 2
        assert handle.attrs["state"] == "complete"
        assert set(handle.keys()) == {"assets"}
        manifest = json.loads(handle.attrs["manifest_json"])
        for dataset_path, spec in manifest["tensors"].items():
            dataset = handle[dataset_path]
            assert spec["digest_algorithm"] == \
                "sha256(pyqcu-logical-tensor-v1)"
            assert len(spec["sha256"]) == 64
            assert dataset.attrs["digest_algorithm"] == \
                spec["digest_algorithm"]
            assert dataset.attrs["sha256"] == spec["sha256"]


def test_inspect_revalidates_content_without_torch_allocation_or_transfer(
        tmp_path, monkeypatch):
    fine, levels, identity, metadata, stats = _fixture()
    path = tmp_path / "inspect-only.h5"
    expected = make_strict_runtime_cache_manifest(
        fine_blocked_v=fine, levels=levels)
    save_strict_runtime_cache(
        path, identity=identity, fine_blocked_v=fine, levels=levels,
        metadata=metadata, stats=stats)

    monkeypatch.setattr(
        strict_cache.torch, "empty",
        lambda *_args, **_kwargs: pytest.fail("inspect must not allocate"))
    monkeypatch.setattr(
        strict_cache.torch, "from_numpy",
        lambda *_args, **_kwargs: pytest.fail("inspect must not transfer"))

    inspected = inspect_strict_runtime_cache(
        path, identity=identity, expected_manifest=expected)
    assert inspected.hit, (inspected.reason, inspected.detail)
    assert inspected.assets is None
    assert inspected.metadata == metadata
    assert inspected.stats == stats
    assert inspected.evidence is not None
    assert inspected.evidence["tensor_count"] == expected["tensor_count"]

    with h5py.File(path, "r+") as handle:
        dataset = handle["assets/fine_blocked_v"]
        index = (0,) * dataset.ndim
        dataset[index] = dataset[index] + np.complex64(1.0)
    tampered = inspect_strict_runtime_cache(
        path, identity=identity, expected_manifest=expected)
    assert not tampered.hit
    assert tampered.reason == "tensor_digest_mismatch"


@pytest.mark.parametrize(
    ("dtype", "expected_alias"),
    [
        (torch.complex64, "c64"),
        (torch.complex128, "c128"),
    ],
)
def test_c64_c128_roundtrip_accepts_digest_free_expected_contract(
        tmp_path, dtype, expected_alias):
    fine, levels, identity, metadata, stats = _fixture(dtype)
    path = tmp_path / f"strict-runtime-{expected_alias}.h5"
    expected = make_strict_runtime_cache_manifest(
        fine_blocked_v=fine, levels=levels)
    assert all(set(spec) == {"shape", "dtype", "nbytes"}
               for spec in expected["tensors"].values())
    expected["dtype"] = expected_alias
    for spec in expected["tensors"].values():
        spec["dtype"] = expected_alias

    save_strict_runtime_cache(
        path, identity=identity, fine_blocked_v=fine, levels=levels,
        metadata=metadata, stats=stats)
    loaded = load_strict_runtime_cache(
        path, identity=identity, expected_manifest=expected)

    assert loaded.hit, (loaded.reason, loaded.detail)
    _assert_assets_equal(loaded, fine, levels)
    assert loaded.manifest["dtype"] == (
        "complex64" if dtype == torch.complex64 else "complex128")


def test_single_tensor_value_tamper_is_rejected_before_device_transfer(
        tmp_path, monkeypatch):
    fine, levels, identity, _, _ = _fixture()
    path = tmp_path / "tampered-value.h5"
    save_strict_runtime_cache(
        path, identity=identity, fine_blocked_v=fine, levels=levels)

    dataset_path = "assets/levels/1/null_vectors"
    with h5py.File(path, "r+") as handle:
        dataset = handle[dataset_path]
        index = (0,) * dataset.ndim
        dataset[index] = dataset[index] + np.complex64(1.0 + 0.5j)

    transfers = []
    allocations = []
    original_from_numpy = strict_cache.torch.from_numpy
    original_empty = strict_cache.torch.empty

    def tracked_from_numpy(array):
        transfers.append(int(array.nbytes))
        return original_from_numpy(array)

    def tracked_empty(*args, **kwargs):
        allocations.append((args, kwargs))
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(strict_cache.torch, "from_numpy", tracked_from_numpy)
    monkeypatch.setattr(strict_cache.torch, "empty", tracked_empty)
    loaded = load_strict_runtime_cache(path, identity=identity)

    assert not loaded.hit
    assert loaded.reason == "tensor_digest_mismatch"
    assert loaded.assets is None
    assert transfers == []
    assert allocations == []


def test_tensor_digest_attribute_tamper_is_structured_miss(tmp_path):
    fine, levels, identity, _, _ = _fixture()
    path = tmp_path / "tampered-digest-attr.h5"
    save_strict_runtime_cache(
        path, identity=identity, fine_blocked_v=fine, levels=levels)

    with h5py.File(path, "r+") as handle:
        handle["assets/fine_blocked_v"].attrs.modify("sha256", "0" * 64)

    loaded = load_strict_runtime_cache(path, identity=identity)
    assert not loaded.hit
    assert loaded.reason == "tensor_digest_metadata_mismatch"
    assert loaded.assets is None


def test_coordinated_manifest_digest_tamper_is_checked_against_data(tmp_path):
    fine, levels, identity, _, _ = _fixture()
    path = tmp_path / "tampered-manifest-digest.h5"
    save_strict_runtime_cache(
        path, identity=identity, fine_blocked_v=fine, levels=levels)

    dataset_path = "assets/levels/0/onsite_pair"
    fake_digest = "f" * 64
    with h5py.File(path, "r+") as handle:
        manifest = json.loads(handle.attrs["manifest_json"])
        manifest["tensors"][dataset_path]["sha256"] = fake_digest
        _replace_manifest(handle, manifest)
        handle[dataset_path].attrs.modify("sha256", fake_digest)

    loaded = load_strict_runtime_cache(path, identity=identity)
    assert not loaded.hit
    assert loaded.reason == "tensor_digest_mismatch"
    assert loaded.assets is None


def test_manifest_digest_tamper_is_cross_checked_against_dataset_attrs(tmp_path):
    fine, levels, identity, _, _ = _fixture()
    path = tmp_path / "tampered-manifest-only.h5"
    save_strict_runtime_cache(
        path, identity=identity, fine_blocked_v=fine, levels=levels)

    dataset_path = "assets/levels/0/preconditioned_links"
    with h5py.File(path, "r+") as handle:
        manifest = json.loads(handle.attrs["manifest_json"])
        manifest["tensors"][dataset_path]["sha256"] = "e" * 64
        _replace_manifest(handle, manifest)

    loaded = load_strict_runtime_cache(path, identity=identity)
    assert not loaded.hit
    assert loaded.reason == "tensor_digest_metadata_mismatch"
    assert loaded.assets is None


class _TrackingArray(np.ndarray):
    def __new__(cls, value):
        result = np.asarray(value).view(cls)
        result.read_nbytes = []
        return result

    def __array_finalize__(self, source):
        self.read_nbytes = getattr(source, "read_nbytes", [])

    def __getitem__(self, index):
        result = super().__getitem__(index)
        if isinstance(result, np.ndarray):
            self.read_nbytes.append(int(result.nbytes))
        return result


def test_tensor_io_is_chunked_and_load_keeps_one_hdf5_handle(
        tmp_path, monkeypatch):
    fine, levels, identity, _, _ = _fixture()
    tracked_fine = _TrackingArray(fine.numpy())
    path = tmp_path / "streamed.h5"
    monkeypatch.setattr(strict_cache, "_TENSOR_IO_CHUNK_BYTES", 64, raising=False)

    save_strict_runtime_cache(
        path, identity=identity, fine_blocked_v=tracked_fine, levels=levels)
    assert len(tracked_fine.read_nbytes) > 1
    assert max(tracked_fine.read_nbytes) <= 64

    original_reader = getattr(strict_cache, "_read_dataset_chunk", None)
    assert original_reader is not None
    reads = []

    def tracked_reader(dataset, selection):
        array = original_reader(dataset, selection)
        reads.append(int(array.nbytes))
        return array

    monkeypatch.setattr(strict_cache, "_read_dataset_chunk", tracked_reader)
    original_file = h5py.File
    handles = []

    def counted_file(*args, **kwargs):
        handles.append(args[1] if len(args) > 1 else kwargs.get("mode"))
        return original_file(*args, **kwargs)

    monkeypatch.setattr(strict_cache.h5py, "File", counted_file)
    loaded = load_strict_runtime_cache(path, identity=identity)

    assert loaded.hit, (loaded.reason, loaded.detail)
    assert handles == ["r"]
    assert len(reads) > loaded.manifest["tensor_count"]
    assert max(reads) <= 64


def test_tensor_digest_is_independent_of_stream_chunk_boundaries(
        tmp_path, monkeypatch):
    fine, levels, identity, _, _ = _fixture()
    paths = []
    digest_maps = []

    for chunk_bytes in (16, 1 << 20):
        path = tmp_path / f"chunk-{chunk_bytes}.h5"
        monkeypatch.setattr(
            strict_cache, "_TENSOR_IO_CHUNK_BYTES", chunk_bytes,
            raising=False)
        save_strict_runtime_cache(
            path, identity=identity, fine_blocked_v=fine, levels=levels)
        with h5py.File(path, "r") as handle:
            manifest = json.loads(handle.attrs["manifest_json"])
        paths.append(path)
        digest_maps.append({
            dataset_path: spec["sha256"]
            for dataset_path, spec in manifest["tensors"].items()
        })

    assert digest_maps[0] == digest_maps[1]

    monkeypatch.setattr(
        strict_cache, "_TENSOR_IO_CHUNK_BYTES", 24, raising=False)
    for path in paths:
        loaded = load_strict_runtime_cache(path, identity=identity)
        assert loaded.hit, (loaded.reason, loaded.detail)
        _assert_assets_equal(loaded, fine, levels)


def test_missing_identity_mismatch_and_same_identity_are_explicit(tmp_path):
    fine, levels, identity, metadata, stats = _fixture()
    path = tmp_path / "strict-runtime.h5"

    missing = load_strict_runtime_cache(path, identity=identity)
    assert not missing.hit and missing.reason == "not_found"

    first = save_strict_runtime_cache(
        path, identity=identity, fine_blocked_v=fine, levels=levels,
        metadata=metadata, stats=stats)
    before = sha256(path.read_bytes()).hexdigest()

    second = save_strict_runtime_cache(
        path, identity=identity, fine_blocked_v=fine, levels=levels,
        metadata={"would": "not overwrite"}, stats={})
    assert first.written and not second.written
    assert second.reason == "already_exists"
    assert sha256(path.read_bytes()).hexdigest() == before

    other_identity = dict(identity, operator_sha256="sha256:different")
    miss = load_strict_runtime_cache(path, identity=other_identity)
    assert not miss.hit and miss.reason == "identity_mismatch"
    assert miss.cached_identity == identity

    with pytest.raises(StrictRuntimeCacheConflictError, match="identity 不匹配"):
        save_strict_runtime_cache(
            path, identity=other_identity, fine_blocked_v=fine, levels=levels)
    assert sha256(path.read_bytes()).hexdigest() == before
    assert not list(tmp_path.glob(".strict-runtime.h5.*.tmp"))


def test_same_identity_corrupt_target_is_never_accepted_as_existing(tmp_path):
    fine, levels, identity, _, _ = _fixture()
    path = tmp_path / "corrupt-same-identity.h5"
    save_strict_runtime_cache(
        path, identity=identity, fine_blocked_v=fine, levels=levels)
    with h5py.File(path, "r+") as handle:
        dataset = handle["assets/fine_blocked_v"]
        index = (0,) * dataset.ndim
        dataset[index] = dataset[index] + np.complex64(1.0)

    with pytest.raises(
            StrictRuntimeCacheConflictError, match="tensor_digest_mismatch"):
        save_strict_runtime_cache(
            path, identity=identity, fine_blocked_v=fine, levels=levels)


@pytest.mark.parametrize("same_identity", [True, False])
def test_atomic_publish_never_clobbers_a_concurrent_writer(
        tmp_path, monkeypatch, same_identity):
    fine, levels, identity, _, _ = _fixture()
    competitor_identity = (
        identity if same_identity else
        dict(identity, operator_sha256="sha256:concurrent-other"))
    competitor = tmp_path / "competitor.h5"
    target = tmp_path / "raced.h5"
    save_strict_runtime_cache(
        competitor, identity=competitor_identity,
        fine_blocked_v=fine, levels=levels)
    competitor_digest = sha256(competitor.read_bytes()).hexdigest()

    original_link = __import__("os").link
    raced = False

    def racing_link(source, destination):
        nonlocal raced
        if not raced and str(destination) == str(target):
            raced = True
            original_link(competitor, target)
        return original_link(source, destination)

    monkeypatch.setattr("pyqcu.cuda._strict_cache.os.link", racing_link)
    if same_identity:
        result = save_strict_runtime_cache(
            target, identity=identity, fine_blocked_v=fine, levels=levels)
        assert not result.written
        assert result.reason == "concurrent_same_identity"
    else:
        with pytest.raises(StrictRuntimeCacheConflictError, match="identity 不匹配"):
            save_strict_runtime_cache(
                target, identity=identity, fine_blocked_v=fine, levels=levels)
    assert raced
    assert sha256(target.read_bytes()).hexdigest() == competitor_digest
    assert not list(tmp_path.glob(".raced.h5.*.tmp"))


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("schema", "schema_mismatch"),
        ("version", "version_mismatch"),
        ("version_type", "version_mismatch"),
        ("shape", "shape_mismatch"),
        ("dtype", "dtype_mismatch"),
        ("nbytes_attr", "dataset_metadata_mismatch"),
        ("metadata_json", "metadata_invalid"),
    ],
)
def test_corrupt_schema_version_identity_shape_dtype_are_precise_misses(
        tmp_path, mutation, reason):
    fine, levels, identity, metadata, stats = _fixture()
    path = tmp_path / f"corrupt-{mutation}.h5"
    save_strict_runtime_cache(
        path, identity=identity, fine_blocked_v=fine, levels=levels,
        metadata=metadata, stats=stats)

    with h5py.File(path, "r+") as handle:
        if mutation == "schema":
            handle.attrs.modify("schema", "wrong.schema")
        elif mutation == "version":
            handle.attrs.modify("schema_version", 1)
        elif mutation == "version_type":
            del handle.attrs["schema_version"]
            handle.attrs["schema_version"] = 2.5
        elif mutation == "metadata_json":
            handle.attrs.modify("metadata_json", "{broken-json")
        elif mutation == "nbytes_attr":
            dataset = handle["assets/levels/0/onsite_pair"]
            changed_nbytes = float(dataset.attrs["nbytes"]) + 0.5
            del dataset.attrs["nbytes"]
            dataset.attrs["nbytes"] = changed_nbytes
        else:
            dataset_path = "assets/levels/0/onsite_pair"
            original = handle[dataset_path][...]
            attrs = dict(handle[dataset_path].attrs)
            del handle[dataset_path]
            if mutation == "shape":
                changed = original.reshape(-1)
            else:
                changed = original.astype("complex128")
            dataset = handle.create_dataset(dataset_path, data=changed)
            for key, value in attrs.items():
                dataset.attrs[key] = value

    loaded = load_strict_runtime_cache(path, identity=identity)
    assert not loaded.hit
    assert loaded.reason == reason
    assert loaded.assets is None


def test_expected_manifest_and_level_contract_are_strict(tmp_path):
    fine, levels, identity, _, _ = _fixture()
    path = tmp_path / "strict-runtime.h5"
    save_strict_runtime_cache(
        path, identity=identity, fine_blocked_v=fine, levels=levels)

    expected = make_strict_runtime_cache_manifest(
        fine_blocked_v=fine, levels=levels)
    changed = json.loads(json.dumps(expected))
    fine_spec = changed["tensors"]["assets/fine_blocked_v"]
    fine_spec["shape"][-1] += 1
    fine_spec["nbytes"] = prod(fine_spec["shape"]) * 8
    changed["total_bytes"] = sum(
        spec["nbytes"] for spec in changed["tensors"].values())
    mismatch = load_strict_runtime_cache(
        path, identity=identity, expected_manifest=changed)
    assert not mismatch.hit
    assert mismatch.reason == "expected_manifest_mismatch"

    invalid_dtype = json.loads(json.dumps(expected))
    invalid_dtype["dtype"] = ["complex64"]
    with pytest.raises(ValueError, match="manifest dtype"):
        load_strict_runtime_cache(
            path, identity=identity, expected_manifest=invalid_dtype)

    float_nbytes = json.loads(json.dumps(expected))
    float_nbytes["tensors"]["assets/fine_blocked_v"]["nbytes"] = float(
        float_nbytes["tensors"]["assets/fine_blocked_v"]["nbytes"])
    with pytest.raises(ValueError, match="nbytes"):
        load_strict_runtime_cache(
            path, identity=identity, expected_manifest=float_nbytes)

    bad_level_zero = [dict(levels[0], null_vectors=fine)]
    with pytest.raises(ValueError, match="fine_blocked_v"):
        save_strict_runtime_cache(
            tmp_path / "bad-zero.h5", identity=identity,
            fine_blocked_v=fine, levels=bad_level_zero)

    missing_recursive = [levels[0], dict(levels[1], null_vectors=None)]
    with pytest.raises(ValueError, match="缺少递归所需"):
        make_strict_runtime_cache_manifest(
            fine_blocked_v=fine, levels=missing_recursive)


def test_json_metadata_is_strict_and_symlink_is_not_followed(tmp_path):
    fine, levels, identity, _, _ = _fixture()
    path = tmp_path / "strict-runtime.h5"
    with pytest.raises(ValueError, match="NaN/Inf"):
        save_strict_runtime_cache(
            path, identity=identity, fine_blocked_v=fine, levels=levels,
            stats={"bad": float("nan")})

    real = tmp_path / "real.h5"
    save_strict_runtime_cache(
        real, identity=identity, fine_blocked_v=fine, levels=levels)
    link = tmp_path / "linked.h5"
    link.symlink_to(real)
    loaded = load_strict_runtime_cache(link, identity=identity)
    assert not loaded.hit and loaded.reason == "not_regular_file"
    with pytest.raises(StrictRuntimeCacheConflictError, match="非普通"):
        save_strict_runtime_cache(
            link, identity=identity, fine_blocked_v=fine, levels=levels)
