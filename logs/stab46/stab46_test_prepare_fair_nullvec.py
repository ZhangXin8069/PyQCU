"""Canonical full-null-vector 预处理器的秒级、CPU-only 协议测试。"""

from __future__ import annotations

import builtins
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

import prepare_fair_nullvec as prep


def _tiny_spec(tmp_path: Path, *, output_name: str = "full.h5") -> prep.PreparationSpec:
    return prep.PreparationSpec(
        source_path=tmp_path / "odd.h5",
        gauge_path=tmp_path / "gauge.h5",
        output_path=tmp_path / output_name,
        lattice=(2, 2, 2, 4),
        block=(1, 1, 1, 1),
        nvec=2,
        mass=0.05,
    )


def _write_inputs(spec: prep.PreparationSpec) -> None:
    source = (
        np.arange(np.prod(spec.expected_source_shape), dtype=np.float32)
        .reshape(spec.expected_source_shape)
        .astype(np.complex64)
    )
    gauge = np.zeros(spec.expected_gauge_shape, dtype=np.complex64)
    gauge.real[...] = np.arange(gauge.size, dtype=np.float32).reshape(gauge.shape)
    with h5py.File(spec.source_path, "w") as handle:
        handle.create_dataset(spec.source_dataset, data=source)
    with h5py.File(spec.gauge_path, "w") as handle:
        handle.create_dataset(spec.gauge_dataset, data=gauge)


def _full_vectors(spec: prep.PreparationSpec) -> list[np.ndarray]:
    shape = spec.expected_output_shape[1:]
    return [
        np.full(shape, complex(index + 1, -(index + 1)), dtype=np.complex64)
        for index in range(spec.nvec)
    ]


def _factory(vectors, calls):
    def generate(spec, fingerprints):
        calls.append((spec, fingerprints))
        yield from vectors

    return generate


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fingerprints_and_identity_cover_physics_and_layout(tmp_path):
    spec = _tiny_spec(tmp_path)
    _write_inputs(spec)

    fingerprints = prep.fingerprint_inputs(spec)
    identity = prep.build_identity(spec, fingerprints)

    assert fingerprints["source"]["shape"] == list(spec.expected_source_shape)
    assert fingerprints["gauge"]["shape"] == list(spec.expected_gauge_shape)
    assert identity["source"]["sha256"] == fingerprints["source"]["sha256"]
    assert identity["gauge"]["sha256"] == fingerprints["gauge"]["sha256"]
    assert identity["physics"]["mass"] == pytest.approx(0.05)
    assert identity["physics"]["kappa"] == pytest.approx(1.0 / 8.1)
    assert identity["source"]["parity"] == "odd"
    assert identity["output"]["layout"] == "[nvec,spin,color,x,y,z,t]"
    assert identity["reconstruction"]["version"] == 2


def test_create_verify_and_matching_reuse_are_atomic(tmp_path):
    spec = _tiny_spec(tmp_path)
    _write_inputs(spec)
    vectors = _full_vectors(spec)
    calls = []

    created = prep.prepare(
        spec,
        vector_factory=_factory(vectors, calls),
        git_revision_fn=lambda: "0123456789abcdef",
    )
    assert created["status"] == "created"
    assert spec.output_path.is_file()
    assert len(calls) == 1
    assert not list(tmp_path.glob(f".{spec.output_path.name}.tmp.*"))

    verified = prep.prepare(spec, mode="verify-only")
    assert verified["status"] == "verified"
    assert verified["output_sha256"] == created["output_sha256"]

    reused = prep.prepare(
        spec,
        vector_factory=lambda *_: pytest.fail("matching output must bypass CUDA"),
    )
    assert reused["status"] == "reused"

    with h5py.File(spec.output_path, "r") as handle:
        metadata = json.loads(handle.attrs["metadata_json"])
        np.testing.assert_array_equal(handle[spec.output_dataset][0], vectors[0])
        assert metadata["source_dataset_sha256"] == (
            prep.fingerprint_inputs(spec)["source"]["sha256"]
        )
        assert metadata["gauge_sha256"] == prep.fingerprint_inputs(spec)["gauge"]["sha256"]
        assert metadata["physics"]["mass"] == pytest.approx(spec.mass)
        assert metadata["physics"]["kappa"] == pytest.approx(spec.kappa)
        assert metadata["reconstruction"]["algorithm"] == prep.RECONSTRUCTION_ALGORITHM
        assert metadata["reconstruction"]["version"] == prep.RECONSTRUCTION_VERSION
        assert metadata["gamma_basis"] == prep.GAMMA_BASIS
        assert handle.attrs["gamma_basis"] == prep.GAMMA_BASIS
        assert handle[spec.output_dataset].attrs["gamma_basis"] == prep.GAMMA_BASIS
        assert metadata["memory_strategy"]["mode"] == "stream-one-vector"
        assert metadata["memory_strategy"]["full_batch_resident"] is False
        assert metadata["provenance"]["git_revision"] == "0123456789abcdef"


def test_legacy_basis_metadata_upgrade_preserves_dataset_and_is_idempotent(
        tmp_path):
    spec = _tiny_spec(tmp_path)
    _write_inputs(spec)
    created = prep.prepare(
        spec,
        vector_factory=_factory(_full_vectors(spec), []),
        git_revision_fn=lambda: "legacy-revision",
    )
    output_sha256 = created["output_sha256"]

    with h5py.File(spec.output_path, "r+") as handle:
        metadata = json.loads(handle.attrs["metadata_json"])
        metadata["identity"]["output"].pop("gamma_basis")
        metadata["output"].pop("gamma_basis")
        metadata.pop("gamma_basis")
        legacy_identity_hash = prep._sha256_json(metadata["identity"])
        metadata["identity_sha256"] = legacy_identity_hash
        handle.attrs["identity_sha256"] = legacy_identity_hash
        del handle.attrs["gamma_basis"]
        del handle[spec.output_dataset].attrs["gamma_basis"]
        handle.attrs["metadata_json"] = prep._json_bytes(metadata).decode("utf-8")
        handle.flush()

    upgraded = prep.upgrade_gamma_basis_metadata(
        spec.output_path, spec.output_dataset)
    assert upgraded["status"] == "upgraded"
    assert upgraded["output_sha256"] == output_sha256
    assert Path(upgraded["backup"]).is_file()
    verified = prep.prepare(spec, mode="verify-only")
    assert verified["output_sha256"] == output_sha256
    repeated = prep.upgrade_gamma_basis_metadata(
        spec.output_path, spec.output_dataset)
    assert repeated["status"] == "already-upgraded"


def test_existing_identity_mismatch_fails_without_overwrite(tmp_path):
    spec = _tiny_spec(tmp_path)
    _write_inputs(spec)
    prep.prepare(
        spec,
        vector_factory=_factory(_full_vectors(spec), []),
        git_revision_fn=lambda: "abc",
    )
    before = _file_sha256(spec.output_path)

    with h5py.File(spec.source_path, "r+") as handle:
        handle[spec.source_dataset][0, 0, 0, 0, 0, 0, 0, 0, 0, 0] += 1

    with pytest.raises(prep.PreparationError, match="identity mismatch"):
        prep.prepare(
            spec,
            vector_factory=lambda *_: pytest.fail("must fail before reconstruction"),
        )
    assert _file_sha256(spec.output_path) == before


def test_verify_detects_payload_hash_corruption(tmp_path):
    spec = _tiny_spec(tmp_path)
    _write_inputs(spec)
    prep.prepare(
        spec,
        vector_factory=_factory(_full_vectors(spec), []),
        git_revision_fn=lambda: "abc",
    )
    with h5py.File(spec.output_path, "r+") as handle:
        handle[spec.output_dataset][0, 0, 0, 0, 0, 0, 0] += 3

    with pytest.raises(prep.PreparationError, match="hash mismatch"):
        prep.prepare(spec, mode="verify-only")


def test_failed_generation_leaves_no_output_or_temporary(tmp_path):
    spec = _tiny_spec(tmp_path)
    _write_inputs(spec)

    def fail_after_one(_spec, _fingerprints):
        yield _full_vectors(spec)[0]
        raise RuntimeError("synthetic reconstruction failure")

    with pytest.raises(RuntimeError, match="synthetic reconstruction failure"):
        prep.prepare(
            spec,
            vector_factory=fail_after_one,
            git_revision_fn=lambda: "abc",
        )
    assert not spec.output_path.exists()
    assert not list(tmp_path.glob(f".{spec.output_path.name}.tmp.*"))


def test_atomic_publish_never_replaces_an_existing_name(tmp_path):
    temporary = tmp_path / ".candidate.tmp.unique"
    output = tmp_path / "canonical.h5"
    temporary.write_bytes(b"new")
    output.write_bytes(b"old")

    assert prep._publish_no_replace(temporary, output) is False
    assert output.read_bytes() == b"old"
    assert temporary.read_bytes() == b"new"


def test_dry_run_and_verify_only_never_import_cuda_stack(tmp_path, monkeypatch, capsys):
    spec = _tiny_spec(tmp_path)
    _write_inputs(spec)
    vectors = _full_vectors(spec)
    prep.prepare(
        spec,
        vector_factory=_factory(vectors, []),
        git_revision_fn=lambda: "abc",
    )

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("pyqcu"):
            raise AssertionError(f"protocol-only mode imported {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    common = [
        "--source", str(spec.source_path),
        "--gauge", str(spec.gauge_path),
        "--output", str(spec.output_path),
        "--lattice", *(str(value) for value in spec.lattice),
        "--block", *(str(value) for value in spec.block),
        "--nvec", str(spec.nvec),
    ]
    assert prep.main([*common, "--verify-only"]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["status"] == "verified"

    spec.output_path.unlink()
    assert prep.main([*common, "--dry-run"]) == 0
    planned = json.loads(capsys.readouterr().out)
    assert planned["status"] == "planned"
    assert not spec.output_path.exists()


def test_output_path_cannot_alias_an_input(tmp_path):
    spec = _tiny_spec(tmp_path)
    _write_inputs(spec)
    bad = prep.PreparationSpec(
        source_path=spec.source_path,
        gauge_path=spec.gauge_path,
        output_path=spec.source_path,
        lattice=spec.lattice,
        block=spec.block,
        nvec=spec.nvec,
        mass=spec.mass,
    )

    with pytest.raises(prep.PreparationError, match="must differ"):
        prep.prepare(bad, mode="dry-run")


def test_post_publish_verification_retries_only_transient_hdf5_lock(
        tmp_path, monkeypatch):
    spec = _tiny_spec(tmp_path)
    calls = []

    def transient(_spec, _identity):
        calls.append(1)
        if len(calls) < 3:
            try:
                raise BlockingIOError(11, "temporarily locked")
            except OSError as cause:
                raise prep.PreparationError("locked") from cause
        return {"status": "verified"}

    monkeypatch.setattr(prep.time, "sleep", lambda _seconds: None)
    result = prep._verify_after_publish(
        spec, {"identity": 1}, verifier=transient)
    assert result == {"status": "verified"}
    assert len(calls) == 3

    def permanent(_spec, _identity):
        try:
            raise OSError(5, "I/O error")
        except OSError as cause:
            raise prep.PreparationError("broken") from cause

    with pytest.raises(prep.PreparationError, match="broken"):
        prep._verify_after_publish(
            spec, {"identity": 1}, verifier=permanent)
