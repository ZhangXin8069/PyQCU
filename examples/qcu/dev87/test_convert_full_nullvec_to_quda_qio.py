"""QIO 转换协议的秒级 CPU 测试；不导入 CUDA、PyQUDA 或 QUDA。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

import convert_full_nullvec_to_quda_qio as converter


LATTICE = (4, 4, 4, 4)
BLOCK = (2, 2, 2, 1)
NVEC = 2


def _canonical(
    path: Path,
    *,
    source_nvec: int = NVEC,
    values: np.ndarray | None = None,
) -> np.ndarray:
    shape = (NVEC, 4, 3, *LATTICE)
    if values is None:
        real = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        values = np.asarray(real + 1j * (real + 0.25), dtype=np.complex64)
    odd = (*LATTICE[:3], LATTICE[3] // 2)
    coarse = tuple(x // b for x, b in zip(odd, BLOCK))
    source_shape = (
        source_nvec,
        12,
        coarse[0], BLOCK[0],
        coarse[1], BLOCK[1],
        coarse[2], BLOCK[2],
        coarse[3], BLOCK[3],
    )
    source_sha = "a" * 64
    identity = {
        "schema": converter.CANONICAL_SCHEMA,
        "source": {
            "dataset": "lonv",
            "sha256": source_sha,
            "shape": list(source_shape),
            "dtype": "complex64",
            "layout": "blocked odd E12",
            "parity": "odd",
        },
        "geometry": {
            "lattice_xyzt": list(LATTICE),
            "block_xyzt_on_odd_storage": list(BLOCK),
            "nvec": NVEC,
        },
        "reconstruction": {
            "algorithm": "test Clover block elimination",
            "version": 1,
            "rhs": "zero",
            "input_parity": "odd",
        },
        "output": {
            "dataset": "null",
            "shape": list(shape),
            "dtype": "complex64",
            "layout": converter.CANONICAL_LAYOUT,
            "parity": converter.CANONICAL_PARITY,
            "gamma_basis": converter.GAMMA_BASIS,
        },
    }
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset("null", data=values, chunks=(1, *shape[1:]))
        digest = converter._logical_dataset_hash(dataset, "null")
        metadata = {
            "schema": converter.CANONICAL_SCHEMA,
            "identity": identity,
            "source_dataset_sha256": source_sha,
            "output": {**identity["output"], "sha256": digest},
            "gamma_basis": converter.GAMMA_BASIS,
        }
        dataset.attrs["sha256"] = digest
        dataset.attrs["gamma_basis"] = converter.GAMMA_BASIS
        handle.attrs["schema"] = converter.CANONICAL_SCHEMA
        handle.attrs["source_dataset_sha256"] = source_sha
        handle.attrs["output_sha256"] = digest
        handle.attrs["gamma_basis"] = converter.GAMMA_BASIS
        handle.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
    return values


def _adapter(path: Path, *, corrupt_roundtrip: bool = False) -> Path:
    corruption = (
        "data = bytearray(data); data[0] ^= 1; data = bytes(data)"
        if corrupt_roundtrip
        else "pass"
    )
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        f"marker={converter.ADAPTER_MARKER!r}\n"
        "if sys.argv[1] == 'probe':\n"
        " print(marker + json.dumps({'schema':'pyqcu.quda-vectorio-adapter/v1',"
        "'backend':'test-double','qio_enabled':True,'qmp_enabled':True,"
        "'single_rank_only':True,'writer_api':'test'})); raise SystemExit(0)\n"
        "if sys.argv[1] != 'convert' or len(sys.argv) != 10: raise SystemExit(64)\n"
        "raw, qio, roundtrip = map(pathlib.Path, sys.argv[2:5])\n"
        "data = raw.read_bytes()\n"
        "qio.write_bytes(b'FAKE-QIO\\0' + data)\n"
        f"{corruption}\n"
        "roundtrip.write_bytes(data)\n"
        "print(marker + json.dumps({'status':'ok','bytes':len(data)}))\n"
    )
    path.chmod(0o755)
    return path


def _spec(tmp_path: Path, *, adapter: Path | None = None) -> converter.ConversionSpec:
    return converter.ConversionSpec(
        input_path=tmp_path / "canonical.h5",
        output_prefix=tmp_path / "quda-null",
        adapter_path=adapter,
        quda_prefix=tmp_path / "quda-install",
        lattice=LATTICE,
        block=BLOCK,
        nvec=NVEC,
        workspace_root=tmp_path,
        allow_test_adapter=True,
        timeout=30.0,
    )


def test_checkerboard_staging_matches_quda_full_site_index(tmp_path: Path) -> None:
    values = _canonical(tmp_path / "canonical.h5")
    spec = _spec(tmp_path, adapter=tmp_path / "unused")
    info = converter.inspect_canonical(spec)
    assert info["site_subset"] == "full"
    assert info["gamma_basis"] == converter.GAMMA_BASIS

    raw = tmp_path / "packed.raw"
    summary = converter._stage_checkerboard_raw(spec, raw)
    observed = np.fromfile(raw, dtype=np.complex64).reshape(NVEC, -1, 4, 3)

    t, z, y, x = np.indices((4, 4, 4, 4))
    even = ((t + z + y + x).reshape(-1) & 1) == 0
    lex = np.transpose(values[0], (5, 4, 3, 2, 0, 1)).reshape(-1, 4, 3)
    expected = np.concatenate((lex[even], lex[~even]), axis=0)
    np.testing.assert_array_equal(observed[0], expected)
    assert summary["resident_full_vector_count"] == 0
    assert summary["full_batch_resident"] is False
    assert summary["mapped_virtual_bytes"] == 0
    assert summary["staging_output"] == "positional file writes; no full-file mmap"


def test_conversion_publishes_benchmark_compatible_manifest(tmp_path: Path) -> None:
    values = _canonical(tmp_path / "canonical.h5")
    adapter = _adapter(tmp_path / "adapter")
    spec = _spec(tmp_path, adapter=adapter)

    result = converter.convert(spec)

    assert result["status"] == "created"
    assert spec.artifact_path.name == "quda-null_level_0_nvec_2"
    assert spec.artifact_path.is_file()
    manifest = json.loads(spec.manifest_path.read_text())
    assert manifest["schema"] == converter.MANIFEST_SCHEMA
    assert manifest["source_sha256"] == "a" * 64
    assert manifest["canonical_dataset_sha256"] == result["canonical_dataset_sha256"]
    assert manifest["qio_prefix"] == "quda-null"
    assert manifest["block_xyzt"] == list(BLOCK)
    assert manifest["layout"]["site_subset"] == "QUDA_FULL_SITE_SUBSET"
    assert manifest["layout"]["gamma_basis"] == converter.GAMMA_BASIS
    assert manifest["round_trip"]["byte_exact"] is True
    assert manifest["round_trip"]["relative_l2"] == 0.0
    assert manifest["round_trip"]["compared_real_scalars"] == (
        values.nbytes // np.dtype("float32").itemsize
    )
    assert manifest["artifacts"] == [{
        "format": "USQCD QIO singlefile",
        "path": spec.artifact_path.name,
        "sha256": converter._sha256_file(spec.artifact_path),
        "size_bytes": spec.artifact_path.stat().st_size,
    }]
    assert converter.verify_conversion(spec)["status"] == "verified"

    import bench_strict_vs_quda as benchmark

    original_nvec, original_block, original_lattice = (
        benchmark.NVECS, benchmark.BLOCK, benchmark.LATTICE)
    try:
        benchmark.NVECS = NVEC
        benchmark.BLOCK = BLOCK
        benchmark.LATTICE = LATTICE
        accepted = benchmark._verify_quda_nullvec_conversion({
            "inputs": {"quda_qio": {
                "prefix": str(spec.output_prefix),
                "conversion_manifest": str(spec.manifest_path),
            }},
            "input_fingerprints": {"null_vectors": {
                "sha256": manifest["canonical_dataset_sha256"],
            }},
        })
    finally:
        benchmark.NVECS = original_nvec
        benchmark.BLOCK = original_block
        benchmark.LATTICE = original_lattice
    assert accepted["source_sha256"] == "a" * 64
    assert accepted["canonical_dataset_sha256"] == manifest[
        "canonical_dataset_sha256"
    ]
    assert accepted["conversion_tool_commit"] == manifest["conversion_tool_commit"]


def test_e24_odd_basis_is_explicitly_rejected(tmp_path: Path) -> None:
    _canonical(tmp_path / "canonical.h5", source_nvec=2 * NVEC)
    with pytest.raises(converter.ConversionError) as raised:
        converter.inspect_canonical(_spec(tmp_path, adapter=tmp_path / "unused"))
    assert raised.value.code == "e24_odd_basis_forbidden"


def test_missing_real_qio_install_is_structured_failure(tmp_path: Path) -> None:
    _canonical(tmp_path / "canonical.h5")
    spec = _spec(tmp_path, adapter=None)
    with pytest.raises(converter.ConversionError) as raised:
        converter.convert(spec)
    assert raised.value.code == "quda_qio_install_missing"
    assert not spec.artifact_path.exists()
    assert not spec.manifest_path.exists()


def test_non_quda_adapter_is_forbidden_without_test_gate(tmp_path: Path) -> None:
    _canonical(tmp_path / "canonical.h5")
    adapter = _adapter(tmp_path / "adapter")
    base = _spec(tmp_path, adapter=adapter)
    spec = converter.ConversionSpec(
        input_path=base.input_path,
        output_prefix=base.output_prefix,
        adapter_path=adapter,
        quda_prefix=base.quda_prefix,
        lattice=LATTICE,
        block=BLOCK,
        nvec=NVEC,
        workspace_root=tmp_path,
        allow_test_adapter=False,
    )
    with pytest.raises(converter.ConversionError) as raised:
        converter.convert(spec)
    assert raised.value.code == "non_quda_adapter_forbidden"


def test_corrupt_roundtrip_fails_before_publish(tmp_path: Path) -> None:
    _canonical(tmp_path / "canonical.h5")
    adapter = _adapter(tmp_path / "adapter", corrupt_roundtrip=True)
    spec = _spec(tmp_path, adapter=adapter)
    with pytest.raises(converter.ConversionError) as raised:
        converter.convert(spec)
    assert raised.value.code == "qio_roundtrip_mismatch"
    assert not spec.artifact_path.exists()
    assert not spec.manifest_path.exists()


def test_manifest_artifact_tamper_is_detected(tmp_path: Path) -> None:
    _canonical(tmp_path / "canonical.h5")
    spec = _spec(tmp_path, adapter=_adapter(tmp_path / "adapter"))
    converter.convert(spec)
    with spec.artifact_path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(converter.ConversionError) as raised:
        converter.verify_conversion(spec)
    assert raised.value.code == "qio_artifact_size_mismatch"


def test_manifest_must_bind_exact_quda_filename(tmp_path: Path) -> None:
    _canonical(tmp_path / "canonical.h5")
    spec = _spec(tmp_path, adapter=_adapter(tmp_path / "adapter"))
    converter.convert(spec)
    manifest = json.loads(spec.manifest_path.read_text())
    decoy = tmp_path / "decoy.qio"
    decoy.write_bytes(spec.artifact_path.read_bytes())
    manifest["artifacts"][0]["path"] = decoy.name
    spec.manifest_path.write_text(json.dumps(manifest, sort_keys=True))

    with pytest.raises(converter.ConversionError) as raised:
        converter.verify_conversion(spec)
    assert raised.value.code == "conversion_artifact_path_mismatch"


def test_manifest_publish_failure_is_recoverable_without_clobber(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _canonical(tmp_path / "canonical.h5")
    spec = _spec(tmp_path, adapter=_adapter(tmp_path / "adapter"))
    publish = converter._publish_no_replace

    def fail_manifest_once(source: Path, destination: Path) -> None:
        if destination == spec.manifest_path:
            raise converter.ConversionError("injected_manifest_failure", str(destination))
        publish(source, destination)

    monkeypatch.setattr(converter, "_publish_no_replace", fail_manifest_once)
    with pytest.raises(converter.ConversionError) as raised:
        converter.convert(spec)
    assert raised.value.code == "injected_manifest_failure"
    assert spec.artifact_path.is_file()
    assert not spec.manifest_path.exists()
    orphan_sha256 = converter._sha256_file(spec.artifact_path)

    monkeypatch.setattr(converter, "_publish_no_replace", publish)
    result = converter.convert(spec)
    assert result["status"] == "created"
    assert converter._sha256_file(spec.artifact_path) == orphan_sha256
    assert spec.manifest_path.is_file()


def test_workspace_escape_and_existing_output_fail_closed(tmp_path: Path) -> None:
    _canonical(tmp_path / "canonical.h5")
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    spec = converter.ConversionSpec(
        input_path=tmp_path / "canonical.h5",
        output_prefix=outside / "qio",
        adapter_path=tmp_path / "adapter",
        quda_prefix=tmp_path / "quda-install",
        lattice=LATTICE,
        block=BLOCK,
        nvec=NVEC,
        workspace_root=tmp_path,
        allow_test_adapter=True,
    )
    with pytest.raises(converter.ConversionError) as raised:
        converter.inspect_canonical(spec)
    assert raised.value.code == "path_outside_workspace"

    adapter = _adapter(tmp_path / "adapter")
    normal = _spec(tmp_path, adapter=adapter)
    normal.artifact_path.write_bytes(b"preexisting")
    with pytest.raises(converter.ConversionError) as raised:
        converter.convert(normal)
    assert raised.value.code == "output_exists"
    assert normal.artifact_path.read_bytes() == b"preexisting"


def test_cli_dependency_error_is_machine_readable(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _canonical(tmp_path / "canonical.h5")
    rc = converter.main([
        "--input", str(tmp_path / "canonical.h5"),
        "--output-prefix", str(tmp_path / "qio"),
        "--quda-prefix", str(tmp_path / "missing-quda"),
        "--lattice", *map(str, LATTICE),
        "--block", *map(str, BLOCK),
        "--nvec", str(NVEC),
    ])
    # CLI 固定限制到仓库根；测试输入在 pytest tmpdir，先触发边界错误也是结构化失败。
    assert rc == 2
    line = capsys.readouterr().out.strip()
    assert line.startswith(converter.RESULT_MARKER)
    payload = json.loads(line[len(converter.RESULT_MARKER):])
    assert payload["status"] == "failed"
    assert payload["code"] == "path_outside_workspace"


def test_embedded_adapter_requires_real_qio_writer() -> None:
    assert "write_spinor_field" in converter.ADAPTER_CPP
    assert "read_spinor_field" in converter.ADAPTER_CPP
    assert "QUDA_FULL_SITE_SUBSET" in converter.ADAPTER_CPP
    assert "if(NOT QUDA_QIO)" in converter.ADAPTER_CMAKE
    assert "if(NOT QUDA_QMP)" in converter.ADAPTER_CMAKE
    assert "project(pyqcu_quda_qio_adapter LANGUAGES C CXX)" in converter.ADAPTER_CMAKE
    assert "find_package(MPI REQUIRED COMPONENTS C CXX)" in converter.ADAPTER_CMAKE
    assert "cxx_std_20" in converter.ADAPTER_CMAKE
