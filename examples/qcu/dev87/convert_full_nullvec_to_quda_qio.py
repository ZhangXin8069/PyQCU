#!/usr/bin/env python3
"""将 canonical full near-null vectors 转成 QUDA VectorIO/QIO 资产。

默认输入由 ``prepare_fair_nullvec.py`` 生成：

``data/L16x32x32x48_nvec12_full_c64.h5:/null``

输出文件名严格遵循 QUDA ``MG::loadVectors`` 的契约：

``PREFIX_level_0_nvec_12``

本工具不会把 E24 odd basis 补零、改轴或伪装成 full vectors。输入必须是经过
Clover 零右端块消元重构的 ``[nvec,4,3,X,Y,Z,T] complex64`` full field，且
canonical metadata、逻辑 dataset SHA256、原始 E12 身份必须自洽。

QIO 不是裸二进制格式。正式转换必须调用带 QMP+QIO 的 QUDA
``write_spinor_field``（即 ``VectorIO`` 的底层 writer）。可传入已构建 adapter，
也可给工作区内的 QUDA install prefix，由本脚本在临时目录编译小型 adapter。
依赖缺失、round-trip 不精确或协议不匹配时均失败关闭，不发布资产或 manifest。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import h5py
import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

DEFAULT_INPUT = REPO / "data" / "L16x32x32x48_nvec12_full_c64.h5"
DEFAULT_PREFIX = REPO / "data" / "L16x32x32x48_nvec12_quda"
DEFAULT_QUDA_PREFIX = REPO / "data" / "quda-qio-install"

CANONICAL_SCHEMA = "pyqcu.canonical-full-nullvec/v1"
MANIFEST_SCHEMA = "pyqcu.quda-nullvec-conversion/v1"
MANIFEST_VERSION = 1
ADAPTER_SCHEMA = "pyqcu.quda-vectorio-adapter/v1"
HASH_ALGORITHM = "sha256(logical-hdf5-dataset-v1)"
CANONICAL_LAYOUT = "[nvec,spin,color,x,y,z,t]"
CANONICAL_PARITY = "full (even reconstructed; odd copied verbatim)"
QIO_LAYOUT = (
    "QUDA_SPACE_SPIN_COLOR_FIELD_ORDER; full-site checkerboard index "
    "[even lex,odd lex][spin][color][complex], x fastest"
)
GAMMA_BASIS = "QUDA_DEGRAND_ROSSI_GAMMA_BASIS"
ADAPTER_MARKER = "PYQCU_QIO_ADAPTER="
RESULT_MARKER = "PYQCU_QIO_CONVERSION="


class ConversionError(RuntimeError):
    """带稳定错误码的输入、依赖、QIO 或持久化失败。"""

    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = str(code)
        self.detail = str(detail)

    def as_dict(self) -> dict[str, Any]:
        return {"status": "failed", "code": self.code, "detail": self.detail}


@dataclass(frozen=True)
class ConversionSpec:
    input_path: Path = DEFAULT_INPUT
    input_dataset: str = "null"
    output_prefix: Path = DEFAULT_PREFIX
    manifest_path: Optional[Path] = None
    adapter_path: Optional[Path] = None
    quda_prefix: Optional[Path] = DEFAULT_QUDA_PREFIX
    lattice: tuple[int, int, int, int] = (16, 32, 32, 48)
    block: tuple[int, int, int, int] = (2, 2, 2, 2)
    nvec: int = 12
    workspace_root: Path = REPO
    allow_test_adapter: bool = False
    timeout: float = 600.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_path", Path(self.input_path))
        object.__setattr__(self, "output_prefix", Path(self.output_prefix))
        object.__setattr__(self, "workspace_root", Path(self.workspace_root))
        object.__setattr__(self, "lattice", tuple(int(x) for x in self.lattice))
        object.__setattr__(self, "block", tuple(int(x) for x in self.block))
        object.__setattr__(self, "nvec", int(self.nvec))
        object.__setattr__(self, "timeout", float(self.timeout))
        if self.manifest_path is None:
            object.__setattr__(
                self, "manifest_path", Path(f"{self.output_prefix}.conversion.json")
            )
        else:
            object.__setattr__(self, "manifest_path", Path(self.manifest_path))
        if self.adapter_path is not None:
            object.__setattr__(self, "adapter_path", Path(self.adapter_path))
        if self.quda_prefix is not None:
            object.__setattr__(self, "quda_prefix", Path(self.quda_prefix))
        if len(self.lattice) != 4 or any(x <= 0 or x % 2 for x in self.lattice):
            raise ValueError("lattice must contain four positive even extents")
        if len(self.block) != 4 or any(x <= 0 for x in self.block):
            raise ValueError("block must contain four positive extents")
        if self.nvec <= 0:
            raise ValueError("nvec must be positive")
        if not self.input_dataset:
            raise ValueError("input_dataset must be non-empty")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be finite and positive")

    @property
    def artifact_path(self) -> Path:
        return Path(f"{self.output_prefix}_level_0_nvec_{self.nvec}")

    @property
    def expected_shape(self) -> tuple[int, ...]:
        return (self.nvec, 4, 3, *self.lattice)

    @property
    def expected_e12_shape(self) -> tuple[int, ...]:
        odd = (*self.lattice[:3], self.lattice[3] // 2)
        if any(x % b for x, b in zip(odd, self.block)):
            raise ValueError("odd storage lattice must be divisible by block")
        coarse = tuple(x // b for x, b in zip(odd, self.block))
        return (
            self.nvec,
            12,
            coarse[0], self.block[0],
            coarse[1], self.block[1],
            coarse[2], self.block[2],
            coarse[3], self.block[3],
        )


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _decode_attr(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _dataset_digest(dataset: str, shape: Sequence[int], dtype: Any) -> Any:
    digest = hashlib.sha256()
    digest.update(_json_bytes({
        "dataset": str(dataset),
        "shape": tuple(int(x) for x in shape),
        "dtype": str(np.dtype(dtype)),
    }))
    return digest


def _logical_dataset_hash(value: h5py.Dataset, dataset: str) -> str:
    digest = _dataset_digest(dataset, value.shape, value.dtype)
    if value.ndim == 0:
        digest.update(np.ascontiguousarray(value[...]).tobytes(order="C"))
    else:
        for index in range(value.shape[0]):
            digest.update(np.ascontiguousarray(value[index]).tobytes(order="C"))
    return digest.hexdigest()


def _sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _maxrss_bytes(who: int) -> int:
    value = int(resource.getrusage(who).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _regular_file(path: Path, code: str) -> None:
    if path.is_symlink():
        raise ConversionError(code, f"symlink is not allowed: {path}")
    try:
        mode = path.stat().st_mode
    except FileNotFoundError as exc:
        raise ConversionError(code, f"missing file: {path}") from exc
    if not stat.S_ISREG(mode):
        raise ConversionError(code, f"not a regular file: {path}")


def _require_inside(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve(strict=False)
    root_resolved = root.resolve(strict=True)
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ConversionError(
            "path_outside_workspace", f"{label}={resolved} is outside {root_resolved}"
        ) from exc
    return resolved


def _validate_paths(spec: ConversionSpec) -> None:
    root = spec.workspace_root
    for label, path in (
        ("input", spec.input_path),
        ("output_prefix", spec.output_prefix),
        ("artifact", spec.artifact_path),
        ("manifest", spec.manifest_path),
    ):
        _require_inside(Path(path), root, label)
    if spec.adapter_path is not None:
        _require_inside(spec.adapter_path, root, "adapter")
    if spec.quda_prefix is not None:
        _require_inside(spec.quda_prefix, root, "quda_prefix")
    input_path = spec.input_path.resolve(strict=False)
    artifact_path = spec.artifact_path.resolve(strict=False)
    manifest_path = spec.manifest_path.resolve(strict=False)
    if input_path in {artifact_path, manifest_path}:
        raise ConversionError("path_collision", "input and output paths must differ")
    if artifact_path == manifest_path:
        raise ConversionError("path_collision", "QIO artifact and manifest must differ")
    if artifact_path.parent != manifest_path.parent:
        raise ConversionError(
            "publication_parent_mismatch",
            "QIO artifact and manifest must share one parent for recoverable publish",
        )


def _expect_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConversionError("invalid_canonical_metadata", f"{label} must be an object")
    return value


def inspect_canonical(spec: ConversionSpec) -> dict[str, Any]:
    """完整验证 canonical full-vector 身份并返回转换所需摘要。"""
    _validate_paths(spec)
    _regular_file(spec.input_path, "canonical_input_missing")
    try:
        with h5py.File(spec.input_path, "r") as handle:
            schema = _decode_attr(handle.attrs.get("schema", ""))
            if schema != CANONICAL_SCHEMA:
                raise ConversionError(
                    "invalid_canonical_schema",
                    f"schema={schema!r}, expected={CANONICAL_SCHEMA!r}",
                )
            raw_metadata = handle.attrs.get("metadata_json")
            if raw_metadata is None:
                raise ConversionError(
                    "invalid_canonical_metadata", "root metadata_json is missing"
                )
            try:
                metadata = json.loads(_decode_attr(raw_metadata))
            except (TypeError, ValueError) as exc:
                raise ConversionError(
                    "invalid_canonical_metadata", f"metadata_json: {exc}"
                ) from exc
            metadata = _expect_mapping(metadata, "metadata")
            identity = _expect_mapping(metadata.get("identity"), "metadata.identity")
            source = _expect_mapping(identity.get("source"), "identity.source")
            output = _expect_mapping(identity.get("output"), "identity.output")
            geometry = _expect_mapping(identity.get("geometry"), "identity.geometry")
            reconstruction = _expect_mapping(
                identity.get("reconstruction"), "identity.reconstruction"
            )

            source_shape = tuple(int(x) for x in source.get("shape", ()))
            if source_shape and source_shape[0] == 2 * spec.nvec:
                raise ConversionError(
                    "e24_odd_basis_forbidden",
                    "E24 odd-only basis cannot be reinterpreted as 12 full vectors",
                )
            if source_shape != spec.expected_e12_shape:
                raise ConversionError(
                    "invalid_e12_source_contract",
                    f"source shape={source_shape}, expected={spec.expected_e12_shape}",
                )
            if source.get("parity") != "odd" or source.get("dtype") != "complex64":
                raise ConversionError(
                    "invalid_e12_source_contract",
                    f"source parity/dtype={source.get('parity')!r}/{source.get('dtype')!r}",
                )
            source_sha256 = str(source.get("sha256", ""))
            if len(source_sha256) != 64:
                raise ConversionError(
                    "invalid_e12_source_contract", "source dataset SHA256 is missing"
                )
            if reconstruction.get("input_parity") != "odd" or reconstruction.get("rhs") != "zero":
                raise ConversionError(
                    "invalid_full_reconstruction",
                    f"reconstruction={dict(reconstruction)!r}",
                )
            if tuple(geometry.get("lattice_xyzt", ())) != spec.lattice:
                raise ConversionError(
                    "canonical_geometry_mismatch", repr(geometry.get("lattice_xyzt"))
                )
            stored_block = geometry.get("block_xyzt_on_odd_storage")
            if tuple(stored_block or ()) != spec.block or int(geometry.get("nvec", -1)) != spec.nvec:
                raise ConversionError("canonical_geometry_mismatch", repr(dict(geometry)))
            if (output.get("layout") != CANONICAL_LAYOUT or
                    output.get("parity") != CANONICAL_PARITY or
                    output.get("gamma_basis") != GAMMA_BASIS):
                raise ConversionError(
                    "canonical_layout_mismatch",
                    "layout/parity/gamma_basis="
                    f"{output.get('layout')!r}/{output.get('parity')!r}/"
                    f"{output.get('gamma_basis')!r}",
                )
            if tuple(output.get("shape", ())) != spec.expected_shape or output.get("dtype") != "complex64":
                raise ConversionError("canonical_layout_mismatch", repr(dict(output)))
            if spec.input_dataset not in handle or not isinstance(
                handle[spec.input_dataset], h5py.Dataset
            ):
                raise ConversionError(
                    "canonical_dataset_missing", repr(spec.input_dataset)
                )
            value = handle[spec.input_dataset]
            if tuple(int(x) for x in value.shape) != spec.expected_shape:
                raise ConversionError(
                    "canonical_dataset_shape_mismatch",
                    f"shape={value.shape}, expected={spec.expected_shape}",
                )
            if value.dtype != np.dtype("complex64"):
                raise ConversionError(
                    "canonical_dataset_dtype_mismatch", str(value.dtype)
                )
            if (metadata.get("gamma_basis") != GAMMA_BASIS or
                    _decode_attr(handle.attrs.get("gamma_basis", "")) != GAMMA_BASIS or
                    _decode_attr(value.attrs.get("gamma_basis", "")) != GAMMA_BASIS):
                raise ConversionError(
                    "canonical_gamma_basis_mismatch",
                    "canonical metadata must explicitly bind the full vectors "
                    f"to {GAMMA_BASIS}",
                )
            observed = _logical_dataset_hash(value, spec.input_dataset)
            hashes = {
                "metadata.output": _expect_mapping(
                    metadata.get("output"), "metadata.output"
                ).get("sha256"),
                "dataset": _decode_attr(value.attrs.get("sha256", "")),
                "root": _decode_attr(handle.attrs.get("output_sha256", "")),
            }
            if any(item != observed for item in hashes.values()):
                raise ConversionError(
                    "canonical_dataset_digest_mismatch",
                    f"observed={observed}, stored={hashes!r}",
                )
            metadata_source = metadata.get("source_dataset_sha256")
            root_source = _decode_attr(handle.attrs.get("source_dataset_sha256", ""))
            if metadata_source != source_sha256 or root_source != source_sha256:
                raise ConversionError(
                    "canonical_source_digest_mismatch",
                    f"identity={source_sha256}, metadata={metadata_source}, root={root_source}",
                )
    except OSError as exc:
        raise ConversionError("canonical_hdf5_unreadable", repr(exc)) from exc
    stat_result = spec.input_path.stat()
    return {
        "path": str(spec.input_path.resolve()),
        "dataset": spec.input_dataset,
        "shape": list(spec.expected_shape),
        "dtype": "complex64",
        "layout": CANONICAL_LAYOUT,
        "site_subset": "full",
        "parity": "full",
        "gamma_basis": GAMMA_BASIS,
        "basis_transform": "identity",
        "dataset_sha256": observed,
        "source_sha256": source_sha256,
        "source_contract": {
            "kind": "E12 odd-Schur basis",
            "shape": list(spec.expected_e12_shape),
            "parity": "odd",
            "full_reconstruction": dict(reconstruction),
        },
        "file_size_bytes": int(stat_result.st_size),
        "metadata": dict(metadata),
    }


def _pwrite_all(descriptor: int, value: np.ndarray, offset: int) -> None:
    payload = memoryview(np.ascontiguousarray(value)).cast("B")
    written = 0
    while written < len(payload):
        count = os.pwrite(descriptor, payload[written:], offset + written)
        if count <= 0:
            raise OSError("pwrite returned no progress")
        written += count


def _stage_checkerboard_raw(spec: ConversionSpec, output: Path) -> dict[str, Any]:
    """逐 t-slab 定位写 checkerboard order，不映射或加载完整向量。"""
    expected_bytes = math.prod(spec.expected_shape) * np.dtype("complex64").itemsize
    x, y, z, t_extent = spec.lattice
    volume = math.prod(spec.lattice)
    site_bytes = 4 * 3 * np.dtype("complex64").itemsize
    spatial_even = (
        np.arange(z, dtype=np.int16)[:, None, None]
        + np.arange(y, dtype=np.int16)[None, :, None]
        + np.arange(x, dtype=np.int16)[None, None, :]
    ).reshape(-1) % 2 == 0
    descriptor = -1
    maximum_slab_bytes = 0
    maximum_parity_chunk_bytes = 0
    try:
        descriptor = os.open(
            output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.ftruncate(descriptor, expected_bytes)
        with h5py.File(spec.input_path, "r") as source:
            value = source[spec.input_dataset]
            for index in range(spec.nvec):
                even_offset = 0
                odd_offset = volume // 2
                for t in range(t_extent):
                    # [s,c,x,y,z] -> one small x-fast [z,y,x,s,c] slab.
                    source_slab = np.asarray(value[index, ..., t])
                    if (not np.isfinite(source_slab.real).all() or
                            not np.isfinite(source_slab.imag).all()):
                        raise ConversionError(
                            "canonical_nonfinite",
                            f"vector {index}, t={t} contains non-finite values",
                        )
                    slab = np.ascontiguousarray(
                        np.transpose(source_slab, (4, 3, 2, 0, 1))
                    ).reshape(-1, 4, 3)
                    slab_sites = int(slab.shape[0])
                    maximum_slab_bytes = max(
                        maximum_slab_bytes,
                        int(source_slab.nbytes + slab.nbytes),
                    )
                    even = spatial_even if t % 2 == 0 else ~spatial_even
                    count = int(np.count_nonzero(even))
                    even_values = np.ascontiguousarray(slab[even])
                    maximum_parity_chunk_bytes = max(
                        maximum_parity_chunk_bytes, int(even_values.nbytes))
                    _pwrite_all(
                        descriptor, even_values,
                        (index * volume + even_offset) * site_bytes)
                    del even_values
                    odd_values = np.ascontiguousarray(slab[~even])
                    maximum_parity_chunk_bytes = max(
                        maximum_parity_chunk_bytes, int(odd_values.nbytes))
                    _pwrite_all(
                        descriptor, odd_values,
                        (index * volume + odd_offset) * site_bytes)
                    del odd_values, slab, source_slab
                    even_offset += count
                    odd_offset += slab_sites - count
                if even_offset != volume // 2 or odd_offset != volume:
                    raise ConversionError(
                        "checkerboard_staging_mismatch",
                        f"vector={index}, even={even_offset}, odd={odd_offset}",
                    )
        os.fsync(descriptor)
    except OSError as exc:
        raise ConversionError("staging_io_failed", repr(exc)) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    size = output.stat().st_size
    if size != expected_bytes:
        raise ConversionError(
            "staging_size_mismatch", f"size={size}, expected={expected_bytes}"
        )
    return {
        "size_bytes": size,
        "sha256": _sha256_file(output),
        "layout": QIO_LAYOUT,
        "resident_full_vector_count": 0,
        "full_batch_resident": False,
        "staging_output": "positional file writes; no full-file mmap",
        "maximum_transpose_slab": "one t-slice",
        "maximum_slab_pair_bytes": maximum_slab_bytes,
        "maximum_parity_chunk_bytes": maximum_parity_chunk_bytes,
        "mapped_virtual_bytes": 0,
    }


ADAPTER_CPP = r'''#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include <qmp.h>
#include <quda.h>
#include <comm_quda.h>
#include <qio_field.h>

static constexpr const char *marker = "PYQCU_QIO_ADAPTER=";

static std::uint64_t checked_volume(const int X[4]) {
  std::uint64_t v = 1;
  for (int d = 0; d < 4; ++d) {
    if (X[d] <= 0 || (X[d] & 1)) throw std::runtime_error("invalid even lattice");
    v *= static_cast<std::uint64_t>(X[d]);
  }
  return v;
}

int main(int argc, char **argv) {
  if (argc == 2 && std::string(argv[1]) == "probe") {
    std::cout << marker
              << "{\"schema\":\"pyqcu.quda-vectorio-adapter/v1\","
                 "\"backend\":\"quda-qio\",\"qio_enabled\":true,"
                 "\"qmp_enabled\":true,\"single_rank_only\":true,"
                 "\"writer_api\":\"write_spinor_field(VectorIO QIO backend)\","
                 "\"precision\":\"QUDA_SINGLE_PRECISION\"}" << std::endl;
    return 0;
  }
  if (argc != 10 || std::string(argv[1]) != "convert") {
    std::cerr << "usage: adapter convert RAW QIO ROUNDTRIP X Y Z T NVEC\n";
    return 64;
  }
  const char *raw_path = argv[2];
  const char *qio_path = argv[3];
  const char *roundtrip_path = argv[4];
  int X[4] = {std::stoi(argv[5]), std::stoi(argv[6]), std::stoi(argv[7]), std::stoi(argv[8])};
  int nvec = std::stoi(argv[9]);
  if (nvec <= 0) throw std::runtime_error("invalid nvec");
  std::uint64_t bytes = checked_volume(X) * static_cast<std::uint64_t>(nvec) * 24u * sizeof(float);
  int src_fd = open(raw_path, O_RDONLY);
  if (src_fd < 0) throw std::runtime_error(std::string("open raw: ") + std::strerror(errno));
  struct stat source_stat {};
  if (fstat(src_fd, &source_stat) != 0 || static_cast<std::uint64_t>(source_stat.st_size) != bytes)
    throw std::runtime_error("raw size mismatch");
  int dst_fd = open(roundtrip_path, O_RDWR | O_CREAT | O_EXCL, 0600);
  if (dst_fd < 0) throw std::runtime_error(std::string("open roundtrip: ") + std::strerror(errno));
  if (ftruncate(dst_fd, static_cast<off_t>(bytes)) != 0) throw std::runtime_error("ftruncate failed");
  void *src = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, src_fd, 0);
  void *dst = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, dst_fd, 0);
  if (src == MAP_FAILED || dst == MAP_FAILED) throw std::runtime_error("mmap failed");
  if (madvise(src, bytes, MADV_SEQUENTIAL) != 0 ||
      madvise(dst, bytes, MADV_SEQUENTIAL) != 0)
    throw std::runtime_error("madvise sequential failed");
  std::uint64_t vector_bytes = bytes / static_cast<std::uint64_t>(nvec);
  std::vector<const void *> input(nvec);
  std::vector<void *> output(nvec);
  for (int i = 0; i < nvec; ++i) {
    input[i] = static_cast<const char *>(src) + i * vector_bytes;
    output[i] = static_cast<char *>(dst) + i * vector_bytes;
  }

  QMP_thread_level_t provided;
  if (QMP_init_msg_passing(&argc, &argv, QMP_THREAD_FUNNELED, &provided) != QMP_SUCCESS)
    throw std::runtime_error("QMP_init_msg_passing failed");
  int dims[4] = {1, 1, 1, 1};
  int map[4] = {3, 2, 1, 0};
  if (QMP_declare_logical_topology_map(dims, 4, map, 4) != QMP_SUCCESS)
    throw std::runtime_error("QMP_declare_logical_topology_map failed");
  initCommsGridQuda(4, dims, nullptr, nullptr);
  if (QMP_get_number_of_nodes() != 1) throw std::runtime_error("adapter is single-rank only");
  write_spinor_field(qio_path, input.data(), QUDA_SINGLE_PRECISION, X,
                     QUDA_FULL_SITE_SUBSET, QUDA_INVALID_PARITY, 3, 4, nvec, 0, nullptr, false);
  if (madvise(src, bytes, MADV_DONTNEED) != 0)
    throw std::runtime_error("madvise input release failed");
  read_spinor_field(qio_path, output.data(), QUDA_SINGLE_PRECISION, X,
                    QUDA_FULL_SITE_SUBSET, QUDA_INVALID_PARITY, 3, 4, nvec, 0, nullptr);
  if (msync(dst, bytes, MS_SYNC) != 0) throw std::runtime_error("msync failed");
  if (madvise(dst, bytes, MADV_DONTNEED) != 0)
    throw std::runtime_error("madvise output release failed");
  quda::comm_finalize();
  QMP_finalize_msg_passing();
  munmap(src, bytes);
  munmap(dst, bytes);
  close(src_fd);
  close(dst_fd);
  struct rusage usage {};
  long maxrss_kib = getrusage(RUSAGE_SELF, &usage) == 0 ? usage.ru_maxrss : -1;
  std::cout << marker << "{\"status\":\"ok\",\"bytes\":" << bytes
            << ",\"mapped_virtual_bytes\":" << (2u * bytes)
            << ",\"maxrss_kib\":" << maxrss_kib << "}" << std::endl;
  return 0;
}
'''


ADAPTER_CMAKE = r'''cmake_minimum_required(VERSION 3.18)
project(pyqcu_quda_qio_adapter LANGUAGES C CXX)
find_package(MPI REQUIRED COMPONENTS C CXX)
find_package(QUDA CONFIG REQUIRED)
if(NOT QUDA_QIO)
  message(FATAL_ERROR "QUDA was built without QIO")
endif()
if(NOT QUDA_QMP)
  message(FATAL_ERROR "QUDA QIO adapter requires QMP")
endif()
add_executable(pyqcu_quda_qio_adapter adapter.cpp)
target_compile_features(pyqcu_quda_qio_adapter PRIVATE cxx_std_20)
target_link_libraries(pyqcu_quda_qio_adapter PRIVATE QUDA::quda)
'''


def _run_checked(
    command: Sequence[str], *, timeout: float, code: str,
    environment: Optional[Mapping[str, str]] = None,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            [str(x) for x in command],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=None if environment is None else dict(environment),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ConversionError(code, repr(exc)) from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()[-6000:]
        raise ConversionError(code, f"rc={result.returncode}: {detail}")
    return result


def _adapter_environment(spec: ConversionSpec) -> dict[str, str]:
    environment = dict(os.environ)
    library_paths: list[str] = []
    if spec.quda_prefix is not None:
        for name in ("lib", "lib64"):
            candidate = spec.quda_prefix / name
            if candidate.is_dir():
                library_paths.append(str(candidate.resolve()))
    existing = environment.get("LD_LIBRARY_PATH")
    if existing:
        library_paths.append(existing)
    if library_paths:
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(library_paths)
    return environment


def _parse_adapter_output(text: str) -> dict[str, Any]:
    for line in reversed(text.splitlines()):
        if line.startswith(ADAPTER_MARKER):
            try:
                value = json.loads(line[len(ADAPTER_MARKER):])
            except json.JSONDecodeError as exc:
                raise ConversionError("invalid_adapter_response", repr(exc)) from exc
            if not isinstance(value, dict):
                break
            return value
    raise ConversionError("invalid_adapter_response", text[-2000:])


def _find_quda_config(prefix: Path) -> Path:
    candidates = (
        prefix / "lib" / "cmake" / "QUDA" / "QUDAConfig.cmake",
        prefix / "lib64" / "cmake" / "QUDA" / "QUDAConfig.cmake",
    )
    for path in candidates:
        if path.is_file() and not path.is_symlink():
            return path
    raise ConversionError(
        "quda_qio_install_missing",
        f"no QUDAConfig.cmake under workspace prefix {prefix}",
    )


def _build_adapter(spec: ConversionSpec, directory: Path) -> tuple[Path, dict[str, Any]]:
    if spec.quda_prefix is None:
        raise ConversionError(
            "qio_adapter_unavailable", "provide --adapter or --quda-prefix"
        )
    config = _find_quda_config(spec.quda_prefix)
    directory.mkdir(parents=True, exist_ok=False)
    source = directory / "adapter.cpp"
    cmake = directory / "CMakeLists.txt"
    source.write_text(ADAPTER_CPP)
    cmake.write_text(ADAPTER_CMAKE)
    build = directory / "build"
    _run_checked(
        [
            "cmake", "-S", str(directory), "-B", str(build),
            f"-DCMAKE_PREFIX_PATH={spec.quda_prefix.resolve()}",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        timeout=spec.timeout,
        code="qio_adapter_configure_failed",
    )
    _run_checked(
        ["cmake", "--build", str(build), "--parallel", "2"],
        timeout=spec.timeout,
        code="qio_adapter_build_failed",
    )
    executable = build / "pyqcu_quda_qio_adapter"
    _regular_file(executable, "qio_adapter_build_failed")
    return executable, {
        "mode": "built-from-installed-quda",
        "quda_prefix": str(spec.quda_prefix.resolve()),
        "quda_config": str(config.resolve()),
        "quda_config_sha256": _sha256_file(config),
    }


def _probe_adapter(spec: ConversionSpec, adapter: Path) -> dict[str, Any]:
    _regular_file(adapter, "qio_adapter_unavailable")
    if not os.access(adapter, os.X_OK):
        raise ConversionError("qio_adapter_unavailable", f"not executable: {adapter}")
    result = _run_checked(
        [str(adapter), "probe"], timeout=min(spec.timeout, 30.0),
        code="qio_adapter_probe_failed",
        environment=_adapter_environment(spec),
    )
    probe = _parse_adapter_output(result.stdout)
    if probe.get("schema") != ADAPTER_SCHEMA:
        raise ConversionError("qio_adapter_protocol_mismatch", repr(probe))
    if probe.get("qio_enabled") is not True or probe.get("qmp_enabled") is not True:
        raise ConversionError("qio_adapter_capability_missing", repr(probe))
    if probe.get("single_rank_only") is not True:
        raise ConversionError("qio_adapter_protocol_mismatch", repr(probe))
    if probe.get("backend") != "quda-qio" and not spec.allow_test_adapter:
        raise ConversionError(
            "non_quda_adapter_forbidden", f"backend={probe.get('backend')!r}"
        )
    return probe


def _roundtrip_summary(source: Path, observed: Path) -> dict[str, Any]:
    source_size = source.stat().st_size
    observed_size = observed.stat().st_size
    if source_size != observed_size or source_size % np.dtype("float32").itemsize:
        raise ConversionError(
            "qio_roundtrip_size_mismatch",
            f"source={source_size}, roundtrip={observed_size}",
        )
    sum_sq = 0.0
    ref_sq = 0.0
    max_abs = 0.0
    chunk_bytes = 8 << 20
    compared_scalars = 0
    maximum_host_chunk_bytes = 0
    source_digest = hashlib.sha256()
    observed_digest = hashlib.sha256()
    with source.open("rb") as lhs, observed.open("rb") as rhs:
        while True:
            left = lhs.read(chunk_bytes)
            right = rhs.read(chunk_bytes)
            if len(left) != len(right):
                raise ConversionError(
                    "qio_roundtrip_size_mismatch",
                    f"stream chunks differ: {len(left)} != {len(right)}")
            if not left:
                break
            source_digest.update(left)
            observed_digest.update(right)
            a = np.frombuffer(left, dtype=np.float32)
            b = np.frombuffer(right, dtype=np.float32)
            a64 = a.astype(np.float64)
            diff = a64 - b
            if diff.size:
                max_abs = max(max_abs, float(np.max(np.abs(diff))))
            sum_sq += float(np.dot(diff, diff))
            ref_sq += float(np.dot(a64, a64))
            compared_scalars += int(a.size)
            maximum_host_chunk_bytes = max(
                maximum_host_chunk_bytes,
                len(left) + len(right) + int(a64.nbytes + diff.nbytes),
            )
    source_sha = source_digest.hexdigest()
    observed_sha = observed_digest.hexdigest()
    expected_scalars = source_size // np.dtype("float32").itemsize
    if compared_scalars != expected_scalars:
        raise ConversionError(
            "qio_roundtrip_scalar_count_mismatch",
            f"compared={compared_scalars}, expected={expected_scalars}",
        )
    relative_l2 = math.sqrt(sum_sq / ref_sq) if ref_sq > 0.0 else math.sqrt(sum_sq)
    summary = {
        "method": "independent bounded streaming comparison after QUDA QIO read_spinor_field",
        "compared_real_scalars": compared_scalars,
        "chunk_bytes_per_file": chunk_bytes,
        "maximum_host_chunk_bytes": maximum_host_chunk_bytes,
        "source_staging_sha256": source_sha,
        "roundtrip_staging_sha256": observed_sha,
        "byte_exact": source_sha == observed_sha,
        "max_abs": max_abs,
        "relative_l2": relative_l2,
        "gate": "byte_exact for c64 -> c64 VectorIO/QIO round-trip",
    }
    if not summary["byte_exact"] or max_abs != 0.0 or relative_l2 != 0.0:
        raise ConversionError("qio_roundtrip_mismatch", repr(summary))
    return summary


def _git_provenance(repository: Path) -> dict[str, Any]:
    def run(*args: str) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", "-C", str(repository), *args],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        value = result.stdout.strip()
        return value if result.returncode == 0 and value else None

    revision = run("rev-parse", "HEAD")
    status = run("status", "--short", "--untracked-files=no")
    return {
        "path": str(repository.resolve()),
        "revision": revision,
        "dirty": bool(status) if status is not None else None,
    }


def _relative(path: Path, parent: Path) -> str:
    return os.path.relpath(path.resolve(), parent.resolve())


def _publish_no_replace(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except FileExistsError as exc:
        raise ConversionError("output_exists", str(destination)) from exc
    except OSError as exc:
        raise ConversionError("atomic_publish_failed", repr(exc)) from exc


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def convert(spec: ConversionSpec) -> dict[str, Any]:
    """执行真实 QIO 转换、独立 round-trip 检查并原子 no-clobber 发布。"""
    canonical = inspect_canonical(spec)
    spec.artifact_path.parent.mkdir(parents=True, exist_ok=True)
    spec.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if spec.manifest_path.is_symlink() or os.path.lexists(spec.manifest_path):
        raise ConversionError("output_exists", str(spec.manifest_path))
    recovering_orphan = os.path.lexists(spec.artifact_path)
    if recovering_orphan:
        _regular_file(spec.artifact_path, "output_exists")

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(
        prefix=f".{spec.artifact_path.name}.tmp.", dir=spec.artifact_path.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        build_provenance: dict[str, Any]
        if spec.adapter_path is None:
            adapter, build_provenance = _build_adapter(spec, temporary / "adapter")
        else:
            adapter = spec.adapter_path
            build_provenance = {"mode": "provided", "path": str(adapter.resolve())}
        probe = _probe_adapter(spec, adapter)
        build_provenance["adapter_sha256"] = _sha256_file(adapter)
        build_provenance["probe"] = probe

        raw = temporary / "canonical.checkerboard.c64.raw"
        roundtrip = temporary / "roundtrip.checkerboard.c64.raw"
        qio_temporary = temporary / "vectors.qio"
        staging = _stage_checkerboard_raw(spec, raw)
        command = [
            str(adapter), "convert", str(raw), str(qio_temporary), str(roundtrip),
            *(str(x) for x in spec.lattice), str(spec.nvec),
        ]
        result = _run_checked(
            command, timeout=spec.timeout, code="qio_adapter_conversion_failed",
            environment=_adapter_environment(spec),
        )
        response = _parse_adapter_output(result.stdout)
        if response.get("status") != "ok":
            raise ConversionError("qio_adapter_conversion_failed", repr(response))
        _regular_file(qio_temporary, "qio_artifact_missing")
        _regular_file(roundtrip, "qio_roundtrip_missing")
        if qio_temporary.stat().st_size <= 0:
            raise ConversionError("qio_artifact_empty", str(qio_temporary))
        round_trip = _roundtrip_summary(raw, roundtrip)
        artifact = {
            "path": _relative(spec.artifact_path, spec.manifest_path.parent),
            "size_bytes": int(qio_temporary.stat().st_size),
            "sha256": _sha256_file(qio_temporary),
            "format": "USQCD QIO singlefile",
        }
        if recovering_orphan:
            existing_size = int(spec.artifact_path.stat().st_size)
            existing_sha256 = _sha256_file(spec.artifact_path)
            if (existing_size != artifact["size_bytes"] or
                    existing_sha256 != artifact["sha256"]):
                raise ConversionError(
                    "output_exists",
                    "existing orphan QIO does not match the freshly verified "
                    f"asset: {spec.artifact_path}",
                )
        pyqcu_git = _git_provenance(REPO)
        quda_git = _git_provenance(REPO / "refer" / "git-rep" / "quda")
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "schema_version": MANIFEST_VERSION,
            "conversion_tool_commit": pyqcu_git["revision"],
            # Original E12 odd-Schur provenance.  Fair benchmarking binds the
            # converted QIO field to canonical_dataset_sha256 below.
            "source_sha256": canonical["source_sha256"],
            "canonical_dataset_sha256": canonical["dataset_sha256"],
            "canonical_hash_algorithm": HASH_ALGORITHM,
            "nvec": spec.nvec,
            "block_xyzt": list(spec.block),
            "lattice_xyzt": list(spec.lattice),
            "qio_prefix": _relative(spec.output_prefix, spec.manifest_path.parent),
            "expected_quda_filename": spec.artifact_path.name,
            "precision": {
                "canonical": "complex64",
                "qio": "QUDA_SINGLE_PRECISION",
                "real_storage": "float32",
            },
            "layout": {
                "canonical": CANONICAL_LAYOUT,
                "qio_host": QIO_LAYOUT,
                "field_order": "QUDA_SPACE_SPIN_COLOR_FIELD_ORDER",
                "site_subset": "QUDA_FULL_SITE_SUBSET",
                "parity": "full (even and odd; QUDA_INVALID_PARITY metadata)",
                "gamma_basis": GAMMA_BASIS,
                "basis_transform": "identity",
            },
            "canonical": {key: value for key, value in canonical.items() if key != "metadata"},
            "artifacts": [artifact],
            "round_trip": round_trip,
            "memory_strategy": {
                **staging,
                "adapter_io": "file-backed mmap with sequential/DONTNEED advice",
                "adapter_mapped_virtual_bytes": response.get(
                    "mapped_virtual_bytes"),
                "adapter_maxrss_bytes": (
                    None if int(response.get("maxrss_kib", -1)) < 0 else
                    int(response["maxrss_kib"]) * 1024),
                "python_maxrss_bytes_at_publish": _maxrss_bytes(
                    resource.RUSAGE_SELF),
                "rss_scope": (
                    "measured process high-water; page cache and virtual mappings "
                    "are reported separately and are not claimed as resident heap"),
                "temporary_raw_files_removed_after_publish": True,
            },
            "provenance": {
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "tool": str(Path(__file__).resolve()),
                "tool_sha256": _sha256_file(Path(__file__).resolve()),
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "pyqcu_git": pyqcu_git,
                "quda_source_git": quda_git,
                "adapter": build_provenance,
                "adapter_conversion_response": response,
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
        manifest_temporary = temporary / "conversion.json"
        manifest_temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        _fsync_file(qio_temporary)
        _fsync_file(manifest_temporary)
        if not recovering_orphan:
            _publish_no_replace(qio_temporary, spec.artifact_path)
            _fsync_directory(spec.artifact_path.parent)
        # If this second link fails, the verified QIO remains as a recoverable
        # orphan.  A retry regenerates and byte-compares it before publishing
        # the manifest; no existing user file is ever removed or overwritten.
        _publish_no_replace(manifest_temporary, spec.manifest_path)
        _fsync_directory(spec.manifest_path.parent)
    verified = verify_conversion(spec)
    verified["status"] = "created"
    return verified


def verify_conversion(spec: ConversionSpec) -> dict[str, Any]:
    """验证 v1 manifest、canonical 身份和每个 QIO artifact 的大小/摘要。"""
    canonical = inspect_canonical(spec)
    _regular_file(spec.manifest_path, "conversion_manifest_missing")
    try:
        manifest = json.loads(spec.manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ConversionError("invalid_conversion_manifest", repr(exc)) from exc
    if (not isinstance(manifest, dict) or
            manifest.get("schema") != MANIFEST_SCHEMA or
            manifest.get("schema_version") != MANIFEST_VERSION):
        raise ConversionError("invalid_conversion_manifest_schema", repr(manifest))
    if manifest.get("source_sha256") != canonical["source_sha256"]:
        raise ConversionError("source_digest_mismatch", repr(manifest.get("source_sha256")))
    if manifest.get("canonical_dataset_sha256") != canonical["dataset_sha256"]:
        raise ConversionError(
            "canonical_digest_mismatch", repr(manifest.get("canonical_dataset_sha256"))
        )
    if (manifest.get("canonical_hash_algorithm") != HASH_ALGORITHM or
            int(manifest.get("nvec", -1)) != spec.nvec or
            tuple(manifest.get("block_xyzt", ())) != spec.block or
            tuple(manifest.get("lattice_xyzt", ())) != spec.lattice or
            manifest.get("expected_quda_filename") != spec.artifact_path.name):
        raise ConversionError("conversion_config_mismatch", repr(manifest))
    if manifest.get("precision") != {
            "canonical": "complex64",
            "qio": "QUDA_SINGLE_PRECISION",
            "real_storage": "float32",
    }:
        raise ConversionError(
            "conversion_precision_mismatch", repr(manifest.get("precision")))
    layout = manifest.get("layout")
    expected_layout = {
        "canonical": CANONICAL_LAYOUT,
        "qio_host": QIO_LAYOUT,
        "field_order": "QUDA_SPACE_SPIN_COLOR_FIELD_ORDER",
        "site_subset": "QUDA_FULL_SITE_SUBSET",
        "parity": "full (even and odd; QUDA_INVALID_PARITY metadata)",
        "gamma_basis": GAMMA_BASIS,
        "basis_transform": "identity",
    }
    if layout != expected_layout:
        raise ConversionError("conversion_layout_mismatch", repr(layout))
    expected_prefix = Path(str(manifest.get("qio_prefix", "")))
    if not expected_prefix.is_absolute():
        expected_prefix = spec.manifest_path.parent / expected_prefix
    if expected_prefix.resolve() != spec.output_prefix.resolve():
        raise ConversionError("conversion_prefix_mismatch", str(expected_prefix))
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 1:
        raise ConversionError("conversion_artifacts_missing", repr(artifacts))
    expected_artifact = spec.artifact_path.resolve(strict=False)
    verified = []
    for item in artifacts:
        if not isinstance(item, Mapping):
            raise ConversionError("invalid_conversion_artifact", repr(item))
        path = Path(str(item.get("path", "")))
        if not path.is_absolute():
            path = spec.manifest_path.parent / path
        _require_inside(path, spec.workspace_root, "manifest artifact")
        if path.resolve(strict=False) != expected_artifact:
            raise ConversionError(
                "conversion_artifact_path_mismatch",
                f"artifact={path.resolve(strict=False)}, expected={expected_artifact}",
            )
        if item.get("format") != "USQCD QIO singlefile":
            raise ConversionError("invalid_conversion_artifact", repr(item))
        _regular_file(path, "qio_artifact_missing")
        size = path.stat().st_size
        digest = _sha256_file(path)
        if size != int(item.get("size_bytes", -1)):
            raise ConversionError("qio_artifact_size_mismatch", str(path))
        if digest != item.get("sha256"):
            raise ConversionError("qio_artifact_digest_mismatch", str(path))
        verified.append({"path": str(path.resolve()), "size_bytes": size, "sha256": digest})
    round_trip = manifest.get("round_trip")
    if not isinstance(round_trip, Mapping) or round_trip.get("byte_exact") is not True:
        raise ConversionError("roundtrip_evidence_missing", repr(round_trip))
    return {
        "status": "verified",
        "manifest_path": str(spec.manifest_path.resolve()),
        "qio_prefix": str(spec.output_prefix.resolve()),
        "artifacts": verified,
        "source_sha256": canonical["source_sha256"],
        "canonical_dataset_sha256": canonical["dataset_sha256"],
        "round_trip": dict(round_trip),
        "manifest": manifest,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert canonical full near-null vectors to QUDA VectorIO/QIO."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--dataset", default="null")
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_PREFIX)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--quda-prefix", type=Path, default=DEFAULT_QUDA_PREFIX)
    parser.add_argument("--lattice", type=int, nargs=4, default=(16, 32, 32, 48))
    parser.add_argument("--block", type=int, nargs=4, default=(2, 2, 2, 2))
    parser.add_argument("--nvec", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=600.0)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--verify-only", action="store_true")
    return parser


def _spec_from_args(args: argparse.Namespace) -> ConversionSpec:
    return ConversionSpec(
        input_path=args.input,
        input_dataset=args.dataset,
        output_prefix=args.output_prefix,
        manifest_path=args.manifest,
        adapter_path=args.adapter,
        quda_prefix=args.quda_prefix,
        lattice=tuple(args.lattice),
        block=tuple(args.block),
        nvec=args.nvec,
        timeout=args.timeout,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = _parser().parse_args(argv)
        spec = _spec_from_args(args)
        if args.dry_run:
            canonical = inspect_canonical(spec)
            result = {
                "status": "planned",
                "canonical": {key: value for key, value in canonical.items() if key != "metadata"},
                "qio_prefix": str(spec.output_prefix.resolve()),
                "artifact": str(spec.artifact_path.resolve()),
                "manifest": str(spec.manifest_path.resolve()),
                "would_write": False,
            }
        elif args.verify_only:
            result = verify_conversion(spec)
        else:
            result = convert(spec)
    except ConversionError as exc:
        print(RESULT_MARKER + json.dumps(exc.as_dict(), ensure_ascii=False, sort_keys=True))
        return 2
    except (OSError, ValueError) as exc:
        failure = ConversionError("invalid_conversion_request", repr(exc))
        print(RESULT_MARKER + json.dumps(failure.as_dict(), ensure_ascii=False, sort_keys=True))
        return 2
    print(RESULT_MARKER + json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
