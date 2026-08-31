#!/usr/bin/env python3
"""为 PyQCU/QUDA 公平 benchmark 生成 canonical full null vectors。

默认输入是 dev87 的 12 个 odd-Schur blocked ``lonv`` 与同一规范场。
生产路径逐个向量调用已验证的 Clover 零右端块重构：

    psi_e = C_ee^{-1} (kappa D_eo psi_o),  psi_o = lonv

输出为 ``/null=[12,4,3,16,32,32,48] complex64``。写入使用一个
HDF5 writer、同目录唯一临时文件和无覆盖原子发布；既有输出只有在身份、
shape 与逻辑 dataset SHA256 全部匹配时才复用。

常用命令：

  python -B examples/qcu/dev87/prepare_fair_nullvec.py --dry-run
  python -B examples/qcu/dev87/prepare_fair_nullvec.py
  python -B examples/qcu/dev87/prepare_fair_nullvec.py --verify-only

本工具只生成 canonical HDF5；QUDA QIO 转换和 QIO round-trip 校验属于后续阶段。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

import h5py
import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

DEFAULT_SOURCE = REPO / "data" / "L16x32x32x48_lv1_E12_nvi1_t1e-2.h5"
DEFAULT_GAUGE = REPO / "data" / "gauge_16x32x32x48_m0.05_seed42_c64.h5"
DEFAULT_OUTPUT = REPO / "data" / "L16x32x32x48_nvec12_full_c64.h5"

SCHEMA = "pyqcu.canonical-full-nullvec/v1"
SCHEMA_VERSION = 1
HASH_ALGORITHM = "sha256(logical-hdf5-dataset-v1)"
RECONSTRUCTION_ALGORITHM = (
    "Clover zero-rhs block elimination via "
    "applyCloverBistabCgReconstructQcu"
)
RECONSTRUCTION_VERSION = 2
SOURCE_LAYOUT = "[nvec,spin_color,Cx,bx,Cy,by,Cz,bz,Ct_odd,bt]"
OUTPUT_LAYOUT = "[nvec,spin,color,x,y,z,t]"
SOURCE_PARITY = "odd"
OUTPUT_PARITY = "full (even reconstructed; odd copied verbatim)"
GAMMA_BASIS = "QUDA_DEGRAND_ROSSI_GAMMA_BASIS"
CLOVER_BUILDER = "examples.qcu.dev87.common.make_clover_tensors/applyCloversQcu"
CLOVER_SIGMA = 0.1


class PreparationError(RuntimeError):
    """输入、身份、持久化或 reconstruction 协议不满足。"""


@dataclass(frozen=True)
class PreparationSpec:
    """一次 canonical full-vector 生成的完整、可哈希规格。"""

    source_path: Path = DEFAULT_SOURCE
    gauge_path: Path = DEFAULT_GAUGE
    output_path: Path = DEFAULT_OUTPUT
    source_dataset: str = "lonv"
    gauge_dataset: str = "g"
    output_dataset: str = "null"
    lattice: tuple[int, int, int, int] = (16, 32, 32, 48)
    block: tuple[int, int, int, int] = (2, 2, 2, 2)
    nvec: int = 12
    mass: float = 0.05
    device: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_path", Path(self.source_path))
        object.__setattr__(self, "gauge_path", Path(self.gauge_path))
        object.__setattr__(self, "output_path", Path(self.output_path))
        object.__setattr__(self, "lattice", tuple(int(x) for x in self.lattice))
        object.__setattr__(self, "block", tuple(int(x) for x in self.block))
        object.__setattr__(self, "nvec", int(self.nvec))
        object.__setattr__(self, "mass", float(self.mass))
        if len(self.lattice) != 4 or any(x <= 0 or x % 2 for x in self.lattice):
            raise ValueError("lattice must contain four positive even extents")
        if len(self.block) != 4 or any(x <= 0 for x in self.block):
            raise ValueError("block must contain four positive extents")
        if self.nvec <= 0:
            raise ValueError("nvec must be positive")
        if not math.isfinite(self.mass) or abs(2.0 * self.mass + 8.0) < 1.0e-15:
            raise ValueError("mass must be finite and define a finite kappa")
        if not self.source_dataset or not self.gauge_dataset or not self.output_dataset:
            raise ValueError("dataset names must be non-empty")
        if any(extent % width for extent, width in zip(self.odd_shape, self.block)):
            raise ValueError(
                f"odd lattice {self.odd_shape} is not divisible by block {self.block}"
            )

    @property
    def kappa(self) -> float:
        return 1.0 / (2.0 * self.mass + 8.0)

    @property
    def odd_shape(self) -> tuple[int, int, int, int]:
        return (*self.lattice[:3], self.lattice[3] // 2)

    @property
    def expected_source_shape(self) -> tuple[int, ...]:
        coarse = tuple(x // b for x, b in zip(self.odd_shape, self.block))
        return (
            self.nvec,
            12,
            coarse[0], self.block[0],
            coarse[1], self.block[1],
            coarse[2], self.block[2],
            coarse[3], self.block[3],
        )

    @property
    def expected_gauge_shape(self) -> tuple[int, ...]:
        return (2, 3, 3, 4, *self.lattice[:3], self.lattice[3] // 2)

    @property
    def expected_output_shape(self) -> tuple[int, ...]:
        return (self.nvec, 4, 3, *self.lattice)


VectorFactory = Callable[
    [PreparationSpec, Mapping[str, Mapping[str, Any]]], Iterable[np.ndarray]
]


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _dataset_digest(dataset: str, shape: Sequence[int], dtype: Any) -> Any:
    digest = hashlib.sha256()
    digest.update(_json_bytes({
        "dataset": str(dataset),
        "shape": tuple(int(x) for x in shape),
        "dtype": str(np.dtype(dtype)),
    }))
    return digest


def _update_digest(digest: Any, value: Any) -> None:
    digest.update(np.ascontiguousarray(value).tobytes(order="C"))


def _hash_dataset_object(value: h5py.Dataset, dataset: str) -> str:
    shape = tuple(int(x) for x in value.shape)
    digest = _dataset_digest(dataset, shape, value.dtype)
    if value.ndim == 0:
        _update_digest(digest, value[()])
    else:
        for index in range(shape[0]):
            _update_digest(digest, value[index])
    return digest.hexdigest()


def _hash_numpy_array(value: np.ndarray, dataset: str) -> str:
    array = np.asarray(value)
    digest = _dataset_digest(dataset, array.shape, array.dtype)
    if array.ndim == 0:
        _update_digest(digest, array)
    else:
        for index in range(array.shape[0]):
            _update_digest(digest, array[index])
    return digest.hexdigest()


def fingerprint_dataset(path: Path, dataset: str) -> Dict[str, Any]:
    """按第一轴流式计算与 benchmark 一致的逻辑 HDF5 dataset SHA256。"""
    path = Path(path)
    if not path.is_file():
        raise PreparationError(f"missing input file: {path}")
    started = time.perf_counter()
    with h5py.File(path, "r") as handle:
        if dataset not in handle or not isinstance(handle[dataset], h5py.Dataset):
            raise PreparationError(f"{path} does not contain dataset {dataset!r}")
        value = handle[dataset]
        shape = tuple(int(x) for x in value.shape)
        dtype = str(value.dtype)
        sha256 = _hash_dataset_object(value, dataset)
    stat = path.stat()
    return {
        "algorithm": HASH_ALGORITHM,
        "sha256": sha256,
        "path": str(path.resolve()),
        "dataset": dataset,
        "shape": list(shape),
        "dtype": dtype,
        "file_size_bytes": int(stat.st_size),
        "file_mtime_ns": int(stat.st_mtime_ns),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _validate_paths(spec: PreparationSpec) -> None:
    output = spec.output_path.resolve(strict=False)
    for name, path in (("source", spec.source_path), ("gauge", spec.gauge_path)):
        if output == path.resolve(strict=False):
            raise PreparationError(f"output path must differ from {name} input path")


def fingerprint_inputs(spec: PreparationSpec) -> Dict[str, Dict[str, Any]]:
    """验证正式输入的 shape/dtype，并返回其稳定身份指纹。"""
    _validate_paths(spec)
    source = fingerprint_dataset(spec.source_path, spec.source_dataset)
    gauge = fingerprint_dataset(spec.gauge_path, spec.gauge_dataset)
    expected = {
        "source": (list(spec.expected_source_shape), "complex64"),
        "gauge": (list(spec.expected_gauge_shape), "complex64"),
    }
    for name, value in (("source", source), ("gauge", gauge)):
        shape, dtype = expected[name]
        if value["shape"] != shape:
            raise PreparationError(
                f"{name} shape mismatch: observed={value['shape']} expected={shape}"
            )
        if value["dtype"] != dtype:
            raise PreparationError(
                f"{name} dtype mismatch: observed={value['dtype']} expected={dtype}"
            )
    return {"source": source, "gauge": gauge}


def build_identity(
    spec: PreparationSpec,
    fingerprints: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """构造决定输出数值的身份；文件路径与 Git revision 不参与身份。"""
    source = fingerprints["source"]
    gauge = fingerprints["gauge"]
    return {
        "schema": SCHEMA,
        "source": {
            "dataset": spec.source_dataset,
            "sha256": source["sha256"],
            "shape": list(spec.expected_source_shape),
            "dtype": "complex64",
            "layout": SOURCE_LAYOUT,
            "parity": SOURCE_PARITY,
        },
        "gauge": {
            "dataset": spec.gauge_dataset,
            "sha256": gauge["sha256"],
            "shape": list(spec.expected_gauge_shape),
            "dtype": "complex64",
            "layout": "[parity,row_color,col_color,mu,x,y,z,t_half]",
        },
        "clover": {
            "builder": CLOVER_BUILDER,
            "sigma": CLOVER_SIGMA,
            "gauge_sha256": gauge["sha256"],
        },
        "physics": {
            "mass": spec.mass,
            "kappa": spec.kappa,
        },
        "geometry": {
            "lattice_xyzt": list(spec.lattice),
            "block_xyzt_on_odd_storage": list(spec.block),
            "nvec": spec.nvec,
        },
        "reconstruction": {
            "algorithm": RECONSTRUCTION_ALGORITHM,
            "version": RECONSTRUCTION_VERSION,
            "equation": "psi_e=C_ee^-1(kappa*D_eo*psi_o); psi_o=source_odd",
            "rhs": "zero",
            "input_parity": SOURCE_PARITY,
        },
        "output": {
            "dataset": spec.output_dataset,
            "shape": list(spec.expected_output_shape),
            "dtype": "complex64",
            "layout": OUTPUT_LAYOUT,
            "parity": OUTPUT_PARITY,
            "gamma_basis": GAMMA_BASIS,
        },
    }


def _decode_attr(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def verify_canonical_output(
    spec: PreparationSpec,
    identity: Mapping[str, Any],
) -> Dict[str, Any]:
    """完整验证既有输出；任何不匹配都 fail，不修复也不覆盖。"""
    path = spec.output_path
    if path.is_symlink():
        raise PreparationError(f"output must be a regular non-symlink file: {path}")
    if not path.is_file():
        raise PreparationError(f"canonical output is missing: {path}")
    expected_identity_hash = _sha256_json(identity)
    try:
        with h5py.File(path, "r") as handle:
            schema = _decode_attr(handle.attrs.get("schema", ""))
            if schema != SCHEMA:
                raise PreparationError(
                    f"output schema mismatch: observed={schema!r} expected={SCHEMA!r}"
                )
            if int(handle.attrs.get("schema_version", -1)) != SCHEMA_VERSION:
                raise PreparationError("output schema version mismatch")
            raw_metadata = handle.attrs.get("metadata_json")
            if raw_metadata is None:
                raise PreparationError("output metadata_json is missing")
            try:
                metadata = json.loads(_decode_attr(raw_metadata))
            except (TypeError, ValueError) as exc:
                raise PreparationError(f"invalid output metadata_json: {exc}") from exc
            stored_identity = metadata.get("identity")
            stored_identity_hash = metadata.get("identity_sha256")
            if (
                metadata.get("schema") != SCHEMA
                or int(metadata.get("schema_version", -1)) != SCHEMA_VERSION
            ):
                raise PreparationError("output metadata schema/version mismatch")
            if stored_identity != identity or stored_identity_hash != expected_identity_hash:
                raise PreparationError(
                    "output identity mismatch: "
                    f"stored={stored_identity_hash!r} expected={expected_identity_hash!r}"
                )
            root_identity_hash = _decode_attr(handle.attrs.get("identity_sha256", ""))
            if root_identity_hash != expected_identity_hash:
                raise PreparationError("output root identity hash mismatch")
            root_source_hash = _decode_attr(
                handle.attrs.get("source_dataset_sha256", "")
            )
            root_gauge_hash = _decode_attr(handle.attrs.get("gauge_sha256", ""))
            if root_source_hash != identity["source"]["sha256"]:
                raise PreparationError("output root source dataset SHA256 mismatch")
            if root_gauge_hash != identity["gauge"]["sha256"]:
                raise PreparationError("output root gauge SHA256 mismatch")
            if spec.output_dataset not in handle:
                raise PreparationError(
                    f"output dataset {spec.output_dataset!r} is missing"
                )
            value = handle[spec.output_dataset]
            if tuple(int(x) for x in value.shape) != spec.expected_output_shape:
                raise PreparationError(
                    f"output shape mismatch: observed={value.shape} "
                    f"expected={spec.expected_output_shape}"
                )
            if str(value.dtype) != "complex64":
                raise PreparationError(
                    f"output dtype mismatch: observed={value.dtype} expected=complex64"
                )
            observed_hash = _hash_dataset_object(value, spec.output_dataset)
            metadata_hash = metadata.get("output", {}).get("sha256")
            dataset_hash = _decode_attr(value.attrs.get("sha256", ""))
            root_hash = _decode_attr(handle.attrs.get("output_sha256", ""))
            if not (
                observed_hash == metadata_hash == dataset_hash == root_hash
            ):
                raise PreparationError(
                    "output hash mismatch: "
                    f"observed={observed_hash} metadata={metadata_hash!r} "
                    f"dataset={dataset_hash!r} root={root_hash!r}"
                )
            if _decode_attr(value.attrs.get("sha256_algorithm", "")) != HASH_ALGORITHM:
                raise PreparationError("output dataset hash algorithm mismatch")
            if _decode_attr(value.attrs.get("layout", "")) != OUTPUT_LAYOUT:
                raise PreparationError("output dataset layout metadata mismatch")
            if _decode_attr(value.attrs.get("parity", "")) != OUTPUT_PARITY:
                raise PreparationError("output dataset parity metadata mismatch")
            if (_decode_attr(value.attrs.get("gamma_basis", "")) != GAMMA_BASIS or
                    _decode_attr(handle.attrs.get("gamma_basis", "")) != GAMMA_BASIS or
                    metadata.get("gamma_basis") != GAMMA_BASIS):
                raise PreparationError("output gamma-basis metadata mismatch")
            if metadata.get("source_dataset_sha256") != identity["source"]["sha256"]:
                raise PreparationError("metadata source dataset SHA256 mismatch")
            if metadata.get("gauge_sha256") != identity["gauge"]["sha256"]:
                raise PreparationError("metadata gauge SHA256 mismatch")
            revision = metadata.get("provenance", {}).get("git_revision")
            if not isinstance(revision, str) or not revision:
                raise PreparationError("metadata Git revision is missing")
    except OSError as exc:
        raise PreparationError(f"cannot verify canonical output {path}: {exc}") from exc
    return {
        "status": "verified",
        "path": str(path.resolve()),
        "dataset": spec.output_dataset,
        "shape": list(spec.expected_output_shape),
        "dtype": "complex64",
        "identity_sha256": expected_identity_hash,
        "output_sha256": observed_hash,
        "metadata": metadata,
    }


def _verify_after_publish(
    spec: PreparationSpec,
    identity: Mapping[str, Any],
    *,
    attempts: int = 20,
    delay_seconds: float = 0.05,
    verifier: Optional[Callable[[PreparationSpec, Mapping[str, Any]], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Retry only the short-lived HDF5 lock observed after hard-link publish."""
    if attempts < 1 or delay_seconds < 0.0:
        raise ValueError("invalid post-publish verification retry policy")
    check = verify_canonical_output if verifier is None else verifier
    last: Optional[PreparationError] = None
    for attempt in range(attempts):
        try:
            return check(spec, identity)
        except PreparationError as exc:
            cause = exc.__cause__
            if not isinstance(cause, OSError) or cause.errno != 11:
                raise
            last = exc
            if attempt + 1 < attempts:
                time.sleep(delay_seconds)
    assert last is not None
    raise last


def _git_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PreparationError(f"cannot determine Git revision: {exc}") from exc
    revision = result.stdout.strip()
    if result.returncode != 0 or not revision:
        raise PreparationError(
            f"cannot determine Git revision: {result.stderr.strip() or result.returncode}"
        )
    return revision


def _metadata(
    spec: PreparationSpec,
    fingerprints: Mapping[str, Mapping[str, Any]],
    identity: Mapping[str, Any],
    output_sha256: str,
    git_revision: str,
) -> Dict[str, Any]:
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "identity": identity,
        "identity_sha256": _sha256_json(identity),
        "source_dataset_sha256": identity["source"]["sha256"],
        "gauge_sha256": identity["gauge"]["sha256"],
        "physics": identity["physics"],
        "layout": OUTPUT_LAYOUT,
        "parity": OUTPUT_PARITY,
        "gamma_basis": GAMMA_BASIS,
        "reconstruction": identity["reconstruction"],
        "clover": identity["clover"],
        "memory_strategy": {
            "mode": "stream-one-vector",
            "resident_full_vector_count": 1,
            "resident_blocked_vector_count": 1,
            "full_batch_resident": False,
        },
        "output": {**identity["output"], "sha256": output_sha256},
        "inputs": {
            "source_path": fingerprints["source"]["path"],
            "source_dataset": spec.source_dataset,
            "gauge_path": fingerprints["gauge"]["path"],
            "gauge_dataset": spec.gauge_dataset,
        },
        "provenance": {
            "git_revision": git_revision,
            "tool": str(Path(__file__).resolve()),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python": sys.version.split()[0],
        },
    }


def _dataset_chunks(spec: PreparationSpec) -> tuple[int, ...]:
    _, spin, color, _x, y, z, t = spec.expected_output_shape
    return (1, spin, color, 1, min(y, 16), min(z, 16), min(t, 16))


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_temporary(
    temporary: Path,
    spec: PreparationSpec,
    fingerprints: Mapping[str, Mapping[str, Any]],
    identity: Mapping[str, Any],
    vectors: Iterable[np.ndarray],
    git_revision: str,
) -> str:
    """用唯一一个 HDF5 writer 写完 dataset 与全部元数据。"""
    digest = _dataset_digest(
        spec.output_dataset, spec.expected_output_shape, np.dtype("complex64")
    )
    iterator = iter(vectors)
    try:
        with h5py.File(temporary, "w") as handle:
            output = handle.create_dataset(
                spec.output_dataset,
                shape=spec.expected_output_shape,
                dtype=np.complex64,
                chunks=_dataset_chunks(spec),
                fletcher32=True,
            )
            for index in range(spec.nvec):
                try:
                    vector = np.asarray(next(iterator))
                except StopIteration as exc:
                    raise PreparationError(
                        f"reconstruction produced {index} vectors; expected {spec.nvec}"
                    ) from exc
                expected = spec.expected_output_shape[1:]
                if vector.shape != expected:
                    raise PreparationError(
                        f"vector {index} shape mismatch: observed={vector.shape} "
                        f"expected={expected}"
                    )
                if vector.dtype != np.dtype("complex64"):
                    raise PreparationError(
                        f"vector {index} dtype mismatch: observed={vector.dtype} "
                        "expected=complex64"
                    )
                if not np.isfinite(vector.real).all() or not np.isfinite(vector.imag).all():
                    raise PreparationError(f"vector {index} contains non-finite values")
                contiguous = np.ascontiguousarray(vector)
                output[index] = contiguous
                _update_digest(digest, contiguous)
            try:
                next(iterator)
            except StopIteration:
                pass
            else:
                raise PreparationError(
                    f"reconstruction produced more than {spec.nvec} vectors"
                )
            output_sha256 = digest.hexdigest()
            metadata = _metadata(
                spec, fingerprints, identity, output_sha256, git_revision
            )
            output.attrs["sha256_algorithm"] = HASH_ALGORITHM
            output.attrs["sha256"] = output_sha256
            output.attrs["layout"] = OUTPUT_LAYOUT
            output.attrs["parity"] = OUTPUT_PARITY
            output.attrs["gamma_basis"] = GAMMA_BASIS
            handle.attrs["schema"] = SCHEMA
            handle.attrs["schema_version"] = SCHEMA_VERSION
            handle.attrs["identity_sha256"] = metadata["identity_sha256"]
            handle.attrs["source_dataset_sha256"] = metadata["source_dataset_sha256"]
            handle.attrs["gauge_sha256"] = metadata["gauge_sha256"]
            handle.attrs["output_sha256"] = output_sha256
            handle.attrs["gamma_basis"] = GAMMA_BASIS
            handle.attrs["metadata_json"] = _json_bytes(metadata).decode("utf-8")
            handle.flush()
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()
    _fsync_file(temporary)
    return output_sha256


def _publish_no_replace(temporary: Path, output: Path) -> bool:
    """同文件系统 hard-link 发布；目标存在时原子失败，绝不覆盖。"""
    try:
        os.link(temporary, output)
    except FileExistsError:
        return False
    except OSError as exc:
        raise PreparationError(
            f"atomic no-replace publish failed for {output}: {exc}"
        ) from exc
    temporary.unlink()
    _fsync_directory(output.parent)
    return True


def _select_cuda_device(torch: Any, requested: Optional[int]) -> Any:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise PreparationError("CUDA is required for Clover block reconstruction")
    if requested is None:
        requested = int(os.environ.get("QCU_DEVICE_ID", "0"))
    if requested < 0 or requested >= torch.cuda.device_count():
        raise PreparationError(
            f"CUDA device {requested} is unavailable; count={torch.cuda.device_count()}"
        )
    torch.cuda.set_device(requested)
    return torch.device("cuda", requested)


def _cuda_vectors(
    spec: PreparationSpec,
    fingerprints: Mapping[str, Mapping[str, Any]],
) -> Iterable[np.ndarray]:
    """逐 vector 重构，避免 12-vector full 张量同时常驻 CPU/GPU。"""
    try:
        import torch
        from pyqcu import tools
        from pyqcu.cuda import define, qcu
    except (ImportError, OSError) as exc:
        raise PreparationError(f"cannot import PyQCU CUDA runtime: {exc}") from exc
    if int(define.size) != 1:
        raise PreparationError(
            f"canonical reconstruction currently requires one MPI rank; got {define.size}"
        )
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    try:
        from common import make_clover_tensors
    except (ImportError, OSError) as exc:
        raise PreparationError(f"cannot import dev87 Clover builder: {exc}") from exc

    device = _select_cuda_device(torch, spec.device)
    with h5py.File(spec.gauge_path, "r") as handle:
        gauge_np = np.ascontiguousarray(handle[spec.gauge_dataset][...])
    observed_gauge = _hash_numpy_array(gauge_np, spec.gauge_dataset)
    if observed_gauge != fingerprints["gauge"]["sha256"]:
        raise PreparationError(
            "gauge changed after preflight: "
            f"observed={observed_gauge} expected={fingerprints['gauge']['sha256']}"
        )

    gauge = torch.from_numpy(gauge_np).to(
        device=device, dtype=torch.complex64
    ).contiguous()
    del gauge_np
    ce = cei = coo = coi = None
    reconstruct_ptrs = reconstruct_params = argv = None
    try:
        ce, cei, coo, coi, _unused_ptrs, params, argv = make_clover_tensors(
            gauge,
            list(spec.lattice),
            spec.mass,
            dtype=torch.complex64,
            data_type=define._LAT_C64_,
        )
        reconstruct_params = params.clone().contiguous()
        reconstruct_params[define._SET_PLAN_] = 1
        reconstruct_ptrs = define.set_ptrs.clone()
        first_set_index = int(reconstruct_params[define._SET_INDEX_])

        parity_shape = (2, 4, 3, *spec.odd_shape)
        zero_rhs = torch.zeros(parity_shape, dtype=torch.complex64, device=device)
        reconstructed_eo = torch.empty_like(zero_rhs)
        source_digest = _dataset_digest(
            spec.source_dataset,
            spec.expected_source_shape,
            np.dtype("complex64"),
        )
        with h5py.File(spec.source_path, "r") as handle:
            source = handle[spec.source_dataset]
            for index in range(spec.nvec):
                blocked_np = np.ascontiguousarray(source[index])
                _update_digest(source_digest, blocked_np)
                blocked = torch.from_numpy(blocked_np).to(device=device).contiguous()
                odd = blocked.reshape(4, 3, *spec.odd_shape)
                reconstruct_params[define._SET_INDEX_] = first_set_index + index
                qcu.applyInitQcu(reconstruct_ptrs, reconstruct_params, argv)
                try:
                    qcu.applyCloverBistabCgReconstructQcu(
                        reconstructed_eo,
                        zero_rhs,
                        odd,
                        gauge,
                        ce,
                        coo,
                        cei,
                        coi,
                        reconstruct_ptrs,
                        reconstruct_params,
                    )
                finally:
                    qcu.applyEndQcu(reconstruct_ptrs, reconstruct_params)
                full = tools.poooxyzt2oooxyzt(reconstructed_eo).contiguous()
                full_np = np.ascontiguousarray(full.detach().cpu().numpy())
                yield full_np
                del blocked, odd, full, full_np
        observed_source = source_digest.hexdigest()
        if observed_source != fingerprints["source"]["sha256"]:
            raise PreparationError(
                "source lonv changed after preflight: "
                f"observed={observed_source} "
                f"expected={fingerprints['source']['sha256']}"
            )
        torch.cuda.synchronize(device)
    finally:
        # Every per-vector LatticeSet is closed in the loop; only tensor
        # references remain here for deterministic cleanup.
        del ce, cei, coo, coi
        del reconstruct_ptrs, reconstruct_params, argv, gauge


def prepare(
    spec: PreparationSpec,
    *,
    mode: str = "create",
    vector_factory: Optional[VectorFactory] = None,
    git_revision_fn: Optional[Callable[[], str]] = None,
) -> Dict[str, Any]:
    """执行 preflight、复用/验证或无覆盖原子生成。"""
    if mode not in {"create", "dry-run", "verify-only"}:
        raise ValueError(f"unsupported mode: {mode}")
    fingerprints = fingerprint_inputs(spec)
    identity = build_identity(spec, fingerprints)
    identity_sha256 = _sha256_json(identity)

    if spec.output_path.is_symlink() or os.path.lexists(spec.output_path):
        verified = _verify_after_publish(spec, identity)
        verified["status"] = {
            "create": "reused",
            "dry-run": "reusable",
            "verify-only": "verified",
        }[mode]
        verified["input_fingerprints"] = fingerprints
        return verified

    if mode == "verify-only":
        raise PreparationError(f"canonical output is missing: {spec.output_path}")
    if mode == "dry-run":
        return {
            "status": "planned",
            "path": str(spec.output_path.resolve()),
            "dataset": spec.output_dataset,
            "shape": list(spec.expected_output_shape),
            "dtype": "complex64",
            "identity_sha256": identity_sha256,
            "input_fingerprints": fingerprints,
            "would_start_cuda": False,
            "would_write": False,
        }

    spec.output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{spec.output_path.name}.tmp.",
        dir=spec.output_path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    factory = _cuda_vectors if vector_factory is None else vector_factory
    revision_fn = _git_revision if git_revision_fn is None else git_revision_fn
    try:
        revision = str(revision_fn()).strip()
        if not revision:
            raise PreparationError("Git revision must be non-empty")
        vectors = factory(spec, fingerprints)
        _write_temporary(
            temporary,
            spec,
            fingerprints,
            identity,
            vectors,
            revision,
        )
        published = _publish_no_replace(temporary, spec.output_path)
        verified = verify_canonical_output(spec, identity)
        verified["status"] = "created" if published else "reused-after-race"
        verified["input_fingerprints"] = fingerprints
        return verified
    finally:
        if temporary.exists():
            temporary.unlink()


def upgrade_gamma_basis_metadata(
    path: Path = DEFAULT_OUTPUT,
    dataset: str = "null",
) -> Dict[str, Any]:
    """为已验证的 schema-v1 canonical 文件补上显式 gamma-basis 契约。

    数值 dataset 不重写；升级前把旧 metadata/attrs 写入同目录 no-clobber
    JSON 旁路备份。任何数值摘要或旧 identity 不一致都会失败关闭。
    """

    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise PreparationError(f"canonical output must be a regular file: {path}")
    backup = Path(f"{path}.pre-gamma-basis-metadata.json")
    try:
        with h5py.File(path, "r") as handle:
            if dataset not in handle or not isinstance(handle[dataset], h5py.Dataset):
                raise PreparationError(f"canonical dataset is missing: {dataset!r}")
            value = handle[dataset]
            observed_hash = _hash_dataset_object(value, dataset)
            raw_metadata = handle.attrs.get("metadata_json")
            if raw_metadata is None:
                raise PreparationError("canonical metadata_json is missing")
            metadata = json.loads(_decode_attr(raw_metadata))
            identity = metadata.get("identity")
            if not isinstance(identity, dict) or not isinstance(
                    identity.get("output"), dict):
                raise PreparationError("canonical identity/output is invalid")
            stored_hashes = (
                metadata.get("output", {}).get("sha256"),
                _decode_attr(value.attrs.get("sha256", "")),
                _decode_attr(handle.attrs.get("output_sha256", "")),
            )
            if any(item != observed_hash for item in stored_hashes):
                raise PreparationError(
                    f"refuse basis upgrade: output digest mismatch {stored_hashes!r}")
            old_identity_hash = _sha256_json(identity)
            if (metadata.get("identity_sha256") != old_identity_hash or
                    _decode_attr(handle.attrs.get("identity_sha256", "")) !=
                    old_identity_hash):
                raise PreparationError("refuse basis upgrade: identity hash mismatch")
            basis_values = (
                identity["output"].get("gamma_basis"),
                metadata.get("gamma_basis"),
                _decode_attr(value.attrs.get("gamma_basis", "")) or None,
                _decode_attr(handle.attrs.get("gamma_basis", "")) or None,
            )
            if any(item not in (None, GAMMA_BASIS) for item in basis_values):
                raise PreparationError(
                    f"refuse basis upgrade: conflicting gamma basis {basis_values!r}")
            already_complete = all(item == GAMMA_BASIS for item in basis_values)
            old_root_attrs = {
                str(name): _decode_attr(item)
                for name, item in handle.attrs.items()
            }
            old_dataset_attrs = {
                str(name): _decode_attr(item)
                for name, item in value.attrs.items()
            }
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        if isinstance(exc, PreparationError):
            raise
        raise PreparationError(f"cannot inspect canonical basis metadata: {exc}") from exc

    if already_complete:
        return {
            "status": "already-upgraded",
            "path": str(path.resolve()),
            "dataset": dataset,
            "output_sha256": observed_hash,
            "identity_sha256": old_identity_hash,
            "gamma_basis": GAMMA_BASIS,
            "backup": str(backup.resolve()) if backup.exists() else None,
        }

    backup_payload = {
        "schema": "pyqcu.canonical-basis-metadata-backup/v1",
        "path": str(path.resolve()),
        "dataset": dataset,
        "output_sha256": observed_hash,
        "identity_sha256": old_identity_hash,
        "root_attrs": old_root_attrs,
        "dataset_attrs": old_dataset_attrs,
        "metadata": metadata,
    }
    backup_text = json.dumps(
        backup_payload, sort_keys=True, ensure_ascii=False,
        allow_nan=False, indent=2) + "\n"
    if backup.exists():
        try:
            existing_backup = json.loads(backup.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise PreparationError(f"basis backup is unreadable: {backup}") from exc
        if existing_backup != backup_payload:
            raise PreparationError(f"basis backup conflicts with current metadata: {backup}")
    else:
        try:
            with backup.open("x") as handle:
                handle.write(backup_text)
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_directory(backup.parent)
        except OSError as exc:
            raise PreparationError(f"cannot publish basis backup: {exc}") from exc

    upgraded_identity = json.loads(json.dumps(identity))
    upgraded_identity["output"]["gamma_basis"] = GAMMA_BASIS
    upgraded_identity_hash = _sha256_json(upgraded_identity)
    upgraded_metadata = json.loads(json.dumps(metadata))
    upgraded_metadata["identity"] = upgraded_identity
    upgraded_metadata["identity_sha256"] = upgraded_identity_hash
    upgraded_metadata["gamma_basis"] = GAMMA_BASIS
    upgraded_metadata["output"]["gamma_basis"] = GAMMA_BASIS
    upgraded_text = _json_bytes(upgraded_metadata).decode("utf-8")
    try:
        with h5py.File(path, "r+") as handle:
            value = handle[dataset]
            value.attrs["gamma_basis"] = GAMMA_BASIS
            handle.attrs["gamma_basis"] = GAMMA_BASIS
            handle.attrs["identity_sha256"] = upgraded_identity_hash
            handle.attrs["metadata_json"] = upgraded_text
            handle.flush()
        _fsync_file(path)
    except OSError as exc:
        raise PreparationError(
            f"basis metadata upgrade failed; recovery metadata is in {backup}: {exc}") from exc

    with h5py.File(path, "r") as handle:
        value = handle[dataset]
        final_metadata = json.loads(_decode_attr(handle.attrs["metadata_json"]))
        if (_hash_dataset_object(value, dataset) != observed_hash or
                final_metadata.get("identity_sha256") != upgraded_identity_hash or
                final_metadata.get("gamma_basis") != GAMMA_BASIS or
                _decode_attr(value.attrs.get("gamma_basis", "")) != GAMMA_BASIS or
                _decode_attr(handle.attrs.get("gamma_basis", "")) != GAMMA_BASIS):
            raise PreparationError(
                f"basis metadata post-verify failed; recovery metadata is in {backup}")
    return {
        "status": "upgraded",
        "path": str(path.resolve()),
        "dataset": dataset,
        "output_sha256": observed_hash,
        "old_identity_sha256": old_identity_hash,
        "identity_sha256": upgraded_identity_hash,
        "gamma_basis": GAMMA_BASIS,
        "backup": str(backup.resolve()),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate canonical full Clover null vectors for fair QUDA comparison."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--source-dataset", default="lonv")
    parser.add_argument("--gauge", type=Path, default=DEFAULT_GAUGE)
    parser.add_argument("--gauge-dataset", default="g")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-dataset", default="null")
    parser.add_argument("--lattice", type=int, nargs=4, default=(16, 32, 32, 48),
                        metavar=("X", "Y", "Z", "T"))
    parser.add_argument("--block", type=int, nargs=4, default=(2, 2, 2, 2),
                        metavar=("BX", "BY", "BZ", "BT"))
    parser.add_argument("--nvec", type=int, default=12)
    parser.add_argument("--mass", type=float, default=0.05)
    parser.add_argument("--device", type=int, default=None,
                        help="CUDA device index; default QCU_DEVICE_ID or 0")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true",
                       help="hash/validate inputs and existing output; no CUDA or writes")
    modes.add_argument("--verify-only", action="store_true",
                       help="verify an existing canonical output; no CUDA or writes")
    modes.add_argument(
        "--upgrade-gamma-basis-metadata", action="store_true",
        help="add the explicit DeGrand-Rossi basis contract to a legacy output",
    )
    return parser


def _spec_from_args(args: argparse.Namespace) -> PreparationSpec:
    return PreparationSpec(
        source_path=args.source,
        gauge_path=args.gauge,
        output_path=args.output,
        source_dataset=args.source_dataset,
        gauge_dataset=args.gauge_dataset,
        output_dataset=args.output_dataset,
        lattice=tuple(args.lattice),
        block=tuple(args.block),
        nvec=args.nvec,
        mass=args.mass,
        device=args.device,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    mode = "verify-only" if args.verify_only else "dry-run" if args.dry_run else "create"
    try:
        if args.upgrade_gamma_basis_metadata:
            result = upgrade_gamma_basis_metadata(args.output, args.output_dataset)
        else:
            result = prepare(_spec_from_args(args), mode=mode)
    except (PreparationError, OSError, ValueError) as exc:
        print(f"prepare_fair_nullvec: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
