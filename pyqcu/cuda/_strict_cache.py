"""Strict MultiGrid CUDA runtime assets 的流式 HDF5 cache。

该模块只持久化求解期真正需要的 packed 资产：fine transition 的 blocked
``V``、每条 transition 的 ``Yhat/(X, X^-1)``，以及递归 transition 的
``V``。写入期间始终只保留一个 HDF5 句柄，并逐张量执行 device -> CPU ->
HDF5；读取先逐块复核所有逻辑内容摘要，再逐张量、逐块执行
HDF5 -> CPU -> device，避免额外聚合整个 hierarchy 的 host/device 副本。

cache 是按 identity 不可变的。发布使用同目录唯一临时文件和 ``link(2)``
的 no-clobber 语义：完整文件才会原子出现，并发写入不会覆盖既有文件。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from itertools import product as cartesian_product
import json
from math import isfinite, prod
from numbers import Integral
import os
from pathlib import Path
import tempfile
from typing import Any

import h5py
import numpy as np
import torch


STRICT_RUNTIME_CACHE_SCHEMA = "pyqcu.strict-runtime-cache"
STRICT_RUNTIME_CACHE_VERSION = 2
STRICT_RUNTIME_TENSOR_DIGEST = "sha256(pyqcu-logical-tensor-v1)"

_STATE_COMPLETE = "complete"
_TENSOR_IO_CHUNK_BYTES = 8 << 20
_SUPPORTED_DTYPES = frozenset({
    "bool", "uint8", "int8", "int16", "int32", "int64",
    "float16", "float32", "float64", "complex64", "complex128",
})
_TORCH_DTYPES = {
    torch.bool: "bool",
    torch.uint8: "uint8",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.complex64: "complex64",
    torch.complex128: "complex128",
}
_TORCH_DTYPES_BY_NAME = {name: dtype for dtype, name in _TORCH_DTYPES.items()}
_EXPECTED_DTYPE_ALIASES = {"c64": "complex64", "c128": "complex128"}
_ROOT_ATTRS = frozenset({
    "schema", "schema_version", "state",
    "identity_json", "identity_sha256",
    "metadata_json", "metadata_sha256",
    "stats_json", "stats_sha256",
    "manifest_json", "manifest_sha256",
})
_DATASET_ATTRS = frozenset({
    "dtype", "shape_json", "nbytes", "digest_algorithm", "sha256",
})
_STRUCTURAL_TENSOR_SPEC_KEYS = frozenset({"shape", "dtype", "nbytes"})
_DIGEST_TENSOR_SPEC_KEYS = frozenset({"digest_algorithm", "sha256"})


class StrictRuntimeCacheConflictError(RuntimeError):
    """目标 cache 已存在，但不能证明它与待写 identity 相同。"""


class _CacheMiss(Exception):
    def __init__(self, reason: str, detail: str):
        self.reason = reason
        self.detail = detail
        super().__init__(detail)


@dataclass
class StrictRuntimeLevelAssets:
    """一条 strict hierarchy transition 的运行期资产。"""

    preconditioned_links: Any
    onsite_pair: Any
    null_vectors: Any | None = None
    level: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": int(self.level),
            "preconditioned_links": self.preconditioned_links,
            "onsite_pair": self.onsite_pair,
            "null_vectors": self.null_vectors,
        }


@dataclass
class StrictRuntimeCacheAssets:
    """一次 cache hit 后返回的完整、device-resident 资产集合。"""

    fine_blocked_v: Any
    levels: tuple[StrictRuntimeLevelAssets, ...]

    @property
    def preconditioned_links(self) -> tuple[Any, ...]:
        return tuple(level.preconditioned_links for level in self.levels)

    @property
    def onsite_pair(self) -> tuple[Any, ...]:
        return tuple(level.onsite_pair for level in self.levels)

    @property
    def null_vectors(self) -> tuple[Any, ...]:
        """仅返回后续 transition 的递归 V；fine V 单独存放。"""
        return tuple(level.null_vectors for level in self.levels[1:])

    def to_runtime_levels(self) -> list[dict[str, Any]]:
        """转换成 ``QcuStrictAssetBinding`` 可消费的 level mappings。"""
        return [level.to_dict() for level in self.levels]


@dataclass
class StrictRuntimeCacheLoadResult:
    """结构化 cache 查询结果；cache 内容问题均表现为明确 miss。"""

    hit: bool
    reason: str
    detail: str
    path: str
    assets: StrictRuntimeCacheAssets | None = None
    identity: dict[str, Any] | None = None
    cached_identity: dict[str, Any] | None = None
    metadata: Any = None
    stats: Any = None
    manifest: dict[str, Any] | None = None
    evidence: dict[str, Any] | None = None

    @property
    def miss_reason(self) -> str | None:
        return None if self.hit else self.reason


@dataclass(frozen=True)
class StrictRuntimeCacheWriteResult:
    """不可变 cache 写入结果。"""

    written: bool
    reason: str
    detail: str
    path: str
    identity_sha256: str
    tensor_count: int
    logical_bytes: int


def _normalise_json(value: Any, name: str) -> Any:
    """转成严格 JSON 数据树；拒绝非字符串 key、NaN/Inf 和对象回退。"""

    active: set[int] = set()

    def visit(item: Any, where: str) -> Any:
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, float):
            if not isfinite(item):
                raise ValueError(f"{where} 不允许 NaN/Inf")
            return item
        if isinstance(item, Mapping):
            marker = id(item)
            if marker in active:
                raise ValueError(f"{where} 含循环引用")
            active.add(marker)
            result: dict[str, Any] = {}
            for key, child in item.items():
                if not isinstance(key, str):
                    raise TypeError(f"{where} 的 JSON object key 必须是 str")
                result[key] = visit(child, f"{where}.{key}")
            active.remove(marker)
            return result
        if isinstance(item, (list, tuple)):
            marker = id(item)
            if marker in active:
                raise ValueError(f"{where} 含循环引用")
            active.add(marker)
            result = [visit(child, f"{where}[{index}]")
                      for index, child in enumerate(item)]
            active.remove(marker)
            return result
        raise TypeError(
            f"{where}={type(item).__name__} 不是 JSON 可序列化值")

    return visit(value, name)


def _json_text(value: Any, name: str) -> tuple[Any, str]:
    normalised = _normalise_json(value, name)
    return normalised, json.dumps(
        normalised, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False)


def _digest(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _dtype_name(value: Any, label: str) -> str:
    if isinstance(value, torch.dtype):
        if value not in _TORCH_DTYPES:
            raise TypeError(f"{label} dtype={value} 不受 HDF5 runtime cache 支持")
        return _TORCH_DTYPES[value]
    try:
        name = np.dtype(value).name
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} dtype={value!r} 无法规范化") from exc
    if name not in _SUPPORTED_DTYPES:
        raise TypeError(f"{label} dtype={name} 不受 runtime cache 支持")
    return name


def _tensor_spec(value: Any, label: str) -> dict[str, Any]:
    if torch.is_tensor(value):
        shape = tuple(int(extent) for extent in value.shape)
        dtype = _dtype_name(value.dtype, label)
    elif isinstance(value, np.ndarray):
        shape = tuple(int(extent) for extent in value.shape)
        dtype = _dtype_name(value.dtype, label)
    else:
        raise TypeError(f"{label} 必须是 torch.Tensor 或 numpy.ndarray")
    if not shape or any(extent <= 0 for extent in shape):
        raise ValueError(f"{label} shape 必须为非空正整数序列，得到 {shape}")
    nbytes = int(prod(shape)) * int(np.dtype(dtype).itemsize)
    return {"shape": list(shape), "dtype": dtype, "nbytes": nbytes}


def _normalise_levels(
        levels: Sequence[Mapping[str, Any] | StrictRuntimeLevelAssets],
) -> tuple[StrictRuntimeLevelAssets, ...]:
    if isinstance(levels, (str, bytes)):
        raise TypeError("levels 必须是 transition 资产序列")
    try:
        values = tuple(levels)
    except TypeError as exc:
        raise TypeError("levels 必须是 transition 资产序列") from exc
    if not values:
        raise ValueError("strict runtime cache 至少需要一条 transition")

    result: list[StrictRuntimeLevelAssets] = []
    for index, value in enumerate(values):
        if isinstance(value, StrictRuntimeLevelAssets):
            level = value
        elif isinstance(value, Mapping):
            missing = [name for name in (
                "preconditioned_links", "onsite_pair") if name not in value]
            if missing:
                raise ValueError(f"levels[{index}] 缺少资产 {missing}")
            level = StrictRuntimeLevelAssets(
                preconditioned_links=value["preconditioned_links"],
                onsite_pair=value["onsite_pair"],
                null_vectors=value.get("null_vectors"),
                level=value.get("level", index),
            )
        else:
            raise TypeError(
                f"levels[{index}] 必须是 mapping 或 StrictRuntimeLevelAssets")
        if (isinstance(level.level, bool) or
                not isinstance(level.level, Integral) or
                int(level.level) != index):
            raise ValueError(
                f"levels[{index}].level 必须等于 transition index {index}")
        if index == 0 and level.null_vectors is not None:
            raise ValueError(
                "transition 0 的 V 必须只通过 fine_blocked_v 提供，避免重复缓存")
        if index > 0 and level.null_vectors is None:
            raise ValueError(
                f"transition {index} 缺少递归所需 null_vectors")
        result.append(StrictRuntimeLevelAssets(
            preconditioned_links=level.preconditioned_links,
            onsite_pair=level.onsite_pair,
            null_vectors=level.null_vectors,
            level=index,
        ))
    return tuple(result)


def _asset_entries(
        fine_blocked_v: Any,
        levels: tuple[StrictRuntimeLevelAssets, ...],
) -> list[tuple[str, Any]]:
    entries: list[tuple[str, Any]] = [
        ("assets/fine_blocked_v", fine_blocked_v),
    ]
    for level in levels:
        prefix = f"assets/levels/{level.level}"
        entries.extend((
            (f"{prefix}/preconditioned_links", level.preconditioned_links),
            (f"{prefix}/onsite_pair", level.onsite_pair),
        ))
        if level.level > 0:
            entries.append((f"{prefix}/null_vectors", level.null_vectors))
    return entries


def _build_manifest(
        fine_blocked_v: Any,
        levels: tuple[StrictRuntimeLevelAssets, ...],
) -> dict[str, Any]:
    tensors: dict[str, dict[str, Any]] = {}
    dtypes: set[str] = set()
    for path, value in _asset_entries(fine_blocked_v, levels):
        spec = _tensor_spec(value, path)
        tensors[path] = spec
        dtypes.add(spec["dtype"])
    if len(dtypes) != 1:
        raise ValueError(
            f"strict runtime 资产 dtype 必须完全一致，得到 {sorted(dtypes)}")
    return {
        "layout": "fine_blocked_v; per-transition Yhat/onsite; transition>=1 V",
        "level_count": len(levels),
        "tensor_count": len(tensors),
        "dtype": next(iter(dtypes)),
        "total_bytes": sum(spec["nbytes"] for spec in tensors.values()),
        "tensors": tensors,
    }


def make_strict_runtime_cache_manifest(
        *, fine_blocked_v: Any,
        levels: Sequence[Mapping[str, Any] | StrictRuntimeLevelAssets],
) -> dict[str, Any]:
    """返回不含内容摘要、可用于加载端的结构 expected-contract。

    文件内 v2 manifest 会额外包含每个 tensor 的 digest algorithm/SHA256；
    调用方在资产尚未生成时只需提供这里返回的 shape/dtype/nbytes 契约。
    """
    return _build_manifest(fine_blocked_v, _normalise_levels(levels))


def _iter_tensor_slices(
        shape: Sequence[int], itemsize: int,
) -> Any:
    """按 C-order 连续区间切片，单块不超过目标 host workspace。"""

    extents = tuple(int(value) for value in shape)
    max_elements = max(1, int(_TENSOR_IO_CHUNK_BYTES) // int(itemsize))
    if prod(extents) <= max_elements:
        yield (slice(None),) * len(extents)
        return

    suffix_elements = 1
    split_axis = len(extents) - 1
    for axis in range(len(extents) - 1, -1, -1):
        if suffix_elements * extents[axis] > max_elements:
            split_axis = axis
            break
        suffix_elements *= extents[axis]
    chunk_extent = max(
        1, min(extents[split_axis], max_elements // suffix_elements))
    prefix_ranges = [range(extent) for extent in extents[:split_axis]]
    prefixes = cartesian_product(*prefix_ranges) if prefix_ranges else [()]
    trailing = (slice(None),) * (len(extents) - split_axis - 1)
    for prefix in prefixes:
        for start in range(0, extents[split_axis], chunk_extent):
            stop = min(extents[split_axis], start + chunk_extent)
            yield (*prefix, slice(start, stop), *trailing)


def _normalise_numpy_chunk(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if not array.dtype.isnative:
        array = array.astype(array.dtype.newbyteorder("="), copy=False)
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    return array


def _source_chunk_to_numpy(
        value: Any, selection: tuple[Any, ...], label: str,
) -> np.ndarray:
    if torch.is_tensor(value):
        host = value.detach()[selection].to(device="cpu").contiguous()
        if hasattr(host, "resolve_conj"):
            host = host.resolve_conj()
        if hasattr(host, "resolve_neg"):
            host = host.resolve_neg()
        array = host.numpy()
    else:
        array = value[selection]
    result = _normalise_numpy_chunk(array)
    _dtype_name(result.dtype, label)
    return result


def _read_dataset_chunk(
        dataset: h5py.Dataset, selection: tuple[Any, ...],
) -> np.ndarray:
    return _normalise_numpy_chunk(dataset[selection])


def _new_tensor_digest(path: str, spec: Mapping[str, Any]) -> Any:
    header = {
        "protocol": STRICT_RUNTIME_TENSOR_DIGEST,
        "path": path,
        "dtype": spec["dtype"],
        "shape": list(spec["shape"]),
        "nbytes": int(spec["nbytes"]),
    }
    _, text = _json_text(header, "tensor_digest_header")
    digest = sha256()
    digest.update(text.encode("utf-8"))
    digest.update(b"\n")
    return digest


def _update_tensor_digest(
        digest: Any, array: np.ndarray, dtype: str,
) -> None:
    canonical_dtype = np.dtype(dtype).newbyteorder("<")
    canonical = np.ascontiguousarray(
        array.astype(canonical_dtype, copy=False))
    digest.update(memoryview(canonical).cast("B"))


def _full_manifest_copy(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "layout": manifest["layout"],
        "level_count": manifest["level_count"],
        "tensor_count": manifest["tensor_count"],
        "dtype": manifest["dtype"],
        "total_bytes": manifest["total_bytes"],
        "tensors": {
            path: dict(spec) for path, spec in manifest["tensors"].items()
        },
    }


def _write_cache_file(
        path: str,
        *,
        identity_text: str,
        metadata_text: str,
        stats_text: str,
        manifest: dict[str, Any],
        fine_blocked_v: Any,
        levels: tuple[StrictRuntimeLevelAssets, ...],
) -> None:
    """使用一个 HDF5 handle 分块写入，并同步生成内容摘要。"""
    full_manifest = _full_manifest_copy(manifest)
    with h5py.File(path, "w") as handle:
        handle.attrs["schema"] = STRICT_RUNTIME_CACHE_SCHEMA
        handle.attrs["schema_version"] = STRICT_RUNTIME_CACHE_VERSION
        handle.attrs["state"] = "writing"
        for name, text in (
                ("identity", identity_text),
                ("metadata", metadata_text),
                ("stats", stats_text)):
            handle.attrs[f"{name}_json"] = text
            handle.attrs[f"{name}_sha256"] = _digest(text)

        assets = handle.create_group("assets")
        assets.create_group("levels")
        for level in levels:
            handle["assets/levels"].create_group(str(level.level))

        for dataset_path, value in _asset_entries(fine_blocked_v, levels):
            expected = manifest["tensors"][dataset_path]
            dataset = handle.create_dataset(
                dataset_path, shape=tuple(expected["shape"]),
                dtype=np.dtype(expected["dtype"]))
            digest = _new_tensor_digest(dataset_path, expected)
            written_bytes = 0
            for selection in _iter_tensor_slices(
                    expected["shape"], np.dtype(expected["dtype"]).itemsize):
                array = _source_chunk_to_numpy(
                    value, selection, dataset_path)
                actual_dtype = _dtype_name(array.dtype, dataset_path)
                if actual_dtype != expected["dtype"]:
                    raise RuntimeError(
                        f"{dataset_path} 在写入期间 dtype 变化："
                        f"expected={expected['dtype']}, actual={actual_dtype}")
                dataset[selection] = array
                written_bytes += int(array.nbytes)
                _update_tensor_digest(digest, array, expected["dtype"])
            if written_bytes != expected["nbytes"]:
                raise RuntimeError(
                    f"{dataset_path} 分块写入字节数不匹配："
                    f"expected={expected['nbytes']}, actual={written_bytes}")
            tensor_digest = digest.hexdigest()
            full_spec = full_manifest["tensors"][dataset_path]
            full_spec["digest_algorithm"] = STRICT_RUNTIME_TENSOR_DIGEST
            full_spec["sha256"] = tensor_digest
            dataset.attrs["dtype"] = expected["dtype"]
            dataset.attrs["shape_json"] = json.dumps(
                expected["shape"], separators=(",", ":"))
            dataset.attrs["nbytes"] = expected["nbytes"]
            dataset.attrs["digest_algorithm"] = STRICT_RUNTIME_TENSOR_DIGEST
            dataset.attrs["sha256"] = tensor_digest
            del dataset

        _, manifest_text = _json_text(full_manifest, "manifest")
        handle.attrs["manifest_json"] = manifest_text
        handle.attrs["manifest_sha256"] = _digest(manifest_text)

        handle.attrs.modify("state", _STATE_COMPLETE)
        handle.flush()


def _fsync_file(path: str) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: str) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _attr_text(attrs: Any, name: str, reason: str) -> str:
    if name not in attrs:
        raise _CacheMiss(reason, f"缺少 HDF5 root attribute {name!r}")
    value = attrs[name]
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise _CacheMiss(reason, f"attribute {name!r} 不是 UTF-8") from exc
    if isinstance(value, (str, np.str_)):
        return str(value)
    raise _CacheMiss(reason, f"attribute {name!r} 必须是 UTF-8 字符串")


def _parse_json_bundle(handle: h5py.File, name: str) -> tuple[Any, str]:
    reason = f"{name}_invalid"
    text = _attr_text(handle.attrs, f"{name}_json", reason)
    expected_digest = _attr_text(handle.attrs, f"{name}_sha256", reason)
    if _digest(text) != expected_digest:
        raise _CacheMiss(reason, f"{name} JSON digest 不匹配")
    try:
        parsed = json.loads(
            text, parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"非法常量 {token}")))
        normalised, canonical = _json_text(parsed, name)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise _CacheMiss(reason, f"{name} JSON 无效：{exc}") from exc
    if canonical != text:
        raise _CacheMiss(reason, f"{name} JSON 不是 canonical encoding")
    return normalised, text


def _validate_root(handle: h5py.File) -> None:
    actual_attrs = set(handle.attrs.keys())
    if actual_attrs != _ROOT_ATTRS:
        missing = sorted(_ROOT_ATTRS - actual_attrs)
        unknown = sorted(actual_attrs - _ROOT_ATTRS)
        raise _CacheMiss(
            "schema_invalid",
            f"root attributes 不完整：missing={missing}, unknown={unknown}")
    schema = _attr_text(handle.attrs, "schema", "schema_mismatch")
    if schema != STRICT_RUNTIME_CACHE_SCHEMA:
        raise _CacheMiss(
            "schema_mismatch",
            f"schema 应为 {STRICT_RUNTIME_CACHE_SCHEMA!r}，得到 {schema!r}")
    version = handle.attrs["schema_version"]
    if (isinstance(version, (bool, np.bool_)) or
            not isinstance(version, Integral)):
        raise _CacheMiss("version_mismatch", "schema_version 必须是整数")
    version = int(version)
    if version != STRICT_RUNTIME_CACHE_VERSION:
        raise _CacheMiss(
            "version_mismatch",
            f"version 应为 {STRICT_RUNTIME_CACHE_VERSION}，得到 {version}")
    state = _attr_text(handle.attrs, "state", "incomplete")
    if state != _STATE_COMPLETE:
        raise _CacheMiss("incomplete", f"cache state={state!r}")


def _expected_tensor_paths(level_count: int) -> set[str]:
    paths = {"assets/fine_blocked_v"}
    for level in range(level_count):
        prefix = f"assets/levels/{level}"
        paths.update((f"{prefix}/preconditioned_links",
                      f"{prefix}/onsite_pair"))
        if level > 0:
            paths.add(f"{prefix}/null_vectors")
    return paths


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and
        all(character in "0123456789abcdef" for character in value)
    )


def _validate_manifest_common(
        value: Any, *, expected_contract: bool,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _CacheMiss("manifest_invalid", "manifest 必须是 JSON object")
    required = {
        "layout", "level_count", "tensor_count", "dtype", "total_bytes",
        "tensors",
    }
    if set(value) != required:
        raise _CacheMiss(
            "manifest_invalid",
            f"manifest keys 不匹配：missing={sorted(required - set(value))}, "
            f"unknown={sorted(set(value) - required)}")
    if value["layout"] != (
            "fine_blocked_v; per-transition Yhat/onsite; transition>=1 V"):
        raise _CacheMiss("manifest_invalid", "manifest layout 不受支持")
    level_count = value["level_count"]
    if (isinstance(level_count, bool) or not isinstance(level_count, int) or
            level_count < 1):
        raise _CacheMiss("manifest_invalid", "level_count 必须是正整数")
    tensors = value["tensors"]
    if not isinstance(tensors, dict):
        raise _CacheMiss("manifest_invalid", "manifest.tensors 必须是 object")
    expected_paths = _expected_tensor_paths(level_count)
    if set(tensors) != expected_paths:
        raise _CacheMiss(
            "asset_layout_mismatch",
            f"tensor paths 不匹配：missing={sorted(expected_paths - set(tensors))}, "
            f"unknown={sorted(set(tensors) - expected_paths)}")
    raw_dtype = value["dtype"]
    if not isinstance(raw_dtype, str):
        raise _CacheMiss(
            "manifest_invalid", f"manifest dtype={raw_dtype!r} 无效")
    dtype = (
        _EXPECTED_DTYPE_ALIASES.get(raw_dtype, raw_dtype)
        if expected_contract else raw_dtype)
    if dtype not in _SUPPORTED_DTYPES:
        raise _CacheMiss(
            "manifest_invalid", f"manifest dtype={raw_dtype!r} 无效")

    total_bytes = 0
    normalised_tensors: dict[str, dict[str, Any]] = {}
    digest_modes: set[bool] = set()
    for path, spec in tensors.items():
        if not isinstance(spec, dict):
            raise _CacheMiss("manifest_invalid", f"{path} tensor spec 无效")
        spec_keys = set(spec)
        has_digest = bool(spec_keys & _DIGEST_TENSOR_SPEC_KEYS)
        allowed_keys = set(_STRUCTURAL_TENSOR_SPEC_KEYS)
        if has_digest:
            allowed_keys.update(_DIGEST_TENSOR_SPEC_KEYS)
        if (spec_keys != allowed_keys or
                (not expected_contract and not has_digest)):
            raise _CacheMiss("manifest_invalid", f"{path} tensor spec 无效")
        digest_modes.add(has_digest)
        shape = spec["shape"]
        if (not isinstance(shape, list) or not shape or
                any(isinstance(x, bool) or not isinstance(x, int) or x <= 0
                    for x in shape)):
            raise _CacheMiss("manifest_invalid", f"{path} shape 无效：{shape!r}")
        raw_spec_dtype = spec["dtype"]
        if not isinstance(raw_spec_dtype, str):
            raise _CacheMiss(
                "manifest_invalid",
                f"{path} dtype={raw_spec_dtype!r} 无效")
        spec_dtype = (
            _EXPECTED_DTYPE_ALIASES.get(raw_spec_dtype, raw_spec_dtype)
            if expected_contract else raw_spec_dtype)
        if spec_dtype != dtype:
            raise _CacheMiss(
                "dtype_mismatch",
                f"{path} dtype={raw_spec_dtype!r} "
                f"与 manifest dtype={raw_dtype!r} 不同")
        expected_nbytes = int(prod(shape)) * int(np.dtype(dtype).itemsize)
        if (isinstance(spec["nbytes"], bool) or
                not isinstance(spec["nbytes"], int) or
                spec["nbytes"] != expected_nbytes):
            raise _CacheMiss(
                "manifest_invalid",
                f"{path} nbytes 应为 {expected_nbytes}，得到 {spec['nbytes']!r}")
        normalised_spec = {
            "shape": list(shape),
            "dtype": dtype,
            "nbytes": expected_nbytes,
        }
        if has_digest:
            if spec["digest_algorithm"] != STRICT_RUNTIME_TENSOR_DIGEST:
                raise _CacheMiss(
                    "manifest_invalid",
                    f"{path} digest_algorithm={spec['digest_algorithm']!r} 无效")
            if not _valid_sha256(spec["sha256"]):
                raise _CacheMiss(
                    "manifest_invalid", f"{path} sha256 不是小写 64 位摘要")
            normalised_spec.update({
                "digest_algorithm": STRICT_RUNTIME_TENSOR_DIGEST,
                "sha256": spec["sha256"],
            })
        normalised_tensors[path] = normalised_spec
        total_bytes += expected_nbytes
    if expected_contract and len(digest_modes) > 1:
        raise _CacheMiss(
            "manifest_invalid", "expected manifest 不允许混合有/无摘要 tensor spec")
    tensor_count = value["tensor_count"]
    if (isinstance(tensor_count, bool) or
            not isinstance(tensor_count, int) or
            tensor_count != len(tensors)):
        raise _CacheMiss("manifest_invalid", "tensor_count 与 tensors 不一致")
    manifest_bytes = value["total_bytes"]
    if (isinstance(manifest_bytes, bool) or
            not isinstance(manifest_bytes, int) or
            manifest_bytes != total_bytes):
        raise _CacheMiss("manifest_invalid", "total_bytes 与 tensor specs 不一致")
    return {
        "layout": value["layout"],
        "level_count": level_count,
        "tensor_count": tensor_count,
        "dtype": dtype,
        "total_bytes": total_bytes,
        "tensors": normalised_tensors,
    }


def _validate_manifest(value: Any) -> dict[str, Any]:
    return _validate_manifest_common(value, expected_contract=False)


def _validate_expected_manifest(value: Any) -> dict[str, Any]:
    return _validate_manifest_common(value, expected_contract=True)


def _manifest_matches_expected(
        manifest: Mapping[str, Any], expected: Mapping[str, Any],
) -> bool:
    for key in ("layout", "level_count", "tensor_count", "dtype", "total_bytes"):
        if manifest[key] != expected[key]:
            return False
    if set(manifest["tensors"]) != set(expected["tensors"]):
        return False
    for path, expected_spec in expected["tensors"].items():
        observed_spec = manifest["tensors"][path]
        if any(observed_spec.get(key) != value
               for key, value in expected_spec.items()):
            return False
    return True


def _dataset_paths(handle: h5py.File, level_count: int) -> dict[str, h5py.Dataset]:
    if set(handle.keys()) != {"assets"} or not isinstance(
            handle["assets"], h5py.Group):
        raise _CacheMiss("asset_layout_mismatch", "root 必须只含 assets group")
    assets = handle["assets"]
    if set(assets.keys()) != {"fine_blocked_v", "levels"}:
        raise _CacheMiss("asset_layout_mismatch", "assets group 内容不匹配")
    if not isinstance(assets["fine_blocked_v"], h5py.Dataset):
        raise _CacheMiss("asset_layout_mismatch", "fine_blocked_v 不是 dataset")
    levels = assets["levels"]
    if not isinstance(levels, h5py.Group):
        raise _CacheMiss("asset_layout_mismatch", "levels 不是 group")
    expected_levels = {str(index) for index in range(level_count)}
    if set(levels.keys()) != expected_levels:
        raise _CacheMiss("asset_layout_mismatch", "levels group 编号不连续")

    datasets = {"assets/fine_blocked_v": assets["fine_blocked_v"]}
    for level in range(level_count):
        group = levels[str(level)]
        if not isinstance(group, h5py.Group):
            raise _CacheMiss(
                "asset_layout_mismatch", f"transition {level} 不是 group")
        names = {"preconditioned_links", "onsite_pair"}
        if level > 0:
            names.add("null_vectors")
        if set(group.keys()) != names:
            raise _CacheMiss(
                "asset_layout_mismatch",
                f"transition {level} datasets 不匹配")
        for name in names:
            item = group[name]
            if not isinstance(item, h5py.Dataset):
                raise _CacheMiss(
                    "asset_layout_mismatch",
                    f"transition {level}/{name} 不是 dataset")
            datasets[f"assets/levels/{level}/{name}"] = item
    return datasets


def _validate_datasets(
        datasets: Mapping[str, h5py.Dataset],
        manifest: Mapping[str, Any],
) -> None:
    for path, spec in manifest["tensors"].items():
        dataset = datasets[path]
        actual_shape = tuple(int(x) for x in dataset.shape)
        expected_shape = tuple(spec["shape"])
        if actual_shape != expected_shape:
            raise _CacheMiss(
                "shape_mismatch",
                f"{path} shape 应为 {expected_shape}，得到 {actual_shape}")
        try:
            actual_dtype = _dtype_name(dataset.dtype, path)
        except TypeError as exc:
            raise _CacheMiss(
                "dtype_mismatch",
                f"{path} 含不受支持的 HDF5 dtype={dataset.dtype}") from exc
        if actual_dtype != spec["dtype"]:
            raise _CacheMiss(
                "dtype_mismatch",
                f"{path} dtype 应为 {spec['dtype']}，得到 {actual_dtype}")
        if set(dataset.attrs.keys()) != _DATASET_ATTRS:
            raise _CacheMiss(
                "dataset_metadata_mismatch",
                f"{path} dataset attributes 不完整")
        stored_dtype = _attr_text(
            dataset.attrs, "dtype", "dataset_metadata_mismatch")
        stored_shape = _attr_text(
            dataset.attrs, "shape_json", "dataset_metadata_mismatch")
        try:
            stored_shape_value = json.loads(stored_shape)
        except json.JSONDecodeError as exc:
            raise _CacheMiss(
                "dataset_metadata_mismatch",
                f"{path} shape_json 无效") from exc
        if stored_dtype != spec["dtype"] or stored_shape_value != spec["shape"]:
            raise _CacheMiss(
                "dataset_metadata_mismatch",
                f"{path} dataset attributes 与 manifest 不一致")
        stored_nbytes = dataset.attrs["nbytes"]
        valid_nbytes = (
            not isinstance(stored_nbytes, (bool, np.bool_)) and
            isinstance(stored_nbytes, Integral) and
            int(stored_nbytes) == spec["nbytes"])
        if not valid_nbytes:
            raise _CacheMiss(
                "dataset_metadata_mismatch",
                f"{path} nbytes attribute 与 manifest 不一致")
        stored_algorithm = _attr_text(
            dataset.attrs, "digest_algorithm",
            "tensor_digest_metadata_mismatch")
        stored_digest = _attr_text(
            dataset.attrs, "sha256", "tensor_digest_metadata_mismatch")
        if (stored_algorithm != spec["digest_algorithm"] or
                stored_digest != spec["sha256"] or
                not _valid_sha256(stored_digest)):
            raise _CacheMiss(
                "tensor_digest_metadata_mismatch",
                f"{path} digest attrs 与 manifest 不一致")


def _verify_tensor_digests(
        datasets: Mapping[str, h5py.Dataset],
        manifest: Mapping[str, Any],
) -> None:
    """在任何 torch/device 分配前，逐块复核全部 tensor 内容。"""

    for path, spec in manifest["tensors"].items():
        dataset = datasets[path]
        digest = _new_tensor_digest(path, spec)
        observed_bytes = 0
        try:
            for selection in _iter_tensor_slices(
                    spec["shape"], np.dtype(spec["dtype"]).itemsize):
                array = _read_dataset_chunk(dataset, selection)
                observed_bytes += int(array.nbytes)
                _update_tensor_digest(digest, array, spec["dtype"])
        except (OSError, ValueError, TypeError) as exc:
            raise _CacheMiss(
                "tensor_read_failed", f"读取 {path} 以复核摘要失败：{exc}") from exc
        if observed_bytes != spec["nbytes"]:
            raise _CacheMiss(
                "tensor_digest_mismatch",
                f"{path} 摘要读取字节数应为 {spec['nbytes']}，"
                f"得到 {observed_bytes}")
        observed = digest.hexdigest()
        if observed != spec["sha256"]:
            raise _CacheMiss(
                "tensor_digest_mismatch",
                f"{path} logical SHA256 不匹配："
                f"expected={spec['sha256']}, observed={observed}")


def _load_dataset_to_device(
        dataset: h5py.Dataset, spec: Mapping[str, Any], device: Any,
) -> Any:
    target = torch.empty(
        tuple(spec["shape"]), dtype=_TORCH_DTYPES_BY_NAME[spec["dtype"]],
        device=device)
    for selection in _iter_tensor_slices(
            spec["shape"], np.dtype(spec["dtype"]).itemsize):
        array = _read_dataset_chunk(dataset, selection)
        host = torch.from_numpy(array)
        target[selection].copy_(host)
        del host, array
    return target


def _read_existing_identity(path: str) -> tuple[dict[str, Any], str]:
    try:
        with h5py.File(path, "r") as handle:
            _validate_root(handle)
            identity, identity_text = _parse_json_bundle(handle, "identity")
            # A same-identity target is reusable only when the complete v2
            # asset bundle is valid.  This closes the race where a concurrent
            # writer publishes a truncated/corrupt file between our initial
            # miss and no-clobber link.
            _parse_json_bundle(handle, "metadata")
            _parse_json_bundle(handle, "stats")
            manifest_value, _ = _parse_json_bundle(handle, "manifest")
            manifest = _validate_manifest(manifest_value)
            datasets = _dataset_paths(handle, manifest["level_count"])
            _validate_datasets(datasets, manifest)
            _verify_tensor_digests(datasets, manifest)
    except _CacheMiss as miss:
        raise StrictRuntimeCacheConflictError(
            f"拒绝覆盖既有 cache {path!r}：{miss.reason}: {miss.detail}") from miss
    except (OSError, ValueError, TypeError) as exc:
        raise StrictRuntimeCacheConflictError(
            f"拒绝覆盖无法验证的既有 cache {path!r}：{exc}") from exc
    if not isinstance(identity, dict):
        raise StrictRuntimeCacheConflictError(
            f"拒绝覆盖既有 cache {path!r}：identity 不是 JSON object")
    return identity, identity_text


def _check_existing_target(path: str, identity_text: str) -> bool:
    """返回是否已有同 identity cache；任何其他既有目标都拒绝覆盖。"""
    if not os.path.lexists(path):
        return False
    if os.path.islink(path) or not os.path.isfile(path):
        raise StrictRuntimeCacheConflictError(
            f"拒绝写入既有非普通 cache 文件 {path!r}")
    cached_identity, cached_text = _read_existing_identity(path)
    if cached_text != identity_text:
        raise StrictRuntimeCacheConflictError(
            f"拒绝覆盖 identity 不匹配的既有 cache {path!r}；"
            f"cached_sha256={_digest(cached_text)}, "
            f"requested_sha256={_digest(identity_text)}, "
            f"cached_identity={cached_identity!r}")
    return True


def save_strict_runtime_cache(
        path: str | os.PathLike[str],
        *,
        identity: Mapping[str, Any],
        fine_blocked_v: Any,
        levels: Sequence[Mapping[str, Any] | StrictRuntimeLevelAssets],
        metadata: Any = None,
        stats: Any = None,
) -> StrictRuntimeCacheWriteResult:
    """流式写入并原子发布一个 identity 不可变的 strict runtime cache。

    若目标已是同 identity 的完整 cache，函数幂等返回 ``already_exists``；
    若目标身份不同、损坏或是符号链接/目录，则抛出
    :class:`StrictRuntimeCacheConflictError`，绝不覆盖。
    """

    target = os.path.abspath(os.fspath(path))
    if not isinstance(identity, Mapping):
        raise TypeError("identity 必须是非空 JSON object")
    identity_value, identity_text = _json_text(identity, "identity")
    if not isinstance(identity_value, dict) or not identity_value:
        raise ValueError("identity 必须是非空 JSON object")
    metadata_value = {} if metadata is None else metadata
    stats_value = {} if stats is None else stats
    _, metadata_text = _json_text(metadata_value, "metadata")
    _, stats_text = _json_text(stats_value, "stats")
    level_values = _normalise_levels(levels)
    manifest = _build_manifest(fine_blocked_v, level_values)
    identity_digest = _digest(identity_text)

    parent = os.path.dirname(target) or os.curdir
    os.makedirs(parent, exist_ok=True)
    if _check_existing_target(target, identity_text):
        return StrictRuntimeCacheWriteResult(
            written=False,
            reason="already_exists",
            detail="同 identity cache 已存在；未覆盖",
            path=target,
            identity_sha256=identity_digest,
            tensor_count=manifest["tensor_count"],
            logical_bytes=manifest["total_bytes"],
        )

    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{Path(target).name}.", suffix=".tmp", dir=parent)
    os.close(descriptor)
    published = False
    try:
        _write_cache_file(
            temporary,
            identity_text=identity_text,
            metadata_text=metadata_text,
            stats_text=stats_text,
            manifest=manifest,
            fine_blocked_v=fine_blocked_v,
            levels=level_values,
        )
        _fsync_file(temporary)
        try:
            # 同目录 hard-link 是原子 no-clobber 发布；随后移除临时名字。
            os.link(temporary, target)
            published = True
        except FileExistsError:
            if _check_existing_target(target, identity_text):
                return StrictRuntimeCacheWriteResult(
                    written=False,
                    reason="concurrent_same_identity",
                    detail="并发写入者已发布同 identity cache；未覆盖",
                    path=target,
                    identity_sha256=identity_digest,
                    tensor_count=manifest["tensor_count"],
                    logical_bytes=manifest["total_bytes"],
                )
            raise AssertionError("unreachable")
        _fsync_directory(parent)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass

    if not published:  # pragma: no cover - defensive invariant
        raise RuntimeError("strict runtime cache 未完成原子发布")
    return StrictRuntimeCacheWriteResult(
        written=True,
        reason="published",
        detail="cache 已由同目录唯一临时文件原子发布",
        path=target,
        identity_sha256=identity_digest,
        tensor_count=manifest["tensor_count"],
        logical_bytes=manifest["total_bytes"],
    )


def _miss_result(
        path: str, reason: str, detail: str,
        *, identity: dict[str, Any], cached_identity: dict[str, Any] | None = None,
) -> StrictRuntimeCacheLoadResult:
    return StrictRuntimeCacheLoadResult(
        hit=False, reason=reason, detail=detail, path=path,
        identity=identity, cached_identity=cached_identity)


def _cache_evidence(
        path: str, *, identity_text: str, metadata_text: str,
        stats_text: str, manifest_text: str,
        manifest: Mapping[str, Any]) -> dict[str, Any]:
    """返回已逐张量验证的、可写入 benchmark JSON 的语义摘要。"""

    return {
        "schema": STRICT_RUNTIME_CACHE_SCHEMA,
        "schema_version": STRICT_RUNTIME_CACHE_VERSION,
        "path": path,
        "file_size_bytes": int(os.stat(path).st_size),
        "identity_sha256": _digest(identity_text),
        "metadata_sha256": _digest(metadata_text),
        "stats_sha256": _digest(stats_text),
        "manifest_sha256": _digest(manifest_text),
        "tensor_count": int(manifest["tensor_count"]),
        "logical_bytes": int(manifest["total_bytes"]),
        "tensor_digests": {
            tensor_path: {
                "digest_algorithm": spec["digest_algorithm"],
                "sha256": spec["sha256"],
                "nbytes": int(spec["nbytes"]),
            }
            for tensor_path, spec in manifest["tensors"].items()
        },
    }


def load_strict_runtime_cache(
        path: str | os.PathLike[str],
        *,
        identity: Mapping[str, Any],
        device: Any = "cpu",
        expected_manifest: Mapping[str, Any] | None = None,
) -> StrictRuntimeCacheLoadResult:
    """严格验证并逐张量加载 cache，返回明确的 hit/miss 原因。

    ``expected_manifest`` 可由 :func:`make_strict_runtime_cache_manifest`
    生成，允许仅含结构字段，也接受 ``c64/c128`` 精度别名。无论是否提供，
    文件内完整 manifest、dataset attrs 和逻辑内容 SHA256 都会在任何 device
    transfer 前交叉校验。cache miss 不抛异常，调用参数非法则直接抛出。
    """

    target = os.path.abspath(os.fspath(path))
    if not isinstance(identity, Mapping):
        raise TypeError("identity 必须是非空 JSON object")
    identity_value, identity_text = _json_text(identity, "identity")
    if not isinstance(identity_value, dict) or not identity_value:
        raise ValueError("identity 必须是非空 JSON object")

    expected_value: dict[str, Any] | None = None
    if expected_manifest is not None:
        try:
            normalised, _ = _json_text(expected_manifest, "expected_manifest")
            expected_value = _validate_expected_manifest(normalised)
        except _CacheMiss as miss:
            raise ValueError(
                f"expected_manifest 非法：{miss.reason}: {miss.detail}") from miss

    if not os.path.lexists(target):
        return _miss_result(
            target, "not_found", "cache 文件不存在", identity=identity_value)
    if os.path.islink(target) or not os.path.isfile(target):
        return _miss_result(
            target, "not_regular_file",
            "cache 路径必须是普通文件且不能是符号链接",
            identity=identity_value)

    cached_identity: dict[str, Any] | None = None
    try:
        with h5py.File(target, "r") as handle:
            _validate_root(handle)
            loaded_identity, loaded_identity_text = _parse_json_bundle(
                handle, "identity")
            if not isinstance(loaded_identity, dict):
                raise _CacheMiss("identity_invalid", "identity 必须是 JSON object")
            cached_identity = loaded_identity
            if loaded_identity_text != identity_text:
                raise _CacheMiss(
                    "identity_mismatch",
                    f"cached_sha256={_digest(loaded_identity_text)}, "
                    f"requested_sha256={_digest(identity_text)}")

            metadata, metadata_text = _parse_json_bundle(handle, "metadata")
            stats, stats_text = _parse_json_bundle(handle, "stats")
            manifest_value, manifest_text = _parse_json_bundle(
                handle, "manifest")
            manifest = _validate_manifest(manifest_value)
            if (expected_value is not None and
                    not _manifest_matches_expected(manifest, expected_value)):
                raise _CacheMiss(
                    "expected_manifest_mismatch",
                    "cache tensor shape/dtype manifest 与调用方期望不一致")

            datasets = _dataset_paths(handle, manifest["level_count"])
            _validate_datasets(datasets, manifest)
            _verify_tensor_digests(datasets, manifest)
            evidence = _cache_evidence(
                target, identity_text=loaded_identity_text,
                metadata_text=metadata_text, stats_text=stats_text,
                manifest_text=manifest_text, manifest=manifest)

            loaded: dict[str, Any] = {}
            for dataset_path in manifest["tensors"]:
                try:
                    tensor = _load_dataset_to_device(
                        datasets[dataset_path],
                        manifest["tensors"][dataset_path], device)
                except (OSError, KeyError) as exc:
                    raise _CacheMiss(
                        "tensor_read_failed",
                        f"读取 {dataset_path} 失败：{exc}") from exc
                except (RuntimeError, TypeError, ValueError) as exc:
                    raise _CacheMiss(
                        "device_transfer_failed",
                        f"{dataset_path} 转移到 device={device!r} 失败：{exc}") from exc
                loaded[dataset_path] = tensor

            levels: list[StrictRuntimeLevelAssets] = []
            for level in range(manifest["level_count"]):
                prefix = f"assets/levels/{level}"
                levels.append(StrictRuntimeLevelAssets(
                    preconditioned_links=loaded[
                        f"{prefix}/preconditioned_links"],
                    onsite_pair=loaded[f"{prefix}/onsite_pair"],
                    null_vectors=(
                        None if level == 0 else
                        loaded[f"{prefix}/null_vectors"]),
                    level=level,
                ))
            assets = StrictRuntimeCacheAssets(
                fine_blocked_v=loaded["assets/fine_blocked_v"],
                levels=tuple(levels),
            )
    except _CacheMiss as miss:
        return _miss_result(
            target, miss.reason, miss.detail, identity=identity_value,
            cached_identity=cached_identity)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        return _miss_result(
            target, "unreadable", f"HDF5 cache 无法读取：{exc}",
            identity=identity_value, cached_identity=cached_identity)

    return StrictRuntimeCacheLoadResult(
        hit=True,
        reason="hit",
        detail="schema/version/identity/shape/dtype/content digest 全部验证通过",
        path=target,
        assets=assets,
        identity=identity_value,
        cached_identity=cached_identity,
        metadata=metadata,
        stats=stats,
        manifest=manifest,
        evidence=evidence,
    )


def inspect_strict_runtime_cache(
        path: str | os.PathLike[str], *, identity: Mapping[str, Any],
        expected_manifest: Mapping[str, Any] | None = None,
) -> StrictRuntimeCacheLoadResult:
    """无 device transfer 地流式验证 cache，供父进程 merge/fair gate 使用。"""

    target = os.path.abspath(os.fspath(path))
    if not isinstance(identity, Mapping):
        raise TypeError("identity 必须是非空 JSON object")
    identity_value, identity_text = _json_text(identity, "identity")
    if not isinstance(identity_value, dict) or not identity_value:
        raise ValueError("identity 必须是非空 JSON object")

    expected_value: dict[str, Any] | None = None
    if expected_manifest is not None:
        try:
            normalised, _ = _json_text(
                expected_manifest, "expected_manifest")
            expected_value = _validate_expected_manifest(normalised)
        except _CacheMiss as miss:
            raise ValueError(
                f"expected_manifest 非法：{miss.reason}: {miss.detail}") from miss

    if not os.path.lexists(target):
        return _miss_result(
            target, "not_found", "cache 文件不存在", identity=identity_value)
    if os.path.islink(target) or not os.path.isfile(target):
        return _miss_result(
            target, "not_regular_file",
            "cache 路径必须是普通文件且不能是符号链接",
            identity=identity_value)

    cached_identity: dict[str, Any] | None = None
    try:
        with h5py.File(target, "r") as handle:
            _validate_root(handle)
            loaded_identity, loaded_identity_text = _parse_json_bundle(
                handle, "identity")
            if not isinstance(loaded_identity, dict):
                raise _CacheMiss("identity_invalid", "identity 必须是 JSON object")
            cached_identity = loaded_identity
            if loaded_identity_text != identity_text:
                raise _CacheMiss(
                    "identity_mismatch",
                    f"cached_sha256={_digest(loaded_identity_text)}, "
                    f"requested_sha256={_digest(identity_text)}")

            metadata, metadata_text = _parse_json_bundle(handle, "metadata")
            stats, stats_text = _parse_json_bundle(handle, "stats")
            manifest_value, manifest_text = _parse_json_bundle(
                handle, "manifest")
            manifest = _validate_manifest(manifest_value)
            if (expected_value is not None and
                    not _manifest_matches_expected(manifest, expected_value)):
                raise _CacheMiss(
                    "expected_manifest_mismatch",
                    "cache tensor shape/dtype manifest 与调用方期望不一致")
            datasets = _dataset_paths(handle, manifest["level_count"])
            _validate_datasets(datasets, manifest)
            _verify_tensor_digests(datasets, manifest)
            evidence = _cache_evidence(
                target, identity_text=loaded_identity_text,
                metadata_text=metadata_text, stats_text=stats_text,
                manifest_text=manifest_text, manifest=manifest)
    except _CacheMiss as miss:
        return _miss_result(
            target, miss.reason, miss.detail, identity=identity_value,
            cached_identity=cached_identity)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        return _miss_result(
            target, "unreadable", f"HDF5 cache 无法读取：{exc}",
            identity=identity_value, cached_identity=cached_identity)

    return StrictRuntimeCacheLoadResult(
        hit=True,
        reason="hit",
        detail="schema/version/identity/shape/dtype/content digest 全部验证通过；未执行 device transfer",
        path=target,
        assets=None,
        identity=identity_value,
        cached_identity=cached_identity,
        metadata=metadata,
        stats=stats,
        manifest=manifest,
        evidence=evidence,
    )


# 与 pyqcu.tools.save/load 命名并存的显式读写别名。
write_strict_runtime_cache = save_strict_runtime_cache
read_strict_runtime_cache = load_strict_runtime_cache


__all__ = [
    "STRICT_RUNTIME_CACHE_SCHEMA",
    "STRICT_RUNTIME_CACHE_VERSION",
    "STRICT_RUNTIME_TENSOR_DIGEST",
    "StrictRuntimeCacheAssets",
    "StrictRuntimeCacheConflictError",
    "StrictRuntimeCacheLoadResult",
    "StrictRuntimeCacheWriteResult",
    "StrictRuntimeLevelAssets",
    "inspect_strict_runtime_cache",
    "load_strict_runtime_cache",
    "make_strict_runtime_cache_manifest",
    "read_strict_runtime_cache",
    "save_strict_runtime_cache",
    "write_strict_runtime_cache",
]
