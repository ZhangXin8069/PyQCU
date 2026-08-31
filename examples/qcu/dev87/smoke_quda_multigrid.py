#!/usr/bin/env python3
"""8^4 单 rank QUDA Clover-MultiGrid Nc=24 setup-only 快速闸门。

本脚本只构造单位规范场并调用 ``dirac.loadGauge``。对 Clover-Wilson 来说，
这一步会建立 QUDA 的 MultiGrid 层级和粗算子；脚本不会调用 ``invert``，因此
输出的 ``setup_seconds`` 不含任何求解时间。所有会影响本闸门的参数都在 setup
前以整列 ``setattr`` 写回并立即读回，setup 后再读回一次，避免 PyQUDA 的
“数组 getter 返回副本”造成静默配置失败。

QUDA 的 QudaMultigridParam 没有独立的 ``coarse_spin`` 成员。Clover-Wilson
默认细自旋为 4，``spin_block_size[0]=2`` 因而表示
``coarse_spin = 4 / 2 = 2``，对应第一粗层颜色
``coarse_color = n_vec[0] * coarse_spin = 12 * 2 = 24``。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA_DIR = REPO / "data"

LATTICE = (8, 8, 8, 8)
FINE_SPIN = 4
MASS = 0.05
SETUP_TOLERANCE = 1.0e-6
SETUP_MAX_ITER = 1000
NVECS = 12
COARSE_SPIN = 2
COARSE_COLOR = NVECS * COARSE_SPIN
BLOCK = (2, 2, 2, 2)
COARSE_LATTICE = tuple(extent // width for extent, width in zip(LATTICE, BLOCK))
LEVELS = 2
NU_PRE = 0
NU_POST = 2
N_BLOCK_ORTHO = 2
DEVICE_PRECISION = "single"
HOST_GAUGE_DTYPE = "complex128"

DEFAULT_RESOURCE_PATH = DATA_DIR / "quda-multigrid-8^4-nc24-resource"
DEFAULT_OUTPUT = HERE / "out" / "quda_multigrid_setup_smoke.json"
PASS_MARKER = "DEV87_QUDA_MG8_NC24_PASS"
SCHEMA = "pyqcu.quda-multigrid-setup-smoke"
SCHEMA_VERSION = 1

_PARAM_ARRAY_FIELDS = (
    "geo_block_size",
    "spin_block_size",
    "n_vec",
    "n_block_ortho",
    "nu_pre",
    "nu_post",
    "setup_use_mma",
    "dslash_use_mma",
    "transfer_use_mma",
    "vec_load",
)
_PRECISION_FIELDS = (
    "cuda_prec",
    "cuda_prec_sloppy",
    "cuda_prec_refinement_sloppy",
    "cuda_prec_precondition",
    "cuda_prec_eigensolver",
)
_CLOVER_PRECISION_FIELDS = tuple(f"clover_{field}" for field in _PRECISION_FIELDS)


class SmokeFailure(RuntimeError):
    """可写入 JSON 的前置、配置或 setup 闸门失败。"""

    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = str(code)
        self.detail = str(detail)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 << 20)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def _file_summary(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise SmokeFailure("library_missing", str(path))
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _load_bench_helpers() -> Any:
    """加载 benchmark 的标准库 helper；此时仍不导入 PyQUDA/CUDA。"""
    text = str(HERE)
    if text not in sys.path:
        sys.path.insert(0, text)
    import bench_strict_vs_quda as bench
    return bench


def _apply_cli_environment(args: argparse.Namespace) -> None:
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    if args.quda_install is not None:
        prefix = Path(args.quda_install).expanduser().resolve()
        os.environ["QUDA_INSTALL"] = str(prefix)
        os.environ["QUDA_PATH"] = str(prefix)
        lib_dir = prefix / "lib"
        old_ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = str(lib_dir) + (
            os.pathsep + old_ld if old_ld else "")


def _resource_path(value: Path | str) -> Path:
    """Resolve a QUDA resource directory and reject paths outside repo/data."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = DATA_DIR / path
    path = path.resolve()
    try:
        path.relative_to(DATA_DIR.resolve())
    except ValueError as exc:
        raise SmokeFailure(
            "resource_outside_repository_data",
            f"resource={path}, required_root={DATA_DIR.resolve()}") from exc
    return path


def _single_rank_report() -> dict[str, Any]:
    observed: dict[str, int] = {}
    for name in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "WORLD_SIZE"):
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            size = int(raw)
        except ValueError:
            continue
        observed[name] = size
        if size > 1:
            raise SmokeFailure(
                "multi_rank_unsupported", f"{name}={size}; smoke is single-rank")
    return {"grid_size": [1, 1, 1, 1], "environment_sizes": observed}


def _quda_prefix_and_library() -> tuple[Path, Path]:
    text = os.environ.get("QUDA_INSTALL") or os.environ.get("QUDA_PATH")
    if not text:
        raise SmokeFailure(
            "quda_install_missing",
            "set QUDA_INSTALL/QUDA_PATH or pass --quda-install")
    prefix = Path(text).expanduser().resolve()
    library = prefix / "lib" / "libquda.so"
    if not library.is_file() or library.is_symlink():
        raise SmokeFailure("quda_library_missing", str(library))
    return prefix, library


def _unit_gauge_qdp(np: Any, lattice: Sequence[int] = LATTICE) -> Any:
    """返回连续 QDP host gauge ``(mu,t,z,y,x,row,col)``，恒为 complex128。"""
    x, y, z, t = (int(value) for value in lattice)
    eye = np.eye(3, dtype=np.complex128)
    return np.ascontiguousarray(np.broadcast_to(eye, (4, t, z, y, x, 3, 3)))


def _normalise(value: Any) -> Any:
    """把 QUDA enum、bytes、嵌套数组变为可比较/可 JSON 化的值。"""
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).split(b"\0", 1)[0].decode("utf-8", errors="replace")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, (bool, int, float, str)):
        return enum_value
    if isinstance(value, Mapping):
        return {str(key): _normalise(item) for key, item in value.items()}
    if isinstance(value, Sequence):
        return [_normalise(item) for item in value]
    try:
        return [_normalise(item) for item in value]
    except TypeError:
        return repr(value)


def _equal(left: Any, right: Any) -> bool:
    return _normalise(left) == _normalise(right)


def _read_array(target: Any, field: str) -> list[Any]:
    if not hasattr(target, field):
        raise SmokeFailure("quda_param_missing", field)
    try:
        value = getattr(target, field)
        if isinstance(value, (str, bytes, bytearray)):
            raise TypeError("not an array")
        return copy.deepcopy(list(value))
    except Exception as exc:
        raise SmokeFailure("quda_param_read_failed", f"{field}: {exc!r}") from exc


def _set_whole_array(target: Any, field: str, values: Sequence[Any]) -> list[Any]:
    """整列 ``setattr``，随后整列读回并比较，而不是修改 getter 的副本。"""
    expected = copy.deepcopy(list(values))
    if not hasattr(target, field):
        raise SmokeFailure("quda_param_missing", field)
    try:
        setattr(target, field, copy.deepcopy(expected))
        observed = _read_array(target, field)
    except SmokeFailure:
        raise
    except Exception as exc:
        raise SmokeFailure(
            "quda_param_assignment_failed", f"{field}: {exc!r}") from exc
    if not _equal(observed, expected):
        raise SmokeFailure(
            "quda_param_readback_mismatch",
            f"{field}: observed={_normalise(observed)!r}, "
            f"expected={_normalise(expected)!r}")
    return _normalise(observed)


def _set_array_item_whole_column(
        target: Any, field: str, index: int, value: Any) -> list[Any]:
    """修改一项但仍整列写回/整列读回，覆盖 PyQUDA getter-copy 语义。"""
    column = _read_array(target, field)
    if index < 0 or index >= len(column):
        raise SmokeFailure(
            "quda_param_index_missing",
            f"{field}[{index}] unavailable in length {len(column)}")
    column[index] = value
    return _set_whole_array(target, field, column)


def _set_scalar(target: Any, field: str, value: Any) -> Any:
    if not hasattr(target, field):
        raise SmokeFailure("quda_param_missing", field)
    try:
        setattr(target, field, value)
        observed = getattr(target, field)
    except Exception as exc:
        raise SmokeFailure(
            "quda_param_assignment_failed", f"{field}: {exc!r}") from exc
    if not _equal(observed, value):
        raise SmokeFailure(
            "quda_param_readback_mismatch",
            f"{field}: observed={_normalise(observed)!r}, "
            f"expected={_normalise(value)!r}")
    return _normalise(observed)


def _replace_geometry_prefix(row: Sequence[Any], prefix: Sequence[int]) -> list[Any]:
    result = list(row)
    if len(result) < len(prefix):
        raise SmokeFailure("quda_geometry_invalid", repr(row))
    result[:len(prefix)] = [int(value) for value in prefix]
    return result


def _configure_multigrid(
        mg_param: Any, *, boolean_false: Any, boolean_true: Any,
        compute_null_vector_yes: Any) -> dict[str, Any]:
    """配置一层 fine->coarse 转移，并记录每次整列写回的即时结果。"""
    writes: dict[str, Any] = {}
    writes["n_level"] = _set_scalar(mg_param, "n_level", LEVELS)

    geometry = _read_array(mg_param, "geo_block_size")
    if len(geometry) != LEVELS:
        raise SmokeFailure(
            "quda_geometry_level_mismatch",
            f"expected {LEVELS} geometry rows, got {len(geometry)}")
    desired_geometry = copy.deepcopy(geometry)
    desired_geometry[0] = _replace_geometry_prefix(desired_geometry[0], BLOCK)
    # QUDA 的 terminal geometry 是 block 后的粗格尺寸，而不是 fine lattice。
    desired_geometry[1] = _replace_geometry_prefix(
        desired_geometry[1], COARSE_LATTICE)
    writes["geo_block_size"] = _set_whole_array(
        mg_param, "geo_block_size", desired_geometry)

    n_vec = _read_array(mg_param, "n_vec")
    if len(n_vec) < LEVELS:
        raise SmokeFailure("quda_param_array_too_short", "n_vec")
    writes["n_vec"] = _set_whole_array(mg_param, "n_vec", [NVECS] * len(n_vec))

    spin_block = _read_array(mg_param, "spin_block_size")
    if len(spin_block) < LEVELS:
        raise SmokeFailure("quda_param_array_too_short", "spin_block_size")
    writes["spin_block_size"] = _set_array_item_whole_column(
        mg_param, "spin_block_size", 0, FINE_SPIN // COARSE_SPIN)

    n_block_ortho = _read_array(mg_param, "n_block_ortho")
    writes["n_block_ortho"] = _set_whole_array(
        mg_param, "n_block_ortho", [N_BLOCK_ORTHO] * len(n_block_ortho))

    for field, value in (("nu_pre", NU_PRE), ("nu_post", NU_POST)):
        column = _read_array(mg_param, field)
        writes[field] = _set_whole_array(
            mg_param, field, [value] * len(column))

    for field in ("setup_use_mma", "dslash_use_mma", "transfer_use_mma"):
        column = _read_array(mg_param, field)
        writes[field] = _set_whole_array(
            mg_param, field, [boolean_false] * len(column))

    vec_load = _read_array(mg_param, "vec_load")
    writes["vec_load"] = _set_whole_array(
        mg_param, "vec_load", [boolean_false] * len(vec_load))

    writes["compute_null_vector"] = _set_scalar(
        mg_param, "compute_null_vector", compute_null_vector_yes)
    writes["generate_all_levels"] = _set_scalar(
        mg_param, "generate_all_levels", boolean_true)
    return writes


def _snapshot_multigrid(mg_param: Any) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for field in ("n_level", "compute_null_vector", "generate_all_levels"):
        if not hasattr(mg_param, field):
            raise SmokeFailure("quda_param_missing", field)
        snapshot[field] = _normalise(getattr(mg_param, field))
    for field in _PARAM_ARRAY_FIELDS:
        snapshot[field] = _normalise(_read_array(mg_param, field))

    spin_block = snapshot["spin_block_size"]
    n_vec = snapshot["n_vec"]
    if not isinstance(spin_block, list) or not isinstance(n_vec, list):
        raise SmokeFailure("quda_param_snapshot_invalid", "spin_block_size/n_vec")
    try:
        resolved_spin_block = int(spin_block[0])
        resolved_n_vec = int(n_vec[0])
    except (IndexError, TypeError, ValueError) as exc:
        raise SmokeFailure("quda_param_snapshot_invalid", "active MG columns") from exc
    if resolved_spin_block <= 0:
        resolved_coarse_spin = None
        resolved_coarse_color = None
    else:
        resolved_coarse_spin = FINE_SPIN // resolved_spin_block
        resolved_coarse_color = resolved_n_vec * resolved_coarse_spin
    snapshot["derived_coarse_spin"] = resolved_coarse_spin
    snapshot["derived_coarse_color"] = resolved_coarse_color
    return snapshot


def _active_geometry(snapshot: Mapping[str, Any]) -> list[list[int]] | None:
    value = snapshot.get("geo_block_size")
    if not isinstance(value, list) or len(value) < LEVELS:
        return None
    rows: list[list[int]] = []
    for row in value[:LEVELS]:
        if not isinstance(row, list) or len(row) < 4:
            return None
        try:
            rows.append([int(item) for item in row[:4]])
        except (TypeError, ValueError):
            return None
    return rows


def _evaluate_multigrid_contract(snapshot: Mapping[str, Any]) -> dict[str, bool]:
    geometry = _active_geometry(snapshot)
    checks: dict[str, bool] = {
        "n_level": _equal(snapshot.get("n_level"), LEVELS),
        "geometry_block": geometry is not None and geometry[0] == list(BLOCK),
        "geometry_terminal": (
            geometry is not None and geometry[1] == list(COARSE_LATTICE)),
        "n_vec": _equal(snapshot.get("n_vec"), [NVECS] * len(snapshot.get("n_vec", []))),
        "spin_block_size": (
            isinstance(snapshot.get("spin_block_size"), list)
            and snapshot["spin_block_size"][:LEVELS] == [FINE_SPIN // COARSE_SPIN, 1]
        ),
        "coarse_spin": _equal(snapshot.get("derived_coarse_spin"), COARSE_SPIN),
        "coarse_color": _equal(snapshot.get("derived_coarse_color"), COARSE_COLOR),
        "n_block_ortho": _equal(
            snapshot.get("n_block_ortho"),
            [N_BLOCK_ORTHO] * len(snapshot.get("n_block_ortho", []))),
        "nu_pre": _equal(
            snapshot.get("nu_pre"), [NU_PRE] * len(snapshot.get("nu_pre", []))),
        "nu_post": _equal(
            snapshot.get("nu_post"), [NU_POST] * len(snapshot.get("nu_post", []))),
    }
    for field in ("setup_use_mma", "dslash_use_mma", "transfer_use_mma", "vec_load"):
        values = snapshot.get(field)
        checks[f"{field}_off"] = (
            isinstance(values, list) and all(_equal(value, 0) for value in values))
    checks["compute_null_vector"] = _equal(snapshot.get("compute_null_vector"), 1)
    checks["generate_all_levels"] = _equal(snapshot.get("generate_all_levels"), 1)
    return checks


def _precision_number(single_precision: Any) -> int:
    value = getattr(single_precision, "value", single_precision)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SmokeFailure("quda_precision_invalid", repr(single_precision)) from exc


def _snapshot_precision(dirac: Any, mg_param: Any) -> dict[str, Any]:
    objects = {
        "gauge": getattr(dirac, "gauge_param", None),
        "outer_invert": getattr(dirac, "invert_param", None),
        "mg_invert": getattr(getattr(dirac, "multigrid", None), "inv_param", None),
    }
    result: dict[str, Any] = {}
    for name, target in objects.items():
        if target is None:
            raise SmokeFailure("quda_precision_param_missing", name)
        fields = list(_PRECISION_FIELDS)
        if name in ("outer_invert", "mg_invert"):
            fields.extend(_CLOVER_PRECISION_FIELDS)
        result[name] = {}
        for field in fields:
            if not hasattr(target, field):
                raise SmokeFailure("quda_precision_field_missing", f"{name}.{field}")
            result[name][field] = _normalise(getattr(target, field))
    result["mg_precision_null"] = _normalise(_read_array(mg_param, "precision_null"))
    return result


def _evaluate_precision_contract(
        snapshot: Mapping[str, Any], expected: int) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for name, values in snapshot.items():
        if name == "mg_precision_null":
            checks[name] = (
                isinstance(values, list)
                and all(_equal(value, expected) for value in values))
            continue
        if not isinstance(values, Mapping):
            checks[name] = False
            continue
        for field, value in values.items():
            checks[f"{name}.{field}"] = _equal(value, expected)
    return checks


def _parse_instance_list(value: Any) -> list[int]:
    found: set[int] = set()
    for token in re.split(r"[,;\s]+", str(value).strip()):
        if not token:
            continue
        try:
            found.add(int(token))
        except ValueError:
            continue
    return sorted(found)


def _build_evidence(cmake: Mapping[str, Any], single_precision: Any) -> dict[str, Any]:
    features = dict(cmake.get("features") or {})
    instances = _parse_instance_list(features.get("QUDA_MULTIGRID_NVEC_LIST", ""))
    precision_raw = features.get("QUDA_PRECISION")
    try:
        precision_mask = int(precision_raw)
    except (TypeError, ValueError):
        precision_mask = None
    single_bit = _precision_number(single_precision)
    checks = {
        "multigrid_instances_12_and_24":
            NVECS in instances and COARSE_COLOR in instances,
        "single_precision_compiled": (
            precision_mask is not None and (precision_mask & single_bit) == single_bit),
    }
    return {
        "cmake": _normalise(cmake),
        "features": _normalise(features),
        "multigrid_nvec_instances": instances,
        "precision_mask": precision_mask,
        "single_precision_bit": single_bit,
        "checks": checks,
    }


def _library_evidence(
        prefix: Path, libquda: Path, reduction_runtime: Mapping[str, Any],
        qmp_runtime: Mapping[str, Any]) -> dict[str, Any]:
    quda_summary = _file_summary(libquda)
    qmp_path = Path(str(qmp_runtime.get("library", ""))).expanduser().resolve()
    qmp_summary = _file_summary(qmp_path)
    checks = {
        "libquda_path": _equal(
            reduction_runtime.get("library"), quda_summary["path"]),
        "libquda_digest": _equal(
            reduction_runtime.get("library_sha256"), quda_summary["sha256"]),
        "libqmp_path": _equal(qmp_runtime.get("library"), qmp_summary["path"]),
        "libqmp_digest": _equal(
            qmp_runtime.get("library_sha256"), qmp_summary["sha256"]),
        "qmp_initialized": qmp_runtime.get("initialized") is True,
        "qmp_funneled": (
            _equal(qmp_runtime.get("thread_level_required"), 1)
            and (qmp_runtime.get("initialized_here") is not True
                 or _equal(qmp_runtime.get("thread_level_provided"), 1))
        ),
    }
    if reduction_runtime.get("required") is True:
        checks["wsl2_sync_guard"] = (
            reduction_runtime.get("enabled") is True
            and reduction_runtime.get("marker_present") is True)
    else:
        checks["wsl2_sync_guard"] = True
    return {
        "install_prefix": str(prefix),
        "libquda": quda_summary,
        "libqmp": qmp_summary,
        "wsl2_guard": _normalise(reduction_runtime),
        "qmp_runtime": _normalise(qmp_runtime),
        "checks": checks,
    }


def _fixed_config(rank: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "lattice_xyzt": list(LATTICE),
        "rank": None if rank is None else dict(rank),
        "setup_only": True,
        "operator": "Clover-Wilson",
        "mass": MASS,
        "setup_tolerance": SETUP_TOLERANCE,
        "setup_max_iter": SETUP_MAX_ITER,
        "host_gauge_dtype": HOST_GAUGE_DTYPE,
        "host_gauge_layout": "QDP (mu,t,z,y,x,row,col), contiguous",
        "device_precision": DEVICE_PRECISION,
        "n_vec": NVECS,
        "coarse_spin": COARSE_SPIN,
        "coarse_color": COARSE_COLOR,
        "block_xyzt": list(BLOCK),
        "coarse_lattice_xyzt": list(COARSE_LATTICE),
        "n_level": LEVELS,
        "nu_pre": NU_PRE,
        "nu_post": NU_POST,
        "n_block_ortho": N_BLOCK_ORTHO,
        "mma": {"setup": False, "dslash": False, "transfer": False},
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    resource = _resource_path(args.resource_path)
    rank = _single_rank_report()

    # WSL2 guard/QMP 必须先于 PyQUDA import；两者均复用正式 benchmark helper。
    bench = _load_bench_helpers()
    try:
        reduction_runtime = bench._prepare_quda_reduction_runtime()
        prefix, libquda = _quda_prefix_and_library()
        qmp_runtime = bench._initialize_quda_qmp_runtime(reduction_runtime)
    except getattr(bench, "BenchmarkFailure", RuntimeError) as exc:
        raise SmokeFailure(str(getattr(exc, "code", "quda_runtime_guard")), str(exc)) from exc

    runtime_evidence = dict(reduction_runtime)
    runtime_evidence["qmp"] = qmp_runtime

    # 直到 QMP 初始化并通过 V100 guard 后才导入 CUDA/PyQUDA。
    import numpy as np
    import torch

    device = bench._select_v100(torch)
    device_report = bench._torch_runtime_provenance(torch, device)
    import pyquda
    import pyquda_utils.core as core
    from pyquda.enum_quda import QudaBoolean, QudaComputeNullVector, QudaPrecision
    from pyquda.field import LatticeGauge

    resource.mkdir(parents=True, exist_ok=True)  # 闸门不删除 tuning/resource 文件
    torch.cuda.synchronize(device)
    runtime_started = time.perf_counter()
    pyquda.init(
        grid_size=[1, 1, 1, 1],
        latt_size=list(LATTICE),
        backend="torch",
        backend_target="cuda",
        enable_nvshmem=False,
        enable_tuning=False,
        resource_path=str(resource),
        enable_device_memory_pool=False,
        enable_pinned_memory_pool=False,
    )
    torch.cuda.synchronize(device)
    runtime_seconds = time.perf_counter() - runtime_started

    qdp = _unit_gauge_qdp(np)
    info = core.LatticeInfo(list(LATTICE), 1, 1.0)
    input_started = time.perf_counter()
    gauge_eo = np.ascontiguousarray(info.evenodd(qdp, True))
    # 即使设备精度为 single，这里的 QDP host staging 仍保持 complex128。
    gauge_field = LatticeGauge(
        info, 4, torch.from_numpy(gauge_eo).to(device))
    input_seconds = time.perf_counter() - input_started

    single = QudaPrecision.QUDA_SINGLE_PRECISION
    expected_precision = _precision_number(single)
    dirac = None
    cleanup_errors: list[str] = []
    writes: dict[str, Any]
    before: dict[str, Any]
    after: dict[str, Any]
    precision_after: dict[str, Any]
    setup_seconds: float
    memory_after_setup: dict[str, int]
    try:
        dirac = core.getClover(
            info, MASS, SETUP_TOLERANCE, SETUP_MAX_ITER,
            clover_csw_t=1.0, multigrid=[list(BLOCK)])
        dirac.setPrecision(
            cuda=single,
            sloppy=single,
            precondition=single,
            refinement_sloppy=single,
            eigensolver=single,
        )
        mg_obj = getattr(dirac, "multigrid", None)
        mg_param = None if mg_obj is None else getattr(mg_obj, "param", None)
        if mg_param is None:
            raise SmokeFailure("quda_multigrid_param_missing", "dirac.multigrid.param")
        writes = _configure_multigrid(
            mg_param,
            boolean_false=QudaBoolean.QUDA_BOOLEAN_FALSE,
            boolean_true=QudaBoolean.QUDA_BOOLEAN_TRUE,
            compute_null_vector_yes=(
                QudaComputeNullVector.QUDA_COMPUTE_NULL_VECTOR_YES),
        )
        before = _snapshot_multigrid(mg_param)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        setup_started = time.perf_counter()
        dirac.loadGauge(gauge_field)
        torch.cuda.synchronize(device)
        setup_seconds = time.perf_counter() - setup_started
        # 这是 setup 后的第二次读回，所有 PASS 判断均只依据此快照。
        after = _snapshot_multigrid(mg_param)
        precision_after = _snapshot_precision(dirac, mg_param)
        # 在释放 QUDA 层级前采样，避免把 cleanup 后的较小 allocator 数值
        # 误标为 setup 后驻留量；cleanup 仍不计入 setup_seconds。
        memory_after_setup = {
            "cuda_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "cuda_reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
    finally:
        if dirac is not None:
            cleanup_errors.extend(bench._close_quda_dirac(dirac))

    try:
        provenance = bench._benchmark_provenance(
            pyquda=pyquda, reduction_runtime=runtime_evidence)
        cmake = provenance.get("cmake_features", {})
    except Exception as exc:
        raise SmokeFailure("provenance_failed", repr(exc)) from exc

    mg_checks = _evaluate_multigrid_contract(after)
    precision_checks = _evaluate_precision_contract(precision_after, expected_precision)
    build = _build_evidence(cmake, single)
    library = _library_evidence(prefix, libquda, reduction_runtime, qmp_runtime)
    checks: dict[str, bool] = {}
    checks.update({f"multigrid.{key}": value for key, value in mg_checks.items()})
    checks.update({f"precision.{key}": value for key, value in precision_checks.items()})
    checks.update({f"build.{key}": value for key, value in build["checks"].items()})
    checks.update({f"library.{key}": value for key, value in library["checks"].items()})
    passed = bool(checks) and all(checks.values())
    failed_checks = [name for name, value in checks.items() if not value]

    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "ok" if passed else "failed",
        "pass_marker": PASS_MARKER if passed else None,
        "pass_marker_expected": PASS_MARKER,
        "config": _fixed_config(rank),
        "resource": {
            "path": str(resource),
            "required_root": str(DATA_DIR.resolve()),
            "cleanup": "not performed",
        },
        "device": device_report,
        "gauge": {
            "host_dtype": str(qdp.dtype),
            "host_shape": list(qdp.shape),
            "host_contiguous": bool(qdp.flags.c_contiguous),
            "unit": True,
            "qdp_layout": "(mu,t,z,y,row,col)",
        },
        "multigrid": {
            "requested": _fixed_config(rank),
            "writeback_readback_before_setup": writes,
            "resolved_after_setup": after,
            "precision_resolved_after_setup": precision_after,
            "checks": mg_checks,
        },
        "build": build,
        "library": library,
        "timing": {
            "runtime_init_seconds": float(runtime_seconds),
            "input_prepare_seconds": float(input_seconds),
            "setup_seconds": float(setup_seconds),
            "setup_boundary": "dirac.loadGauge plus surrounding CUDA synchronize only",
            "solve_executed": False,
        },
        "memory": {
            **memory_after_setup,
            "note": "PyTorch allocator values exclude QUDA native allocations",
        },
        "checks": checks,
        "passed": passed,
        "provenance": {
            **provenance,
            "environment": bench._environment_snapshot(),
            "runtime": device_report,
            "reduction_runtime": runtime_evidence,
            "cleanup_errors": cleanup_errors,
        },
        "error": None if passed else {
            "code": "setup_contract_gate",
            "detail": "resolved parameters or library/QMP/build evidence mismatch",
            "failed_checks": failed_checks,
        },
    }


def _base_failure(args: argparse.Namespace, exc: BaseException) -> dict[str, Any]:
    try:
        resource = str(_resource_path(args.resource_path))
    except Exception:
        resource = str(args.resource_path)
    if isinstance(exc, SmokeFailure):
        error = {"code": exc.code, "detail": exc.detail}
    else:
        error = {"code": type(exc).__name__, "detail": str(exc)}
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
        "pass_marker": None,
        "pass_marker_expected": PASS_MARKER,
        "config": _fixed_config(),
        "resource": {
            "path": resource,
            "required_root": str(DATA_DIR.resolve()),
            "cleanup": "not performed",
        },
        "error": error,
    }


def _write_json(path: Path, record: Mapping[str, Any]) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        record, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "8^4 单 rank、unit gauge、single precision 的 QUDA Clover-MG "
            "Nc24 setup-only smoke；成功时输出 DEV87_QUDA_MG8_NC24_PASS。"))
    parser.add_argument(
        "--resource-path", type=Path, default=DEFAULT_RESOURCE_PATH,
        help="QUDA resource/tuning 路径；相对路径按仓库 data/ 解释，必须位于 data/ 内")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="JSON 输出路径")
    parser.add_argument(
        "--cuda-visible-devices", default=None,
        help="在导入 CUDA/PyQUDA 前设置 CUDA_VISIBLE_DEVICES，例如 2")
    parser.add_argument(
        "--quda-install", default=None,
        help="QUDA 安装前缀；同时设置 QUDA_INSTALL/QUDA_PATH 并把其 lib 置于 LD_LIBRARY_PATH 首位")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _apply_cli_environment(args)
    try:
        record = _run(args)
    except Exception as exc:
        record = _base_failure(args, exc)
    try:
        _write_json(args.output, record)
    except Exception as exc:
        print(f"QUDA MultiGrid smoke JSON 写入失败: {exc}", file=sys.stderr, flush=True)
        return 1

    print(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    if record.get("status") == "ok" and record.get("pass_marker") == PASS_MARKER:
        print(PASS_MARKER, flush=True)
        return 0
    print(f"QUDA MultiGrid setup smoke failed: {record.get('error')}",
          file=sys.stderr, flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
