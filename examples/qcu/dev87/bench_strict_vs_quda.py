#!/usr/bin/env python3
"""严格 MultiGrid 与 QUDA 的可恢复、公平性能采集器。

正式运行固定使用 ``16x32x32x48``，并以 dev87 的 HDF5 ``g``/``fi``
及预先验证的 canonical full ``null`` 为唯一输入。E12 odd-Schur cache 只在
独立预处理器中做一次零右端块消元；双方都直接消费同一 full dataset。父进程只负责输入指纹、隔离 worker、
超时和 JSON 合并；PyQCU/QUDA 均在独立进程内初始化 CUDA，避免一个后端
污染另一个后端的 context、allocator 或峰值显存。

默认流程不会把 QUDA 自行生成的随机 near-null vectors 当作公平对照。
QUDA 侧必须提供由同一 canonical ``null`` 转换出的 QIO 前缀及转换清单；缺少时明确
标记 ``skipped``。这项约束是有意的：只有双方输入摘要、精度和求解配置均
一致时，输出才包含速度比。

常用命令（正式 benchmark 很慢，本文件的单元测试不会执行它）：

  python examples/qcu/dev87/bench_strict_vs_quda.py --list
  python examples/qcu/dev87/bench_strict_vs_quda.py --dry-run
  python examples/qcu/dev87/bench_strict_vs_quda.py --side pyqcu \
      --cache-expect hit --output pyqcu.json
  python examples/qcu/dev87/bench_strict_vs_quda.py --side quda \
      --quda-nullvec-prefix /path/to/qio-prefix \
      --quda-nullvec-manifest /path/to/conversion.json --output quda.json
  python examples/qcu/dev87/bench_strict_vs_quda.py \
      --merge pyqcu.json quda.json --output combined.json

统计口径：2 次不计时 warmup（第一次同时采集首次 solve 的显存峰值），
随后默认 5 次独立零初值 steady solve；
报告 wall-time median/MAD。setup、输入读取、运行时初始化、steady solve 和
真残差复算分别计时，不能互相混入。真残差统一为 PyQCU Wilson/Clover
归一化 ``||b-Dx||_2/||b||_2``；QUDA 解先乘 ``mass+4`` 再进入同一复算。
"""

from __future__ import annotations

import argparse
import atexit
import base64
import copy
import ctypes
from contextlib import contextmanager
import hashlib
import inspect
import json
import math
import os
import platform
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

SCHEMA_NAME = "pyqcu.strict-vs-quda.benchmark"
SCHEMA_VERSION = 1
MEMORY_SCHEMA_VERSION = 2
QIO_MANIFEST_SCHEMA = "pyqcu.quda-nullvec-conversion/v1"
QIO_MANIFEST_VERSION = 1
QIO_GAMMA_BASIS = "QUDA_DEGRAND_ROSSI_GAMMA_BASIS"
QIO_HOST_LAYOUT = (
    "QUDA_SPACE_SPIN_COLOR_FIELD_ORDER; full-site checkerboard index "
    "[even lex,odd lex][spin][color][complex], x fastest"
)
WORKER_PREFIX = "PYQCU_STRICT_BENCH_RESULT="
WORKER_PAYLOAD_ENV = "PYQCU_STRICT_BENCH_PAYLOAD_B64"
WORKER_RESULT_DIR_ENV = "PYQCU_STRICT_BENCH_RESULT_DIR"

DEFAULT_LATTICE = (16, 32, 32, 48)
LATTICE = DEFAULT_LATTICE
MASS = 0.05
SEED = 42
SOURCE_DATASET = "fi"
GAUGE_DATASET = "g"
NULLVEC_DATASET = "null"
NVECS = 12
COARSE_DOF = 2 * NVECS
BLOCK = (2, 2, 2, 2)
LEVELS = 2
COARSE_SPIN = 2
# Match the formal QUDA setting below (QUDA_MATPC_ODD_ODD).  Keeping both
# sides on the same Schur block matters for finite-tolerance iteration counts
# and therefore for a fair performance comparison.
TARGET_PARITY = 1
WARMUPS = 2
DEFAULT_REPEATS = 5
PHASE_ORDER = ("cold", "warmup", "steady")
DEFAULT_PHASES = ("warmup", "steady")
DEVICE_TARGETS = ("auto", "v100", "p100")
MPI_RANK_ENV = "PYQCU_STRICT_BENCH_MPI_RANK"
MPI_SIZE_ENV = "PYQCU_STRICT_BENCH_MPI_SIZE"
MPI_GRID_ENV = "PYQCU_STRICT_BENCH_PROCESS_GRID"
DEFAULT_TIMEOUT = 1800.0
DEFAULT_RESTART = 16
DEFAULT_MAX_KRYLOV_BYTES = 512 << 20
DEFAULT_STRICT_GALERKIN_COLUMN_BATCH_C64 = 12
DEFAULT_STRICT_GALERKIN_COLUMN_BATCH_C128 = 1
DEFAULT_STRICT_GALERKIN_MAX_WORKSPACE_C64 = 4 << 30
DEFAULT_STRICT_GALERKIN_MAX_WORKSPACE_C128 = 1 << 30
DEFAULT_STRICT_GALERKIN_PROJECTION_BATCH = 4
DEFAULT_MAX_ITER = 1000
DEFAULT_COARSE_MAX_ITER = 200
DEFAULT_NU_PRE = 1
DEFAULT_NU_POST = 1
PROFILE_NAMES = ("formal", "smoke")
DEFAULT_PROFILE = "formal"
STRICT_CACHE_IDENTITY_SCHEMA = "pyqcu.strict-runtime-cache-identity/v1"
STRICT_CACHE_FORMAT_VERSION = 2
STRICT_ASSET_SEMANTICS_VERSION = 3
STRICT_CACHE_DIR = REPO / "data" / "strict_runtime_cache"

GAUGE_PATH = REPO / "data" / "gauge_16x32x32x48_m0.05_seed42_c64.h5"
NULLVEC_PATH = REPO / "data" / "L16x32x32x48_nvec12_full_c64.h5"
DEFAULT_OUTPUT = HERE / "out" / "strict_vs_quda_benchmark.json"

TERMINAL_STATUSES = {
    "ok", "skipped", "failed", "timeout", "side_unsupported",
}
SIDE_NAMES = ("pyqcu", "quda")
ENV_ALLOWLIST = (
    "CUDA_VISIBLE_DEVICES",
    "QCU_DEVICE_ID",
    "QUDA_INSTALL",
    "QUDA_PATH",
    "QUDA_BUILD_DIR",
    "QUDA_SOURCE_DIR",
    "DEV87_REDUCE_SYNC",
    "PYQCU_STRICT_TRACE_FILE",
    "PYQCU_STRICT_LINK_HALO_CACHE",
    "PYQCU_STRICT_COARSE_TOL_FACTOR",
    "PYQCU_QUDA_TRACE_FILE",
    "QUDA_MG_TRACE_FILE",
    "PYQCU_STRICT_BENCH_MPI_RANK",
    "PYQCU_STRICT_BENCH_MPI_SIZE",
    "PYQCU_STRICT_BENCH_PROCESS_GRID",
    "PYQCU_STRICT_BENCH_RESULT_DIR",
    "PYQCU_MPI_DEVICE_AWARE",
    "PYQCU_MPI_DEVICE_ID",
    "OMP_NUM_THREADS",
)
_QMP_RUNTIME_HOLD: List[Any] = []


def _strict_coarse_tolerance(config: Mapping[str, Any]) -> float:
    value = float(config["coarse_tolerance"])
    raw = os.environ.get("PYQCU_STRICT_COARSE_TOL_FACTOR")
    if raw is None or not raw.strip():
        return value
    try:
        factor = float(raw)
    except ValueError as exc:
        raise BenchmarkFailure(
            "invalid_strict_coarse_tol_factor", raw) from exc
    if not math.isfinite(factor) or factor <= 0.0:
        raise BenchmarkFailure(
            "invalid_strict_coarse_tol_factor", raw)
    return min(0.5, value * factor)


class BenchmarkSkip(RuntimeError):
    """环境或公平输入缺失；这是明确 skip，不是成功。"""

    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = str(code)
        self.detail = str(detail)


class BenchmarkFailure(RuntimeError):
    """协议、公平性或运行期失败。"""

    def __init__(self, code: str, detail: str,
                 context: Optional[Mapping[str, Any]] = None):
        super().__init__(detail)
        self.code = str(code)
        self.detail = str(detail)
        self.context = {} if context is None else copy.deepcopy(dict(context))


class BenchmarkUnsupported(BenchmarkFailure):
    """The requested side/protocol is valid but cannot be represented safely."""

    def __init__(self, code: str, detail: str,
                 context: Optional[Mapping[str, Any]] = None):
        super().__init__(code, detail, context)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _normalise_phases(value: Any) -> Tuple[str, ...]:
    if isinstance(value, str):
        items = [item.strip().lower() for item in value.split(",") if item.strip()]
    else:
        try:
            items = [str(item).strip().lower() for item in value]
        except TypeError as exc:
            raise ValueError("phases must be a comma-separated string or sequence") from exc
    if not items:
        raise ValueError("phases must not be empty")
    unknown = sorted(set(items) - set(PHASE_ORDER))
    if unknown:
        raise ValueError(f"unknown phases: {unknown}; expected {list(PHASE_ORDER)}")
    if len(set(items)) != len(items):
        raise ValueError("phases must not contain duplicates")
    selected = set(items)
    return tuple(phase for phase in PHASE_ORDER if phase in selected)


def _phases_from_args(args: argparse.Namespace) -> Tuple[str, ...]:
    return _normalise_phases(getattr(args, "phases", DEFAULT_PHASES))


def _phase_counts(
        phases: Sequence[str],
        steady_repeats: int = DEFAULT_REPEATS) -> Dict[str, int]:
    selected = set(_normalise_phases(phases))
    return {
        "cold": 1 if "cold" in selected else 0,
        "warmup": WARMUPS if "warmup" in selected else 0,
        "steady": int(steady_repeats) if "steady" in selected else 0,
    }


def _phase_request(payload: Mapping[str, Any]) -> Dict[str, Any]:
    request = payload.get("phase_request")
    if request is None:
        execution = payload.get("execution") or {}
        phases = execution.get("phases") or DEFAULT_PHASES
        if isinstance(phases, Mapping):
            phases = phases.get("selected") or DEFAULT_PHASES
        selected = set(_normalise_phases(phases))
        return {
            "cold": "cold" in selected,
            "warmup": "warmup" in selected,
            "steady": "steady" in selected,
            "cache_expect": (
                (execution.get("strict_cache") or {}).get("expect", "any")),
            "legacy": True,
        }
    if not isinstance(request, Mapping):
        raise TypeError("phase_request must be a mapping")
    return {
        "cold": bool(request.get("cold", False)),
        "warmup": bool(request.get("warmup", False)),
        "steady": bool(request.get("steady", False)),
        "cache_expect": str(request.get("cache_expect", "any")),
        "legacy": False,
    }


def _default_process_grid(ranks: int) -> Tuple[int, int, int, int]:
    return (1, 1, 1, 1) if int(ranks) == 1 else (int(ranks), 1, 1, 1)


def _normalise_process_grid(
        value: Optional[Sequence[int]], ranks: int) -> Tuple[int, int, int, int]:
    if value is None:
        grid = _default_process_grid(ranks)
    else:
        try:
            grid = tuple(int(item) for item in value)
        except (TypeError, ValueError) as exc:
            raise ValueError("process grid must contain four positive integers") from exc
        if len(grid) != 4 or any(item <= 0 for item in grid):
            raise ValueError("process grid must contain four positive integers")
    if math.prod(grid) != int(ranks):
        raise ValueError(
            f"process grid {grid} has {math.prod(grid)} ranks, "
            f"expected --mpi-ranks={int(ranks)}")
    return grid  # type: ignore[return-value]


def _rank_coordinate(rank: int, grid: Sequence[int]) -> Tuple[int, int, int, int]:
    coordinate = [0, 0, 0, 0]
    remainder = int(rank)
    for axis in range(4):
        stride = math.prod(grid[axis + 1:])
        coordinate[axis] = remainder // stride
        remainder %= stride
    return tuple(coordinate)  # type: ignore[return-value]


def _worker_mpi_environment_context() -> Optional[Dict[str, int]]:
    rank_names = ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK")
    size_names = ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE")
    rank_value = next((os.environ[name] for name in rank_names
                       if os.environ.get(name) is not None), None)
    size_value = next((os.environ[name] for name in size_names
                       if os.environ.get(name) is not None), None)
    if rank_value is None and size_value is None:
        return None
    if rank_value is None or size_value is None:
        raise BenchmarkFailure(
            "mpi_environment_incomplete",
            f"incomplete MPI environment: rank={rank_value!r}, size={size_value!r}")
    result = {"rank": int(rank_value), "size": int(size_value)}
    local_rank = os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
    local_size = os.environ.get("OMPI_COMM_WORLD_LOCAL_SIZE")
    if local_rank is not None:
        result["local_rank"] = int(local_rank)
    if local_size is not None:
        result["local_size"] = int(local_size)
    return result


def _worker_mpi_context(payload: Mapping[str, Any]) -> Dict[str, Any]:
    env_context = _worker_mpi_environment_context()
    explicit_rank = payload.get("mpi_rank")
    explicit_size = payload.get("mpi_size")
    if env_context is not None:
        rank = int(env_context["rank"])
        size = int(env_context["size"])
    else:
        rank = 0 if explicit_rank is None else int(explicit_rank)
        size = 1 if explicit_size is None else int(explicit_size)
    parallel = (payload.get("execution") or {}).get("parallel") or {}
    grid = parallel.get("process_grid")
    if grid is None:
        grid = payload.get("mpi_process_grid")
    grid = _normalise_process_grid(grid, size)
    if rank < 0 or rank >= size:
        raise BenchmarkFailure(
            "mpi_rank_invalid", f"rank={rank}, size={size}")
    coordinate = _rank_coordinate(rank, grid)
    return {
        "rank": rank,
        "size": size,
        "process_grid": list(grid),
        "rank_coordinate": list(coordinate),
        "local_rank": int(env_context.get(
            "local_rank", rank)) if env_context is not None else rank,
        "local_size": int(env_context.get(
            "local_size", size)) if env_context is not None else size,
    }


def _rank_trace_path(path: str, rank: int) -> str:
    suffix = f".rank{int(rank):05d}"
    return path if path.endswith(suffix) else path + suffix


def _activate_rank_trace_paths(context: Mapping[str, Any]) -> Dict[str, str]:
    """Give every MPI rank a private native trace file."""
    rank = int(context["rank"])
    size = int(context["size"])
    result: Dict[str, str] = {}
    for name in (
            "PYQCU_STRICT_TRACE_FILE", "QUDA_MG_TRACE_FILE",
            "PYQCU_QUDA_TRACE_FILE"):
        value = os.environ.get(name)
        if not value:
            continue
        if size > 1:
            value = _rank_trace_path(value, rank)
            os.environ[name] = value
        path = Path(value).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
        result[name] = str(path)
    return result


def _side_trace_path(side: str) -> Optional[Path]:
    variable = (
        "PYQCU_STRICT_TRACE_FILE" if side == "pyqcu" else
        "QUDA_MG_TRACE_FILE")
    value = os.environ.get(variable)
    return Path(value).resolve() if value else None


def _strict_runtime_cache_identity(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the immutable physical identity of reusable strict assets."""
    config = payload["protocol"]
    fingerprints = payload["input_fingerprints"]
    mpi_context = payload.get("mpi_context")
    if not isinstance(mpi_context, Mapping):
        execution = payload.get("execution") or {}
        parallel = execution.get("parallel") or {}
        ranks = int(parallel.get("ranks", 1))
        grid = _normalise_process_grid(parallel.get("process_grid"), ranks)
        rank = int(payload.get("mpi_rank", 0))
        mpi_context = {
            "rank": rank,
            "size": ranks,
            "process_grid": list(grid),
            "rank_coordinate": list(_rank_coordinate(rank, grid)),
        }

    def input_identity(name: str) -> Dict[str, Any]:
        value = fingerprints[name]
        return {
            "algorithm": value["algorithm"],
            "sha256": value["sha256"],
            "shape": list(value["shape"]),
            "dtype": value["dtype"],
        }

    identity = {
        "schema": STRICT_CACHE_IDENTITY_SCHEMA,
        "runtime_cache_format_version": STRICT_CACHE_FORMAT_VERSION,
        "asset_semantics_version": (
            2 if int(config["levels"]) == 2 else
            STRICT_ASSET_SEMANTICS_VERSION),
        "gauge": input_identity("gauge"),
        "null_vectors": input_identity("null_vectors"),
        "lattice_xyzt": list(config["lattice_xyzt"]),
        "mass": float(config["mass"]),
        "kappa": float(config["kappa"]),
        "precision": config["precision"]["name"],
        "levels": int(config["levels"]),
        "block_xyzt": list(config["block_xyzt"]),
        "nvec": int(config["nvec"]),
        "coarse_spin": int(config["coarse_spin"]),
        "coarse_dof": int(config["coarse_dof"]),
        "target_parity": int(config["target_parity"]),
        "n_block_ortho": 2,
        # PyQCU's explicit gamma matrices match QUDA's DeGrand-Rossi table
        # (despite an older project note calling them Dirac-Pauli).
        "gamma_basis": "QUDA_DEGRAND_ROSSI_GAMMA_BASIS (pyqcu.lattice.gamma)",
        "boundary_conditions": (
            "periodic xyzt; single MPI rank"
            if int(mpi_context["size"]) == 1 else
            "periodic xyzt; distributed rank-local shard"),
        "clover_convention": {
            "python_builder": "pyqcu.dslash.make_clover",
            "cuda_sigma": 0.1,
        },
        "canonical_nullvec": {
            "schema": "pyqcu.canonical-full-nullvec/v1",
            "reconstruction_version": 2,
        },
        "transfer_semantics": "fine spin 4 -> coarse spin 2 chiral aggregation; CGS2",
        "coarsening_operator": "R(X^-1 D)P",
        "runtime_assets": "fine blocked V; per-transition Yhat/(X,X^-1)",
    }
    if int(mpi_context["size"]) > 1:
        identity["mpi_shard"] = {
            "ranks": int(mpi_context["size"]),
            "process_grid": list(mpi_context["process_grid"]),
            "rank": int(mpi_context["rank"]),
            "rank_coordinate": list(mpi_context["rank_coordinate"]),
        }
    block_precision = config.get("pyqcu_strict_block_precision")
    if (isinstance(block_precision, Mapping) and
            block_precision.get("effective") == "single" and
            config["precision"]["name"] == "c128"):
        identity["coarse_block_setup_precision"] = "complex64"
    return identity


def _strict_runtime_cache_path(
        identity: Mapping[str, Any], cache_dir: Path = STRICT_CACHE_DIR) -> Path:
    return Path(cache_dir) / f"strict_runtime_{_sha256_json(identity)[:24]}.h5"


def _strict_runtime_expected_manifest(
        config: Mapping[str, Any],
        mpi_context: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Exact recursive asset contract for the selected benchmark hierarchy."""
    lattice = tuple(int(x) for x in config["lattice_xyzt"])
    if mpi_context is not None:
        grid = tuple(int(x) for x in mpi_context["process_grid"])
        if any(extent % width for extent, width in zip(lattice, grid)):
            raise ValueError(
                f"lattice={lattice} is not divisible by process_grid={grid}")
        lattice = tuple(
            extent // width for extent, width in zip(lattice, grid))
    blocks = [
        tuple(int(x) for x in value)
        for value in config.get(
            "block_xyzt_per_level",
            [config["block_xyzt"]] * (int(config["levels"]) - 1))
    ]
    if len(blocks) != int(config["levels"]) - 1:
        raise ValueError("block_xyzt_per_level length must equal levels-1")
    if not blocks:
        raise ValueError(
            "levels=1 has no strict runtime transition cache; "
            "PyQCU strict MG-1 is unsupported")
    coarse_dof = int(config["coarse_dof"])
    dtype = {
        "c64": "complex64",
        "c128": "complex128",
    }[str(config["precision"]["name"])]
    itemsize = int(config["precision"]["complex_bytes"])

    def spec(shape: Sequence[int]) -> Dict[str, Any]:
        values = [int(x) for x in shape]
        return {
            "shape": values,
            "dtype": dtype,
            "nbytes": math.prod(values) * itemsize,
        }

    tensors: Dict[str, Dict[str, Any]] = {}
    fine_shape = lattice
    for transition, block in enumerate(blocks):
        coarse = tuple(
            extent // width for extent, width in zip(fine_shape, block))
        fine_dof = 12 if transition == 0 else coarse_dof
        blocked_shape: List[int] = [coarse_dof, fine_dof]
        for extent, width in zip(coarse, block):
            blocked_shape.extend((extent, width))
        if transition == 0:
            tensors["assets/fine_blocked_v"] = spec(blocked_shape)
        else:
            tensors[
                f"assets/levels/{transition}/null_vectors"] = spec(
                    blocked_shape)
        tensors[
            f"assets/levels/{transition}/preconditioned_links"] = spec(
                [2, 4, coarse_dof, coarse_dof, *coarse])
        tensors[
            f"assets/levels/{transition}/onsite_pair"] = spec(
                [2, coarse_dof, coarse_dof, *coarse])
        fine_shape = coarse
    return {
        "layout": "fine_blocked_v; per-transition Yhat/onsite; transition>=1 V",
        "level_count": len(blocks),
        "tensor_count": len(tensors),
        "dtype": dtype,
        "total_bytes": sum(value["nbytes"] for value in tensors.values()),
        "tensors": tensors,
    }


def _safe_tail(text: Optional[str], limit: int = 4000) -> str:
    value = "" if text is None else str(text)
    return value[-limit:]


def _append_trace_marker(path: os.PathLike[str] | str,
                         label: str, index: int) -> None:
    """Append a flush-only solve boundary to a native trace."""
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="ascii") as handle:
        handle.write(f"{label}\t{int(index)}\t{time.time():.17g}\n")
        handle.flush()


@contextmanager
def _capture_native_stdout(path: Optional[os.PathLike[str] | str]):
    """Capture C/C++ stdout for an optional diagnostic QUDA trace.

    ``printfQuda`` writes through the process file descriptor rather than
    Python's ``sys.stdout`` object, so ``contextlib.redirect_stdout`` is not
    sufficient here.  The context is never entered by the default benchmark;
    trace runs use it only around the caller-owned ``invertQuda`` call.
    """
    if path is None or str(path) == "":
        yield
        return
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    saved_fd = os.dup(1)
    target_fd = os.open(
        str(target), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        sys.stdout.flush()
        ctypes.CDLL(None).fflush(None)
        os.dup2(target_fd, 1)
        yield
    finally:
        sys.stdout.flush()
        ctypes.CDLL(None).fflush(None)
        os.dup2(saved_fd, 1)
        os.close(target_fd)
        os.close(saved_fd)


def _float_field(fields: Sequence[str], index: int) -> float:
    return float(fields[index])


def _int_field(fields: Sequence[str], index: int) -> int:
    return int(fields[index])


def _parse_pyqcu_trace(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    sections: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for line_number, raw in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if not raw:
            continue
        fields = raw.split("\t")
        kind = fields[0]
        try:
            if kind == "trace_version":
                version = _int_field(fields, 1)
                if version not in (1, 2, 3):
                    raise BenchmarkFailure(
                        "pyqcu_trace_version_unsupported",
                        f"{path}: trace_version={version}")
                continue
            if kind == "solve_begin":
                if current is not None:
                    raise ValueError("nested solve_begin")
                current = {
                    "solve_index": len(sections),
                    "rhs_norm": _float_field(fields, 1),
                    "events": [],
                }
                continue
            if kind == "solve_end":
                if current is None:
                    raise ValueError("solve_end without solve_begin")
                current["end"] = {
                    "iterations": _int_field(fields, 1),
                    "converged": bool(_int_field(fields, 2)),
                    "residual": _float_field(fields, 3),
                    "relative": _float_field(fields, 4),
                    "elapsed_seconds": _float_field(fields, 5),
                }
                sections.append(current)
                current = None
                continue
            if current is None:
                raise ValueError("event outside solve")
            event: Dict[str, Any] = {"kind": kind}
            if kind == "initial_residual":
                event.update({
                    "iteration": _int_field(fields, 1),
                    "absolute": _float_field(fields, 2),
                    "relative": _float_field(fields, 3),
                    "elapsed_seconds": _float_field(fields, 4),
                })
            elif kind == "iteration":
                event.update({
                    "iteration": _int_field(fields, 1),
                    "cycle_iteration": _int_field(fields, 2),
                    "estimate": _float_field(fields, 3),
                    "estimate_relative": _float_field(fields, 4),
                    "next_norm": _float_field(fields, 5),
                    "elapsed_seconds": _float_field(fields, 6),
                })
            elif kind == "restart_residual":
                event.update({
                    "iteration": _int_field(fields, 1),
                    "absolute": _float_field(fields, 2),
                    "relative": _float_field(fields, 3),
                    "elapsed_seconds": _float_field(fields, 4),
                })
            elif kind == "stage":
                event.update({
                    "outer_iteration": _int_field(fields, 1),
                    "level": _int_field(fields, 2),
                    "name": fields[3],
                    "seconds": _float_field(fields, 4),
                    "elapsed_seconds": _float_field(fields, 5),
                })
            elif kind == "residual":
                event.update({
                    "outer_iteration": _int_field(fields, 1),
                    "level": _int_field(fields, 2),
                    "name": fields[3],
                    "absolute": _float_field(fields, 4),
                    "relative": _float_field(fields, 5),
                    "elapsed_seconds": _float_field(fields, 6),
                })
            elif kind == "iteration_count":
                event.update({
                    "outer_iteration": _int_field(fields, 1),
                    "level": _int_field(fields, 2),
                    "name": fields[3],
                    "value": _int_field(fields, 4),
                    "elapsed_seconds": _float_field(fields, 5),
                })
            else:
                raise ValueError(f"unknown record {kind!r}")
            current["events"].append(event)
        except (IndexError, ValueError) as exc:
            raise BenchmarkFailure(
                "pyqcu_trace_invalid",
                f"{path}:{line_number}: {exc}; raw={raw!r}") from exc
    if current is not None:
        raise BenchmarkFailure(
            "pyqcu_trace_unterminated", f"{path}: unterminated solve")
    return sections


def _parse_quda_mg_trace(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    sections: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for line_number, raw in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if not raw:
            continue
        fields = raw.split("\t")
        kind = fields[0]
        try:
            if kind == "trace_version":
                continue
            if kind == "python_solve_begin":
                if current is not None:
                    raise ValueError("nested python_solve_begin")
                current = {
                    "solve_index": _int_field(fields, 1),
                    "events": [],
                }
                continue
            if kind == "python_solve_end":
                if current is None:
                    raise ValueError("python_solve_end without begin")
                sections.append(current)
                current = None
                continue
            if current is None:
                raise ValueError("event outside solve")
            event: Dict[str, Any] = {"kind": kind}
            if kind in ("cycle_begin", "cycle_end"):
                event["cycle"] = _int_field(fields, 1)
                event["level"] = _int_field(fields, 2)
                if kind == "cycle_begin":
                    event["levels"] = _int_field(fields, 3)
                    event["outer_iteration"] = _int_field(fields, 4)
                else:
                    event["relative"] = _float_field(fields, 6)
            elif kind == "stage":
                event.update({
                    "cycle": _int_field(fields, 1),
                    "level": _int_field(fields, 2),
                    "phase": fields[3],
                    "seconds": _float_field(fields, 4),
                    "pre_iterations": _int_field(fields, 5),
                    "post_iterations": _int_field(fields, 6),
                    "coarse_iterations": _int_field(fields, 7),
                    "outer_iteration": _int_field(fields, 8),
                })
            elif kind == "residual":
                event.update({
                    "cycle": _int_field(fields, 1),
                    "level": _int_field(fields, 2),
                    "phase": fields[3],
                    "r2": _float_field(fields, 4),
                    "rn": _float_field(fields, 5),
                    "b2": _float_field(fields, 6),
                    "relative": _float_field(fields, 7),
                    "outer_iteration": _int_field(fields, 8),
                })
            elif kind == "outer_begin":
                event.update({
                    "iteration": _int_field(fields, 1),
                    "r2": _float_field(fields, 2),
                    "rn": _float_field(fields, 3),
                    "b2": _float_field(fields, 4),
                    "relative": _float_field(fields, 5),
                })
            elif kind == "outer_iteration":
                event.update({
                    "iteration": _int_field(fields, 1),
                    "cycle_iteration": _int_field(fields, 2),
                    "iterated_r2": _float_field(fields, 3),
                    "iterated_rn": _float_field(fields, 4),
                    "b2": _float_field(fields, 5),
                    "iterated_relative": _float_field(fields, 6),
                    "true_r2": _float_field(fields, 7),
                    "true_rn": _float_field(fields, 8),
                    "true_relative": _float_field(fields, 9),
                })
            else:
                raise ValueError(f"unknown record {kind!r}")
            current["events"].append(event)
        except (IndexError, ValueError) as exc:
            raise BenchmarkFailure(
                "quda_mg_trace_invalid",
                f"{path}:{line_number}: {exc}; raw={raw!r}") from exc
    if current is not None:
        raise BenchmarkFailure(
            "quda_mg_trace_unterminated", f"{path}: unterminated solve")
    return sections


def _empty_level_accumulator(level: int) -> Dict[str, Any]:
    return {
        "level": int(level),
        "total_seconds": 0.0,
        "total_iterations": 0,
        "pre_smoother_seconds": 0.0,
        "post_smoother_seconds": 0.0,
        "restriction_seconds": 0.0,
        "prolongation_seconds": 0.0,
        "coarse_solver_seconds": 0.0,
        "pre_smoother_iterations": 0,
        "post_smoother_iterations": 0,
        "restriction_calls": 0,
        "prolongation_calls": 0,
        "recursive_solve_calls": 0,
        "coarse_solver_iterations": 0,
        "coarse_solver_calls": 0,
        "residual_sequence": [],
        "finest_iterations": None,
        "_cycles": set(),
    }


def _pyqcu_stage_kind(name: str) -> Optional[str]:
    if name in ("fine_pre_smoother", "level_pre_smoother"):
        return "pre_smoother_seconds"
    if name in ("fine_post_smoother", "level_post_smoother"):
        return "post_smoother_seconds"
    if name in ("fine_restriction", "level_restriction"):
        return "restriction_seconds"
    if name in ("fine_prolongation", "level_prolongation"):
        return "prolongation_seconds"
    if name in (
            "coarse_vcycle", "level_recursive_solve", "coarsest_bicgstab"):
        return "coarse_solver_seconds"
    return None


def _quda_stage_kind(name: str) -> Optional[str]:
    if name == "pre_smoother":
        return "pre_smoother_seconds"
    if name == "post_smoother":
        return "post_smoother_seconds"
    if name == "restrict":
        return "restriction_seconds"
    if name == "prolongate":
        return "prolongation_seconds"
    if name in ("coarse_solver", "coarsest_solve"):
        return "coarse_solver_seconds"
    return None


_PYQCU_ITERATION_COUNT_FIELDS = frozenset({
    "pre_smoother_iterations",
    "post_smoother_iterations",
    "restriction_calls",
    "prolongation_calls",
    "recursive_solve_calls",
    "coarse_solver_iterations",
    "coarse_solver_calls",
})


def _finalise_mg_levels(
        accumulators: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for value in accumulators:
        total = float(value["total_seconds"])
        known = sum(float(value[key]) for key in (
            "pre_smoother_seconds", "post_smoother_seconds",
            "restriction_seconds", "prolongation_seconds",
            "coarse_solver_seconds"))
        other = total - known
        tolerance = max(1.0e-6, 1.0e-2 * max(1.0, abs(total), abs(known)))
        if other < -tolerance:
            raise BenchmarkFailure(
                "mg_level_timing_negative_other",
                f"level={value['level']} total={total}, known={known}, "
                f"other={other}")
        other = max(0.0, other)
        item = {
            "level": int(value["level"]),
            "total_seconds": total,
            "total_iterations": int(value["total_iterations"]),
            "pre_smoother_seconds": float(value["pre_smoother_seconds"]),
            "post_smoother_seconds": float(value["post_smoother_seconds"]),
            "restriction_seconds": float(value["restriction_seconds"]),
            "prolongation_seconds": float(value["prolongation_seconds"]),
            "coarse_solver_seconds": float(value["coarse_solver_seconds"]),
            "other_seconds": other,
            "pre_smoother_iterations": int(
                value["pre_smoother_iterations"]),
            "post_smoother_iterations": int(
                value["post_smoother_iterations"]),
            "smoother_iterations": (
                int(value["pre_smoother_iterations"]) +
                int(value["post_smoother_iterations"])),
            "restriction_calls": int(value["restriction_calls"]),
            "prolongation_calls": int(value["prolongation_calls"]),
            "recursive_solve_calls": int(value["recursive_solve_calls"]),
            "coarse_solver_iterations": int(
                value["coarse_solver_iterations"]),
            "coarse_solver_calls": int(value["coarse_solver_calls"]),
        }
        if int(value["level"]) == 0:
            item["finest_iterations"] = int(value["total_iterations"])
            sequence = list(value["residual_sequence"])
            item["residual_sequence"] = sequence
            item["final_residual"] = (
                float(sequence[-1]["relative"]) if sequence else None)
        result.append(item)
    return result


def _mg_levels_from_trace(
        side: str, path: Optional[Path], solve_indices: Sequence[int],
        level_count: int) -> Optional[List[Dict[str, Any]]]:
    if path is None or not path.is_file():
        return None
    sections = (
        _parse_pyqcu_trace(path) if side == "pyqcu"
        else _parse_quda_mg_trace(path))
    selected: List[Mapping[str, Any]] = []
    for index in solve_indices:
        if index < 0 or index >= len(sections):
            raise BenchmarkFailure(
                f"{side}_trace_solve_missing",
                f"{path}: requested solve={index}, available={len(sections)}")
        selected.append(sections[index])
    if not selected:
        return None

    accumulators = [_empty_level_accumulator(level)
                    for level in range(int(level_count))]
    for section in selected:
        outer_iterations: Dict[int, Mapping[str, Any]] = {}
        events = section.get("events", [])
        if side == "pyqcu":
            end = section.get("end")
            if not isinstance(end, Mapping):
                raise BenchmarkFailure(
                    "pyqcu_trace_incomplete", f"{path}: solve has no end")
            iterations = int(end["iterations"])
            for acc in accumulators:
                acc["total_iterations"] += iterations
            for event in events:
                if event["kind"] == "initial_residual":
                    accumulators[0]["residual_sequence"].append({
                        "solve_index": int(section["solve_index"]),
                        "iteration": int(event["iteration"]),
                        "relative": float(event["relative"]),
                        "kind": "initial_true_residual",
                    })
                elif event["kind"] == "iteration":
                    accumulators[0]["residual_sequence"].append({
                        "solve_index": int(section["solve_index"]),
                        "iteration": int(event["iteration"]),
                        "relative": float(event["estimate_relative"]),
                        "kind": "arnoldi_estimate",
                    })
                elif event["kind"] == "stage":
                    level = int(event["level"])
                    if level < 0 or level >= len(accumulators):
                        raise BenchmarkFailure(
                            "pyqcu_trace_level_invalid",
                            f"{path}: level={level}")
                    acc = accumulators[level]
                    acc["total_seconds"] += float(event["seconds"])
                    kind = _pyqcu_stage_kind(str(event["name"]))
                    if kind is not None:
                        acc[kind] += float(event["seconds"])
                elif event["kind"] == "iteration_count":
                    level = int(event["level"])
                    if level < 0 or level >= len(accumulators):
                        raise BenchmarkFailure(
                            "pyqcu_trace_level_invalid",
                            f"{path}: level={level}")
                    name = str(event["name"])
                    if name not in _PYQCU_ITERATION_COUNT_FIELDS:
                        raise BenchmarkFailure(
                            "pyqcu_trace_iteration_count_invalid",
                            f"{path}: unknown counter {name!r}")
                    accumulators[level][name] += int(event["value"])
        else:
            for event in events:
                if event["kind"] == "outer_iteration":
                    outer_iterations[int(event["iteration"])] = event
                elif event["kind"] == "stage":
                    level = int(event["level"])
                    if level < 0 or level >= len(accumulators):
                        raise BenchmarkFailure(
                            "quda_mg_trace_level_invalid",
                            f"{path}: level={level}")
                    acc = accumulators[level]
                    seconds = float(event["seconds"])
                    acc["total_seconds"] += seconds
                    kind = _quda_stage_kind(str(event["phase"]))
                    if kind is not None:
                        acc[kind] += seconds
                    pre = int(event["pre_iterations"])
                    post = int(event["post_iterations"])
                    coarse = int(event["coarse_iterations"])
                    if event["phase"] == "pre_smoother_iterations" and pre >= 0:
                        acc["pre_smoother_iterations"] += pre
                    elif event["phase"] == "post_smoother_iterations" and post >= 0:
                        acc["post_smoother_iterations"] += post
                    elif event["phase"] == "coarse_solver_iterations" and coarse >= 0:
                        acc["coarse_solver_iterations"] += coarse
                    elif event["phase"] == "coarsest_solve" and pre >= 0 and post >= 0:
                        acc["pre_smoother_iterations"] += max(0, post - pre)
            for acc in accumulators:
                acc["total_iterations"] += len(outer_iterations)
            for iteration in sorted(outer_iterations):
                event = outer_iterations[iteration]
                true_r2 = float(event["true_r2"])
                relative = (
                    float(event["true_relative"])
                    if true_r2 >= 0.0 else
                    float(event["iterated_relative"]))
                accumulators[0]["residual_sequence"].append({
                    "solve_index": int(section["solve_index"]),
                    "iteration": int(iteration),
                    "relative": relative,
                    "kind": (
                        "gcr_true_residual" if true_r2 >= 0.0 else
                        "gcr_iterated_residual"),
                })
    for acc in accumulators:
        acc.pop("_cycles", None)
    return _finalise_mg_levels(accumulators)


def _fallback_mg_levels(
        config: Mapping[str, Any], timing: Mapping[str, Any],
        iteration_samples: Sequence[int],
        residual_samples: Sequence[float]) -> List[Dict[str, Any]]:
    cold = timing.get("cold_seconds")
    warmups = timing.get("warmups") or []
    steady = timing.get("steady") or {}
    steady_samples = steady.get("samples_seconds") or []
    total_seconds = (
        (0.0 if cold is None else float(cold)) +
        sum(float(item.get("seconds", 0.0)) for item in warmups) +
        sum(float(value) for value in steady_samples))
    total_iterations = sum(int(value) for value in iteration_samples)
    accumulators = [_empty_level_accumulator(level)
                    for level in range(int(config["levels"]))]
    accumulators[0]["total_seconds"] = total_seconds
    for acc in accumulators:
        acc["total_iterations"] = total_iterations
    accumulators[0]["residual_sequence"] = [
        {"solve_index": index, "iteration": None,
         "relative": float(value), "kind": "case_true_residual"}
        for index, value in enumerate(residual_samples)
    ]
    return _finalise_mg_levels(accumulators)


def _git_provenance(repository: Path = REPO) -> Dict[str, Any]:
    def run(*args: str) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", "-C", str(repository), *args], capture_output=True,
                text=True, timeout=10, check=False)
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    dirty_text = run("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": commit,
        "dirty_tracked": None if dirty_text is None else bool(dirty_text),
        "repository": str(repository.resolve()),
    }


def _library_provenance(path: Optional[os.PathLike[str] | str]) -> Dict[str, Any]:
    """Return the resolved binary path and digest without loading the library."""
    if path is None or str(path) == "":
        return {"path": None, "sha256": None, "exists": False}
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        return {"path": str(candidate), "sha256": None, "exists": False}
    return {
        "path": str(candidate),
        "sha256": _sha256_file(candidate),
        "exists": True,
    }


def _pyqcu_library_provenance(
        path: Optional[os.PathLike[str] | str] = None) -> Dict[str, Any]:
    """Return PyQCU's loaded C++ library identity and cubin architectures."""
    if path is None:
        candidates: List[Path] = [
            REPO / "cpp" / "cuda" / "qcu" / "libqcu.so"]
        for entry in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
            if entry:
                candidates.append(Path(entry).expanduser() / "libqcu.so")
        selected = next(
            (candidate for candidate in candidates if candidate.is_file()),
            candidates[0])
    else:
        selected = Path(path).expanduser()
    report = _library_provenance(selected)
    report["requested_architectures"] = os.environ.get(
        "QCU_CUDA_ARCHITECTURES")
    report["architectures"] = []
    report["listing_error"] = None
    if not report["exists"]:
        return report
    tool = shutil.which("cuobjdump")
    if tool is None:
        report["listing_error"] = "cuobjdump not found"
        return report
    try:
        completed = subprocess.run(
            [tool, "--list-elf", report["path"]],
            check=False, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        report["listing_error"] = repr(exc)
        return report
    if completed.returncode != 0:
        report["listing_error"] = (
            f"cuobjdump exited {completed.returncode}: "
            f"{completed.stderr.strip()[-500:]}")
        return report
    report["architectures"] = sorted(set(
        f"sm_{match}" for match in
        re.findall(r"\.sm_(\d+)\.cubin", completed.stdout)))
    if not report["architectures"]:
        report["listing_error"] = "cuobjdump reported no sm_* cubin"
    return report


_QUDA_CMAKE_FEATURE_KEYS = (
    "QUDA_INTERFACE_QDP",
    "QUDA_QIO",
    "QUDA_QMP",
    "BUILD_QDP_INTERFACE",
    "HAVE_QIO",
    "QMP_COMMS",
    "QUDA_RECONSTRUCT",
    "QUDA_PRECISION",
    "QUDA_MULTIGRID_NVEC_LIST",
    "QUDA_MULTIGRID_DOUBLE",
)


def _cmake_value(raw: str) -> Any:
    value = raw.strip()
    upper = value.upper()
    if upper == "ON":
        return True
    if upper == "OFF":
        return False
    try:
        return int(value)
    except ValueError:
        return value


def _cmake_nvec_list(value: Any) -> List[int]:
    if isinstance(value, (list, tuple)):
        values = value
    elif isinstance(value, str):
        values = value.replace(";", ",").replace(" ", ",").split(",")
    else:
        values = []
    result: List[int] = []
    for item in values:
        text = str(item).strip()
        if not text:
            continue
        try:
            result.append(int(text))
        except ValueError:
            return []
    return result


def _normalise_quda_cmake_features(raw: Mapping[str, Any]) -> Dict[str, Any]:
    def first(*keys: str) -> Any:
        for key in keys:
            if key in raw:
                return raw[key]
        return None

    precision = first("QUDA_PRECISION")
    reconstruct = first("QUDA_RECONSTRUCT")
    multigrid_double = first("QUDA_MULTIGRID_DOUBLE")
    if multigrid_double is None:
        # Compile-time double MG is also exposed by the generated config header
        # when a cache predating the CMake option is being inspected.
        multigrid_double = first("GPU_MULTIGRID_DOUBLE")
    return {
        "qdp_interface": first(
            "QUDA_INTERFACE_QDP", "BUILD_QDP_INTERFACE"),
        "qio": first("QUDA_QIO", "HAVE_QIO"),
        "qmp": first("QUDA_QMP", "QMP_COMMS"),
        "reconstruct": reconstruct,
        "precision": precision,
        "multigrid_nvec_list": _cmake_nvec_list(
            first("QUDA_MULTIGRID_NVEC_LIST")),
        "multigrid_double": multigrid_double,
    }


def _quda_cmake_provenance() -> Dict[str, Any]:
    """Read only the bounded CMake cache locations used by this benchmark."""
    candidates: List[Path] = []
    configured = os.environ.get("QUDA_BUILD_DIR")
    if configured:
        configured_path = Path(configured).expanduser()
        candidates.append(
            configured_path / "CMakeCache.txt"
            if configured_path.is_dir() else configured_path)
    # The production build is deliberately outside the source reference tree;
    # retaining its cache path makes patched-vs-upstream provenance explicit.
    candidates.append(REPO / "data" / "quda-qio-build" / "attempt2" /
                      "CMakeCache.txt")
    seen: set[Path] = set()
    for cache_path in candidates:
        cache_path = cache_path.resolve()
        if cache_path in seen or not cache_path.is_file():
            continue
        seen.add(cache_path)
        features: Dict[str, Any] = {}
        try:
            for line in cache_path.read_text(
                    encoding="utf-8", errors="replace").splitlines():
                if not line or line.startswith("#") or ":" not in line:
                    continue
                key_type, separator, raw = line.partition("=")
                if not separator or ":" not in key_type:
                    continue
                key = key_type.split(":", 1)[0]
                if key in _QUDA_CMAKE_FEATURE_KEYS:
                    features[key] = _cmake_value(raw)
        except OSError:
            continue
        return {
            "cache_path": str(cache_path),
            "cache_sha256": _sha256_file(cache_path),
            "features": features,
            "normalized": _normalise_quda_cmake_features(features),
        }
    return {
        "cache_path": None,
        "cache_sha256": None,
        "features": {},
        "normalized": _normalise_quda_cmake_features({}),
    }


def _quda_cmake_feature_mismatches(
        features: Mapping[str, Any], precision: str,
        n_level: int = 2) -> List[str]:
    normalized = features.get("normalized")
    if not isinstance(normalized, Mapping):
        return ["normalized CMake features missing"]
    errors: List[str] = []
    for key in ("qdp_interface", "qio", "qmp"):
        if normalized.get(key) is not True:
            errors.append(f"{key}={normalized.get(key)!r}, expected=True")
    if normalized.get("reconstruct") != 7:
        errors.append(
            f"reconstruct={normalized.get('reconstruct')!r}, expected=7")
    required_precision = 4 if str(precision) == "c64" else 8
    actual_precision = normalized.get("precision")
    if (isinstance(actual_precision, bool) or
            not isinstance(actual_precision, int) or
            actual_precision & required_precision != required_precision):
        errors.append(
            f"precision={actual_precision!r} lacks bit {required_precision}")
    if int(n_level) > 1:
        nvec = normalized.get("multigrid_nvec_list")
        if not isinstance(nvec, list) or 12 not in nvec or 24 not in nvec:
            errors.append(f"multigrid_nvec_list={nvec!r} must contain 12 and 24")
    if str(precision) == "c128" and normalized.get("multigrid_double") is not True:
        errors.append(
            "multigrid_double="
            f"{normalized.get('multigrid_double')!r}, expected=True")
    return errors


def _quda_patch_variant(
        reduction_runtime: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    release = platform.release()
    wsl2 = "microsoft" in release.lower() or "wsl" in release.lower()
    runtime = {} if reduction_runtime is None else reduction_runtime
    marker = runtime.get("marker_present") is True
    enabled = runtime.get("enabled") is True
    if wsl2:
        return {
            "name": ("dev87_wsl2_reduce_sync" if enabled and marker else
                     "wsl2_unpatched_or_unverified"),
            "wsl2": True,
            "environment_scoped": True,
            "limitation": (
                "仅限启用 DEV87_REDUCE_SYNC 的 WSL2 patched QUDA 构建；"
                "不得解释为可移植的 upstream QUDA 证据"),
            "marker_present": marker,
        }
    return {
        "name": "dev87_reduce_sync_opt_in" if marker else "upstream_or_unpatched",
        "wsl2": False,
        "environment_scoped": False,
        "limitation": None,
        "marker_present": marker,
    }


def _benchmark_provenance(
        *, pyquda: Any = None,
        reduction_runtime: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Use stable, separately named provenance fields for both workers."""
    runtime = {} if reduction_runtime is None else reduction_runtime
    qmp_runtime = runtime.get("qmp")
    prefix = os.environ.get("QUDA_INSTALL") or os.environ.get("QUDA_PATH")
    libquda_path = runtime.get("library")
    if libquda_path is None and prefix:
        libquda_path = str(Path(prefix).expanduser() / "lib" / "libquda.so")
    libqmp_path = (
        qmp_runtime.get("library") if isinstance(qmp_runtime, Mapping) else None)
    if libqmp_path is None and prefix:
        libqmp_path = str(Path(prefix).expanduser() / "lib" / "libqmp.so")

    source_text = os.environ.get("QUDA_SOURCE_DIR")
    source_repository = (
        Path(source_text).expanduser() if source_text else
        REPO / "refer" / "git-rep" / "quda")
    module_path = getattr(pyquda, "__file__", None)
    return {
        "pyqcu_git": _git_provenance(),
        "pyqcu_library": _pyqcu_library_provenance(),
        "quda_source_git": _git_provenance(source_repository),
        "quda_libraries": {
            "libquda": _library_provenance(libquda_path),
            "libqmp": _library_provenance(libqmp_path),
        },
        "pyquda_module": {
            "path": (None if module_path is None else
                     str(Path(module_path).expanduser().resolve())),
            "version": (None if pyquda is None else
                        getattr(pyquda, "__version__", None)),
        },
        "cmake_features": _quda_cmake_provenance(),
        "patch_variant": _quda_patch_variant(runtime),
    }


def _environment_snapshot() -> Dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "env": {key: os.environ[key] for key in ENV_ALLOWLIST if key in os.environ},
    }


def _precision_spec(name: str) -> Dict[str, Any]:
    normalized = str(name).lower()
    if normalized == "c64":
        return {
            "name": "c64",
            "complex_bytes": 8,
            "real": "float32",
            "default_tolerance": 1.0e-6,
            "true_residual_gate": 5.0e-6,
        }
    if normalized == "c128":
        return {
            "name": "c128",
            "complex_bytes": 16,
            "real": "float64",
            # The recursive MG solve reaches this gate on the benchmark
            # geometry without turning the comparison into a fp64 floor test.
            "default_tolerance": 1.0e-8,
            "true_residual_gate": 5.0e-8,
        }
    raise ValueError("precision must be c64 or c128")


def _quda_coarse_precision(
        precision: str, requested: str) -> Dict[str, Any]:
    requested = str(requested).lower()
    if requested not in ("auto", "double", "single"):
        raise ValueError(
            "QUDA coarse precision must be auto, double or single")
    if str(precision) == "c64" and requested == "double":
        raise ValueError(
            "c64 cannot request double-precision QUDA coarse fields")
    effective = (
        "double" if (
            str(precision) == "c128" and requested in ("auto", "double"))
        else "single")
    return {
        "requested": requested,
        "effective": effective,
        "semantics": (
            "fine c128 solve with double coarse MG fields"
            if effective == "double" else
            "fine c128/c64 solve with single coarse MG fields"
            if str(precision) == "c128" else
            "c64 solve with single coarse MG fields"),
    }


def _quda_qdp_host_dtype(precision: str) -> Any:
    """Return the host dtype accepted by PyQUDA's QDP ``_NDArray`` bridge.

    QUDA's device precision remains independently controlled by
    ``QudaPrecision``.  PyQUDA 0.10.x rejects a complex64 QDP host array, even
    when the requested device precision is single, so both protocol precisions
    use a complex128 host staging array.
    """
    import numpy as np

    if str(precision).lower() not in ("c64", "c128"):
        raise ValueError("precision must be c64 or c128")
    return np.dtype(np.complex128)


def _quda_qdp_torch_host_dtype(torch_module: Any, precision: str) -> Any:
    """Return the matching PyTorch host staging dtype for the QDP bridge."""
    _quda_qdp_host_dtype(precision)
    return torch_module.complex128


def _formal_profile_defaults(precision: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the precision-dependent values locked by the formal profile."""
    complex_bytes = int(precision["complex_bytes"])
    return {
        "repeats": DEFAULT_REPEATS,
        "tolerance": float(precision["default_tolerance"]),
        "restart": DEFAULT_RESTART,
        "max_iter": DEFAULT_MAX_ITER,
        "max_krylov_bytes": DEFAULT_MAX_KRYLOV_BYTES * complex_bytes // 8,
        "strict_galerkin_column_batch": (
            DEFAULT_STRICT_GALERKIN_COLUMN_BATCH_C64
            if precision["name"] == "c64" else
            DEFAULT_STRICT_GALERKIN_COLUMN_BATCH_C128),
        "strict_galerkin_max_workspace_bytes": (
            DEFAULT_STRICT_GALERKIN_MAX_WORKSPACE_C64
            if precision["name"] == "c64" else
            DEFAULT_STRICT_GALERKIN_MAX_WORKSPACE_C128),
    }


def _profile_name(args: argparse.Namespace) -> str:
    profile = str(getattr(args, "profile", DEFAULT_PROFILE)).lower()
    if profile not in PROFILE_NAMES:
        raise ValueError("profile must be formal or smoke")
    return profile


def _canonical_config(args: argparse.Namespace) -> Dict[str, Any]:
    profile = _profile_name(args)
    precision = _precision_spec(args.precision)
    quda_coarse_precision = _quda_coarse_precision(
        precision["name"], args.quda_coarse_precision)
    lattice = tuple(int(value) for value in args.lattice)
    block = tuple(int(value) for value in args.block)
    levels = int(args.levels)
    if len(lattice) != 4 or any(value <= 0 or value % 2 for value in lattice):
        raise ValueError("--lattice must contain four positive even extents")
    if len(block) != 4 or any(value <= 0 for value in block):
        raise ValueError("--block must contain four positive extents")
    if not 1 <= levels <= 5:
        raise ValueError("--levels must be in [1,5]")
    for axis, (extent, width) in enumerate(zip(lattice, block)):
        if extent % width:
            raise ValueError(
                f"--lattice axis {axis} extent {extent} is not divisible "
                f"by --block width {width}")
    for transition in range(levels - 1):
        coarse = tuple(
            extent // (width ** (transition + 1))
            for extent, width in zip(lattice, block))
        if any(extent < 2 or extent % 2 for extent in coarse):
            raise ValueError(
                f"level {transition + 1} coarse shape {coarse} must have "
                "even extents >= 2")
    coarse_dof = COARSE_DOF
    tolerance = precision["default_tolerance"] if args.tol is None else float(args.tol)
    if tolerance <= 0.0 or not math.isfinite(tolerance):
        raise ValueError("--tol must be a finite positive number")
    coarse_tolerance = (
        min(0.1, 3000.0 * tolerance)
        if args.coarse_tol is None else float(args.coarse_tol))
    if (coarse_tolerance <= 0.0 or coarse_tolerance > 0.5 or
            not math.isfinite(coarse_tolerance)):
        raise ValueError("--coarse-tol must be finite and in (0, 0.5]")
    smoother_steps = int(args.nu)
    if smoother_steps < 0:
        raise ValueError("--nu must be non-negative")
    block_levels = (
        [list(value) for value in args.block_level]
        if args.block_level else
        [list(block) for _ in range(levels - 1)])
    if len(block_levels) != levels - 1:
        raise ValueError(
            "--block-level must be supplied exactly levels-1 times")
    for level, level_block in enumerate(block_levels):
        if any(value <= 0 for value in level_block):
            raise ValueError(
                f"--block-level[{level}] must contain positive extents")
    formal_defaults = _formal_profile_defaults(precision)
    if profile == "formal":
        violations: List[str] = []
        if int(args.repeats) != formal_defaults["repeats"]:
            violations.append(
                f"--repeats={formal_defaults['repeats']} (got {args.repeats})")
        if tolerance != formal_defaults["tolerance"]:
            violations.append(
                f"--tol={formal_defaults['tolerance']!r} (got {tolerance!r})")
        if int(args.restart) != formal_defaults["restart"]:
            violations.append(
                f"--restart={formal_defaults['restart']} (got {args.restart})")
        if int(args.max_iter) != formal_defaults["max_iter"]:
            violations.append(
                f"--max-iter={formal_defaults['max_iter']} (got {args.max_iter})")
        if (args.max_krylov_bytes is not None and
                int(args.max_krylov_bytes) != formal_defaults["max_krylov_bytes"]):
            violations.append(
                "--max-krylov-bytes="
                f"{formal_defaults['max_krylov_bytes']} (got {args.max_krylov_bytes})")
        if (args.strict_galerkin_column_batch is not None and
                int(args.strict_galerkin_column_batch) !=
                formal_defaults["strict_galerkin_column_batch"]):
            violations.append(
                "--strict-galerkin-column-batch="
                f"{formal_defaults['strict_galerkin_column_batch']} "
                f"(got {args.strict_galerkin_column_batch})")
        if (args.strict_galerkin_max_workspace_bytes is not None and
                int(args.strict_galerkin_max_workspace_bytes) !=
                formal_defaults["strict_galerkin_max_workspace_bytes"]):
            violations.append(
                "--strict-galerkin-max-workspace-bytes="
                f"{formal_defaults['strict_galerkin_max_workspace_bytes']} "
                f"(got {args.strict_galerkin_max_workspace_bytes})")
        if violations:
            raise ValueError(
                "formal profile locks the benchmark protocol; "
                + "; ".join(violations)
                + ". Use --profile smoke for exploratory parameters.")
    true_gate = max(float(precision["true_residual_gate"]), 5.0 * tolerance)
    max_krylov_bytes = (
        DEFAULT_MAX_KRYLOV_BYTES * int(precision["complex_bytes"]) // 8
        if args.max_krylov_bytes is None else int(args.max_krylov_bytes))
    setup_column_batch = (
        (DEFAULT_STRICT_GALERKIN_COLUMN_BATCH_C64
         if precision["name"] == "c64" else
         DEFAULT_STRICT_GALERKIN_COLUMN_BATCH_C128)
        if args.strict_galerkin_column_batch is None else
        int(args.strict_galerkin_column_batch))
    setup_workspace_bytes = (
        (DEFAULT_STRICT_GALERKIN_MAX_WORKSPACE_C64
         if precision["name"] == "c64" else
         DEFAULT_STRICT_GALERKIN_MAX_WORKSPACE_C128)
        if args.strict_galerkin_max_workspace_bytes is None else
        int(args.strict_galerkin_max_workspace_bytes))
    setup_projection_batch = (
        DEFAULT_STRICT_GALERKIN_PROJECTION_BATCH
        if args.strict_galerkin_projection_batch is None else
        int(args.strict_galerkin_projection_batch))
    if setup_projection_batch <= 0:
        raise ValueError(
            "--strict-galerkin-projection-batch must be positive")
    setup_workspace_lower_bound = int(
        4 * setup_column_batch * 12 * math.prod(lattice)
        * int(precision["complex_bytes"]))
    fine_vector_bytes = (
        12 * math.prod(lattice) // 2 * int(precision["complex_bytes"]))
    coarse_vector_bytes = (
        coarse_dof * math.prod(
            extent // width for extent, width in zip(lattice, block))
        * int(precision["complex_bytes"]))
    fixed_workspace = 5 * fine_vector_bytes + 2 * coarse_vector_bytes
    per_restart = 2 * fine_vector_bytes
    effective_restart = min(
        int(args.restart), int(args.max_iter),
        max(0, (max_krylov_bytes - fixed_workspace) // per_restart))
    config = {
        "profile": profile,
        "lattice_xyzt": list(lattice),
        "mass": MASS,
        "kappa": 1.0 / (2.0 * MASS + 8.0),
        "seed": SEED,
        "precision": precision,
        "levels": levels,
        "block_xyzt": list(block),
        "block_xyzt_per_level": block_levels,
        "nvec": NVECS,
        "coarse_spin": COARSE_SPIN,
        "coarse_dof": coarse_dof,
        "target_parity": TARGET_PARITY,
        "null_vector_contract": {
            "source": "12 canonical full near-null vectors",
            "full_reconstruction": "precomputed once by zero-rhs Clover block elimination",
            "transfer": "full-field chiral aggregation with coarse_spin=2",
        },
        "outer_solver": "restarted-right-fgmres/gcr",
        "quda_strategy": str(args.quda_strategy),
        "quda_coarse_precision": quda_coarse_precision,
        "pyqcu_strict_block_precision": {
            "requested": str(args.pyqcu_strict_block_precision),
            "effective": (
                "single" if (
                    str(args.pyqcu_strict_block_precision) == "single" or
                    precision["name"] == "c64")
                else "double"),
        },
        "pyqcu_strict_block_device": str(args.pyqcu_strict_block_device),
        "pyqcu_strict_offload_null_basis": bool(
            args.pyqcu_strict_offload_null_basis),
        "reference_solver": {
            "kind": str(args.reference_solver),
            "warmups": int(args.reference_warmups),
            "timing_semantics": (
                "same caller-owned input/output and true-residual gate; "
                "outside MultiGrid and excluded from speedup"),
        },
        "restart_requested": int(args.restart),
        "restart_effective": int(effective_restart),
        "max_krylov_bytes": int(max_krylov_bytes),
        "pyqcu_strict_setup": {
            "probe_mode": "colored",
            "column_batch_size": int(setup_column_batch),
            "projection_site_batch_size": (
                int(setup_projection_batch)),
            "max_workspace_bytes": int(setup_workspace_bytes),
            "workspace_four_arena_lower_bound_bytes": (
                setup_workspace_lower_bound),
            "require_exact_batch": True,
        },
        "fused_workspace_formula": "(2m+5)B_f+2B_c",
        "fused_workspace_terms": {
            "B_f": "one compact fine-parity vector in bytes",
            "B_c": "one full first-coarse-level vector in bytes",
        },
        "max_iter": int(args.max_iter),
        "tolerance": tolerance,
        "true_residual_gate": true_gate,
        "nu_pre": smoother_steps,
        "nu_post": smoother_steps,
        "coarse_max_iter": DEFAULT_COARSE_MAX_ITER,
        "coarse_tolerance": coarse_tolerance,
        "initial_guess": "zero for every warmup and measured solve",
        "warmups": WARMUPS,
        "repeats": int(args.repeats),
        "statistic": {
            "center": "median",
            "spread": "median absolute deviation from median",
            "warmups_included": False,
        },
        "timing_contract": {
            "input_io": "outside setup and solve",
            "runtime_init": "CUDA/PyQUDA module initialization, outside setup",
            "setup": (
                "host inputs -> gauge/clover/runtime ready; includes cache lookup/restore, "
                "excludes cache persistence"),
            "cache_lookup_or_restore": "reported as a disjoint setup component",
            "cache_persist": "reported separately and excluded from setup total",
            "steady_solve": "zero-x0 solve call bracketed by CUDA synchronization",
            "true_residual": "computed after each timed solve and excluded from solve time",
            "first_solve_memory": (
                "the first zero-x0 warmup is sampled for lazy solver/workspace "
                "allocation; excluded from warmup/repeat timing statistics"),
            "memory_probe": (
                "one additional post-timing solve with 10 ms device-wide "
                "cudaMemGetInfo sampling; excluded from warmups/repeats/statistics"),
        },
        "residual_contract": {
            "definition": "||b-D_pyqcu*x_canonical||_2/||b||_2",
            "operator": "full periodic Wilson+Clover, csw=1, PyQCU normalization",
            "quda_solution_scale": MASS + 4.0,
        },
    }
    if levels == 1:
        config["mg_level_semantics"] = {
            "quda": (
                "QUDA_MG_LEVEL=1: no transfer/coarse level; the MG "
                "preconditioner executes the fine-level coarsest branch and "
                "the outer GCR remains the total-iteration authority."),
            "pyqcu": (
                "unsupported by the current strict fused FGMRES C++ hierarchy, "
                "which requires at least two levels"),
            "pair_supported": False,
            "fail_closed": True,
        }
    if config["restart_requested"] <= 0:
        raise ValueError("--restart must be positive")
    if config["max_krylov_bytes"] <= 0:
        raise ValueError("--max-krylov-bytes must be positive")
    if not 1 <= setup_column_batch <= coarse_dof:
        raise ValueError(
            f"--strict-galerkin-column-batch must be in [1,{coarse_dof}]")
    if setup_workspace_bytes <= 0:
        raise ValueError(
            "--strict-galerkin-max-workspace-bytes must be positive")
    if setup_workspace_bytes < setup_workspace_lower_bound:
        raise ValueError(
            "--strict-galerkin-max-workspace-bytes cannot hold the requested "
            f"four-arena batch: budget={setup_workspace_bytes}, "
            f"minimum={setup_workspace_lower_bound}")
    if config["max_iter"] <= 0:
        raise ValueError("--max-iter must be positive")
    if config["repeats"] <= 0:
        raise ValueError("--repeats must be positive")
    if config["restart_effective"] < 1:
        minimum = fixed_workspace + per_restart
        raise ValueError(
            "--max-krylov-bytes cannot hold restart=1: "
            f"budget={max_krylov_bytes}, minimum={minimum}")
    config["config_hash"] = _sha256_json(config)
    return config


def _input_plan(args: argparse.Namespace) -> Dict[str, Any]:
    lattice = tuple(int(value) for value in args.lattice)
    lattice_tag = "x".join(str(value) for value in lattice)
    gauge_path = (
        Path(args.gauge_path).expanduser().resolve()
        if args.gauge_path else
        REPO / "data" / f"gauge_{lattice_tag}_m{MASS:.2f}_seed{SEED}_c64.h5")
    nullvec_path = (
        Path(args.nullvec_path).expanduser().resolve()
        if args.nullvec_path else
        REPO / "data" /
        f"L{lattice_tag}_nvec{NVECS}_full_c64.h5")
    return {
        "gauge": {
            "path": str(gauge_path),
            "dataset": GAUGE_DATASET,
            "storage_precision": "c64",
            "exists": gauge_path.is_file(),
        },
        "source": {
            "path": str(gauge_path),
            "dataset": SOURCE_DATASET,
            "storage_precision": "c64",
            "exists": gauge_path.is_file(),
        },
        "null_vectors": {
            "path": str(nullvec_path),
            "dataset": NULLVEC_DATASET,
            "layout": "[nvec,spin,color,x,y,z,t]",
            "parity": "full (even reconstructed once; odd copied verbatim)",
            "storage_precision": "c64",
            "exists": nullvec_path.is_file(),
        },
        "quda_qio": {
            "prefix": args.quda_nullvec_prefix,
            "conversion_manifest": args.quda_nullvec_manifest,
            "required_for_fair_quda_run": int(args.levels) > 1,
        },
    }


def _activate_runtime_globals(
        config: Mapping[str, Any], inputs: Mapping[str, Any]) -> None:
    """Bind process-local geometry to the worker's serialized protocol."""
    global LATTICE, BLOCK, LEVELS, COARSE_DOF, GAUGE_PATH, NULLVEC_PATH
    LATTICE = tuple(int(value) for value in config["lattice_xyzt"])
    BLOCK = tuple(int(value) for value in config["block_xyzt"])
    LEVELS = int(config["levels"])
    COARSE_DOF = int(config["coarse_dof"])
    GAUGE_PATH = Path(inputs["gauge"]["path"]).expanduser().resolve()
    NULLVEC_PATH = Path(inputs["null_vectors"]["path"]).expanduser().resolve()


def _execution_plan(args: argparse.Namespace) -> Dict[str, Any]:
    cache_dir = Path(args.strict_cache_dir).resolve()
    try:
        cache_dir.relative_to(REPO.resolve())
    except ValueError as exc:
        raise ValueError(
            f"--strict-cache-dir must stay inside {REPO.resolve()}: {cache_dir}") from exc
    ranks = int(args.mpi_ranks)
    if ranks <= 0:
        raise ValueError("--mpi-ranks must be positive")
    grid = _normalise_process_grid(args.process_grid, ranks)
    lattice = tuple(int(value) for value in args.lattice)
    if any(extent % width for extent, width in zip(lattice, grid)):
        raise ValueError(
            f"lattice={lattice} is not divisible by process_grid={grid}")
    phases = _phases_from_args(args)
    return {
        "strict_cache": {
            "directory": str(cache_dir),
            "expect": str(args.cache_expect),
        },
        "phases": {
            "selected": list(phases),
            "counts": _phase_counts(phases, int(args.repeats)),
            "cold_separate_process": (
                "cold" in phases and
                any(phase in phases for phase in ("warmup", "steady"))),
        },
        "device": {
            "requested": str(args.device),
        },
        "parallel": {
            "ranks": ranks,
            "process_grid": list(grid),
            "launcher": "mpirun" if ranks > 1 else "direct",
            "rank_local_cache": ranks > 1,
            "rank_suffixed_trace": ranks > 1,
        },
    }


def _side_placeholder(side: str, selected: bool, dry_run: bool) -> Dict[str, Any]:
    return {
        "side": side,
        "status": "planned" if selected else "not_selected",
        "dry_run": bool(dry_run),
        "reason": None,
    }


def build_document(args: argparse.Namespace, *, dry_run: bool) -> Dict[str, Any]:
    """构造不触发重型依赖的 v1 JSON 文档。供 CLI 与协议测试复用。"""
    config = _canonical_config(args)
    selected = set(SIDE_NAMES if args.side == "both" else (args.side,))
    phases = _phases_from_args(args)
    split_cold_steady = (
        "cold" in phases and
        any(phase in phases for phase in ("warmup", "steady")))
    if (not dry_run and config["profile"] == "formal" and
            "pyqcu" in selected and not split_cold_steady and
            args.cache_expect != "hit"):
        raise ValueError(
            "formal profile requires --cache-expect hit when pyqcu is selected; "
            "dry-run may display cache-expect any")
    now = _utc_now()
    document = {
        "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "run_id": config["config_hash"][:20],
        "state": "dry-run" if dry_run else "partial",
        "profile": config["profile"],
        "created_at": now,
        "updated_at": now,
        "selected_sides": sorted(selected),
        "protocol": config,
        "inputs": _input_plan(args),
        "execution": _execution_plan(args),
        "input_fingerprints": None,
        "collector": {
            "path": str(Path(__file__).resolve()),
            "git": _git_provenance(),
            "environment": _environment_snapshot(),
            "timeout_seconds_per_side": float(args.timeout),
            "resume": bool(args.resume),
        },
        "sides": {
            side: _side_placeholder(side, side in selected, dry_run)
            for side in SIDE_NAMES
        },
        "comparison": (
            {
                "status": "smoke-pending",
                "profile": config["profile"],
                "fair": False,
                "reasons": ["smoke: no backend executed"] if dry_run else [],
                "speedup_pyqcu_over_quda": None,
            }
            if config["profile"] == "smoke" else
            {
                "status": "pending",
                "profile": config["profile"],
                "fair": None,
                "reasons": ["dry-run: no backend executed"] if dry_run else [],
                "speedup_pyqcu_over_quda": None,
            }
        ),
    }
    errors = validate_document(document, allow_planned=True)
    if errors:
        raise RuntimeError("internal protocol error: " + "; ".join(errors))
    return document


def _hash_hdf5_dataset(path: Path, dataset: str) -> Dict[str, Any]:
    """按第一轴流式计算逻辑 dataset SHA256，避免一次复制多 GiB。"""
    try:
        import h5py
        import numpy as np
    except ImportError as exc:
        raise BenchmarkSkip("missing_hdf5_dependency", repr(exc)) from exc
    if not path.is_file():
        raise BenchmarkSkip("missing_input", f"missing input file: {path}")
    digest = hashlib.sha256()
    started = time.perf_counter()
    with h5py.File(path, "r") as handle:
        if dataset not in handle:
            raise BenchmarkFailure(
                "missing_dataset", f"{path} does not contain dataset {dataset!r}")
        value = handle[dataset]
        shape = tuple(int(x) for x in value.shape)
        dtype = str(value.dtype)
        digest.update(_json_bytes({"dataset": dataset, "shape": shape, "dtype": dtype}))
        if value.ndim == 0:
            digest.update(np.ascontiguousarray(value[()]).tobytes())
        else:
            for index in range(shape[0]):
                digest.update(np.ascontiguousarray(value[index]).tobytes(order="C"))
    return {
        "algorithm": "sha256(logical-hdf5-dataset-v1)",
        "sha256": digest.hexdigest(),
        "path": str(path.resolve()),
        "dataset": dataset,
        "shape": list(shape),
        "dtype": dtype,
        "file_size_bytes": int(path.stat().st_size),
        "file_mtime_ns": int(path.stat().st_mtime_ns),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _hash_loaded_array(array: Any, dataset: str) -> str:
    """使用与 HDF5 流式指纹完全相同的逻辑布局验证 worker 实际输入。"""
    import numpy as np

    value = np.asarray(array)
    shape = tuple(int(x) for x in value.shape)
    digest = hashlib.sha256()
    digest.update(_json_bytes({
        "dataset": dataset,
        "shape": shape,
        "dtype": str(value.dtype),
    }))
    if value.ndim == 0:
        digest.update(np.ascontiguousarray(value).tobytes())
    else:
        for index in range(shape[0]):
            digest.update(np.ascontiguousarray(value[index]).tobytes(order="C"))
    return digest.hexdigest()


def _verify_loaded_input(
        name: str, array: Any, fingerprint: Mapping[str, Any]) -> None:
    observed = _hash_loaded_array(array, str(fingerprint["dataset"]))
    expected = fingerprint.get("sha256")
    if observed != expected:
        raise BenchmarkFailure(
            "input_changed_after_preflight",
            f"{name} loaded sha256={observed}, preflight sha256={expected}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(8 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _binary_contains(path: Path, marker: bytes, chunk_bytes: int = 8 << 20) -> bool:
    overlap = b""
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                return False
            combined = overlap + chunk
            if marker in combined:
                return True
            overlap = combined[-max(0, len(marker) - 1):]


def _prepare_quda_reduction_runtime() -> Dict[str, Any]:
    """Fail closed on WSL2 unless the opt-in QUDA sync fallback is loaded."""
    release = platform.release()
    required = "microsoft" in release.lower() or "wsl" in release.lower()
    report: Dict[str, Any] = {
        "required": required,
        "platform_release": release,
        "enabled": False,
        "library": None,
        "library_sha256": None,
        "marker_present": None,
    }
    if not required:
        prefix_text = os.environ.get("QUDA_INSTALL") or os.environ.get("QUDA_PATH")
        if prefix_text:
            library = Path(prefix_text).expanduser().resolve() / "lib" / "libquda.so"
            if library.is_file():
                try:
                    ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
                except OSError as exc:
                    raise BenchmarkFailure(
                        "quda_library_preload_failed",
                        f"{library}: {exc!r}") from exc
                report.update({
                    "library": str(library),
                    "library_sha256": _sha256_file(library),
                    "marker_present": _binary_contains(
                        library, b"DEV87_REDUCE_SYNC"),
                })
        return report

    configured = os.environ.get("DEV87_REDUCE_SYNC")
    if configured is None:
        os.environ["DEV87_REDUCE_SYNC"] = "1"
        configured = "1"
    if configured.strip().lower() in ("", "0", "false", "no", "off"):
        raise BenchmarkFailure(
            "quda_wsl2_reduce_sync_disabled",
            f"DEV87_REDUCE_SYNC={configured!r}")

    prefix_text = os.environ.get("QUDA_INSTALL") or os.environ.get("QUDA_PATH")
    if not prefix_text:
        raise BenchmarkFailure(
            "quda_wsl2_install_missing",
            "QUDA_INSTALL or QUDA_PATH must identify the patched install")
    library_dir = (Path(prefix_text).resolve() / "lib")
    library = library_dir / "libquda.so"
    if not library.is_file() or library.is_symlink():
        raise BenchmarkFailure("quda_wsl2_library_missing", str(library))

    marker = b"DEV87_REDUCE_SYNC"
    marker_present = _binary_contains(library, marker)
    if not marker_present:
        raise BenchmarkFailure("quda_wsl2_reduce_sync_missing", str(library))
    try:
        ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
    except OSError as exc:
        raise BenchmarkFailure(
            "quda_library_preload_failed", f"{library}: {exc!r}") from exc
    report.update({
        "enabled": True,
        "library": str(library),
        "library_sha256": _sha256_file(library),
        "marker_present": True,
    })
    return report


def _initialize_quda_qmp_runtime(
        reduction_runtime: Mapping[str, Any],
        topology: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """Initialize the QMP transport before PyQUDA calls initCommsGridQuda."""
    expected_topology = tuple(
        int(value) for value in (
            topology if topology is not None else (1, 1, 1, 1)))
    if len(expected_topology) != 4 or any(
            value <= 0 for value in expected_topology):
        raise ValueError("QMP topology must contain four positive extents")
    quda_library = reduction_runtime.get("library")
    if isinstance(quda_library, str):
        qmp_library = Path(quda_library).resolve().with_name("libqmp.so")
    else:
        prefix = os.environ.get("QUDA_INSTALL") or os.environ.get("QUDA_PATH")
        if not prefix:
            raise BenchmarkFailure(
                "quda_qmp_install_missing",
                "QUDA_INSTALL or QUDA_PATH is required for libqmp.so")
        qmp_library = Path(prefix).resolve() / "lib" / "libqmp.so"
    if not qmp_library.is_file() or qmp_library.is_symlink():
        raise BenchmarkFailure("quda_qmp_library_missing", str(qmp_library))

    try:
        qmp = ctypes.CDLL(str(qmp_library), mode=ctypes.RTLD_GLOBAL)
        qmp.QMP_is_initialized.argtypes = []
        qmp.QMP_is_initialized.restype = ctypes.c_int
        qmp.QMP_logical_topology_is_declared.argtypes = []
        qmp.QMP_logical_topology_is_declared.restype = ctypes.c_int
        qmp.QMP_declare_logical_topology.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        qmp.QMP_declare_logical_topology.restype = ctypes.c_int
        qmp.QMP_get_logical_number_of_dimensions.argtypes = []
        qmp.QMP_get_logical_number_of_dimensions.restype = ctypes.c_int
        qmp.QMP_get_logical_dimensions.argtypes = []
        qmp.QMP_get_logical_dimensions.restype = ctypes.POINTER(ctypes.c_int)
        qmp.QMP_finalize_msg_passing.argtypes = []
        qmp.QMP_finalize_msg_passing.restype = None
    except (OSError, AttributeError) as exc:
        raise BenchmarkFailure(
            "quda_qmp_load_failed", f"{qmp_library}: {exc!r}") from exc

    initialized_here = qmp.QMP_is_initialized() != 1
    provided_value: Optional[int] = None
    if initialized_here:
        qmp.QMP_init_msg_passing.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_char_p)),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
        ]
        qmp.QMP_init_msg_passing.restype = ctypes.c_int
        argc = ctypes.c_int(1)
        argv_storage = (ctypes.c_char_p * 2)(
            b"pyqcu-strict-quda-worker", None)
        argv = ctypes.cast(argv_storage, ctypes.POINTER(ctypes.c_char_p))
        provided = ctypes.c_int(-1)
        status = int(qmp.QMP_init_msg_passing(
            ctypes.byref(argc), ctypes.byref(argv), 1,
            ctypes.byref(provided)))
        if status != 0 or qmp.QMP_is_initialized() != 1:
            raise BenchmarkFailure(
                "quda_qmp_init_failed",
                f"status={status}, provided={provided.value}")
        provided_value = int(provided.value)
        # PyQUDA registers endQuda after this helper returns.  atexit is LIFO,
        # therefore endQuda executes before QMP finalization.
        atexit.register(qmp.QMP_finalize_msg_passing)
        _QMP_RUNTIME_HOLD.extend((qmp, argv_storage, argv))
    else:
        _QMP_RUNTIME_HOLD.append(qmp)
    # QIO's QUDA layout helper assumes that QMP already owns a logical
    # topology.  A plain single-rank QMP init does not necessarily declare
    # one; without this explicit declaration qio_field.cpp passes an
    # uninitialised dimension array to QMP_declare_logical_topology and aborts
    # as soon as a MultiGrid QIO null-vector file is read.
    topology_declared = bool(qmp.QMP_logical_topology_is_declared())
    topology = None
    if not topology_declared:
        topology_dims = (ctypes.c_int * 4)(*expected_topology)
        status = int(qmp.QMP_declare_logical_topology(topology_dims, 4))
        if status != 0 or not qmp.QMP_logical_topology_is_declared():
            raise BenchmarkFailure(
                "quda_qmp_topology_init_failed",
                f"status={status}; unable to declare {list(expected_topology)}")
        topology_declared = True
    if topology_declared:
        ndim = int(qmp.QMP_get_logical_number_of_dimensions())
        dims_ptr = qmp.QMP_get_logical_dimensions()
        if ndim <= 0 or not bool(dims_ptr):
            raise BenchmarkFailure(
                "quda_qmp_topology_readback_failed",
                f"ndim={ndim}")
        topology = [int(dims_ptr[index]) for index in range(ndim)]
        if topology != list(expected_topology):
            raise BenchmarkFailure(
                "quda_qmp_topology_mismatch",
                f"expected {list(expected_topology)}, got {topology}")
    return {
        "library": str(qmp_library),
        "library_sha256": _sha256_file(qmp_library),
        "initialized": True,
        "initialized_here": initialized_here,
        "thread_level_required": 1,
        "thread_level_provided": provided_value,
        "logical_topology_declared": topology_declared,
        "logical_topology": topology,
    }


def _fingerprint_inputs(document: MutableMapping[str, Any]) -> Dict[str, Any]:
    plans = document["inputs"]
    protocol = document.get("protocol", {})
    levels = int(protocol.get("levels", LEVELS))
    lattice = tuple(int(value) for value in protocol.get(
        "lattice_xyzt", LATTICE))
    nvec = int(protocol.get("nvec", NVECS))
    gauge = _hash_hdf5_dataset(Path(plans["gauge"]["path"]), plans["gauge"]["dataset"])
    source = _hash_hdf5_dataset(Path(plans["source"]["path"]), plans["source"]["dataset"])
    checkerboard_shape = [*lattice[:3], lattice[3] // 2]
    expected_shapes = {
        "gauge": [2, 3, 3, 4, *checkerboard_shape],
        "source": [2, 4, 3, *checkerboard_shape],
    }
    result = {"gauge": gauge, "source": source}
    if levels > 1:
        nullvec = _hash_hdf5_dataset(
            Path(plans["null_vectors"]["path"]),
            plans["null_vectors"]["dataset"])
        # The formal protocol consumes the canonical full vectors directly.
        # The old E12 odd-Schur packing is provenance only and must never be
        # mistaken for the runtime transfer field.
        expected_shapes["null_vectors"] = [nvec, 4, 3, *lattice]
        result["null_vectors"] = nullvec
    for name, fingerprint in result.items():
        if fingerprint["shape"] != expected_shapes[name]:
            raise BenchmarkFailure(
                "input_shape_mismatch",
                f"{name} shape={fingerprint['shape']} expected={expected_shapes[name]}")
    result["bundle_hash"] = _sha256_json({
        key: {"sha256": value["sha256"], "shape": value["shape"],
              "dtype": value["dtype"]}
        for key, value in result.items() if isinstance(value, dict)
    })
    return result


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _input_fingerprint_identity(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Select immutable input facts used to decide whether resume is safe."""
    return {
        key: copy.deepcopy(value.get(key))
        for key in (
            "algorithm", "sha256", "path", "dataset", "shape", "dtype",
            "file_size_bytes")
    }


def _input_fingerprint_mismatches(
        previous: Any, current: Mapping[str, Any]) -> List[str]:
    if not isinstance(previous, Mapping):
        return ["previous input_fingerprints is missing"]
    mismatches: List[str] = []
    if previous.get("bundle_hash") != current.get("bundle_hash"):
        mismatches.append(
            "bundle_hash changed: "
            f"previous={previous.get('bundle_hash')!r}, "
            f"current={current.get('bundle_hash')!r}")
    names = ["gauge", "source"]
    if "null_vectors" in previous or "null_vectors" in current:
        names.append("null_vectors")
    for name in names:
        old_value = previous.get(name)
        new_value = current.get(name)
        if not isinstance(old_value, Mapping):
            mismatches.append(f"previous {name} fingerprint is missing")
        elif not isinstance(new_value, Mapping):
            mismatches.append(f"current {name} fingerprint is missing")
        elif _input_fingerprint_identity(old_value) != \
                _input_fingerprint_identity(new_value):
            mismatches.append(f"{name} fingerprint changed")
    return mismatches


def _validate_input_fingerprints(
        value: Any, *, required: bool = False,
        levels: int = LEVELS) -> List[str]:
    if value is None:
        return ["input_fingerprints missing"] if required else []
    if not isinstance(value, Mapping):
        return ["input_fingerprints must be an object"]
    errors: List[str] = []
    if not _is_sha256(value.get("bundle_hash")):
        errors.append("input_fingerprints.bundle_hash is not sha256")
    entries: Dict[str, Any] = {}
    valid_entries: Dict[str, Any] = {}
    names = ["gauge", "source"]
    if int(levels) > 1 or (
            isinstance(value, Mapping) and "null_vectors" in value):
        names.append("null_vectors")
    for name in names:
        entry = value.get(name)
        if not isinstance(entry, Mapping):
            errors.append(f"input_fingerprints.{name} missing")
            continue
        entries[name] = entry
        entry_errors = False
        if entry.get("algorithm") != "sha256(logical-hdf5-dataset-v1)":
            errors.append(f"input_fingerprints.{name}.algorithm invalid")
            entry_errors = True
        if not _is_sha256(entry.get("sha256")):
            errors.append(f"input_fingerprints.{name}.sha256 invalid")
            entry_errors = True
        if not isinstance(entry.get("path"), str) or not entry["path"]:
            errors.append(f"input_fingerprints.{name}.path invalid")
            entry_errors = True
        if not isinstance(entry.get("dataset"), str) or not entry["dataset"]:
            errors.append(f"input_fingerprints.{name}.dataset invalid")
            entry_errors = True
        shape = entry.get("shape")
        if (not isinstance(shape, list) or not shape or
                any(isinstance(extent, bool) or not isinstance(extent, int) or
                    extent <= 0 for extent in shape)):
            errors.append(f"input_fingerprints.{name}.shape invalid")
            entry_errors = True
        if not isinstance(entry.get("dtype"), str) or not entry["dtype"]:
            errors.append(f"input_fingerprints.{name}.dtype invalid")
            entry_errors = True
        file_size = entry.get("file_size_bytes")
        if (isinstance(file_size, bool) or not isinstance(file_size, int) or
                file_size < 0):
            errors.append(f"input_fingerprints.{name}.file_size_bytes invalid")
            entry_errors = True
        if not entry_errors:
            valid_entries[name] = entry
    if len(entries) == len(names) and len(valid_entries) == len(names):
        expected_bundle = _sha256_json({
            key: {
                "sha256": item["sha256"],
                "shape": item["shape"],
                "dtype": item["dtype"],
            }
            for key, item in valid_entries.items()
        })
        if value.get("bundle_hash") != expected_bundle:
            errors.append("input_fingerprints.bundle_hash does not match entries")
    return errors


def _median_mad(values: Sequence[float]) -> Dict[str, Any]:
    samples = [float(value) for value in values]
    if not samples or any(not math.isfinite(value) or value < 0.0 for value in samples):
        raise BenchmarkFailure("invalid_samples", f"invalid timing samples: {samples}")
    center = float(statistics.median(samples))
    mad = float(statistics.median(abs(value - center) for value in samples))
    return {"samples_seconds": samples, "median_seconds": center, "mad_seconds": mad}


def _validate_median_mad(
        summary: Any, repeats: int, label: str) -> List[str]:
    if not isinstance(summary, Mapping):
        return [f"{label} summary missing"]
    samples = summary.get("samples_seconds")
    if (not isinstance(samples, list) or len(samples) != int(repeats) or
            any(isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(float(value)) or float(value) < 0.0
                for value in samples)):
        return [f"{label} samples invalid"]
    expected_median = float(statistics.median(float(value) for value in samples))
    expected_mad = float(statistics.median(
        abs(float(value) - expected_median) for value in samples))
    errors: List[str] = []
    for key, expected in (
            ("median_seconds", expected_median), ("mad_seconds", expected_mad)):
        actual = summary.get(key)
        if (isinstance(actual, bool) or not isinstance(actual, (int, float)) or
                not math.isfinite(float(actual)) or float(actual) != expected):
            errors.append(
                f"{label}.{key} does not match recomputed value {expected!r}")
    return errors


def _iteration_summary(values: Sequence[int]) -> Dict[str, Any]:
    samples = [int(value) for value in values]
    if not samples or any(value < 0 for value in samples):
        raise BenchmarkFailure("invalid_iterations", f"invalid iterations: {samples}")
    return {
        "samples": samples,
        "min": min(samples),
        "max": max(samples),
        "median": float(statistics.median(samples)),
    }


def _nvidia_smi_used(
        pid: Optional[int] = None,
        device_uuid: Optional[str] = None,
) -> Dict[str, Any]:
    """读取目标 GPU 上的本进程/设备显存快照。"""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "process_used_bytes": None,
                "gpu_used_bytes": None, "error": "nvidia-smi not found"}
    pid = os.getpid() if pid is None else int(pid)
    result: Dict[str, Any] = {
        "available": True,
        "device_uuid": device_uuid,
        "process_used_bytes": None,
        "gpu_used_bytes": None,
        "error": None,
    }
    try:
        process = subprocess.run(
            [executable, "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
             "--format=csv,noheader,nounits"], capture_output=True,
            text=True, timeout=10, check=False)
        if process.returncode == 0:
            used_mib = 0
            found = False
            for line in process.stdout.splitlines():
                fields = [part.strip() for part in line.split(",")]
                if len(fields) < 3:
                    continue
                try:
                    if (int(fields[1]) == pid and
                            (device_uuid is None or fields[0] == device_uuid)):
                        used_mib += int(float(fields[2]))
                        found = True
                except ValueError:
                    continue
            if found:
                result["process_used_bytes"] = used_mib << 20
        gpu = subprocess.run(
            [executable, "--query-gpu=uuid,memory.used",
             "--format=csv,noheader,nounits"], capture_output=True,
            text=True, timeout=10, check=False)
        if gpu.returncode == 0:
            values = []
            for line in gpu.stdout.splitlines():
                fields = [part.strip() for part in line.split(",")]
                if len(fields) < 2 or (
                        device_uuid is not None and fields[0] != device_uuid):
                    continue
                try:
                    values.append(int(float(fields[1])) << 20)
                except ValueError:
                    pass
            if values:
                result["gpu_used_bytes"] = max(values)
        if result["process_used_bytes"] is None and result["gpu_used_bytes"] is None:
            result["error"] = _safe_tail(process.stderr or gpu.stderr, 500) or "no numeric sample"
    except (OSError, subprocess.SubprocessError) as exc:
        result.update(available=False, error=repr(exc))
    return result


def _max_optional(values: Iterable[Optional[int]]) -> Optional[int]:
    present = [int(value) for value in values if value is not None]
    return max(present) if present else None


class _CudaDeviceMemorySampler:
    """Low-overhead device-wide cudaMemGetInfo sampler.

    This observes native CUDA allocations that are invisible to PyTorch's
    allocator.  It is intentionally reported as device-wide (other processes
    may contribute), and formal solve timing uses a separate untimed probe.
    """

    def __init__(self, torch: Any, device: Any, interval_seconds: float = 0.01):
        self.torch = torch
        self.device = device
        self.interval_seconds = float(interval_seconds)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._samples = 0
        self._high_water = 0
        self._initial_used: Optional[int] = None
        self._total = 0
        self._errors: List[str] = []
        self._started_at: Optional[float] = None
        self._stopped_at: Optional[float] = None
        self._join_timed_out = False

    def _sample(self) -> None:
        try:
            free, total = self.torch.cuda.mem_get_info(self.device)
            used = int(total) - int(free)
            self._samples += 1
            self._total = int(total)
            if self._initial_used is None:
                self._initial_used = used
            self._high_water = max(self._high_water, used)
        except Exception as exc:  # best-effort telemetry must not mask solve
            if not self._errors:
                self._errors.append(repr(exc))

    def _run(self) -> None:
        self._sample()
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def start(self) -> "_CudaDeviceMemorySampler":
        if self._thread is not None:
            return self
        self._stop.clear()
        self._started_at = time.perf_counter()
        self._stopped_at = None
        self._join_timed_out = False
        self._thread = threading.Thread(
            target=self._run, name="pyqcu-cuda-memory-sampler", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> Dict[str, Any]:
        thread = self._thread
        if thread is not None:
            self._stop.set()
            try:
                thread.join(timeout=max(1.0, 5.0 * self.interval_seconds))
                self._join_timed_out = bool(thread.is_alive())
            except Exception as exc:  # telemetry cleanup must not mask solve
                self._join_timed_out = True
                if not self._errors:
                    self._errors.append(f"sampler join failed: {exc!r}")
            if not self._join_timed_out:
                self._thread = None
            elif not self._errors:
                self._errors.append("sampler thread did not stop before join timeout")
        self._stopped_at = time.perf_counter()
        duration = (
            None if self._started_at is None else
            max(0.0, self._stopped_at - self._started_at))
        return {
            "available": self._samples > 0,
            "scope": "device-wide cudaMemGetInfo; may include other processes",
            "device": str(self.device),
            "unit": "bytes",
            "interval_seconds": self.interval_seconds,
            "duration_seconds": duration,
            "sample_count": self._samples,
            "device_total_bytes": self._total or None,
            "device_used_initial_bytes": self._initial_used,
            "device_used_max_observed_bytes": (
                self._high_water if self._samples else None),
            "join_timed_out": self._join_timed_out,
            "errors": list(self._errors),
        }


def _load_h5_array(path: str, dataset: str) -> Any:
    try:
        import h5py
        import numpy as np
    except ImportError as exc:
        raise BenchmarkSkip("missing_hdf5_dependency", repr(exc)) from exc
    with h5py.File(path, "r") as handle:
        if dataset not in handle:
            raise BenchmarkFailure("missing_dataset", f"{path}: {dataset}")
        return np.ascontiguousarray(handle[dataset][...])


def _load_h5_local_array(
        path: str, dataset: str, global_lattice: Sequence[int],
        mpi_context: Mapping[str, Any], *, parity_compressed: bool) -> Any:
    """Read only this rank's rectangular slab from a global HDF5 dataset.

    The gauge/source files are checkerboard-compressed on the final axis.
    Since every local extent and origin is even, the checkerboard parity on a
    rank block agrees with the global parity, so a rectangular HDF5 hyperslab
    is already the required rank-local compressed field.
    """
    try:
        import h5py
        import numpy as np
    except ImportError as exc:
        raise BenchmarkSkip("missing_hdf5_dependency", repr(exc)) from exc
    size = int(mpi_context["size"])
    if size <= 1:
        return _load_h5_array(path, dataset)
    grid = tuple(int(value) for value in mpi_context["process_grid"])
    coordinate = tuple(int(value) for value in mpi_context["rank_coordinate"])
    local = [int(global_lattice[d]) // grid[d] for d in range(4)]
    starts = [coordinate[d] * local[d] for d in range(4)]
    expected = (
        [int(global_lattice[0]), int(global_lattice[1]),
         int(global_lattice[2]),
         int(global_lattice[3]) // (2 if parity_compressed else 1)]
    )
    t_count = local[3] // (2 if parity_compressed else 1)
    t_start = starts[3] // (2 if parity_compressed else 1)
    try:
        from mpi4py import MPI
        driver = "mpio" if bool(getattr(
            __import__("pyqcu.tools", fromlist=["HAS_MPI_SUPPORT"]),
            "HAS_MPI_SUPPORT", False)) else None
        comm = MPI.COMM_WORLD
    except (ImportError, OSError):
        driver = None
        comm = None
    if driver is None:
        raise BenchmarkSkip(
            "hdf5_mpi_unavailable",
            "rank-sliced HDF5 reads require h5py MPI support")
    with h5py.File(path, "r", driver=driver, comm=comm) as handle:
        if dataset not in handle:
            raise BenchmarkFailure("missing_dataset", f"{path}: {dataset}")
        value = handle[dataset]
        if tuple(int(x) for x in value.shape[-4:]) != tuple(expected):
            raise BenchmarkFailure(
                "input_shape_mismatch",
                f"{dataset} shape={tuple(value.shape)} expected_tail={expected}")
        selection = [slice(None)] * (value.ndim - 4) + [
            slice(starts[0], starts[0] + local[0]),
            slice(starts[1], starts[1] + local[1]),
            slice(starts[2], starts[2] + local[2]),
            slice(t_start, t_start + t_count),
        ]
        return np.ascontiguousarray(value[tuple(selection)])


def _verify_local_input_slab(
        name: str, array: Any, fingerprint: Mapping[str, Any],
        local_lattice: Sequence[int], *, parity_compressed: bool) -> None:
    import numpy as np

    expected = [
        int(local_lattice[0]), int(local_lattice[1]),
        int(local_lattice[2]),
        int(local_lattice[3]) // (2 if parity_compressed else 1),
    ]
    if tuple(int(x) for x in np.asarray(array).shape[-4:]) != tuple(expected):
        raise BenchmarkFailure(
            "input_local_shape_mismatch",
            f"{name} local shape={tuple(np.asarray(array).shape)} "
            f"expected_tail={expected}")
    if str(np.asarray(array).dtype) != str(fingerprint["dtype"]):
        raise BenchmarkFailure(
            "input_local_dtype_mismatch",
            f"{name} dtype={np.asarray(array).dtype} "
            f"expected={fingerprint['dtype']}")
    path = Path(str(fingerprint["path"]))
    if (not path.is_file() or
            int(path.stat().st_size) != int(fingerprint["file_size_bytes"])):
        raise BenchmarkFailure(
            "input_changed_after_preflight",
            f"{name} file metadata changed: {path}")


def _torch_runtime_provenance(torch: Any, device: Any) -> Dict[str, Any]:
    index = int(torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(index)
    return {
        "torch": getattr(torch, "__version__", None),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "device_index": index,
        "device_name": torch.cuda.get_device_name(index),
        # Recent PyTorch exposes ``uuid`` as a private ``_CUuuid`` object,
        # which looks printable but is not accepted by ``json.dumps``.
        "device_uuid": (
            None if getattr(properties, "uuid", None) is None else
            str(properties.uuid)),
        "total_memory_bytes": int(properties.total_memory),
        "compute_capability": [int(properties.major), int(properties.minor)],
        "device": str(device),
    }


def _select_accelerator(
        torch: Any, requested: str = "v100", rank: int = 0,
        world_size: int = 1) -> Any:
    requested = str(requested).lower()
    if requested not in DEVICE_TARGETS:
        raise ValueError(f"device target must be one of {DEVICE_TARGETS}")
    if not torch.cuda.is_available():
        raise BenchmarkSkip("cuda_unavailable", "torch.cuda.is_available() is false")

    override = os.environ.get("QCU_DEVICE_ID")
    if override is not None and override.strip():
        indices = [int(value.strip()) for value in override.split(",")
                   if value.strip()]
        if not indices:
            raise ValueError("QCU_DEVICE_ID must contain at least one device index")
        if int(world_size) > len(indices):
            raise BenchmarkSkip(
                "insufficient_visible_devices",
                f"world_size={world_size}, QCU_DEVICE_ID entries={indices}")
        index = indices[int(rank)]
    else:
        names = [torch.cuda.get_device_name(index)
                 for index in range(torch.cuda.device_count())]
        if requested == "auto":
            family_matches = [
                index for index, name in enumerate(names)
                if "P100" in name or "V100" in name]
            family_matches.sort(
                key=lambda index: 0 if "P100" in names[index] else 1)
        else:
            marker = requested.upper()
            family_matches = [
                index for index, name in enumerate(names)
                if marker in name]
        matches = family_matches
        if not matches:
            raise BenchmarkSkip(
                f"{requested}_unavailable",
                f"no {requested} device found")
        if int(world_size) > len(matches):
            raise BenchmarkSkip(
                "insufficient_target_devices",
                f"world_size={world_size}, matching {requested} devices={matches}")
        index = matches[int(rank)]
    torch.cuda.set_device(index)
    name = torch.cuda.get_device_name(index)
    properties = torch.cuda.get_device_properties(index)
    compute_capability = (int(properties.major), int(properties.minor))
    if requested == "v100" and "V100" not in name:
        raise BenchmarkSkip(
            "wrong_gpu", f"requested V100, selected {name!r}")
    if requested == "p100" and "P100" not in name:
        raise BenchmarkSkip(
            "wrong_gpu", f"requested P100, selected {name!r}")
    if requested == "auto" and (
            "V100" not in name and "P100" not in name):
        raise BenchmarkSkip(
            "unsupported_gpu", f"requested auto V100/P100, selected {name!r}")
    if requested == "p100" and compute_capability != (6, 0):
        raise BenchmarkSkip(
            "wrong_compute_capability",
            f"P100 requires sm_60, selected {compute_capability}")
    if requested == "v100" and compute_capability != (7, 0):
        raise BenchmarkSkip(
            "wrong_compute_capability",
            f"V100 requires sm_70, selected {compute_capability}")
    return torch.device("cuda", index)


def _select_v100(torch: Any) -> Any:
    """Backward-compatible wrapper for callers that require the old default."""
    return _select_accelerator(torch, "v100", 0)


def _canonical_true_residual(
        solution_eo: Any, rhs_eo: Any, gauge_eo: Any, mass: float,
        *, full_gauge: Any = None, clover: Any = None) -> float:
    """双方共用的 full Wilson/Clover 真相对残差。"""
    try:
        import torch
        from pyqcu import dslash, tools
    except (ImportError, OSError) as exc:
        raise BenchmarkSkip("missing_pyqcu_residual_dependency", repr(exc)) from exc
    kappa = 1.0 / (2.0 * float(mass) + 8.0)
    if full_gauge is None:
        full_gauge = tools.poooxyzt2oooxyzt(gauge_eo)
    if clover is None:
        real_dtype = torch.float32 if gauge_eo.dtype == torch.complex64 else torch.float64
        clover = dslash.make_clover(
            full_gauge,
            kappa=torch.tensor([kappa], dtype=real_dtype,
                               device=full_gauge.device),
            u_0=torch.ones([1], dtype=real_dtype, device=full_gauge.device))
    full_solution = tools.poooxyzt2oooxyzt(solution_eo)
    full_rhs = tools.poooxyzt2oooxyzt(rhs_eo)
    residual = dslash.give_wilson(full_solution, full_gauge, kappa, True)
    residual = residual + dslash.give_clover(full_solution, clover) - full_rhs
    denominator = max(float(tools.norm(full_rhs)), 1.0e-30)
    value = float(tools.norm(residual)) / denominator
    if not math.isfinite(value):
        raise BenchmarkFailure("nonfinite_true_residual", repr(value))
    return value


def _gather_local_blocks(
        value: Any, process_grid: Sequence[int], root: int = 0) -> Any:
    """Gather a rank-local ``...xyzt`` tensor into the global lattice."""
    import numpy as np
    import torch
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    array = np.ascontiguousarray(value.detach().cpu().numpy())
    grid = tuple(int(item) for item in process_grid)
    local = tuple(int(item) for item in array.shape[-4:])
    global_shape = (
        *array.shape[:-4],
        *(local[axis] * grid[axis] for axis in range(4)),
    )
    blocks = comm.gather(array, root=root)
    # MPI_Gather may return on the root once receives complete while a peer is
    # still flushing its send.  Keep the communicator in lockstep before the
    # root starts GPU work; otherwise the root can reach MPI_Finalize first on
    # WSL/OpenMPI and strand the sender in the collective.
    comm.Barrier()
    if rank != root:
        return None
    whole = np.empty(global_shape, dtype=array.dtype)
    for block_rank, block in enumerate(blocks):
        coordinate = _rank_coordinate(block_rank, grid)
        selection = [slice(None)] * (array.ndim - 4)
        selection.extend(
            slice(coordinate[axis] * local[axis],
                  (coordinate[axis] + 1) * local[axis])
            for axis in range(4))
        whole[tuple(selection)] = block
    return torch.from_numpy(whole).to(device=value.device)


def _canonical_true_residual_mpi(
        solution_eo: Any, rhs_eo: Any, gauge_eo: Any, mass: float,
        process_grid: Sequence[int]) -> float:
    """Global true residual for rank-local parity fields."""
    import torch
    from mpi4py import MPI
    from pyqcu import dslash, tools

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    device = solution_eo.device
    local_solution = tools.poooxyzt2oooxyzt(solution_eo)
    local_rhs = tools.poooxyzt2oooxyzt(rhs_eo)
    local_gauge = tools.poooxyzt2oooxyzt(gauge_eo)
    global_solution = _gather_local_blocks(
        local_solution, process_grid)
    global_rhs = _gather_local_blocks(local_rhs, process_grid)
    global_gauge = _gather_local_blocks(local_gauge, process_grid)
    if rank == 0:
        kappa = 1.0 / (2.0 * float(mass) + 8.0)
        real_dtype = (
            torch.float32 if gauge_eo.dtype == torch.complex64 else
            torch.float64)
        clover = dslash.make_clover(
            global_gauge,
            kappa=torch.tensor([kappa], dtype=real_dtype, device=device),
            u_0=torch.ones([1], dtype=real_dtype, device=device))
        residual = dslash.give_wilson(
            global_solution, global_gauge, kappa, True)
        residual = (
            residual + dslash.give_clover(global_solution, clover) -
            global_rhs)
        numerator_local = residual
        denominator_local = global_rhs
    else:
        numerator_local = torch.zeros(1, dtype=solution_eo.dtype, device=device)
        denominator_local = torch.zeros(
            1, dtype=solution_eo.dtype, device=device)
    numerator = float(tools.norm(numerator_local))
    denominator = max(float(tools.norm(denominator_local)), 1.0e-30)
    value = numerator / denominator
    if not math.isfinite(value):
        raise BenchmarkFailure("nonfinite_true_residual", repr(value))
    return value


def _unpack_odd_null_vectors(
        blocked: Any, *, lattice: Sequence[int] = LATTICE,
        block: Sequence[int] = BLOCK, nvec: int = NVECS) -> Any:
    """把 legacy blocked cache 还原为 ``nvec`` 个 odd-Schur vectors。"""
    lattice4 = tuple(int(value) for value in lattice)
    block4 = tuple(int(value) for value in block)
    odd_shape = (*lattice4[:3], lattice4[3] // 2)
    if any(extent % width for extent, width in zip(odd_shape, block4)):
        raise BenchmarkFailure(
            "nullvec_geometry_mismatch",
            f"odd_shape={odd_shape} is not divisible by block={block4}")
    coarse = tuple(extent // width for extent, width in zip(odd_shape, block4))
    expected = (
        int(nvec), 12,
        coarse[0], block4[0], coarse[1], block4[1],
        coarse[2], block4[2], coarse[3], block4[3])
    if tuple(int(x) for x in blocked.shape) != expected:
        raise BenchmarkFailure(
            "nullvec_shape_mismatch", f"blocked shape={tuple(blocked.shape)} expected={expected}")
    return blocked.reshape(int(nvec), 4, 3, *odd_shape)


def _reconstruct_full_null_vectors(
        blocked: Any, rhs: Any, gauge: Any, clover_ee: Any,
        clover_oo: Any, clover_ee_inv: Any, clover_oo_inv: Any,
        params: Any, argv: Any) -> Any:
    """由 odd-Schur 向量和 ``b=0`` 块消元重构 full near-null vectors。"""
    import torch
    from pyqcu import tools
    from pyqcu.cuda import define, qcu

    odd = _unpack_odd_null_vectors(blocked)
    eo = torch.empty(
        (NVECS, 2, 4, 3, *LATTICE[:3], LATTICE[3] // 2),
        dtype=blocked.dtype, device=blocked.device)
    zero_rhs = torch.zeros_like(rhs)
    reconstruct_params = params.clone().contiguous()
    reconstruct_params[define._SET_PLAN_] = 1
    reconstruct_ptrs = define.set_ptrs.clone()
    first_set_index = int(reconstruct_params[define._SET_INDEX_])
    for index in range(NVECS):
        reconstruct_params[define._SET_INDEX_] = first_set_index + index
        qcu.applyInitQcu(reconstruct_ptrs, reconstruct_params, argv)
        try:
            qcu.applyCloverBistabCgReconstructQcu(
                eo[index], zero_rhs, odd[index], gauge, clover_ee,
                clover_oo, clover_ee_inv, clover_oo_inv,
                reconstruct_ptrs, reconstruct_params)
        finally:
            qcu.applyEndQcu(reconstruct_ptrs, reconstruct_params)
    torch.cuda.synchronize(blocked.device)
    full = tools.poooxyzt2oooxyzt(eo.movedim(1, 0)).contiguous()
    del eo, zero_rhs
    return full


def _configured_strict_runtime_assets(
        *, argv: Any, params: Any, gauge: Any,
        clover_ee: Any, clover_oo: Any, clover_ee_inv: Any,
        clover_oo_inv: Any, fine_null: Any,
        assets: Sequence[Mapping[str, Any]],
        level_specs: Sequence[Mapping[str, Any]],
        coarse_max_iter: int, coarse_tol: float, restart: int,
        target_parity: int = TARGET_PARITY,
        smoother_steps: int = DEFAULT_NU_PRE) -> Dict[str, Any]:
    """Create the fused runtime from validated resident transition assets."""
    import torch
    from pyqcu.cuda import define, qcu
    from pyqcu.cuda._schur_op import CudaSchurOp
    from pyqcu.solver._quda_multigrid import QcuStrictAssetBinding

    if len(assets) != len(level_specs) or not assets:
        raise ValueError("strict runtime assets/level_specs 必须是等长非空序列")

    configured = params.clone().contiguous()
    controls = argv.clone().contiguous()
    configured[define._PARITY_] = int(target_parity)
    configured[define._MG_NUM_LEVEL_] = len(level_specs) + 1
    configured[define._MG_MU_PRE_] = int(smoother_steps)
    for level, spec in enumerate(level_specs, start=1):
        shape = tuple(int(x) for x in spec["shape"])
        if len(shape) != 4:
            raise ValueError(f"strict level {level} shape 必须有四维")
        base = define._MG_LEVEL1_E_ + (level - 1) * define._MG_PARAMS_SIZE_
        configured[base:base + define._MG_PARAMS_SIZE_] = torch.tensor(
            [int(spec["dof"]), *shape, int(coarse_max_iter),
             int(configured[define._DATA_TYPE_]), int(restart)],
            dtype=configured.dtype)
        controls[define._MG_LEVEL1_ATOL_ + level - 1] = float(coarse_tol)

    schur = None
    binding = None
    initialized = False
    try:
        schur = CudaSchurOp(
            controls, gauge, clover_ee, clover_oo, clover_ee_inv,
            clover_oo_inv, device=gauge.device, params=configured)
        binding = QcuStrictAssetBinding(
            schur.set_ptrs, assets, start_level=1, retain_raw_links=False)
        coarse_workspace = int(qcu.applyMultigridStrictInitQcu(
            schur.set_ptrs, schur.params, 1))
        initialized = True
        return {
            "schur": schur,
            "binding": binding,
            "fine_null_vectors": fine_null,
            "coarse_workspace_bytes": coarse_workspace,
            "initialized": initialized,
        }
    except Exception:
        if initialized and schur is not None:
            qcu.applyMultigridStrictEndQcu(schur.set_ptrs, schur.params)
        if binding is not None:
            binding.close()
        if schur is not None:
            schur.release()
        raise


def _configured_strict_runtime(
        hierarchy: Any, argv: Any, params: Any, gauge: Any,
        clover_ee: Any, clover_oo: Any, clover_ee_inv: Any,
        clover_oo_inv: Any, *,
        release_setup_before_runtime: bool = False) -> Dict[str, Any]:
    """创建 fused strict 所需最小 runtime，避免额外 Python Krylov arena。"""
    import torch

    fine_null_source = hierarchy.transfers[0].to_qcu_blocked()
    assets = hierarchy.qcu_strict_transition_assets(
        dtype=None if release_setup_before_runtime else gauge.dtype,
        device=None if release_setup_before_runtime else gauge.device,
        include_raw_links=False, runtime_start_level=1)
    level_specs = [
        {"dof": int(operator.dof), "shape": list(operator.shape)}
        for operator in hierarchy.operators[1:]
    ]
    setup_release = None
    if release_setup_before_runtime:
        # The exported assets and fine_null keep their storages alive.  Drop
        # the duplicate hierarchy references before C++ applyInitQcu allocates
        # its LatticeSet scratch; this is the cold c128 capacity path.
        setup_release = hierarchy.release_python_setup_assets()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(gauge.device)
    for asset in assets:
        null_vectors = asset.get("null_vectors")
        if null_vectors is not None:
            asset["null_vectors"] = null_vectors.to(
                dtype=gauge.dtype, device=gauge.device).contiguous()
    fine_null = fine_null_source.to(
        dtype=gauge.dtype, device=gauge.device).contiguous()
    del fine_null_source
    runtime = _configured_strict_runtime_assets(
        argv=argv, params=params, gauge=gauge,
        clover_ee=clover_ee, clover_oo=clover_oo,
        clover_ee_inv=clover_ee_inv, clover_oo_inv=clover_oo_inv,
        fine_null=fine_null, assets=assets, level_specs=level_specs,
        coarse_max_iter=hierarchy.coarse_max_iter,
        coarse_tol=hierarchy.coarse_tol, restart=hierarchy.restart,
        target_parity=hierarchy.target_parity,
        smoother_steps=int(getattr(
            hierarchy, "nu_pre", DEFAULT_NU_PRE)))
    runtime["transition_assets"] = assets
    runtime["level_specs"] = level_specs
    if setup_release is not None:
        runtime["python_setup_release"] = setup_release
    return runtime


def _close_strict_runtime(runtime: Optional[Mapping[str, Any]]) -> None:
    if not runtime:
        return
    from pyqcu.cuda import qcu
    first: Optional[BaseException] = None
    schur = runtime.get("schur")
    if runtime.get("initialized") and schur is not None:
        try:
            qcu.applyMultigridStrictEndQcu(schur.set_ptrs, schur.params)
        except BaseException as exc:  # cleanup must continue
            first = exc
    binding = runtime.get("binding")
    if binding is not None:
        try:
            binding.close()
        except BaseException as exc:
            first = first or exc
    if schur is not None:
        try:
            schur.release()
        except BaseException as exc:
            first = first or exc
    if first is not None:
        raise first


def _seal_pyqcu_hierarchy_runtime(
        hierarchy: Any, torch: Any, device: Any) -> Tuple[Any, Dict[str, Any]]:
    """保存 setup 诊断后释放已由 C++ runtime 接管的 Python 资产。"""
    setup_stats = copy.deepcopy(
        getattr(hierarchy, "strict_setup_stats", None))
    allocated_before = int(torch.cuda.memory_allocated(device))
    release_report = dict(hierarchy.seal_cuda_runtime(
        runtime_assets_bound=True))
    torch.cuda.synchronize(device)
    allocated_after = int(torch.cuda.memory_allocated(device))
    release_report["allocator_released_bytes"] = max(
        0, allocated_before - allocated_after)
    return setup_stats, release_report


def _validate_strict_setup_contract(
        stats: Any, config: Mapping[str, Any]) -> None:
    """Fail closed when a cache/build silently changes formal setup batching."""
    setup = config.get("pyqcu_strict_setup")
    if not isinstance(setup, Mapping):
        raise BenchmarkFailure(
            "strict_setup_contract_missing", "protocol.pyqcu_strict_setup")
    if not isinstance(stats, list) or len(stats) != int(config["levels"]) - 1:
        raise BenchmarkFailure(
            "strict_setup_stats_invalid",
            f"expected {int(config['levels']) - 1} transition stats, got {stats!r}")
    requested_columns = int(setup["column_batch_size"])
    requested_projection = int(setup["projection_site_batch_size"])
    workspace_cap = int(setup["max_workspace_bytes"])
    for level, entry in enumerate(stats):
        if not isinstance(entry, Mapping):
            raise BenchmarkFailure(
                "strict_setup_stats_invalid", f"level {level}: {entry!r}")
        mode = entry.get("effective_probe_mode", entry.get("probe_mode"))
        columns = entry.get("column_batch_size")
        projection = entry.get("projection_site_batch_size")
        memory = entry.get("memory")
        workspace = (
            memory.get("workspace_upper_bytes")
            if isinstance(memory, Mapping) else None)
        mismatches: List[str] = []
        if mode != setup["probe_mode"]:
            mismatches.append(f"mode={mode!r}")
        if columns != requested_columns:
            mismatches.append(
                f"column_batch_size={columns!r}, requested={requested_columns}")
        if projection != requested_projection:
            mismatches.append(
                "projection_site_batch_size="
                f"{projection!r}, requested={requested_projection}")
        if not isinstance(workspace, int) or workspace > workspace_cap:
            mismatches.append(
                f"workspace_upper_bytes={workspace!r}, cap={workspace_cap}")
        if mismatches:
            raise BenchmarkFailure(
                "strict_setup_contract_mismatch",
                f"level {level}: " + "; ".join(mismatches))


def _validate_cached_setup_stats(
        stats: Any, config: Mapping[str, Any]) -> None:
    """Validate reusable cache geometry without replaying build batching."""
    if not isinstance(stats, list) or len(stats) != int(config["levels"]) - 1:
        raise BenchmarkFailure(
            "strict_cache_setup_stats_invalid",
            f"expected {int(config['levels']) - 1} cached transition stats, "
            f"got {stats!r}")
    shape = tuple(int(value) for value in config["lattice_xyzt"])
    blocks = [
        tuple(int(value) for value in block)
        for block in config["block_xyzt_per_level"]
    ]
    for level, entry in enumerate(stats):
        if not isinstance(entry, Mapping):
            raise BenchmarkFailure(
                "strict_cache_setup_stats_invalid",
                f"level {level}: expected mapping, got {entry!r}")
        shape = tuple(
            extent // width for extent, width in zip(shape, blocks[level]))
        expected_sites = math.prod(shape)
        values = {
            "coarse_sites": expected_sites,
            "coarse_dof": int(config["coarse_dof"]),
        }
        mismatches = [
            f"{key}={entry.get(key)!r}, expected={expected!r}"
            for key, expected in values.items()
            if entry.get(key) != expected
        ]
        for key in ("column_batch_size", "projection_site_batch_size"):
            value = entry.get(key)
            if (isinstance(value, bool) or not isinstance(value, int) or
                    value <= 0):
                mismatches.append(f"{key}={value!r} must be a positive integer")
        memory = entry.get("memory")
        workspace = (
            memory.get("workspace_upper_bytes")
            if isinstance(memory, Mapping) else None)
        if (isinstance(workspace, bool) or not isinstance(workspace, int) or
                workspace <= 0):
            mismatches.append(
                f"workspace_upper_bytes={workspace!r} must be a positive integer")
        if mismatches:
            raise BenchmarkFailure(
                "strict_cache_setup_stats_invalid",
                f"level {level}: " + "; ".join(mismatches))


def _run_pyqcu_worker(payload: Mapping[str, Any]) -> Dict[str, Any]:
    global LATTICE
    config = payload["protocol"]
    inputs = payload["inputs"]
    phase_request = _phase_request(payload)
    mpi_context = payload.get("mpi_context") or _worker_mpi_context(payload)
    if int(config["levels"]) == 1:
        raise BenchmarkUnsupported(
            "pyqcu_mg_level_1_unsupported",
            "QUDA n_level=1 uses a fine-level coarsest MG branch; the "
            "current PyQCU strict fused C++ hierarchy requires >=2 levels. "
            "The collector fails closed instead of comparing different solvers.")
    if int(mpi_context["size"]) > 1:
        try:
            from pyqcu.cuda._strict_mpi import strict_mpi_capabilities
            capabilities = strict_mpi_capabilities()
        except Exception as exc:
            raise BenchmarkUnsupported(
                "pyqcu_distributed_strict_preflight_unavailable",
                repr(exc)) from exc
        if not all(capabilities.get(name, False) for name in (
                "setup_halo", "fine_halo", "full_halo", "compact_halo",
                "rp_coarse_halo", "global_reduction", "fused_fgmres")):
            raise BenchmarkUnsupported(
                "pyqcu_distributed_strict_unsupported",
                f"strict MPI capabilities={capabilities}; the collector "
                "will not fall back to serial or fake distributed results.")
    _activate_runtime_globals(config, inputs)
    cache_execution = payload["execution"]["strict_cache"]
    cache_expect = str(phase_request["cache_expect"])
    cache_identity = _strict_runtime_cache_identity(payload)
    cache_path = _strict_runtime_cache_path(
        cache_identity, Path(cache_execution["directory"]))
    if cache_expect == "miss" and cache_path.exists():
        raise BenchmarkFailure(
            "strict_runtime_cache_expectation_failed",
            f"expected miss but target exists: {cache_path}")
    if cache_expect == "hit" and not cache_path.is_file():
        raise BenchmarkFailure(
            "strict_runtime_cache_expectation_failed",
            f"expected hit but target is absent: {cache_path}")
    runtime_started = time.perf_counter()
    try:
        import numpy as np
        import torch
        from pyqcu import dslash, tools
        from pyqcu.cuda import define, qcu
        from pyqcu.cuda._strict_cache import (
            StrictRuntimeCacheConflictError,
            load_strict_runtime_cache,
            save_strict_runtime_cache,
        )
        from pyqcu.solver import QudaStrictMultigrid
        sys.path.insert(0, str(HERE))
        from common import (
            _slice_last4, global_parity_to_local, local_geometry,
            make_clover_tensors)
    except (ImportError, OSError) as exc:
        raise BenchmarkSkip("missing_pyqcu_dependency", repr(exc)) from exc

    device = _select_accelerator(
        torch,
        str((payload["execution"].get("device") or {}).get("requested", "v100")),
        int(mpi_context.get("local_rank", mpi_context["rank"])),
        int(mpi_context.get("local_size", mpi_context["size"])))
    if int(mpi_context["size"]) > 1:
        os.environ["PYQCU_MPI_DEVICE_ID"] = str(int(device.index))
    device_uuid = _torch_runtime_provenance(torch, device)["device_uuid"]
    torch.cuda.synchronize(device)
    runtime_init_s = time.perf_counter() - runtime_started
    precision = config["precision"]["name"]
    complex_dtype = torch.complex64 if precision == "c64" else torch.complex128
    data_code = define._LAT_C64_ if precision == "c64" else define._LAT_C128_

    io_started = time.perf_counter()
    global_lattice = tuple(int(value) for value in LATTICE)
    fingerprints = payload["input_fingerprints"]
    distributed_read = int(mpi_context["size"]) > 1
    gauge_np = _load_h5_local_array(
        inputs["gauge"]["path"], inputs["gauge"]["dataset"],
        global_lattice, mpi_context, parity_compressed=True)
    source_np = _load_h5_local_array(
        inputs["source"]["path"], inputs["source"]["dataset"],
        global_lattice, mpi_context, parity_compressed=True)
    null_np = _load_h5_local_array(
        inputs["null_vectors"]["path"], inputs["null_vectors"]["dataset"],
        global_lattice, mpi_context, parity_compressed=False)
    local_lattice, local_starts, _ = local_geometry(
        global_lattice, grid=mpi_context["process_grid"],
        rank=mpi_context["rank"], require_even=True)
    if distributed_read:
        _verify_local_input_slab(
            "gauge", gauge_np, fingerprints["gauge"], local_lattice,
            parity_compressed=True)
        _verify_local_input_slab(
            "source", source_np, fingerprints["source"], local_lattice,
            parity_compressed=True)
        _verify_local_input_slab(
            "null_vectors", null_np, fingerprints["null_vectors"],
            local_lattice, parity_compressed=False)
    else:
        _verify_loaded_input("gauge", gauge_np, fingerprints["gauge"])
        _verify_loaded_input("source", source_np, fingerprints["source"])
        _verify_loaded_input(
            "null_vectors", null_np, fingerprints["null_vectors"])
    if int(mpi_context["size"]) > 1:
        LATTICE = tuple(local_lattice)
    else:
        local_starts = [0, 0, 0, 0]
    input_io_s = time.perf_counter() - io_started

    runtime: Optional[Dict[str, Any]] = None
    hierarchy = None
    setup_memory_before = None
    setup_sampler: Optional[_CudaDeviceMemorySampler] = None
    cache_report: Dict[str, Any] = {}
    try:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        setup_memory_before = {
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "nvidia_smi": _nvidia_smi_used(device_uuid=device_uuid),
        }
        setup_sampler = _CudaDeviceMemorySampler(torch, device).start()
        setup_started = time.perf_counter()
        if int(mpi_context["size"]) > 1:
            gauge = torch.from_numpy(gauge_np).to(
                device=device, dtype=complex_dtype).contiguous()
            rhs = torch.from_numpy(source_np).to(
                device=device, dtype=complex_dtype).contiguous()
            local_null_np = null_np
        else:
            gauge = torch.from_numpy(gauge_np).to(
                device=device, dtype=complex_dtype).contiguous()
            rhs = torch.from_numpy(source_np).to(
                device=device, dtype=complex_dtype).contiguous()
            local_null_np = null_np
        ce, cei, coo, coi, _unused_s, params, argv = make_clover_tensors(
            gauge, LATTICE, MASS, grid=mpi_context["process_grid"],
            rank=mpi_context["rank"], dtype=complex_dtype,
            data_type=data_code)
        params[define._MAX_ITER_] = int(config["max_iter"])
        argv[define._ATOL_] = float(config["tolerance"])
        full_gauge = tools.poooxyzt2oooxyzt(gauge).contiguous()
        real_dtype = torch.float32 if complex_dtype == torch.complex64 else torch.float64
        kappa_tensor = torch.tensor(
            [config["kappa"]], dtype=real_dtype, device=device)
        clover_full = dslash.make_clover(
            full_gauge,
            kappa=kappa_tensor,
            u_0=torch.ones([1], dtype=real_dtype, device=device))

        expected_manifest = _strict_runtime_expected_manifest(
            config, mpi_context)
        # Keep cache restore timing disjoint from preceding Gauge/Clover CUDA
        # work.  The wait belongs to setup, not to cache I/O.
        torch.cuda.synchronize(device)
        cache_load_started = time.perf_counter()
        cache_result = load_strict_runtime_cache(
            cache_path, identity=cache_identity, device=device,
            expected_manifest=expected_manifest)
        torch.cuda.synchronize(device)
        cache_load_s = time.perf_counter() - cache_load_started
        cache_write_s = 0.0
        cache_report = {
            "path": str(cache_path.resolve()),
            "identity_sha256": _sha256_json(cache_identity),
            "hit": bool(cache_result.hit),
            "load_reason": cache_result.reason,
            "load_detail": cache_result.detail,
            "load_seconds": cache_load_s,
            "write": None,
            "expectation": cache_expect,
            "evidence": copy.deepcopy(cache_result.evidence),
        }
        level_specs = []
        level_shape = list(LATTICE)
        level_blocks = [
            tuple(int(value) for value in block)
            for block in config["block_xyzt_per_level"]
        ]
        for block in level_blocks:
            level_shape = [
                extent // width
                for extent, width in zip(level_shape, block)
            ]
            level_specs.append({
                "dof": int(config["coarse_dof"]),
                "shape": list(level_shape),
            })

        if cache_result.hit:
            if cache_expect == "miss":
                raise BenchmarkFailure(
                    "strict_runtime_cache_expectation_failed",
                    f"expected miss but cache verified: {cache_path}")
            assert cache_result.assets is not None
            cached_assets = cache_result.assets
            strict_setup_stats = cache_result.stats
            _validate_cached_setup_stats(strict_setup_stats, config)
            cache_report["build_contract"] = {
                "validation": "cache assets must match requested geometry; "
                              "construction batching is provenance only",
                "requested_setup": copy.deepcopy(config["pyqcu_strict_setup"]),
                "cached_setup_stats": copy.deepcopy(strict_setup_stats),
            }
            runtime = _configured_strict_runtime_assets(
                argv=argv, params=params, gauge=gauge,
                clover_ee=ce, clover_oo=coo,
                clover_ee_inv=cei, clover_oo_inv=coi,
                fine_null=cached_assets.fine_blocked_v,
                assets=cached_assets.to_runtime_levels(),
                level_specs=level_specs,
                coarse_max_iter=int(config["coarse_max_iter"]),
                coarse_tol=_strict_coarse_tolerance(config),
                restart=int(config["restart_requested"]),
                target_parity=int(config["target_parity"]),
                smoother_steps=int(config["nu_pre"]))
            setup_release = {
                "sealed": True,
                "source": "strict_runtime_cache",
                "allocator_released_bytes": 0,
                "loaded_logical_bytes": int(cache_result.manifest["total_bytes"]),
            }
            del cached_assets, cache_result
            del gauge_np, source_np, null_np
        else:
            if cache_result.reason != "not_found":
                raise BenchmarkFailure(
                    "strict_runtime_cache_invalid",
                    f"{cache_path}: {cache_result.reason}: {cache_result.detail}")
            if cache_expect == "hit":
                raise BenchmarkFailure(
                    "strict_runtime_cache_expectation_failed",
                    f"expected hit but cache missed: {cache_path}")
            del cache_result
            expected_null_shape = (NVECS, 4, 3, *LATTICE)
            if tuple(int(x) for x in local_null_np.shape) != expected_null_shape:
                raise BenchmarkFailure(
                    "canonical_nullvec_shape_mismatch",
                    f"shape={tuple(local_null_np.shape)}, "
                    f"expected={expected_null_shape}")
            null_vectors = torch.from_numpy(local_null_np).to(
                device=device, dtype=complex_dtype).contiguous()
            # hierarchy 持有 device null vectors；CPU HDF5 arrays 不跨入 setup。
            del gauge_np, source_np, null_np
            from mpi4py import MPI
            hierarchy = QudaStrictMultigrid(
                U=full_gauge,
                clover_term=clover_full,
                kappa=kappa_tensor,
                u_0=torch.ones(
                    [1], dtype=real_dtype, device=device),
                null_vectors=[null_vectors],
                dof_list=[12] + [COARSE_DOF] * max(0, LEVELS - 1),
                block_size=[
                    list(value)
                    for value in config["block_xyzt_per_level"]],
                max_level=LEVELS,
                propagate_null_vectors=bool(LEVELS > 2),
                n_block_ortho=2,
                materialize_coarse=True,
                use_parity=True,
                target_parity=int(config["target_parity"]),
                nu_pre=int(config["nu_pre"]),
                nu_post=int(config["nu_post"]),
                coarse_max_iter=int(config["coarse_max_iter"]),
                coarse_tol=_strict_coarse_tolerance(config),
                restart=int(config["restart_requested"]),
                max_iter=int(config["max_iter"]),
                tol=float(config["tolerance"]),
                setup_method="random",
                setup_iters=0,
                strict_galerkin_mode=str(
                    config["pyqcu_strict_setup"]["probe_mode"]),
                strict_galerkin_column_batch=int(
                    config["pyqcu_strict_setup"]["column_batch_size"]),
                strict_galerkin_projection_batch=int(
                    config["pyqcu_strict_setup"]["projection_site_batch_size"]),
                strict_galerkin_block_dtype=(
                    torch.float32
                    if (config["pyqcu_strict_block_precision"]["effective"] ==
                        "single" and complex_dtype == torch.complex128)
                    else None),
                strict_galerkin_block_device=(
                    torch.device("cpu")
                    if config["pyqcu_strict_block_device"] == "cpu"
                    else None),
                strict_offload_null_basis=bool(
                    config["pyqcu_strict_offload_null_basis"]),
                strict_galerkin_max_workspace_bytes=int(
                    config["pyqcu_strict_setup"]["max_workspace_bytes"]),
                process_grid=mpi_context["process_grid"],
                comm=MPI.COMM_WORLD,
                verbose=False,
            )
            # hierarchy 已持有输入引用。删除局部别名后，runtime seal 才能真正
            # 回收 transfer.B/V，而不是只从 hierarchy 对象上解除引用。
            del null_vectors
            hierarchy.setup()
            _validate_strict_setup_contract(
                hierarchy.strict_setup_stats, config)
            runtime = _configured_strict_runtime(
                hierarchy, argv, params, gauge, ce, coo, cei, coi,
                release_setup_before_runtime=True)
            strict_setup_stats = copy.deepcopy(hierarchy.strict_setup_stats)
            transition_assets = runtime.pop("transition_assets")
            runtime.pop("level_specs")
            cache_levels = []
            for level, asset in enumerate(transition_assets):
                cache_levels.append({
                    "level": level,
                    "preconditioned_links": asset["preconditioned_links"],
                    "onsite_pair": asset["onsite_pair"],
                    "null_vectors": (
                        None if level == 0 else asset["null_vectors"]),
                })
            # All hierarchy/Galerkin kernels must finish before persistence is
            # timed; otherwise subtracting cache_write_s would under-report
            # cold setup and misattribute GPU wait time to HDF5.
            torch.cuda.synchronize(device)
            cache_write_started = time.perf_counter()
            try:
                write_result = save_strict_runtime_cache(
                    cache_path,
                    identity=cache_identity,
                    fine_blocked_v=runtime["fine_null_vectors"],
                    levels=cache_levels,
                    metadata={
                        "level_specs": level_specs,
                        "input_bundle_hash": fingerprints["bundle_hash"],
                        "coarsening_operator": "R(X^-1 D)P",
                    },
                    stats=strict_setup_stats)
            except StrictRuntimeCacheConflictError as exc:
                raise BenchmarkFailure(
                    "strict_runtime_cache_conflict", str(exc)) from exc
            torch.cuda.synchronize(device)
            cache_write_s = time.perf_counter() - cache_write_started
            cache_report["write"] = {
                "written": bool(write_result.written),
                "reason": write_result.reason,
                "detail": write_result.detail,
                "logical_bytes": int(write_result.logical_bytes),
                "seconds": cache_write_s,
            }
            del cache_levels, transition_assets
            strict_setup_stats, setup_release = _seal_pyqcu_hierarchy_runtime(
                hierarchy, torch, device)
        torch.cuda.synchronize(device)
        setup_finished = time.perf_counter()
        setup_device_memory = setup_sampler.stop()
        setup_sampler = None
        # Cache persistence is reported separately and never inflates setup.
        # Sampler join/telemetry cleanup is outside the setup timer.
        setup_s = setup_finished - setup_started - cache_write_s
        setup_noncache_s = max(0.0, setup_s - cache_load_s)
        setup_memory = {
            "baseline": setup_memory_before,
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "nvidia_smi": _nvidia_smi_used(device_uuid=device_uuid),
            "device_wide_sampler": setup_device_memory,
            "timing_semantics": (
                "instrumented setup; sampler stop excluded from setup_seconds"),
        }

        # The Python setup hierarchy has been sealed above; return its unused
        # allocator blocks before allocating the steady-solve workspace.  This
        # is outside both setup and solve timing and is especially important
        # for the 32 GiB c128 hierarchy.
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)

        # Keep these protocol facts next to the completed setup.  They used to
        # be accidentally inserted into _strict_runtime_cache_identity(),
        # where the worker-local torch/device variables do not exist.
        pyqcu_timing_boundary = {
            "requested": "caller_preallocated",
            "supported": True,
            "formal_eligible": True,
            "zero_initial_guess_before_timer": True,
            "preserve_source": "not_applicable; PyQCU rhs is not mutated",
            "timed_operation":
                "pyqcu.cuda.qcu.applyMultigridStrictFgmresQcu(output,rhs,...)",
            "performance_call": "not applicable",
        }
        provenance = _benchmark_provenance()
        provenance.update({
            "runtime": _torch_runtime_provenance(torch, device),
            "strict_setup_stats": copy.deepcopy(strict_setup_stats),
        })

        schur = runtime["schur"]
        fine_null = runtime["fine_null_vectors"]
        output = torch.empty_like(rhs)
        requested_restart = int(config["restart_requested"])
        fine_elements = rhs.numel() // 2
        coarse_elements = int(fine_null.shape[0])
        for axis in (2, 4, 6, 8):
            coarse_elements *= int(fine_null.shape[axis])
        effective_restart = int(config["restart_effective"])
        required_workspace = int(
            ((2 * effective_restart + 5) * fine_elements + 2 * coarse_elements)
            * rhs.element_size())
        if required_workspace > int(config["max_krylov_bytes"]):
            raise BenchmarkFailure(
                "krylov_contract_drift",
                f"workspace={required_workspace}, budget={config['max_krylov_bytes']}")

        def solve_once(
                *, sample_device_memory: bool = False,
        ) -> Tuple[float, Dict[str, Any], float, Dict[str, Any]]:
            output.zero_()
            schur.params[define._MG_USE_INIT_GUESS_] = 0
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            sampler = (
                _CudaDeviceMemorySampler(torch, device).start()
                if sample_device_memory else None)
            started = time.perf_counter()
            try:
                result = qcu.applyMultigridStrictFgmresQcu(
                    output, rhs, gauge, ce, coo, cei, coi, fine_null,
                    schur.set_ptrs, schur.params, effective_restart,
                    int(config["max_iter"]), float(config["tolerance"]),
                    int(config["nu_pre"]), int(config["nu_post"]),
                    int(config["max_krylov_bytes"]))
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
            finally:
                device_memory = None if sampler is None else sampler.stop()
            memory_sample = {
                "cuda_peak_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(device)),
                "cuda_peak_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(device)),
                "nvidia_smi": _nvidia_smi_used(device_uuid=device_uuid),
                "device_wide_sampler": device_memory,
            }
            residual = (
                _canonical_true_residual_mpi(
                    output, rhs, gauge, MASS,
                    mpi_context["process_grid"])
                if int(mpi_context["size"]) > 1 else
                _canonical_true_residual(
                    output, rhs, gauge, MASS,
                    full_gauge=full_gauge, clover=clover_full))
            return elapsed, dict(result), residual, memory_sample

        first_solve_memory = None
        cold_seconds = None
        cold_result_payload = None
        if phase_request["cold"]:
            elapsed, result, residual, memory_sample = solve_once(
                sample_device_memory=True)
            cold_seconds = elapsed
            cold_result_payload = {
                "seconds": elapsed,
                "iterations": int(result["iterations"]),
                "converged": bool(result["converged"]),
                "true_residual_rel": residual,
                "backend_final_true_residual": float(
                    result["final_true_residual"]),
                "fused_workspace_bytes": int(result["allocated_bytes"]),
                "memory": memory_sample,
            }
            first_solve_memory = {
                "excluded_from_formal_timing": True,
                **copy.deepcopy(cold_result_payload),
            }

        warmup_results = []
        if phase_request["warmup"]:
            for warmup_index in range(WARMUPS):
                elapsed, result, residual, memory_sample = solve_once(
                    sample_device_memory=(
                        warmup_index == 0 and cold_seconds is None))
                if warmup_index == 0 and first_solve_memory is None:
                    first_solve_memory = {
                        "excluded_from_formal_timing": True,
                        "seconds": elapsed,
                        "iterations": int(result["iterations"]),
                        "converged": bool(result["converged"]),
                        "true_residual_rel": residual,
                        "backend_final_true_residual": float(
                            result["final_true_residual"]),
                        "fused_workspace_bytes": int(result["allocated_bytes"]),
                        "memory": memory_sample,
                    }
                warmup_results.append({
                    "seconds": elapsed,
                    "iterations": int(result["iterations"]),
                    "converged": bool(result["converged"]),
                    "true_residual_rel": residual,
                })

        steady_baseline = {
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "nvidia_smi": _nvidia_smi_used(device_uuid=device_uuid),
        }
        timings: List[float] = []
        iterations: List[int] = []
        converged: List[bool] = []
        residuals: List[float] = []
        backend_residuals: List[float] = []
        solve_memory_samples: List[Dict[str, Any]] = []
        allocated_bytes: List[int] = []
        if phase_request["steady"]:
            for _ in range(int(config["repeats"])):
                elapsed, result, residual, memory_sample = solve_once()
                timings.append(elapsed)
                iterations.append(int(result["iterations"]))
                converged.append(bool(result["converged"]))
                residuals.append(residual)
                backend_residuals.append(float(result["final_true_residual"]))
                allocated_bytes.append(int(result["allocated_bytes"]))
                solve_memory_samples.append(memory_sample)

        gate = float(config["true_residual_gate"])
        # A separate post-timing solve samples cudaMemGetInfo frequently enough
        # to observe native C++ transient allocations without perturbing the
        # formal 2+N timing samples.
        memory_probe: Dict[str, Any] = {}
        probe_residual: Optional[float] = None
        if phase_request["steady"]:
            _probe_elapsed, probe_result, probe_residual, memory_probe = solve_once(
                sample_device_memory=True)
            memory_probe.update({
                "excluded_from_formal_timing": True,
                "iterations": int(probe_result["iterations"]),
                "converged": bool(probe_result["converged"]),
                "true_residual_rel": probe_residual,
            })
            probe_pass = bool(probe_result["converged"]) and probe_residual <= gate
        else:
            probe_pass = True
        reference_report = None
        reference_pass = True
        reference_requested = any(
            phase_request[name] for name in ("cold", "warmup", "steady"))
        if (reference_requested and
                config["reference_solver"]["kind"] == "bicgstab"):
            from pyqcu.cuda._schur_op import CudaSchurOp
            reference_op = None
            try:
                reference_argv = argv.clone().contiguous()
                reference_argv[define._ATOL_] = float(config["tolerance"])
                reference_op = CudaSchurOp(
                    reference_argv, gauge, ce, coo, cei, coi,
                    device=device, params=params)
                reference_op.params[define._SET_PLAN_] = 1
                reference_op.params[define._MAX_ITER_] = int(config["max_iter"])
                reference_output = torch.empty_like(rhs)

                def reference_once() -> Tuple[float, float, int]:
                    reference_output.zero_()
                    torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    qcu.applyCloverBistabCgQcu(
                        reference_output, rhs, gauge, ce, coo, cei, coi,
                        reference_op.set_ptrs, reference_op.params)
                    iterations = int(qcu.getCloverBistabCgIterationsQcu(
                        reference_op.set_ptrs, reference_op.params))
                    torch.cuda.synchronize(device)
                    elapsed = time.perf_counter() - started
                    residual = (
                        _canonical_true_residual_mpi(
                            reference_output, rhs, gauge, MASS,
                            mpi_context["process_grid"])
                        if int(mpi_context["size"]) > 1 else
                        _canonical_true_residual(
                            reference_output, rhs, gauge, MASS,
                            full_gauge=full_gauge, clover=clover_full))
                    return elapsed, residual, iterations

                reference_cold = None
                if phase_request["cold"]:
                    seconds, residual, reference_iterations = reference_once()
                    reference_cold = {
                        "seconds": seconds,
                        "true_residual_rel": residual,
                        "iterations": reference_iterations,
                    }
                reference_warmups = []
                if phase_request["warmup"]:
                    for _ in range(WARMUPS):
                        seconds, residual, reference_iterations = reference_once()
                        reference_warmups.append({
                            "seconds": seconds,
                            "true_residual_rel": residual,
                            "iterations": reference_iterations,
                        })
                reference_samples = []
                if phase_request["steady"]:
                    for _ in range(int(config["repeats"])):
                        seconds, residual, reference_iterations = reference_once()
                        reference_samples.append({
                            "seconds": seconds,
                            "true_residual_rel": residual,
                            "iterations": reference_iterations,
                        })
                residual_samples = [
                    item["true_residual_rel"]
                    for item in ([reference_cold] if reference_cold else []) +
                    reference_warmups + reference_samples
                    if item is not None
                ]
                reference_report = {
                    "kind": "bicgstab",
                    "implementation": "pyqcu.cuda.qcu.applyCloverBistabCgQcu",
                    "cold": reference_cold,
                    "warmups": reference_warmups,
                    "steady": (
                        _median_mad([
                            item["seconds"] for item in reference_samples])
                        if reference_samples else
                        {"samples_seconds": [], "median_seconds": None,
                         "mad_seconds": None}),
                    "samples": reference_samples,
                    "true_residual_max_rel": (
                        max(residual_samples) if residual_samples else None),
                    "iterations": (
                        _median_mad([
                            item["iterations"] for item in reference_samples])
                        if reference_samples else None),
                    "excluded_from_speedup": True,
                }
                reference_pass = all(
                    value <= gate for value in residual_samples)
            finally:
                if reference_op is not None:
                    reference_op.release()
        phase_checks = [probe_pass, reference_pass]
        if cold_result_payload is not None:
            phase_checks.extend((
                bool(cold_result_payload["converged"]),
                float(cold_result_payload["true_residual_rel"]) <= gate))
        if warmup_results:
            phase_checks.extend(
                bool(item["converged"]) and
                float(item["true_residual_rel"]) <= gate
                for item in warmup_results)
        if phase_request["steady"]:
            phase_checks.extend((
                all(converged),
                all(value <= gate for value in residuals)))
        all_converged = all(phase_checks)
        measured_iterations = (
            list(iterations) if phase_request["steady"] else
            ([int(cold_result_payload["iterations"])]
             if cold_result_payload is not None else []) +
            [int(item["iterations"]) for item in warmup_results])
        measured_residuals = (
            list(residuals) if phase_request["steady"] else
            ([float(cold_result_payload["true_residual_rel"])]
             if cold_result_payload is not None else []) +
            [float(item["true_residual_rel"]) for item in warmup_results])
        timing_payload = {
            "input_io_seconds": input_io_s,
            "input_read_distributed": bool(distributed_read),
            "input_read_scope": (
                "rank-sliced HDF5 hyperslab"
                if distributed_read else "full HDF5 dataset"),
            "runtime_init_seconds": runtime_init_s,
            "setup_seconds": setup_s,
            "setup_mode": (
                "runtime_cache_restore" if cache_report["hit"] else
                "cold_hierarchy_build"),
            "setup_noncache_seconds": setup_noncache_s,
            "cache_restore_seconds": (
                cache_load_s if cache_report["hit"] else None),
            "cache_miss_probe_seconds": (
                None if cache_report["hit"] else cache_load_s),
            "cache_persist_seconds": cache_write_s,
            "warmups": warmup_results,
            "steady": (
                _median_mad(timings) if timings else
                {"samples_seconds": [], "median_seconds": None,
                 "mad_seconds": None}),
        }
        if cold_seconds is not None:
            timing_payload["cold_seconds"] = cold_seconds

        trace_path = _side_trace_path("pyqcu")
        by_phase: Dict[str, List[Dict[str, Any]]] = {}
        if phase_request["cold"]:
            levels = _mg_levels_from_trace(
                "pyqcu", trace_path, [0], int(config["levels"]))
            if levels is not None:
                by_phase["cold"] = levels
        offset = 1 if phase_request["cold"] else 0
        if phase_request["warmup"]:
            levels = _mg_levels_from_trace(
                "pyqcu", trace_path,
                list(range(offset, offset + WARMUPS)),
                int(config["levels"]))
            if levels is not None:
                by_phase["warmup"] = levels
            offset += WARMUPS
        if phase_request["steady"]:
            levels = _mg_levels_from_trace(
                "pyqcu", trace_path,
                list(range(offset, offset + int(config["repeats"]))),
                int(config["levels"]))
            if levels is not None:
                by_phase["steady"] = levels
        mg_levels = (
            by_phase.get("steady") or by_phase.get("warmup") or
            by_phase.get("cold"))
        if mg_levels is None:
            mg_levels = _fallback_mg_levels(
                config, timing_payload, measured_iterations,
                measured_residuals)
        binding_report = runtime["binding"].memory_report()
        fine_transfer_bytes = int(
            fine_null.numel() * fine_null.element_size())
        coarse_asset_bytes = int(binding_report["resident_bytes"])
        memory = {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "setup": setup_memory,
            "first_solve": first_solve_memory,
            "steady": {
                "baseline": steady_baseline,
                "cuda_peak_allocated_bytes": max(
                    (sample["cuda_peak_allocated_bytes"]
                     for sample in solve_memory_samples), default=0),
                "cuda_peak_reserved_bytes": max(
                    (sample["cuda_peak_reserved_bytes"]
                     for sample in solve_memory_samples), default=0),
                "nvidia_smi_process_max_observed_bytes": _max_optional(
                    sample["nvidia_smi"].get("process_used_bytes")
                    for sample in solve_memory_samples),
                "nvidia_smi_gpu_max_observed_bytes": _max_optional(
                    sample["nvidia_smi"].get("gpu_used_bytes")
                    for sample in solve_memory_samples),
                "samples": solve_memory_samples,
                "snapshot_semantics": (
                    "nvidia-smi values are post-solve observations, not peaks"),
                "untimed_device_memory_probe": memory_probe,
            },
            "strict_owned": {
                "coarse_workspace_bytes": int(runtime["coarse_workspace_bytes"]),
                "fine_transfer_bytes": fine_transfer_bytes,
                "coarse_asset_resident_bytes": coarse_asset_bytes,
                "asset_resident_bytes": fine_transfer_bytes + coarse_asset_bytes,
                "omitted_raw_bytes": int(binding_report["omitted_raw_bytes"]),
                "fused_workspace_bytes": max(allocated_bytes, default=0),
                "setup_release": setup_release,
            },
        }
        return {
            "side": "pyqcu",
            "status": "ok" if all_converged else "failed",
            "reason": None if all_converged else {
                "code": "convergence_or_true_residual_gate",
                "detail": f"converged={converged}, residuals={residuals}, gate={gate}",
            },
            "config_hash": config["config_hash"],
            "input_bundle_hash": payload["input_fingerprints"]["bundle_hash"],
            "timing": timing_payload,
            "reference_solver": reference_report,
            "iterations": _iteration_summary(measured_iterations),
            "mg_levels": mg_levels,
            "mg_levels_by_phase": by_phase,
            "iteration_semantics": (
                "finest total_iterations is the outer right-preconditioned "
                "FGMRES total; smoother and coarse iterations are separate"),
            "iteration_kind": "outer_right_preconditioned_fgmres",
            "iteration_scope": (
                "finest-level outer FGMRES columns; smoother and coarse "
                "solver iterations are recorded separately in MG trace"),
            "converged_samples": (
                converged if phase_request["steady"] else
                [bool(cold_result_payload["converged"])]
                if cold_result_payload is not None else
                [bool(item["converged"]) for item in warmup_results]),
            "converged": all_converged,
            "true_residual": {
                "samples_rel": measured_residuals,
                "max_rel": (
                    max(measured_residuals) if measured_residuals else math.inf),
                "gate": gate,
                "pass": all_converged,
                "untimed_probe_rel": probe_residual,
                "backend_schur_absolute_samples": backend_residuals,
            },
            "krylov": {
                "requested_restart": requested_restart,
                "effective_restart": effective_restart,
                "max_krylov_bytes": int(config["max_krylov_bytes"]),
                # A cold-only worker (``--phases cold``) never records a
                # solve workspace sample; the parent merges the steady worker
                # record, so an empty list must not abort the cold process.
                "effective_workspace_bytes": max(allocated_bytes, default=0),
                "workspace_formula": "(2m+5)B_f+2B_c",
                "fine_vector_bytes": int(fine_elements * rhs.element_size()),
                "coarse_vector_bytes": int(coarse_elements * rhs.element_size()),
            },
            "memory": memory,
            "runtime_cache": cache_report,
            "timing_boundary": pyqcu_timing_boundary,
            "provenance": provenance,
        }
    finally:
        if setup_sampler is not None:
            setup_sampler.stop()
        if runtime is not None:
            _close_strict_runtime(runtime)
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)


def _verify_quda_nullvec_conversion(payload: Mapping[str, Any]) -> Dict[str, Any]:
    qio = payload["inputs"]["quda_qio"]
    prefix = qio.get("prefix")
    manifest_path = qio.get("conversion_manifest")
    if not prefix or not manifest_path:
        raise BenchmarkSkip(
            "shared_nullvec_qio_missing",
            "QUDA requires --quda-nullvec-prefix and --quda-nullvec-manifest; "
            "native random vectors are intentionally not benchmarked")
    path = Path(manifest_path)
    if not path.is_file():
        raise BenchmarkSkip("shared_nullvec_manifest_missing", str(path))
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkFailure("invalid_nullvec_manifest", repr(exc)) from exc
    if (manifest.get("schema") != QIO_MANIFEST_SCHEMA or
            manifest.get("schema_version") != QIO_MANIFEST_VERSION):
        raise BenchmarkFailure(
            "invalid_nullvec_manifest_schema",
            f"schema/version={manifest.get('schema')!r}/"
            f"{manifest.get('schema_version')!r}, expected="
            f"{QIO_MANIFEST_SCHEMA!r}/{QIO_MANIFEST_VERSION}")
    expected = payload["input_fingerprints"]["null_vectors"]["sha256"]
    observed = manifest.get("canonical_dataset_sha256")
    if observed != expected:
        raise BenchmarkFailure(
            "nullvec_conversion_digest_mismatch",
            "manifest canonical_dataset_sha256="
            f"{observed!r}, expected={expected!r}")
    e12_source = manifest.get("source_sha256")
    try:
        valid_e12_source = (
            isinstance(e12_source, str) and len(e12_source) == 64 and
            int(e12_source, 16) >= 0)
    except ValueError:
        valid_e12_source = False
    if not valid_e12_source:
        raise BenchmarkFailure(
            "nullvec_conversion_source_digest_invalid",
            f"manifest source_sha256={e12_source!r}")
    if (int(manifest.get("nvec", -1)) != NVECS or
            list(manifest.get("block_xyzt", [])) != list(BLOCK) or
            list(manifest.get("lattice_xyzt", [])) != list(LATTICE)):
        raise BenchmarkFailure("nullvec_conversion_config_mismatch", repr(manifest))
    if manifest.get("precision") != {
            "canonical": "complex64",
            "qio": "QUDA_SINGLE_PRECISION",
            "real_storage": "float32",
    }:
        raise BenchmarkFailure(
            "nullvec_conversion_precision_mismatch",
            repr(manifest.get("precision")))
    expected_layout = {
        "canonical": "[nvec,spin,color,x,y,z,t]",
        "qio_host": QIO_HOST_LAYOUT,
        "field_order": "QUDA_SPACE_SPIN_COLOR_FIELD_ORDER",
        "site_subset": "QUDA_FULL_SITE_SUBSET",
        "parity": "full (even and odd; QUDA_INVALID_PARITY metadata)",
        "gamma_basis": QIO_GAMMA_BASIS,
        "basis_transform": "identity",
    }
    if manifest.get("layout") != expected_layout:
        raise BenchmarkFailure(
            "nullvec_conversion_layout_mismatch", repr(manifest.get("layout")))
    manifest_prefix = manifest.get("qio_prefix")
    cli_prefix_path = Path(str(prefix)).resolve()
    manifest_prefix_path = Path(str(manifest_prefix)) if manifest_prefix else None
    if manifest_prefix_path is not None and not manifest_prefix_path.is_absolute():
        manifest_prefix_path = path.parent / manifest_prefix_path
    if (manifest_prefix_path is not None and
            manifest_prefix_path.resolve() != cli_prefix_path):
        raise BenchmarkFailure(
            "nullvec_conversion_prefix_mismatch",
            f"CLI prefix={prefix!r}, manifest prefix={manifest_prefix!r}")
    expected_artifact_path = Path(
        f"{cli_prefix_path}_level_0_nvec_{NVECS}").resolve()
    if manifest.get("expected_quda_filename") != expected_artifact_path.name:
        raise BenchmarkFailure(
            "nullvec_conversion_filename_mismatch",
            f"manifest={manifest.get('expected_quda_filename')!r}, "
            f"expected={expected_artifact_path.name!r}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 1:
        raise BenchmarkFailure(
            "nullvec_conversion_artifacts_missing",
            "manifest must list exactly the QIO file QUDA will load")
    verified_artifacts = []
    verification_started = time.perf_counter()
    for item in artifacts:
        if not isinstance(item, Mapping):
            raise BenchmarkFailure("invalid_nullvec_artifact", repr(item))
        artifact_path = Path(str(item.get("path", "")))
        if not artifact_path.is_absolute():
            artifact_path = path.parent / artifact_path
        if artifact_path.resolve() != expected_artifact_path:
            raise BenchmarkFailure(
                "nullvec_conversion_artifact_path_mismatch",
                f"manifest artifact={artifact_path.resolve()}, "
                f"QUDA loads={expected_artifact_path}")
        if artifact_path.is_symlink() or not artifact_path.is_file():
            raise BenchmarkSkip("shared_nullvec_qio_missing", str(artifact_path))
        if item.get("format") != "USQCD QIO singlefile":
            raise BenchmarkFailure(
                "invalid_nullvec_artifact", repr(item))
        size = int(artifact_path.stat().st_size)
        if size != int(item.get("size_bytes", -1)):
            raise BenchmarkFailure(
                "nullvec_qio_size_mismatch",
                f"{artifact_path}: size={size}, expected={item.get('size_bytes')}")
        digest = _sha256_file(artifact_path)
        if digest != item.get("sha256"):
            raise BenchmarkFailure(
                "nullvec_qio_digest_mismatch",
                f"{artifact_path}: sha256={digest}, expected={item.get('sha256')}")
        verified_artifacts.append({
            "path": str(artifact_path.resolve()),
            "size_bytes": size,
            "sha256": digest,
        })
    round_trip = manifest.get("round_trip")
    if not isinstance(round_trip, Mapping) or round_trip.get("byte_exact") is not True:
        raise BenchmarkFailure(
            "nullvec_conversion_roundtrip_missing", repr(round_trip))
    return {
        "prefix": str(cli_prefix_path),
        "manifest": str(path.resolve()),
        "source_sha256": e12_source,
        "canonical_dataset_sha256": observed,
        "conversion_tool_commit": manifest.get("conversion_tool_commit"),
        "artifacts": verified_artifacts,
        "verification_seconds": time.perf_counter() - verification_started,
    }


def _normalise_quda_scalar(value: Any) -> Any:
    """Make enum/int/char-buffer values comparable and JSON friendly."""
    if isinstance(value, bytes):
        return value.split(b"\0", 1)[0].decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value.split("\0", 1)[0]
    enum_name = getattr(value, "name", None)
    if isinstance(enum_name, str):
        return enum_name
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, (bool, int, float, str)):
        return enum_value
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    return repr(value)


def _quda_values_equal(observed: Any, expected: Any) -> bool:
    expected_enum_value = getattr(expected, "value", None)
    if isinstance(expected_enum_value, (bool, int, float)):
        observed_value = getattr(observed, "value", observed)
        if isinstance(observed_value, (bool, int, float)):
            return observed_value == expected_enum_value
    left = _normalise_quda_scalar(observed)
    right = _normalise_quda_scalar(expected)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    return left == right


def _quda_vec_infile_row(value: Any, width: int = 256) -> List[int]:
    """Encode one Quda ``char[256]`` row for PyQUDA's property setter."""
    if isinstance(value, str):
        raw = value.encode("utf-8")
    elif isinstance(value, bytes):
        raw = value.split(b"\0", 1)[0]
    else:
        try:
            raw = bytes(int(item) & 0xFF for item in value)
        except (TypeError, ValueError) as exc:
            raise BenchmarkFailure(
                "quda_config_assignment_failed",
                f"vec_infile value={value!r} is not byte-like") from exc
        raw = raw.split(b"\0", 1)[0]
    if len(raw) >= width:
        raise BenchmarkFailure(
            "quda_config_assignment_failed",
            f"vec_infile path exceeds {width - 1} bytes")
    return list(raw + b"\0" * (width - len(raw)))


def _set_indexed(target: Any, field: str, index: int, value: Any) -> None:
    """Set a Quda array property and verify the reassigned backing struct.

    PyQUDA exposes several C arrays through getters that return a fresh Python
    list.  Mutating that list alone therefore changes nothing in the Quda
    struct.  ``vec_infile`` is additionally exposed as fixed-width character
    rows, so it needs a full 256-byte row before invoking its setter.
    """
    if not hasattr(target, field):
        raise BenchmarkFailure("quda_config_field_missing", field)
    try:
        original = getattr(target, field)
        updated = list(original)
    except Exception as exc:
        raise BenchmarkFailure(
            "quda_config_read_failed", f"{field}: {exc!r}") from exc
    if (field != "vec_infile" and
            (index < 0 or index >= len(updated))):
        raise BenchmarkFailure(
            "quda_config_index_missing",
            f"{field}[{index}] unavailable in sequence of length {len(updated)}")
    if field == "vec_infile":
        # The Cython getter returns truncated bytes and may expose
        # uninitialized bytes for unused rows.  Do not round-trip those rows:
        # the setter requires exactly n_level rows of exactly 256 chars.
        try:
            row_count = int(getattr(target, "n_level"))
        except Exception:
            row_count = len(updated)
        if index < 0 or row_count <= index or row_count <= 0:
            raise BenchmarkFailure(
                "quda_config_index_missing",
                f"vec_infile[{index}] unavailable for n_level={row_count}")
        updated = [[0] * 256 for _ in range(row_count)]
        expected_value: Any = _quda_vec_infile_row(value)
        updated[index] = expected_value
    else:
        expected_value = value
        updated[index] = value
    try:
        setattr(target, field, updated)
    except Exception as exc:
        raise BenchmarkFailure(
            "quda_config_assignment_failed",
            f"{field}[{index}]={value!r}: {exc!r}") from exc
    try:
        observed_sequence = getattr(target, field)
        observed_values = list(observed_sequence)
        observed = observed_values[index]
    except Exception as exc:
        raise BenchmarkFailure(
            "quda_config_readback_failed", f"{field}[{index}]: {exc!r}") from exc
    compare_expected = value if field != "vec_infile" else value
    if not _quda_values_equal(observed, compare_expected):
        raise BenchmarkFailure(
            "quda_config_readback_mismatch",
            f"{field}[{index}] read back {_normalise_quda_scalar(observed)!r}, "
            f"expected {_normalise_quda_scalar(compare_expected)!r}")
    if field == "vec_infile":
        nonempty = [
            (row, _normalise_quda_scalar(item))
            for row, item in enumerate(observed_values)
            if row != index and _normalise_quda_scalar(item) != ""
        ]
        if nonempty:
            raise BenchmarkFailure(
                "quda_config_readback_mismatch",
                f"vec_infile unused rows are not empty: {nonempty!r}")


def _configure_quda_shared_nullvec(mg_param: Any, conversion: Mapping[str, Any]) -> None:
    try:
        from pyquda.enum_quda import QudaBoolean, QudaComputeNullVector
    except ImportError as exc:
        raise BenchmarkSkip("missing_pyquda_enum", repr(exc)) from exc
    try:
        # QUDA's MG constructor dispatches to ``loadVectors`` only in the
        # compute-null-vector branch when ``vec_load`` and ``vec_infile`` are
        # set.  With COMPUTE_NULL_VECTOR_NO plus a non-empty infile, this
        # QUDA version enters ``generateNullVectors`` instead, which is both
        # unintended for the shared canonical basis and can diverge before
        # the worker has a chance to emit its sentinel.
        mg_param.compute_null_vector = QudaComputeNullVector.QUDA_COMPUTE_NULL_VECTOR_YES
        observed_compute = getattr(mg_param, "compute_null_vector")
        if _quda_enum_name(observed_compute, QudaComputeNullVector) != (
                QudaComputeNullVector.QUDA_COMPUTE_NULL_VECTOR_YES.name):
            raise BenchmarkFailure(
                "quda_config_readback_mismatch",
                "compute_null_vector did not read back as YES")
        prefix = str(conversion["prefix"])
        # Only transition 0 is user supplied.  Deeper transitions are formed
        # by R from that same basis, matching PyQCU's propagation contract.
        _set_indexed(
            mg_param, "vec_load", 0, QudaBoolean.QUDA_BOOLEAN_TRUE)
        _set_indexed(mg_param, "vec_infile", 0, prefix)
        transition_count = max(0, int(getattr(mg_param, "n_level")) - 1)
        for index in range(1, transition_count):
            _set_indexed(
                mg_param, "vec_load", index,
                QudaBoolean.QUDA_BOOLEAN_FALSE)
            # With generate_all_levels=false QUDA expects the parent MG to
            # restrict transition 0's vectors into this level.  Leaving the
            # default ``num_setup_iter=1`` would still enter generateNullVectors
            # and silently replace the shared coarse basis.
            _set_indexed(mg_param, "num_setup_iter", index, 0)
    except BenchmarkFailure:
        raise
    except Exception as exc:
        raise BenchmarkFailure("quda_nullvec_injection_failed", repr(exc)) from exc


def _quda_enum_name(value: Any, enum_type: Any = None) -> Any:
    if enum_type is not None:
        raw = getattr(value, "value", value)
        try:
            return enum_type(raw).name
        except (TypeError, ValueError):
            pass
    return _normalise_quda_scalar(value)


def _quda_sequence(target: Any, field: str, enum_type: Any = None) -> List[Any]:
    try:
        values = list(getattr(target, field))
    except Exception as exc:
        raise BenchmarkFailure(
            "quda_config_readback_failed", f"{field}: {exc!r}") from exc
    return [_quda_enum_name(value, enum_type) for value in values]


def _quda_parameter_snapshot(
        invert: Any, mg_param: Any, enum_types: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize the post-setup values exposed by PyQUDA's Cython structs."""
    n_level = int(getattr(mg_param, "n_level"))
    transition_count = max(0, n_level - 1)
    level_count = max(1, n_level)

    def enum(name: str, value: Any) -> Any:
        return _quda_enum_name(value, enum_types.get(name))

    def sequence(field: str, count: int, enum_name: Optional[str] = None) -> List[Any]:
        values = _quda_sequence(
            mg_param, field, enum_types.get(enum_name) if enum_name else None)
        return values[:count]

    precision_fields = (
        "cuda_prec", "cuda_prec_sloppy", "cuda_prec_refinement_sloppy",
        "cuda_prec_precondition", "cuda_prec_eigensolver")
    return {
        "invert": {
            "inv_type": enum("inv_type", getattr(invert, "inv_type")),
            "matpc_type": enum("matpc_type", getattr(invert, "matpc_type")),
            "solve_type": enum("solve_type", getattr(invert, "solve_type")),
            "solution_type": enum(
                "solution_type", getattr(invert, "solution_type")),
            "use_init_guess": enum(
                "use_init_guess", getattr(invert, "use_init_guess")),
            "preserve_source": enum(
                "preserve_source", getattr(invert, "preserve_source")),
            "gcrNkrylov": int(getattr(invert, "gcrNkrylov")),
            "maxiter": int(getattr(invert, "maxiter")),
            "tol": float(getattr(invert, "tol")),
            "precision": {
                field: enum("precision", getattr(invert, field))
                for field in precision_fields
            },
        },
        "multigrid": {
            "n_level": n_level,
            "coarsest_level_index": n_level - 1,
            "compute_null_vector": enum(
                "compute_null_vector", getattr(
                    mg_param, "compute_null_vector")),
            "generate_all_levels": enum(
                "boolean", getattr(mg_param, "generate_all_levels")),
            "run_verify": enum(
                "boolean", getattr(mg_param, "run_verify")),
            "transition": {
                "n_vec": sequence("n_vec", transition_count),
                "n_block_ortho": sequence("n_block_ortho", transition_count),
                "vec_load": sequence("vec_load", transition_count, "boolean"),
                "vec_infile": sequence("vec_infile", transition_count),
                "num_setup_iter": sequence(
                    "num_setup_iter", transition_count),
                "setup_use_mma": sequence(
                    "setup_use_mma", transition_count, "boolean"),
                "dslash_use_mma": sequence(
                    "dslash_use_mma", transition_count, "boolean"),
                "transfer_use_mma": sequence(
                    "transfer_use_mma", transition_count, "boolean"),
            },
            "levels": {
                "nu_pre": sequence("nu_pre", level_count),
                "nu_post": sequence("nu_post", level_count),
                "smoother": sequence("smoother", level_count, "inv_type"),
                "smoother_tol": sequence("smoother_tol", level_count),
                "coarse_solver": sequence(
                    "coarse_solver", level_count, "inv_type"),
                "coarse_solver_maxiter": sequence(
                    "coarse_solver_maxiter", level_count),
                "coarse_solver_tol": sequence(
                    "coarse_solver_tol", level_count),
            },
            "precision_null": sequence(
                "precision_null", level_count, "precision"),
        },
    }


def _quda_expected_parameters(
        config: Mapping[str, Any], conversion_prefix: Optional[str]) -> Dict[str, Any]:
    precision_name = (
        "QUDA_SINGLE_PRECISION"
        if config["precision"]["name"] == "c64" else
        "QUDA_DOUBLE_PRECISION")
    coarse_precision_name = (
        "QUDA_DOUBLE_PRECISION"
        if config["quda_coarse_precision"]["effective"] == "double" else
        "QUDA_SINGLE_PRECISION")
    n_level = int(config["levels"])
    transition_count = max(0, n_level - 1)
    library_strategy = config.get("quda_strategy", "aligned") == "library"
    if library_strategy:
        level_parameters = {
            "nu_pre": [0] * n_level,
            "nu_post": [8] * n_level,
            "smoother": ["QUDA_CA_GCR_INVERTER"] * n_level,
            "smoother_tol": [0.25] * n_level,
            "coarse_solver": (
                ["QUDA_GCR_INVERTER"] * max(0, n_level - 1) +
                ["QUDA_CA_GCR_INVERTER"]),
            "coarse_solver_maxiter": [16] * n_level,
            "coarse_solver_tol": [0.25] * n_level,
        }
    else:
        level_parameters = {
            "nu_pre": [int(config["nu_pre"])] * n_level,
            "nu_post": [int(config["nu_post"])] * n_level,
            "smoother": ["QUDA_MR_INVERTER"] * n_level,
            "smoother_tol": [0.0] * n_level,
            "coarse_solver": (
                ["QUDA_GCR_INVERTER"] * max(0, n_level - 1) +
                ["QUDA_CA_GCR_INVERTER"]),
            "coarse_solver_maxiter": (
                [1] * max(0, n_level - 1) +
                [int(config["coarse_max_iter"])]),
            "coarse_solver_tol": [
                float(config["coarse_tolerance"])] * n_level,
        }
    active = {
        "transition": {
            "n_vec": [int(config["nvec"])] * transition_count,
            "n_block_ortho": [2] * transition_count,
            "vec_load": (
                (["QUDA_BOOLEAN_TRUE"] +
                 ["QUDA_BOOLEAN_FALSE"] * max(0, transition_count - 1))
                if transition_count > 0 else []),
            "vec_infile": (
                ([None if conversion_prefix is None else str(conversion_prefix)] +
                 [""] * max(0, transition_count - 1))
                if transition_count > 0 else []),
            "num_setup_iter": (
                [1] + [0] * max(0, transition_count - 1)
                if transition_count > 1 else
                [1] if transition_count == 1 else []),
            "setup_use_mma": ["QUDA_BOOLEAN_FALSE"] * transition_count,
            "dslash_use_mma": ["QUDA_BOOLEAN_FALSE"] * transition_count,
            "transfer_use_mma": ["QUDA_BOOLEAN_FALSE"] * transition_count,
        },
        "levels": level_parameters,
    }
    return {
        "invert": {
            "inv_type": "QUDA_GCR_INVERTER",
            "matpc_type": "QUDA_MATPC_ODD_ODD",
            # With MATPC_ODD_ODD, PyQUDA/QUDA resolves the outer solve to
            # DIRECT_PC odd though the public solution is MAT_SOLUTION.
            # This is the actual odd-even preconditioned path used by QUDA
            # and must be part of the formal contract.
            "solve_type": "QUDA_DIRECT_PC_SOLVE",
            "solution_type": "QUDA_MAT_SOLUTION",
            "use_init_guess": "QUDA_USE_INIT_GUESS_NO",
            "preserve_source": "QUDA_PRESERVE_SOURCE_YES",
            "gcrNkrylov": int(config["restart_effective"]),
            "maxiter": int(config["max_iter"]),
            "tol": float(config["tolerance"]),
            "precision": {
                "cuda_prec": precision_name,
                "cuda_prec_eigensolver": precision_name,
                "cuda_prec_sloppy": coarse_precision_name,
                "cuda_prec_refinement_sloppy": coarse_precision_name,
                "cuda_prec_precondition": coarse_precision_name,
            },
        },
        "multigrid": {
            "n_level": n_level,
            "coarsest_level_index": n_level - 1,
            "compute_null_vector": (
                "QUDA_COMPUTE_NULL_VECTOR_YES" if n_level > 1 else
                "QUDA_COMPUTE_NULL_VECTOR_NO"),
            "generate_all_levels": (
                "QUDA_BOOLEAN_FALSE" if n_level != 2 else
                "QUDA_BOOLEAN_TRUE"),
            "run_verify": "QUDA_BOOLEAN_FALSE",
            "precision_null": [coarse_precision_name] * n_level,
            **active,
        },
    }


def _quda_parameter_mismatches(
        actual: Mapping[str, Any], expected: Mapping[str, Any]) -> List[str]:
    mismatches: List[str] = []
    for group in ("invert", "multigrid"):
        actual_group = actual.get(group)
        expected_group = expected.get(group)
        if not isinstance(actual_group, Mapping) or not isinstance(expected_group, Mapping):
            mismatches.append(f"{group} missing")
            continue
        for key, wanted in expected_group.items():
            got = actual_group.get(key)
            if key == "precision":
                if not isinstance(got, Mapping) or got != wanted:
                    mismatches.append(f"{group}.{key}={got!r}, expected={wanted!r}")
            elif key in ("transition", "levels"):
                if not isinstance(got, Mapping) or got != wanted:
                    mismatches.append(f"{group}.{key}={got!r}, expected={wanted!r}")
            elif got != wanted:
                mismatches.append(f"{group}.{key}={got!r}, expected={wanted!r}")
    return mismatches


def _quda_invert_output_contract(invert_quda: Any) -> Dict[str, Any]:
    """Check the low-level PyQUDA API used for caller-owned output.

    ``FermionDirac.invert`` allocates a new ``LatticeFermion`` by design, but
    its implementation delegates to ``pyquda.quda.invertQuda(h_x, h_b,
    param)``.  The latter is the fair timing entry point because ``h_x`` can
    be allocated and reused by the caller.
    """
    try:
        signature = inspect.signature(invert_quda)
        parameters = list(signature.parameters.values())
    except (TypeError, ValueError) as exc:
        return {
            "requested": "caller_preallocated",
            "supported": False,
            "formal_eligible": False,
            "api": "uninspectable",
            "detail": repr(exc),
        }
    if (len(parameters) == 3 and
            all(parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for parameter in parameters)):
        return {
            "requested": "caller_preallocated",
            "supported": True,
            "formal_eligible": True,
            "api": str(signature),
            "method": "pyquda.quda.invertQuda(h_x,h_b,param)",
            "output_argument": parameters[0].name,
            "detail": "low-level invertQuda writes the caller-owned output",
        }
    return {
        "requested": "caller_preallocated",
        "supported": False,
        "formal_eligible": False,
        "api": str(signature),
        "detail": (
            "pyquda.quda.invertQuda does not expose the expected "
            "(h_x,h_b,param) caller-output API"),
    }


def _close_quda_dirac(dirac: Any) -> List[str]:
    """Best-effort QUDA cleanup that never masks the primary failure."""
    errors: List[str] = []
    if dirac is None:
        return errors
    try:
        if getattr(dirac, "multigrid", None) is not None:
            dirac.multigrid.destroy()
    except BaseException as exc:
        errors.append(f"multigrid.destroy: {exc!r}")
    try:
        dirac.freeGauge()
    except BaseException as exc:
        errors.append(f"freeGauge: {exc!r}")
    return errors


def _run_quda_worker(payload: Mapping[str, Any]) -> Dict[str, Any]:
    config = payload["protocol"]
    inputs = payload["inputs"]
    phase_request = _phase_request(payload)
    mpi_context = payload.get("mpi_context") or _worker_mpi_context(payload)
    _activate_runtime_globals(config, inputs)
    conversion = (
        _verify_quda_nullvec_conversion(payload)
        if int(config["levels"]) > 1 else
        {
            "required": False,
            "prefix": None,
            "reason": "QUDA MG-1 has no transfer level or null-vector basis",
        })
    reduction_runtime = _prepare_quda_reduction_runtime()
    try:
        reduction_runtime["qmp"] = _initialize_quda_qmp_runtime(
            reduction_runtime, mpi_context["process_grid"])
    except Exception as exc:
        if int(mpi_context["size"]) > 1:
            raise BenchmarkUnsupported(
                "quda_mpi_unsupported",
                f"QMP/QUDA MPI initialization failed for grid "
                f"{mpi_context['process_grid']}: {exc!r}") from exc
        raise
    try:
        import numpy as np
        import torch
        import pyquda  # noqa: F401
        import pyquda_utils.core as core
        from pyquda_comm.pointer import Pointer
        from pyquda.field import LatticeFermion, LatticeGauge
        from pyquda.quda import invertQuda
        from pyquda.enum_quda import (
            QudaBoolean, QudaInverterType, QudaPrecision, QudaMatPCType,
            QudaPreserveSource, QudaSolutionType, QudaSolveType,
            QudaUseInitGuess, QudaComputeNullVector, QudaVerbosity,
            QudaDslashType)
        from pyquda.dirac import general as dirac_general
        sys.path.insert(0, str(HERE))
        from run_quda_py import field_to_scxyzt, reconstruct_full_b
        from common import (
            full_gauge_numpy, full_to_qdp, global_parity_to_local,
            local_geometry)
        from pyqcu import tools
    except (ImportError, OSError) as exc:
        raise BenchmarkSkip("missing_quda_dependency", repr(exc)) from exc

    device = _select_accelerator(
        torch,
        str((payload["execution"].get("device") or {}).get("requested", "v100")),
        int(mpi_context.get("local_rank", mpi_context["rank"])),
        int(mpi_context.get("local_size", mpi_context["size"])))
    device_uuid = _torch_runtime_provenance(torch, device)["device_uuid"]
    trace_path = os.environ.get("PYQCU_QUDA_TRACE_FILE")
    trace_enabled = trace_path is not None and trace_path != ""
    mg_trace_path = os.environ.get("QUDA_MG_TRACE_FILE")
    mg_trace_enabled = mg_trace_path is not None and mg_trace_path != ""
    if mg_trace_enabled:
        trace_target = Path(mg_trace_path).resolve()
        trace_target.parent.mkdir(parents=True, exist_ok=True)
        if trace_target.exists():
            trace_target.unlink()
    precision = config["precision"]["name"]
    complex_dtype = torch.complex64 if precision == "c64" else torch.complex128
    # PyQUDA's QDP _NDArray bridge rejects complex64 host arrays.  This host
    # staging dtype is independent from QUDA's requested device precision.
    quda_qdp_host_dtype = _quda_qdp_host_dtype(precision)
    quda_precision = (QudaPrecision.QUDA_SINGLE_PRECISION
                      if precision == "c64" else QudaPrecision.QUDA_DOUBLE_PRECISION)

    io_started = time.perf_counter()
    fingerprints = payload["input_fingerprints"]
    gauge_np = _load_h5_local_array(
        inputs["gauge"]["path"], inputs["gauge"]["dataset"],
        LATTICE, mpi_context, parity_compressed=True)
    source_np = _load_h5_local_array(
        inputs["source"]["path"], inputs["source"]["dataset"],
        LATTICE, mpi_context, parity_compressed=True)
    if int(mpi_context["size"]) > 1:
        local_lattice, _, _ = local_geometry(
            tuple(int(value) for value in LATTICE),
            grid=mpi_context["process_grid"],
            rank=mpi_context["rank"], require_even=True)
        _verify_local_input_slab(
            "gauge", gauge_np, fingerprints["gauge"], local_lattice,
            parity_compressed=True)
        _verify_local_input_slab(
            "source", source_np, fingerprints["source"], local_lattice,
            parity_compressed=True)
    else:
        _verify_loaded_input("gauge", gauge_np, fingerprints["gauge"])
        _verify_loaded_input("source", source_np, fingerprints["source"])
    input_io_s = time.perf_counter() - io_started

    runtime_started = time.perf_counter()
    try:
        pyquda.init(
            grid_size=list(mpi_context["process_grid"]),
            latt_size=list(LATTICE), backend="torch",
            backend_target="cuda", enable_nvshmem=False)
    except Exception as exc:
        if int(mpi_context["size"]) > 1:
            raise BenchmarkUnsupported(
                "quda_mpi_unsupported",
                f"PyQUDA/QUDA could not initialize MPI grid "
                f"{mpi_context['process_grid']}: {exc!r}") from exc
        raise
    torch.cuda.synchronize(device)
    info = core.LatticeInfo(list(LATTICE), 1, 1.0)
    runtime_init_s = time.perf_counter() - runtime_started

    input_prepare_started = time.perf_counter()
    global_lattice = tuple(int(value) for value in LATTICE)
    if int(mpi_context["size"]) > 1:
        local_lattice, _, _ = local_geometry(
            global_lattice, grid=mpi_context["process_grid"],
            rank=mpi_context["rank"], require_even=True)
        local_gauge_cpu = torch.from_numpy(gauge_np).to(
            dtype=_quda_qdp_torch_host_dtype(torch, precision))
        local_source_cpu = torch.from_numpy(source_np).to(
            dtype=_quda_qdp_torch_host_dtype(torch, precision))
    else:
        local_gauge_cpu = torch.from_numpy(gauge_np).to(
            dtype=_quda_qdp_torch_host_dtype(torch, precision))
        local_source_cpu = None
    qdp = full_to_qdp(full_gauge_numpy(local_gauge_cpu)).astype(
        quda_qdp_host_dtype, copy=False)
    gauge_eo = info.evenodd(np.ascontiguousarray(qdp), True)
    if local_source_cpu is None:
        source_full = reconstruct_full_b(source_np).reshape(12, *LATTICE)
    else:
        source_full = tools.poooxyzt2oooxyzt(local_source_cpu)
        source_full = source_full.reshape(12, *local_lattice)
    if isinstance(source_full, torch.Tensor):
        source_full = source_full.detach().cpu().numpy()
    tzyxsc = np.ascontiguousarray(np.transpose(
        source_full.astype(quda_qdp_host_dtype, copy=False), (4, 3, 2, 1, 0)))
    rhs_eo = np.ascontiguousarray(info.evenodd(tzyxsc, False))
    input_prepare_s = time.perf_counter() - input_prepare_started

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    setup_baseline = {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "nvidia_smi": _nvidia_smi_used(device_uuid=device_uuid),
    }
    setup_sampler: Optional[_CudaDeviceMemorySampler] = (
        _CudaDeviceMemorySampler(torch, device).start())
    setup_started = time.perf_counter()
    dirac = None
    try:
        gauge_field = LatticeGauge(
            info, 4, torch.from_numpy(np.ascontiguousarray(gauge_eo)).to(device))
        rhs_field = LatticeFermion(info, torch.from_numpy(rhs_eo).to(device))
        if LEVELS == 1:
            mg_param, mg_inv_param = dirac_general.newQudaMultigridParam(
                QudaDslashType.QUDA_CLOVER_WILSON_DSLASH,
                MASS, float(config["kappa"]), [])
            # PyQUDA appends one terminal level to the requested block list.
            # Override it to represent QUDA_MG_LEVEL=1 exactly: only level 0
            # exists and no transfer/coarse operator is constructed.
            mg_param.n_level = 1
            mg_param.compute_null_vector = (
                QudaComputeNullVector.QUDA_COMPUTE_NULL_VECTOR_NO)
            mg_param.generate_all_levels = QudaBoolean.QUDA_BOOLEAN_FALSE
            mg_instance = core.Multigrid(mg_param, mg_inv_param)
            mg_instance.setParam()
            dirac = core.getClover(
                info, MASS, float(config["tolerance"]),
                int(config["max_iter"]), clover_csw_t=1.0,
                multigrid=mg_instance)
        else:
            dirac = core.getClover(
                info, MASS, float(config["tolerance"]),
                int(config["max_iter"]), clover_csw_t=1.0,
                multigrid=[
                    list(value)
                    for value in config["block_xyzt_per_level"]
                ])
        coarse_precision = (
            QudaPrecision.QUDA_SINGLE_PRECISION
            if config["quda_coarse_precision"]["effective"] == "single"
            else quda_precision)
        dirac.setPrecision(
            cuda=quda_precision, sloppy=coarse_precision,
            precondition=coarse_precision,
            refinement_sloppy=coarse_precision,
            eigensolver=quda_precision)
        invert = dirac.invert_param
        invert.inv_type = QudaInverterType.QUDA_GCR_INVERTER
        # The strict PyQCU side solves the odd Schur block.  PyQUDA defaults
        # to EVEN_EVEN unless this is assigned explicitly, so relying on the
        # constructor default silently makes the two finite-tolerance solves
        # incomparable.
        invert.matpc_type = QudaMatPCType.QUDA_MATPC_ODD_ODD
        if not hasattr(invert, "gcrNkrylov"):
            raise BenchmarkFailure(
                "quda_restart_field_missing", "invert_param.gcrNkrylov")
        invert.gcrNkrylov = int(config["restart_effective"])
        invert.maxiter = int(config["max_iter"])
        invert.tol = float(config["tolerance"])
        if trace_enabled:
            # QUDA's outer Solver::PrintStats is the desired trace.  Keep the
            # MG preconditioner quiet so nested coarse solves do not obscure
            # the one outer GCR curve being compared with Strict FGMRES.
            invert.verbosity = QudaVerbosity.QUDA_VERBOSE
            invert.verbosity_precondition = QudaVerbosity.QUDA_SILENT
        if mg_trace_enabled and hasattr(invert, "verbosity"):
            invert.verbosity = QudaVerbosity.QUDA_VERBOSE
        if hasattr(invert, "use_init_guess"):
            invert.use_init_guess = QudaUseInitGuess.QUDA_USE_INIT_GUESS_NO
        else:
            raise BenchmarkFailure(
                "quda_config_field_missing", "invert_param.use_init_guess")
        if hasattr(invert, "preserve_source"):
            invert.preserve_source = QudaPreserveSource.QUDA_PRESERVE_SOURCE_YES
        else:
            raise BenchmarkFailure(
                "quda_config_field_missing", "invert_param.preserve_source")

        mg_obj = getattr(dirac, "multigrid", None)
        mg_param = None if mg_obj is None else getattr(mg_obj, "param", None)
        if mg_param is None:
            raise BenchmarkFailure(
                "quda_multigrid_param_missing", "dirac.multigrid.param")
        n_level = int(getattr(mg_param, "n_level"))
        transition_count = max(0, n_level - 1)
        for level in range(transition_count):
            _set_indexed(mg_param, "n_vec", level, NVECS)
            _set_indexed(mg_param, "n_block_ortho", level, 2)
            for field in ("setup_use_mma", "dslash_use_mma", "transfer_use_mma"):
                _set_indexed(mg_param, field, level,
                             QudaBoolean.QUDA_BOOLEAN_FALSE)
        if config.get("quda_strategy", "aligned") == "aligned":
            for level in range(max(1, n_level)):
                _set_indexed(mg_param, "nu_pre", level, int(config["nu_pre"]))
                _set_indexed(mg_param, "nu_post", level, int(config["nu_post"]))
                _set_indexed(
                    mg_param, "coarse_solver", level,
                    (QudaInverterType.QUDA_GCR_INVERTER
                     if level < n_level - 1 else
                     QudaInverterType.QUDA_CA_GCR_INVERTER))
                _set_indexed(
                    mg_param, "coarse_solver_maxiter", level,
                    1 if level < n_level - 1 else
                    int(config["coarse_max_iter"]))
                _set_indexed(
                    mg_param, "coarse_solver_tol", level,
                    float(config["coarse_tolerance"]))
                _set_indexed(
                    mg_param, "smoother", level,
                    QudaInverterType.QUDA_MR_INVERTER)
                _set_indexed(mg_param, "smoother_tol", level, 0.0)
        if trace_enabled and hasattr(mg_param, "verbosity"):
            for level in range(max(1, n_level)):
                _set_indexed(
                    mg_param, "verbosity", level,
                    QudaVerbosity.QUDA_SILENT)
        if LEVELS > 1:
            _configure_quda_shared_nullvec(mg_param, conversion)
        if LEVELS > 2:
            if not hasattr(mg_param, "generate_all_levels"):
                raise BenchmarkFailure(
                    "quda_multigrid_param_missing", "generate_all_levels")
            mg_param.generate_all_levels = QudaBoolean.QUDA_BOOLEAN_FALSE
        if not hasattr(mg_param, "run_verify"):
            raise BenchmarkFailure(
                "quda_multigrid_param_missing", "run_verify")
        # The independent full-operator true-residual gate is the correctness
        # authority for this benchmark.  QUDA's expensive recursive setup
        # verification remains available outside the timed comparison path.
        mg_param.run_verify = QudaBoolean.QUDA_BOOLEAN_FALSE
        timing_boundary = _quda_invert_output_contract(invertQuda)
        enum_types = {
            "boolean": QudaBoolean,
            "inv_type": QudaInverterType,
            "matpc_type": QudaMatPCType,
            "solve_type": QudaSolveType,
            "solution_type": QudaSolutionType,
            "compute_null_vector": QudaComputeNullVector,
            "use_init_guess": QudaUseInitGuess,
            "preserve_source": QudaPreserveSource,
            "precision": QudaPrecision,
        }
        expected_parameters = _quda_expected_parameters(
            config,
            None if conversion.get("prefix") is None else str(conversion["prefix"]))
        build_provenance = _benchmark_provenance(
            pyquda=pyquda, reduction_runtime=reduction_runtime)
        cmake_mismatches = _quda_cmake_feature_mismatches(
            build_provenance["cmake_features"], precision, int(config["levels"]))
        if config["profile"] == "formal" and cmake_mismatches:
            raise BenchmarkFailure(
                "quda_build_feature_contract_mismatch",
                "; ".join(cmake_mismatches),
                context={
                    "timing_boundary": timing_boundary,
                    "quda_parameters": {
                        "requested": expected_parameters,
                        "actual": None,
                        "mismatches": cmake_mismatches,
                    },
                    "provenance": build_provenance,
                })
        if config["profile"] == "formal" and not timing_boundary["supported"]:
            raise BenchmarkFailure(
                "quda_preallocated_output_unsupported",
                timing_boundary["detail"],
                context={
                    "timing_boundary": timing_boundary,
                    "quda_parameters": {
                        "requested": expected_parameters,
                        "actual": None,
                        "mismatches": ["caller-preallocated output unsupported"],
                    },
                })
        dirac.loadGauge(gauge_field)
        resolved_parameters = _quda_parameter_snapshot(
            invert, mg_param, enum_types)
        parameter_mismatches = _quda_parameter_mismatches(
            resolved_parameters, expected_parameters)
        if parameter_mismatches:
            raise BenchmarkFailure(
                "quda_parameter_contract_mismatch",
                "; ".join(parameter_mismatches),
                context={
                    "timing_boundary": timing_boundary,
                    "quda_parameters": {
                        "requested": expected_parameters,
                        "actual": resolved_parameters,
                        "mismatches": parameter_mismatches,
                    },
                })
        torch.cuda.synchronize(device)
        setup_finished = time.perf_counter()
        setup_device_memory = setup_sampler.stop()
        setup_sampler = None
        setup_s = setup_finished - setup_started
    except BaseException:
        if setup_sampler is not None:
            setup_sampler.stop()
        _close_quda_dirac(dirac)
        raise
    setup_memory = {
        "baseline": setup_baseline,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "nvidia_smi": _nvidia_smi_used(device_uuid=device_uuid),
        "device_wide_sampler": setup_device_memory,
        "timing_semantics": (
            "instrumented setup; sampler stop excluded from setup_seconds"),
        "note": (
            "PyTorch allocator excludes QUDA native allocations; the "
            "device-wide sampler captures transients but may include other processes"),
    }

    try:
        # The output is a caller-owned field.  Allocate it after QUDA setup and
        # before any warmup/timed solve; the low-level API below writes it in
        # place and never allocates a new solution inside the timed region.
        solution_field = LatticeFermion(info)
        torch.cuda.synchronize(device)
        if int(mpi_context["size"]) > 1:
            gauge_for_residual = local_gauge_cpu.to(
                device=device, dtype=complex_dtype).contiguous()
            rhs_for_residual = local_source_cpu.to(
                device=device, dtype=complex_dtype).contiguous()
        else:
            gauge_for_residual = torch.from_numpy(gauge_np).to(
                device=device, dtype=complex_dtype).contiguous()
            rhs_for_residual = torch.from_numpy(source_np).to(
                device=device, dtype=complex_dtype).contiguous()
        from pyqcu import dslash
        full_gauge = tools.poooxyzt2oooxyzt(gauge_for_residual).contiguous()
        real_dtype = (
            torch.float32 if complex_dtype == torch.complex64 else torch.float64)
        clover = dslash.make_clover(
            full_gauge,
            kappa=torch.tensor(
                [config["kappa"]], dtype=real_dtype, device=device),
            u_0=torch.ones([1], dtype=real_dtype, device=device))
    except BaseException:
        _close_quda_dirac(dirac)
        raise

    trace_solve_index = 0

    def solve_once(
            *, sample_device_memory: bool = False,
    ) -> Tuple[float, int, bool, float, Dict[str, Any]]:
        nonlocal trace_solve_index
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        sampler = (
            _CudaDeviceMemorySampler(torch, device).start()
            if sample_device_memory else None)
        started = time.perf_counter()
        try:
            if timing_boundary["supported"]:
                solution_field.data.zero_()
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                if mg_trace_enabled:
                    _append_trace_marker(
                        mg_trace_path, "python_solve_begin", trace_solve_index)
                with _capture_native_stdout(trace_path):
                    invertQuda(
                        solution_field.data_ptr, rhs_field.data_ptr, invert)
                if mg_trace_enabled:
                    _append_trace_marker(
                        mg_trace_path, "python_solve_end", trace_solve_index)
                trace_solve_index += 1
                solution = solution_field
            else:
                # Smoke may still exercise the high-level API, but this path
                # is explicitly marked non-formal by timing_boundary.
                with _capture_native_stdout(trace_path):
                    solution = dirac.invert(rhs_field)
                if mg_trace_enabled:
                    _append_trace_marker(
                        mg_trace_path, "python_solve_end", trace_solve_index)
                trace_solve_index += 1
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
        finally:
            device_memory = None if sampler is None else sampler.stop()
        iterations = int(invert.iter)
        memory_sample = {
            "cuda_peak_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)),
            "cuda_peak_reserved_bytes": int(
                torch.cuda.max_memory_reserved(device)),
            "nvidia_smi": _nvidia_smi_used(device_uuid=device_uuid),
            "device_wide_sampler": device_memory,
        }
        solution_full = field_to_scxyzt(info, solution)
        canonical = np.ascontiguousarray(solution_full * (MASS + 4.0))
        canonical_tensor = torch.from_numpy(canonical).to(
            device=device, dtype=complex_dtype)
        canonical_eo = tools.oooxyzt2poooxyzt(canonical_tensor).contiguous()
        residual = (
            _canonical_true_residual_mpi(
                canonical_eo, rhs_for_residual, gauge_for_residual, MASS,
                mpi_context["process_grid"])
            if int(mpi_context["size"]) > 1 else
            _canonical_true_residual(
                canonical_eo, rhs_for_residual, gauge_for_residual, MASS,
                full_gauge=full_gauge, clover=clover))
        converged = bool(residual <= float(config["true_residual_gate"]))
        return elapsed, iterations, converged, residual, memory_sample

    first_solve_memory = None
    cold_seconds = None
    cold_result_payload = None
    warmup_results = []
    try:
        if phase_request["cold"]:
            (elapsed, iterations, converged, residual,
             memory_sample) = solve_once(sample_device_memory=True)
            cold_seconds = elapsed
            cold_result_payload = {
                "seconds": elapsed,
                "iterations": iterations,
                "converged": converged,
                "true_residual_rel": residual,
                "memory": memory_sample,
            }
            first_solve_memory = {
                "excluded_from_formal_timing": True,
                **copy.deepcopy(cold_result_payload),
            }
        if phase_request["warmup"]:
            for warmup_index in range(WARMUPS):
                (elapsed, iterations, converged, residual,
                 memory_sample) = solve_once(
                    sample_device_memory=(
                        warmup_index == 0 and cold_seconds is None))
                if warmup_index == 0 and first_solve_memory is None:
                    first_solve_memory = {
                        "excluded_from_formal_timing": True,
                        "seconds": elapsed,
                        "iterations": iterations,
                        "converged": converged,
                        "true_residual_rel": residual,
                        "memory": memory_sample,
                    }
                warmup_results.append({
                    "seconds": elapsed,
                    "iterations": iterations,
                    "converged": converged,
                    "true_residual_rel": residual,
                })
    except BaseException:
        _close_quda_dirac(dirac)
        raise

    steady_baseline = {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "nvidia_smi": _nvidia_smi_used(device_uuid=device_uuid),
    }
    timings: List[float] = []
    iterations_list: List[int] = []
    converged_list: List[bool] = []
    residuals: List[float] = []
    solve_memory_samples: List[Dict[str, Any]] = []
    memory_probe: Dict[str, Any] = {}
    cleanup_errors: List[str] = []
    try:
        if phase_request["steady"]:
            for _ in range(int(config["repeats"])):
                (elapsed, iterations, converged, residual,
                 memory_sample) = solve_once()
                timings.append(elapsed)
                iterations_list.append(iterations)
                converged_list.append(converged)
                residuals.append(residual)
                solve_memory_samples.append(memory_sample)
            (_probe_elapsed, probe_iterations, probe_converged,
             probe_residual, memory_probe) = solve_once(
                sample_device_memory=True)
            memory_probe.update({
                "excluded_from_formal_timing": True,
                "iterations": probe_iterations,
                "converged": probe_converged,
                "true_residual_rel": probe_residual,
            })
        reference_report = None
        if (phase_request["steady"] and
                config["reference_solver"]["kind"] == "bicgstab"):
            invert.inv_type = QudaInverterType.QUDA_BICGSTAB_INVERTER
            invert.tol = float(config["tolerance"])
            invert.maxiter = int(config["max_iter"])
            invert.use_init_guess = QudaUseInitGuess.QUDA_USE_INIT_GUESS_NO
            if hasattr(invert, "inv_type_precondition"):
                invert.inv_type_precondition = (
                    QudaInverterType.QUDA_INVALID_INVERTER)
            if not hasattr(invert, "preconditioner"):
                raise BenchmarkFailure(
                    "quda_config_field_missing", "invert_param.preconditioner")
            # getClover() attaches the MG instance directly.  Clearing only
            # inv_type_precondition is insufficient: QUDA's solver factory
            # rejects BiCGStab whenever this pointer remains non-null.
            invert.preconditioner = Pointer("void")

            def reference_once() -> Tuple[float, int, float]:
                solution_field.data.zero_()
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                with _capture_native_stdout(trace_path):
                    invertQuda(
                        solution_field.data_ptr, rhs_field.data_ptr, invert)
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                solution_full = field_to_scxyzt(info, solution_field)
                canonical = np.ascontiguousarray(
                    solution_full * (MASS + 4.0))
                canonical_tensor = torch.from_numpy(canonical).to(
                    device=device, dtype=complex_dtype)
                canonical_eo = tools.oooxyzt2poooxyzt(
                    canonical_tensor).contiguous()
                residual = (
                    _canonical_true_residual_mpi(
                        canonical_eo, rhs_for_residual, gauge_for_residual,
                        MASS, mpi_context["process_grid"])
                    if int(mpi_context["size"]) > 1 else
                    _canonical_true_residual(
                        canonical_eo, rhs_for_residual, gauge_for_residual,
                        MASS, full_gauge=full_gauge, clover=clover))
                return elapsed, int(invert.iter), residual

            reference_warmups = []
            for _ in range(int(config["reference_solver"]["warmups"])):
                seconds, reference_iterations, residual = reference_once()
                reference_warmups.append({
                    "seconds": seconds,
                    "iterations": reference_iterations,
                    "true_residual_rel": residual,
                })
            reference_samples = []
            for _ in range(int(config["repeats"])):
                seconds, reference_iterations, residual = reference_once()
                reference_samples.append({
                    "seconds": seconds,
                    "iterations": reference_iterations,
                    "true_residual_rel": residual,
                })
            reference_report = {
                "kind": "bicgstab",
                "implementation": "pyquda.quda.invertQuda("
                                   "QUDA_BICGSTAB_INVERTER)",
                "warmups": reference_warmups,
                "steady": _median_mad([
                    item["seconds"] for item in reference_samples]),
                "iterations": _iteration_summary([
                    item["iterations"] for item in reference_samples]),
                "samples": reference_samples,
                "true_residual_max_rel": max(
                    item["true_residual_rel"]
                    for item in reference_samples),
                "excluded_from_speedup": True,
            }
    finally:
        cleanup_errors = _close_quda_dirac(dirac)

    phase_checks = []
    if cold_result_payload is not None:
        phase_checks.extend((
            bool(cold_result_payload["converged"]),
            float(cold_result_payload["true_residual_rel"]) <=
            float(config["true_residual_gate"])))
    if warmup_results:
        phase_checks.extend(
            bool(item["converged"]) and
            float(item["true_residual_rel"]) <=
            float(config["true_residual_gate"])
            for item in warmup_results)
    if phase_request["steady"]:
        phase_checks.extend((
            all(converged_list),
            bool(memory_probe.get("converged")),
            float(memory_probe.get("true_residual_rel", math.inf)) <=
            float(config["true_residual_gate"])))
    all_converged = all(phase_checks)
    measured_iterations = (
        list(iterations_list) if phase_request["steady"] else
        ([int(cold_result_payload["iterations"])]
         if cold_result_payload is not None else []) +
        [int(item["iterations"]) for item in warmup_results])
    measured_residuals = (
        list(residuals) if phase_request["steady"] else
        ([float(cold_result_payload["true_residual_rel"])]
         if cold_result_payload is not None else []) +
        [float(item["true_residual_rel"]) for item in warmup_results])
    timing_payload = {
        "input_io_seconds": input_io_s,
        "input_prepare_seconds": input_prepare_s,
        "input_read_distributed": bool(int(mpi_context["size"]) > 1),
        "input_read_scope": (
            "rank-sliced HDF5 hyperslab"
            if int(mpi_context["size"]) > 1 else "full HDF5 dataset"),
        "runtime_init_seconds": runtime_init_s,
        "setup_seconds": setup_s,
        "warmups": warmup_results,
        "steady": (
            _median_mad(timings) if timings else
            {"samples_seconds": [], "median_seconds": None,
             "mad_seconds": None}),
    }
    if cold_seconds is not None:
        timing_payload["cold_seconds"] = cold_seconds

    trace_path_quda = _side_trace_path("quda")
    by_phase: Dict[str, List[Dict[str, Any]]] = {}
    if phase_request["cold"]:
        levels = _mg_levels_from_trace(
            "quda", trace_path_quda, [0], int(config["levels"]))
        if levels is not None:
            by_phase["cold"] = levels
    offset = 1 if phase_request["cold"] else 0
    if phase_request["warmup"]:
        levels = _mg_levels_from_trace(
            "quda", trace_path_quda,
            list(range(offset, offset + WARMUPS)),
            int(config["levels"]))
        if levels is not None:
            by_phase["warmup"] = levels
        offset += WARMUPS
    if phase_request["steady"]:
        levels = _mg_levels_from_trace(
            "quda", trace_path_quda,
            list(range(offset, offset + int(config["repeats"]))),
            int(config["levels"]))
        if levels is not None:
            by_phase["steady"] = levels
    mg_levels = (
        by_phase.get("steady") or by_phase.get("warmup") or
        by_phase.get("cold"))
    if mg_levels is None:
        mg_levels = _fallback_mg_levels(
            config, timing_payload, measured_iterations,
            measured_residuals)
    timing_boundary = dict(timing_boundary)
    timing_boundary.update({
        "formal_eligible": bool(timing_boundary.get("formal_eligible")) and
        bool(timing_boundary.get("supported")),
        "zero_initial_guess_before_timer": True,
        "preserve_source": resolved_parameters["invert"]["preserve_source"],
        "timed_operation": "pyquda.quda.invertQuda(h_x,h_b,param)",
        "performance_call": "omitted; no high-level allocation/performance call in timer",
    })
    provenance = _benchmark_provenance(
        pyquda=pyquda, reduction_runtime=reduction_runtime)
    provenance.update({
        "runtime": _torch_runtime_provenance(torch, device),
        "shared_nullvec_conversion": conversion,
        "reduction_runtime": reduction_runtime,
        "cleanup_errors": cleanup_errors,
    })
    return {
        "side": "quda",
        "status": "ok" if all_converged else "failed",
        "reason": None if all_converged else {
            "code": "convergence_or_true_residual_gate",
            "detail": f"converged={converged_list}, residuals={residuals}",
        },
        "config_hash": config["config_hash"],
        "input_bundle_hash": payload["input_fingerprints"]["bundle_hash"],
        "timing": timing_payload,
        "reference_solver": reference_report,
        "iterations": _iteration_summary(measured_iterations),
        "mg_levels": mg_levels,
        "mg_levels_by_phase": by_phase,
        "iteration_semantics": (
            "finest total_iterations is the outer GCR total; smoother and "
            "coarse iterations are separate per-level counters"),
        "iteration_kind": "outer_gcr",
        "iteration_scope": (
            "outer GCR iterations; recursive MG and coarse solver "
            "iterations are recorded separately in QUDA_MG_TRACE"),
        "converged_samples": (
            converged_list if phase_request["steady"] else
            ([bool(cold_result_payload["converged"])]
             if cold_result_payload is not None else []) +
            [bool(item["converged"]) for item in warmup_results]),
        "converged": all_converged,
        "true_residual": {
            "samples_rel": measured_residuals,
            "max_rel": (
                max(measured_residuals) if measured_residuals else math.inf),
            "gate": float(config["true_residual_gate"]),
            "pass": all_converged,
            "untimed_probe_rel": memory_probe.get("true_residual_rel"),
        },
        "krylov": {
            "requested_restart": int(config["restart_requested"]),
            "effective_restart": int(invert.gcrNkrylov),
            "max_krylov_bytes": None,
            "effective_workspace_bytes": None,
            "note": "QUDA native workspace is not exposed by PyQUDA",
        },
        "timing_boundary": timing_boundary,
        "quda_parameters": {
            "requested": expected_parameters,
            "actual": resolved_parameters,
            "mismatches": [],
        },
        "quda_input_contract": {
            "qdp_host_dtype": str(quda_qdp_host_dtype),
            "device_precision": quda_precision.name,
        },
        "memory": {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "setup": setup_memory,
            "first_solve": first_solve_memory,
            "steady": {
                "baseline": steady_baseline,
                "cuda_peak_allocated_bytes": max(
                    (sample["cuda_peak_allocated_bytes"]
                     for sample in solve_memory_samples), default=0),
                "cuda_peak_reserved_bytes": max(
                    (sample["cuda_peak_reserved_bytes"]
                     for sample in solve_memory_samples), default=0),
                "nvidia_smi_process_max_observed_bytes": _max_optional(
                    sample["nvidia_smi"].get("process_used_bytes")
                    for sample in solve_memory_samples),
                "nvidia_smi_gpu_max_observed_bytes": _max_optional(
                    sample["nvidia_smi"].get("gpu_used_bytes")
                    for sample in solve_memory_samples),
                "samples": solve_memory_samples,
                "snapshot_semantics": (
                    "nvidia-smi values are post-solve observations, not peaks"),
                "untimed_device_memory_probe": memory_probe,
            },
        },
        "provenance": provenance,
    }


def _worker_record(side: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    started = time.perf_counter()
    mpi_context = _worker_mpi_context(payload)
    os.environ[MPI_RANK_ENV] = str(mpi_context["rank"])
    os.environ[MPI_SIZE_ENV] = str(mpi_context["size"])
    os.environ[MPI_GRID_ENV] = ",".join(
        str(value) for value in mpi_context["process_grid"])
    trace_paths = _activate_rank_trace_paths(mpi_context)
    payload = copy.deepcopy(dict(payload))
    payload["mpi_context"] = mpi_context
    payload["trace_paths"] = trace_paths
    try:
        if side == "pyqcu":
            record = _run_pyqcu_worker(payload)
        elif side == "quda":
            record = _run_quda_worker(payload)
        else:
            raise BenchmarkFailure("unknown_side", side)
    except BenchmarkSkip as exc:
        record = {
            "side": side,
            "status": "skipped",
            "reason": {"code": exc.code, "detail": exc.detail},
            "config_hash": payload["protocol"]["config_hash"],
            "input_bundle_hash": (
                payload.get("input_fingerprints") or {}).get("bundle_hash"),
            "provenance": _benchmark_provenance(),
        }
    except BenchmarkUnsupported as exc:
        record = {
            "side": side,
            "status": "side_unsupported",
            "reason": {"code": exc.code, "detail": exc.detail},
            "config_hash": payload["protocol"]["config_hash"],
            "input_bundle_hash": (
                payload.get("input_fingerprints") or {}).get("bundle_hash"),
            "provenance": _benchmark_provenance(),
        }
        if exc.context:
            record.update(exc.context)
    except BenchmarkFailure as exc:
        record = {
            "side": side,
            "status": "failed",
            "reason": {"code": exc.code, "detail": exc.detail},
            "config_hash": payload["protocol"]["config_hash"],
            "input_bundle_hash": (
                payload.get("input_fingerprints") or {}).get("bundle_hash"),
            "provenance": _benchmark_provenance(),
        }
        if exc.context:
            record.update(exc.context)
    except BaseException as exc:  # worker must return an honest failure record
        record = {
            "side": side,
            "status": "failed",
            "reason": {
                "code": "unhandled_exception",
                "detail": repr(exc),
                "traceback_tail": _safe_tail(traceback.format_exc()),
            },
            "config_hash": payload["protocol"]["config_hash"],
            "input_bundle_hash": (
                payload.get("input_fingerprints") or {}).get("bundle_hash"),
            "provenance": _benchmark_provenance(),
        }
    record["worker_wall_seconds"] = time.perf_counter() - started
    record["completed_at"] = _utc_now()
    record["mpi"] = copy.deepcopy(mpi_context)
    record["trace_paths"] = copy.deepcopy(trace_paths)
    execution = payload.get("execution") or {}
    record["parallel_contract"] = copy.deepcopy(execution.get("parallel"))
    record["phase_contract"] = copy.deepcopy(execution.get("phases"))
    record["device_contract"] = copy.deepcopy(execution.get("device"))
    _write_rank_worker_record(record, mpi_context)
    return record


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _terminate_process_group(process: subprocess.Popen[str], grace_seconds: float = 2.0) -> None:
    """先 TERM 后 KILL 整个新 session，避免 benchmark 孙进程残留。"""
    process_group = process.pid
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    deadline = time.monotonic() + float(grace_seconds)
    while time.monotonic() < deadline:
        process.poll()  # reap the session leader when it exits
        if not _process_group_exists(process_group):
            break
        time.sleep(0.02)
    if _process_group_exists(process_group):
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"unable to terminate process group {process.pid}") from exc


def run_process_group(
        command: Sequence[str], timeout: float,
        *, env: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """运行独立 session，并在超时时终止完整进程组。"""
    started = time.perf_counter()
    process = subprocess.Popen(
        list(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=None if env is None else dict(env),
        start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=float(timeout))
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        partial_stdout = exc.stdout or ""
        partial_stderr = exc.stderr or ""
        if isinstance(partial_stdout, bytes):
            partial_stdout = partial_stdout.decode(errors="replace")
        if isinstance(partial_stderr, bytes):
            partial_stderr = partial_stderr.decode(errors="replace")
        _terminate_process_group(process)
        tail_stdout, tail_stderr = process.communicate()
        stdout = tail_stdout if tail_stdout else str(partial_stdout)
        stderr = tail_stderr if tail_stderr else str(partial_stderr)
        timed_out = True
    return {
        "command": list(command),
        "returncode": process.returncode,
        "timed_out": timed_out,
        "wall_seconds": time.perf_counter() - started,
        "stdout": stdout,
        "stderr": stderr,
    }


def _extract_worker_records(stdout: str) -> List[Dict[str, Any]]:
    # Native CUDA backends may append diagnostics to the same pipe after the
    # Python sentinel (and a few drivers can split writes at arbitrary byte
    # boundaries).  Do not require the sentinel to occupy an entire line or
    # the JSON object to be the last bytes in stdout.  raw_decode still
    # requires one complete JSON object, so unrelated log text cannot become a
    # successful worker record.
    decoder = json.JSONDecoder()
    positions: List[int] = []
    start = 0
    while True:
        position = stdout.find(WORKER_PREFIX, start)
        if position < 0:
            break
        positions.append(position)
        start = position + len(WORKER_PREFIX)
    records: List[Dict[str, Any]] = []
    for position in positions:
        payload = stdout[position + len(WORKER_PREFIX):].lstrip()
        try:
            value, _end = decoder.raw_decode(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _extract_worker_record(stdout: str) -> Optional[Dict[str, Any]]:
    records = _extract_worker_records(stdout)
    return records[-1] if records else None


def _write_rank_worker_record(
        record: Mapping[str, Any], mpi_context: Mapping[str, Any]) -> None:
    directory = os.environ.get(WORKER_RESULT_DIR_ENV)
    if not directory:
        return
    target = Path(directory).resolve() / (
        f"rank{int(mpi_context['rank']):05d}.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"),
                   allow_nan=False),
        encoding="utf-8")
    os.replace(temporary, target)


def _read_rank_worker_records(directory: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not directory.is_dir():
        return records
    for path in sorted(directory.glob("rank*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BenchmarkFailure(
                "mpi_rank_result_invalid", f"{path}: {exc!r}") from exc
        if not isinstance(value, dict):
            raise BenchmarkFailure(
                "mpi_rank_result_invalid", f"{path}: result is not an object")
        records.append(value)
    return records


def _worker_log_tail(stdout: str) -> str:
    return _safe_tail("\n".join(
        line for line in stdout.splitlines()
        if not line.startswith(WORKER_PREFIX)))


def _validate_cache_evidence(value: Any, label: str) -> List[str]:
    if not isinstance(value, Mapping):
        return [f"{label} missing"]
    errors: List[str] = []
    if value.get("schema") != "pyqcu.strict-runtime-cache":
        errors.append(f"{label}.schema invalid")
    if value.get("schema_version") != STRICT_CACHE_FORMAT_VERSION:
        errors.append(f"{label}.schema_version invalid")
    for key in (
            "identity_sha256", "metadata_sha256", "stats_sha256",
            "manifest_sha256"):
        if not _is_sha256(value.get(key)):
            errors.append(f"{label}.{key} invalid")
    for key in ("file_size_bytes", "tensor_count", "logical_bytes"):
        number = value.get(key)
        if (isinstance(number, bool) or not isinstance(number, int) or
                number < 0):
            errors.append(f"{label}.{key} invalid")
    tensor_digests = value.get("tensor_digests")
    if not isinstance(tensor_digests, Mapping):
        errors.append(f"{label}.tensor_digests missing")
    else:
        if value.get("tensor_count") != len(tensor_digests):
            errors.append(f"{label}.tensor_count does not match tensor_digests")
        logical_bytes = 0
        for path, tensor in tensor_digests.items():
            if not isinstance(path, str) or not isinstance(tensor, Mapping):
                errors.append(f"{label}.tensor_digests entry invalid")
                continue
            if not isinstance(tensor.get("digest_algorithm"), str):
                errors.append(f"{label}.{path}.digest_algorithm invalid")
            if not _is_sha256(tensor.get("sha256")):
                errors.append(f"{label}.{path}.sha256 invalid")
            nbytes = tensor.get("nbytes")
            if (isinstance(nbytes, bool) or not isinstance(nbytes, int) or
                    nbytes < 0):
                errors.append(f"{label}.{path}.nbytes invalid")
            else:
                logical_bytes += nbytes
        if isinstance(value.get("logical_bytes"), int) and \
                value.get("logical_bytes") != logical_bytes:
            errors.append(f"{label}.logical_bytes does not match tensors")
    if not isinstance(value.get("path"), str) or not value["path"]:
        errors.append(f"{label}.path invalid")
    return errors


def _manifest_contract_mismatches(
        observed: Any, expected: Mapping[str, Any]) -> List[str]:
    if not isinstance(observed, Mapping):
        return ["inspected manifest missing"]
    mismatches: List[str] = []
    for key in ("layout", "level_count", "tensor_count", "dtype", "total_bytes"):
        if observed.get(key) != expected.get(key):
            mismatches.append(
                f"manifest.{key}={observed.get(key)!r}, "
                f"expected={expected.get(key)!r}")
    observed_tensors = observed.get("tensors")
    expected_tensors = expected.get("tensors")
    if not isinstance(observed_tensors, Mapping) or not isinstance(expected_tensors, Mapping):
        mismatches.append("manifest.tensors missing")
    elif set(observed_tensors) != set(expected_tensors):
        mismatches.append("manifest tensor paths differ")
    else:
        for path, expected_spec in expected_tensors.items():
            actual_spec = observed_tensors[path]
            for key in ("shape", "dtype", "nbytes"):
                if actual_spec.get(key) != expected_spec.get(key):
                    mismatches.append(
                        f"manifest.tensors[{path!r}].{key}="
                        f"{actual_spec.get(key)!r}, expected={expected_spec.get(key)!r}")
    return mismatches


def _inspect_formal_runtime_cache(
        document: Mapping[str, Any], record: Mapping[str, Any]) -> Dict[str, Any]:
    """Re-verify a worker cache record without CUDA before formal merge."""
    cache = record.get("runtime_cache")
    if not isinstance(cache, Mapping):
        raise BenchmarkFailure("merge_cache_evidence_missing", "runtime_cache missing")
    fingerprints = document.get("input_fingerprints")
    if not isinstance(fingerprints, Mapping):
        raise BenchmarkFailure(
            "merge_input_fingerprint_missing", "formal merge has no input fingerprints")
    try:
        from pyqcu.cuda._strict_cache import inspect_strict_runtime_cache
        cache_directory = Path(
            document["execution"]["strict_cache"]["directory"]).resolve()
    except Exception as exc:
        raise BenchmarkFailure(
            "merge_cache_contract_invalid", repr(exc)) from exc

    rank_entries = cache.get("ranks")
    if not isinstance(rank_entries, list) or not rank_entries:
        rank_entries = [cache]
    inspected_rank_evidence: List[Dict[str, Any]] = []
    for entry in rank_entries:
        if not isinstance(entry, Mapping):
            raise BenchmarkFailure(
                "merge_cache_evidence_invalid", "rank cache entry is not a mapping")
        evidence = entry.get("evidence")
        evidence_errors = _validate_cache_evidence(
            evidence, "runtime_cache.evidence")
        if evidence_errors:
            raise BenchmarkFailure(
                "merge_cache_evidence_invalid", "; ".join(evidence_errors))
        rank = int(entry.get("rank", 0))
        context = {
            "rank": rank,
            "size": int(
                ((document.get("execution") or {}).get("parallel") or {}).get(
                    "ranks", 1)),
            "process_grid": (
                ((document.get("execution") or {}).get("parallel") or {}).get(
                    "process_grid") or [1, 1, 1, 1]),
        }
        context["rank_coordinate"] = list(
            _rank_coordinate(context["rank"], context["process_grid"]))
        try:
            identity = _strict_runtime_cache_identity({
                "protocol": document["protocol"],
                "input_fingerprints": fingerprints,
                "mpi_context": context,
            })
            expected_manifest = _strict_runtime_expected_manifest(
                document["protocol"], context)
            expected_path = _strict_runtime_cache_path(
                identity, cache_directory).resolve()
            recorded_path = Path(str(entry.get("path"))).resolve()
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise BenchmarkFailure(
                "merge_cache_contract_invalid", repr(exc)) from exc
        if recorded_path != expected_path:
            raise BenchmarkFailure(
                "merge_cache_path_mismatch",
                f"rank={rank} recorded={recorded_path}, expected={expected_path}")
        recorded_identity = entry.get("identity_sha256")
        if (recorded_identity is None and isinstance(evidence, Mapping)):
            recorded_identity = evidence.get("identity_sha256")
        expected_identity = _sha256_json(identity)
        if recorded_identity != expected_identity:
            raise BenchmarkFailure(
                "merge_cache_identity_mismatch",
                f"rank={rank} worker identity_sha256="
                f"{recorded_identity!r}, "
                f"expected={expected_identity!r}")
        try:
            inspected = inspect_strict_runtime_cache(
                recorded_path, identity=identity,
                expected_manifest=expected_manifest)
        except Exception as exc:
            raise BenchmarkFailure(
                "merge_cache_inspection_unavailable", repr(exc)) from exc
        if not inspected.hit:
            raise BenchmarkFailure(
                "merge_cache_inspection_failed",
                f"rank={rank} {recorded_path}: "
                f"{inspected.reason}: {inspected.detail}")
        inspected_evidence = inspected.evidence
        manifest_errors = _manifest_contract_mismatches(
            inspected.manifest, expected_manifest)
        if manifest_errors:
            raise BenchmarkFailure(
                "merge_cache_manifest_mismatch", "; ".join(manifest_errors))
        for key in (
                "path", "identity_sha256", "metadata_sha256", "stats_sha256",
                "manifest_sha256", "tensor_digests", "tensor_count",
                "logical_bytes", "file_size_bytes"):
            if evidence.get(key) != inspected_evidence.get(key):
                raise BenchmarkFailure(
                    "merge_cache_evidence_mismatch",
                    f"rank={rank} {key}: worker={evidence.get(key)!r}, "
                    f"current={inspected_evidence.get(key)!r}")
        inspected_rank_evidence.append(copy.deepcopy(dict(inspected_evidence)))
    result = copy.deepcopy(inspected_rank_evidence[0])
    if len(inspected_rank_evidence) > 1:
        result["rank_evidence"] = inspected_rank_evidence
    return result


def _aggregate_mg_levels(
        level_sets: Sequence[Sequence[Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    if not level_sets:
        return []
    counts = {len(levels) for levels in level_sets}
    if len(counts) != 1:
        raise BenchmarkFailure(
            "rank_mg_level_count_mismatch", f"level counts={sorted(counts)}")
    result: List[Dict[str, Any]] = []
    for level_index in range(next(iter(counts))):
        levels = [levels[level_index] for levels in level_sets]
        total = max(float(item["total_seconds"]) for item in levels)
        # Keep one rank's stage breakdown intact.  Taking an independent
        # maximum for every stage can assemble a sum larger than the slowest
        # rank's total when different ranks are slowest in different stages.
        timing_rank_index = max(
            range(len(levels)),
            key=lambda index: float(levels[index]["total_seconds"]))
        timing_rank = levels[timing_rank_index]
        merged = {
            "level": int(levels[0]["level"]),
            "total_seconds": total,
            "total_iterations": max(
                int(item["total_iterations"]) for item in levels),
            "pre_smoother_seconds": float(
                timing_rank["pre_smoother_seconds"]),
            "post_smoother_seconds": float(
                timing_rank["post_smoother_seconds"]),
            "restriction_seconds": float(
                timing_rank["restriction_seconds"]),
            "prolongation_seconds": float(
                timing_rank["prolongation_seconds"]),
            "coarse_solver_seconds": float(
                timing_rank["coarse_solver_seconds"]),
            "timing_rank_index": int(timing_rank_index),
        }
        known = sum(float(merged[key]) for key in (
            "pre_smoother_seconds", "post_smoother_seconds",
            "restriction_seconds", "prolongation_seconds",
            "coarse_solver_seconds"))
        other = total - known
        tolerance = max(1.0e-6, 1.0e-2 * max(1.0, abs(total), abs(known)))
        if other < -tolerance:
            raise BenchmarkFailure(
                "rank_mg_level_timing_negative_other",
                f"level={merged['level']} total={total}, known={known}")
        merged["other_seconds"] = max(0.0, other)
        for key in (
                "pre_smoother_iterations", "post_smoother_iterations",
                "restriction_calls", "prolongation_calls",
                "recursive_solve_calls", "coarse_solver_iterations",
                "coarse_solver_calls"):
            merged[key] = max(int(item.get(key, 0)) for item in levels)
        merged["smoother_iterations"] = max(
            int(item.get(
                "smoother_iterations",
                int(item.get("pre_smoother_iterations", 0)) +
                int(item.get("post_smoother_iterations", 0))))
            for item in levels)
        if level_index == 0:
            merged["finest_iterations"] = int(
                levels[0].get("finest_iterations",
                              merged["total_iterations"]))
            merged["residual_sequence"] = copy.deepcopy(
                levels[0].get("residual_sequence", []))
            final_values = [
                float(item["final_residual"]) for item in levels
                if item.get("final_residual") is not None]
            merged["final_residual"] = (
                max(final_values) if final_values else None)
            merged["rank_final_residuals"] = [
                item.get("final_residual") for item in levels]
        result.append(merged)
    return result


def _aggregate_side_records(
        side: str, records: Sequence[Mapping[str, Any]],
        mpi_context: Mapping[str, Any]) -> Dict[str, Any]:
    expected_ranks = int(mpi_context["size"])
    if len(records) != expected_ranks:
        raise BenchmarkFailure(
            "mpi_worker_record_count",
            f"{side}: expected {expected_ranks} rank records, got {len(records)}")
    ranked: List[Mapping[str, Any]] = []
    seen = set()
    for record in records:
        mpi = record.get("mpi") or {}
        rank = int(mpi.get("rank", len(ranked)))
        if rank in seen:
            raise BenchmarkFailure("mpi_duplicate_rank", f"{side}: rank={rank}")
        seen.add(rank)
        ranked.append(record)
    ranked.sort(key=lambda item: int((item.get("mpi") or {}).get("rank", 0)))
    if seen != set(range(expected_ranks)):
        raise BenchmarkFailure(
            "mpi_rank_set_mismatch",
            f"{side}: ranks={sorted(seen)}, expected={list(range(expected_ranks))}")

    statuses = [str(record.get("status")) for record in ranked]
    summary_ranks = [{
        "rank": int((record.get("mpi") or {}).get("rank", 0)),
        "status": record.get("status"),
        "reason": copy.deepcopy(record.get("reason")),
        "trace_paths": copy.deepcopy(record.get("trace_paths") or {}),
        "device": copy.deepcopy(
            ((record.get("provenance") or {}).get("runtime") or {})),
    } for record in ranked]
    base = copy.deepcopy(dict(ranked[0]))
    base["rank_results"] = summary_ranks
    base["mpi"] = copy.deepcopy(dict(mpi_context))
    base["rank_aggregation"] = {
        "mode": "rank-wise maximum for wall times/bytes/iterations; "
                "element-wise conjunction for convergence",
        "ranks": expected_ranks,
        "process_grid": list(mpi_context["process_grid"]),
    }
    if "side_unsupported" in statuses:
        base["status"] = "side_unsupported"
        base["reason"] = next(
            (copy.deepcopy(record.get("reason")) for record in ranked
             if record.get("status") == "side_unsupported"),
            {"code": "side_unsupported", "detail": side})
        return base
    if "timeout" in statuses:
        base["status"] = "timeout"
        base["reason"] = next(
            copy.deepcopy(record.get("reason")) for record in ranked
            if record.get("status") == "timeout")
        return base
    if "failed" in statuses:
        base["status"] = "failed"
        base["reason"] = next(
            copy.deepcopy(record.get("reason")) for record in ranked
            if record.get("status") == "failed")
        return base
    if "skipped" in statuses:
        base["status"] = "skipped"
        base["reason"] = next(
            copy.deepcopy(record.get("reason")) for record in ranked
            if record.get("status") == "skipped")
        return base

    timing_sources = [record["timing"] for record in ranked]
    merged_timing = copy.deepcopy(dict(timing_sources[0]))
    for key in (
            "input_io_seconds", "runtime_init_seconds", "setup_seconds",
            "setup_noncache_seconds", "cache_restore_seconds",
            "cache_miss_probe_seconds", "cache_persist_seconds",
            "cold_seconds"):
        values = [
            float(source[key]) for source in timing_sources
            if source.get(key) is not None]
        if values:
            merged_timing[key] = max(values)
    warmup_sources = [source.get("warmups") or [] for source in timing_sources]
    warmup_count = min((len(values) for values in warmup_sources), default=0)
    merged_timing["warmups"] = []
    for index in range(warmup_count):
        items = [values[index] for values in warmup_sources]
        merged_timing["warmups"].append({
            "seconds": max(float(item["seconds"]) for item in items),
            "iterations": max(int(item["iterations"]) for item in items),
            "converged": all(bool(item["converged"]) for item in items),
            "true_residual_rel": max(
                float(item["true_residual_rel"]) for item in items),
        })
    steady_sources = [source.get("steady") or {} for source in timing_sources]
    steady_samples = [
        [float(value) for value in source.get("samples_seconds", [])]
        for source in steady_sources]
    steady_count = min((len(values) for values in steady_samples), default=0)
    merged_timing["steady"] = (
        _median_mad([
            max(values[index] for values in steady_samples)
            for index in range(steady_count)])
        if steady_count else
        {"samples_seconds": [], "median_seconds": None,
         "mad_seconds": None})

    iteration_sources = [
        [int(value) for value in record["iterations"]["samples"]]
        for record in ranked]
    iteration_count = min(
        (len(values) for values in iteration_sources), default=0)
    base["iterations"] = _iteration_summary([
        max(values[index] for values in iteration_sources)
        for index in range(iteration_count)])
    residual_sources = [
        [float(value) for value in record["true_residual"]["samples_rel"]]
        for record in ranked]
    residual_count = min(
        (len(values) for values in residual_sources), default=0)
    residual_samples = [
        max(values[index] for values in residual_sources)
        for index in range(residual_count)]
    base["converged_samples"] = [
        all(bool(record["converged_samples"][index]) for record in ranked)
        for index in range(iteration_count)]
    base["converged"] = all(bool(record["converged"]) for record in ranked)
    base["true_residual"].update({
        "samples_rel": residual_samples,
        "max_rel": max(residual_samples) if residual_samples else math.inf,
        "pass": base["converged"],
    })
    base["timing"] = merged_timing

    level_sets = [record.get("mg_levels") for record in ranked]
    if all(isinstance(value, list) for value in level_sets):
        base["mg_levels"] = _aggregate_mg_levels(level_sets)
    phase_sets: Dict[str, List[List[Mapping[str, Any]]]] = {}
    for record in ranked:
        by_phase = record.get("mg_levels_by_phase") or {}
        for phase, levels in by_phase.items():
            if isinstance(levels, list):
                phase_sets.setdefault(phase, []).append(levels)
    if phase_sets:
        base["mg_levels_by_phase"] = {
            phase: _aggregate_mg_levels(level_sets)
            for phase, level_sets in phase_sets.items()
            if len(level_sets) == expected_ranks
        }
    if side == "pyqcu":
        caches = [record.get("runtime_cache") for record in ranked]
        if all(isinstance(value, Mapping) for value in caches):
            base["runtime_cache"] = copy.deepcopy(dict(caches[0]))
            base["runtime_cache"]["ranks"] = [
                {
                    "rank": int((record.get("mpi") or {}).get("rank", 0)),
                    "path": cache.get("path"),
                    "hit": cache.get("hit"),
                    "expectation": cache.get("expectation"),
                    "evidence": copy.deepcopy(cache.get("evidence")),
                }
                for record, cache in zip(ranked, caches)
            ]
            base["runtime_cache"]["hit"] = all(
                bool(cache.get("hit")) for cache in caches)
            expectations = {cache.get("expectation") for cache in caches}
            base["runtime_cache"]["expectation"] = (
                next(iter(expectations)) if len(expectations) == 1 else "mixed")
    memories = [record.get("memory") for record in ranked]
    if all(isinstance(value, Mapping) for value in memories):
        base["memory_rank_summary"] = {
            "setup_peak_allocated_bytes": _max_optional(
                (value.get("setup") or {}).get("cuda_peak_allocated_bytes")
                for value in memories),
            "setup_peak_reserved_bytes": _max_optional(
                (value.get("setup") or {}).get("cuda_peak_reserved_bytes")
                for value in memories),
            "steady_peak_allocated_bytes": _max_optional(
                (value.get("steady") or {}).get("cuda_peak_allocated_bytes")
                for value in memories),
            "steady_peak_reserved_bytes": _max_optional(
                (value.get("steady") or {}).get("cuda_peak_reserved_bytes")
                for value in memories),
            "per_rank": [
                {
                    "rank": int((record.get("mpi") or {}).get("rank", 0)),
                    "setup_peak_allocated_bytes": (
                        (record.get("memory") or {}).get("setup") or {}
                    ).get("cuda_peak_allocated_bytes"),
                    "steady_peak_allocated_bytes": (
                        (record.get("memory") or {}).get("steady") or {}
                    ).get("cuda_peak_allocated_bytes"),
                }
                for record in ranked
            ],
        }
    provenance = copy.deepcopy(base.get("provenance") or {})
    provenance["runtime_ranks"] = [
        copy.deepcopy(((record.get("provenance") or {}).get("runtime") or {}))
        for record in ranked]
    base["provenance"] = provenance
    return base


def _phase_selection(document: Mapping[str, Any]) -> Tuple[str, ...]:
    execution = document.get("execution") or {}
    phases = (execution.get("phases") or {}).get("selected")
    return _normalise_phases(phases or DEFAULT_PHASES)


def _combine_phase_records(
        cold: Mapping[str, Any], warm_steady: Mapping[str, Any]) -> Dict[str, Any]:
    for record in (cold, warm_steady):
        if record.get("status") != "ok":
            return copy.deepcopy(dict(record))
    record = copy.deepcopy(dict(warm_steady))
    cold_timing = cold.get("timing") or {}
    cold_seconds = float(cold_timing.get("cold_seconds", 0.0))
    record["timing"]["cold_seconds"] = cold_seconds
    record["timing"]["cold_setup_seconds"] = cold_timing.get("setup_seconds")
    record["timing"]["cold_cache_persist_seconds"] = cold_timing.get(
        "cache_persist_seconds")
    record["iterations_cold"] = copy.deepcopy(cold.get("iterations"))
    record["converged_cold"] = bool(cold.get("converged"))
    record["true_residual_cold"] = copy.deepcopy(cold.get("true_residual"))
    record["memory_cold"] = copy.deepcopy(cold.get("memory"))
    record["phase_results"] = {
        "cold": {
            "seconds": cold_seconds,
            "iterations": copy.deepcopy(cold.get("iterations")),
            "converged": bool(cold.get("converged")),
        },
        "warmup_count": len(record.get("timing", {}).get("warmups") or []),
        "steady_count": len(
            (record.get("timing", {}).get("steady") or {}).get(
                "samples_seconds", [])),
    }
    by_phase = copy.deepcopy(record.get("mg_levels_by_phase") or {})
    by_phase["cold"] = copy.deepcopy(cold.get("mg_levels"))
    record["mg_levels_by_phase"] = by_phase
    if record.get("runtime_cache") is not None:
        record["runtime_cache"]["cold"] = copy.deepcopy(
            cold.get("runtime_cache"))
        record["runtime_cache"]["steady"] = copy.deepcopy(
            warm_steady.get("runtime_cache"))
    return record


def _launch_side_once(
        side: str, document: Mapping[str, Any], timeout: float,
        phase_request: Mapping[str, Any],
        phase_tag: Optional[str] = None) -> Dict[str, Any]:
    payload = {
        "protocol": document["protocol"],
        "inputs": document["inputs"],
        "execution": document["execution"],
        "input_fingerprints": document["input_fingerprints"],
        "collector_git": document["collector"]["git"],
        "phase_request": copy.deepcopy(dict(phase_request)),
    }
    encoded = base64.b64encode(_json_bytes(payload)).decode("ascii")
    env = os.environ.copy()
    env[WORKER_PAYLOAD_ENV] = encoded
    parallel = (document.get("execution") or {}).get("parallel") or {}
    ranks = int(parallel.get("ranks", 1))
    grid = _normalise_process_grid(parallel.get("process_grid"), ranks)
    env[MPI_SIZE_ENV] = str(ranks)
    env[MPI_GRID_ENV] = ",".join(str(value) for value in grid)
    if phase_tag:
        for name in (
                "PYQCU_STRICT_TRACE_FILE", "QUDA_MG_TRACE_FILE",
                "PYQCU_QUDA_TRACE_FILE"):
            if env.get(name):
                env[name] = f"{env[name]}.{phase_tag}"
    rank_result_dir: Optional[tempfile.TemporaryDirectory[str]] = None
    if ranks > 1:
        launcher = shutil.which("mpirun") or shutil.which("mpiexec")
        if launcher is None:
            raise BenchmarkFailure(
                "mpi_launcher_missing",
                "mpirun/mpiexec is required when --mpi-ranks > 1")
        command = [
            launcher, "-np", str(ranks),
            sys.executable, str(Path(__file__).resolve()),
            "--_worker", side,
        ]
        rank_result_dir = tempfile.TemporaryDirectory(
            prefix=f"pyqcu-strict-{side}-ranks-")
        env[WORKER_RESULT_DIR_ENV] = rank_result_dir.name
    else:
        command = [sys.executable, str(Path(__file__).resolve()), "--_worker", side]
    try:
        process = run_process_group(command, timeout, env=env)
        rank_records = (
            _read_rank_worker_records(Path(rank_result_dir.name))
            if rank_result_dir is not None else [])
    finally:
        if rank_result_dir is not None:
            rank_result_dir.cleanup()
    if process["timed_out"]:
        return {
            "side": side,
            "status": "timeout",
            "reason": {
                "code": "side_timeout",
                "detail": f"exceeded {timeout:.3f} seconds; process group terminated",
            },
            "config_hash": document["protocol"]["config_hash"],
            "input_bundle_hash": document["input_fingerprints"]["bundle_hash"],
            "process": {
                "returncode": process["returncode"],
                "wall_seconds": process["wall_seconds"],
                "stdout_tail": _safe_tail(process["stdout"]),
                "stderr_tail": _safe_tail(process["stderr"]),
            },
            "completed_at": _utc_now(),
        }
    records = (
        rank_records if len(rank_records) == ranks else
        _extract_worker_records(process["stdout"]))
    if not records:
        return {
            "side": side,
            "status": "failed",
            "reason": {
                "code": "worker_protocol_missing",
                "detail": f"returncode={process['returncode']}; no result sentinel",
            },
            "config_hash": document["protocol"]["config_hash"],
            "input_bundle_hash": document["input_fingerprints"]["bundle_hash"],
            "process": {
                "returncode": process["returncode"],
                "wall_seconds": process["wall_seconds"],
                "stdout_tail": _safe_tail(process["stdout"]),
                "stderr_tail": _safe_tail(process["stderr"]),
            },
            "completed_at": _utc_now(),
        }
    if ranks > 1:
        record = _aggregate_side_records(
            side, records,
            {
                "rank": 0,
                "size": ranks,
                "process_grid": list(grid),
                "rank_coordinate": [0, 0, 0, 0],
            })
    else:
        record = copy.deepcopy(records[-1])
    record["process"] = {
        "returncode": process["returncode"],
        "wall_seconds": process["wall_seconds"],
        "stdout_tail": _worker_log_tail(process["stdout"]),
        "stderr_tail": _safe_tail(process["stderr"]),
    }
    if process["returncode"] != 0 and record.get("status") == "ok":
        record["status"] = "failed"
        record["reason"] = {
            "code": "worker_nonzero_after_success",
            "detail": f"returncode={process['returncode']}",
        }
    return record


def _launch_side(side: str, document: Mapping[str, Any], timeout: float) -> Dict[str, Any]:
    phases = _phase_selection(document)
    split = (
        "cold" in phases and
        any(phase in phases for phase in ("warmup", "steady")))
    cache_expect = (
        (document.get("execution") or {}).get("strict_cache") or {}
    ).get("expect", "any")
    if split:
        cold_cache_expect = "hit" if cache_expect == "hit" else "miss"
        cold = _launch_side_once(
            side, document, timeout,
            {
                "cold": True,
                "warmup": False,
                "steady": False,
                "cache_expect": cold_cache_expect,
            },
            phase_tag="cold",
        )
        main_cache_expect = (
            "hit" if document["protocol"].get("profile") == "formal"
            else ("any" if cache_expect == "miss" else cache_expect))
        warm_steady = _launch_side_once(
            side, document, timeout,
            {
                "cold": False,
                "warmup": "warmup" in phases,
                "steady": "steady" in phases,
                "cache_expect": main_cache_expect,
            },
            phase_tag="warmup_steady",
        )
        return _combine_phase_records(cold, warm_steady)
    return _launch_side_once(
        side, document, timeout,
        {
            "cold": "cold" in phases,
            "warmup": "warmup" in phases,
            "steady": "steady" in phases,
            "cache_expect": cache_expect,
        })


def _comparison(document: Mapping[str, Any]) -> Dict[str, Any]:
    profile = document["protocol"].get("profile")
    sides = document["sides"]
    left = sides.get("pyqcu", {})
    right = sides.get("quda", {})
    reasons: List[str] = []
    if left.get("status") != "ok":
        reasons.append(f"pyqcu status={left.get('status')}")
    if right.get("status") != "ok":
        reasons.append(f"quda status={right.get('status')}")
    expected_config = document["protocol"]["config_hash"]
    expected_input = (document.get("input_fingerprints") or {}).get("bundle_hash")
    for side, record in (("pyqcu", left), ("quda", right)):
        if record.get("status") == "ok" and record.get("config_hash") != expected_config:
            reasons.append(f"{side} config hash mismatch")
        if record.get("status") == "ok" and record.get("input_bundle_hash") != expected_input:
            reasons.append(f"{side} input bundle hash mismatch")
        true_residual = record.get("true_residual") or {}
        if record.get("status") == "ok" and not true_residual.get("pass"):
            reasons.append(f"{side} true residual gate failed")
        timing = ((record.get("timing") or {}).get("steady") or {})
        if record.get("status") == "ok" and not isinstance(timing.get("median_seconds"), (int, float)):
            reasons.append(f"{side} steady median missing")

    if profile == "smoke":
        # Smoke 只回答“两个后端是否按本次探索配置跑通”；它不构成公平
        # 性能证据，因此不要求 cache hit、同一 GPU UUID，也绝不产生 speedup。
        for error in validate_document(document, allow_planned=True):
            reasons.append(f"document evidence invalid: {error}")
        reasons = list(dict.fromkeys(reasons))
        if reasons:
            terminal = all(
                sides.get(side, {}).get("status") in TERMINAL_STATUSES
                for side in SIDE_NAMES)
            return {
                "status": "smoke-unavailable" if terminal else "smoke-pending",
                "profile": "smoke",
                "fair": False,
                "reasons": reasons,
                "speedup_pyqcu_over_quda": None,
            }
        pyqcu_median = float(left["timing"]["steady"]["median_seconds"])
        quda_median = float(right["timing"]["steady"]["median_seconds"])
        if pyqcu_median <= 0.0 or quda_median <= 0.0:
            return {
                "status": "smoke-failed",
                "profile": "smoke",
                "fair": False,
                "reasons": ["non-positive timing median"],
                "speedup_pyqcu_over_quda": None,
            }
        return {
            "status": "smoke-pass",
            "profile": "smoke",
            "fair": False,
            "reasons": [],
            "speedup_pyqcu_over_quda": None,
            "pyqcu_median_seconds": pyqcu_median,
            "quda_median_seconds": quda_median,
        }

    if left.get("status") == "ok":
        cache = left.get("runtime_cache")
        if not isinstance(cache, Mapping):
            reasons.append("pyqcu runtime cache evidence missing")
        else:
            if cache.get("expectation") != "hit":
                reasons.append("pyqcu formal record was not requested as cache hit")
            if cache.get("hit") is not True:
                reasons.append("pyqcu formal runtime cache was not a verified hit")
    if left.get("status") == "ok" and right.get("status") == "ok":
        runtime_left = ((left.get("provenance") or {}).get("runtime") or {})
        runtime_right = ((right.get("provenance") or {}).get("runtime") or {})
        ranks_left = (
            (left.get("provenance") or {}).get("runtime_ranks") or
            [runtime_left])
        ranks_right = (
            (right.get("provenance") or {}).get("runtime_ranks") or
            [runtime_right])
        uuid_left = [item.get("device_uuid") for item in ranks_left]
        uuid_right = [item.get("device_uuid") for item in ranks_right]
        if (not uuid_left or not uuid_right or
                not all(isinstance(value, str) and value
                        for value in uuid_left + uuid_right)):
            reasons.append("backend GPU UUID evidence missing")
        elif len(uuid_left) != len(uuid_right):
            reasons.append(
                f"backend GPU rank count mismatch: "
                f"pyqcu={len(uuid_left)}, quda={len(uuid_right)}")
        elif uuid_left != uuid_right:
            reasons.append(
                f"backend GPU UUID mismatch by rank: "
                f"pyqcu={uuid_left}, quda={uuid_right}")
    for error in validate_document(document, allow_planned=True):
        reasons.append(f"document evidence invalid: {error}")
    reasons = list(dict.fromkeys(reasons))
    if reasons:
        terminal = all(sides.get(side, {}).get("status") in TERMINAL_STATUSES
                       for side in SIDE_NAMES)
        return {
            "status": "unavailable" if terminal else "pending",
            "profile": profile,
            "fair": False if terminal else None,
            "reasons": reasons,
            "speedup_pyqcu_over_quda": None,
        }
    pyqcu_median = float(left["timing"]["steady"]["median_seconds"])
    quda_median = float(right["timing"]["steady"]["median_seconds"])
    if pyqcu_median <= 0.0 or quda_median <= 0.0:
        return {
            "status": "failed",
            "profile": profile,
            "fair": False,
            "reasons": ["non-positive timing median"],
            "speedup_pyqcu_over_quda": None,
        }
    # >1 表示 PyQCU 更快：QUDA wall / PyQCU wall。
    return {
        "status": "pass",
        "profile": profile,
        "fair": True,
        "reasons": [],
        "speedup_pyqcu_over_quda": quda_median / pyqcu_median,
        "pyqcu_median_seconds": pyqcu_median,
        "quda_median_seconds": quda_median,
    }


def _update_state(document: MutableMapping[str, Any]) -> None:
    document["comparison"] = _comparison(document)
    selected = document["selected_sides"]
    terminal = all(document["sides"][side]["status"] in TERMINAL_STATUSES
                   for side in selected)
    both_ok = all(document["sides"][side]["status"] == "ok" for side in SIDE_NAMES)
    smoke_pass = (
        document["protocol"].get("profile") == "smoke" and
        document["comparison"].get("status") == "smoke-pass")
    if both_ok and (
            document["comparison"].get("fair") is True or smoke_pass):
        document["state"] = "complete"
    elif terminal and len(selected) == 2:
        document["state"] = "complete-unavailable"
    else:
        document["state"] = "partial"
    document["updated_at"] = _utc_now()


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _emit(document: Mapping[str, Any], output: Optional[str]) -> None:
    if output:
        _atomic_write(Path(output), document)
        print(f"[result] {Path(output).resolve()}", file=sys.stderr)
    else:
        print(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False))


def _load_document(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkFailure("invalid_result_json", f"{path}: {exc!r}") from exc
    if not isinstance(value, dict):
        raise BenchmarkFailure("invalid_result_json", f"{path}: root is not an object")
    errors = validate_document(value, allow_planned=True)
    if errors:
        raise BenchmarkFailure("invalid_result_schema", f"{path}: {'; '.join(errors)}")
    return value


def _compatible(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    if left["schema"] != right["schema"]:
        raise BenchmarkFailure("merge_schema_mismatch", repr((left["schema"], right["schema"])))
    if left["protocol"]["config_hash"] != right["protocol"]["config_hash"]:
        raise BenchmarkFailure("merge_config_mismatch", "config_hash differs")
    left_fp = left.get("input_fingerprints")
    right_fp = right.get("input_fingerprints")
    if left_fp and right_fp:
        mismatches = _input_fingerprint_mismatches(left_fp, right_fp)
        if mismatches:
            raise BenchmarkFailure(
                "merge_input_mismatch", "; ".join(mismatches))


def _resume_side_compatible(
        side: str, record: Mapping[str, Any],
        requested_execution: Mapping[str, Any],
        expected_input_bundle_hash: Optional[str] = None) -> bool:
    """Return whether a successful side record satisfies this resume request."""
    if record.get("status") != "ok":
        return False
    if (expected_input_bundle_hash is not None and
            record.get("input_bundle_hash") != expected_input_bundle_hash):
        return False
    for key, record_key in (
            ("parallel", "parallel_contract"),
            ("phases", "phase_contract"),
            ("device", "device_contract")):
        recorded_contract = record.get(record_key)
        requested_contract = requested_execution.get(key)
        if recorded_contract is not None and recorded_contract != requested_contract:
            return False
    if side != "pyqcu":
        return True
    policy = requested_execution.get("strict_cache")
    cache = record.get("runtime_cache")
    if not isinstance(policy, Mapping) or not isinstance(cache, Mapping):
        return False
    requested_expectation = policy.get("expect")
    if requested_expectation not in ("any", "miss", "hit"):
        return False
    if requested_expectation != "any" and cache.get("expectation") != requested_expectation:
        return False
    if requested_expectation == "hit" and cache.get("hit") is not True:
        return False
    if requested_expectation == "miss" and cache.get("hit") is not False:
        return False
    path = cache.get("path")
    directory = policy.get("directory")
    if not isinstance(path, str) or not isinstance(directory, str):
        return False
    try:
        return Path(path).resolve().parent == Path(directory).resolve()
    except OSError:
        return False


def merge_documents(paths: Sequence[str]) -> Dict[str, Any]:
    if not paths:
        raise ValueError("--merge requires at least one input")
    documents = [_load_document(Path(path)) for path in paths]
    merged = json.loads(json.dumps(documents[0]))
    merged["selected_sides"] = []
    pyqcu_execution_source: Optional[Dict[str, Any]] = None
    quda_qio_source: Optional[Dict[str, Any]] = None
    for document in documents:
        _compatible(merged, document)
        if merged.get("input_fingerprints") is None and document.get("input_fingerprints"):
            merged["input_fingerprints"] = document["input_fingerprints"]
        for side in SIDE_NAMES:
            candidate = document["sides"].get(side, {})
            status = candidate.get("status")
            if status in TERMINAL_STATUSES and side == "pyqcu":
                # The cache policy is a PyQCU-side execution contract.  A
                # QUDA-only document carries the parser default (usually
                # ``any``), so choose the policy from the actual PyQCU side
                # regardless of input-document order.
                execution = document.get("execution")
                if isinstance(execution, Mapping):
                    execution_copy = copy.deepcopy(dict(execution))
                    if (pyqcu_execution_source is not None and
                            pyqcu_execution_source != execution_copy):
                        raise BenchmarkFailure(
                            "merge_execution_mismatch",
                            "conflicting PyQCU execution/cache contracts")
                    pyqcu_execution_source = execution_copy
            if status in TERMINAL_STATUSES and side == "quda":
                # QIO input metadata belongs to the QUDA side.  Preserve it
                # when merging a QUDA-only record into a PyQCU-first document;
                # otherwise formal validation loses the prefix and rebuilds a
                # different expected vec_infile contract.
                inputs = document.get("inputs")
                qio = inputs.get("quda_qio") if isinstance(inputs, Mapping) else None
                if isinstance(qio, Mapping):
                    has_qio = bool(qio.get("prefix") or qio.get("conversion_manifest"))
                    if has_qio:
                        qio_copy = copy.deepcopy(dict(qio))
                        if (quda_qio_source is not None and
                                quda_qio_source != qio_copy):
                            raise BenchmarkFailure(
                                "merge_quda_qio_mismatch",
                                "conflicting QUDA QIO input contracts")
                        quda_qio_source = qio_copy
            if status in TERMINAL_STATUSES:
                existing = merged["sides"].get(side, {})
                if existing.get("status") == "ok" and status == "ok":
                    if _sha256_json(existing) != _sha256_json(candidate):
                        raise BenchmarkFailure(
                            "merge_duplicate_success", f"conflicting successful {side} records")
                else:
                    merged["sides"][side] = candidate
            if side in document.get("selected_sides", []):
                merged["selected_sides"].append(side)
    if pyqcu_execution_source is not None:
        merged["execution"] = pyqcu_execution_source
    if quda_qio_source is not None:
        inputs = merged.setdefault("inputs", {})
        inputs["quda_qio"] = quda_qio_source
    merged["selected_sides"] = sorted(set(merged["selected_sides"]))
    if set(merged["selected_sides"]) == set(SIDE_NAMES):
        merged["selected_sides"] = list(SIDE_NAMES)
    merged["merged_from"] = [str(Path(path).resolve()) for path in paths]
    _update_state(merged)
    if (merged["protocol"].get("profile") == "formal" and
            merged["sides"].get("pyqcu", {}).get("status") == "ok"):
        inspected_evidence = _inspect_formal_runtime_cache(
            merged, merged["sides"]["pyqcu"])
        merged["sides"]["pyqcu"]["runtime_cache"]["merge_inspection"] = {
            "method": "inspect_strict_runtime_cache",
            "device_transfer": False,
            "verified": True,
            "evidence": inspected_evidence,
        }
        _update_state(merged)
    errors = validate_document(merged, allow_planned=True)
    if errors:
        raise BenchmarkFailure("merged_schema_invalid", "; ".join(errors))
    return merged


def _validate_provenance_contract(
        value: Any, label: str, *, require_quda_build: bool = False) -> List[str]:
    """Validate the split provenance fields emitted by a successful worker."""
    if not isinstance(value, Mapping):
        return [f"{label} missing"]
    errors: List[str] = []
    for key in ("pyqcu_git", "quda_source_git", "quda_libraries",
                "pyquda_module", "cmake_features", "patch_variant"):
        if not isinstance(value.get(key), Mapping):
            errors.append(f"{label}.{key} missing")
    runtime = value.get("runtime")
    if not isinstance(runtime, Mapping):
        errors.append(f"{label}.runtime missing")
    elif not isinstance(runtime.get("device_uuid"), str) or not runtime["device_uuid"]:
        errors.append(f"{label}.runtime.device_uuid missing")
    runtime_ranks = value.get("runtime_ranks")
    if runtime_ranks is not None:
        if not isinstance(runtime_ranks, list) or not runtime_ranks:
            errors.append(f"{label}.runtime_ranks invalid")
        else:
            for rank, item in enumerate(runtime_ranks):
                if (not isinstance(item, Mapping) or
                        not isinstance(item.get("device_uuid"), str) or
                        not item["device_uuid"]):
                    errors.append(
                        f"{label}.runtime_ranks[{rank}].device_uuid missing")
    if isinstance(value.get("pyqcu_git"), Mapping):
        if not isinstance(value["pyqcu_git"].get("repository"), str):
            errors.append(f"{label}.pyqcu_git.repository missing")
    if isinstance(value.get("quda_source_git"), Mapping):
        if not isinstance(value["quda_source_git"].get("repository"), str):
            errors.append(f"{label}.quda_source_git.repository missing")
    if isinstance(value.get("patch_variant"), Mapping):
        patch = value["patch_variant"]
        for key in ("name", "wsl2", "environment_scoped", "limitation"):
            if key not in patch:
                errors.append(f"{label}.patch_variant.{key} missing")
    if not require_quda_build:
        return errors

    libraries = value.get("quda_libraries")
    if isinstance(libraries, Mapping):
        for name in ("libquda", "libqmp"):
            library = libraries.get(name)
            if not isinstance(library, Mapping):
                errors.append(f"{label}.quda_libraries.{name} missing")
                continue
            if (not isinstance(library.get("path"), str) or
                    not library["path"]):
                errors.append(f"{label}.quda_libraries.{name}.path missing")
            if not _is_sha256(library.get("sha256")):
                errors.append(f"{label}.quda_libraries.{name}.sha256 invalid")
            if library.get("exists") is not True:
                errors.append(f"{label}.quda_libraries.{name}.exists is not true")
    module = value.get("pyquda_module")
    if isinstance(module, Mapping) and (
            not isinstance(module.get("path"), str) or not module["path"]):
        errors.append(f"{label}.pyquda_module.path missing")
    cmake = value.get("cmake_features")
    if isinstance(cmake, Mapping):
        if (not isinstance(cmake.get("cache_path"), str) or
                not cmake["cache_path"]):
            errors.append(f"{label}.cmake_features.cache_path missing")
        if not _is_sha256(cmake.get("cache_sha256")):
            errors.append(f"{label}.cmake_features.cache_sha256 invalid")
    return errors


def _validate_mg_levels(value: Any, level_count: int, side: str) -> List[str]:
    errors: List[str] = []
    if not isinstance(value, list) or len(value) != int(level_count):
        return [f"{side} mg_levels must contain {int(level_count)} entries"]
    time_fields = (
        "total_seconds", "pre_smoother_seconds", "post_smoother_seconds",
        "restriction_seconds", "prolongation_seconds",
        "coarse_solver_seconds", "other_seconds")
    iteration_fields = (
        "total_iterations", "pre_smoother_iterations",
        "post_smoother_iterations", "smoother_iterations",
        "restriction_calls",
        "prolongation_calls", "recursive_solve_calls",
        "coarse_solver_iterations", "coarse_solver_calls")
    finest_total: Optional[int] = None
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            errors.append(f"{side} mg_levels[{index}] invalid")
            continue
        if raw.get("level") != index:
            errors.append(f"{side} mg_levels[{index}].level invalid")
        for key in time_fields:
            number = raw.get(key)
            if (isinstance(number, bool) or
                    not isinstance(number, (int, float)) or
                    not math.isfinite(float(number)) or float(number) < 0.0):
                errors.append(f"{side} mg_levels[{index}].{key} invalid")
        for key in iteration_fields:
            number = raw.get(key)
            if (isinstance(number, bool) or not isinstance(number, int) or
                    number < 0):
                errors.append(f"{side} mg_levels[{index}].{key} invalid")
        if all(isinstance(raw.get(key), int) for key in (
                "pre_smoother_iterations", "post_smoother_iterations",
                "smoother_iterations")):
            if raw["smoother_iterations"] != (
                    raw["pre_smoother_iterations"] +
                    raw["post_smoother_iterations"]):
                errors.append(
                    f"{side} mg_levels[{index}].smoother_iterations mismatch")
        if all(isinstance(raw.get(key), (int, float))
               for key in time_fields[:6]):
            known = sum(float(raw[key]) for key in time_fields[1:6])
            other = float(raw["total_seconds"]) - known
            tolerance = max(
                1.0e-6,
                1.0e-2 * max(
                    1.0, abs(float(raw["total_seconds"])), abs(known)))
            expected_other = max(0.0, other)
            if other < -tolerance or not math.isclose(
                    expected_other, float(raw["other_seconds"]),
                    rel_tol=1.0e-9, abs_tol=1.0e-9):
                errors.append(
                    f"{side} mg_levels[{index}] timing conservation failed")
        total_iterations = raw.get("total_iterations")
        if isinstance(total_iterations, int) and not isinstance(
                total_iterations, bool):
            if finest_total is None:
                finest_total = total_iterations
            elif total_iterations != finest_total:
                errors.append(
                    f"{side} mg_levels[{index}].total_iterations "
                    "must equal the finest outer total")
        if index == 0:
            if raw.get("finest_iterations") != total_iterations:
                errors.append(
                    f"{side} mg_levels[0].finest_iterations invalid")
            if not isinstance(raw.get("residual_sequence"), list):
                errors.append(
                    f"{side} mg_levels[0].residual_sequence missing")
    return errors


def validate_document(document: Mapping[str, Any], *, allow_planned: bool = False) -> List[str]:
    """轻量内建 schema 校验；不依赖 jsonschema，dry-run 也能执行。"""
    errors: List[str] = []
    schema = document.get("schema")
    if schema != {"name": SCHEMA_NAME, "version": SCHEMA_VERSION}:
        errors.append("schema name/version mismatch")
    protocol = document.get("protocol")
    if not isinstance(protocol, Mapping):
        errors.append("protocol must be an object")
        return errors
    profile = protocol.get("profile")
    if profile not in PROFILE_NAMES:
        errors.append("profile must be formal or smoke")
    if document.get("profile") != profile:
        errors.append("document/profile does not match protocol/profile")
    protocol_lattice = protocol.get("lattice_xyzt")
    protocol_block = protocol.get("block_xyzt")
    if (not isinstance(protocol_lattice, list) or
            len(protocol_lattice) != 4 or
            any(isinstance(value, bool) or not isinstance(value, int) or
                value <= 0 or value % 2 for value in protocol_lattice)):
        errors.append("protocol.lattice_xyzt must contain four positive even integers")
        protocol_lattice = list(DEFAULT_LATTICE)
    if (not isinstance(protocol_block, list) or
            len(protocol_block) != 4 or
            any(isinstance(value, bool) or not isinstance(value, int) or
                value <= 0 for value in protocol_block)):
        errors.append("protocol.block_xyzt must contain four positive integers")
        protocol_block = list(BLOCK)
    protocol_levels = protocol.get("levels")
    if (isinstance(protocol_levels, bool) or
            not isinstance(protocol_levels, int) or
            not 1 <= protocol_levels <= 5):
        errors.append("protocol.levels must be in [1,5]")
        protocol_levels = LEVELS
    if protocol.get("warmups") != WARMUPS:
        errors.append("warmups must be 2")
    repeats = protocol.get("repeats")
    if (isinstance(repeats, bool) or not isinstance(repeats, int) or
            repeats <= 0):
        errors.append("repeats must be a positive integer")
    execution = document.get("execution")
    selected_phases = DEFAULT_PHASES
    if isinstance(execution, Mapping):
        phase_contract = execution.get("phases")
        if isinstance(phase_contract, Mapping):
            try:
                selected_phases = _normalise_phases(
                    phase_contract.get("selected"))
            except ValueError as exc:
                errors.append(f"execution.phases invalid: {exc}")
    precision = protocol.get("precision")
    if not isinstance(precision, Mapping) or precision.get("name") not in ("c64", "c128"):
        errors.append("precision must be c64/c128")
    else:
        expected_precision = _precision_spec(str(precision["name"]))
        if profile == "formal":
            formal_defaults = _formal_profile_defaults(expected_precision)
            fixed_values = {
                "repeats": formal_defaults["repeats"],
                "restart_requested": formal_defaults["restart"],
                "max_iter": formal_defaults["max_iter"],
                "max_krylov_bytes": formal_defaults["max_krylov_bytes"],
                "tolerance": formal_defaults["tolerance"],
            }
            for key, expected in fixed_values.items():
                if protocol.get(key) != expected:
                    errors.append(
                        f"formal profile requires {key}={expected!r}")
            fine_vector_bytes = (
                12 * math.prod(protocol_lattice) // 2
                * int(expected_precision["complex_bytes"]))
            coarse_vector_bytes = (
                int(protocol.get("coarse_dof", COARSE_DOF)) * math.prod(
                    extent // width for extent, width in
                    zip(protocol_lattice, protocol_block))
                * int(expected_precision["complex_bytes"]))
            expected_restart_effective = min(
                formal_defaults["restart"], formal_defaults["max_iter"],
                max(
                    0,
                    (formal_defaults["max_krylov_bytes"] -
                     (5 * fine_vector_bytes + 2 * coarse_vector_bytes)) //
                    (2 * fine_vector_bytes)))
            if protocol.get("restart_effective") != expected_restart_effective:
                errors.append(
                    "formal profile requires "
                    f"restart_effective={expected_restart_effective!r}")
            setup = protocol.get("pyqcu_strict_setup")
            if not isinstance(setup, Mapping):
                errors.append("formal pyqcu_strict_setup missing")
            else:
                setup_values = {
                    "probe_mode": "colored",
                    "column_batch_size": (
                        formal_defaults["strict_galerkin_column_batch"]),
                    "projection_site_batch_size": (
                        DEFAULT_STRICT_GALERKIN_PROJECTION_BATCH),
                    "max_workspace_bytes": (
                        formal_defaults["strict_galerkin_max_workspace_bytes"]),
                    "workspace_four_arena_lower_bound_bytes": int(
                        4 * formal_defaults["strict_galerkin_column_batch"] *
                        12 * math.prod(protocol_lattice) *
                        int(expected_precision["complex_bytes"])),
                    "require_exact_batch": True,
                }
                for key, expected in setup_values.items():
                    if setup.get(key) != expected:
                        errors.append(
                            f"formal profile requires pyqcu_strict_setup"
                            f".{key}={expected!r}")
            for key in (
                    "complex_bytes", "real", "default_tolerance",
                    "true_residual_gate"):
                if precision.get(key) != expected_precision[key]:
                    errors.append(
                        f"formal precision field {key} does not match"
                        f" {expected_precision[key]!r}")
    expected_hash = protocol.get("config_hash")
    if isinstance(expected_hash, str):
        unhashed = dict(protocol)
        unhashed.pop("config_hash", None)
        if _sha256_json(unhashed) != expected_hash:
            errors.append("config_hash does not match protocol")
    else:
        errors.append("config_hash missing")
    input_fingerprints = document.get("input_fingerprints")
    sides = document.get("sides")
    if not isinstance(sides, Mapping):
        errors.append("sides must be an object")
        return errors
    successful_sides = [
        side for side in SIDE_NAMES
        if isinstance(sides.get(side), Mapping) and
        sides[side].get("status") == "ok"
    ]
    errors.extend(_validate_input_fingerprints(
        input_fingerprints, required=bool(successful_sides),
        levels=protocol_levels))
    expected_input_hash = (
        input_fingerprints.get("bundle_hash")
        if isinstance(input_fingerprints, Mapping) else None)
    valid_statuses = TERMINAL_STATUSES | {"not_selected"}
    if allow_planned:
        valid_statuses.add("planned")
    for side in SIDE_NAMES:
        record = sides.get(side)
        if not isinstance(record, Mapping):
            errors.append(f"missing side record: {side}")
            continue
        if record.get("side") != side:
            errors.append(f"side identity mismatch: {side}")
        status = record.get("status")
        if status not in valid_statuses:
            errors.append(f"invalid {side} status={status!r}")
        if status == "ok":
            parallel = (
                execution.get("parallel") if isinstance(execution, Mapping)
                else None)
            expected_ranks = (
                int(parallel.get("ranks", 1))
                if isinstance(parallel, Mapping) else 1)
            rank_results = record.get("rank_results")
            if expected_ranks > 1 and (
                    not isinstance(rank_results, list) or
                    len(rank_results) != expected_ranks):
                errors.append(
                    f"{side} rank_results must contain {expected_ranks} entries")
            timing = record.get("timing")
            steady = timing.get("steady") if isinstance(timing, Mapping) else None
            samples = steady.get("samples_seconds") if isinstance(steady, Mapping) else None
            expected_steady = repeats if "steady" in selected_phases else 0
            expected_warmups = WARMUPS if "warmup" in selected_phases else 0
            if (not isinstance(samples, list) or
                    len(samples) != expected_steady):
                errors.append(
                    f"{side} steady samples must contain "
                    f"{expected_steady} entries")
            elif expected_steady and isinstance(repeats, int) and not isinstance(
                    repeats, bool):
                errors.extend(_validate_median_mad(
                    steady, repeats, f"{side} timing.steady"))
            warmups = timing.get("warmups") if isinstance(timing, Mapping) else None
            if (not isinstance(warmups, list) or
                    len(warmups) != expected_warmups):
                errors.append(
                    f"{side} warmups must contain {expected_warmups} entries")
            if "cold" in selected_phases and not isinstance(
                    timing.get("cold_seconds"), (int, float)):
                errors.append(f"{side} timing.cold_seconds missing")
            has_iteration_contract = (
                record.get("iteration_semantics") is not None or
                record.get("mg_levels") is not None)
            if has_iteration_contract:
                if not isinstance(record.get("iteration_semantics"), str):
                    errors.append(f"{side} iteration_semantics missing")
                errors.extend(_validate_mg_levels(
                    record.get("mg_levels"), protocol_levels, side))
            true_residual = record.get("true_residual")
            if not isinstance(true_residual, Mapping) or not isinstance(
                    true_residual.get("samples_rel"), list):
                errors.append(f"{side} true residual samples missing")
            if record.get("config_hash") != expected_hash:
                errors.append(f"{side} config_hash mismatch")
            if record.get("input_bundle_hash") != expected_input_hash:
                errors.append(f"{side} input_bundle_hash mismatch")
            boundary = record.get("timing_boundary")
            if not isinstance(boundary, Mapping):
                errors.append(f"{side} timing boundary missing")
            else:
                if boundary.get("zero_initial_guess_before_timer") is not True:
                    errors.append(
                        f"{side} timing boundary does not prove zero initial guess")
                if profile == "formal":
                    if boundary.get("supported") is not True:
                        errors.append(f"{side} formal timing output contract unsupported")
                    if boundary.get("formal_eligible") is not True:
                        errors.append(f"{side} formal timing boundary is not eligible")
            errors.extend(_validate_provenance_contract(
                record.get("provenance"), f"{side} provenance",
                require_quda_build=(profile == "formal" and side == "quda")))
            memory = record.get("memory")
            if not isinstance(memory, Mapping):
                errors.append(f"{side} memory evidence missing")
            else:
                if memory.get("schema_version") != MEMORY_SCHEMA_VERSION:
                    errors.append(f"{side} memory schema version mismatch")
                setup_memory = memory.get("setup")
                setup_sampler = (
                    setup_memory.get("device_wide_sampler")
                    if isinstance(setup_memory, Mapping) else None)
                first_solve = memory.get("first_solve")
                if not isinstance(first_solve, Mapping):
                    errors.append(f"{side} first_solve memory evidence missing")
                    first_sampler = None
                else:
                    if first_solve.get("excluded_from_formal_timing") is not True:
                        errors.append(
                            f"{side} first_solve must be excluded from timing")
                    first_solve_memory = first_solve.get("memory")
                    first_sampler = (
                        first_solve_memory.get("device_wide_sampler")
                        if isinstance(first_solve_memory, Mapping) else None)
                steady_memory = memory.get("steady")
                probe = (
                    steady_memory.get("untimed_device_memory_probe")
                    if isinstance(steady_memory, Mapping) else None)
                probe_sampler = (
                    probe.get("device_wide_sampler")
                    if isinstance(probe, Mapping) else None)
                for label, sampler in (
                        ("setup device sampler", setup_sampler),
                        ("first solve device sampler", first_sampler),
                        ("steady untimed device sampler", probe_sampler)):
                    if not isinstance(sampler, Mapping):
                        errors.append(f"{side} {label} missing")
                        continue
                    if sampler.get("available") is not True:
                        errors.append(f"{side} {label} unavailable")
                    if sampler.get("unit") != "bytes":
                        errors.append(f"{side} {label} unit must be bytes")
                    if sampler.get("join_timed_out") is not False:
                        errors.append(f"{side} {label} thread did not stop")
                    if sampler.get("errors") != []:
                        errors.append(f"{side} {label} reported sampling errors")
                    if not isinstance(sampler.get("device"), str):
                        errors.append(f"{side} {label} device missing")
                    samples = sampler.get("sample_count")
                    if not isinstance(samples, int) or samples <= 0:
                        errors.append(f"{side} {label} sample_count invalid")
                    maximum = sampler.get("device_used_max_observed_bytes")
                    if not isinstance(maximum, int) or maximum < 0:
                        errors.append(f"{side} {label} max_observed invalid")
                if not isinstance(probe, Mapping):
                    errors.append(f"{side} untimed device memory probe missing")
                elif probe.get("excluded_from_formal_timing") is not True:
                    errors.append(
                        f"{side} device memory probe must be excluded from timing")
            if side == "pyqcu":
                cache = record.get("runtime_cache")
                if not isinstance(cache, Mapping):
                    errors.append("pyqcu runtime cache evidence missing")
                else:
                    expectation = cache.get("expectation")
                    if expectation not in ("any", "miss", "hit"):
                        errors.append("pyqcu runtime cache expectation invalid")
                    if expectation == "hit" and cache.get("hit") is not True:
                        errors.append("pyqcu cache-hit record did not hit")
                    if expectation == "miss" and cache.get("hit") is not False:
                        errors.append("pyqcu cache-miss record unexpectedly hit")
                    if (profile == "formal" and
                            expectation != "hit"):
                        errors.append(
                            "formal pyqcu record requires cache expectation hit")
                    if (profile == "formal" and
                            expectation == "hit" and cache.get("hit") is not True):
                        errors.append(
                            "formal pyqcu record requires a verified cache hit")
                    if not isinstance(cache.get("path"), str):
                        errors.append("pyqcu runtime cache path missing")
                    if cache.get("hit") is True:
                        errors.extend(_validate_cache_evidence(
                            cache.get("evidence"), "pyqcu runtime_cache.evidence"))
                        if isinstance(input_fingerprints, Mapping):
                            try:
                                identity_payload = {
                                    "protocol": protocol,
                                    "input_fingerprints": input_fingerprints,
                                }
                                mpi_context = record.get("mpi")
                                if isinstance(mpi_context, Mapping):
                                    identity_payload["mpi_context"] = mpi_context
                                identity = _strict_runtime_cache_identity(
                                    identity_payload)
                                expected_identity_sha = _sha256_json(identity)
                                expected_cache_path = _strict_runtime_cache_path(
                                    identity,
                                    Path(document["execution"]["strict_cache"][
                                        "directory"])).resolve()
                                recorded_cache_path = Path(
                                    str(cache.get("path"))).resolve()
                                if cache.get("identity_sha256") != expected_identity_sha:
                                    errors.append(
                                        "pyqcu runtime cache identity mismatch")
                                if recorded_cache_path != expected_cache_path:
                                    errors.append(
                                        "pyqcu runtime cache path does not match identity")
                                evidence = cache.get("evidence")
                                if (isinstance(evidence, Mapping) and
                                        evidence.get("identity_sha256") !=
                                        expected_identity_sha):
                                    errors.append(
                                        "pyqcu runtime cache evidence identity mismatch")
                                if (isinstance(evidence, Mapping) and
                                        Path(str(evidence.get("path"))).resolve() !=
                                        recorded_cache_path):
                                    errors.append(
                                        "pyqcu runtime cache evidence path mismatch")
                            except (KeyError, TypeError, ValueError, OSError):
                                errors.append(
                                    "pyqcu runtime cache identity cannot be recomputed")
            if side == "quda" and profile == "formal":
                parameters = record.get("quda_parameters")
                if not isinstance(parameters, Mapping):
                    errors.append("formal quda_parameters missing")
                else:
                    requested = parameters.get("requested")
                    actual = parameters.get("actual")
                    mismatches = parameters.get("mismatches")
                    if not isinstance(requested, Mapping):
                        errors.append("formal quda requested parameters missing")
                    if not isinstance(actual, Mapping):
                        errors.append("formal quda resolved parameters missing")
                    if mismatches != []:
                        errors.append("formal quda parameter mismatches are non-empty")
                    if isinstance(requested, Mapping) and isinstance(actual, Mapping):
                        if actual != requested:
                            errors.append("formal quda resolved parameters differ")
                        mg = requested.get("multigrid")
                        if not isinstance(mg, Mapping):
                            errors.append("formal quda multigrid parameters missing")
                        else:
                            n_level = mg.get("n_level")
                            coarsest = mg.get("coarsest_level_index")
                            if (isinstance(n_level, bool) or
                                    not isinstance(n_level, int) or n_level < 1):
                                errors.append("formal quda n_level invalid")
                            elif coarsest != n_level - 1:
                                errors.append(
                                    "formal quda coarsest_level_index is not n_level-1")
                            levels = mg.get("levels")
                            transition = mg.get("transition")
                            if not isinstance(levels, Mapping):
                                errors.append("formal quda level parameters missing")
                            else:
                                for field in (
                                        "nu_pre", "nu_post", "coarse_solver",
                                        "coarse_solver_maxiter", "coarse_solver_tol"):
                                    values = levels.get(field)
                                    if (not isinstance(values, list) or
                                            isinstance(n_level, bool) or
                                            not isinstance(n_level, int) or
                                            len(values) < n_level):
                                        errors.append(
                                            f"formal quda levels.{field} does not cover n_level")
                                if (isinstance(n_level, int) and
                                        isinstance(levels.get("coarse_solver"), list) and
                                        len(levels["coarse_solver"]) >= n_level and
                                        levels["coarse_solver"][n_level - 1] !=
                                        "QUDA_CA_GCR_INVERTER"):
                                    errors.append(
                                        "formal quda coarse solver is not written at coarsest index")
                            if not isinstance(transition, Mapping):
                                errors.append("formal quda transition parameters missing")
                            elif (isinstance(n_level, int) and
                                  not isinstance(n_level, bool)):
                                for field in (
                                        "n_vec", "n_block_ortho", "vec_load",
                                        "vec_infile", "setup_use_mma", "dslash_use_mma",
                                        "transfer_use_mma"):
                                    values = transition.get(field)
                                    if (not isinstance(values, list) or
                                            len(values) != max(0, n_level - 1)):
                                        errors.append(
                                            f"formal quda transition.{field} length mismatch")
                qio_contract = record.get("quda_input_contract")
                if not isinstance(qio_contract, Mapping):
                    errors.append("formal quda input contract missing")
                else:
                    if qio_contract.get("qdp_host_dtype") != "complex128":
                        errors.append(
                            "formal quda QDP host gauge dtype must be complex128")
                    expected_device_precision = (
                        "QUDA_SINGLE_PRECISION"
                        if isinstance(precision, Mapping) and
                        precision.get("name") == "c64" else
                        "QUDA_DOUBLE_PRECISION")
                    if qio_contract.get("device_precision") != expected_device_precision:
                        errors.append("formal quda device precision mismatch")
                provenance = record.get("provenance")
                if isinstance(provenance, Mapping):
                    cmake = provenance.get("cmake_features")
                    if isinstance(cmake, Mapping):
                        errors.extend(_quda_cmake_feature_mismatches(
                            cmake, str(precision.get("name")),
                            int(protocol_levels)))
                if isinstance(parameters, Mapping):
                    requested = parameters.get("requested")
                    if isinstance(requested, Mapping):
                        qio = document.get("inputs", {}).get("quda_qio", {})
                        prefix = qio.get("prefix") if isinstance(qio, Mapping) else None
                        if prefix:
                            prefix = str(Path(str(prefix)).expanduser().resolve())
                        try:
                            expected_parameters = _quda_expected_parameters(
                                protocol, prefix)
                        except (KeyError, TypeError, ValueError, OSError):
                            errors.append("formal quda expected parameters cannot be built")
                        else:
                            if requested != expected_parameters:
                                errors.append(
                                    "formal quda requested parameters differ from protocol")
    comparison = document.get("comparison")
    if not isinstance(comparison, Mapping):
        errors.append("comparison must be an object")
    else:
        if comparison.get("profile") != profile:
            errors.append("comparison/profile does not match protocol/profile")
        if profile == "smoke":
            if comparison.get("fair") is True:
                errors.append("smoke comparison must never be fair")
            if comparison.get("speedup_pyqcu_over_quda") is not None:
                errors.append("smoke comparison must not contain speedup")
            if comparison.get("status") not in {
                    "smoke-pending", "smoke-pass", "smoke-unavailable",
                    "smoke-failed"}:
                errors.append("invalid smoke comparison status")
    return errors


def _list_payload(args: argparse.Namespace) -> Dict[str, Any]:
    document = build_document(args, dry_run=True)
    return {
        "schema": document["schema"],
        "profile": document["profile"],
        "formal_lattice": list(document["protocol"]["lattice_xyzt"]),
        "formal_levels": int(document["protocol"]["levels"]),
        "sides": {
            "pyqcu": {
                "worker": "internal strict fused FGMRES",
                "requires": [
                    "CUDA V100/P100 as requested", "PyTorch",
                    "PyQCU Cython/CUDA", "h5py"],
            },
            "quda": {
                "worker": "internal PyQUDA MG/GCR",
                "requires": [
                    "CUDA V100/P100 as requested", "PyQUDA/QUDA", "h5py",
                    "QIO null-vector prefix converted from canonical full null vectors",
                    "v1 conversion manifest with matching source_sha256 and QIO artifact hashes",
                ],
            },
        },
        "inputs": document["inputs"],
        "execution": document["execution"],
        "statistics": {"warmups": WARMUPS, "default_repeats": DEFAULT_REPEATS,
                       "summary": "median/MAD"},
        "commands": {
            "dry_run": f"{sys.executable} {Path(__file__).resolve()} --dry-run",
            "pyqcu_only": "--profile formal --side pyqcu --cache-expect hit "
                           "--output pyqcu.json",
            "quda_only": "--profile formal --side quda --quda-nullvec-prefix PREFIX "
                         "--quda-nullvec-manifest MANIFEST --output quda.json",
            "merge": "--merge pyqcu.json quda.json --output combined.json",
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fair 16x32x32x48 strict PyQCU-vs-QUDA benchmark collector")
    parser.add_argument("--side", choices=("pyqcu", "quda", "both"), default="both")
    parser.add_argument(
        "--profile", choices=PROFILE_NAMES, default=DEFAULT_PROFILE,
        help="formal fixes the reproducibility protocol; smoke permits exploration")
    parser.add_argument(
        "--device", choices=DEVICE_TARGETS, default="v100",
        help="accelerator family; QCU_DEVICE_ID remains an explicit override")
    parser.add_argument(
        "--mpi-ranks", type=int, default=1,
        help="number of MPI ranks launched for each collector side")
    parser.add_argument(
        "--process-grid", type=int, nargs=4, default=None,
        metavar=("PX", "PY", "PZ", "PT"),
        help="4D process grid; defaults to [mpi-ranks,1,1,1]")
    parser.add_argument(
        "--phases", default="warmup,steady",
        help="comma-separated phase subset from cold,warmup,steady")
    parser.add_argument("--precision", choices=("c64", "c128"), default="c64")
    parser.add_argument(
        "--quda-coarse-precision", choices=("auto", "double", "single"),
        default="auto",
        help="QUDA MG coarse-field precision; use single to bound c128 memory")
    parser.add_argument(
        "--pyqcu-strict-block-precision", choices=("auto", "single"),
        default="auto",
        help="PyQCU strict Galerkin block precision; single bounds c128 memory")
    parser.add_argument(
        "--pyqcu-strict-block-device", choices=("auto", "cpu"),
        default="auto",
        help="PyQCU strict Galerkin block staging device")
    parser.add_argument(
        "--pyqcu-strict-offload-null-basis", action="store_true",
        help="offload the fine null basis between strict setup transitions")
    parser.add_argument(
        "--quda-strategy", choices=("aligned", "library"),
        default="aligned",
        help="aligned one-cycle settings or QUDA/PyQUDA library defaults")
    parser.add_argument(
        "--lattice", type=int, nargs=4, default=list(DEFAULT_LATTICE),
        metavar=("LX", "LY", "LZ", "LT"))
    parser.add_argument("--levels", type=int, default=LEVELS)
    parser.add_argument(
        "--block", type=int, nargs=4, default=list(BLOCK),
        metavar=("BX", "BY", "BZ", "BT"))
    parser.add_argument(
        "--block-level", type=int, nargs=4, action="append", default=[],
        metavar=("BX", "BY", "BZ", "BT"),
        help="per-transition coarsening block; repeat levels-1 times")
    parser.add_argument("--gauge-path", default=None)
    parser.add_argument("--nullvec-path", default=None)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                        help="per-side child process timeout in seconds")
    parser.add_argument("--output", default=None,
                        help=f"atomic JSON output (formal default recommendation: {DEFAULT_OUTPUT})")
    parser.add_argument("--restart", type=int, default=DEFAULT_RESTART)
    parser.add_argument(
        "--max-krylov-bytes", type=int, default=None,
        help="outer workspace cap; default scales 512 MiB by complex element size")
    parser.add_argument(
        "--strict-galerkin-column-batch", type=int, default=None,
        help="PyQCU cold-setup colored probe width; c64 default 12, c128 default 1")
    parser.add_argument(
        "--strict-galerkin-projection-batch", type=int, default=None,
        help="PyQCU strict Galerkin projection-site batch size")
    parser.add_argument(
        "--strict-galerkin-max-workspace-bytes", type=int, default=None,
        help="PyQCU cold-setup workspace cap; independent of outer Krylov memory")
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument("--tol", type=float, default=None)
    parser.add_argument(
        "--coarse-tol", type=float, default=None,
        help="shared PyQCU/QUDA coarse-solver tolerance in (0, 0.5]")
    parser.add_argument(
        "--nu", type=int, default=DEFAULT_NU_PRE,
        help="shared symmetric pre/post smoother steps")
    parser.add_argument("--quda-nullvec-prefix", default=None)
    parser.add_argument("--quda-nullvec-manifest", default=None)
    parser.add_argument(
        "--strict-cache-dir", default=str(STRICT_CACHE_DIR),
        help="workspace-local strict runtime cache directory")
    parser.add_argument(
        "--cache-expect", choices=("any", "miss", "hit"), default="any",
        help="formal execution requires hit; dry-run/smoke may use any")
    parser.add_argument("--resume", action="store_true",
                        help="reuse successful compatible side records from --output")
    parser.add_argument("--dry-run", action="store_true",
                        help="emit plan/schema without hashing inputs or starting workers")
    parser.add_argument(
        "--allow-trace", action="store_true",
        help="allow diagnostic trace env vars in a formal-profile invocation")
    parser.add_argument(
        "--reference-solver", choices=("none", "bicgstab"),
        default="bicgstab",
        help="plain BiCGStab performance reference recorded beside MG")
    parser.add_argument("--reference-warmups", type=int, default=1)
    parser.add_argument("--list", action="store_true",
                        help="list protocol, dependencies and commands; starts no worker")
    parser.add_argument("--merge", nargs="+", metavar="JSON",
                        help="merge compatible single-side/partial result documents")
    parser.add_argument("--_worker", choices=SIDE_NAMES, help=argparse.SUPPRESS)
    return parser


def _run_parent(args: argparse.Namespace) -> Tuple[Dict[str, Any], int]:
    if args.timeout <= 0.0 or not math.isfinite(args.timeout):
        raise ValueError("--timeout must be a finite positive number")
    if args.reference_warmups < 0:
        raise ValueError("--reference-warmups must be non-negative")
    profile = _profile_name(args)
    active_trace_env = [
        name for name in (
            "PYQCU_STRICT_TRACE_FILE", "PYQCU_QUDA_TRACE_FILE",
            "QUDA_MG_TRACE_FILE")
        if os.environ.get(name)
    ]
    if profile == "formal" and active_trace_env and not args.allow_trace:
        raise ValueError(
            "formal speedup requires tracing disabled; unset "
            + ", ".join(active_trace_env)
            + " or use --allow-trace for a diagnostic run")
    if (profile == "formal" and args.side in ("pyqcu", "both") and
            args.cache_expect != "hit" and not (
                "cold" in _phases_from_args(args) and
                any(phase in _phases_from_args(args)
                    for phase in ("warmup", "steady")))):
        raise ValueError(
            "formal profile requires --cache-expect hit when pyqcu is selected; "
            "use --profile smoke for exploratory cache policies")
    document = build_document(args, dry_run=False)
    document["collector"]["diagnostic_trace_environment"] = active_trace_env
    selected = list(SIDE_NAMES if args.side == "both" else (args.side,))
    requested_qio = dict(document["inputs"]["quda_qio"])
    requested_execution = copy.deepcopy(document["execution"])

    try:
        # Always fingerprint the current files before considering reuse.  A
        # resume record is never trusted merely because it already contains a
        # fingerprint: the source file, path, dataset, shape, dtype and size
        # must be re-observed in this invocation.
        current_fingerprints = _fingerprint_inputs(document)
        if args.resume:
            if not args.output:
                raise ValueError("--resume requires --output")
            output_path = Path(args.output)
            if output_path.is_file():
                previous = _load_document(output_path)
                stale = _input_fingerprint_mismatches(
                    previous.get("input_fingerprints"), current_fingerprints)
                if stale:
                    raise BenchmarkFailure(
                        "resume_stale_input", "; ".join(stale))
                _compatible(document, previous)
                document = previous
                # 恢复 PyQCU partial 后，允许本次补充 QUDA QIO 适配器；这不
                # 改变 canonical config/input hash，只改变 QUDA 的读取入口。
                document["inputs"]["quda_qio"] = requested_qio
                document["execution"] = requested_execution
                document["input_fingerprints"] = current_fingerprints
                document["selected_sides"] = sorted(set(
                    document.get("selected_sides", [])) | set(selected))
                document["collector"]["resume"] = True
                document["collector"]["timeout_seconds_per_side"] = float(args.timeout)
            else:
                document["input_fingerprints"] = current_fingerprints
        else:
            document["input_fingerprints"] = current_fingerprints
    except BenchmarkSkip as exc:
        for side in selected:
            document["sides"][side] = {
                "side": side,
                "status": "skipped",
                "reason": {"code": exc.code, "detail": exc.detail},
            }
        _update_state(document)
        return document, 2
    except BenchmarkFailure as exc:
        for side in selected:
            document["sides"][side] = {
                "side": side,
                "status": "failed",
                "reason": {"code": exc.code, "detail": exc.detail},
            }
            if exc.context:
                document["sides"][side].update(exc.context)
        _update_state(document)
        return document, 1

    for side in selected:
        if args.resume and _resume_side_compatible(
                side, document["sides"].get(side, {}), requested_execution,
                document["input_fingerprints"]["bundle_hash"]):
            continue
        document["sides"][side] = _launch_side(side, document, float(args.timeout))
        _update_state(document)
        if args.output:
            _atomic_write(Path(args.output), document)

    _update_state(document)
    validation_errors = validate_document(document, allow_planned=True)
    if validation_errors:
        return document, 1
    statuses = [document["sides"][side]["status"] for side in selected]
    if any(status in ("failed", "timeout") for status in statuses):
        return document, 1
    if any(status == "skipped" for status in statuses):
        return document, 2
    if any(status == "side_unsupported" for status in statuses):
        return document, 2
    if set(selected) == set(SIDE_NAMES):
        comparison = document["comparison"]
        if profile == "formal" and comparison.get("fair") is not True:
            return document, 1
        if profile == "smoke" and comparison.get("status") != "smoke-pass":
            return document, 1
    return document, 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args._worker:
        encoded = os.environ.get(WORKER_PAYLOAD_ENV)
        if not encoded:
            record = {
                "side": args._worker,
                "status": "failed",
                "reason": {"code": "worker_payload_missing", "detail": WORKER_PAYLOAD_ENV},
            }
        else:
            try:
                payload = json.loads(base64.b64decode(encoded).decode("utf-8"))
                record = _worker_record(args._worker, payload)
            except BaseException as exc:
                record = {
                    "side": args._worker,
                    "status": "failed",
                    "reason": {"code": "worker_payload_invalid", "detail": repr(exc)},
                }
        print(WORKER_PREFIX + json.dumps(
            record, ensure_ascii=False, separators=(",", ":"), allow_nan=False), flush=True)
        return 0 if record.get("status") in ("ok", "skipped") else 1

    if args.list:
        print(json.dumps(_list_payload(args), ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    if args.merge:
        document = merge_documents(args.merge)
        _emit(document, args.output)
        return 0 if document["comparison"]["status"] in (
            "pass", "pending", "smoke-pass", "smoke-pending") else 2
    if args.dry_run:
        document = build_document(args, dry_run=True)
        _emit(document, args.output)
        return 0

    document, code = _run_parent(args)
    _emit(document, args.output)
    return code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BenchmarkFailure, ValueError) as exc:
        print(f"bench_strict_vs_quda: {exc}", file=sys.stderr)
        raise SystemExit(1)
