#!/usr/bin/env python3
"""可复用的 QUDA reduction 快速 smoke。

本脚本故意不在模块导入阶段加载 PyQUDA。运行时先复用
``bench_strict_vs_quda.py`` 的 WSL2 reduction guard 与 QMP 初始化 helper，
再初始化 PyQUDA；这样可以覆盖 ``QMP_init -> PyQUDA.init`` 的真实顺序。

默认问题是单 rank、周期边界的 ``4^4`` 单位规范场。源为
``psi(x)=(-1)^x psi_0``，即 ``k_x=pi`` 的自由 Wilson 本征模。因为其它三个
方向的动量为零，Wilson 自旋项在正负跳跃中成对抵消，
``D_W psi = (mass + 2) psi``；CPU 校验因此可以独立地用这个已知本征值计算
``||D_W x-b||_2 / ||b||_2``，不依赖 QUDA 的 residual 实现或 PyQCU 算子。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA_DIR = REPO / "data"
DEFAULT_RESOURCE_PATH = DATA_DIR / "quda-reduction-smoke-resource"
DEFAULT_OUTPUT = HERE / "out" / "quda_reduction_smoke.json"
DEFAULT_LATTICE = (4, 4, 4, 4)
DEFAULT_MASS = 0.05
DEFAULT_TOLERANCE = 5.0e-10
DEFAULT_REPEATS = 20
DEFAULT_MAX_ITER = 100
PASS_MARKER = "DEV87_REDUCTION_SMOKE_PASS"
SCHEMA = "pyqcu.quda-reduction-smoke"
SCHEMA_VERSION = 1


class SmokeFailure(RuntimeError):
    """可报告给 JSON 的前置或运行期失败。"""

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
    if not path.is_file():
        raise SmokeFailure("library_missing", str(path))
    return {
        "path": str(path.resolve()),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _load_bench_helpers() -> Any:
    """只加载 benchmark 的纯 Python helper；此处仍未 import pyquda。"""
    here_text = str(HERE)
    if here_text not in sys.path:
        sys.path.insert(0, here_text)
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


def _quda_prefix_and_library() -> tuple[Path, Path]:
    text = os.environ.get("QUDA_INSTALL") or os.environ.get("QUDA_PATH")
    if not text:
        raise SmokeFailure(
            "quda_install_missing",
            "set QUDA_INSTALL/QUDA_PATH or pass --quda-install")
    prefix = Path(text).expanduser().resolve()
    library = prefix / "lib" / "libquda.so"
    if not library.is_file():
        raise SmokeFailure("quda_library_missing", str(library))
    return prefix, library


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


def _select_visible_v100(torch: Any) -> tuple[Any, dict[str, Any]]:
    if not torch.cuda.is_available():
        raise SmokeFailure("cuda_unavailable", "torch.cuda.is_available() is false")
    count = int(torch.cuda.device_count())
    names = [str(torch.cuda.get_device_name(i)) for i in range(count)]
    matches = [i for i, name in enumerate(names) if "V100" in name]
    if not matches:
        raise SmokeFailure(
            "v100_unavailable",
            f"visible CUDA devices do not contain V100: {names!r}")
    index = matches[0]
    torch.cuda.set_device(index)
    selected_name = str(torch.cuda.get_device_name(index))
    if "V100" not in selected_name:
        raise SmokeFailure("wrong_gpu", repr(selected_name))
    device = torch.device("cuda", index)
    return device, {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "visible_count": count,
        "visible_names": names,
        "selected_index": index,
        "selected_name": selected_name,
    }


def _unit_gauge_qdp(np: Any, lattice: Sequence[int]) -> Any:
    _, _, _, t = (int(v) for v in lattice)
    x, y, z, _ = (int(v) for v in lattice)
    eye = np.eye(3, dtype=np.complex128)
    # PyQUDA 的 QDP host layout 是 (mu,t,z,y,x,row,col)，x 最快。
    return np.ascontiguousarray(np.broadcast_to(
        eye, (4, t, z, y, x, 3, 3)))


def _kx_pi_mode(np: Any, lattice: Sequence[int]) -> Any:
    x, y, z, t = (int(v) for v in lattice)
    mode = np.zeros((4, 3, x, y, z, t), dtype=np.complex128)
    phase = (-1.0) ** np.arange(x, dtype=np.int64)
    mode[0, 0] = phase[:, None, None, None]
    return np.ascontiguousarray(mode)


def _to_lattice_fermion(np: Any, torch: Any, info: Any, mode: Any, device: Any) -> Any:
    from pyquda.field import LatticeFermion

    tzyxsc = np.ascontiguousarray(np.transpose(mode, (5, 4, 3, 2, 0, 1)))
    eo = info.evenodd(tzyxsc, False)
    return LatticeFermion(
        info, torch.from_numpy(np.ascontiguousarray(eo)).to(device))


def _to_lattice_gauge(np: Any, torch: Any, info: Any, qdp: Any, device: Any) -> Any:
    from pyquda.field import LatticeGauge

    eo = info.evenodd(qdp, True)
    return LatticeGauge(
        info, 4, torch.from_numpy(np.ascontiguousarray(eo)).to(device))


def _field_to_scxyzt(np: Any, field: Any, info: Any) -> Any:
    data = field.data
    if hasattr(data, "detach"):
        data = data.detach()
    if hasattr(data, "cpu"):
        data = data.cpu()
    data = np.asarray(data)
    # info.lexico 返回 (t,z,y,x,s,c)，转回本脚本的 (s,c,x,y,z,t)。
    lex = np.asarray(info.lexico(np.ascontiguousarray(data), False))
    return np.ascontiguousarray(np.transpose(lex, (4, 5, 3, 2, 1, 0)))


def _cpu_mode_true_residual(solution: Any, rhs: Any, mass: float) -> float:
    eigenvalue = float(mass) + 2.0
    residual = eigenvalue * solution - rhs
    denominator = max(float(np_norm(rhs)), 1.0e-300)
    value = float(np_norm(residual)) / denominator
    if not math.isfinite(value):
        raise SmokeFailure("cpu_true_residual_nonfinite", repr(value))
    return value


def np_norm(value: Any) -> float:
    """延迟使用 NumPy 的 norm，保持校验函数与 PyTorch 解耦。"""
    # value 是 numpy ndarray；调用其实现不涉及 CUDA 或 PyQCU。
    import numpy as np
    return float(np.linalg.norm(np.asarray(value).ravel()))


def _scalar_true_res(invert: Any) -> float:
    if not hasattr(invert, "true_res"):
        raise SmokeFailure(
            "quda_true_res_unavailable",
            "PyQUDA invert_param does not expose true_res")
    raw = getattr(invert, "true_res")
    try:
        raw = raw[0]
    except (IndexError, KeyError, TypeError):
        pass
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise SmokeFailure("quda_true_res_invalid", repr(raw)) from exc
    if not math.isfinite(value):
        raise SmokeFailure("quda_true_res_nonfinite", repr(value))
    return value


def _set_required(target: Any, field: str, value: Any) -> None:
    if not hasattr(target, field):
        raise SmokeFailure("quda_param_missing", field)
    setattr(target, field, value)


def _summary(values: Sequence[float]) -> dict[str, Any]:
    values = [float(v) for v in values]
    return {
        "samples": values,
        "min_seconds": min(values),
        "median_seconds": float(statistics.median(values)),
        "max_seconds": max(values),
        "total_seconds": float(sum(values)),
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    lattice = tuple(int(v) for v in args.lat)
    if len(lattice) != 4 or any(v <= 0 for v in lattice):
        raise SmokeFailure("invalid_lattice", repr(lattice))
    if args.repeats <= 0 or args.max_iter <= 0 or args.tol <= 0:
        raise SmokeFailure("invalid_solver_argument", "repeats/max-iter/tol must be positive")
    if args.mass + 2.0 <= 0.0:
        raise SmokeFailure("invalid_mass", f"mass={args.mass}")

    resource = _resource_path(args.resource_path)
    rank = _single_rank_report()

    # 这两个 helper 必须发生在 import pyquda 之前；QMP helper 自己保持
    # ctypes 对象和 argv 生命周期，并以 atexit LIFO 保证 endQuda 先执行。
    bench = _load_bench_helpers()
    wsl2_guard = bench._prepare_quda_reduction_runtime()
    prefix, libquda = _quda_prefix_and_library()
    qmp_runtime = bench._initialize_quda_qmp_runtime(wsl2_guard)
    library = {
        "install_prefix": str(prefix),
        "libquda": _file_summary(libquda),
        "libqmp": _file_summary(Path(str(qmp_runtime["library"]))),
        "wsl2_guard": dict(wsl2_guard),
        "qmp_runtime": dict(qmp_runtime),
    }

    # 到这里才允许加载 CUDA/PyQUDA；CUDA_VISIBLE_DEVICES 已经生效。
    import numpy as np
    import torch
    device, device_report = _select_visible_v100(torch)

    # 先确认 CUDA_VISIBLE_DEVICES 映射到的可见卡确实是 V100，再让
    # PyQUDA 读取 CUDA/MPI 运行时，避免错误设备的初始化副作用。
    import pyquda
    import pyquda_utils.core as core
    from pyquda.enum_quda import QudaBoolean, QudaInverterType, QudaPrecision

    resource.mkdir(parents=True, exist_ok=True)  # 永不由本脚本删除
    torch.cuda.synchronize(device)

    runtime_started = time.perf_counter()
    pyquda.init(
        grid_size=[1, 1, 1, 1],
        latt_size=list(lattice),
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

    mode = _kx_pi_mode(np, lattice)
    qdp = _unit_gauge_qdp(np, lattice)
    info = core.LatticeInfo(list(lattice), 1, 1.0)
    input_started = time.perf_counter()
    gauge_field = _to_lattice_gauge(np, torch, info, qdp, device)
    rhs_field = _to_lattice_fermion(np, torch, info, mode, device)
    input_seconds = time.perf_counter() - input_started

    precision = QudaPrecision.QUDA_DOUBLE_PRECISION
    dirac = None
    cleanup_errors: list[str] = []
    try:
        dirac = core.getWilson(
            info, float(args.mass), float(args.tol), int(args.max_iter))
        dirac.setPrecision(
            cuda=precision,
            sloppy=precision,
            precondition=precision,
            refinement_sloppy=precision,
            eigensolver=precision,
        )
        invert = dirac.invert_param
        _set_required(invert, "inv_type", QudaInverterType.QUDA_BICGSTAB_INVERTER)
        _set_required(invert, "tol", float(args.tol))
        _set_required(invert, "maxiter", int(args.max_iter))
        _set_required(invert, "use_init_guess", QudaBoolean.QUDA_BOOLEAN_FALSE)
        _set_required(invert, "compute_true_res", QudaBoolean.QUDA_BOOLEAN_TRUE)

        setup_started = time.perf_counter()
        dirac.loadGauge(gauge_field)
        torch.cuda.synchronize(device)
        setup_seconds = time.perf_counter() - setup_started

        solve_seconds: list[float] = []
        iterations: list[int] = []
        quda_true_res: list[float] = []
        cpu_true_res: list[float] = []
        expected = float(args.mass) + 2.0
        for _ in range(int(args.repeats)):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            solution = dirac.invert(rhs_field)
            torch.cuda.synchronize(device)
            solve_seconds.append(time.perf_counter() - started)

            iterations.append(int(getattr(invert, "iter")))
            quda_true_res.append(_scalar_true_res(invert))
            solution_full = _field_to_scxyzt(np, solution, info)
            cpu_true_res.append(_cpu_mode_true_residual(
                solution_full, mode, float(args.mass)))
            del solution

        quda_pass = max(quda_true_res) <= float(args.tol)
        cpu_pass = max(cpu_true_res) <= float(args.tol)
        passed = bool(quda_pass and cpu_pass)
        error = None if passed else {
            "code": "true_residual_gate",
            "detail": (
                f"quda_max={max(quda_true_res):.3e}, "
                f"cpu_max={max(cpu_true_res):.3e}, "
                f"threshold={args.tol:.3e}"),
        }
        return {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "status": "ok" if passed else "failed",
            "pass_marker": PASS_MARKER if passed else None,
            "pass_marker_expected": PASS_MARKER,
            "config": {
                "lattice_xyzt": list(lattice),
                "rank": rank,
                "precision": "double",
                "operator": "Wilson",
                "inverter": "BiCGStab",
                "mass": float(args.mass),
                "tolerance": float(args.tol),
                "max_iter": int(args.max_iter),
                "repeats": int(args.repeats),
            },
            "resource": {
                "path": str(resource),
                "required_root": str(DATA_DIR.resolve()),
                "cleanup": "not performed",
            },
            "library": library,
            "device": device_report,
            "physics": {
                "gauge": "unit",
                "boundary": "periodic",
                "momentum": ["pi", 0, 0, 0],
                "mode": "psi(x)=(-1)^x psi_0; one spin-color component",
                "expected_eigenvalue": expected,
                "rhs_norm": np_norm(mode),
            },
            "iterations": {
                "samples": iterations,
                "min": min(iterations),
                "max": max(iterations),
                "max_iter": int(args.max_iter),
            },
            "true_residual": {
                "threshold": float(args.tol),
                "quda_true_res": {
                    "samples": quda_true_res,
                    "max": max(quda_true_res),
                    "pass": quda_pass,
                },
                "cpu_relative": {
                    "samples": cpu_true_res,
                    "max": max(cpu_true_res),
                    "pass": cpu_pass,
                },
                "pass": passed,
            },
            "timing": {
                "input_prepare_seconds": float(input_seconds),
                "runtime_init_seconds": float(runtime_seconds),
                "setup_seconds": float(setup_seconds),
                "solve": _summary(solve_seconds),
                "solve_timing_note": "CUDA synchronized before and after each invert",
            },
            "cleanup_errors": cleanup_errors,
            "error": error,
        }
    finally:
        if dirac is not None:
            try:
                dirac.freeGauge()
            except BaseException as exc:  # 清理不能覆盖主异常
                cleanup_errors.append(repr(exc))


def _base_failure(args: argparse.Namespace, exc: BaseException) -> dict[str, Any]:
    requested = str(args.resource_path)
    try:
        requested = str(_resource_path(args.resource_path))
    except Exception:
        pass
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
        "config": {
            "lattice_xyzt": list(args.lat),
            "precision": "double",
            "operator": "Wilson",
            "inverter": "BiCGStab",
            "mass": float(args.mass),
            "tolerance": float(args.tol),
            "max_iter": int(args.max_iter),
            "repeats": int(args.repeats),
        },
        "resource": {
            "path": requested,
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
            "4^4 单 rank 单位规范场上的 double Wilson BiCGStab reduction smoke；"
            "成功时输出 DEV87_REDUCTION_SMOKE_PASS。"))
    parser.add_argument("--lat", type=int, nargs=4, default=list(DEFAULT_LATTICE),
                        metavar=("X", "Y", "Z", "T"),
                        help="周期格点尺寸（默认 4 4 4 4）")
    parser.add_argument("--mass", type=float, default=DEFAULT_MASS,
                        help="Wilson mass（默认 0.05）")
    parser.add_argument("--tol", type=float, default=DEFAULT_TOLERANCE,
                        help="QUDA 与独立 CPU 真残差阈值（默认 5e-10）")
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER,
                        help="BiCGStab 最大迭代数（默认 100）")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                        help="零初值 solve 次数（默认 20）")
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
        print(f"smoke JSON 写入失败: {exc}", file=sys.stderr, flush=True)
        return 1

    print(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    if record.get("status") == "ok" and record.get("pass_marker") == PASS_MARKER:
        print(PASS_MARKER, flush=True)
        return 0
    error = record.get("error")
    print(f"QUDA reduction smoke failed: {error}", file=sys.stderr, flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
