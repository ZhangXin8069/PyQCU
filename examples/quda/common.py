"""QUDA single-function test helpers and explicit PyQCU↔QDP layout adapters."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import textwrap
import time
from typing import Any, Callable, Sequence
import numpy as np
import torch
from pyqcu import dslash, tools

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("PYQCU_DATA_DIR", str(ROOT / "data"))).expanduser()
LAT = (16, 16, 16, 16)
MASS = 0.05
_DISPLAY_ENV = ("RUN_QUDA_TESTS", "QUDA_UNSAFE_INPROCESS")


class RunReporter:
    """Dependency-free progress and timing output for QUDA reference checks."""

    def __init__(self, *, enabled: bool = True, total: int | None = None) -> None:
        self.enabled = bool(enabled)
        self.total = total
        self.current = 0
        self.started = time.perf_counter()
        self.timings: list[tuple[str, float]] = []
        self.device: torch.device | None = None

    def _synchronize(self) -> None:
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def run(self, label: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        self.current += 1
        suffix = f"/{self.total}" if self.total else ""
        if self.enabled:
            if self.total:
                width = 20
                filled = int(width * (self.current - 1) / self.total)
                bar = "#" * filled + "-" * (width - filled)
                print(f"[进度 |{bar}| {self.current}{suffix}] {label} ...", flush=True)
            else:
                print(f"[进度 {self.current}{suffix}] {label} ...", flush=True)
        self._synchronize()
        started = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
            self._synchronize()
        except Exception:
            elapsed = time.perf_counter() - started
            self.timings.append((label, elapsed))
            print(f"[失败] {label}: {elapsed:.3f} s", flush=True)
            raise
        elapsed = time.perf_counter() - started
        self.timings.append((label, elapsed))
        if self.enabled:
            print(f"[完成] {label}: {elapsed:.3f} s", flush=True)
        return result

    def finish(self, status: str = "PASS") -> None:
        total = time.perf_counter() - self.started
        print(f"[总耗时] {total:.3f} s | status={status}", flush=True)
        if self.timings:
            details = ", ".join(f"{name}={elapsed:.3f}s" for name, elapsed in self.timings)
            print(f"[分项耗时] {details}", flush=True)


def _even_positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0 or parsed % 2:
        raise argparse.ArgumentTypeError("格点维度必须为正偶数")
    return parsed


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--lat", nargs=4, type=_even_positive, default=LAT,
                   metavar=("X", "Y", "Z", "T"), help="全局格点大小（默认: 16 16 16 16）")
    p.add_argument("--mass", type=float, default=MASS)
    p.add_argument("--no-doc", action="store_true", help="不在开头打印本脚本使用说明")
    p.add_argument("--no-progress", action="store_true", help="隐藏分步进度，只保留结果与耗时")
    return p


def run_setup(description: str, doc: str | None = None, *, argv: Sequence[str] | None = None,
              total: int | None = None) -> tuple[argparse.Namespace, RunReporter]:
    # Pytest invokes main() with its own command-line flags; standalone calls
    # still consume the process arguments as usual.
    if argv is None and os.environ.get("PYTEST_CURRENT_TEST"):
        argv = []
    p = parser(description)
    args = p.parse_args(argv)
    if not args.no_doc:
        print(f"=== {sys.argv[0]} 使用说明 ===")
        if doc:
            print(textwrap.dedent(doc).strip())
        print(f"运行示例: python {sys.argv[0]} --lat {' '.join(map(str, args.lat))}")
        print(f"查看全部参数: python {sys.argv[0]} --help")
        print("=== 输入参数 ===")
        env = {key: os.environ.get(key) for key in _DISPLAY_ENV if os.environ.get(key) is not None}
        print(json.dumps({"description": description, **vars(args), "env": env}, ensure_ascii=False, sort_keys=True))
    else:
        print("=== 输入参数 ===")
        env = {key: os.environ.get(key) for key in _DISPLAY_ENV if os.environ.get(key) is not None}
        print(json.dumps({"description": description, **vars(args), "env": env}, ensure_ascii=False, sort_keys=True))
    return args, RunReporter(enabled=not args.no_progress, total=total)


def pyqcu_gauge_to_quda(u: np.ndarray | torch.Tensor) -> np.ndarray:
    a = np.asarray(u.detach().cpu() if isinstance(u, torch.Tensor) else u)
    if a.ndim != 7 or a.shape[:3] != (3, 3, 4): raise ValueError("PyQCU gauge must be [3,3,4,X,Y,Z,T]")
    return np.ascontiguousarray(np.transpose(a, (2, 6, 5, 4, 3, 0, 1)))


def quda_gauge_to_pyqcu(qdp: np.ndarray) -> np.ndarray:
    a = np.asarray(qdp)
    if a.ndim != 7 or a.shape[-2:] != (3, 3): raise ValueError("QDP gauge must be [4,T,Z,Y,X,3,3]")
    return np.ascontiguousarray(np.transpose(a, (5, 6, 0, 4, 3, 2, 1)))


def pyqcu_fermion_to_quda(field: np.ndarray | torch.Tensor) -> np.ndarray:
    a = np.asarray(field.detach().cpu() if isinstance(field, torch.Tensor) else field)
    if a.ndim != 6 or a.shape[:2] != (4, 3): raise ValueError("PyQCU fermion must be [4,3,X,Y,Z,T]")
    s, c, x, y, z, t = a.shape
    full = np.transpose(a, (5, 4, 3, 2, 0, 1))  # [t,z,y,x,s,c]
    out = np.empty((2, t, z, y, x // 2, s, c), dtype=a.dtype)
    for it in range(t):
        for iz in range(z):
            for iy in range(y):
                for q in (0, 1):
                    start = (q - (it + iz + iy)) & 1
                    out[q, it, iz, iy] = full[it, iz, iy, start::2]
    return np.ascontiguousarray(out)


def quda_fermion_to_pyqcu(field: np.ndarray) -> np.ndarray:
    a = np.asarray(field)
    if a.ndim != 7 or a.shape[0] != 2: raise ValueError("QDP fermion must be [2,T,Z,Y,X/2,4,3]")
    _, t, z, y, xh, s, c = a.shape
    full = np.empty((t, z, y, 2 * xh, s, c), dtype=a.dtype)
    for it in range(t):
        for iz in range(z):
            for iy in range(y):
                for q in (0, 1):
                    start = (q - (it + iz + iy)) & 1
                    full[it, iz, iy, start::2] = a[q, it, iz, iy]
    return np.ascontiguousarray(np.transpose(full, (4, 5, 3, 2, 1, 0)))


def identity_gauge(lat=LAT, dtype=torch.complex64):
    x, y, z, t = lat
    eye = torch.eye(3, dtype=dtype)
    return eye.view(3, 3, 1, 1, 1, 1, 1).expand(3, 3, 4, x, y, z, t).contiguous()


def source(lat=LAT, seed=123):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn((4, 3, *lat), generator=g, dtype=torch.float32).to(torch.complex64)


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(a - b) / torch.clamp(torch.linalg.vector_norm(b), min=1e-30))


def quda_available() -> bool:
    try:
        import pyquda  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False


def quda_status() -> str:
    return "available" if quda_available() else "SKIP: pyquda/CUDA unavailable"


def require_quda():
    if not quda_available():
        raise RuntimeError("QUDA single-function test requires pyquda and CUDA")
    import pyquda
    return pyquda


def run_quda_mat(field: torch.Tensor, *, clover: bool = False, mass: float = MASS) -> torch.Tensor:
    """Run one QUDA ``Dirac.mat`` call and return PyQCU ``[s,c,x,y,z,t]``.

    This path is opt-in because loading QUDA and its MPI runtime is process-wide;
    the default unit tests remain deterministic and dependency-light.
    """
    pyquda = require_quda()
    import pyquda_utils.core as core
    from pyquda.field import LatticeFermion, LatticeGauge
    lat = tuple(int(v) for v in field.shape[-4:])
    u = identity_gauge(lat).numpy()
    pyquda.init(grid_size=[1, 1, 1, 1], latt_size=list(lat), backend="torch",
                backend_target="cuda", enable_nvshmem=False,
                enable_tuning=False, enable_device_memory_pool=False,
                enable_pinned_memory_pool=False)
    info = core.LatticeInfo(list(lat), 1, 1.0)
    try:
        qg = torch.from_numpy(info.evenodd(pyqcu_gauge_to_quda(u).astype(np.complex128), True)).to("cuda")
        full_tzyxsc = np.ascontiguousarray(np.transpose(field.detach().cpu().numpy(), (5, 4, 3, 2, 0, 1))).astype(np.complex128)
        qf = torch.from_numpy(info.evenodd(full_tzyxsc, False)).to("cuda")
        gauge = LatticeGauge(info, 4, qg)
        fermion = LatticeFermion(info, qf)
        if clover:
            d = core.getClover(info, mass, 1e-5, 100, clover_csw_t=1.0)
        else:
            d = core.getWilson(info, mass, 1e-5, 100)
        d.loadGauge(gauge)
        result = d.mat(fermion)
        return torch.from_numpy(quda_fermion_to_pyqcu(result.data.detach().cpu().numpy()))
    finally:
        close = locals().get("d")
        for name in ("destroy", "close", "end"):
            method = getattr(close, name, None) if close is not None else None
            if callable(method):
                try: method()
                except Exception: pass


def run_quda_solve(rhs: torch.Tensor, *, clover: bool = False,
                   multigrid: bool = False, mass: float = MASS) -> torch.Tensor:
    """Run one QUDA BiCGStab solve and return the solution in PyQCU layout."""
    pyquda = require_quda()
    import pyquda_utils.core as core
    from pyquda.field import LatticeFermion, LatticeGauge
    lat = tuple(int(v) for v in rhs.shape[-4:])
    pyquda.init(grid_size=[1, 1, 1, 1], latt_size=list(lat), backend="torch",
                backend_target="cuda", enable_nvshmem=False,
                enable_tuning=False, enable_device_memory_pool=False,
                enable_pinned_memory_pool=False)
    info = core.LatticeInfo(list(lat), 1, 1.0)
    d = None
    try:
        gauge = LatticeGauge(info, 4, torch.from_numpy(info.evenodd(pyqcu_gauge_to_quda(identity_gauge(lat).numpy()).astype(np.complex128), True)).to("cuda"))
        full_tzyxsc = np.ascontiguousarray(np.transpose(rhs.detach().cpu().numpy(), (5, 4, 3, 2, 0, 1))).astype(np.complex128)
        qrhs = torch.from_numpy(info.evenodd(full_tzyxsc, False)).to("cuda")
        b = LatticeFermion(info, qrhs)
        mg = [[2, 2, 2, 2]] if multigrid else None
        d = core.getClover(info, mass, 1e-5, 100, clover_csw_t=1.0, multigrid=mg) if clover else core.getWilson(info, mass, 1e-5, 100, multigrid=mg)
        d.loadGauge(gauge)
        x = d.invert(b)
        return torch.from_numpy(quda_fermion_to_pyqcu(x.data.detach().cpu().numpy()))
    finally:
        for name in ("destroy", "close", "end"):
            method = getattr(d, name, None) if d is not None else None
            if callable(method):
                try: method()
                except Exception: pass


def layout_roundtrip() -> dict:
    u = identity_gauge((2,2,2,4)).numpy()
    f = source((2,2,2,4)).numpy()
    return {"gauge_error": float(np.max(np.abs(quda_gauge_to_pyqcu(pyqcu_gauge_to_quda(u)) - u))),
            "fermion_error": float(np.max(np.abs(quda_fermion_to_pyqcu(pyqcu_fermion_to_quda(f)) - f)))}
