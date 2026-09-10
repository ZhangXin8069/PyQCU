"""Shared helpers for the QCU single-function checks.

The helpers deliberately keep the C++ bridge optional: on a CPU-only machine the
same scripts still exercise the PyTorch reference and layout contracts, while a
CUDA run executes exactly one QCU operation per lifecycle.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
from contextlib import suppress
import json
import os
import sys
import textwrap
import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch

from pyqcu import dslash, tools
from pyqcu.cuda import define


DEFAULT_LATTICE = (16, 16, 16, 16)
_DISPLAY_ENV = (
    "QCU_DEVICE", "QCU_SEED", "QCU_ATOL", "QCU_VERBOSE", "QCU_STRICT_NUMERIC",
    "QCU_EXERCISE_DSLASH", "QCU_EXERCISE_SCHUR", "QCU_EXERCISE_LAPLACIAN",
)


@dataclass
class Context:
    params: torch.Tensor
    argv: torch.Tensor
    set_ptrs: torch.Tensor
    device: torch.device
    dtype: torch.dtype
    lattice: tuple[int, int, int, int]
    mass: float
    reporter: "RunReporter | None" = None


class RunReporter:
    """Small dependency-free progress/timing reporter for standalone checks."""

    def __init__(self, *, enabled: bool = True, total: int | None = None) -> None:
        self.enabled = bool(enabled)
        self.total = total
        self.current = 0
        self.started = time.perf_counter()
        self.timings: list[tuple[str, float]] = []
        # ``timings`` measures the complete reported step (for example, a
        # QCU lifecycle including init/end).  ``function_timings`` records the
        # operation itself so the number can be compared with another backend.
        self.function_timings: list[tuple[str, float]] = []
        self._timing_details: list[tuple[str, float, tuple[tuple[str, float], ...]]] = []
        self.device: torch.device | None = None

    def _synchronize(self) -> None:
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def measure_function(self, label: str, fn: Callable[..., Any], *args: Any,
                         **kwargs: Any) -> Any:
        """Measure only the function under test, including asynchronous GPU work.

        The surrounding ``run`` call still measures the complete test step.  A
        separate measurement is needed for QCU/QUDA because setup, lifecycle
        management, layout conversion and destruction are not part of the
        operation being compared.
        """
        self._synchronize()
        started = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
            self._synchronize()
        except BaseException:
            # Synchronize before recording an exceptional CUDA call too; this
            # keeps the reported value meaningful when a kernel failed after
            # being queued.  Do not swallow the original exception.
            with suppress(Exception):
                self._synchronize()
            elapsed = time.perf_counter() - started
            self.function_timings.append((label, elapsed))
            raise
        elapsed = time.perf_counter() - started
        self.function_timings.append((label, elapsed))
        return result

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
        function_count = len(self.function_timings)
        started = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
            self._synchronize()
        except Exception:
            elapsed = time.perf_counter() - started
            self.timings.append((label, elapsed))
            function_timings = tuple(self.function_timings[function_count:])
            self._timing_details.append((label, elapsed, function_timings))
            detail = self._function_detail(function_timings)
            print(f"[失败] {label}: {elapsed:.3f} s{detail}", flush=True)
            raise
        elapsed = time.perf_counter() - started
        self.timings.append((label, elapsed))
        function_timings = tuple(self.function_timings[function_count:])
        self._timing_details.append((label, elapsed, function_timings))
        if self.enabled:
            detail = self._function_detail(function_timings)
            print(f"[完成] {label}: {elapsed:.3f} s{detail}", flush=True)
        return result

    @staticmethod
    def _function_detail(function_timings: tuple[tuple[str, float], ...]) -> str:
        if not function_timings:
            return ""
        details = ", ".join(f"{name}={elapsed:.6f} s" for name, elapsed in function_timings)
        return f" | 被测函数: {details}"

    def skip(self, reason: str) -> None:
        print(f"[跳过] {reason}", flush=True)

    def finish(self, status: str = "PASS") -> None:
        total = time.perf_counter() - self.started
        print(f"[总耗时] {total:.3f} s | status={status}", flush=True)
        if self.timings:
            details = []
            for label, elapsed, function_timings in self._timing_details:
                suffix = self._function_detail(function_timings)
                details.append(f"{label}={elapsed:.3f}s{suffix}")
            # Keep a fallback for reporters created before a timing detail was
            # recorded (and for callers that append to ``timings`` directly).
            if not details:
                details = [f"{name}={elapsed:.3f}s" for name, elapsed in self.timings]
            print(f"[分项耗时] {', '.join(details)}", flush=True)
        if self.function_timings:
            details = ", ".join(f"{name}={elapsed:.6f}s" for name, elapsed in self.function_timings)
            print(f"[被测函数耗时] {details}", flush=True)


def _even_positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0 or parsed % 2:
        raise argparse.ArgumentTypeError("格点维度必须为正偶数")
    return parsed


def _print_run_documentation(description: str, doc: str | None, args: argparse.Namespace) -> None:
    script = sys.argv[0]
    print(f"=== {script} 使用说明 ===")
    if doc:
        print(textwrap.dedent(doc).strip())
    print(f"运行示例: python {script} --lat {' '.join(map(str, args.lat))}")
    print(f"查看全部参数: python {script} --help")
    print("=== 输入参数 ===")
    env = {key: os.environ.get(key) for key in _DISPLAY_ENV if os.environ.get(key) is not None}
    print(json.dumps({"description": description, **vars(args), "env": env}, ensure_ascii=False, sort_keys=True))


def run_setup(description: str, doc: str | None = None, *, argv: Sequence[str] | None = None,
              total: int | None = None) -> tuple[argparse.Namespace, RunReporter]:
    """Parse CLI options and print the default quick-start documentation."""
    p = parser(description)
    args = p.parse_args(argv)
    if not args.no_doc:
        _print_run_documentation(description, doc, args)
    else:
        print("=== 输入参数 ===")
        env = {key: os.environ.get(key) for key in _DISPLAY_ENV if os.environ.get(key) is not None}
        print(json.dumps({"description": description, **vars(args), "env": env}, ensure_ascii=False, sort_keys=True))
    reporter = RunReporter(enabled=not args.no_progress, total=total)
    return args, reporter


def qcu_or_none():
    """Return the Cython bridge, or ``None`` when CUDA/libqcu is unavailable."""
    if not torch.cuda.is_available():
        return None
    try:
        from pyqcu.cuda import qcu
    except Exception:
        return None
    return qcu


def require_qcu():
    qcu = qcu_or_none()
    if qcu is None:
        raise RuntimeError("QCU single-function test requires CUDA and libqcu.so")
    return qcu


def make_context(
    lattice: Sequence[int] = DEFAULT_LATTICE, mass: float = 0.05,
    *, plan: int = 0, parity: int = 0, max_iter: int = 200,
    device: str | None = None, reporter: RunReporter | None = None,
) -> Context:
    """Create private ABI tensors for one test (never mutate module globals)."""
    lat = tuple(int(v) for v in lattice)
    if len(lat) != 4 or any(v <= 0 or v % 2 for v in lat):
        raise ValueError("lattice dimensions must be positive and even")
    if device is None:
        device = os.environ.get("QCU_DEVICE", "cuda:0")
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA device is unavailable")
    torch.cuda.set_device(dev)
    if reporter is not None:
        reporter.device = dev
    dtype = torch.complex64
    p = torch.zeros(define._PARAMS_SIZE_, dtype=torch.int32)
    p[define._LAT_X_], p[define._LAT_Y_], p[define._LAT_Z_], p[define._LAT_T_] = lat
    p[define._LAT_XYZT_] = int(torch.tensor(lat).prod())
    p[define._GRID_X_], p[define._GRID_Y_], p[define._GRID_Z_], p[define._GRID_T_] = tools.give_grid_size()
    p[define._PARITY_] = int(parity)
    p[define._NODE_RANK_] = define.rank
    p[define._NODE_SIZE_] = define.size
    p[define._DAGGER_] = 0
    p[define._MAX_ITER_] = int(max_iter)
    p[define._DATA_TYPE_] = define._LAT_C64_
    p[define._SET_INDEX_] = 0
    p[define._SET_PLAN_] = int(plan)
    p[define._VERBOSE_] = int(os.environ.get("QCU_VERBOSE", "0"))
    p[define._SEED_] = int(os.environ.get("QCU_SEED", "42"))
    p[define._TEST_IN_CPU_] = 0
    a = torch.zeros(define._ARGV_SIZE_, dtype=torch.float32)
    a[define._MASS_] = float(mass)
    a[define._ATOL_] = float(os.environ.get("QCU_ATOL", "1e-7"))
    a[define._SIGMA_] = 0.1
    s = torch.zeros(define._SET_PTRS_SIZE_, dtype=torch.int64)
    return Context(p, a, s, dev, dtype, lat, float(mass), reporter)


def lat_shape(ctx: Context) -> list[int]:
    return [ctx.lattice[0], ctx.lattice[1], ctx.lattice[2], ctx.lattice[3] // 2]


def allocate_state(ctx: Context, *, clover: bool = False):
    """Allocate QCU parity layout fields and a deterministic random source."""
    shape = lat_shape(ctx)
    g = torch.zeros((2, 3, 3, 4, *shape), dtype=ctx.dtype, device=ctx.device)
    eye = torch.eye(3, dtype=ctx.dtype, device=ctx.device)
    g[:, :, :, :, ...] = 0
    # A unit gauge is deterministic and makes the PyTorch reference independent
    # of the C++ random-number implementation.
    g[:, :, :, :, ...] = eye.view(1, 3, 3, 1, 1, 1, 1, 1)
    gen = torch.Generator(device="cpu").manual_seed(1234)
    f = torch.randn((2, 4, 3, *shape), generator=gen, dtype=torch.float32)
    f = f.to(dtype=ctx.dtype, device=ctx.device).contiguous()
    out = torch.zeros_like(f)
    if not clover:
        return g.contiguous(), f, out
    cshape = (4, 3, 4, 3, *shape)
    ce = torch.zeros(cshape, dtype=ctx.dtype, device=ctx.device)
    co = torch.zeros_like(ce)
    cei = torch.zeros_like(ce)
    coi = torch.zeros_like(ce)
    return g.contiguous(), f, out, ce, co, cei, coi


def lifecycle_call(ctx: Context, name: str, *args):
    """Call one QCU symbol with the required init/index/end protocol."""
    def invoke():
        qcu = require_qcu()
        fn = getattr(qcu, name)
        qcu.applyInitQcu(ctx.set_ptrs, ctx.params, ctx.argv)
        try:
            if ctx.reporter is None:
                return fn(*args, ctx.set_ptrs, ctx.params)
            return ctx.reporter.measure_function(name, fn, *args, ctx.set_ptrs, ctx.params)
        finally:
            # Every operation gets a fresh scratch slot before destruction.
            ctx.params[define._SET_INDEX_] += 1
            qcu.applyEndQcu(ctx.set_ptrs, ctx.params)
    if ctx.reporter is None:
        return invoke()
    return ctx.reporter.run(name, invoke)


def full_gauge(gauge_eo: torch.Tensor) -> torch.Tensor:
    return tools.poooxyzt2oooxyzt(gauge_eo)


def full_fermion(field_eo: torch.Tensor) -> torch.Tensor:
    return tools.poooxyzt2oooxyzt(field_eo)


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denom = float(torch.linalg.vector_norm(expected).detach().cpu())
    if denom == 0.0:
        return float(torch.linalg.vector_norm(actual).detach().cpu())
    return float((torch.linalg.vector_norm(actual - expected) / denom).detach().cpu())


def pure_wilson_reference(field_eo: torch.Tensor, gauge_eo: torch.Tensor,
                          mass: float, *, with_I: bool = True) -> torch.Tensor:
    kappa = 1.0 / (2.0 * mass + 8.0)
    return dslash.give_wilson(full_fermion(field_eo), full_gauge(gauge_eo),
                              torch.tensor([kappa]), with_I=with_I)


def pure_clover_reference(field_eo: torch.Tensor, gauge_eo: torch.Tensor,
                          mass: float) -> torch.Tensor:
    kappa = 1.0 / (2.0 * mass + 8.0)
    u = full_gauge(gauge_eo)
    cl = dslash.make_clover(u, kappa=torch.tensor([kappa]))
    return pure_wilson_reference(field_eo, gauge_eo, mass, with_I=True) + dslash.give_clover(full_fermion(field_eo), cl)


def assert_roundtrip_layout() -> None:
    x = torch.arange(2 * 3 * 4 * 2 * 2 * 2 * 4, dtype=torch.float32).reshape(2, 3, 4, 2, 2, 2, 4).to(torch.complex64)
    assert torch.equal(tools.poooxyzt2oooxyzt(tools.oooxyzt2poooxyzt(x)), x)


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--lat", nargs=4, type=_even_positive, default=DEFAULT_LATTICE, metavar=("X", "Y", "Z", "T"),
                   help="全局格点大小（默认: 16 16 16 16）")
    p.add_argument("--mass", type=float, default=0.05)
    p.add_argument("--device", default=None)
    p.add_argument("--pure-only", action="store_true", help="only run the PyTorch reference")
    p.add_argument("--no-doc", action="store_true", help="不在开头打印本脚本使用说明")
    p.add_argument("--no-progress", action="store_true", help="隐藏分步进度，只保留结果与耗时")
    return p


def cpu_reference_smoke(kind: str, mass: float = 0.05) -> float:
    """Exercise the pure-PyTorch path for a small deterministic field."""
    lat = (2, 2, 2, 4)
    shape = (lat[0], lat[1], lat[2], lat[3] // 2)
    g = torch.zeros((2, 3, 3, 4, *shape), dtype=torch.complex64)
    eye = torch.eye(3, dtype=torch.complex64)
    g[...] = eye.view(1, 3, 3, 1, 1, 1, 1, 1)
    f = torch.arange(2 * 4 * 3 * int(torch.tensor(shape).prod()), dtype=torch.float32).reshape(2, 4, 3, *shape).to(torch.complex64)
    if kind == "wilson":
        y = pure_wilson_reference(f, g, mass, with_I=True)
        return float(torch.linalg.vector_norm(y).item())
    if kind == "clover":
        y = pure_clover_reference(f, g, mass)
        return float(torch.linalg.vector_norm(y).item())
    if kind == "roundtrip":
        assert_roundtrip_layout()
        return 0.0
    raise ValueError(kind)
