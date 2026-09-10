"""Shared helpers for the QCU single-function checks.

The helpers deliberately keep the C++ bridge optional: on a CPU-only machine the
same scripts still exercise the PyTorch reference and layout contracts, while a
CUDA run executes exactly one QCU operation per lifecycle.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import torch

from pyqcu import dslash, tools
from pyqcu.cuda import define


@dataclass
class Context:
    params: torch.Tensor
    argv: torch.Tensor
    set_ptrs: torch.Tensor
    device: torch.device
    dtype: torch.dtype
    lattice: tuple[int, int, int, int]
    mass: float


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
    lattice: Sequence[int] = (4, 4, 4, 8), mass: float = 0.05,
    *, plan: int = 0, parity: int = 0, max_iter: int = 200,
    device: str | None = None,
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
    return Context(p, a, s, dev, dtype, lat, float(mass))


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
    qcu = require_qcu()
    fn = getattr(qcu, name)
    qcu.applyInitQcu(ctx.set_ptrs, ctx.params, ctx.argv)
    try:
        result = fn(*args, ctx.set_ptrs, ctx.params)
    finally:
        # Every operation gets a fresh scratch slot before destruction.
        ctx.params[define._SET_INDEX_] += 1
        qcu.applyEndQcu(ctx.set_ptrs, ctx.params)
    return result


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
    p.add_argument("--lat", nargs=4, type=int, default=(4, 4, 4, 8), metavar=("X", "Y", "Z", "T"))
    p.add_argument("--mass", type=float, default=0.05)
    p.add_argument("--device", default=None)
    p.add_argument("--pure-only", action="store_true", help="only run the PyTorch reference")
    return p


def cpu_reference_smoke(kind: str) -> float:
    """Exercise the pure-PyTorch path for a small deterministic field."""
    lat = (2, 2, 2, 4)
    shape = (lat[0], lat[1], lat[2], lat[3] // 2)
    g = torch.zeros((2, 3, 3, 4, *shape), dtype=torch.complex64)
    eye = torch.eye(3, dtype=torch.complex64)
    g[...] = eye.view(1, 3, 3, 1, 1, 1, 1, 1)
    f = torch.arange(2 * 4 * 3 * int(torch.tensor(shape).prod()), dtype=torch.float32).reshape(2, 4, 3, *shape).to(torch.complex64)
    if kind == "wilson":
        y = pure_wilson_reference(f, g, 0.05, with_I=True)
        return float(torch.linalg.vector_norm(y).item())
    if kind == "clover":
        y = pure_clover_reference(f, g, 0.05)
        return float(torch.linalg.vector_norm(y).item())
    if kind == "roundtrip":
        assert_roundtrip_layout()
        return 0.0
    raise ValueError(kind)
