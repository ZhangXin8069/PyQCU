"""Installation self-test for the packaged PyQCU layers.

The self-test uses only packaged public APIs. Each layer is independent:

* ``torch`` always runs on CPU;
* ``kupl`` runs when the native CPU library can be loaded;
* ``cuda`` runs when the Cython bridge imports and a CUDA device is visible.

A missing native backend is reported as ``SKIP``, not as a failure. Any layer
that runs must pass its independent residual check.
"""

from __future__ import annotations

import time
from typing import Any, Iterable

import torch

from . import __version__


def _validate_lattice(lattice_shape: Iterable[int]) -> tuple[int, int, int, int]:
    shape = tuple(int(value) for value in lattice_shape)
    if len(shape) != 4 or any(value <= 0 or value % 2 for value in shape):
        raise ValueError("self-test lattice must be four positive even integers")
    return shape  # type: ignore[return-value]


def _source(shape: tuple[int, int, int, int], seed: int,
            device: torch.device) -> torch.Tensor:
    compact = (*shape[:3], shape[3] // 2)
    generator = torch.Generator().manual_seed(seed)
    return torch.complex(
        torch.randn((2, 4, 3, *compact), generator=generator),
        torch.randn((2, 4, 3, *compact), generator=generator),
    ).to(device)


def _torch_layer(shape: tuple[int, int, int, int]) -> dict[str, Any]:
    from . import lattice
    from .torch import dslash, solver

    started = time.perf_counter()
    full_gauge = torch.empty((3, 3, 4, *shape), dtype=torch.complex64)
    lattice.generate_gauge_field(
        full_gauge, seed=7, sigma=0.1, verbose=False)
    if not lattice.check_su3(full_gauge, tol=1e-3, verbose=False):
        raise RuntimeError("generated gauge field failed the SU(3) check")
    kappa = torch.tensor([0.125])
    clover = dslash.make_clover(U=full_gauge, kappa=kappa)
    source = torch.randn((4, 3, *shape), dtype=torch.complex64)
    operator = dslash.operator(
        U=full_gauge, clover_term=clover, kappa=kappa, verbose=False)
    history: list[float] = []
    solution = solver.bistabcg(
        b=source, matvec=operator.matvec, tol=1e-5, max_iter=200,
        verbose=False, history=history)
    residual = float(
        torch.linalg.vector_norm(operator.matvec(solution) - source)
        / torch.linalg.vector_norm(source))
    if not residual < 1e-3:
        raise RuntimeError(
            f"Torch BiCGStab residual {residual:.3e} is not below 1e-3")
    return {
        "status": "PASS",
        "backend": "torch",
        "device": "cpu",
        "iterations": max(len(history) - 1, 0),
        "true_residual": residual,
        "seconds": time.perf_counter() - started,
    }


def _kupl_layer(shape: tuple[int, int, int, int]) -> dict[str, Any]:
    try:
        from .kupl import qcu
        from .kupl.linear_operator import KuplLinearOperator
    except Exception as error:
        return {
            "status": "SKIP",
            "backend": "kupl",
            "reason": f"{type(error).__name__}: {error}",
        }

    from . import lattice, tools
    from .api import BiCGStabControl, ResidualScope
    from .cuda import QcuRuntime
    from .solver import Krylov
    from .torch import dslash

    started = time.perf_counter()
    full_gauge = torch.empty((3, 3, 4, *shape), dtype=torch.complex64)
    lattice.generate_gauge_field(
        full_gauge, sigma=0.1, seed=59, verbose=False)
    gauge = tools.oooxyzt2poooxyzt(full_gauge).contiguous()
    source = _source(shape, 61, torch.device("cpu"))
    runtime = QcuRuntime.from_lattice(
        shape, mass=0.17, atol=1e-7, max_iter=200,
        device="cpu", plan=1, grid=(1, 1, 1, 1), qcu_module=qcu)
    operator = KuplLinearOperator(runtime=runtime, gauge=gauge, parity=1)
    result = Krylov(
        backend="kupl", method="bicgstab", operator=operator,
        control=BiCGStabControl(rtol=1e-6, max_iter=200),
    ).solve_result(source)
    full_source = tools.poooxyzt2oooxyzt(source)
    full_solution = tools.poooxyzt2oooxyzt(result.x)
    residual = dslash.give_wilson(
        full_solution, full_gauge, torch.tensor([runtime.kappa]),
        with_I=True) - full_source
    relative = float(
        torch.linalg.vector_norm(residual)
        / torch.linalg.vector_norm(full_source))
    if not result.converged:
        raise RuntimeError("KUPL native BiCGStab did not converge")
    if result.residual_scope is not ResidualScope.SCHUR:
        raise RuntimeError(
            f"KUPL residual scope is {result.residual_scope}, expected SCHUR")
    if not 0 < int(result.krylov_iterations) < 200:
        raise RuntimeError(
            f"KUPL iteration count {result.krylov_iterations} is invalid")
    if not relative < 2.0e-4:
        raise RuntimeError(
            f"KUPL full residual {relative:.3e} is not below 2e-4")
    return {
        "status": "PASS",
        "backend": "kupl",
        "device": "cpu",
        "library": str(qcu.library_path()),
        "iterations": int(result.krylov_iterations),
        "true_residual": relative,
        "residual_scope": str(result.residual_scope),
        "seconds": time.perf_counter() - started,
    }


def _cuda_layer(shape: tuple[int, int, int, int]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "status": "SKIP",
            "backend": "cuda",
            "reason": "torch.cuda.is_available() is False",
        }
    try:
        from .cuda import qcu as cuda_qcu
        from .cuda.linear_operator import CudaLinearOperator
    except Exception as error:
        return {
            "status": "SKIP",
            "backend": "cuda",
            "reason": f"{type(error).__name__}: {error}",
        }
    if not getattr(cuda_qcu, "__file__", None):
        return {
            "status": "SKIP",
            "backend": "cuda",
            "reason": "Cython bridge is not built in this installation",
        }

    from . import lattice, tools
    from .api import BiCGStabControl, ResidualScope
    from .cuda import QcuRuntime
    from .solver import Krylov
    from .torch import dslash

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    started = time.perf_counter()
    full_gauge = torch.empty(
        (3, 3, 4, *shape), dtype=torch.complex64, device=device)
    lattice.generate_gauge_field(
        full_gauge, sigma=0.1, seed=59, verbose=False)
    gauge = tools.oooxyzt2poooxyzt(full_gauge).contiguous()
    source = _source(shape, 61, device)
    runtime = QcuRuntime.from_lattice(
        shape, mass=0.17, atol=1e-7, max_iter=200,
        device=device, plan=1, grid=(1, 1, 1, 1), qcu_module=cuda_qcu)
    operator = CudaLinearOperator(runtime=runtime, gauge=gauge, parity=1)
    result = Krylov(
        backend="cuda", method="bicgstab", operator=operator,
        control=BiCGStabControl(rtol=1e-6, max_iter=200),
    ).solve_result(source)
    torch.cuda.synchronize(device)
    full_source = tools.poooxyzt2oooxyzt(source)
    full_solution = tools.poooxyzt2oooxyzt(result.x)
    residual = dslash.give_wilson(
        full_solution, full_gauge, torch.tensor([runtime.kappa]),
        with_I=True) - full_source
    relative = float(
        torch.linalg.vector_norm(residual)
        / torch.linalg.vector_norm(full_source))
    if not result.converged:
        raise RuntimeError("CUDA native BiCGStab did not converge")
    if result.residual_scope is not ResidualScope.SCHUR:
        raise RuntimeError(
            f"CUDA residual scope is {result.residual_scope}, expected SCHUR")
    iterations = (
        None if result.krylov_iterations is None
        else int(result.krylov_iterations))
    if iterations is not None and not 0 < iterations < 200:
        raise RuntimeError(
            f"CUDA iteration count {result.krylov_iterations} is invalid")
    if not relative < 2.0e-4:
        raise RuntimeError(
            f"CUDA full residual {relative:.3e} is not below 2e-4")
    return {
        "status": "PASS",
        "backend": "cuda",
        "device": torch.cuda.get_device_name(device),
        "capability": ".".join(
            str(value) for value in torch.cuda.get_device_capability(device)),
        "iterations": iterations,
        "true_residual": relative,
        "residual_scope": str(result.residual_scope),
        "seconds": time.perf_counter() - started,
    }


_LAYERS = {
    "torch": _torch_layer,
    "kupl": _kupl_layer,
    "cuda": _cuda_layer,
}


def run_all(
        lattice_shape: Iterable[int] = (4, 4, 4, 4),
        include: Iterable[str] | None = None) -> dict[str, Any]:
    """Run the requested layers and return a JSON-safe report."""
    shape = _validate_lattice(lattice_shape)
    names = tuple(include) if include is not None else tuple(_LAYERS)
    layers: dict[str, dict[str, Any]] = {}
    for name in names:
        function = _LAYERS.get(name)
        if function is None:
            raise ValueError(
                f"unknown self-test layer {name!r}; choose from "
                f"{', '.join(_LAYERS)}")
        try:
            layers[name] = function(shape)
        except Exception as error:
            layers[name] = {
                "status": "FAIL",
                "backend": name,
                "error": f"{type(error).__name__}: {error}",
            }
    status = "FAIL" if any(
        layer["status"] == "FAIL" for layer in layers.values()) else "PASS"
    return {
        "package": "pyqcu",
        "version": __version__,
        "lattice": list(shape),
        "layers": layers,
        "status": status,
    }


__all__ = ["run_all"]
