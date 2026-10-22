"""Unified LinearOperator and Krylov driver regression tests."""

from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch

from pyqcu.api import (
    BiCGStabControl,
    CGControl,
    CGOperator,
    CallableLinearOperator,
    LatticeGeometry,
    LinearOperator,
    NormalOperator,
    ResidualScope,
    SolveStatus,
)
from pyqcu.solver import Krylov
from pyqcu.torch import dslash
from pyqcu.torch.dslash import TorchLinearOperator
from pyqcu.torch.solver import TorchBiCGStabDriver, TorchCGDriver


class _FakeRuntime:
    def __init__(self):
        self.params = torch.zeros(58, dtype=torch.int32)
        self.argv = torch.zeros(8, dtype=torch.float32)
        self.lattice = (2, 2, 2, 2)
        self.device = torch.device("cpu")
        self.calls = []

    def call(self, name, *args, synchronize=True):
        self.calls.append(name)
        output = args[0]
        source = args[1]
        output.copy_(source)
        return None

    @contextmanager
    def operation(self, name, synchronize=True):
        self.calls.append(name)

        class Operation:
            def __call__(inner_self, *args):
                args[0].copy_(args[1])

            def read(inner_self, getter):
                return 3

        yield Operation()


def _complex_diagonal(values):
    diagonal = torch.tensor(values, dtype=torch.complex64)

    def apply(value):
        return diagonal.reshape((-1,) + (1,) * (value.ndim - 1)) * value

    return apply


def test_callable_and_normal_operator_expose_uniform_protocol():
    geometry = LatticeGeometry((2, 2, 2, 2), dtype="c64")
    shift = 2.0 + 0.5j
    base = CallableLinearOperator(
        backend="torch",
        geometry=geometry,
        residual_scope=ResidualScope.FULL,
        capabilities={"adjoint": True, "python_krylov": True},
        apply_fn=lambda value, parity: shift * value,
        adjoint_fn=lambda value, parity: shift.conjugate() * value,
        normal_fn=lambda value, parity: abs(shift) ** 2 * value,
    )
    assert isinstance(base, LinearOperator)
    assert base.residual_scope is ResidualScope.FULL

    normal = NormalOperator(base)
    value = torch.ones(8, dtype=torch.complex64)
    assert normal.residual_scope is ResidualScope.NORMAL
    assert torch.allclose(
        normal.apply(value), abs(shift) ** 2 * value)
    assert normal.capabilities["hpd_cg"] is True


def test_torch_bicgstab_and_cg_return_solve_result_contract():
    geometry = LatticeGeometry((2, 2, 2, 2), dtype="c64")
    diagonal = CallableLinearOperator(
        backend="torch",
        geometry=geometry,
        residual_scope=ResidualScope.FULL,
        capabilities={"adjoint": True, "hpd_cg": True, "python_krylov": True},
        apply_fn=lambda value, parity: _complex_diagonal(
            [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])(value),
        adjoint_fn=lambda value, parity: _complex_diagonal(
            [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])(value),
        normal_fn=lambda value, parity: _complex_diagonal(
            [9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0])(value),
    )
    rhs = torch.arange(1, 9).to(dtype=torch.complex64)
    expected = rhs / torch.arange(3, 11).to(dtype=torch.complex64)

    bicg = TorchBiCGStabDriver(
        diagonal, BiCGStabControl(rtol=1e-6, max_iter=20))
    bicg_result = bicg.solve_result(rhs)
    assert bicg_result.status is SolveStatus.CONVERGED
    assert bicg_result.converged is True
    assert bicg_result.krylov_iterations is not None
    assert bicg_result.final_residual is not None
    assert bicg_result.residual_scope is ResidualScope.FULL
    torch.testing.assert_close(bicg_result.x, expected, rtol=2e-5, atol=2e-6)

    cg = TorchCGDriver(diagonal, CGControl(rtol=1e-6, max_iter=20))
    cg_result = cg.solve_result(rhs)
    assert cg_result.status is SolveStatus.CONVERGED
    torch.testing.assert_close(cg_result.x, expected, rtol=2e-5, atol=2e-6)

    selected = Krylov(
        backend="torch",
        method="bistabcg",
        operator=diagonal,
        control=BiCGStabControl(rtol=1e-6, max_iter=20),
    ).solve_result(rhs)
    assert selected.method == "bicgstab"
    torch.testing.assert_close(selected.x, expected, rtol=2e-5, atol=2e-6)

    with pytest.raises(ValueError, match="hpd_cg"):
        TorchCGDriver(
            CallableLinearOperator(
                backend="torch",
                geometry=geometry,
                capabilities={"hpd_cg": False},
                apply_fn=lambda value, parity: value,
            ),
            CGControl(),
        ).solve_result(rhs)


def test_torch_linear_operator_full_and_gamma5_adjoint():
    shape = (2, 2, 2, 2)
    gauge = torch.zeros(3, 3, 4, *shape, dtype=torch.complex64)
    for color in range(3):
        gauge[color, color] = 1.0
    fine = dslash.operator(
        U=gauge, kappa=torch.tensor([0.12]), support_parity=True,
        verbose=False)
    operator = TorchLinearOperator(fine)
    assert isinstance(operator, LinearOperator)
    assert operator.residual_scope is ResidualScope.FULL

    left = torch.randn(12, *shape, dtype=torch.complex64)
    right = torch.randn(12, *shape, dtype=torch.complex64)
    lhs = torch.vdot(left.flatten(), operator.apply(right).flatten())
    rhs = torch.vdot(
        operator.apply_adjoint(left).flatten(), right.flatten())
    torch.testing.assert_close(lhs, rhs, rtol=2e-5, atol=2e-6)


def test_torch_normal_operator_cg_solves_explicit_normal_system():
    geometry = LatticeGeometry((2, 2, 2, 2), dtype="c64")
    shift = torch.tensor(
        [2.0 + 1.0j, 3.0 - 2.0j] * 4, dtype=torch.complex64)

    def apply(value, parity):
        return shift * value

    def adjoint(value, parity):
        return shift.conj() * value

    base = CallableLinearOperator(
        backend="torch",
        geometry=geometry,
        residual_scope=ResidualScope.FULL,
        capabilities={"adjoint": True},
        apply_fn=apply,
        adjoint_fn=adjoint,
    )
    normal = NormalOperator(base)
    target = torch.ones(8, dtype=torch.complex64)
    rhs = base.apply_adjoint(target)
    result = TorchCGDriver(
        normal,
        CGControl(
            rtol=1e-6,
            max_iter=30,
            operator_kind=CGOperator.NORMAL,
            residual_scope=ResidualScope.NORMAL,
        ),
    ).solve_result(rhs)
    assert result.status is SolveStatus.CONVERGED
    assert result.residual_scope is ResidualScope.NORMAL
    expected = target / shift
    torch.testing.assert_close(result.x, expected, rtol=2e-5, atol=2e-6)


def test_cuda_and_kupl_operator_adapters_apply_without_python_krylov_loop():
    from pyqcu.cuda.linear_operator import CudaLinearOperator
    from pyqcu.kupl.linear_operator import KuplLinearOperator

    source = torch.ones(12, 2, 2, 2, 1, dtype=torch.complex64)
    gauge = torch.ones(2, 3, 3, 4, 2, 2, 2, 1, dtype=torch.complex64)
    for cls, backend in ((CudaLinearOperator, "cuda"),
                         (KuplLinearOperator, "kupl")):
        runtime = _FakeRuntime()
        operator = cls(runtime=runtime, gauge=gauge)
        output = operator.apply(source)
        assert torch.equal(output, source)
        assert runtime.calls == ["applyWilsonBistabCgDslashQcu"]
        capabilities = operator.capability_report()
        assert capabilities.backend == backend
        assert capabilities.python_krylov is False


def test_cuda_and_kupl_native_driver_call_one_whole_solve():
    from pyqcu.cuda.linear_operator import CudaLinearOperator
    from pyqcu.kupl.linear_operator import KuplLinearOperator

    source = torch.ones(2, 4, 3, 2, 2, 2, 1, dtype=torch.complex64) * 2.0
    gauge = torch.ones(2, 3, 3, 4, 2, 2, 2, 1, dtype=torch.complex64)
    for cls, backend, solver_name, expected_iterations in (
            (CudaLinearOperator, "cuda", "applyWilsonBistabCgQcu", None),
            (KuplLinearOperator, "kupl", "applyWilsonBistabCgQcu", 3)):
        runtime = _FakeRuntime()
        operator = cls(runtime=runtime, gauge=gauge)
        driver = Krylov(
            backend=backend,
            method="bicgstab",
            operator=operator,
            control=BiCGStabControl(rtol=1e-6, max_iter=10),
        )
        result = driver.solve_result(source)
        assert result.status is SolveStatus.CONVERGED
        assert result.krylov_iterations == expected_iterations
        assert result.final_residual is None
        assert result.residual_scope is ResidualScope.SCHUR
        assert solver_name in runtime.calls
        assert driver.capabilities.native_krylov is True
        assert driver.capabilities.python_krylov is False

    cuda_runtime = _FakeRuntime()
    cuda_operator = CudaLinearOperator(runtime=cuda_runtime, gauge=gauge)
    cg_result = Krylov(
        backend="cuda",
        method="cg",
        operator=cuda_operator,
        control=CGControl(
            rtol=1e-6, max_iter=10, operator_kind=CGOperator.NORMAL),
    ).solve_result(source)
    assert cg_result.status is SolveStatus.CONVERGED
    assert cg_result.residual_scope is ResidualScope.NORMAL
    assert "applyWilsonCgQcu" in cuda_runtime.calls
    with pytest.raises(ValueError, match="NORMAL"):
        Krylov(
            backend="cuda",
            method="cg",
            operator=cuda_operator,
            control=CGControl(
                rtol=1e-6, max_iter=10, operator_kind=CGOperator.HPD),
        ).solve_result(source)
    with pytest.raises(ValueError, match="relative tolerance"):
        Krylov(
            backend="cuda",
            method="bicgstab",
            operator=cuda_operator,
            control=BiCGStabControl(rtol=1e-6, atol=1e-8, max_iter=10),
        ).solve_result(source)


def test_cuda_clover_operator_uses_clover_bicgstab_only():
    from pyqcu.cuda.linear_operator import CudaLinearOperator

    runtime = _FakeRuntime()
    gauge = torch.ones(2, 3, 3, 4, 2, 2, 2, 1, dtype=torch.complex64)
    clover = tuple(
        torch.ones(4, 3, 4, 3, 2, 2, 2, 1, dtype=torch.complex64)
        for _ in range(4))
    operator = CudaLinearOperator(
        runtime=runtime, gauge=gauge, parity=1, clover=clover)
    compact = torch.ones(4, 3, 2, 2, 2, 1, dtype=torch.complex64)
    operator.apply(compact)
    assert runtime.calls[-1] == "applyCloverBistabCgDslashQcu"

    source = torch.ones(2, 4, 3, 2, 2, 2, 1, dtype=torch.complex64)
    result = Krylov(
        backend="cuda",
        method="bicgstab",
        operator=operator,
        control=BiCGStabControl(rtol=1e-6, max_iter=10),
    ).solve_result(source)
    assert result.status is SolveStatus.CONVERGED
    assert result.krylov_iterations == 3
    assert "applyCloverBistabCgQcu" in runtime.calls
    with pytest.raises(ValueError, match="cg=True"):
        Krylov(
            backend="cuda",
            method="cg",
            operator=operator,
            control=CGControl(
                rtol=1e-6, max_iter=10, operator_kind=CGOperator.NORMAL),
        )


def test_krylov_selector_rejects_unknown_backend_and_method():
    operator = object()
    with pytest.raises(ValueError, match="backend"):
        Krylov(
            backend="unknown", method="bicgstab", operator=operator,
            control=BiCGStabControl())
    with pytest.raises(ValueError, match="method"):
        Krylov(
            backend="cuda", method="unknown", operator=operator,
            control=BiCGStabControl())
