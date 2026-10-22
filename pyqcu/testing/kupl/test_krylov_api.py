"""Real KUPL whole-solve regression for the unified Krylov facade."""

from __future__ import annotations

import pytest
import torch

from pyqcu import lattice, tools
from pyqcu.api import BiCGStabControl, CGControl, CGOperator, ResidualScope
from pyqcu.cuda import QcuRuntime, define
from pyqcu.kupl.linear_operator import KuplLinearOperator
from pyqcu.solver import Krylov
from pyqcu.torch import dslash


def _kupl_qcu():
    try:
        from pyqcu.kupl import qcu
        return qcu
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(str(exc))


def _full_relative_residual(runtime, full_gauge, full_source, solution):
    full_solution = tools.poooxyzt2oooxyzt(solution)
    residual = dslash.give_wilson(
        full_solution,
        full_gauge,
        torch.tensor([runtime.kappa]),
        with_I=True,
    ) - full_source
    return float(
        torch.linalg.vector_norm(residual)
        / torch.linalg.vector_norm(full_source))


@pytest.mark.skipif(define.size != 1, reason="KUPL API smoke uses one MPI rank")
def test_kupl_krylov_facade_matches_wilson_full_residual():
    qcu = _kupl_qcu()
    lattice_shape = (4, 4, 4, 4)
    compact = (*lattice_shape[:3], lattice_shape[3] // 2)
    full_gauge = torch.empty(
        (3, 3, 4, *lattice_shape), dtype=torch.complex64)
    lattice.generate_gauge_field(
        full_gauge, sigma=0.1, seed=59, verbose=False)
    gauge = tools.oooxyzt2poooxyzt(full_gauge).contiguous()
    generator = torch.Generator().manual_seed(61)
    source = torch.complex(
        torch.randn((2, 4, 3, *compact), generator=generator),
        torch.randn((2, 4, 3, *compact), generator=generator))
    full_source = tools.poooxyzt2oooxyzt(source)
    runtime = QcuRuntime.from_lattice(
        lattice_shape,
        mass=0.17,
        atol=1.0e-7,
        max_iter=200,
        device="cpu",
        plan=1,
        grid=(1, 1, 1, 1),
        qcu_module=qcu,
    )
    operator = KuplLinearOperator(runtime=runtime, gauge=gauge, parity=1)

    bicg = Krylov(
        backend="kupl",
        method="bicgstab",
        operator=operator,
        control=BiCGStabControl(rtol=1.0e-6, max_iter=200),
    ).solve_result(source)
    assert bicg.converged is True
    assert bicg.residual_scope is ResidualScope.SCHUR
    assert 0 < bicg.krylov_iterations < 200
    assert _full_relative_residual(
        runtime, full_gauge, full_source, bicg.x) < 2.0e-4

    cg = Krylov(
        backend="kupl",
        method="cg",
        operator=operator,
        control=CGControl(
            rtol=1.0e-6,
            max_iter=200,
            operator_kind=CGOperator.NORMAL,
        ),
    ).solve_result(source)
    assert cg.converged is True
    assert cg.residual_scope is ResidualScope.NORMAL
    assert 0 < cg.krylov_iterations < 200
    assert _full_relative_residual(
        runtime, full_gauge, full_source, cg.x) < 2.0e-4


@pytest.mark.skipif(define.size != 1, reason="KUPL API smoke uses one MPI rank")
def test_kupl_krylov_facade_clover_bicgstab():
    qcu = _kupl_qcu()
    lattice_shape = (4, 4, 4, 4)
    compact = (*lattice_shape[:3], lattice_shape[3] // 2)
    full_gauge = torch.empty(
        (3, 3, 4, *lattice_shape), dtype=torch.complex64)
    lattice.generate_gauge_field(
        full_gauge, sigma=0.12, seed=67, verbose=False)
    gauge = tools.oooxyzt2poooxyzt(full_gauge).contiguous()
    generator = torch.Generator().manual_seed(71)
    source = torch.complex(
        torch.randn((2, 4, 3, *compact), generator=generator),
        torch.randn((2, 4, 3, *compact), generator=generator))
    runtime = QcuRuntime.from_lattice(
        lattice_shape,
        mass=0.17,
        csw=0.8,
        atol=1.0e-7,
        max_iter=200,
        device="cpu",
        plan=1,
        grid=(1, 1, 1, 1),
        qcu_module=qcu,
    )
    ce, cei = runtime.make_clover(gauge, with_inverse=True, parity=0)
    co, coi = runtime.make_clover(gauge, with_inverse=True, parity=1)
    operator = KuplLinearOperator(
        runtime=runtime, gauge=gauge, parity=1,
        clover=(ce, co, cei, coi))
    result = Krylov(
        backend="kupl",
        method="bicgstab",
        operator=operator,
        control=BiCGStabControl(rtol=1.0e-6, max_iter=200),
    ).solve_result(source)
    assert result.converged is True
    assert result.residual_scope is ResidualScope.SCHUR
    assert 0 < result.krylov_iterations < 200

    full_source = tools.poooxyzt2oooxyzt(source)
    full_solution = tools.poooxyzt2oooxyzt(result.x)
    clover = dslash.make_clover(
        full_gauge, kappa=torch.tensor([runtime.kappa]), csw=0.8)
    fine = dslash.operator(
        U=full_gauge,
        clover_term=clover,
        kappa=torch.tensor([runtime.kappa]),
        support_parity=True,
        verbose=False,
    )
    residual = fine.matvec(full_solution) - full_source
    relative = float(
        torch.linalg.vector_norm(residual)
        / torch.linalg.vector_norm(full_source))
    assert relative < 3.0e-4
