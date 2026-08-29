"""QUDA 风格 Python MultiGrid 的小格回归与代数验证。

这些测试只使用 CPU、4^4 格点和复数单精度，目标是验证 transfer/coarse
operator 的代数关系与边界约定，而不是测量性能。旧的 ``multigrid`` 仍
作为独立实现导出，故这里也明确检查两者没有互相覆盖。
"""

from __future__ import annotations

from itertools import product

import pytest
import torch

from pyqcu import dslash, lattice, tools
from pyqcu.solver import (
    ParitySchurOperator,
    QudaCoarseOperator,
    QudaMultigrid,
    QudaTransfer,
    multigrid,
)
from pyqcu.solver._gmres import fgmres


DTYPE = torch.complex64
FINE_SHAPE = (4, 4, 4, 4)
BLOCK = (2, 2, 2, 2)


def _random_nulls(seed: int = 123):
    torch.manual_seed(seed)
    first = torch.randn(2, 4, 3, *FINE_SHAPE, dtype=DTYPE)
    second = torch.randn(2, 2, 2, 2, 2, 2, 2, dtype=DTYPE)
    return first, second


def _identity_diagonal(dof: int, shape, value: complex = 1.0):
    diagonal = torch.eye(dof, dtype=DTYPE) * value
    return diagonal.reshape(dof, dof, 1, 1, 1, 1).expand(
        dof, dof, *shape).clone()


def _fine_matvec_with_hop(diagonal, link, backward=None):
    def matvec(value):
        local = torch.einsum("ijxyzt,jxyzt->ixyzt", diagonal, value)
        hop = torch.einsum(
            "ijxyzt,jxyzt->ixyzt", link,
            torch.roll(value, shifts=-1, dims=1))
        if backward is not None:
            hop = hop + torch.einsum(
                "ijxyzt,jxyzt->ixyzt", backward,
                torch.roll(value, shifts=1, dims=1))
        return local + hop

    return matvec


def _apply_qcu_blocked_transfer(blocked, coarse, fine_shape, block_size):
    """用 QCU blocked V 的内存逻辑直接实现 ``P``，供布局回归使用。"""
    E, e = (int(blocked.shape[0]), int(blocked.shape[1]))
    coarse_shape = tuple(n // b for n, b in zip(fine_shape, block_size))
    Xc, Yc, Zc, Tc = coarse_shape
    bx, by, bz, bt = block_size
    fine_blocked = torch.einsum(
        "EeXxYyZzTt,EXYZT->eXxYyZzTt", blocked, coarse)
    return fine_blocked.reshape(e, *fine_shape)


def _restrict_qcu_blocked_transfer(blocked, fine, fine_shape, block_size):
    """用 QCU blocked V 的内存逻辑直接实现 ``R=V^dagger``。"""
    Xc, Yc, Zc, Tc = (n // b for n, b in zip(fine_shape, block_size))
    bx, by, bz, bt = block_size
    fine_blocked = fine.reshape(
        int(fine.shape[0]), Xc, bx, Yc, by, Zc, bz, Tc, bt)
    return torch.einsum(
        "EeXxYyZzTt,eXxYyZzTt->EXYZT", blocked.conj(), fine_blocked)


def _make_full_qcu_blocks(dof, shape):
    """构造支持所有 33 个 canonical 位移的合成粗算子。"""
    identity = torch.eye(dof, dtype=DTYPE).reshape(
        dof, dof, 1, 1, 1, 1).expand(dof, dof, *shape).clone()
    blocks = {(0, 0, 0, 0): identity}
    for displacement in product((-1, 0, 1), repeat=4):
        if displacement == (0, 0, 0, 0) or sum(x != 0 for x in displacement) > 2:
            continue
        torch.manual_seed(1000 + sum((i + 2) * (v + 1)
                                     for i, v in enumerate(displacement)))
        blocks[displacement] = torch.randn(
            dof, dof, *shape, dtype=DTYPE) * 0.01
    return blocks


def test_new_and_legacy_multigrid_are_parallel_exports():
    assert multigrid is not QudaMultigrid


def test_fgmres_zero_rhs_stops_without_nan():
    zero = torch.zeros(12, *FINE_SHAPE, dtype=DTYPE)
    result = fgmres(
        zero, lambda value: 2.0 * value, tol=1e-8, if_rtol=True,
        max_iter=4, verbose=False)
    assert torch.equal(result, zero)


def test_fgmres_breakdown_keeps_singular_case_finite():
    source = torch.ones(12, *FINE_SHAPE, dtype=DTYPE)
    result = fgmres(
        source, lambda value: torch.zeros_like(value), tol=1e-8,
        if_rtol=True, max_iter=2, verbose=False)
    assert torch.isfinite(result).all()


def test_transfer_galerkin_and_quda_yhat_conventions():
    null, _ = _random_nulls()
    transfer = QudaTransfer(null, FINE_SHAPE, block_size=BLOCK)
    coarse = torch.randn(
        transfer.coarse_dof, *transfer.coarse_shape, dtype=DTYPE)
    fine = torch.randn(12, *FINE_SHAPE, dtype=DTYPE)

    rp_error = torch.linalg.norm(
        transfer.restrict(transfer.prolong(coarse)) - coarse
    ) / torch.linalg.norm(coarse)
    lhs = torch.vdot(transfer.prolong(coarse).flatten(), fine.flatten())
    rhs = torch.vdot(coarse.flatten(), transfer.restrict(fine).flatten())
    adjoint_error = torch.abs(lhs - rhs) / (
        torch.linalg.norm(transfer.prolong(coarse)) * torch.linalg.norm(fine))
    assert float(rp_error) < 2e-5
    assert float(adjoint_error) < 2e-5
    assert transfer.orthogonality_error() < 2e-5

    diagonal = _identity_diagonal(12, FINE_SHAPE, value=1.2)
    link = torch.randn(12, 12, *FINE_SHAPE, dtype=DTYPE) * 0.01
    backward = torch.randn(12, 12, *FINE_SHAPE, dtype=DTYPE) * 0.01
    fine_operator = _fine_matvec_with_hop(diagonal, link, backward)
    coarse_operator = QudaCoarseOperator(
        transfer, fine_operator, materialize=True)
    coarse_trial = torch.randn(
        coarse_operator.dof, *coarse_operator.shape, dtype=DTYPE)
    galerkin = transfer.restrict(fine_operator(transfer.prolong(coarse_trial)))
    assert torch.linalg.norm(
        coarse_operator.apply(coarse_trial) - galerkin
    ) / torch.linalg.norm(galerkin) < 2e-5

    expected_pc = torch.einsum(
        "ijxyzt,jxyzt->ixyzt", coarse_operator.X_inv,
        coarse_operator.apply(coarse_trial))
    assert torch.linalg.norm(
        coarse_operator.preconditioned_full_apply(coarse_trial) - expected_pc
    ) / torch.linalg.norm(expected_pc) < 2e-5


def test_qcu_blocked_transfer_layout_matches_python_transfer():
    """验证 QCU 的 10 维 blocked V 与 Python ``P/R`` 的逐元素约定。"""
    null, _ = _random_nulls(20260829)
    transfer = QudaTransfer(null, FINE_SHAPE, block_size=BLOCK)
    blocked = transfer.to_qcu_blocked()
    assert tuple(blocked.shape) == transfer.qcu_blocked_shape

    coarse = torch.randn(
        transfer.coarse_dof, *transfer.coarse_shape, dtype=DTYPE)
    fine = torch.randn(transfer.fine_dof, *transfer.fine_shape, dtype=DTYPE)
    prolong_ref = transfer.prolong(coarse)
    prolong_blocked = _apply_qcu_blocked_transfer(
        blocked, coarse, transfer.fine_shape, transfer.block_size)
    restrict_ref = transfer.restrict(fine)
    restrict_blocked = _restrict_qcu_blocked_transfer(
        blocked, fine, transfer.fine_shape, transfer.block_size)
    assert torch.allclose(prolong_blocked, prolong_ref, rtol=1e-5, atol=1e-5)
    assert torch.allclose(restrict_blocked, restrict_ref, rtol=1e-5, atol=1e-5)


def test_qcu_stencil_pack_matches_all_33_periodic_slots():
    """验证 33 点槽位顺序及 2 点周期维的重复邻居分摊。"""
    null, _ = _random_nulls(20260830)
    transfer = QudaTransfer(null, FINE_SHAPE, block_size=BLOCK)
    operator = QudaCoarseOperator(transfer, lambda value: value)
    operator.blocks = _make_full_qcu_blocks(transfer.coarse_dof, transfer.coarse_shape)

    sit, hop_nn, hop_diag = operator.to_qcu_stencil()
    trial = torch.randn(
        transfer.coarse_dof, *transfer.coarse_shape, dtype=DTYPE)
    packed = tools.apply_stencil(hop_nn, hop_diag, sit, trial)
    direct = operator.apply(trial)
    assert tuple(sit.shape) == (transfer.coarse_dof, transfer.coarse_dof,
                                *transfer.coarse_shape)
    assert tuple(hop_nn.shape) == (2, 4, transfer.coarse_dof,
                                   transfer.coarse_dof, *transfer.coarse_shape)
    assert tuple(hop_diag.shape) == (2, 2, 6, transfer.coarse_dof,
                                     transfer.coarse_dof, *transfer.coarse_shape)
    assert torch.allclose(packed, direct, rtol=1e-5, atol=1e-5)


def test_qcu_stencil_degenerate_extent_and_strict_support_guard():
    """验证 extent=1 折入 sitting，并拒绝无法表达的宽支撑。"""
    null, _ = _random_nulls(20260832)
    transfer = QudaTransfer(null, FINE_SHAPE, block_size=FINE_SHAPE)
    operator = QudaCoarseOperator(transfer, lambda value: value)
    identity = _identity_diagonal(transfer.coarse_dof, transfer.coarse_shape)
    local_shift = identity * 0.25
    operator.blocks = {
        (0, 0, 0, 0): identity,
        (1, 0, 0, 0): local_shift,
    }
    sit, hop_nn, hop_diag = operator.to_qcu_stencil()
    assert torch.allclose(sit, identity + local_shift)
    assert torch.count_nonzero(hop_nn) == 0
    assert torch.count_nonzero(hop_diag) == 0

    wider = QudaTransfer(null, FINE_SHAPE, block_size=(1, 1, 1, 1))
    unsupported = QudaCoarseOperator(wider, lambda value: value)
    unsupported.blocks = {
        (0, 0, 0, 0): _identity_diagonal(
            unsupported.dof, unsupported.shape),
        (2, 0, 0, 0): _identity_diagonal(
            unsupported.dof, unsupported.shape),
    }
    with pytest.raises(ValueError, match="33 点 stencil"):
        unsupported.to_qcu_stencil(strict=True)


def test_multigrid_qcu_transition_assets_are_paired_and_guard_lazy_mode():
    """验证多层导出把每条 P 与下一层 stencil 正确配对。"""
    null, next_null = _random_nulls(20260833)
    diagonal = _identity_diagonal(12, FINE_SHAPE, value=2.0)
    mg = QudaMultigrid(
        fine_matvec=lambda value: 2.0 * value,
        fine_diagonal=diagonal,
        fine_adjoint=lambda value: 2.0 * value,
        lat_size=FINE_SHAPE,
        null_vectors=[null, next_null],
        block_size=[BLOCK, (1, 1, 1, 1)],
        max_level=3,
        materialize_coarse=True,
        use_parity=False,
        verbose=False,
    )
    assets = mg.qcu_transition_assets()
    assert len(assets) == 2
    for level, asset in enumerate(assets):
        transfer = mg.transfers[level]
        operator = mg.operators[level + 1]
        assert asset["level"] == level
        assert tuple(asset["null_vectors"].shape) == transfer.qcu_blocked_shape
        assert tuple(asset["sitting"].shape) == (
            operator.dof, operator.dof, *operator.shape)
        assert asset["stencil"][0] is asset["sitting"]
    lazy = QudaMultigrid(
        fine_matvec=lambda value: 2.0 * value,
        fine_diagonal=diagonal,
        lat_size=FINE_SHAPE,
        null_vectors=[null],
        block_size=BLOCK,
        max_level=2,
        materialize_coarse=False,
        use_parity=False,
        verbose=False,
    )
    with pytest.raises(RuntimeError, match="matrix-free"):
        lazy.qcu_transition_assets()


def test_staggered_transfer_preserves_parity_blocks():
    torch.manual_seed(314159)
    null = torch.randn(2, 1, 3, *FINE_SHAPE, dtype=DTYPE)
    transfer = QudaTransfer(
        null, FINE_SHAPE, fine_spin=1, fine_color=3, coarse_spin=2,
        block_size=BLOCK, spin_block_size=0)
    coarse = torch.randn(
        transfer.coarse_dof, *transfer.coarse_shape, dtype=DTYPE)
    fine = torch.randn(3, *FINE_SHAPE, dtype=DTYPE)

    rp_error = torch.linalg.norm(
        transfer.restrict(transfer.prolong(coarse)) - coarse
    ) / torch.linalg.norm(coarse)
    lhs = torch.vdot(transfer.prolong(coarse).flatten(), fine.flatten())
    rhs = torch.vdot(coarse.flatten(), transfer.restrict(fine).flatten())
    adjoint_error = torch.abs(lhs - rhs) / (
        torch.linalg.norm(transfer.prolong(coarse)) * torch.linalg.norm(fine))
    assert transfer.coarse_spin == 2
    assert transfer.coarse_shape == (2, 2, 2, 2)
    assert float(rp_error) < 2e-5
    assert float(adjoint_error) < 2e-5
    assert transfer.orthogonality_error() < 2e-5


def test_checkerboard_schur_reconstructs_even_equation():
    null, _ = _random_nulls(456)
    transfer = QudaTransfer(null, FINE_SHAPE, block_size=BLOCK)
    diagonal = _identity_diagonal(12, FINE_SHAPE, value=1.4)
    link = torch.randn(12, 12, *FINE_SHAPE, dtype=DTYPE) * 0.005
    backward = torch.randn(12, 12, *FINE_SHAPE, dtype=DTYPE) * 0.005
    coarse_operator = QudaCoarseOperator(
        transfer, _fine_matvec_with_hop(diagonal, link, backward), materialize=True)
    schur = ParitySchurOperator(coarse_operator)
    rhs = torch.randn(coarse_operator.dof, *coarse_operator.shape, dtype=DTYPE)
    odd = torch.randn(
        coarse_operator.dof, schur.checkerboard.volume, dtype=DTYPE)
    reconstructed = schur.reconstruct(rhs, odd)
    residual = rhs - coarse_operator.apply(reconstructed)
    even = schur.checkerboard.extract(residual, 0)
    assert torch.linalg.norm(even) / torch.linalg.norm(rhs) < 2e-5


def test_multilevel_vcycle_and_max_level_one():
    null, next_null = _random_nulls(789)
    diagonal = _identity_diagonal(12, FINE_SHAPE, value=2.0)
    link = torch.randn(12, 12, *FINE_SHAPE, dtype=DTYPE) * 0.002
    backward = torch.randn(12, 12, *FINE_SHAPE, dtype=DTYPE) * 0.002
    matvec = _fine_matvec_with_hop(diagonal, link, backward)
    mg = QudaMultigrid(
        fine_matvec=matvec,
        fine_diagonal=diagonal,
        fine_adjoint=matvec,
        lat_size=FINE_SHAPE,
        null_vectors=[null, next_null],
        block_size=[BLOCK, (1, 1, 1, 1)],
        max_level=3,
        materialize_coarse=True,
        use_parity=True,
        nu_pre=1,
        nu_post=1,
        coarse_max_iter=30,
        coarse_tol=1e-7,
        max_iter=8,
        tol=1e-5,
        verbose=False,
    )
    diagnostics = mg.diagnostics()
    assert diagnostics["transfer_RP"] < 2e-5
    assert diagnostics["galerkin_RDP"] < 2e-5
    source = torch.randn(4, 3, *FINE_SHAPE, dtype=DTYPE)
    solution = mg.solve(source)
    residual = mg.apply(solution) - source
    assert torch.linalg.norm(residual) / torch.linalg.norm(source) < 2e-4

    one_level = QudaMultigrid(
        fine_matvec=matvec,
        fine_diagonal=diagonal,
        fine_adjoint=matvec,
        lat_size=FINE_SHAPE,
        null_vectors=[null],
        block_size=BLOCK,
        max_level=1,
        materialize_coarse=False,
        use_parity=True,
        coarse_max_iter=30,
        coarse_tol=1e-7,
        max_iter=8,
        tol=1e-5,
        verbose=False,
    )
    one_level_solution = one_level.solve(source)
    one_level_residual = one_level.apply(one_level_solution) - source
    assert torch.linalg.norm(one_level_residual) / torch.linalg.norm(source) < 2e-4


def test_matrix_free_mode_keeps_coarse_operator_lazy():
    null, _ = _random_nulls(9753)
    diagonal = _identity_diagonal(12, FINE_SHAPE, value=2.0)

    mg = QudaMultigrid(
        fine_matvec=lambda value: 2.0 * value,
        fine_diagonal=diagonal,
        fine_adjoint=lambda value: 2.0 * value,
        lat_size=FINE_SHAPE,
        null_vectors=[null],
        block_size=BLOCK,
        max_level=2,
        materialize_coarse=False,
        use_parity=False,
        nu_pre=1,
        nu_post=1,
        coarse_max_iter=8,
        coarse_tol=1e-5,
        max_iter=5,
        tol=1e-5,
        verbose=False,
    )
    mg.setup()
    assert mg.operators[1].blocks is None
    source = torch.randn(12, *FINE_SHAPE, dtype=DTYPE)
    solution = mg.solve(source)
    assert torch.isfinite(solution).all()
    assert torch.linalg.norm(mg.apply(solution) - source) / torch.linalg.norm(source) < 2e-4


def test_wilson_and_clover_gauge_setup():
    torch.manual_seed(2468)
    gauge = torch.zeros(3, 3, 4, *FINE_SHAPE, dtype=DTYPE)
    lattice.generate_gauge_field(gauge, seed=2468, sigma=0.02, verbose=False)
    null, _ = _random_nulls(1357)
    clover = torch.zeros(4, 3, 4, 3, *FINE_SHAPE, dtype=DTYPE)
    for spin in range(4):
        for color in range(3):
            clover[spin, color, spin, color] = 0.03

    for clover_term in (None, clover):
        mg = QudaMultigrid(
            U=gauge,
            clover_term=clover_term,
            kappa=torch.tensor([0.1]),
            null_vectors=null,
            block_size=BLOCK,
            max_level=2,
            materialize_coarse=True,
            use_parity=True,
            verbose=False,
        )
        diagnostics = mg.diagnostics()
        assert diagnostics["transfer_RP"] < 2e-5
        assert diagnostics["galerkin_RDP"] < 2e-5
        assert diagnostics["schur_reconstruct_even"] < 3e-4
