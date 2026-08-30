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
    Checkerboard,
    CompactParityLayout,
    CompactParityOperator,
    ParitySchurOperator,
    QudaCoarseOperator,
    QudaMatPCOperator,
    QudaMultigrid,
    QudaStrictMultigrid,
    QudaTransfer,
    multigrid,
)
from pyqcu.solver._quda_multigrid import _FineOperator
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


def _component_diagonal(shape=FINE_SHAPE):
    diagonal = torch.zeros(12, 12, *shape, dtype=DTYPE)
    for component in range(12):
        diagonal[component, component] = float(component + 1)
    return diagonal


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


def _apply_quda_stored_links(links, value, onsite=None):
    """按 QUDA ``Y/Yhat`` 的 forward/backward link 存储约定作用。"""
    result = (torch.zeros_like(value) if onsite is None else
              torch.einsum("ijxyzt,jxyzt->ixyzt", onsite, value))
    for dim in range(4):
        result = result + torch.einsum(
            "ijxyzt,jxyzt->ixyzt", links[0, dim],
            torch.roll(value, shifts=-1, dims=1 + dim))
        backward_at_target = torch.roll(
            links[1, dim], shifts=1, dims=2 + dim).conj().transpose(0, 1)
        result = result + torch.einsum(
            "ijxyzt,jxyzt->ixyzt", backward_at_target,
            torch.roll(value, shifts=1, dims=1 + dim))
    return result


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


def test_single_parity_transfer_is_adjoint_and_keeps_full_coarse_geometry():
    """QUDA 的 parity transfer 只裁 fine 场，coarse 场仍是完整格。"""
    null, _ = _random_nulls(20260846)
    transfer = QudaTransfer(null, FINE_SHAPE, block_size=BLOCK)
    checkerboard = Checkerboard(FINE_SHAPE)
    coarse = torch.randn(
        transfer.coarse_dof, *transfer.coarse_shape, dtype=DTYPE)

    for parity in (0, 1):
        fine_compact = torch.randn(
            transfer.fine_dof, checkerboard.volume, dtype=DTYPE)
        prolong_compact = transfer.prolong_parity(coarse, parity)
        restricted = transfer.restrict_parity(fine_compact, parity)
        lhs = torch.vdot(prolong_compact.flatten(), fine_compact.flatten())
        rhs = torch.vdot(coarse.flatten(), restricted.flatten())
        scale = torch.linalg.norm(prolong_compact) * torch.linalg.norm(fine_compact)

        assert tuple(restricted.shape) == (
            transfer.coarse_dof, *transfer.coarse_shape)
        assert tuple(prolong_compact.shape) == (
            transfer.fine_dof, checkerboard.volume)
        assert float(torch.abs(lhs - rhs) / scale) < 2e-5


def test_quda_matpc_matches_left_preconditioned_block_elimination():
    """逐层 MATPC 必须是 ``I-Hhat_pq Hhat_qp``，而非永久缩格。"""
    null, _ = _random_nulls(20260847)
    transfer = QudaTransfer(null, FINE_SHAPE, block_size=BLOCK)
    diagonal = _component_diagonal()
    link = torch.randn(12, 12, *FINE_SHAPE, dtype=DTYPE) * 0.001
    backward = torch.randn(12, 12, *FINE_SHAPE, dtype=DTYPE) * 0.001
    coarse_operator = QudaCoarseOperator(
        transfer, _fine_matvec_with_hop(diagonal, link, backward),
        materialize=True)

    for parity in (0, 1):
        matpc = QudaMatPCOperator(coarse_operator, parity=parity)
        other = 1 - parity
        trial = torch.randn(
            coarse_operator.dof, matpc.checkerboard.volume, dtype=DTYPE)
        full = matpc.checkerboard.embed(trial, parity, coarse_operator.dof)
        first = coarse_operator.preconditioned_full_apply(full)
        first_other = matpc.checkerboard.extract(first, other)
        first_full = matpc.checkerboard.embed(
            first_other, other, coarse_operator.dof)
        second = coarse_operator.preconditioned_full_apply(first_full)
        expected = trial - matpc.checkerboard.extract(second, parity)
        assert torch.allclose(matpc.apply(trial), expected,
                              rtol=2e-5, atol=2e-6)

        rhs = torch.randn(
            coarse_operator.dof, *coarse_operator.shape, dtype=DTYPE)
        reconstructed = matpc.reconstruct(rhs, trial)
        residual = rhs - coarse_operator.apply(reconstructed)
        eliminated = matpc.checkerboard.extract(residual, other)
        assert float(torch.linalg.norm(eliminated)) / float(
            torch.linalg.norm(rhs)) < 2e-5


def test_strict_quda_hierarchy_coarsens_full_preconditioned_operator():
    """二次粗化保持 spin=2/full geometry，并逐层满足 ``R X^-1 D P``。"""
    torch.manual_seed(20260848)
    diagonal = _component_diagonal()
    link = torch.randn(12, 12, *FINE_SHAPE, dtype=DTYPE) * 0.0005
    backward = torch.randn(12, 12, *FINE_SHAPE, dtype=DTYPE) * 0.0005
    matvec = _fine_matvec_with_hop(diagonal, link, backward)
    first, second = _random_nulls(20260849)
    mg = QudaStrictMultigrid(
        fine_matvec=matvec,
        fine_diagonal=diagonal,
        fine_adjoint=matvec,
        lat_size=FINE_SHAPE,
        null_vectors=[first, second],
        dof_list=[12, 4, 4],
        block_size=[BLOCK, (1, 1, 1, 1)],
        max_level=3,
        materialize_coarse=True,
        use_parity=True,
        setup_iters=0,
        target_parity=0,
        verbose=False,
    ).setup()

    assert [operator.spin for operator in mg.operators] == [4, 2, 2]
    assert [operator.dof for operator in mg.operators] == [12, 4, 4]
    assert [operator.shape for operator in mg.operators] == [
        FINE_SHAPE, (2, 2, 2, 2), (2, 2, 2, 2)]
    assert all(transfer.coarse_spin == 2 for transfer in mg.transfers)

    strict_assets = mg.qcu_strict_transition_assets()
    assert len(strict_assets) == 2
    for level, asset in enumerate(strict_assets):
        transfer = mg.transfers[level]
        child = mg.operators[level + 1]
        assert asset["operator_kind"] == "quda_full_preconditioned"
        assert asset["slot_order"] == (
            "null_vectors", "raw_links", "preconditioned_links",
            "onsite_pair")
        assert tuple(asset["fine_shape"]) == transfer.fine_shape
        assert tuple(asset["coarse_shape"]) == child.shape
        assert tuple(asset["null_vectors"].shape) == transfer.qcu_blocked_shape
        assert tuple(asset["raw_links"].shape) == (
            2, 4, child.dof, child.dof, *child.shape)

    for level, transfer in enumerate(mg.transfers):
        trial = torch.randn(
            transfer.coarse_dof, *transfer.coarse_shape, dtype=DTYPE)
        direct = transfer.restrict(
            mg.coarsening_operators[level].apply(transfer.prolong(trial)))
        actual = mg.operators[level + 1].apply(trial)
        assert float(torch.linalg.norm(actual - direct)) / float(
            torch.linalg.norm(direct)) < 2e-5
        assert mg.operators[level + 1].X is not None
        assert mg.operators[level + 1].X_inv is not None
        assert mg.operators[level + 1].Yhat_forward is not None

    source = torch.randn(4, 3, *FINE_SHAPE, dtype=DTYPE)
    solution = mg.solve(source)
    residual = mg.apply(solution) - source
    assert float(torch.linalg.norm(residual)) / float(
        torch.linalg.norm(source)) < 2e-4


def test_strict_qcu_assets_preserve_quda_y_yhat_storage_and_actions():
    """strict 四槽资产必须逐元素重现 raw D 与 ``X^-1 H``。"""
    null, _ = _random_nulls(20260850)
    transfer = QudaTransfer(null, FINE_SHAPE, block_size=BLOCK)
    diagonal = _component_diagonal()
    link = torch.randn(12, 12, *FINE_SHAPE, dtype=DTYPE) * 0.001
    backward = torch.randn(12, 12, *FINE_SHAPE, dtype=DTYPE) * 0.001
    operator = QudaCoarseOperator(
        transfer, _fine_matvec_with_hop(diagonal, link, backward),
        materialize=True)
    assets = operator.to_qcu_strict_assets()

    E = operator.dof
    shape = operator.shape
    assert tuple(assets["raw_links"].shape) == (2, 4, E, E, *shape)
    assert tuple(assets["preconditioned_links"].shape) == (
        2, 4, E, E, *shape)
    assert tuple(assets["onsite_pair"].shape) == (2, E, E, *shape)
    assert all(value.is_contiguous() for value in assets.values())

    trial = torch.randn(E, *shape, dtype=DTYPE)
    raw = _apply_quda_stored_links(
        assets["raw_links"], trial, assets["onsite_pair"][0])
    preconditioned_hopping = _apply_quda_stored_links(
        assets["preconditioned_links"], trial)
    assert torch.allclose(raw, operator.apply(trial), rtol=2e-5, atol=2e-6)
    assert torch.allclose(
        preconditioned_hopping, operator.preconditioned_apply(trial),
        rtol=2e-5, atol=2e-6)
    assert torch.allclose(assets["onsite_pair"][0], operator.X)
    assert torch.allclose(assets["onsite_pair"][1], operator.X_inv)


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


def test_quda_test_vector_setup_matches_zero_guess_solve():
    """TEST_VECTOR_SETUP 应等价于从零初值求解 ``D B_new = B``。"""
    torch.manual_seed(20260834)
    initial = torch.randn(1, 4, 3, *FINE_SHAPE, dtype=DTYPE)
    diagonal = _identity_diagonal(12, FINE_SHAPE, value=2.0)
    mg = QudaMultigrid(
        fine_matvec=lambda value: 2.0 * value,
        fine_diagonal=diagonal,
        lat_size=FINE_SHAPE,
        null_vectors=initial,
        block_size=BLOCK,
        max_level=2,
        materialize_coarse=False,
        use_parity=False,
        setup_method="test",
        setup_iters=1,
        setup_tol=1e-7,
        setup_max_iter=8,
        setup_post_orthonormalize=False,
        verbose=False,
    )
    mg.setup()
    assert torch.allclose(mg._null_vectors[0], initial / 2.0,
                          rtol=2e-5, atol=2e-6)
    assert mg.setup_history[0]["setup_type"] == "test"
    assert mg.setup_history[0]["operator"] == "full"


@pytest.mark.parametrize(
    ("method", "operator_kind"),
    [("inverse", "full"), ("cg", "normal"), ("ca-cg", "normal"),
     ("krylov", "full"), ("gcr", "full")],
)
def test_quda_setup_solver_family_is_finite(method, operator_kind):
    """QUDA setup solver 家族都应产生有限且可继续 transfer 的基。"""
    torch.manual_seed(20260835)
    initial = torch.randn(1, 4, 3, *FINE_SHAPE, dtype=DTYPE)
    diagonal = _component_diagonal()

    def matvec(value):
        return torch.einsum("ijxyzt,jxyzt->ixyzt", diagonal, value)

    mg = QudaMultigrid(
        fine_matvec=matvec,
        fine_adjoint=matvec,
        fine_diagonal=diagonal,
        lat_size=FINE_SHAPE,
        null_vectors=initial,
        block_size=BLOCK,
        max_level=2,
        materialize_coarse=False,
        use_parity=False,
        setup_method=method,
        setup_iters=1,
        setup_tol=1e-3,
        setup_max_iter=3,
        setup_krylov=2,
        setup_post_orthonormalize=False,
        verbose=False,
    )
    mg.setup()
    generated = mg._null_vectors[0]
    assert torch.isfinite(generated).all()
    assert mg.setup_history[0]["operator"] == operator_kind
    assert mg.transfers[0].orthogonality_error() < 2e-5


def test_quda_normal_setup_requires_custom_adjoint():
    diagonal = _identity_diagonal(12, FINE_SHAPE, value=2.0)
    null, _ = _random_nulls(20260836)
    mg = QudaMultigrid(
        fine_matvec=lambda value: 2.0 * value,
        fine_diagonal=diagonal,
        lat_size=FINE_SHAPE,
        null_vectors=null[:1],
        block_size=BLOCK,
        max_level=2,
        materialize_coarse=False,
        use_parity=False,
        setup_method="cg",
        setup_iters=1,
        setup_max_iter=2,
        verbose=False,
    )
    with pytest.raises(RuntimeError, match="fine_adjoint"):
        mg.setup()


def test_quda_wilson_gamma5_adjoint_enables_normal_setup():
    """物理 Wilson 路径未显式给 adjoint 时自动使用 γ5 D γ5。"""
    gauge = torch.zeros(3, 3, 4, *FINE_SHAPE, dtype=DTYPE)
    for color in range(3):
        gauge[color, color] = 1.0
    null, _ = _random_nulls(20260837)
    mg = QudaMultigrid(
        U=gauge,
        kappa=torch.tensor([0.1]),
        null_vectors=null[:1],
        block_size=BLOCK,
        max_level=2,
        materialize_coarse=False,
        use_parity=False,
        setup_method="cg",
        setup_iters=1,
        setup_max_iter=2,
        setup_tol=1e-3,
        setup_post_orthonormalize=False,
        verbose=False,
    )
    trial = torch.randn(12, *FINE_SHAPE, dtype=DTYPE)
    image = torch.randn_like(trial)
    lhs = torch.vdot(mg._fine.apply(trial).flatten(), image.flatten())
    rhs = torch.vdot(trial.flatten(), mg._fine.adjoint_apply(image).flatten())
    assert float(torch.abs(lhs - rhs)) / (
        float(torch.linalg.norm(mg._fine.apply(trial))) *
        float(torch.linalg.norm(image))) < 2e-5
    mg.setup()
    assert mg._fine_adjoint_kind == "gamma5"
    assert mg.setup_history[0]["operator"] == "normal"


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


def test_compact_parity_layout_roundtrip_matches_qcu_mapping():
    """紧凑 odd/even 布局应与 QCU 的空间 parity-dependent t 配对一致。"""
    layout = CompactParityLayout(FINE_SHAPE)
    field = torch.arange(12 * 4 * 4 * 4 * 4, dtype=torch.float32).reshape(
        12, *FINE_SHAPE)
    even_indices = set(layout.indices[0])
    odd_indices = set(layout.indices[1])
    assert even_indices.isdisjoint(odd_indices)
    assert even_indices | odd_indices == set(range(field[0].numel()))
    for parity in (0, 1):
        compact = layout.extract(field, parity)
        restored = layout.embed(compact, parity, dof=12)
        assert torch.equal(
            restored.reshape(12, -1)[:, layout.indices[parity]],
            field.reshape(12, -1)[:, layout.indices[parity]],
        )
        for x, y, z, t_half in product(*[
                range(extent) for extent in layout.compact_shape]):
            t_full = 2 * t_half + ((parity - x - y - z) & 1)
            assert (x + y + z + t_full) % 2 == parity


def test_compact_schur_matches_dslash_parity_and_adjoint():
    """compact Schur 应与现有 dslash.operator 的 odd 路径严格同构。"""
    gauge = torch.zeros(3, 3, 4, *FINE_SHAPE, dtype=DTYPE)
    for color in range(3):
        gauge[color, color] = 1.0
    dslash_operator = dslash.operator(
        U=gauge, kappa=torch.tensor([0.1]), support_parity=True,
        verbose=False)
    diagonal = _identity_diagonal(12, FINE_SHAPE)
    fine_adjoint = QudaMultigrid._gamma5_hermitian_adjoint(
        dslash_operator.matvec, 4, 3, FINE_SHAPE)
    fine = _FineOperator(
        dslash_operator.matvec, FINE_SHAPE, 4, 3,
        diagonal=diagonal, adjoint=fine_adjoint)
    schur = CompactParityOperator(fine)
    torch.manual_seed(20260840)
    trial = torch.randn(12, *schur.shape, dtype=DTYPE)
    image = torch.randn_like(trial)
    assert torch.allclose(
        schur.apply(trial), dslash_operator.matvec_parity(trial),
        rtol=2e-5, atol=2e-6)
    lhs = torch.vdot(schur.apply(trial).flatten(), image.flatten())
    rhs = torch.vdot(trial.flatten(), schur.adjoint_apply(image).flatten())
    assert float(torch.abs(lhs - rhs)) / (
        float(torch.linalg.norm(schur.apply(trial))) *
        float(torch.linalg.norm(image))) < 2e-5

    full_rhs = torch.randn(12, *FINE_SHAPE, dtype=DTYPE)
    odd_trial = torch.randn(12, *schur.shape, dtype=DTYPE)
    reconstructed = schur.reconstruct(full_rhs, odd_trial)
    residual = full_rhs - fine.apply(reconstructed)
    eliminated = schur.layout.extract(residual, schur.eliminated_parity)
    assert float(torch.linalg.norm(eliminated)) / float(
        torch.linalg.norm(full_rhs)) < 2e-5


def test_compact_test_vector_setup_uses_schur_operator():
    """TEST_VECTOR_SETUP 在 compact 模式应求解 odd ``S B_new = B``。"""
    torch.manual_seed(20260841)
    initial = torch.randn(1, 4, 3, *FINE_SHAPE, dtype=DTYPE)
    diagonal = _identity_diagonal(12, FINE_SHAPE, value=2.0)
    mg = QudaMultigrid(
        fine_matvec=lambda value: 2.0 * value,
        fine_diagonal=diagonal,
        fine_adjoint=lambda value: 2.0 * value,
        lat_size=FINE_SHAPE,
        null_vectors=initial,
        block_size=BLOCK,
        max_level=2,
        materialize_coarse=False,
        use_parity=True,
        setup_operator="schur",
        setup_method="test",
        setup_iters=1,
        setup_tol=1e-7,
        setup_max_iter=8,
        setup_post_orthonormalize=False,
        verbose=False,
    )
    mg.setup()
    initial_flat = initial.reshape(1, 12, *FINE_SHAPE)
    expected = mg._fine_compact.layout.extract_vectors(initial_flat, 1) / 2.0
    assert torch.allclose(mg._null_vectors[0], expected,
                          rtol=2e-5, atol=2e-6)
    assert mg.setup_history[0]["operator"] == "schur"
    assert mg.operators[0].shape == (4, 4, 4, 2)


def test_compact_multilevel_galerkin_solve_and_qcu_assets():
    """compact 多层应递归构造 ``R S_o P``，并可重构 full 解/资产。"""
    torch.manual_seed(20260842)
    diagonal = _identity_diagonal(12, FINE_SHAPE, value=1.7)
    link = torch.randn(12, 12, *FINE_SHAPE, dtype=DTYPE) * 0.003
    backward = torch.randn(12, 12, *FINE_SHAPE, dtype=DTYPE) * 0.003
    matvec = _fine_matvec_with_hop(diagonal, link, backward)
    first, _ = _random_nulls(20260843)
    second = torch.randn(2, 1, 2, 2, 2, 2, 1, dtype=DTYPE)
    mg = QudaMultigrid(
        fine_matvec=matvec,
        fine_diagonal=diagonal,
        fine_adjoint=matvec,
        lat_size=FINE_SHAPE,
        null_vectors=[first, second],
        dof_list=[12, 2, 2],
        block_size=[BLOCK, (1, 1, 1, 1)],
        max_level=3,
        materialize_coarse=True,
        use_parity=True,
        setup_operator="schur",
        setup_iters=0,
        nu_pre=1,
        nu_post=1,
        coarse_max_iter=24,
        coarse_tol=1e-6,
        max_iter=8,
        tol=1e-5,
        verbose=False,
    )
    diagnostics = mg.diagnostics()
    assert diagnostics["transfer_RP"] < 2e-5
    assert diagnostics["galerkin_RDP"] < 2e-5
    assert diagnostics["schur_reconstruct_even"] < 2e-5
    assert all(operator.dof == expected for operator, expected in zip(
        mg.operators, (12, 2, 2)))

    source = torch.randn(4, 3, *FINE_SHAPE, dtype=DTYPE)
    solution = mg.solve(source)
    residual = mg.apply(solution) - source
    assert float(torch.linalg.norm(residual)) / float(
        torch.linalg.norm(source)) < 2e-4
    direct_solution = mg.solve_parity(source, tol=1e-5, max_iter=12)
    direct_residual = mg.apply(direct_solution) - source
    assert float(torch.linalg.norm(direct_residual)) / float(
        torch.linalg.norm(source)) < 2e-4

    assets = mg.qcu_transition_assets()
    assert len(assets) == 2
    for level, asset in enumerate(assets):
        transfer = mg.transfers[level]
        assert asset["compact_parity"] is True
        assert asset["parity"] == 1
        assert asset["eliminated_parity"] == 0
        assert asset["operator_kind"] == (
            "compact_schur" if level == 0 else "compact_galerkin")
        assert tuple(asset["fine_full_shape"]) == FINE_SHAPE
        assert tuple(asset["fine_shape"]) == transfer.fine_shape
        assert asset["fine_dof"] == transfer.fine_dof
        assert asset["coarse_dof"] == transfer.coarse_dof
        assert tuple(asset["null_vectors"].shape) == transfer.qcu_blocked_shape


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


def test_compact_wilson_and_clover_gauge_setup():
    """实际 Wilson/Clover + Gauge 输入也应进入紧凑 odd hierarchy。"""
    torch.manual_seed(20260844)
    gauge = torch.zeros(3, 3, 4, *FINE_SHAPE, dtype=DTYPE)
    lattice.generate_gauge_field(gauge, seed=20260844, sigma=0.02,
                                 verbose=False)
    null, _ = _random_nulls(20260845)
    clover = torch.zeros(4, 3, 4, 3, *FINE_SHAPE, dtype=DTYPE)
    for spin in range(4):
        for color in range(3):
            clover[spin, color, spin, color] = 0.03

    for clover_term in (None, clover):
        mg = QudaMultigrid(
            U=gauge,
            clover_term=clover_term,
            kappa=torch.tensor([0.1]),
            null_vectors=null[:1],
            block_size=BLOCK,
            max_level=2,
            materialize_coarse=False,
            use_parity=True,
            setup_operator="schur",
            setup_iters=0,
            verbose=False,
        )
        diagnostics = mg.diagnostics()
        assert diagnostics["transfer_RP"] < 2e-5
        assert diagnostics["galerkin_RDP"] < 2e-5
        assert diagnostics["schur_reconstruct_even"] < 3e-4
