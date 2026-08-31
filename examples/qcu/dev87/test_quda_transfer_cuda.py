"""小格 CUDA 回归：QUDA 风格 Python transfer/stencil 到 QCU kernel。

该测试不依赖外部 QUDA 二进制；它用同一个 ``QudaTransfer`` 生成正交化
基，用合成的 33 点粗算子验证两个边界最容易出错的接口：

* Python ``[spin,color,coarse_spin,nvec,X,Y,Z,T]`` 到 QCU
  ``[E,e,Xc,bx,Yc,by,Zc,bz,Tc,bt]``；
* 目标点位移到 C++ ``sit/hop_nn/hop_diag``，包括粗格尺寸为 2 时的
  ``+1/-1`` 重合邻点分摊。

运行：

    source ./env.sh
    pytest -q examples/qcu/dev87/test_quda_transfer_cuda.py
"""

from __future__ import annotations

from itertools import product
import weakref

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("需要 CUDA 设备", allow_module_level=True)

try:
    from pyqcu.cuda import CudaStrictMultigridSolver, define, qcu
    from pyqcu.solver import (Checkerboard, CompactParityLayout,
                              QudaCoarseOperator, QcuStrictAssetBinding,
                              QudaMatPCOperator, QudaStrictMultigrid,
                              QudaTransfer)
except (ImportError, OSError) as exc:  # pragma: no cover - 环境相关
    pytest.skip(f"QCU Cython/CUDA 后端不可用: {exc}", allow_module_level=True)


DTYPE = torch.complex64
FINE_SHAPE = (4, 4, 4, 4)
BLOCK = (2, 2, 2, 2)


def _params(lat, E, coarse, index):
    p = define.params.clone()
    p[define._LAT_X_] = lat[0]
    p[define._LAT_Y_] = lat[1]
    p[define._LAT_Z_] = lat[2]
    p[define._LAT_T_] = lat[3]
    p[define._LAT_XYZT_] = int(torch.tensor(lat).prod().item())
    p[define._GRID_X_] = p[define._GRID_Y_] = 1
    p[define._GRID_Z_] = p[define._GRID_T_] = 1
    p[define._NODE_RANK_] = 0
    p[define._NODE_SIZE_] = 1
    p[define._DATA_TYPE_] = define._LAT_C64_
    p[define._SET_INDEX_] = index
    p[define._SET_PLAN_] = 1
    p[define._VERBOSE_] = 0
    p[define._MG_NUM_LEVEL_] = 2
    p[define._MG_LEVEL1_E_] = E
    p[define._MG_LEVEL1_X_] = coarse[0]
    p[define._MG_LEVEL1_Y_] = coarse[1]
    p[define._MG_LEVEL1_Z_] = coarse[2]
    p[define._MG_LEVEL1_T_] = coarse[3]
    a = define.argv.clone()
    return p, a


def _full_blocks(dof, shape, device):
    identity = torch.eye(dof, dtype=DTYPE).reshape(
        dof, dof, 1, 1, 1, 1).expand(dof, dof, *shape).clone()
    blocks = {(0, 0, 0, 0): identity.to(device=device)}
    for displacement in product((-1, 0, 1), repeat=4):
        if displacement == (0, 0, 0, 0):
            continue
        if sum(value != 0 for value in displacement) > 2:
            continue
        seed = 7000 + sum((axis + 3) * (value + 2)
                          for axis, value in enumerate(displacement))
        torch.manual_seed(seed)
        blocks[displacement] = (
            torch.randn(dof, dof, *shape, dtype=DTYPE) * 0.01
        ).to(device=device)
    return blocks


def _relative(a, b):
    denominator = torch.linalg.norm(b).item()
    return float(torch.linalg.norm(a - b).item() / max(denominator, 1e-30))


def _synthetic_strict_operator(shape, seed):
    """构造对角占优的 full-coarse ``X+Y``，避免测试依赖 fine Clover。"""
    dof = 4
    torch.manual_seed(seed)
    metadata_null = torch.randn(2, 2, 2, *shape, dtype=DTYPE)
    metadata_transfer = QudaTransfer(
        metadata_null, shape, fine_spin=2, fine_color=2,
        coarse_spin=2, block_size=(1, 1, 1, 1))
    operator = QudaCoarseOperator(metadata_transfer, lambda value: value)
    identity = torch.eye(dof, dtype=DTYPE).reshape(
        dof, dof, 1, 1, 1, 1).expand(dof, dof, *shape).clone()
    operator.blocks = {(0, 0, 0, 0): 1.7 * identity}
    for dim in range(4):
        plus = tuple(1 if axis == dim else 0 for axis in range(4))
        operator.blocks[plus] = -0.01 * identity
        if shape[dim] > 2:
            minus = tuple(-1 if axis == dim else 0 for axis in range(4))
            operator.blocks[minus] = -0.01 * identity
    operator.X = operator.blocks[(0, 0, 0, 0)]
    operator._build_links()
    return operator


def _strict_hierarchy(depth):
    first = _synthetic_strict_operator((4, 4, 4, 4), 20260860)
    operators = [first]
    transfers = []
    if depth == 2:
        torch.manual_seed(20260861)
        null = torch.randn(2, 2, 2, 4, 4, 4, 4, dtype=DTYPE)
        transfer = QudaTransfer(
            null, first.shape, fine_spin=2, fine_color=2,
            coarse_spin=2, block_size=(2, 2, 2, 2))
        child = QudaCoarseOperator(
            transfer, first.preconditioned_full_apply, materialize=True)
        operators.append(child)
        transfers.append(transfer)
    return operators, transfers


def _strict_params(operators, index=0, parity=0, smoother_steps=2):
    params, argv = _params(
        operators[0].shape, operators[0].dof, operators[0].shape, index)
    params[define._PARITY_] = parity
    params[define._MG_NUM_LEVEL_] = len(operators) + 1
    params[define._MG_MU_PRE_] = smoother_steps
    for offset, operator in enumerate(operators):
        base = define._MG_LEVEL1_E_ + offset * define._MG_PARAMS_SIZE_
        params[base:base + 8] = torch.tensor([
            operator.dof, *operator.shape, 200, define._LAT_C64_, 0],
            dtype=params.dtype)
        argv[define._MG_LEVEL1_ATOL_ + offset] = 1.0e-6
    return params, argv


def _bind_strict_assets(set_ptrs, operators, transfers, device):
    """只驻留运行期必需的 ``V/Yhat/(X,Xinv)``，raw ``Y`` 保持空槽。"""
    assets = []
    for transition, operator in enumerate(operators):
        packed = operator.to_qcu_strict_assets(
            device=device, include_raw_links=False)
        packed["null_vectors"] = None
        if transition > 0:
            packed["null_vectors"] = transfers[transition - 1].to_qcu_blocked(
                device=device)
        assets.append(packed)
    return QcuStrictAssetBinding(
        set_ptrs, assets, start_level=1, retain_raw_links=False)


def _reference_strict_vcycle(operators, transfers, full_rhs, parity,
                             smoother_steps, level=0):
    matpc = QudaMatPCOperator(operators[level], parity=parity)
    pc_rhs = matpc.rhs(full_rhs)
    if level == len(operators) - 1:
        target = matpc.solve(
            pc_rhs, tol=1.0e-8, max_iter=200, direct_solve_max=4096)
        return matpc.reconstruct(full_rhs, target)

    target = torch.zeros_like(pc_rhs)
    residual = pc_rhs.clone()

    def smooth():
        nonlocal target, residual
        for _ in range(smoother_steps):
            image = matpc.apply(residual)
            denominator = torch.vdot(image.reshape(-1), image.reshape(-1))
            if float(torch.abs(denominator)) <= 1.0e-20:
                break
            alpha = (torch.vdot(image.reshape(-1), residual.reshape(-1)) /
                     denominator)
            target = target + alpha * residual
            residual = residual - alpha * image

    smooth()
    child_rhs = transfers[level].restrict_parity(residual, parity)
    child_correction = _reference_strict_vcycle(
        operators, transfers, child_rhs, parity, smoother_steps, level + 1)
    target = target + transfers[level].prolong_parity(
        child_correction, parity)
    residual = pc_rhs - matpc.apply(target)
    smooth()
    return matpc.reconstruct(full_rhs, target)


def _expected_strict_workspace_bytes(operators):
    # CUDA 实现按 256 B 对齐 complex64 arena 切片。
    alignment = 256 // torch.empty((), dtype=DTYPE).element_size()

    def aligned(elements):
        return ((elements + alignment - 1) // alignment) * alignment

    compact = [operator.dof * int(torch.tensor(operator.shape).prod()) // 2
               for operator in operators]
    full = [2 * value for value in compact]
    persistent = sum(2 * aligned(value) for value in compact)
    persistent += sum(aligned(value) for value in full[1:])
    arena = 3 * aligned(max(compact)) + 4 * aligned(compact[-1])
    arena += aligned(max(full[1:], default=0))
    # strict_dot_{pair,many} uses one partial complex value per block.  The
    # fine-grid reduction length is the full fine vector (2*compact[0]); the
    # coarse reductions use compact lengths.  Keep this mirror of
    # strict_reduction_blocks() in sync with the CUDA arena accounting.
    reduction_n = max([full[0], *compact])
    reduction_blocks = max(1, min(1024, (reduction_n + 256 * 8 - 1) // (256 * 8)))
    arena += aligned(2 * reduction_blocks)
    return (persistent + arena) * torch.empty((), dtype=DTYPE).element_size()


def _identity_fine_clover_fields(shape, device):
    compact_shape = (*shape[:3], shape[3] // 2)
    gauge = torch.zeros(
        2, 3, 3, 4, *compact_shape, dtype=DTYPE)
    for parity in range(2):
        for color in range(3):
            gauge[parity, color, color] = 1.0
    clover = torch.zeros(
        4, 3, 4, 3, *compact_shape, dtype=DTYPE)
    for spin in range(4):
        for color in range(3):
            clover[spin, color, spin, color] = 1.0
    return (gauge.to(device), clover.to(device), clover.to(device),
            clover.to(device), clover.to(device))


def _nontrivial_fine_clover_fields(shape, device):
    """Build positive-definite, site-dependent Clover blocks for MATPC gates."""
    compact_shape = (*shape[:3], shape[3] // 2)
    sites = int(torch.tensor(compact_shape).prod().item())
    generator = torch.Generator().manual_seed(20260873)
    random = torch.randn(
        sites, 12, 12, dtype=DTYPE, generator=generator)
    hermitian = random + random.conj().transpose(-1, -2)
    identity = torch.eye(12, dtype=DTYPE).expand(sites, -1, -1)
    even = 1.5 * identity + 0.01 * hermitian
    odd = 1.35 * identity - 0.008 * hermitian

    def as_field(matrix):
        return matrix.permute(1, 2, 0).reshape(
            12, 12, *compact_shape).contiguous()

    even_field = as_field(even)
    odd_field = as_field(odd)
    even_inverse = as_field(torch.linalg.inv(even))
    odd_inverse = as_field(torch.linalg.inv(odd))
    gauge = torch.zeros(2, 3, 3, 4, *compact_shape, dtype=DTYPE)
    diagonal = torch.arange(3)
    gauge[:, diagonal, diagonal] = 1.0
    return (
        gauge.to(device),
        even_field.reshape(4, 3, 4, 3, *compact_shape).to(device),
        odd_field.reshape(4, 3, 4, 3, *compact_shape).to(device),
        even_inverse.reshape(4, 3, 4, 3, *compact_shape).to(device),
        odd_inverse.reshape(4, 3, 4, 3, *compact_shape).to(device),
        even_field,
        odd_field,
    )


def _fine_full_clover_term(shape, even_field, odd_field):
    full = torch.empty(12, 12, *shape, dtype=DTYPE)
    for x, y, z, t in product(*(range(extent) for extent in shape)):
        source = even_field if ((x + y + z + t) & 1) == 0 else odd_field
        full[:, :, x, y, z, t] = source[:, :, x, y, z, t // 2]
    identity = torch.eye(12, dtype=DTYPE).reshape(
        12, 12, 1, 1, 1, 1)
    return (full - identity).reshape(4, 3, 4, 3, *shape).contiguous()


def test_quda_transfer_and_stencil_match_qcu_kernels():
    # 先在 CPU 生成随机输入，再搬到 CUDA；这样也兼容没有对应 PyTorch
    # architecture kernel 的旧卡，真正需要 GPU kernel 的仅是 QCU/参考运算。
    torch.manual_seed(20260831)
    device = torch.device("cuda")
    null = torch.randn(2, 4, 3, *FINE_SHAPE, dtype=DTYPE).to(device)
    transfer = QudaTransfer(null, FINE_SHAPE, block_size=BLOCK)
    blocked = transfer.to_qcu_blocked()
    operator = QudaCoarseOperator(transfer, lambda value: value)
    operator.blocks = _full_blocks(
        transfer.coarse_dof, transfer.coarse_shape, device)
    sit, hop_nn, hop_diag = operator.to_qcu_stencil()

    fine = torch.randn(12, *FINE_SHAPE, dtype=DTYPE).to(device)
    coarse = torch.randn(
        transfer.coarse_dof, *transfer.coarse_shape, dtype=DTYPE).to(device)
    direct_fine = transfer.prolong(coarse)
    direct_coarse = transfer.restrict(fine)

    set_ptrs = define.set_ptrs.clone()
    initialized = []
    try:
        # 每个 C++ 调用使用独立 set_index；这遵守 applyInit/applyEnd 的
        # 生命周期约定，也避免把两个不同 operation 的 scratch 混用。
        p_restrict, av = _params(FINE_SHAPE, transfer.coarse_dof,
                                 transfer.coarse_shape, 0)
        p_prolong, _ = _params(FINE_SHAPE, transfer.coarse_dof,
                               transfer.coarse_shape, 1)
        p_wide, _ = _params(transfer.coarse_shape, transfer.coarse_dof,
                            transfer.coarse_shape, 2)
        for params in (p_restrict, p_prolong, p_wide):
            qcu.applyInitQcu(set_ptrs, params, av)
            initialized.append(params)

        coarse_cpp = torch.empty_like(coarse)
        fine_cpp = torch.empty_like(fine)
        qcu.applyMultigridRestrictQcu(
            coarse_cpp, fine, blocked, set_ptrs, p_restrict)
        qcu.applyMultigridProLongQcu(
            fine_cpp, coarse, blocked, set_ptrs, p_prolong)
        assert _relative(coarse_cpp, direct_coarse) < 2e-5
        assert _relative(fine_cpp, direct_fine) < 2e-5

        coarse_cpp = torch.empty_like(coarse)
        qcu.applyMultigridCoarseDslashWideQcu(
            coarse_cpp, coarse, sit, hop_nn, hop_diag, set_ptrs, p_wide)
        assert _relative(coarse_cpp, operator.apply(coarse)) < 2e-5
    finally:
        for params in reversed(initialized):
            qcu.applyEndQcu(set_ptrs, params)
        torch.cuda.synchronize()


def test_quda_strict_transfer_coarse_and_matpc_match_cuda_kernels():
    """strict ``P/R``, ``X/Y/Yhat`` 与逐层 MATPC 的逐元素 CUDA 闭环。"""
    torch.manual_seed(20260850)
    device = torch.device("cuda")
    null = torch.randn(2, 4, 3, *FINE_SHAPE, dtype=DTYPE).to(device)
    transfer = QudaTransfer(null, FINE_SHAPE, block_size=BLOCK)
    blocked = transfer.to_qcu_blocked()

    E = transfer.coarse_dof
    shape = transfer.coarse_shape
    operator = QudaCoarseOperator(transfer, lambda value: value)
    operator.blocks = {}
    identity = torch.eye(E, dtype=DTYPE, device=device).reshape(
        E, E, 1, 1, 1, 1).expand(E, E, *shape).clone()
    operator.blocks[(0, 0, 0, 0)] = 1.7 * identity
    for dim in range(4):
        # extent=2 时 +mu/-mu 是同一 canonical neighbour；materialize
        # 后只有一个合并 block，_build_links 会把它均分给两个方向。
        displacement = [0, 0, 0, 0]
        displacement[dim] = 1
        torch.manual_seed(20260851 + dim)
        operator.blocks[tuple(displacement)] = (
            torch.randn(E, E, *shape, dtype=DTYPE) * 0.003
        ).to(device)
    operator.X = operator.blocks[(0, 0, 0, 0)]
    operator._build_links()
    assets = operator.to_qcu_strict_assets()

    set_ptrs = define.set_ptrs.clone()
    initialized = []

    def initialize(lat, coarse, index):
        params, av = _params(lat, E, coarse, index)
        qcu.applyInitQcu(set_ptrs, params, av)
        initialized.append(params)
        return params

    try:
        trial = torch.randn(E, *shape, dtype=DTYPE).to(device)
        raw_cpp = torch.empty_like(trial)
        p_raw = initialize(shape, shape, 0)
        qcu.applyMultigridStrictCoarseQcu(
            raw_cpp, trial, assets["raw_links"], assets["onsite_pair"],
            set_ptrs, p_raw, 0)
        assert _relative(raw_cpp, operator.apply(trial)) < 2e-5

        hopping_cpp = torch.empty_like(trial)
        p_hop = initialize(shape, shape, 1)
        qcu.applyMultigridStrictCoarseQcu(
            hopping_cpp, trial, assets["preconditioned_links"],
            assets["onsite_pair"], set_ptrs, p_hop, -1)
        assert _relative(
            hopping_cpp, operator.preconditioned_apply(trial)) < 2e-5

        checkerboard = Checkerboard(shape)
        for parity in (0, 1):
            compact = checkerboard.extract(trial, parity).reshape(
                E, shape[0], shape[1], shape[2], shape[3] // 2).contiguous()
            matpc = QudaMatPCOperator(operator, parity=parity)
            matpc_cpp = torch.empty_like(compact)
            scratch = torch.empty_like(compact)
            p_matpc = initialize(shape, shape, len(initialized))
            qcu.applyMultigridStrictMatPCQcu(
                matpc_cpp, compact, assets["preconditioned_links"], scratch,
                set_ptrs, p_matpc, parity)
            expected = matpc.apply(
                compact.reshape(E, -1)).reshape_as(compact)
            assert _relative(matpc_cpp, expected) < 2e-5

            full_rhs = torch.randn(E, *shape, dtype=DTYPE).to(device)
            prepared_cpp = torch.empty_like(compact)
            p_prepare = initialize(shape, shape, len(initialized))
            qcu.applyMultigridStrictPrepareQcu(
                prepared_cpp, full_rhs, assets["preconditioned_links"],
                assets["onsite_pair"], scratch, set_ptrs, p_prepare, parity)
            expected_rhs = matpc.rhs(full_rhs).reshape_as(compact)
            assert _relative(prepared_cpp, expected_rhs) < 2e-5

            reconstructed_cpp = torch.empty_like(full_rhs)
            p_reconstruct = initialize(shape, shape, len(initialized))
            qcu.applyMultigridStrictReconstructQcu(
                reconstructed_cpp, full_rhs, compact,
                assets["preconditioned_links"], assets["onsite_pair"],
                scratch, set_ptrs, p_reconstruct, parity)
            expected_full = matpc.reconstruct(
                full_rhs, compact.reshape(E, -1))
            assert _relative(reconstructed_cpp, expected_full) < 2e-5

        fine = torch.randn(12, *FINE_SHAPE, dtype=DTYPE).to(device)
        coarse = torch.randn(E, *shape, dtype=DTYPE).to(device)
        fine_checkerboard = Checkerboard(FINE_SHAPE)
        for parity in (0, 1):
            fine_compact = fine_checkerboard.extract(fine, parity).reshape(
                12, FINE_SHAPE[0], FINE_SHAPE[1], FINE_SHAPE[2],
                FINE_SHAPE[3] // 2).contiguous()
            coarse_cpp = torch.empty_like(coarse)
            p_restrict = initialize(FINE_SHAPE, shape, len(initialized))
            qcu.applyMultigridStrictRestrictQcu(
                coarse_cpp, fine_compact, blocked, set_ptrs, p_restrict,
                parity)
            assert _relative(
                coarse_cpp, transfer.restrict_parity(
                    fine_compact.reshape(12, -1), parity)) < 2e-5

            fine_cpp = torch.empty_like(fine_compact)
            p_prolong = initialize(FINE_SHAPE, shape, len(initialized))
            qcu.applyMultigridStrictProLongQcu(
                fine_cpp, coarse, blocked, set_ptrs, p_prolong, parity)
            expected = transfer.prolong_parity(coarse, parity).reshape_as(
                fine_compact)
            assert _relative(fine_cpp, expected) < 2e-5
    finally:
        for params in reversed(initialized):
            qcu.applyEndQcu(set_ptrs, params)
        torch.cuda.synchronize()


@pytest.mark.parametrize("depth", (1, 2), ids=lambda value: f"depth{value}")
@pytest.mark.parametrize("parity", (0, 1), ids=lambda value: f"parity{value}")
def test_quda_strict_recursive_vcycle_matches_reference_and_arena(depth, parity):
    """递归 coarse V-cycle 应匹配参考，且临时显存遵守 arena 精确预算。"""
    operators, transfers = _strict_hierarchy(depth)
    torch.manual_seed(20260862 + depth)
    full_rhs_cpu = torch.randn(
        operators[0].dof, *operators[0].shape, dtype=DTYPE)
    expected = _reference_strict_vcycle(
        operators, transfers, full_rhs_cpu, parity=parity, smoother_steps=2)

    device = torch.device("cuda")
    full_rhs = full_rhs_cpu.to(device)
    full_out = torch.empty_like(full_rhs)
    params, argv = _strict_params(operators, parity=parity, smoother_steps=2)
    set_ptrs = define.set_ptrs.clone()
    qcu.applyInitQcu(set_ptrs, params, argv)
    binding = _bind_strict_assets(
        set_ptrs, operators, transfers, device)
    persistent_initialized = False
    try:
        persistent_bytes = qcu.applyMultigridStrictInitQcu(
            set_ptrs, params, 1)
        persistent_initialized = True
        workspace_bytes = qcu.applyMultigridStrictVCycleQcu(
            full_out, full_rhs, set_ptrs, params, 1)
        repeated_out = torch.empty_like(full_out)
        repeated_bytes = qcu.applyMultigridStrictVCycleQcu(
            repeated_out, full_rhs, set_ptrs, params, 1)
        torch.cuda.synchronize()
        assert persistent_bytes == _expected_strict_workspace_bytes(operators)
        assert workspace_bytes == _expected_strict_workspace_bytes(operators)
        assert repeated_bytes == workspace_bytes
        assert _relative(repeated_out, full_out) == 0.0
        assert binding.memory_report()["omitted_raw_bytes"] > 0
        naive_per_level = sum(
            11 * operator.dof * int(torch.tensor(operator.shape).prod()) // 2
            for operator in operators) * full_rhs.element_size()
        assert workspace_bytes < naive_per_level

        if depth == 1:
            residual = operators[0].apply(full_out.cpu()) - full_rhs_cpu
            relative_residual = float(
                torch.linalg.norm(residual) / torch.linalg.norm(full_rhs_cpu))
            assert relative_residual < 2e-5
        else:
            assert _relative(full_out.cpu(), expected) < 2e-4
    finally:
        # binding 明确延长资产生命周期到 C++ 同步完成之后。
        assert not binding.closed
        if persistent_initialized:
            qcu.applyMultigridStrictEndQcu(set_ptrs, params)
        binding.close()
        qcu.applyEndQcu(set_ptrs, params)
        torch.cuda.synchronize()


def test_fine_clover_prepare_and_reconstruct_match_existing_solver():
    """锚定 odd Schur RHS/重构与现有 Clover BiCGStab 的 κ/奇偶约定。"""
    device = torch.device("cuda")
    shape = FINE_SHAPE
    gauge, clover_ee, clover_oo, clover_ee_inv, clover_oo_inv = (
        _identity_fine_clover_fields(shape, device))
    params, argv = _params(shape, 4, BLOCK, 0)
    params[define._MAX_ITER_] = 1000
    params[define._SET_PLAN_] = 1
    params[define._MG_USE_INIT_GUESS_] = 0
    argv[define._MASS_] = 0.1
    argv[define._ATOL_] = 1.0e-7
    set_ptrs = define.set_ptrs.clone()
    torch.manual_seed(20260866)
    full_rhs = torch.randn(
        2, 4, 3, *shape[:3], shape[3] // 2, dtype=DTYPE).to(device)
    full_solution = torch.empty_like(full_rhs)
    prepared = torch.empty_like(full_rhs[1])
    reconstructed = torch.empty_like(full_rhs)
    schur_image = torch.empty_like(prepared)

    qcu.applyInitQcu(set_ptrs, params, argv)
    try:
        qcu.applyCloverBistabCgPrepareQcu(
            prepared, full_rhs, gauge, clover_ee, clover_oo,
            clover_ee_inv, clover_oo_inv, set_ptrs, params)
        qcu.applyCloverBistabCgQcu(
            full_solution, full_rhs, gauge, clover_ee, clover_oo,
            clover_ee_inv, clover_oo_inv, set_ptrs, params)
        qcu.applyCloverBistabCgDslashQcu(
            schur_image, full_solution[1], gauge, clover_ee, clover_oo,
            clover_ee_inv, clover_oo_inv, set_ptrs, params)
        qcu.applyCloverBistabCgReconstructQcu(
            reconstructed, full_rhs, full_solution[1], gauge, clover_ee,
            clover_oo, clover_ee_inv, clover_oo_inv, set_ptrs, params)
        torch.cuda.synchronize()
        assert _relative(schur_image, prepared) < 2e-5
        assert _relative(reconstructed, full_solution) < 2e-6
    finally:
        qcu.applyEndQcu(set_ptrs, params)
        torch.cuda.synchronize()


def test_strict_fine_matpc_nontrivial_clover_matches_python_both_parities():
    """严格细层归一化 MATPC 必须同时匹配非平凡 Clover 的两种奇偶。"""
    shape = FINE_SHAPE
    compact = (*shape[:3], shape[3] // 2)
    device = torch.device("cuda")
    (gauge, clover_ee, clover_oo, clover_ee_inv, clover_oo_inv,
     even_field, odd_field) = _nontrivial_fine_clover_fields(shape, device)
    full_gauge = torch.zeros(3, 3, 4, *shape, dtype=DTYPE)
    diagonal = torch.arange(3)
    full_gauge[diagonal, diagonal] = 1.0
    full_clover_term = _fine_full_clover_term(
        shape, even_field, odd_field)
    kappa = 0.12
    reference = QudaStrictMultigrid(
        U=full_gauge,
        clover_term=full_clover_term,
        kappa=torch.tensor([kappa]),
        max_level=1,
        hierarchy_mode="strict",
        use_parity=True,
        setup_iters=0,
        verbose=False,
    )
    params, argv = _params(shape, 12, compact, 0)
    params[define._PARITY_] = 1
    argv[define._MASS_] = (1.0 / kappa - 8.0) / 2.0
    set_ptrs = define.set_ptrs.clone()
    qcu.applyInitQcu(set_ptrs, params, argv)
    try:
        torch.manual_seed(20260874)
        for parity in (0, 1):
            source = torch.randn(
                12, *compact, dtype=DTYPE, device=device)
            actual = torch.empty_like(source)
            qcu.applyMultigridStrictFineMatPCQcu(
                actual, source, gauge, clover_ee, clover_oo,
                clover_ee_inv, clover_oo_inv, set_ptrs, params, parity)
            expected = QudaMatPCOperator(
                reference._fine, parity=parity).apply(
                    source.reshape(12, -1).cpu()).reshape_as(source.cpu())
            assert _relative(actual.cpu(), expected) < 3.0e-5
    finally:
        qcu.applyEndQcu(set_ptrs, params)
        torch.cuda.synchronize()


def test_strict_fused_nontrivial_clover_matches_python_matpc_both_parities():
    """fused FGMRES 的 Clover RHS prepare/reconstruct 必须双奇偶闭环。"""
    shape = FINE_SHAPE
    compact = (*shape[:3], shape[3] // 2)
    device = torch.device("cuda")
    (gauge, clover_ee, clover_oo, clover_ee_inv, clover_oo_inv,
     even_field, odd_field) = _nontrivial_fine_clover_fields(shape, device)
    full_gauge = torch.zeros(3, 3, 4, *shape, dtype=DTYPE)
    diagonal = torch.arange(3)
    full_gauge[diagonal, diagonal] = 1.0
    full_clover_term = _fine_full_clover_term(
        shape, even_field, odd_field)
    kappa = 0.12
    rhs = torch.randn(
        2, 4, 3, *shape[:3], shape[3] // 2,
        dtype=DTYPE, device=device)
    layout = CompactParityLayout(shape)

    for parity in (0, 1):
        hierarchy = QudaStrictMultigrid(
            U=full_gauge,
            clover_term=full_clover_term,
            kappa=torch.tensor([kappa]),
            null_vectors=[torch.randn(
                2, 4, 3, *shape, dtype=DTYPE)],
            dof_list=[12, 4],
            block_size=BLOCK,
            max_level=2,
            hierarchy_mode="strict",
            use_parity=True,
            materialize_coarse=True,
            setup_iters=0,
            nu_pre=1,
            nu_post=1,
            coarse_max_iter=60,
            coarse_tol=1.0e-6,
            restart=8,
            max_iter=100,
            tol=1.0e-6,
            verbose=False,
            target_parity=parity,
        )
        reference = QudaMatPCOperator(hierarchy._fine, parity=parity)
        params, argv = _params(shape, 4, BLOCK, 0)
        params[define._PARITY_] = parity
        params[define._MG_USE_INIT_GUESS_] = 0
        argv[define._MASS_] = (1.0 / kappa - 8.0) / 2.0
        argv[define._ATOL_] = 1.0e-7
        vector_bytes = 12 * int(torch.tensor(shape).prod()) // 2 * 8
        coarse_vector_bytes = 4 * 2**4 * 8
        budget = 21 * vector_bytes + 2 * coarse_vector_bytes
        with CudaStrictMultigridSolver(
                hierarchy, argv, gauge, clover_ee, clover_oo,
                clover_ee_inv, clover_oo_inv, params,
                restart=8, max_krylov_bytes=budget,
                release_setup_assets=False) as solver:
            solution = solver.solve(rhs)

        rhs_compact = rhs.detach().cpu().reshape(2, 12, *compact)
        full_rhs = (
            layout.embed(rhs_compact[0], 0, 12) +
            layout.embed(rhs_compact[1], 1, 12))
        solution_compact = solution.detach().cpu().reshape(2, 12, *compact)
        target = solution_compact[parity].reshape(12, -1)
        expected_target = reference.rhs(full_rhs)
        residual = reference.apply(target) - expected_target
        relative_residual = float(
            torch.linalg.norm(residual) /
            torch.linalg.norm(expected_target))
        assert relative_residual < 3.0e-4

        actual_full = (
            layout.embed(solution_compact[0], 0, 12) +
            layout.embed(solution_compact[1], 1, 12))
        expected_full = reference.reconstruct(full_rhs, target)
        assert _relative(actual_full, expected_full) < 3.0e-4


def test_cuda_strict_solver_converges_with_bounded_krylov_arena():
    """完整 fine Schur + strict coarse V-cycle 应收敛且严格服从显存预算。"""
    shape = FINE_SHAPE
    diagonal = torch.eye(12, dtype=DTYPE).reshape(
        12, 12, 1, 1, 1, 1).expand(12, 12, *shape).clone() * 1.5
    torch.manual_seed(20260867)
    null = torch.randn(2, 4, 3, *shape, dtype=DTYPE)
    hierarchy = QudaStrictMultigrid(
        fine_matvec=lambda value: 1.5 * value,
        fine_adjoint=lambda value: 1.5 * value,
        fine_diagonal=diagonal,
        lat_size=shape,
        null_vectors=[null],
        dof_list=[12, 4],
        block_size=BLOCK,
        max_level=2,
        materialize_coarse=True,
        use_parity=True,
        target_parity=1,
        nu_pre=1,
        nu_post=1,
        coarse_max_iter=40,
        coarse_tol=1.0e-6,
        restart=6,
        max_iter=30,
        tol=1.0e-6,
        setup_iters=0,
        verbose=False,
    )
    device = torch.device("cuda")
    gauge, clover_ee, clover_oo, clover_ee_inv, clover_oo_inv = (
        _identity_fine_clover_fields(shape, device))
    params, argv = _params(shape, 4, BLOCK, 0)
    argv[define._MASS_] = 0.1
    argv[define._ATOL_] = 1.0e-7
    vector_bytes = 12 * int(torch.tensor(shape).prod()) // 2 * 8
    coarse_vector_bytes = 4 * 2**4 * 8
    fused_budget = 11 * vector_bytes + 2 * coarse_vector_bytes
    full_rhs = torch.randn(
        2, 4, 3, *shape[:3], shape[3] // 2, dtype=DTYPE).to(device)

    with CudaStrictMultigridSolver(
            hierarchy, argv, gauge, clover_ee, clover_oo,
            clover_ee_inv, clover_oo_inv, params,
            restart=6, max_krylov_bytes=fused_budget) as solver:
        owned_refs = (weakref.ref(solver.fine_null_vectors),)
        set_ptrs = solver.set_ptrs
        set_index = solver.schur.set_index
        report = solver.memory_report()
        assert report["effective_restart"] == 3
        assert report["outer_arena_bytes"] == 11 * vector_bytes
        assert report["coarse_io_bytes"] == 2 * coarse_vector_bytes
        assert report["fused_workspace_planned_bytes"] == fused_budget
        assert report["fused_workspace_resident_bytes"] == 0
        assert report["python_outer_arena_bytes"] == 0
        assert report["python_coarse_io_bytes"] == 0
        assert report["omitted_raw_bytes"] > 0
        assert report["hierarchy_sealed"]
        assert report["setup_detached_storage_bytes"] > 0
        assert report["lattice_scratch_requested_bytes"] == 3 * vector_bytes
        assert report["borrowed_gauge_clover_bytes"] > 0
        assert report["known_live_device_bytes"] > report["accounted_owned_bytes"]
        assert hierarchy._cuda_runtime_sealed
        assert hierarchy.transfers == [] and hierarchy.operators == []
        solution = solver.solve(full_rhs)
        prepared = torch.empty(12, *shape[:3], shape[3] // 2,
                               dtype=DTYPE, device=device)
        image = torch.empty_like(prepared)
        qcu.applyCloverBistabCgPrepareQcu(
            prepared, full_rhs, gauge, clover_ee, clover_oo,
            clover_ee_inv, clover_oo_inv, solver.set_ptrs, solver.params)
        qcu.applyCloverBistabCgDslashQcu(
            image, solution[1].reshape_as(prepared), gauge,
            clover_ee, clover_oo, clover_ee_inv, clover_oo_inv,
            solver.set_ptrs, solver.params)
        torch.cuda.synchronize()
        assert solver.converged
        assert _relative(image, prepared) < 3.0e-5
        live_report = solver.memory_report()
        assert live_report["fused_workspace_resident_bytes"] == fused_budget
        assert live_report["accounted_owned_bytes"] > report[
            "accounted_owned_bytes"]

    assert solver.closed
    assert all(reference() is None for reference in owned_refs)
    assert int(set_ptrs[set_index]) == 0
    solver.close()  # explicit close is idempotent
    with pytest.raises(RuntimeError, match="已关闭"):
        solver.memory_report()
