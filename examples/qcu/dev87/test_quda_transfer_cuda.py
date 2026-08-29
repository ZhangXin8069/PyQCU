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

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("需要 CUDA 设备", allow_module_level=True)

try:
    from pyqcu.cuda import define, qcu
    from pyqcu.solver import QudaCoarseOperator, QudaTransfer
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
