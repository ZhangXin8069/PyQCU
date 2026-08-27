"""CUDA C++ Multigrid 的粗层 breakdown 回归。

粗算子故意取零，但限制后的粗 RHS 保持非零；BiStabCG 因而会遇到
``<r_tilde, A p> = 0``。实现应停止当前粗求解并保留有限解，而不能让
复数除零产生 NaN，再经 prolongation 污染细层输出。

运行：
    source ./env.sh
    pytest -q examples/qcu/dev87/test_mg_breakdown.py
"""

import pytest


@pytest.mark.skipif("not __import__('torch').cuda.is_available()",
                    reason="需要 CUDA 设备")
def test_zero_coarse_operator_does_not_poison_solution():
    import torch

    from pyqcu.cuda import define, qcu

    X, Y, Z, T = 4, 4, 4, 8
    Xc, Yc, Zc, Tc = 2, 2, 2, 2
    E = 12
    device = torch.device("cuda")
    dt = torch.complex64
    data_type = define._LAT_C64_

    params = define.params.clone()
    argv = define.argv.clone()
    set_ptrs = define.set_ptrs.clone()
    params[define._LAT_X_] = X
    params[define._LAT_Y_] = Y
    params[define._LAT_Z_] = Z
    params[define._LAT_T_] = T
    params[define._LAT_XYZT_] = X * Y * Z * T
    params[define._GRID_X_] = 1
    params[define._GRID_Y_] = 1
    params[define._GRID_Z_] = 1
    params[define._GRID_T_] = 1
    params[define._NODE_RANK_] = 0
    params[define._NODE_SIZE_] = 1
    params[define._DATA_TYPE_] = data_type
    params[define._SET_INDEX_] = 0
    params[define._SET_PLAN_] = 1
    params[define._PARITY_] = 0
    params[define._MAX_ITER_] = 8
    params[define._VERBOSE_] = 0
    params[define._MG_NUM_LEVEL_] = 2
    params[define._MG_LEVEL1_E_] = E
    params[define._MG_LEVEL1_X_] = Xc
    params[define._MG_LEVEL1_Y_] = Yc
    params[define._MG_LEVEL1_Z_] = Zc
    params[define._MG_LEVEL1_T_] = Tc
    params[define._MG_LEVEL1_MAX_ITER_] = 8
    params[define._MG_LEVEL1_DATA_TYPE_] = data_type
    params[define._MG_LEVEL1_NUM_RESTART_] = 1
    params[define._MG_USE_DEFLATE_] = 0
    params[define._MG_USE_GCR_] = 0
    params[define._MG_MU_PRE_] = 0
    params[define._MG_USE_INIT_GUESS_] = 0
    argv[define._MASS_] = 0.05
    argv[define._ATOL_] = 1e-6
    argv[define._MG_LEVEL1_ATOL_] = 1e-3

    ls = (X, Y, Z, T // 2)
    # [parity, row-color, col-color, direction, X, Y, Z, T/2]
    eye3 = torch.eye(3, dtype=dt, device=device)
    gauge = eye3.reshape(1, 3, 3, 1, 1, 1, 1, 1).expand(
        2, 3, 3, 4, *ls).contiguous()
    # [spin, color, spin, color, X, Y, Z, T/2]
    eye12 = torch.eye(12, dtype=dt, device=device).reshape(
        4, 3, 4, 3, 1, 1, 1, 1)
    clover = eye12.expand(4, 3, 4, 3, *ls).contiguous()
    clover_inv = clover.clone()
    rhs = torch.ones((2, 4, 3, *ls), dtype=dt, device=device)
    out = torch.empty_like(rhs)

    # [E, 12, X, Y, Z, T/2]；非零 null vector 使 restrict 的 RHS 非零。
    null_vecs = torch.ones((E, 12, *ls), dtype=dt, device=device)
    hop_nn = torch.zeros((2, 4, E, E, Xc, Yc, Zc, Tc),
                         dtype=dt, device=device)
    hop_diag = torch.zeros((2, 2, 6, E, E, Xc, Yc, Zc, Tc),
                           dtype=dt, device=device)
    sitting = torch.zeros((E, E, Xc, Yc, Zc, Tc),
                          dtype=dt, device=device)
    base = 30
    set_ptrs[base + 0] = null_vecs.data_ptr()
    set_ptrs[base + 1] = hop_nn.data_ptr()
    set_ptrs[base + 2] = hop_diag.data_ptr()
    set_ptrs[base + 3] = sitting.data_ptr()

    initialized = False
    try:
        qcu.applyInitQcu(set_ptrs, params, argv)
        initialized = True
        qcu.applyCloverMultigridQcu(
            out, rhs, gauge, clover, clover, clover_inv, clover_inv,
            set_ptrs, params)
        torch.cuda.synchronize()
        assert bool(torch.isfinite(out).all().item())
    finally:
        if initialized:
            qcu.applyEndQcu(set_ptrs, params)
