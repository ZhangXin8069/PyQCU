"""CudaSchurOp vs Python matvec_parity 等价性验证（共享 clover 组件）。

bug36 同族思路: 执行从未运行的 pyqcu/cuda/_schur_op.py；
共享 sitting.M_e/M_o/M_e_inv/M_o_inv 消除 clover 约定差异，
纯对比两侧 Schur matvec 计算路径。
"""
import traceback
import torch
from pyqcu import tools, dslash, lattice
from pyqcu.cuda import qcu, define
from pyqcu.cuda.define import params, argv, set_ptrs
from pyqcu.cuda._schur_op import CudaSchurOp

try:
    LAT = [16, 16, 16, 8]
    MASS = 0.0
    KAPPA = torch.Tensor([0.125])
    DT = torch.complex64
    DEV = torch.device('cuda')

    # ---- params 模板(conftest.wilson 协议, mass=0) ----
    for ax, v in zip((define._LAT_X_, define._LAT_Y_, define._LAT_Z_,
                      define._LAT_T_), LAT):
        params[ax] = v
    params[define._LAT_XYZT_] = LAT[0] * LAT[1] * LAT[2] * LAT[3]
    (params[define._GRID_X_], params[define._GRID_Y_],
     params[define._GRID_Z_], params[define._GRID_T_]) = tools.give_grid_size()
    params[define._PARITY_] = 0
    params[define._NODE_RANK_] = define.rank
    params[define._NODE_SIZE_] = define.size
    params[define._DAGGER_] = 0
    params[define._MAX_ITER_] = 1000
    params[define._DATA_TYPE_] = define._LAT_C64_
    params[define._SET_INDEX_] = 0
    params[define._SET_PLAN_] = -1
    params[define._VERBOSE_] = 0
    params[define._SEED_] = 42
    av = argv.to(dtype=define.dtype(params[define._DATA_TYPE_]).to_real()).clone()
    av[define._MASS_] = MASS
    av[define._ATOL_] = 1e-9
    av[define._SIGMA_] = 0.1

    # ---- C++ gauss 规范场(独立源) ----
    gauge_eo = torch.zeros([2, 3, 3, 4] + define.lat_shape(params),
                           dtype=DT, device=DEV)
    qcu.applyInitQcu(set_ptrs, params, av)
    qcu.applyGaussGaugeQcu(gauge_eo, set_ptrs, params)
    torch.cuda.synchronize()
    U = tools.poooxyzt2oooxyzt(input_array=gauge_eo)

    # ---- Python 算子(clover 组件供双方共享) ----
    clover = torch.zeros([4, 3, 4, 3] + [LAT[0], LAT[1], LAT[2], LAT[3]],
                         dtype=DT, device=DEV)
    op_py = dslash.operator(U=U, kappa=KAPPA, clover_term=clover,
                            support_parity=True, verbose=False)

    # ---- CudaSchurOp(共享 Python clover 组件) ----
    ptpl = params.clone()
    schur = CudaSchurOp(av.clone(), gauge_eo, op_py.sitting.M_e,
                        op_py.sitting.M_o, op_py.sitting.M_e_inv,
                        op_py.sitting.M_o_inv, device=DEV, params=ptpl)
    print(f"[SCHUR] set_index={schur.set_index}", flush=True)

    torch.manual_seed(42)
    x = torch.randn([12] + LAT[:3] + [LAT[3] // 2], dtype=DT, device=DEV)
    y_cpp = schur.matvec(x).clone()
    y_ref = op_py.matvec_parity(src_o=x)
    torch.cuda.synchronize()
    denom = float(tools.norm(y_ref))
    rel = float(tools.norm(y_cpp - y_ref)) / max(denom, 1e-30)
    print(f"[SCHUR] |y_cpp|={float(tools.norm(y_cpp)):.4f} "
          f"|y_ref|={denom:.4f} rel_diff={rel:.2e}", flush=True)
    assert rel < 1e-4, f"C++/Python Schur mismatch rel={rel:.2e}"

    schur.release()
    print(f"[SCHUR][SUMMARY] EQUIVALENCE PASS", flush=True)
except Exception:
    traceback.print_exc()
    print("[SCHUR][SUMMARY] FAIL")
