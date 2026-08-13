#!/usr/bin/env python3
"""dev74 —— 布局对照实验：applyCloverBistabCgDslashQcu vs Python matvec_parity。

目的：确定 C++ Schur 算子（A_oo - k^2 D_oe A_ee^-1 D_eo）的输入/输出布局，
为 mg_dev74_dslash.py（CudaSchurOp）提供依据。
"""
import torch, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pyqcu import tools, dslash
from pyqcu.cuda import qcu
import pyqcu.cuda.define as define
from pyqcu.cuda.define import params, argv, set_ptrs
from mg_pyref_expt import setup_gpu

Lx, Ly, Lz, Lt = 8, 8, 8, 16
MASS = 0.05
KAPPA = 1.0 / (2 * MASS + 8)
ATOL = 1e-6

# ---- Python 参照 ----
U_full, b_full, clover, KAPPA, av, (g, fi, ce, coo, cei, coi) = setup_gpu(
    Lx, Ly, Lz, Lt, MASS, ATOL=ATOL)
op = dslash.operator(U=U_full, clover_term=clover,
                     kappa=torch.Tensor([KAPPA]),
                     support_parity=True, verbose=False)
S_py = op.matvec_parity
ls_odd = [Lx, Ly, Lz, Lt // 2]
torch.manual_seed(7)
x_o = torch.randn([12] + ls_odd, dtype=torch.complex64, device="cuda")
y_py = S_py(x_o)
print(f"Python S(x_o): shape={tuple(y_py.shape)} norm={float(tools.norm(y_py)):.6e}")

# ---- C++ 准备 ----
dt = define.dtype(define._LAT_C64_)
params[define._SET_INDEX_] = 0
params[define._SET_PLAN_] = -1
qcu.applyInitQcu(set_ptrs, params, av)
qcu.applyGaussGaugeQcu(g, set_ptrs, params)
params[define._SET_INDEX_] += 1
params[define._SET_PLAN_] = 2
params[define._PARITY_] = 0
qcu.applyInitQcu(set_ptrs, params, av)
qcu.applyCloversQcu(ce, cei, g, set_ptrs, params)
params[define._SET_INDEX_] += 1
params[define._SET_PLAN_] = 2
params[define._PARITY_] = 1
qcu.applyInitQcu(set_ptrs, params, av)
qcu.applyCloversQcu(coo, coi, g, set_ptrs, params)
params[define._SET_INDEX_] += 1
params[define._SET_PLAN_] = 1
qcu.applyInitQcu(set_ptrs, params, av)

# ---- 布局尝试 1：输入/输出都是 [12,X,Y,Z,T/2] ----
y_cpp = torch.zeros_like(x_o)
qcu.applyCloverBistabCgDslashQcu(y_cpp, x_o, g, ce, coo, cei, coi,
                                 set_ptrs, params)
torch.cuda.synchronize()
err1 = float(tools.norm(y_cpp - y_py) / tools.norm(y_py))
print(f"try1 [12,XYZT/2]: rel_err = {err1:.6e}")

# ---- 布局尝试 2：输入/输出 [2,12,X,Y,Z,T/2]（poooxyzt，取 p=1 奇）----
x_po = torch.stack([torch.zeros_like(x_o), x_o])       # p=1 = 奇子格
y_po = torch.zeros_like(x_po)
qcu.applyCloverBistabCgDslashQcu(y_po, x_po, g, ce, coo, cei, coi,
                                 set_ptrs, params)
torch.cuda.synchronize()
err2 = float(tools.norm(y_po[1] - y_py) / tools.norm(y_py))
print(f"try2 [2,12,XYZT/2] p=1: rel_err = {err2:.6e}")

# ---- 布局尝试 3：输入 [2,12,...] 偶子格放 p=0 ----
x_po0 = torch.stack([x_o, torch.zeros_like(x_o)])
y_po0 = torch.zeros_like(x_po0)
qcu.applyCloverBistabCgDslashQcu(y_po0, x_po0, g, ce, coo, cei, coi,
                                 set_ptrs, params)
torch.cuda.synchronize()
err3 = float(tools.norm(y_po0[0] - y_py) / tools.norm(y_py))
print(f"try3 [2,12,XYZT/2] p=0: rel_err = {err3:.6e}")

qcu.applyEndQcu(set_ptrs, params)
