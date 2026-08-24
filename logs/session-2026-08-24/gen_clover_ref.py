"""with_data clover 参考生成器 — L8³ κ=1.0（2026-08-24）。

U 由 C++ applyGaussGaugeQcu(seed=42) 独立生成；clover_term/inv 为 Python
make_clover+add_I/inverse 当前态冻结；另以 C++ applyCloversQcu 交叉校验
Python clover_term（信息性报告）。原 L32Y16 规格达 GB 级故参数化至 L8³。
"""
import torch
from pyqcu import tools, dslash, lattice
from pyqcu.cuda import qcu, define
from pyqcu.cuda.define import params, argv, set_ptrs

LAT = [8, 8, 8, 8]
MASS = 0.0
KAPPA = torch.Tensor([1.0])
DT = torch.complex64
DATA = 'examples/data'
DEV = torch.device('cuda')

params[define._LAT_X_] = LAT[0]
params[define._LAT_Y_] = LAT[1]
params[define._LAT_Z_] = LAT[2]
params[define._LAT_T_] = LAT[3]
params[define._LAT_XYZT_] = LAT[0] * LAT[1] * LAT[2] * LAT[3]
params[define._GRID_X_], params[define._GRID_Y_], params[
    define._GRID_Z_], params[define._GRID_T_] = tools.give_grid_size()
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
argv = argv.to(dtype=define.dtype(params[define._DATA_TYPE_]).to_real())
argv[define._MASS_] = MASS
argv[define._ATOL_] = 1e-9
argv[define._SIGMA_] = 0.1

# 1) C++ 独立生成规范场
gauge_eo = torch.zeros(size=[2, 3, 3, 4] + define.lat_shape(params),
                       dtype=DT, device=DEV)
qcu.applyInitQcu(set_ptrs, params, argv)
qcu.applyGaussGaugeQcu(gauge_eo, set_ptrs, params)
torch.cuda.synchronize()
U = tools.poooxyzt2oooxyzt(input_array=gauge_eo)
print(f"[GENC] U su3={lattice.check_su3(U, verbose=False)}", flush=True)

# 2) Python 冻结 clover term/inv
term = dslash.make_clover(U=U, kappa=KAPPA, verbose=False)
term = dslash.add_I(clover_term=term, verbose=False)
inv = dslash.inverse(clover_term=term, verbose=False)

# 3) C++ applyClovers 交叉校验(信息性)
ls = define.lat_shape(params)
ce = torch.zeros([4, 3, 4, 3] + ls, dtype=DT, device=DEV)
cei = torch.zeros_like(ce)
coo = torch.zeros_like(ce)
coi = torch.zeros_like(ce)
params[define._SET_INDEX_] += 1
params[define._SET_PLAN_] = 2
params[define._PARITY_] = 0
qcu.applyInitQcu(set_ptrs, params, argv)
qcu.applyCloversQcu(ce, cei, gauge_eo, set_ptrs, params)
params[define._SET_INDEX_] += 1
params[define._PARITY_] = 1
qcu.applyInitQcu(set_ptrs, params, argv)
qcu.applyCloversQcu(coo, coi, gauge_eo, set_ptrs, params)
torch.cuda.synchronize()
cpp_term = tools.poooxyzt2oooxyzt(input_array=torch.stack([ce, coo]))
d_cpp = float(tools.norm(cpp_term - term) / tools.norm(term))
print(f"[GENC] C++ applyClovers vs python term: rel={d_cpp:.2e} (信息性)", flush=True)

for name, tensor, suffix in (("U", U.cpu(), "ccdxyzt"),
                             ("clover_term", term.cpu(), "scscxyzt"),
                             ("clover_inv_term", inv.cpu(), "scscxyzt")):
    fn = f"{DATA}/refer.clover.{name}.L8K1.{suffix}.c64.h5"
    tools.gridoooxyzt2hdf5oooxyzt(input_tensor=tensor, file_name=fn,
                                  lat_size=LAT, verbose=False)
    print(f"[GENC] wrote {fn}", flush=True)
print("[GENC] ALL DONE", flush=True)
