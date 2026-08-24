"""with_data 参考 HDF5 生成器 — 跨后端独立来源（2026-08-24 bug37 后续）。

U 与求解解 x 由 C++ CUDA 后端生成（applyGaussGaugeQcu/applyWilsonBistabCgQcu，
与 PyTorch 层完全异构的独立实现）；dest 为 Python give_wilson 当前态冻结
（守护 Python 算子未来漂移）。规格：L16^3, mass=0 (kappa=0.125), c64, seed=42。
"""
import torch
from pyqcu import tools, dslash, lattice
from pyqcu.cuda import qcu, define
from pyqcu.cuda.define import params, argv, set_ptrs

LAT = [16, 16, 16, 16]
MASS = 0.0                    # kappa = 1/(2*mass+8) = 0.125
KAPPA = torch.Tensor([0.125])
DT = torch.complex64
DATA = 'examples/data'

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

gauge_eo = torch.zeros(size=[2, 3, 3, 4] + define.lat_shape(params),
                       dtype=DT, device=torch.device('cuda'))
qcu.applyInitQcu(set_ptrs, params, argv)
qcu.applyGaussGaugeQcu(gauge_eo, set_ptrs, params)
torch.cuda.synchronize()
U_full = tools.poooxyzt2oooxyzt(input_array=gauge_eo).cpu()
print(f"[GEN] U ok su3={lattice.check_su3(U_full, verbose=False)}", flush=True)

torch.manual_seed(42)
src_full = _src = torch.randn(size=[4, 3] + LAT, dtype=DT,
                              device=torch.device('cuda'))
src_eo = tools.oooxyzt2poooxyzt(input_array=src_full)

params[define._SET_INDEX_] += 1
params[define._SET_PLAN_] = 1
params[define._PARITY_] = 0
qcu.applyInitQcu(set_ptrs, params, argv)
x_eo = torch.zeros_like(src_eo)
qcu.applyWilsonBistabCgQcu(x_eo, src_eo, gauge_eo, set_ptrs, params)
torch.cuda.synchronize()
x_full = tools.poooxyzt2oooxyzt(input_array=x_eo).cpu()
res = float(tools.norm(dslash.give_wilson(
    src=x_full.to(CUDA_DEV := torch.device('cuda')), U=U_full.to(CUDA_DEV),
    kappa=KAPPA, with_I=True, verbose=False) - src_full.to(CUDA_DEV))
    / tools.norm(src_full.to(CUDA_DEV)))
print(f"[GEN] C++ solve residual check = {res:.2e}", flush=True)

dest_full = dslash.give_wilson(src=_src, U=U_full.to(CUDA_DEV), kappa=KAPPA,
                               with_I=True, verbose=False).cpu()
b_full = src_full.cpu()

for name, tensor in (("U", U_full), ("src", b_full), ("dest", dest_full),
                     ("x", x_full), ("b", b_full)):
    fn = f"{DATA}/refer.wilson.{name}.L16K0_125.ccdxyzt.c64.h5" \
        if name == "U" else \
        f"{DATA}/refer.wilson.{name}.L16K0_125.scxyzt.c64.h5"
    tools.gridoooxyzt2hdf5oooxyzt(input_tensor=tensor.to(torch.device("cpu")),
                                  file_name=fn, lat_size=LAT, verbose=False)
    print(f"[GEN] wrote {fn}", flush=True)
print("[GEN] ALL DONE", flush=True)
