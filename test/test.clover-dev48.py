from pyquda import pyquda as quda
from pyquda_utils import core
from pyquda.field import LatticeGauge
import cupy as cp
import numpy as np
from pyqcu.cuda import define, io, qcu, eigen, cg, bistabcg, amg, linalg, gauge
from time import perf_counter
from opt_einsum import contract
from pyqcu.cuda.set import params, argv, set_ptrs
params[define._LAT_X_] = 8
params[define._LAT_Y_] = 8
params[define._LAT_Z_] = 8
params[define._LAT_T_] = 8
params[define._LAT_XYZT_] = params[define._LAT_X_] * \
    params[define._LAT_Y_] * params[define._LAT_Z_] * params[define._LAT_T_]
params[define._DATA_TYPE_] = define._LAT_C64_
sigma = 0.1
seed = 12138
params[define._NODE_RANK_] = define.rank
params[define._NODE_SIZE_] = define.size
argv[define._MASS_] = -3.5
argv[define._TOL_] = 1e-12
kappa = 1 / (2 * argv[define._MASS_] + 8)
print(define.dtype(params[define._DATA_TYPE_]))
qcu_src = cp.ones(params[define._LAT_XYZT_]*define._LAT_SC_,
                  dtype=define.dtype(params[define._DATA_TYPE_]))
qcu_src = io.fermion2psctzyx(qcu_src, params)
print("Src data:", qcu_src.data)
print("Src shape:", qcu_src.shape)
argv = argv.astype(qcu_src.real.dtype)
print("Arguments:", argv)
print("Arguments data:", argv.data)
print("Arguments dtype:", argv.dtype)
print("Demo is running...")
print("Set pointers:", set_ptrs)
print("Set pointers data:", set_ptrs.data)
# qcu_gauge = gauge.give_gauss_SU3(sigma=sigma, seed=seed,
#                          dtype=qcu_src.dtype, size=params[define._LAT_XYZT_]*define._LAT_D_)
qcu_gauge = cp.ones(params[define._LAT_XYZT_] *
                    define._LAT_DCC_, dtype=qcu_src.dtype)
qcu_gauge = io.gauge2dptzyxcc(qcu_gauge, params)
qcu_gauge = io.dptzyxcc2ccdptzyx(qcu_gauge)
print("qcu_gauge data:", qcu_gauge.data)
print("qcu_gauge shape:", qcu_gauge.shape)
wilson_cg_params = params.copy()
wilson_cg_params[define._SET_INDEX_] = 0
wilson_cg_params[define._SET_PLAN_] = define._SET_PLAN1_
qcu.applyInitQcu(set_ptrs, wilson_cg_params, argv)
wilson_dslash_eo_params = params.copy()
wilson_dslash_eo_params[define._SET_INDEX_] = 1
wilson_dslash_eo_params[define._SET_PLAN_] = define._SET_PLAN0_
wilson_dslash_eo_params[define._PARITY_] = define._EVEN_
wilson_dslash_eo_params[define._DAGGER_] = define._NO_USE_
qcu.applyInitQcu(set_ptrs, wilson_dslash_eo_params, argv)
wilson_dslash_eo_dag_params = params.copy()
wilson_dslash_eo_dag_params[define._SET_INDEX_] = 2
wilson_dslash_eo_dag_params[define._SET_PLAN_] = define._SET_PLAN0_
wilson_dslash_eo_dag_params[define._PARITY_] = define._EVEN_
wilson_dslash_eo_dag_params[define._DAGGER_] = define._USE_
qcu.applyInitQcu(set_ptrs, wilson_dslash_eo_dag_params, argv)
wilson_dslash_oe_params = params.copy()
wilson_dslash_oe_params[define._SET_INDEX_] = 3
wilson_dslash_oe_params[define._SET_PLAN_] = define._SET_PLAN0_
wilson_dslash_oe_params[define._PARITY_] = define._ODD_
wilson_dslash_oe_params[define._DAGGER_] = define._NO_USE_
qcu.applyInitQcu(set_ptrs, wilson_dslash_oe_params, argv)
wilson_dslash_oe_dag_params = params.copy()
wilson_dslash_oe_dag_params[define._SET_INDEX_] = 4
wilson_dslash_oe_dag_params[define._SET_PLAN_] = define._SET_PLAN0_
wilson_dslash_oe_dag_params[define._PARITY_] = define._ODD_
wilson_dslash_oe_dag_params[define._DAGGER_] = define._USE_
qcu.applyInitQcu(set_ptrs, wilson_dslash_oe_dag_params, argv)
clover_even = cp.zeros((define._LAT_S_, define._LAT_C_, define._LAT_S_, define._LAT_C_,
                       params[define._LAT_T_], params[define._LAT_Z_], params[define._LAT_Y_], int(params[define._LAT_X_]/define._LAT_P_),), dtype=qcu_src.dtype)
clover_odd = cp.zeros((define._LAT_S_, define._LAT_C_, define._LAT_S_, define._LAT_C_,
                       params[define._LAT_T_], params[define._LAT_Z_], params[define._LAT_Y_], int(params[define._LAT_X_]/define._LAT_P_),), dtype=qcu_src.dtype)
clover_dslash_eo_params = params.copy()
clover_dslash_eo_params[define._SET_INDEX_] = 5
clover_dslash_eo_params[define._SET_PLAN_] = define._SET_PLAN2_
clover_dslash_eo_params[define._PARITY_] = define._EVEN_
clover_dslash_eo_params[define._DAGGER_] = define._NO_USE_
qcu.applyInitQcu(set_ptrs, clover_dslash_eo_params, argv)
clover_dslash_oe_params = params.copy()
clover_dslash_oe_params[define._SET_INDEX_] = 6
clover_dslash_oe_params[define._SET_PLAN_] = define._SET_PLAN2_
clover_dslash_oe_params[define._PARITY_] = define._ODD_
clover_dslash_oe_params[define._DAGGER_] = define._NO_USE_
qcu.applyInitQcu(set_ptrs, clover_dslash_oe_params, argv)
qcu_clover_src = cp.zeros_like(qcu_src)
print(qcu_clover_src.shape)
qcu_clover_src = (linalg.initialize_random_vector(
    qcu_clover_src.flatten())).reshape(qcu_clover_src.shape)
qcu_clover_dest = cp.zeros_like(qcu_clover_src)
_qcu_clover_dest = cp.zeros_like(qcu_clover_src)
qcu.applyCloverQcu(clover_even, qcu_gauge, set_ptrs, clover_dslash_eo_params)
qcu.applyCloverDslashQcu(_qcu_clover_dest, qcu_clover_src,
                         qcu_gauge, set_ptrs, clover_dslash_eo_params)
qcu.applyDslashQcu(qcu_clover_dest, qcu_clover_src, clover_even,
                   qcu_gauge, set_ptrs, clover_dslash_eo_params)
print(cp.linalg.norm(_qcu_clover_dest - qcu_clover_dest))
qcu.applyCloverQcu(clover_odd, qcu_gauge, set_ptrs, clover_dslash_oe_params)
qcu.applyCloverDslashQcu(_qcu_clover_dest, qcu_clover_src,
                         qcu_gauge, set_ptrs, clover_dslash_oe_params)
qcu.applyDslashQcu(qcu_clover_dest, qcu_clover_src, clover_odd,
                   qcu_gauge, set_ptrs, clover_dslash_oe_params)
print(cp.linalg.norm(_qcu_clover_dest - qcu_clover_dest))
grid_size = [1, 1, 1, 1]
latt_size = [params[define._LAT_X_], params[define._LAT_Y_],
             params[define._LAT_Z_], params[define._LAT_T_]]
xi_0, nu = 1.0, 1.0
kappa = 1.0
mass = 1 / (2 * kappa) - 4
coeff = 1.0
coeff_r, coeff_t = 1.0, 1.0
core.init(grid_size, latt_size, -1, xi_0 / nu, resource_path=".cache")
latt_info = core.getDefaultLattice()
Lx, Ly, Lz, Lt = latt_info.size
quda_dslash = core.getDefaultDirac(mass, 1e-12, 1000, xi_0, coeff_t, coeff_r)
quda_gauge = LatticeGauge(latt_info=latt_info)
# print(type(qcu_gauge))
# print(qcu_gauge.dtype)
# print(qcu_gauge.shape)
# print(type(quda_gauge.data))
# print(quda_gauge.data.dtype)
# print(quda_gauge.data.shape)
# quda_gauge.data[:] = io.ccdptzyx2dptzyxcc(
#     qcu_gauge).astype(quda_gauge.data.dtype)
# quda_src = io.psctzyx2ptzyxsc(qcu_src, params)
# quda_dest = cp.zeros_like(quda_src)
# quda_dslash.loadGauge(quda_gauge)
# quda.dslashQuda(quda_dest, quda_src, quda_dslash.invert_param,
#                 0)
