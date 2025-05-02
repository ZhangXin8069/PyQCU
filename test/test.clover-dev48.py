from pyquda_utils import core
from pyquda.enum_quda import QudaParity
from pyquda.field import Ns, Nc, LatticeGauge
from pyquda import init, pyquda as quda
import cupy as cp
import numpy as np
import functools
from pyqcu import define, io, qcu, eigen, cg, bistabcg, amg, linalg, gauge, demo
from time import perf_counter
from opt_einsum import contract
from pyqcu.set import params, argv, set_ptrs
params[define._LAT_X_] = 32
params[define._LAT_Y_] = 16
params[define._LAT_Z_] = 32
params[define._LAT_T_] = 32
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
gauge_filename = f"quda_wilson-clover-dslash-gauge_-{params[define._LAT_X_]}-{params[define._LAT_Y_]}-{params[define._LAT_Z_]}-{params[define._LAT_T_]}-{params[define._LAT_XYZT_]}-{params[define._GRID_X_]}-{params[define._GRID_Y_]}-{params[define._GRID_Z_]}-{params[define._GRID_T_]}-{params[define._PARITY_]}-{params[define._NODE_RANK_]}-{params[define._NODE_SIZE_]}-{params[define._DAGGER_]}-f.h5"
print("Parameters:", params)
print("Gauge filename:", gauge_filename)
gauge = io.hdf5_xxxtzyx2grid_xxxtzyx(params, gauge_filename)
fermion_in_filename = gauge_filename.replace("gauge", "fermion-in")
print("Fermion in filename:", fermion_in_filename)
fermion_in = io.hdf5_xxxtzyx2grid_xxxtzyx(
    params, fermion_in_filename)
fermion_out_filename = gauge_filename.replace("gauge", "fermion-out")
print("Fermion out filename:", fermion_out_filename)
quda_fermion_out = io.hdf5_xxxtzyx2grid_xxxtzyx(
    params, fermion_out_filename)
fermion_out = cp.zeros_like(fermion_in)
print("Fermion out data:", fermion_out.data)
print("Fermion out shape:", fermion_out.shape)
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
                       params[define._LAT_T_], params[define._LAT_Z_], params[define._LAT_Y_], int(params[define._LAT_X_]/define._LAT_P_),), dtype=fermion_in.dtype)
clover_odd = cp.zeros((define._LAT_S_, define._LAT_C_, define._LAT_S_, define._LAT_C_,
                       params[define._LAT_T_], params[define._LAT_Z_], params[define._LAT_Y_], int(params[define._LAT_X_]/define._LAT_P_),), dtype=fermion_in.dtype)
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
clover_fermion_in = cp.zeros_like(fermion_in)
print(clover_fermion_in.shape)
clover_fermion_in = (linalg.initialize_random_vector(
    clover_fermion_in.flatten())).reshape(clover_fermion_in.shape)
clover_fermion_out = cp.zeros_like(clover_fermion_in)
_clover_fermion_out = cp.zeros_like(clover_fermion_in)
qcu.applyCloverQcu(clover_even, gauge, set_ptrs, clover_dslash_eo_params)
qcu.applyCloverDslashQcu(_clover_fermion_out, clover_fermion_in,
                         gauge, set_ptrs, clover_dslash_eo_params)
qcu.applyDslashQcu(clover_fermion_out, clover_fermion_in, clover_even,
                   gauge, set_ptrs, clover_dslash_eo_params)
print(cp.linalg.norm(_clover_fermion_out - clover_fermion_out))
qcu.applyCloverQcu(clover_odd, gauge, set_ptrs, clover_dslash_oe_params)
qcu.applyCloverDslashQcu(_clover_fermion_out, clover_fermion_in,
                         gauge, set_ptrs, clover_dslash_oe_params)
qcu.applyDslashQcu(clover_fermion_out, clover_fermion_in, clover_odd,
                   gauge, set_ptrs, clover_dslash_oe_params)
print(cp.linalg.norm(_clover_fermion_out - clover_fermion_out))


def pdslash_no_dag(src):
    tmp0 = cp.zeros_like(src)
    tmp1 = cp.zeros_like(src)
    qcu.applyWilsonDslashQcu(
        tmp0, src, gauge, set_ptrs, wilson_dslash_eo_params)
    qcu.applyWilsonDslashQcu(
        tmp1, tmp0, gauge, set_ptrs, wilson_dslash_oe_params)
    return src-kappa**2*tmp1


def pdslash_dag(src):
    tmp0 = cp.zeros_like(src)
    tmp1 = cp.zeros_like(src)
    qcu.applyWilsonDslashQcu(
        tmp0, src, gauge, set_ptrs, wilson_dslash_eo_dag_params)
    qcu.applyWilsonDslashQcu(
        tmp1, tmp0, gauge, set_ptrs, wilson_dslash_oe_dag_params)
    return src-kappa**2*tmp1


def cg_dslash(src):
    return pdslash_dag(pdslash_no_dag(src))


def dslash_no_dag(src):
    dest = cp.zeros_like(src)
    qcu.applyWilsonDslashQcu(
        dest, src, gauge, set_ptrs, wilson_dslash_eo_params)
    return dest


def dslash_dag(src):
    dest = cp.zeros_like(src)
    qcu.applyWilsonDslashQcu(
        dest, src, gauge, set_ptrs, wilson_dslash_eo_dag_params)
    return dest


def dslash(src):
    return dslash_no_dag(src)


def bistabcg_dslash(src):
    return pdslash_no_dag(src)


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
dslash = core.getDefaultDirac(mass, 1e-12, 1000, xi_0, coeff_t, coeff_r)
quda_gauge = LatticeGauge(
    latt_info=latt_info)
dslash.loadGauge(quda_gauge)
dslash.destroy()
