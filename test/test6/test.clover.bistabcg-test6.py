import os
import numpy as np
import cupy as cp
from pyqcu.cuda import define, io, qcu
params = np.array([0]*define._PARAMS_SIZE_, dtype=np.int32)
params[define._LAT_X_] = 32
params[define._LAT_Y_] = 64
params[define._LAT_Z_] = 64
params[define._LAT_T_] = 64
params[define._LAT_XYZT_] = params[define._LAT_X_] * \
    params[define._LAT_Y_]*params[define._LAT_Z_]*params[define._LAT_T_]
params[define._GRID_X_], params[define._GRID_Y_], params[define._GRID_Z_], params[
    define._GRID_T_] = define.split_into_four_factors(define.size)
params[define._PARITY_] = 0
params[define._NODE_RANK_] = define.rank
params[define._NODE_SIZE_] = define.size
params[define._DAGGER_] = 0
params[define._MAX_ITER_] = 1000
params[define._DATA_TYPE_] = define._LAT_C64_
params[define._SET_INDEX_] = 0
params[define._SET_PLAN_] = 0
params[define._MG_X_] = 1
params[define._MG_Y_] = 1
params[define._MG_Z_] = 1
params[define._MG_T_] = 1
params[define._LAT_E_] = 24
params[define._VERBOSE_] = 1
params[define._SEED_] = 42
argv = np.array([0.0]*define._ARGV_SIZE_,
                dtype=define.dtype_half(params[define._DATA_TYPE_]))
argv[define._MASS_] = 0.05
argv[define._TOL_] = 1e-12
argv[define._SIGMA_] = 0.1
set_ptrs = np.array(10*[0], dtype=np.int64)  # maybe more than 10?
#############################
if os.path.exists("_gauge-test8.h5") and os.path.exists("_fermion_in-test8.h5") and os.path.exists("_wilson_fermion_out-test8.h5") and os.path.exists("_clover_fermion_out-test8.h5"):
    gauge = io.hdf5_xxxtzyx2grid_xxxtzyx(params, "_gauge-test8.h5")
    fermion_in = io.hdf5_xxxtzyx2grid_xxxtzyx(params, "_fermion_in-test8.h5")
    _wilson_fermion_out = io.hdf5_xxxtzyx2grid_xxxtzyx(
        params, "_wilson_fermion_out-test8.h5")
    _clover_fermion_out = io.hdf5_xxxtzyx2grid_xxxtzyx(
        params, "_clover_fermion_out-test8.h5")
    wilson_fermion_out = cp.zeros_like(_wilson_fermion_out)
    clover_fermion_out = cp.zeros_like(_clover_fermion_out)
gauge = io.give_none_gauge(params, save=False)
fermion_in = io.give_none_fermion_in(params, save=False)
wilson_fermion_out = io.give_none_fermion_out(params, save=False)
clover_fermion_out = io.give_none_fermion_out(params, save=False)
#############################
gauge = cp.zeros_like(gauge)
fermion_in = cp.ones_like(fermion_in)
wilson_fermion_out = cp.zeros_like(fermion_in)
clover_fermion_out = cp.zeros_like(fermion_in)
#############################
qcu.applyInitQcu(set_ptrs, params, argv)
qcu.applyGaussGaugeQcu(gauge, set_ptrs, params)
qcu.applyEndQcu(set_ptrs, params)
#############################
params[define._VERBOSE_] = 1
params[define._SET_INDEX_] += 1
params[define._SET_PLAN_] = 1
#############################
qcu.applyInitQcu(set_ptrs, params, argv)
qcu.applyWilsonBistabCgQcu(
    wilson_fermion_out, fermion_in, gauge, set_ptrs, params)
qcu.applyEndQcu(set_ptrs, params)
#############################
clover_ee = io.give_none_clover(params, save=False)
clover_ee_inv = cp.zeros_like(clover_ee)
clover_oo = cp.zeros_like(clover_ee)
clover_oo_inv = cp.zeros_like(clover_ee)
#############################
params[define._VERBOSE_] = 1
params[define._SET_INDEX_] += 1
params[define._SET_PLAN_] = 2
params[define._PARITY_] = 0
#############################
qcu.applyInitQcu(set_ptrs, params, argv)
qcu.applyCloversQcu(clover_ee, clover_ee_inv, gauge, set_ptrs, params)
qcu.applyEndQcu(set_ptrs, params)
#############################
params[define._VERBOSE_] = 1
params[define._SET_INDEX_] += 1
params[define._SET_PLAN_] = 2
params[define._PARITY_] = 1
#############################
qcu.applyInitQcu(set_ptrs, params, argv)
qcu.applyCloversQcu(clover_oo, clover_oo_inv, gauge, set_ptrs, params)
qcu.applyEndQcu(set_ptrs, params)
#############################
params[define._VERBOSE_] = 1
params[define._SET_INDEX_] += 1
params[define._SET_PLAN_] = 1
#############################
qcu.applyInitQcu(set_ptrs, params, argv)
qcu.applyCloverBistabCgQcu(clover_fermion_out, fermion_in,
                           gauge, clover_ee, clover_oo, clover_ee_inv, clover_oo_inv,  set_ptrs, params)
qcu.applyEndQcu(set_ptrs, params)
#############################
if os.path.exists("_gauge-test8.h5") and os.path.exists("_fermion_in-test8.h5") and os.path.exists("_wilson_fermion_out-test8.h5") and os.path.exists("_clover_fermion_out-test8.h5"):
    io.grid_xxxtzyx2hdf5_xxxtzyx(
        wilson_fermion_out, params, "_wilson_fermion_out-test8.h5")
    io.grid_xxxtzyx2hdf5_xxxtzyx(
        clover_fermion_out, params, "_clover_fermion_out-test8.h5")
    print(
        f"@cp.linalg.norm(wilson_fermion_out-_wilson_fermion_out)/cp.linalg.norm(wilson_fermion_out):{cp.linalg.norm(wilson_fermion_out-_wilson_fermion_out)/cp.linalg.norm(wilson_fermion_out)}@")
    print(
        f"@cp.linalg.norm(clover_fermion_out-_clover_fermion_out)/cp.linalg.norm(clover_fermion_out):{cp.linalg.norm(clover_fermion_out-_clover_fermion_out)/cp.linalg.norm(clover_fermion_out)}@")
#############################
