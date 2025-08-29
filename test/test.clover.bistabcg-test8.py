import os
import cupy as cp
from pyqcu.cuda import define
from pyqcu.cuda import io
from pyqcu.cuda import qcu
from pyqcu.cuda.tmp_set import params, argv, set_ptrs
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
