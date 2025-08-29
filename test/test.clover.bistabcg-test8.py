import cupy as cp
from pyqcu.cuda import define
from pyqcu.cuda import io
from pyqcu.cuda import qcu
from pyqcu.cuda.tmp_set import params, argv, set_ptrs
#############################
gauge = io.get_or_give(params, "gauge-test8.h5")
clover = io.get_or_give(params, "clover-test8.h5")
fermion_in = io.get_or_give(params, "fermion_in-test8.h5")
_fermion_out = io.get_or_give(params, "fermion_out-test8.h5")
fermion_out = cp.zeros_like(_fermion_out)
clover_ee = cp.zeros_like(clover)
clover_ee_inv = cp.zeros_like(clover)
clover_oo = cp.zeros_like(clover)
clover_oo_inv = cp.zeros_like(clover)
#############################
params[define._VERBOSE_] = 1
params[define._SET_INDEX_] += 1
params[define._SET_PLAN_] = 1
#############################
qcu.applyInitQcu(set_ptrs, params, argv)
qcu.applyWilsonBistabCgQcu(
    fermion_out, fermion_in, gauge, set_ptrs, params)
qcu.applyEndQcu(set_ptrs, params)
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
qcu.applyCloverBistabCgQcu(fermion_out, fermion_in,
                           gauge, clover_ee, clover_oo, clover_ee_inv, clover_oo_inv,  set_ptrs, params)
qcu.applyEndQcu(set_ptrs, params)
#############################
io.grid_xxxtzyx2hdf5_xxxtzyx(fermion_out, params, "fermion_out-test8.h5")
#############################
print(
    f"@cp.linalg.norm(fermion_out-_fermion_out)/cp.linalg.norm(fermion_out):{cp.linalg.norm(fermion_out-_fermion_out)/cp.linalg.norm(fermion_out)}@")
#############################
