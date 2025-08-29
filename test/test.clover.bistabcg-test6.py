import cupy as cp
from pyqcu.cuda import define
from pyqcu.cuda import io
# import pyqcu.cuda.gauge as Gauge
from pyqcu.cuda import qcu
from pyqcu.cuda.set import params, argv, set_ptrs
#############################
params[define._LAT_X_] = 32
params[define._LAT_Y_] = 64
params[define._LAT_Z_] = 64
params[define._LAT_T_] = 64
params[define._LAT_XYZT_] = params[define._LAT_X_] * \
    params[define._LAT_Y_]*params[define._LAT_Z_]*params[define._LAT_T_]
params[define._GRID_X_], params[define._GRID_Y_], params[define._GRID_Z_], params[
    define._GRID_T_] = define.split_into_four_factors(define.size)
params[define._DATA_TYPE_] = define._LAT_C128_
params[define._MAX_ITER_] = 1000
argv[define._MASS_] = 0.05
argv = argv.astype(define.dtype_half(params[define._DATA_TYPE_]))
#############################
gauge = io.get_or_give_gauge(params, "gauge.h5")
clover = io.get_or_give_clover(params, "clover.h5")
fermion_in = io.get_or_give_fermion_in(params, "fermion_in.h5")
#############################
params[define._VERBOSE_] = 1
params[define._SET_INDEX_] += 1
params[define._SET_PLAN_] = 0
#############################
gauge = cp.zeros_like(gauge)
fermion_in = cp.ones_like(fermion_in)
fermion_in = cp.random.randn(fermion_in.size).astype(
    fermion_in.dtype).reshape(fermion_in.shape)
fermion_out = cp.zeros_like(fermion_in)
clover_ee = cp.zeros_like(clover)
clover_ee_inv = cp.zeros_like(clover)
clover_oo = cp.zeros_like(clover)
clover_oo_inv = cp.zeros_like(clover)
#############################
# _gauge = cp.zeros_like(gauge)
# qcu.applyInitQcu(set_ptrs, params, argv)
# qcu.applyGaussGaugeQcu(_gauge, set_ptrs, params)
# qcu.applyEndQcu(set_ptrs, params)
# io.grid_xxxtzyx2hdf5_xxxtzyx(_gauge, params, "_gauge.h5")
#############################
qcu.applyInitQcu(set_ptrs, params, argv)
qcu.applyGaussGaugeQcu(gauge, set_ptrs, params)
qcu.applyEndQcu(set_ptrs, params)
#############################
# gauge = Gauge.give_gauss_SU3(
#     dtype=gauge.dtype, size=gauge.size//define._LAT_CC_).transpose(1, 2, 0).reshape(gauge.shape)
#############################
# Gauge.test_su3(gauge[:, :, -1, -1, -1, -1, -1, -1])
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
#############################
qcu.applyInitQcu(set_ptrs, params, argv)
qcu.applyCloversQcu(clover_ee, clover_ee_inv, gauge, set_ptrs, params)
qcu.applyEndQcu(set_ptrs, params)
#############################
params[define._VERBOSE_] = 1
params[define._SET_INDEX_] += 1
params[define._SET_PLAN_] = 2
#############################
qcu.applyInitQcu(set_ptrs, params, argv)
qcu.applyCloversQcu(clover_oo, clover_oo_inv, gauge, set_ptrs, params)
qcu.applyEndQcu(set_ptrs, params)
#############################
params[define._VERBOSE_] = 1
params[define._SET_INDEX_] += 1
params[define._SET_PLAN_] = 1
#############################
# clover_ee = io.clover2I(input_array=clover_ee)
# clover_oo = io.clover2I(input_array=clover_oo)
# clover_ee_inv = io.clover2I(input_array=clover_ee_inv)
# clover_oo_inv = io.clover2I(input_array=clover_oo_inv)
#############################
# print(clover_ee.shape)
# print(clover_ee.size)
# print(clover_ee[:, :, :, :, -1, -1, -1, -1].reshape(12,12))
# print(cp.einsum('abcd,cdAB->abAB', clover_ee[:, :, :, :, -1, -1, -1, -1],
#       clover_ee_inv[:, :, :, :, -1, -1, -1, -1]))
#############################
print("@fermion_out.data.ptr:", fermion_out.data.ptr)
print("@fermion_in.data.ptr:", fermion_in.data.ptr)
print("@gauge.data.ptr:", gauge.data.ptr)
print("@clover_ee.data.ptr:", clover_ee.data.ptr)
print("@clover_oo.data.ptr:", clover_oo.data.ptr)
print("@clover_ee_inv.data.ptr:", clover_ee_inv.data.ptr)
print("@clover_oo_inv.data.ptr:", clover_oo_inv.data.ptr)
print("@set_ptrs.ctypes.data:", set_ptrs.ctypes.data)
print("@params.ctypes.data:", params.ctypes.data)
# #############################
qcu.applyInitQcu(set_ptrs, params, argv)
qcu.applyCloverBistabCgQcu(fermion_out, fermion_in,
                           gauge, clover_ee, clover_oo, clover_ee_inv, clover_oo_inv,  set_ptrs, params)
qcu.applyEndQcu(set_ptrs, params)
# #############################
io.try_give(gauge, params, "gauge.h5")
io.try_give(clover, params, "clover.h5")
io.try_give(fermion_in, params, "fermion_in.h5")
io.try_give(fermion_out, params, "fermion_out.h5")
#############################
