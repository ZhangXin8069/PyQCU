import cupy as cp
from pyqcu.cuda import define
from pyqcu.cuda import io
from pyqcu.cuda import qcu
from pyqcu.cuda.set import params, argv, set_ptrs
#############################
params[define._LAT_X_] = 16
params[define._LAT_Y_] = 16
params[define._LAT_Z_] = 16
params[define._LAT_T_] = 16
params[define._LAT_XYZT_] = params[define._LAT_X_] * \
    params[define._LAT_Y_]*params[define._LAT_Z_]*params[define._LAT_T_]
params[define._GRID_X_], params[define._GRID_Y_], params[define._GRID_Z_], params[
    define._GRID_T_] = define.split_into_four_factors(define.size)
params[define._DATA_TYPE_] = define._LAT_C128_
params[define._MAX_ITER_] = 1000
argv[define._MASS_] = 0.05
argv = argv.astype(define.dtype_half(params[define._DATA_TYPE_]))
#############################
gauge = io.get_or_give(params, "gauge-test8.h5")
clover = io.get_or_give(params, "clover-test8.h5")
fermion_in = io.get_or_give(params, "fermion_in-test8.h5")
fermion_out = io.get_or_give(params, "fermion_out-test8.h5")
#############################
params[define._VERBOSE_] = 1
params[define._SET_INDEX_] += 1
params[define._SET_PLAN_] = 0
#############################
gauge = cp.zeros_like(gauge)
clover = cp.zeros_like(clover)
fermion_in = cp.ones_like(fermion_in)
fermion_out = cp.zeros_like(fermion_in)
#############################
qcu.applyInitQcu(set_ptrs, params, argv)
qcu.applyGaussGaugeQcu(gauge, set_ptrs, params)
qcu.applyEndQcu(set_ptrs, params)
#############################
io.try_give(gauge, params, "gauge-test8.h5")
io.try_give(clover, params, "clover-test8.h5")
io.try_give(fermion_in, params, "fermion_in-test8.h5")
io.try_give(fermion_out, params, "fermion_out-test8.h5")
#############################
