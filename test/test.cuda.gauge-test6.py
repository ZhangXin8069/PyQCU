import numpy as np
import cupy as cp
from pyqcu.cuda import define
from pyqcu.cuda import io
from pyqcu.cuda import qcu
from pyqcu.cuda.set import params, argv, set_ptrs
print('My rank is ', define.rank)
params[define._LAT_X_] = 4
params[define._LAT_Y_] = 8
params[define._LAT_Z_] = 16
params[define._LAT_T_] = 8
params[define._LAT_XYZT_] = params[define._LAT_X_] * \
    params[define._LAT_Y_]*params[define._LAT_Z_]*params[define._LAT_T_]
params[define._GRID_X_] = 1
params[define._GRID_Y_] = 1
params[define._GRID_Z_] = 1
params[define._GRID_T_] = 1
params[define._PARITY_] = 0
params[define._NODE_RANK_] = define.rank
params[define._NODE_SIZE_] = define.size
params[define._DATA_TYPE_] = define._LAT_C64_
dtype = define.dtype(_data_type_=params[define._DATA_TYPE_])
params[define._VERBOSE_] = 0
print("Parameters:", params)
argv[define._MASS_] = 0.0
print("Arguments:", argv)
#############################
gauge = cp.zeros(shape=[define._LAT_C_, define._LAT_C_, define._LAT_D_, params[define._LAT_T_],
                 params[define._LAT_Z_], params[define._LAT_Y_], params[define._LAT_X_]], dtype=dtype)
gauge = io.xxxtzyx2pxxxtzyx(input_array=gauge)
print("Gauge Shape:", gauge.shape)
#############################
qcu.applyInitQcu(set_ptrs, params, argv)
qcu.applyGaussGaugeQcu(gauge, set_ptrs, params)
qcu.applyEndQcu(set_ptrs, params)
#############################
