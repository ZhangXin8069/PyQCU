import pyqcu.cuda.define as define
import numpy as np
params = np.array([0]*define._PARAMS_SIZE_, dtype=np.int32)
params[define._LAT_X_] = 32
params[define._LAT_Y_] = 32
params[define._LAT_Z_] = 32
params[define._LAT_T_] = 32
params[define._LAT_XYZT_] = 32**4
params[define._GRID_X_] = 1
params[define._GRID_Y_] = 1
params[define._GRID_Z_] = 1
params[define._GRID_T_] = 1
params[define._PARITY_] = 0
params[define._NODE_RANK_] = 0
params[define._NODE_SIZE_] = 1
params[define._DAGGER_] = 0
params[define._MAX_ITER_] = 1e4
params[define._DATA_TYPE_] = define._LAT_C64_
params[define._SET_INDEX_] = 0
params[define._SET_PLAN_] = 0
params[define._MG_X_] = 4
params[define._MG_Y_] = 4
params[define._MG_Z_] = 4
params[define._MG_T_] = 8
params[define._LAT_E_] = 24
params[define._VERBOSE_] = 0
print("Parameters:", params)
print("Parameters data:", params.data)
argv = np.array([0.0]*define._ARGV_SIZE_, dtype=np.float32)
argv[define._MASS_] = -3.5  # make kappa=1.0
argv[define._TOL_] = 1e-9
print("Arguments:", argv)
print("Arguments data:", argv.data)
set_ptrs = np.array(10*[0], dtype=np.int64)  # maybe more than 10?
print("Set pointers:", set_ptrs)
print("Set pointers data:", set_ptrs.data)
