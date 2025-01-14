import define
import numpy as np
params = np.array([0]*define._PARAMS_SIZE_, dtype=np.int32)
params[define._LAT_X_] = 32
params[define._LAT_Y_] = 32
params[define._LAT_Z_] = 32
params[define._LAT_T_] = 32
params[define._LAT_XYZT_] = 1048576
params[define._GRID_X_] = 1
params[define._GRID_Y_] = 1
params[define._GRID_Z_] = 1
params[define._GRID_T_] = 1
params[define._PARITY_] = 0
params[define._NODE_RANK_] = 0
params[define._NODE_SIZE_] = 1
params[define._DAGGER_] = 0
params[define._MAX_ITER_] = 1e4
params[define._DATA_TYPE_] = 0
params[define._SET_INDEX_] = 2
params[define._SET_PLAN_] = 0
print("Parameters:", params)
argv = np.array([0.0]*define._ARGV_SIZE_, dtype=np.float32)
argv[define._MASS_] = 0.0
argv[define._TOL_] = 1e-9
print("Arguments:", argv)
set_ptrs = np.array(params, dtype=np.int64)
print("Set pointers:", set_ptrs)
print("Set pointers data:", set_ptrs.data)
