import re
import numpy as np
import cupy as cp
from pyqcu.define import _X_, _Y_, _Z_, _T_, _LAT_X_, _LAT_Y_, _LAT_Z_, _LAT_T_, _LAT_XYZT_, _GRID_X_, _GRID_Y_, _GRID_Z_, _GRID_T_, _PARITY_, _NODE_RANK_, _NODE_SIZE_, _DAGGER_, _MAX_ITER_, _SET_INDEX_, _SET_PLAN_, _PARAMS_SIZE_, _LAT_DCC_, _LAT_SC_, _DATA_TYPE_, rank, size, _ARGV_SIZE_, _MASS_, _TOL_

print('My rank is ', rank)
if rank == 0:
    params = np.array([0]*_PARAMS_SIZE_, dtype=np.int32)
    params[_LAT_X_] = 32
    params[_LAT_Y_] = 32
    params[_LAT_Z_] = 32
    params[_LAT_T_] = 32
    params[_LAT_XYZT_] = 1048576
    params[_GRID_X_] = 1
    params[_GRID_Y_] = 1
    params[_GRID_Z_] = 1
    params[_GRID_T_] = 1
    params[_PARITY_] = 0
    params[_NODE_RANK_] = 0
    params[_NODE_SIZE_] = 1
    params[_DAGGER_] = 0
    params[_MAX_ITER_] = 1e3
    params[_DATA_TYPE_] = 0
    params[_SET_INDEX_] = 2
    params[_SET_PLAN_] = 1
    argv = np.array([0.0]*_ARGV_SIZE_, dtype=np.float32)
    argv[_MASS_] = 0.0
    argv[_TOL_] = 1e-9
    gauge_filename = f"quda_wilson-bistabcg-gauge_-{params[_LAT_X_]}-{params[_LAT_Y_]}-{params  [_LAT_Z_]}-{params[_LAT_T_]}-{params[_LAT_XYZT_]}-{params[_GRID_X_]}-{params[_GRID_Y_]}-{params[_GRID_Z_]}-{params[_GRID_T_]}-{params[_PARITY_]}-{params[_NODE_RANK_]}-{params[_NODE_SIZE_]}-{params[_DAGGER_]}-f.bin"
    gauge = cp.fromfile(gauge_filename, dtype=cp.complex64,
                        count=params[_LAT_XYZT_]*_LAT_DCC_)
    print("Gauge:", gauge)
    print("Gauge data:", gauge.data)
    print("Gauge shape:", gauge.shape)
    #############################
    params = np.array([0]*_PARAMS_SIZE_, dtype=np.int32)
    params[_GRID_X_] = 2
    params[_GRID_Y_] = 1
    params[_GRID_Z_] = 1
    params[_GRID_T_] = 1
    params[_NODE_RANK_] = rank
    params[_NODE_SIZE_] = size
