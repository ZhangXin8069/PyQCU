import re
import numpy as np
import cupy as cp
from pyqcu import define
from pyqcu import io
from pyqcu import qcu

print('My rank is ', define.rank)
if define.rank == 0:
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
    params[define._MAX_ITER_] = 1e3
    params[define._DATA_TYPE_] = 0
    params[define._SET_INDEX_] = 2
    params[define._SET_PLAN_] = -1
    argv = np.array([0.0]*define._ARGV_SIZE_, dtype=np.float32)
    argv[define._MASS_] = 0.0
    argv[define._TOL_] = 1e-9
    print("Parameters:", params)
    print("Arguments:", argv)
    #############################
    gauge_filename = f"quda_wilson-dslash-gauge_-{params[define._LAT_X_]}-{params[define._LAT_Y_]}-{params  [define._LAT_Z_]}-{params[define._LAT_T_]}-{params[define._LAT_XYZT_]}-{params[define._GRID_X_]}-{params[define._GRID_Y_]}-{params[define._GRID_Z_]}-{params[define._GRID_T_]}-{params[define._PARITY_]}-{params[define._NODE_RANK_]}-{params[define._NODE_SIZE_]}-{params[define._DAGGER_]}-f.bin"
    print("Gauge filename:", gauge_filename)
    gauge = cp.fromfile(gauge_filename, dtype=cp.complex64,
                        count=params[define._LAT_XYZT_]*define._LAT_DCC_)
    print("Gauge:", gauge)
    print("Gauge data:", gauge.data)
    print("Gauge shape:", gauge.shape)
    #############################
    # ccdpdtzyx -> ccdzyx
    laplacian_gauge = cp.sum(a=io.gauge2ccdptzyx(gauge, params), axis=(3, 5))
    print("Laplacian Gauge:", laplacian_gauge)
    print("Laplacian Gauge data:", laplacian_gauge.data)
    print("Laplacian Gauge shape:", laplacian_gauge.shape)
    #############################
    params[define._LAT_XYZT_] /= params[define._LAT_T_]
    params[define._LAT_T_] = 1
    laplacian_gauge = io.array2xxx(laplacian_gauge)
    print("Laplacian Gauge:", laplacian_gauge)
    print("Laplacian Gauge data:", laplacian_gauge.data)
    print("Laplacian Gauge shape:", laplacian_gauge.shape)
    #############################
    laplacian_out = cp.zeros(
        params[define._LAT_XYZT_]*define._LAT_SC_, dtype=cp.complex64)
    print("Laplacian out:", laplacian_out)
    print("Laplacian out data:", laplacian_out.data)
    print("Laplacian out shape:", laplacian_out.shape)
    laplacian_in = cp.ones(
        params[define._LAT_XYZT_]*define._LAT_SC_, dtype=cp.complex64)
    print("Laplacian in:", laplacian_in)
    print("Laplacian in data:", laplacian_in.data)
    print("Laplacian in shape:", laplacian_in.shape)
    #############################
    set_ptrs = np.array(params, dtype=np.int64)
    print("Set pointers:", set_ptrs)
    print("Set pointers data:", set_ptrs.data)
    qcu.applyInitQcu(set_ptrs, params, argv)
    qcu.applyLaplacianQcu(laplacian_out, laplacian_in, gauge, set_ptrs, params)
    print("Laplacian out:", laplacian_out)
    print("Laplacian out data:", laplacian_out.data)
    print("Laplacian out shape:", laplacian_out.shape)
    qcu.applyEndQcu(set_ptrs, params)
