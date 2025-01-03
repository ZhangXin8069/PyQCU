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
    params[define._SET_PLAN_] = 0
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
    gauge = io.gauge2ccdptzyx(gauge, params)
    print("Gauge:", gauge)
    print("Gauge data:", gauge.data)
    print("Gauge shape:", gauge.shape)
    fermion_in_filename = gauge_filename.replace("gauge", "fermion-in")
    print("Fermion in filename:", fermion_in_filename)
    fermion_in = cp.fromfile(fermion_in_filename, dtype=cp.complex64,
                             count=params[define._LAT_XYZT_]*define._LAT_HALF_SC_)
    fermion_in = io.fermion2sctzyx(fermion_in, params)
    print("Fermion in:", fermion_in)
    print("Fermion in data:", fermion_in.data)
    print("Fermion in shape:", fermion_in.shape)
    fermion_out_filename = gauge_filename.replace("gauge", "fermion-out")
    print("Fermion out filename:", fermion_out_filename)
    fermion_out = cp.zeros(
        params[define._LAT_XYZT_]*define._LAT_HALF_SC_, dtype=cp.complex64)
    fermion_out = io.fermion2sctzyx(fermion_out, params)
    print("Fermion out:", fermion_out)
    print("Fermion out data:", fermion_out.data)
    print("Fermion out shape:", fermion_out.shape)
    #############################
    set_ptrs = np.array(params, dtype=np.int64)
    print("Set pointers:", set_ptrs)
    print("Set pointers data:", set_ptrs.data)
    qcu.applyInitQcu(set_ptrs, params, argv)
    qcu.applyWilsonDslashQcu(fermion_out, fermion_in, gauge, set_ptrs, params)
    print("Fermion out:", fermion_out)
    print("Fermion out data:", fermion_out.data)
    print("Fermion out shape:", fermion_out.shape)
    quda_fermion_out = cp.fromfile(
        fermion_out_filename, dtype=cp.complex64, count=params[define._LAT_XYZT_]*define._LAT_HALF_SC_)
    fermion_out = io.array2xxx(fermion_out)
    print("QUDA Fermion out:", quda_fermion_out)
    print("QUDA Fermion out data:", quda_fermion_out.data)
    print("QUDA Fermion out shape:", quda_fermion_out.shape)
    print("Difference:", cp.linalg.norm(fermion_out -
          quda_fermion_out)/cp.linalg.norm(quda_fermion_out))
    qcu.applyEndQcu(set_ptrs, params)
    
