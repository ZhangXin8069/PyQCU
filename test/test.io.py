import re
import numpy as np
import cupy as cp
from pyqcu import define
from pyqcu import io
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
    params[define._SET_PLAN_] = 1
    argv = np.array([0.0]*define._ARGV_SIZE_, dtype=np.float32)
    argv[define._MASS_] = 0.0
    argv[define._TOL_] = 1e-9
    #############################
    gauge_filename = f"quda_wilson-bistabcg-gauge_-{params[define._LAT_X_]}-{params[define._LAT_Y_]}-{params  [define._LAT_Z_]}-{params[define._LAT_T_]}-{params[define._LAT_XYZT_]}-{params[define._GRID_X_]}-{params[define._GRID_Y_]}-{params[define._GRID_Z_]}-{params[define._GRID_T_]}-{params[define._PARITY_]}-{params[define._NODE_RANK_]}-{params[define._NODE_SIZE_]}-{params[define._DAGGER_]}-f.bin"
    print("Gauge filename:", gauge_filename)
    gauge = cp.fromfile(gauge_filename, dtype=cp.complex64,
                        count=params[define._LAT_XYZT_]*define._LAT_DCC_)
    print("Gauge:", gauge)
    print("Gauge data:", gauge.data)
    print("Gauge shape:", gauge.shape)
    #############################
    # x->y->z->t->p->d->c->c
    _gauge = gauge.reshape((define._LAT_C_, define._LAT_C_, define._LAT_D_, define._LAT_P_,
                            params[define._LAT_T_], params[define._LAT_Z_], params[define._LAT_Y_], int(params[define._LAT_X_]/define._LAT_P_)))
    U = _gauge[:, :, 0, 0, 0, 0, 0, 0].reshape(define._LAT_C_ * define._LAT_C_)
    _U = cp.array([0.0+0.0j]*9, dtype=cp.complex64)
    _U[6] = (U[1] * U[5] - U[2] * U[4]).conj()
    _U[7] = (U[2] * U[3] - U[0] * U[5]).conj()
    _U[8] = (U[0] * U[4] - U[1] * U[3]).conj()
    print("U:", U)
    print("_U:", _U)
    print("Gauge:", gauge.size)
    #############################
    gauge_filename = f"quda_wilson-bistabcg-gauge_-{params[define._LAT_X_]}-{params[define._LAT_Y_]}-{params  [define._LAT_Z_]}-{params[define._LAT_T_]}-{params[define._LAT_XYZT_]}-{params[define._GRID_X_]}-{params[define._GRID_Y_]}-{params[define._GRID_Z_]}-{params[define._GRID_T_]}-{params[define._PARITY_]}-{params[define._NODE_RANK_]}-{params[define._NODE_SIZE_]}-{params[define._DAGGER_]}-f.bin"
    print("Gauge filename:", gauge_filename)
    gauge = cp.fromfile(gauge_filename, dtype=cp.complex64,
                        count=params[define._LAT_XYZT_]*define._LAT_DCC_)
    print("Gauge:", gauge)
    print("Gauge data:", gauge.data)
    print("Gauge shape:", gauge.shape)
    qcu_gauge = io.gauge2ccdptzyx(gauge, params)
    #############################
    gauge_filename = f"wilson-bistabcg-gauge_-{params[define._LAT_X_]}-{params[define._LAT_Y_]}-{params  [define._LAT_Z_]}-{params[define._LAT_T_]}-{params[define._LAT_XYZT_]}-{params[define._GRID_X_]}-{params[define._GRID_Y_]}-{params[define._GRID_Z_]}-{params[define._GRID_T_]}-{params[define._PARITY_]}-{params[define._NODE_RANK_]}-{params[define._NODE_SIZE_]}-{params[define._DAGGER_]}-f.bin"
    print("Gauge filename:", gauge_filename)
    gauge = cp.fromfile(gauge_filename, dtype=cp.complex64,
                        count=params[define._LAT_XYZT_]*define._LAT_DCC_)
    print("Gauge:", gauge)
    print("Gauge data:", gauge.data)
    print("Gauge shape:", gauge.shape)
    quda_gauge = io.gauge2dptzyxcc(gauge, params)
    #############################
    _gauge = io.ccdptzyx2dptzyxcc(qcu_gauge)
    print("differece:", cp.linalg.norm(_gauge-quda_gauge))
