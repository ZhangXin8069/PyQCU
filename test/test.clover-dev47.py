import re
import numpy as np
import cupy as cp
from pyqcu import define
from pyqcu import io
from pyqcu import qcu
from pyqcu.set import params, argv, set_ptrs
import h5py
print('My rank is ', define.rank)
params[define._LAT_Y_] /= 2
params[define._LAT_XYZT_] /= 2
params[define._SET_PLAN_] = 2
gauge_filename = f"quda_wilson-clover-dslash-gauge_-{params[define._LAT_X_]}-{params[define._LAT_Y_]}-{params  [define._LAT_Z_]}-{params[define._LAT_T_]}-{params[define._LAT_XYZT_]}-{params[define._GRID_X_]}-{params[define._GRID_Y_]}-{params[define._GRID_Z_]}-{params[define._GRID_T_]}-{params[define._PARITY_]}-{params[define._NODE_RANK_]}-{params[define._NODE_SIZE_]}-{params[define._DAGGER_]}-f.h5"
params[define._GRID_T_] = 1
params[define._NODE_RANK_] = define.rank
params[define._NODE_SIZE_] = define.size
params[define._DATA_TYPE_] = define._LAT_C64_
print("Parameters:", params)
#############################
print("Gauge filename:", gauge_filename)
gauge = io.hdf5_xxxtzyx2grid_xxxtzyx(params, gauge_filename)
fermion_in_filename = gauge_filename.replace("gauge", "fermion-in")
print("Fermion in filename:", fermion_in_filename)
fermion_in = io.hdf5_xxxtzyx2grid_xxxtzyx(params, fermion_in_filename)
fermion_out_filename = gauge_filename.replace("gauge", "fermion-out")
print("Fermion out filename:", fermion_out_filename)
quda_fermion_out = io.hdf5_xxxtzyx2grid_xxxtzyx(params, fermion_out_filename)
#############################
fermion_out = cp.zeros_like(fermion_in)
print("Fermion out data:", fermion_out.data)
print("Fermion out shape:", fermion_out.shape)
#############################
params[define._DATA_TYPE_] = define._LAT_C128_
_fermion_in = fermion_in.astype(define.dtype(params[define._DATA_TYPE_]))
_gauge = gauge.astype(define.dtype(params[define._DATA_TYPE_]))
_fermion_out = fermion_out.astype(define.dtype(params[define._DATA_TYPE_]))
_argv = argv.astype(define.dtype(params[define._DATA_TYPE_]))
qcu.applyInitQcu(set_ptrs, params, _argv)
clover_even = cp.zeros((define._LAT_S_, define._LAT_C_, define._LAT_S_, define._LAT_C_,
                       params[define._LAT_T_], params[define._LAT_Z_], params[define._LAT_Y_], int(params[define._LAT_X_]/define._LAT_P_),), dtype=_fermion_in.dtype)
qcu.applyCloverQcu(clover_even, _gauge, set_ptrs, params)
qcu.applyDslashQcu(_fermion_out, _fermion_in,
                   clover_even, _gauge, set_ptrs, params)
# qcu.applyCloverDslashQcu(_fermion_out, _fermion_in, _gauge, set_ptrs, params)
qcu.applyEndQcu(set_ptrs, params)
params[define._DATA_TYPE_] = define._LAT_C64_
fermion_in = _fermion_in.astype(define.dtype(params[define._DATA_TYPE_]))
gauge = _gauge.astype(define.dtype(params[define._DATA_TYPE_]))
fermion_out = _fermion_out.astype(define.dtype(params[define._DATA_TYPE_]))
argv = _argv.astype(define.dtype(params[define._DATA_TYPE_]))
#############################
print("Fermion out data:", fermion_out.data)
print("Fermion out shape:", fermion_out.shape)
print("QUDA Fermion out data:", quda_fermion_out.data)
print("QUDA Fermion out shape:", quda_fermion_out.shape)
print("Difference:", cp.linalg.norm(fermion_out -
      quda_fermion_out)/cp.linalg.norm(quda_fermion_out))
#############################
io.grid_xxxtzyx2hdf5_xxxtzyx(fermion_out, params)
