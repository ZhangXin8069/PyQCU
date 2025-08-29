import numpy as np
import cupy as cp
from pyqcu.cuda import define
from pyqcu.cuda import io
from pyqcu.cuda import qcu
from pyqcu.cuda.set import params, argv, set_ptrs
params = np.array([0]*define._PARAMS_SIZE_, dtype=np.int32)
params[define._LAT_X_] = 32
params[define._LAT_Y_] = 16
params[define._LAT_Z_] = 32
params[define._LAT_T_] = 32
params[define._LAT_XYZT_] = params[define._LAT_X_] * \
    params[define._LAT_Y_]*params[define._LAT_Z_]*params[define._LAT_T_]
params[define._GRID_X_], params[define._GRID_Y_], params[define._GRID_Z_], params[
    define._GRID_T_] = define.split_into_four_factors(define.size)
params[define._PARITY_] = 0
params[define._NODE_RANK_] = define.rank
params[define._NODE_SIZE_] = define.size
params[define._DAGGER_] = 0
params[define._MAX_ITER_] = 1000
params[define._DATA_TYPE_] = define._LAT_C64_
params[define._SET_INDEX_] = 0
params[define._SET_PLAN_] = 2
params[define._MG_X_] = 1
params[define._MG_Y_] = 1
params[define._MG_Z_] = 1
params[define._MG_T_] = 1
params[define._LAT_E_] = 24
params[define._VERBOSE_] = 1
params[define._SEED_] = 42
argv = np.array([0.0]*define._ARGV_SIZE_,
                dtype=define.dtype_half(params[define._DATA_TYPE_]))
argv[define._MASS_] = 0.00
argv[define._TOL_] = 1e-12
argv[define._SIGMA_] = 0.1
set_ptrs = np.array(10*[0], dtype=np.int64)  # maybe more than 10?
gauge_filename = f"quda_wilson-clover-dslash-gauge_-32-16-32-32-524288-1-1-1-1-0-0-1-0-f.h5"
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
qcu.applyInitQcu(set_ptrs, params, argv)
qcu.applyCloverDslashQcu(fermion_out, fermion_in, gauge, set_ptrs, params)
qcu.applyEndQcu(set_ptrs, params)
#############################
print("Fermion out data:", fermion_out.data)
print("Fermion out shape:", fermion_out.shape)
print("QUDA Fermion out data:", quda_fermion_out.data)
print("QUDA Fermion out shape:", quda_fermion_out.shape)
print("Difference:", cp.linalg.norm(fermion_out -
      quda_fermion_out)/cp.linalg.norm(quda_fermion_out))
#############################