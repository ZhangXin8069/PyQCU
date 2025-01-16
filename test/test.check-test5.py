import cupy as cp
from pyqcu import define
from pyqcu import io
from pyqcu.set import params
print('My rank is ', define.rank)
params[define._LAT_Y_] /= 2
params[define._LAT_XYZT_] /= 2
params[define._SET_PLAN_] = 2
params[define._GRID_T_] = 1
# params[define._GRID_T_] = 2
params[define._NODE_RANK_] = define.rank
params[define._NODE_SIZE_] = define.size
print("Parameters:", params)
#############################
fermion_out_filename = f"quda_wilson-clover-dslash-fermion-out_-{params[define._LAT_X_]}-{params[define._LAT_Y_]}-{params  [define._LAT_Z_]}-{params[define._LAT_T_]}-{params[define._LAT_XYZT_]}-{params[define._GRID_X_]}-{params[define._GRID_Y_]}-{params[define._GRID_Z_]}-{params[define._GRID_T_]}-{params[define._PARITY_]}-{params[define._NODE_RANK_]}-{params[define._NODE_SIZE_]}-{params[define._DAGGER_]}-f.h5"
# quda_fermion_out = io.hdf5_xxxtzyx2grid_xxxtzyx(params, fermion_out_filename)

quda_fermion_out = cp.fromfile(
    fermion_out_filename.replace(".h5", ".bin"), dtype=cp.complex64, count=params[define._LAT_XYZT_]*define._LAT_HALF_SC_)
quda_fermion_out = io.fermion2sctzyx(quda_fermion_out, params)
#############################
fermion_out = io.hdf5_xxxtzyx2grid_xxxtzyx(params)
#############################
print("Fermion out data:", fermion_out.data)
print("Fermion out shape:", fermion_out.shape)
print("QUDA Fermion out data:", quda_fermion_out.data)
print("QUDA Fermion out shape:", quda_fermion_out.shape)
print("Difference:", cp.linalg.norm(fermion_out -
      quda_fermion_out)/cp.linalg.norm(quda_fermion_out))
