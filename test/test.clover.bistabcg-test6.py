import cupy as cp
from pyqcu.cuda import define
from pyqcu.cuda import io
from pyqcu.cuda import qcu
from pyqcu.cuda.set import params, argv, set_ptrs
print('My rank is ', define.rank)
print(f"print(cp.cuda.Device().id):{print(cp.cuda.Device().id)}")
params[define._LAT_Y_] /= 2
params[define._LAT_XYZT_] /= 2
params[define._SET_PLAN_] = 2
params[define._DATA_TYPE_] = define._LAT_C64_
params[define._VERBOSE_] = 1
params[define._PARITY_] = 0
gauge_filename = f"quda_wilson-bistabcg-gauge_-32-32-32-32-1048576-1-1-1-1-0-0-1-0-f.h5"
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
fermion_in = cp.ones_like(fermion_in)
fermion_out = cp.zeros_like(fermion_in)
clover_ee = cp.zeros(shape=[define._LAT_S_, define._LAT_C_] +
                     list(fermion_in.shape), dtype=fermion_in.dtype)
clover_ee_inv = cp.zeros_like(clover_ee)
clover_oo = cp.zeros_like(clover_ee)
clover_oo_inv = cp.zeros_like(clover_ee)
#############################
params[define._VERBOSE_] = 1
params[define._PARITY_] = 0
params[define._SET_INDEX_] += 1
params[define._SET_PLAN_] = 2
#############################
qcu.applyInitQcu(set_ptrs, params, argv)
qcu.applyCloversQcu(clover_ee, clover_ee_inv, gauge, set_ptrs, params)
qcu.applyEndQcu(set_ptrs, params)
#############################
params[define._VERBOSE_] = 1
params[define._PARITY_] = 1
params[define._SET_INDEX_] += 1
params[define._SET_PLAN_] = 2
#############################
qcu.applyInitQcu(set_ptrs, params, argv)
qcu.applyCloversQcu(clover_oo, clover_oo_inv, gauge, set_ptrs, params)
qcu.applyEndQcu(set_ptrs, params)
#############################
params[define._VERBOSE_] = 1
params[define._PARITY_] = 0
params[define._SET_INDEX_] += 1
params[define._SET_PLAN_] = 1
#############################
clover_ee = io.clover2I(input_array=clover_ee)
clover_oo = io.clover2I(input_array=clover_oo)
clover_ee_inv = io.clover2I(input_array=clover_ee_inv)
clover_oo_inv = io.clover2I(input_array=clover_oo_inv)
print("@fermion_out.data.ptr:", fermion_out.data.ptr)
print("@fermion_in.data.ptr:", fermion_in.data.ptr)
print("@gauge.data.ptr:", gauge.data.ptr)
print("@clover_ee.data.ptr:", clover_ee.data.ptr)
print("@clover_oo.data.ptr:", clover_oo.data.ptr)
print("@clover_ee_inv.data.ptr:", clover_ee_inv.data.ptr)
print("@clover_oo_inv.data.ptr:", clover_oo_inv.data.ptr)
print("@set_ptrs.ctypes.data:", set_ptrs.ctypes.data)
print("@params.ctypes.data:", params.ctypes.data)
#############################
qcu.applyInitQcu(set_ptrs, params, argv)
qcu.applyCloverBistabCgQcu(fermion_out, fermion_in,
                           gauge, clover_ee, clover_oo, clover_ee_inv, clover_oo_inv,  set_ptrs, params)
qcu.applyEndQcu(set_ptrs, params)
#############################
