import re
import numpy as np
import cupy as cp
from pyqcu import define
from pyqcu import io
from pyqcu import qcu
from pyqcu.set import params, argv, set_ptrs
import h5py
print('My rank is ', define.rank)
print('My rank is ', define.rank)
params[define._SET_PLAN_] = 1
print("Parameters:", params)
#############################
gauge_filename = f"quda_wilson-bistabcg-gauge_-{params[define._LAT_X_]}-{params[define._LAT_Y_]}-{params  [define._LAT_Z_]}-{params[define._LAT_T_]}-{params[define._LAT_XYZT_]}-{params[define._GRID_X_]}-{params[define._GRID_Y_]}-{params[define._GRID_Z_]}-{params[define._GRID_T_]}-{params[define._PARITY_]}-{params[define._NODE_RANK_]}-{params[define._NODE_SIZE_]}-{params[define._DAGGER_]}-f.bin"
print("Gauge filename:", gauge_filename)
gauge = cp.fromfile(gauge_filename, dtype=cp.complex64,
                    count=params[define._LAT_XYZT_]*define._LAT_DCC_)
print("Gauge:", gauge)
print("Gauge data:", gauge.data)
print("Gauge shape:", gauge.shape)
fermion_in_filename = gauge_filename.replace("gauge", "fermion-in")
print("Fermion in filename:", fermion_in_filename)
fermion_in = cp.fromfile(fermion_in_filename, dtype=cp.complex64,
                         count=params[define._LAT_XYZT_]*define._LAT_SC_)
print("Fermion in:", fermion_in)
print("Fermion in data:", fermion_in.data)
print("Fermion in shape:", fermion_in.shape)
fermion_out_filename = gauge_filename.replace("gauge", "fermion-out")
print("Fermion out filename:", fermion_out_filename)
fermion_out = cp.zeros(
    params[define._LAT_XYZT_]*define._LAT_SC_, dtype=cp.complex64)
print("Fermion out:", fermion_out)
print("Fermion out data:", fermion_out.data)
print("Fermion out shape:", fermion_out.shape)
#############################
set_ptrs = np.array(params, dtype=np.int64)
print("Set pointers:", set_ptrs)
print("Set pointers data:", set_ptrs.data)
qcu.applyInitQcu(set_ptrs, params, argv)
qcu.applyWilsonBistabCgQcu(fermion_out, fermion_in, gauge, set_ptrs, params)
print("Fermion out:", fermion_out)
print("Fermion out data:", fermion_out.data)
print("Fermion out shape:", fermion_out.shape)
quda_fermion_out = cp.fromfile(
    fermion_out_filename, dtype=cp.complex64, count=params[define._LAT_XYZT_]*define._LAT_SC_)
print("QUDA Fermion out:", quda_fermion_out)
print("QUDA Fermion out data:", quda_fermion_out.data)
print("QUDA Fermion out shape:", quda_fermion_out.shape)
print("Difference:", cp.linalg.norm(fermion_out -
      quda_fermion_out)/cp.linalg.norm(quda_fermion_out))
qcu.applyEndQcu(set_ptrs, params)
gauge = io.gauge2ccdptzyx(gauge, params)
fermion_in = io.fermion2psctzyx(fermion_in, params)
quda_fermion_out = io.fermion2psctzyx(quda_fermion_out, params)
io.grid_xxxtzyx2hdf5_xxxtzyx(
    gauge, params, gauge_filename.replace(".bin", ".h5"))
io.grid_xxxtzyx2hdf5_xxxtzyx(
    fermion_in, params, fermion_in_filename.replace(".bin", ".h5"))
io.grid_xxxtzyx2hdf5_xxxtzyx(
    quda_fermion_out, params, fermion_out_filename.replace(".bin", ".h5"))
