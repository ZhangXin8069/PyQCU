import ctypes
from pyqcu import qcu
import cupy as cp
import numpy as np
import re
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print('My rank is ', rank)

gauge_filename = "wilson-bistabcg-gauge_-32-32-32-32-1048576-1-1-1-1-0-0-1-0-f.bin"
pattern = r"-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-"
match = re.search(pattern, gauge_filename)
if match:
    param = [int(num) for num in match.groups()]
    print("Extracted integers:", param)
    param.append(1000)
    py_params = np.array(param, dtype=np.int32)
    print("NumPy Array:", py_params)
    print("Numpy data pointer:", py_params.data)
    py_argv = np.array([0,1e-9], dtype=np.float32)
    print("Argv:", py_argv)
    print("Argv data pointer:", py_argv.data)
    _LAT_XYZT_ = 4
    _LAT_DCC_ = 36
    _LAT_SC_ = 12
    _EVEN_ODD = 2
    size = py_params[_LAT_XYZT_]*_LAT_DCC_
    gauge_filename = gauge_filename.replace("gauge", "gauge")
    py_gauge = cp.fromfile(gauge_filename, dtype=cp.complex64, count=size)
    print("Gauge:", py_gauge)
    print("Gauge data:", py_gauge.data)
    size = py_params[_LAT_XYZT_]*_LAT_SC_
    fermion_in_filename = gauge_filename.replace("gauge", "fermion-in")
    py_fermion_in = cp.fromfile(
        fermion_in_filename, dtype=cp.complex64, count=size)
    print("Fermion in:", py_fermion_in)
    print("Fermion in data:", py_fermion_in.data)
    fermion_out_filename = gauge_filename.replace("gauge", "fermion-out")
    py_fermion_out = cp.fromfile(
        fermion_out_filename, dtype=cp.complex64, count=size)
    print("Fermion out:", py_fermion_out)
    print("Fermion out data:", py_fermion_out.data)
    qcu.applyBistabCgQcu(py_fermion_out, py_fermion_in, py_gauge, py_params, py_argv)
else:
    print("No match found!")
