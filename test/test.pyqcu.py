import ctypes
from pyqcu import qcu
import cupy as cp
import numpy as np
import re
gauge_filename = "wilson-bistabcg-gauge_-32-32-32-32-1048576-1-1-1-1-0-0-1-0-f.bin"
pattern = r"-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-"
match = re.search(pattern, gauge_filename)
if match:
    param = [int(num) for num in match.groups()]
    print("Extracted integers:", param)
    py_params = np.array(param, dtype=int)
    py_argv = np.array(py_params, dtype=np.float32)
    params = py_params.ctypes.data
    argv = py_argv.ctypes.data
    print(params)
    print(type(params))
    print("NumPy Array:", py_params)
    _LAT_XYZT_ = 4
    _LAT_DCC_ = 36
    _LAT_SC_ = 12
    _EVEN_ODD = 2
    size = py_params[_LAT_XYZT_]*_LAT_DCC_
    gauge_filename = gauge_filename.replace("gauge", "gauge")
    py_gauge = cp.fromfile(gauge_filename, dtype=cp.complex64, count=size)
    gauge = py_gauge.data.ptr
    size = py_params[_LAT_XYZT_]*_LAT_SC_
    fermion_in_filename = gauge_filename.replace("gauge", "fermion-in")
    py_fermion_in = cp.fromfile(
        fermion_in_filename, dtype=cp.complex64, count=size)
    fermion_in = py_fermion_in.data.ptr
    fermion_out_filename = gauge_filename.replace("gauge", "fermion-out")
    py_fermion_out = cp.fromfile(
        fermion_out_filename, dtype=cp.complex64, count=size)
    fermion_out = py_fermion_out.data.ptr
    print(gauge)
    qcu.applyBistabCgQcu(fermion_out, fermion_in, gauge, params, argv)
    # qcu.applyBistabCgQcu(py_fermion_out, py_fermion_in, py_gauge, py_params, py_argv)
else:
    print("No match found!")
    exit(1)
