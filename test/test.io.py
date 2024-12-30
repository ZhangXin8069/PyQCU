from pyqcu import qcu
import cupy as cp
import numpy as np
import re
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print('My rank is ', rank)
gauge_filename = "quda_wilson-bistabcg-gauge_-32-32-32-32-1048576-1-1-1-1-0-0-1-0-f.bin"
pattern = r"-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-"
match = re.search(pattern, gauge_filename)
# if match:
#     param = [int(num) for num in match.groups()]
#     print("Extracted integers:", param)
#     param.append(1000)
#     param.append(5)
#     param.append(1)
#     params = np.array(param, dtype=np.int32)
#     print("NumPy Array:", params)
#     print("Numpy data pointer:", params.data)
#     argv = np.array([0, 1e-9], dtype=np.float32)
#     print("Argv:", argv)
#     print("Argv data pointer:", argv.data)
#     set_ptrs = np.array(params, dtype=np.int64)
#     print("Set ptrs:", set_ptrs)
#     print("Set ptrs data pointer:", set_ptrs.data)
#     qcu.applyInitQcu(set_ptrs, params, argv)
#     _LAT_XYZT_ = 4
#     _LAT_DCC_ = 36
#     _LAT_SC_ = 12
#     _EVEN_ODD = 2
#     size = params[_LAT_XYZT_]*_LAT_DCC_
#     gauge_filename = gauge_filename.replace("gauge", "gauge")
#     gauge = cp.fromfile(gauge_filename, dtype=cp.complex64, count=size)
#     print("Gauge:", gauge)
#     print("Gauge data:", gauge.data)
#     size = params[_LAT_XYZT_]*_LAT_SC_
#     fermion_in_filename = gauge_filename.replace("gauge", "fermion-in")
#     fermion_in = cp.fromfile(
#         fermion_in_filename, dtype=cp.complex64, count=size)
#     print("Fermion in:", fermion_in)
#     print("Fermion in data:", fermion_in.data)
#     fermion_out_filename = gauge_filename.replace("gauge", "fermion-out")
#     quda_fermion_out = cp.fromfile(
#         fermion_out_filename, dtype=cp.complex64, count=size)
#     fermion_out = cp.zeros(size, dtype=cp.complex64)
#     print("Fermion out:", fermion_out)
#     print("Fermion out data:", fermion_out.data)
#     qcu.applyWilsonBistabCgQcu(fermion_out, fermion_in, gauge, set_ptrs, params)
#     fermion_out_filename = fermion_out_filename.replace("quda", "pyqcu")
#     fermion_out_filename = fermion_out_filename.replace("bistabcg", "cg")
#     fermion_out.tofile(fermion_out_filename)
#     print("Fermion out diff:", cp.linalg.norm(fermion_out -
#           quda_fermion_out)/cp.linalg.norm(quda_fermion_out))
#     qcu.applyEndQcu(set_ptrs, params)
#     print("params:", params)
#     print("argv:", argv)
#     print("set_ptrs:", set_ptrs)
# else:
#     print("No match found!")
param = [int(num) for num in match.groups()]
params = np.array(param, dtype=np.int32)
_LAT_XYZT_ = 4
_LAT_DCC_ = 36
_LAT_SC_ = 12
_EVEN_ODD = 2
size = params[_LAT_XYZT_]*_LAT_DCC_
gauge_filename = gauge_filename.replace("gauge", "gauge")
gauge = cp.fromfile(gauge_filename, dtype=cp.complex64, count=size)
print("Gauge:", gauge)
print("Gauge data:", gauge.data)
size = params[_LAT_XYZT_]*_LAT_SC_
from pyqcu import io
print(rank)
print(params)
params = np.array([0]*, dtype=np.int32)
