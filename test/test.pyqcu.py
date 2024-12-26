from pyqcu import qcu
import cupy as cp
import numpy as np
import re
filename = "wilson-bistabcg-gauge_-32-32-32-32-1048576-1-1-1-1-0-0-1-0-f.bin"
pattern = r"-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-"
match = re.search(pattern, filename)
if match:
    param = [int(num) for num in match.groups()]
    print("Extracted integers:", param)
    params = np.array(param, dtype=int)
    argv = cp.array(params, dtype=cp.double)
    print("NumPy Array:", params)
    _LAT_XYZT_ = 4
    _LAT_DCC_ = 36
    _LAT_SC_ = 12
    _EVEN_ODD = 2
    size = params[_LAT_XYZT_]*_LAT_DCC_
    gauge = cp.fromfile(filename, dtype=cp.complex64, count=size)
    filename.replace("gauge", "fermion-in")
    size = params[_LAT_XYZT_]*_LAT_SC_
    fermion_in = cp.fromfile(filename, dtype=cp.complex64, count=size)
    filename.replace("fermion-in", "fermion-out")
    fermion_out = cp.fromfile(filename, dtype=cp.complex64, count=size)
    qcu.applyBistabCgQcu(fermion_out, fermion_in, gauge, params, argv)
else:
    print("No match found!")
    exit(1)

