import cupy as cp
from pyqcu.cuda import define
from pyqcu.cuda import io
from pyqcu.cuda import qcu
from pyqcu.cuda.tmp_set import params, argv, set_ptrs
#############################
gauge = io.get_or_give(params, "_gauge-test8.h5")
clover = io.get_or_give(params, "_clover-test8.h5")
fermion_in = io.get_or_give(params, "_fermion_in-test8.h5")
fermion_out = io.get_or_give(params, "_fermion_out-test8.h5")
#############################
gauge = cp.zeros_like(gauge)
clover = cp.zeros_like(clover)
fermion_in = cp.ones_like(fermion_in)
fermion_out = cp.zeros_like(fermion_in)
#############################
qcu.applyInitQcu(set_ptrs, params, argv)
qcu.applyGaussGaugeQcu(gauge, set_ptrs, params)
qcu.applyEndQcu(set_ptrs, params)
#############################
io.grid_xxxtzyx2hdf5_xxxtzyx(gauge, params, "_gauge-test8.h5")
io.grid_xxxtzyx2hdf5_xxxtzyx(clover, params, "_clover-test8.h5")
io.grid_xxxtzyx2hdf5_xxxtzyx(fermion_in, params, "_fermion_in-test8.h5")
io.grid_xxxtzyx2hdf5_xxxtzyx(fermion_out, params, "_fermion_out-test8.h5")
#############################
