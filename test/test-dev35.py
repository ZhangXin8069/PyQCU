# %%
# %%
import cupy as cp
import numpy as np
from pyqcu import define
from pyqcu import io
from pyqcu import qcu
from pyqcu.set import params, argv, set_ptrs
import h5py
print('My rank is ', define.rank)
# qcu.applyInitQcu(set_ptrs, params, argv)
gauge_filename = f"quda_wilson-dslash-gauge_-{params[define._LAT_X_]}-{params[define._LAT_Y_]}-{params  [define._LAT_Z_]}-{params[define._LAT_T_]}-{params[define._LAT_XYZT_]}-{params[define._GRID_X_]}-{params[define._GRID_Y_]}-{params[define._GRID_Z_]}-{params[define._GRID_T_]}-{params[define._PARITY_]}-{params[define._NODE_RANK_]}-{params[define._NODE_SIZE_]}-{params[define._DAGGER_]}-f.bin"
print("Gauge filename:", gauge_filename)
gauge = cp.fromfile(gauge_filename, dtype=cp.complex64,
                    count=params[define._LAT_XYZT_]*define._LAT_DCC_)
gauge = io.gauge2ccdptzyx(gauge, params)
print("Gauge data:", gauge.data)
print("Gauge shape:", gauge.shape)
params[define._NODE_RANK_] = define.rank
params[define._NODE_SIZE_] = define.size
params[define._GRID_T_] = 2
params[define._GRID_Z_] = 1
params[define._GRID_Y_] = 1
params[define._GRID_X_] = 4
print("Params:", params)

print("params[define._NODE_RANK_]", params[define._NODE_RANK_])
print("params[define._NODE_SIZE_]", params[define._NODE_SIZE_])
_ = io.xxxtzyx2grid_xxxtzyx(gauge, params)

io.grid_xxxtzyx2hdf5_xxxtzyx(_, params, file_name='__xxxtzyx.h5')


# %%

# qcu.applyEndQcu(set_ptrs, params)
