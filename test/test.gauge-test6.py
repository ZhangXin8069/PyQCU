import numpy as np
import cupy as cp
from pyqcu.cuda import define
from pyqcu.cuda import io
from pyqcu.cuda import qcu
from pyqcu.cuda.set import params, argv, set_ptrs
from pyqcu.cuda.gauge import Gauge
print('My rank is ', define.rank)
params[define._LAT_X_] = 16
params[define._LAT_Y_] = 16
params[define._LAT_Z_] = 16
params[define._LAT_T_] = 16
params[define._LAT_XYZT_] = params[define._LAT_X_] * \
    params[define._LAT_Y_]*params[define._LAT_Z_]*params[define._LAT_T_]
params[define._GRID_X_] = 1
params[define._GRID_Y_] = 1
params[define._GRID_Z_] = 1
params[define._GRID_T_] = 1
params[define._PARITY_] = 0
params[define._NODE_RANK_] = define.rank
params[define._NODE_SIZE_] = define.size
params[define._DATA_TYPE_] = define._LAT_C64_
dtype = define.dtype(_data_type_=params[define._DATA_TYPE_])
params[define._VERBOSE_] = 0
print("Parameters:", params)
#############################
gauge = Gauge(latt_size=params[:4], dtype=dtype, verbose=True)
U = gauge.generate_gauge_field(sigma=0.1, seed=42)
gauge.check_su3(U=U)