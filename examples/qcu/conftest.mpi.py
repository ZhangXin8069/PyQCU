import torch
from pyqcu import tools, dslash, lattice
from pyqcu.cuda import qcu, define
from pyqcu.cuda.define import params, argv, set_ptrs
params[define._LAT_X_] = 8
params[define._LAT_Y_] = 8
params[define._LAT_Z_] = 8
params[define._LAT_T_] = 8
params[define._LAT_XYZT_] = params[define._LAT_X_] * \
    params[define._LAT_Y_]*params[define._LAT_Z_]*params[define._LAT_T_]
params[define._GRID_X_], params[define._GRID_Y_], params[define._GRID_Z_], params[
    define._GRID_T_] = tools.give_grid_size()
params[define._PARITY_] = 0
params[define._NODE_RANK_] = define.rank
params[define._NODE_SIZE_] = define.size
params[define._DAGGER_] = 0
params[define._MAX_ITER_] = 1000
params[define._DATA_TYPE_] = define._LAT_C128_
params[define._SET_INDEX_] = 0
params[define._SET_PLAN_] = 1
params[define._MG_X_] = 1
params[define._MG_Y_] = 1
params[define._MG_Z_] = 1
params[define._MG_T_] = 1
params[define._LAT_E_] = 24
params[define._VERBOSE_] = 1
params[define._SEED_] = 42
params[define._TEST_IN_CPU_] = 1
argv = argv.to(dtype=define.dtype(params[define._DATA_TYPE_]).to_real())
argv[define._MASS_] = 0.05
argv[define._TOL_] = 1e-9
argv[define._SIGMA_] = 0.1
#############################
print(params)
print(argv)
print(set_ptrs)

params[define._VERBOSE_] = 1
params[define._SET_INDEX_] = 0
params[define._SET_PLAN_] = 0
params[define._PARITY_] = 0
qcu.applyInitQcu(set_ptrs, params, argv)
# PASS
qcu.applyEndQcu(set_ptrs, params)
print("set_ptrs:", set_ptrs)