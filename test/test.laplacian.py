from opt_einsum import contract
import re
import cupy as cp
import numpy as np
from time import perf_counter
from pyqcu.cuda import define, linalg
from pyqcu.cuda import io
from pyqcu.cuda import qcu
print('My rank is ', define.rank)
#############################
params = np.array([0]*define._PARAMS_SIZE_, dtype=np.int32)
params[define._LAT_X_] = 4
params[define._LAT_Y_] = 4
params[define._LAT_Z_] = 4
params[define._LAT_T_] = 1
params[define._LAT_XYZT_] = params[define._LAT_X_] * \
    params[define._LAT_Y_]*params[define._LAT_Z_]*params[define._LAT_T_]
params[define._GRID_X_], params[define._GRID_Y_], params[define._GRID_Z_], params[
    define._GRID_T_] = define.split_into_four_factors(define.size)
params[define._PARITY_] = 0
params[define._NODE_RANK_] = define.rank
params[define._NODE_SIZE_] = define.size
params[define._DAGGER_] = 0
params[define._MAX_ITER_] = 1000
params[define._DATA_TYPE_] = define._LAT_C128_
params[define._SET_INDEX_] = 0
params[define._SET_PLAN_] = -1
params[define._MG_X_] = 1
params[define._MG_Y_] = 1
params[define._MG_Z_] = 1
params[define._MG_T_] = 1
params[define._LAT_E_] = 24
params[define._VERBOSE_] = 1
params[define._SEED_] = 42
argv = np.array([0.0]*define._ARGV_SIZE_,
                dtype=define.dtype_half(params[define._DATA_TYPE_]))
argv[define._MASS_] = 0.05
argv[define._TOL_] = 1e-12
argv[define._SIGMA_] = 0.1
set_ptrs = np.array(10*[0], dtype=np.int64)  # maybe more than 10?
#############################
laplacian_in = cp.array([range(define._LAT_C_*params[define._LAT_XYZT_])]).reshape(
    define._LAT_C_, params[define._LAT_Z_], params[define._LAT_Y_], params[define._LAT_X_]).astype(define.dtype(params[define._DATA_TYPE_]))
laplacian_out = cp.zeros_like(laplacian_in)
gauge = cp.array([range(define._LAT_3D_*define._LAT_CC_*params[define._LAT_XYZT_])]).reshape(
    define._LAT_C_, define._LAT_C_, define._LAT_3D_, params[define._LAT_Z_], params[define._LAT_Y_], params[define._LAT_X_]).astype(define.dtype(params[define._DATA_TYPE_]))
#############################
# gauge = cp.ones_like(gauge)
gauge = linalg.initialize_random_vector(gauge)
# laplacian_in = cp.ones_like(laplacian_in)
laplacian_in = linalg.initialize_random_vector(laplacian_in)
laplacian_out = cp.zeros_like(laplacian_out)
#############################
qcu.applyInitQcu(set_ptrs, params, argv)
t0 = perf_counter()
qcu.applyLaplacianQcu(laplacian_out, laplacian_in,
                      gauge, set_ptrs, params)
t1 = perf_counter()
qcu.applyEndQcu(set_ptrs, params)
print(f'PyQCU cost time: {t1 - t0} sec')
print("norm of Laplacian out:", cp.linalg.norm(laplacian_out))
#############################


def _Laplacian(F, U):
    Lx, Ly, Lz = params[define._LAT_X_], params[define._LAT_Y_], params[define._LAT_Z_]
    U_dag = U.transpose(0, 1, 2, 3, 5, 4).conj()  # dzyxcc
    F = F.reshape(Lz, Ly, Lx, define._LAT_C_, -1)  # zyxc
    print(f"U.shape:{U.shape}")
    print(f"U.dtype:{U.dtype}")
    print(f"F.shape:{F.shape}")
    print(f"F.dtype:{F.dtype}")
    t0 = perf_counter()
    dest = (
        # - for SA with evals , + for LA with (12 - evals)
        6 * F
        - (
            contract("zyxab,zyxbc->zyxac", U[0], cp.roll(F, -1, 2))
            + contract("zyxab,zyxbc->zyxac", U[1], cp.roll(F, -1, 1))
            + contract("zyxab,zyxbc->zyxac", U[2], cp.roll(F, -1, 0))
            + cp.roll(contract("zyxab,zyxbc->zyxac", U_dag[0], F), 1, 2)
            + cp.roll(contract("zyxab,zyxbc->zyxac", U_dag[1], F), 1, 1)
            + cp.roll(contract("zyxab,zyxbc->zyxac", U_dag[2], F), 1, 0)
        )
    ).reshape(Lz * Ly * Lx * define._LAT_C_, -1)
    t1 = perf_counter()
    print(f'cupy cost time: {t1 - t0} sec')
    return dest


#############################
_gauge = io.ccdzyx2dzyxcc(io.gauge2ccdzyx(
    gauge, params))
_laplacian_in = io.czyx2zyxc(io.laplacian2czyx(
    laplacian_in, params))
# print(
#     f"contract('zyxab,zyxb->zyxa',_gauge[0], cp.roll(_laplacian_in, -1, 2)):{contract('zyxab,zyxb->zyxa',_gauge[0], cp.roll(_laplacian_in, -1, 2))}")
print(f"_gauge.shape:{_gauge.shape}")
print(f"_gauge.dtype:{_gauge.dtype}")
print(f"_laplacian.shape:{_laplacian_in.shape}")
print(f"_laplacian.dtype:{_laplacian_in.dtype}")
t0 = perf_counter()
_laplacian_out = _Laplacian(
    _laplacian_in, _gauge)
t1 = perf_counter()
print(f'PyQUDA cost time: {t1 - t0} sec')
print("norm of PyQuda Laplacian out:",
      cp.linalg.norm(_laplacian_out))
#############################
_laplacian_out = io.zyxc2czyx(io.laplacian2zyxc(_laplacian_out, params))
# print(f"gauge:{gauge}")
# print(f"laplacian_in:{laplacian_in}")
# print(f"laplacian_out:{laplacian_out}")
# print(f"_gauge:{_gauge}")
# print(f"_laplacian_in:{_laplacian_in}")
# print(f"_laplacian_out:{_laplacian_out}")
print("Difference between QUDA and PyQuda Laplacian out:",
      cp.linalg.norm(_laplacian_out - laplacian_out)/cp.linalg.norm(_laplacian_out))
#############################
