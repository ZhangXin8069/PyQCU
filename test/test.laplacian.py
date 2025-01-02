import re
import cupy as cp
import numpy as np
from pyqcu import define
from pyqcu import io
from pyqcu import qcu

print('My rank is ', define.rank)
if define.rank == 0:
    params = np.array([0]*define._PARAMS_SIZE_, dtype=np.int32)
    params[define._LAT_X_] = 4
    params[define._LAT_Y_] = 4
    params[define._LAT_Z_] = 4
    params[define._LAT_T_] = 1
    params[define._LAT_XYZT_] = params[define._LAT_X_] * \
        params[define._LAT_Y_] * \
        params[define._LAT_Z_] * params[define._LAT_T_]
    params[define._GRID_X_] = 1
    params[define._GRID_Y_] = 1
    params[define._GRID_Z_] = 1
    params[define._GRID_T_] = 1
    params[define._PARITY_] = 0
    params[define._NODE_RANK_] = 0
    params[define._NODE_SIZE_] = 1
    params[define._DAGGER_] = 0
    params[define._MAX_ITER_] = 1e3
    params[define._DATA_TYPE_] = 0
    params[define._SET_INDEX_] = 2
    params[define._SET_PLAN_] = -1
    argv = np.array([0.0]*define._ARGV_SIZE_, dtype=np.float32)
    argv[define._MASS_] = 0.0
    argv[define._TOL_] = 1e-9
    print("Parameters:", params)
    print("Arguments:", argv)
    #############################
    laplacian_out = cp.zeros(
        shape=params[define._LAT_XYZT_]*define._LAT_C_, dtype=cp.complex64)
    print("Laplacian out:", laplacian_out)
    print("Laplacian out data:", laplacian_out.data)
    print("Laplacian out shape:", laplacian_out.shape)
    print("norm of Laplacian out:", cp.linalg.norm(laplacian_out))
    laplacian_in = cp.array([range(params[define._LAT_XYZT_]*define._LAT_C_)])
    laplacian_in = cp.array(laplacian_in+1j*laplacian_in, dtype=cp.complex64)
    print("Laplacian in:", laplacian_in)
    print("Laplacian in data:", laplacian_in.data)
    print("Laplacian in shape:", laplacian_in.shape)
    print("norm of Laplacian in:", cp.linalg.norm(laplacian_in))
    laplacian_gauge = cp.array(
        [range(params[define._LAT_XYZT_]*define._LAT_DCC_)])
    laplacian_gauge = cp.array(
        laplacian_gauge+1j*laplacian_gauge, dtype=cp.complex64)
    print("Laplacian gauge:", laplacian_gauge)
    print("Laplacian gauge data:", laplacian_gauge.data)
    print("Laplacian gauge shape:", laplacian_gauge.shape)
    print("norm of Laplacian gauge:", cp.linalg.norm(laplacian_gauge))
    #############################
    set_ptrs = np.array(params, dtype=np.int64)
    print("Set pointers:", set_ptrs)
    print("Set pointers data:", set_ptrs.data)
    qcu.applyInitQcu(set_ptrs, params, argv)
    qcu.applyLaplacianQcu(laplacian_out, laplacian_in,
                          laplacian_gauge, set_ptrs, params)
    print("Laplacian out:", laplacian_out)
    print("Laplacian out data:", laplacian_out.data)
    print("Laplacian out shape:", laplacian_out.shape)
    print("norm of Laplacian out:", cp.linalg.norm(laplacian_out))
    # _=io.laplacian2czyx(laplacian_out, params)
    # for x in range(params[define._LAT_X_]):
    #     for y in range(params[define._LAT_Y_]):
    #         for z in range(params[define._LAT_Z_]):
    #             for c in range(define._LAT_C_):
    #                 # print(f"({x}, {y}, {z}, {c}):", _[z, y, x, c])
    #                 print(f"({c}, {x}, {y}, {z}):", _[c, x, y, z])
    qcu.applyEndQcu(set_ptrs, params)
    #############################
    pyquda_laplacian_gauge = io.ccdzyx2dzyxcc(io.laplacian_gauge2ccdzyx(
        laplacian_gauge, params))
    pyquda_laplacian_in = io.czyx2zyxc(io.laplacian2czyx(
        laplacian_in, params))
    print("PyQuda Laplacian gauge:", pyquda_laplacian_gauge)
    print("PyQuda Laplacian gauge data:", pyquda_laplacian_gauge.data)
    print("PyQuda Laplacian gauge shape:", pyquda_laplacian_gauge.shape)
    print("norm of PyQuda Laplacian gauge:", cp.linalg.norm(pyquda_laplacian_gauge))
    print("PyQuda Laplacian in:", pyquda_laplacian_in)
    print("PyQuda Laplacian in data:", pyquda_laplacian_in.data)
    print("PyQuda Laplacian in shape:", pyquda_laplacian_in.shape)
    print("norm of PyQuda Laplacian in:", cp.linalg.norm(pyquda_laplacian_in))
    from opt_einsum import contract

    def pyquda_Laplacian(F, U):
        Lx, Ly, Lz, Lt = params[define._LAT_X_], params[define._LAT_Y_], params[define._LAT_Z_], params[define._LAT_T_]
        U_dag = U.transpose(0, 1, 2, 3, 5, 4).conj()
        F = F.reshape(Lz, Ly, Lx, define._LAT_C_, -1)
        return (
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
    pyquda_Laplacian_out = pyquda_Laplacian(
        pyquda_laplacian_in, pyquda_laplacian_gauge)
    _ = io.zyxc2czyx(
        io.laplacian2zyxc(pyquda_Laplacian_out, params))
    print("PyQuda Laplacian out[0]:", _[0])
    print("PyQuda Laplacian out[0] data:", _[0].data)
    print("PyQuda Laplacian out[0] shape:", _[0].shape)
    print("norm of PyQuda Laplacian out[0]:", cp.linalg.norm(_[0]))
    _ = io.array2xxx(_)
    print("PyQuda Laplacian out:", _)
    print("PyQuda Laplacian out data:", _.data)
    print("PyQuda Laplacian out shape:", _.shape)
    print("norm of PyQuda Laplacian out:", cp.linalg.norm(_))
