import h5py
import numpy as np
import cupy as cp
import mpi4py.MPI as MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
# Copy from ../extern/qcu/include/defin.h
_BLOCK_SIZE_ = 128
_MAIN_RANK_ = 0
_a_ = 0
_b_ = 1
_c_ = 2
_d_ = 3
_tmp0_ = 0
_tmp1_ = 1
_rho_prev_ = 2
_rho_ = 3
_alpha_ = 4
_beta_ = 5
_omega_ = 6
_send_tmp_ = 7
_norm2_tmp_ = 8
_diff_tmp_ = 9
_lat_4dim_ = 10
_vals_size_ = 11
_NO_USE_ = 0
_USE_ = 1
_X_ = 0
_Y_ = 1
_Z_ = 2
_T_ = 3
_LAT_X_ = 0
_LAT_Y_ = 1
_LAT_Z_ = 2
_LAT_T_ = 3
_LAT_XYZT_ = 4
_GRID_X_ = 5
_GRID_Y_ = 6
_GRID_Z_ = 7
_GRID_T_ = 8
_PARITY_ = 9
_NODE_RANK_ = 10
_NODE_SIZE_ = 11
_DAGGER_ = 12
_MAX_ITER_ = 13
_DATA_TYPE_ = 14
_LAT_C8_ = 0
_LAT_C16_ = 1
_LAT_C32_ = 2
_LAT_C64_ = 3
_LAT_C128_ = 4
_LAT_C256_ = 5
_LAT_R8_ = 6
_LAT_R16_ = 7
_LAT_R32_ = 8
_LAT_R64_ = 9
_LAT_R128_ = 10
_SET_INDEX_ = 15
_SET_PLAN_ = 16
_SET_PLAN0_ = 0  # just for wilson dslash
_SET_PLAN1_ = 1  # just for wilson bistabcg and cg
_SET_PLAN2_ = 2  # just for clover dslash
_SET_PLAN3_ = 3
_MG_X_ = 17
_MG_Y_ = 18
_MG_Z_ = 19
_MG_T_ = 20
_LAT_E_ = 21
_PARAMS_SIZE_ = 22
_MASS_ = 0
_TOL_ = 1
_ARGV_SIZE_ = 2
_DIM_ = 4
_1DIM_ = 4
_2DIM_ = 6
_3DIM_ = 4
_B_X_ = 0
_F_X_ = 1
_B_Y_ = 2
_F_Y_ = 3
_B_Z_ = 4
_F_Z_ = 5
_B_T_ = 6
_F_T_ = 7
_BX_BY_ = 8
_FX_BY_ = 9
_BX_FY_ = 10
_FX_FY_ = 11
_BX_BZ_ = 12
_FX_BZ_ = 13
_BX_FZ_ = 14
_FX_FZ_ = 15
_BX_BT_ = 16
_FX_BT_ = 17
_BX_FT_ = 18
_FX_FT_ = 19
_BY_BZ_ = 20
_FY_BZ_ = 21
_BY_FZ_ = 22
_FY_FZ_ = 23
_BY_BT_ = 24
_FY_BT_ = 25
_BY_FT_ = 26
_FY_FT_ = 27
_BZ_BT_ = 28
_FZ_BT_ = 29
_BZ_FT_ = 30
_FZ_FT_ = 31
_B_X_B_Y_ = 0
_F_X_B_Y_ = 1
_B_X_F_Y_ = 2
_F_X_F_Y_ = 3
_B_X_B_Z_ = 4
_F_X_B_Z_ = 5
_B_X_F_Z_ = 6
_F_X_F_Z_ = 7
_B_X_B_T_ = 8
_F_X_B_T_ = 9
_B_X_F_T_ = 10
_F_X_F_T_ = 11
_B_Y_B_Z_ = 12
_F_Y_B_Z_ = 13
_B_Y_F_Z_ = 14
_F_Y_F_Z_ = 15
_B_Y_B_T_ = 16
_F_Y_B_T_ = 17
_B_Y_F_T_ = 18
_F_Y_F_T_ = 19
_B_Z_B_T_ = 20
_F_Z_B_T_ = 21
_B_Z_F_T_ = 22
_F_Z_F_T_ = 23
_WARDS_ = 8
_WARDS_2DIM_ = 24
_XY_ = 0
_XZ_ = 1
_XT_ = 2
_YZ_ = 3
_YT_ = 4
_ZT_ = 5
_YZT_ = 0
_XZT_ = 1
_XYT_ = 2
_XYZ_ = 3
_EVEN_ = 0
_ODD_ = 1
_EVEN_ODD_ = 2
_LAT_P_ = 2
_LAT_C_ = 3
_LAT_S_ = 4
_LAT_CC_ = 9
_LAT_1C_ = 3
_LAT_2C_ = 6
_LAT_3C_ = 9
_LAT_HALF_SC_ = 6
_LAT_SC_ = 12
_LAT_SCSC_ = 144
_LAT_D_ = 4
_LAT_DCC_ = 36
_LAT_PDCC_ = 72
_B_ = 0
_F_ = 1
_BF_ = 2
_REAL_IMAG_ = 2
_OUTPUT_SIZE_ = 10
_BACKWARD_ = -1
_NOWARD_ = 0
_FORWARD_ = 1
_SR_ = 2
_LAT_EXAMPLE_ = 32
_GRID_EXAMPLE_ = 1
_MEM_POOL_ = 0
_CHECK_ERROR_ = 1

def dtype(_data_type_=_LAT_C64_):
    if _data_type_ == _LAT_C8_:
        print("Doesn't support complex8")
        return None
    elif _data_type_ == _LAT_C16_:
        print("Doesn't support complex16")
        return None
    elif _data_type_ == _LAT_C32_:
        print("Doesn't support complex32")
        return None
    elif _data_type_ == _LAT_C64_:
        return cp.complex64
    elif _data_type_ == _LAT_C128_:
        return cp.complex128
    elif _data_type_ == _LAT_C256_:
        print("Doesn't support complex256")
        return None
    elif _data_type_ == _LAT_R8_:
        print("Doesn't support real8")
        return None
    elif _data_type_ == _LAT_R16_:
        return cp.float16
    elif _data_type_ == _LAT_R32_:
        return cp.float32
    elif _data_type_ == _LAT_R64_:
        return cp.float64
    elif _data_type_ == _LAT_R128_:
        print("Doesn't support real128")
        return None
