import mpi4py.MPI as MPI
from pyqcu import tools
import torch
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
# Copy from PyQCU/cpp/cuda/qcu/include/define.h
_SET_PTRS_SIZE_ = 10
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
_LAT_C16_ = 0
_LAT_C32_ = 1
_LAT_C64_ = 2
_LAT_C128_ = 3
_LAT_C256_ = 4
_LAT_R8_ = 5
_LAT_R16_ = 6
_LAT_R32_ = 7
_LAT_R64_ = 8
_LAT_R128_ = 9
_SET_INDEX_ = 15
_SET_PLAN_ = 16
_SET_PLAN_N_2_ = -2  # just for laplacian
_SET_PLAN_N_1_ = -1  # just for gauss gauge
_SET_PLAN0_ = 0     # for wilson dslash
_SET_PLAN1_ = 1     # just for bistabcg and cg
_SET_PLAN2_ = 2     # for clover dslash
_MG_X_ = 17
_MG_Y_ = 18
_MG_Z_ = 19
_MG_T_ = 20
_LAT_E_ = 21
_LAT_P_ = 2
_VERBOSE_ = 22
_SEED_ = 23
_TEST_IN_CPU_ = 24
_PARAMS_SIZE_ = 25
_MASS_ = 0
_ATOL_ = 1
_SIGMA_ = 2
_ARGV_SIZE_ = 3
def dtype(_data_type_=_LAT_C64_) -> torch.dtype:
    if _data_type_ == _LAT_C16_:
        print("Doesn't support complex16")
        return None
    elif _data_type_ == _LAT_C32_:
        return torch.complex64
    elif _data_type_ == _LAT_C64_:
        return torch.complex64
    elif _data_type_ == _LAT_C128_:
        return torch.complex128
    elif _data_type_ == _LAT_C256_:
        print("Doesn't support complex256")
        return None
    elif _data_type_ == _LAT_R8_:
        print("Doesn't support real8")
        return None
    elif _data_type_ == _LAT_R16_:
        return torch.float16
    elif _data_type_ == _LAT_R32_:
        return torch.float32
    elif _data_type_ == _LAT_R64_:
        return torch.float64
    elif _data_type_ == _LAT_R128_:
        print("Doesn't support real128")
        return None
params = torch.Tensor(
    [0]*_PARAMS_SIZE_).to(dtype=torch.int32, device=torch.device('cpu'))
params[_LAT_X_] = 32
params[_LAT_Y_] = 32
params[_LAT_Z_] = 32
params[_LAT_T_] = 32
params[_LAT_XYZT_] = params[_LAT_X_] * \
    params[_LAT_Y_]*params[_LAT_Z_]*params[_LAT_T_]
params[_GRID_X_], params[_GRID_Y_], params[_GRID_Z_], params[
    _GRID_T_] = tools.give_grid_size()
params[_PARITY_] = 0
params[_NODE_RANK_] = rank
params[_NODE_SIZE_] = size
params[_DAGGER_] = 0
params[_MAX_ITER_] = 1000
params[_DATA_TYPE_] = _LAT_C64_
params[_SET_INDEX_] = 0
params[_SET_PLAN_] = 0
params[_MG_X_] = 4
params[_MG_Y_] = 4
params[_MG_Z_] = 4
params[_MG_T_] = 8
params[_LAT_E_] = 24
params[_VERBOSE_] = 1
params[_SEED_] = 42
params[_TEST_IN_CPU_] = 0
argv = torch.Tensor([0.0]*_ARGV_SIZE_).to(dtype=dtype(_LAT_C64_).to_real(),
                                          device=torch.device('cpu'))
argv[_MASS_] = -3.5  # make kappa=1.0
argv[_ATOL_] = 1e-9
argv[_SIGMA_] = 0.1
set_ptrs = torch.Tensor([0]*_SET_PTRS_SIZE_).to(dtype=torch.int64,
                                                device=torch.device('cpu'))  # maybe more than 10?
