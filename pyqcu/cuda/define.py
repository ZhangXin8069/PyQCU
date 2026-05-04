import mpi4py.MPI as MPI
from pyqcu import tools
import torch
from typing import List, Optional
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
_SET_PTRS_SIZE_ = 10
_LAT_P_ = 2
# Copy from PyQCU/cpp/cuda/qcu/include/define.h
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
_SET_PLAN1_ = 1     # just for bistabcg and cg and the whole dslash for them
_SET_PLAN2_ = 2     # for clover dslash
_MG_NUM_LEVEL_ = 17
_MG_LEVEL_INDEX_ = 18
_MG_LEVEL1_E_ = 19
_MG_LEVEL1_X_ = 20
_MG_LEVEL1_Y_ = 21
_MG_LEVEL1_Z_ = 22
_MG_LEVEL1_T_ = 23
_MG_LEVEL1_MAX_ITER_ = 24
_MG_LEVEL1_DATA_TYPE_ = 25
_MG_LEVEL1_NUM_RESTART_ = 26
_MG_LEVEL2_E_ = 27
_MG_LEVEL2_X_ = 28
_MG_LEVEL2_Y_ = 29
_MG_LEVEL2_Z_ = 30
_MG_LEVEL2_T_ = 31
_MG_LEVEL2_MAX_ITER_ = 32
_MG_LEVEL2_DATA_TYPE_ = 33
_MG_LEVEL2_NUM_RESTART_ = 34
_MG_LEVEL3_E_ = 35
_MG_LEVEL3_X_ = 36
_MG_LEVEL3_Y_ = 37
_MG_LEVEL3_Z_ = 38
_MG_LEVEL3_T_ = 39
_MG_LEVEL3_MAX_ITER_ = 40
_MG_LEVEL3_DATA_TYPE_ = 41
_MG_LEVEL3_NUM_RESTART_ = 42
_MG_LEVEL4_E_ = 43
_MG_LEVEL4_X_ = 44
_MG_LEVEL4_Y_ = 45
_MG_LEVEL4_Z_ = 46
_MG_LEVEL4_T_ = 47
_MG_LEVEL4_MAX_ITER_ = 48
_MG_LEVEL4_DATA_TYPE_ = 49
_MG_LEVEL4_NUM_RESTART_ = 50
_MG_PARAMS_SIZE_ = 8
_VERBOSE_ = 51
_SEED_ = 52
_TEST_IN_CPU_ = 53
_PARAMS_SIZE_ = 54
_MASS_ = 0
_ATOL_ = 1
_SIGMA_ = 2
_MG_LEVEL1_ATOL_ = 3
_MG_LEVEL2_ATOL_ = 4
_MG_LEVEL3_ATOL_ = 5
_MG_LEVEL4_ATOL_ = 6
_ARGV_SIZE_ = 7
_LAT_C64_IN_TENSOR_ = torch.Tensor(_LAT_C64_, device=torch.device('cpu'))


def dtype(_data_type_: Optional[torch.Tensor] = _LAT_C64_IN_TENSOR_) -> torch.dtype:
    if _data_type_ == _LAT_C16_:
        print("Doesn't support complex16")
    elif _data_type_ == _LAT_C32_:
        return torch.complex64
    elif _data_type_ == _LAT_C64_:
        return torch.complex64
    elif _data_type_ == _LAT_C128_:
        return torch.complex128
    elif _data_type_ == _LAT_C256_:
        print("Doesn't support complex256")
    elif _data_type_ == _LAT_R8_:
        print("Doesn't support real8")
    elif _data_type_ == _LAT_R16_:
        return torch.float16
    elif _data_type_ == _LAT_R32_:
        return torch.float32
    elif _data_type_ == _LAT_R64_:
        return torch.float64
    elif _data_type_ == _LAT_R128_:
        print("Doesn't support real128")
    raise


def lat_shape(params: torch.Tensor) -> List[int]:
    return [int(params[_LAT_X_]), int(params[_LAT_Y_]), int(params[_LAT_Z_]), int(params[_LAT_T_]//_LAT_P_)]


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
params[_VERBOSE_] = 1
params[_SEED_] = 42
params[_TEST_IN_CPU_] = 0
argv = torch.Tensor([0.0]*_ARGV_SIZE_).to(dtype=dtype(params[_DATA_TYPE_]).to_real(),
                                          device=torch.device('cpu'))
argv[_MASS_] = -3.5  # make kappa=1.0
argv[_ATOL_] = 1e-9
argv[_SIGMA_] = 0.1
set_ptrs = torch.Tensor([0]*_SET_PTRS_SIZE_).to(dtype=torch.int64,
                                                device=torch.device('cpu'))  # maybe more than 10?
