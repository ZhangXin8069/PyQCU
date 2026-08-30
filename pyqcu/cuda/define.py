import mpi4py.MPI as MPI
from pyqcu import tools
import torch
from typing import List, Optional
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
_SET_PTRS_SIZE_ = 100
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
_DATA_TYPE_SIZE_ = 10
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
_MG_USE_GCR_ = 54
_MG_USE_DEFLATE_ = 55
_MG_MU_PRE_ = 56
_MG_USE_INIT_GUESS_ = 57
_SET_PTRS_STRICT_COARSE_BASE_ = 60
_SET_PTRS_STRICT_STRIDE_ = 4
_SET_PTRS_STRICT_NULL_ = 0
_SET_PTRS_STRICT_RAW_LINKS_ = 1
_SET_PTRS_STRICT_PRECONDITIONED_LINKS_ = 2
_SET_PTRS_STRICT_ONSITE_PAIR_ = 3
_SET_PTRS_STRICT_HIERARCHY_ = 80
# _MG_USE_GCR_ is a backwards-compatible mode bit mask.  Keep these values
# synchronized with cpp/cuda/qcu/include/define.h; the params ABI stays 58.
_MG_MODE_GCR_ = 1
_MG_MODE_MR_SMOOTHER_ = 2
# Alternative coarse smoothers / outer solver.
_MG_MODE_CHEBYSHEV_ = 4
_MG_MODE_CA_GCR_ = 8
# Recursive cycle selection; zero means the historical V-cycle.
_MG_MODE_W_CYCLE_ = 16
_MG_MODE_F_CYCLE_ = 32
_MG_MODE_K_CYCLE_ = 64
_MG_MODE_BICGSTABL_ = 128
_MG_MODE_CYCLE_MASK_ = _MG_MODE_W_CYCLE_ | _MG_MODE_F_CYCLE_ | _MG_MODE_K_CYCLE_
_PARAMS_SIZE_ = 58
_MASS_ = 0
_ATOL_ = 1
_SIGMA_ = 2
_MG_LEVEL1_ATOL_ = 3
_MG_LEVEL2_ATOL_ = 4
_MG_LEVEL3_ATOL_ = 5
_MG_LEVEL4_ATOL_ = 6
_ARGV_SIZE_ = 7
_LAT_C64_IN_TENSOR_ = torch.Tensor([_LAT_C64_], device=torch.device('cpu'))


def dtype(_data_type_: Optional[torch.Tensor] = _LAT_C64_IN_TENSOR_) -> torch.dtype:
    if _data_type_ == torch.Tensor([_LAT_C16_], device=torch.device('cpu')):
        raise ValueError(f"Unsupported QCU data type: _LAT_C16_ (complex16) at constant {_data_type_.item()}")
    elif _data_type_ == torch.Tensor([_LAT_C32_], device=torch.device('cpu')):
        return torch.complex32
    elif _data_type_ == torch.Tensor([_LAT_C64_], device=torch.device('cpu')):
        return torch.complex64
    elif _data_type_ == torch.Tensor([_LAT_C128_], device=torch.device('cpu')):
        return torch.complex128
    elif _data_type_ == torch.Tensor([_LAT_C256_], device=torch.device('cpu')):
        raise ValueError(f"Unsupported QCU data type: _LAT_C256_ (complex256)")
    elif _data_type_ == torch.Tensor([_LAT_R8_], device=torch.device('cpu')):
        raise ValueError(f"Unsupported QCU data type: _LAT_R8_ (real8)")
    elif _data_type_ == torch.Tensor([_LAT_R16_], device=torch.device('cpu')):
        return torch.float16
    elif _data_type_ == torch.Tensor([_LAT_R32_], device=torch.device('cpu')):
        return torch.float32
    elif _data_type_ == torch.Tensor([_LAT_R64_], device=torch.device('cpu')):
        return torch.float64
    elif _data_type_ == torch.Tensor([_LAT_R128_], device=torch.device('cpu')):
        raise ValueError(f"Unsupported QCU data type: _LAT_R128_ (real128)")
    raise ValueError(f"Unsupported QCU data type constant: {_data_type_.item()}")


def epytd(torch_dtype: Optional[torch.dtype]) -> torch.Tensor:
    for i in range(_DATA_TYPE_SIZE_):
        _data_type_ = torch.Tensor([i], device=torch.device('cpu'))
        # BUGFIX 2026-08-02: dtype() raises for unsupported constants (e.g.
        # _LAT_C16_ at i=0). Skip those during the reverse-mapping search
        # instead of letting the raise abort the whole loop. Otherwise epytd()
        # can never map complex64/128, which blocks the multigrid with_cuda_qcu
        # path (pyqcu/solver/_multigrid.py::init calls epytd on dtype_list[0]).
        try:
            if dtype(_data_type_=_data_type_) == torch_dtype:
                return _data_type_
        except ValueError:
            continue
    # BUGFIX 2026-07-28: explicit error with torch dtype that failed mapping.
    raise ValueError(f"No QCU data type constant maps to torch dtype: {torch_dtype}")


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
