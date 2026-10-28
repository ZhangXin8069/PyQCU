"""Canonical flat-ABI constants shared by the independent KUPL backend."""

from __future__ import annotations

import math
from typing import Any, List, Optional

from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

_SET_PTRS_SIZE_ = 100
_PARAMS_SIZE_ = 58
_ARGV_SIZE_ = 8
_LAT_P_ = 2

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
_SET_INDEX_ = 15
_SET_PLAN_ = 16

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
LAT_C64 = _LAT_C64_
LAT_C128 = _LAT_C128_

_SET_PLAN_N_2_ = -2
_SET_PLAN_N_1_ = -1
_SET_PLAN0_ = 0
_SET_PLAN1_ = 1
_SET_PLAN2_ = 2

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

_SET_PTRS_COARSE_BASE_ = 30
_SET_PTRS_STRICT_COARSE_BASE_ = 60
_SET_PTRS_STRICT_STRIDE_ = 4
_SET_PTRS_STRICT_NULL_ = 0
_SET_PTRS_STRICT_RAW_LINKS_ = 1
_SET_PTRS_STRICT_PRECONDITIONED_LINKS_ = 2
_SET_PTRS_STRICT_ONSITE_PAIR_ = 3
_SET_PTRS_STRICT_HIERARCHY_ = 80
_SET_PTRS_LEGACY_LAYOUT_ = 99
_LEGACY_LAYOUT_FULL_SITE_ = 0
_LEGACY_LAYOUT_ODD_SCHUR_ = 1

_MG_MODE_GCR_ = 1
_MG_MODE_MR_SMOOTHER_ = 2
_MG_MODE_CHEBYSHEV_ = 4
_MG_MODE_CA_GCR_ = 8
_MG_MODE_W_CYCLE_ = 16
_MG_MODE_F_CYCLE_ = 32
_MG_MODE_K_CYCLE_ = 64
_MG_MODE_BICGSTABL_ = 128
_MG_MODE_CYCLE_MASK_ = (
    _MG_MODE_W_CYCLE_ | _MG_MODE_F_CYCLE_ | _MG_MODE_K_CYCLE_)

_MASS_ = 0
_ATOL_ = 1
_SIGMA_ = 2
_MG_LEVEL1_ATOL_ = 3
_MG_LEVEL2_ATOL_ = 4
_MG_LEVEL3_ATOL_ = 5
_MG_LEVEL4_ATOL_ = 6
_CSW_ = 7


def _data_type_value(value: Any) -> int:
    return int(value.item()) if hasattr(value, "item") else int(value)


def dtype(_data_type_: Any = _LAT_C64_):
    """Map a QCU dtype constant to the corresponding torch dtype."""
    import torch

    value = _data_type_value(_data_type_)
    mapping = {
        _LAT_C32_: torch.complex32,
        _LAT_C64_: torch.complex64,
        _LAT_C128_: torch.complex128,
        _LAT_R16_: torch.float16,
        _LAT_R32_: torch.float32,
        _LAT_R64_: torch.float64,
    }
    try:
        return mapping[value]
    except KeyError as exc:
        raise ValueError(f"Unsupported QCU data type constant: {value}") from exc


def epytd(torch_dtype: Any):
    """Map a torch dtype to a QCU data-type tensor."""
    import torch

    mapping = {
        torch.complex64: _LAT_C64_,
        torch.complex128: _LAT_C128_,
        torch.float16: _LAT_R16_,
        torch.float32: _LAT_R32_,
        torch.float64: _LAT_R64_,
    }
    try:
        value = mapping[torch_dtype]
    except KeyError as exc:
        raise ValueError(
            f"No QCU data type constant maps to torch dtype: {torch_dtype}"
        ) from exc
    return torch.tensor([value], dtype=torch.int32, device="cpu")


def lat_shape(params: Any) -> List[int]:
    return [
        int(params[_LAT_X_]), int(params[_LAT_Y_]),
        int(params[_LAT_Z_]), int(params[_LAT_T_] // _LAT_P_),
    ]


def make_argv(mass: float, *, atol: float = 1.0e-6, sigma: float = 0.1,
              csw: float = 1.0, mg_atol: Optional[List[float]] = None,
              data_type: int = _LAT_C64_, array_module: str = "torch"):
    """Build the canonical eight-slot QCU ``argv`` tensor."""
    if not math.isfinite(float(mass)):
        raise ValueError("mass must be finite")
    if not math.isfinite(float(atol)) or float(atol) <= 0.0:
        raise ValueError("atol must be finite and positive")
    if not math.isfinite(float(sigma)):
        raise ValueError("sigma must be finite")
    if not math.isfinite(float(csw)):
        raise ValueError("csw must be finite")
    values = [float(mass), float(atol), float(sigma), 0.0, 0.0, 0.0, 0.0,
              float(csw)]
    if mg_atol is not None:
        for index, value in enumerate(mg_atol[:4]):
            values[_MG_LEVEL1_ATOL_ + index] = float(value)
    if array_module == "numpy":
        import numpy as np

        numpy_dtype = np.float32 if int(data_type) == _LAT_C64_ else np.float64
        return np.asarray(values, dtype=numpy_dtype)
    if array_module != "torch":
        raise ValueError("array_module must be 'torch' or 'numpy'")
    import torch

    real_dtype = dtype(data_type).to_real()
    return torch.tensor(values, dtype=real_dtype, device="cpu")


__all__ = [name for name in globals() if name.startswith("_") or name in {
    "LAT_C64", "LAT_C128", "comm", "dtype", "epytd", "lat_shape",
    "make_argv", "rank", "size",
}]
