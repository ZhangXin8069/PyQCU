import numpy as np
import torch
from pyqcu import cuda


def applyInitQcu(_set_ptrs: torch.ndarray = torch.Tensor([0]*cuda._SET_PTRS_SIZE_, dtype=torch.int64),
                 _params: torch.ndarray = torch.Tensor(
                     [0]*cuda._PARAMS_SIZE_, dtype=torch.int32),
                 _argv: torch.ndarray = torch.Tensor([0.0]*cuda._SET_PTRS_SIZE_, dtype=cuda.dtype_half(cuda._LAT_C64_))) -> None: ...


def applyEndQcu(_set_ptrs: torch.ndarray = torch.Tensor([0]*cuda._SET_PTRS_SIZE_, dtype=torch.int64),
                _params: torch.ndarray = torch.Tensor([0]*cuda._PARAMS_SIZE_, dtype=torch.int32)) -> None: ...


def testWilsonDslashQcu(_fermion_ou: cp_ndarray = torch.Tensor([0.0], dtype=cuda.dtype(cuda._LAT_C64_)),
                        _fermion_i: cp_ndarray = torch.Tensor(
                            [0.0], dtype=cuda.dtype(cuda._LAT_C64_)),
                        _gaug: cp_ndarray = torch.Tensor(
                            [0.0], dtype=cuda.dtype(cuda._LAT_C64_)),
                        _set_ptrs: torch.ndarray = torch.Tensor(
                            [0]*cuda._SET_PTRS_SIZE_, dtype=torch.int64),
                        _params: torch.ndarray = torch.Tensor([0]*cuda._PARAMS_SIZE_, dtype=torch.int32)) -> None: ...


def applyWilsonDslashQcu(_fermion_ou: cp_ndarray = torch.Tensor([0.0], dtype=cuda.dtype(cuda._LAT_C64_)),
                         _fermion_i: cp_ndarray = torch.Tensor(
                             [0.0], dtype=cuda.dtype(cuda._LAT_C64_)),
                         _gaug: cp_ndarray = torch.Tensor(
                             [0.0], dtype=cuda.dtype(cuda._LAT_C64_)),
                         _set_ptrs: torch.ndarray = torch.Tensor(
                             [0]*cuda._SET_PTRS_SIZE_, dtype=torch.int64),
                         _params: torch.ndarray = torch.Tensor([0]*cuda._PARAMS_SIZE_, dtype=torch.int32)) -> None: ...


def testCloverDslashQcu(_fermion_ou: cp_ndarray = torch.Tensor([0.0], dtype=cuda.dtype(cuda._LAT_C64_)),
                        _fermion_i: cp_ndarray = torch.Tensor(
                            [0.0], dtype=cuda.dtype(cuda._LAT_C64_)),
                        _gaug: cp_ndarray = torch.Tensor(
                            [0.0], dtype=cuda.dtype(cuda._LAT_C64_)),
                        _set_ptrs: torch.ndarray = torch.Tensor(
                            [0]*cuda._SET_PTRS_SIZE_, dtype=torch.int64),
                        _params: torch.ndarray = torch.Tensor([0]*cuda._PARAMS_SIZE_, dtype=torch.int32)) -> None: ...


def applyCloverDslashQcu(_fermion_ou: cp_ndarray = torch.Tensor([0.0], dtype=cuda.dtype(cuda._LAT_C64_)),
                         _fermion_i: cp_ndarray = torch.Tensor(
                             [0.0], dtype=cuda.dtype(cuda._LAT_C64_)),
                         _gaug: cp_ndarray = torch.Tensor(
                             [0.0], dtype=cuda.dtype(cuda._LAT_C64_)),
                         _set_ptrs: torch.ndarray = torch.Tensor(
                             [0]*cuda._SET_PTRS_SIZE_, dtype=torch.int64),
                         _params: torch.ndarray = torch.Tensor([0]*cuda._PARAMS_SIZE_, dtype=torch.int32)) -> None: ...


def applyWilsonBistabCgQcu(_fermion_ou: cp_ndarray = torch.Tensor([0.0], dtype=cuda.dtype(cuda._LAT_C64_)),
                           _fermion_i: cp_ndarray = torch.Tensor(
                               [0.0], dtype=cuda.dtype(cuda._LAT_C64_)),
                           _gaug: cp_ndarray = torch.Tensor(
                               [0.0], dtype=cuda.dtype(cuda._LAT_C64_)),
                           _set_ptrs: torch.ndarray = torch.Tensor(
                               [0]*cuda._SET_PTRS_SIZE_, dtype=torch.int64),
                           _params: torch.ndarray = torch.Tensor([0]*cuda._PARAMS_SIZE_, dtype=torch.int32)) -> None: ...


def applyWilsonBistabCgDslashQcu(
    _fermion_ou: cp_ndarray = torch.Tensor(
        [0.0], dtype=cuda.dtype(cuda._LAT_C64_)),
    _fermion_i: cp_ndarray = torch.Tensor(
        [0.0], dtype=cuda.dtype(cuda._LAT_C64_)),
    _gaug: cp_ndarray = torch.Tensor([0.0], dtype=cuda.dtype(cuda._LAT_C64_)),
    _set_ptrs: torch.ndarray = torch.Tensor(
        [0]*cuda._SET_PTRS_SIZE_, dtype=torch.int64),
    _params: torch.ndarray = torch.Tensor([0]*cuda._PARAMS_SIZE_, dtype=torch.int32)) -> None: ...


def applyWilsonCgQcu(_fermion_ou: cp_ndarray = torch.Tensor([0.0], dtype=cuda.dtype(cuda._LAT_C64_)),
                     _fermion_i: cp_ndarray = torch.Tensor(
                         [0.0], dtype=cuda.dtype(cuda._LAT_C64_)),
                     _gaug: cp_ndarray = torch.Tensor(
                         [0.0], dtype=cuda.dtype(cuda._LAT_C64_)),
                     _set_ptrs: torch.ndarray = torch.Tensor(
                         [0]*cuda._SET_PTRS_SIZE_, dtype=torch.int64),
                     _params: torch.ndarray = torch.Tensor([0]*cuda._PARAMS_SIZE_, dtype=torch.int32)) -> None: ...


def applyWilsonCgDslashQcu(_fermion_ou: cp_ndarray = torch.Tensor([0.0], dtype=cuda.dtype(cuda._LAT_C64_)),
                           _fermion_i: cp_ndarray = torch.Tensor(
                               [0.0], dtype=cuda.dtype(cuda._LAT_C64_)),
                           _gaug: cp_ndarray = torch.Tensor(
                               [0.0], dtype=cuda.dtype(cuda._LAT_C64_)),
                           _set_ptrs: torch.ndarray = torch.Tensor(
                               [0]*cuda._SET_PTRS_SIZE_, dtype=torch.int64),
                           _params: torch.ndarray = torch.Tensor([0]*cuda._PARAMS_SIZE_, dtype=torch.int32)) -> None: ...


def applyLaplacianQcu(_laplacian_ou: cp_ndarray = torch.Tensor([0.0], dtype=cuda.dtype(cuda._LAT_C64_)),
                      _laplacian_i: cp_ndarray = torch.Tensor(
                          [0.0], dtype=cuda.dtype(cuda._LAT_C64_)),
                      _gaug: cp_ndarray = torch.Tensor(
                          [0.0], dtype=cuda.dtype(cuda._LAT_C64_)),
                      _set_ptrs: torch.ndarray = torch.Tensor(
                          [0]*cuda._SET_PTRS_SIZE_, dtype=torch.int64),
                      _params: torch.ndarray = torch.Tensor([0]*cuda._PARAMS_SIZE_, dtype=torch.int32)) -> None: ...


def applyCloverQcu(_clove: cp_ndarray = torch.Tensor([0.0], dtype=cuda.dtype(cuda._LAT_C64_)),
                   _gaug: cp_ndarray = torch.Tensor(
                       [0.0], dtype=cuda.dtype(cuda._LAT_C64_)),
                   _set_ptrs: torch.ndarray = torch.Tensor(
                       [0]*cuda._SET_PTRS_SIZE_, dtype=torch.int64),
                   _params: torch.ndarray = torch.Tensor([0]*cuda._PARAMS_SIZE_, dtype=torch.int32)) -> None: ...


def applyCloversQcu(_clove: cp_ndarray = torch.Tensor([0.0], dtype=cuda.dtype(cuda._LAT_C64_)),
                    _clover_in: cp_ndarray = torch.Tensor(
                        [0.0], dtype=cuda.dtype(cuda._LAT_C64_)),
                    _gaug: cp_ndarray = torch.Tensor(
                        [0.0], dtype=cuda.dtype(cuda._LAT_C64_)),
                    _set_ptrs: torch.ndarray = torch.Tensor(
                        [0]*cuda._SET_PTRS_SIZE_, dtype=torch.int64),
                    _params: torch.ndarray = torch.Tensor([0]*cuda._PARAMS_SIZE_, dtype=torch.int32)) -> None: ...


def applyDslashQcu(_fermion_ou: cp_ndarray = torch.Tensor([0.0], dtype=cuda.dtype(cuda._LAT_C64_)),
                   _fermion_i: cp_ndarray = torch.Tensor(
                       [0.0], dtype=cuda.dtype(cuda._LAT_C64_)),
                   _gaug: cp_ndarray = torch.Tensor(
                       [0.0], dtype=cuda.dtype(cuda._LAT_C64_)),
                   _clove: cp_ndarray = torch.Tensor(
                       [0.0], dtype=cuda.dtype(cuda._LAT_C64_)),
                   _set_ptrs: torch.ndarray = torch.Tensor(
                       [0]*cuda._SET_PTRS_SIZE_, dtype=torch.int64),
                   _params: torch.ndarray = torch.Tensor([0]*cuda._PARAMS_SIZE_, dtype=torch.int32)) -> None: ...


def applyGaussGaugeQcu(_gaug: cp_ndarray = torch.Tensor([0.0], dtype=cuda.dtype(cuda._LAT_C64_)),
                       _set_ptrs: torch.ndarray = torch.Tensor(
                           [0]*cuda._SET_PTRS_SIZE_, dtype=torch.int64),
                       _params: torch.ndarray = torch.Tensor([0]*cuda._PARAMS_SIZE_, dtype=torch.int32)) -> None: ...


def applyCloverBistabCgQcu(_fermion_ou: cp_ndarray = torch.Tensor([0.0], dtype=cuda.dtype(cuda._LAT_C64_)),
                           _fermion_i: cp_ndarray = torch.Tensor(
                               [0.0], dtype=cuda.dtype(cuda._LAT_C64_)),
                           _gaug: cp_ndarray = torch.Tensor(
                               [0.0], dtype=cuda.dtype(cuda._LAT_C64_)),
                           _clover_e: cp_ndarray = torch.Tensor(
                               [0.0], dtype=cuda.dtype(cuda._LAT_C64_)),
                           _clover_o: cp_ndarray = torch.Tensor(
                               [0.0], dtype=cuda.dtype(cuda._LAT_C64_)),
                           _clover_ee_in: cp_ndarray = torch.Tensor(
                               [0.0], dtype=cuda.dtype(cuda._LAT_C64_)),
                           _clover_oo_in: cp_ndarray = torch.Tensor(
                               [0.0], dtype=cuda.dtype(cuda._LAT_C64_)),
                           _set_ptrs: torch.ndarray = torch.Tensor(
                               [0]*cuda._SET_PTRS_SIZE_, dtype=torch.int64),
                           _params: torch.ndarray = torch.Tensor([0]*cuda._PARAMS_SIZE_, dtype=torch.int32)) -> None: ...
