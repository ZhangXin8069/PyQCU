import numpy as np
import cupy as cp
from pyqcu.cuda import define
from pyqcu.cuda.define import cp_ndarray


def applyInitQcu(_set_ptrs: np.ndarray = np.array([0]*define._SET_PTRS_SIZE_, dtype=np.int64),
                 _params: np.ndarray = np.array(
                     [0]*define._PARAMS_SIZE_, dtype=np.int32),
                 _argv: np.ndarray = np.array([0.0]*define._SET_PTRS_SIZE_, dtype=define.dtype_half(define._LAT_C64_))) -> None: ...


def applyEndQcu(_set_ptrs: np.ndarray = np.array([0]*define._SET_PTRS_SIZE_, dtype=np.int64),
                _params: np.ndarray = np.array([0]*define._PARAMS_SIZE_, dtype=np.int32)) -> None: ...


def testWilsonDslashQcu(_fermion_ou: cp_ndarray = cp.array([0.0], dtype=define.dtype(define._LAT_C64_)),
                        _fermion_i: cp_ndarray = cp.array(
                            [0.0], dtype=define.dtype(define._LAT_C64_)),
                        _gaug: cp_ndarray = cp.array(
                            [0.0], dtype=define.dtype(define._LAT_C64_)),
                        _set_ptrs: np.ndarray = np.array(
                            [0]*define._SET_PTRS_SIZE_, dtype=np.int64),
                        _params: np.ndarray = np.array([0]*define._PARAMS_SIZE_, dtype=np.int32)) -> None: ...


def applyWilsonDslashQcu(_fermion_ou: cp_ndarray = cp.array([0.0], dtype=define.dtype(define._LAT_C64_)),
                         _fermion_i: cp_ndarray = cp.array(
                             [0.0], dtype=define.dtype(define._LAT_C64_)),
                         _gaug: cp_ndarray = cp.array(
                             [0.0], dtype=define.dtype(define._LAT_C64_)),
                         _set_ptrs: np.ndarray = np.array(
                             [0]*define._SET_PTRS_SIZE_, dtype=np.int64),
                         _params: np.ndarray = np.array([0]*define._PARAMS_SIZE_, dtype=np.int32)) -> None: ...


def testCloverDslashQcu(_fermion_ou: cp_ndarray = cp.array([0.0], dtype=define.dtype(define._LAT_C64_)),
                        _fermion_i: cp_ndarray = cp.array(
                            [0.0], dtype=define.dtype(define._LAT_C64_)),
                        _gaug: cp_ndarray = cp.array(
                            [0.0], dtype=define.dtype(define._LAT_C64_)),
                        _set_ptrs: np.ndarray = np.array(
                            [0]*define._SET_PTRS_SIZE_, dtype=np.int64),
                        _params: np.ndarray = np.array([0]*define._PARAMS_SIZE_, dtype=np.int32)) -> None: ...


def applyCloverDslashQcu(_fermion_ou: cp_ndarray = cp.array([0.0], dtype=define.dtype(define._LAT_C64_)),
                         _fermion_i: cp_ndarray = cp.array(
                             [0.0], dtype=define.dtype(define._LAT_C64_)),
                         _gaug: cp_ndarray = cp.array(
                             [0.0], dtype=define.dtype(define._LAT_C64_)),
                         _set_ptrs: np.ndarray = np.array(
                             [0]*define._SET_PTRS_SIZE_, dtype=np.int64),
                         _params: np.ndarray = np.array([0]*define._PARAMS_SIZE_, dtype=np.int32)) -> None: ...


def applyWilsonBistabCgQcu(_fermion_ou: cp_ndarray = cp.array([0.0], dtype=define.dtype(define._LAT_C64_)),
                           _fermion_i: cp_ndarray = cp.array(
                               [0.0], dtype=define.dtype(define._LAT_C64_)),
                           _gaug: cp_ndarray = cp.array(
                               [0.0], dtype=define.dtype(define._LAT_C64_)),
                           _set_ptrs: np.ndarray = np.array(
                               [0]*define._SET_PTRS_SIZE_, dtype=np.int64),
                           _params: np.ndarray = np.array([0]*define._PARAMS_SIZE_, dtype=np.int32)) -> None: ...


def applyWilsonBistabCgDslashQcu(
    _fermion_ou: cp_ndarray = cp.array(
        [0.0], dtype=define.dtype(define._LAT_C64_)),
    _fermion_i: cp_ndarray = cp.array(
        [0.0], dtype=define.dtype(define._LAT_C64_)),
    _gaug: cp_ndarray = cp.array([0.0], dtype=define.dtype(define._LAT_C64_)),
    _set_ptrs: np.ndarray = np.array(
        [0]*define._SET_PTRS_SIZE_, dtype=np.int64),
    _params: np.ndarray = np.array([0]*define._PARAMS_SIZE_, dtype=np.int32)) -> None: ...


def applyWilsonCgQcu(_fermion_ou: cp_ndarray = cp.array([0.0], dtype=define.dtype(define._LAT_C64_)),
                     _fermion_i: cp_ndarray = cp.array(
                         [0.0], dtype=define.dtype(define._LAT_C64_)),
                     _gaug: cp_ndarray = cp.array(
                         [0.0], dtype=define.dtype(define._LAT_C64_)),
                     _set_ptrs: np.ndarray = np.array(
                         [0]*define._SET_PTRS_SIZE_, dtype=np.int64),
                     _params: np.ndarray = np.array([0]*define._PARAMS_SIZE_, dtype=np.int32)) -> None: ...


def applyWilsonCgDslashQcu(_fermion_ou: cp_ndarray = cp.array([0.0], dtype=define.dtype(define._LAT_C64_)),
                           _fermion_i: cp_ndarray = cp.array(
                               [0.0], dtype=define.dtype(define._LAT_C64_)),
                           _gaug: cp_ndarray = cp.array(
                               [0.0], dtype=define.dtype(define._LAT_C64_)),
                           _set_ptrs: np.ndarray = np.array(
                               [0]*define._SET_PTRS_SIZE_, dtype=np.int64),
                           _params: np.ndarray = np.array([0]*define._PARAMS_SIZE_, dtype=np.int32)) -> None: ...


def applyLaplacianQcu(_laplacian_ou: cp_ndarray = cp.array([0.0], dtype=define.dtype(define._LAT_C64_)),
                      _laplacian_i: cp_ndarray = cp.array(
                          [0.0], dtype=define.dtype(define._LAT_C64_)),
                      _gaug: cp_ndarray = cp.array(
                          [0.0], dtype=define.dtype(define._LAT_C64_)),
                      _set_ptrs: np.ndarray = np.array(
                          [0]*define._SET_PTRS_SIZE_, dtype=np.int64),
                      _params: np.ndarray = np.array([0]*define._PARAMS_SIZE_, dtype=np.int32)) -> None: ...


def applyCloverQcu(_clove: cp_ndarray = cp.array([0.0], dtype=define.dtype(define._LAT_C64_)),
                   _gaug: cp_ndarray = cp.array(
                       [0.0], dtype=define.dtype(define._LAT_C64_)),
                   _set_ptrs: np.ndarray = np.array(
                       [0]*define._SET_PTRS_SIZE_, dtype=np.int64),
                   _params: np.ndarray = np.array([0]*define._PARAMS_SIZE_, dtype=np.int32)) -> None: ...


def applyCloversQcu(_clove: cp_ndarray = cp.array([0.0], dtype=define.dtype(define._LAT_C64_)),
                    _clover_in: cp_ndarray = cp.array(
                        [0.0], dtype=define.dtype(define._LAT_C64_)),
                    _gaug: cp_ndarray = cp.array(
                        [0.0], dtype=define.dtype(define._LAT_C64_)),
                    _set_ptrs: np.ndarray = np.array(
                        [0]*define._SET_PTRS_SIZE_, dtype=np.int64),
                    _params: np.ndarray = np.array([0]*define._PARAMS_SIZE_, dtype=np.int32)) -> None: ...


def applyDslashQcu(_fermion_ou: cp_ndarray = cp.array([0.0], dtype=define.dtype(define._LAT_C64_)),
                   _fermion_i: cp_ndarray = cp.array(
                       [0.0], dtype=define.dtype(define._LAT_C64_)),
                   _gaug: cp_ndarray = cp.array(
                       [0.0], dtype=define.dtype(define._LAT_C64_)),
                   _clove: cp_ndarray = cp.array(
                       [0.0], dtype=define.dtype(define._LAT_C64_)),
                   _set_ptrs: np.ndarray = np.array(
                       [0]*define._SET_PTRS_SIZE_, dtype=np.int64),
                   _params: np.ndarray = np.array([0]*define._PARAMS_SIZE_, dtype=np.int32)) -> None: ...


def applyGaussGaugeQcu(_gaug: cp_ndarray = cp.array([0.0], dtype=define.dtype(define._LAT_C64_)),
                       _set_ptrs: np.ndarray = np.array(
                           [0]*define._SET_PTRS_SIZE_, dtype=np.int64),
                       _params: np.ndarray = np.array([0]*define._PARAMS_SIZE_, dtype=np.int32)) -> None: ...


def applyCloverBistabCgQcu(_fermion_ou: cp_ndarray = cp.array([0.0], dtype=define.dtype(define._LAT_C64_)),
                           _fermion_i: cp_ndarray = cp.array(
                               [0.0], dtype=define.dtype(define._LAT_C64_)),
                           _gaug: cp_ndarray = cp.array(
                               [0.0], dtype=define.dtype(define._LAT_C64_)),
                           _clover_e: cp_ndarray = cp.array(
                               [0.0], dtype=define.dtype(define._LAT_C64_)),
                           _clover_o: cp_ndarray = cp.array(
                               [0.0], dtype=define.dtype(define._LAT_C64_)),
                           _clover_ee_in: cp_ndarray = cp.array(
                               [0.0], dtype=define.dtype(define._LAT_C64_)),
                           _clover_oo_in: cp_ndarray = cp.array(
                               [0.0], dtype=define.dtype(define._LAT_C64_)),
                           _set_ptrs: np.ndarray = np.array(
                               [0]*define._SET_PTRS_SIZE_, dtype=np.int64),
                           _params: np.ndarray = np.array([0]*define._PARAMS_SIZE_, dtype=np.int32)) -> None: ...
