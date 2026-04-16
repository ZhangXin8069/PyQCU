import torch
from pyqcu.cuda import define


def applyInitQcu(_set_ptrs: torch.Tensor = torch.Tensor([0]*define._SET_PTRS_SIZE_).to(dtype=torch.int64, device=torch.device('cpu')),
                 _params: torch.Tensor = torch.Tensor(
    [0]*define._PARAMS_SIZE_).to(dtype=torch.int32, device=torch.device('cpu')),
    _argv: torch.Tensor = torch.Tensor([0.0]*define._SET_PTRS_SIZE_, dtype=define.dtype_half(define._LAT_C64_))) -> None: ...


def applyEndQcu(_set_ptrs: torch.Tensor = torch.Tensor([0]*define._SET_PTRS_SIZE_).to(dtype=torch.int64, device=torch.device('cpu')),
                _params: torch.Tensor = torch.Tensor([0]*define._PARAMS_SIZE_).to(dtype=torch.int32, device=torch.device('cpu'))) -> None: ...


def testWilsonDslashQcu(_fermion_out: torch.Tensor = torch.Tensor([0.0]).to(dtype=define.dtype(define._LAT_C64_), device=torch.device('cuda')),
                        _fermion_in: torch.Tensor = torch.Tensor(
    [0.0]).to(dtype=define.dtype(define._LAT_C64_), device=torch.device('cuda')),
    _gauge: torch.Tensor = torch.Tensor(
    [0.0]).to(dtype=define.dtype(define._LAT_C64_), device=torch.device('cuda')),
    _set_ptrs: torch.Tensor = torch.Tensor(
    [0]*define._SET_PTRS_SIZE_).to(dtype=torch.int64, device=torch.device('cpu')),
    _params: torch.Tensor = torch.Tensor([0]*define._PARAMS_SIZE_).to(dtype=torch.int32, device=torch.device('cpu'))) -> None: ...


def applyWilsonDslashQcu(_fermion_out: torch.Tensor = torch.Tensor([0.0]).to(dtype=define.dtype(define._LAT_C64_), device=torch.device('cuda')),
                         _fermion_in: torch.Tensor = torch.Tensor(
    [0.0]).to(dtype=define.dtype(define._LAT_C64_), device=torch.device('cuda')),
    _gauge: torch.Tensor = torch.Tensor(
    [0.0]).to(dtype=define.dtype(define._LAT_C64_), device=torch.device('cuda')),
    _set_ptrs: torch.Tensor = torch.Tensor(
    [0]*define._SET_PTRS_SIZE_).to(dtype=torch.int64, device=torch.device('cpu')),
    _params: torch.Tensor = torch.Tensor([0]*define._PARAMS_SIZE_).to(dtype=torch.int32, device=torch.device('cpu'))) -> None: ...


def testCloverDslashQcu(_fermion_out: torch.Tensor = torch.Tensor([0.0]).to(dtype=define.dtype(define._LAT_C64_), device=torch.device('cuda')),
                        _fermion_in: torch.Tensor = torch.Tensor(
    [0.0]).to(dtype=define.dtype(define._LAT_C64_), device=torch.device('cuda')),
    _gauge: torch.Tensor = torch.Tensor(
    [0.0]).to(dtype=define.dtype(define._LAT_C64_), device=torch.device('cuda')),
    _set_ptrs: torch.Tensor = torch.Tensor(
    [0]*define._SET_PTRS_SIZE_).to(dtype=torch.int64, device=torch.device('cpu')),
    _params: torch.Tensor = torch.Tensor([0]*define._PARAMS_SIZE_).to(dtype=torch.int32, device=torch.device('cpu'))) -> None: ...


def applyCloverDslashQcu(_fermion_out: torch.Tensor = torch.Tensor([0.0]).to(dtype=define.dtype(define._LAT_C64_), device=torch.device('cuda')),
                         _fermion_in: torch.Tensor = torch.Tensor(
    [0.0]).to(dtype=define.dtype(define._LAT_C64_), device=torch.device('cuda')),
    _gauge: torch.Tensor = torch.Tensor(
    [0.0]).to(dtype=define.dtype(define._LAT_C64_), device=torch.device('cuda')),
    _set_ptrs: torch.Tensor = torch.Tensor(
    [0]*define._SET_PTRS_SIZE_).to(dtype=torch.int64, device=torch.device('cpu')),
    _params: torch.Tensor = torch.Tensor([0]*define._PARAMS_SIZE_).to(dtype=torch.int32, device=torch.device('cpu'))) -> None: ...


def applyWilsonBistabCgQcu(_fermion_out: torch.Tensor = torch.Tensor([0.0]).to(dtype=define.dtype(define._LAT_C64_), device=torch.device('cuda')),
                           _fermion_in: torch.Tensor = torch.Tensor(
    [0.0]).to(dtype=define.dtype(define._LAT_C64_), device=torch.device('cuda')),
    _gauge: torch.Tensor = torch.Tensor(
    [0.0]).to(dtype=define.dtype(define._LAT_C64_), device=torch.device('cuda')),
    _set_ptrs: torch.Tensor = torch.Tensor(
    [0]*define._SET_PTRS_SIZE_).to(dtype=torch.int64, device=torch.device('cpu')),
    _params: torch.Tensor = torch.Tensor([0]*define._PARAMS_SIZE_).to(dtype=torch.int32, device=torch.device('cpu'))) -> None: ...


def applyWilsonBistabCgDslashQcu(
    _fermion_out: torch.Tensor = torch.Tensor(
        [0.0]).to(dtype=define.dtype(define._LAT_C64_), device=torch.device('cuda')),
    _fermion_in: torch.Tensor = torch.Tensor(
        [0.0]).to(dtype=define.dtype(define._LAT_C64_), device=torch.device('cuda')),
    _gauge: torch.Tensor = torch.Tensor(
        [0.0]).to(dtype=define.dtype(define._LAT_C64_), device=torch.device('cuda')),
    _set_ptrs: torch.Tensor = torch.Tensor(
        [0]*define._SET_PTRS_SIZE_).to(dtype=torch.int64, device=torch.device('cpu')),
    _params: torch.Tensor = torch.Tensor([0]*define._PARAMS_SIZE_).to(dtype=torch.int32, device=torch.device('cpu'))) -> None: ...


def applyWilsonCgQcu(_fermion_out: torch.Tensor = torch.Tensor([0.0]).to(dtype=define.dtype(define._LAT_C64_), device=torch.device('cuda')),
                     _fermion_in: torch.Tensor = torch.Tensor(
    [0.0]).to(dtype=define.dtype(define._LAT_C64_), device=torch.device('cuda')),
    _gauge: torch.Tensor = torch.Tensor(
    [0.0]).to(dtype=define.dtype(define._LAT_C64_), device=torch.device('cuda')),
    _set_ptrs: torch.Tensor = torch.Tensor(
    [0]*define._SET_PTRS_SIZE_).to(dtype=torch.int64, device=torch.device('cpu')),
    _params: torch.Tensor = torch.Tensor([0]*define._PARAMS_SIZE_).to(dtype=torch.int32, device=torch.device('cpu'))) -> None: ...


def applyWilsonCgDslashQcu(_fermion_out: torch.Tensor = torch.Tensor([0.0]).to(dtype=define.dtype(define._LAT_C64_), device=torch.device('cuda')),
                           _fermion_in: torch.Tensor = torch.Tensor(
    [0.0]).to(dtype=define.dtype(define._LAT_C64_), device=torch.device('cuda')),
    _gauge: torch.Tensor = torch.Tensor(
    [0.0]).to(dtype=define.dtype(define._LAT_C64_), device=torch.device('cuda')),
    _set_ptrs: torch.Tensor = torch.Tensor(
    [0]*define._SET_PTRS_SIZE_).to(dtype=torch.int64, device=torch.device('cpu')),
    _params: torch.Tensor = torch.Tensor([0]*define._PARAMS_SIZE_).to(dtype=torch.int32, device=torch.device('cpu'))) -> None: ...


def applyLaplacianQcu(_laplacian_ou: torch.Tensor = torch.Tensor([0.0]).to(dtype=define.dtype(define._LAT_C64_), device=torch.device('cuda')),
                      _laplacian_i: torch.Tensor = torch.Tensor(
    [0.0]).to(dtype=define.dtype(define._LAT_C64_), device=torch.device('cuda')),
    _gauge: torch.Tensor = torch.Tensor(
    [0.0]).to(dtype=define.dtype(define._LAT_C64_), device=torch.device('cuda')),
    _set_ptrs: torch.Tensor = torch.Tensor(
    [0]*define._SET_PTRS_SIZE_).to(dtype=torch.int64, device=torch.device('cpu')),
    _params: torch.Tensor = torch.Tensor([0]*define._PARAMS_SIZE_).to(dtype=torch.int32, device=torch.device('cpu'))) -> None: ...


def applyCloverQcu(_clover: torch.Tensor = torch.Tensor([0.0]).to(dtype=define.dtype(define._LAT_C64_), device=torch.device('cuda')),
                   _gauge: torch.Tensor = torch.Tensor(
    [0.0]).to(dtype=define.dtype(define._LAT_C64_), device=torch.device('cuda')),
    _set_ptrs: torch.Tensor = torch.Tensor(
    [0]*define._SET_PTRS_SIZE_).to(dtype=torch.int64, device=torch.device('cpu')),
    _params: torch.Tensor = torch.Tensor([0]*define._PARAMS_SIZE_).to(dtype=torch.int32, device=torch.device('cpu'))) -> None: ...


def applyCloversQcu(_clover: torch.Tensor = torch.Tensor([0.0]).to(dtype=define.dtype(define._LAT_C64_), device=torch.device('cuda')),
                    _clover_in: torch.Tensor = torch.Tensor(
    [0.0]).to(dtype=define.dtype(define._LAT_C64_), device=torch.device('cuda')),
    _gauge: torch.Tensor = torch.Tensor(
    [0.0]).to(dtype=define.dtype(define._LAT_C64_), device=torch.device('cuda')),
    _set_ptrs: torch.Tensor = torch.Tensor(
    [0]*define._SET_PTRS_SIZE_).to(dtype=torch.int64, device=torch.device('cpu')),
    _params: torch.Tensor = torch.Tensor([0]*define._PARAMS_SIZE_).to(dtype=torch.int32, device=torch.device('cpu'))) -> None: ...


def applyDslashQcu(_fermion_out: torch.Tensor = torch.Tensor([0.0]).to(dtype=define.dtype(define._LAT_C64_), device=torch.device('cuda')),
                   _fermion_in: torch.Tensor = torch.Tensor(
    [0.0]).to(dtype=define.dtype(define._LAT_C64_), device=torch.device('cuda')),
    _gauge: torch.Tensor = torch.Tensor(
    [0.0]).to(dtype=define.dtype(define._LAT_C64_), device=torch.device('cuda')),
    _clover: torch.Tensor = torch.Tensor(
    [0.0]).to(dtype=define.dtype(define._LAT_C64_), device=torch.device('cuda')),
    _set_ptrs: torch.Tensor = torch.Tensor(
    [0]*define._SET_PTRS_SIZE_).to(dtype=torch.int64, device=torch.device('cpu')),
    _params: torch.Tensor = torch.Tensor([0]*define._PARAMS_SIZE_).to(dtype=torch.int32, device=torch.device('cpu'))) -> None: ...


def applyGaussGaugeQcu(_gauge: torch.Tensor = torch.Tensor([0.0]).to(dtype=define.dtype(define._LAT_C64_), device=torch.device('cuda')),
                       _set_ptrs: torch.Tensor = torch.Tensor(
    [0]*define._SET_PTRS_SIZE_).to(dtype=torch.int64, device=torch.device('cpu')),
    _params: torch.Tensor = torch.Tensor([0]*define._PARAMS_SIZE_).to(dtype=torch.int32, device=torch.device('cpu'))) -> None: ...


def applyCloverBistabCgQcu(_fermion_out: torch.Tensor = torch.Tensor([0.0]).to(dtype=define.dtype(define._LAT_C64_), device=torch.device('cuda')),
                           _fermion_in: torch.Tensor = torch.Tensor(
    [0.0]).to(dtype=define.dtype(define._LAT_C64_), device=torch.device('cuda')),
    _gauge: torch.Tensor = torch.Tensor(
    [0.0]).to(dtype=define.dtype(define._LAT_C64_), device=torch.device('cuda')),
    _clover_e: torch.Tensor = torch.Tensor(
    [0.0]).to(dtype=define.dtype(define._LAT_C64_), device=torch.device('cuda')),
    _clover_o: torch.Tensor = torch.Tensor(
    [0.0]).to(dtype=define.dtype(define._LAT_C64_), device=torch.device('cuda')),
    _clover_ee_in: torch.Tensor = torch.Tensor(
    [0.0]).to(dtype=define.dtype(define._LAT_C64_), device=torch.device('cuda')),
    _clover_oo_in: torch.Tensor = torch.Tensor(
    [0.0]).to(dtype=define.dtype(define._LAT_C64_), device=torch.device('cuda')),
    _set_ptrs: torch.Tensor = torch.Tensor(
    [0]*define._SET_PTRS_SIZE_).to(dtype=torch.int64, device=torch.device('cpu')),
    _params: torch.Tensor = torch.Tensor([0]*define._PARAMS_SIZE_).to(dtype=torch.int32, device=torch.device('cpu'))) -> None: ...
