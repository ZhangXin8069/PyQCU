import torch
from pyqcu.cuda import define


def applyInitQcu(_set_ptrs: torch.Tensor = torch.Tensor([0]*define._SET_PTRS_SIZE_).to(dtype=torch.int64, device=torch.device('cpu')), _params: torch.Tensor = torch.Tensor([0]*define._PARAMS_SIZE_).to(dtype=torch.int32, device=torch.device('cpu')), _argv: torch.Tensor = torch.Tensor([0.0]*define._SET_PTRS_SIZE_).to(dtype=define.dtype().to_real(), device=torch.device('cpu'))) -> None:
    """
    # follow above, most values in params should be set.
    pay attention to the dtype of argv. (to_real)
    """
    ...


def applyEndQcu(_set_ptrs: torch.Tensor = torch.Tensor([0]*define._SET_PTRS_SIZE_).to(dtype=torch.int64, device=torch.device('cpu')), _params: torch.Tensor = torch.Tensor([0]*define._PARAMS_SIZE_).to(dtype=torch.int32, device=torch.device('cpu'))) -> None:
    """
    # Using this function causes bugs in subsequent functions; it has been deprecated. Do not use it!
    # follow above, most values in params should be set.
    """
    ...


def testWilsonDslashQcu(_fermion_out: torch.Tensor = torch.Tensor([0.0]).to(device=torch.device('cuda')), _fermion_in: torch.Tensor = torch.Tensor([0.0]).to(device=torch.device('cuda')), _gauge: torch.Tensor = torch.Tensor([0.0]).to(device=torch.device('cuda')), _set_ptrs: torch.Tensor = torch.Tensor([0]*define._SET_PTRS_SIZE_).to(dtype=torch.int64, device=torch.device('cpu')), _params: torch.Tensor = torch.Tensor([0]*define._PARAMS_SIZE_).to(dtype=torch.int32, device=torch.device('cpu'))) -> None:
    """
    # follow above, most values in params should be set.
    even-odd, parity in params should be set to 0 or 1, fermion_out:[scxyzt] in [pscxyzt], fermion_in:[scxyzt] in [pscxyzt], gauge:[pccdxyzt].
    """
    ...


def applyWilsonDslashQcu(_fermion_out: torch.Tensor = torch.Tensor([0.0]).to(device=torch.device('cuda')), _fermion_in: torch.Tensor = torch.Tensor([0.0]).to(device=torch.device('cuda')), _gauge: torch.Tensor = torch.Tensor([0.0]).to(device=torch.device('cuda')), _set_ptrs: torch.Tensor = torch.Tensor([0]*define._SET_PTRS_SIZE_).to(dtype=torch.int64, device=torch.device('cpu')), _params: torch.Tensor = torch.Tensor([0]*define._PARAMS_SIZE_).to(dtype=torch.int32, device=torch.device('cpu'))) -> None:
    """
    # follow above, most values in params should be set.
    even-odd, parity in params should be set to 0 or 1, fermion_out:[scxyzt] in [pscxyzt], fermion_in:[scxyzt] in [pscxyzt], gauge:[pccdxyzt].
    """
    ...


def testCloverDslashQcu(_fermion_out: torch.Tensor = torch.Tensor([0.0]).to(device=torch.device('cuda')), _fermion_in: torch.Tensor = torch.Tensor([0.0]).to(device=torch.device('cuda')), _gauge: torch.Tensor = torch.Tensor([0.0]).to(device=torch.device('cuda')), _set_ptrs: torch.Tensor = torch.Tensor([0]*define._SET_PTRS_SIZE_).to(dtype=torch.int64, device=torch.device('cpu')), _params: torch.Tensor = torch.Tensor([0]*define._PARAMS_SIZE_).to(dtype=torch.int32, device=torch.device('cpu'))) -> None:
    """
    # follow above, most values in params should be set.
    even-odd, parity in params should be set to 0 or 1, fermion_out:[scxyzt] in [pscxyzt], fermion_in:[scxyzt] in [pscxyzt], gauge:[pccdxyzt].
    """
    ...


def applyCloverDslashQcu(_fermion_out: torch.Tensor = torch.Tensor([0.0]).to(device=torch.device('cuda')), _fermion_in: torch.Tensor = torch.Tensor([0.0]).to(device=torch.device('cuda')), _gauge: torch.Tensor = torch.Tensor([0.0]).to(device=torch.device('cuda')), _set_ptrs: torch.Tensor = torch.Tensor([0]*define._SET_PTRS_SIZE_).to(dtype=torch.int64, device=torch.device('cpu')), _params: torch.Tensor = torch.Tensor([0]*define._PARAMS_SIZE_).to(dtype=torch.int32, device=torch.device('cpu'))) -> None:
    """
    # follow above, most values in params should be set.
    even-odd, parity in params should be set to 0 or 1, fermion_out:[scxyzt] in [pscxyzt], fermion_in:[scxyzt] in [pscxyzt], gauge:[pccdxyzt].
    """
    ...


def applyWilsonBistabCgQcu(_fermion_out: torch.Tensor = torch.Tensor([0.0]).to(device=torch.device('cuda')), _fermion_in: torch.Tensor = torch.Tensor([0.0]).to(device=torch.device('cuda')), _gauge: torch.Tensor = torch.Tensor([0.0]).to(device=torch.device('cuda')), _set_ptrs: torch.Tensor = torch.Tensor([0]*define._SET_PTRS_SIZE_).to(dtype=torch.int64, device=torch.device('cpu')), _params: torch.Tensor = torch.Tensor([0]*define._PARAMS_SIZE_).to(dtype=torch.int32, device=torch.device('cpu'))) -> None:
    """
    # follow above, most values in params should be set.
    even-odd, parity in params should be set to 0 or 1, fermion_out:[pscxyzt], fermion_in:[scxyzt] in [pscxyzt], gauge:[pccdxyzt].
    """
    ...


def applyWilsonBistabCgDslashQcu(_fermion_out: torch.Tensor = torch.Tensor([0.0]).to(device=torch.device('cuda')), _fermion_in: torch.Tensor = torch.Tensor([0.0]).to(device=torch.device('cuda')), _gauge: torch.Tensor = torch.Tensor([0.0]).to(device=torch.device('cuda')), _set_ptrs: torch.Tensor = torch.Tensor([0]*define._SET_PTRS_SIZE_).to(dtype=torch.int64, device=torch.device('cpu')), _params: torch.Tensor = torch.Tensor([0]*define._PARAMS_SIZE_).to(dtype=torch.int32, device=torch.device('cpu'))) -> None:
    """
    # follow above, most values in params should be set.
    even-odd, parity in params should be set to 0 or 1, fermion_out:[scxyzt] in [pscxyzt], fermion_in:[scxyzt] in [pscxyzt], gauge:[pccdxyzt].
    """
    ...


def applyWilsonCgQcu(_fermion_out: torch.Tensor = torch.Tensor([0.0]).to(device=torch.device('cuda')), _fermion_in: torch.Tensor = torch.Tensor([0.0]).to(device=torch.device('cuda')), _gauge: torch.Tensor = torch.Tensor([0.0]).to(device=torch.device('cuda')), _set_ptrs: torch.Tensor = torch.Tensor([0]*define._SET_PTRS_SIZE_).to(dtype=torch.int64, device=torch.device('cpu')), _params: torch.Tensor = torch.Tensor([0]*define._PARAMS_SIZE_).to(dtype=torch.int32, device=torch.device('cpu'))) -> None:
    """
    # follow above, most values in params should be set.
    even-odd, parity in params should be set to 0 or 1, fermion_out:[pscxyzt], fermion_in:[scxyzt] in [pscxyzt], gauge:[pccdxyzt].
    """
    ...


def applyWilsonCgDslashQcu(_fermion_out: torch.Tensor = torch.Tensor([0.0]).to(device=torch.device('cuda')), _fermion_in: torch.Tensor = torch.Tensor([0.0]).to(device=torch.device('cuda')), _gauge: torch.Tensor = torch.Tensor([0.0]).to(device=torch.device('cuda')), _set_ptrs: torch.Tensor = torch.Tensor([0]*define._SET_PTRS_SIZE_).to(dtype=torch.int64, device=torch.device('cpu')), _params: torch.Tensor = torch.Tensor([0]*define._PARAMS_SIZE_).to(dtype=torch.int32, device=torch.device('cpu'))) -> None:
    """
    # follow above, most values in params should be set.
    even-odd, parity in params should be set to 0 or 1, fermion_out:[scxyzt] in [pscxyzt], fermion_in:[scxyzt] in [pscxyzt], gauge:[pccdxyzt].
    """
    ...


def applyLaplacianQcu(_laplacian_ou: torch.Tensor = torch.Tensor([0.0]).to(device=torch.device('cuda')), _laplacian_i: torch.Tensor = torch.Tensor([0.0]).to(device=torch.device('cuda')), _gauge: torch.Tensor = torch.Tensor([0.0]).to(device=torch.device('cuda')), _set_ptrs: torch.Tensor = torch.Tensor([0]*define._SET_PTRS_SIZE_).to(dtype=torch.int64, device=torch.device('cpu')), _params: torch.Tensor = torch.Tensor([0]*define._PARAMS_SIZE_).to(dtype=torch.int32, device=torch.device('cpu'))) -> None:
    """
    # follow above, most values in params should be set.
    no even-odd, fermion_out:[scxyz1], fermion_in:[scxyz1] in [scxyz1], gauge:[ccdxyz1].
    """
    ...


def applyCloverQcu(_clover: torch.Tensor = torch.Tensor([0.0]).to(device=torch.device('cuda')), _gauge: torch.Tensor = torch.Tensor([0.0]).to(device=torch.device('cuda')), _set_ptrs: torch.Tensor = torch.Tensor([0]*define._SET_PTRS_SIZE_).to(dtype=torch.int64, device=torch.device('cpu')), _params: torch.Tensor = torch.Tensor([0]*define._PARAMS_SIZE_).to(dtype=torch.int32, device=torch.device('cpu'))) -> None:
    """
    # follow above, most values in params should be set.
    even-odd, parity in params should be set to 0 or 1, clover:[scscxyzt] in [pscscxyzt], gauge:[pccdxyzt].
    """


def applyCloversQcu(_clover: torch.Tensor = torch.Tensor([0.0]).to(device=torch.device('cuda')), _clover_inv: torch.Tensor = torch.Tensor([0.0]).to(device=torch.device('cuda')), _gauge: torch.Tensor = torch.Tensor([0.0]).to(device=torch.device('cuda')), _set_ptrs: torch.Tensor = torch.Tensor([0]*define._SET_PTRS_SIZE_).to(dtype=torch.int64, device=torch.device('cpu')), _params: torch.Tensor = torch.Tensor([0]*define._PARAMS_SIZE_).to(dtype=torch.int32, device=torch.device('cpu'))) -> None:
    """
    # follow above, most values in params should be set.
    even-odd, parity in params should be set to 0 or 1, clover:[scscxyzt] in [pscscxyzt], clover_inv:[scscxyzt] in [pscscxyzt], gauge:[pccdxyzt].
    """


def applyDslashQcu(_fermion_out: torch.Tensor = torch.Tensor([0.0]).to(device=torch.device('cuda')), _fermion_in: torch.Tensor = torch.Tensor([0.0]).to(device=torch.device('cuda')), _gauge: torch.Tensor = torch.Tensor([0.0]).to(device=torch.device('cuda')), _clover: torch.Tensor = torch.Tensor([0.0]).to(device=torch.device('cuda')), _set_ptrs: torch.Tensor = torch.Tensor([0]*define._SET_PTRS_SIZE_).to(dtype=torch.int64, device=torch.device('cpu')), _params: torch.Tensor = torch.Tensor([0]*define._PARAMS_SIZE_).to(dtype=torch.int32, device=torch.device('cpu'))) -> None:
    """
    # follow above, most values in params should be set.
    even-odd, parity in params should be set to 0 or 1, fermion_out:[scxyzt] in [pscxyzt], fermion_in:[scxyzt] in [pscxyzt], clover_inv:[scscxyzt] in [pscscxyzt], gauge:[pccdxyzt].
    """
    ...


def applyGaussGaugeQcu(_gauge: torch.Tensor = torch.Tensor([0.0]).to(device=torch.device('cuda')), _set_ptrs: torch.Tensor = torch.Tensor([0]*define._SET_PTRS_SIZE_).to(dtype=torch.int64, device=torch.device('cpu')), _params: torch.Tensor = torch.Tensor([0]*define._PARAMS_SIZE_).to(dtype=torch.int32, device=torch.device('cpu'))) -> None:
    """
    # follow above, most values in params should be set.
    even-odd, gauge:[pccdxyzt].
    """
    ...


def applyCloverBistabCgQcu(_fermion_out: torch.Tensor = torch.Tensor([0.0]).to(device=torch.device('cuda')), _fermion_in: torch.Tensor = torch.Tensor([0.0]).to(device=torch.device('cuda')), _gauge: torch.Tensor = torch.Tensor([0.0]).to(device=torch.device('cuda')), _clover_ee: torch.Tensor = torch.Tensor([0.0]).to(device=torch.device('cuda')), _clover_oo: torch.Tensor = torch.Tensor([0.0]).to(device=torch.device('cuda')), _clover_ee_inv: torch.Tensor = torch.Tensor([0.0]).to(device=torch.device('cuda')), _clover_oo_inv: torch.Tensor = torch.Tensor([0.0]).to(device=torch.device('cuda')), _set_ptrs: torch.Tensor = torch.Tensor([0]*define._SET_PTRS_SIZE_).to(dtype=torch.int64, device=torch.device('cpu')), _params: torch.Tensor = torch.Tensor([0]*define._PARAMS_SIZE_).to(dtype=torch.int32, device=torch.device('cpu'))) -> None:
    """
    # follow above, most values in params should be set.
    even-odd, parity in params should be set to 0 or 1, fermion_out:[scxyzt] in [pscxyzt], fermion_in:[scxyzt] in [pscxyzt], gauge:[pccdxyzt].
    """
    ...
