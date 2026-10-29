from __future__ import annotations

import pytest
import torch

from pyqcu import lattice, tools
from pyqcu.torch import dslash
from pyqcu.cuda import QcuRuntime, define


def _qcu_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from pyqcu.cuda import qcu
        return hasattr(qcu, "applyCloverQcu")
    except Exception:
        return False


def _test_device() -> str:
    for index in range(torch.cuda.device_count()):
        if torch.cuda.get_device_capability(index)[0] >= 7:
            return f"cuda:{index}"
    pytest.skip("no CUDA device supported by the current PyTorch build")


@pytest.mark.skipif(not _qcu_available(), reason="QCU CUDA extension unavailable")
@pytest.mark.parametrize("csw", [0.0, 0.35, 1.7])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128],
                         ids=["c64", "c128"])
def test_cuda_clover_matches_python_csw(csw: float, dtype: torch.dtype) -> None:
    lattice_shape = (8, 8, 8, 8)
    full_gauge = torch.empty((3, 3, 4, *lattice_shape), dtype=dtype)
    lattice.generate_gauge_field(full_gauge, sigma=0.15, seed=17, verbose=False)

    mass = 0.1
    runtime = QcuRuntime.from_lattice(
        lattice_shape, mass=mass, csw=csw, dtype=dtype,
        device=_test_device(), plan=2)
    gauge_parity = tools.oooxyzt2poooxyzt(full_gauge).to(
        runtime.device).contiguous()
    cpp_clover = runtime.make_clover(gauge_parity, parity=0)
    expected = dslash.make_clover(
        full_gauge, kappa=torch.tensor([runtime.kappa]), csw=csw)
    expected = tools.oooxyzt2poooxyzt(expected)[0]
    bare_cpp = dslash.cut_I(cpp_clover).detach().cpu()
    difference = float(torch.linalg.vector_norm(bare_cpp - expected))
    expected_norm = float(torch.linalg.vector_norm(expected))
    if expected_norm == 0.0:
        assert difference < 1.0e-6
    else:
        tolerance = 2.0e-3 if dtype == torch.complex64 else 2.0e-5
        assert difference / expected_norm < tolerance


@pytest.mark.skipif(not _qcu_available(), reason="QCU CUDA extension unavailable")
def test_qcu_init_rejects_invalid_control_abi() -> None:
    from pyqcu.cuda import qcu

    params = torch.zeros(define._PARAMS_SIZE_, dtype=torch.int32)
    params[define._DATA_TYPE_] = define._LAT_C64_
    set_ptrs = torch.zeros(define._SET_PTRS_SIZE_, dtype=torch.int64)
    with pytest.raises(ValueError, match="argv"):
        qcu.applyInitQcu(
            set_ptrs, params, torch.zeros(7, dtype=torch.float32))
    with pytest.raises(TypeError, match="argv dtype"):
        qcu.applyInitQcu(
            set_ptrs, params, torch.zeros(8, dtype=torch.float64))


@pytest.mark.skipif(not _qcu_available(), reason="QCU CUDA extension unavailable")
def test_cuda_clover_parity_refresh_within_initialized_slot() -> None:
    from pyqcu.cuda import qcu

    lattice_shape = (4, 4, 4, 4)
    full_gauge = torch.empty(
        (3, 3, 4, *lattice_shape), dtype=torch.complex64)
    lattice.generate_gauge_field(
        full_gauge, sigma=0.2, seed=53, verbose=False)
    runtime = QcuRuntime.from_lattice(
        lattice_shape, mass=0.1, csw=0.8, dtype=torch.complex64,
        device=_test_device(), plan=define._SET_PLAN2_)
    gauge = tools.oooxyzt2poooxyzt(full_gauge).to(
        runtime.device).contiguous()
    shape = (4, 3, 4, 3, 4, 4, 4, 2)
    even = torch.empty(shape, dtype=gauge.dtype, device=gauge.device)
    odd = torch.empty_like(even)
    even_inv = torch.empty_like(even)
    odd_inv = torch.empty_like(even)

    qcu.applyInitQcu(runtime.set_ptrs, runtime.params, runtime.argv)
    try:
        runtime.params[define._PARITY_] = 0
        qcu.applyCloversQcu(
            even, even_inv, gauge, runtime.set_ptrs, runtime.params)
        runtime.params[define._PARITY_] = 1
        qcu.applyCloversQcu(
            odd, odd_inv, gauge, runtime.set_ptrs, runtime.params)
    finally:
        qcu.applyEndQcu(runtime.set_ptrs, runtime.params)

    reference_odd, _ = runtime.make_clover(
        gauge, with_inverse=True, parity=1)
    parity_difference = float(torch.linalg.vector_norm(odd - even))
    reference_difference = float(torch.linalg.vector_norm(odd - reference_odd))
    assert parity_difference / float(torch.linalg.vector_norm(even)) > 1.0e-3
    assert reference_difference / float(torch.linalg.vector_norm(reference_odd)) < 1.0e-6


@pytest.mark.skipif(not _qcu_available(), reason="QCU CUDA extension unavailable")
def test_test_clover_dslash_uses_configured_csw() -> None:
    lattice_shape = (4, 4, 4, 4)
    full_gauge = torch.empty((3, 3, 4, *lattice_shape), dtype=torch.complex64)
    lattice.generate_gauge_field(full_gauge, sigma=0.2, seed=31, verbose=False)
    runtime = QcuRuntime.from_lattice(
        lattice_shape, mass=0.1, csw=0.35, dtype=torch.complex64,
        device=_test_device(), plan=2)
    gauge = tools.oooxyzt2poooxyzt(full_gauge).to(
        runtime.device).contiguous()
    compact = (*lattice_shape[:3], lattice_shape[3] // 2)
    generator = torch.Generator().manual_seed(37)
    source = torch.complex(
        torch.randn((4, 3, *compact), generator=generator),
        torch.randn((4, 3, *compact), generator=generator),
    ).to(runtime.device)
    production = torch.empty_like(source)
    diagnostic = torch.empty_like(source)
    runtime.call("applyCloverDslashQcu", production, source, gauge)
    runtime.call("testCloverDslashQcu", diagnostic, source, gauge)
    relative = float(
        torch.linalg.vector_norm(production - diagnostic) /
        torch.linalg.vector_norm(production))
    assert relative < 1.0e-6
