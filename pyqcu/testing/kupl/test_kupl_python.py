from __future__ import annotations

import ctypes
from pathlib import Path

import pytest
import torch

from pyqcu import dslash, lattice, tools
from pyqcu.cuda import QcuRuntime, define
from pyqcu.testing.qcu.single_function_common import (
    qcu_clover_dslash_reference, qcu_wilson_dslash_reference)
from pyqcu.testing.kupl.check_abi import _declarations


_HEADER_CTYPES = {
    "double": ctypes.c_double,
    "double*": ctypes.POINTER(ctypes.c_double),
    "int": ctypes.c_int,
    "int*": ctypes.POINTER(ctypes.c_int),
    "long long": ctypes.c_longlong,
    "unsigned long long": ctypes.c_ulonglong,
    "unsigned long long*": ctypes.POINTER(ctypes.c_ulonglong),
}


def _kupl_qcu():
    try:
        from pyqcu.kupl import qcu
        return qcu
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(str(exc))


def test_kupl_exports_the_complete_qcu_symbol_set() -> None:
    qcu = _kupl_qcu()
    root = Path(__file__).resolve().parents[3]
    cuda_header = root / "cpp/cuda/qcu/python/pyqcu.h"
    kupl_header = root / "cpp/kupl/qcu/include/kupl/qcu.h"
    cuda_declarations = _declarations(cuda_header)
    kupl_declarations = _declarations(kupl_header)
    cuda_names = set(cuda_declarations)
    kupl_names = set(kupl_declarations)
    assert len(kupl_names) == 37
    assert len(qcu._SIGNATURES) == 37
    assert kupl_names == cuda_names
    assert kupl_declarations == cuda_declarations
    for name in sorted(cuda_names):
        assert getattr(qcu._LIBRARY, name)
        assert callable(getattr(qcu, name))
        return_type, arguments = kupl_declarations[name]
        expected_argtypes = tuple(_HEADER_CTYPES[value] for value in arguments)
        assert qcu._SIGNATURES[name] == expected_argtypes
        assert getattr(qcu._LIBRARY, name).argtypes == expected_argtypes
        expected_restype = ctypes.c_int if return_type == "int" else None
        assert getattr(qcu._LIBRARY, name).restype is expected_restype


def test_kupl_binding_rejects_noncontiguous_tensors() -> None:
    qcu = _kupl_qcu()
    with pytest.raises(ValueError, match="contiguous"):
        qcu._pointer(torch.empty((2, 3)).transpose(0, 1))


@pytest.mark.skipif(define.size != 1, reason="Python binding smoke uses one MPI rank")
def test_kupl_set_ptr_slots_have_independent_lifetimes() -> None:
    qcu = _kupl_qcu()
    first = QcuRuntime.from_lattice(
        (4, 4, 4, 4), mass=0.17, csw=0.8, device="cpu", plan=0,
        grid=(1, 1, 1, 1), qcu_module=qcu)
    second = QcuRuntime.from_lattice(
        (4, 4, 4, 4), mass=0.17, csw=0.8, device="cpu", plan=0,
        grid=(1, 1, 1, 1), qcu_module=qcu)
    shared = torch.zeros(define._SET_PTRS_SIZE_, dtype=torch.int64)
    first.set_ptrs = shared
    second.set_ptrs = shared
    first.params[define._SET_INDEX_] = 0
    second.params[define._SET_INDEX_] = 1
    qcu.applyInitQcu(first.set_ptrs, first.params, first.argv)
    qcu.applyInitQcu(second.set_ptrs, second.params, second.argv)
    assert int(shared[0]) != 0
    assert int(shared[1]) != 0
    qcu.applyEndQcu(first.set_ptrs, first.params)
    assert int(shared[0]) == 0
    assert int(shared[1]) != 0
    assert qcu.getCloverBistabCgIterationsQcu(
        second.set_ptrs, second.params) == 0
    qcu.applyEndQcu(second.set_ptrs, second.params)
    assert all(int(value) == 0 for value in shared)


@pytest.mark.skipif(define.size != 1, reason="Python binding smoke uses one MPI rank")
def test_kupl_python_clover_matches_reference() -> None:
    qcu = _kupl_qcu()
    lattice_shape = (4, 4, 4, 4)
    full_gauge = torch.empty((3, 3, 4, *lattice_shape), dtype=torch.complex64)
    lattice.generate_gauge_field(full_gauge, sigma=0.2, seed=23, verbose=False)
    gauge = tools.oooxyzt2poooxyzt(full_gauge).contiguous()

    runtime = QcuRuntime.from_lattice(
        lattice_shape, mass=0.17, csw=0.8, device="cpu", plan=2,
        grid=(1, 1, 1, 1), qcu_module=qcu)
    clover, clover_inv = runtime.make_clover(gauge, with_inverse=True, parity=0)
    expected = tools.oooxyzt2poooxyzt(dslash.make_clover(
        full_gauge, kappa=torch.tensor([runtime.kappa]), csw=0.8))[0]
    bare = dslash.cut_I(clover)
    relative_error = float(
        torch.linalg.vector_norm(bare - expected)
        / torch.linalg.vector_norm(expected))
    assert relative_error < 5.0e-4

    matrices = clover.permute(4, 5, 6, 7, 0, 1, 2, 3).reshape(-1, 12, 12)
    inverses = clover_inv.permute(4, 5, 6, 7, 0, 1, 2, 3).reshape(-1, 12, 12)
    product = matrices @ inverses
    identity = torch.eye(12, dtype=clover.dtype).expand_as(product)
    inverse_error = float(
        torch.linalg.vector_norm(product - identity)
        / torch.linalg.vector_norm(identity))
    assert inverse_error < 5.0e-4

    runtime.configure(csw=0.0)
    identity_clover, identity_inverse = runtime.make_clover(
        gauge, with_inverse=True, parity=0)
    identity_error = float(
        torch.linalg.vector_norm(
            identity_clover.permute(4, 5, 6, 7, 0, 1, 2, 3).reshape(-1, 12, 12)
            - torch.eye(12, dtype=clover.dtype))
        / torch.linalg.vector_norm(torch.eye(12, dtype=clover.dtype)))
    inverse_identity_error = float(
        torch.linalg.vector_norm(
            identity_inverse.permute(4, 5, 6, 7, 0, 1, 2, 3).reshape(-1, 12, 12)
            - torch.eye(12, dtype=clover.dtype))
        / torch.linalg.vector_norm(torch.eye(12, dtype=clover.dtype)))
    assert identity_error < 1.0e-6
    assert inverse_identity_error < 1.0e-6


@pytest.mark.skipif(define.size != 1, reason="Python binding smoke uses one MPI rank")
@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    [(torch.complex64, 2.0e-5), (torch.complex128, 2.0e-12)],
    ids=["c64", "c128"])
def test_kupl_python_gaussian_gauge_is_su3(
        dtype: torch.dtype, tolerance: float) -> None:
    qcu = _kupl_qcu()
    lattice_shape = (4, 4, 4, 4)
    runtime = QcuRuntime.from_lattice(
        lattice_shape, mass=0.17, csw=0.8, sigma=0.12, dtype=dtype,
        device="cpu", plan=define._SET_PLAN_N_1_, grid=(1, 1, 1, 1),
        seed=61, qcu_module=qcu)
    compact = (4, 4, 4, 2)
    gauge = torch.empty((2, 3, 3, 4, *compact), dtype=dtype)
    runtime.call("applyGaussGaugeQcu", gauge)
    assert torch.isfinite(gauge.real).all()
    assert torch.isfinite(gauge.imag).all()

    matrices = gauge.permute(0, 3, 4, 5, 6, 7, 1, 2).reshape(-1, 3, 3)
    identity = torch.eye(3, dtype=dtype).expand(matrices.shape[0], 3, 3)
    unitarity = (
        matrices.conj().transpose(-1, -2) @ matrices - identity)
    assert float(torch.linalg.vector_norm(unitarity) / torch.linalg.vector_norm(identity)) < tolerance
    determinants = torch.linalg.det(matrices)
    assert float(torch.max(torch.abs(determinants - 1))) < tolerance * 20

    repeated = torch.empty_like(gauge)
    runtime.call("applyGaussGaugeQcu", repeated)
    assert torch.equal(gauge, repeated)


@pytest.mark.skipif(define.size != 1, reason="Python binding smoke uses one MPI rank")
def test_kupl_python_dslash_matches_qcu_semantics() -> None:
    qcu = _kupl_qcu()
    lattice_shape = (4, 4, 4, 4)
    full_gauge = torch.empty((3, 3, 4, *lattice_shape), dtype=torch.complex64)
    lattice.generate_gauge_field(full_gauge, sigma=0.15, seed=29, verbose=False)
    gauge = tools.oooxyzt2poooxyzt(full_gauge).contiguous()
    compact = (*lattice_shape[:3], lattice_shape[3] // 2)
    generator = torch.Generator().manual_seed(37)
    source = torch.complex(
        torch.randn((4, 3, *compact), generator=generator),
        torch.randn((4, 3, *compact), generator=generator))

    runtime = QcuRuntime.from_lattice(
        lattice_shape, mass=0.17, csw=0.8, device="cpu", plan=0,
        grid=(1, 1, 1, 1), qcu_module=qcu)
    wilson = torch.empty_like(source)
    runtime.call("applyWilsonDslashQcu", wilson, source, gauge)
    expected_wilson = qcu_wilson_dslash_reference(source, gauge, parity=0)
    assert torch.allclose(wilson, expected_wilson, rtol=2.0e-4, atol=2.0e-5)

    runtime.configure(plan=2)
    clover_dslash = torch.empty_like(source)
    runtime.call("applyCloverDslashQcu", clover_dslash, source, gauge)
    expected_clover = qcu_clover_dslash_reference(
        source, gauge, mass=0.17, parity=0, csw=0.8)
    assert torch.allclose(
        clover_dslash, expected_clover, rtol=5.0e-4, atol=5.0e-5)

    identity = torch.eye(12, dtype=torch.complex64).reshape(
        4, 3, 4, 3, 1, 1, 1, 1).expand(4, 3, 4, 3, *compact).contiguous()
    supplied = torch.empty_like(source)
    runtime.call("applyDslashQcu", supplied, source, gauge, identity)
    assert torch.allclose(supplied, expected_wilson, rtol=2.0e-4, atol=2.0e-5)


@pytest.mark.skipif(define.size != 1, reason="Python binding smoke uses one MPI rank")
def test_kupl_python_clover_bistabcg_residual() -> None:
    qcu = _kupl_qcu()
    lattice_shape = (4, 4, 4, 4)
    compact = (*lattice_shape[:3], lattice_shape[3] // 2)
    full_gauge = torch.empty((3, 3, 4, *lattice_shape), dtype=torch.complex64)
    lattice.generate_gauge_field(full_gauge, sigma=0.12, seed=41, verbose=False)
    gauge = tools.oooxyzt2poooxyzt(full_gauge).contiguous()
    generator = torch.Generator().manual_seed(43)
    source = torch.complex(
        torch.randn((2, 4, 3, *compact), generator=generator),
        torch.randn((2, 4, 3, *compact), generator=generator))

    runtime = QcuRuntime.from_lattice(
        lattice_shape, mass=0.17, csw=0.8, atol=1.0e-7, max_iter=200,
        device="cpu", plan=1, grid=(1, 1, 1, 1), qcu_module=qcu)
    ce, cei = runtime.make_clover(gauge, with_inverse=True, parity=0)
    co, coi = runtime.make_clover(gauge, with_inverse=True, parity=1)
    runtime.configure(plan=1)
    solution = torch.empty_like(source)
    with runtime.operation("applyCloverBistabCgQcu") as operation:
        operation(solution, source, gauge, ce, co, cei, coi)
        iterations = operation.read("getCloverBistabCgIterationsQcu")
    assert 0 < iterations < 200

    full_source = tools.poooxyzt2oooxyzt(source)
    full_solution = tools.poooxyzt2oooxyzt(solution)
    clover = dslash.make_clover(
        full_gauge, kappa=torch.tensor([runtime.kappa]), csw=0.8)
    operator = dslash.operator(
        U=full_gauge, clover_term=clover, kappa=torch.tensor([runtime.kappa]),
        support_parity=True, verbose=False)
    expected_rhs = operator.give_b_parity(
        source[0].reshape(12, *compact), source[1].reshape(12, *compact))
    compact_rhs = torch.empty_like(source[0])
    with runtime.operation("applyCloverBistabCgPrepareQcu") as operation:
        status = operation(compact_rhs, source, gauge, ce, co, cei, coi)
    assert status == 0
    assert torch.allclose(
        compact_rhs.reshape(12, *compact), expected_rhs,
        rtol=5.0e-4, atol=5.0e-5)

    expected_even = operator.give_x_e(
        source[0].reshape(12, *compact), solution[1].reshape(12, *compact))
    reconstructed = torch.empty_like(source)
    with runtime.operation("applyCloverBistabCgReconstructQcu") as operation:
        status = operation(
            reconstructed, source, solution[1], gauge, ce, co, cei, coi)
    assert status == 0
    assert torch.allclose(
        reconstructed[0].reshape(12, *compact), expected_even,
        rtol=5.0e-4, atol=5.0e-5)

    residual = (
        dslash.give_wilson(
            full_solution, full_gauge, torch.tensor([runtime.kappa]),
            with_I=True)
        + dslash.give_clover(full_solution, clover)
        - full_source)
    relative_residual = float(
        torch.linalg.vector_norm(residual)
        / torch.linalg.vector_norm(full_source))
    assert relative_residual < 2.0e-4


@pytest.mark.skipif(define.size != 1, reason="Python binding smoke uses one MPI rank")
def test_kupl_python_wilson_solvers_and_schur() -> None:
    qcu = _kupl_qcu()
    lattice_shape = (4, 4, 4, 4)
    compact = (*lattice_shape[:3], lattice_shape[3] // 2)
    full_gauge = torch.empty((3, 3, 4, *lattice_shape), dtype=torch.complex64)
    lattice.generate_gauge_field(full_gauge, sigma=0.1, seed=47, verbose=False)
    gauge = tools.oooxyzt2poooxyzt(full_gauge).contiguous()
    generator = torch.Generator().manual_seed(53)
    source = torch.complex(
        torch.randn((2, 4, 3, *compact), generator=generator),
        torch.randn((2, 4, 3, *compact), generator=generator))

    runtime = QcuRuntime.from_lattice(
        lattice_shape, mass=0.17, atol=1.0e-7, max_iter=200,
        device="cpu", plan=1, grid=(1, 1, 1, 1), qcu_module=qcu)
    solution = torch.empty_like(source)
    with runtime.operation("applyWilsonBistabCgQcu") as operation:
        operation(solution, source, gauge)
        bistab_iterations = operation.read(
            "getCloverBistabCgIterationsQcu")
    assert 0 < bistab_iterations < 200

    full_source = tools.poooxyzt2oooxyzt(source)
    full_solution = tools.poooxyzt2oooxyzt(solution)
    residual = dslash.give_wilson(
        full_solution, full_gauge, torch.tensor([runtime.kappa]),
        with_I=True) - full_source
    relative_residual = float(
        torch.linalg.vector_norm(residual)
        / torch.linalg.vector_norm(full_source))
    assert relative_residual < 2.0e-4

    cg_solution = torch.empty_like(source)
    with runtime.operation("applyWilsonCgQcu") as operation:
        operation(cg_solution, source, gauge)
        cg_iterations = operation.read("getCloverBistabCgIterationsQcu")
    assert 0 < cg_iterations < 200
    cg_residual = dslash.give_wilson(
        tools.poooxyzt2oooxyzt(cg_solution), full_gauge,
        torch.tensor([runtime.kappa]), with_I=True) - full_source
    assert float(
        torch.linalg.vector_norm(cg_residual)
        / torch.linalg.vector_norm(full_source)) < 2.0e-4

    schur_input = source[1]
    expected_first = qcu_wilson_dslash_reference(schur_input, gauge, parity=1)
    expected_schur = schur_input - runtime.kappa ** 2 * (
        qcu_wilson_dslash_reference(expected_first, gauge, parity=0))
    schur_output = torch.empty_like(schur_input)
    runtime.call("applyWilsonBistabCgDslashQcu", schur_output, schur_input, gauge)
    assert torch.allclose(schur_output, expected_schur, rtol=2.0e-4, atol=2.0e-5)


@pytest.mark.skipif(define.size != 1, reason="Python binding smoke uses one MPI rank")
def test_kupl_python_laplacian() -> None:
    qcu = _kupl_qcu()
    shape = (3, 2, 2)
    params = torch.zeros(define._PARAMS_SIZE_, dtype=torch.int32)
    params[define._LAT_X_] = shape[0]
    params[define._LAT_Y_] = shape[1]
    params[define._LAT_Z_] = shape[2]
    params[define._LAT_T_] = 1
    params[define._LAT_XYZT_] = int(torch.tensor(shape).prod())
    params[define._GRID_X_] = 1
    params[define._GRID_Y_] = 1
    params[define._GRID_Z_] = 1
    params[define._GRID_T_] = 1
    params[define._NODE_RANK_] = 0
    params[define._NODE_SIZE_] = 1
    params[define._DATA_TYPE_] = define._LAT_C64_
    params[define._SET_PLAN_] = define._SET_PLAN_N_2_
    argv = define.make_argv(0.1, csw=1.0)
    runtime = QcuRuntime(
        params=params, argv=argv,
        set_ptrs=torch.zeros(define._SET_PTRS_SIZE_, dtype=torch.int64),
        device=torch.device("cpu"), lattice=(*shape, 1), mass=0.1, csw=1.0,
        qcu_module=qcu)

    gauge = torch.zeros((3, 3, 3, *shape), dtype=torch.complex64)
    eye = torch.eye(3, dtype=torch.complex64)
    gauge[:, :, :] = eye.reshape(3, 3, 1, 1, 1, 1)
    generator = torch.Generator().manual_seed(59)
    source = torch.complex(
        torch.randn((3, *shape), generator=generator),
        torch.randn((3, *shape), generator=generator))
    expected = 6.0 * source
    for direction, dim in enumerate((1, 2, 3)):
        expected -= torch.roll(source, shifts=1, dims=dim)
        expected -= torch.roll(source, shifts=-1, dims=dim)
    output = torch.empty_like(source)
    runtime.call("applyLaplacianQcu", output, source, gauge)
    assert torch.allclose(output, expected, rtol=2.0e-5, atol=2.0e-6)
