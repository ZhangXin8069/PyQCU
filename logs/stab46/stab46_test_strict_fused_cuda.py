"""CUDA regression for the fused strict fine-grid right-FGMRES backend."""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("需要 CUDA 设备", allow_module_level=True)

try:
    from pyqcu.cuda import CudaStrictMultigridSolver, define, qcu
    from pyqcu.solver import QudaStrictMultigrid
except (ImportError, OSError) as exc:  # pragma: no cover - environment
    pytest.skip(f"QCU Cython/CUDA 后端不可用: {exc}", allow_module_level=True)


DTYPE = torch.complex64
SHAPE = (4, 4, 4, 4)
BLOCK = (2, 2, 2, 2)


def _params():
    params = define.params.clone()
    params[define._LAT_X_] = SHAPE[0]
    params[define._LAT_Y_] = SHAPE[1]
    params[define._LAT_Z_] = SHAPE[2]
    params[define._LAT_T_] = SHAPE[3]
    params[define._LAT_XYZT_] = int(torch.tensor(SHAPE).prod())
    params[define._GRID_X_] = params[define._GRID_Y_] = 1
    params[define._GRID_Z_] = params[define._GRID_T_] = 1
    params[define._NODE_RANK_] = 0
    params[define._NODE_SIZE_] = 1
    params[define._PARITY_] = 1
    params[define._DATA_TYPE_] = define._LAT_C64_
    params[define._SET_INDEX_] = 0
    params[define._SET_PLAN_] = 1
    params[define._VERBOSE_] = 0
    params[define._MG_USE_INIT_GUESS_] = 0
    argv = define.argv.clone()
    argv[define._MASS_] = 0.1
    argv[define._ATOL_] = 1.0e-7
    return params, argv


def _identity_fine_assets(device):
    compact = (*SHAPE[:3], SHAPE[3] // 2)
    gauge = torch.zeros(2, 3, 3, 4, *compact, dtype=DTYPE)
    for parity in range(2):
        for color in range(3):
            gauge[parity, color, color] = 1.0
    clover = torch.zeros(4, 3, 4, 3, *compact, dtype=DTYPE)
    for spin in range(4):
        for color in range(3):
            clover[spin, color, spin, color] = 1.0
    gauge = gauge.to(device)
    clover = clover.to(device)
    return gauge, clover, clover.clone(), clover.clone(), clover.clone()


def _hierarchy():
    diagonal = torch.eye(12, dtype=DTYPE).reshape(
        12, 12, 1, 1, 1, 1).expand(12, 12, *SHAPE).clone() * 1.5
    torch.manual_seed(20260871)
    null = torch.randn(2, 4, 3, *SHAPE, dtype=DTYPE)
    return QudaStrictMultigrid(
        fine_matvec=lambda value: 1.5 * value,
        fine_adjoint=lambda value: 1.5 * value,
        fine_diagonal=diagonal,
        lat_size=SHAPE,
        null_vectors=[null],
        dof_list=[12, 4],
        block_size=BLOCK,
        max_level=2,
        materialize_coarse=True,
        use_parity=True,
        target_parity=1,
        nu_pre=1,
        nu_post=1,
        coarse_max_iter=40,
        coarse_tol=1.0e-6,
        restart=3,
        max_iter=30,
        tol=1.0e-6,
        setup_iters=0,
        verbose=False,
    )


def _relative(left, right):
    denominator = torch.linalg.norm(right).item()
    return float(torch.linalg.norm(left - right).item() /
                 max(denominator, 1.0e-30))


@pytest.fixture
def strict_solver():
    device = torch.device("cuda")
    hierarchy = _hierarchy()
    hierarchy.setup()
    params, argv = _params()
    assets = _identity_fine_assets(device)
    vector_bytes = 12 * 4**4 // 2 * 8
    first_coarse = hierarchy.operators[1]
    coarse_elements = int(first_coarse.dof)
    for extent in first_coarse.shape:
        coarse_elements *= int(extent)
    budget = 11 * vector_bytes + 2 * coarse_elements * 8
    solver = CudaStrictMultigridSolver(
        hierarchy, argv, *assets, params, restart=3,
        max_krylov_bytes=budget, verbose=False)
    try:
        yield solver, assets
    finally:
        solver.close()
        torch.cuda.synchronize()


def _fused_call(solver, assets, out, rhs, budget, restart=3, max_iter=30):
    return qcu.applyMultigridStrictFgmresQcu(
        out, rhs, *assets, solver.fine_null_vectors,
        solver.set_ptrs, solver.params, restart, max_iter, 1.0e-6,
        solver.nu_pre, solver.nu_post, budget)


def test_strict_fused_solver_api_reuses_arena_and_warm_x0(
        strict_solver):
    solver, assets = strict_solver
    torch.manual_seed(20260872)
    rhs = torch.randn(
        2, 4, 3, *SHAPE[:3], SHAPE[3] // 2, dtype=DTYPE).cuda()

    api_solution = solver.solve(rhs).clone()
    assert solver.converged
    assert 0 < solver.iterations <= 30
    assert solver.last_restart == 3
    restart = 3
    fine_n = rhs.numel() // 2
    coarse_n = solver.fine_null_vectors.shape[0]
    for axis in (2, 4, 6, 8):
        coarse_n *= solver.fine_null_vectors.shape[axis]
    expected_bytes = int(
        ((2 * restart + 5) * fine_n + 2 * coarse_n) * rhs.element_size())
    assert solver.memory_report()[
        "fused_workspace_resident_bytes"] == expected_bytes

    solver.params[define._MG_USE_INIT_GUESS_] = 0
    fused_solution = torch.empty_like(rhs)
    result = _fused_call(
        solver, assets, fused_solution, rhs, expected_bytes, restart=restart)
    assert result["converged"]
    assert 0 < result["iterations"] <= 30
    assert result["allocated_bytes"] == expected_bytes
    assert _relative(fused_solution, api_solution) < 8.0e-5

    prepared = torch.empty_like(rhs[1])
    image = torch.empty_like(prepared)
    qcu.applyCloverBistabCgPrepareQcu(
        prepared, rhs, *assets, solver.set_ptrs, solver.params)
    qcu.applyCloverBistabCgDslashQcu(
        image, fused_solution[1], *assets, solver.set_ptrs, solver.params)
    torch.cuda.synchronize()
    true_residual = float(torch.linalg.norm(prepared - image))
    assert true_residual <= 1.2e-6 * float(torch.linalg.norm(prepared))
    assert abs(result["final_true_residual"] - true_residual) <= max(
        2.0e-6, 2.0e-4 * true_residual)

    # Same slot-80 configuration must reuse the C++ allocation and remain
    # deterministic; the fused entry performs no torch allocation.
    repeated = torch.empty_like(rhs)
    allocated_before = torch.cuda.memory_allocated()
    repeated_result = _fused_call(
        solver, assets, repeated, rhs, expected_bytes, restart=restart)
    torch.cuda.synchronize()
    assert repeated_result["allocated_bytes"] == expected_bytes
    assert torch.cuda.memory_allocated() == allocated_before
    assert _relative(repeated, fused_solution) == 0.0

    # Warm start consumes the prefilled odd component.  A converged x0 needs
    # no Arnoldi iteration, but still receives a fresh true-residual check and
    # complete even reconstruction.
    warm = torch.empty_like(rhs)
    solver.solve(rhs, x0=fused_solution, out=warm)
    assert solver.converged
    assert solver.iterations == 0
    assert solver.memory_report()[
        "fused_workspace_resident_bytes"] == expected_bytes
    assert _relative(warm, fused_solution) < 3.0e-6


def test_strict_fused_fgmres_rejects_budget_restart_shape_and_dtype(
        strict_solver):
    solver, assets = strict_solver
    rhs = torch.randn(
        2, 4, 3, *SHAPE[:3], SHAPE[3] // 2, dtype=DTYPE).cuda()
    out = torch.empty_like(rhs)
    fine_n = rhs.numel() // 2
    coarse_n = int(solver.fine_null_vectors.shape[0])
    for axis in (2, 4, 6, 8):
        coarse_n *= int(solver.fine_null_vectors.shape[axis])
    required = ((2 * 3 + 5) * fine_n + 2 * coarse_n) * rhs.element_size()

    with pytest.raises(MemoryError, match="budget is insufficient"):
        _fused_call(solver, assets, out, rhs, required - 1)
    with pytest.raises(ValueError, match="restart"):
        _fused_call(solver, assets, out, rhs, required, restart=4, max_iter=3)
    with pytest.raises(ValueError, match="shapes must match"):
        _fused_call(solver, assets, out[0], rhs, required)

    mismatched_params = solver.params.clone()
    mismatched_params[define._DATA_TYPE_] = define._LAT_C128_
    with pytest.raises(RuntimeError, match="FgmresQcu failed"):
        qcu.applyMultigridStrictFgmresQcu(
            out, rhs, *assets, solver.fine_null_vectors,
            solver.set_ptrs, mismatched_params, 3, 30, 1.0e-6,
            solver.nu_pre, solver.nu_post, required)


def test_strict_fused_fgmres_complex128_dispatch_and_residual():
    """Double dispatch must use double argv/assets and return a true norm."""
    device = torch.device("cuda")
    hierarchy = _hierarchy()
    params, argv = _params()
    params[define._DATA_TYPE_] = define._LAT_C128_
    argv = argv.to(torch.float64)
    assets = tuple(
        value.to(torch.complex128)
        for value in _identity_fine_assets(device))
    vector_bytes = 12 * 4**4 // 2 * 16
    solver = CudaStrictMultigridSolver(
        hierarchy, argv, *assets, params, restart=2,
        max_krylov_bytes=9 * vector_bytes, verbose=False)
    try:
        rhs = torch.randn(
            2, 4, 3, *SHAPE[:3], SHAPE[3] // 2,
            dtype=torch.complex128).cuda()
        out = torch.empty_like(rhs)
        fine_n = rhs.numel() // 2
        coarse_n = int(solver.fine_null_vectors.shape[0])
        for axis in (2, 4, 6, 8):
            coarse_n *= int(solver.fine_null_vectors.shape[axis])
        expected = ((2 * 2 + 5) * fine_n + 2 * coarse_n) * 16
        result = qcu.applyMultigridStrictFgmresQcu(
            out, rhs, *assets, solver.fine_null_vectors,
            solver.set_ptrs, solver.params, 2, 30, 1.0e-6,
            solver.nu_pre, solver.nu_post, expected)
        prepared = torch.empty_like(rhs[1])
        image = torch.empty_like(prepared)
        qcu.applyCloverBistabCgPrepareQcu(
            prepared, rhs, *assets, solver.set_ptrs, solver.params)
        qcu.applyCloverBistabCgDslashQcu(
            image, out[1], *assets, solver.set_ptrs, solver.params)
        torch.cuda.synchronize()
        measured = float(torch.linalg.norm(prepared - image))
        assert result["converged"]
        assert result["allocated_bytes"] == expected
        assert abs(result["final_true_residual"] - measured) <= max(
            1.0e-12, 2.0e-10 * measured)
    finally:
        solver.close()
        torch.cuda.synchronize()
