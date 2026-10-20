"""CPU-only layout tests for benchmark_qcu_lime_mpi."""

from types import SimpleNamespace

import numpy as np
import torch

from pyqcu.testing.qcu import benchmark_qcu_lime_mpi as benchmark


def test_parity_layout_roundtrip() -> None:
    rng = np.random.default_rng(7)
    full = (
        rng.standard_normal((2, 4, 3, 4, 4, 8))
        + 1j * rng.standard_normal((2, 4, 3, 4, 4, 8))
    ).astype(np.complex64)
    packed = benchmark._parity_compress(full)
    assert packed.shape == (2, 2, 4, 3, 4, 4, 4)
    np.testing.assert_array_equal(benchmark._parity_expand(packed), full)


def test_global_rhs_uses_cross_backend_float64_stream() -> None:
    lattice = (4, 4, 4, 4)
    actual = benchmark._generate_global_rhs(
        lattice, np.dtype(np.complex64), 17)
    rng = np.random.default_rng(17)
    expected = (
        rng.standard_normal((4, 3, *lattice))
        + 1j * rng.standard_normal((4, 3, *lattice))
    ).astype(np.complex64)
    np.testing.assert_array_equal(actual, expected)


def test_independent_true_residual_accepts_manufactured_solution() -> None:
    from pyqcu import dslash, tools

    lattice = (4, 4, 4, 4)
    gauge = torch.broadcast_to(
        torch.eye(3, dtype=torch.complex64)[:, :, None, None, None, None, None],
        (3, 3, 4, *lattice),
    ).clone()
    solution = (
        torch.randn((4, 3, *lattice), dtype=torch.float32)
        + 1j * torch.randn((4, 3, *lattice), dtype=torch.float32)
    ).to(torch.complex64)
    kappa = torch.tensor([1.0 / 8.1], dtype=torch.float32)
    u0 = torch.ones([1], dtype=torch.float32)
    clover = dslash.make_clover(gauge, kappa=kappa, u_0=u0, csw=1.0)
    gauge_eo = tools.oooxyzt2poooxyzt(gauge)
    solution_eo = tools.oooxyzt2poooxyzt(solution)
    clover_term_eo = tools.oooxyzt2poooxyzt(clover)
    eye = torch.eye(12, dtype=torch.complex64).reshape(
        4, 3, 4, 3, 1, 1, 1, 1)
    clover_eo = (clover_term_eo[0] + eye, clover_term_eo[1] + eye)
    inverse_eo = tuple(dslash.inverse(item) for item in clover_eo)
    hopping_eo = (
        -dslash.give_wilson_eo(
            solution_eo[1], gauge_eo, torch.tensor([1.0]), u0),
        -dslash.give_wilson_oe(
            solution_eo[0], gauge_eo, torch.tensor([1.0]), u0),
    )
    transported_eo = tuple(
        benchmark._apply_onsite_matrix(hopping, inverse)
        for hopping, inverse in zip(hopping_eo, inverse_eo, strict=True))
    rhs_eo = tuple(
        benchmark._apply_onsite_matrix(solution_eo[parity], clover_eo[parity])
        - float(kappa.item()) * hopping_eo[parity]
        for parity in range(2))

    class FakeQcu:
        @staticmethod
        def applyInitQcu(set_ptrs, params, argv):
            set_ptrs[int(params[15])] = 1

        @staticmethod
        def applyEndQcu(set_ptrs, params):
            set_ptrs[int(params[15])] = 0

        @staticmethod
        def applyDslashQcu(out, src, gauge, inverse, set_ptrs, params):
            del src, gauge, inverse, set_ptrs
            out.copy_(transported_eo[int(params[9])])

    define = SimpleNamespace(
        _SET_PTRS_COARSE_BASE_=30, _SET_INDEX_=15, _SET_PLAN_=16,
        _SET_PLAN0_=0, _PARITY_=9)
    params = torch.zeros(58, dtype=torch.int32)
    set_ptrs = torch.zeros(100, dtype=torch.int64)
    comm = SimpleNamespace(allreduce=lambda value, op=None: value)
    residual = benchmark._independent_true_residual(
        comm=comm, torch=torch, qcu=FakeQcu, define=define, params=params,
        argv=torch.zeros(8), set_ptrs=set_ptrs, device=torch.device("cpu"),
        solution_eo=solution_eo, rhs_eo=rhs_eo, gauge_eo=gauge_eo,
        clover_even=clover_eo[0], clover_odd=clover_eo[1],
        clover_even_inv=inverse_eo[0], clover_odd_inv=inverse_eo[1],
        mass=0.05,
    )
    assert residual < 2.0e-6
    assert int(params[define._SET_INDEX_]) == 1


def test_qdp_rank_slab_matches_canonical_slice() -> None:
    rng = np.random.default_rng(11)
    lattice = (4, 4, 4, 8)
    grid = (2, 1, 1, 2)
    qdp = (
        rng.standard_normal((4, 8, 4, 4, 4, 3, 3))
        + 1j * rng.standard_normal((4, 8, 4, 4, 4, 3, 3))
    ).astype(np.complex64)
    common = benchmark._load_common()
    canonical = common.qdp_gauge_to_qcu(qdp)

    for rank in range(4):
        local_shape, starts, _ = benchmark._local_geometry(
            lattice, grid, rank)
        actual_qdp = benchmark._slice_qdp_gauge(qdp, starts, local_shape)
        actual = common.qdp_gauge_to_qcu(actual_qdp)
        slices = [slice(None)] * 3
        slices.extend(
            slice(starts[axis], starts[axis] + local_shape[axis])
            for axis in range(4))
        expected = canonical[tuple(slices)]
        np.testing.assert_array_equal(actual, expected)


def test_lifecycle_ends_before_advancing_and_recycles_slot() -> None:
    events = []

    class FakeQcu:
        @staticmethod
        def applyInitQcu(set_ptrs, params, argv):
            events.append(("init", int(params[15])))

        @staticmethod
        def applyEndQcu(set_ptrs, params):
            events.append(("end", int(params[15])))

    define = SimpleNamespace(
        _SET_PTRS_COARSE_BASE_=30, _SET_INDEX_=15, _SET_PLAN_=16,
        _PARITY_=9)
    dummy_torch = SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda device: None))
    dummy_comm = SimpleNamespace(Barrier=lambda: events.append('barrier'))
    params = np.zeros(58, dtype=np.int32)
    params[define._SET_INDEX_] = 29
    set_ptrs = np.zeros(100, dtype=np.int64)

    result = benchmark._run_lifecycle(
        comm=dummy_comm, torch=dummy_torch, qcu=FakeQcu(), define=define,
        params=params,
        argv=np.zeros(8), set_ptrs=set_ptrs, device=None, plan=1,
        parity=0,
        operation=lambda: events.append("operation"),
        getter=lambda: events.append("getter") or 3, max_iter=10)

    assert [event[0] if isinstance(event, tuple) else event
            for event in events] == [
                "init", "barrier", "operation", "getter", "end"]
    assert events[0][1] == 29 and events[-1][1] == 29
    assert int(params[define._SET_INDEX_]) == 0
    assert result["iterations"] == 3 and result["native_converged"]
