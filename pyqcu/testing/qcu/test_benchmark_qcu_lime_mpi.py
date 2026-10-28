"""CPU-only layout tests for benchmark_qcu_lime_mpi."""

from types import SimpleNamespace
from unittest.mock import patch

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
    from pyqcu.torch import dslash

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
    # The residual helper receives the compact parity representation and
    # gathers complete fields inside MPI.  This CPU-only test uses one rank,
    # so an identity gather exercises the same A=I+T stripping logic.
    solution_eo = torch.from_numpy(benchmark._parity_compress(solution))
    gauge_eo = torch.from_numpy(benchmark._parity_compress(gauge))
    clover_applied = dslash.add_I(clover)
    clover_eo = torch.from_numpy(benchmark._parity_compress(clover_applied))
    rhs = (
        dslash.give_wilson(solution, gauge, kappa, u0, True)
        + dslash.give_clover(solution, clover)
    )
    rhs_eo = torch.from_numpy(benchmark._parity_compress(rhs))
    broadcast_calls = []

    class FakeComm:
        @staticmethod
        def bcast(value, root=0):
            broadcast_calls.append((value, root))
            return value

    def passthrough(value, process_grid, root=0, comm=None):
        assert comm is not None
        return value

    with patch.object(benchmark, "_gather_local_blocks",
                      side_effect=passthrough):
        residual = benchmark._independent_true_residual(
            torch=torch, solution_eo=solution_eo, rhs_eo=rhs_eo,
            gauge_eo=gauge_eo, clover_even=clover_eo[0],
            clover_odd=clover_eo[1], mass=0.05, process_grid=(1, 1, 1, 1),
            comm=FakeComm(),
        )

    assert residual < 2.0e-6, residual
    assert len(broadcast_calls) == 1 and broadcast_calls[0][1] == 0


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


def test_device_record_uses_hip_uuid_when_torch_omits_it() -> None:
    properties = SimpleNamespace(
        uuid=None, name="DCU", total_memory=32 << 30, major=9, minor=0)
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(
        get_device_properties=lambda device: properties))
    device = SimpleNamespace(index=0)

    with patch.object(
            benchmark, "_hip_device_uuid", return_value="hip:0123456789abcdef"):
        record = benchmark._device_record(fake_torch, device)

    assert record["uuid"] == "hip:0123456789abcdef"
    assert record["uuid_source"] == "hip"
    assert record["compute_capability"] == [9, 0]
