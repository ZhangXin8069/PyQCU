"""Mock regressions for KUPL ctypes pointer and lifecycle mapping."""

from __future__ import annotations

import ctypes

import pytest
import torch

from pyqcu.cuda import define
from pyqcu.kupl import qcu
from pyqcu.kupl.multigrid import strict_multigrid


def _controls():
    params = torch.zeros(define._PARAMS_SIZE_, dtype=torch.int32)
    argv = torch.zeros(define._ARGV_SIZE_, dtype=torch.float32)
    set_ptrs = torch.zeros(define._SET_PTRS_SIZE_, dtype=torch.int64)
    return set_ptrs, params, argv


def test_strict_init_passes_ull_pointer_and_preserves_tensor_output(monkeypatch) -> None:
    calls = []

    class FakeLibrary:
        def applyMultigridStrictInitQcu(self, *args):
            calls.append(args)
            args[3][0] = 4096
            return 0

        def applyMultigridStrictVCycleQcu(self, *args):
            calls.append(args)
            args[5][0] = 2048
            return 0

    monkeypatch.setattr(qcu, "_LIBRARY", FakeLibrary())
    set_ptrs, params, _ = _controls()
    allocated_bytes = torch.zeros(1, dtype=torch.int64)

    status = qcu.applyMultigridStrictInitQcu(
        set_ptrs, params, 1, allocated_bytes)

    assert status == 0
    assert isinstance(
        calls[0][3], ctypes.POINTER(ctypes.c_ulonglong))
    assert calls[0][3] is not allocated_bytes
    assert int(allocated_bytes[0]) == 4096

    full_out = torch.zeros(1, dtype=torch.complex64)
    full_rhs = torch.zeros_like(full_out)
    vcycle_bytes = torch.zeros(1, dtype=torch.int64)
    status = qcu.applyMultigridStrictVCycleQcu(
        full_out, full_rhs, set_ptrs, params, 1, vcycle_bytes)
    assert status == 0
    assert isinstance(calls[1][5], ctypes.POINTER(ctypes.c_ulonglong))
    assert int(vcycle_bytes[0]) == 2048


def test_strict_fgmres_wrapper_supplies_both_dof_counts(monkeypatch) -> None:
    calls = []

    class FakeLibrary:
        def applyMultigridStrictFgmresQcu(self, *args):
            calls.append(args)
            args[27][0] = 3
            args[28][0] = 1
            args[29][0] = 1.0e-8
            args[30][0] = 8192
            return 0

    monkeypatch.setattr(qcu, "_LIBRARY", FakeLibrary())
    set_ptrs, params, _ = _controls()
    fine_dims = (2, 2, 2, 4)
    coarse_dims = (1, 1, 1, 2)
    full_shape = (2, 4, 3, *fine_dims[:3], fine_dims[3] // 2)
    full_out = torch.zeros(full_shape, dtype=torch.complex64)
    full_rhs = torch.zeros_like(full_out)
    compact_shape = (*fine_dims[:3], fine_dims[3] // 2)
    gauge = torch.zeros(2, 3, 3, 4, *compact_shape, dtype=torch.complex64)
    clover = torch.zeros(4, 3, 4, 3, *compact_shape, dtype=torch.complex64)
    null_vectors = torch.zeros(
        3, 12, 1, 2, 1, 2, 1, 2, 2, 2, dtype=torch.complex64)
    iterations = torch.zeros(1, dtype=torch.int32)
    converged = torch.zeros(1, dtype=torch.int32)
    residual = torch.zeros(1, dtype=torch.float64)
    allocated = torch.zeros(1, dtype=torch.int64)

    status = qcu.applyMultigridStrictFgmresQcu(
        full_out, full_rhs, gauge, clover, clover, clover, clover,
        null_vectors, set_ptrs, params, fine_dims, coarse_dims, 8, 3, 10,
        1.0e-8, 1, 1, 512 << 20, iterations, converged, residual, allocated)

    assert status == 0
    call = calls[0]
    assert call[10] == 12
    assert call[11:15] == fine_dims
    assert call[15] == 3
    assert call[16:20] == coarse_dims
    assert call[20] == 8
    assert isinstance(call[27], ctypes.POINTER(ctypes.c_int))
    assert isinstance(call[28], ctypes.POINTER(ctypes.c_int))
    assert isinstance(call[29], ctypes.POINTER(ctypes.c_double))
    assert isinstance(call[30], ctypes.POINTER(ctypes.c_ulonglong))
    assert int(iterations[0]) == 3
    assert int(converged[0]) == 1
    assert float(residual[0]) == pytest.approx(1.0e-8)
    assert int(allocated[0]) == 8192


def test_kupl_pointer_rejects_non_cpu_tensor(monkeypatch) -> None:
    class FakeLibrary:
        pass

    monkeypatch.setattr(qcu, "_LIBRARY", FakeLibrary())
    with pytest.raises(ValueError, match="CPU"):
        qcu._pointer(torch.empty(1, device="meta"))


def test_strict_setup_failure_releases_strict_and_lattice_resources(
        monkeypatch) -> None:
    calls = []

    monkeypatch.setattr(
        strict_multigrid.qcu, "applyInitQcu",
        lambda *args: calls.append("init"))
    monkeypatch.setattr(
        strict_multigrid.qcu, "applyMultigridStrictInitQcu",
        lambda *args: (calls.append("strict-init") or 1))
    monkeypatch.setattr(
        strict_multigrid.qcu, "applyMultigridStrictEndQcu",
        lambda *args: (calls.append("strict-end") or 0))
    monkeypatch.setattr(
        strict_multigrid.qcu, "applyEndQcu",
        lambda *args: calls.append("end"))

    set_ptrs, params, argv = _controls()
    asset = torch.zeros(1, dtype=torch.complex64)
    solver = strict_multigrid.StrictMultigrid(
        params, argv, set_ptrs, asset, asset, asset, asset, asset, asset,
        (2, 2, 2, 4), (1, 1, 1, 2))

    with pytest.raises(RuntimeError, match="hierarchy init"):
        solver.setup()

    assert calls == ["init", "strict-init", "strict-end", "end"]
    assert not solver._setup_done


def test_strict_solve_fails_closed_on_nonconvergence(monkeypatch) -> None:
    monkeypatch.setattr(strict_multigrid.qcu, "applyInitQcu", lambda *args: None)
    monkeypatch.setattr(
        strict_multigrid.qcu, "applyMultigridStrictInitQcu",
        lambda *args: 0)
    monkeypatch.setattr(
        strict_multigrid.qcu, "applyMultigridStrictEndQcu",
        lambda *args: 0)
    monkeypatch.setattr(strict_multigrid.qcu, "applyEndQcu", lambda *args: None)

    observed = {}

    def fake_fgmres(*args):
        observed["null_vectors"] = args[7]
        observed["fine_dims"] = args[10]
        observed["coarse_dims"] = args[11]
        args[-4][0] = 10
        args[-3][0] = 0
        args[-2][0] = 2.0
        args[-1][0] = 16
        return 0

    monkeypatch.setattr(
        strict_multigrid.qcu, "applyMultigridStrictFgmresQcu", fake_fgmres)
    set_ptrs, params, argv = _controls()
    fine_dims = (2, 2, 2, 4)
    coarse_dims = (1, 1, 1, 2)
    asset = torch.zeros(1, dtype=torch.complex64)
    null_vectors = torch.zeros(
        2, 12, 1, 2, 1, 2, 1, 2, 1, 2, dtype=torch.complex64)
    solver = strict_multigrid.StrictMultigrid(
        params, argv, set_ptrs, asset, asset, asset, asset, asset,
        null_vectors, fine_dims, coarse_dims)
    solver.setup()

    with pytest.raises(RuntimeError, match="未收敛"):
        solver.solve(torch.zeros(12, *fine_dims, dtype=torch.complex64))

    assert observed["null_vectors"] is null_vectors
    assert observed["fine_dims"] == fine_dims
    assert observed["coarse_dims"] == coarse_dims
    solver.close()


def test_strict_solve_maps_cold_and_warm_start_flags(monkeypatch) -> None:
    monkeypatch.setattr(strict_multigrid.qcu, "applyInitQcu", lambda *args: None)
    monkeypatch.setattr(
        strict_multigrid.qcu, "applyMultigridStrictInitQcu",
        lambda *args: 0)
    monkeypatch.setattr(
        strict_multigrid.qcu, "applyMultigridStrictEndQcu",
        lambda *args: 0)
    monkeypatch.setattr(strict_multigrid.qcu, "applyEndQcu", lambda *args: None)
    observed_flags = []

    def fake_fgmres(*args):
        observed_flags.append(int(args[9][define._MG_USE_INIT_GUESS_]))
        args[-4][0] = 2
        args[-3][0] = 1
        args[-2][0] = 1.0e-8
        args[-1][0] = 32
        return 0

    monkeypatch.setattr(
        strict_multigrid.qcu, "applyMultigridStrictFgmresQcu", fake_fgmres)
    set_ptrs, params, argv = _controls()
    fine_dims = (2, 2, 2, 4)
    asset = torch.zeros(1, dtype=torch.complex64)
    solver = strict_multigrid.StrictMultigrid(
        params, argv, set_ptrs, asset, asset, asset, asset, asset, asset,
        fine_dims, (1, 1, 1, 2))
    solver.setup()
    rhs = torch.zeros(12, *fine_dims, dtype=torch.complex64)

    solver.solve(rhs)
    initial = torch.ones_like(rhs)
    warm_output = solver.solve(rhs, x0=initial)

    assert observed_flags == [0, 1]
    assert torch.equal(
        warm_output,
        initial.reshape(2, 4, 3, *fine_dims[:3], fine_dims[3] // 2))
    solver.close()
