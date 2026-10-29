"""Torch/MPI reduction and halo optimization regressions for 1/2/4 ranks."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from math import prod
import os
from statistics import median
from time import perf_counter
from typing import Any
from unittest.mock import patch

import numpy as np
from mpi4py import MPI

import pyqcu.cann as _torch
from pyqcu.api import BiCGStabControl, ResidualScope, SolveStatus
from pyqcu.solver import Krylov
from pyqcu.torch import dslash
from pyqcu.torch.dslash import TorchLinearOperator
from pyqcu.torch.solver import (
    LegacyMultigrid,
    StrictMultigrid,
    TorchBiCGStabDriver,
)
from pyqcu.torch.tools import _linalg
from pyqcu.torch.tools import _mpi_roll, norm_many, vdot_many


def _process_grid(size: int) -> tuple[int, int, int, int]:
    try:
        return {
            1: (1, 1, 1, 1),
            2: (2, 1, 1, 1),
            4: (2, 2, 1, 1),
        }[int(size)]
    except KeyError as exc:
        raise RuntimeError("run this regression with one, two, or four ranks") from exc


def _rank_coordinate(
        rank: int, process_grid: tuple[int, int, int, int]
        ) -> tuple[int, int, int, int]:
    remainder = int(rank)
    coordinate = []
    for axis in range(4):
        stride = prod(process_grid[axis + 1:])
        coordinate.append(remainder // stride)
        remainder %= stride
    return tuple(coordinate)  # type: ignore[return-value]


def _global_phase_gauge(
        local_shape: tuple[int, int, int, int],
        process_grid: tuple[int, int, int, int], rank: int,
        dtype: Any = _torch.complex128,
        device: Any = _torch.device("cpu")) -> Any:
    coordinate = _rank_coordinate(rank, process_grid)
    origin = tuple(coordinate[axis] * local_shape[axis] for axis in range(4))
    axes = [
        _torch.torch.arange(local_shape[axis], dtype=_torch.float64,
                            device=device) + origin[axis]
        for axis in range(4)
    ]
    x, y, z, t = _torch.torch.meshgrid(*axes, indexing="ij")
    base = x + 2.0 * y + 3.0 * z + 5.0 * t
    gauge = _torch.zeros(
        (3, 3, 4, *local_shape), dtype=dtype, device=device)
    for direction in range(4):
        phases = [
            0.037 * _torch.torch.remainder(
                base + 2.0 * direction + color, 13.0)
            for color in range(3)
        ]
        phases[2] = -(phases[0] + phases[1])
        for color, phase in enumerate(phases):
            gauge[color, color, direction] = (
                _torch.torch.cos(phase) + 1j * _torch.torch.sin(phase))
    return gauge


def _global_phase_field(
        dof: int, local_shape: tuple[int, int, int, int],
        process_grid: tuple[int, int, int, int], rank: int,
        offset: float = 0.0, dtype: Any = _torch.complex128,
        device: Any = _torch.device("cpu")) -> Any:
    coordinate = _rank_coordinate(rank, process_grid)
    origin = tuple(coordinate[axis] * local_shape[axis] for axis in range(4))
    axes = [
        _torch.torch.arange(local_shape[axis], dtype=_torch.float64,
                            device=device) + origin[axis]
        for axis in range(4)
    ]
    x, y, z, t = _torch.torch.meshgrid(*axes, indexing="ij")
    index = _torch.torch.arange(
        dof, dtype=_torch.float64, device=device).reshape(
            dof, 1, 1, 1, 1)
    phase = 0.071 * (
        index + 3.0 * x + 5.0 * y + 7.0 * z + 11.0 * t + float(offset))
    return (_torch.torch.cos(phase) + 1j * _torch.torch.sin(phase)).to(
        dtype=dtype)


def _assemble_global(value: Any, process_grid: tuple[int, int, int, int],
                     comm: Any = MPI.COMM_WORLD) -> Any:
    arrays = comm.allgather(
        np.ascontiguousarray(value.detach().cpu().numpy()))

    def build(prefix: tuple[int, ...], axis: int) -> np.ndarray:
        if axis == 4:
            rank = 0
            for extent, coordinate in zip(process_grid, prefix):
                rank = rank * extent + coordinate
            return arrays[rank]
        return np.concatenate(
            [build((*prefix, index), axis + 1)
             for index in range(process_grid[axis])],
            axis=value.ndim - 4 + axis)

    return _torch.as_tensor(
        build((), 0), dtype=value.dtype, device=value.device)


def _periodic_operator(gauge: Any, kappa: Any, u_0: Any) -> Any:
    """Build a single-process full-periodic action without another MPI halo."""
    operator = dslash.operator(
        U=None, clover_term=None, kappa=kappa, u_0=u_0, verbose=False)
    hopper = operator.hopping
    hopper.U = gauge
    hopper.grid_size = [1, 1, 1, 1]
    hopper.M_plus_list = [
        dslash.give_hopping_plus(
            ward=ward, U=gauge, kappa=kappa, u_0=u_0, verbose=False)
        for ward in range(4)
    ]
    hopper.M_minus_list = [
        dslash.give_hopping_minus(
            ward=ward, U=gauge, U_head=None, kappa=kappa,
            u_0=u_0, verbose=False)
        for ward in range(4)
    ]
    return operator


def _independent_global_residual(
        solution: Any, rhs: Any, gauge: Any,
        process_grid: tuple[int, int, int, int], comm: Any,
        kappa: Any, u_0: Any
        ) -> tuple[float, float, Any]:
    global_solution = _assemble_global(solution, process_grid, comm)
    global_rhs = _assemble_global(rhs, process_grid, comm)
    global_gauge = _assemble_global(gauge, process_grid, comm)
    operator = _periodic_operator(global_gauge, kappa, u_0)
    image = operator.matvec(global_solution)
    residual = image - global_rhs
    absolute = float(_torch.torch.norm(residual).item())
    rhs_norm = float(_torch.torch.norm(global_rhs).item())
    return absolute, absolute / max(rhs_norm, 1.0e-30), global_solution


def _global_relative_difference(
        actual: Any, expected: Any,
        process_grid: tuple[int, int, int, int], comm: Any) -> float:
    global_actual = _assemble_global(actual, process_grid, comm)
    global_expected = _assemble_global(expected, process_grid, comm)
    numerator = float(
        _torch.torch.norm(global_actual - global_expected).item())
    denominator = float(_torch.torch.norm(global_expected).item())
    return numerator / max(denominator, 1.0e-30)


def _solution_digest(value: Any) -> str:
    array = np.ascontiguousarray(value.detach().cpu().numpy())
    return hashlib.sha256(array.tobytes()).hexdigest()


@contextmanager
def _collective_counters():
    counters = {
        "allreduce_calls": 0,
        "allreduce_scalars": 0,
        "face_staging_calls": 0,
        "face_staging_bytes": 0,
    }
    original_allreduce = _linalg._allreduce_scalars
    original_face_buffer = _mpi_roll._face_buffer

    def counted_allreduce(local_values, comm=None):
        counters["allreduce_calls"] += 1
        counters["allreduce_scalars"] += len(local_values)
        return original_allreduce(local_values, comm=comm)

    def counted_face_buffer(tensor, axis, index):
        buffer = original_face_buffer(tensor, axis, index)
        counters["face_staging_calls"] += 1
        counters["face_staging_bytes"] += int(buffer.nbytes)
        return buffer

    with patch.object(_linalg, "_allreduce_scalars", counted_allreduce), \
            patch.object(_mpi_roll, "_face_buffer", counted_face_buffer):
        yield counters


def _emit_metric(name: str, **payload: Any) -> None:
    if int(MPI.COMM_WORLD.Get_rank()) != 0:
        return
    print(
        "PYQCU_MPI_METRIC " + json.dumps(
            {"name": name, **payload}, sort_keys=True),
        flush=True)


class _CountingComm:
    def __init__(self, size: int = 2, rank: int = 0):
        self._size = int(size)
        self._rank = int(rank)
        self.allreduce_calls = 0

    def Get_size(self) -> int:
        return self._size

    def Get_rank(self) -> int:
        return self._rank

    def Allreduce(self, sendbuf, recvbuf, op=None) -> None:
        del op
        self.allreduce_calls += 1
        np.copyto(recvbuf, sendbuf)

    def Sendrecv(self, sendbuf, dest, sendtag, recvbuf, source,
                 recvtag) -> None:
        del dest, sendtag, source, recvtag
        np.copyto(recvbuf, sendbuf)


def test_batched_reductions_use_one_collective() -> None:
    old_value = os.environ.get("PYQCU_TORCH_BATCH_REDUCTIONS")
    try:
        os.environ["PYQCU_TORCH_BATCH_REDUCTIONS"] = "1"
        comm = _CountingComm()
        left = _torch.tensor(
            [1.0 + 2.0j, 3.0 - 1.0j], dtype=_torch.complex128)
        right = _torch.tensor(
            [0.5 - 1.0j, 2.0 + 3.0j], dtype=_torch.complex128)

        dots = vdot_many(((left, right), (right, right)), comm=comm)

        assert comm.allreduce_calls == 1
        assert abs(dots[0].item() - _torch.vdot(left, right).item()) < 1.0e-12
        assert abs(dots[1].item() - _torch.vdot(right, right).item()) < 1.0e-12

        comm.allreduce_calls = 0
        with _mpi_roll.distributed_roll(
                comm, (2, 1, 1, 1), local_extents=(2, 2, 2, 2)):
            norms = norm_many((left, right))
        assert comm.allreduce_calls == 1
        assert abs(norms[0] - float(_torch.norm(left))) < 1.0e-12
        assert abs(norms[1] - float(_torch.norm(right))) < 1.0e-12
    finally:
        if old_value is None:
            os.environ.pop("PYQCU_TORCH_BATCH_REDUCTIONS", None)
        else:
            os.environ["PYQCU_TORCH_BATCH_REDUCTIONS"] = old_value


def test_single_face_halo_staging_and_legacy_fallback() -> None:
    comm = _CountingComm(size=2, rank=0)
    tensor = _torch.torch.arange(
        2 * 2 * 2 * 2 * 2, dtype=_torch.float32).reshape(
        2, 2, 2, 2, 2)
    staged = []
    original = _mpi_roll._face_buffer

    def counted(tensor_arg, axis, index):
        staged.append((axis, index))
        return original(tensor_arg, axis, index)

    old_value = os.environ.get("PYQCU_TORCH_HALO_SINGLE_FACE")
    try:
        with patch.object(_mpi_roll, "_face_buffer", counted):
            with _mpi_roll.distributed_roll(
                    comm, (2, 1, 1, 1), local_extents=(2, 2, 2, 2)):
                os.environ["PYQCU_TORCH_HALO_SINGLE_FACE"] = "1"
                _mpi_roll.roll(tensor, shifts=-1, dims=1)
                assert len(staged) == 1

                staged.clear()
                os.environ["PYQCU_TORCH_HALO_SINGLE_FACE"] = "0"
                _mpi_roll.roll(tensor, shifts=-1, dims=1)
                assert len(staged) == 2
    finally:
        if old_value is None:
            os.environ.pop("PYQCU_TORCH_HALO_SINGLE_FACE", None)
        else:
            os.environ["PYQCU_TORCH_HALO_SINGLE_FACE"] = old_value


def test_distributed_roll_matches_global_periodic_reference() -> None:
    comm = MPI.COMM_WORLD
    size = int(comm.Get_size())
    rank = int(comm.Get_rank())
    process_grid = _process_grid(size)
    local_shape = (2, 2, 2, 2)
    tensor = _global_phase_field(
        2, local_shape, process_grid, rank, offset=17.0)
    global_tensor = _assemble_global(tensor, process_grid, comm)
    coordinate = _rank_coordinate(rank, process_grid)
    origin = tuple(
        coordinate[axis] * local_shape[axis] for axis in range(4))
    local_slice = tuple(
        slice(origin[axis], origin[axis] + local_shape[axis])
        for axis in range(4))
    active_faces = sum(value > 1 for value in process_grid)
    face_buffers_per_exchange = (
        1 if os.environ.get(
            "PYQCU_TORCH_HALO_SINGLE_FACE", "1").strip() != "0" else 2)

    with _collective_counters() as counters:
        with _mpi_roll.distributed_roll(
                comm, process_grid, local_extents=local_shape):
            for axis in range(4):
                for shift in (-1, 1):
                    actual = _mpi_roll.roll(
                        tensor, shifts=shift, dims=1 + axis)
                    expected = _torch.torch.roll(
                        global_tensor, shifts=shift, dims=1 + axis)
                    expected = expected[(Ellipsis, *local_slice)]
                    assert _torch.torch.equal(actual, expected), (
                        rank, axis, shift)

    assert counters["face_staging_calls"] == (
        2 * active_faces * face_buffers_per_exchange)
    gathered = comm.allgather(counters)
    assert all(item == gathered[0] for item in gathered), gathered
    _emit_metric(
        "distributed_roll", ranks=size, process_grid=process_grid,
        local_shape=local_shape, **counters)


def test_bicgstab_collective_count_and_fallback() -> None:
    class DiagonalOperator:
        residual_scope = ResidualScope.FULL
        capabilities = {"python_krylov": True}

        def apply(self, value):
            return value * _torch.torch.arange(
                2, 5, dtype=value.real.dtype, device=value.device).reshape(
                    3, 1)

    rhs = _torch.torch.ones((3, 2), dtype=_torch.complex128)

    def count_reductions(disabled: bool) -> int:
        old_value = os.environ.get("PYQCU_TORCH_BATCH_REDUCTIONS")
        comm = _CountingComm()
        try:
            os.environ["PYQCU_TORCH_BATCH_REDUCTIONS"] = (
                "0" if disabled else "1")
            with _mpi_roll.distributed_roll(
                    comm, (2, 1, 1, 1), local_extents=(2, 2, 2, 2)):
                TorchBiCGStabDriver(
                    DiagonalOperator(),
                    BiCGStabControl(rtol=1.0e-14, max_iter=1),
                ).solve_result(rhs)
        finally:
            if old_value is None:
                os.environ.pop("PYQCU_TORCH_BATCH_REDUCTIONS", None)
            else:
                os.environ["PYQCU_TORCH_BATCH_REDUCTIONS"] = old_value
        return comm.allreduce_calls

    assert count_reductions(disabled=False) == 6
    assert count_reductions(disabled=True) == 10


def test_torch_krylov_is_globally_reduced_and_rank_symmetric() -> None:
    comm = MPI.COMM_WORLD
    size = int(comm.Get_size())
    rank = int(comm.Get_rank())
    process_grid = _process_grid(size)
    os.environ["PYQCU_PROCESS_GRID"] = ",".join(
        str(value) for value in process_grid)

    local_shape = (2, 2, 2, 2)
    gauge = _global_phase_gauge(local_shape, process_grid, rank)
    kappa = _torch.tensor([0.05], dtype=_torch.float64)
    u_0 = _torch.tensor([1.0], dtype=_torch.float64)
    operator = dslash.operator(
        U=gauge,
        kappa=kappa,
        u_0=u_0,
        verbose=False,
    )
    adapted = TorchLinearOperator(operator, mode="full")
    exact = _global_phase_field(
        12, local_shape, process_grid, rank, offset=101.0)
    rhs = operator.matvec(exact)

    with _collective_counters() as counters:
        result = Krylov(
            backend="torch",
            method="bicgstab",
            operator=adapted,
            control=BiCGStabControl(
                rtol=1.0e-6,
                max_iter=100,
                true_residual_every=5,
                require_true_residual=True,
                breakdown_restarts=3,
            ),
        ).solve_result(rhs)

    assert result.status is SolveStatus.CONVERGED, (
        result.status, result.final_residual, result.krylov_iterations,
        result.diagnostics.get("recurrence_history"))
    assert result.converged is True
    independent, independent_relative, global_solution = (
        _independent_global_residual(
            result.x, rhs, gauge, process_grid, comm, kappa, u_0))
    assert independent_relative < 2.0e-6, independent_relative
    solution_relative = _global_relative_difference(
        result.x, exact, process_grid, comm)
    assert solution_relative < 1.0e-5, solution_relative
    residuals = comm.allgather(float(result.final_residual))
    iterations = comm.allgather(int(result.krylov_iterations))
    assert max(residuals) - min(residuals) <= 1.0e-12
    assert len(set(iterations)) == 1
    assert residuals[0] <= 1.0e-6 * norm_many((rhs,))[0]
    gathered_counters = comm.allgather(counters)
    assert all(item == gathered_counters[0] for item in gathered_counters)
    _emit_metric(
        "torch_krylov", ranks=size, process_grid=process_grid,
        local_shape=local_shape, iterations=iterations[0],
        native_relative=float(result.relative_true_residual),
        independent_relative=independent_relative,
        solution_relative=solution_relative,
        solution_sha256=_solution_digest(global_solution),
        **counters)


def test_torch_strict_setup_is_rank_symmetric() -> None:
    comm = MPI.COMM_WORLD
    size = int(comm.Get_size())
    rank = int(comm.Get_rank())
    process_grid = _process_grid(size)
    os.environ["PYQCU_PROCESS_GRID"] = ",".join(
        str(value) for value in process_grid)
    local_shape = (4, 4, 4, 4)
    gauge = _global_phase_gauge(local_shape, process_grid, rank)
    kappa = _torch.tensor([0.05], dtype=_torch.float64)
    u_0 = _torch.tensor([1.0], dtype=_torch.float64)
    exact = _global_phase_field(
        12, local_shape, process_grid, rank, offset=101.0)
    reference_operator = dslash.operator(
        U=gauge, clover_term=None, kappa=kappa, u_0=u_0, verbose=False)
    rhs = reference_operator.matvec(exact)
    null = exact.unsqueeze(0)
    hierarchy = StrictMultigrid(
        U=gauge,
        clover_term=None,
        kappa=kappa,
        u_0=u_0,
        lat_size=local_shape,
        null_vectors=[null],
        nvec_list=[1],
        dof_list=[12, 4],
        block_size=(2, 2, 2, 2),
        max_level=2,
        materialize_coarse=True,
        use_parity=True,
        target_parity=1,
        setup_iters=0,
        strict_galerkin_mode="colored",
        nu_pre=1,
        nu_post=1,
        coarse_max_iter=30,
        coarse_tol=1.0e-6,
        restart=10,
        max_iter=30,
        tol=1.0e-6,
        process_grid=(process_grid if size > 1 else None),
        comm=(comm if size > 1 else None),
        verbose=False,
    )
    try:
        started = perf_counter()
        with _collective_counters() as setup_counters:
            hierarchy.setup()
        setup_seconds = perf_counter() - started
        diagnostics = hierarchy.diagnostics()
        gathered = comm.allgather(diagnostics)
        assert all(item["transfer_RP"] < 2.0e-5 for item in gathered), gathered
        assert all(item["galerkin_RDP"] < 2.0e-5 for item in gathered), gathered
        assert all(
            item["transfer_block_ortho"] < 2.0e-5 for item in gathered), gathered
        assert _mpi_roll.current_context() is None

        started = perf_counter()
        with _collective_counters() as solve_counters:
            solution = hierarchy.solve(rhs)
        solve_seconds = perf_counter() - started
        independent, relative, global_solution = (
            _independent_global_residual(
                solution, rhs, gauge, process_grid, comm, kappa, u_0))
        gathered_relative = comm.allgather(relative)
        assert max(gathered_relative) - min(gathered_relative) < 1.0e-10
        assert relative < 1.0e-5, gathered_relative
        solution_relative = _global_relative_difference(
            solution, exact, process_grid, comm)
        assert solution_relative < 1.0e-5, solution_relative
        assert _mpi_roll.current_context() is None
        gathered_setup = comm.allgather(setup_counters)
        gathered_solve = comm.allgather(solve_counters)
        assert all(item == gathered_setup[0] for item in gathered_setup)
        assert all(item == gathered_solve[0] for item in gathered_solve)
        _emit_metric(
            "torch_strict", ranks=size, process_grid=process_grid,
            local_shape=local_shape,
            iterations=hierarchy.last_solve_report.get("iterations"),
            native_relative=hierarchy.last_solve_report.get(
                "native_relative_residual"),
            independent_relative=relative,
            solution_relative=solution_relative,
            setup_seconds=setup_seconds, solve_seconds=solve_seconds,
            solution_sha256=_solution_digest(global_solution),
            setup=setup_counters,
            solve=solve_counters)

        if size > 1:
            old_face_mode = os.environ.get(
                "PYQCU_TORCH_HALO_SINGLE_FACE")
            try:
                modes = {
                    "single_face": "1",
                    "two_face": "0",
                }
                for face_mode in modes.values():
                    os.environ["PYQCU_TORCH_HALO_SINGLE_FACE"] = face_mode
                    hierarchy.solve(rhs)
                elapsed = {label: [] for label in modes}
                mode_counters = {label: [] for label in modes}
                mode_solutions = {}
                # Interleave in ABBAAB order so host drift cannot be mistaken
                # for a one-sided face-staging effect.
                sequence = (
                    "single_face", "two_face", "two_face",
                    "single_face", "single_face", "two_face")
                for label in sequence:
                    os.environ["PYQCU_TORCH_HALO_SINGLE_FACE"] = modes[label]
                    started = perf_counter()
                    with _collective_counters() as counters:
                        mode_solutions[label] = hierarchy.solve(rhs)
                    elapsed[label].append(perf_counter() - started)
                    mode_counters[label].append(dict(counters))
                for label in modes:
                    gathered_mode = comm.allgather(mode_counters[label])
                    assert all(
                        item == gathered_mode[0] for item in gathered_mode)
                    mode_absolute, mode_relative, _ = (
                        _independent_global_residual(
                            mode_solutions[label], rhs, gauge,
                            process_grid, comm, kappa, u_0))
                    assert mode_relative < 1.0e-5, mode_relative
                    _emit_metric(
                        "torch_strict_halo_ab", ranks=size, mode=label,
                        process_grid=process_grid,
                        local_shape=local_shape,
                        seconds_median=median(elapsed[label]),
                        seconds_samples=elapsed[label],
                        independent_absolute=mode_absolute,
                        independent_relative=mode_relative,
                        **mode_counters[label][-1])
            finally:
                if old_face_mode is None:
                    os.environ.pop("PYQCU_TORCH_HALO_SINGLE_FACE", None)
                else:
                    os.environ["PYQCU_TORCH_HALO_SINGLE_FACE"] = old_face_mode
    finally:
        hierarchy.close()


def test_torch_legacy_solve_is_rank_symmetric() -> None:
    comm = MPI.COMM_WORLD
    size = int(comm.Get_size())
    rank = int(comm.Get_rank())
    process_grid = _process_grid(size)
    os.environ["PYQCU_PROCESS_GRID"] = ",".join(
        str(value) for value in process_grid)
    local_shape = (4, 4, 4, 4)
    gauge = _global_phase_gauge(local_shape, process_grid, rank)
    kappa = _torch.tensor([0.05], dtype=_torch.float64)
    u_0 = _torch.tensor([1.0], dtype=_torch.float64)
    solver = LegacyMultigrid(
        dtype_list=[_torch.complex128],
        device_list=[_torch.device("cpu")],
        U=gauge,
        clover_term=None,
        kappa=kappa,
        u_0=u_0,
        max_level=1,
        min_size=2,
        dof_list=[12],
        tol=1.0e-6,
        max_iter=50,
        num_restart=0,
        support_parity=True,
        seed=20261029,
        process_grid=process_grid,
        verbose=False,
    )
    assert solver.with_cuda_qcu is False
    try:
        exact = _global_phase_field(
            12, local_shape, process_grid, rank, offset=101.0)
        rhs = solver.op_list[0].matvec(exact)
        started = perf_counter()
        with _collective_counters() as counters:
            solution = solver.solve(rhs.reshape(4, 3, *local_shape))
        solve_seconds = perf_counter() - started
        independent, relative, global_solution = (
            _independent_global_residual(
                solution.reshape(12, *local_shape), rhs, gauge,
                process_grid, comm, kappa, u_0))
        gathered = comm.allgather(relative)
        assert max(gathered) - min(gathered) < 1.0e-10
        assert relative < 1.0e-5, gathered
        solution_relative = _global_relative_difference(
            solution.reshape(12, *local_shape), exact, process_grid, comm)
        assert solution_relative < 1.0e-5, solution_relative
        stop_reasons = comm.allgather(
            solver.last_solve_report.get("stop_reason"))
        iterations = comm.allgather(
            int(solver.last_solve_report.get("iterations", 0)))
        assert len(set(stop_reasons)) == 1
        assert len(set(iterations)) == 1
        gathered_counters = comm.allgather(counters)
        assert all(item == gathered_counters[0] for item in gathered_counters)
        _emit_metric(
            "torch_legacy", ranks=size, process_grid=process_grid,
            local_shape=local_shape, iterations=iterations[0],
            native_relative=solver.last_solve_report.get(
                "native_relative_residual"),
            independent_relative=relative,
            solution_relative=solution_relative,
            solve_seconds=solve_seconds,
            solution_sha256=_solution_digest(global_solution),
            **counters)
    finally:
        solver.close()


def _real_configured_benchmark() -> None:
    """Measure the same real-gauge slab with batched and scalar reductions."""
    import h5py

    comm = MPI.COMM_WORLD
    size = int(comm.Get_size())
    rank = int(comm.Get_rank())
    process_grid = _process_grid(size)
    global_shape = tuple(
        int(value) for value in os.environ.get(
            "PYQCU_TORCH_BENCH_LATTICE", "8,8,8,16").split(","))
    if len(global_shape) != 4 or any(
            global_shape[axis] % process_grid[axis]
            for axis in range(4)):
        raise ValueError(
            f"benchmark lattice {global_shape} is not divisible by "
            f"process grid {process_grid}")
    local_shape = tuple(
        global_shape[axis] // process_grid[axis] for axis in range(4))
    coordinate = _rank_coordinate(rank, process_grid)
    origin = tuple(
        coordinate[axis] * local_shape[axis] for axis in range(4))
    slices = tuple(
        slice(origin[axis], origin[axis] + local_shape[axis])
        for axis in range(4))
    gauge_path = os.environ["PYQCU_TORCH_BENCH_GAUGE"]
    source_path = os.environ["PYQCU_TORCH_BENCH_SOURCE"]
    with h5py.File(gauge_path, "r") as handle:
        gauge_np = np.ascontiguousarray(handle["data"][
            :, :, :, slices[0], slices[1], slices[2], slices[3]])
    with h5py.File(source_path, "r") as handle:
        source_np = np.ascontiguousarray(handle["data"][
            :, :, slices[0], slices[1], slices[2], slices[3]])

    requested_device = os.environ.get(
        "PYQCU_TORCH_BENCH_DEVICE", "cpu").strip().lower()
    local_rank = 0
    if requested_device == "cuda":
        if not _torch.torch.cuda.is_available():
            raise RuntimeError("CUDA benchmark requested without CUDA")
        local_comm = comm.Split_type(MPI.COMM_TYPE_SHARED, 0)
        try:
            local_rank = int(local_comm.Get_rank())
        finally:
            local_comm.Free()
        if local_rank >= int(_torch.torch.cuda.device_count()):
            raise RuntimeError(
                f"MPI local rank {local_rank} has no private GPU")
        _torch.torch.cuda.set_device(local_rank)
        device = _torch.device("cuda", local_rank)
    elif requested_device == "cpu":
        device = _torch.device("cpu")
    else:
        raise ValueError(
            f"unsupported PYQCU_TORCH_BENCH_DEVICE={requested_device}")

    os.environ["PYQCU_PROCESS_GRID"] = ",".join(
        str(value) for value in process_grid)
    mass = float(os.environ.get("PYQCU_TORCH_BENCH_MASS", "-0.277"))
    csw = float(os.environ.get("PYQCU_TORCH_BENCH_CSW", "0.0"))
    rtol = float(os.environ.get("PYQCU_TORCH_BENCH_RTOL", "1.0e-30"))
    kappa_value = 1.0 / (2.0 * mass + 8.0)
    real_dtype = (_torch.float32 if gauge_np.dtype == np.complex64
                  else _torch.float64)
    kappa = _torch.tensor([kappa_value], dtype=real_dtype, device=device)
    u_0 = _torch.tensor([1.0], dtype=real_dtype, device=device)
    gauge = _torch.as_tensor(gauge_np, device=device)
    rhs = _torch.as_tensor(source_np, device=device).reshape(
        12, *local_shape)
    clover_term = (
        dslash.make_clover(
            gauge, kappa=kappa, u_0=u_0, csw=csw, verbose=False)
        if csw != 0.0 else None)
    operator_kind = "clover-wilson" if clover_term is not None else "wilson"
    operator = dslash.operator(
        U=gauge, clover_term=clover_term, kappa=kappa, u_0=u_0, verbose=False)
    adapted = TorchLinearOperator(operator, mode="full")
    max_iter = int(os.environ.get("PYQCU_TORCH_BENCH_ITERATIONS", "5"))
    warmups = int(os.environ.get("PYQCU_TORCH_BENCH_WARMUPS", "1"))
    repeats = int(os.environ.get("PYQCU_TORCH_BENCH_REPEATS", "3"))
    if max_iter <= 0 or warmups < 0 or repeats <= 0:
        raise ValueError("benchmark iteration/warmup/repeat counts are invalid")

    if device.type == "cuda":
        properties = _torch.torch.cuda.get_device_properties(device)
        device_name = str(properties.name)
        device_uuid = str(getattr(properties, "uuid", ""))
    else:
        device_name = "cpu"
        device_uuid = ""
    device_records = comm.allgather({
        "rank": rank,
        "local_rank": local_rank,
        "device": device_name,
        "device_uuid": device_uuid,
    })
    _emit_metric(
        "torch_real_benchmark_start", ranks=size,
        process_grid=process_grid, global_shape=global_shape,
        local_shape=local_shape, device=device_name,
        device_uuid=device_uuid, gauge_path=gauge_path,
        source_path=source_path, max_iter=max_iter,
        mass=mass, kappa=kappa_value, csw=csw, rtol=rtol,
        operator=operator_kind, main_function=(
            "pyqcu.solver.Krylov(backend=torch, method=bicgstab)"),
        warmups=warmups, repeats=repeats,
        device_records=device_records)

    old_batch = os.environ.get("PYQCU_TORCH_BATCH_REDUCTIONS")
    try:
        for label, batch_value in (("batched", "1"), ("scalar", "0")):
            os.environ["PYQCU_TORCH_BATCH_REDUCTIONS"] = batch_value
            for _ in range(warmups):
                Krylov(
                    backend="torch", method="bicgstab",
                    operator=adapted,
                    control=BiCGStabControl(
                        rtol=rtol, max_iter=max_iter,
                        true_residual_every=None,
                        require_true_residual=False,
                        breakdown_restarts=0),
                ).solve_result(rhs)
            elapsed = []
            samples = []
            sample_counters = []
            for _ in range(repeats):
                comm.Barrier()
                if device.type == "cuda":
                    _torch.torch.cuda.synchronize(device)
                started = perf_counter()
                with _collective_counters() as counters:
                    result = Krylov(
                        backend="torch", method="bicgstab",
                        operator=adapted,
                        control=BiCGStabControl(
                            rtol=rtol, max_iter=max_iter,
                            true_residual_every=None,
                            require_true_residual=False,
                            breakdown_restarts=0),
                    ).solve_result(rhs)
                if device.type == "cuda":
                    _torch.torch.cuda.synchronize(device)
                comm.Barrier()
                elapsed.append(perf_counter() - started)
                samples.append({
                    "iterations": int(result.krylov_iterations),
                    "relative_residual": float(
                        result.relative_true_residual),
                    "allreduce_calls": int(counters["allreduce_calls"]),
                    "allreduce_scalars": int(
                        counters["allreduce_scalars"]),
                })
                sample_counters.append(dict(counters))
            gathered_counters = comm.allgather(sample_counters)
            assert all(
                item == gathered_counters[0] for item in gathered_counters)
            _emit_metric(
                "torch_real_benchmark", ranks=size, mode=label,
                process_grid=process_grid, global_shape=global_shape,
                local_shape=local_shape, device=device_name,
                device_uuid=device_uuid, seconds_median=median(elapsed),
                seconds_samples=elapsed, samples=samples,
                allreduce_calls=samples[0]["allreduce_calls"],
                allreduce_scalars=samples[0]["allreduce_scalars"])
    finally:
        if old_batch is None:
            os.environ.pop("PYQCU_TORCH_BATCH_REDUCTIONS", None)
        else:
            os.environ["PYQCU_TORCH_BATCH_REDUCTIONS"] = old_batch


if __name__ == "__main__":
    if os.environ.get("PYQCU_TORCH_REAL_BENCHMARK", "0") == "1":
        _real_configured_benchmark()
        raise SystemExit(0)
    test_batched_reductions_use_one_collective()
    test_single_face_halo_staging_and_legacy_fallback()
    test_distributed_roll_matches_global_periodic_reference()
    test_bicgstab_collective_count_and_fallback()
    test_torch_krylov_is_globally_reduced_and_rank_symmetric()
    test_torch_strict_setup_is_rank_symmetric()
    test_torch_legacy_solve_is_rank_symmetric()
    if int(MPI.COMM_WORLD.Get_rank()) == 0:
        print("torch MPI optimizations: PASS")
