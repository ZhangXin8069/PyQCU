"""CPU regressions for the standalone Torch MPI worker."""

from __future__ import annotations

import numpy as np
import pytest
from mpi4py import MPI

import pyqcu.cann as _torch
from pyqcu import tools
from pyqcu.testing.simple.mpi._common import (
    canonical_fermion,
    canonical_gauge,
)
from pyqcu.testing.simple.mpi.worker import (
    _torch_hierarchy_extents,
    _torch_true_residual,
)
from pyqcu.torch import dslash
from pyqcu.torch.solver._legacy_multigrid import LegacyMultigrid


def _identity_gauge(lattice: tuple[int, int, int, int]) -> np.ndarray:
    gauge = np.zeros((3, 3, 4, *lattice), dtype=np.complex64)
    eye = np.eye(3, dtype=np.complex64)
    for direction in range(4):
        gauge[:, :, direction] = eye[..., None, None, None, None]
    return np.ascontiguousarray(gauge)


def test_torch_hierarchy_lowers_min_size_for_small_local_extent() -> None:
    effective, extents = _torch_hierarchy_extents(
        (8, 2, 8, 16), max_level=2, min_size=4,
        mg_grid=(2, 2, 2, 2))
    assert effective == 2
    assert extents == [[8, 2, 8, 16]]


def test_legacy_rejects_an_empty_hierarchy() -> None:
    cpu = _torch.device("cpu")
    gauge = _torch.zeros(
        (3, 3, 4, 2, 2, 2, 2), dtype=_torch.complex64, device=cpu)
    with pytest.raises(ValueError, match="has no Legacy levels"):
        LegacyMultigrid(
            dtype_list=[_torch.complex64],
            device_list=[cpu],
            U=gauge,
            clover_term=None,
            max_level=1,
            min_size=4,
            dof_list=[12],
            verbose=False,
        )


def test_torch_independent_residual_matches_native_operator(
        monkeypatch: pytest.MonkeyPatch) -> None:
    if int(MPI.COMM_WORLD.Get_size()) != 1:
        pytest.skip("this residual oracle is a single-rank CPU regression")
    monkeypatch.delenv("PYQCU_PROCESS_GRID", raising=False)

    lattice = (2, 2, 2, 2)
    gauge = _identity_gauge(lattice)
    rng = np.random.default_rng(20261029)
    source = (
        rng.normal(size=(4, 3, *lattice)) +
        1j * rng.normal(size=(4, 3, *lattice))).astype(np.complex64)
    solution = (
        rng.normal(size=(4, 3, *lattice)) +
        1j * rng.normal(size=(4, 3, *lattice))).astype(np.complex64)
    kappa = 0.125
    cpu = _torch.device("cpu")
    gauge_tensor = _torch.as_tensor(
        gauge, dtype=_torch.complex64, device=cpu)
    kappa_tensor = _torch.tensor([kappa], dtype=_torch.float32, device=cpu)
    clover = dslash.make_clover(
        gauge_tensor,
        kappa=kappa_tensor,
        u_0=_torch.tensor([1.0], dtype=_torch.float32, device=cpu),
        support_parallel=False,
        verbose=False,
        csw=1.0,
    )
    clover_onsite = dslash.add_I(clover)
    clover_parity = tools.oooxyzt2poooxyzt(
        clover_onsite.reshape(12, 12, *lattice)).detach().cpu().numpy()
    parity_shape = (*lattice[:3], lattice[3] // 2)

    independent = _torch_true_residual(
        _torch.as_tensor(solution, dtype=_torch.complex64, device=cpu),
        canonical_fermion(source).reshape(2, 12, *parity_shape),
        canonical_gauge(gauge),
        clover_parity,
        kappa,
        MPI.COMM_WORLD,
        (0, 0, 0, 0),
        (1, 1, 1, 1),
        lattice,
    )

    operator = dslash.operator(
        U=gauge_tensor,
        clover_term=clover,
        kappa=kappa_tensor,
        u_0=_torch.tensor([1.0], dtype=_torch.float32, device=cpu),
        verbose=False,
    )
    native_image = operator.matvec(
        _torch.as_tensor(
            solution.reshape(12, *lattice),
            dtype=_torch.complex64,
            device=cpu,
        )
    ).detach().cpu().numpy()
    source_flat = source.reshape(12, *lattice)
    native = float(
        np.linalg.norm(native_image - source_flat) /
        max(np.linalg.norm(source_flat), np.finfo(float).tiny))

    assert independent == pytest.approx(native, rel=2.0e-5, abs=1.0e-7)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
