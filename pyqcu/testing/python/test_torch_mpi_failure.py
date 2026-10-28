"""CPU/MPI failure-propagation regressions for the Torch simple worker.

Run this module with exactly two ranks.  The first case reproduces a
rank-local field-loading failure immediately before Torch enters Clover halo
exchanges.  The second checks that a rank-0 HDF5 failure is broadcast before
any subsequent collective.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from mpi4py import MPI

import pyqcu.testing.simple.mpi.worker as worker
from pyqcu.testing.simple._common import read_h5, read_h5_metadata


WORLD = MPI.COMM_WORLD

pytestmark = pytest.mark.skipif(
    WORLD.Get_size() != 2, reason="requires exactly two MPI ranks")


def _identity_gauge(lattice: tuple[int, int, int, int]) -> np.ndarray:
    gauge = np.zeros((3, 3, 4, *lattice), dtype=np.complex64)
    eye = np.eye(3, dtype=np.complex64)
    for direction in range(4):
        gauge[:, :, direction] = eye[..., None, None, None, None]
    return gauge


def _torch_args(output: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        backend="torch",
        torch_device="cpu",
        grid=[2, 1, 1, 1],
        lat=[4, 2, 2, 4],
        gauge="injected-gauge.h5",
        source="injected-source.h5",
        mass=-0.277,
        csw=1.160920226,
        atol=1.0e-6,
        max_iter=10,
        omp_threads=1,
        dtype="c64",
        levels=1,
        min_size=2,
        dof=[12],
        mg_grid=[2, 2, 2, 2],
        restart=0,
        coarse_max_iter=20,
        coarse_tol_factor=1.0e4,
        seed=42,
        warmup=0,
        repeat=1,
        coarse_cache=None,
        coarse_cache_level=[],
        skip_independent_residual=True,
        verbose=False,
        output=output,
        report="injected-report.json",
    )


def test_rank_local_setup_failure_is_collective_before_clover_halo(
        monkeypatch: pytest.MonkeyPatch) -> None:
    rank = int(WORLD.Get_rank())
    lattice = (4, 2, 2, 4)
    gauge = _identity_gauge(lattice)
    source = np.zeros((4, 3, *lattice), dtype=np.complex64)

    monkeypatch.setattr(
        worker, "read_gauge",
        lambda *args, **kwargs: (gauge, {}, "injected"))

    def read_source(*args, **kwargs):
        if rank == 1:
            raise OSError("simulated rank-local source failure")
        return source, {}, "injected"

    monkeypatch.setattr(worker, "read_source", read_source)

    try:
        with pytest.raises(
                RuntimeError,
                match="simulated rank-local source failure") as raised:
            worker._run_torch(_torch_args(), WORLD)
    finally:
        WORLD.Barrier()
    messages = WORLD.allgather(str(raised.value))
    assert len(set(messages)) == 1


def test_rank_zero_hdf5_failure_is_collective_and_preserves_final_output(
        monkeypatch: pytest.MonkeyPatch) -> None:
    rank = int(WORLD.Get_rank())
    shared_root = None
    if rank == 0:
        shared_root = Path(tempfile.mkdtemp(prefix="pyqcu-torch-output-"))
        output = shared_root / "solution.h5"
        output.write_bytes(b"old")
        shared_root_text = str(shared_root)
    else:
        shared_root_text = None
    shared_root = Path(WORLD.bcast(shared_root_text, root=0))
    output = shared_root / "solution.h5"

    def fail_write_h5(path, array, **kwargs):
        assert rank == 0
        Path(path).write_bytes(b"partial")
        raise OSError("simulated rank-zero HDF5 write failure")

    monkeypatch.setattr(worker, "write_h5", fail_write_h5)
    args = _torch_args(output=str(output))
    local_value = np.zeros((4, 3, 4, 2, 2, 4), dtype=np.complex64)
    parts = [(tuple(0 for _ in range(4)), local_value)] if rank == 0 else None
    try:
        with pytest.raises(
                RuntimeError,
                match="simulated rank-zero HDF5 write failure") as raised:
            worker._collective_call(
                WORLD, "Torch rank-0 HDF5 output",
                lambda: worker._torch_publish_solution(
                    args, rank, (4, 2, 2, 4), mpi_size=WORLD.Get_size(),
                    torch_num_threads=1,
                    kappa=1.0 / (2.0 * args.mass + 8.0), parts=parts))
        WORLD.Barrier()
        messages = WORLD.allgather(str(raised.value))
        assert len(set(messages)) == 1
        checks = WORLD.allgather(
            rank != 0 or (
                output.read_bytes() == b"old" and
                not list(shared_root.glob("*.partial"))))
        assert all(checks)
    finally:
        if rank == 0:
            shutil.rmtree(shared_root)
        WORLD.Barrier()


def test_rank_zero_hdf5_output_publishes_atomically() -> None:
    rank = int(WORLD.Get_rank())
    shared_root = None
    if rank == 0:
        shared_root_text = tempfile.mkdtemp(prefix="pyqcu-torch-output-")
    else:
        shared_root_text = None
    shared_root = Path(WORLD.bcast(shared_root_text, root=0))
    output = shared_root / "solution.h5"
    args = _torch_args(output=str(output))
    local_value = np.ones((4, 3, 4, 2, 2, 4), dtype=np.complex64)
    parts = [(tuple(0 for _ in range(4)), local_value)] if rank == 0 else None
    try:
        worker._collective_call(
            WORLD, "Torch rank-0 HDF5 output",
            lambda: worker._torch_publish_solution(
                args, rank, (4, 2, 2, 4), mpi_size=WORLD.Get_size(),
                torch_num_threads=1,
                kappa=1.0 / (2.0 * args.mass + 8.0), parts=parts))
        WORLD.Barrier()
        checks = WORLD.allgather(
            rank != 0 or (
                output.is_file() and
                not (shared_root / "solution.h5.partial").exists()))
        assert all(checks)
        publication_ok = True
        if rank == 0:
            published = read_h5(output)
            metadata = read_h5_metadata(output)
            publication_ok = (
                published.shape == (4, 3, 4, 2, 2, 4) and
                bool(np.all(published == local_value)) and
                metadata["backend"] == "torch" and
                metadata["mpi_size"] == WORLD.Get_size())
        assert WORLD.bcast(publication_ok, root=0)
    finally:
        if rank == 0:
            shutil.rmtree(shared_root)
        WORLD.Barrier()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
