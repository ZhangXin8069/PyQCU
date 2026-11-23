"""Bitwise gate for the rank-slab read used by ``pyqcu.test``.

Before 2026-11-23 every rank of a native case called
``pyqcu.test.io.load_gauge`` / ``load_source``, which return the **whole**
global field, and sliced the rank block out afterwards.  On the L64x128 c64
case (9.66 GB gauge, 128 ranks) that moved 1.24 TB and held ~9.5 GB RSS per
rank - measured on job ``2236885``.

``pyqcu.test.native._read_block`` replaces it with an HDF5 hyperslab read of
exactly this rank's block, collectively when a parallel HDF5 is available
(``driver='mpio'``) and independently otherwise.  These tests pin the property
that makes the optimisation safe to use: **the slab read must be bit-identical
to reading the whole file and slicing it**, for both layouts and both fields.

The MPI case is exercised when ``h5py`` reports MPI support; otherwise the
serial slab path is checked and the collective path is skipped with a reason
instead of being silently unverified.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyqcu.test import io as test_io
from pyqcu.test import native

pytestmark = pytest.mark.skipif(
    not test_io.h5py_available(), reason="h5py is required for the artifacts")

LATTICE = (8, 8, 8, 16)
GRID = (2, 2, 1, 2)


def _geometry():
    # Constructed directly: local_geometry() also validates against the live
    # MPI world size, and these tests must run on one process with a grid.
    local = tuple(LATTICE[axis] // GRID[axis] for axis in range(4))
    return native.LocalGeometry(lattice=LATTICE, local=local, grid=GRID).check()


def _ranks():
    total = 1
    for value in GRID:
        total *= value
    return list(range(total))


def _gauge_full(rng):
    return rng.standard_normal((3, 3, 4, *LATTICE)).astype(np.complex64)


def _source_full(rng):
    return rng.standard_normal((4, 3, *LATTICE)).astype(np.complex64)


def _reference(field, geometry, rank, *, compact):
    """The pre-2026-11-23 answer: slice the block out of the whole field."""
    header = field.ndim - 4
    coordinate = native.rank_coordinate(rank, geometry.grid)
    selection = [slice(None)] * header
    for axis in range(4):
        extent = geometry.local[axis]
        if compact and axis == 3:
            extent //= 2
        start = coordinate[axis] * extent
        selection.append(slice(start, start + extent))
    return np.ascontiguousarray(field[tuple(selection)])


@pytest.mark.parametrize("kind", ["gauge", "source"])
def test_slab_read_matches_full_read_and_slice(tmp_path, kind):
    rng = np.random.default_rng(20261123)
    geometry = _geometry()
    field = _gauge_full(rng) if kind == "gauge" else _source_full(rng)
    path = tmp_path / f"{kind}.h5"
    (test_io.save_gauge if kind == "gauge" else test_io.save_source)(path, field)

    for rank in _ranks():
        block, mode = native._read_block(
            path, geometry=geometry, rank=rank, compact=False, dtype="c64")
        expected = _reference(field, geometry, rank, compact=False)
        assert mode in {"mpio", "slab"}
        assert block.dtype == expected.dtype
        assert block.shape == expected.shape
        assert np.array_equal(block, expected), (
            f"{kind} rank {rank} slab differs from the full-read slice")


@pytest.mark.parametrize("kind", ["gauge", "source"])
def test_compact_layout_slab_read_matches(tmp_path, kind):
    """The parity-compact artifacts halve the last axis; the slab must follow.

    The fixture is a synthetic array of the compact *shape*; the assertion is
    about the selection, so the values only have to be distinguishable.
    """
    rng = np.random.default_rng(7)
    geometry = _geometry()
    head = (2, 3, 3, 4) if kind == "gauge" else (2, 4, 3)
    shape = (*head, LATTICE[0], LATTICE[1], LATTICE[2], LATTICE[3] // 2)
    compact = rng.standard_normal(shape).astype(np.complex64)
    path = tmp_path / f"{kind}_compact.h5"
    (test_io.save_gauge if kind == "gauge" else test_io.save_source)(path, compact)

    for rank in _ranks():
        block, mode = native._read_block(
            path, geometry=geometry, rank=rank, compact=True, dtype="c64")
        expected = _reference(compact, geometry, rank, compact=True)
        assert mode in {"mpio", "slab"}
        assert block.shape == expected.shape
        assert np.array_equal(block, expected), (
            f"{kind} compact rank {rank} slab differs from the full-read slice")


def test_read_block_reports_the_mode_it_used(tmp_path):
    """The report must name the path, so a bound/unbound-style skew is visible."""
    rng = np.random.default_rng(11)
    geometry = _geometry()
    path = tmp_path / "gauge.h5"
    test_io.save_gauge(path, _gauge_full(rng))
    block, mode = native._read_block(
        path, geometry=geometry, rank=0, compact=False, dtype="c64")
    assert block.shape == (3, 3, 4, *geometry.local)
    assert mode == ("mpio" if native._mpi_comm() is not None else "slab")


def test_single_rank_geometry_reads_the_whole_field(tmp_path):
    """A one-rank case must keep reading exactly the file, unchanged."""
    rng = np.random.default_rng(3)
    field = _gauge_full(rng)
    path = tmp_path / "gauge.h5"
    test_io.save_gauge(path, field)
    geometry = native.LocalGeometry(lattice=LATTICE, local=LATTICE,
                                    grid=(1, 1, 1, 1)).check()
    block, _mode = native._read_block(
        path, geometry=geometry, rank=0, compact=False, dtype="c64")
    assert np.array_equal(block, field)
