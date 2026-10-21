"""CPU-only round-trip tests for backend-neutral coarse-grid caches."""

from pathlib import Path

import numpy as np

from pyqcu.testing.simple.kupl_multigrid import (
    load_coarse_assets,
    write_coarse_assets,
)


def test_coarse_cache_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(23)
    arrays = tuple(
        (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
            np.complex64)
        for shape in ((2, 2, 2), (2, 2, 3), (2, 2, 4), (2, 2, 5))
    )
    path = tmp_path / "coarse.h5"

    write_coarse_assets(path, *arrays)
    loaded = load_coarse_assets(path, np.dtype(np.complex64))

    assert path.is_file()
    for levels, expected in zip(loaded, arrays, strict=True):
        assert len(levels) == 1
        np.testing.assert_array_equal(levels[0], expected)


def test_coarse_cache_rejects_missing_dataset(tmp_path: Path) -> None:
    import h5py

    path = tmp_path / "incomplete.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("lonv", data=np.zeros((1,), dtype=np.complex64))

    try:
        load_coarse_assets(path, np.dtype(np.complex64))
    except ValueError as exc:
        assert "missing datasets" in str(exc)
    else:
        raise AssertionError("incomplete coarse cache was accepted")
