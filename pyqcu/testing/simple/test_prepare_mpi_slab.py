import json

import h5py
import numpy as np

from pyqcu.testing.simple.prepare_mpi_slab import main


def test_prepare_mpi_slab_writes_contiguous_rank_files(tmp_path):
    lattice = (2, 2, 2, 4)
    grid = (1, 1, 1, 2)
    source = tmp_path / "gauge.h5"
    output = tmp_path / "slabs"
    with h5py.File(source, "w") as handle:
        handle.create_dataset(
            "data",
            data=np.arange(3 * 3 * 4 * 2 * 2 * 2 * 4, dtype=np.float32)
            .astype(np.complex64).reshape(3, 3, 4, *lattice))

    assert main([
        "--input", str(source), "--lat", *map(str, lattice),
        "--grid", *map(str, grid), "--output-dir", str(output),
        "--kind", "gauge",
    ]) == 0

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["ranks"] == 2
    assert manifest["lattice"] == list(lattice)
    assert manifest["grid"] == list(grid)
    assert "input_sha256" not in manifest
    for rank in range(2):
        with h5py.File(output / f"rank_{rank:04d}.h5", "r") as handle:
            assert handle["data"].shape == (3, 3, 4, 2, 2, 2, 2)
            assert handle.attrs["rank"] == rank
