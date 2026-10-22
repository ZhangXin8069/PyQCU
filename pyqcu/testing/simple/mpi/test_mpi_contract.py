"""Fast contract tests for the simple MPI launcher."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pyqcu.testing.simple._common import write_lime_gauge
from pyqcu.testing.simple.mpi._common import (
    default_grid,
    is_lime_file,
    lime_metadata,
    mass_and_csw,
    read_gauge,
)
from pyqcu.testing.simple.mpi.solve_case import (
    REPO,
    _launch_environment,
    build_mpirun_command,
)


def test_default_grid_keeps_even_local_extents():
    lattice = (8, 8, 8, 16)
    grid = default_grid(lattice, 4)
    assert grid == (1, 4, 1, 1)
    assert all((lattice[axis] // grid[axis]) % 2 == 0 for axis in range(4))


def test_lime_magic_case_and_small_slice(tmp_path):
    rng = np.random.default_rng(7)
    qdp_gauge = (
        rng.normal(size=(4, 4, 4, 4, 4, 3, 3)) +
        1j * rng.normal(size=(4, 4, 4, 4, 4, 3, 3))).astype(np.complex128)
    path = tmp_path / "reference.LIME"
    write_lime_gauge(path, qdp_gauge, {
        "csw": 1.25,
        "reference_mass": -0.2,
    })

    assert is_lime_file(path)
    metadata = lime_metadata(path)
    assert tuple(metadata["lattice_xyzt"]) == (4, 4, 4, 4)
    assert metadata["csw"] == 1.25
    assert metadata["reference_mass"] == -0.2

    gauge, read_metadata, input_format = read_gauge(
        path, (2, 2, 2, 4), np.dtype(np.complex64))
    assert input_format == "LIME/SciDAC"
    assert gauge.shape == (3, 3, 4, 2, 2, 2, 4)
    assert gauge.dtype == np.complex64
    assert read_metadata["csw"] == 1.25


def test_lime_public_record_xml_roundtrip(tmp_path):
    gauge = np.zeros((4, 2, 2, 2, 2, 3, 3), dtype=np.complex128)
    for direction in range(4):
        gauge[direction, ..., 0, 0] = 1.0
        gauge[direction, ..., 1, 1] = 1.0
        gauge[direction, ..., 2, 2] = 1.0
    path = tmp_path / "public-record.lime"
    write_lime_gauge(path, gauge, {"csw": 0.8, "reference_mass": 0.17})
    raw = path.read_bytes()
    private = b"scidac-private-record-xml"
    public = b"scidac-record-xml"
    needle = private + b"\0" * (128 - len(private))
    replacement = public + b"\0" * (128 - len(public))
    assert needle in raw
    raw = raw.replace(needle, replacement, 1)
    private_file = b"scidac-private-file-xml"
    public_file = b"scidac-file-xml"
    file_needle = private_file + b"\0" * (128 - len(private_file))
    file_replacement = public_file + b"\0" * (128 - len(public_file))
    assert file_needle in raw
    path.write_bytes(raw.replace(file_needle, file_replacement, 1))

    read_gauge_data, metadata, input_format = read_gauge(
        path, (2, 2, 2, 2), np.dtype(np.complex64))
    assert input_format == "LIME/SciDAC"
    assert metadata["csw"] == 0.8
    assert read_gauge_data.shape == (3, 3, 4, 2, 2, 2, 2)


def test_lime_metadata_precedence():
    mass, csw = mass_and_csw({"reference_mass": -0.277, "csw": 1.16}, None, None)
    assert mass == -0.277
    assert csw == 1.16
    assert mass_and_csw({"reference_mass": -0.277}, 0.05, 0.8) == (0.05, 0.8)


def test_kupl_hybrid_launch_command():
    worker = ["python", "-m", "pyqcu.testing.simple.mpi.worker"]
    command = build_mpirun_command(
        mpirun="mpirun", ranks=4, bind_to="numa",
        map_by="ppr:1:numa", oversubscribe=True,
        allow_run_as_root=True, worker_arguments=worker)
    assert command[:8] == [
        "mpirun", "--allow-run-as-root", "--oversubscribe",
        "--bind-to", "numa", "--map-by", "ppr:1:numa", "-np"]
    assert command[8:10] == ["4", "python"]


def test_mpi_cuda_explicit_library_variant():
    args = SimpleNamespace(
        backend="cuda", omp_threads=1, libqcu="sm70")
    env = _launch_environment(args)
    expected = REPO / "data" / "libqcu-sm70"
    assert env["QCU_LIBCU_VARIANT"] == "libqcu-sm70"
    assert env["LD_LIBRARY_PATH"].split(":")[0] == str(expected)
