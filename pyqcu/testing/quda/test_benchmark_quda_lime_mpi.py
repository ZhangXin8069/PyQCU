"""Pure-Python checks for the distributed LIME QUDA benchmark helpers."""

from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("benchmark_quda_lime_mpi.py")
SPEC = importlib.util.spec_from_file_location("benchmark_quda_lime_mpi", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
BENCH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCH
SPEC.loader.exec_module(BENCH)


def _record(handle, name: str, payload: bytes) -> None:
    handle.write(BENCH.LIME_MAGIC)
    handle.write(struct.pack(">HHQ", 1, 0, len(payload)))
    handle.write(name.encode("ascii").ljust(128, b"\0"))
    handle.write(payload)
    handle.write(b"\0" * ((-len(payload)) % 8))


def test_lime_slab_and_global_rhs_layout(tmp_path) -> None:
    lattice = (4, 4, 4, 4)
    raw = np.zeros((4, 4, 4, 4, 4, 3, 3), dtype=np.complex128)
    for x in range(4):
        raw[:, :, :, x, 0, 0, 0] = 100 + x
    file_xml = (
        "<scidacFile><dims>4 4 4 4</dims></scidacFile>"
    ).encode()
    record_xml = (
        "<scidacRecord><datacount>4</datacount><colors>3</colors>"
        "<precision>D</precision></scidacRecord>"
    ).encode()
    public_record_xml = b"<Params><clovCoeff>1.25</clovCoeff><Mass>-0.25</Mass></Params>"
    ildg_xml = (
        "<ildgFormat><precision>64</precision><lx>4</lx><ly>4</ly>"
        "<lz>4</lz><lt>4</lt></ildgFormat>"
    ).encode()
    path = tmp_path / "tiny.lime"
    with path.open("wb") as handle:
        _record(handle, "scidac-private-file-xml", file_xml)
        _record(handle, "scidac-private-record-xml", record_xml)
        _record(handle, "scidac-record-xml", public_record_xml)
        _record(handle, "ildg-format", ildg_xml)
        _record(handle, "ildg-binary-data", np.asarray(raw, dtype=">c16").tobytes())

    metadata = BENCH.read_lime_metadata(path)
    assert metadata.lattice_xyzt == lattice
    assert metadata.csw == 1.25
    assert metadata.reference_mass == -0.25
    local_gauge = BENCH.read_lime_local_gauge(metadata, (2, 1, 1, 1), (1, 0, 0, 0))
    assert local_gauge.shape == (4, 4, 4, 4, 2, 3, 3)
    assert local_gauge[0, 0, 0, 0, 0, 0, 0] == 102
    assert local_gauge[0, 0, 0, 0, 1, 0, 0] == 103

    global_rhs = BENCH._global_rhs_canonical(lattice, 17)
    rng = np.random.default_rng(17)
    expected_rhs = (
        rng.standard_normal((4, 3, *lattice))
        + 1j * rng.standard_normal((4, 3, *lattice))
    ).astype(np.complex64)
    np.testing.assert_array_equal(global_rhs, expected_rhs)
    local_rhs = BENCH.slice_global_rhs(global_rhs, lattice, (2, 1, 1, 1), (1, 0, 0, 0))
    assert local_rhs.shape == (4, 3, 2, 4, 4, 4)
    np.testing.assert_array_equal(local_rhs, global_rhs[:, :, 2:4])

    c128_rhs = BENCH._global_rhs_canonical(lattice, 17, "c128")
    assert c128_rhs.dtype == np.dtype(np.complex128)
    rng = np.random.default_rng(17)
    expected_c128 = (
        rng.standard_normal((4, 3, *lattice))
        + 1j * rng.standard_normal((4, 3, *lattice))
    ).astype(np.complex128)
    np.testing.assert_array_equal(c128_rhs, expected_c128)
    assert BENCH._field_dtype("c64") == np.dtype(np.complex64)
    assert BENCH._field_dtype("c128") == np.dtype(np.complex128)

    assert BENCH._validate_geometry(4, lattice, (2, 1, 1, 2)) == (2, 4, 4, 2)
    assert BENCH._process_coord(3, (2, 1, 1, 2)) == (1, 0, 0, 1)
    assert BENCH._validate_geometry(1, (24, 24, 24, 64), (1, 1, 1, 1)) == (24, 24, 24, 64)
    assert BENCH._validate_geometry(2, (24, 24, 24, 64), (1, 1, 1, 2)) == (24, 24, 24, 32)
    assert BENCH._validate_geometry(4, (24, 24, 24, 64), (1, 1, 2, 2)) == (24, 24, 12, 32)

    assert BENCH.kappa_from_mass(0.05) == 1.0 / 8.1
    mad = BENCH.median_and_mad([1.0, 2.0, 10.0])
    assert mad["median_seconds"] == 2.0
    assert mad["mad_seconds"] == 1.0


def test_mg_cli_contract() -> None:
    args = BENCH.build_parser().parse_args([
        "--gauge", "gauge.lime", "--solver", "mg", "--levels", "2",
        "--nvec", "6", "--mg-block", "2", "2", "2", "2",
    ])
    assert args.solver == "mg"
    assert args.levels == 2
    assert args.nvec == 6
    assert args.mg_block == [2, 2, 2, 2]
