"""Fast contract tests for the simple MPI launcher."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

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
    main as solve_case_main,
    _parse_args,
    _launch_environment,
    build_mpirun_command,
)
from pyqcu.testing.simple.mpi.benchmark_kupl import (
    _ranks_per_numa,
    main as benchmark_kupl_main,
)
from pyqcu.testing.simple.mpi.quda_residual import _add_clover_identity
import pyqcu.testing.simple.mpi.solve_case as solve_case_module
import pyqcu.testing.simple.mpi.benchmark_kupl as benchmark_module


def test_default_grid_keeps_even_local_extents():
    lattice = (8, 8, 8, 16)
    grid = default_grid(lattice, 4)
    assert grid == (1, 4, 1, 1)
    assert all((lattice[axis] // grid[axis]) % 2 == 0 for axis in range(4))


def test_clover_residual_operator_adds_unit_onsite_term():
    additive = np.zeros((4, 3, 4, 3, 2, 2, 2, 4), dtype=np.complex64)
    additive[0, 0, 1, 2] = 3.0 + 4.0j
    operator = _add_clover_identity(additive)
    matrix = operator.reshape(12, 12, 2, 2, 2, 4)
    diagonal = np.arange(12)
    assert np.all(matrix[diagonal, diagonal] == 1.0 + 0.0j)
    assert matrix.reshape(4, 3, 4, 3, 2, 2, 2, 4)[0, 0, 1, 2, 0, 0, 0, 0] == (
        3.0 + 4.0j)


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


def test_kupl_library_binding_environment():
    args = SimpleNamespace(
        backend="kupl", omp_threads=24, kupl_numa_bind="library",
        kupl_cpu_node=None, kupl_fused_bicg_dots=True,
        kupl_disable_clover_cache=False, kupl_allow_oversubscribe=False,
        kupl_profile="kunpeng-2numa", kupl_memory_policy="preferred",
        kupl_memory_node="local+16", kupl_hugepage=True,
        kupl_topology_report=True,
    )
    env = _launch_environment(args)
    assert env["KUPL_NUMA_BIND"] == "1"
    assert env["KUPL_MEMORY_POLICY"] == "preferred"
    assert env["KUPL_MEMORY_NODE"] == "local+16"
    assert env["KUPL_HUGEPAGE"] == "1"
    assert env["KUPL_TOPOLOGY_REPORT"] == "1"
    assert env["PYQCU_KUPL_PROFILE"] == "kunpeng-2numa"
    assert "OMP_PROC_BIND" not in env
    assert "OMP_PLACES" not in env


def test_kupl_generic_plan_clears_inherited_runtime_environment(monkeypatch):
    for name, value in {
        "KUPL_MEMORY_POLICY": "bind",
        "KUPL_MEMORY_NODE": "local+16",
        "KUPL_REQUIRE_MEMORY_BIND": "1",
        "KUPL_HUGEPAGE": "1",
        "KUPL_TOPOLOGY_REPORT": "1",
        "PYQCU_KUPL_PROFILE": "kunpeng-2numa",
        "OMP_PROC_BIND": "close",
        "OMP_PLACES": "cores",
        "GOMP_CPU_AFFINITY": "0-7",
        "KMP_AFFINITY": "compact",
        "KUPL_REQUIRE_TOPOLOGY": "1",
        "KUPL_SYSFS_NODE_ROOT": "/stale/topology",
        "KUPL_ALLOW_OVERSUBSCRIBE": "1",
        "KUPL_CPU_NODE": "3",
        "KUPL_FUSED_BICG_DOTS": "0",
        "KUPL_DISABLE_CLOVER_CACHE": "1",
    }.items():
        monkeypatch.setenv(name, value)
    args = SimpleNamespace(
        backend="kupl", omp_threads=4, kupl_numa_bind="scheduler",
        kupl_cpu_node=None, kupl_fused_bicg_dots=True,
        kupl_disable_clover_cache=False, kupl_allow_oversubscribe=False,
        kupl_profile="generic", kupl_memory_policy=None,
        kupl_memory_node=None, kupl_hugepage=False,
        kupl_topology_report=False,
    )
    env = _launch_environment(args)
    assert env["KUPL_NUMA_BIND"] == "0"
    for name in (
        "KUPL_MEMORY_POLICY", "KUPL_MEMORY_NODE", "KUPL_REQUIRE_MEMORY_BIND",
        "KUPL_HUGEPAGE", "KUPL_TOPOLOGY_REPORT", "PYQCU_KUPL_PROFILE",
        "OMP_PROC_BIND", "OMP_PLACES", "GOMP_CPU_AFFINITY", "KMP_AFFINITY",
        "KUPL_REQUIRE_TOPOLOGY", "KUPL_SYSFS_NODE_ROOT",
        "KUPL_ALLOW_OVERSUBSCRIBE", "KUPL_CPU_NODE",
    ):
        assert name not in env
    assert env["KUPL_FUSED_BICG_DOTS"] == "1"
    assert env["KUPL_DISABLE_CLOVER_CACHE"] == "0"


def test_kupl_library_binding_forces_mpi_bind_none():
    args = _parse_args([
        "--backend", "kupl", "--ranks", "2", "--omp-threads", "2",
        "--kupl-numa-bind", "library",
    ])
    assert args.bind_to == "none"
    assert args.map_by == "ppr:1:numa"


def test_c128_dry_run_uses_real_dtype_paths(capsys):
    assert solve_case_main([
        "--backend", "kupl", "--dtype", "c128", "--ranks", "2",
        "--omp-threads", "2", "--levels", "1", "--restart", "0",
        "--output", "relative-solution.h5", "--report", "relative-report.json",
        "--dry-run",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    command = payload["command"]
    assert command[command.index("--output") + 1] == str(
        REPO / "data" / "relative-solution.h5")
    assert command[command.index("--report") + 1] == str(
        REPO / "data" / "relative-report.json")
    assert any("gauge_global_c128.h5" in argument for argument in command)
    assert any("source_global_c128.h5" in argument for argument in command)


def test_cuda_rejects_kupl_runtime_options(capsys):
    with pytest.raises(SystemExit):
        _parse_args(["--backend", "cuda", "--kupl-hugepage"])
    assert "KUPL runtime options require --backend kupl" in capsys.readouterr().err


def test_scheduler_multiple_ranks_per_numa_is_rejected(capsys):
    with pytest.raises(SystemExit):
        _parse_args([
            "--backend", "kupl", "--ranks", "4", "--omp-threads", "4",
            "--bind-to", "numa", "--map-by", "ppr:2:numa",
        ])
    assert "overlap cpusets" in capsys.readouterr().err


def test_mpi_benchmark_rejects_overlapping_scheduler_binding(capsys):
    assert _ranks_per_numa("ppr:2:numa") == 2
    with pytest.raises(SystemExit):
        benchmark_kupl_main([
            "--ranks", "4", "--threads", "4", "--bind-to", "numa",
            "--map-by", "ppr:2:numa", "--dry-run",
        ])
    assert "overlap cpusets" in capsys.readouterr().err


def test_scheduler_rank_count_cannot_exceed_local_numa(monkeypatch, capsys):
    monkeypatch.setattr(solve_case_module, "_local_numa_count", lambda: 2)
    with pytest.raises(SystemExit):
        _parse_args([
            "--backend", "kupl", "--ranks", "4", "--omp-threads", "2",
            "--bind-to", "numa", "--map-by", "ppr:1:numa",
        ])
    assert "more MPI ranks than local NUMA" in capsys.readouterr().err


def test_mpi_benchmark_rank_count_cannot_exceed_local_numa(
        monkeypatch, capsys):
    monkeypatch.setattr(benchmark_module, "_local_numa_count", lambda: 2)
    with pytest.raises(SystemExit):
        benchmark_module.main([
            "--ranks", "4", "--threads", "2", "--bind-to", "numa",
            "--map-by", "ppr:1:numa", "--dry-run",
        ])
    assert "more MPI ranks than local NUMA" in capsys.readouterr().err
