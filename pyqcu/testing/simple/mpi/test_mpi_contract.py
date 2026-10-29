"""Fast contract tests for the simple MPI launcher."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import h5py
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
    _build_coarse_cache,
    main as solve_case_main,
    _parse_args,
    _launch_environment,
    _worker_environment_prefix,
    build_mpirun_command,
)
from pyqcu.testing.simple.mpi.benchmark_kupl import (
    _ranks_per_numa,
    main as benchmark_kupl_main,
)
from pyqcu.testing.simple.mpi._distributed_mg_cache import (
    _asset_coarse_step,
    _asset_slab_selection,
    _hdf5_asset_patch,
    _hdf5_gauge_canonical_sha256,
    _hdf5_gauge_patch,
    _patch_axis,
    _periodic_asset_patch,
)
from pyqcu.testing.simple.mpi.quda_residual import _add_clover_identity
from pyqcu.tools._legacy_mg_cache import (
    build_legacy_mg_cache_manifest,
    legacy_mg_cache_identity,
    legacy_mg_cache_shapes,
    logical_array_sha256,
    logical_array_sha256_chunked,
    logical_dataset_sha256,
    validate_legacy_mg_cache,
    write_legacy_mg_asset_subset,
)
import pyqcu.testing.simple.mpi.solve_case as solve_case_module
import pyqcu.testing.simple.mpi.benchmark_kupl as benchmark_module
from pyqcu.tools._argparse import BooleanOptionalAction


def test_boolean_optional_action_supports_positive_and_negative_forms():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", action=BooleanOptionalAction, default=True)
    assert parser.parse_args(["--feature"]).feature is True
    assert parser.parse_args(["--no-feature"]).feature is False


def test_default_grid_keeps_even_local_extents():
    lattice = (8, 8, 8, 16)
    grid = default_grid(lattice, 4)
    assert grid == (1, 4, 1, 1)
    assert all((lattice[axis] // grid[axis]) % 2 == 0 for axis in range(4))


def test_distributed_mg_lonv_patch_keeps_interleaved_axes():
    lonv = np.arange(
        2 * 12 * 4 * 2 * 4 * 2 * 4 * 2 * 4 * 2,
        dtype=np.float32,
    ).reshape(2, 12, 4, 2, 4, 2, 4, 2, 4, 2)
    patch = _periodic_asset_patch(lonv, "lonv", (0, 0, 0, 0),
                                  (2, 3, 2, 4))
    assert patch.shape == (2, 12, 2, 2, 3, 2, 2, 2, 4, 2)
    np.testing.assert_array_equal(patch, lonv[:, :, :2, :, :3, :, :2, :, :4, :])


def test_distributed_mg_patch_axis_keeps_alignment_when_narrow():
    start, extent, center = _patch_axis(8, 2, 3, 2)
    assert (start, extent, center) == (0, 8, 6)


def test_distributed_mg_cache_time_uses_odd_schur_coarse_stride():
    assert [_asset_coarse_step((2, 4, 8, 16), axis) for axis in range(4)] == [
        2, 4, 8, 32]


def test_hdf5_gauge_patch_wraps_without_collective(tmp_path):
    rng = np.random.default_rng(19)
    gauge = (
        rng.normal(size=(3, 3, 4, 4, 4, 4, 4)) +
        1j * rng.normal(size=(3, 3, 4, 4, 4, 4, 4))).astype(np.complex64)
    path = tmp_path / "gauge.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=gauge)
    starts = (3, 3, 3, 3)
    extent = (2, 2, 2, 2)
    actual = _hdf5_gauge_patch(
        path, starts, extent, np.dtype(np.complex64))
    expected = gauge
    for axis, (start, width) in enumerate(zip(starts, extent, strict=True)):
        expected = np.take(
            expected, (start + np.arange(width)) % 4, axis=axis + 3)
    np.testing.assert_array_equal(actual, expected)
    with h5py.File(path, "r") as handle:
        expected_sha = logical_dataset_sha256(handle["data"])
    assert _hdf5_gauge_canonical_sha256(path) == expected_sha


def test_hdf5_lonv_patch_matches_in_memory_periodic_read(tmp_path):
    lonv = np.arange(
        2 * 12 * 4 * 2 * 3 * 2 * 2 * 2 * 5 * 2,
        dtype=np.float32,
    ).reshape(2, 12, 4, 2, 3, 2, 2, 2, 5, 2)
    path = tmp_path / "lonv.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("lonv", data=lonv)
    starts = (3, 2, 1, 4)
    extent = (2, 2, 2, 3)
    actual = _hdf5_asset_patch(
        path, "lonv", starts, extent, np.dtype(np.float32),
        max_first_axis_extent=1)
    expected = _periodic_asset_patch(lonv, "lonv", starts, extent)
    np.testing.assert_array_equal(actual, expected)
    selection = _asset_slab_selection((1, 2, 0, 1), (2, 1, 2, 1), "lonv")
    assert selection[2] == slice(1, 3)
    assert selection[4] == slice(2, 3)
    assert selection[6] == slice(0, 2)
    assert selection[8] == slice(1, 2)


def test_staged_cache_can_manifest_streamed_existing_datasets(tmp_path):
    lattice = (4, 4, 4, 4)
    mg_grid = (2, 2, 2, 2)
    shapes = legacy_mg_cache_shapes(lattice, mg_grid, (12, 2))
    identity = legacy_mg_cache_identity(
        gauge_canonical_sha256="1" * 64,
        gauge_file_sha256="2" * 64,
        lattice=lattice, dtype=np.dtype(np.complex64), mass=-0.277,
        csw=1.160920225783, dof=(12, 2), mg_grid=mg_grid,
        nv_iters=1, seed=42)
    datasets = {
        name: np.zeros(shape, dtype=np.complex64)
        for name, shape in shapes.items()
    }
    path = tmp_path / "cache.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("lonv", data=datasets["lonv"])
        handle.attrs["pyqcu_cache_schema"] = "pyqcu.legacy-mg-cache.v1"
        handle.attrs["pyqcu_cache_layout"] = "odd-schur"
    write_legacy_mg_asset_subset(
        path, {name: datasets[name] for name in ("hnn", "hdg", "sit")},
        identity=identity)
    manifest = build_legacy_mg_cache_manifest(
        path, identity, datasets=None, setup_method="test",
        setup_backend="unit")
    with h5py.File(path, "r") as handle:
        assert manifest["assets"]["lonv"]["sha256"] == (
            logical_dataset_sha256(handle["lonv"]))
    from pyqcu.tools._legacy_mg_cache import write_legacy_mg_cache_manifest

    write_legacy_mg_cache_manifest(path, manifest)
    validate_legacy_mg_cache(path, identity, verify_file_sha256=False)


def test_chunked_dataset_digest_matches_array_digest(tmp_path):
    value = np.arange(3 * 5 * 7 * 11, dtype=np.complex64).reshape(3, 5, 7, 11)
    path = tmp_path / "chunked.h5"
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset("value", data=value)
        assert logical_dataset_sha256(
            dataset, max_chunk_bytes=97) == logical_array_sha256(value)
        assert logical_array_sha256_chunked(
            value, max_chunk_bytes=97) == logical_array_sha256(value)


def test_solve_case_builds_coarse_cache_with_requested_mpi_grid(
        tmp_path, monkeypatch):
    cache = tmp_path / "coarse.h5"
    cache.touch()
    captured = []

    def fake_run(command, **kwargs):
        captured.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(solve_case_module.subprocess, "run", fake_run)
    args = SimpleNamespace(
        lat=[8, 8, 8, 16], mass=-0.277, csw=1.160920226,
        dtype="c64", dof=[12, 2], levels=2, mg_grid=[2, 2, 2, 2],
        nv_iters=1, seed=42, build_device="cpu", build_batch_size=1,
        grid=[1, 2, 1, 1], mpirun="mpirun", ranks=2, bind_to="none",
        map_by=None, oversubscribe=True, allow_run_as_root=True,
    )
    case = {
        "gauge_path": str(tmp_path / "gauge.h5"),
        "mass": -0.277,
        "csw": 1.160920226,
    }
    _build_coarse_cache(args, case, cache)

    command, kwargs = captured[0]
    assert command[:4] == ["mpirun", "--allow-run-as-root",
                           "--oversubscribe", "--bind-to"]
    assert command[command.index("-np") + 1] == "2"
    grid_index = command.index("--grid")
    assert command[grid_index:grid_index + 5] == [
        "--grid", "1", "2", "1", "1"]
    assert kwargs["cwd"] == REPO
    assert kwargs["check"] is False


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


def test_worker_command_pins_current_checkout_pythonpath():
    prefix = _worker_environment_prefix()

    assert prefix[0].endswith("/env")
    assert prefix[1].startswith(f"PYTHONPATH={REPO}")
    assert prefix[2] == f"PYQCU_ROOT={REPO}"


def test_mpi_cuda_explicit_library_variant(tmp_path, monkeypatch):
    import pyqcu.testing.simple._runner as runner

    variant_dir = tmp_path / "data" / "libqcu-sm70"
    variant_dir.mkdir(parents=True)
    (variant_dir / "libqcu.so").touch()
    monkeypatch.setattr(runner, "REPO", tmp_path)
    args = SimpleNamespace(
        backend="cuda", omp_threads=1, libqcu="sm70")
    env = _launch_environment(args)
    assert env["QCU_LIBCU_VARIANT"] == "libqcu-sm70"
    assert env["LD_LIBRARY_PATH"].split(":")[0] == str(variant_dir)


def test_torch_cpu_defaults_to_one_rank_per_core_and_fixed_threads():
    args = _parse_args([
        "--backend", "torch", "--torch-device", "cpu", "--ranks", "4",
        "--omp-threads", "1", "--levels", "1", "--restart", "0",
        "--warmup", "1", "--repeat", "3",
    ])
    assert args.bind_to == "core"
    assert args.map_by == "slot:PE=1"
    assert args.warmup == 1
    assert args.repeat == 3
    env = _launch_environment(args)
    for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS",
    ):
        assert env[name] == "1"
    assert env["OMP_DYNAMIC"] == "FALSE"


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
    output = capsys.readouterr().out
    assert "null" not in output
    payload = json.loads(output)
    command = payload["command"]
    worker_index = command.index("-np") + 2
    assert command[worker_index].endswith("/env")
    assert command[worker_index + 1].startswith(f"PYTHONPATH={REPO}")
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
