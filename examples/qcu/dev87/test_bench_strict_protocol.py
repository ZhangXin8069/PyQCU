"""秒级协议测试；不得启动 CUDA、PyQUDA 或正式 16x32x32x48 benchmark。"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import bench_strict_vs_quda as bench


def _args(*extra: str):
    return bench._parser().parse_args(list(extra))


def _synthetic_fingerprints():
    """Return a schema-valid, file-free input bundle for protocol tests."""
    entries = {
        "gauge": {
            "sha256": "a" * 64,
            "path": "/synthetic/gauge.h5",
            "dataset": "g",
            "shape": [2, 3, 3, 4, 16, 32, 32, 24],
            "dtype": "complex64",
            "file_size_bytes": 1,
        },
        "source": {
            "sha256": "b" * 64,
            "path": "/synthetic/source.h5",
            "dataset": "fi",
            "shape": [2, 4, 3, 16, 32, 32, 24],
            "dtype": "complex64",
            "file_size_bytes": 1,
        },
        "null_vectors": {
            "sha256": "c" * 64,
            "path": "/synthetic/null.h5",
            "dataset": "null",
            "shape": [12, 4, 3, 16, 32, 32, 48],
            "dtype": "complex64",
            "file_size_bytes": 1,
        },
    }
    for value in entries.values():
        value["algorithm"] = "sha256(logical-hdf5-dataset-v1)"
    entries["bundle_hash"] = bench._sha256_json({
        key: {
            "sha256": value["sha256"],
            "shape": value["shape"],
            "dtype": value["dtype"],
        }
        for key, value in entries.items()
    })
    return entries


def _synthetic_provenance(document):
    precision = document["protocol"]["precision"]["name"]
    precision_bit = 4 if precision == "c64" else 8
    return {
        "pyqcu_git": {"repository": "/synthetic/pyqcu"},
        "quda_source_git": {"repository": "/synthetic/quda"},
        "quda_libraries": {
            "libquda": {
                "path": "/synthetic/libquda.so",
                "sha256": "d" * 64,
                "exists": True,
            },
            "libqmp": {
                "path": "/synthetic/libqmp.so",
                "sha256": "e" * 64,
                "exists": True,
            },
        },
        "pyquda_module": {"path": "/synthetic/pyquda/__init__.py"},
        "cmake_features": {
            "cache_path": "/synthetic/CMakeCache.txt",
            "cache_sha256": "f" * 64,
            "normalized": {
                "qdp_interface": True,
                "qio": True,
                "qmp": True,
                "reconstruct": 7,
                "precision": precision_bit,
                "multigrid_nvec_list": [12, 24],
            },
        },
        "patch_variant": {
            "name": "synthetic",
            "wsl2": False,
            "environment_scoped": False,
            "limitation": None,
        },
        "runtime": {"device_uuid": "GPU-test-v100"},
    }


def _synthetic_cache(document):
    identity = bench._strict_runtime_cache_identity({
        "protocol": document["protocol"],
        "input_fingerprints": document["input_fingerprints"],
    })
    identity_sha256 = bench._sha256_json(identity)
    path = bench._strict_runtime_cache_path(
        identity, Path(document["execution"]["strict_cache"]["directory"]))
    evidence = {
        "schema": "pyqcu.strict-runtime-cache",
        "schema_version": bench.STRICT_CACHE_FORMAT_VERSION,
        "identity_sha256": identity_sha256,
        "metadata_sha256": "1" * 64,
        "stats_sha256": "2" * 64,
        "manifest_sha256": "3" * 64,
        "file_size_bytes": 0,
        "tensor_count": 0,
        "logical_bytes": 0,
        "tensor_digests": {},
        "path": str(path),
    }
    expectation = document["execution"]["strict_cache"]["expect"]
    return {
        "path": str(path),
        "identity_sha256": identity_sha256,
        "hit": True,
        "expectation": expectation,
        "evidence": evidence,
    }


def _successful_side(side: str, document, seconds: float):
    # These fixtures intentionally satisfy the current result schema.  They
    # remain file/CUDA-free; physical cache inspection is mocked only in the
    # merge test below, while production formal workers keep the hard gate.
    document["input_fingerprints"] = _synthetic_fingerprints()
    repeats = document["protocol"]["repeats"]
    samples = [seconds] * repeats
    summary = bench._median_mad(samples)
    sampler = {
        "available": True,
        "scope": "device-wide cudaMemGetInfo; test fixture",
        "device": "cuda:0",
        "unit": "bytes",
        "interval_seconds": 0.01,
        "duration_seconds": 1.0,
        "sample_count": 2,
        "device_total_bytes": 1000,
        "device_used_initial_bytes": 100,
        "device_used_max_observed_bytes": 200,
        "join_timed_out": False,
        "errors": [],
    }
    record = {
        "side": side,
        "status": "ok",
        "reason": None,
        "config_hash": document["protocol"]["config_hash"],
        "input_bundle_hash": document["input_fingerprints"]["bundle_hash"],
        "timing": {
            "input_io_seconds": 0.1,
            "runtime_init_seconds": 0.2,
            "setup_seconds": 1.0,
            "warmups": [
                {"seconds": seconds, "iterations": 4, "converged": True,
                 "true_residual_rel": 1.0e-7}
                for _ in range(bench.WARMUPS)
            ],
            "steady": summary,
        },
        "iterations": bench._iteration_summary([4] * repeats),
        "converged_samples": [True] * repeats,
        "converged": True,
        "true_residual": {
            "samples_rel": [1.0e-7] * repeats,
            "max_rel": 1.0e-7,
            "gate": document["protocol"]["true_residual_gate"],
            "pass": True,
        },
        "krylov": {
            "requested_restart": document["protocol"]["restart_requested"],
            "effective_restart": document["protocol"]["restart_requested"],
            "max_krylov_bytes": document["protocol"]["max_krylov_bytes"],
            "effective_workspace_bytes": 1024,
        },
        "memory": {
            "schema_version": bench.MEMORY_SCHEMA_VERSION,
            "setup": {"cuda_peak_allocated_bytes": 1,
                      "cuda_peak_reserved_bytes": 2,
                      "device_wide_sampler": copy.deepcopy(sampler)},
            "first_solve": {
                "excluded_from_formal_timing": True,
                "seconds": seconds,
                "iterations": 4,
                "converged": True,
                "true_residual_rel": 1.0e-7,
                "memory": copy.deepcopy({
                    "cuda_peak_allocated_bytes": 3,
                    "cuda_peak_reserved_bytes": 4,
                    "device_wide_sampler": sampler,
                }),
            },
            "steady": {"cuda_peak_allocated_bytes": 3,
                       "cuda_peak_reserved_bytes": 4,
                       "nvidia_smi_process_max_observed_bytes": 5,
                       "untimed_device_memory_probe": {
                           "excluded_from_formal_timing": True,
                           "device_wide_sampler": copy.deepcopy(sampler),
                       }},
        },
        "timing_boundary": {
            "requested": "caller_preallocated",
            "supported": True,
            "formal_eligible": True,
            "zero_initial_guess_before_timer": True,
            "preserve_source": "test",
            "timed_operation": "test fixture",
            "performance_call": "test fixture",
        },
        "provenance": _synthetic_provenance(document),
    }
    if side == "pyqcu":
        record["runtime_cache"] = _synthetic_cache(document)
    if side == "quda":
        record["quda_parameters"] = {
            "requested": bench._quda_expected_parameters(
                document["protocol"], None),
            "actual": bench._quda_expected_parameters(
                document["protocol"], None),
            "mismatches": [],
        }
        record["quda_input_contract"] = {
            "qdp_host_dtype": "complex128",
            "device_precision": (
                "QUDA_SINGLE_PRECISION"
                if document["protocol"]["precision"]["name"] == "c64"
                else "QUDA_DOUBLE_PRECISION"),
        }
    return record


def test_dry_run_schema_is_fixed_and_dependency_free():
    document = bench.build_document(_args("--dry-run"), dry_run=True)
    assert bench.validate_document(document, allow_planned=True) == []
    assert document["state"] == "dry-run"
    assert document["profile"] == "formal"
    assert document["protocol"]["profile"] == "formal"
    assert document["protocol"]["lattice_xyzt"] == [16, 32, 32, 48]
    assert document["protocol"]["warmups"] == 2
    assert document["protocol"]["repeats"] == 5
    assert document["protocol"]["fused_workspace_formula"] == "(2m+5)B_f+2B_c"
    assert document["protocol"]["restart_effective"] == 4
    assert document["protocol"]["precision"]["name"] == "c64"
    assert document["inputs"]["null_vectors"]["path"].endswith(
        "L16x32x32x48_nvec12_full_c64.h5")
    assert document["inputs"]["null_vectors"]["dataset"] == "null"
    assert document["inputs"]["null_vectors"]["layout"] == \
        "[nvec,spin,color,x,y,z,t]"
    assert document["input_fingerprints"] is None
    assert all(document["sides"][side]["status"] == "planned"
               for side in bench.SIDE_NAMES)


def test_worker_result_extractor_tolerates_native_logs_after_sentinel():
    record = {"side": "pyqcu", "status": "ok", "worker_wall_seconds": 1.0}
    stdout = (
        "native setup log\n"
        + bench.WORKER_PREFIX
        + json.dumps(record, separators=(",", ":"))
        + "\n :0\nmove_wards[_FX_FY_]:0\n")
    assert bench._extract_worker_record(stdout) == record


@pytest.mark.parametrize("extra", [
    ("--repeats", "1"),
    ("--tol", "1e-5"),
    ("--restart", "8"),
    ("--max-iter", "999"),
    ("--max-krylov-bytes", str(1024 << 20)),
    ("--strict-galerkin-column-batch", "8"),
    ("--strict-galerkin-max-workspace-bytes", str(3 << 30)),
])
def test_formal_profile_rejects_exploratory_protocol_values(extra):
    with pytest.raises(ValueError, match="formal profile"):
        bench.build_document(
            _args("--dry-run", "--profile", "formal", *extra),
            dry_run=True)


def test_formal_profile_cache_any_is_rejected_before_execution():
    with pytest.raises(ValueError, match="formal profile requires --cache-expect hit"):
        bench._run_parent(_args(
            "--profile", "formal", "--side", "pyqcu", "--cache-expect", "any"))


def test_smoke_both_success_is_nonfair_and_cli_merge_succeeds(
        tmp_path, monkeypatch):
    args = _args(
        "--profile", "smoke", "--side", "both", "--repeats", "1",
        "--tol", "1e-3", "--restart", "8", "--max-iter", "20",
        "--cache-expect", "any")
    fingerprints = {
        "bundle_hash": "smoke-inputs",
        "gauge": {"sha256": "g"},
        "source": {"sha256": "b"},
        "null_vectors": {"sha256": "v"},
    }
    monkeypatch.setattr(bench, "_fingerprint_inputs", lambda _document: copy.deepcopy(fingerprints))
    monkeypatch.setattr(
        bench, "_launch_side",
        lambda side, document, _timeout: _successful_side(side, document, 1.0))

    document, code = bench._run_parent(args)
    assert code == 0
    assert document["profile"] == "smoke"
    assert document["state"] == "complete"
    assert document["comparison"]["status"] == "smoke-pass"
    assert document["comparison"]["profile"] == "smoke"
    assert document["comparison"]["fair"] is False
    assert document["comparison"]["speedup_pyqcu_over_quda"] is None

    left_path = tmp_path / "smoke-left.json"
    right_path = tmp_path / "smoke-right.json"
    left_path.write_text(json.dumps(document), encoding="utf-8")
    right_path.write_text(json.dumps(document), encoding="utf-8")
    output_path = tmp_path / "smoke-merged.json"
    merge_code = bench.main([
        "--merge", str(left_path), str(right_path),
        "--output", str(output_path)])
    assert merge_code == 0
    merged = json.loads(output_path.read_text(encoding="utf-8"))
    assert merged["comparison"]["status"] == "smoke-pass"
    assert merged["comparison"]["profile"] == "smoke"
    assert merged["comparison"]["fair"] is False
    assert merged["comparison"]["speedup_pyqcu_over_quda"] is None


def test_input_fingerprint_contract_accepts_canonical_full_nullvec(
        tmp_path, monkeypatch):
    import h5py
    import numpy as np

    lattice = (2, 2, 2, 2)
    nvec = 2
    path = tmp_path / "inputs.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "gauge", data=np.zeros((2, 3, 3, 4, 2, 2, 2, 1), np.complex64))
        handle.create_dataset(
            "source", data=np.zeros((2, 4, 3, 2, 2, 2, 1), np.complex64))
        handle.create_dataset(
            "null", data=np.zeros((nvec, 4, 3, *lattice), np.complex64))

    monkeypatch.setattr(bench, "LATTICE", lattice)
    monkeypatch.setattr(bench, "NVECS", nvec)
    inputs = {
        "gauge": {"path": str(path), "dataset": "gauge"},
        "source": {"path": str(path), "dataset": "source"},
        "null_vectors": {"path": str(path), "dataset": "null"},
    }
    fingerprints = bench._fingerprint_inputs({"inputs": inputs})
    assert fingerprints["null_vectors"]["shape"] == [nvec, 4, 3, *lattice]
    assert fingerprints["null_vectors"]["dtype"] == "complex64"


def test_c128_and_median_mad_contract():
    document = bench.build_document(
        _args("--dry-run", "--profile", "smoke", "--precision", "c128",
              "--repeats", "3"),
        dry_run=True)
    assert document["protocol"]["precision"]["complex_bytes"] == 16
    assert document["protocol"]["max_krylov_bytes"] == 1024 << 20
    assert document["protocol"]["restart_effective"] == 4
    assert document["protocol"]["tolerance"] == 1.0e-10
    assert document["protocol"]["true_residual_gate"] == 5.0e-10
    assert bench._median_mad([1.0, 2.0, 100.0]) == {
        "samples_seconds": [1.0, 2.0, 100.0],
        "median_seconds": 2.0,
        "mad_seconds": 1.0,
    }


def test_strict_setup_memory_is_independent_of_outer_krylov_budget():
    default = bench.build_document(_args("--dry-run"), dry_run=True)
    changed_krylov = bench.build_document(_args(
        "--dry-run", "--profile", "smoke", "--max-krylov-bytes",
        str(1024 << 20)), dry_run=True)
    setup = default["protocol"]["pyqcu_strict_setup"]

    assert setup == {
        "probe_mode": "colored",
        "column_batch_size": 12,
        "projection_site_batch_size": 4,
        "max_workspace_bytes": 4 << 30,
        "workspace_four_arena_lower_bound_bytes": 3623878656,
        "require_exact_batch": True,
    }
    assert changed_krylov["protocol"]["pyqcu_strict_setup"] == setup

    double = bench.build_document(_args(
        "--dry-run", "--precision", "c128"), dry_run=True)
    assert double["protocol"]["pyqcu_strict_setup"]["column_batch_size"] == 1
    assert double["protocol"]["pyqcu_strict_setup"]["max_workspace_bytes"] == 1 << 30


def test_strict_setup_cli_is_hashed_and_fails_closed_on_invalid_caps():
    default = bench.build_document(_args("--dry-run"), dry_run=True)
    custom = bench.build_document(_args(
        "--dry-run", "--profile", "smoke", "--strict-galerkin-column-batch", "8",
        "--strict-galerkin-max-workspace-bytes", str(3 << 30)), dry_run=True)
    assert custom["protocol"]["config_hash"] != default["protocol"]["config_hash"]

    for columns in ("0", "25"):
        with pytest.raises(ValueError, match="strict-galerkin-column-batch"):
            bench.build_document(_args(
                "--dry-run", "--profile", "smoke",
                "--strict-galerkin-column-batch", columns),
                dry_run=True)
    with pytest.raises(ValueError, match="cannot hold"):
        bench.build_document(_args(
            "--dry-run", "--profile", "smoke",
            "--strict-galerkin-column-batch", "12",
            "--strict-galerkin-max-workspace-bytes", str(512 << 20)),
            dry_run=True)


def test_torch_runtime_provenance_is_strict_json_serializable():
    class PrivateUuid:
        def __str__(self):
            return "GPU-deadbeef"

    class Properties:
        uuid = PrivateUuid()
        total_memory = 32 << 30
        major = 7
        minor = 0

    class FakeCuda:
        @staticmethod
        def current_device():
            return 0

        @staticmethod
        def get_device_properties(_index):
            return Properties()

        @staticmethod
        def get_device_name(_index):
            return "Tesla V100"

    class FakeVersion:
        cuda = "12.4"

    class FakeTorch:
        __version__ = "test"
        version = FakeVersion()
        cuda = FakeCuda()

    provenance = bench._torch_runtime_provenance(FakeTorch, "cuda:0")
    assert provenance["device_uuid"] == "GPU-deadbeef"
    json.dumps(provenance, allow_nan=False)


def test_device_memory_sampler_reports_device_wide_high_water():
    class Cuda:
        samples = iter(((900, 1000), (700, 1000), (800, 1000)))

        @classmethod
        def mem_get_info(cls, _device):
            return next(cls.samples)

    class Torch:
        cuda = Cuda

    sampler = bench._CudaDeviceMemorySampler(Torch, "cuda:0")
    sampler._sample()
    sampler._sample()
    sampler._sample()
    report = sampler.stop()
    assert report["available"] is True
    assert report["sample_count"] == 3
    assert report["device_total_bytes"] == 1000
    assert report["device_used_initial_bytes"] == 100
    assert report["device_used_max_observed_bytes"] == 300
    assert report["unit"] == "bytes"
    assert report["join_timed_out"] is False
    assert report["scope"].startswith("device-wide")


def test_device_memory_sampler_join_timeout_is_fail_safe():
    class Cuda:
        @staticmethod
        def mem_get_info(_device):
            raise AssertionError("stop must not sample on the caller thread")

    class Torch:
        cuda = Cuda

    class StuckThread:
        def __init__(self):
            self.join_timeouts = []

        def join(self, timeout=None):
            self.join_timeouts.append(timeout)

        @staticmethod
        def is_alive():
            return True

    sampler = bench._CudaDeviceMemorySampler(Torch, "cuda:7")
    stuck = StuckThread()
    sampler._thread = stuck
    report = sampler.stop()

    assert report["available"] is False
    assert report["join_timed_out"] is True
    assert report["device"] == "cuda:7"
    assert sampler._thread is stuck
    assert stuck.join_timeouts == [1.0]


def test_nvidia_smi_snapshot_is_filtered_by_device_uuid(monkeypatch):
    outputs = iter((
        "GPU-other, 42, 900\nGPU-target, 42, 200\n",
        "GPU-other, 900\nGPU-target, 200\n",
    ))

    monkeypatch.setattr(bench.shutil, "which", lambda _name: "/bin/nvidia-smi")

    def run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=["nvidia-smi"], returncode=0, stdout=next(outputs), stderr="")

    monkeypatch.setattr(bench.subprocess, "run", run)
    report = bench._nvidia_smi_used(pid=42, device_uuid="GPU-target")

    assert report["device_uuid"] == "GPU-target"
    assert report["process_used_bytes"] == 200 << 20
    assert report["gpu_used_bytes"] == 200 << 20


def test_quda_cleanup_attempts_all_resources_without_masking_error():
    class Multigrid:
        @staticmethod
        def destroy():
            raise RuntimeError("destroy failed")

    class Dirac:
        multigrid = Multigrid()

        def __init__(self):
            self.gauge_freed = False

        def freeGauge(self):
            self.gauge_freed = True

    dirac = Dirac()
    errors = bench._close_quda_dirac(dirac)

    assert dirac.gauge_freed is True
    assert len(errors) == 1
    assert "destroy failed" in errors[0]


def test_wsl2_quda_reduction_guard_requires_patched_first_library(
        tmp_path, monkeypatch):
    install = tmp_path / "quda-install"
    library_dir = install / "lib"
    library_dir.mkdir(parents=True)
    library = library_dir / "libquda.so"
    library.write_bytes(b"prefix\0DEV87_REDUCE_SYNC\0suffix")

    monkeypatch.setattr(bench.platform, "release", lambda: "WSL2-microsoft-standard")
    monkeypatch.setenv("QUDA_INSTALL", str(install))
    monkeypatch.setenv("LD_LIBRARY_PATH", f"{library_dir}:/other")
    monkeypatch.delenv("DEV87_REDUCE_SYNC", raising=False)

    report = bench._prepare_quda_reduction_runtime()
    assert report["required"] is True
    assert report["enabled"] is True
    assert report["library"] == str(library.resolve())
    assert report["library_sha256"] == bench._sha256_file(library)
    assert report["marker_present"] is True
    assert os.environ["DEV87_REDUCE_SYNC"] == "1"

    library.write_bytes(b"unpatched")
    try:
        bench._prepare_quda_reduction_runtime()
    except bench.BenchmarkFailure as exc:
        assert exc.code == "quda_wsl2_reduce_sync_missing"
    else:
        raise AssertionError("unpatched WSL2 QUDA must fail closed")


def test_quda_qmp_runtime_initializes_funneled_and_keeps_argv_alive(
        tmp_path, monkeypatch):
    import ctypes

    library_dir = tmp_path / "quda-install" / "lib"
    library_dir.mkdir(parents=True)
    quda_library = library_dir / "libquda.so"
    qmp_library = library_dir / "libqmp.so"
    quda_library.write_bytes(b"quda")
    qmp_library.write_bytes(b"qmp")

    state = {
        "initialized": False,
        "required": None,
        "finalized": False,
        "topology": None,
    }

    class Function:
        def __init__(self, callback):
            self.callback = callback
            self.argtypes = None
            self.restype = None

        def __call__(self, *args):
            return self.callback(*args)

    class Qmp:
        def __init__(self):
            self.QMP_is_initialized = Function(
                lambda: int(state["initialized"]))
            self.QMP_logical_topology_is_declared = Function(
                lambda: int(state["topology"] is not None))
            self.QMP_declare_logical_topology = Function(
                self._declare_topology)
            self.QMP_get_logical_number_of_dimensions = Function(
                lambda: 0 if state["topology"] is None else 4)
            self.QMP_get_logical_dimensions = Function(
                lambda: state["topology"])
            self.QMP_finalize_msg_passing = Function(self._finalize)
            self.QMP_init_msg_passing = Function(self._initialize)

        @staticmethod
        def _finalize():
            state["finalized"] = True

        @staticmethod
        def _initialize(_argc, _argv, required, provided):
            state["initialized"] = True
            state["required"] = required
            ctypes.cast(provided, ctypes.POINTER(ctypes.c_int))[0] = required
            return 0

        @staticmethod
        def _declare_topology(dims, ndim):
            assert ndim == 4
            state["topology"] = (ctypes.c_int * ndim)(
                *(int(dims[index]) for index in range(ndim)))
            return 0

    qmp = Qmp()
    load_modes = []

    def load_library(path, mode):
        assert Path(path) == qmp_library
        load_modes.append(mode)
        return qmp

    registered = []
    monkeypatch.setattr(bench.ctypes, "CDLL", load_library)
    monkeypatch.setattr(bench.atexit, "register", registered.append)
    monkeypatch.setattr(bench, "_QMP_RUNTIME_HOLD", [])

    report = bench._initialize_quda_qmp_runtime(
        {"library": str(quda_library)})

    assert report["initialized"] is True
    assert report["initialized_here"] is True
    assert report["thread_level_required"] == 1
    assert report["thread_level_provided"] == 1
    assert state["required"] == 1
    assert [int(state["topology"][index]) for index in range(4)] == [1, 1, 1, 1]
    assert report["logical_topology_declared"] is True
    assert report["logical_topology"] == [1, 1, 1, 1]
    assert state["finalized"] is False
    assert load_modes == [ctypes.RTLD_GLOBAL]
    assert registered == [qmp.QMP_finalize_msg_passing]
    assert bench._QMP_RUNTIME_HOLD[0] is qmp
    assert len(bench._QMP_RUNTIME_HOLD) == 3

    registered.clear()
    second = bench._initialize_quda_qmp_runtime(
        {"library": str(quda_library)})
    assert second["initialized_here"] is False
    assert second["thread_level_provided"] is None
    assert registered == []


def test_strict_runtime_cache_identity_is_physical_and_rhs_independent():
    document = bench.build_document(_args("--dry-run"), dry_run=True)
    fingerprints = {
        name: {
            "algorithm": "sha256(logical-hdf5-dataset-v1)",
            "sha256": name * 8,
            "shape": [1, 2],
            "dtype": "complex64",
        }
        for name in ("gauge", "source", "null_vectors")
    }
    fingerprints["bundle_hash"] = "bundle-a"
    payload = {
        "protocol": document["protocol"],
        "input_fingerprints": fingerprints,
    }
    first = bench._strict_runtime_cache_identity(payload)
    payload = copy.deepcopy(payload)
    payload["input_fingerprints"]["source"]["sha256"] = "different-rhs"
    payload["input_fingerprints"]["bundle_hash"] = "bundle-b"
    second = bench._strict_runtime_cache_identity(payload)

    assert first == second
    assert "source" not in first
    assert first["coarsening_operator"] == "R(X^-1 D)P"
    assert bench._strict_runtime_cache_path(first) == \
        bench._strict_runtime_cache_path(second)

    custom_document = bench.build_document(_args(
        "--dry-run", "--profile", "smoke", "--strict-galerkin-column-batch", "8",
        "--strict-galerkin-max-workspace-bytes", str(3 << 30)), dry_run=True)
    custom_payload = {
        "protocol": custom_document["protocol"],
        "input_fingerprints": copy.deepcopy(fingerprints),
    }
    assert custom_document["protocol"]["config_hash"] != \
        document["protocol"]["config_hash"]
    assert bench._strict_runtime_cache_identity(custom_payload) == first


def test_strict_setup_stats_must_match_requested_batch():
    config = bench.build_document(_args("--dry-run"), dry_run=True)["protocol"]
    stats = [{
        "effective_probe_mode": "colored",
        "column_batch_size": 12,
        "projection_site_batch_size": 4,
        "memory": {"workspace_upper_bytes": 3629097984},
    }]
    bench._validate_strict_setup_contract(stats, config)

    stats[0]["column_batch_size"] = 8
    with pytest.raises(bench.BenchmarkFailure) as error:
        bench._validate_strict_setup_contract(stats, config)
    assert error.value.code == "strict_setup_contract_mismatch"


def test_strict_runtime_expected_manifest_matches_formal_geometry():
    config = bench.build_document(_args("--dry-run"), dry_run=True)["protocol"]
    manifest = bench._strict_runtime_expected_manifest(config)

    assert manifest["level_count"] == 1
    assert manifest["tensor_count"] == 3
    assert manifest["dtype"] == "complex64"
    assert manifest["tensors"]["assets/fine_blocked_v"]["shape"] == [
        24, 12, 8, 2, 16, 2, 16, 2, 24, 2]
    assert manifest["tensors"][
        "assets/levels/0/preconditioned_links"]["shape"] == [
            2, 4, 24, 24, 8, 16, 16, 24]
    assert manifest["total_bytes"] == sum(
        value["nbytes"] for value in manifest["tensors"].values())


def test_formal_cache_contract_is_accepted_before_a_file_exists(tmp_path):
    from pyqcu.cuda._strict_cache import load_strict_runtime_cache

    for precision, expected_dtype in (("c64", "complex64"),
                                      ("c128", "complex128")):
        config = bench.build_document(
            _args("--dry-run", "--precision", precision),
            dry_run=True)["protocol"]
        manifest = bench._strict_runtime_expected_manifest(config)
        result = load_strict_runtime_cache(
            tmp_path / f"missing-{precision}.h5",
            identity={"case": precision},
            expected_manifest=manifest)
        assert manifest["dtype"] == expected_dtype
        assert result.hit is False
        assert result.reason == "not_found"


def test_cache_execution_policy_is_explicit_and_workspace_scoped():
    cache_dir = bench.REPO / "data" / "isolated-strict-cache"
    document = bench.build_document(_args(
        "--dry-run", "--strict-cache-dir", str(cache_dir),
        "--cache-expect", "hit"), dry_run=True)

    assert document["execution"]["strict_cache"] == {
        "directory": str(cache_dir.resolve()),
        "expect": "hit",
    }
    assert bench._strict_runtime_cache_path(
        {"case": "isolated"}, cache_dir).parent == cache_dir


def test_pyqcu_runtime_seal_preserves_setup_stats_and_reports_allocator_delta():
    class FakeHierarchy:
        strict_setup_stats = {"phase": {"bytes": 123}}

        def seal_cuda_runtime(self, *, runtime_assets_bound=False):
            assert runtime_assets_bound is True
            self.strict_setup_stats["phase"]["bytes"] = -1
            return {
                "sealed": True,
                "detached_setup_storage_bytes": 80,
            }

    class FakeCuda:
        def __init__(self):
            self.samples = iter((100, 35))
            self.synchronized = []

        def memory_allocated(self, device):
            return next(self.samples)

        def synchronize(self, device):
            self.synchronized.append(device)

    class FakeTorch:
        cuda = FakeCuda()

    setup_stats, report = bench._seal_pyqcu_hierarchy_runtime(
        FakeHierarchy(), FakeTorch, "cuda:0")
    assert setup_stats == {"phase": {"bytes": 123}}
    assert report["sealed"] is True
    assert report["allocator_released_bytes"] == 65
    assert FakeTorch.cuda.synchronized == ["cuda:0"]


def test_legacy_odd_nullvec_layout_unpacks_without_reinterpreting_e24():
    import numpy as np

    lattice = (4, 4, 4, 8)
    block = (2, 2, 2, 2)
    shape = (2, 12, 2, 2, 2, 2, 2, 2, 2, 2)
    blocked = np.arange(np.prod(shape)).reshape(shape)
    odd = bench._unpack_odd_null_vectors(
        blocked, lattice=lattice, block=block, nvec=2)
    assert odd.shape == (2, 4, 3, 4, 4, 4, 4)
    assert np.shares_memory(odd, blocked)

    wrong_e24 = np.empty((4, *shape[1:]))
    try:
        bench._unpack_odd_null_vectors(
            wrong_e24, lattice=lattice, block=block, nvec=2)
    except bench.BenchmarkFailure as exc:
        assert exc.code == "nullvec_shape_mismatch"
    else:
        raise AssertionError("E24 parity basis must not masquerade as 2 full vectors")


def test_quda_without_shared_qio_is_explicit_skip():
    document = bench.build_document(_args("--side", "quda"), dry_run=False)
    document["input_fingerprints"] = {
        "bundle_hash": "bundle",
        "null_vectors": {"sha256": "nullvec"},
    }
    payload = {
        "protocol": document["protocol"],
        "inputs": document["inputs"],
        "input_fingerprints": document["input_fingerprints"],
    }
    record = bench._worker_record("quda", payload)
    assert record["status"] == "skipped"
    assert record["reason"]["code"] == "shared_nullvec_qio_missing"


def test_single_side_documents_merge_only_after_fair_gates(monkeypatch):
    left = bench.build_document(
        _args("--side", "pyqcu", "--cache-expect", "hit"), dry_run=False)
    right = bench.build_document(_args("--side", "quda"), dry_run=False)
    fingerprints = {
        "bundle_hash": "same-inputs",
        "gauge": {"sha256": "g"},
        "source": {"sha256": "b"},
        "null_vectors": {"sha256": "v"},
    }
    left["input_fingerprints"] = copy.deepcopy(fingerprints)
    right["input_fingerprints"] = copy.deepcopy(fingerprints)
    left["sides"]["pyqcu"] = _successful_side("pyqcu", left, 1.0)
    right["sides"]["quda"] = _successful_side("quda", right, 2.0)
    bench._update_state(left)
    bench._update_state(right)

    # The fixture contains schema-valid synthetic evidence but no physical
    # HDF5 cache.  The physical inspector has its own dedicated tests; here
    # exercise the merge wiring without weakening the production gate.
    monkeypatch.setattr(
        bench, "_inspect_formal_runtime_cache",
        lambda _document, record: copy.deepcopy(
            record["runtime_cache"]["evidence"]))
    documents = {"left.json": left, "right.json": right}
    monkeypatch.setattr(
        bench, "_load_document", lambda path: copy.deepcopy(documents[path.name]))
    merged = bench.merge_documents(["left.json", "right.json"])
    assert bench.validate_document(merged, allow_planned=True) == []
    assert merged["state"] == "complete"
    assert merged["comparison"]["fair"] is True
    assert merged["comparison"]["speedup_pyqcu_over_quda"] == 2.0


def test_single_side_merge_preserves_quda_qio_contract(monkeypatch):
    left = bench.build_document(
        _args("--side", "pyqcu", "--cache-expect", "hit"), dry_run=False)
    right = bench.build_document(_args(
        "--side", "quda", "--quda-nullvec-prefix", "/qio/shared",
        "--quda-nullvec-manifest", "/qio/shared.json"), dry_run=False)
    fingerprints = {
        "bundle_hash": "same-inputs",
        "gauge": {"sha256": "g"},
        "source": {"sha256": "b"},
        "null_vectors": {"sha256": "v"},
    }
    left["input_fingerprints"] = copy.deepcopy(fingerprints)
    right["input_fingerprints"] = copy.deepcopy(fingerprints)
    left["sides"]["pyqcu"] = _successful_side("pyqcu", left, 1.0)
    right["sides"]["quda"] = _successful_side("quda", right, 2.0)
    expected = bench._quda_expected_parameters(right["protocol"], "/qio/shared")
    right["sides"]["quda"]["quda_parameters"] = {
        "requested": expected,
        "actual": copy.deepcopy(expected),
        "mismatches": [],
    }
    bench._update_state(left)
    bench._update_state(right)
    monkeypatch.setattr(
        bench, "_inspect_formal_runtime_cache",
        lambda _document, record: copy.deepcopy(
            record["runtime_cache"]["evidence"]))
    documents = {"left.json": left, "right.json": right}
    monkeypatch.setattr(
        bench, "_load_document", lambda path: copy.deepcopy(documents[path.name]))

    merged = bench.merge_documents(["left.json", "right.json"])

    assert merged["inputs"]["quda_qio"] == right["inputs"]["quda_qio"]
    assert bench.validate_document(merged, allow_planned=True) == []
    assert merged["comparison"]["fair"] is True


def test_fair_comparison_requires_verified_hit_clean_sampler_and_same_gpu():
    document = bench.build_document(
        _args("--side", "both", "--cache-expect", "hit"), dry_run=False)
    document["input_fingerprints"] = {
        "bundle_hash": "same-inputs",
        "gauge": {"sha256": "g"},
        "source": {"sha256": "b"},
        "null_vectors": {"sha256": "v"},
    }
    document["sides"]["pyqcu"] = _successful_side("pyqcu", document, 1.0)
    document["sides"]["quda"] = _successful_side("quda", document, 2.0)
    bench._update_state(document)
    assert document["comparison"]["fair"] is True

    cold = copy.deepcopy(document)
    cold["sides"]["pyqcu"]["runtime_cache"].update({
        "hit": False,
        "expectation": "miss",
    })
    bench._update_state(cold)
    assert cold["comparison"]["fair"] is False
    assert any("not requested as cache hit" in reason
               for reason in cold["comparison"]["reasons"])

    sampler_error = copy.deepcopy(document)
    sampler_error["sides"]["quda"]["memory"]["setup"][
        "device_wide_sampler"]["errors"] = ["probe failed"]
    bench._update_state(sampler_error)
    assert sampler_error["comparison"]["fair"] is False
    assert any("reported sampling errors" in reason
               for reason in sampler_error["comparison"]["reasons"])

    other_gpu = copy.deepcopy(document)
    other_gpu["sides"]["quda"]["provenance"]["runtime"][
        "device_uuid"] = "GPU-other-v100"
    bench._update_state(other_gpu)
    assert other_gpu["comparison"]["fair"] is False
    assert any("GPU UUID mismatch" in reason
               for reason in other_gpu["comparison"]["reasons"])


def test_success_schema_rejects_missing_memory_evidence():
    document = bench.build_document(
        _args("--side", "pyqcu", "--cache-expect", "hit"), dry_run=False)
    document["input_fingerprints"] = {
        "bundle_hash": "same-inputs",
        "gauge": {"sha256": "g"},
        "source": {"sha256": "b"},
        "null_vectors": {"sha256": "v"},
    }
    record = _successful_side("pyqcu", document, 1.0)
    del record["memory"]
    document["sides"]["pyqcu"] = record

    errors = bench.validate_document(document, allow_planned=True)

    assert "pyqcu memory evidence missing" in errors


def test_process_group_timeout_path_is_bounded():
    command = [
        sys.executable,
        "-c",
        ("import subprocess,sys,time; "
         "subprocess.Popen([sys.executable,'-c',"
         "'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)']); "
         "time.sleep(0.25); print('spawned', flush=True); time.sleep(30)"),
    ]
    result = bench.run_process_group(command, timeout=0.4)
    assert result["timed_out"] is True
    assert result["returncode"] is not None
    assert result["wall_seconds"] < 5.0
    assert "spawned" in result["stdout"]


def test_resume_keeps_success_and_accepts_later_quda_adapter(monkeypatch):
    previous = bench.build_document(
        _args("--side", "pyqcu", "--cache-expect", "hit"), dry_run=False)
    previous["input_fingerprints"] = {
        "bundle_hash": "same-inputs",
        "gauge": {"sha256": "g"},
        "source": {"sha256": "b"},
        "null_vectors": {"sha256": "v"},
    }
    previous["sides"]["pyqcu"] = _successful_side("pyqcu", previous, 1.0)
    bench._update_state(previous)

    original_is_file = Path.is_file

    def is_file(path):
        if path.name == "resume.json":
            return True
        return original_is_file(path)

    captured = {}
    monkeypatch.setattr(
        bench, "_fingerprint_inputs",
        lambda _document: copy.deepcopy(previous["input_fingerprints"]))
    monkeypatch.setattr(Path, "is_file", is_file)
    monkeypatch.setattr(bench, "_load_document", lambda _path: copy.deepcopy(previous))
    monkeypatch.setattr(bench, "_atomic_write", lambda *_args, **_kwargs: None)

    def launch(side, document, timeout):
        captured.update(document["inputs"]["quda_qio"])
        return {
            "side": side,
            "status": "skipped",
            "reason": {"code": "probe", "detail": str(timeout)},
        }

    monkeypatch.setattr(bench, "_launch_side", launch)
    document, code = bench._run_parent(_args(
        "--side", "quda", "--resume", "--output", "resume.json",
        "--quda-nullvec-prefix", "/qio/shared",
        "--quda-nullvec-manifest", "/qio/shared.json"))
    assert code == 2
    assert document["sides"]["pyqcu"]["status"] == "ok"
    assert captured == {
        "prefix": "/qio/shared",
        "conversion_manifest": "/qio/shared.json",
        "required_for_fair_quda_run": True,
    }


def test_resume_does_not_relabel_cache_policy_or_directory():
    document = bench.build_document(
        _args("--side", "pyqcu", "--cache-expect", "hit"), dry_run=False)
    document["input_fingerprints"] = {
        "bundle_hash": "same-inputs",
        "gauge": {"sha256": "g"},
        "source": {"sha256": "b"},
        "null_vectors": {"sha256": "v"},
    }
    record = _successful_side("pyqcu", document, 1.0)
    requested = copy.deepcopy(document["execution"])
    requested["strict_cache"]["expect"] = "hit"
    assert bench._resume_side_compatible("pyqcu", record, requested)

    relabeled = copy.deepcopy(record)
    relabeled["runtime_cache"]["expectation"] = "any"
    assert not bench._resume_side_compatible("pyqcu", relabeled, requested)

    moved = copy.deepcopy(requested)
    moved["strict_cache"]["directory"] = str(
        bench.REPO / "data" / "another-cache")
    assert not bench._resume_side_compatible("pyqcu", record, moved)


def test_cli_dry_run_stdout_is_valid_json_and_starts_no_worker():
    script = Path(bench.__file__).resolve()
    result = subprocess.run(
        [sys.executable, str(script), "--dry-run", "--side", "pyqcu"],
        capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stderr
    document = json.loads(result.stdout)
    assert bench.validate_document(document, allow_planned=True) == []
    assert document["selected_sides"] == ["pyqcu"]
    assert document["sides"]["pyqcu"]["status"] == "planned"
    assert document["sides"]["quda"]["status"] == "not_selected"
