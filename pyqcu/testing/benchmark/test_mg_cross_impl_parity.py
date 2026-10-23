from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from pyqcu.testing.benchmark import mg_cross_impl_parity as parity


def _write_dataset(path: Path, dataset: str, shape: tuple[int, ...]) -> None:
    with h5py.File(path, "a") as handle:
        handle.create_dataset(
            dataset, data=np.arange(np.prod(shape), dtype=np.float32).reshape(shape))


def _args(tmp_path: Path) -> argparse.Namespace:
    gauge = tmp_path / "gauge.h5"
    source = tmp_path / "source.h5"
    null = tmp_path / "null.h5"
    cache = tmp_path / "cache.h5"
    _write_dataset(gauge, "data", (3, 3, 4, 2, 2, 2, 2))
    _write_dataset(source, "data", (4, 3, 2, 2, 2, 2))
    _write_dataset(null, "null", (2, 4, 3, 2, 2, 2, 2))
    _write_dataset(cache, "lonv", (2, 12, 1, 2, 1, 2, 1, 2, 1, 2))
    _write_dataset(cache, "hnn", (2, 4, 2, 12, 1, 1, 1, 1))
    _write_dataset(cache, "hdg", (2, 2, 6, 2, 12, 1, 1, 1, 1))
    _write_dataset(cache, "sit", (2, 12, 1, 1, 1, 1))
    return argparse.Namespace(
        gauge=gauge, source=source, nullvec=null, legacy_cache=cache,
        null_dataset="null", nvec=2, levels=2, block=(2, 2, 2, 2),
        lat=(2, 2, 2, 2), mass=0.05, csw=1.0, correction_interval=5,
        coarse_max_iter=20, coarse_tolerance=1e-4, restart=10,
        tolerance=1e-6,
    )


def test_parity_contract_records_shared_identities(tmp_path: Path):
    contract = parity.build_contract(_args(tmp_path))

    assert contract["schema"] == "pyqcu.mg-cross-impl-parity-contract.v1"
    assert contract["trace"]["enabled"] is False
    assert contract["algorithms"]["legacy"]["outer_solver"] == "BiCGStab"
    assert contract["algorithms"]["strict"]["outer_solver"].endswith("FGMRES")
    assert contract["coarse_assets"]["legacy"]["datasets"]["lonv"]
    assert contract["coarse_assets"]["strict"]["recipe_sha256"]
    assert contract["acceptance"]["strict_iteration_delta_max"] == 2
    assert contract["acceptance"]["legacy_iteration_relative_delta_max"] == 0.20


def test_torch_iterations_use_solver_report_not_history_length(tmp_path: Path):
    raw = {
        "status": "PASS",
        "iterations": {"samples": [23], "stop_reasons": ["native_tolerance"]},
        "residual": {
            "native_relative": 8.0e-7,
            "independent_relative": 6.5e-7,
        },
        "coarse_execution": {"last_solve_report": {"iterations": 23}},
        "timing": {"setup_seconds": 1.0},
    }

    normalized = parity._normalize_torch(raw, "cuda:0")

    assert normalized["iterations"] == 23
    assert normalized["solver_report"]["iterations"] == 23
    assert parity._side_pass(normalized, 1e-6)


def test_solution_relative_difference_is_l2_and_max(tmp_path: Path):
    left = tmp_path / "left.h5"
    right = tmp_path / "right.h5"
    with h5py.File(left, "w") as handle:
        handle.create_dataset("data", data=np.array([1.0, 2.0], dtype=np.complex64))
    with h5py.File(right, "w") as handle:
        handle.create_dataset("data", data=np.array([1.0, 2.1], dtype=np.complex64))

    result = parity._solution_difference(left, right)

    assert result["relative_l2"] > 0.0
    assert result["relative_max"] > result["relative_l2"]
