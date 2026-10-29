"""CPU compatibility probe for a CUDA-generated Legacy-MG cache."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from pyqcu.testing.simple._common import read_h5
from pyqcu.testing.simple.kupl_multigrid import build_coarse_assets


REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_ROOT = REPO_ROOT / "data/qcu_mg_cache_mpi_validation_20261029"
CUDA_CACHE = CASE_ROOT / "cache_cuda_rank1_E2_fixed.h5"
GAUGE = CASE_ROOT / "gauge_8x8x8x16_c64.h5"


def test_kupl_native_stencil_matches_cuda_legacy_cache() -> None:
    if not CUDA_CACHE.is_file() or not GAUGE.is_file():
        return

    probe_count = 2
    with h5py.File(CUDA_CACHE, "r") as handle:
        lonv = np.ascontiguousarray(handle["lonv"][:probe_count])
        expected = {
            name: np.ascontiguousarray(handle[name])
            for name in ("hnn", "hdg", "sit")
        }

    actual_lonv, hop_nn, hop_diag, sitting = build_coarse_assets(
        read_h5(GAUGE), (8, 8, 8, 16), -0.277,
        1.0 / (2.0 * -0.277 + 8.0), 1.160920226,
        np.dtype(np.complex64), 2, [12, probe_count], [2, 2, 2, 2],
        1, 42, device_name="cpu", supplied_null_vectors=lonv,
        validate_galerkin=False)
    actual = {
        "hnn": hop_nn[0],
        "hdg": hop_diag[0],
        "sit": sitting[0],
    }

    np.testing.assert_array_equal(actual_lonv[0], lonv)
    for name, expected_value in expected.items():
        relative = np.linalg.norm(
            actual[name] - expected_value) / max(
                np.linalg.norm(expected_value), 1.0)
        assert relative < 2.0e-5, (name, relative)
