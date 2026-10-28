"""CPU-only fairness tests for the Legacy/Strict MG benchmark."""

from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyqcu._multigrid_warnings import (
    MultigridDepthWarning,
    warn_excessive_multigrid_levels,
)
from pyqcu.testing.qcu.multigrid.benchmarks.legacy_vs_strict import (
    _combine_device_peaks,
    _decode_legacy_cache_manifest,
    _legacy_cache_manifest,
    _legacy_cache_manifest_json,
    _legacy_cache_block_from_lonv_shape,
    _legacy_cache_physical_shape,
    _legacy_parity_block,
    _physical_coarse_shapes,
    _parse_legacy_run_text,
    _read_log_growth,
    _strict_coarse_execution,
    _strict_hierarchy_transitions,
    _validate_full_block,
    _validate_legacy_cache_block,
    _validate_legacy_cache_manifest,
)


def test_full_block_maps_to_legacy_parity_block() -> None:
    full = _validate_full_block((2, 3, 4, 6))
    assert full == (2, 3, 4, 6)
    assert _legacy_parity_block(full) == (2, 3, 4, 3)
    assert _physical_coarse_shapes((16, 24, 32, 48), full, 2) == [
        (8, 8, 8, 8)]


def test_full_block_rejects_odd_or_nonpositive_time() -> None:
    with pytest.raises(ValueError, match="bt must be a positive even integer"):
        _validate_full_block((2, 2, 2, 3))
    with pytest.raises(ValueError, match="bt must be a positive even integer"):
        _validate_full_block((2, 2, 2, 0))


def test_legacy_cache_block_is_recovered_and_checked_from_lonv_shape() -> None:
    # lonv is [E, e, Xc, bx, Yc, by, Zc, bz, Tc, bt].
    shape = (8, 12, 4, 2, 5, 3, 6, 2, 7, 1)
    assert _legacy_cache_block_from_lonv_shape(shape, level=1) == (2, 3, 2, 1)
    assert _validate_legacy_cache_block(shape, (2, 3, 2, 1), 1) == (
        2, 3, 2, 1)
    with pytest.raises(
            ValueError,
            match=r"level 2 block mismatch: actual=\(2, 3, 2, 1\) "
                  r"expected=\(2, 2, 2, 1\)"):
        _validate_legacy_cache_block(shape, (2, 2, 2, 1), 2)


def _geometry_warning(variant: str, block: tuple[int, int, int, int]) -> str:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", MultigridDepthWarning)
        warned = warn_excessive_multigrid_levels(
            2, variant=variant, backend="cpu",
            lattice=(8, 8, 8, 16), block=block,
            process_grid=(1, 1, 1, 1))
    assert warned
    assert len(caught) == 1
    return str(caught[0].message)


def test_legacy_and_strict_warn_with_the_same_physical_coarse_time() -> None:
    full_block = (2, 2, 2, 4)
    legacy = _geometry_warning("legacy", _legacy_parity_block(full_block))
    strict = _geometry_warning("strict", full_block)
    assert "coarsest=(4, 4, 4, 4)" in legacy
    assert "coarsest=(4, 4, 4, 4)" in strict


def test_legacy_cache_reports_and_validates_full_physical_shape() -> None:
    # lonv is [E, e, Xc, bx, Yc, by, Zc, bz, Tc, bt].
    shape = (8, 12, 4, 2, 6, 2, 8, 2, 5, 1)
    assert _legacy_cache_physical_shape(shape, level=1) == (4, 6, 8, 10)
    assert _physical_coarse_shapes(
        (8, 12, 16, 20), (2, 2, 2, 2), 2) == [(4, 6, 8, 10)]
    with pytest.raises(ValueError, match="physical coarse shape"):
        _legacy_cache_physical_shape((8, 12, 4, 2, 6, 2, 8, 2, 0, 1))


def test_cache_provenance_fails_closed_on_missing_or_mismatched_input() -> None:
    manifest = _legacy_cache_manifest(
        level=1, lattice=(8, 8, 8, 16), full_block=(2, 2, 2, 2),
        legacy_block=(2, 2, 2, 1), physical_shape=(4, 4, 4, 8),
        gauge_sha256="a" * 64, mass=0.05, csw=1.1, coarse_dof=12,
        nv_iters=3, local_stencil=True, local_stencil_batch_size=4,
        null_batch_chunk=None)
    encoded = _legacy_cache_manifest_json(manifest)
    decoded = _decode_legacy_cache_manifest(
        encoded, path=Path("cache.h5"), level=1)
    assert _validate_legacy_cache_manifest(
        decoded, manifest, path=Path("cache.h5"), level=1) == manifest
    with pytest.raises(ValueError, match="lacks provenance"):
        _decode_legacy_cache_manifest(
            None, path=Path("cache.h5"), level=1)
    with pytest.raises(ValueError, match="gauge_sha256"):
        _validate_legacy_cache_manifest(
            {**decoded, "gauge_sha256": "b" * 64}, manifest,
            path=Path("cache.h5"), level=1)


def test_legacy_run_parser_requires_observed_consistent_vcycle_counters() -> None:
    text = """
Total iterations: 17
FINAL TRUE residual (full-op) = 4.0e-8  relative = 2.5e-7
FINAL STOP: reason=full_residual converged=true full_relative=2.5e-7 tol=1e-6 full_gate_checks=1 correction_attempts=2 corrections_accepted=1 corrections_rejected=1
PROF_SECTIONS: fine_iter=10.0ms vcycle=20.0ms n_vcycles=2 coarse_iters=7
"""
    parsed = _parse_legacy_run_text(text)
    assert parsed["complete"]
    assert parsed["iterations"] == 17
    assert parsed["native_true_residual"] == pytest.approx(2.5e-7)
    coarse = parsed["coarse_execution"]
    assert coarse["counters_consistent"]
    assert coarse["executed"] is True
    assert coarse["attempted_vcycles"] == 2
    assert coarse["coarse_iterations"] == 7

    inconsistent = _parse_legacy_run_text(
        text.replace("correction_attempts=2", "correction_attempts=1"))
    assert inconsistent["coarse_execution"]["counters_consistent"] is False
    assert inconsistent["coarse_execution"]["executed"] is None
    assert inconsistent["coarse_execution"]["effective_mode"] == "unverified"


def test_log_growth_ignores_previous_solve_fragment(tmp_path: Path) -> None:
    log = tmp_path / "clover_multigrid.log"
    old = "Total iterations: 99\nPROF_SECTIONS: n_vcycles=9\n"
    log.write_text(old, encoding="utf-8")
    offset = log.stat().st_size
    log.write_text(old + "Total iterations: 4\n", encoding="utf-8")
    fragment, new_offset = _read_log_growth(log, offset)
    assert fragment == "Total iterations: 4\n"
    assert new_offset == log.stat().st_size


def test_strict_transition_geometry_is_read_from_materialized_hierarchy() -> None:
    transfer = SimpleNamespace(
        block_size=(2, 2, 2, 2), fine_shape=(8, 8, 8, 16),
        coarse_shape=(4, 4, 4, 8))
    assert _strict_hierarchy_transitions(
        [transfer], (8, 8, 8, 16), (2, 2, 2, 2), 2) == [{
            "level": 1,
            "block": [2, 2, 2, 2],
            "fine_shape": [8, 8, 8, 16],
            "coarse_shape": [4, 4, 4, 8],
        }]
    bad = SimpleNamespace(
        block_size=(2, 2, 2, 1), fine_shape=(8, 8, 8, 16),
        coarse_shape=(4, 4, 4, 16))
    with pytest.raises(ValueError, match="block mismatch"):
        _strict_hierarchy_transitions(
            [bad], (8, 8, 8, 16), (2, 2, 2, 2), 2)


def test_strict_workload_uses_backend_outer_iterations_not_config() -> None:
    workload = _strict_coarse_execution([0, 5])
    assert workload["outer_iterations"] == [0, 5]
    assert workload["top_level_vcycle_calls"] == [0, 5]
    assert workload["effective_mode"] == "recursive_vcycle"
    assert _strict_coarse_execution([0])["effective_mode"] == "zero_work"


def test_phase_memory_high_water_marks_are_combined() -> None:
    setup = {
        "sample_interval_seconds": 0.05, "sample_count": 4,
        "baseline_used_bytes": 100, "peak_used_bytes": 180,
    }
    solve = {
        "sample_interval_seconds": 0.05, "sample_count": 6,
        "baseline_used_bytes": 190, "peak_used_bytes": 260,
    }
    combined = _combine_device_peaks((setup, solve))
    assert combined is not None
    assert combined["baseline_used_bytes"] == 100
    assert combined["peak_used_bytes"] == 260
    assert combined["peak_delta_over_baseline_bytes"] == 160
    assert combined["sample_count"] == 10
