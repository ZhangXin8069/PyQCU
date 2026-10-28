"""CPU-only tests for the CUDA pair-reduction sampler and aggregators."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from pyqcu.testing.simple.mpi import cuda_pair_reduce_samples as samples


def _profile(*, variant: str) -> dict[str, int | float]:
    candidate_logical = 12
    candidate_pair = candidate_logical if variant == "B" else 0
    candidate_scalar = 0 if variant == "B" else 2 * candidate_logical
    unchanged_pair = 4
    real = 5
    pair = candidate_pair + unchanged_pair
    logical = candidate_logical + unchanged_pair
    return {
        "fine_iter": 1.5,
        "mpi_logical_complex_reductions": logical,
        "mpi_pair_reductions": pair,
        "mpi_scalar_reductions": candidate_scalar,
        "mpi_real_reductions": real,
        "mpi_allreduce_calls": pair + candidate_scalar + real,
        "mpi_unmodified_pair_reductions": unchanged_pair,
        "mpi_candidate_logical_reductions": candidate_logical,
        "mpi_candidate_pair_reductions": candidate_pair,
        "mpi_candidate_scalar_reductions": candidate_scalar,
        "mpi_pair_fine_dot_host": 4 if variant == "B" else 0,
        "mpi_pair_coarse_dot_slot": 8 if variant == "B" else 0,
        "mpi_pair_coarse_dot_host": 0,
        "mpi_pair_dot_coarse": 0,
        "mpi_pair_mixed_dot_host": 0,
    }


def _group(variant: str, slot: int, base: float) -> dict:
    label = (
        f"a-baseline-{slot}" if variant == "A" else
        f"b-optimized-{slot}")
    timed = []
    for index in range(5):
        wall = base + index * 0.01
        timed.append({
            "phase": "timed",
            "index": index,
            "wall_seconds": wall,
            "native_relative_residual": 1.882062e-7,
            "native_true_residual": 8.368379e-5,
            "iterations": 16,
            "logical_solution_sha256": "same-logical-sha",
            "logical_solution_shape": [4, 3, 8, 8, 8, 16],
            "report_status": "PASS",
            "converged": True,
            "atol": 1.0e-6,
            "ranks": [
                {"rank": 0, "solve_seconds": wall - 0.002},
                {"rank": 1, "solve_seconds": wall},
            ],
            "profile": _profile(variant=variant),
        })
    return {
        "schema": samples.SCHEMA,
        "kind": "group",
        "status": "PASS",
        "label": label,
        "variant": variant,
        "slot": slot,
        "warmups": [{"phase": "warmup", "index": 0}],
        "samples": timed,
    }


def _groups() -> list[dict]:
    return [
        _group("A", 0, 1.00),
        _group("B", 1, 0.90),
        _group("B", 2, 0.91),
        _group("A", 3, 0.99),
    ]


def test_parse_native_profile_uses_last_record_and_site_counters():
    text = "\n".join((
        "Total iterations: 99",
        "PROF_SECTIONS: mpi_pair_reductions=99",
        "Total iterations: 16",
        "FINAL TRUE residual (full-op) = 8.368379e-05  relative = "
        "1.882062e-07",
        "PROF_SECTIONS: fine_iter=58.6ms vcycle=57.4ms n_vcycles=1 "
        "mpi_logical_complex_reductions=16 mpi_pair_reductions=16 "
        "mpi_scalar_reductions=0 mpi_real_reductions=5 "
        "mpi_allreduce_calls=21 mpi_unmodified_pair_reductions=4 "
        "mpi_candidate_logical_reductions=12 "
        "mpi_candidate_pair_reductions=12 "
        "mpi_candidate_scalar_reductions=0 mpi_pair_fine_dot_host=4 "
        "mpi_pair_coarse_dot_slot=8 mpi_pair_coarse_dot_host=0 "
        "mpi_pair_dot_coarse=0 ck(k/d/s)=0.0/0.0/0.0",
    ))
    parsed = samples.parse_native_profile(text)
    assert parsed["iterations"] == 16
    assert parsed["native_true_residual"] == pytest.approx(8.368379e-5)
    assert parsed["native_relative_residual"] == pytest.approx(1.882062e-7)
    assert parsed["profile"]["fine_iter"] == pytest.approx(58.6)
    assert parsed["profile"]["mpi_pair_reductions"] == 16
    assert parsed["profile"]["mpi_pair_coarse_dot_slot"] == 8
    assert "ck(k/d/s)" not in parsed["profile"]


def test_abba_aggregate_reports_warm_medians_pairs_and_reduction_routes():
    report = samples.aggregate_group_reports(_groups())
    assert report["status"] == "PASS"
    assert all(report["gates"].values())
    assert report["group_order"] == [
        "a-baseline-0", "b-optimized-1",
        "b-optimized-2", "a-baseline-3"]
    assert report["variants"]["A"]["wall_seconds"]["samples"] == 10
    assert report["paired_differences"]["all_samples"][
        "difference_seconds"]["samples"] == 10
    assert len(report["paired_differences"]["pairs"]) == 2
    assert report["same_library_variability"]["A"][
        "group_median_spread_seconds"] == pytest.approx(0.01)
    assert report["variants"]["A"]["reduction_evidence"][
        "scalar_reductions"]["median"] == 24
    assert report["variants"]["B"]["reduction_evidence"][
        "pair_reductions"]["median"] == 16


def test_abba_aggregate_rejects_pair_route_for_baseline():
    groups = _groups()
    groups[0]["samples"][0]["profile"] = _profile(variant="B")
    report = samples.aggregate_group_reports(groups)
    assert report["status"] == "INCOMPLETE"
    assert report["gates"]["baseline_scalar_route"] is False


def test_logical_sha256_ignores_hdf5_layout_and_metadata(tmp_path: Path):
    array = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    first = tmp_path / "plain.h5"
    second = tmp_path / "chunked.h5"
    with h5py.File(first, "w") as handle:
        handle.create_dataset("data", data=array)
        handle.attrs["layout"] = "plain"
    with h5py.File(second, "w") as handle:
        handle.create_dataset(
            "data", data=array, chunks=(1, 2, 2), compression="gzip")
        handle.attrs["layout"] = "chunked"
    left = samples.logical_hdf5_sha256(first)
    right = samples.logical_hdf5_sha256(second)
    assert left == right
    assert left["shape"] == [2, 3, 4]
    assert left["dtype"] == "float32"


def test_atomic_json_failure_removes_partial(monkeypatch, tmp_path: Path):
    def fail_replace(source, destination):
        raise OSError("publish failed")

    monkeypatch.setattr(samples.os, "replace", fail_replace)
    target = tmp_path / "aggregate.json"
    with pytest.raises(OSError, match="publish failed"):
        samples.atomic_write_json(target, {"status": "PASS"})
    assert not target.exists()
    assert not list(tmp_path.glob("*.partial"))
    assert not list(tmp_path.glob(".*.partial"))


def test_failed_sample_cleanup_keeps_diagnostics(tmp_path: Path):
    (tmp_path / "solution.h5").write_bytes(b"partial solution")
    (tmp_path / "report.json").write_text("{}", encoding="utf-8")
    (tmp_path / ".solution.h5.123.partial").write_bytes(b"partial")
    (tmp_path / "stdout.txt").write_text("diagnostic", encoding="utf-8")
    removed = samples.cleanup_failed_sample(tmp_path)
    assert {Path(path).name for path in removed} == {
        "solution.h5", "report.json", ".solution.h5.123.partial"}
    assert (tmp_path / "stdout.txt").read_text(encoding="utf-8") == "diagnostic"


def test_cpu_residual_aggregation_checks_all_timed_solutions(tmp_path: Path):
    input_root = tmp_path / "input"
    residual_root = tmp_path / "residual"
    groups = _groups()
    for group in groups:
        group_root = input_root / group["label"]
        group_root.mkdir(parents=True)
        (group_root / "group.json").write_text(
            json.dumps(group), encoding="utf-8")
        for sample in group["samples"]:
            sample_root = (
                residual_root / group["label"] / "samples" /
                f"sample-{sample['index']:03d}")
            sample_root.mkdir(parents=True)
            (sample_root / "residual.json").write_text(json.dumps({
                "status": "PASS",
                "converged": True,
                "relative_true_residual": 2.0462001595e-7,
            }), encoding="utf-8")
    report = samples.aggregate_residual_reports(input_root, residual_root)
    assert report["status"] == "PASS"
    assert report["residual_summary"]["samples"] == 20
    assert all(report["gates"].values())
