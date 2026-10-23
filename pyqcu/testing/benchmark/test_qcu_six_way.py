"""CPU-only protocol tests for the six-way benchmark orchestrator."""

from __future__ import annotations

from types import SimpleNamespace

from pyqcu.testing.benchmark import qcu_six_way as bench


def test_parse_cuda_trace_counts_coarse_stages_and_halo_calls(tmp_path):
    trace = tmp_path / "trace.tsv"
    trace.write_text(
        "solve_end\t0\t0\n"
        "stage\t1\t0\touter_iteration\n"
        "stage\t1\t0\tcoarse_vcycle\n"
        "iteration_count\t1\t0\trestriction_calls\t4\n"
        "iteration_count\t1\t0\tprolongation_calls\t4\n",
        encoding="utf-8",
    )
    parsed = bench._parse_cuda_trace(trace)
    assert parsed["solve_end_count"] == 1
    assert parsed["outer_iteration_stages"] == 1
    assert parsed["coarse_vcycle_stages"] == 1
    assert parsed["restriction_calls"] == 4
    assert parsed["prolongation_calls"] == 4


def test_parse_kupl_legacy_distinguishes_attempt_from_effective_mode():
    text = (
        "deflation residual=1.0 corrections=0\n"
        "deflation residual=1.0 corrections=0\n"
        "final iterations=48\n"
    )
    parsed = bench._parse_kupl_legacy(text)
    assert parsed["attempted_vcycles"] == 2
    assert parsed["rejected_attempts"] == 2
    assert parsed["executed"] is True
    assert parsed["effective_mode"] == "fine_only_after_rejected_correction"
    assert parsed["median"] == 48.0


def test_summary_rows_keep_fine_only_and_missing_residual_explicit():
    sides = {
        "cuda-legacy-fine-only": {
            "backend": "cuda",
            "strategy": "legacy",
            "status": "PASS",
            "timing": {"steady_median_seconds": 0.01},
            "iterations": {"median": 16},
            "coarse_execution": {
                "executed": False,
                "effective_mode": "fine_only",
            },
            "memory": {"device_memory_applicable": False},
            "residual": {"independent_relative": None},
        },
    }
    row = bench._summary_rows(sides)[0]
    assert row["coarse_executed"] is False
    assert row["effective_mode"] == "fine_only"
    assert row["true_residual"] is None
    assert row["peak_device_bytes"] is None


def test_worker_command_keeps_legacy_cache_ownership_explicit(tmp_path):
    args = SimpleNamespace(
        gauge=tmp_path / "gauge.h5",
        source=tmp_path / "source.h5",
        nullvec=tmp_path / "null.h5",
        null_dataset="null",
        cuda_legacy_cache=tmp_path / "cuda-cache.h5",
        kupl_legacy_cache=tmp_path / "kupl-cache.h5",
        kupl_strict_assets=tmp_path / "strict.h5",
        kupl_strict_abi="both",
        solution_dir=tmp_path / "solutions",
        lat=(16, 16, 16, 16),
        mass=0.05,
        csw=1.0,
        tolerance=1e-6,
        coarse_tolerance=1e-4,
        max_iter=200,
        coarse_max_iter=200,
        levels=2,
        block=(2, 2, 2, 2),
        nvec=12,
        nu=1,
        restart=20,
        correction_interval=5,
        warmups=1,
        repeats=3,
        seed=42,
        device="cuda:0",
        max_workspace_bytes=512 << 20,
    )

    torch_command = bench._worker_command(args, "torch-legacy", tmp_path)
    kupl_command = bench._worker_command(args, "kupl-legacy", tmp_path)
    strict_command = bench._worker_command(args, "kupl-strict", tmp_path)

    torch_cache = torch_command[torch_command.index("--legacy-cache") + 1]
    kupl_cache = kupl_command[kupl_command.index("--legacy-cache") + 1]
    assert torch_cache == str(args.cuda_legacy_cache)
    assert kupl_cache == str(args.kupl_legacy_cache)
    assert strict_command[strict_command.index("--kupl-strict-abi") + 1] == "both"
