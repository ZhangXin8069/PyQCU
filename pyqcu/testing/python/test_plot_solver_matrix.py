"""Smoke tests for normalized solver matrix plotting."""

from __future__ import annotations

import json

import pytest

from pyqcu.testing.simple.mpi import plot_solver_matrix


def test_solver_matrix_plot_writes_vector_and_raster(tmp_path):
    pytest.importorskip("matplotlib")
    records = []
    for ranks, wall in ((1, 0.2), (2, 0.11), (4, 0.07)):
        for backend, solver, factor in (
                ("cuda", "bicgstab", 1.0), ("quda", "bicgstab", 1.1)):
            records.append({
                "backend": backend,
                "solver": solver,
                "ranks": ranks,
                "grid": [ranks, 1, 1, 1],
                "wall_seconds": wall * factor,
                "iterations": 17 + ranks % 2,
                "true_residual": 8.0e-7,
            })
    source = tmp_path / "records.json"
    source.write_text(json.dumps({"metadata": {"lattice": [32, 32, 32, 64]},
                                  "records": records}), encoding="utf-8")
    output = tmp_path / "plots"
    assert plot_solver_matrix.main([
        "--input", str(source), "--out-dir", str(output)]) == 0
    assert (output / "large_volume_summary.csv").is_file()
    assert (output / "large_volume_wall.svg").is_file()
    assert (output / "large_volume_wall.png").is_file()
    assert (output / "large_volume_residual.svg").is_file()
