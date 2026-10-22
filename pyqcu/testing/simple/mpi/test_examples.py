"""Fast contract tests for the flat simple-MPI example wrappers."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from pyqcu.testing.simple.mpi.mpi_ranks_power_of_two import (
    is_power_of_two,
    validate_rank_count,
)


EXAMPLE_DIR = Path(__file__).resolve().parent
HELPER = EXAMPLE_DIR / "mpi_ranks_power_of_two.py"
SOLVE_CASE = EXAMPLE_DIR / "solve_case.py"

EXAMPLES = (
    ("example_cuda_legacy_2rank.sh", "cuda", 2, 2),
    ("example_cuda_legacy_3level_2rank.sh", "cuda", 2, 3),
    ("example_kupl_legacy_2rank.sh", "kupl", 2, 2),
    ("example_kupl_legacy_4rank.sh", "kupl", 4, 2),
)


def _option_value(command: list[str], option: str) -> str:
    positions = [index for index, value in enumerate(command) if value == option]
    assert positions, f"missing {option} in {command!r}"
    index = positions[-1] + 1
    assert index < len(command)
    return command[index]


def _dry_run_env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    for name in ("PYQCU_RANKS", "PYQCU_GRID", "PYQCU_OMP_THREADS"):
        env.pop(name, None)
    env.update({
        "DRY_RUN": "1",
        "PYQCU_PYTHON": sys.executable,
        "PYQCU_GAUGE": str(tmp_path / "gauge with spaces.h5"),
        "PYQCU_LIME": str(tmp_path / "ignored.lime"),
        "PYQCU_SOURCE": str(tmp_path / "source with spaces.h5"),
        "PYQCU_COARSE_CACHE": str(tmp_path / "coarse level 1.h5"),
        "PYQCU_COARSE_CACHE_LEVEL": str(tmp_path / "coarse level 2.h5"),
        "PYQCU_DTYPE": "c64",
        "PYQCU_ATOL": "1e-6",
    })
    return env


def test_rank_helper_accepts_powers_of_two():
    for ranks in (1, 2, 4, 8, 16):
        assert is_power_of_two(ranks)
        assert validate_rank_count(ranks) == ranks


@pytest.mark.parametrize("ranks", (0, -2, 3, 6, 12))
def test_rank_helper_rejects_invalid_counts(ranks):
    assert not is_power_of_two(ranks)
    with pytest.raises(ValueError, match="positive power of two"):
        validate_rank_count(ranks)

    completed = subprocess.run(
        [sys.executable, str(HELPER), str(ranks)],
        text=True, capture_output=True, check=False)
    assert completed.returncode == 2
    assert "positive power of two" in completed.stderr
    assert str(ranks) in completed.stderr


@pytest.mark.parametrize(
    ("name", "backend", "ranks", "levels"), EXAMPLES)
def test_example_dry_run_is_single_safe_command(
        tmp_path, name, backend, ranks, levels):
    script = EXAMPLE_DIR / name
    assert script.is_file()
    assert os.access(script, os.X_OK)
    text = script.read_text(encoding="utf-8")
    assert "set -euo pipefail" in text
    assert "mpi_ranks_power_of_two.py" in text
    assert "MultiGpuMultigrid" not in text
    assert "examples/" not in text

    env = _dry_run_env(tmp_path)
    env["PYQCU_RANKS"] = str(ranks)
    completed = subprocess.run(
        ["bash", str(script)], cwd=tmp_path, env=env,
        text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert len(completed.stdout.splitlines()) == 1
    command = shlex.split(completed.stdout)
    assert command[1] == str(SOLVE_CASE)
    assert _option_value(command, "--backend") == backend
    assert _option_value(command, "--ranks") == str(ranks)
    assert _option_value(command, "--levels") == str(levels)
    assert _option_value(command, "--dtype") == "c64"
    assert _option_value(command, "--atol") == "1e-6"
    assert _option_value(command, "--gauge") == str(tmp_path / "gauge with spaces.h5")
    assert "--lime" not in command

    if levels == 3:
        assert "Legacy default depth: 2 levels." in completed.stderr
        assert "experimental" in completed.stderr
        assert _option_value(command, "--coarse-cache") == \
            str(tmp_path / "coarse level 1.h5")
        assert _option_value(command, "--coarse-cache-level") == \
            str(tmp_path / "coarse level 2.h5")
        assert "--skip-independent-residual" not in command
    else:
        assert completed.stderr == ""


@pytest.mark.parametrize("name", [example[0] for example in EXAMPLES])
def test_every_example_rejects_non_power_of_two_ranks(tmp_path, name):
    env = _dry_run_env(tmp_path)
    env["PYQCU_RANKS"] = "3"
    completed = subprocess.run(
        ["bash", str(EXAMPLE_DIR / name)], cwd=tmp_path, env=env,
        text=True, capture_output=True, check=False)
    assert completed.returncode == 2
    assert "positive power of two" in completed.stderr


def test_three_level_example_rejects_residual_skip(tmp_path):
    env = _dry_run_env(tmp_path)
    script = EXAMPLE_DIR / "example_cuda_legacy_3level_2rank.sh"
    completed = subprocess.run(
        ["bash", str(script), "--skip-independent-residual"],
        cwd=tmp_path, env=env, text=True, capture_output=True, check=False)
    assert completed.returncode == 2
    assert "independent true-residual" in completed.stderr


def test_missing_live_input_fails_before_solver(tmp_path):
    env = os.environ.copy()
    env.pop("PYQCU_GAUGE", None)
    env.pop("PYQCU_LIME", None)
    env["DRY_RUN"] = "0"
    env["PYQCU_PYTHON"] = sys.executable
    script = EXAMPLE_DIR / "example_kupl_legacy_2rank.sh"
    completed = subprocess.run(
        ["bash", str(script)], cwd=tmp_path, env=env,
        text=True, capture_output=True, check=False)
    assert completed.returncode == 2
    assert "PYQCU_GAUGE" in completed.stderr
    assert "PYQCU_LIME" in completed.stderr
