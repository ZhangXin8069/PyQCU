"""Shared correctness-lattice policy for every backend test entry point."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import pyqcu.testing as testing
from pyqcu.testing._config import CORRECTNESS_LATTICE
from pyqcu.testing.qcu.single_function_common import (
    DEFAULT_LATTICE as QCU_DEFAULT_LATTICE,
    parser as qcu_parser,
)
from pyqcu.testing.quda.common import LAT as QUDA_LATTICE, parser as quda_parser


def test_correctness_lattice_is_shared_across_backends() -> None:
    expected = (8, 8, 8, 16)
    assert CORRECTNESS_LATTICE == expected
    assert QCU_DEFAULT_LATTICE == expected
    assert QUDA_LATTICE == expected
    assert testing.HISTORICAL_WILSON_REFERENCE_LATTICE == (16, 16, 16, 16)


def test_python_test_helpers_default_to_correctness_lattice() -> None:
    expected = list(CORRECTNESS_LATTICE)
    for function in (
        testing.test_lattice,
        testing.test_dslash_wilson,
        testing.test_dslash_parity,
        testing.test_solver,
        testing.test_smear_stout,
        testing.test_smear_wuppertal,
        testing.test_npu_emulation,
        testing.test_multigrid_multithread,
        testing.test_multi_gpu_multigrid,
    ):
        default = inspect.signature(function).parameters["lat_size"].default
        assert default == expected, f"{function.__name__} default is {default}"


def test_single_function_parsers_default_to_correctness_lattice() -> None:
    expected = CORRECTNESS_LATTICE
    assert qcu_parser("QCU correctness").parse_args([]).lat == expected
    assert quda_parser("QUDA correctness").parse_args([]).lat == expected


def test_simple_entry_sources_use_the_shared_correctness_lattice() -> None:
    root = Path(__file__).resolve().parents[3]
    files = (
        "pyqcu/testing/simple/qcu_bistabcg.py",
        "pyqcu/testing/simple/quda_bistabcg.py",
        "pyqcu/testing/simple/kupl_bistabcg.py",
        "pyqcu/testing/simple/qcu_multigrid.py",
        "pyqcu/testing/simple/quda_multigrid.py",
        "pyqcu/testing/simple/kupl_multigrid.py",
        "pyqcu/testing/simple/mpi/qcu_multigrid.py",
        "pyqcu/testing/simple/mpi/quda_multigrid.py",
        "pyqcu/testing/simple/mpi/kupl_multigrid.py",
        "pyqcu/testing/simple/mpi/compare_three_way.py",
    )
    for relative in files:
        text = (root / relative).read_text(encoding="utf-8")
        assert "8, 8, 8, 16" in text, relative
        assert "8, 8, 16, 16" not in text, relative


def test_historical_data_mode_requires_an_explicit_lattice() -> None:
    with pytest.raises(ValueError, match="historical 16x16x16x16"):
        testing.test_dslash_wilson(with_data=True)
