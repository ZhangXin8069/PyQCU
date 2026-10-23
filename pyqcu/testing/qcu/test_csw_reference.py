from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

from pyqcu.torch import dslash
from pyqcu.cuda import QcuRuntime, define
from pyqcu.cuda._multi_gpu import _coarse_cache_paths
from pyqcu.testing.simple._common import csw_matches, resolve_csw


def test_csw_metadata_precedence() -> None:
    assert resolve_csw(0.7, {"csw": 1.2}) == pytest.approx(0.7)
    assert resolve_csw(None, {"csw": 1.2}) == pytest.approx(1.2)
    assert resolve_csw(None, {}) == pytest.approx(1.160920226)
    assert csw_matches(1.160920226, 1.1609202260000002)
    assert not csw_matches(1.0, 1.160920226)


def test_csw_abi_constants_match_cpp_header() -> None:
    header = Path(__file__).resolve().parents[3] / "cpp/cuda/qcu/include/define.h"
    text = header.read_text(encoding="utf-8")
    csw = int(re.search(r"#define _CSW_ (\d+)", text).group(1))
    size = int(re.search(r"#define _ARGV_SIZE_ (\d+)", text).group(1))
    coarse = int(re.search(
        r"#define _SET_PTRS_COARSE_BASE_ (\d+)", text).group(1))
    assert csw == define._CSW_ == 7
    assert size == define._ARGV_SIZE_ == 8
    assert coarse == define._SET_PTRS_COARSE_BASE_ == 30


def test_multigrid_tolerances_do_not_consume_csw_slot() -> None:
    source = (
        Path(__file__).resolve().parents[3]
        / "cpp/cuda/qcu/include/lattice_clover_multigrid.h"
    ).read_text(encoding="utf-8")
    assert "tol_idx <= _MG_LEVEL4_ATOL_" in source
    assert "tol_idx < _ARGV_SIZE_" not in source


def test_multigrid_cache_identity_includes_physics() -> None:
    base = dict(cache_dir="/tmp/cache", lat_full=(8, 8, 8, 16), level=1,
                E=24, nv_iters=2, nv_tol=1e-2)
    first = _coarse_cache_paths(
        **base, mass=0.05, csw=0.35, seed=42,
        cache_tag="generated:42")[0]
    second = _coarse_cache_paths(
        **base, mass=0.05, csw=1.7, seed=42,
        cache_tag="generated:42")[0]
    third = _coarse_cache_paths(
        **base, mass=0.05, csw=0.35, seed=43,
        cache_tag="generated:43")[0]
    assert len({first, second, third}) == 3


@pytest.mark.skipif(define.size > 4, reason="small runtime test grid")
def test_argv_csw_is_explicit_and_lifecycle_reuses_slots() -> None:
    class DummyQcu:
        def __init__(self) -> None:
            self.events: list[tuple[str, int]] = []

        @staticmethod
        def _index(params: torch.Tensor) -> int:
            return int(params[define._SET_INDEX_])

        def applyInitQcu(self, set_ptrs, params, argv) -> None:
            index = self._index(params)
            assert int(set_ptrs[index]) == 0
            assert float(argv[define._CSW_]) == pytest.approx(0.0)
            set_ptrs[index] = 1
            self.events.append(("init", index))

        def applyEndQcu(self, set_ptrs, params) -> None:
            index = self._index(params)
            assert int(set_ptrs[index]) == 1
            set_ptrs[index] = 0
            self.events.append(("end", index))

        def applyDummyQcu(self, _field, set_ptrs, params) -> None:
            index = self._index(params)
            assert int(set_ptrs[index]) == 1
            self.events.append(("op", index))

    dummy = DummyQcu()
    runtime = QcuRuntime.from_lattice(
        (4, 4, 4, 4), mass=0.1, csw=0.0,
        device="cpu", grid=(define.size, 1, 1, 1), qcu_module=dummy)
    field = torch.empty(1, dtype=torch.complex64)
    sentinel = 0x5A5A
    runtime.set_ptrs[define._SET_PTRS_COARSE_BASE_] = sentinel
    for _ in range(101):
        runtime.call("applyDummyQcu", field)

    assert float(runtime.argv[define._CSW_]) == 0.0
    assert runtime.kappa == pytest.approx(1.0 / 8.2)
    expected_slot = 100 % define._SET_PTRS_COARSE_BASE_
    assert dummy.events[-3:] == [
        ("init", expected_slot), ("op", expected_slot), ("end", expected_slot)]
    assert all(int(value) == 0
               for value in runtime.set_ptrs[:define._SET_PTRS_COARSE_BASE_])
    assert int(runtime.set_ptrs[define._SET_PTRS_COARSE_BASE_]) == sentinel
    assert all(index < define._SET_PTRS_COARSE_BASE_
               for _, index in dummy.events)


def test_runtime_cleans_up_when_init_raises() -> None:
    class FailingQcu:
        @staticmethod
        def applyInitQcu(set_ptrs, params, argv) -> None:
            set_ptrs[int(params[define._SET_INDEX_])] = 1
            raise RuntimeError("injected init failure")

        @staticmethod
        def applyEndQcu(set_ptrs, params) -> None:
            set_ptrs[int(params[define._SET_INDEX_])] = 0

        @staticmethod
        def unused(*args) -> None:
            raise AssertionError("operation must not run after init failure")

    runtime = QcuRuntime.from_lattice(
        (4, 4, 4, 4), mass=0.1, csw=0.0, device="cpu",
        grid=(define.size, 1, 1, 1), qcu_module=FailingQcu())
    with pytest.raises(RuntimeError, match="injected init failure"):
        runtime.call("unused")
    assert all(int(value) == 0 for value in runtime.set_ptrs)


def test_runtime_rejects_reentrant_transactions() -> None:
    class DummyQcu:
        @staticmethod
        def applyInitQcu(set_ptrs, params, argv) -> None:
            set_ptrs[int(params[define._SET_INDEX_])] = 1

        @staticmethod
        def applyEndQcu(set_ptrs, params) -> None:
            set_ptrs[int(params[define._SET_INDEX_])] = 0

        @staticmethod
        def applyDummyQcu(_field, set_ptrs, params) -> None:
            assert int(set_ptrs[int(params[define._SET_INDEX_])]) == 1

    runtime = QcuRuntime.from_lattice(
        (4, 4, 4, 4), mass=0.1, csw=0.0, device="cpu",
        grid=(define.size, 1, 1, 1), qcu_module=DummyQcu())
    field = torch.empty(1, dtype=torch.complex64)
    with runtime.operation("applyDummyQcu") as outer:
        outer(field)
        with pytest.raises(RuntimeError, match="not reentrant"):
            with runtime.operation("applyDummyQcu"):
                pass
        outer(field)
    assert all(int(value) == 0
               for value in runtime.set_ptrs[:define._SET_PTRS_COARSE_BASE_])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mass": -4.0},
        {"parity": 2},
        {"max_iter": -1},
    ],
)
def test_runtime_constructor_validates_controls(kwargs) -> None:
    with pytest.raises(ValueError):
        QcuRuntime.from_lattice(
            (4, 4, 4, 4), mass=kwargs.get("mass", 0.1),
            parity=kwargs.get("parity", 0),
            max_iter=kwargs.get("max_iter", 10), device="cpu",
            grid=(define.size, 1, 1, 1))


def test_make_clover_is_linear_in_csw() -> None:
    lattice = (2, 2, 2, 2)
    gauge = torch.empty((3, 3, 4, *lattice), dtype=torch.complex64)
    eye = torch.eye(3, dtype=gauge.dtype)
    gauge.copy_(eye.reshape(3, 3, 1, 1, 1, 1, 1).expand_as(gauge))
    kappa = torch.tensor([0.125])

    zero = dslash.make_clover(gauge, kappa=kappa, csw=0.0)
    unit = dslash.make_clover(gauge, kappa=kappa, csw=1.0)
    doubled = dslash.make_clover(gauge, kappa=kappa, csw=2.0)

    assert torch.count_nonzero(zero) == 0
    assert torch.allclose(doubled, 2.0 * unit, rtol=1.0e-6, atol=1.0e-7)
