"""Tests for the backend-neutral MultiGrid depth warning."""

import warnings

import pytest

from pyqcu._multigrid_warnings import (
    MultigridDepthWarning,
    warn_excessive_multigrid_levels,
)


def test_legacy_two_levels_do_not_warn() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warned = warn_excessive_multigrid_levels(
            2, variant="legacy", backend="cuda", context="test")
    assert not warned
    assert not caught


def test_legacy_three_levels_warn() -> None:
    with pytest.warns(MultigridDepthWarning, match="validated depth 2"):
        warned = warn_excessive_multigrid_levels(
            3, variant="legacy", backend="cuda", context="test")
    assert warned


def test_same_depth_warning_depends_on_lattice_geometry() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        large = warn_excessive_multigrid_levels(
            2, variant="legacy", backend="cuda",
            lattice=(32, 32, 32, 64), block=(2, 2, 2, 2),
            process_grid=(1, 1, 1, 1))
    assert not large
    assert not caught

    with pytest.warns(MultigridDepthWarning, match="coarsest volume"):
        small = warn_excessive_multigrid_levels(
            2, variant="legacy", backend="cuda",
            lattice=(8, 8, 8, 16), block=(2, 2, 2, 2),
            process_grid=(1, 1, 1, 1))
    assert small


def test_process_grid_and_block_enter_geometry_warning() -> None:
    with pytest.warns(MultigridDepthWarning, match="next coarsening"):
        warned = warn_excessive_multigrid_levels(
            2, variant="legacy", backend="cuda",
            lattice=(24, 24, 24, 64), block=(3, 2, 2, 2),
            process_grid=(2, 1, 1, 1))
    assert warned


def test_strict_three_levels_do_not_warn() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warned = warn_excessive_multigrid_levels(
            3, variant="strict", backend="torch", context="test")
    assert not warned
    assert not caught


def test_strict_four_levels_warn() -> None:
    with pytest.warns(MultigridDepthWarning, match="validated depth 3"):
        warned = warn_excessive_multigrid_levels(
            4, variant="strict", backend="torch", context="test")
    assert warned
