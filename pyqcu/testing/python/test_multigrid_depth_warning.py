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
