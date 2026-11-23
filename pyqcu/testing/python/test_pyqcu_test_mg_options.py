"""The MG knobs must survive ``native.case`` all the way to the runner.

``pyqcu.test.common.Case`` only declares the cross-runner fields.  Before
2026-11-24 ``native.case()`` validated ``restart``/``mg_levels``/``mg_grid``/
``coarse_tol_factor`` through ``_options`` and then discarded them, so the
runner fell back to the ``NativeCase`` defaults.  Every row of an MG parameter
sweep recorded ``restart=5, coarse_tol_factor=1e4`` whatever the command line
said - six runs of the same point, indistinguishable from "the knobs have no
effect".

These tests pin the value at the boundary that was broken: the ``Case`` carries
what the caller set, and ``_options(case)`` resolves it back unchanged.
"""

from __future__ import annotations

import pytest

from pyqcu.test import native

LATTICE = (8, 8, 8, 16)


@pytest.mark.parametrize("restart", [1, 2, 5])
def test_restart_reaches_the_option(restart):
    built = native.case("mg", backend="kupl", lattice=LATTICE,
                        restart=restart, mg_levels=2)
    assert getattr(built, "restart", None) == restart
    assert native._options(built, kind="mg", backend="kupl").restart == restart


@pytest.mark.parametrize("factor", [1e1, 1e2, 1e4])
def test_coarse_tol_factor_reaches_the_option(factor):
    built = native.case("mg", backend="kupl", lattice=LATTICE,
                        coarse_tol_factor=factor, mg_levels=2)
    assert getattr(built, "coarse_tol_factor", None) == pytest.approx(factor)
    resolved = native._options(built, kind="mg", backend="kupl")
    assert resolved.coarse_tol_factor == pytest.approx(factor)


def test_mg_grid_and_levels_reach_the_option():
    built = native.case("mg", backend="kupl", lattice=LATTICE,
                        mg_grid=(2, 2, 2, 4), mg_levels=2)
    assert getattr(built, "mg_grid", None) == (2, 2, 2, 4)
    resolved = native._options(built, kind="mg", backend="kupl")
    assert tuple(resolved.mg_grid) == (2, 2, 2, 4)
    assert resolved.mg_levels == 2


def test_only_two_levels_are_supported_today():
    """mg-3 and deeper are refused, not silently downgraded.

    The Legacy-MG port drives exactly one coarse transition, so ``mg-3``/
    ``mg-**`` cannot exist until a deeper hierarchy is implemented.  Pinning
    the refusal keeps a sweep from "measuring" a level count that was never
    used.
    """
    with pytest.raises(ValueError, match="mg_levels"):
        native.case("mg", backend="kupl", lattice=LATTICE, mg_levels=3)


def test_unset_knobs_keep_the_runner_default():
    """An unset option must not become an explicit ``None`` override."""
    built = native.case("mg", backend="kupl", lattice=LATTICE)
    assert getattr(built, "restart", None) is None
    resolved = native._options(built, kind="mg", backend="kupl")
    assert resolved.restart == native.NativeCase().restart
    assert resolved.coarse_tol_factor == native.NativeCase().coarse_tol_factor
