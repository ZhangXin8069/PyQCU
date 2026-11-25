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


def test_mg_3_is_wired_when_the_lattice_holds_the_hierarchy():
    """mg-3 is accepted and its hierarchy derived - and still refused to run.

    Until 2026-11-24 ``mg_levels > 2`` was rejected before the lattice was
    looked at, so a lattice with room for three levels could not express it.
    The option layer now resolves the depth and derives the geometry.  That is
    *wiring*, not runnability: KUPL builds the fine transition only
    (``apply_multigrid.cpp:232,281`` take gauge+Clover terms) and the cache
    stores one transition, so the run refuses.  Both halves are asserted - a
    change that lets mg-3 reach an ABI it cannot fill is as much a regression as
    one that refuses the option again.

    The KUPL Legacy-MG accepts 1..5 levels and builds the hierarchy in a loop
    (``num_levels_ < 1 || num_levels_ > 5`` in lattice_clover_multigrid.cpp),
    and this port wires one asset set per transition
    (``native._configure_legacy_mg``: params block ``19 + 8*(level-1)``,
    ``set_ptrs`` quartet ``30 + 4*(level-1)``, ``argv`` tolerance
    ``3 + (level-1)``).  ``8x8x8x16`` has room for two levels only - the level-1
    coarse box is 4^4 and halves to 2^4 - so the depth that must be accepted is
    a lattice that fits three.
    """
    deep = (16, 16, 16, 32)
    built = native.case("mg", backend="kupl", lattice=deep, mg_levels=3,
                        dof=(12, 12, 12))
    assert getattr(built, "mg_levels", None) == 3
    resolved = native._options(built, kind="mg", backend="kupl")
    assert resolved.mg_levels == 3
    chain = native._mg_level_chain(deep, resolved.mg_grid, 3, resolved.dof)
    assert [tuple(level["dims"]) for level in chain] == [
        (16, 16, 16, 16), (8, 8, 8, 8), (4, 4, 4, 4)]
    # Wiring is asserted to the end of the chain the run would consume: three
    # levels means two transitions, and the second one's E is the first one's.
    assert [level["E"] for level in chain] == [12, 12, 12]
    assert len(chain) == 3
    identity = native.cache_module.mg_identity(
        lattice=deep, mg_grid=resolved.mg_grid, dof=resolved.dof,
        dtype="c64", mass=0.17, csw=0.0, seed=42, nv_iters=2)
    shapes = native._mg_level_shapes(identity, chain)
    record = native._mg_chain_record(chain, shapes, ["cache", "cache"])
    assert [entry["level"] for entry in record] == [1, 2]
    assert [entry["coarse"] for entry in record] == [
        [8, 8, 8, 8], [4, 4, 4, 4]]
    assert record[1]["E_parent"] == 12
    # A missing root is the only thing that may refuse a depth this lattice can
    # hold, and it must name the transition rather than the backend.
    refusal = native._missing_transition_root(resolved, 2, 3)
    assert "transition 2" in str(refusal)
    assert "at most" not in str(refusal)


def test_the_port_rejects_a_depth_the_lattice_cannot_hold():
    """An unrepresentable depth must name the level and the extent.

    The C++ builds every level in one constructor and validates that each coarse
    extent divides its parent (lattice_clover_multigrid.cpp:238-244), so a
    truncated hierarchy would leave the solver reading default ``params`` blocks
    for levels that were never wired.
    """
    with pytest.raises(
            ValueError,
            match=r"level 2 \(axis 0\) would have extent 2, below the "
                  r"coarsest extent 4"):
        native.case("mg", backend="kupl", lattice=LATTICE, mg_levels=3,
                    dof=(12, 12, 12))


def test_mg_3_needs_one_component_count_per_level():
    """``dof`` is per level; a shorter tuple cannot describe level 2."""
    with pytest.raises(ValueError, match="one dof component count per level"):
        native.case("mg", backend="kupl", lattice=(16, 16, 16, 32),
                    mg_levels=3)


def test_the_port_keeps_the_solver_level_limit():
    """1..5 is the solver's own limit - it must stay a usage error above it."""
    with pytest.raises(ValueError, match="accepts 1..5 levels"):
        native.case("mg", backend="kupl", lattice=(16, 16, 16, 32),
                    mg_levels=6, dof=(12, 12, 12, 12, 12, 12))


def test_coarse_tol_factor_reaches_the_harness_options():
    """The boundary that actually broke: Test -> Options -> runner kwargs.

    ``native.case`` carrying the attribute is not enough - the harness builds
    runner kwargs from ``Options``, so a value that stops at the Case is
    silently replaced by the runner default.  Six sweep points recorded 1e4
    while the CLI asked for 1e2 and 1e1 because of exactly this.
    """
    from pyqcu.test import common
    from pyqcu.test.__main__ import _build_test, _resolve_config, build_parser

    argv = ["--mode", "solve", "--native", "mg", "--dtype", "c64",
            "--lattice-size", "8", "8", "8", "16",
            "--lattice-grid", "2", "2", "2", "2", "--ranks", "8",
            "--coarse-tol-factor", "100", "--restart", "1"]
    config = _resolve_config(build_parser().parse_args(argv))
    test = _build_test(config)
    assert test.coarse_tol_factor == pytest.approx(100.0)
    options = common.Options(lattice=(8, 8, 8, 16),
                            lattice_grid=(2, 2, 2, 2),
                            coarse_tol_factor=test.coarse_tol_factor,
                            restart=test.restart)
    kwargs = options.as_kwargs()
    assert kwargs.get("coarse_tol_factor") == pytest.approx(100.0)
    assert kwargs.get("restart") == 1


def test_unset_knobs_keep_the_runner_default():
    """An unset option must not become an explicit ``None`` override."""
    built = native.case("mg", backend="kupl", lattice=LATTICE)
    assert getattr(built, "restart", None) is None
    resolved = native._options(built, kind="mg", backend="kupl")
    assert resolved.restart == native.NativeCase().restart
    assert resolved.coarse_tol_factor == native.NativeCase().coarse_tol_factor
