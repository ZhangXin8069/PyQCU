"""Multi-level Legacy-MG wiring in :mod:`pyqcu.test.native` (mg-3 .. mg-5).

The KUPL Legacy-MG solves a hierarchy of 1..5 levels in one constructor
(``num_levels_ < 1 || num_levels_ > 5`` and the per-level loop in
``cpp/kupl/qcu/src/lattice_clover_multigrid.cpp:195,231-275``).  ``pyqcu.test``
wired exactly one coarse transition until 2026-11-24: ``mg_levels > 2`` was
refused outright, and the single block shape would have been repeated for every
level.  The wiring is now per transition, and these tests pin the three things
that make it a hierarchy rather than one repeated block:

* the derived geometry chain (level ``L`` halves level ``L-1``, and a level
  below the coarsest extent or with an odd extent is refused by name),
* the placement of every level's ``E``/extents/``set_ptrs``/tolerance in its own
  ABI block (``19 + 8*(level-1)``, ``30 + 4*(level-1)``, ``3 + (level-1)``),
* the one-transition cache: a mg-2 case keeps the shapes and the wiring it always
  had, and a deeper case never claims a one-transition cache covers it.

Wiring is not running: KUPL has no coarse-to-coarser operator builder, so a
``mg_levels`` above 2 is still refused - now by a named ValueError from
:func:`pyqcu.test.native._resolve_mg_assets` instead of a blanket option check.
"""

from __future__ import annotations

import pytest

from pyqcu.test import cache as cache_module
from pyqcu.test import native

#: The port's tier-0 lattice: two levels at most (coarse 4^4 halves to 2^4).
SHALLOW = (8, 8, 8, 16)
#: Room for three levels with the default block grid.
DEEP = (16, 16, 16, 32)
#: Default coarsening of the first transition.
GRID = (2, 2, 2, 2)


class _SyntheticTensor:
    """A stand-in with the only two things the wiring reads."""

    def __init__(self, shape, address: int) -> None:
        self.shape = tuple(int(value) for value in shape)
        self._address = int(address)

    def data_ptr(self) -> int:
        return self._address


class _SyntheticControls:
    """``native._Controls`` shaped stand-in: the three ABI arrays + ``define``."""

    def __init__(self, define) -> None:
        self.define = define
        self.params = [0] * 58
        self.argv = [0.0] * 8
        self.set_ptrs = [0] * 100


def _define():
    from pyqcu.cuda import define

    return define


def _option(**overrides):
    """Resolve a ``NativeCase`` without importing torch or probing a backend."""
    values = {"lattice": DEEP, "mg_levels": 3, "dof": (12, 12, 12),
              "backend": "kupl"}
    values.update(overrides)
    return native._options(None, kind="mg", **values)


def _synthetic_transition(shape_lonv, base_address: int):
    """One transition's four tensors with distinct, checkable addresses."""
    extents = tuple(shape_lonv[2::2])
    components = shape_lonv[0]
    return {
        "lonv": _SyntheticTensor(shape_lonv, base_address + 0),
        "hnn": _SyntheticTensor((2, 4, components, components, *extents),
                                base_address + 1),
        "hdg": _SyntheticTensor((2, 2, 6, components, components, *extents),
                                base_address + 2),
        "sit": _SyntheticTensor((components, components, *extents),
                                base_address + 3),
    }


def _identity(**overrides):
    values = {"lattice": DEEP, "mg_grid": GRID, "dof": (12, 12, 12),
              "dtype": "c64", "mass": 0.17, "csw": 0.0, "seed": 42,
              "nv_iters": 2}
    values.update(overrides)
    return cache_module.mg_identity(**values)


# ---------------------------------------------------------------------------
# hierarchy derivation
# ---------------------------------------------------------------------------
def test_chain_halves_every_extent_once_per_level():
    chain = native._mg_level_chain(DEEP, GRID, 3, (12, 12, 12))
    assert [tuple(level["dims"]) for level in chain] == [
        (16, 16, 16, 16), (8, 8, 8, 8), (4, 4, 4, 4)]
    # The odd-Schur fine field halves time once before the block grid applies.
    assert chain[0]["dims"][3] == DEEP[3] // 2
    assert [level["block"] for level in chain] == [
        None, (2, 2, 2, 2), (2, 2, 2, 2)]
    assert [level["parent"] for level in chain] == [None, 0, 1]
    assert [level["E"] for level in chain] == [12, 12, 12]


def test_chain_honours_a_non_default_block_grid_for_level_one():
    chain = native._mg_level_chain((16, 16, 16, 32), (2, 2, 2, 4), 2,
                                   (12, 8))
    assert tuple(chain[1]["dims"]) == (8, 8, 8, 4)
    assert tuple(chain[1]["block"]) == (2, 2, 2, 4)
    assert chain[1]["E"] == 8


def test_chain_stops_early_and_says_which_level_it_cannot_reach():
    extents, blocked = native._mg_level_extents(SHALLOW, GRID)
    # Levels 0 and 1 exist (8^3x8 odd-Schur, then 4^4); level 2 would be 2^4.
    assert extents == ((8, 8, 8, 8), (4, 4, 4, 4))
    assert blocked == {"level": 2, "axis": 0, "extent": 4, "half": 2,
                       "reason": "small"}
    # The same call with a depth limit stops at the limit, not at a diagnosis.
    assert native._mg_level_extents(DEEP, GRID, 2) == (
        ((16, 16, 16, 16), (8, 8, 8, 8)), None)


def test_an_unrepresentable_depth_names_the_level_and_the_extent():
    with pytest.raises(ValueError) as raised:
        native._mg_level_chain(SHALLOW, GRID, 3, (12, 12, 12))
    message = str(raised.value)
    assert "level 2" in message and "axis 0" in message
    assert "extent 2" in message and "coarsest extent 4" in message
    assert "at most 2 level(s)" in message


def test_an_odd_level_extent_is_refused_by_name():
    """An odd extent has no integer half; it must not be floored silently.

    18 on X gives an odd 9 at level 1, which the halving chain cannot split.
    The message must name the level and the extent rather than report a
    too-small block it would never have had.
    """
    with pytest.raises(ValueError) as raised:
        native._mg_level_chain((18, 16, 16, 32), GRID, 3, (12, 12, 12))
    message = str(raised.value)
    assert "level 2 (axis 0) cannot be built" in message
    assert "level 1 is 9 on that axis" in message
    assert "odd extent has no integer half" in message
    assert "at most 2 level(s)" in message


def test_the_chain_rejects_an_odd_time_extent():
    with pytest.raises(ValueError, match="even time extent"):
        native._mg_level_chain((16, 16, 16, 17), GRID, 2, (12, 12))


def test_the_chain_rejects_a_block_grid_that_does_not_divide_the_fine_field():
    with pytest.raises(ValueError, match="not divisible by mg_grid 3"):
        native._mg_level_chain((16, 16, 16, 32), (3, 2, 2, 2), 2, (12, 12))


def test_the_chain_requires_one_component_count_per_level():
    with pytest.raises(ValueError, match="one dof component count per level"):
        native._mg_level_chain(DEEP, GRID, 3, (12, 12))


def test_the_chain_keeps_the_solver_level_limit():
    with pytest.raises(ValueError, match="accepts 1..5 levels"):
        native._mg_level_chain(DEEP, GRID, 6, (12,) * 6)


def test_mg_1_and_2_never_consult_the_chain():
    """mg-1/mg-2 must keep today's single-transition behaviour.

    ``_options`` derives the hierarchy only for a depth above 2, so an input the
    chain would refuse (here a ``mg_grid`` that does not divide the fine field)
    still resolves at the option layer.  The asset path validates the same
    geometry itself (``pyqcu.tools._legacy_mg_cache.legacy_mg_geometry``), and
    ``_mg_case`` derives the chain for any depth above 1, so the case still
    fails - with the geometry error instead of the solver's own abort.
    """
    option = _option(lattice=SHALLOW, mg_levels=2, dof=(12, 12),
                     mg_grid=(3, 2, 2, 2))
    assert option.mg_levels == 2


# ---------------------------------------------------------------------------
# per-transition shapes
# ---------------------------------------------------------------------------
def test_the_first_transition_keeps_exactly_the_cache_shapes():
    identity = _identity()
    chain = native._mg_level_chain(DEEP, GRID, 3, (12, 12, 12))
    shapes = native._mg_level_shapes(identity, chain)
    canonical = cache_module.asset_shapes(identity)
    assert shapes[0] == canonical
    # mg-2 asks for one transition, so its shape list is that single mapping.
    assert len(native._mg_level_shapes(
        _identity(dof=(12, 12)),
        native._mg_level_chain(SHALLOW, GRID, 2, (12, 12)))) == 1


def test_a_deeper_transition_scales_every_axis_by_its_own_factor():
    identity = _identity()
    chain = native._mg_level_chain(DEEP, GRID, 3, (12, 12, 12))
    shapes = native._mg_level_shapes(identity, chain)
    lonv = shapes[1]["lonv"]
    # (E, E_parent, X, bx, Y, by, Z, bz, T, bt) of the 8^3x8 -> 4^4 transition.
    assert lonv == (12, 12, 4, 2, 4, 2, 4, 2, 4, 2)
    assert shapes[1]["sit"] == (12, 12, 4, 4, 4, 4)
    assert shapes[1]["hnn"] == (2, 4, 12, 12, 4, 4, 4, 4)
    assert shapes[1]["hdg"] == (2, 2, 6, 12, 12, 4, 4, 4, 4)
    assert set(shapes[1]) == set(native.MG_TRANSITION_ASSETS)


# ---------------------------------------------------------------------------
# ABI placement
# ---------------------------------------------------------------------------
def test_each_level_gets_its_own_params_block_and_pointer_quartet():
    define = _define()
    option = _option(mg_levels=3, dof=(12, 12, 8), coarse_max_iter=200,
                     restart=5, atol=1.0e-5, coarse_tol_factor=1.0e4)
    controls = _SyntheticControls(define)
    chain = native._mg_level_chain(option.lattice, option.mg_grid, 3,
                                   option.dof)
    assets = [
        _synthetic_transition((12, 12, 8, 2, 8, 2, 8, 2, 8, 2), 1000),
        # Level 2 has 8 coarse components over 12 parent components.
        _synthetic_transition((8, 12, 4, 2, 4, 2, 4, 2, 4, 2), 2000),
    ]
    parameters = native._configure_legacy_mg(option, controls, assets, chain)

    assert controls.params[define._MG_NUM_LEVEL_] == 3
    assert controls.params[define._MG_USE_DEFLATE_] == 1
    assert controls.set_ptrs[native.LEGACY_LAYOUT_SLOT] == \
        native.LEGACY_LAYOUT_ODD_SCHUR
    for level, expected in ((1, (12, 8, 8, 8, 8, 1000)),
                            (2, (8, 4, 4, 4, 4, 2000))):
        base = native.MG_LEVEL_BASE + (level - 1) * native.MG_LEVEL_STRIDE
        E, *coarse, address = expected
        assert controls.params[base + 0] == E
        assert controls.params[base + 1:base + 5] == coarse
        assert controls.params[base + native.MG_LEVEL_MAX_ITER] == 200
        assert controls.params[base + native.MG_LEVEL_DATA_TYPE] == \
            controls.params[define._DATA_TYPE_]
        assert controls.params[base + native.MG_LEVEL_NUM_RESTART] == 5
        slot = native.SLOT_LIMIT + 4 * (level - 1)
        assert controls.set_ptrs[slot:slot + 4] == [
            address + offset for offset in range(4)]
        # The tolerance sits in argv[3 + (level-1)] with the 3^(level-1) ramp.
        tolerance = option.atol * option.coarse_tol_factor * (3 ** (level - 1))
        assert controls.argv[native.ARGV_MG_LEVEL_BASE + level - 1] == \
            pytest.approx(tolerance)
        record = parameters["mg_level_parameters"][level - 1]
        assert record["level"] == level and record["E"] == E
        assert record["coarse"] == coarse and record["slot"] == slot
        assert record["atol"] == pytest.approx(tolerance)
    # Level 3's block was never written: a truncated hierarchy would read as a
    # zero-extent level instead of an error.
    assert controls.params[native.MG_LEVEL_BASE + 2 * native.MG_LEVEL_STRIDE] \
        == 0
    assert parameters["mg_layout"] == "odd-schur"


def test_mg_2_is_wired_exactly_as_before():
    """One asset set, one block at 19, one quartet at 30 - unchanged."""
    define = _define()
    option = _option(lattice=SHALLOW, mg_levels=2, dof=(12, 12))
    controls = _SyntheticControls(define)
    assets = [_synthetic_transition((12, 12, 4, 2, 4, 2, 4, 2, 4, 2), 4000)]
    parameters = native._configure_legacy_mg(option, controls, assets)

    assert controls.params[define._MG_NUM_LEVEL_] == 2
    assert controls.params[native.MG_LEVEL_BASE:
                           native.MG_LEVEL_BASE + 5] == [12, 4, 4, 4, 4]
    assert controls.params[native.MG_LEVEL_BASE
                           + native.MG_LEVEL_MAX_ITER] == \
        option.coarse_max_iter
    assert controls.params[native.MG_LEVEL_BASE
                           + native.MG_LEVEL_NUM_RESTART] == option.restart
    assert controls.argv[native.ARGV_MG_LEVEL_BASE] == pytest.approx(
        option.atol * option.coarse_tol_factor)
    assert controls.argv[native.ARGV_MG_LEVEL_BASE + 1] == 0.0
    assert controls.set_ptrs[native.SLOT_LIMIT:native.SLOT_LIMIT + 4] == [
        4000, 4001, 4002, 4003]
    assert controls.set_ptrs[native.SLOT_LIMIT + 4:
                             native.SLOT_LIMIT + 8] == [0, 0, 0, 0]
    assert parameters["mg_level_parameters"] == [{
        "level": 1, "E": 12, "coarse": [4, 4, 4, 4],
        "slot": native.SLOT_LIMIT,
        "coarse_max_iter": option.coarse_max_iter,
        "restart": option.restart,
        "atol": pytest.approx(option.atol * option.coarse_tol_factor),
    }]


def test_wiring_rejects_an_asset_that_contradicts_the_derived_chain():
    define = _define()
    option = _option(mg_levels=3, dof=(12, 12, 12))
    controls = _SyntheticControls(define)
    chain = native._mg_level_chain(option.lattice, option.mg_grid, 3,
                                   option.dof)
    # Level 2 claims 8 coarse components where the chain derives 12.
    assets = [_synthetic_transition((12, 12, 8, 2, 8, 2, 8, 2, 8, 2), 1000),
              _synthetic_transition((8, 8, 4, 2, 4, 2, 4, 2, 4, 2), 2000)]
    with pytest.raises(ValueError, match="does not match the derived hierarchy"):
        native._configure_legacy_mg(option, controls, assets, chain)


def test_wiring_needs_one_asset_set_per_transition():
    define = _define()
    option = _option(mg_levels=3, dof=(12, 12, 12))
    controls = _SyntheticControls(define)
    with pytest.raises(ValueError, match="needs 2 coarse transitions"):
        native._configure_legacy_mg(option, controls, [])


# ---------------------------------------------------------------------------
# asset provenance
# ---------------------------------------------------------------------------
def test_the_row_records_the_chain_and_where_each_transition_came_from():
    chain = native._mg_level_chain(DEEP, GRID, 3, (12, 12, 12))
    shapes = native._mg_level_shapes(_identity(), chain)
    record = native._mg_chain_record(chain, shapes, ["native", "unbuilt"])
    assert [entry["level"] for entry in record] == [1, 2]
    assert [entry["source"] for entry in record] == ["native", "unbuilt"]
    assert [entry["coarse"] for entry in record] == [
        [8, 8, 8, 8], [4, 4, 4, 4]]
    assert record[0]["E"] == 12 and record[1]["E_parent"] == 12
    # The lonv shape of each transition is carried so a reader can see the block.
    assert record[0]["shape"] == list(shapes[0]["lonv"])
    assert record[1]["shape"] == list(shapes[1]["lonv"])


def test_a_transition_without_a_root_is_refused_by_name():
    """Every transition past the first must name the flag that would supply it.

    A deeper hierarchy is legitimate now that the probe builder exists
    (``pyqcu.testing.simple.mg_transition``), but the port does not build it: the
    portable package ships ``pyqcu/test`` and the builder is a checkout-only
    developer tool, so a run must consume roots that already exist.  Refusing by
    transition number is what keeps a missing root from silently becoming a
    two-level hierarchy.
    """
    option = _option(mg_levels=3, dof=(12, 12, 12))
    second = native._missing_transition_root(option, 2, 3)
    assert isinstance(second, ValueError)
    message = str(second)
    assert "mg_levels=3 needs a cache root for multigrid transition 2" in message
    assert "--coarse-cache-level number 1" in message
    assert "level 1 -> level 2" in message
    # The first transition names the other flag, so a reader is never told to pass
    # the flag that belongs to a different transition.
    first = str(native._missing_transition_root(option, 1, 3))
    assert "--cache-dir" in first and "--coarse-cache-level" not in first.split(
        "the gauge-built")[0]


class _Untouchable:
    """Fails if the refusal reads anything: it must run before any building."""

    def __getattr__(self, name: str):
        raise AssertionError(
            f"the mg-3 refusal touched {name!r}; it must happen before the "
            f"asset resolution allocates or builds anything")


def test_the_run_path_refuses_a_depth_without_its_roots():
    """``_mg_case`` resolves assets through here: refuse, and touch nothing.

    This is the production refusal, not a helper.  A keyed-but-empty resolution
    must raise before the controls are read, so that a depth whose roots were not
    supplied cannot reach the null-vector/stencil build with a ``None`` control
    block.  ``_Untouchable`` reports any read as "the refusal touched ...", and a
    regression that let such a case reach the ABI would abort the MPI process
    instead (``api::guarded``), so this one has to fail loudly.
    """
    option = _option(mg_levels=3, dof=(12, 12, 12))
    nothing = _Untouchable()
    with pytest.raises(ValueError, match="needs a cache root for multigrid "
                                         "transition 2"):
        native._resolve_mg_assets(option, nothing, nothing, nothing, nothing,
                                  nothing, nothing)


def test_a_three_level_hierarchy_consumes_exactly_two_roots(tmp_path):
    """`depth` levels means `depth - 1` transitions, and no more.

    Transition 1 is ``--cache-dir``; the ones owed a ``--coarse-cache-level`` are
    2..depth-1.  Asking for transition ``depth`` is the off-by-one that made a
    three-level run refuse with "needs a cache root for multigrid transition 3
    (level 2 -> level 3)" while the caller had supplied every root that exists, so
    the count is asserted directly rather than through the message.
    """
    roots = []
    for index in range(2):
        path = tmp_path / f"l{index}.h5"
        path.write_text("placeholder", encoding="utf-8")
        roots.append(str(path))

    three = _option(mg_levels=3, dof=(12, 12, 12),
                    coarse_cache_level=(roots[0],))
    resolved = native._mg_transition_roots(three, 3)
    assert len(resolved) == 2, "three levels have exactly two transitions"
    assert resolved[0] is None, "transition 1 is --cache-dir, not a level root"
    assert resolved[1] is not None and resolved[1].name == "l0.h5"

    # A fourth level owes one more root and must ask for exactly one more.  It
    # needs a lattice that holds four levels: 16^3x32 halves to extent 2 at level
    # 3, which the chain refuses, so the four-level case uses 32^3x64.
    four = _option(lattice=(32, 32, 32, 64), mg_levels=4,
                   dof=(12, 12, 12, 12), coarse_cache_level=tuple(roots))
    resolved4 = native._mg_transition_roots(four, 4)
    assert len(resolved4) == 3
    assert [path.name for path in resolved4[1:]] == ["l0.h5", "l1.h5"]
    with pytest.raises(ValueError, match="transition 3"):
        native._mg_transition_roots(three, 4)
