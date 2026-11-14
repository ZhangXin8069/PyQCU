"""Unit tests for the uniform (balanced) 4D process-grid decomposition."""

from __future__ import annotations

import itertools
import math

import pytest

from pyqcu.testing.simple._common import choose_balanced_grid
from pyqcu.testing.simple._runner import default_kupl_grid


def _local(lattice, grid):
    return tuple(lattice[axis] // grid[axis] for axis in range(4))


def _assert_valid(lattice, grid, size):
    assert grid is not None
    assert math.prod(grid) == size
    assert all(lattice[axis] % grid[axis] == 0 for axis in range(4))
    local = _local(lattice, grid)
    assert all(extent > 0 and extent % 2 == 0 for extent in local)
    # grid_t must divide the parity-compressed time extent, otherwise a coarse
    # block (--mg-grid) cannot divide the local time axis.
    assert (lattice[3] // 2) % grid[3] == 0
    return local


def _reference_candidates(lattice, size):
    """Brute-force reference for the ranking used by choose_balanced_grid."""
    spans = [[value for value in range(1, lattice[axis] + 1)
              if lattice[axis] % value == 0 and
              (lattice[axis] // value) % 2 == 0] for axis in range(4)]
    spans[3] = [value for value in spans[3] if (lattice[3] // 2) % value == 0]
    found = []
    for grid in itertools.product(*spans):
        if math.prod(grid) != size:
            continue
        local = _local(lattice, grid)
        found.append((max(local) / min(local), max(local), min(local), grid))
    return found


@pytest.mark.parametrize(("lattice", "size", "expected"), [
    # 32³×12: even time is only 12, so t carries few ranks.
    ((32, 32, 32, 12), 32, (2, 4, 4, 1)),
    ((32, 32, 32, 12), 64, (4, 4, 4, 1)),
    # 24³×64: 64 is the long axis, so the balanced grid splits t as well.
    ((24, 24, 24, 64), 32, (2, 2, 2, 4)),
    ((24, 24, 24, 64), 64, (2, 2, 2, 8)),
    # 32³×64: the reference aspect-ratio-1.0 decomposition.
    ((32, 32, 32, 64), 32, (2, 2, 2, 4)),
    ((32, 32, 32, 64), 64, (2, 2, 2, 8)),
])
def test_reference_decompositions(lattice, size, expected):
    grid = choose_balanced_grid(lattice, size)
    assert grid == expected
    local = _assert_valid(lattice, grid, size)
    assert max(local) / min(local) == min(
        candidate[0] for candidate in _reference_candidates(lattice, size))


def test_single_rank_is_the_trivial_grid():
    assert choose_balanced_grid((24, 24, 24, 64), 1) == (1, 1, 1, 1)
    assert default_kupl_grid((24, 24, 24, 64), 1) == (1, 1, 1, 1)


def test_odd_lattice_has_no_even_local_grid():
    assert choose_balanced_grid((24, 24, 24, 63), 4) is None
    assert default_kupl_grid((7, 7, 7, 8), 2) is None


def test_unreachable_rank_count_returns_none():
    # (2,2,2,4) admits at most two ranks with even local extents.
    assert choose_balanced_grid((2, 2, 2, 4), 8) is None


def test_balanced_choice_beats_the_legacy_y_first_split():
    """The old heuristic produced 1×4×1×1 for L24³×64 at 4 ranks."""
    lattice, size = (24, 24, 24, 64), 4
    grid = choose_balanced_grid(lattice, size)
    assert grid == (1, 1, 1, 4)
    legacy = (1, 4, 1, 1)
    new_local, legacy_local = _local(lattice, grid), _local(lattice, legacy)
    assert (max(new_local) / min(new_local) <
            max(legacy_local) / min(legacy_local))


def test_axis_preference_breaks_aspect_ratio_ties():
    # All four single-axis splits of 8⁴ into two even halves tie on aspect 2.
    assert choose_balanced_grid((8, 8, 8, 8), 2) == (1, 1, 1, 2)
    assert choose_balanced_grid(
        (8, 8, 8, 8), 2, axis_preference=(3, 0, 1, 2)) == (1, 1, 2, 1)


def test_result_is_always_valid_and_deterministic():
    for lattice in ((8, 8, 8, 16), (24, 24, 24, 64), (32, 32, 32, 12),
                    (16, 16, 16, 32), (2, 2, 2, 4)):
        for size in (1, 2, 3, 4, 6, 8, 16, 32):
            grid = choose_balanced_grid(lattice, size)
            if grid is None:
                continue
            _assert_valid(lattice, grid, size)
            assert grid == choose_balanced_grid(lattice, size)


def test_helper_is_pure():
    lattice, size = [24, 24, 24, 64], 32
    first = choose_balanced_grid(lattice, size)
    second = choose_balanced_grid(tuple(lattice), size)
    assert first == second == (2, 2, 2, 4)
    assert lattice == [24, 24, 24, 64]


def test_invalid_arguments_are_rejected():
    with pytest.raises(ValueError):
        choose_balanced_grid((8, 8, 8), 2)
    with pytest.raises(ValueError):
        choose_balanced_grid((8, 8, 8, 8), 0)
    with pytest.raises(ValueError):
        choose_balanced_grid((8, 8, 8, 8), 2, axis_preference=(0, 0, 1, 2))


def test_mpi_default_grid_matches_the_shared_ranking():
    from pyqcu.testing.simple.mpi._common import default_grid

    lattice, size = (24, 24, 24, 64), 32
    assert default_grid(lattice, size) == choose_balanced_grid(lattice, size)
    with pytest.raises(ValueError):
        default_grid((2, 2, 2, 4), 8)
