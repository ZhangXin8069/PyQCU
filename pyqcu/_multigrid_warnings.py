"""Backend-neutral warnings for MultiGrid hierarchy configuration."""

from __future__ import annotations

import warnings
from math import prod
from typing import Sequence


LEGACY_VALIDATED_LEVELS = 2
STRICT_VALIDATED_LEVELS = 3


class MultigridDepthWarning(UserWarning):
    """The requested hierarchy is deeper than the validated default path."""


def warn_excessive_multigrid_levels(
        levels: int, *, variant: str, backend: str,
        context: str | None = None,
        lattice: Sequence[int] | None = None,
        block: Sequence[int] | Sequence[Sequence[int]] = (2, 2, 2, 2),
        process_grid: Sequence[int] | None = None,
        minimum_extent: int = 4,
        minimum_coarse_volume: int = 1024) -> bool:
    """Warn when depth or coarse geometry exceeds validated bounds.

    Legacy-MultiGrid is the production default and is validated at two
    levels. Additional Legacy levels remain experimental and have shown no
    iteration reduction at three levels, with correctness failures in
    distributed three-level and single-rank four-level tests. Strict-
    MultiGrid is the explicit comparison path and supports three levels;
    deeper Strict hierarchies remain experimental.
    """

    depth = int(levels)
    variant_name = str(variant).strip().lower().replace("_", "-")
    threshold = (
        LEGACY_VALIDATED_LEVELS if variant_name == "legacy" else
        STRICT_VALIDATED_LEVELS)
    reasons: list[str] = []
    if depth > threshold:
        reasons.append(f"depth {depth} exceeds validated depth {threshold}")
    coarse_shape: tuple[int, int, int, int] | None = None
    next_shape: tuple[int, int, int, int] | None = None
    if depth > 1 and lattice is not None:
        fine = tuple(int(value) for value in lattice)
        raw_blocks = list(block)
        if raw_blocks and isinstance(raw_blocks[0], Sequence) and not isinstance(
                raw_blocks[0], (str, bytes)):
            width_sets = [tuple(int(value) for value in item)
                          for item in raw_blocks]  # type: ignore[arg-type]
        else:
            width_sets = [tuple(int(value) for value in raw_blocks)] * (depth - 1)
        if len(fine) != 4 or any(len(widths) != 4 for widths in width_sets):
            raise ValueError("lattice and every block must contain four extents")
        if len(width_sets) != depth - 1:
            raise ValueError("block transitions must equal levels - 1")
        if any(value <= 0 for value in (*fine, *(
                value for widths in width_sets for value in widths))):
            raise ValueError("lattice and block extents must be positive")
        if process_grid is not None:
            grid = tuple(int(value) for value in process_grid)
            if len(grid) != 4 or any(value <= 0 for value in grid):
                raise ValueError("process_grid must contain four positive extents")
            if any(fine[axis] % grid[axis] for axis in range(4)):
                reasons.append(
                    f"lattice {fine} is not divisible by process grid {grid}")
                fine = tuple(max(1, fine[axis] // grid[axis]) for axis in range(4))
            else:
                fine = tuple(fine[axis] // grid[axis] for axis in range(4))
        # QCU stores the parity-compressed time axis before coarsening.
        shape = [fine[0], fine[1], fine[2], fine[3] // 2]
        divisible = True
        for widths in width_sets:
            if any(shape[axis] % widths[axis] for axis in range(4)):
                divisible = False
                shape = [
                    max(1, shape[axis] // widths[axis]) for axis in range(4)
                ]
            else:
                shape = [shape[axis] // widths[axis] for axis in range(4)]
        coarse_shape = tuple(shape)  # type: ignore[assignment]
        if not divisible:
            reasons.append(
                f"lattice/process-grid geometry is not divisible by block {widths}")
        if any(extent < minimum_extent for extent in coarse_shape):
            reasons.append(
                f"coarsest shape {coarse_shape} has an extent below "
                f"{minimum_extent}")
        if prod(coarse_shape) < minimum_coarse_volume:
            reasons.append(
                f"coarsest volume {prod(coarse_shape)} is below "
                f"{minimum_coarse_volume}")
        last_widths = width_sets[-1]
        next_shape = tuple(max(1, extent // last_widths[axis])
                           for axis, extent in enumerate(coarse_shape))
        if any(extent < minimum_extent for extent in next_shape):
            reasons.append(
                f"next coarsening {next_shape} would fall below extent "
                f"{minimum_extent}")
    if not reasons:
        return False
    location = f" ({context})" if context else ""
    if variant_name == "legacy":
        guidance = (
            "Legacy-MultiGrid defaults to 2 levels; deeper Legacy hierarchies "
            "are experimental and must pass native plus independent residual "
            "gates before use.")
    else:
        guidance = (
            "Strict-MultiGrid validates 3 levels; deeper Strict hierarchies "
            "are experimental and require an explicit independent residual gate.")
    geometry = (
        "" if coarse_shape is None else
        f" Geometry: coarsest={coarse_shape}, next={next_shape}.")
    warnings.warn(
        f"{backend} {variant_name} MultiGrid levels={depth}{location}: "
        f"{'; '.join(reasons)}. {guidance}{geometry}",
        MultigridDepthWarning,
        stacklevel=3,
    )
    return True


__all__ = [
    "LEGACY_VALIDATED_LEVELS",
    "MultigridDepthWarning",
    "STRICT_VALIDATED_LEVELS",
    "warn_excessive_multigrid_levels",
]
