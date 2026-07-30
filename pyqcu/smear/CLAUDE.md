# CLAUDE.md — pyqcu.smear

Gauge field smearing — spatial smoothing of gauge links to reduce UV noise.

## Files

| File | Purpose |
|------|---------|
| `_stout.py` | Stout smearing algorithm |

## Key Function

- **`stout_smear(U, rho, nstep, verbose)`** — apply nstep iterations of stout smearing with parameter rho

## Fixed Bug

The `nstep>1` loop previously did not update `U` between steps, effectively degrading multi-step smearing to 1 step. Ensure each iteration feeds its output as the input for the next step.

## Data Layout

Gauge field: `[3, 3, 4, Lx, Ly, Lz, Lt]` = `[color, color, direction, x, y, z, t]`
