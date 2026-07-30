# CLAUDE.md — pyqcu.lattice

Lattice QCD fundamentals: gamma matrices, Gell-Mann matrices, SU(3) group utilities, and gauge field generation.

## Module-level Data (computed at import time)

- **`gamma`** — 4×4×4 gamma matrices γ₀, γ₁, γ₂, γ₃ (shape `[4, 4, 4]`, on CPU)
- **`gamma_5`** — γ₅ = γ₀γ₁γ₂γ₃ (shape `[4, 4]`, on CPU)
- **`gamma_gamma`** — six γ_μ γ_ν commutator products: `[γ_x,γ_y], [γ_x,γ_z], [γ_x,γ_t], [γ_y,γ_z], [γ_y,γ_t], [γ_z,γ_t]` (shape `[6, 4, 4]`)
- **`I`** — 4×4 identity matrix
- **`gell_mann`** — eight Gell-Mann matrices λ₁…λ₈ (shape `[8, 3, 3]`), SU(3) generators

## Ward Index Convention

Ward indices use **negative indexing** because spacetime dimensions are always the last four axes of any tensor (`...xyzt` layout). Example: `wards['x'] = -4`, `wards['t'] = -1`. This makes indexing robust regardless of prefix dimensions (spin, color, parity).

## Key Functions

- **`check_su3(U, tol, verbose)`** — verifies unitarity (U U^† = I), determinant = 1, and minor identities for an SU(3) gauge field
- **`generate_gauge_field(U, seed, sigma, verbose)`** — generates random SU(3) gauge links via exponential map of random Gell-Mann combinations

## Data Layout

Gauge field `U`: shape `[3, 3, 4, Lx, Ly, Lz, Lt]` = `[color, color, direction, x, y, z, t]`
