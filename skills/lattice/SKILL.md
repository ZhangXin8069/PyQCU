---
name: lattice
description: pyqcu.lattice 目录的完整生成 skill：gamma/Gell-Mann 矩阵、SU(3) 检查、规范场生成与 Ward 负索引约定。
---
# CLAUDE.md — pyqcu.lattice

Lattice QCD fundamentals: gamma matrices, Gell-Mann matrices, SU(3) group utilities, and gauge field generation.

## Module-level Data (computed at import time, on CPU, complex64)

- **`gamma`** — 4×4×4 gamma matrices γ₀, γ₁, γ₂, γ₃ in the Dirac-Pauli representation (γ₀ anti-hermitian, γ_i hermitian). Shape `[4, 4, 4]`.
- **`gamma_5`** — γ₅ = γ₀γ₁γ₂γ₃. Shape `[4, 4]`.
- **`gamma_gamma`** — six γ_μ γ_ν products: [γ_x,γ_y], [γ_x,γ_z], [γ_x,γ_t], [γ_y,γ_z], [γ_y,γ_t], [γ_z,γ_t]. Shape `[6, 4, 4]`. Used as σ_{μν} matrices in the clover term.
- **`I`** — 4×4 identity matrix (complex64)
- **`minus_I`** — −I (precomputed)
- **`gell_mann`** — eight Gell-Mann matrices λ₁…λ₈ (SU(3) generators, traceless hermitian). Shape `[8, 3, 3]`. λ₁,λ₄,λ₆ are real; λ₂,λ₅,λ₇ are i×real.

## Ward Index Convention

Ward indices use **negative indexing** because spacetime dimensions are always the last four axes (`...xyzt` layout):

```python
wards['x'] = -4    # last 4th axis
wards['y'] = -3    # last 3rd axis
wards['z'] = -2    # last 2nd axis
wards['t'] = -1    # last axis
wards['t_p'] = -1  # parity-split temporal (same index as t)
```

This makes indexing robust regardless of prefix dimensions (spin, color, parity, etc.).

### Ward key lists
- **`ward_keys`** = `['x', 'y', 'z', 't']` — standard 4D directions
- **`ward_p_keys`** = `['x', 'y', 'z', 't_p']` — parity-aware (t_p for temporal with even/odd mask)
- **`ward_ward_keys`** = `['xy', 'xz', 'xt', 'yz', 'yt', 'zt']` — 6 plane directions for clover

### Ward mapping for gamma_gamma indexing
```python
ward_wards['xy'] = {'mu': -4, 'nu': -3, 'ward': -6}  # gamma_gamma index 0
ward_wards['xz'] = {'mu': -4, 'nu': -2, 'ward': -5}  # gamma_gamma index 1
# ... etc.
```

## Exported Functions

### `check_su3(U, tol=1e-3, verbose=True) → bool`

Verifies SU(3) properties of a gauge field:
1. **Unitarity:** U^H U ≈ I (uses `_torch.allclose` with `atol=tol`)
2. **Determinant:** det(U) ≈ 1 (uses raw `torch.linalg.det` — no NPU equivalent needed)
3. **Minor identities:** Each column is the cross product of the other two (with conjugation)

Returns `True` only if all three checks pass.

### `generate_gauge_field(U, sigma=0.1, seed=None, verbose=False) → torch.Tensor`

Generates random SU(3) gauge links via exponential map:
1. Sample 8 random Gaussian coefficients per site per direction
2. Form Hermitian matrix H = Σ_a c_a λ_a
3. Compute U = exp(i · σ · H) via `torch.matrix_exp`
4. Rearrange to `[3, 3, 4, Lx, Ly, Lz, Lt]` layout

Writes result in-place into `U`. Returns `U`.

### `give_support_multi() → bool`

Returns `True` if `MPI.COMM_WORLD.size > 1` (multi-process run).

## Data Layout

Gauge field `U`: shape `[3, 3, 4, Lx, Ly, Lz, Lt]` = `[color, color, direction, x, y, z, t]`

## Other Module-Level Data

In addition to the matrix data above, the module imports `mpi4py.MPI`, `pyqcu.cann as _torch`, and raw `torch` (for `torch.linalg.det` and `torch.matrix_exp` which have no NPU wrappers).
