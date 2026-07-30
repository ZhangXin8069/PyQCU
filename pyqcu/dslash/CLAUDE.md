# CLAUDE.md — pyqcu.dslash

Wilson and Clover Dirac operators — the core linear operators of lattice QCD.

## Files

| File | Purpose |
|------|---------|
| `_wilson.py` | Wilson hopping term D_w: spatial derivative with γ_μ matrices and parallel transport |
| `_clover.py` | Clover term (chromo-magnetic field strength F_{μν} contribution) |
| `_operator.py` | Composed Dirac operator = hopping + sitting, with even-odd preconditioning and coarse-grid construction |

## Key Functions Exported

### Wilson (`_wilson.py`)
- `give_wilson(src, U, kappa)` — full Wilson operator
- `give_wilson_eo` / `give_wilson_oe` — even-odd preconditioned variants
- `give_hopping_plus(U, dir)` / `give_hopping_minus(U, dir)` — directional hopping matrices M_μ^+/M_μ^- for the 4 spacetime directions

### Clover (`_clover.py`)
- `make_clover(U)` — build the clover term (field strength tensor F_{μν})
- `add_I(clover)` / `cut_I(clover)` — add/remove identity from clover matrix
- `inverse(clover)` — invert clover matrix (use batched `torch.linalg.inv`, NOT per-site loop)
- `give_clover_ee` / `give_clover_oo` — split clover into even/odd parity blocks

### Operator (`_operator.py`)
- `hopping` class — precomputes M_plus/M_minus for all 4 directions; handles MPI halo exchange for fermion boundaries on each matvec
- `sitting` class — clover term with optional even/odd split and inverse precomputation
- `operator` class — composes hopping + sitting; provides `matvec()`, `matvec_eo()`, `matvec_oe()`, `matvec_parity()`

## Coarse-Grid Operator (Galerkin Projection)

When `fine_hopping`, `fine_sitting`, and `local_ortho_null_vecs` are all provided, the operator builds a coarse-grid operator via Galerkin projection P^T D_fine P. For each null-space basis vector:
1. Prolong a delta-source from coarse to fine grid
2. Apply the fine hopping operator (± directions)
3. Restrict the result back to coarse grid
4. Also project the fine sitting operator

## Anti-Patterns

- **Never** use `self.sitting` (an object) as a truthy check; use `self.sitting.clover_term is not None`
- **Never** loop `torch.linalg.inv` per-site; use batched inversion
