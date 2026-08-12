# CLAUDE.md — pyqcu.dslash

Wilson and Clover Dirac operators — the core linear operators of lattice QCD.

## Files

| File | Purpose |
|------|---------|
| `_wilson.py` | Wilson hopping term D_w: spatial derivative with γ_μ matrices and parallel transport. Has per-module `force_use_npu` flag. |
| `_clover.py` | Clover term (chromo-magnetic field strength F_{μν} contribution). Four-plaquette clover construction with MPI boundary exchange. |
| `_operator.py` | Composed Dirac operator = hopping + sitting, with even-odd preconditioning, coarse-grid Galerkin projection, and MPI halo exchange. |

## Exported API

### Wilson (`_wilson.py`)

| Function | Purpose |
|----------|---------|
| `give_wilson(src, U, kappa, u_0, with_I, verbose)` | Full Wilson operator D_w = I − κ/u_0 · Σ_μ [(1−γ_μ)U_μ δ_{x+μ,y} + (1+γ_μ)U^†_{x−μ,μ} δ_{x−μ,y}] |
| `give_wilson_eo(src_o, U_eo, kappa, u_0, verbose)` | Even-odd Wilson (even dest from odd src) |
| `give_wilson_oe(src_e, U_eo, kappa, u_0, verbose)` | Odd-even Wilson (odd dest from even src) |
| `give_hopping_plus(ward, U, kappa, u_0, verbose)` | Directional hopping matrix M_μ^+ = −κ/u_0 · (I−γ_μ) ⊗ U_μ, shape `[12, 12, Lx, Ly, Lz, Lt]` |
| `give_hopping_minus(ward, U, U_head, kappa, u_0, verbose)` | Directional hopping matrix M_μ^- = −κ/u_0 · (I+γ_μ) ⊗ U^†_{x−μ,μ}, with halo exchange via `U_head` |
| `give_wilson_plus(ward, src, hopping, src_tail, parity, verbose)` | Apply hopping_plus to src: einsum("Eexyzt,exyzt→Exyzt", M_plus, rolled_src). Handles MPI `src_tail` boundary and parity masking. |
| `give_wilson_minus(ward, src, hopping, src_head, parity, verbose)` | Apply hopping_minus to src: einsum("Eexyzt,exyzt→Exyzt", M_minus, rolled_src). Handles MPI `src_head` boundary and parity masking. |

The eo/oe variants use `ward_p_keys` (x, y, z, t_p) — the `t_p` direction handles parity-split temporal hopping with even/odd masks.

### Clover (`_clover.py`)

| Function | Purpose |
|----------|---------|
| `make_clover(U, kappa, u_0, support_parallel, verbose)` | Build clover term from four-plaquette F_{μν} construction with 12 shifted gauge links per μν pair. Returns `[4,3,4,3,Lx,Ly,Lz,Lt]`. |
| `add_I(clover_term, verbose)` | Add identity: M = I + clover_term. Reshapes to `[12,12,N]`, adds I, reshapes back. |
| `cut_I(clover_term, verbose)` | Remove identity: M = clover_term − I |
| `inverse(clover_term, verbose)` | Batched 12×12 matrix inversion via `torch.linalg.inv` (NOT per-site loop!) |
| `give_clover(src, clover_term, verbose)` | Apply clover term: einsum("SCscxyzt,scxyzt→SCxyzt", clover, src) |
| `give_clover_ee(src_e, clover_e)` | Even-even clover application (delegates to `give_clover`) |
| `give_clover_oo(src_o, clover_o)` | Odd-odd clover application (delegates to `give_clover`) |

**Clover coefficient note:** Uses `_clover_factor = −0.125 · κ/u_0`. Standard convention with c_sw=1 gives −κ/(16·u_0). The factor of 2 may be due to the anti-hermitian part convention. Cross-validate against QUDA/Chroma before changing.

When `support_parallel=True`, `make_clover` performs MPI halo exchange for all 4 corners and edges of each μν plaquette (head, tail, head-tail, head-head, tail-tail).

### Operator (`_operator.py`)

Three classes compose the Dirac operator:

#### `hopping` class
- **Init:** Precomputes `M_plus_list[4]` and `M_minus_list[4]` via `give_hopping_plus`/`give_hopping_minus`. Performs MPI halo exchange for gauge field boundaries at init time. If `support_parity=True`, also splits into even/odd sub-blocks (`M_e_plus_list`, `M_o_plus_list`, etc.).
- **`matvec_plus(ward, src)` / `matvec_minus(ward, src)`:** Apply directional hopping with MPI halo exchange for fermion boundaries (send head to minus rank, receive tail from plus rank, and vice versa).
- **`matvec(src)`:** Sum over all 4 directions: Σ_μ (matvec_plus(μ) + matvec_minus(μ))

#### `sitting` class
- **Init:** Takes `clover_term` (can be None for pure Wilson). Adds I to get M = I + T. If `support_parity=True`, splits M into even/odd (`M_e`, `M_o`) and optionally precomputes inverses (`M_e_inv`, `M_o_inv`) unless provided externally.
- **`matvec(src)`:** Apply sitting term. If `clover_term is None`, returns `src` unchanged (identity).

#### `operator` class
- **Init:** Creates `hopping` and `sitting` instances. If `fine_hopping`, `fine_sitting`, and `local_ortho_null_vecs` are provided, builds a coarse-grid operator via Galerkin projection P^T D_fine P.
- **`matvec(src)`:** hopping.matvec + sitting.matvec. Auto-detects `[4,3,...]` vs `[12,...]` layout.
- **`matvec_eo(src_o)`:** Even-dest from odd-src hopping (used in preconditioned solves)
- **`matvec_oe(src_e)`:** Odd-dest from even-src hopping
- **`matvec_ee(src_e)` / `matvec_oo(src_o)`:** Even/odd sitting application
- **`matvec_ee_inv(src_e)` / `matvec_oo_inv(src_o)`:** Even/odd sitting inverse
- **`matvec_parity(src_o)`:** Parity-preconditioned operator: M_oo − M_oe · M_ee^{-1} · M_eo
- **`matvec_parity4fermion(fermion_in_o)`:** Same but auto-reshapes `[4,3,...]` ↔ `[12,...]`
- **`give_b_parity(b_e, b_o)`:** Preconditioned RHS: −M_oe · M_ee^{-1} · b_e + b_o
- **`give_x_e(b_e, x_o)`:** Recover even solution: M_ee^{-1} · (b_e − M_eo · x_o)
- **`matvec_eeo(src_e, src_o)` / `matvec_oeo(src_e, src_o)`:** Combined e→e+o→e and e→o+o→o
- **`matvec_all(src)`:** Full operator via parity-split/recombine: split src, apply eeo+oeo, recombine

**MPI halo exchange** in `matvec_eo`/`matvec_oe` is guarded by `grid_size[ward] != 1` — no MPI overhead for single-process directions.

## Coarse-Grid Operator (Galerkin Projection)

When `fine_hopping`, `fine_sitting`, and `local_ortho_null_vecs` are all provided, the operator builds a coarse-grid operator via P^T D_fine P:

1. For each null-space basis vector `e` and each direction `ward`:
   - Prolong a delta-source from coarse to fine grid
   - Apply the fine hopping operator (plus and minus directions)
   - Restrict the result back to coarse grid
   - Even/odd separation uses step=2 along the current direction
2. Also project the fine sitting operator: prolong → sitting.matvec → restrict

## Anti-Patterns

- **Never** use `self.sitting` (an object) as a truthy check; use `self.sitting.clover_term is not None`
- **Never** loop `torch.linalg.inv` per-site; use batched inversion (permute to batch dim, invert all at once, permute back)
- **Never** add `MPI.Barrier()` before/after blocking `Sendrecv` — it's redundant and slows down execution
