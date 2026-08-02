# CLAUDE.md — pyqcu

Top-level Python package for QCU: CUDA-accelerated lattice QCD library. Implements Wilson/Clover Dirac operators, BiStabCG and multigrid solvers, stout smearing, and gauge field generation — all MPI-distributed across a 4D process grid.

## Two-Layer Architecture

1. **Pure Python** (`dslash/`, `solver/`, `smear/`) — PyTorch-based implementations for CPU, CUDA GPU, or Ascend NPU (via `pyqcu.cann`).
2. **C++ CUDA backend** (`cuda/` → `cpp/cuda/qcu/`) — Hand-tuned CUDA kernels with MPI halo exchange, accessed through a Cython bridge (`pyqcu.cuda.qcu`).

The multigrid solver can mix both layers: finest-level smoothing via the C++ backend (`with_cuda_qcu=True`) and coarser levels in pure Python.

## Subpackages

| Package | Purpose |
|---------|---------|
| `lattice/` | Gamma matrices, Gell-Mann matrices, SU(3) checks, gauge field generation |
| `dslash/` | Wilson & Clover Dirac operators, hopping/sitting decomposition, even-odd preconditioning, coarse-grid Galerkin projection |
| `solver/` | BiCGStab(l) solver, adaptive multigrid (AMG) V-cycle solver, GMRES stub |
| `smear/` | Stout gauge field smearing (iterative, MPI-capable) |
| `tools/` | MPI grid helpers, HDF5 I/O (parallel + serial), einsum (TileLang JIT), linear algebra, multigrid restrict/prolong/null-vectors |
| `testing/` | Integration tests for all components |
| `cuda/` | Cython bridge to `libqcu.so` + parameter constants (`define.py`) |
| `cann/` | Torch compatibility layer for Ascend NPU (complex ops decomposition) |
| `dtk/` | Placeholder for DCU/ROCm backend (no implementation yet) |
| `maca/` | Placeholder for Maca backend (no implementation yet) |

## Key Convention

All code imports `pyqcu.cann as _torch` instead of `torch` directly. On CUDA/CPU it delegates to torch; on NPU it decomposes complex ops into real/imaginary parts (Ascend NPU doesn't natively support complex tensors).

## Data Layout Conventions

| Tensor | Shape | Notes |
|--------|-------|-------|
| Gauge field (U) | `[3, 3, 4, Lx, Ly, Lz, Lt]` | `[color, color, direction, x, y, z, t]` |
| Fermion field | `[4, 3, Lx, Ly, Lz, Lt]` | `[spin, color, x, y, z, t]` |
| Clover term | `[4, 3, 4, 3, Lx, Ly, Lz, Lt]` | `[spin, color, spin, color, x, y, z, t]` |
| Parity-split (prefix `p`) | `[2, ...original...]` | `p=0` is even sites, `p=1` is odd (prepended dim) |
| Flattened spin×color | `[12, ...]` or `[E, ...]` | E = degrees of freedom per site |

Spacetime dimensions are always the last four axes (`...xyzt` layout). Ward indices use negative indexing (`wards['x'] = -4`, `wards['t'] = -1`) to be robust against arbitrary prefix dimensions.

## Build & Run

```bash
source ./env.sh                # LD_LIBRARY_PATH, PYTHONPATH, MPI flags
bash ./build.sh                # build libqcu.so (C++ CUDA backend)
bash ./install.sh              # build Cython extension in-place

# Tests
cd examples && pytest .
mpirun -np 4 python examples/pyqcu/conftest.py
```

## Logging Convention

All modules use: `PYQCU::MODULE::SUBMODULE:\n message`

---

## Complete Skills (Agent-Produced Subdirectories)

The content of each subdirectory below was produced with Claude Code assistance. Per repo convention, the complete skill that generates that content is reproduced verbatim below (source: the subdirectory's own `CLAUDE.md`), so the full knowledge is available directly at this level.

### Complete Skill: `lattice/` (source: `lattice/CLAUDE.md`)

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

### Complete Skill: `dslash/` (source: `dslash/CLAUDE.md`)

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

### Complete Skill: `solver/` (source: `solver/CLAUDE.md`)

# CLAUDE.md — pyqcu.solver

Iterative solvers for the Dirac equation D ψ = η.

## Files

| File | Purpose |
|------|---------|
| `_bistabcg.py` | BiCGStab(l) solver (Bi-stabilized Conjugate Gradient) |
| `_multigrid.py` | Adaptive multigrid (AMG) V-cycle solver with CUDA acceleration at finest level |
| `_gmres.py` | GMRES solver — **placeholder stub, not yet implemented** |

## Exported API

### `bistabcg(b, matvec, tol=1e-6, max_iter=1000, x0=None, if_rtol=False, verbose=True) → torch.Tensor`

Standard BiCGStab solver. Takes a callable `matvec(src) → dest`.

**Breakdown detection** (added R2): raises `RuntimeError` on:
- `rho ≈ 0` (r_tilde orthogonal to r — loss of orthogonality)
- `vdot(r_tilde, v) ≈ 0` (pivot breakdown)
- `vdot(t, t) ≈ 0` (t is zero/near-zero)

**Tolerance:** `if_rtol=True` uses `tol * ||b||`; otherwise uses absolute `tol`.

### `multigrid` class

Adaptive multigrid V-cycle solver with configurable multi-level hierarchy.

**Constructor parameters:**
- `dtype_list`, `device_list` — per-level data types and devices
- `U`, `clover_term`, `kappa`, `u_0` — physical parameters
- `clover_ee_inv`, `clover_oo_inv` — if both provided, enables `with_cuda_qcu=True` (C++ backend at finest level)
- `min_size=4` — minimum lattice size per direction before coarsening stops
- `max_level=4` — maximum number of MG levels
- `mg_grid_size=[2,2,2,2]` — coarsening factor per direction
- `dof_list=[12,24,24,...]` — degrees of freedom per level
- `tol`, `max_iter`, `num_restart=5` — convergence parameters
- `num_convergence_sample=50` — window for adaptive level-back detection
- `support_parity=False` — use even-odd preconditioning

**Key methods:**

| Method | Purpose |
|--------|---------|
| `init()` | Build null-space vectors via inverse iteration, local-orthogonalize, construct coarse-grid operators (Galerkin) |
| `solve(b, x0)` | Solve D x = b. Returns `[4, 3, Lx, Ly, Lz, Lt]` tensor. |
| `cycle(level)` | Recursive V-cycle: BiCGStab smoothing → restrict residual → recurse → prolong correction → continue smoothing |
| `adaptive(iter)` | Level-back: drops to coarsest level if convergence stalls (≥3 counts in sample window) |
| `levels_back()` | Reset adaptive state |
| `plot(save_path)` | Plot convergence history (matplotlib, only on root rank) |

**Execution layers:**
- **Level 0 (finest):** Can use C++ CUDA backend (`with_cuda_qcu=True`) for BiCGStab smoothing
- **Level 1 (first coarse):** Can use C++ CUDA backend for coarse dslash via `_coarse_dslash_cuda()`
- **Levels 2+:** Pure Python einsum-based operators

**C++ backend integration (level 0):**
- `applyInitQcu`/`applyEndQcu` manage scratch buffer lifecycle
- `applyCloverBistabCgQcu` performs the full BiCGStab solve
- `applyCloverBistabCgDslashQcu` performs a single parity-preconditioned dslash
- `_SET_INDEX_` must be incremented between successive calls

**C++ backend integration (level 1):**
- `applyMultigridRestrictQcu`/`applyMultigridProLongQcu` for inter-grid transfers
- `applyMultigridCoarseDslashQcu` for coarse-grid operator application
- Hopping matrices packed as `[2, 4, E, E, Xc, Yc, Zc, Tc]` (pm dir Eout Ein XYZT)

**BiCGStab state reset after coarse-grid correction (R3 fix):** After a coarse-grid correction `x = x + e_fine`, the residual `r` changes, so all BiCGStab state must be reinitialized: `r_tilde = r.clone()`, reset `p/v/s/t` to zero, reset `rho_prev/alpha/omega` to 1.0. Without this, `rho = vdot(r_tilde_old, r_new)` gives meaningless results.

**Convergence tracking:** Records `r_norm` twice per iteration (before and after coarse-grid correction). Plot shows both.

**Debug helper:** `_verify_coarse_dslash(level, tol)` compares CUDA coarse dslash against Python einsum reference.

### Complete Skill: `smear/` (source: `smear/CLAUDE.md`)

# CLAUDE.md — pyqcu.smear

Gauge field smearing — spatial smoothing of gauge links to reduce UV noise.

## Files

| File | Purpose |
|------|---------|
| `_stout.py` | Stout smearing algorithm (copied/adapted from EasyDistillation's elemental generator) |

## Exported API

### `stout_smear(U, nstep=1, rho=0.12, support_parallel=False) → torch.Tensor`

Apply nstep iterations of stout smearing with parameter rho.

**Algorithm (per step):**

1. **Compute Q_μ = staple sum** for each direction μ: sum over ν≠μ of two 3-link staples (U_ν U_μ U^†_ν forward + U^†_ν U_μ U_ν backward)
2. **Project to su(3) algebra:** Q ← ρ · Q · U^†, then anti-hermitize: Q ← i/2 · (Q^† − Q) − (1/3) Tr(Q) · I
3. **Compute SU(3) projection coefficients f₀, f₁, f₂** via the Morningstar-Peardon method:
   - c₀ = Re(Tr(Q³))/3, c₁ = Re(Tr(Q²))/2
   - θ = arccos(c₀ / (2(c₁/3)^(3/2)))
   - u = √(c₁/3) · cos(θ/3), w = √c₁ · sin(θ/3)
   - f₀, f₁, f₂ expressed in terms of e^{iu}, e^{2iu}, cos(w), sinc(w)
4. **Parity handling** (when c₀ < 0): f₀ → f₀^*, f₁ → −f₁^*, f₂ → f₂^* (standard path); NPU path uses real/imag decomposition
5. **Update U:** U_new = (f₀·I + f₁·Q + f₂·Q²) · U

**Numerical stability:**
- c₁ clamped to min 1e-15 (prevents c₀_max = 0)
- ratio clamped to [−1+1e-15, 1−1e-15] for arccos domain
- sinc(w) uses Taylor expansion for |w| ≤ 0.05, sin(w)/w otherwise
- Denominator 9u² − w² has 1e-15 epsilon to prevent division by zero

**MPI support:** When `support_parallel=True`, MPI boundary data (U_head, U_tail, U_head_tail) is recomputed each step since U changes with each smearing step.

## Key Anti-Pattern (Fixed)

The `nstep>1` loop previously did not update `U` between steps — the loop variable was properly rebound but the MPI boundary data was computed outside the loop. Fixed by moving MPI exchange inside the step loop.

## Data Layout

Gauge field: `[3, 3, 4, Lx, Ly, Lz, Lt]` = `[color, color, direction, x, y, z, t]`

Returned tensor has the same shape.

## NPU Support

Has per-module `force_use_npu` flag. On NPU, the parity sign convention for f₀/f₁/f₂ uses explicit real/imag decomposition:
- f₀: imag = −imag (conj)
- f₁: real = −real, imag unchanged (conj + leading minus cancel)
- f₂: real = −real, imag unchanged (same as f₁)

### Complete Skill: `tools/` (source: `tools/CLAUDE.md`)

# CLAUDE.md — pyqcu.tools

Utility modules for MPI grid management, HDF5 I/O, linear algebra, tensor operations, multigrid transfers, and TileLang JIT kernels.

## Files

| File | Purpose |
|------|---------|
| `_define.py` | MPI grid size factorization, rank neighbors, parity splitting (`oooxyzt2poooxyzt`/`poooxyzt2oooxyzt`), dimension reordering (ccdxyzt↔ccdptzyx, scxyzt↔psctzyx), dtype conversion tables, device setup, slice helpers, prime factorization |
| `_io.py` | HDF5 I/O with MPI parallel I/O (`driver='mpio'`, `h5py`) and serial gather/scatter fallback (`comm.gather` + `comm.scatter`) |
| `_linalg.py` | Vector dot product (`vdot`) and norm (`norm`) via `_torch` |
| `_einsum.py` | TileLang JIT-compiled einsum kernels — currently `Eexyzt_exyzt2Exyzt` (optional, try/except import) |
| `_matul.py` | TileLang-based matrix multiply kernels: `matmul_gpu(M,N,K,...)` and `matmul_cpu(M,N,K,...)` (optional) |
| `_multigrid.py` | Null vector generation (`give_null_vecs`), local orthogonalization (`local_orthogonalize`), restrict/prolong operators — all with NPU-compatible fallback paths |
| `_roll.py` | Tensor rolling utilities |

## Exported API

### MPI Grid (`_define.py`)

| Function | Purpose |
|----------|---------|
| `give_grid_size()` | Auto-factor MPI communicator size into 4D grid `[gx, gy, gz, gt]` via prime factorization (sorted ascending) |
| `give_grid_index(rank)` | Convert flat rank to 4D grid index `[ix, iy, iz, it]` |
| `give_rank_plus(ward, rank)` | Neighbor rank in +direction |
| `give_rank_minus(ward, rank)` | Neighbor rank in −direction |
| `give_rank_plus_plus(ward_a, ward_b, rank)` | Diagonal neighbor (+a, +b) |
| `give_rank_plus_minus(ward_a, ward_b, rank)` | Diagonal neighbor (+a, −b) |
| `give_rank_minus_minus(ward_a, ward_b, rank)` | Diagonal neighbor (−a, −b) |
| `give_rank_minus_plus(ward_a, ward_b, rank)` | Diagonal neighbor (−a, +b) |
| `set_device(device, verbose)` | Set CUDA/NPU device based on MPI rank (round-robin assignment) |

### Parity Splitting (`_define.py`)

- **`oooxyzt2poooxyzt(input_array, verbose) → [2, ..., t, z, y, x//2]`** — Standard layout → parity-split. Separates even/odd sites based on (x+y+z+t) % 2. Splits along the fastest-varying (x) dimension.
- **`poooxyzt2oooxyzt(input_array, verbose) → [..., t, z, y, x]`** — Reverse: parity-split → standard layout. Recombines even/odd halves.

Both support NPU via explicit real/imaginary handling.

### Even-Odd Mask (`_define.py`)

- **`give_eo_mask(oootzy_t_p, eo, verbose)`** — Returns boolean mask for even (`eo=0`) or odd (`eo=1`) sites. Uses `(x+y+z) % 2` checkerboard. Results cached by shape+device+eo key.

### Dimension Reordering (`_define.py`)

HDF5 I/O uses dimension order `zyxt` (fastest to slowest: t, z, y, x):

- **`ccdxyzt2ccdptzyx(ccdxyzt) → [c,c,d,p,t,z,y,x]`** — Gauge field to file layout
- **`ccdptzyx2ccdxyzt(ccdptzyx) → [c,c,d,x,y,z,t]`** — File layout to gauge field
- **`scxyzt2psctzyx(scxyzt) → [p,s,c,t,z,y,x]`** — Fermion field to file layout
- **`psctzyx2scxyzt(psctzyx) → [s,c,x,y,z,t]`** — File layout to fermion field

### MPI Gather/Scatter (`_define.py`)

- **`local_xyzt2whole_xyzt(local_array, root) → Tensor | None`** — Gather distributed tensor chunks into a full global tensor on root rank. Uses `comm.Gather`.
- **`whole_xyzt2local_xyzt(dtype, device, whole_shape, whole_array, root) → Tensor`** — Scatter a global tensor (or shape template) to all ranks. Uses `comm.Scatter`. Each rank gets its grid block.

### Slice Helpers (`_define.py`)

- **`slice_dim(dims_num, ward, start, stop, step, point)`** — Build Python slice tuple for indexing along a specific ward dimension (using negative indexing). For `point`, returns integer index.
- **`slice_dim_dim(dims_num, ward_a, ..., ward_b, ...)`** — Two-dimension slice
- **`slice_dim_none_dim(dims_num, ward, ..., ward_none)`** — Slice with one skipped dimension

### Memory Helpers (`_define.py`)

- **`to_contiguous_real(tensor, channel, *shape)`** — Extract real/imag channel from complex tensor and return a truly stride-1 contiguous real tensor. Uses `empty + copy_` pattern instead of `.contiguous()` for correctness on single-element tensors.

### HDF5 I/O (`_io.py`)

- **`gridoooxyzt2hdf5oooxyzt(input_tensor, file_name, lat_size, verbose)`** — Write distributed tensor to HDF5. MPI path uses `h5py.File(..., driver='mpio')`; serial path uses `comm.gather` to root.
- **`hdf5oooxyzt2gridoooxyzt(file_name, lat_size, device, verbose)`** — Read HDF5 into distributed tensor. MPI path uses `h5py.File(..., driver='mpio')`; serial path uses root-read + `comm.scatter`.

**MPI support detection:** `HAS_MPI_SUPPORT = check_mpi_support()` at module import time. Tests h5py config and tries creating a test file with `driver='mpio'`. Can be manually overridden.

**Serial fallback note:** `comm.scatter` uses pickle serialization; may hit 2GB limit for very large lattices (>64⁴ float32). MPI I/O path preferred for production.

### Linear Algebra (`_linalg.py`)

- **`norm(input, p='fro', dim=None, keepdim=False)`** — Frobenius/vector norm via `_torch.norm`
- **`vdot(input, other)`** — Complex inner product `Σ conj(a_i) * b_i` via `_torch.vdot`

### Multigrid Utilities (`_multigrid.py`)

- **`give_null_vecs(null_vecs, matvec, bistabcg, normalize, ortho_r, ortho_null_vecs, verbose)`** — Generate near-null-space vectors via inverse iteration: v_i = v_i − A^{-1} A v_i. Optionally orthogonalizes against previous vectors. `null_vecs` parameter is used as shape/dtype/device template only; values are overwritten with random init.
- **`local_orthogonalize(null_vecs, coarse_lat_size, normalize, verbose)`** — Block-local Gram-Schmidt orthogonalization via batched QR decomposition. Splits null vectors into coarse-grid blocks, applies QR per block. NPU path avoids >8-dim tensors.
- **`restrict(local_ortho_null_vecs, fine_vec)`** — P^T v_fine = Σ v_fine · null_vec^†. Standard path uses 10-dim einsum; NPU path reshapes to ≤8 dims.
- **`prolong(local_ortho_null_vecs, coarse_vec)`** — P v_coarse = Σ null_vec · v_coarse. Standard path uses 10-dim einsum; NPU path reshapes to ≤8 dims.

**NPU compatibility:** NPU limits tensors to ≤8 dimensions, so restrict/prolong/orthogonalize all have `_npu` variants that use reshape/permute chains to stay within this limit. Cross-validated against standard path (max diff ~1e-7 for float32).

### TileLang Integration (`_einsum.py`, `_matul.py`)

Optional — try/except import at package level; silently degrades if TileLang unavailable.

- **`Eexyzt_exyzt2Exyzt(Eexyzt, exyzt)`** — JIT-compiled TileLang kernel for specific einsum pattern used in Wilson dslash (disabled by default; `tools_Eexyzt_exyzt2Exyzt = False`)
- **`matmul_gpu(M, N, K, block_M, block_N, block_K)`** / **`matmul_cpu(M, N, K, ...)`** — TileLang kernel definitions for matrix multiply benchmarking

Kernels use `warp_size = 128` from `_define`.

### Dtype Conversion Tables (`_define.py`)

- **`np2torch_dtype`**, **`torch2np_dtype`** — bidirectional NumPy ↔ PyTorch dtype maps
- **`torch2tl_dtype`** — PyTorch → TileLang dtype map (float16/32/64 only)

## Logging Convention

`PYQCU::TOOLS::<SUBMODULE>::\n message`

### Complete Skill: `testing/` (source: `testing/CLAUDE.md`)

# CLAUDE.md — pyqcu.testing

Integration tests for all PyQCU components. Tests are Python functions imported by `examples/*/conftest.py` entry points.

## Architecture

All test functions live in `pyqcu/testing/__init__.py`. They import from all PyQCU subpackages (`lattice`, `solver`, `dslash`, `tools`, `smear`). Each `examples/*/conftest.py` acts as a pytest entry point that imports specific test functions and calls them. The conftest files are manually edited to uncomment the test(s) to run.

The module imports `tilelang` at module level (with try/except fallback) for `test_matmul`.

## Test Functions

### `test_lattice(lat_size, dtype, device)`
Tests SU(3) gauge generation + gamma matrix algebra.
- Generates random gauge field, runs `check_su3`
- Verifies γ_μ² = I for all 4 gamma matrices
- **Assertion:** `check_su3` must return True

### `test_dslash_wilson(kappa, lat_size, dtype, device, with_data, support_parallel)`
Tests Wilson Dirac operator.
- `with_data=False`: Generates random gauge field + source, applies full Wilson operator and eo/oe preconditioned variants
- `with_data=True`: Loads reference HDF5 data (`refer.wilson.*.L32K0_125.*.h5`), validates operator.matvec against known result
- **Assertion:** Relative difference < 1e-4

### `test_dslash_parity(lat_size, kappa, dtype, device)`
Tests parity-preconditioned Wilson+Clover operator with MPI.
- Distributes gauge field across MPI grid
- Root rank computes full operator result as reference
- All ranks compare local parity-preconditioned operator against reference
- Tests both `matvec_all` and `matvec_eeo`/`matvec_oeo` paths

### `test_dslash_clover(device, with_data, dtype)`
Tests Clover term construction.
- `with_data=True`: Loads reference data, validates clover term and inverse against known results
- `with_data=False`: Tests parallel vs serial clover construction across MPI grid

### `test_solver(kind, method, kappa, lat_size, dtype, device, with_data, max_level, num_restart, support_parity)`
Tests BiStabCG and multigrid solvers.
- `method='bistabcg'`: Standard or parity-preconditioned BiCGStab
- `method='multigrid'`: Full multigrid V-cycle with `init()` + `solve()` + `plot()`
- `with_data=True`: Validates against reference Wilson data
- **Assertion:** Relative error < 1e-3

### `test_matmul()`
Benchmarks TileLang JIT-compiled matrix multiply vs PyTorch (cuBLAS/MKL).
- GPU: 4096×4096 matmul, TileLang vs cuBLAS
- CPU: 1024×1024 matmul, TileLang (LLVM or C backend) vs MKL/OneDNN
- Prints TFLOPS comparison table

### `test_smear_stout(lat_size, device, dtype)`
Tests stout smearing across MPI grid.
- Distributes gauge field, root computes whole-grid reference
- All ranks compare local parallel smear against reference
- Verifies SU(3) before and after smearing

## Running Tests

```bash
cd examples && pytest .                              # all conftest.py files
mpirun -np 4 python examples/pyqcu/conftest.py       # single file with MPI
```

## Logging Convention

All test output uses: `PYQCU::TESTING::<MODULE>::\n message`

## Important Notes

- Tests use `tools.local_xyzt2whole_xyzt` / `tools.whole_xyzt2local_xyzt` for MPI reference comparison
- Reference HDF5 data lives in `examples/data/`
- The `path` variable in tests is computed from `pyqcu.__file__` to locate data files
- **R3 fix:** Tests now include `assert` statements so pytest can detect failures

### Complete Skill: `cuda/` (source: `cuda/CLAUDE.md`)

# CLAUDE.md — pyqcu.cuda

Cython bridge package for the C++ CUDA backend (`libqcu.so`).

## Files

| File | Purpose |
|------|---------|
| `__init__.py` | Makes `pyqcu.cuda` a proper Python package (added 2026-07-28 R3; was missing, causing `pip install` failures) |
| `qcu/qcu.pyx` | Cython extension source — wraps C functions from `pyqcu.h` as `applyInitQcu`, `applyWilsonDslashQcu`, etc. |
| `qcu/qcu.pxd` | Cython declaration file — `cdef extern` block matching `pyqcu.h` |
| `qcu/qcu.pyi` | Type stub (155 lines) — full type annotations, docstrings, and default values for IDE support |
| `define.py` | Parameter constants (`_LAT_X_`, `_SET_PLAN_`, etc.) and dtype conversion helpers (`dtype()`, `epytd()`) |

## Public API

```python
from pyqcu.cuda import qcu      # Cython bridge to libqcu.so
from pyqcu.cuda import define   # Parameter constants, dtype helpers, pre-built params/argv/set_ptrs tensors
```

## Cython Extension — C Functions Exposed

| Function | Purpose | Plan |
|----------|---------|------|
| `applyInitQcu` / `applyEndQcu` | Allocate / free scratch buffers | — |
| `applyWilsonDslashQcu` | Wilson dslash | 0 |
| `applyCloverDslashQcu` | Clover dslash | 2 |
| `applyWilsonBistabCgQcu` / `applyWilsonBistabCgDslashQcu` | Wilson BiStabCG solver + its dslash | 1 |
| `applyWilsonCgQcu` / `applyWilsonCgDslashQcu` | Wilson CG solver + its dslash | 1 |
| `applyCloverBistabCgQcu` / `applyCloverBistabCgDslashQcu` | Clover BiStabCG (needs clover_ee/oo + inverses) | 1 |
| `applyCloverQcu` / `applyCloversQcu` | Build Clover term (and its inverse) | 2 |
| `applyDslashQcu` | Combined Wilson+Clover dslash | 0+2 |
| `applyLaplacianQcu` | Laplacian operator | -2 |
| `applyGaussGaugeQcu` | Gaussian gauge field generation | -1 |
| `applyMultigridRestrictQcu` / `applyMultigridProLongQcu` | MG restrict/prolong with null vectors | MG |
| `applyMultigridCoarseDslashQcu` | Coarse-grid dslash (hopping + sitting) | MG |
| `applyCloverMultigridQcu` | Full Clover multigrid V-cycle solver | MG |

All functions take raw pointers cast to `long long` from `tensor.contiguous().data_ptr()`.

## Parameter Protocol

Three flat tensors bridge Python ↔ C++:

- **`params`** (int32, size 54) — lattice dims (`_LAT_X_`…`_LAT_XYZT_`), grid sizes (`_GRID_X_`…), data types (`_DATA_TYPE_`), iteration counts (`_MAX_ITER_`), plan selection (`_SET_PLAN_`), verbosity (`_VERBOSE_`), parity (`_PARITY_`), multigrid level configs (`_MG_LEVEL1_X_`…, `_MG_NUM_LEVEL_`)
- **`argv`** (float, size 7) — physical parameters: `_MASS_` (idx 0), `_ATOL_` (1), `_SIGMA_` (2), per-level MG tolerances (3–6)
- **`set_ptrs`** (int64, size 100) — scratch pointers managed by the C++ runtime

Index constants in `define.py` MUST stay in sync with `cpp/cuda/qcu/include/define.h`.

`define.py` also provides pre-built tensors `params`, `argv`, and `set_ptrs` for convenience. They are modified in-place by the solver code.

## Critical: `_SET_INDEX_` Increment

Between successive C++ calls within the same `applyInitQcu`/`applyEndQcu` lifecycle, you MUST increment `params[define._SET_INDEX_]` by 1. Failing to do so causes scratch buffer reuse conflicts that produce wrong results.

Exception: coarse-grid dslash resets `_SET_INDEX_` to 0 (different MG level, no overlap with fine-level ops).

## Data Type Mapping

- `define.dtype(data_type)` — QCU internal constant (`_LAT_C64_`, `_LAT_R32_`, etc.) → PyTorch dtype
- `define.epytd(torch_dtype)` — PyTorch dtype → QCU internal constant
- `define.lat_shape(params)` — extract `[Lt, Lz, Ly, Lx]` from params tensor

## Plan Selection

| Plan Constant | Value | Purpose |
|---------------|-------|---------|
| `_SET_PLAN_N_2_` | -2 | Laplacian |
| `_SET_PLAN_N_1_` | -1 | Gauss gauge generation |
| `_SET_PLAN0_` | 0 | Wilson dslash |
| `_SET_PLAN1_` | 1 | BiStabCG / CG (and their dslash) |
| `_SET_PLAN2_` | 2 | Clover dslash |

## Call Lifecycle

```python
qcu.applyInitQcu(set_ptrs, params, argv)          # allocate
# ... operations with _SET_INDEX_ += 1 between calls ...
qcu.applyEndQcu(set_ptrs, params)                  # free
```

### Complete Skill: `cann/` (source: `cann/CLAUDE.md`)

# CLAUDE.md — pyqcu.cann

Torch compatibility layer for Ascend NPU. All Python code in PyQCU imports `pyqcu.cann as _torch` instead of using `torch` directly.

## Problem

Ascend NPU does not natively support complex tensors. This module wraps torch operations, decomposing complex ops into real/imaginary parts on NPU while passing through directly on CUDA/CPU.

## Behavior

- **CUDA/CPU path** (`device.type != 'npu'` and `force_use_npu=False`): delegates to `torch.*` unchanged
- **NPU path** (`device.type == 'npu'` or `force_use_npu=True`): decomposes complex ops into real/imaginary parts

## Global Flag

`pyqcu.cann.force_use_npu = True` — force NPU code paths on CPU for testing without NPU hardware. This affects only the `cann` layer; some modules (`dslash/_wilson.py`, `tools/_define.py`, `tools/_multigrid.py`, `smear/_stout.py`) also have their own per-module `force_use_npu` flag for deeper NPU workarounds (e.g., tensor dimension limits).

## Functions Provided

Always use these instead of raw torch calls anywhere complex tensors might run on NPU:

| Category | Functions | Notes |
|----------|-----------|-------|
| Math | `abs`, `vdot`, `norm`, `sqrt`, `matmul` | `vdot` → conj-flatten-sum; `norm` → abs-then-norm; `sqrt` → CPU fallback |
| Reduction/shape | `roll`, `allclose`, `einsum` | `roll` → roll real/imag separately; `allclose` → check real + imag separately |
| Creation | `zeros`, `zeros_like`, `randn`, `randn_like`, `eye` | Creates real parts then combines to complex |
| Linear algebra | `linalg_qr` | Falls back to CPU on NPU for complex inputs |

### Uses of raw `torch`

- `torch.linalg.det` — used in `lattice.check_su3()` for SU(3) determinant check. No equivalent in `_torch`; works on NPU for real matrices.
- `torch.matrix_exp` — used in `lattice.generate_gauge_field()` for exponential map.

## Einsum on NPU

General N-operand complex einsum uses a combinatorial approach. For Z = Π(a_k + i·b_k):

- Iterates all 2ⁿ sign combinations (bitmask k → real/imag selection for each operand)
- Even number of imaginary parts → contributes to real part with sign = (-1)^(n_imag/2)
- Odd number of imaginary parts → contributes to imaginary part with sign = (-1)^((n_imag-1)/2)
- 2-operand special case: explicit ac-bd + i(ad+bc) formula (faster)

## Key Implementation Details

- `eye(n, m, ...)` — creates real identity then casts to complex dtype on NPU
- `zeros(*args, ...)` / `randn(*args, ...)` — creates separate real + imag tensors and combines
- `sqrt(input)` — sends complex input to CPU, computes sqrt, sends back (NPU doesn't support complex sqrt)
- `matmul(input, other)` — uses explicit (ac-bd) + i(ad+bc) decomposition

## Subdirectory

`qcu/` — placeholder stub (empty `PASS` file), no implementation yet.

### Complete Skill: `dtk/` (source: `dtk/CLAUDE.md`)

# CLAUDE.md — pyqcu.dtk

Placeholder for DCU/ROCm (AMD GPU) backend. No implementation yet.

Contains only an empty `PASS` file as a directory placeholder.

### Complete Skill: `maca/` (source: `maca/CLAUDE.md`)

# CLAUDE.md — pyqcu.maca

Placeholder for Maca backend. No implementation yet.

Contains only an empty `PASS` file as a directory placeholder.

