---
name: solver
description: pyqcu.solver 目录的完整生成 skill：BiCGStab(l)、FGMRES 与 legacy/strict QUDA-style MultiGrid，含奇偶边界、Galerkin 层级和 CUDA 持久显存约束。
---
# pyqcu.solver

Iterative solvers for the Dirac equation D ψ = η.

## CUDA 混合路径约定（2026-08-14）

- `_matvec`/`_restrict_cuda`/`_prolong_cuda`/`_coarse_dslash_cuda` 的 C++ 调用后必须
  `torch.cuda.synchronize()`（C++ 私有流写固定输出缓冲，与 torch 默认流无跨流同步）。
- BiCGStab breakdown 自动重启（保留 x/r，重置影子向量与系数，阈值 1e-20 +
  alpha/omega 有限性检查），不再抛 RuntimeError。

## Files

| File | Purpose |
|------|---------|
| `_bistabcg.py` | BiCGStab(l) solver (Bi-stabilized Conjugate Gradient) |
| `_multigrid.py` | Adaptive multigrid (AMG) V-cycle solver with CUDA acceleration at finest level |
| `_gmres.py` | Python restarted-FGMRES reference with right preconditioning, complex Givens rotations, warm start and restart true-residual checks; production strict CUDA solve uses the fused C++ entry |
| `_quda_multigrid.py` | QUDA-style transfer, coarse operator, MATPC and recursive strict reference hierarchy; coexists with legacy `_multigrid.py` |

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
- legacy `applyInitQcu`/`applyEndQcu` operation sequences must increment `_SET_INDEX_` between calls; Strict `CudaSchurOp` sequences use one fixed per-instance slot and follow the exception below

**C++ backend integration (level 1):**
- `applyMultigridRestrictQcu`/`applyMultigridProLongQcu` for inter-grid transfers
- `applyMultigridCoarseDslashQcu` for coarse-grid operator application
- Hopping matrices packed as `[2, 4, E, E, Xc, Yc, Zc, Tc]` (pm dir Eout Ein XYZT)

**BiCGStab state reset after coarse-grid correction (R3 fix):** After a coarse-grid correction `x = x + e_fine`, the residual `r` changes, so all BiCGStab state must be reinitialized: `r_tilde = r.clone()`, reset `p/v/s/t` to zero, reset `rho_prev/alpha/omega` to 1.0. Without this, `rho = vdot(r_tilde_old, r_new)` gives meaningless results.

**Convergence tracking:** Records `r_norm` twice per iteration (before and after coarse-grid correction). Plot shows both.

## Strict QUDA-style MultiGrid Invariants

Keep the legacy implementation available for comparison. Select Strict with `hierarchy_mode="strict"` or `QudaStrictMultigrid`; `setup_operator="schur"` is the legacy compact odd-Schur/setup-vector option, not the hierarchy selector. Under Strict the coarse geometry remains full even when that setup option is supplied. The strict hierarchy instead uses

\[
D_l=X_l+H_l,\qquad \widehat D_l=X_l^{-1}D_l,\qquad
D_{l+1}=R_l\widehat D_lP_l,
\]

with `R=P†`, full coarse lattices, and `coarse_spin=2` (`E=2*nvec`) at every Wilson/Clover coarse level. `P` is the blocked null-vector map from a full coarse field to the selected fine parity; `R` is its adjoint and returns a full coarse field. The parity boundary is narrow: R/P may consume or produce one compact fine parity while always mapping to/from a full coarse field; MATPC acts on the selected compact parity as `I-Hhat_pq Hhat_qp`. Never parity-crop coarse assets or substitute the legacy hopping-only coarse dslash.

The formal QUDA comparison fixes the spin convention to `QUDA_DEGRAND_ROSSI_GAMMA_BASIS`; include this basis in hierarchy/cache identity checks instead of assuming an implicit basis transform.

The current strict runtime accepts only all-level `matpc/direct_pc`; it rejects standard staggered/KD, odd or sub-two coarse extents, and unsupported non-nearest-neighbor stencil support instead of silently changing the operator. Coarse Galerkin setup applies the full left-preconditioned operator `X^-1 D` before forming `R(X^-1 D)P`; the fine odd-Clover Schur/MATPC path is used for smoothing and the solve, not as a reason to checkerboard coarse assets. User-facing canonical null vectors `[nvec,4,3,X,Y,Z,T]` must be converted before the C++ call to C-order blocked `[E,12,Xc,bx,Yc,by,Zc,bz,Tc,bt]`. The CUDA strict solver prepares the fine odd-Clover Schur system, applies fine MR → parity R → recursive coarse V-cycle → parity P → fine MR as a right preconditioner to restarted FGMRES, then reconstructs the full solution.

At every coarse transition, `X` is the local coarse block, `Y` is the four-direction forward/backward link set, and `Yhat=X^-1Y` is the link set used by the preconditioned coarse action. Runtime onsite storage is the pair `(X,X^-1)`; raw `Y` may be kept for setup diagnostics but is not required by the ordinary Strict solve. The fine boundary carries the physical Gauge and Clover even/odd blocks plus their inverses; coarse levels carry Galerkin `X/Y/Yhat`, not a second physical Gauge/Clover field.

The recursive hierarchy is allocated by strict init; the outer FGMRES workspace is allocated lazily by C++ on the first solve and reused. Its exact size is `(2*m+5)*B_f + 2*B_c`; Python keeps neither a duplicate Krylov arena nor duplicate coarse-I/O tensors. The current CUDA path fixes fine `target_parity=1` and coarse recursion `start_level=1`; coarse fields are full and only the fine-side MATPC/R/P views are compact. Bind packed `V/Yhat/onsite` assets first, then call `seal_cuda_runtime(runtime_assets_bound=True)` to detach the Python setup hierarchy; this seal is destructive, so copy `strict_setup_stats` first when they are needed. Packed raw `Y` is omitted by default and should be retained only for setup/diagnostics.

`params[57]` is `_MG_USE_INIT_GUESS_`. Set it to `0` for a cold solve; when it is `1`, prefill `fermion_out` and the fused entry consumes its odd half as the initial guess before reconstructing the full result. This flag is part of the existing `int32[58]` ABI, not a new parameter array.

Strict CUDA 的调用顺序必须是 `hierarchy.setup()` → 构造 `CudaSchurOp`（内部 `applyInitQcu`）→ 绑定运行期资产 → `applyMultigridStrictInitQcu` → 重复 V-cycle/FGMRES → `applyMultigridStrictEndQcu` → `CudaSchurOp.release()`（内部 `applyEndQcu`）。整个 Strict 生命周期保持该实例的 `_SET_INDEX_` 不变；不要套用 legacy 的递增或 coarse-grid reset 规则。

Runtime-cache schema v2 protects every logical tensor with a streaming SHA256. A load verifies the entire tensor set before creating/transferring device tensors, uses about 8 MiB host chunks, and intentionally performs two logical reads on a hit (verification plus transfer). Digest failure is fail-closed and must leave the device hierarchy unbound. If a concurrent publisher wins the same-identity target, the loser may reuse it only after fully verifying the manifest, every dataset attr, and every tensor SHA256.

`memory_report()` reports explicitly owned runtime assets, not all process/device memory. Memory regressions must track setup/export/bind/seal/init/first solve/steady solve/close, distinguish planned from first-solve resident fused bytes, deduplicate shared storages, include backend `LatticeSet` and allocator high-water overhead, and verify repeated solves have stable live bytes. Galerkin setup planning counts four simultaneous full-field arenas. The library default setup cap is `512 MiB`; the formal `16×32×32×48` profile instead uses colored `C=12` with `4 GiB` for c64 and `C=1` with `1 GiB` for c128. The outer fused `max_krylov_bytes` is independent: solver API default `512 MiB`, formal c64 `512 MiB`, formal c128 `1 GiB`. `strict_galerkin_mode="auto"` compares modeled site-batch and colored call/memory costs; record requested/effective mode, `C`, projection batch `K`, cap and stats, and let formal validation fail on silent shrink. Formal benchmark memory probing uses schema version 2 and an independent untimed device-wide `cudaMemGetInfo` measurement named `device_used_max_observed_bytes`; `setup_seconds` ends before sampler stop. A join timeout keeps the sampler handle live and invalidates the record rather than silently accepting incomplete evidence. A memory cap may reduce FGMRES restart but must not change the relative true-residual criterion.

MPI stage 1 has c64/c128 global dot/norm reduction (`global_reduction=True`) and rank-symmetric preflight. Setup/full/compact halos and the distributed fused solve remain disabled (`setup_halo=False`, `full_halo=False`, `compact_halo=False`, `fused_fgmres=False`), so every production multi-rank strict solve must remain rejected. Do not describe this stage as distributed MultiGrid support.

## 口径与容差语义（2026-08-24）

- **clover 双实现「12% 分歧」已证伪**：系对照实验 argv MASS=0 ⇒ κ=0.125 错配伪影；
  正确口径下 C++ applyClovers ≡ Python make_clover+add_I（rel=9.5e-09）。
  遇「双实现分歧」先查单位/约定口径（κ 吸收），再怀疑实现。
- **null 向量逆迭代容差**：`give_null_vecs*` 的 nv_tol 是**绝对容差**语义；逆迭代收敛到
  近似精确解时 v−x≈舍入噪声，归一化后得到噪声向量。须改宽松相对容差（ddamg 配方）。
  诊断指标 ‖Sv‖/‖v‖：有效谱尺度 ≈0.38–0.50；≈0.98 即噪声向量。
- ghost Ritz 值须过残差验证才算数（γ5S² Lanczos [0.0028..0.083] 区间全为 ghost）。

## Debug helper

`_verify_coarse_dslash(level, tol)` compares CUDA coarse dslash against Python einsum reference.
