# Strict QUDA-style multigrid invariants

The `StrictMultigrid` hierarchy: its operator recursion, connectivity, runtime
asset layout, and the history of its setup budgets. Split out of `SKILL.md` to
keep the skill entry point under the 500-line limit; the content is unchanged.
The Legacy-MG route, which is the QCU default, is documented in the skill itself.

Keep the legacy implementation available for comparison. Select Strict with `hierarchy_mode="strict"` or the public `StrictMultigrid`; `setup_operator="schur"` is the legacy compact odd-Schur/setup-vector option, not the hierarchy selector. Under Strict the coarse geometry remains full even when that setup option is supplied. The strict hierarchy instead uses

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

Current strict MPI provides rank-symmetric preflight, c64/c128 global reduction, setup/full/compact/R-P halos and distributed fused FGMRES. Multi-rank runs use one process/rank/device and must pass capability preflight plus an independent true residual; a reduction-only smoke is not sufficient evidence.

### 2026-10-29 L32 Strict 容量与预算

A100-40GB、`32x32x32x64` c64、两层、`nvec=12`、`coarse_spin=2`、
`restart=16` 的实测边界：

- 已有 Legacy E12 cache 的 `lonv` 可直接经
  `prepare_fair_nullvec.py` 重构 canonical full basis；不要重复生成 null-space。
- 1 MPI rank 在 `StrictTransfer._block_orthogonalise` 的 96 MiB 分配处 OOM；
  2/4 MPI rank（每 rank 一张 A100）成功，cold setup 为 `201.073/111.873 s`，
  steady sync-max 为 `0.476122/0.356964 s`。L32 Strict 不能假设单卡可容纳
  full basis、full Clover 和正交化临时量；4-rank 相对 2-rank 仅 `1.334x`。
- fused FGMRES 的 `restart=16` workspace 实需 `1,887,436,800 B`。512 MiB 预算
  会 fail-closed；正式 L32 使用 2 GiB。必须记录 required/requested，不能静默缩小
  restart 冒充同一性能点。
- `final_residual` 的 native 字段可能使用 absolute/aggregate 口径；L32 2-rank
  实测为 `4.646653e-3`，而独立 full-operator 相对残差为 `6.253315e-7`。验收只看
  后者。
- 旧 Strict trace 的顶层 `restriction_calls` 曾漏记 `precondition_fine()` 的直接
  fine restriction/prolongation。源码已补计数，8³×16 回归由 0 纠正为 9；历史
  pre-fix JSON 仍应结合 level-1 `coarse_solver_calls/coarse_solver_iterations`、
  stage 和 independent residual 交叉解释。

### 2026-10-29 L48 builder 资源门

真实 `48x48x48x96` Legacy E12 cache 的 4-rank CUDA builder 已实测：null-vector 后
早期实现曾因 overlap stencil 走 CPU 而空占 GPU；根因是 patch LatticeSet halo
固定使用 `MPI_COMM_WORLD`，并与 Python collective 串序。现行实现为 patch 创建
private singleton communicator，并把多 GiB `lonv` 改为 HDF5 MPI-IO。
`8x8x8x16` 4-rank E2 已记录 `stencil_device=cuda:*`、4/4 slab 验签通过，并由 KUPL
直接消费得到独立真残差 `4.110297e-7`。L48 的 5/8 分钟审计、GPU util、输出增长和
独立残差必须按新 job id 重新记录。`1604062` 已在 5/8 分钟看到四卡 100%，但最终
因 root 汇总 `hnn/hdg/sit` 触发 54,976 MiB cgroup OOM；后续四个张量全部使用
MPI-IO owned slab，`1604161` 并提前释放 staging patch，但仍在 `8m32s` OOM。
最终 batch 16、chunked write、已完成 `hnn` 复用和 page-cache eviction 使
`1604374` 发布 18.7 GB L48 cache（setup `499.163295 s`，4/4 slab PASS）；
`1604447` 的 4-rank Legacy-MG 为 20 iterations、`0.740402 s`、native full
residual `2.243699e-7`。Python independent residual 仍受 cgroup 限制，L48 尚无
QUDA 同窗，不能报告 L48 相对加速比。
