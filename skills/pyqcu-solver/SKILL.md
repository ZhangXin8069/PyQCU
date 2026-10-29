---
name: pyqcu-solver
description: |
  当任务涉及 pyqcu.torch.solver、pyqcu.solver.multigrid、Krylov，
  或需要比较、选择、排障 Legacy/Strict MultiGrid 与 cache 时使用。
---
# pyqcu.torch.solver + pyqcu.solver.multigrid

Iterative solvers for the Dirac equation D ψ = η.

`Legacy-MultiGrid` is the QCU default multigrid solver and the canonical
development/debug/optimization target. `Strict-MultiGrid` is explicit-only:
use it for comparison, algorithm reference, or special fallback, never as an
implicit default.

MultiGrid depth is guarded by both validated depth and actual coarse geometry.
Legacy is validated at two levels and warns above two; Strict is validated at
three levels and warns above three. Both variants also warn when the
post-process-grid, parity-compressed coarse lattice becomes too small, has a
volume below the safe threshold, is not divisible by the block, or leaves no
room for another stable transition. The warning is deliberately not a silent
fallback or a correctness claim. Experimental depths must pass both native and
independent full-operator residual gates, and distributed Legacy depth greater
than two is not a supported performance path.

## MultiGrid cache 路线选择

比较、选择、加载或排障 `33-tensor cache` 与 `Strict cache` 时，先读取
`references/mg-cache-comparison.md`。最少保持以下边界：

- 33-tensor cache 直接保存 `A_c=P†SP` 的 1 onsite + 8 nearest + 24 diagonal block，
  对应 compact odd coarse geometry。
- Strict cache 保存 `V/Yhat/(X,X^-1)`，粗层保持 full lattice；raw `Y` 不是普通
  solve 的必需运行期资产。
- 两种 cache 的槽序、geometry、C++ kernel 和生命周期不同，不能改名或重排后互载。
- cache 性能必须分别记录验证/加载、setup、首次 workspace 和 solve；不要把两者边界混成
  同一 `setup_seconds`。
- 同输入 c64 两级、`nvec=12` 的现有 CUDA 记录中，Legacy 的实际 fine-only solve
  设备峰值低于 Strict；Strict 的 full-coarse assets、fine transfer 和 FGMRES arena
  更大。Legacy 记录尚未执行 coarse V-cycle，不能写成 MG 算法级普适结论。详见
  [`docs/legacy_vs_strict_mg_memory_20261023.md`](../../docs/legacy_vs_strict_mg_memory_20261023.md)。

需要从正式文档核对 legacy/Strict/QUDA 的算法路径、默认值和粗算子表示时，读取
[`pyqcu-docs/references/extracted-guidance.md`](../pyqcu-docs/references/extracted-guidance.md#4-qcustrict-与-quda-multigrid)。
跨 Torch/CUDA/KUPL 的统一架构见
[`docs/qcu_unified_backend_strategy_20261023.md`](../../docs/qcu_unified_backend_strategy_20261023.md)。
`pyqcu.api` 的 P0 配置、结果、能力、资产清单和 Protocol 已实现，现有
`Multigrid.solve_result()` 只映射后端已暴露字段；`create_backend()` 和完整 backend
工厂尚未实现。契约主责技能为 [`pyqcu-api/SKILL.md`](../pyqcu-api/SKILL.md)，正式原文见
[`docs/api_contract.md`](../../docs/api_contract.md)。

## CUDA execution topology

CUDA/GPGPU 生产程序保持一 Python 进程 = 一 MPI rank = 一设备；跨设备 MG 通过
MPI 格点分域。`MultiGpuMultigrid` 是显式调用的测试专用例外，不得作为 solver
默认驱动。完整规则见
[`pyqcu-cuda/SKILL.md`](../pyqcu-cuda/SKILL.md#cuda-production-topology)。

## 三后端统一接口与资源复用

Krylov/BiCGStab/CG、Legacy/Strict MultiGrid 和 MG cache 的公共签名、返回对象、能力门禁
与后端差异边界以
[`pyqcu-api/SKILL.md`](../pyqcu-api/SKILL.md#三后端统一接口强制规范) 为唯一权威源。
高层调用统一使用 `Solver`/`Krylov`、`Multigrid` 和 `MultigridCache`；backend-specific
类只负责适配。新增公共入口前先复用或扩展现有选择器，不建立平行包装。

存在 identity 已验证的 MG cache、已构建后端库、已生成测试输入或可续构建对象时，直接复用；
cache 命中必须先通过 manifest、gauge/physical identity、shape/dtype 和 tensor digest 校验，
然后跳过 Galerkin/null-space 重建。只有资源缺失、identity 不匹配或内容损坏时才重建，并在
报告中区分 `cache_hit`、验证/加载时间与真实 solve 时间。

## CUDA 混合路径约定（2026-08-14）

- `_matvec`/`_restrict_cuda`/`_prolong_cuda`/`_coarse_dslash_cuda` 的 C++ 调用后必须
  `torch.cuda.synchronize()`（C++ 私有流写固定输出缓冲，与 torch 默认流无跨流同步）。
- BiCGStab breakdown 自动重启（保留 x/r，重置影子向量与系数，阈值 1e-20 +
  alpha/omega 有限性检查），不再抛 RuntimeError。

## Files

| File | Purpose |
|------|---------|
| `api/` | 后端无关的 `SolveResult`、Capabilities、AssetManifest、控制配置和 Protocol |
| `solver/krylov/` | `Krylov(backend, method, operator, control)` 统一选择器 |
| `solver/_native_qcu.py` | CUDA/KUPL compact-Schur LinearOperator 与原生整次 solve 适配 |
| `torch/dslash/_adapter.py` | `TorchLinearOperator` full/Schur 与 γ5-Hermitian adjoint |
| `torch/solver/_krylov_drivers.py` | Torch BiCGStab/CG `SolveResult` driver |
| `torch/solver/_bistabcg.py` | BiCGStab(l) solver (Bi-stabilized Conjugate Gradient) |
| `torch/solver/_legacy_multigrid.py` | `LegacyMultigrid`: adaptive 33-stencil V-cycle with optional CUDA acceleration at the finest level |
| `torch/solver/multigrid/` | Concrete Torch `StrictMultigrid` / `LegacyMultigrid` hierarchy entries |
| `solver/multigrid/` | Backend-neutral `Multigrid` selector retained at top level |
| `torch/solver/_gmres.py` | Python restarted-FGMRES reference with right preconditioning, complex Givens rotations, warm start and restart true-residual checks; production strict CUDA solve uses the fused C++ entry |
| `torch/solver/_mr.py` | `mr` — MR 最小残差迭代（quda `inv_mr` 思想，非 Krylov 平滑器）：`p=A†r; α=ω·<p,p>/<Ap,Ap>`，每迭代一次 A + 一次 A†，内存 O(1) 向量数；作 MG smoother 或粗略预求解 |
| `torch/solver/_multishift_cg.py` | `multishift_cg` — 多质量 CG（quda `updateAlphaZeta` 递推）：一次 matvec 序列同时解全部移位系统 |
| `torch/solver/_cacg.py` | `cacg` — CA-CG（s-step CG，quda `inv_ca_cg.cpp` 思想）：MGS 正交化基 + 极小残差块解，每外迭代一次同步块 |
| `torch/solver/_lanczos.py` | `tr_lanczos` — thick-restart Lanczos 特征底座（Wu–Simon arrowhead 重启，deflation 用 Ritz 对），服务 GMRES-DR/eigCG 类 deflation |
| `torch/solver/_multigrid_core.py` | Strict transfer, coarse operator, MATPC and recursive hierarchy reference implementation |

各模块的算法细节、与 quda 的有意偏差、以及实测收敛数字（含反模式，如 multishift
的 σ₀ 折叠不可移除、tr_lanczos 的 H 必须对称双写）见 `pyqcu/torch/solver/AGENTS.md`
的「导出 API」与「2026-08-22 整合新增」两节。

## Exported API

### `Krylov(backend="torch"|"cuda"|"kupl", method="bicgstab"|"bistabcg"|"cg", operator=..., control=...)`

- `bistabcg` 是 `bicgstab` 的兼容别名。
- Torch 使用 `TorchLinearOperator` 与 `_krylov_drivers.py` 的 Python 迭代。
- CUDA/KUPL 只接受后端绑定的 compact-Schur Operator，driver 每次调用一次原生 solve；
  `operator.apply()` 不得用于 Python Krylov 循环。
- CUDA/KUPL CG 只接受 `CGOperator.NORMAL`，求解 `S†S`；普通 Wilson 或默认 HPD 必须拒绝。
- 原生 ABI 未返回残差时 `final_residual=None`，正常返回只表示 C++ 最终 full-residual 门通过。

### `bistabcg(b, matvec, tol=1e-6, max_iter=1000, x0=None, if_rtol=False, verbose=True) → torch.Tensor`

Standard BiCGStab solver. Takes a callable `matvec(src) → dest`.

**Breakdown detection** (added R2): raises `RuntimeError` on:
- `rho ≈ 0` (r_tilde orthogonal to r — loss of orthogonality)
- `vdot(r_tilde, v) ≈ 0` (pivot breakdown)
- `vdot(t, t) ≈ 0` (t is zero/near-zero)

**Tolerance:** `if_rtol=True` uses `tol * ||b||`; otherwise uses absolute `tol`.

### `LegacyMultigrid` class

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
- legacy 普通事务按 `applyInitQcu → operation → applyEndQcu → _SET_INDEX_ += 1`
  执行；不能在 end 前递增。Strict `CudaSchurOp` 序列使用固定私有槽位，见下文例外

**C++ backend integration (level 1):**
- `applyMultigridRestrictQcu`/`applyMultigridProLongQcu` for inter-grid transfers
- `applyMultigridCoarseDslashQcu` for coarse-grid operator application
- Hopping matrices packed as `[2, 4, E, E, Xc, Yc, Zc, Tc]` (pm dir Eout Ein XYZT)

**BiCGStab state reset after coarse-grid correction (R3 fix):** After a coarse-grid correction `x = x + e_fine`, the residual `r` changes, so all BiCGStab state must be reinitialized: `r_tilde = r.clone()`, reset `p/v/s/t` to zero, reset `rho_prev/alpha/omega` to 1.0. Without this, `rho = vdot(r_tilde_old, r_new)` gives meaningless results.

**Convergence tracking:** Records `r_norm` twice per iteration (before and after coarse-grid correction). Plot shows both.

## Legacy CUDA selection and stopping semantics

`simple/qcu_multigrid.py` is one driver, not the definition of every Legacy entry.
Its current defaults are two levels, `dof=[12,48]`, block `2^4`, `restart=5`,
`coarse_max_iter=200`, `coarse_tol_factor=1e4`, and `params[54:58]=0`. This selects
odd-Schur BiCGStab plus periodic additive coarse correction, not a right-preconditioned
QUDA-style MG. Setting `levels>1` alone does not select FGMRES/CA-GCR preconditioning.

| Backend | Legacy | Strict | Fail-closed boundary |
|---|---|---|---|
| Torch/CANN | Yes | Yes | Torch reference path; no claim of CUDA performance |
| CUDA QCU | Yes | Yes, fused right-FGMRES | Unsupported geometry/operator modes must not fall back to legacy kernels |
| KUPL QCU | Yes | Yes | Same-precision c64/c128 only; CUDA-only GCR/CA-GCR modes must be rejected |

- The default BiCGStab path checks the relative Schur residual every 4 iterations and
  runs a 30-step calibration before comparing 12-iteration windows. Two consecutive
  windows with less than 15% improvement permanently disable correction for that solve;
  near convergence also skips correction. Single-rank true Schur refresh executes at
  iteration counts satisfying `total % 48 == 0`; the old nearby comment saying 50 is
  stale.
- The default correction path has no fine smoother. Right-preconditioned MG applies a
  fixed-step CG/MR/Chebyshev smoother; `_MG_MU_PRE_ <= 0` falls back to 4 steps and the
  path has no fine post-smoother. MR and Chebyshev modes are mutually exclusive.
- Optional outer modes are BiCGStabL(2), FGMRES(10), and CA-GCR block 4. BiCGStabL
  performs a true restart every 8 cycles, FGMRES can gate a non-improving preconditioner
  to identity, and CA-GCR falls back to FGMRES on Gram breakdown. Default cycle is V;
  with two levels, W/F/K have no effective child-level difference.
- A cold coarse solve divides `max_iter` across pre/post relaxers; a warm coarse solve
  uses 32 pre + 32 post steps. The terminal is guarded BiCGStab despite the historical
  `multigrid_coarse_solve_cg` name. Single rank with `vec_sz < 262144` may use the fused
  cooperative kernel; multi-rank disables cooperative fusion and CUDA Graph. Coarse RHS
  below `1e-4` is skipped, and breakdown discards that correction.
- Large-volume production local-stencil callers explicitly use
  `BatchedLocalSchur(W=8)`. The class constructor default remains `W=10`; do not reduce
  to `W=6`, which failed nontrivial periodic-boundary regression.
- `n_vcycles=0` means no coarse correction executed. A configured `restart=5` or
  Strict `restart=20` is a Krylov/restart parameter, not a cycle count. Never derive an
  MG speedup from those fields. See
  [`docs/qcu_quda_multigrid_algorithms.md`](../../docs/qcu_quda_multigrid_algorithms.md)
  and [`docs/multigrid_deprecation_audit_20261022.md`](../../docs/multigrid_deprecation_audit_20261022.md).

### 2026-10-23 Legacy 多层正确性边界

Legacy 多级结果的正确性门是求解返回后独立重建 full Wilson/Clover true residual，
不能只看 C++ native residual。旧记录中 2-rank 三层和 1-rank 四层的独立残差
`0.24734981087334906`/`0.24726009368896484` 已定位为递归 V-cycle 返回时错误恢复
最细层 volume 的 C++ bug；修复后 `24×24×24×64` c64 的单 rank 3L/4L 和 2-rank
3L/4L 均通过，独立残差分别约为 `2.60e-7`（单 rank 3L/4L）和 `2.45e-7`
（2-rank 3L/4L）。

Legacy correction 现采用确定性 fail-safe：post residual 未改善则回滚并关闭本
solve 的后续 correction，不再依赖浮点斜率临界门。验证深度仍为 Legacy 两层、
Strict 三层；3L/4L 只作为显式实验路径，depth warning 说明几何/深度风险，不能代替
correctness gate。完整矩阵、实际 cycle 计数和证据链接见
[`pyqcu-benchmark/references/qcu-multigrid-20261023.md`](../pyqcu-benchmark/references/qcu-multigrid-20261023.md)。

## Strict QUDA-style MultiGrid Invariants

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
多 rank overlap stencil 按当前实现切换到 CPU。5 分钟审计时 4 个 rank 各约
98% CPU，而 4 张 A100 均为 0%、每卡驻留约 8.7 GiB；作业因 GPU 空占被主动取消，
没有 cache 或 solve 结果。按 L32 单 rank GPU stencil `153.889 s × 5.0625`
仅得约 779 秒的量级估计，不能作为 L48 实测。后续必须实现 GPU-native 多 rank
overlap stencil/独立 Clover halo communicator，或复用已验证 cache，才可重启 L48。

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

## 多层粗基与迭代口径（2026-09-16）

`StrictMultigrid(..., propagate_null_vectors=True)` propagates each later
transition's basis as `R` applied to the preceding global `B`, matching
QUDA's `generate_all_levels=false` path.  It is strict-only; the benchmark
uses one canonical full fine basis and derives the next coarse basis
deterministically.

`StrictMultigrid.outer_iterations` counts outer FGMRES columns for the CUDA backend;
`iterations` is only a compatibility alias.  The earlier `4.3817x` value was
withdrawn: QUDA had been left on a different smoother/coarse-iteration budget.
For a matched one-recursive-cycle comparison, rerun with QUDA MR smoother,
`smoother_tol=0`, and `coarse_solver_maxiter=1` outside the coarsest level.
The current sm70 c64 `16·32·32·48` three-level result is 14 outer iterations
in `0.655570 s` versus QUDA's 39 outer GCR iterations in `1.314577 s`
(`2.0052x`), both odd--odd MATPC.  Report the old value nowhere except as a
retraction.

The same run must include per-side plain BiCGStab reference records, excluded
from the speedup.  In the current large c64 point PyQCU MG/BiCGStab are
`0.655570/1.668540 s`, while QUDA MG/BiCGStab are
`1.314577/0.329849 s`; this contrast is required context, not a substitute
for the MG-vs-MG ratio.  Compare `--quda-strategy aligned` as the matched-cycle
primary and `--quda-strategy library` as a sensitivity point.

Keep the same-lattice two-level ablation in the report: on
`16·32·32·48` c64, PyQCU/QUDA are `1.944663/2.183002 s` at two levels and
`0.655570/1.314577 s` at three levels.  The corresponding PyQCU speedup grows
from `1.1226x` to `2.0052x`, which isolates the third-level benefit from
input, precision and strategy differences.

`_cg_orthogonalise()` now keeps only a read-only source and one destination
buffer per CGS pass instead of cloning both source and work.  `_block_orthogonalise()`
also allocates the C++ blocked layout as the primary storage and exposes the
Python order as a permuted view, so `to_qcu_blocked()` is zero-copy for the
matching dtype/device.

The c128 `16·32·32·48` full-double hierarchy now completes with the extended
setup protocol `strict_galerkin_column_batch=4`,
`strict_galerkin_projection_batch=16`, CPU canonical-block staging, and
null-basis offload.  With parity-bucket source ordering, the two PyQCU
transition setup times are 351.33 s and 202.72 s, total 585.28 s; the
source color count drops from 22 to 16 and C=4 calls from 132 to 96 per
level.  After `seal_cuda_runtime()` the benchmark must retain the detached-storage
evidence and release unused allocator blocks with `torch.cuda.empty_cache()` when the
caching allocator does not reuse them; otherwise the steady workspace can OOM.  The
call is an allocator-capacity remedy, not a solver-protocol step. The resulting
formal cache-hit solve is 20 outer iterations, `2.061594 s`, true residual
`1.78798e-8`; QUDA double gives 60 outer iterations, `16.166481 s`, and
`1.79244e-8`.  The conservative 7.8417x MG ratio is formal for solve
(the first independent run was 8.3507x).  The cold C=1 setup now also
completes via `release_python_setup_assets()` before `applyInitQcu`:
setup `2709.45 s`, steady `2.067186 s`, 20 iterations, and
`1.78798e-8` residual.  Keep this cold-start time separate from the
cache-hit solve ratio.
The cache identity binds physical assets, and a hit validates geometry plus
every tensor SHA rather than replaying the historical construction batch.

The experimental `strict_galerkin_block_dtype=torch.float32` path reduces
setup cost but changes coarse-block precision; on c128 small it increases
outer iterations from 14 to 35.  Treat it as a capacity/tuning experiment,
not as a substitute for the double-coarse main protocol.

## 2026-10-21 真残差与局部 stencil 构建

不要把 `applyDslashQcu(inverse_clover) + onsite` 的临时推导直接当作通用 true
residual 公式；在真实 `24³×64` c64 上该公式即使对单 rank 正确解也报约
`0.746`。可信 QCU 主口径是先保留 even/odd Clover 项，root-gather 后由独立
PyQCU `give_wilson + give_clover` 计算；计时与残差计算必须分离。

large-volume legacy MG 的第一层 stencil 在 `E_c*Nc` 较大时由生产调用方采用
`BatchedLocalSchur(W=8) + build_stencil_local(batch_size=4)`；类默认仍为
`W=10`。其局部窗口在
中心 `c±1` 块必须与全格 `P†DP` 等价；K=1/K>1 回归见
`test_stencil_batch_local.py`。缓存命中后正式 solve 只报告 worker MG 时间，
setup/load、首次 workspace 分配和残差探针必须另列。
