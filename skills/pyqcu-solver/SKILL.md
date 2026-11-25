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

> **警告：先验算 κ，再解释任何失败（`logs/v20261123.txt` §176）。**
> `mass=-0.277` 给出 `kappa = 1/(2*(-0.277)+8) = 0.1343003`，比自由
> `kappa_c = 1/8` 高 **7.4%**。单位 gauge（clover 项恒为零）或 `sigma=0.1`
> 近单位 gauge 上，Wilson-Clover 在 `cos p_x = 1/(2*kappa) - 3 = 0.723` 处有
> 精确/近零模，Schur 补奇异。实测（8 rank、grid `2 2 2 1`、跑满 1000 次迭代）：
> 相对残差停在 **7–12%**，各 rank 逐位相同且有限，不是 NaN、不是归约错。
>
> 由此三条判据，缺一条就会误判：
> - 这种输入上普通 BiCGStab 失败**不说明**求解器有问题；
> - null-vector 构建失败**不说明** builder 有问题——同一个集体 builder 已在
>   32/16/128 rank 上 PASS（§176）；
> - 在该输入上做的基比较**作废**：§176 把 §173 的 overlap `0.115` 与
>   “legacy 多抓 6.1x”一并撤销。**已在 `κ < κ_c` 上重测**（§182）：
>   [8,8,8,16]、`mass=+0.1`、两套基各由对应 builder 在计算节点构建并 `PASS`，
>   单向量门禁 ρ=1.000000、稠密随机向量达容量 `1/96`；结论是 legacy 基在
>   k≥2 上多抓 **1.07–1.38×（中位约 1.3×）**，不是 6.1×——符号对、量级错五倍。
>   ρ 随残差变光滑从 0.015 升到 0.088（容量 0.0104，上界 0.1021），仍远小于 1。
>   完整表见 `logs/v20261123.txt` §182；6.1× 与 overlap `0.115` 仍然不得引用。
>
> 收紧 `--atol` 无效，换 seed 无效；只能换**热化** gauge 重跑。远端判读表见
> [`pyqcu-testing/references/remote-l48-experience.md`](../pyqcu-testing/references/remote-l48-experience.md)。

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

### 2026-11-24 QCU-MG 层数上限是 ABI 布局，不是算法

- KUPL Legacy-MG：`cpp/kupl/qcu/src/lattice_clover_multigrid.cpp:312` 接受
  `num_levels_` **1..5**，层级由 `for (level = 1; level < num_levels_; ...)` 循环
  建立；CUDA Strict-MG：`cpp/cuda/qcu/src/apply_multigrid_strict.cu:3182` 接受
  **2..5**，并要求有效的 coarse start。
- 上限来自 ABI 槽位，不是算法：每个粗层（`level >= 1`）的 params 块从
  `19 + 8*(level-1)` 开始（`E,X,Y,Z,T,max_iter,dtype,restart`），`set_ptrs`
  四元组从 `30 + 4*(level-1)` 开始（`lonv,hnn,hdg,sit`），`argv` 容差在
  `3 + (level-1)`。这些槽位已预留，所以 3/4/5 层是现成的求解器配置。
- `pyqcu.test` 的 port 只接一条粗化 transition（一套 coarse 资产），所以今天
  `--mg-levels 3` 跑不起来。**这是 plumbing，不是物理，也不是求解器上限**；
  四处闸门逐条可改（`logs/v20261123.txt` §171、§177）：
  1. `pyqcu/test/native.py:4117-4129` 的 `if depth > 2: raise refusal`——唯一的
     CLI 门，删掉它只是把失败搬到 builder；
  2. 唯一的 builder 只建 transition 1，而 `_configure_legacy_mg` 本身已是 N 层
     （`native.py:4271` 的 `for level in range(1, depth)`），
     `pyqcu/testing/simple/mpi/worker.py:207-221` 也已按层各取一个 cache root；
  3. `pyqcu/tools/_legacy_mg_cache.py:132-136`（`legacy_mg_cache_shapes`）
     硬编码 `fine_dof = 12`，而 transition 2 的 fine dof 是 level 1 的
     `E`（实测 `4`）；
  4. ABI 包装 `cpp/kupl/qcu/src/apply_multigrid.cpp:149,167` 在 restrict/prolong
     里传 `e = kVectorSize`（=12）。第一条 3 层实验取 `E_1 = 12` 可绕开宽变体。
  求解器侧不用动：`:975-978` 的 `multigrid_coarse_dslash_wide` 不带 gauge 参数，
  `:986-1001` 把 `c.E, p.E` 与两侧 dims 直接交给通用 restrict/prolong。
- 接受不等于已验证：Legacy 的正式验证深度仍是两层、Strict 是三层，3–5 层必须
  逐层过独立全算子残差门后才能引用。

### 2026-11-26 Legacy-MG 的层级是递归的：没有 coarse gauge，也不要造一个

`op(0)` 是细层 odd-Schur Clover 算子，`op(L) = R_L · op(L-1) · P_L`：每层粗算子
是**上一层算子**的 Galerkin 投影，上一层可以已经是 33 点 stencil（§177）。全树
搜索 `coarse_gauge`/`block_average`/`coarsen_gauge`/`coarse_clover` 只命中 QUDA
散文，没有任何实现——因为不需要。

- **不要**把 coarse gauge + coarse Clover 交给
  `applyMultigridBuildNullVectorsQcu` / `applyMultigridBuildStencilLocalQcu`：
  Wilson-Clover 是 onsite + 1 hop，33 点 stencil 含 2 hop 耦合，任何 gauge 都
  复现不出来，两个 builder 会静默构建**另一个**算子的层级。
- 第 2 条 transition 的构建必须发生在 Python/torch，那里 matvec 是参数。已验证
  的原语、数值证据与锚点见
  [`references/mg-cache-comparison.md`](references/mg-cache-comparison.md#深层-transition递归而非-coarse-gauge)
  的“深层 transition”一节。
- legacy 与 Strict/QUDA 粗层表示的正式对照见
  [`docs/qcu_quda_multigrid_coarse_operator.md`](../../docs/qcu_quda_multigrid_coarse_operator.md)。

### 2026-11-24 KUPL Legacy-MG 在本实现中无法加速（实测，勿再扫参数）

先读 `logs/v20261123.txt` §142（转移审计通过）、§145（`capture_bound` 是代数）、
§150/§152（关闭仪表的三路对照）、§153（间隔即细层 restart，对照仍有混淆）。
以下每条都修正过早期版本的错误结论，不要再引用 §121/§129/§133 的旧措辞。

- **转移对是对的**：逐 dof 审计给出 `mismatches=0`、`gather_set_equals_scatter_set=1`、
  `support == block_sites * E`、`outside_block=0`、`roundtrip_err` 2.4e-07–8.3e-07。
  不存在索引/parity 映射错误——**不要去改 `restrict()`/`prolong()`**。
- **`capture_bound`/`reduction_bound` 不是测量值**，是代数
  `sqrt(E/(e*block))` 与 `sqrt(1-capture²)`，`E=2, e=12` 时对 block 16/32/256 分别
  给出 `0.1020621 / 0.07216878 / 0.02551552`，与"实测"七位吻合。粗化时它下降只是
  block 变大，**与转移无关**，不能作为证据。
- **真实测量**（关闭 `KUPL_MG_DEBUG` 才有意义，`diagnose_correction` 计在同一计时器
  内）：一次修正只去掉不到 1% 的残差，而一次细迭代去掉约 45%；`overlap` 0.013–0.046；
  `best_ratio` 0.991–0.993；即使最优阻尼也只保留约 29% 的修正量。
- **零空间基曾经是错的**：旧循环是 `v -= S⁻¹(S v) = v − v`，存的是内层求解器残差
  （白噪声）。已改为真逆迭代 `v ← S⁻¹v` 并每轮归一化，且缓存身份带
  `nv_builder="inverse-iteration/2"`。**用真基重测，结论不变**（§152）。
- **`E` 不是杠杆**：2→4 让 `best_ratio` 从 0.9941 到 0.9934、`level0` 从 18 到 24（更差）。
- **调度的混淆**（§153）：修正间隔**就是**细层 restart 长度
  （`chunk = min(remaining, correction_interval - since_correction)`），所以
  `KUPL_MG_CORRECTION_INTERVAL=2` 同时把 BiCGStab 的重启从 20 变成 2。三路对照里
  `--mg-levels 1` 那一臂其实是**不重启**的 BiCGStab，因此 4.647 / 6.504 / 7.947 s
  比较的是三种 Krylov 策略，**不能**把 1.71× 全部归给 MG。要归因必须先让一个 chunk
  能跨多次重启——这是结构改动，未做。
- 结论：本实现的两级 Legacy-MG 没有净收益。倍数本身在 2026-11-25 的一次同分配、
  同 `restart`、真复用缓存的四臂对照中已测得（下一节，**MG/plain = 1.68×–1.72×**）；
  在那之前的一切倍数仍然不得引用。

### 2026-11-25 KUPL Legacy-MG 的**合法**减速倍数：MG/plain = 1.68×–1.72×

上一节最后一句（"倍数不得引用"）已被一次**条件完备**的测量取代。作业 `2258877`，
一个 dsub 分配、单节点 `cn12139`、8 rank、`grid 2 2 2 1`、`mass=+0.1`（`κ=0.121951
< κ_c=0.125`，算符非奇异）、`atol=1e-5`、`--repeat 3 --warmup 1`，四臂**串行跑在同一
分配内**：

| 臂 | ABI 符号 | 秒（3 次中位） | 迭代 | 残差 | 缓存 |
|---|---|---:|---:|---:|---|
| plain BiCGStab | `applyCloverBistabCgQcu` | **1.1856** | 28 | 4.4086e-06 | 未请求 |
| mg-2 + legacy 基 | `applyCloverMultigridQcu` | **1.9922** | `n/a` | 4.8466e-06 | `cache_reused=True` |
| mg-2 + inverse 基 | `applyCloverMultigridQcu` | **2.0421** | `n/a` | 4.6316e-06 | `cache_reused=True` |

```
MG/plain = 1.680×（legacy 基）   1.722×（inverse 基）
```

**这次为什么可以引用**（逐条对照 §153 的混淆项）：

- **同分配**：四行同一个 dsub、同一节点、同一库 md5 `305a152cdc68`。§114 禁止的跨
  分配比较完全没有出现。
- **`restart` 相同**：四行的 `metrics.restart` **全部为 5**，所以 §153 的"修正间隔即
  restart 长度"混淆没有被引入——两臂的细层 Krylov 重启长度一致。
- **缓存是真复用**：两臂都记录 `cache_reused=True`、`asset_source=cache`、
  `cache_seconds=null`，用的是 `2258741` 产出并按 `--nv-iters 2` 建的产物。
- **差异远超噪声**：同作业内基准漂移 +0.39%（1.1856 → 1.1903），单臂样本离散
  0.13–0.36%，而 68–72% 的 MG 惩罚约为漂移的 **170×**。
- **跨分配一致性佐证**：纯求解残差与另一分配的纯求解**逐位相同**
  （`4.408633403218057e-06`）而墙钟不同——这正是 §114 要防的情形，反过来说明
  残差可比、墙钟不可跨分配比。

**两套基仍然互相不可区分**（1.9922 vs 2.0421，2.5%；另一作业 `2258741` 为 0.25%），
即 **null-vector 构建器的选择不改变代价**。所以"换基能否加速"是否定答案，"MG 是否
值得"也是否定答案——与本节的机制解释一致：一次修正去掉不到 1%，一次细迭代去掉约
45%，一次修正却要付约 2.3 次细层应用的代价。

**`--atol` 必须比 dtype 的验收门更紧**：c128 的验收门是 `1e-08`；`--atol 1e-5` 的
c128 16 rank 单元真残差 `7.17e-07`、`residual_verified=true`，仍被判 FAIL（原因串把
`7.167e-07` 与 `1.000e-08` 门比较）；改 `--atol 1e-9` 后到 `1e-11`/`5.3e-10` 并
PASS（`logs/v20261123.txt` §128、§134）。门是对的，只能收紧求解器 `--atol`；该
flag 在提交 `0bab08ee` 之前根本到不了 runner。

## Strict QUDA-style MultiGrid Invariants

See [references/strict-multigrid-invariants.md](./references/strict-multigrid-invariants.md)
for the operator recursion, coarse connectivity, runtime asset layout, and the
L32/L48 setup-budget history. Legacy-MG is the QCU default; Strict is opt-in via
`hierarchy_mode="strict"` or `StrictMultigrid`.

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
