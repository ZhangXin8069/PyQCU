---
name: pyqcu-cuda
description: 当任务涉及 pyqcu.cuda、libqcu.so Cython 桥、params/argv/set_ptrs ABI 或 CUDA 显存生命周期时使用。
---
# pyqcu.cuda

Cython bridge package for the C++ CUDA backend (`libqcu.so`).

## Files

| File | Purpose |
|------|---------|
| `__init__.py` | Makes `pyqcu.cuda` a proper Python package (added 2026-07-28 R3; was missing, causing `pip install` failures) |
| `qcu/qcu.pyx` | Cython extension source — wraps C functions from `pyqcu.h` as `applyInitQcu`, `applyWilsonDslashQcu`, etc. |
| `qcu/qcu_api.pxd` | Cython declarations — aliases C symbols, marks calls `nogil`, and must match `pyqcu.h` |
| `qcu/qcu.pyi` | Type stub (390 lines) — full type annotations, docstrings, and default values for IDE support。注意 `pyqcu/cuda/qcu.pyi` 是指向 `qcu/qcu.pyi` 的符号链接，不是第二份拷贝 |
| `define.py` | Parameter constants (`_LAT_X_`, `_SET_PLAN_`, etc.) and dtype conversion helpers (`dtype()`, `epytd()`) |
| `_runtime.py` | `QcuRuntime` — lifecycle-safe flat-ABI wrapper with private control arrays, free-slot selection, `mass`/`csw` configuration, and `operation()` contexts for getters |
| `linear_operator.py` | `CudaLinearOperator` — compact Wilson/Clover Schur apply and native whole-solve adapter |
| `krylov.py` | `CudaBiCGStabDriver` / `CudaCGDriver` — one native C++ solve per request |
| `_schur_op.py` | `CudaSchurOp` — C++ Schur odd-site operator wrapper (per-instance independent params/set_ptrs, thread-safe slot allocator; matvec uses preallocated buffer + device sync + clone to avoid deep pool/stream races) |
| `_multi_gpu.py` | **TEST-ONLY, explicit opt-in**: `MultiGpuMultigrid` — single-process multi-thread multi-GPU Legacy MG driver (one-thread-one-GPU; consistency + independent-problem modes; single-MPI-rank constraint); not the production topology and not a single-solve domain decomposition |
| `_strict_multigrid.py` | `StrictMultigrid` — fine odd-Clover Schur + persistent strict coarse V-cycle + C++ fused right-FGMRES；通过 `pyqcu.cuda.multigrid` 导出 |
| `_strict_cache.py` | strict runtime cache schema v2；逐 tensor 流式摘要、完整验签后再传入 device |
| `_strict_mpi.py` | strict MPI rank 对称 preflight 与能力门禁；当前声明 setup/full/compact/R-P halo、全局归约和分布式 fused FGMRES 均可用 |
| `_logs.py` | C++ 后端收敛日志诊断 API：`parse_mg_log(path)` → (残差列表, `PROF_SECTIONS` ms 字典, 总迭代数)；`parse_convergence_histories(path, offset)` → (histories, 新偏移) 增量收集逐次求解的 `CONVERGENCE_HISTORY`。2026-10-22 审计基线下仓内没有代码或测试导入该模块；补定向解析测试前，不应把“模块存在”写成已验证的规范入口。默认路径 `<repo>/logs/clover_multigrid.log` |

## CUDA Production Topology

All new production CUDA programs follow one invariant:

```text
one Python process = one MPI rank = one GPU/DCU/GPGPU
```

Cross-device parallelism is implemented with MPI ranks and 4D lattice domain
decomposition. Do not create multiple production device workers inside one
Python process. `MultiGpuMultigrid` is a test-only exception: importing
`pyqcu`, `pyqcu.cuda`, or `pyqcu.solver` does not construct or start it; only an
explicit import and construction enable the path. Use it only for consistency,
throughput, or diagnostic tests, never as the default driver for a new CUDA
program.

Each production process owns one rank-local `LatticeSet`/solver context.
`CUDA_VISIBLE_DEVICES` and device selection must be fixed before importing or
initializing PyTorch, the Cython bridge, or native QCU. MPI rank count should
match participating devices; oversubscribed ranks are correctness smoke only,
not an acceleration path.

Multi-rank benchmark entry points accept only positive powers of two and must
reject other counts before MPI/solver startup. Two P100 ranks require a Python
environment compatible with `sm_60` and a native library set covering
`sm_60`/`sm_70`; V100+P100 mixed-device timing is not a valid
same-architecture result.

The normative project document is
[`docs/cuda_execution_topology.md`](../../docs/cuda_execution_topology.md).
The cross-document extraction index is
[`pyqcu-docs/references/extracted-guidance.md`](../pyqcu-docs/references/extracted-guidance.md#3-cuda-生产执行拓扑).
The 2026-10-23 multi-rank and MultiGrid benchmark constraints are in
[`pyqcu-benchmark/references/qcu-multigrid-20261023.md`](../pyqcu-benchmark/references/qcu-multigrid-20261023.md).

## Public API

```python
from pyqcu.cuda import qcu      # Cython bridge to libqcu.so
from pyqcu.cuda import define   # Parameter constants, dtype helpers, pre-built params/argv/set_ptrs tensors
from pyqcu.cuda import QcuRuntime  # lifecycle-safe flat-ABI calls
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
| `applyMultigridStrictCoarseQcu` / `applyMultigridStrictMatPCQcu` | Full-coarse `X/Yhat` operator and compact-parity MATPC | strict MG |
| `applyMultigridStrictPrepareQcu` / `applyMultigridStrictReconstructQcu` | Full↔odd-Schur preparation/reconstruction | strict MG |
| `applyMultigridStrictRestrictQcu` / `applyMultigridStrictProLongQcu` | Compact fine-parity R/P with full coarse output/input | strict MG |
| `applyMultigridStrictInitQcu` / `applyMultigridStrictVCycleQcu` / `applyMultigridStrictEndQcu` | Allocate, reuse, and release the persistent strict hierarchy/arena | strict MG |
| `applyMultigridStrictFgmresQcu` | Fused restarted right-FGMRES with MPI global reductions, including warm x0 and exact workspace accounting | strict MG |

All functions take raw pointers cast to `long long` from `tensor.contiguous().data_ptr()`.

## Parameter Protocol

Three flat tensors bridge Python ↔ C++:

- **`params`** (CPU int32[58]) — lattice dims (`_LAT_X_`…`_LAT_XYZT_`), grid sizes (`_GRID_X_`…), data types (`_DATA_TYPE_`), iteration counts (`_MAX_ITER_`), plan selection (`_SET_PLAN_`), verbosity (`_VERBOSE_`), parity (`_PARITY_`), multigrid level configs and controls. The tail is `_MG_USE_GCR_` (54), `_MG_USE_DEFLATE_` (55), `_MG_MU_PRE_` (56), and `_MG_USE_INIT_GUESS_` (57); slot 57 is a 0/1 warm-start flag and does not extend the ABI.
- Multi-rank CUDA Legacy must set `params[_MG_USE_DEFLATE_]` explicitly. Leaving it unset can produce `n_vcycles=0`, which is a fine-only result and must not be reported as coarse MultiGrid performance. Verify actual `n_vcycles`/coarse iteration and correction counters instead of inferring execution from `levels` or `restart`.
- **`argv`** (real dtype, size 8) — physical parameters: `_MASS_` (0), `_ATOL_` (1), `_SIGMA_` (2), per-level MG tolerances (3–6), `_CSW_` (7); c64 使用 float32，c128 使用 float64。Clover 项使用 `-kappa*csw/8` 且 `csw=0` 合法
- **`set_ptrs`** (CPU int64[100]) — LatticeSet/scratch pointers managed by the C++ runtime. Strict transition `t` uses slots `60+4*t+0..3` for blocked `V`, optional raw `Y`, `Yhat`, and onsite pair `(X,X^-1)`; slot `80` stores the persistent hierarchy handle.

Index constants in `define.py` MUST stay in sync with `cpp/cuda/qcu/include/define.h`.

`define.py` also provides pre-built tensors `params`, `argv`, and `set_ptrs` for convenience. They are modified in-place by the solver code.

## Strict MultiGrid Boundary and Lifetime

- `hierarchy_mode="strict"`（或 `StrictMultigrid`）才选择 Strict full-coarse 语义；`setup_operator="schur"` 是 legacy compact odd-Schur/setup-vector 入口，不能当作 Strict 的别名。Strict 下即使传入该值，也不得因此把 coarse 几何或资产改成 compact。
- Strict 当前只接受每一层均为 `coarse_grid_solution_type="matpc"` 与 `smoother_solve_type="direct_pc"`；不得静默回退到 legacy hopping-only coarse dslash。
- Coarse operators and vectors retain the full lattice shape; Wilson/Clover coarse levels use `coarse_spin=2` and `E=2*nvec`. `P` 把 full coarse field 映射到选定 fine parity，`R=P†` 把该 compact fine field 限制回 full coarse field。Checkerboarding is confined to fine-side R/P, MATPC, prepare and reconstruct, never to coarse geometry.
- 当前 CUDA fused 入口固定 `target_parity=1`、`start_level=1`：fine full field 为 `[2,4,3,X,Y,Z,T/2]`，MATPC/Schur 工作向量为 `[12,X,Y,Z,T/2]`，每个 coarse 输入/输出仍为 `[E,Xc,Yc,Zc,Tc]` full field；粗层不得再次 checkerboard。用户侧 canonical null vector `[nvec,4,3,X,Y,Z,T]` 必须先转换为 C++ ABI 的 C-order blocked `[E,12,Xc,bx,Yc,by,Zc,bz,Tc,bt]`，不能把 7-D canonical 张量直接传给 fused 入口。
- 生命周期必须为 `hierarchy.setup()` → 创建 `CudaSchurOp`（内部执行 `applyInitQcu`）→ 绑定 fine/inter-level runtime assets → `applyMultigridStrictInitQcu` → 重复 `applyMultigridStrictVCycleQcu`/`applyMultigridStrictFgmresQcu` → `applyMultigridStrictEndQcu` → `CudaSchurOp.release()`（内部执行 `applyEndQcu`）。所有路径均应以 `try/finally` 保证逆序清理。
- Only after packed runtime assets are bound may setup tensors be detached with `hierarchy.seal_cuda_runtime(runtime_assets_bound=True)`. CUDA `StrictMultigrid` does this by default when `release_setup_assets=True`; preserve `strict_setup_stats` before sealing if later diagnostics need them, because sealing deliberately disables Python setup/apply/export reuse.
- Runtime binding keeps `Yhat`, onsite data and inter-level blocked null vectors. Packed raw `Y` is diagnostic/setup data and is omitted by default; request it only for raw-operator checks.
- Strict fine FGMRES consumes Gauge `[2,3,3,4,X,Y,Z,T/2]` and Clover even/odd plus inverses `[4,3,4,3,X,Y,Z,T/2]`; coarse levels do not receive a second physical Gauge/Clover pair, but use Galerkin `X/Y/Yhat` assets. `Yhat` is the left-preconditioned link (`X^-1Y`) and raw backward links retain the QUDA storage-site/adjoint convention.
- Runtime caches use schema v2 and bind a streaming SHA256 to every logical tensor. Loading first verifies the complete tensor set and only then creates/transfers device tensors; host I/O is chunked at about 8 MiB. A hit intentionally reads tensor bytes twice—once for verification and once for device transfer—without materializing a full host hierarchy. A same-identity target won by a competing publisher may be reused only after complete verification of the manifest, all dataset attrs, and all tensor SHA256 values; never trust identity or path existence alone.
- Galerkin setup memory planning must count four simultaneous full-field arenas. The library default `strict_galerkin_max_workspace_bytes=512 MiB` is an API default, not the formal benchmark contract. The formal `16×32×32×48` profile uses colored `C=12` with a `4 GiB` c64 setup cap, and `C=1` with a `1 GiB` c128 setup cap. The outer fused `max_krylov_bytes` is separate: the solver API default is `512 MiB`, while the formal profile uses `512 MiB` for c64 and `1 GiB` for c128. Never infer either budget from the other.
- With `strict_galerkin_mode="auto"`, compare site-batch and colored operator-call/memory models under the requested cap, choose site-batch only when it fits and is no slower by call count, and record requested/effective mode, column batch `C`, projection-site batch `K`, cap and observed stats. The generic builders may shrink `C/K` to fit; a formal benchmark sets an exact-batch contract and must fail closed on any shrink.
- `applyMultigridStrictFgmresQcu` lazily allocates the outer workspace in C++ on the first solve and reuses it for the same geometry/restart. Its exact device size is `(2*m+5)*B_f + 2*B_c`, where `B_f` is one compact fine-parity vector and `B_c` one full first-coarse vector; Python owns no duplicate Krylov arena or coarse-I/O tensors. CUDA `StrictMultigrid` shortens `m` before the call when `max_krylov_bytes` requires it.
- Current strict MPI declares `setup_halo=True`, `fine_halo=True`, `full_halo=True`, `compact_halo=True`, `rp_coarse_halo=True`, `global_reduction=True`, and `fused_fgmres=True`. Rank-dependent metadata/control errors are rejected through rank-symmetric preflight before collectives. Multi-rank production runs still need one process/rank/device, capability preflight and an independent true-residual gate; passing reduction-only smoke is not sufficient evidence.

Treat `memory_report()` as an explicit-ownership ledger, not whole-process peak memory. Before the first solve, fused resident bytes are zero and only planned bytes are reported; after the first solve, resident bytes must equal the formula above. For a lifecycle audit, record baseline → hierarchy setup → packed export/bind → runtime seal → strict init → first/steady solve → `close()`, deduplicate tensor storages, and separately include Gauge/Clover inputs, `LatticeSet` scratch/halos, allocator reserved/high-water state, and caller RHS/output storage. In formal benchmarks, sampler start delegates its first `cudaMemGetInfo` call to the worker thread; stop takes no final sample and retains the thread handle on join timeout so validation can fail closed. Keep the independent device-wide probe untimed, require memory schema version 2, use `device_used_max_observed_bytes`, and query `nvidia-smi` by target GPU UUID with observational fields named only `max_observed`.

The C++ `allocated_bytes` values returned by Strict init/V-cycle/fused FGMRES
describe their owned arenas only; they are not a process-wide measurement and
must not replace the device-wide probe or the ownership ledger above.

For a direct Legacy/Strict capacity decision, use the separate process-isolated
A/B records linked from
[`docs/legacy_vs_strict_mg_memory_20261023.md`](../../docs/legacy_vs_strict_mg_memory_20261023.md).
On the measured c64 two-level, `nvec=12` inputs, Legacy fine-only solve peaks
at 0.596/1.533/4.113 GiB versus Strict 0.789/2.326/6.715 GiB for
`8x8x8x16`, `12x12x12x24`, and `16x16x16x32`. Strict includes hierarchy setup and
lazy FGMRES workspace in that run; Legacy records have `n_vcycles=0`, so this
is a current-path capacity result, not a completed coarse-cycle algorithm
comparison. Strict `24x24x24x64` has no JSON record.

## Critical: `_SET_INDEX_` Increment

For an ordinary transaction, call `applyEndQcu` at the index used by
`applyInitQcu`, then increment `params[define._SET_INDEX_]`. Incrementing before
`applyEndQcu` leaves the allocated object behind. `QcuRuntime` implements this
ordering and selects a reusable free slot.

`MultiGpuMultigrid` defaults to `run_reference=True` so consistency tests can
compare MG against BiStabCG. Explicit test-only throughput runs may use
`run_reference=False`: it skips the redundant per-device reference solve and
the second ordinary `LatticeSet` initialization. Worker cleanup must keep
`applyEndQcu` in `finally`, at the same slot index used for initialization.

Legacy exception: coarse-grid dslash may reset `_SET_INDEX_` to 0 because it uses a different legacy MG level. Strict exception: `CudaSchurOp` allocates one private LatticeSet slot and all strict primitives, strict init/VCycle/fused-FGMRES and strict end must keep the same `_SET_INDEX_` for that instance; do **not** increment it between Strict calls, or the backend will look up a different/null `set_ptrs` entry. The strict asset area (`set_ptrs[60..79]`) and persistent hierarchy handle (`set_ptrs[80]`) do not replace this LatticeSet slot.

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
qcu.applyInitQcu(set_ptrs, params, argv)  # allocate at current slot
# ... one operation ...
qcu.applyEndQcu(set_ptrs, params)          # free the current slot
params[_SET_INDEX_] += 1                   # advance for the next transaction
```

Prefer `QcuRuntime.call()` / `QcuRuntime.operation()`: they select a free slot,
perform this sequence exception-safely, and support >100 sequential operations.
Do not increment before `applyEndQcu`; it frees by the current index. Strict
multi-call lifecycles keep one private index unchanged throughout.

Strict calls use the fixed per-instance slot instead:

```python
hierarchy.setup()
schur = CudaSchurOp(argv, gauge, clover_ee, clover_oo,
                    clover_ee_inv, clover_oo_inv, params=params)
# bind strict assets; CudaSchurOp construction already called applyInitQcu
qcu.applyMultigridStrictInitQcu(schur.set_ptrs, schur.params, 1)
try:
    qcu.applyMultigridStrictFgmresQcu(..., schur.set_ptrs, schur.params, ...)
finally:
    qcu.applyMultigridStrictEndQcu(schur.set_ptrs, schur.params)
    schur.release()  # applyEndQcu；期间不要修改 _SET_INDEX_
```

---

## Related skill documents

`qcu/` 子目录（`qcu.pyx` / `qcu_api.pxd` / `qcu.pyi`）自身是独立技能范围，细节以那两份文档为准。
本文**不复制**其正文——此前内嵌于此的 `qcu/` 文件表、C API 表与调用生命周期副本已于
2026-10-15 删除，避免同一内容在同一文件内出现两遍、并与其源文档多处漂移：

| 范围 | 技能 |
|---|---|
| `pyqcu/cuda/qcu/`（Cython 桥全貌，含 `qcu.pyi` 类型约定） | `skills/pyqcu-cpp-qcu/SKILL.md` |
| C++↔Cython 边界两侧（`pyqcu.h` 40 个入口、`qcu_api.pxd`、`qcu.pyx`） | `skills/pyqcu-cpp-qcu-python/SKILL.md` |
| QUDA 对照侧的 refer/quda 构建与 PyQUDA 调用契约 | `skills/pyqcu-testing-qcu/SKILL.md`（2026-10-15 自本文件移入） |

两条只在此处出现、故予保留的约定：

- strict 入口先返回状态并校验 shape/contiguity，之后才进入 `nogil` 代码段。
- `qcu_api.pxd` 必须与 `cpp/cuda/qcu/python/pyqcu.h` 的 C 声明逐字一致；任何漂移都可能造成**静默内存损坏**。

---

## Strict trace v2（2026-09-16）

`apply_multigrid_strict.cu` keeps tracing disabled unless
`PYQCU_STRICT_TRACE_FILE` is set.  Trace version 2 adds
`residual <outer> <level> <phase> <absolute> <relative> <elapsed>` alongside
the existing stage and outer-FGMRES records.  Residual probes use dedicated
trace-only workspaces and the same fine/coarse MATPC operator as the solve;
with no environment variable, no file and no trace workspace are created.

`StrictMultigrid.outer_iterations` is the authoritative count of
outer right-preconditioned FGMRES columns.  `iterations` remains a
compatibility alias and must not be described as the sum of smoother or
coarse-solver iterations.

## 2026-10-21 真实 LIME、MG 输入与多卡通信

测试专用的 `MultiGpuMultigrid` 现支持注入外部 `gauge/source`、`input_tag`、
真实 MG 残差校验和 setup/solve 分段计时。真实输入应传入 QCU parity 布局；
粗算子缓存 key 必须绑定 gauge/source SHA，不能继续使用 `generated:<seed>`。

本集群 A100 + OpenMPI 的 CUDA-aware 路径会在分布式 `wilson_dslash.run_oe`
首个 halo 阻塞；`PYQCU_MPI_DEVICE_AWARE=0` 的 pinned-host staging 可正常完成。
真实 LIME 矩阵脚本默认必须使用 host staging；只有在新 MPI/驱动组合上通过
单步 halo 门禁后才能恢复 device-aware 并报告吞吐。

大格 legacy MG setup 默认在 `E_c * Nc >= 200000` 时走
`BatchedLocalSchur + build_stencil_local`；`PYQCU_MG_LOCAL_STENCIL=0/1` 可
强制 A/B，`PYQCU_MG_BUILD_PROGRESS=1` 输出 level、null-space、stencil 阶段。
长任务必须至少记录 `setup_seconds`、worker `mg_time` 和全算子残差，禁止只报
含缓存构建的总墙钟。

## 2026-10-21 local stencil gather 与 W=8

`BatchedLocalSchur._flat_window_index` 将周期窗口 `[K,4,W]` 一次展平，
`_slicem` 以 expanded base + `torch.gather` 批量复制 K 个窗口；每次 local
operator 调用只构造一次索引。V100 同形状 microbenchmark 中 K=4 为
`4.027 -> 0.272 ms`，wrap/non-wrap 逐位一致。

生产 local Schur 窗口由 `W=10` 收紧到经 full-grid 与周期边界回归验证的 `W=8`；
同形状单次 local-op 中位 `27.337 -> 13.725 ms`。真实 `24³×64` c64 二层 MG
stencil `598.107 -> 323.745 s`（45.87%），总 cold setup
`616.343 -> 327.587 s`（46.85%），MG 仍为 20 iterations、true residual
`4.1247e-7`。不得继续缩小到 `W=6`：非平凡矩阵边界回归已出现局部元素不符。

### legacy Clover BiCGStab 单流化

单 rank Clover BiCGStab 使用 main stream 排队所有 reduction、标量更新、dslash 与
融合的 `r/x` 更新，每迭代只同步一次；`beta/rho_prev` 合并，`r=s-ωt` 与
`x+=αp+ωs` 合并。停机改用刚更新的 residual，因此真实外迭代由 19 降到 18。
Clover multiply 逐行直读，不再物化 144-element 线程本地矩阵。

真实 `24³×64` c64、10 样本：`0.189074 -> 0.157841 s`（16.52%），18 iterations，
独立 true residual `6.6952e-7`。同一时段 QUDA wall 中位 `0.137520 s`，CUDA 仍慢
约 14.8%。不要按 `make_clover` 的 6×6 稀疏结构裁剪通用入口：strict nontrivial
Clover 回归证明外部矩阵可含跨块项，稀疏候选已撤销。

### 成对点积归约（2026-10-21）

`bistabcg_give_1omega` 前的 `t^H s` 与 `t^H t` 已合并为
`bistabcg_dot_pair_complete`。kernel 每个 block 同时累加两个点积，最后一个 block
按固定 partial 顺序完成汇总；completion counter 每次 launch 前异步清零，禁止用
浮点 atomicAdd 破坏确定性。multi-rank 五流参考路径不变。

同一动态库 ABBA、真实 `24³×64` c64、20 个 steady 样本：Clover BiCGStab
`0.161293 -> 0.157776 s`（2.18%，1.0223x），均 18 iterations，独立相对真残差
`6.6952e-7 -> 6.6950e-7`。二层 MG 后端 `0.1746 s` 不变，生命周期
`0.417192 -> 0.414053 s`；因此该收益归因于 null-vector/BiCG 构建，不得外推为
V-cycle 提速。tier0 20 项、cuda-strict 11 项、fused-FGMRES 3 项及非平凡 Clover
strict 门禁通过。

### 6x6 结构门控与通用回退（2026-10-21）

legacy Clover-MG 在 `init` 中扫描 `clover_ee_inv` 和 `clover_oo` 的 off-block
条目。只有全部严格为零时才启用 6x6 multiply 内核；检测到任意非零、NaN 或外部
通用 Clover 矩阵时保持完整 12x12 内核。探测只在每次 MG 调用开始时做一次，使用
设备 flag 加一次 D2H，不新增 ABI 参数。

生产 `24^3x64` c64、二层 cache-hit、两轮 ABBA × 10 steady：MG backend
`0.175150 -> 0.147050 s`（降低 16.04%，1.1911x），lifecycle
`0.414565 -> 0.386002 s`（降低 6.89%）；均为 20 iterations、真残差
`4.1247345e-7`。Stage16 与 Stage17 最终解逐元素相同。向 off-block 注入
`1e-20` 后，检测自动回退通用路径，通用路径 3 次中位 `0.179510 s` 且残差不变。
不得在未检测的情况下无条件裁剪通用入口。

该探测也用于 legacy Clover BiCGStab 的完整 solve `init`，但不在单 matvec
wrapper 中执行，避免 stencil build 重复扫描。真实 `24^3x64` c64 ABBA、20 steady：
`0.159155 -> 0.132760 s`（降低 16.58%，1.1988x），lifecycle
`0.197254 -> 0.169155 s`（降低 14.25%），均 18 iterations、真残差
`6.6950e-7`；两版最终解逐元素相同。当前 CUDA wall 比 QUDA `0.137520 s` 快
约 3.59%，但每迭代 `7.376 ms` 对 `7.238 ms` 仍慢约 1.90%，优势来自
18 对 19 次外层迭代。

`r/x` update 与新残差范数的融合候选应在 block-stride 覆盖全部 site 后比较；
本机修正版两轮仅为 `0.128986/0.133303 s`，相对 Stage18 中位改善约 `0.23%`，
第二样本回到噪声上界，未保留。不要把首个低点 `0.1290 s` 单独写成融合收益。

### 收敛同步分块（2026-10-21）

单 rank legacy Clover BiCGStab 每 2 次迭代才执行一次 `r^H r`、D2H 与 host
synchronization；中间迭代只省略不参与控制流的范数，Krylov 轨迹仍完整执行。
真实 `24^3x64` c64 两轮 A/B × 10 steady：`0.127732 -> 0.125572 s`
（降低 1.69%，1.0172x），均 18 iterations、真残差 `6.6950e-7`，最终解逐元素
相同。同热态 QUDA `0.134009 s`，CUDA wall 快 6.72%；每迭代 `6.976 ms`
对 `7.053 ms`，CUDA 快 1.09%。收敛发生在奇数迭代时最多多执行一次，停机上界
仍是下一偶数次检查。

### legacy MG fine 成对归约（2026-10-21）

单 rank legacy Clover-MG 的 fine Schur fast path 复用
`bistabcg_dot_pair_complete`，把每轮 `<t,s>` 与 `<t,t>` 合并为一次归约。
真实 `24^3x64` c64、cache-hit、ABBA 两轮共 20 steady：
MG backend `0.14125 -> 0.13700 s`（`3.01%`），lifecycle wall
`0.372507 -> 0.368274 s`（`1.14%`）；均 20 iterations、`pair_checks=20`，
真残差 `4.1247345e-7 -> 4.1255681e-7`。strict tier1 20+11+3、直接
c64/c128 Clover 8 项、ABI、MG breakdown 与 c128 pair smoke 全绿。
该优化只覆盖单 rank 路径，不能报告为 multi-rank 或 coarse V-cycle 收益。

### legacy MG fine 单点积 fast path（2026-10-21）

fine Schur fast path 的 `rho`、`rtv` 与检查范数统一改用
`bistabcg_dot_complete`；它与成对归约共用 block 数、partials 和 completion
counter，保持固定 partial 汇总顺序，并让 cuBLAS 退出该单 rank 迭代关键链。
以 Stage21 成对归约库为基线，ABBA 两轮 20 steady：MG backend
`0.13760 -> 0.13295 s`（`3.38%`），lifecycle `0.362613 -> 0.356707 s`
（`1.63%`）；相对 Stage20 原始基线累计降低 `5.88%/4.24%`。均 20 iterations、
`single_checks=45`、`pair_checks=20`，真残差 `4.1257200e-7`。strict tier1、
直接 c64/c128 Clover、ABI、c64 breakdown 与 c128 smoke 全部通过。

### 6x6 register specialization rejected（2026-10-21）

将 runtime block-diagonal multiply 改成模板特化、按两个 6x6 子块顺序累加，c64
kernel 仍为 88 registers。三轮 ABBA 共 30 steady 仅改善 MG backend `0.19%`，
lifecycle 慢 `0.09%`，已完整撤销。寄存器峰值不只由 Clover 的 12-row 临时数组
决定，不能仅凭少一半累加器就预期 occupancy 提升。

### E12 coarse V-cycle 生产筛选（2026-10-22）

固定 LIME/c64/cache-hit 的同进程结果：fine-only lifecycle `0.365086 s`；BiCG
加一次 deflate V-cycle `0.401633 s`；FGMRES+MR 最好 `0.832060 s`；FGMRES+CG
`1.160491 s`；CA-GCR+MR `1.911062 s`。deflate 把 backend 真残差降到
`2.666e-7`，但 outer iterations 仍为 20，一次 V-cycle 约 `40 ms`，其中 coarse
check `~30 ms`。E12/单 null-vector 粗空间不足以抵消成本。

生产 fine-only Stage22 记录本身是 `n_vcycles=0`；只有上面“deflate V-cycle”
实验真正执行了一次 coarse correction。不要把 Stage21/22 的 fine-path 收益写成
V-cycle 收益。

按当前 33-tensor 精确 shape，E24 cache 约 `5.22 GB`，E48 约 `18.86 GB`；E12
local-stencil setup 已有约 `29.75 GB` peak。32 GiB V100 下继续扩 E 前必须先实现
compressed/matrix-free coarse operator 或显著降低 setup workspace。

### standalone Clover BiCGStab single-dot path（2026-10-22）

Stage22 的 `bistabcg_dot_complete` 现同时覆盖 standalone Clover BiCGStab 的
`rho`、`rtv` 与每两次迭代检查范数；末尾双点积仍走 pair kernel。ABBA 两轮共 20
steady：`0.123997709 -> 0.120461542 s`（`2.85%`），均 18 iterations，独立真残差
`6.694990e-7 -> 6.694679e-7`。相对 QUDA `0.134009 s`，dev101 历史窗口 wall
快 `10.11%`；dev103 最终同窗主口径改用 CUDA `0.147801 s` 对 QUDA `0.156680 s`，
wall 快 `6.01%`，每迭代仅快 `0.43%`。不得把两个窗口混成同一倍率。
每迭代 `6.692` 对 `7.053 ms`，快 `5.12%`。strict tier1、c64/c128 Clover 与
ABI 全绿；multi-rank 路径不变。

无 cache 冷构建总 setup `356.179 -> 353.635 s`（`0.71%`），但 null-space
`26.524 -> 26.677 s` 在噪声内，stencil 主导 setup。不得把冷构建变化单独归因于
单/成对归约，也不得声称 null-vector 构建显著提速。
