---
name: cuda
description: pyqcu.cuda 目录的完整生成 skill：C++ CUDA 后端（libqcu.so）的 Cython 桥接包；含 strict QUDA-style MultiGrid、params/argv/set_ptrs 参数协议与显存生命周期约束。
---
# pyqcu.cuda

Cython bridge package for the C++ CUDA backend (`libqcu.so`).

## Files

| File | Purpose |
|------|---------|
| `__init__.py` | Makes `pyqcu.cuda` a proper Python package (added 2026-07-28 R3; was missing, causing `pip install` failures) |
| `qcu/qcu.pyx` | Cython extension source — wraps C functions from `pyqcu.h` as `applyInitQcu`, `applyWilsonDslashQcu`, etc. |
| `qcu/qcu_api.pxd` | Cython declarations — aliases C symbols, marks calls `nogil`, and must match `pyqcu.h` |
| `qcu/qcu.pyi` | Type stub (155 lines) — full type annotations, docstrings, and default values for IDE support |
| `define.py` | Parameter constants (`_LAT_X_`, `_SET_PLAN_`, etc.) and dtype conversion helpers (`dtype()`, `epytd()`) |
| `_schur_op.py` | `CudaSchurOp` — C++ Schur odd-site operator wrapper (per-instance independent params/set_ptrs, thread-safe slot allocator; matvec uses preallocated buffer + device sync + clone to avoid deep pool/stream races) |
| `_multi_gpu.py` | `MultiGpuMultigrid` — multi-thread multi-GPU C++ Clover MG driver (one-thread-one-GPU; consistency + independent-problem modes; `save_report` JSON; single-MPI-rank constraint) |
| `_strict_multigrid.py` | `CudaStrictMultigridSolver` — fine odd-Clover Schur + persistent strict coarse V-cycle + C++ fused right-FGMRES |
| `_strict_cache.py` | strict runtime cache schema v2；逐 tensor 流式摘要、完整验签后再传入 device |
| `_strict_mpi.py` | strict MPI rank 对称 preflight 与能力门禁；阶段 1 仅启用 c64/c128 全局标量归约 |

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
| `applyMultigridStrictCoarseQcu` / `applyMultigridStrictMatPCQcu` | Full-coarse `X/Yhat` operator and compact-parity MATPC | strict MG |
| `applyMultigridStrictPrepareQcu` / `applyMultigridStrictReconstructQcu` | Full↔odd-Schur preparation/reconstruction | strict MG |
| `applyMultigridStrictRestrictQcu` / `applyMultigridStrictProLongQcu` | Compact fine-parity R/P with full coarse output/input | strict MG |
| `applyMultigridStrictInitQcu` / `applyMultigridStrictVCycleQcu` / `applyMultigridStrictEndQcu` | Allocate, reuse, and release the persistent strict hierarchy/arena | strict MG |
| `applyMultigridStrictFgmresQcu` | Fused single-rank restarted right-FGMRES, including warm x0 and exact workspace accounting | strict MG |

All functions take raw pointers cast to `long long` from `tensor.contiguous().data_ptr()`.

## Parameter Protocol

Three flat tensors bridge Python ↔ C++:

- **`params`** (CPU int32[58]) — lattice dims (`_LAT_X_`…`_LAT_XYZT_`), grid sizes (`_GRID_X_`…), data types (`_DATA_TYPE_`), iteration counts (`_MAX_ITER_`), plan selection (`_SET_PLAN_`), verbosity (`_VERBOSE_`), parity (`_PARITY_`), multigrid level configs and controls. The tail is `_MG_USE_GCR_` (54), `_MG_USE_DEFLATE_` (55), `_MG_MU_PRE_` (56), and `_MG_USE_INIT_GUESS_` (57); slot 57 is a 0/1 warm-start flag and does not extend the ABI.
- **`argv`** (real dtype, size 7) — physical parameters: `_MASS_` (idx 0), `_ATOL_` (1), `_SIGMA_` (2), per-level MG tolerances (3–6); c64 使用 float32，c128 使用 float64
- **`set_ptrs`** (CPU int64[100]) — LatticeSet/scratch pointers managed by the C++ runtime. Strict transition `t` uses slots `60+4*t+0..3` for blocked `V`, optional raw `Y`, `Yhat`, and onsite pair `(X,X^-1)`; slot `80` stores the persistent hierarchy handle.

Index constants in `define.py` MUST stay in sync with `cpp/cuda/qcu/include/define.h`.

`define.py` also provides pre-built tensors `params`, `argv`, and `set_ptrs` for convenience. They are modified in-place by the solver code.

## Strict MultiGrid Boundary and Lifetime

- `hierarchy_mode="strict"`（或 `QudaStrictMultigrid`）才选择 Strict full-coarse 语义；`setup_operator="schur"` 是 legacy compact odd-Schur/setup-vector 入口，不能当作 Strict 的别名。Strict 下即使传入该值，也不得因此把 coarse 几何或资产改成 compact。
- Strict 当前只接受每一层均为 `coarse_grid_solution_type="matpc"` 与 `smoother_solve_type="direct_pc"`；不得静默回退到 legacy hopping-only coarse dslash。
- Coarse operators and vectors retain the full lattice shape; Wilson/Clover coarse levels use `coarse_spin=2` and `E=2*nvec`. `P` 把 full coarse field 映射到选定 fine parity，`R=P†` 把该 compact fine field 限制回 full coarse field。Checkerboarding is confined to fine-side R/P, MATPC, prepare and reconstruct, never to coarse geometry.
- 当前 CUDA fused 入口固定 `target_parity=1`、`start_level=1`：fine full field 为 `[2,4,3,X,Y,Z,T/2]`，MATPC/Schur 工作向量为 `[12,X,Y,Z,T/2]`，每个 coarse 输入/输出仍为 `[E,Xc,Yc,Zc,Tc]` full field；粗层不得再次 checkerboard。用户侧 canonical null vector `[nvec,4,3,X,Y,Z,T]` 必须先转换为 C++ ABI 的 C-order blocked `[E,12,Xc,bx,Yc,by,Zc,bz,Tc,bt]`，不能把 7-D canonical 张量直接传给 fused 入口。
- 生命周期必须为 `hierarchy.setup()` → 创建 `CudaSchurOp`（内部执行 `applyInitQcu`）→ 绑定 fine/inter-level runtime assets → `applyMultigridStrictInitQcu` → 重复 `applyMultigridStrictVCycleQcu`/`applyMultigridStrictFgmresQcu` → `applyMultigridStrictEndQcu` → `CudaSchurOp.release()`（内部执行 `applyEndQcu`）。所有路径均应以 `try/finally` 保证逆序清理。
- Only after packed runtime assets are bound may setup tensors be detached with `hierarchy.seal_cuda_runtime(runtime_assets_bound=True)`. `CudaStrictMultigridSolver` does this by default when `release_setup_assets=True`; preserve `strict_setup_stats` before sealing if later diagnostics need them, because sealing deliberately disables Python setup/apply/export reuse.
- Runtime binding keeps `Yhat`, onsite data and inter-level blocked null vectors. Packed raw `Y` is diagnostic/setup data and is omitted by default; request it only for raw-operator checks.
- Strict fine FGMRES consumes Gauge `[2,3,3,4,X,Y,Z,T/2]` and Clover even/odd plus inverses `[4,3,4,3,X,Y,Z,T/2]`; coarse levels do not receive a second physical Gauge/Clover pair, but use Galerkin `X/Y/Yhat` assets. `Yhat` is the left-preconditioned link (`X^-1Y`) and raw backward links retain the QUDA storage-site/adjoint convention.
- Runtime caches use schema v2 and bind a streaming SHA256 to every logical tensor. Loading first verifies the complete tensor set and only then creates/transfers device tensors; host I/O is chunked at about 8 MiB. A hit intentionally reads tensor bytes twice—once for verification and once for device transfer—without materializing a full host hierarchy. A same-identity target won by a competing publisher may be reused only after complete verification of the manifest, all dataset attrs, and all tensor SHA256 values; never trust identity or path existence alone.
- Galerkin setup memory planning must count four simultaneous full-field arenas. The library default `strict_galerkin_max_workspace_bytes=512 MiB` is an API default, not the formal benchmark contract. The formal `16×32×32×48` profile uses colored `C=12` with a `4 GiB` c64 setup cap, and `C=1` with a `1 GiB` c128 setup cap. The outer fused `max_krylov_bytes` is separate: the solver API default is `512 MiB`, while the formal profile uses `512 MiB` for c64 and `1 GiB` for c128. Never infer either budget from the other.
- With `strict_galerkin_mode="auto"`, compare site-batch and colored operator-call/memory models under the requested cap, choose site-batch only when it fits and is no slower by call count, and record requested/effective mode, column batch `C`, projection-site batch `K`, cap and observed stats. The generic builders may shrink `C/K` to fit; a formal benchmark sets an exact-batch contract and must fail closed on any shrink.
- `applyMultigridStrictFgmresQcu` lazily allocates the outer workspace in C++ on the first solve and reuses it for the same geometry/restart. Its exact device size is `(2*m+5)*B_f + 2*B_c`, where `B_f` is one compact fine-parity vector and `B_c` one full first-coarse vector; Python owns no duplicate Krylov arena or coarse-I/O tensors. `CudaStrictMultigridSolver` shortens `m` before the call when `max_krylov_bytes` requires it.
- MPI stage 1 implements c64/c128 global dot/norm reduction, so the capability is `global_reduction=True`. Rank-dependent metadata/control errors are rejected through rank-symmetric preflight before collectives. `setup_halo=False`, `full_halo=False`, `compact_halo=False`, and distributed `fused_fgmres=False` remain hard gates; production multi-rank strict solves must still fail closed and must not be reported as supported.

Treat `memory_report()` as an explicit-ownership ledger, not whole-process peak memory. Before the first solve, fused resident bytes are zero and only planned bytes are reported; after the first solve, resident bytes must equal the formula above. For a lifecycle audit, record baseline → hierarchy setup → packed export/bind → runtime seal → strict init → first/steady solve → `close()`, deduplicate tensor storages, and separately include Gauge/Clover inputs, `LatticeSet` scratch/halos, allocator reserved/high-water state, and caller RHS/output storage. In formal benchmarks, sampler start delegates its first `cudaMemGetInfo` call to the worker thread; stop takes no final sample and retains the thread handle on join timeout so validation can fail closed. Keep the independent device-wide probe untimed, require memory schema version 2, use `device_used_max_observed_bytes`, and query `nvidia-smi` by target GPU UUID with observational fields named only `max_observed`.

The C++ `allocated_bytes` values returned by Strict init/V-cycle/fused FGMRES
describe their owned arenas only; they are not a process-wide measurement and
must not replace the device-wide probe or the ownership ledger above.

## Critical: `_SET_INDEX_` Increment

Between successive C++ calls within the same `applyInitQcu`/`applyEndQcu` lifecycle, you MUST increment `params[define._SET_INDEX_]` by 1. Failing to do so causes scratch buffer reuse conflicts that produce wrong results.

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
qcu.applyInitQcu(set_ptrs, params, argv)          # allocate
# ... operations with _SET_INDEX_ += 1 between calls ...
qcu.applyEndQcu(set_ptrs, params)                  # free
```

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

The Cython extension has a separate `qcu/` scope; keep its declarations and
stubs synchronized with the current source rather than copying an old snapshot.

### `qcu/` — `pyqcu/cuda/qcu/qcu.pyx`, `qcu_api.pxd`, `qcu.pyi`

# pyqcu.cuda.qcu

Cython extension module — bridges Python to the C++ CUDA backend `libqcu.so`.

## Files

| File | Purpose |
|------|---------|
| `qcu.pyx` | Cython source: thin wrappers around C functions from `pyqcu.h` |
| `qcu_api.pxd` | Cython declarations: aliased `cdef extern` symbols with `nogil` (must match `pyqcu.h` exactly) |
| `qcu.pyi` | Python type stub for IDE autocomplete |

## C API Surface

The C API is exposed through thin wrappers. Each field argument is a raw tensor data pointer cast to `long long`; strict entry points return status and validate shapes/contiguity before entering `nogil` code.

| Function | Purpose |
|----------|---------|
| `applyInitQcu` / `applyEndQcu` | Allocate / free scratch buffers |
| `applyWilsonDslashQcu` | Wilson dslash |
| `applyCloverDslashQcu` | Clover dslash |
| `applyWilsonBistabCgQcu` / `applyWilsonBistabCgDslashQcu` | Wilson BiStabCG solver + dslash |
| `applyWilsonCgQcu` / `applyWilsonCgDslashQcu` | Wilson CG solver + dslash |
| `applyCloverBistabCgQcu` / `applyCloverBistabCgDslashQcu` | Clover BiStabCG (requires clover_ee/oo + inverses) |
| `applyCloverQcu` / `applyCloversQcu` | Build Clover term (and inverse) |
| `applyDslashQcu` | Combined Wilson+Clover dslash |
| `applyLaplacianQcu` | Laplacian operator |
| `applyGaussGaugeQcu` | Gaussian gauge field generation |
| `applyMultigridRestrictQcu` / `applyMultigridProLongQcu` | MG restrict/prolong with null vectors |
| `applyMultigridCoarseDslashQcu` | Coarse-grid dslash |
| `applyCloverMultigridQcu` | Full Clover multigrid V-cycle |
| `applyMultigridStrict*Qcu` | Strict coarse/MATPC, prepare/reconstruct, parity R/P, and persistent hierarchy lifecycle |
| `applyMultigridStrictFgmresQcu` | Fused single-rank strict right-FGMRES with persistent C++ workspace |

## Call Lifecycle

以下是 legacy 普通 C API 的生命周期；它要求在同一 `applyInitQcu`/`applyEndQcu` 区间的连续操作间递增 `_SET_INDEX_`：

```python
qcu.applyInitQcu(set_ptrs, params, argv)   # allocate buffers
# ... perform operations ...
params[define._SET_INDEX_] += 1              # MUST increment between calls
qcu.applyEndQcu(set_ptrs, params)            # free buffers
```

Strict `CudaSchurOp` 使用固定的 per-instance `_SET_INDEX_` 槽位，严格按顶层 Strict 生命周期调用；Strict 原语之间不得递增或重置该值。

## Synchronization

`qcu_api.pxd` must exactly match the C declarations in `cpp/cuda/qcu/python/pyqcu.h`. Any mismatch can cause silent memory corruption.
