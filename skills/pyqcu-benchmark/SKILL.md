---
name: pyqcu-benchmark
description: 当需要运行或解释 PyQCU/QCU/QUDA 性能基准、公平计时、显存或容量证据时使用。
---
# pyqcu/testing/benchmark

Performance benchmarks comparing PyTorch, TileLang, and C++ CUDA backend implementations.

GPU/DCU/GPGPU benchmarks keep one Python process = one MPI rank = one device.
`MultiGpuMultigrid` benchmarks are explicit test-only evidence; do not treat
their launch shape as the production default. See
[`pyqcu-cuda/SKILL.md`](../pyqcu-cuda/SKILL.md#cuda-production-topology).

Legacy-MultiGrid is the QCU default solver and the primary development/debug/
optimization benchmark target. Strict-MultiGrid rows are explicit comparison,
algorithm-reference, or special-fallback evidence unless a task says otherwise.

Legacy hierarchies deeper than two levels emit `MultigridDepthWarning`; Strict
hierarchies deeper than three do the same. The same warning also considers the
global and rank-local lattice after the process grid and parity compression,
the coarsening block, the coarsest volume and whether another stable transition
is possible. Do not treat a warning-only run as a valid multi-level result.
Record native and independently reconstructed true residuals together, and
mark distributed Legacy depth greater than two as unsupported unless a
dedicated regression proves both correctness and benefit.

MPI benchmarks accept only a positive power of two ranks (`1/2/4/8/...`).
Reject every other count before launching MPI or allocating the solver. The
complete 2026-10-23 rank, quick-matrix, true-residual and evidence guidance is in
[`references/qcu-multigrid-20261023.md`](references/qcu-multigrid-20261023.md);
read it before running the QCU MultiGrid matrix.

Do not treat MG cache files as interchangeable by filename or size. CUDA/Torch
Legacy consume the blocked odd-Schur layout with parity-compressed time
(`lonv=[E,e,X,x,Y,y,Z,z,T/2,t]`), while KUPL's native full-time builder emits a
different coarse shape. Validate `lonv/hnn/hdg/sit` shapes against the selected
backend before setup; fail closed on a mismatch. For a prebuilt odd-Schur cache,
the Torch fine operator may use `parity_only=True`, but never enable that mode
while the same operator is still needed to construct a Galerkin hierarchy.

The MPI cache entry is
`python -m pyqcu.testing.simple.mpi.build_full_mg_cache`, not the serial parent
script. It must run every rank through distributed null-vector construction,
assign each rank a disjoint coarse slab, stage only a periodic overlap halo for
local stencil evaluation, write all four tensors through HDF5 MPI-IO owned slabs,
validate every rank slab against the published HDF5, and publish the manifest
atomically. Never root-gather multi-GiB `lonv`; L48 experience also rules out
root assembly of `hnn/hdg/sit`. The
correct `lonv` patch axes are `(2,4,6,8)`, not the last four tensor axes. Record
MPI ranks/grid, per-rank patch extent, slab reconstruction status, and all
global tensor SHA256 values. A 2/4-rank cache is functionally equivalent to the
1-rank reference only when tensor-level relative differences are reported; MPI
reduction order means bitwise SHA equality must not be required across rank
counts.

Identity-safe reuse, QUDA fallback, formal matrix scope, figure/speedup rules,
required recipe fields, and the real-time document ledger are defined in
[`references/qcu-benchmark-contract.md`](references/qcu-benchmark-contract.md).
Read it before launching a formal or multi-backend benchmark. Every speedup must
show absolute numerator and denominator times with units and sample statistics.

## Reuse the existing drivers first (mandatory)

`pyqcu/testing/simple/` is the normative implementation for every measurement
case, so a new ad-hoc driver that re-assembles the operator calls by hand is a
defect, not a shortcut:

| Case | Existing entry point |
|---|---|
| Production single solve with independent residual and atomic publish | `simple/solve_case.py --backend cuda\|quda\|kupl` |
| Three-backend comparison | `simple/compare_three_way.py` |
| Single-backend BiCGStab | `simple/{kupl,qcu,quda}_bistabcg.py` |
| Single-backend MultiGrid | `simple/{kupl,qcu,quda}_multigrid.py` |
| Input preparation and layout conversion | `simple/prepare_case.py`, `simple/_common.py` |
| MPI orchestration | `simple/mpi/compare_*.py` |

Those drivers already handle layout conversion, the parity Clover assets
(`ce/co/cei/coi`), the `_SET_PLAN_` lifecycle, the kappa/`u_0` device
consistency of `give_wilson`/`give_clover`, the independent full-lattice true
residual and the report fields (`iterations`, `wall_samples`,
`final_true_residual`).

Write a new script only when one of these holds, and state the specific gap and
its reproduction evidence in the conclusion:

1. the existing entry point genuinely does not support the required
   configuration -- read its argparse definition, do not assume;
2. a new measurement dimension is needed and the existing entry point has been
   extended backwards-compatibly instead of forked into a parallel script.

Bypassing the drivers has already produced three wrong conclusions in this
repository: an operator that disagreed with the gauge because Clover assets were
never passed, a single-parity Schur solve that degenerated to a zero RHS for an
even-site point source, and a native `MPI_ABORT` because `_SET_PLAN_` was left at
2 after `make_clover`.

## Source and timing contract (mandatory)

- **Performance runs use a point source only**: `pyqcu.lattice.point_source(lat, [0,0,0,0])`
  (single spacetime point, all spin/color components filled), converted with
  `pyqcu.tools.oooxyzt2poooxyzt` into the parity-split layout the native operators consume.
  A point source excites the whole spectrum, needs more iterations, is closer to production
  and is fully reproducible.
- **Correctness runs use point + random sources** on the small gate lattice
  (`8x8x8x16`) so a source-independent defect cannot hide behind a single RHS.
- Every record must carry source type, source coordinate/seed, gauge identity
  (file or seed+sigma), `mass` and `csw`. Iteration counts from a point source and
  a random source are never comparable and must not be mixed in one comparison.
- Wall time **may** be compared across machines, but only after the hardware and
  environment identity is written down in the same report: GPU model and compute
  capability, CPU model and core count, NUMA topology, `OMP_NUM_THREADS`, compiled
  architectures (`sm_XX` with cubin/PTX), library SHA256 values, driver and CUDA
  version, plus warmup/repeat counts and median/min/max. A wall time without that
  identity is "not comparable" and cannot support a speedup or regression claim.
  CPU and GPU results are never folded into a single speedup.
- **Never compare timings across allocations.** The same unchanged BiCGStab
  script, same shape and flags, measured 1.8185 / 3.8661 / 2.7584 s (medians) in
  three allocations, with bit-identical residuals (6.554010772522588e-07) and the
  same 18 iterations; the cross-allocation swing of one solver is 2.1x, larger
  than the effect under test (`logs/v20261123.txt` §114, L48 c64). Quote the job
  or allocation id with every number, and attribute a solver difference only to
  alternating points measured inside ONE allocation (same job, same nodes).

正式三方基准的输入、同窗比较、历史证据和不可外推条件见
[`pyqcu-docs/references/extracted-guidance.md`](../pyqcu-docs/references/extracted-guidance.md#5-性能与证据解释)。

Legacy/Strict CUDA MG 的运行时峰值必须用同时覆盖 C++ 原生分配的独立设备级采样器，
不能只报 PyTorch allocator。现有同输入 c64 两级、`nvec=12` 记录中 Legacy solve
峰值为 0.596/1.533/4.113 GiB，Strict 为 0.789/2.326/6.715 GiB
（`8x8x8x16`、`12x12x12x24`、`16x16x16x32`）；`24x24x24x64` Strict 缺失。Legacy 同时是
`n_vcycles=0`，所以只能作为当前实际入口的容量证据。详见
[`docs/legacy_vs_strict_mg_memory_20261023.md`](../../docs/legacy_vs_strict_mg_memory_20261023.md)。
这些 JSON 没有 `device_name`，只能确认总显存 `34,359,345,152` bytes；不能用相同
总量代替物理设备身份。Legacy asset-build 另有 `432,013,312`、`1,591,738,368`、
`4,722,786,304` bytes 的三点峰值，它们是独立 setup 阶段，不能与 Strict 全程峰值
拼成同一阶段比较。

## 带宽与启动四项口径（2026-11-15）

同一次运行里会同时出现四种"带宽"，它们的分母与字节模型都不同，**任意两者相除都
得不到加速比或效率**：

| 口径 | 定义 | 典型值（L24 c64，32r×18t） |
|---|---|---|
| profiler 字节 | `KUPL_PROFILE_OPT=1` 自报 `dslash_bytes`、`dslash_seconds` | `1210.3 MB / 0.1171 s ≈ 10.3 GB/s·rank`（aggregate `330.9`） |
| 解析流量 | 每 site 固定 `768 B`（c64：gauge `576` + src `96` + dst `96`）× 本地 site × 调用数 | `1.10 GB / 0.0675 s ≈ 16.3 GB/s·rank`（`522 GB/s` 聚合） |
| STREAM triad | 同节点内存探针（8M double × 3 数组、5 次取最优） | `18.3–20.1 GB/s·rank`，32 rank `585–643 GB/s` |
| solve 级有效带宽 | 模型总流量 ÷ 整个 solve 墙钟 | 旧口径 `149.5 GB/s`（y76，模型 `64.97 GB`） |

- profiler 模型会**低估**字节（`828 MB` vs 解析 `1.10 GB`），也会随 dtype 非线性
  变化（c128 报 `4013 MB` 而 c64 报 `1210 MB`，不是 2×）；不要按 dtype 倍率推算。
- 只有 dslash 级才能与 STREAM 分母比（`16.3/18.3–20.1 ≈ 87%` 视为带宽受限）；
  clover 的 `40 GB/s·rank` **高于**探针峰值，是 cache 复用命中，不是超过硬件。
- 判断"是否带宽受限"必须先配对时间：同一 76 次 dslash，balanced 4D 的字节更少
  （`857.3 MB` 对 `1210.3 MB`）但每 rank 只有 `4.06 GB/s`、t-only 有 `8.32 GB/s`——
  **字节下降、时间上升**，4D 邻居打散局部性。字节数不能单独用来排序路径。

### 冷读与源混杂（必须先排除）

- 读模式：`*_read_mode` = `cold`（本次生成并首次读）/`warm`（本次显式 `prewarm`）/
  `pre-generated`（复用旧文件、页缓存状态未知）。判读顺序是
  `status/returncode` → `*_read_mode` 与 `*_read_seconds` → `atol` → `node/grid/shm`
  → 才看 `wall_seconds`。
- 实测：warm 行 `gauge_read_seconds = 2.4–5.2 s`；cold 行 `29.9–35.6 s`，对应
  `wall_seconds` 被抬高 `33–60×`（`11.44/10.93/20.86 s` 对 `0.24–0.35 s`），
  `dslash_gbps_rank` 掉到 `0.19–0.44`。**这些行不是性能数据**，必须剔除或复测。
- 源语义不可混：点源 `seed 42`（12 it，真残差 `5.385e-05`）、全 1 源（11–12 it，
  `6.756e-05`/`6.832e-05`）、`real1`（全 1 的别名）、随机源（18–19 it）迭代数与
  残差都不同；同一表里出现两种源就不许互算倍率。
- 组态语义也不可混：`gauge=random`（NumPy QR 生成）与真实组态不是同一物理问题，
  只能作正确性/鲁棒性证据。

### c128/c64 倍率

同节点、同 rank×thread、同源、同 grid（L24、32r×18t、ones、cn19699）：
c64 18 it `0.247571 s`（shm0）/`0.240473 s`（shm1），c128 31 it
`0.567444 s`/`0.475555 s` → 墙钟 `2.29×`/`1.98×`，其中**迭代数占 `1.72×`**、
单次迭代 `1.15–1.33×`。报倍率时必须同时给迭代数与 `atol`（c64 `1e-6`、
c128 `1e-9`），否则读者会把迭代差当成算子代价。

### 启动摊销

计算节点上每个 `mpirun` 的固定成本约 `120 s`（import `80.8 s`：numpy `7.8` +
mpi4py `34.9` + 驱动 `38.1`；MPI 起跑约 `39 s`）。所以一行 CSV 的
`elapsed_seconds` 是 `95–211 s`（16 秩≈`100 s`、32 秩 `183–211 s`），而
`wall_seconds` 只有 `0.21–0.57 s`（`builder_seconds` 另计 `4.4–6.1 s`）。
- 作业成本按 `elapsed_seconds` 估，不按 `wall_seconds`；矩阵吞吐由启动主导。
- 进程内 loop 原型（同一 `mpirun` 内重复调用驱动）3 例只需 `107.4 s`，但它会保留
  内存（RSS 589 MB→925 MB）并放大 wall 抖动，且不允许共享 `_SET_PLAN_` 残留；
  正式基准仍一例一进程，loop 只用于快速筛配置。

### 矩阵范围裁定与"不重测"判据（2026-11-15）

完整裁定见 [`docs/kupl_matrix_scope_20261115.md`](../../docs/kupl_matrix_scope_20261115.md)。

- **放弃轴**：thread 9（16×9、32×9、32×9 shm1 复核）、seed24、random gauge、random 源、
  point 源、mg-restart 50。标注固定为"耗时代价过大，已有部分数据保留，不重测/不新增"；
  这些轴的现有单样本（如 `861ac912ac3c7c42` 的 `11.437897 s`）不得进入任何性能表。
- **保留轴**：16×36（已有 3 行）、MG1、r10、L32/c128、MG+shm。
- **不重测判据**（全部同口径才成立）：source 语义与 hash、gauge dtype 与
  `gauge_canonical_sha256`、cache identity、`atol`、`grid`/`ranks`/`threads`/`shm`/
  `solver`/`mg_restart`。任一不同必须重跑，并以新 `run_id` 落账。
- **31 条 PASS 不重跑**（L24 c64 gauge 的 c128 副本 cast 后等价；L32 c128 cache 不重建；
  MG cache 不含 source 哈希）。**L32 的 6 条 FAIL 是"首次有效运行"，不是"重测"**。
- **MPI h5py 只做接口检查**：不做重新编译/安装，ranked cache 不列入后续计划；不要写
  "建议编译/安装 MPI h5py"这类下一步。全批 `hdf5_mpio_read=false`（0 行 `true`），
  读是每 rank 全量 open + 本地 slab，所以 I/O 预算按"文件字节 × rank 数"估
  （32 rank 读 1.2 GB gauge ≈ 38.6 GB 聚合流量），`prewarm` 是唯一低成本重复读。
- **超时只留 partial 输出**：600 s 硬超时的 prepare/cache 行可能已落盘并被判 `PASS`，
  solve 行没有 `reports/<run_id>.json` 一律 `FAIL`。这些行只证明"部分产物可用"，
  不得进入性能表，也不得写成"该格点可跑"；`--resume --skip-failed` 还会让它们永久
  跳过。

## Files

| File | Purpose |
|------|---------|
| `conftest.py` | Benchmark entry point |
| `env.py` | Benchmark environment configuration |

## 2026-10-23 六版本真实输入基准

正式编排入口为 `pyqcu/testing/benchmark/qcu_six_way.py`，隔离 worker 为
`qcu_six_way_worker.py`。默认输入是真实 LIME 的 `8×8×8×16` 前缀，运行
Torch/CUDA/KUPL × Legacy/Strict、QUDA aligned MG，并额外执行 CUDA Legacy
fine-only 消融。结果与解释见
[`docs/qcu_six_way_benchmark_20261023.md`](../../docs/qcu_six_way_benchmark_20261023.md)。

所有后端的正确性门固定为 `8×8×8×16`，包括 Torch/CUDA/KUPL × Legacy/Strict
六项 QCU-MG；`16×16×16×16` 及更大格点只能用于性能、容量或历史数据验证。
合格批次还必须包含
同输入的 QUDA aligned MG 以及 Torch/CUDA/KUPL/QUDA BiCGStab 边界对照。正式矩阵只用
trace-off，并在进程启动前设置 `QUDA_ENABLE_TUNING=0`；trace-on 只能单独用于诊断。
详细执行与证据口径见
[`references/qcu-multigrid-20261023.md`](references/qcu-multigrid-20261023.md)。

不得由 `levels`、`restart` 或 cycle 配置推断 coarse 已执行。每个 Legacy side 必须读取
`n_vcycles`、`coarse_iters`/`coarse_iterations` 和实际 cycle/correction 计数；多 rank
CUDA 未显式设置 `params[_MG_USE_DEFLATE_]` 时结果为 fine-only，不能标成 coarse MG。

QUDA GCR-MG on the A100 native-LAPACK build was made usable with `nvec=12`,
2 levels, `mg_block=2^4`, and `--no-mma`. MMA failed differently by path: `nvec=24`
exhausted A100 memory, while `nvec=12` requested unsupported coarse color `N=24`.
The working no-MMA L32 comparison uses internal null-vector setup and should be reported
as a GCR-MG entry-point comparison, not as an identical Legacy cycle. L32 medians are
QUDA `0.482426/0.314393/0.287465 s` versus CUDA Legacy
`0.214456/0.219506/0.218343 s` for 1/2/4 ranks; all three pass independent full residuals.

本轮确认的三个硬边界：

- CUDA Legacy 在同一 `LatticeSet` 上连续 cold solve 会污染后续 RHS/状态；正式样本必须
  在计时前重建运行时，重建时间单独记录，不能把后端自报残差当作末次解的正确性证据。
- KUPL Strict 的 public 与 raw ABI 已完成对齐；可用的正式入口应使用
  `--kupl-strict-abi both` 交叉校验两条路径。仍不得只验证 raw ABI 就宣称其他公共
  wrapper 可用，ABI 差异必须 fail closed。
- sm70 QUDA 构建为 `QUDA_QIO=OFF`。缺少共享 QIO null-vector 时，QUDA 只能在同 gauge 上
  自生成 null space；该结果可作同配置算法对照，不能写成 byte-identical null-basis replay。

Capacity-bound residual validation must run after native teardown. End the
`LatticeSet`, close the MG hierarchy, delete cache/asset owners and any bound
method closure that references the hierarchy, run GC/allocator cleanup, then
construct a fresh Clover/operator for the independent full residual. Computing
the residual while native scratch and coarse assets are still resident creates
false OOMs and obscures the real solve capacity.

## Usage

```bash
python pyqcu/testing/benchmark/conftest.py
```

2026-08-24：conftest.py 收集期笔误已修（bug37，pytest 正常收集 exit=0，见
`logs/fix-report-2026-08-24.md`）。基准锚点：4096³ fp16 matmul TileLang 热修后
38.7 TFLOPS ≈ cuBLAS 94%（V100，见 tilelang 技能）。

Strict benchmark records use the current bridge ABI: `params` is CPU
`int32[58]`, `argv` is an eight-element real tensor (`float32` for c64 and
`float64` for c128), and `set_ptrs` is CPU `int64[100]`. `params[57]` is
`_MG_USE_INIT_GUESS_`: formal cold solves set it to `0`; only an explicitly
prefilled `fermion_out` warm start may set it to `1`. Do not report a speedup
from a run whose side records use different ABI, initial-guess, or asset
contracts.

For fast iteration, use `run_strict_fast.py --list` to inspect the command
plan, then `--only <gate>` for one named gate or the default tier 0 for data-free CPU algebra. Tier 1 is the small
CUDA handoff gate. Tier 2 is explicit and intentionally expensive: it invokes
the formal real-gauge/QUDA collector and may emit a fair speedup, so it is not
part of the default edit loop. Keep the protocol/cache/QIO tests separate from
tier 0/1 so a documentation or collector edit does not import external QUDA or
allocate a large gauge before the cheap gate passes.

## Strict MultiGrid vs QUDA

Use `pyqcu/testing/qcu/strict/quda_comparison/bench_strict_vs_quda.py` directly, or through
`run_strict_fast.py --tier 2`, for formal timing. The direct collector is the
authoritative protocol implementation; tier 2 is the same formal gate with
runner-level timeout/JSON handling, not a cheap substitute. The PyQCU side must
use `hierarchy_mode="strict"`/`StrictMultigrid`, fixed fine odd
`target_parity=1` and coarse `start_level=1`; `setup_operator="schur"` is a
legacy compact path, not a substitute. The collector fixes `16×32×32×48`, the
Gauge/RHS/canonical-full-null-vector bundle and SHA256 identities, zero initial
guesses, precision, two warmups, five measured solves, and median/MAD. The
formal input null vectors are canonical complex64 `[12,4,3,X,Y,Z,T]`; PyQCU
converts them to the C++ blocked 10-D ABI
`[E,12,Xc,bx,Yc,by,Zc,bz,Tc,bt]`, while QUDA consumes the separately converted
QIO artifact. Do not confuse canonical input layout with the legacy blocked
runtime layout. Each side must converge and pass the independently recomputed
full Wilson/Clover relative true-residual gate before a speedup is emitted.

The QUDA side requires QIO null vectors converted from that same canonical full dataset plus a manifest whose QIO artifact hashes match. The formal gamma basis is `QUDA_DEGRAND_ROSSI_GAMMA_BASIS`. The fairness gate hashes the canonical full dataset and compares it with `canonical_dataset_sha256`; `source_sha256` records only the original E12 odd-Schur provenance and must never be compared with the full dataset digest. QIO round-trip verification compares the staging and read-back files with a bounded two-file 8 MiB streaming scan, rather than mapping both complete assets. Missing QIO evidence must produce an explicit skip; native random QUDA null vectors are not a fair substitute. Keep Gauge/Clover normalization, mass/κ, RHS, precision, topology, level/block geometry, `coarse_spin=2`, smoother/coarse budgets and stopping criterion aligned, and retain config/input hashes in every side record.

Before formal timing, persist two cheap single-rank gates: the `4^4` reduction smoke (`pyqcu/testing/qcu/strict/quda_comparison/smoke_quda_reduction.py`) and an `8^4` Nc24 setup-only probe with `n_vec=12` and `coarse_spin=2`. Formal collection is blocked until both pass. A smoke is `PASS` only when post-setup resolved/read-back parameters and observed capabilities satisfy the gate; requested CLI values are metadata, not evidence. Record each gate's status and requested-versus-resolved values, without embedding transient build digests in this skill.

`QUDA_MULTIGRID_NVEC_LIST` is a comma-separated compile-time instantiation set. For the formal `n_vec=12`, `coarse_spin=2` path, configure `12,24`: `12` is needed by `BlockOrthogonalize` (`B.size`), while `24 = n_vec × coarse_spin` is the coarse color/coarse-operator instance. A list containing only `12` or only `24` ends in `MPI_ABORT`. After setup, read back and record the effective `n_vec`, `coarse_spin`, coarse color, and build-instance configuration; fail closed on missing or mismatched values.

PyQUDA `QudaMultigridParam` array getters return copies. For every array-valued field (`n_vec`, `nu_pre`, `nu_post`, solver arrays, MMA flags, `vec_load`, `vec_infile`, and similar), copy the whole column, modify the copy, assign the whole column with `setattr`, then immediately read it back and compare. `param.n_vec[0] = 12` (and the analogous `vec_load[0] = ...`) is a silent no-op anti-pattern. For c64 device precision, the QDP host gauge remains contiguous `complex128`; `setPrecision(single)` selects device precision and does not justify downcasting the host QDP gauge.

On the PyQCU side, bind packed runtime assets, copy `strict_setup_stats`, then call `seal_cuda_runtime(runtime_assets_bound=True)` before steady solves; report the detached-storage estimate and observed allocator delta. The strict worker lifecycle is setup → `CudaSchurOp`/`applyInitQcu` → asset bind → Strict init → warmup/steady solves → Strict end → `applyEndQcu`; retain the same per-instance `_SET_INDEX_` throughout this sequence. The outer workspace is one C++ fused allocation, lazily created on the first solve and reused, with exact size `(2*m+5)*B_f+2*B_c`; Python outer-Krylov and coarse-I/O resident bytes are zero. Report this separately from the persistent recursive hierarchy/V-cycle arena, packed assets, Gauge/Clover, backend scratch/halos, caller fields, and CUDA allocated/reserved. Omitting raw `Y` is a real steady-state saving only after native setup duplicates have been sealed away.

Formal memory evidence uses schema version 2 as a hard gate for every successful side record. The `first_solve` object is the first zero-initial-guess warmup, not a measured repeat: it enables the device-wide sampler so lazy native/CUDA solver workspace creation (including PyQCU's lazy fused workspace) is observable. Its `excluded_from_formal_timing` field must be `true`; its duration and memory are diagnostic first-allocation evidence and do not enter formal steady timing or speedup statistics. The separate `steady.untimed_device_memory_probe` is also marked `excluded_from_formal_timing=true` and exists only to sample native device-wide allocations after the measured repeats.

Interpret the schema-v2 memory scopes separately. `steady.baseline.allocated_bytes` and `steady.baseline.reserved_bytes` are PyTorch allocator live and reserved bytes immediately after all warmups and before measured repeats. Each `steady.samples[*].cuda_peak_allocated_bytes` and `cuda_peak_reserved_bytes` is collected after resetting PyTorch peak statistics immediately before that measured solve; the aggregate `steady.cuda_peak_*` fields are the maxima across measured repeats. `allocated` is the allocator live-allocation high-water value, while `reserved` is the caching-allocator reservation high-water value; neither is a complete device-wide/native-allocation total. The exact device-wide sampler field is `device_used_max_observed_bytes`, the maximum sampled `cudaMemGetInfo` used-bytes value, and may include other processes. Sampler `start()` only launches the worker—the first `cudaMemGetInfo` call is not executed on the main thread. `stop()` signals and joins without taking a final sample; if join times out, it retains the thread handle and the record fails closed. Formal solve timing uses the separate untimed probe, and `setup_seconds` is captured before sampler stop. Filter `nvidia-smi` observations by the target GPU UUID; `nvidia-smi_*_max_observed_bytes` are post-solve snapshots, not peaks or high-water marks. On QUDA setup or warmup failure, stop any active sampler and best-effort destroy multigrid plus free Gauge without masking the primary exception.

Runtime-cache records use schema v2 with a streaming SHA256 for every logical tensor. A cache hit verifies every tensor before any device transfer, uses host chunks of at most about 8 MiB, and performs two logical reads (digest verification, then device transfer); report cache verification/load time rather than treating a hit as zero-cost. If no-clobber publication encounters a competing target with the same identity, reuse is allowed only after fully validating its manifest, every dataset attribute, and every tensor SHA256; identity equality or file existence alone is insufficient. Formal Galerkin setup uses a four-full-field-arena budget independent of outer Krylov: c64 defaults to colored `C=12` with a `4 GiB` setup cap, while c128 defaults to `C=1` with `1 GiB`. The `512 MiB` c64 value is the outer-Krylov default (scaled to `1 GiB` for c128), not a Galerkin cap. Record requested/effective column batch, projection batch, setup cap and observed setup statistics separately; a formal run must fail rather than silently shrink the batch.

PyQCU cache experiments must set both `--strict-cache-dir` and `--cache-expect {miss,hit}` explicitly, with the cache directory kept inside the repository. Use the smoke profile plus `miss` for cold-cache generation and the formal profile plus `hit` for a measured cache-hit run; a mismatch must fail before heavy imports or device allocation. Reserve `any`, reduced repeats and relaxed tolerances for smoke runs. A smoke document must never emit a formal speedup.

On WSL2, patch only an independent QUDA source shadow with `pyqcu/testing/qcu/strict/quda_comparison/quda_wsl2_reduce_sync.patch`; do not modify `refer/git-rep/quda`. Set `DEV87_REDUCE_SYNC=1` and put the patched install's `lib` first in `LD_LIBRARY_PATH`. Qualify the selected install dynamically on every production smoke: record the actual `libquda.so` and `libqmp.so` paths and SHA256 values, verify the WSL2 marker/path precedence, and require `BUILD_QDP_INTERFACE`, `HAVE_QIO`, `QMP_COMMS`, `QUDA_RECONSTRUCT=7`, plus a `QUDA_PRECISION` bitmask containing the requested precision. For 12 fine near-null vectors with `coarse_spin=2`, `QUDA_MULTIGRID_NVEC_LIST` must contain both `12` and `24`; compiling only either value aborts in the multigrid setup path. Do not hard-code a production digest in this skill or a synthetic fixture; every changed build needs a fresh reduction smoke. Patched WSL2 timing is environment-scoped evidence, not a portable upstream-QUDA claim.

Run and merge sides without rerunning a successful compatible record:

```bash
python pyqcu/testing/qcu/strict/quda_comparison/bench_strict_vs_quda.py --profile formal --side pyqcu --strict-cache-dir data/strict_runtime_cache --cache-expect hit --output pyqcu.json
python pyqcu/testing/qcu/strict/quda_comparison/bench_strict_vs_quda.py --profile formal --side quda --quda-nullvec-prefix PREFIX --quda-nullvec-manifest MANIFEST --output quda.json
python pyqcu/testing/qcu/strict/quda_comparison/bench_strict_vs_quda.py --merge pyqcu.json quda.json --output combined.json
```

Only a merged document with matching config/input hashes, both side statuses `ok`, passing true residuals, and `comparison.fair=true` may report `speedup_pyqcu_over_quda = median(QUDA)/median(PyQCU)`. Do not claim PyQCU is faster until repeated fair runs show a stable value above one; distinguish patched or unhealthy QUDA environments from portable results.

## 历史性能证据

2026-09 至 2026-11 的详细矩阵、trace、Galerkin、Hopper overlap、资源守卫和鲲鹏真-MG 记录见
[`references/qcu-benchmark-history-2026.md`](references/qcu-benchmark-history-2026.md)。
