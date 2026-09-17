---
name: benchmark
description: PyQCU 性能基准 skill：覆盖 examples/benchmark 的通用基准，以及 dev87 strict MultiGrid 对 QUDA 的可复现公平计时与显存口径。
---
# examples/benchmark

Performance benchmarks comparing PyTorch, TileLang, and C++ CUDA backend implementations.

## Files

| File | Purpose |
|------|---------|
| `conftest.py` | Benchmark entry point |
| `env.py` | Benchmark environment configuration |

## Usage

```bash
python examples/benchmark/conftest.py
```

2026-08-24：conftest.py 收集期笔误已修（bug37，pytest 正常收集 exit=0，见
`logs/fix-report-2026-08-24.md`）。基准锚点：4096³ fp16 matmul TileLang 热修后
38.7 TFLOPS ≈ cuBLAS 94%（V100，见 tilelang 技能）。

Strict benchmark records use the current bridge ABI: `params` is CPU
`int32[58]`, `argv` is a seven-element real tensor (`float32` for c64 and
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

Use `examples/qcu/dev87/bench_strict_vs_quda.py` directly, or through
`run_strict_fast.py --tier 2`, for formal timing. The direct collector is the
authoritative protocol implementation; tier 2 is the same formal gate with
runner-level timeout/JSON handling, not a cheap substitute. The PyQCU side must
use `hierarchy_mode="strict"`/`QudaStrictMultigrid`, fixed fine odd
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

Before formal timing, persist two cheap single-rank gates: the `4^4` reduction smoke (`examples/qcu/dev87/smoke_quda_reduction.py`) and an `8^4` Nc24 setup-only probe with `n_vec=12` and `coarse_spin=2`. Formal collection is blocked until both pass. A smoke is `PASS` only when post-setup resolved/read-back parameters and observed capabilities satisfy the gate; requested CLI values are metadata, not evidence. Record each gate's status and requested-versus-resolved values, without embedding transient build digests in this skill.

`QUDA_MULTIGRID_NVEC_LIST` is a comma-separated compile-time instantiation set. For the formal `n_vec=12`, `coarse_spin=2` path, configure `12,24`: `12` is needed by `BlockOrthogonalize` (`B.size`), while `24 = n_vec × coarse_spin` is the coarse color/coarse-operator instance. A list containing only `12` or only `24` ends in `MPI_ABORT`. After setup, read back and record the effective `n_vec`, `coarse_spin`, coarse color, and build-instance configuration; fail closed on missing or mismatched values.

PyQUDA `QudaMultigridParam` array getters return copies. For every array-valued field (`n_vec`, `nu_pre`, `nu_post`, solver arrays, MMA flags, `vec_load`, `vec_infile`, and similar), copy the whole column, modify the copy, assign the whole column with `setattr`, then immediately read it back and compare. `param.n_vec[0] = 12` (and the analogous `vec_load[0] = ...`) is a silent no-op anti-pattern. For c64 device precision, the QDP host gauge remains contiguous `complex128`; `setPrecision(single)` selects device precision and does not justify downcasting the host QDP gauge.

On the PyQCU side, bind packed runtime assets, copy `strict_setup_stats`, then call `seal_cuda_runtime(runtime_assets_bound=True)` before steady solves; report the detached-storage estimate and observed allocator delta. The strict worker lifecycle is setup → `CudaSchurOp`/`applyInitQcu` → asset bind → Strict init → warmup/steady solves → Strict end → `applyEndQcu`; retain the same per-instance `_SET_INDEX_` throughout this sequence. The outer workspace is one C++ fused allocation, lazily created on the first solve and reused, with exact size `(2*m+5)*B_f+2*B_c`; Python outer-Krylov and coarse-I/O resident bytes are zero. Report this separately from the persistent recursive hierarchy/V-cycle arena, packed assets, Gauge/Clover, backend scratch/halos, caller fields, and CUDA allocated/reserved. Omitting raw `Y` is a real steady-state saving only after native setup duplicates have been sealed away.

Formal memory evidence uses schema version 2 as a hard gate for every successful side record. The `first_solve` object is the first zero-initial-guess warmup, not a measured repeat: it enables the device-wide sampler so lazy native/CUDA solver workspace creation (including PyQCU's lazy fused workspace) is observable. Its `excluded_from_formal_timing` field must be `true`; its duration and memory are diagnostic first-allocation evidence and do not enter formal steady timing or speedup statistics. The separate `steady.untimed_device_memory_probe` is also marked `excluded_from_formal_timing=true` and exists only to sample native device-wide allocations after the measured repeats.

Interpret the schema-v2 memory scopes separately. `steady.baseline.allocated_bytes` and `steady.baseline.reserved_bytes` are PyTorch allocator live and reserved bytes immediately after all warmups and before measured repeats. Each `steady.samples[*].cuda_peak_allocated_bytes` and `cuda_peak_reserved_bytes` is collected after resetting PyTorch peak statistics immediately before that measured solve; the aggregate `steady.cuda_peak_*` fields are the maxima across measured repeats. `allocated` is the allocator live-allocation high-water value, while `reserved` is the caching-allocator reservation high-water value; neither is a complete device-wide/native-allocation total. The exact device-wide sampler field is `device_used_max_observed_bytes`, the maximum sampled `cudaMemGetInfo` used-bytes value, and may include other processes. Sampler `start()` only launches the worker—the first `cudaMemGetInfo` call is not executed on the main thread. `stop()` signals and joins without taking a final sample; if join times out, it retains the thread handle and the record fails closed. Formal solve timing uses the separate untimed probe, and `setup_seconds` is captured before sampler stop. Filter `nvidia-smi` observations by the target GPU UUID; `nvidia-smi_*_max_observed_bytes` are post-solve snapshots, not peaks or high-water marks. On QUDA setup or warmup failure, stop any active sampler and best-effort destroy multigrid plus free Gauge without masking the primary exception.

Runtime-cache records use schema v2 with a streaming SHA256 for every logical tensor. A cache hit verifies every tensor before any device transfer, uses host chunks of at most about 8 MiB, and performs two logical reads (digest verification, then device transfer); report cache verification/load time rather than treating a hit as zero-cost. If no-clobber publication encounters a competing target with the same identity, reuse is allowed only after fully validating its manifest, every dataset attribute, and every tensor SHA256; identity equality or file existence alone is insufficient. Formal Galerkin setup uses a four-full-field-arena budget independent of outer Krylov: c64 defaults to colored `C=12` with a `4 GiB` setup cap, while c128 defaults to `C=1` with `1 GiB`. The `512 MiB` c64 value is the outer-Krylov default (scaled to `1 GiB` for c128), not a Galerkin cap. Record requested/effective column batch, projection batch, setup cap and observed setup statistics separately; a formal run must fail rather than silently shrink the batch.

PyQCU cache experiments must set both `--strict-cache-dir` and `--cache-expect {miss,hit}` explicitly, with the cache directory kept inside the repository. Use the smoke profile plus `miss` for cold-cache generation and the formal profile plus `hit` for a measured cache-hit run; a mismatch must fail before heavy imports or device allocation. Reserve `any`, reduced repeats and relaxed tolerances for smoke runs. A smoke document must never emit a formal speedup.

On WSL2, patch only an independent QUDA source shadow with `examples/qcu/dev87/quda_wsl2_reduce_sync.patch`; do not modify `refer/git-rep/quda`. Set `DEV87_REDUCE_SYNC=1` and put the patched install's `lib` first in `LD_LIBRARY_PATH`. Qualify the selected install dynamically on every production smoke: record the actual `libquda.so` and `libqmp.so` paths and SHA256 values, verify the WSL2 marker/path precedence, and require `BUILD_QDP_INTERFACE`, `HAVE_QIO`, `QMP_COMMS`, `QUDA_RECONSTRUCT=7`, plus a `QUDA_PRECISION` bitmask containing the requested precision. For 12 fine near-null vectors with `coarse_spin=2`, `QUDA_MULTIGRID_NVEC_LIST` must contain both `12` and `24`; compiling only either value aborts in the multigrid setup path. Do not hard-code a production digest in this skill or a synthetic fixture; every changed build needs a fresh reduction smoke. Patched WSL2 timing is environment-scoped evidence, not a portable upstream-QUDA claim.

Run and merge sides without rerunning a successful compatible record:

```bash
python examples/qcu/dev87/bench_strict_vs_quda.py --profile formal --side pyqcu --strict-cache-dir /root/PyQCU/data/strict_runtime_cache --cache-expect hit --output pyqcu.json
python examples/qcu/dev87/bench_strict_vs_quda.py --profile formal --side quda --quda-nullvec-prefix PREFIX --quda-nullvec-manifest MANIFEST --output quda.json
python examples/qcu/dev87/bench_strict_vs_quda.py --merge pyqcu.json quda.json --output combined.json
```

Only a merged document with matching config/input hashes, both side statuses `ok`, passing true residuals, and `comparison.fair=true` may report `speedup_pyqcu_over_quda = median(QUDA)/median(PyQCU)`. Do not claim PyQCU is faster until repeated fair runs show a stable value above one; distinguish patched or unhealthy QUDA environments from portable results.

## 2026-09-16 三层与默认关闭 trace

`bench_strict_vs_quda.py` now accepts `--lattice`, `--levels`, `--block`,
`--gauge-path`, and `--nullvec-path`.  Two-level cache identity remains
`asset_semantics_version=2`; recursive three-plus-level identity uses version
3.  For `levels>2`, PyQCU propagates each transition's coarse `V` with `R`,
while QUDA uses `generate_all_levels=false`, `vec_load[0]=true`, and
`num_setup_iter[>0]=0`.  Do not let QUDA generate a second private coarse
basis.

Diagnostic tracing is default-off:

```text
PYQCU_STRICT_TRACE_FILE=/path/pyqcu.tsv
QUDA_MG_TRACE_FILE=/path/quda.tsv
```

PyQCU trace version 2 adds residual events; QUDA emits cycle/stage/residual
events.  `examples/qcu/dev87/summarize_mg_trace.py` aggregates both.
Trace-enabled wall times are diagnostic only; the no-trace formal result is
the speedup authority.

## 2026-09-17 重新校验与 QUDA double MultiGrid

旧 c64 三层 `4.3817x` 结果撤回。它使用 QUDA 原生 CA-GCR/多内层预算，
而 PyQCU 使用一次递归 V-cycle；复测后同格点主口径为 `1.9898x`
(`0.662549` s vs. `1.318356` s)，见
`data/mg_matrix_20260916_round2/formal-c64-l3/benchmark.json`。当前主口径
`--quda-strategy aligned` 明确匹配一次递归预算：MR smoother、
`smoother_tol=0`、非最粗层 `coarse_solver_maxiter=1`。`--quda-strategy
library` 保留 PyQUDA/QUDA 原生 CA-GCR、`nu_post=8`、`coarse_solver_maxiter=16`
并作为敏感性对照，不得把两种策略混进同一个加速比。

QUDA upstream 1.1.0 的 double MG 需要以下源码级修复：

1. 用 CMake `QUDA_MULTIGRID_DOUBLE=ON` 打开 `GPU_MULTIGRID_DOUBLE`。
2. `matrix_tile.cuh` 在 fine/coarse 精度不同的 accessor 上显式构造目标
   `complex<T>`，不能依赖隐式精度转换。
3. double coarse-link 原子累加必须使用 double 存储；若访问器仍把 double
   字段按 `int` 固定点解释，level-1 setup 会出现 NaN。
   实现位于 `coarse_op.in.cu`、`coarsecoarse_op.hpp`、
   `include/kernels/coarse_op_kernel.cuh` 和 `staggered_coarse_op.in.cu`。
4. 建议同一构建使用 `QUDA_PRECISION=12`、`QUDA_RECONSTRUCT=7`、
   `QUDA_INTERFACE_QDP=ON`、`QUDA_QIO=ON`、`QUDA_QMP=ON`、
   `QUDA_MULTIGRID_NVEC_LIST=12,24`、`QUDA_ENABLE_MMA=OFF`。

正式 PyQCU/QUDA 对照现在记录同侧 plain BiCGStab 参考时间与真残差。
参考解不进入 MG-vs-MG 加速比，但必须用于判断某侧 MG 是否异常。QUDA
侧切到 BiCGStab 时还需把 `invert_param.preconditioner` 显式置为
`Pointer("void")`；只把 `inv_type_precondition` 设为 INVALID 不够，
QUDA 的 solver factory 仍会因残留 MG 指针拒绝 BiCGStab。

主口径 c64/c128 结果如下（五次 median；trace 环境变量全部未设置）：

| 格点 / 层数 / 精度 | PyQCU | QUDA aligned | 加速比 | 外迭代 PyQCU/QUDA |
|---|---:|---:|---:|---:|
| `8^3·16` / 2 / c64 | 0.054184 s | 0.399675 s | 7.3763x | 9 / 32 |
| `8^3·16` / 3 / c64 | 0.043200 s | 0.516422 s | 11.9542x | 10 / 33 |
| `16^3·16` / 3 / c64 | 0.149240 s | 0.745390 s | 4.9946x | 31 / 59 |
| `16·32·32·48` / 2 / c64 | 1.944663 s | 2.183002 s | 1.1226x | 11 / 37 |
| `16·32·32·48` / 3 / c64 | 0.662549 s | 1.318356 s | 1.9898x | 14 / 39 |
| `8^3·16` / 3 / c128 | 0.077870 s | 0.963774 s | 12.3768x | 14 / 46 |
| `16^3·16` / 3 / c128 | 0.380444 s | 1.230048 s | 3.2332x | 44 / 87 |
| `16·32·32·48` / 3 / c128 | 2.065148 s | 17.245388 s | 8.3507x | 20 / 60 |

大格 c128 使用 cache-hit formal：缓存由 extended `C=4`、`K=16`、
CPU staging 构建，formal solve 仍按 `C=1` 协议执行。cache identity
只绑定物理资产，cache-hit 校验几何与逐 tensor SHA，不要求历史
`column_batch_size` 等于当前请求；该构建 provenance 必须单独记录。
冷 formal `C=1` setup 仍未完成，不能把 cache-hit 比值当作冷启动
总时间比值。

大格 c64 的二层/三层消融把层级收益单独隔离出来：二者同一输入、同一
precision 和同一 QUDA aligned 策略，三层使 PyQCU 从 1.944663 s 降到
0.662549 s，使 QUDA 从 2.183002 s 降到 1.318356 s。

三层复测的 BiCGStab 对照必须与 MG 比值一起解释：大格 c64 为
PyQCU MG/BiCGStab `0.662549/1.685399 s`、QUDA
`1.318356/0.325375 s`；c128 小格为 PyQCU `0.077870/0.304677 s`、
QUDA `0.963774/0.103475 s`；c128 中格为 PyQCU
`0.380444/0.453031 s`、QUDA `1.230048/0.118572 s`。其中 QUDA
plain BiCGStab 在多点上快于 PyQCU MG，因此 MG-vs-MG 加速比只能归因于
MultiGrid 路径，绝不能外推为 PyQCU 总体求解器优势。

c128 三层大格点 `16·32·32·48` 的早期冷 $C=1$ setup 在 32 GiB
V100 上 OOM 或超时；现在用 extended `C=4`、`K=16`、CPU block
staging、null-basis offload 构建同一物理缓存，再由 formal $C=1$
cache-hit solve。两侧 formal 结果为 PyQCU `2.065148 s` / 20
iterations / `1.78798e-8`，QUDA `17.245388 s` / 60 iterations /
`1.79244e-8`，MG 比值 `8.3507x`。该点进入正式表，但必须同时声明
冷 formal $C=1$ setup 尚未完成。
该容量路径要求在 `seal_cuda_runtime()` 后、steady workspace 分配前调用
`torch.cuda.empty_cache()`；否则 caching allocator 会保留 setup 临时块，
在 32 GiB V100 上以 solve workspace OOM 失败。该释放位于 setup/solve
计时之外，不能计入任何一侧速度比。

同一大格点的 BiCGStab 参考为 PyQCU `3.22998 s`、QUDA
`0.97875 s`；QUDA plain BiCGStab 仍比 PyQCU MG 快约 `2.11x`，
因此上述 MG 比值只能解释为 MultiGrid 路径差异。

`--quda-coarse-precision single` 提供 QUDA mixed-coarse 容量补充：
fine solve 保持 c128，coarse MG 用 single precision。该模式在
`16·32·32·48` 三层真实完成，60 次外层迭代、真残差 `1.79e-8`、
steady solve 约 2.265 s，设备峰值约 9.8 GiB。该记录是 QUDA 容量与
正确性证据，不是 MG-vs-MG 加速比；PyQCU strict 尚未实现同等
mixed-coarse runtime。

`QUDA_RESOURCE_PATH` 应指向仓库内持久 tuning 目录；同一构建复跑若切换
`QUDA_INSTALL`，还必须同步 `QUDA_BUILD_DIR`，否则 CMake provenance 会读取
另一套 precision/reconstruct/nvec 能力并错误判定。

正式 trace-off 与 trace-on 必须分开报告。小格 c64 三层诊断：

| 模式 | PyQCU | QUDA |
|---|---:|---:|
| trace-off | 0.043971 s | 0.505706 s |
| trace-on | 0.125649 s | 0.560432 s |
| 开销因子 | 2.86x | 1.11x |

trace-on 数值只用于分层归因，绝不能再用于正式 speedup。

大格 c128 formal cache-hit 的 PyQCU trace 为：8 次 solve 均为 20 个
外层迭代，trace-on 约 3.31--3.43 s，对 trace-off 的 2.065148 s
约有 1.6x 观测开销。首次 solve 的 coarse V-cycle 约 2.146 s（占
outer 约 63.6%），最粗层 BiCGStab 约 0.297 s；该 trace 无同尺度
QUDA 对照，不能单独归因优势来源。原始 TSV 为
`data/mg_matrix_20260916_round2/formal-c128-l3/pyqcu-formal-trace.tsv`。
trace 的 `solve_end` 次数与正式 `iterations.median` 一致为 20，层号
固定 fine 0/coarse 1/coarsest 2；若看到 20/60 差异，应归因于外层
Krylov 策略，而不能解释成 parity 层计数错位。

多卡方面，当前机器只有一张受这套 PyTorch `sm_70+` 支持的 V100；两张
P100 为 `sm_60`，当前 PyTorch 构建明确不兼容，且 PyQCU/QUDA 本任务构建
目标为 `sm_70`。因此本轮不能把多卡数字并入正式 MG 结论；应记录为环境
能力缺口，而不是把单卡结果冒充多卡验证。
另实测把同一 V100 UUID 在 `CUDA_VISIBLE_DEVICES` 中重复两次会得到
`device_count=0`，不能借此构造同卡的“双逻辑设备”多卡测试；不得把
这种伪多卡输出写入结果。

PyQCU 的 C++ 默认 CMake 架构是 `60-real;60-virtual`，因此 V100 上需显式
用 `QCU_CUDA_ARCHITECTURES='70-real;70-virtual'` 重建并确认
`cuobjdump --list-elf` 显示 `sm_70.cubin`。sm_70 重建后 Tier1 仍全绿；
大格 c64 solve 为 `0.657074 s`（默认 sm60 JIT 为 `0.662549 s`），大格
c128 为 `2.062569 s`（默认 `2.065148 s`），残差和迭代数不变。此前主表
使用 sm60 路径，因此是保守结果；不得把不同 CUDA 架构混成同一速度比。

### 2026-09-17 colored Galerkin 批处理与 CPU staging

`build_strict_galerkin_colored()` 的 support gather、fine-block gather 和
canonical-block scatter 已改为预计算索引与批量写入。`_target_entries()` 与
`_source_color_groups()` 增加缓存，但保留按实际周期目标集合的贪心分组；
不要用固定模数线性着色替代它，`8×16×16×24` 的周期轴上会产生更多色组。
实际采用 16 个坐标 parity 桶优先的确定性顺序：大格源色组从 22 降到
16，C=4 每层 operator calls 从 132 降到 96；小格 c128 `K=4` setup
从 `16.67 s` 降到 `14.66 s`，大格全 double C=4/K=16 冷 setup
从约 `974 s` 降到 `585.28 s`，低于同点 QUDA 的 `630.74 s`。
若按 extended smoke 的冷 setup 加 steady solve 估算，PyQCU 约
`587.34 s`、QUDA 约 `648.10 s`，冷启动总量约快 `1.10x`；该估算不含
输入 I/O/warmup，不能替代 formal solve 的 `8.3507x`。
两版 cache 已审计：22 色旧构建与 16 色 parity 构建的
`identity_sha256`、`manifest_sha256`、`metadata_sha256` 相同，仅
`stats_sha256` 因构造统计不同。需要用 manifest 相等而不是文件大小或
残差相似来证明资产等价。

`--pyqcu-strict-block-device cpu` 与 `--pyqcu-strict-offload-null-basis`
是低显存实验开关，默认关闭；CPU canonical blocks 要求
`retain_blocks=False`，runtime `X/Xinv/Y/Yhat` 仍在目标设备组装。对应的
CUDA 回归为
`test_strict_galerkin_colored_cpu_block_staging_matches_reference`。

真实 c128 smoke 证据（`8^3·16`、三层、`--pyqcu-strict-block-precision
single`、CPU block staging）显示，`K=1` 的 setup 为 `44.86 s`，`K=4`
为 `16.67 s`（约 `2.69x`），两者均通过 `1.84e-8` 真残差门并保持 35 次
外迭代。该收益是 setup 收益，不是 MG solve 加速比；正式对照必须重新运行
QUDA 侧并保持两侧 K 相同。

混合精度 coarse blocks 不能进入主 speedup：c128 小格把 block precision
改为 single 后，MG 外迭代从 double coarse blocks 的 14 次增至 35 次，
steady solve 约为 `0.177628 s`，即使 setup 更快也不是等精度比较。

同输入合成微基准（`8×16×16×32`、c128 blocked basis、c64 blocks、K=256、
44 次算子调用）旧实现 `18.39 s`、新实现 `5.48 s`（约 `3.36x`），峰值
allocated 均约 `13.04 GiB`。该结果只证明 builder 热路径收益，不替代真实
gauge/QUDA 的公平计时。

smoke/formal 的 `require_exact_batch` 会按每个 level 的实际最大色组检查
请求值：小格 `8^3·16` 的 level 0/1 上界分别为 38/4，因此请求 256 或
38 都会被拒绝；正式参数应选择不超过所有 level 上界的值。
