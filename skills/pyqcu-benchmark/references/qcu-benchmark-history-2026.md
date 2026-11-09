# QCU benchmark history (2026-09 – 2026-11)

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
events.  `pyqcu/testing/qcu/strict/quda_comparison/summarize_mg_trace.py` aggregates both.
Trace-enabled wall times are diagnostic only; the no-trace formal result is
the speedup authority.

## 2026-09-17 重新校验与 QUDA double MultiGrid

旧 c64 三层 `4.3817x` 结果撤回。它使用 QUDA 原生 CA-GCR/多内层预算，
而 PyQCU 使用一次递归 V-cycle；sm70 native 复测后同格点主口径为
`2.0052x` (`0.655570` s vs. `1.314577` s)，前一次为 `1.9898x`，见
`logs/data/mg_matrix_20260916_round2/formal-c64-l3/benchmark.json`。当前主口径
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
| `16·32·32·48` / 3 / c64 | 0.655570 s | 1.314577 s | 2.0052x | 14 / 39 |
| `8^3·16` / 3 / c128 | 0.077870 s | 0.963774 s | 12.3768x | 14 / 46 |
| `16^3·16` / 3 / c128 | 0.380444 s | 1.230048 s | 3.2332x | 44 / 87 |
| `16·32·32·48` / 3 / c128 | 2.061594 s | 16.166481 s | 7.8417x | 20 / 60 |

大格 c128 使用 cache-hit formal：缓存由 extended `C=4`、`K=16`、
CPU staging 构建，formal solve 仍按 `C=1` 协议执行。cache identity
只绑定物理资产，cache-hit 校验几何与逐 tensor SHA，不要求历史
`column_batch_size` 等于当前请求；该构建 provenance 必须单独记录。
extended `C=4`、`K=16` cache 路径曾记录 `585.28 s`；后续正式 cold `C=1`
setup 也已成功完成，记录为 `2709.45 s`。必须把 cache 构建、cold formal
setup 和 cache-hit steady solve 分成三个时间口径，不能把 cache-hit 比值当作
冷启动总时间比值。

大格 c64 的二层/三层消融把层级收益单独隔离出来：二者同一输入、同一
precision 和同一 QUDA aligned 策略，三层使 PyQCU 从 1.944663 s 降到
0.655570 s，使 QUDA 从 2.183002 s 降到 1.314577 s。

三层复测的 BiCGStab 对照必须与 MG 比值一起解释：大格 c64 为
PyQCU MG/BiCGStab `0.655570/1.668540 s`、QUDA
`1.314577/0.329849 s`；c128 小格为 PyQCU `0.077870/0.304677 s`、
QUDA `0.963774/0.103475 s`；c128 中格为 PyQCU
`0.380444/0.453031 s`、QUDA `1.230048/0.118572 s`。其中 QUDA
plain BiCGStab 在多点上快于 PyQCU MG，因此 MG-vs-MG 加速比只能归因于
MultiGrid 路径，绝不能外推为 PyQCU 总体求解器优势。

c128 三层大格点 `16·32·32·48` 的冷 `C=1` setup 在
`release_python_setup_assets()` 路径下成功完成：先导出 runtime assets，
再释放重复 Python hierarchy，实测 setup `2709.45 s`、steady
`2.067186 s`、20 iterations、残差 `1.78798e-8`。extended
`C=4`、`K=16`、CPU block staging、null-basis offload 仍是更快的
冷构建路径，并由同一 asset-identity cache 进入 formal `C=1` solve。
cache-hit solve。两侧 formal 保守复测为 PyQCU `2.061594 s` / 20
iterations / `1.78798e-8`，QUDA `16.166481 s` / 60 iterations /
`1.79244e-8`，MG 比值 `7.8417x`；首次独立复测为 `8.3507x`。
该容量路径要求在 `seal_cuda_runtime()` 后、steady workspace 分配前调用
`torch.cuda.empty_cache()`；否则 caching allocator 会保留 setup 临时块，
在 32 GiB V100 上以 solve workspace OOM 失败。该释放位于 setup/solve
计时之外，不能计入任何一侧速度比。

同一大格点的保守复测 BiCGStab 参考为 PyQCU `3.251439 s`、QUDA
`1.043206 s`；QUDA plain BiCGStab 仍比 PyQCU MG 快约 `1.98x`，
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
`logs/data/mg_matrix_20260916_round2/formal-c128-l3/pyqcu-formal-trace.tsv`。
trace 的 `solve_end` 次数与正式 `iterations.median` 一致为 20，层号
固定 fine 0/coarse 1/coarsest 2；若看到 20/60 差异，应归因于外层
Krylov 策略，而不能解释成 parity 层计数错位。

多卡方面，2-rank CUDA 可以使用两张 P100，但必须切换到兼容 `sm_60` 的 Python
环境，并使用同时包含 `sm_60`/`sm_70` 的原生库集合。V100+P100 异构组合会混合性能
和架构变量，不能并入正式同窗结论。历史 V100 单卡结果仍应按其原始环境解释，不能
冒充双 P100 验证。
另实测把同一 V100 UUID 在 `CUDA_VISIBLE_DEVICES` 中重复两次会得到
`device_count=0`，不能借此构造同卡的“双逻辑设备”多卡测试；不得把
这种伪多卡输出写入结果。

PyQCU 的 C++ 默认 CMake 架构是 `60-real;60-virtual`，因此 V100 上需显式
用 `QCU_CUDA_ARCHITECTURES='70-real;70-virtual'` 重建并确认
`cuobjdump --list-elf` 显示 `sm_70.cubin`。sm_70 重建后 Tier1 仍全绿；
大格 c64 solve 为 `0.657074 s`（默认 sm60 JIT 为 `0.662549 s`），大格
c128 为 `2.062569 s`（默认 `2.065148 s`），残差和迭代数不变。此前主表
大格 c64/c128 已改采用 sm70 独立正式复测；小/中格仍保留 sm60 路径，
因此按保守旧架构记录。不得把不同 CUDA 架构混成同一速度比。
每次 worker 记录现在都写入 `provenance.pyqcu_library`，包含 libqcu
路径、SHA256、`QCU_CUDA_ARCHITECTURES` 请求值和 `cuobjdump` 实际
`architectures` 列表；正式比较前必须核对目标架构。

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
输入 I/O/warmup，不能替代 formal solve 的保守 `7.8417x`。
两版 cache 已审计：22 色旧构建与 16 色 parity 构建的
`identity_sha256`、`manifest_sha256`、`metadata_sha256` 相同，仅
`stats_sha256` 因构造统计不同。需要用 manifest 相等而不是文件大小或
残差相似来证明资产等价。

`--pyqcu-strict-block-device cpu` 与 `--pyqcu-strict-offload-null-basis`
是低显存实验开关，默认关闭；CPU canonical blocks 要求
`retain_blocks=False`，runtime `X/Xinv/Y/Yhat` 仍在目标设备组装。对应的
CUDA 回归为
`test_strict_galerkin_colored_cpu_block_staging_matches_reference`。

2026-09-18 的矩阵补跑把大格 c128 路径改成显式生命周期协议：
probe 结束时先释放 `inverse/site_matrix` 和最后一轮 full-field
workspace，再把 canonical blocked basis 下放 CPU；`_finish` 在目标
CUDA 设备上逐方向 stream action blocks 并组装 coarse links。递归更新
null vector 时允许在 CPU 计算 restriction，但上传前必须切回当前
coarse operator 的 packed-asset dtype/device。coarse operator 的
dtype/device 参考不得再读取可能已下放的 `transfer.V`。正式恢复命令为：

```bash
source ./env.sh
source pyqcu/testing/qcu/strict/quda_comparison/quda_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python pyqcu/testing/qcu/strict/quda_comparison/bench_mg_matrix_full.py \
  --execute --resume --timeout 7800 \
  --output-dir data/mg_matrix_full_v100_c128 \
  --cache-root data/strict_cache_matrix/matrix-v100-c128 \
  --only-side pyqcu --only-precision c128 --only-device v100 \
  --only-lattice 16x32x32x48 --only-levels 2 --only-levels 3 \
  --extra-args "--pyqcu-strict-block-device cpu \
  --pyqcu-strict-offload-null-basis --strict-galerkin-column-batch 1 \
  --strict-galerkin-projection-batch 4 \
  --strict-galerkin-max-workspace-bytes 1073741824"
```

矩阵编排器的 `--resume` 是成功单元的唯一复用判据；不要把 `--resume`
继续传给 collector，否则修正执行参数后会与旧 failed JSON 的
`config_hash` 冲突。`QcuStrictAssetBinding` 必须 fail closed 拒绝
任何非 CUDA 资产。当前 V100 大格 c128 PyQCU 结果：L2 trace-off
`10.486293 s` / 15 iterations，L3 trace-off `2.084975 s` /
20 iterations，真残差均 `1.79e-8`；大格 QUDA double 的矩阵补跑在
32 GiB 卡上仍会 OOM，不能把历史或单次慢速 replay 直接并入正式比值。

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

## 2026-09-27 Strict Hopper overlap

`StrictVectorHalo` 现在可把 active face pack、pinned-host staging 和
`MPI_Isend/Irecv` 放在独立交换流上，并在 coarse local-kernel 之后
才等待接收；`PYQCU_MPI_OVERLAP=0` 可回退到原始顺序路径。该优化改变
remote-forward 项的浮点累加顺序，因此只默认用于 complex64 分布路径；
complex128 需要 `PYQCU_STRICT_OVERLAP_C128=1` 才显式启用。大格 c128
三层已在 P100 上复现：默认禁用时 32 个外层迭代、steady `5.10 s`；
错误启用时 451 个外层迭代、`46.28 s`，两者最终真残差都低于 gate。

快速回归：

```bash
source pyqcu/testing/qcu/strict/quda_comparison/p100_env.sh
PYQCU_MPI_OVERLAP=0 mpirun --allow-run-as-root --oversubscribe -np 2 \
  python -B pyqcu/testing/qcu/strict/quda_comparison/strict_mpi_solve_probe.py \
  --shape 8 8 8 16 --grid 2 1 1 1 --restart 20 --max-iter 200 \
  --dtype c64 --galerkin-mode colored
PYQCU_MPI_OVERLAP=1 mpirun --allow-run-as-root --oversubscribe -np 2 \
  python -B pyqcu/testing/qcu/strict/quda_comparison/strict_mpi_solve_probe.py \
  --shape 8 8 8 16 --grid 2 1 1 1 --restart 20 --max-iter 200 \
  --dtype c64 --galerkin-mode colored
python -B pyqcu/testing/qcu/strict/quda_comparison/validate_matrix_trace_evidence.py \
  data/mg_matrix_overlap_20260927_p100/units/*.json
```

## 2026-10-21 真实组态统一计时与资源守卫

QCU、KUPL、QUDA 的 c64/c128 RHS 必须从同一个 PCG64 float64 流生成后再 cast；
不能因 backend dtype 改变随机数采样精度。正式 solve 前应比较全局 RHS SHA256。

QUDA tune-off 必须同时具备：进程启动前 `QUDA_ENABLE_TUNING=0`、运行时回读
`invert.tune=QUDA_TUNE_NO`，以及日志中的 `Autotuning disabled`。PyQUDA 当前
封装可能不暴露 `QudaGaugeParam.tune`；该字段缺失应记为 `not_exposed`，不能
伪造为已验证。QUDA `secs` 是内部 solver 时间，不能与包含 upload/download/
preamble 的完整 `Dirac.invert` 墙钟混在同一比值中。

每次 GPU benchmark 开始后第 10 分钟必须检查：Slurm 状态、进程 CPU/RSS、
GPU util/显存/功耗、阶段日志、缓存文件、stderr。若 GPU 长期 0% 且无阶段进展，
应立即停止并改测试路径；总 GPU 预算按 `sum(ElapsedRaw * gres/gpu)` 从
`sacct` 记账，默认硬上限 20 卡时。

## 2026-11-06 Kunpeng hardware bandwidth and compute accounting

Kunpeng CPU performance conclusions must report the measured machine ceiling on
the same compute-node class, not just solver wall time or an HBM topology flag.
For the 16-domain LX2/HBM profile, record:

- effective HBM bandwidth from the unchanged `refer/kunpeng/lx2-stream-source`
  SVE2 FP64 `Copy`/`Scale`/`Add`/`Triad` probe, with per-domain scaling and a
  synchronized aggregate at the production rank/thread shape;
- live `numactl --hardware`, `numastat -m`, affinity/cpuset and per-rank memory
  node evidence, including the sampled process page distribution;
- measured SVE2 floating-point throughput from a dedicated compute probe, plus a
  separately labelled theoretical FP64 ceiling derived from frequency, physical
  cores and SVE lane width.

The solver report must state the fraction of measured HBM and compute throughput
actually reached. `KUPL_TOPOLOGY_REPORT=1` with `memory_bound=true` proves only
that the memory-policy call succeeded; it does not prove bandwidth saturation.
A single-domain `STREAM` result must not be multiplied by 16 and presented as an
aggregate without running the domains concurrently. Keep raw per-rank samples and
aggregate medians/MAD, and do not compare runs on different node classes or under
different co-tenant contention.

For KUPL BiCGStab versus Legacy-MG, run separate grid matrices. Iteration count is
the MG advantage, but coarse transfer/operator synchronization can erase it at high
rank counts. Also screen MG cache-build time and cache size separately from steady
solve: E/nv_iters and local-stencil batching may leave solve time unchanged while
multiplying setup cost. Login-node grids are only elimination screens; HBM placement
and compute-node medians remain the acceptance authority.

When a cache builder is the blocker, decompose setup separately from steady solve.
The KUPL local-stencil coarse-site loop was serial and dominated build time; a
per-worker-scratch parallel build reduced L8/L16/L24 setup by 12.9x/17.6x/>31x while
preserving gram/Galerkin checks. Report `builder_seconds`, cache-hit solve, and
cache size independently, and enable the parallel builder in targeted regressions so
the serial fallback does not hide setup-only correctness failures.

## 2026-11-06 KUPL true-MG acceptance and A/B correction

The Legacy-MG cache validator must stream logical tensors in strict C order. The
blocked `lonv=[E,e,X,bx,Y,by,Z,bz,T/2,bt]` shape is vulnerable to a Cartesian
multi-axis chunk walk: fixing only the chunk byte budget can hash non-adjacent
axis blocks and produce a false manifest mismatch. Fix prefix coordinates and
split one axis while keeping all trailing axes complete. A regression must compare
`logical_dataset_sha256` with `logical_array_sha256` on the blocked shape, not only
on a small contiguous fixture.

Formal KUPL MG runs must set `KUPL_REQUIRE_MG_CORRECTION=1`. A `levels=2` report
with `fallback=fine_bicgstab`, `coarse_calls=0`, or no accepted correction is a
BiCGStab result and must not enter a MG speedup table. The same gate must be
disabled only for the explicit `levels=1,restart=0` BiCGStab control.

Coarse correction needs a damping line search because a full additive correction
can increase the fine true residual even when the coarse solve itself converges.
Build the coarse correction once, try dampings `1, 0.5, 0.25, 0.125`, and accept
only a decrease. A failed search restores the previous fine solution and residual
before declaring fallback.

Current 2026-11-06 L24 c64 result, 16 ranks x 8 threads and `grid=1 1 1 16`:
BiCGStab median `1.2545952125 s` (16 iterations), true-MG median
`1.3541084125 s` (`fine_calls=37`, `coarse_calls=29`), giving a BiCGStab/MG
ratio of `0.926510`. The MG path was about 7.9% slower in this same login-node
window. The window was contaminated by another user's `mpirun -np 8 ./test` and
must not be promoted to a compute-node speedup claim. The sign of the result is
still important: iteration reduction alone does not establish an MG speedup.

L24 true-MG grid screening found the best observed login-node point at 16 ranks
x 8 threads, `grid=1 1 1 16`, around `1.686 s`; 32-rank grids were slower. L8 and
L16 scans must reject grids whose `grid_t` exceeds the cache's coarse time extent
before timing. L32x64 c64 gauge/source were prepared successfully, but the
one-rank cache build and 32-rank BiCGStab control both exceeded the 10-minute
login-node gate without a report. L32 results remain blocked until setup is made
substantially cheaper or a qualified compute-node allocation is available.

A later same-window L24 `restart=10` A/B at the same 16-rank t-split gave
BiCGStab median `1.39724 s` and true-MG median `0.78976 s`, a `1.7692x`
median speedup (`fine_calls=32`, `coarse_calls=19`). The full-residual
companion run passed at `1.04379 s` with independent residual `6.88873e-5`
against the `2.44140625e-4` c64 acceptance. This supersedes the earlier
`restart=5` negative screen as the current login-node candidate, but still
requires compute-node repeats and does not satisfy the `0.1 s` target.

Larger restart intervals showed the correction-count knee: `restart=20`
gave BiCGStab/MG medians `1.230783/0.705146 s` (`1.7454x`) and `restart=50`
gave `1.091866/0.704241 s` (`1.5504x`), both with `fine_calls=29` and
`coarse_calls=9`. At `restart=50` only initial deflation ran, followed by fine
BiCGStab, so do not label it a periodic V-cycle. Its independent-residual run
passed at `0.947332 s` with residual `5.5308248713e-5`; this is the lowest
absolute L24 login-node solve, still roughly 9.5x above the target.

On the recovered LX2 scheduler, `dsub -q q_hpcapp <script>` was accepted even
though the same script without the CLI queue flag was refused. However, the
first 32-rank compute-node A/B remained in host pre-hook with no stdout/report
and no documented cancel command. Do not treat scheduler acceptance as compute
node health: first run a short writable-output probe, inspect `djob -ll`, and
only then submit a solver matrix. Never bypass host-key verification or touch
another user's processes.

### 2026-11-06 rank-slab cache and grid-specific A/B

The current remote `h5py 3.7.0` has no MPI support
(`h5py.get_config().mpi == False`), so the optional HDF5 MPIO path cannot be
used there. `prepare_mpi_slab.py` writes contiguous `rank_XXXX.h5` slabs for a
fixed lattice/grid, and `kupl_multigrid.py` accepts
`--gauge-slab-dir`/`--source-slab-dir`. Validate the rank file trailing shape;
never reuse a slab directory after changing the process grid.

On login02 with L24 c64, 16 ranks x 8 threads, the slab cache reduced measured
gauge/source reads to about `0.096/0.010 s`. It does not prove compute-node
launch health: 16-rank scheduler jobs still reached the ten-minute gate without
a report. Keep those runs unclassified until a case report and independent
residual exist.

The current login-node grid screen is:

| grid | BiCGStab | true-MG restart=20 | interpretation |
|---|---:|---:|---|
| `1 1 1 16` | 1.116278 s median | 1.037458 s median | positive but noisy |
| `1 1 2 8` | 1.844325 s | 1.840493 s | rejected |
| `1 2 2 4` | 1.084889 s | 1.250455 s | rejected for MG |

The t-only median ratio is `1.076x`, with individual paired ratios spanning
roughly `0.78–1.59x`; this is not a stable compute-node speedup. The
restart=20 full-residual run passed at `1.316182 s`, `verify_status=0`, and
independent residual `5.5308248713e-5` under `2.44140625e-4`. Report setup
reads, solver wall, correction counts, and residual separately.

An `ht=2` boundary split opt-in reduced the BiCG median from `1.099630` to
`1.037594 s`, but the true-MG median moved from `0.902746` to `1.041128 s`.
Because MG is the objective, the split is not a production default; require
separate MG A/B before enabling it.

After the manual periodic-geometry fix, a compute-node 16×8 t-only HBM
BiCGStab run measured `wall_seconds=213.4409 s`, 16 iterations, independent
residual `3.5410827e-6`, and `status=PASS`. HBM-bound initialization reached
`register=41.75-46.30 s` per rank. Treat input prebind, HDF5 slab read,
initialization and the `applyCloverMultigridQcu` wall as separate fields; never
fold setup into the solve wall when comparing against the 0.1 s target.

The paired true-MG case was terminated after the 10-minute gate with no report
(`jobExitCode=143`). It is a timeout/health result, not a valid MG speedup
point.
