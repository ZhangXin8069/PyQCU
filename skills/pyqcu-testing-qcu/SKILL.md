---
name: pyqcu-testing-qcu
description: 当任务涉及 pyqcu/testing/qcu、Cython CUDA 后端测试或 MultiGrid 基准套件时使用。
---
# pyqcu/testing/qcu

C++ CUDA backend tests via the Cython bridge. These exercise `libqcu.so` from Python.

CUDA production topology is one Python process = one MPI rank = one
GPU/DCU/GPGPU. The multi-GPU benchmarks and conftest entries below use the
test-only, explicitly constructed `MultiGpuMultigrid` exception for consistency
or throughput evidence; they are not production launch templates.

See
[`docs/cuda_execution_topology.md`](../../docs/cuda_execution_topology.md).

## Test Files

| File | What it tests |
|------|---------------|
| `conftest.cuda.py` | Basic CUDA availability and context |
| `conftest.mpi.py` | MPI grid setup and halo exchange |
| `conftest.wilson.bistabcg.py` | Wilson BiStabCG via C++ backend |
| `conftest.wilson.bistabcg.dslash.py` | Wilson BiStabCG dslash kernel |
| `conftest.wilson.cg.py` | Wilson CG solver |
| `conftest.clover.py` | Clover term construction |
| `conftest.clover.bistabcg.py` | Clover BiStabCG solver |
| `conftest.clover.bistabcg.dslash.py` | Clover BiStabCG dslash |
| `conftest.clover.multigrid.py` | Clover multigrid V-cycle solver |
| `bench_multi_gpu.py` | Test-only multi-thread multi-GPU Clover-MG steady-state benchmark with `--no-reference` |
| `slurm_multi_gpu_smoke.sh` | Test-only Slurm 2-GPU build, numeric gates, consistency and real-LIME regression |

## 2026-10-23 MPI 与快速矩阵门禁

- `pyqcu/testing/simple/mpi` 使用扁平脚本布局，不创建 `examples/` 子目录。示例必须在
  启动 solver 前验证 rank 数是 2 的正整数次幂，并支持 `DRY_RUN=1` 只输出解析后的命令；
  3/6/12 等 rank 数必须 fail-closed。
- 六项正确性矩阵为 Torch/CUDA/KUPL × Legacy/Strict，统一只在 `8×8×8×16` 运行；
  同批要有
  QUDA aligned MG 和 Torch/CUDA/KUPL/QUDA BiCGStab 对照。计时只用 trace-off，并设置
  `QUDA_ENABLE_TUNING=0`。
- 大格 `24×24×24×64` c64 Strict 两层初始化运行约 33m50s 仍未完成，项目已决定大格
  Strict benchmark 行 SKIP。该记录是未完成下界，不是数值失败或性能结果。
- 长测试必须分阶段输出进度并设置阶段/总超时。GPU 100% util 但长时间无输出不能作为
  健康证明；容器 `ptrace_scope=1` 时 `py-spy`/`gdb` 可能无法附加。

完整矩阵、rank/设备约束和真残差证据见
[`pyqcu-benchmark/references/qcu-multigrid-20261023.md`](../pyqcu-benchmark/references/qcu-multigrid-20261023.md)。

## Usage

```bash
mpirun -np 1 python pyqcu/testing/qcu/conftest.clover.multigrid.py
```

Output: convergence log → `logs/clover_multigrid.log`, performance report → `logs/clover_multigrid_report.log`

### Slurm multi-GPU regression

Submit from the repository root:

```bash
sbatch --gpus=2 --cpus-per-task=64 --time=01:00:00 \
  --output=logs/slurm-%j-multi-gpu.out \
  --error=logs/slurm-%j-multi-gpu.err \
  pyqcu/testing/qcu/slurm_multi_gpu_smoke.sh
```

The script detects the active `sm_*`, builds the C++ backend, runs `install.sh`,
then executes ABI, Wilson/Clover, two-GPU consistency, steady-state benchmark,
and real-LIME gates. Use `--no-reference` only for production MG throughput;
`total_solve_median` is the steady end-to-end timing, while `mg_time` covers
only the C++ solver call.

## 早期 MG 性能基准套件（代码已删除，证据保留）

dev73 的 `multigrid/legacy/` 与 dev74 的 `multigrid/scaling/`（含
`benchmarks/{volume32,volume1632,multi_gpu}`、`legacy/{cleanup_runner,collect_results,diag_coarse}.py`、
`legacy/{integrated_single_thread,integrated_multi_thread,integrated_multi_thread_batched}/`、
`legacy/verify_nullvecs.py`、`legacy/mg_stencil_build.py`、`legacy/cache.py`、`legacy/mg_schur_build.py`、
`legacy/mg_pyref_expt.py`、`legacy/reference_benchmark.py`、`legacy/dslash_adapter.py`、
`legacy/budget.py`、`legacy/benchmark.py` 与 5 处绘图副本）已于 **2026-10-15 删除**，可复用算法全部
收入库内。**归档证据由 tag 保存**：dev73/dev74 对应的报告、LaTeX 表和 PNG 已随 tag
归档并从当前工作区清理；引用时使用
`git show <tag>:<path>`，不要恢复 `logs/<tag>/` 永久副本。

| 已删除来源 | 库内规范位置 |
|---|---|
| `legacy/verify_nullvecs.py` | `pyqcu/testing/__init__.py::verify_nullvecs` |
| `legacy/mg_stencil_build.py` | `pyqcu/torch/tools/_multigrid.py::build_stencil`/`build_stencil_mt`/`apply_stencil` |
| `legacy/cache.py` | `pyqcu/cuda/_multi_gpu.py` + `pyqcu/tools/_io.py` |
| `legacy/mg_schur_build.py` | `pyqcu/torch/tools/_strict_galerkin.py` |
| `legacy/mg_pyref_expt.py` | `pyqcu/torch/solver/_multigrid_core.py` |
| `legacy/reference_benchmark.py` | `pyqcu/cuda/_logs.py` + `pyqcu/torch/solver/_bistabcg.py::bistabcg_history` |
| `legacy/dslash_adapter.py` | `pyqcu/cuda/_schur_op.py` |
| `legacy/budget.py` | `pyqcu/tools/_budget.py` |
| `legacy/benchmark.py` | `pyqcu/tools/_env.py` |
| 5 处绘图副本（legacy/scaling/volume24） | `pyqcu/tools/_bench_plots.py` |

**归档结论（代码已删，结论保留）**：本地小卡上 MG 恒慢于 BiStabCG；V100 最好 2.43×；
3 层优于 2 层；restart 20 优于 10；多线程单卡无收益；`CudaSchurOp` 依赖
`applyCloverBistabCgDslashQcu`。完整映射与结论原文见 `pyqcu/testing/qcu/AGENTS.md`。

## Large-Volume Multigrid 攻坚套件（当前版）

`pyqcu/testing/qcu/multigrid/benchmarks/large_volume/main.py` — 子命令 run / multi / run_gcr / hotspot（带 `--only` 门控），运行期间产物写入 `out/*.json` 与 `logs/`；阶段报告提交并打 tag 后，从工作区删除对应报告和日志副本。

结论（16×32×32×48 统一格子）：粗空间 ρ_V=0.9759（连续谱无孤立低模簇），MG>2 目标不可达；
体积标度 1.5× 体量仅 0.421×，「大格子有利」证伪。但净优化使 V100 上 MG 首次稳定超 BiStabCG
1.13–1.16×，自适应校正门控再降 MG_2L −18%。机制：CUDA Graph 段回放（8 迭代/段）、零拷贝标量、守卫标量内核、粗解开销 3246→4ms、V-cycle 156→60ms。

剖析工具边界：nvprof 可用（权威）；torch.profiler/kineto 捕不到跨线程 C++ 内核；nsys 在 WSL2 失效。

## 单功能 QCU 回归（当前维护入口）

`pyqcu/testing/qcu/single_qcu_*.py` 将 C API 拆成可独立运行的最小测试：每个入口固定小格点、私有
`params/argv/set_ptrs`，严格递增 `_SET_INDEX_`，并把输出还原后交给纯 PyTorch Wilson/Clover
参考。`single_qcu_api.py` 在不启动 CUDA 的情况下闭合检查 `pyqcu.h`、`qcu_api.pxd`、`qcu.pyx`
的导出集合；Strict 与 Clover-MG 入口在缺少缓存/近零向量时只做形状契约检查并明确 skip。

## QUDA 对照侧的构建与调用契约（2026-09-17；2026-10-15 自 `pyqcu-cuda` 移入）

本节讲的是 `strict/quda_comparison/` 的**对照侧**——`refer/git-rep/quda` 的 opt-in 构建
与 PyQUDA/QUDA 的调用口径——不是 `pyqcu.cuda` 桥接层的内容，故归此。

## QUDA double MultiGrid 对照构建（2026-09-17）

`refer/git-rep/quda` 现在是只读上游 submodule，固定到
`e75ee8835f7c93683c6c96975eddc2e476c13154`，不得在其工作树中加入本地修复。
上游仍默认关闭 double recursive MultiGrid；下列条目是本项目历史上的 opt-in
修补记录，不属于当前 submodule 状态：

1. CMake `QUDA_MULTIGRID_DOUBLE=ON` 定义 `GPU_MULTIGRID_DOUBLE`。
2. `matrix_tile.cuh` 的 gauge accessor 对 mixed float/double 做显式
   `complex<T>` 转换。
3. double coarse-link 原子累加改用 double 存储。核心修改在
   `include/kernels/coarse_op_kernel.cuh`：`CacheT` 和 `atomic_helper`
   使用 `typename Arg::store_t`；`coarse_op.in.cu`、`coarsecoarse_op.hpp`、
   `staggered_coarse_op.in.cu` 对 `Float=double` 选择 double atomic store，
   float 路径继续使用确定性 int 固定点。
4. `gauge_norm.cu` 禁止 double fixed-point 检查不再被 double MG 触发。

如需重新启用 double recursive MultiGrid，必须复制上游固定提交到独立的
`data/quda-*` 工作树，在 shadow 中维护补丁并完成真残差回归；不得修改
`refer/git-rep/quda`。上述历史补丁是否适用于当前上游提交也必须重新验证。

以下推荐构建参数仅适用于已应用上述 shadow 补丁的构建：`QUDA_PRECISION=12`、`QUDA_RECONSTRUCT=7`、
`QUDA_INTERFACE_QDP=ON`、`QUDA_QIO=ON`、`QUDA_QMP=ON`、
`QUDA_MULTIGRID_NVEC_LIST=12,24`、`QUDA_ENABLE_MMA=OFF`、
`QUDA_GPU_ARCH=sm_70`。该组合同时覆盖 c64 和 c128，不再分别维护
c64-only/double-only 两个安装。

上述 `sm_70` 参数只适用于 V100 类对照构建。P100 需要兼容 `sm_60` 的 Python
环境，以及同时包含 `sm_60`/`sm_70` 的原生库集合。不要把 V100+P100 异构双卡结果
当作同架构性能结论。

`pyqcu/testing/qcu/strict/quda_comparison/quda_env.sh` 默认指向该组合安装，并设置匹配的
`QUDA_BUILD_DIR`。切换 `QUDA_INSTALL` 时必须同步构建目录，否则正式
provenance 会把另一套 CMake precision/reconstruct/nvec 能力当成当前库。
QUDA 侧调用 plain BiCGStab 参考前，必须同时清除
`inv_type_precondition` 和 `invert_param.preconditioner`（后者置为
`pyquda_comm.pointer.Pointer("void")`）。

`bench_strict_vs_quda.py --quda-coarse-precision single` 必须通过
`dirac.setPrecision(sloppy=..., precondition=..., refinement_sloppy=...)`
同时更新 `gauge_param` 和 `invert_param`。只改 `invert_param` 会留下
double sloppy gauge，coarse single 构建时触发 `Precisions 4 8 do not
match`。正确设置后 QUDA c128 大格点三层可运行，outer fine precision
仍是 double，`precision_null` 与 coarse fields 为 single。

不要把 trace 诊断中的 `PYQCU_STRICT_TRACE_FILE`/`QUDA_MG_TRACE_FILE`
带进正式 speedup；正式 profile 会对活动 trace 环境 fail closed，只有
显式 `--allow-trace` 才能运行诊断。

## 2026-10-21 LIME 分布式对照补充

`benchmark_qcu_lime_mpi.py` 与 `benchmark_quda_lime_mpi.py` 使用同一真实 LIME、
同一 PCG64 全局 RHS 和逐步 loading/setup/warmup/steady 计时。QCU 真残差必须
在求解器返回后单独计算；不能仅凭 `iterations < max_iter` 宣称数值正确。

本集群 CUDA-aware MPI 在首个 distributed Wilson halo 会阻塞，真实 LIME
multi-rank QCU 默认设 `PYQCU_MPI_DEVICE_AWARE=0`。强制 host staging 后可完成
2-rank 收敛；恢复 device-aware 前必须通过 2-rank halo、全算子残差和 10 分钟
阶段审计。

QUDA MG 的 `n_vec` 必须在运行时按实际 `libquda.so` 回读。历史某构建实例只支持
`6,24,32`，请求 `12` 会明确报 `Unsupported nVec 12`；不要依据历史构建文档或
绝对安装路径假定支持 `12,24`。正式 MG 比较应先解析当前 `QUDA_INSTALL` 的 CMake
capability，再选择两侧相同的 coarse spin/nvec 语义。

任何 GPU 作业超过 10 分钟仍在运行时，必须留下一次运行中审计记录；仅有
`RUNNING` 状态不足以证明任务健康。

## 2026-10-29 raw Clover parity 与 Legacy/Strict 性能门

同一 `applyInitQcu` slot 内切换 `params[_PARITY_]` 后，`applyCloverQcu` /
`applyCloversQcu` 必须同步 native/device parity。遗漏时 even/odd Clover 会位级相同，
并把错误 odd term 写入 Legacy cache。最小回归为
`pyqcu/testing/qcu/test_csw_cuda.py::test_cuda_clover_parity_refresh_within_initialized_slot`；
同 gauge 的 KUPL-vs-CUDA `hnn/hdg/sit` 当前均低于 `3e-7`。

多 MPI CUDA Legacy cache builder 不要把 overlap patch 交给仍绑定
`MPI_COMM_WORLD` 的 CUDA Clover halo。此类调用会在 patch `Sendrecv`、Python
`Bcast` 和全局 `Gather` 之间形成 collective 顺序错位。现行 LatticeSet 在
`NODE_SIZE=1`、grid product 为 1、world size 大于 1 时创建私有 singleton
communicator，并把 `set_ptr->comm` 传给 Clover/coarse halo；因此所有 rank 都可使用
本地 GPU patch stencil，报告应记录 `stencil_device=cuda:*`。

多 MiB/GiB `lonv` 不得再经 `comm.gather(root=0)` 和 `scatter`。按 canonical
`[E,12,Cx,bx,Cy,by,Cz,bz,Ct,bt]` 轴 `(2,4,6,8)`，各 rank 应通过 HDF5 MPI-IO
写入 owned coarse hyperslab，再按周期连续 slice 读取本 rank patch；时间 stride 是
`2*mg_grid[t]`。四个张量都应由各 rank 经 MPI-IO 写 owned slab；L48 `1604062`
已证明只去掉 `lonv` gather 后，root 汇总 `hnn/hdg/sit` 仍会 cgroup OOM。写入后
立即释放 patch/local 副本，rank 0 只用固定上限（当前 64 MiB）的多维 dataset
streaming SHA 构造 manifest；只按第一维
读取会让单个 L48 组件切片膨胀到数 GiB。验证证据：`8x8x8x16` 4-rank、
`2x2x1x1`、E2 的 `lonv` MPI-IO `0.354144 s`、GPU stencil `0.137035 s`、其余三张量
MPI-IO `0.884016 s`、总 setup `3.040447 s`，4/4 rank slab 回读通过；同 cache 由 KUPL 求解得到
`verify_status=0` 和独立真残差 `4.110297e-7`。

Strict-MG 性能 benchmark 必须使用合法 4D process grid：每个 rank-local coarse
extent 必须为偶数。`8x8x8x16` 的 4-rank 应使用 `2x2x1x1`，不能使用 y-only
`1x4x1x1`。测后独立 full residual 应在 MPI solve 退出后由独立进程计算，避免同一
`COMM_WORLD` 内重入 distributed dslash 导致 busy-wait。缺少真实 Strict cache 时，
允许用同一 gauge/RHS 的持久化内存资产，但必须记录 cache/setup 时间并标记为
合成或内存资产。

分布式 Strict runtime 的 MG level shape 必须按
`global_lattice / (process_grid * mg_block)` 计算 local coarse shape。cold setup
使用 Torch hierarchy 的 local shape，measure 若误写全局 coarse shape，会在
`applyMultigridStrictFgmresQcu` 报 `strict FGMRES coarse geometry mismatch`。
`8x8x8x16` 的 2-rank 使用 `1x2x1x1`、local coarse shape `[4,2,4,8]`；4-rank 使用
`2x2x1x1`。性能 trace 每次 Slurm 作业必须写入 `trace/<job-id>/rank*.tsv`，不能复用
固定 `trace/rank00000.tsv`；native trace 以 append 打开，旧文件存在时探针应 fail-closed。

Slurm 只分配一个 batch task 时，OpenMPI 可能把多 GPU 作业仍计为一个 slot。启动
`mpirun -np N` 应使用项目统一的 `--bind-to none --oversubscribe`，同时保持 Slurm
`--gpus=N --cpus-per-task=N`，不得借 oversubscribe 申请少于 rank 的 CPU/GPU。
