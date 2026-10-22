---
name: pyqcu-testing-qcu
description: pyqcu/testing/qcu 目录的完整生成 skill：经 Cython 桥测 C++ CUDA 后端；含按职责分类的多重网格性能基准套件（clean/bench/verify/collect/mktable/plots）。
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
| `legacy/mg_stencil_build.py` | `pyqcu/tools/_multigrid.py::build_stencil`/`build_stencil_mt`/`apply_stencil` |
| `legacy/cache.py` | `pyqcu/cuda/_multi_gpu.py` + `pyqcu/tools/_io.py` |
| `legacy/mg_schur_build.py` | `pyqcu/tools/_strict_galerkin.py` |
| `legacy/mg_pyref_expt.py` | `pyqcu/solver/_quda_multigrid.py` |
| `legacy/reference_benchmark.py` | `pyqcu/cuda/_logs.py` + `pyqcu/solver/_bistabcg.py::bistabcg_history` |
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

QUDA MG 的 `n_vec` 必须在运行时按实际 `libquda.so` 回读。当前
`/home/bingxing2/home/scx8080/lib/quda-develop/build` 编译实例支持
`6,24,32`，请求 `12` 会明确报 `Unsupported nVec 12`；不要仅依据历史构建文档
假定支持 `12,24`。正式 MG 比较应先解析实际 CMake capability，再选择两侧相同的
coarse spin/nvec 语义。

任何 GPU 作业超过 10 分钟仍在运行时，必须留下一次运行中审计记录；仅有
`RUNNING` 状态不足以证明任务健康。
