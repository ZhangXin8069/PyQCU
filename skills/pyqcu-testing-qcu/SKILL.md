---
name: pyqcu-testing-qcu
description: pyqcu/testing/qcu 目录的完整生成 skill：经 Cython 桥测 C++ CUDA 后端；含按职责分类的多重网格性能基准套件（clean/bench/verify/collect/mktable/plots）。
---
# pyqcu/testing/qcu

C++ CUDA backend tests via the Cython bridge. These exercise `libqcu.so` from Python.

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

## Usage

```bash
mpirun -np 1 python pyqcu/testing/qcu/conftest.clover.multigrid.py
```

Output: convergence log → `logs/clover_multigrid.log`, performance report → `logs/clover_multigrid_report.log`

## 早期 MG 性能基准套件（代码已删除，证据保留）

dev73 的 `multigrid/legacy/` 与 dev74 的 `multigrid/scaling/`（含
`benchmarks/{volume32,volume1632,multi_gpu}`、`legacy/{cleanup_runner,collect_results,diag_coarse}.py`、
`legacy/{integrated_single_thread,integrated_multi_thread,integrated_multi_thread_batched}/`、
`legacy/verify_nullvecs.py`、`legacy/mg_stencil_build.py`、`legacy/cache.py`、`legacy/mg_schur_build.py`、
`legacy/mg_pyref_expt.py`、`legacy/reference_benchmark.py`、`legacy/dslash_adapter.py`、
`legacy/budget.py`、`legacy/benchmark.py` 与 5 处绘图副本）已于 **2026-10-15 删除**，可复用算法全部
收入库内。**归档证据不随代码删除**：`logs/dev73_5.*`（报告/LaTeX 表/PNG）、`logs/dev74/` 仍是
这些结论的权威出处，引用时直接用 logs 路径。

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

`pyqcu/testing/qcu/multigrid/benchmarks/large_volume/main.py` — 子命令 run / multi / run_gcr / hotspot（带 `--only` 门控），产物镜像 `out/*.json` 与 `logs/dev84/`；报告 `pyqcu/testing/qcu/multigrid/benchmarks/large_volume/report.md`。

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

QUDA 1.1.0 上游默认关闭 double recursive MultiGrid；本项目已在
`refer/git-rep/quda` 加入最小 opt-in 修复：

1. CMake `QUDA_MULTIGRID_DOUBLE=ON` 定义 `GPU_MULTIGRID_DOUBLE`。
2. `matrix_tile.cuh` 的 gauge accessor 对 mixed float/double 做显式
   `complex<T>` 转换。
3. double coarse-link 原子累加改用 double 存储。核心修改在
   `include/kernels/coarse_op_kernel.cuh`：`CacheT` 和 `atomic_helper`
   使用 `typename Arg::store_t`；`coarse_op.in.cu`、`coarsecoarse_op.hpp`、
   `staggered_coarse_op.in.cu` 对 `Float=double` 选择 double atomic store，
   float 路径继续使用确定性 int 固定点。
4. `gauge_norm.cu` 禁止 double fixed-point 检查不再被 double MG 触发。

推荐构建参数：`QUDA_PRECISION=12`、`QUDA_RECONSTRUCT=7`、
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
