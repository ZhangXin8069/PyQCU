---
name: qcu
description: pyqcu/testing/qcu 目录的完整生成 skill：经 Cython 桥测 C++ CUDA 后端；含按职责分类的多重网格性能基准套件（clean/bench/verify/collect/mktable/plots）。
---
# CLAUDE.md — pyqcu/testing/qcu

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

## Legacy Multigrid Benchmark Suite

Development scripts for the early multigrid performance milestone. They benchmark `applyCloverMultigridQcu` against the Clover BiStabCG reference (`applyCloverBistabCgQcu`) across precision / lattice / solver-parameter sweeps, and feed the archived `logs/dev73_5.*` evidence (report, LaTeX tables, PNG figures).

Scripts are archived under `pyqcu/testing/qcu/multigrid/legacy/` (outputs → `logs/dev73/`):

| File | Purpose |
|------|---------|
| `legacy/cleanup_runner.py` | Clean, isolated-process timing of a single config (ref/mg interleaved, min+median speedup) |
| `legacy/reference_benchmark.py` | Extended performance benchmark — precision / lattice / solver-parameter sweeps vs BiStabCG |
| `legacy/verify_results.py` | Correctness checks — SU(3) gauge, solution error, null-vector zero-mode/orthogonality, C++ vs Python coarse dslash |
| `legacy/collect_results.py` | Aggregate clean/bench/verify JSON into `logs/dev73_5_results.json` |
| `legacy/make_tables.py` | Emit LaTeX table snippets (`logs/dev73_5_tbl_*.tex`) for `dev73_5.tex` |
| `legacy/make_plots.py` | Generate convergence / hotspot / speedup / time PNG figures into `logs/` |

The resource-scaling and server-validation suites live in `pyqcu/testing/qcu/multigrid/scaling/` (outputs → `logs/dev74/`); integrated benchmark snapshots are under `pyqcu/testing/qcu/multigrid/legacy/` and archived outputs remain in `logs/`.

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
