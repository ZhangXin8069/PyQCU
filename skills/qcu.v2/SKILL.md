---
name: qcu
description: examples/qcu 目录的完整生成 skill：经 Cython 桥测 C++ CUDA 后端；含 dev73_5 多重网格性能基准套件（clean/bench/verify/collect/mktable/plots）。
---
# CLAUDE.md — examples/qcu

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
mpirun -np 1 python examples/qcu/conftest.clover.multigrid.py
```

Output: convergence log → `logs/clover_multigrid.log`, performance report → `logs/clover_multigrid_report.log`

## Dev73_5 Multigrid Benchmark Suite

Development scripts for the dev73_5 multigrid performance milestone. They benchmark `applyCloverMultigridQcu` against the Clover BiStabCG reference (`applyCloverBistabCgQcu`) across precision / lattice / solver-parameter sweeps, and feed `logs/dev73_5.*` (report, LaTeX tables, PNG figures).

Scripts are archived under `examples/qcu/dev73/` (outputs → `logs/dev73/`):

| File | Purpose |
|------|---------|
| `dev73/mg_dev73_5_clean.py` | Clean, isolated-process timing of a single config (ref/mg interleaved, min+median speedup) |
| `dev73/mg_dev73_5_bench.py` | Extended performance benchmark — precision / lattice / solver-parameter sweeps vs BiStabCG |
| `dev73/mg_dev73_5_verify.py` | Correctness checks — SU(3) gauge, solution error, null-vector zero-mode/orthogonality, C++ vs Python coarse dslash |
| `dev73/mg_dev73_5_collect.py` | Aggregate clean/bench/verify JSON into `logs/dev73_5_results.json` |
| `dev73/mg_dev73_5_mktable.py` | Emit LaTeX table snippets (`logs/dev73_5_tbl_*.tex`) for `dev73_5.tex` |
| `dev73/mg_dev73_5_plots.py` | Generate convergence / hotspot / speedup / time PNG figures into `logs/` |

Newer dev74 / dev74_1 suites live in `examples/qcu/dev74/` (outputs → `logs/dev74/`); the test11/test12 integration suites live in `logs/test11/`, `logs/test12/` (see the `test12` skill).

## Dev84 Multigrid 攻坚套件（当前版）

`examples/qcu/dev84/main.py` — 子命令 run / multi / run_gcr / hotspot（带 `--only` 门控），产物镜像 `out/*.json` 与 `logs/dev84/`；报告 `examples/qcu/dev84/dev84_report.md`。

结论（16×32×32×48 统一格子）：粗空间 ρ_V=0.9759（连续谱无孤立低模簇），MG>2 目标不可达；
体积标度 1.5× 体量仅 0.421×，「大格子有利」证伪。但净优化使 V100 上 MG 首次稳定超 BiStabCG
1.13–1.16×，自适应校正门控再降 MG_2L −18%。机制：CUDA Graph 段回放（8 迭代/段）、零拷贝标量、守卫标量内核、粗解开销 3246→4ms、V-cycle 156→60ms。

剖析工具边界：nvprof 可用（权威）；torch.profiler/kineto 捕不到跨线程 C++ 内核；nsys 在 WSL2 失效。
