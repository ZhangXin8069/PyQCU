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
