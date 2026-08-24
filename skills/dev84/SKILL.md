---
name: dev84
description: examples/qcu/dev84 目录的完整生成 skill：16×32×32×48 MultiGrid 真实加速比攻坚套件（CUDA Graph 段回放/零拷贝标量/守卫标量内核/粗空间诊断 ρ_V），报告 dev84_report.md。
---
# CLAUDE.md — examples/qcu/dev84

Current multigrid real-speedup campaign on the unified lattice 16×32×32×48.
Report: `examples/qcu/dev84/dev84_report.md`; outputs mirror `out/*.json` and `logs/dev84/`.

## Entry Point

```bash
python examples/qcu/dev84/main.py {run|multi|run_gcr|hotspot} [--only ...]
```

## Key Results (V100)

- Coarse space diagnostic ρ_V = 0.9759 (continuum-like spectrum, no isolated low-mode cluster)
  → MG>2 target unreachable on this unified lattice; volume scaling 1.5× costs only 0.421×
  runtime ("bigger lattice helps" falsified).
- Net optimizations still put MG stably above BiStabCG for the first time: 1.13–1.16×;
  adaptive-correction gating lowers MG_2L by another −18% (to 0.798×).
- Mechanisms: CUDA Graph segment replay (8 iterations/segment), zero-copy mapped-memory
  scalars, guarded scalar kernels, coarse-solve overhead 3246→4 ms (~800×),
  V-cycle 156→60 ms (2.6×).

## Profiling Tool Boundaries

nvprof works (authoritative); torch.profiler/kineto cannot capture cross-thread C++ kernels;
nsys fails under WSL2.
