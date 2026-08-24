---
name: session-2026-08-24
description: logs/session-2026-08-24 的完整生成 skill：bug31–37 无人值守会话验证资产（8 脚本 + README，15/15 PASS；覆盖基线/求解器族/MPI/Wuppertal/stencil/Galerkin/等价性）与确定性参考数据再生器。
---
# CLAUDE.md — logs/session-2026-08-24

Verification assets from the unattended session that fixed bug31–bug37 (2026-08-24):
8 scripts + README, all 15/15 PASS. Coverage: baseline, solver family, MPI halo,
Wuppertal smear, stencil, Galerkin projection, dual-implementation equivalence.

## Assets

- `gen_*_ref.py` — deterministic generators rebuilding the gitignored reference HDF5 files
  in `examples/data/`; bit-level reproducible across architectures (solver dual-backend
  cross-check rel=8.6e-07). See the `data` skill.
- README documents session outcomes and operational lessons.

## Operational Lessons

- `mpirun ... python` requires ABSOLUTE script paths (OpenMPI cannot resolve relative
  paths on process side: "could not access").
- Scripts must stay idempotent / re-runnable.
