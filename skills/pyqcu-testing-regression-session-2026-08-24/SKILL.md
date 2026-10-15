---
name: pyqcu-testing-regression-session-2026-08-24
description: pyqcu/testing/regression/session-2026-08-24 的完整生成 skill：bug31–37 无人值守会话验证资产与确定性参考数据再生器。
---
# pyqcu/testing/regression/session-2026-08-24

Verification assets from the unattended session that fixed bug31–bug37 (2026-08-24).
**15 个脚本 + `__init__.py`**，全量关门回归 **15/15 PASS**。Coverage: baseline, solver family,
MPI halo, Wuppertal smear, stencil, Galerkin projection, dual-implementation equivalence，
以及 README 未列表的补充面（NPU 模拟、分层显存、数据源、预算/环境、Schur）。

## Assets

2026-10-15 核对，`ls pyqcu/testing/regression/session-2026-08-24/` 即得：

| 组 | 脚本 |
|---|---|
| README 表格覆盖的 8 项核心 | `r1_baseline.py`、`c2_solver_smoke.py`、`c4_mpi_smoke.py`、`c5_final.py`、`c6_gold.py`、`c7_schur_smoke.py`、`c8_galerkin.py`、`c9_equiv.py` |
| README 未列表的补充脚本 | `c10_npu_emu.py`、`c11_hier.py`、`c12_source.py`、`c13_budget_env.py`、`c14_schur.py` |
| 确定性生成器 | `gen_clover_ref.py`、`gen_wilson_ref.py` — 重建 `pyqcu/testing/data/` 中 gitignored 的参考 HDF5；跨架构位级可复现（求解器双后端交叉校验 rel=8.6e-07）。见 `skills/pyqcu-testing-data/SKILL.md` |

README **不在本目录**，在仓库根的 **`logs/session-2026-08-24/README.md`**：逐脚本的验证内容 / 预期
结果表、运行方式，以及下面的操作性教训。**以该 README 为准**取逐项数字（如 `c2` 7/7 PASS、
`c4` 在 `mpirun -np 2` 与 `-np 4` 均 4/4 PASS，`c6` np=2 rel≈5.1e-08 / np=4 ≈3.9e-08）。

数字口径：README 自述「8 个回归验证脚本」指它表格里的 8 项核心；`15/15 PASS` 是含生成器与全部
补充脚本的全量关门计数。两者不矛盾，勿据其一改删其一。

## Operational Lessons

- `mpirun ... python` requires ABSOLUTE script paths (OpenMPI cannot resolve relative
  paths on process side: "could not access").
- Scripts must stay idempotent / re-runnable.
