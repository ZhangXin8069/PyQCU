---
name: pyqcu-testing-data
description: 当任务涉及 pyqcu/testing/data 的 HDF5 参考数据或 with_data 测试时使用。
---
# pyqcu/testing/data

Reference HDF5 files for test validation. Used when tests are run with `with_data=True` to validate against precomputed expected results.

Contains gauge fields, fermion sources, and expected operator/solver outputs for various lattice sizes.

Current end-to-end correctness defaults to `8x8x8x16`. The L16/L8 Wilson and
Clover files here are historical fixtures. `with_data=True` must receive an explicit
historical lattice and must not silently replace the current correctness lattice.

Reference HDF5 files are gitignored (not committed)。**注意：这些 HDF5 缺失时
`with_data=True` 的用例会在 OSError 层失败，属于已知且预期的情况**（例如 `logs/session-2026-08-24/README.md`
里 `r1_baseline.py` 记的 8/10 PASS、2 FAIL 就是这一层），不要当成代码回归。

Regenerate deterministically via `pyqcu/testing/regression/session-2026-08-24/gen_clover_ref.py`
and `gen_wilson_ref.py` — bit-level reproducible across architectures；solver dual-backend
cross-check rel=8.6e-07。Agent 技能见
`skills/pyqcu-testing-regression-session-2026-08-24/SKILL.md`（生成器的逐项说明与 README 位置在该
文档，此处不重复）。
