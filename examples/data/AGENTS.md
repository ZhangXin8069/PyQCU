# AGENTS.md — examples/data

测试校验用参考 HDF5 文件。`with_data=True` 时用于对照预计算结果。

包含多种格点尺寸的规范场、费米子源与期望的算子/求解器输出。

## 现状（2026-08-24）

**当前目录为空（仅本文件）**：`refer.wilson.*.L32K0_125.*.h5`、`refer.clover.*.L32Y16K1.*.h5`
从未入 git（`git ls-files` 仅跟踪本文件），仓库内无副本、无生成脚本，来源不可考。
影响：`pyqcu/testing` 的 `test_dslash_wilson(with_data=True)`、`test_dslash_clover(with_data=True)`、
`test_solver(with_data=True)` 路径因文件缺失报 OSError——属环境资产缺失，非代码缺陷；
核心算子正确性由非 data 测试路径覆盖。若重建参考数据，须以独立实现生成并入库。
