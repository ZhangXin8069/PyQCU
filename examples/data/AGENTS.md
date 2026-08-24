# AGENTS.md — examples/data

测试校验用参考 HDF5 文件。`with_data=True` 时用于对照预计算结果。

包含多种格点尺寸的规范场、费米子源与期望的算子/求解器输出。

## 现状（2026-08-24 二次更新：Wilson 组已跨后端重建）

原 L32 参考数据从未入库且来源不可考。现以 **C++ CUDA 后端为独立实现源**重建 Wilson 组
（L16³, mass=0⇒κ=0.125, c64, seed42；生成器 `logs/session-2026-08-24/gen_wilson_ref.py`）：
`refer.wilson.{U,src,b,x,dest}.L16K0_125.*.h5` 共 ~44MB（**因 .gitignore `*.h5`
不入库**，由归档生成器 `logs/session-2026-08-24/gen_wilson_ref.py` 确定性再生——
seed42 + C++ 确定性内核；clover 组同理由 gen_clover_ref.py 再生）。
测试分支尺寸同步改为 L16/L8。验证：dslash par 7.1e-08 / nopar 0.0 /
solver 跨后端 rel 8.6e-07（Python BiCGStab vs C++ BistabCg）。

clover 组（L32Y16K1）因原规格 GB 级未重建，`test_dslash_clover(with_data=True)`
仍不可用（如需可同法以小格子生成）。
