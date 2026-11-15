# AGENTS.md — pyqcu-kupl

独立 MPI+OpenMP CPU QCU 后端及 `pyqcu.kupl` ctypes 绑定技能。

- 规范全文：`SKILL.md`
- 核心范围：canonical flat ABI、csw、Clover BiStabCG、NUMA/MPI 绑定、矩阵/CSV/调度、重测判定
- 矩阵（94 列 CSV、作业模型、cache/重测判定、runbook）单一保存在
  `references/kupl-kunpeng-matrix-20261115.md`；**不新设 `pyqcu-kupl-kunpeng-matrix`
  独立技能**（同一内容会形成双源）。历史数字读 `references/kupl-history-2026.md`，
  与 CUDA 的取向差异读 `references/kupl-vs-cuda-algorithms.md`。
- 更新内容时同步库级 `../AGENTS.md` 技能表；不自动提交或推送
- 范围裁定（MPI h5py 接口检查、放弃轴、L32 source/cache 重签、队列）见 [`docs/kupl_matrix_scope_20261115.md`](../../docs/kupl_matrix_scope_20261115.md)。
