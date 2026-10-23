---
name: pyqcu-logs-results
description: 当任务需要读取或更新 logs/results 的最终与剩余修复报告时使用。
---
# logs/results

Final and remaining fix reports. These are the polished, summary versions of fix reports after debug resolution.

## Purpose

When a bug-fixing session completes, the final summary report is written here. These are the authoritative record of what was fixed, what remains, and what was skipped.

## 本目录实际内容（2026-10-15 核对）

| 文件 | 内容（数字取自文件自述，勿跨文件合并计数） |
|---|---|
| `final-report.md` | 2026-07-28 会话最终报告：审查 23 Python / 20+ C++ / 3 Cython 文件，报告 **74 findings**、**14 项 PASS**（判为设计决策）；落地 **12 项 bug 修复** + 1 项性能优化（移除 24 处冗余 `MPI.Barrier`）；8/8 测试通过 |
| `remaining-fixes-report.md` | 2026-07-28 的 **R2 剩余高优先级项**，来源 `git show dev73:logs/dev73/review-2026-07-28-r2.md`；本轮 11 items |

**易错点**：`fix-report-2026-07-28.md` 与 `fix-report-2026-08-24.md` **在仓库根 `logs/`，
不在本目录**——本目录只有上面两份。

- `logs/fix-report-2026-07-28.md`：同一 07-28 会话的修复明细表（61 项 actionable：
  25 已修复 / 15 已记录 / 21 需重构），与 `final-report.md` 的 74 findings 是**两套口径**
  （findings 是审查条目，61 是可执行项），不要相加或互换。
- `logs/fix-report-2026-08-24.md`：无人值守 auto 会话（基线 `56e6ee6 (dev84_2_2)` →
  终态 `1025a90`），标签 bug31–bug36、test16、dev85、bug37；含 bug33/34/35 的定量闭环
  数字（Wuppertal 不动点 44.4→6.75e-08、C9 等价性 1e-06、Galerkin 5.6e-07）。

## 当前 MultiGrid 收尾证据

- `stab62` 最终日志：`logs/v2026102312.txt`；`stab62_1` 经验日志：
  `logs/v202610232206.txt`；机器汇总：
  `logs/qcu_l32x64_mpi_20261023/summary.json`。
- QUDA MG 未完成原因：`logs/qcu_l32x64_mpi_20261023/quda-multigrid-status.json`。
- 单组达到 `1000 s` 时记录 `status=SKIPPED`、
  `reason=elapsed_time_unacceptable`、timeout、退出码和最后完成阶段；不得写成数值失败。
- `logs/**` 中的命令退出码文件必须写成 `*.exit_code.txt`，裸
  `*.exit_code` 不符合允许扩展名。
