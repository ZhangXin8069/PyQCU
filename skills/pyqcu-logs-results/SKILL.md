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

## 结论作废与口径纠正（2026-11-15）

权威报告的价值一半在"哪些旧结论不再成立"。作废一条结论必须同时给三样：
**作废对象、机制、替代口径**（只写"已修正"会让下游继续引用旧数字）。

- 已作废：`atol=1e-4` 下的 `0.128289 s`（`restart=1,cmax=200,tol=1e5`）与
  `0.135579–0.136258 s` 曾被写成"MG 比 BiCGStab 快 26%/21.8%"。机制是
  `coarse_atol = atol × coarse_tol_factor × 3^(levels−1)`，`1e-4 × 1e4 = 1.0`
  让粗层首行即"收敛"、修正恒零并 `fallback_to_fine=true`——实际是 fine
  BiCGStab。替代口径：真实 MG 必须 `atol=1e-6`，且同格点对照下 MG 与 BiCGStab
  在 ±2% 内、端到端因 cache 载入慢 `10–19×`。
- 已作废："多 rank Legacy-MG 无法运行"。机制是 patch gauge 取自 rank 局部 slab
  （4-rank 有 75% 站点错误），修成全局 patch 后 32 rank 可构建
  （L24 `24.476 s`/L32 `53.023 s`）。替代边界：仍须预置全局 `--coarse-cache`。
- 已作废（2026-11-15 鲲鹏矩阵）：
  - "L32 不可跑/算法有问题"——6 条 FAIL 全在 `source` 读取的
    `OSError: Unable to open file (bad object header version number)` 处退出，
    solver 未启动；机制是输入 HDF5 被非原子写损坏。替代口径：缺**一次有效运行**，
    修复 ones（`9cdcfc82…f55a`）后重跑，旧 FAIL 行原样保留。
  - "MG + 共享 halo 不可用"——只有 32 秩 balanced `2 2 2 4` + `shm=1` 触发
    `KUPL shared halo handshake timeout: publish`（非对称调用序列），t-only 的
    32/16 秩均可跑。
  - 口径裂缝：600 s 硬超时的 cache 行 `rc=124` 却 `status=PASS`（cache 与 manifest
    已落盘）。替代口径：这类行只算 **partial 输出**，不能当"该格点可跑"或 cache
    构建成功；solve 行没有 `reports/<run_id>.json` 一律 `FAIL`。
- 引用禁令：放弃轴（thread 9、seed24、random gauge、random 源、point 源、r50）的
  单样本（`861ac912ac3c7c42` 的 `11.437897 s` 等）不得进入均值、加速比或"最优"；
  已有性能数据保留但标注不变，`n=1` 行必须显式写 `n=1`。
- 标注义务：每条性能数字带 `n`（本批大量 `n=1`）、节点名、`atol`、源类型、
  grid/halo 路径；`FAIL`/`TIMEOUT`/被中断的行**不得**进入均值、加速比或"最优"。
- 复测优先：同一 `run_id` 的复测可差 `36×`（`builder` `72.1/78.5/5.64 s`），
  出现离群值时先查 `*_read_mode`/`*_read_seconds` 与作业资源模型，再决定是否作废。
- 台账级剩余问题以严重度清单交付（P1 阻断结论 / P2 可能误读 / P3 可追溯性 /
  P4 整洁），每条带行号或作业号证据与可执行修正；修复与否在下一轮结果里闭环，
  不改历史报告的原始数字。
