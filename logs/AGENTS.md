# AGENTS.md — logs

开发日志、审查报告、bug 修复总结与求解器输出。里程碑报告（`dev*.md`/`.tex`/`.pdf`）、修复报告及其图件版本化；临时输出（`*.log`、`*.json`、`*.aux`）gitignore。

## 文件模式

| 模式 | 用途 |
|---|---|
| `dev<N>.md` / `.tex` / `.pdf` | 开发里程碑报告（如 `dev73_5.md` + 生成的表 `dev73_5_tbl_*.tex` 与图 `dev73_5_*.png`） |
| `stab<N>.md` / `.tex` / `.pdf` | 稳定里程碑总结报告（如 `stab24.*` — dev73→dev73_5 MultiGrid 系列） |
| `bug<N>.md` | bug 发现与代码审查报告 |
| `review-*.md` | 代码审查发现（如 `review-2026-07-28.md`） |
| `fix-report-*.md` | bug 修复总结 |
| `mg-*-report-*.md` / `.tex` / `.pdf` | 多重网格开发报告（如 `mg-v4-report-2026-08-02.*`） |
| `multigrid_report.md` | MG 求解器性能报告 |
| `clover_multigrid.log` | C++ 求解器收敛输出 |
| `*.png` | 性能图、收敛图 |

## 子目录

| 目录 | 用途 |
|---|---|
| `debug/` | 逐轮修复日志（`fix-log*.md`） |
| `results/` | 最终/剩余修复报告（README + dev73_5 报告）；生成报告时须同步更新 README |

## Skills

`skills/` 下子目录的完整生成 skill（原 CLAUDE.md 内容）：`debug`、`results`。
