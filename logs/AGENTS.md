# AGENTS.md — logs

开发日志、审查报告、bug 修复摘要与求解器输出。顶层按文件类型保存
`.json`、`.jsonl`、`.log`、`.tsv`、`.txt` 等日志；里程碑报告
`.md`/`.tex`/`.pdf` 与产出一同按 tag 归档，不使用软链接。目录包括
`dev<N>/`、`stab<N>/`、`bug<N>/`（`logs/dev73/`、
`logs/dev73/stab24/`、`logs/dev74/`、`logs/bug30/`）。
`logs/<tag-name>/**`（stab**/dev**/test**/bug**）在 .gitignore 中全豁免入库；
`*.log`、`*.json` 等默认也受全局忽略规则保护，仅经复核的证据文件显式入库。

`logs/` 下新增的跨目录归档：

- `logs/data/`：从 `data/**` 提升出的 JSON、日志、TSV、TXT 与运行摘要；
- `logs/<tag>/**` 保留运行日志和图表；报告类实体的规范位置是
  `logs/<tag>/...`；张量类实体的规范位置是 `data/logs/<tag>/...`；
- 测试与脚本实体的规范位置是 `pyqcu/testing/**`；被现行实现取代的历史代码
  直接删除，不保留软链接；
- null-vector/粗算子缓存实体位于 `data/logs/nullvec_cache/`。

> 2026-09-27 迁移说明：历史日志中的 `examples/...` 保留为当时的原始记录，
> 其当前映射统一为 `pyqcu/testing/...`；新日志和新报告必须使用新路径。

## 文件模式（位于对应 tag 子目录内）

| 模式 | 用途 |
|---|---|
| `dev<N>.md` / `.tex` / `.pdf` | 开发里程碑报告（如 `dev73/dev73_5.md`；实体在 `logs/dev73/`） |
| `stab<N>.md` / `.tex` / `.pdf` | 稳定里程碑总结报告（如 `stab24/stab24.*`；实体在 `logs/stab24/`） |
| `bug<N>.md` | Bug 发现与代码审查报告（如 `bug30/bug30.md`；实体在 `logs/bug30/`） |
| `review-*.md` | 代码审查发现（如 `dev73/review-2026-07-28.md`；实体同路径映射到 `logs/`） |
| `fix-report-*.md` | Bug 修复摘要，实体位于 `logs/` |
| `mg-*-report-*.md` / `.tex` / `.pdf` | 多重网格开发报告（如 `dev73/mg-v4-report-2026-08-02.*`） |
| `multigrid_report.md` | MG 求解器性能报告 |
| `clover_multigrid.log` | C++ 求解器收敛输出（C++ 端相对路径写 `logs/clover_multigrid.log`） |
| `*.png` | 性能图表、收敛图 |

## 子目录

| 目录 | 用途 |
|---|---|
| `dev73/` | dev73/dev73_5 运行日志与图件；报告实体在 `logs/dev73/` |
| `dev74/` | dev74/dev74_1 运行日志与图件；报告实体在 `logs/dev74/` |
| `bug30/` | bug30 运行归档；报告实体在 `logs/bug30/` |
| `debug/` | 修复过程日志；Markdown 实体在 `logs/debug/` |
| `results/` | 最终/剩余修复运行归档；报告实体在 `logs/results/` |
| `../data/logs/nullvec_cache/` | 共享 null-vector/粗算子缓存实体 |

运行指南与脚本位置：dev73/dev74 套件脚本在 `pyqcu/testing/qcu/dev73/`、
`pyqcu/testing/qcu/dev74/`，产物写入本目录对应 tag 子目录。
