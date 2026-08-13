# AGENTS.md — logs

开发日志、审查报告、bug 修复摘要与求解器输出。里程碑报告与产物按 tag 归档于子目录
`dev<N>/`、`stab<N>/`、`bug<N>/`（`logs/dev73/`、`logs/dev73/stab24/`、`logs/dev74/`、
`logs/bug30/`）；`logs/` 根仅留共享缓存（`nullvec_cache/`）与通用报告
（`fix-report-*.md`）。`logs/<tag-name>/**`（stab**/dev**/test**/bug**）在 .gitignore 中全豁免入库；
`*.log`、`*.json` 等草稿输出仅在 tag 子目录内入库。

## 文件模式（位于对应 tag 子目录内）

| 模式 | 用途 |
|---|---|
| `dev<N>.md` / `.tex` / `.pdf` | 开发里程碑报告（如 `dev73/dev73_5.md`，含生成表 `dev73_5_tbl_*.tex` 与图 `dev73_5_*.png`） |
| `stab<N>.md` / `.tex` / `.pdf` | 稳定里程碑总结报告（如 `dev73/stab24.*` — dev73→dev73_5 MultiGrid 系列） |
| `bug<N>.md` | Bug 发现与代码审查报告 |
| `review-*.md` | 代码审查发现（如 `dev73/review-2026-07-28.md`） |
| `fix-report-*.md` | Bug 修复摘要（根目录留存） |
| `mg-*-report-*.md` / `.tex` / `.pdf` | 多重网格开发报告（如 `dev73/mg-v4-report-2026-08-02.*`） |
| `multigrid_report.md` | MG 求解器性能报告 |
| `clover_multigrid.log` | C++ 求解器收敛输出（C++ 端相对路径写 `logs/clover_multigrid.log`） |
| `*.png` | 性能图表、收敛图 |

## 子目录

| 目录 | 用途 |
|---|---|
| `dev73/` | dev73/dev73_5 里程碑报告、历史 mg_v4/mg-dev/multigrid 系列报告与图件（含 `dev73/stab24/` 子归档） |
| `dev74/` | dev74/dev74_1 里程碑报告、运行指南（`dev74_1_guide.md`/`.tex`/`.pdf`）与图件 |
| `bug30/` | bug30 报告 |
| `debug/` | 每轮修复日志（`fix-log*.md`） |
| `results/` | 最终/剩余修复报告 |
| `nullvec_cache/` | 粗算子缓存（共享，运行产物，不入库） |
| `test11/` | dev74* 整合测试套件（代码+产物同目录；历史版） |
| `test12/` | dev74* 整合测试套件（test11_1 优化版）：代码+脚本在根，**每次运行产物归档版本目录 `test12/v<ts>/`**（含 `env.json` 环境快照），跨环境可横向比对 |

运行指南与脚本位置：dev73/dev74 套件脚本在 `examples/qcu/dev73/`、`examples/qcu/dev74/`，
产物（json/tex/png）直接写入本目录对应 tag 子目录；test11/test12 套件自包含于
`logs/test11/`、`logs/test12/`（含运行脚本 run-local.sh / run-snsc.sh）。
