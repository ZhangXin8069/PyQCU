# PyQCU form 整理记录（2026-10-22）

状态：`finished`

## 目标

按 `~form` 流程对 PyQCU 做只读审计并整改格式问题，重点是 Git 标签名进入路径、
目录索引缺项和本地 form 规则与现状不一致三类问题。整改不改变任何求解器、内核或
Python 模块的行为。

## 审计范围与方法

| 项目 | 命令/依据 |
|---|---|
| 仓库判定 | `git rev-parse --show-toplevel`；库名 `PyQCU` 含大写 → 复杂库 |
| 工作树 | 开始时 `git status --short --branch` 干净，`main` 与 `origin/main` 同步 |
| 通用审计 | `bash /root/configure/skills/form/scripts/form-audit.sh --root /root/PyQCU` |
| 本地审计 | `bash skills/pyqcu-form/form-audit.sh` |
| 标签路径扫描 | `git ls-files \| grep -E '(^|/)(stab\|dev\|bug\|test)[0-9]+(_[0-9]+)*(...)'` |
| 符号扫描 | `git grep -n -E '\b(stab\|dev\|bug\|test)[0-9]+(_[0-9]+)*\b' -- '*.py' '*.pyx' '*.pxd' '*.cpp' '*.h' '*.cu'` |
| 目录/文件名审计 | `git ls-files -s -- refer`、`.gitmodules`、`git submodule status` |
| 根规则同步 | `form-agents-sync.sh --content ... --check` |

## 冲突表摘要

| 对象 | 类型 | 当前名称/位置 | 规则 | 冲突 | 处理 |
|---|---|---|---|---|---|
| `docs/progress_dev99_to_1640349d_20261021.md` | 文件 | 含标签名 `dev99` | 标签名不得作为路径 | 是 | rename 为 `progress_kupl_cuda_mg_20261021.md` |
| `refer/dev71.md`、`dev71.tex`、`dev71.pdf` | 文件 | 含标签名 `dev71` | 同上 | 是 | rename 为 `multigrid_analysis_20260717.*` |
| `AGENTS.md` 目录表 | 文档 | `refer/` 只写 `dev71.*` | 目录说明须与现状一致 | 是 | update |
| `refer/AGENTS.md` | 文档 | 只列 `dev71.*`，未列 `huzc/` | 目录索引完整 | 是 | update，补 5 行 |
| `docs/AGENTS.md` | 文档 | 只列基础指南，未列当前任务文档；与“不保留 tag 阶段报告”表述冲突 | 索引与生命周期说明一致 | 是 | update |
| `docs/progress_until_20261022_worktree.md` | 文档 | 引用旧文件名 | 引用须指向现名 | 是 | update |
| 根 `AGENTS.md` form 块 | 规则 | `refer/**` 一律保留来源名称；豁免计数 `1060` | 自产文档不得用标签名；计数须准确 | 是 | 经 `form-agents-sync.sh` 更新为 `refer/git-rep/**`、`refer/huzc/**`，计数 `1069` |
| `skills/pyqcu-form/SKILL.md` | 技能 | 例外写作 `refer/**` | 与根规则一致 | 是 | update |
| 代码符号（`dev76`/`dev84`/`dev87` 等） | 注释 | 49 个文件中的版本注释 | 注释/日志允许作版本证据 | 否 | keep |
| `logs/**` 中对旧文件名的引用 | 历史日志 | 11 个日志文件 | 历史日志保留既往证据 | 否（例外） | keep |
| `refer/git-rep/*` | submodule | 5 个子库均为 `160000` gitlink | 必须 submodule + README 来源 | 否 | keep |
| `AGENT.md`（单数） | 规则文件 | 全库无 | 统一 `AGENTS.md` | 否 | keep |

## 改动

1. 重命名并保留 Git 历史（`git mv`）：
   - `docs/progress_dev99_to_1640349d_20261021.md` → `docs/progress_kupl_cuda_mg_20261021.md`；
   - `refer/dev71.{md,tex,pdf}` → `refer/multigrid_analysis_20260717.{md,tex,pdf}`。
2. 同步现行引用：根 `AGENTS.md` 目录表、`refer/AGENTS.md`、`docs/AGENTS.md`、
   `docs/progress_until_20261022_worktree.md`。
3. `refer/AGENTS.md` 补齐外部参考脚本 `refer/huzc/{main_pion_C2.py,quda_mg.py}`。
4. 根 `AGENTS.md` 的 form 托管块经 `form-agents-sync.sh` 更新：`refer/` 例外收窄为
   `refer/git-rep/**`、`refer/huzc/**`，历史豁免计数由 `1060` 校正为 `1069`。
5. `skills/pyqcu-form/SKILL.md` 同步同一例外口径。

## 验证证据

| 检查 | 结果 |
|---|---|
| `bash skills/pyqcu-form/form-audit.sh` | `PASS`：33 个技能目录与索引一致，豁免 1069 项既有 finding |
| `git ls-files` 标签名路径扫描 | 仅剩 `logs/**` 内的历史引用（允许） |
| 旧名称全库搜索（排除 `logs/`、`data/`） | 0 命中 |
| `form-agents-sync.sh --check` | 退出码 0 |
| `bash -n build.sh env.sh install.sh ...` | 通过 |
| `bash skills/pyqcu-tag/tag-chain.test.sh` | 通过（临时仓库，不触碰本库标签） |
| `git diff --check` | 通过 |
| 全量 `pytest` | 未运行：需要 CUDA/MPI/QUDA 与大体量数据；本次无代码改动 |

## 未决项

- `docs/progress_kupl_cuda_mg_20261021.md` 已包含在 `dev101` 中；按 `docs/ORGANIZATION.md`
  的 tag 报告生命周期，可在后续清理批次删除工作区副本，本次为保留内容可读性未删除。
- `logs/**` 中的旧路径与旧标签引用属于既往证据，不追改。

## 恢复方法

```bash
git log --follow -- docs/progress_kupl_cuda_mg_20261021.md
git log --follow -- refer/multigrid_analysis_20260717.md
# 需要旧路径内容时
git show <tag>:docs/progress_dev99_to_1640349d_20261021.md
git show dev101:refer/dev71.md
```

全部整改可在 Git 历史中回溯；重命名未改写对象内容（`git mv` 保留相似度）。
