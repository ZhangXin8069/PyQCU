# PyQCU form 格式审计（2026-09-28）

## 目标与边界

本次审计以 `d3072da` 为 Git 基线，范围覆盖已跟踪路径、技能索引、测试资产、
文档/日志/数据边界和本地 form 特化。PyQCU 按库名判定为复杂库，主导语言为
C++/CUDA、Python/Cython 和 Bash。

接受标准：

- 未登记的真实格式冲突清零；
- 已存在的正式资产例外有可执行、可追溯的本地规则；
- 技能目录数与技能表一致；
- shell 语法、tag 回归、本地审计和 `git diff --check` 通过；
- 删除项可由 Git 基线恢复。

## 冲突表摘要

| 对象 | 当前状态 | 规则依据 | 处理 |
|---|---|---|---|
| `pyqcu/testing/python/convergence_history copy*.png` | 4 个未引用、同批引入的临时图，文件名含 `copy` 和空格 | 测试目录不承载一次性图产物 | 删除；可由 `d3072da` 恢复 |
| `skills/tag/tag-chain.test.sh` | 位于被测 `tag-chain.sh` 同目录 | 技能自带定点回归应与其实现共存 | 保留并登记具名例外 |
| `data/**` | 422 个已跟踪证据、数据和自包含构建/安装文件 | `docs/ORGANIZATION.md` 明确允许 | 保留；本地审计按例外过滤 |
| `logs/**` 报告与摘要 | 含 `.md/.tex/.pdf`、图片、`.stdout/.jsonl` 等 | 第一方 tag 报告和运行记录统一存放于 `logs/**` | 保留；本地审计按例外过滤 |
| `docs/张鑫 508应用测试报告-PyQCU/` | 解包 Office XML/媒体资源 | `docs/AGENTS.md` 明确按文档资产保留 | 保留并登记例外 |
| `skills/AGENTS.md` | 实际 35 行技能表却标为 42/42 | 技能表必须与实际目录同步 | 修正为 36/36，并新增 `form` |
| `skills/form/**` | 本地 form 特化缺失 | 复杂库必须提供本地格式规则和验证入口 | 新增 `SKILL.md`、`AGENTS.md`、`form-audit.sh` |
| 根隐藏 agent 日志与缓存 | 未跟踪且已忽略 | 不可由 Git 恢复，未获删除授权 | 保留，不纳入提交 |

## 改动

- 根 `AGENTS.md` 新增“form 格式约定”，记录语言命名、白名单、测试/构建入口、
  文档/日志/数据职责、Git 交付和本地例外。
- `skills/AGENTS.md` 修正技能表计数并加入 `form` 索引。
- 新增 `skills/form/SKILL.md` 和 `skills/form/AGENTS.md`，固化 PyQCU 特化流程。
- 新增 `skills/form/form-audit.sh`，包装通用审计器并仅过滤已登记的本地资产例外。
- 删除 4 个未引用且不可从当前工作树恢复以外再生的临时收敛图。

## 验证证据

以下命令在本轮整改后执行：

```bash
bash skills/form/form-audit.sh
bash -n build.sh env.sh install.sh skills/tag/tag-chain.sh \
  skills/tag/tag-chain.test.sh skills/form/form-audit.sh
bash skills/tag/tag-chain.test.sh
git diff --check
```

结果：

- 本地 form 审计：`PASS`，过滤 744 个已登记既有资产 finding，未登记项为 0；
- shell 语法：退出码 0；
- tag 历史链、dry-run、本地修正和远端 lease 测试：`PASS`；
- 技能目录 36、技能表行 36；现有 `convergence_history copy*.png` 资产路径为 0
  （本审计文档中的历史记录除外）；
- 未跟踪非忽略项仅包含本轮新增的 `docs/form-audit-20260928.md` 和 `skills/form/**`。

全量 pytest 未运行：本次仅修改文档、技能和审计脚本，未修改运行时 Python/C++ 代码。
该缺口不影响上述格式与回归验收，后续功能变更仍须按根 `AGENTS.md` 的测试入口执行。

## 恢复方法

- 恢复删除的图：`git restore --source=d3072da -- <path>`。
- 恢复文档与技能索引：`git restore --source=d3072da -- AGENTS.md skills/AGENTS.md`。
- 撤销本轮新增技能：删除 `skills/form/`，并恢复 `skills/AGENTS.md` 的旧计数。
- 本地审计误报时，先在 `skills/form/form-audit.sh` 增加精确路径规则并执行 shell 语法测试；
  不整体禁用通用审计。
