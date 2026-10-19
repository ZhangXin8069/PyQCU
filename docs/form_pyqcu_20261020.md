# PyQCU form 整理记录（2026-10-20）

## 目标

将根 `AGENTS.md` 的 form 格式约定固化到唯一的 `form-skill` 托管标记块，保留全部
本地规则和历史例外，并补充本地 `pyqcu-form` 技能中标记块、审计例外和历史资产均不得
被绕过的准确说明。本任务不修改代码，不删除或迁移 `data/**` 等历史资产，不执行 Git
提交、推送或标签操作。

## 改动

- 使用 `/root/configure/skills/form/scripts/form-agents-sync.sh` 将根 `AGENTS.md`
  的 form 规则同步为唯一托管块；块外原有架构、反模式和其他本地规则保持不变。
- 托管块完整保留原有命名、目录、测试、文档、日志、数据和 Git 交付规则，并明确：
  - 提交前执行默认上限为 `10,000,000` 字节的大文件闸门；
  - 主测试、MPI 测试和技能内 shell 回归入口；
  - 仅暂存本任务文件、禁止 `git add -A`/`git add .` 和 force push；
  - 当前审计包装器登记并豁免的 `1044` 项既有资产 finding 不得通过删除、迁移、
    整体禁用检查或改写排除项绕过；
  - 根规则原有明确约定优先于 form 通用规则。
- 在 `skills/pyqcu-form/SKILL.md` 补充托管块的唯一维护方式、`--check` 门禁，以及
  历史例外不得绕过和新增例外必须具备精确匹配与回归证据的要求。

## 验证

| 检查 | 结果 |
|---|---|
| `bash skills/pyqcu-form/form-audit.sh` | PASS；`32/32` 个技能目录与索引一致，豁免 `1044` 个既有资产 finding |
| `form-agents-sync.sh --check` | PASS；`AGENTS.md` 已是最新，托管标记起止各 1 个 |
| `bash -n build.sh env.sh install.sh skills/pyqcu-tag/tag-chain.sh skills/pyqcu-tag/tag-chain.test.sh` | PASS；退出码 0 |
| `bash skills/pyqcu-tag/tag-chain.test.sh` | PASS；自检、dry-run、本地修正和远端 lease 重写测试通过 |
| `form-snapshot-verify.sh --quiet` | PASS；退出码 0 |
| `git diff --check` | PASS；无空白错误 |

## 例外与终态

`data/**`、tag 前 `logs/**` 报告、解包 Office 文档资产、技能内定点回归和上游
vendored/generated 内容继续按既有边界保留。本轮没有新增或放宽审计例外。

## 标签交付门

主线程在分支交付前执行全历史标签链检查，发现 `dev37 -> dev38` 与
`dev72 -> bug30` 的 peeled commit 不满足祖先关系，无法只通过修正注释安全恢复。
按 `tag` 技能规则不自动改写已推送历史，因此本轮不创建 `dev97`；文件整改和分支
提交不受影响，标签问题保留为人工处置项。

终态：文件整理完成；标签交付 blocked。改动由主线程逐文件暂存、提交并普通推送。
