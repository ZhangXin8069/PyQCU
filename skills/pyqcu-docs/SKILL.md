---
name: pyqcu-docs
description: |
  当需要定位 PyQCU 的权威项目文档、把 docs/ 内容提取或同步到 skills/、发布或排查
  Read the Docs 中英双语站点（.readthedocs.yaml、Sphinx -W、docs_examples 围栏门）、
  查询安装/环境/维度/测试入口、判断正式文档与阶段报告的生命周期，或追踪 CUDA、
  MultiGrid、KUPL、benchmark 与 profiler 知识来源时使用。
metadata:
  openclaw:
    emoji: 🗂️
---

# pyqcu-docs — 项目文档导航与技能化提取

## 执行前置

先读取 `../../docs/AGENTS.md` 和 `../../docs/ORGANIZATION.md`；它们是文档索引与
目录生命周期的权威来源。涉及具体代码域时，再读取对应目录的 `AGENTS.md` 与技能。
碰到 Read the Docs、`docs/en`/`docs/zh` 双语站点、Sphinx 构建或发布文档门时，
先读 [`references/readthedocs-release.md`](references/readthedocs-release.md)。

## 权威边界

- `docs/**` 是正式独立文档的规范源，负责完整说明、公式、报告与版本证据。
- `skills/**` 是 agent 的按需加载层；只沉淀稳定不变量、命令、反模式和来源映射，
  不复制整篇正式文档，避免双源漂移。
- `logs/**` 只承载当前任务尚未归档的证据；已归档阶段报告通过
  `git show <tag>:<path>` 读取，不把 `logs/<tag>/**` 写成现行路径。
- 带“工作区结果”“尚未提交/打 tag”或 `userinterlude` 状态的报告属于时点证据；
  引用时必须保留窗口、算法、硬件、精度与计时边界，不得提升为永久性能承诺。

完整的 45 份现行 Markdown 文档职责、权威级别和目标技能见
[`references/project-docs-map.md`](references/project-docs-map.md)。需要立即复用安装、
维度、测试、CUDA 拓扑、MultiGrid、性能或 profiler 结论时，读取
[`references/extracted-guidance.md`](references/extracted-guidance.md)。
需要逐份核对全部正式文档的结论是否有技能归宿时，直接读取该文件的
「全部正式文档的结论覆盖索引」。

## Read the Docs 双语站点

`docs/en` 与 `docs/zh` 是两套独立 Sphinx 源，发布到 RTD 项目 `pyqcu`
（`https://pyqcu.readthedocs.io/`）。硬约束：

- slug 是 `pyqcu`，不是 `py-qcu`。PyQCD 用 `py-qcd` 只因 `pyqcd` 被他人占用，是
  那一次的历史替代，不是可推广的命名规则。
- 只有两份 per-language 配置：`docs/en/.readthedocs.yaml` 与
  `docs/zh/.readthedocs.yaml`；仓库根不设 `.readthedocs.yaml`（沿用 PyQCD 约定），
  两份配置在 RTD 网页按语言分别指定。
- `docs/.gitignore` 必须用 `!en/`、`!en/**`、`!zh/`、`!zh/**` 保住源目录；用
  `**/` 把 en/zh 整个忽略，RTD 看到的就是一个没有文档的仓库。
- 每侧 `index.rst` + `overview.md`/`usage.md`/`solvers.md`/`performance.md`/`install.md`/`troubleshooting.md`，正文是
  MyST；`conf.py` 不 import `pyqcu`；zh 侧 `language = "zh_CN"`（Sphinx locale）与
  站点 URL 的 `zh-cn`（RTD 翻译 slug）是两个名字，不能互换。
- 构建门（四条都要过）：

```bash
python -m sphinx -b html -d data/release_check/doctrees-en -W docs/en data/release_check/sphinx-en
python -m sphinx -b html -d data/release_check/doctrees-zh -W docs/zh data/release_check/sphinx-zh
python -m pyqcu.testing.docs_examples --keep-going
python -m pyqcu.testing.release_check --offline   # 已包含双站 -W
```

- 中英围栏块逐字节一致（`python` 与 `bash` 都比）；每个 python 块要有 `assert` 或
  `# docs-skip`/`# pseudo-code`；加入 `solvers.md` 与 `troubleshooting.md` 后
  基线 30 ok / 8 skipped / 0 failed。
- 版本同步点：`docs/{en,zh}/conf.py` 的 `release`、`install.md` 的 `pyqcu==<ver>`
  和两处版本断言、`overview.md` 的版本号随发布版本一起改；漏改 `install.md` 会让
  `docs_examples` 直接失败，不是外观问题。
- RTD 现状：项目 `pyqcu` 已存在（name PyQCU，默认版本 latest），跟踪
  `https://github.com/ZhangXin8069/PyQCU.git`；GitHub 是 Gitee 的自动镜像，推 Gitee
  后 GitHub 自动更新并触发 RTD。成功构建 `duration` 约 35 s；`duration=0` 通常表示
  tracked revision 上没有配置文件，不是 Sphinx 错误。无 RTD token 时不能从 CLI
  手动触发，但推 Gitee 的自动构建不需要 token。
- 判定上线要按页检查 `<title>` 与章节名：重建有分钟级延迟，不同页可能停在不同
  版本；本地构建通过不等于站点已发布。

逐项命令、失败判读、内容边界与发布顺序见
[`references/readthedocs-release.md`](references/readthedocs-release.md)。

## 工作流程

1. 盘点 `docs/**`，把每份文件标为规范指南、当前技术参考、设计提案、时点证据或历史记录。
2. 按 `references/project-docs-map.md` 找到唯一主责技能；跨域内容只建立交叉指针。
3. 提取可执行命令、稳定不变量、默认路径、失败边界和证据口径。历史数字只有在
   输入、实现、硬件和计时边界可比时才进入技能；否则保留来源链接和时间状态。
4. 在目标 `SKILL.md` 加最小上下文指针；长内容写入该技能的 `references/`，
   `SKILL.md` 保持可扫描。
5. 只有新增或删除技能目录时才更新 `skills/AGENTS.md` 技能表与计数；技能内新增
   reference 不改计数。随后验证 frontmatter、本地链接、索引和 `git diff --check`。
6. 运行 `python -B skills/pyqcu-docs/scripts/audit-docs-skills.py`；任何 docs 未映射、
   技能 frontmatter/触发描述、断链、围栏或按需文件指针问题都必须在交付前清零。

完成判据：`docs/**` 每份文件都有明确归宿；新增知识有 `docs/...:line` 来源；
同一规则只有一个权威位置；目标技能能从真实用户触发词进入该参考；专项审计器退出 `0`。
触及双语站点时，双站 `-W`、`docs_examples` 与 `release_check --offline` 必须一起通过。

## 提取规则

- 优先提取会影响未来实现或验收的内容：数据布局、ABI、生命周期、算法默认值、
  正确性门、复现命令和已知反模式。
- 不把完整进度表、失败作业长表或历史报告全文搬入技能；保留结论、证据入口和
  “何时回读原文”的条件。
- 同一事实若已在域技能中完整表达，只补来源映射，不重复正文。
- 文档中的命令必须能从仓库根执行，或在同一段落写明 `cd` 前置目录。
- 报告状态变化后，先更新规范文档和来源映射，再更新技能摘要；不得只改技能数字。

## 错误处理

| 场景 | 处理 |
|---|---|
| `docs/**` 文件名含 Git 标签形片段 | 登记为 form finding；不在技能中复制该路径制度，按 form 决定可恢复重命名 |
| 文档状态为工作区结果或未打 tag | 只作为时点证据引用，并写明状态与日期 |
| 同一内容存在于多个技能 | 选定一个权威技能，其余改为相对链接，避免双重维护 |
| 报告数字缺少统一口径 | 不比较绝对倍率，只记录各自计时/算法/硬件边界 |
| 文档路径失效 | 先查 `docs/AGENTS.md`、Git 历史和 tag，不猜测替代路径 |
| 技能索引与目录不一致 | 仅当技能目录增减时同步 `skills/AGENTS.md` 的目录行和 `N/N` 计数后重跑审计 |
| `audit-docs-skills.py` 失败 | 按 `DOCS-*`、`SKILL-*`、`DESCRIPTION`、`POINTER` 错误码修复正文或索引；不得删除检查项绕过失败 |
| RTD 构建 `success=false`、`duration=0` | 先确认 tracked revision 上有 `.readthedocs.yaml` 且 `docs/.gitignore` 没忽略 `en/`/`zh/`；不要先怀疑 Sphinx |
| `docs_examples` 报 `no assertion` 或版本断言失败 | 补 `assert`/`# docs-skip` 说明原因；版本断言失败按 §版本同步点两侧同改 |
| 中英围栏不一致 | 围栏内注释和命令也算字节；只翻译围栏外文字 |
| 线上页面 404 或版本混杂 | 按页重探 `<title>` 与章节名；构建有分钟级延迟，等待后重试 |

## 注意事项

- `pyqcu-docs` 负责导航和提取规则，不替代 `pyqcu-form` 的目录整改或域技能的技术裁决。
- 正式算法结论优先回到 `qcu_quda_*` 文档与源码；正式性能结论优先回到同窗口报告。
- 技能摘要面向执行，不应包含无法验证的推断、凭据或外部绝对路径。
- RTD 网页侧动作（per-language 配置文件、`zh-cn` 翻译、构建触发）无法从仓库 CLI
  复核；交付时明确列为外部步骤，不能写成已完成。
