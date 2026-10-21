---
name: pyqcu-form
description: |
  当用户要求审计或整改 PyQCU 的命名、目录结构、代码框架、文档、日志、数据、
  测试布局、仓库清洁度与 Git 交付格式时使用；入口为 `~form` 或 `form`。
metadata:
  openclaw:
    emoji: 📐
---

# form — PyQCU 格式治理

## 执行前置

遵循根 `AGENTS.md` 的“form 格式约定”和 `docs/ORGANIZATION.md`。通用命名矩阵读取
`${CONFIGURE_ROOT:-$HOME/configure}/skills/form/references/naming-and-layout.md`；
本库的目录白名单、已知例外、验证入口和交付边界以根规则为准。

PyQCU 是复杂库，主导语言为 C++/CUDA、Python/Cython 和 Bash。先只读审计，再按依赖顺序
分批整改；不得把历史报告、正式证据、自包含构建树或上游 vendored 内容机械迁出。

## 本地结构规则

- 顶层目录只允许 `cpp/`、`data/`、`docs/`、`logs/`、`pyqcu/`、`refer/`、`skills/`。
- 根文件只允许 `AGENTS.md`、`.gitignore`、`LICENSE`、`README.md`、`build.sh`、
  `env.sh`、`install.sh`、`setup.py`。
- **Agent 规则文件名称（硬性）**：所有目录的 agent 指令文件统一使用 `AGENTS.md`；`AGENTS.md` 是唯一名称，`AGENT.md` 是历史输入错误，不是兼容别名。只存在单数文件时直接重命名为 `AGENTS.md`；同目录双文件并存时，以 `AGENTS.md` 为主文件，完整合并并去重 `AGENT.md` 的有效内容后删除 `AGENT.md`，并同步现行脚本、配置、文档和本地规则中的全部引用。历史日志正文可保留旧路径作为既往证据。
- C++ 头文件和 `.cu` 文件使用小写下划线；类型使用大驼峰，宏使用全大写下划线；
  对外 C API 使用小驼峰，内部函数使用小写下划线。
- Python 公共函数使用 `snake_case`，私有函数使用 `_snake_case`，类型使用大驼峰；
  科学接口保留既有领域惯用名称时必须在根 `AGENTS.md` 登记。
- 主测试位于 `pyqcu/testing/**`。技能本身的 shell 回归与实现同目录，并由该技能
  `AGENTS.md` 固定调用入口。
- `docs/**` 保存正式文档和解包 Office 文档资产；`logs/**` 只暂存当前任务尚未归档的
  运行记录、阶段报告和图表。阶段结果先提交并打 tag，随后在后续清理提交中删除工作区
  报告副本，tag 对象是历史报告的唯一规范归档。
- `data/**` 保存数据、SVG、构建/安装树和缓存实体；目录元数据保留在所属目录。

## 本地例外

- `data/**` 的已跟踪内容属于正式证据、自包含构建/安装树或数据资产，不按通用
  “仅元数据”规则迁移。
- `logs/**` 可在打 tag 前暂存第一方 `.md`、`.tex`、`.pdf`、图片、`.stdout`、
  `.jsonl` 和压缩运行摘要；打 tag 后不得保留 `logs/<tag>/**` 报告副本。
- `docs/张鑫 508应用测试报告-PyQCU/` 是按 `docs/AGENTS.md` 保留的解包 Office 资源。
- `skills/pyqcu-tag/tag-chain.test.sh` 是该技能的定点回归，不迁入 `pyqcu/testing/`。
- `refer/git-rep/**`、`refer/huzc/**`、`data/quda-*`、`data/venv-*`、`data/p100-*` 和
  上游 vendored/generated 内容保留来源结构、名称和内容；本库自产文档仍按内容或目的命名，
  不得用 Git 标签名作路径。
- 根 `AGENTS.md` 的 form 标记块由 `form-agents-sync.sh` 唯一维护，必须通过 `--check`；
  不得删除或重复标记、整体禁用审计，或改写排除项来绕过已登记历史例外。新增例外必须
  精确匹配、保留证据并执行审计回归。

## 工作流程

1. 锁定 Git 根目录、分类复杂库、检查工作树和最近标签，输出任务边界。
2. 运行 `bash skills/pyqcu-form/form-audit.sh`，同时检查未跟踪非忽略文件、忽略状态、
   引用、重复产物、冲突标记和文档计数。
3. 建立冲突表，将每项归为 rename/move/update/keep/delete/exception；删除项必须可经
   Git 或 tag 恢复，或由用户明确授权，并先列出候选与恢复点。
4. 先运行可用基线测试，再按目录、文件、符号、文档、技能和清理顺序分批改动。
5. 每批执行最小验证；文档/技能改动至少验证旧引用清零、shell 语法和 Markdown/表格一致。
6. 最终执行 `git diff --check`、本地审计、定向回归和 `git status --short`。
7. 按 form 交付顺序复查 diff、同步文档与技能索引；tag 前确保阶段报告已提交，
   tag 后在独立清理提交中删除工作区报告副本。

## 验证入口

```bash
bash skills/pyqcu-form/form-audit.sh
bash -n build.sh env.sh install.sh skills/pyqcu-tag/tag-chain.sh skills/pyqcu-tag/tag-chain.test.sh
bash skills/pyqcu-tag/tag-chain.test.sh
git cat-file -e <tag>:<archived-report-path>
git show <tag>:<archived-report-path> >/dev/null
cd pyqcu/testing && pytest .
git diff --check
```

全量 pytest 可能依赖 CUDA、MPI、QUDA 或大体积数据；不满足设备条件时记录未运行原因，
但纯格式改动仍必须完成审计、shell 语法、定点回归和引用检查。

## 错误处理

- 工作树已有改动：区分本任务与用户改动，无法安全分离时停止写操作。
- 文件被引用：移动与引用更新必须在同一批完成，旧路径搜索清零后再继续。
- 审计器命中已登记例外：更新 `skills/pyqcu-form/form-audit.sh` 的精确例外规则并测试，不整体禁用规则。
- 工作区仍存在 `logs/<tag>/**` 报告：先确认对应 tag 已包含原件，再在独立清理提交中删除。
- 参考快照校验失败：停止用该快照推导命名，使用明文规则继续。
- 功能回归失败：转 debug 定位根因；不得删除测试或修改期望值绕过。
- 远端漂移或禁止 fast-forward：保留本地改动并报告，禁止 force push。

## 注意事项

- 不把名称机械转换为小写、下划线或驼峰；先判断语言、构件类型和公开边界。
- `.gitignore`、`AGENTS.md`、`README.md` 是目录元数据，可保留在受限内容目录。
- 历史资产的删除以 Git 可恢复为前提；不可恢复的未跟踪日志和缓存不得默认删除。
- tag 报告归档与删除必须分成两个提交：先提交报告并打 tag，再删除报告；同一提交中
  删除会使 tag 无法保存报告。
- 标签只用于版本定位，提交和标签消息不得从 `devN`、`stabN` 等名称反推业务语义。
