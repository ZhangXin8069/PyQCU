# PyQCU form 格式审计（2026-10-19）

## 目标与边界

本次审计以 `2e6510ce` 为 Git 基线，覆盖已跟踪路径、技能目录与索引、审计入口、
文档/日志边界和当前文档中的绝对路径。PyQCU 按库名判定为复杂库，主导语言为
C++/CUDA、Python/Cython 和 Bash。

接受标准：

- 技能目录、`skills/AGENTS.md` 表格和 frontmatter 名称一致；
- 本地 form 审计入口不依赖当前机器固定的 `/root` 路径；
- 已跟踪文档位于其规范职责目录；
- 活动文档不再引用当前机器不存在的硬编码仓库路径；
- shell 语法、tag 回归、本地 form 审计和 `git diff --check` 通过。

## 冲突表摘要

| 对象 | 类型 | 当前状态 | 规则依据 | 处理 |
|---|---|---|---|---|
| `skills/AGENTS.md` | 索引 | 标为 `35/35`，并有 3 个已删除技能行；实际为 32 个目录 | 表格必须与目录逐项一致 | 修正为 `32/32`，删除 `pyqcu-dev80`、`pyqcu-dev84`、`pyqcu-testing-regression-session-2026-08-24` 过期行 |
| `skills/pyqcu-form/form-audit.sh` | 脚本 | 默认硬编码 `/root/configure/...`，在当前环境直接失败 | 审计入口必须可由 `CONFIGURE_ROOT`/`FORM_AUDIT_BIN` 配置并可移植 | 按 `FORM_AUDIT_BIN`、`CONFIGURE_ROOT`、`$HOME/configure` 顺序定位，保留 `/root/configure` 兼容回退 |
| `pyqcu/docs/report_multigrid_python_20260828.*` | 文档 | 历史报告放在 Python 包目录 | 报告按 tag 归档到 `logs/<tag>/` | 迁入 `logs/stab33/` |
| `docs/profiler.md`、`skills/pyqcu-benchmark/SKILL.md`、`skills/pyqcu-md-tex-sync/SKILL.md` | 文档 | 活动说明含 `/root/PyQCU` 硬编码路径 | 仓库内命令应使用相对路径；环境路径应显式配置 | 改为仓库相对路径 |
| `skills/pyqcu-form/form-audit.sh` | 回归 | 只调用通用路径审计，未检查技能索引和 frontmatter | 本地 form 特化必须防止索引漂移 | 增加技能目录计数、缺失/多余索引、frontmatter 名称和表头计数校验 |
| `logs/.DS_Store`、`docs/*.aux/.log/.out/.toc/.synctex.gz` 等 | 忽略文件 | 未跟踪且被 `.gitignore` 排除 | 非可恢复文件不得默认删除 | 保留并记录，不纳入交付 |
| `pyqcu/cuda/qcu/qcu.pxd` 与 `qcu_api.pxd`、两个 `qcu.pyi`、空 `PASS` 文件 | 重复内容 | 内容哈希相同 | 绑定兼容入口和占位后端需要固定实体 | 保留，不按重复文件误删 |

## 改动

- 将技能表及根目录说明中的技能数量更新为实际的 32 个。
- 修正本地 form 审计器的通用审计器定位，并新增技能索引/frontmatter 一致性检查。
- 为 `logs/<tag>/**` 增加精确的标签路径例外，不放宽代码、配置或对象名称的标签边界。
- 将 2026-08-28 纯 Python MultiGrid 报告从 `pyqcu/docs/` 迁移到 `logs/stab33/`，
  保持 tag 报告与运行资产同域。
- 将活动文档中的仓库绝对路径改为相对路径。
- 同步 `logs/AGENTS.md` 的 `stab33/` 归档索引。

## 验证证据

```bash
bash skills/pyqcu-form/form-audit.sh
bash -n build.sh env.sh install.sh skills/pyqcu-form/form-audit.sh \
  skills/pyqcu-tag/tag-chain.sh skills/pyqcu-tag/tag-chain.test.sh
bash skills/pyqcu-tag/tag-chain.test.sh
bash skills/pyqcu-tag/tag-chain.sh --check --remote origin
git diff --check
```

结果：

- 本地 form 审计：`PASS`，32 个技能目录与索引一致，1048 个既有资产 finding
  按本地规则豁免；
- shell 语法：退出码 0；
- tag 历史链、dry-run、本地修正和远端 lease 测试：`PASS`；
- `tag-chain.sh --check --remote origin`：`PASS`，273 个受管标签的 follow 链、
  提交祖先关系和远端对象一致；
- `git diff --check`：退出码 0；
- 旧路径 `pyqcu/docs/report_multigrid_python_20260828.*`：已无活动引用；
- 活动文档中的 `/root/PyQCU`：已清零；仅有 P100 远程部署说明和审计器
  `/root/configure` 兼容回退保留；
- 全量 pytest 未运行：本次未修改运行时 Python、Cython 或 C++ 代码，验证范围限于
  格式、shell 入口、索引和文档资产。

## 恢复方法

- 恢复文本改动：`git restore --source=2e6510ce -- <path>`。
- 恢复报告旧位置：将 `logs/stab33/report_multigrid_python_20260828.*` 移回
  `pyqcu/docs/`，或从基线恢复对应路径。
- 本地审计误报时，在 `skills/pyqcu-form/form-audit.sh` 增加精确规则并重新运行
  shell 语法和审计；不得整体禁用技能索引或标签边界检查。
