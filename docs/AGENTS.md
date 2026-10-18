# AGENTS.md — docs

PyQCU 参考文档。

> 2026-09-27 迁移：顶层 `examples/` 已迁入 `pyqcu/testing/`。本目录旧报告中若
> 出现 `examples/...`，应映射为 `pyqcu/testing/...`；现行文档已统一使用新路径。

## 文件

| 文件 | 内容 |
|---|---|
| `dims.md` | 维度命名方案（`s`=spin、`c`=color、`d`=direction、`p`=parity、`x/y/z/t`=时空）。`ccdxyzt`、`scxyzt`、`psctzyx` 等约定 |
| `env.md` | Python 环境设置 — 必需变量（`QUDA_PATH`、`LD_LIBRARY_PATH`、`PYTHONPATH`） |
| `install.md` | 安装指南 — build.sh + install.sh 工作流 |
| `examples.md` | 测试与示例迁移映射、运行入口与结果解读 |
| `profiler.md` | 剖析指南 — torch.profiler 与 Perfetto |
| `ORGANIZATION.md` | `docs/`、`data/`、`logs/`、`pyqcu/testing/` 的统一分类规则、垃圾清理与构建树例外 |

tag 阶段报告和任务计划不作为 `docs/` 当前文件保留；需要历史版本时使用
`git show <tag>:<path>` 或临时 worktree。

## 目录分类

- `docs/` 顶层只保存正式文档：`.pdf`、`.tex`、`.md`；
  `张鑫 508应用测试报告-PyQCU/` 是解包 Office 资源，按文档资产保留。
- `docs/data/` 保存从 `data/**` 提升出的 `.md`/`.pdf`/`.tex`。tag 阶段报告
  先进入待 tag 提交，tag 创建后在后续清理提交中删除，不保留
  `logs/<tag>/` 工作区副本或旧路径软链接。
- LaTeX 辅助文件、`.report-build-*`、逐页渲染图、agent/codex 会话和临时缓存
  均属于可再生产物，直接删除，不纳入归档。
- Office 文件及非文档资源不在 `docs/` 顶层保存；现有 `.docx`/`.xlsx` 的实体
  位于 `data/docs/`，调用方直接引用实体路径。
