# PyQCU 文件分类约定

`docs/`、`data/`、`logs/`、`pyqcu/testing/` 采用“保留资产实体只存一个规范目录”
的方式整理。历史重复代码、生成缓存与一次性会话垃圾直接删除，不建立软链接或嵌套
归档目录保存。

## 规范目录

| 类型 | 规范位置 | 示例 |
|---|---|---|
| 正式独立文档 | `docs/**` | 指南、规范、跨项目报告 |
| 日志、运行摘要与 tag 报告 | `logs/**` | `.json`、`.log`、`.tsv`、`.txt`、报告 `.md`/`.tex`/`.pdf` |
| 数据、安装与构建产物 | `data/**` | `.h5`、`.so`、`.dat`、`.csv`、`.svg`、venv/build/install 树 |
| 测试及其他代码 | `pyqcu/testing/**` | `.py`、`.sh`、`.patch`、测试入口与历史回归套件 |

跨目录提升时保留来源语义：

- `data/**` 内的 `.md`、`.pdf`、`.tex` 进入 `docs/data/`；
- `logs/<tag>/**` 内的报告继续保留在原 tag 目录，不迁入 `docs/`；
- `data/**` 内的 JSON、日志、TSV、TXT 等进入 `logs/data/`；
- `logs/**` 内的 `.h5`、`.npz`、`.npy`、`.pt` 进入 `data/logs/`；
- Office 文件及非文档资源进入 `data/docs/`。
- `logs/**` 和第一方 `data/**` 内仍需维护的测试/脚本进入 `pyqcu/testing/`；
  已被现行实现取代的历史测试套件直接删除。

仍需保留的跨目录资产迁移时保留来源根和相对路径。整理后的旧路径不再建立软链接，
代码和文档统一引用实体新路径。

LaTeX `.aux/.log/.out/.toc/.nav/.snm/.vrb`、`.report-build-*`、逐页
`report-render-*`、agent/codex 会话、临时文件和生成缓存均直接删除。

## 语义例外

以下目录视为自包含归档或构建工作区，按目录整体保留，不递归拆散其中的文件：

- `data/quda-*`、`data/venv-*`、`data/p100-*`：第三方源码、构建、安装与虚拟环境；
- 解包 Office 资源：正式文档附带的原始资产。

`AGENTS.md` 和 `.gitignore` 是目录元数据，始终留在其所属目录，不参与内容类型迁移。

格式治理规则与本地例外见根 `AGENTS.md` 的“form 格式约定”，可执行审计入口为
`bash skills/form/form-audit.sh`。
