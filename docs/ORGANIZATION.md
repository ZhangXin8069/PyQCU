# PyQCU 文件分类约定

`docs/`、`data/`、`logs/`、`pyqcu/test/`、`pyqcu/testing/` 采用“保留资产实体只存
一个规范目录”的方式整理。历史重复代码、生成缓存与一次性会话垃圾直接删除，不建立
软链接或嵌套归档目录保存。

## 规范目录

| 类型 | 规范位置 | 示例 |
|---|---|---|
| 正式独立文档 | `docs/**` | 指南、规范、跨项目报告 |
| 当前任务未归档日志与阶段报告 | `logs/**` | `.json`、`.log`、`.tsv`、`.txt`、`.stdout`、阶段报告 `.md`/`.tex`/`.pdf` |
| 已归档 tag 报告 | Git tag 对象 | 通过 `git show <tag>:<path>` 读取，工作区不保留副本 |
| 数据、安装与构建产物 | `data/**` | `.h5`、`.so`、`.dat`、`.csv`、`.svg`、venv/build/install 树 |
| 测试（正式快速端口） | `pyqcu/test/**` | `python -m pyqcu.test` 的 `Case`/`policy`/后端 runner；随 sdist/wheel 发行，安装态可运行 |
| 测试（非正式/研究/性能） | `pyqcu/testing/**` | `.py`、`.sh`、`.patch`、集成测试入口、性能基准与历史回归套件；仅源码 checkout |

跨目录提升时保留来源语义：

- `data/**` 内的 `.md`、`.pdf`、`.tex` 进入 `docs/data/`；
- `logs/<tag>/**` 阶段报告先提交并打 tag，归档完成后在后续清理提交中删除；
  `logs/<tag>/**` 报告副本不得继续留在工作区；
- `data/**` 内的 JSON、日志、TSV、TXT 等进入 `logs/data/`；
- `logs/**` 内的 `.h5`、`.npz`、`.npy`、`.pt` 进入 `data/logs/`；
- Office 文件及非文档资源进入 `data/docs/`。
- `logs/**` 和第一方 `data/**` 内仍需维护的测试/脚本进入 `pyqcu/testing/`；
  正式发行测试端口只写 `pyqcu/test/**`，不向 `pyqcu/testing/` 新增正式用例；
  已被现行实现取代的历史测试套件直接删除。

仍需保留的跨目录资产迁移时保留来源根和相对路径。整理后的旧路径不再建立软链接，
代码和文档统一引用实体新路径。

tag 报告采用固定生命周期：

1. 在当前任务中写入 `logs/**` 并完成验证；
2. 将报告纳入待归档提交并创建对应 tag；
3. 在后续清理提交中删除工作区报告副本；
4. 后续引用使用 `git show <tag>:<path>` 或临时 worktree，不恢复永久副本。

删除不能与 tag 指向的归档提交合并，否则 tag 中不会包含报告。

LaTeX `.aux/.log/.out/.toc/.nav/.snm/.vrb`、`.report-build-*`、逐页
`report-render-*`、agent/codex 会话、临时文件和生成缓存均直接删除。

## 测试端口分层（正式 / 非正式）

测试代码只有两个规范位置，按“是否随发行包交付”划分，不按文件新旧划分：

| 端口 | 规范位置 | 交付属性 | 内容边界 |
|---|---|---|---|
| 正式快速端口 | `pyqcu/test/**` | 随 sdist/wheel 发行；安装态 `python -m pyqcu.test` 可运行 | stdlib-only 导入的 `Case`/`policy`/后端 runner、`selfcheck.py`；19 个 tier 0 跨后端正确性门（默认格点 `[8,8,8,16]`、c64、单 rank、无数据文件），可用 `--lattice-size`/`--lattice-grid`/`--mg-grid`/`--mass`/`--kappa`/`--csw`/`--solver`/`--dslash`/`--device`/`--backend cpu/dcu/npu` 等选项改参数；`--mode list/prepare/prewarm/solve/merge/run/validate` 是 full-mode 工作台账（rows/reports/solutions/caches，`--resume`/`--skip-failed`/`--run-label`），`--threads` 只在 torch runner 应用、KUPL/CUDA runner 仅记录 |
| 非正式套件 | `pyqcu/testing/**` | **仅源码 checkout**：`MANIFEST.in` 的 `prune pyqcu/testing` 与 `setup.py` 的 `find_packages(exclude=["pyqcu.testing", "pyqcu.testing.*"])` 双重排除 | 集成/回归/性能/QUDA/PyQUDA/HPCKit 矩阵、参考 HDF5、历史资产；模块级依赖 `torch`、`mpi4py`、`pytest` |

- `pip install pyqcu` 的用户**没有** `pyqcu.testing`；安装态验证只能用 `pyqcu.test`
  或公开 API，不得把 `pyqcu.testing` 写成用户入口。
- 两个方向都不得互相 import，生产模块也不得 import `pyqcu.test`。
- **新增正式用例只进 `pyqcu/test/units.py`**；`pyqcu/testing/**` 不再新增“正式”用例，
  只保留探索性、研究性、性能性和历史回归资产。
- 正式端口的展开选项（`--lattice-size`/`--lattice-grid`/`--mg-grid`、`--gauge`/
  `--gauge-sigma`/`--source`、`--mass`/`--kappa`/`--csw`、`--solver`/`--dslash`、
  `--device`/`--device-id`/`--threads`、`--backend cpu/dcu/npu`）只改运行参数与用例
  筛选，不改变发行边界与包布局；`--mode` 只把同一矩阵编排成 `rows/reports/solutions/
  caches` 台账，`--gauge-file`/`--source-file` 只读取调用者显式给出的 HDF5 输入。
  用例矩阵与生效配置以 `python -m pyqcu.test --list`（或 `--list --json`）为准。
- `pyqcu/test/**` 内不得出现 `test_*.py`（form 审计只在 `<repo>/testing/**` 内免除
  `TEST_LOCATION`），也不得 import `pytest`；`pyqcu/test/README.md` 与
  `pyqcu/test/AGENTS.md` 只是仓库元数据，不进 sdist。
- 打包闸门：`release_check --offline` 保留 `sdist no testing`（`pyqcu/testing/**` 不进包），
  并新增正向条目 `sdist ships pyqcu.test`（`pyqcu-<ver>/pyqcu/test/*.py` 必须随包发行）。
  正式端口的命令、退出码、tier 0 用例与 SKIP 策略见
  [`pyqcu/test/README.md`](../pyqcu/test/README.md)。

## 语义例外

以下目录视为自包含归档或构建工作区，按目录整体保留，不递归拆散其中的文件：

- `data/quda-*`、`data/venv-*`、`data/p100-*`：第三方源码、构建、安装与虚拟环境；
- 解包 Office 资源：正式文档附带的原始资产。

`AGENTS.md` 和 `.gitignore` 是目录元数据，始终留在其所属目录，不参与内容类型迁移。

格式治理规则与本地例外见根 `AGENTS.md` 的“form 格式约定”，可执行审计入口为
`bash skills/pyqcu-form/form-audit.sh`。
