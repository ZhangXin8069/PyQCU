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
| `form_pyqcu_20261020.md` | 2026-10-20 form 整理记录 |
| `form_pyqcu_20261022.md` | 2026-10-22 form 整理记录（tag 名路径整改与目录索引补齐） |
| `progress_kupl_cuda_mg_20261021.md` | KUPL/CUDA 多卡与多重网格进度总结（2026-10-21） |
| `progress_until_20261022_worktree.md` | 当前工作区续接进度总结（2026-10-22） |
| `three_way_benchmark_20261022.md` | CUDA QCU / KUPL QCU / QUDA 三方性能测评与并行扩展报告（2026-10-22） |
| `profiler_solver_performance_history.md` | PyQCU / QCU / QUDA 求解器历史性能、失败与证据索引（2025-08-29 至 2026-10-22） |
| `qcu_quda_multigrid_algorithms.md` | QCU legacy MG 与 QUDA/PyQUDA MG 的算法组成、流程图、默认参数和并排对照 |
| `qcu_quda_multigrid_subalgorithms.md` | QCU/QUDA MG 子算法的独立流程图、数学摘要、默认配置与同类算法优劣 |
| `qcu_quda_multigrid_coarse_operator.md` | QCU legacy 33-tensor 与 Strict/QUDA `X/Y/Yhat` 粗层 dslash 表示辨析及源码依据 |
| `cuda_execution_topology.md` | CUDA 生产拓扑：一 Python 进程 = 一 MPI rank = 一 GPU/DCU/GPGPU；`MultiGpuMultigrid` 测试专用例外 |

已归档的 tag 阶段报告以 tag 对象为准，通过 `git show <tag>:<path>` 或临时 worktree 读取；
当前任务尚未归档的进度总结暂留本目录，纳入后续标签后在清理批次中删除工作区副本。
任务计划不作为 `docs/` 当前文件保留。

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
