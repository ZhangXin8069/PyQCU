# AGENTS.md — docs

PyQCU 参考文档。

> 2026-09-27 迁移：顶层 `examples/` 已迁入 `pyqcu/testing/`。本目录旧报告中若
> 出现 `examples/...`，应映射为 `pyqcu/testing/...`；现行文档已统一使用新路径。

当前 `docs/` 顶层共有 **34 份 Markdown 文档（含本索引）**。下表只维护文件职责；
权威级别、唯一主责技能和提取边界见
[`skills/pyqcu-docs/references/project-docs-map.md`](../skills/pyqcu-docs/references/project-docs-map.md)。

## 核心操作规范

| 文件 | 内容 |
|---|---|
| `ORGANIZATION.md` | `docs/`、`data/`、`logs/`、`pyqcu/testing/` 的统一分类规则、垃圾清理与构建树例外 |
| `dims.md` | 维度命名方案（`s`=spin、`c`=color、`d`=direction、`p`=parity、`x/y/z/t`=时空）。`ccdxyzt`、`scxyzt`、`psctzyx` 等约定 |
| `env.md` | Python 环境设置；`env.sh` 始终设置 `LD_LIBRARY_PATH`/`PYTHONPATH`，仅在仓库内 QUDA 库存在时设置 `QUDA_PATH`，显式 `QUDA_INSTALL` 可覆盖安装树 |
| `install.md` | 安装指南 — CUDA/KUPL 后端选择和 install.sh 工作流 |
| `kupl_only_deployment.md` | 纯 KUPL 无 CUDA 安装、旧版 PyTorch 兼容、测试、生产、Multigrid、升级和故障排查手册 |
| `examples.md` | 测试与示例迁移映射、运行入口与结果解读 |
| `kupl_kunpeng_guide.md` | KUPL CPU 后端在鲲鹏上的构建、MPI/OpenMP 绑定、NUMA/内存策略、测试与性能证据规范 |
| `profiler.md` | 剖析指南 — torch.profiler 与 Perfetto |
| `cuda_execution_topology.md` | CUDA 生产拓扑：一 Python 进程 = 一 MPI rank = 一 GPU/DCU/GPGPU；`MultiGpuMultigrid` 测试专用例外 |
| `torch_backend_layout.md` | Torch/CANN 实现目录、顶层公共层、兼容门面与 multigrid 选择器边界 |
| `api_contract.md` | `pyqcu.api` 的 P0 数据模型/Protocol 与 P1 LinearOperator、统一 Krylov、Torch 迭代和 CUDA/KUPL 原生整次 solve 边界 |

## MultiGrid 技术参考

| 文件 | 内容 |
|---|---|
| `qcu_unified_backend_strategy_20261023.md` | Torch/CUDA/KUPL 的 Legacy/Strict 六实现对比、统一组件架构及扩展 dslash/BiCGStab/CG 方案 |
| `multigrid_setup_cpp_migration.md` | Legacy MG 的 null-vector、局部正交化、33-tensor stencil 从 Python 迁移到 CUDA/KUPL C++ 的边界与验收门 |
| `qcu_quda_multigrid_algorithms.md` | QCU legacy MG 与 QUDA/PyQUDA MG 的算法组成、流程图、默认参数和并排对照 |
| `qcu_quda_multigrid_subalgorithms.md` | QCU/QUDA MG 子算法的独立流程图、数学摘要、默认配置与同类算法优劣 |
| `qcu_quda_multigrid_coarse_operator.md` | QCU legacy 33-tensor 与 Strict/QUDA `X/Y/Yhat` 粗层 dslash 表示辨析及源码依据 |
| `mg_cache_sizes_24x24x24x64_20261023.md` | 真实 L24、c64、两级、`nvec=12` 下 Legacy 33-tensor、Strict schema-v2 cache 与 QUDA null-vector 的逐项尺寸推导 |
| `multigrid_legacy_strict_comparison_20261022.md` | Torch/CUDA/KUPL 的 Legacy/Strict 算法、接口、生命周期与既有证据简要对比（2026-10-22） |

## 性能、状态与风险证据

| 文件 | 内容 |
|---|---|
| `three_way_benchmark_20261022.md` | CUDA QCU / KUPL QCU / QUDA 三方性能测评与并行扩展报告（2026-10-22） |
| `qcu_six_way_benchmark_20261023.md` | Torch/CUDA/KUPL 六版本、QUDA 对照、真实 LIME 前缀、coarse 执行证据与同窗口性能结果（2026-10-23） |
| `qcu_mg_benchmark_decisions_20261023.md` | 大格 Strict 初始化的 benchmark SKIP 决策、适用范围与证据边界（2026-10-23） |
| `qcu_mg_refactor_validation_20261023.md` | 历史 16⁴ 六项矩阵及 L24 深度回归；该格点已被当前 `8x8x8x16` 正确性门替代，仅保留时点证据（2026-10-23） |
| `qcu_mg_unified_interfaces_and_reuse_20261029.md` | 当前工作树的三后端统一接口、Legacy/Strict cache 复用、L32 正式入口、QUDA 构建经验与图表规范（2026-10-29，未打 tag） |
| `qcu_mg_mpi_cache_builder_20261029.md` | Legacy-MG 多 MPI cache builder：分布式 null-vector、overlap stencil、1/2/4-rank slab 验证、Torch CPU 小格门和 KUPL 跨 cache 边界（2026-10-29，未打 tag） |
| `qcu_mg_performance_ledger_20261029.md` | 用户要求的实时性能账本：按批次记录求解器、完整参数、输入/库 SHA、耗时、迭代、真残差、cache、作业审计与失败记录（2026-10-29，未打 tag） |
| `qcu_mg_performance_report_20261029.tex` | 当前工作的 LaTeX 主报告：统一接口、CUDA/KUPL/QUDA 数据、失败边界和证据路径（2026-10-29，未打 tag；当前环境缺少 XeLaTeX） |
| `legacy_vs_strict_mg_status_20261022.md` | Legacy/Strict CUDA 与 KUPL MG 现有 JSON/stdout 的完成状态、数值边界和未验证项（2026-10-22） |
| `legacy_vs_strict_mg_memory_20261023.md` | Legacy/Strict CUDA MG 的设备级运行时显存峰值、测量边界、差异来源和未验证项（2026-10-23） |
| `progress_until_20261022_worktree.md` | 当前工作区续接进度总结（2026-10-22） |
| `profiler_solver_performance_history.md` | PyQCU / QCU / QUDA 求解器历史性能、失败与证据索引（2025-08-29 至 2026-10-22） |

## 历史与治理归档

| 文件 | 内容 |
|---|---|
| `progress_kupl_cuda_mg_20261021.md` | KUPL/CUDA 多卡与多重网格进度总结（2026-10-21） |
| `multigrid_deprecation_audit_20261022.md` | MultiGrid 弃用、历史 CUDA 入口、占位目录和兼容别名的证据审计（2026-10-22） |
| `form_pyqcu_20261020.md` | 2026-10-20 form 整理记录 |
| `form_pyqcu_20261022.md` | 2026-10-22 form 整理记录（tag 名路径整改与目录索引补齐） |

已归档的 tag 阶段报告以 tag 对象为准，通过 `git show <tag>:<path>` 或临时 worktree 读取。
当前任务的活跃草稿、未提交阶段结果和逐轮进度写入 `logs/**`。已经提交、具有稳定
status/date/provenance 的跨项目报告与时点证据可以保留在 `docs/**`，但必须明确
“未打 tag/特定工作树”属性，不能升级为永久结论；`docs/**` 不保存活跃工作草稿。
任务计划不作为 `docs/` 当前文件保留。

## 质量门

- 新增、重命名或删除文档时，同一批次更新本索引、
  `skills/pyqcu-docs/references/project-docs-map.md` 和
  `skills/pyqcu-docs/references/extracted-guidance.md` 的结论覆盖行。
- 文档/skills 改动提交前运行
  `python -B skills/pyqcu-docs/scripts/audit-docs-skills.py`，并完成本地链接、代码围栏、
  `git diff --check` 与 form 审计。

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
