# PyQCU `docs/**` 来源映射

本表覆盖当前 `docs/` 顶层的 17 份 Markdown 文档。`docs/AGENTS.md` 是文件索引；
需要判断“原文应保留在哪里”时先读该文件和 `docs/ORGANIZATION.md`。

## 权威级别

| 级别 | 含义 | 在技能中的使用方式 |
|---|---|---|
| A 规范 | 当前项目约定或操作指南 | 可提炼为硬约束与执行命令 |
| B 技术参考 | 当前实现、算法、布局的系统说明 | 提炼不变量，并保留源码回读入口 |
| C 时点证据 | 有明确日期、窗口和状态的测试/性能结果 | 连同口径与状态引用，不作永久承诺 |
| D 历史记录 | 已完成的整理或阶段归档 | 仅用于追溯决策与恢复方法 |

## 文件映射

| 文档 | 级别 | 主责技能 | 提取内容与边界 |
|---|---|---|---|
| [`docs/AGENTS.md`](../../../docs/AGENTS.md) | A | `pyqcu-docs`, `pyqcu-form` | 文件索引、正式文档边界、tag 生命周期；新增/迁移文档时先更新此索引。 |
| [`docs/ORGANIZATION.md`](../../../docs/ORGANIZATION.md) | A | `pyqcu-form` | `docs/data/logs/testing` 的规范位置、跨目录提升、tag 报告两提交生命周期和语义例外。 |
| [`docs/dims.md`](../../../docs/dims.md) | A | `pyqcu-tools`, `pyqcu-lattice`, `pyqcu-dslash` | `s/c/d/p/x/y/z/t` 轴含义、规范场/费米子/奇偶布局；空间时间始终是最后四轴。 |
| [`docs/env.md`](../../../docs/env.md) | A | `pyqcu-tools`, `pyqcu-testing` | Python 依赖、`PYTHONPATH` 与最小环境入口；永久环境仍需以 `env.sh` 和安装文档为准。 |
| [`docs/install.md`](../../../docs/install.md) | A | `pyqcu-kupl`, `pyqcu-cuda`, `pyqcu-testing-pyquda` | KUPL CMake/CTest、`PYQCU_KUPL_LIB`、QUDA/PyQUDA 安装选择与 ABI 对齐。 |
| [`docs/examples.md`](../../../docs/examples.md) | A | `pyqcu-testing` | 旧 `examples/` 到 `pyqcu/testing/` 的迁移映射、目录分类和常用入口。 |
| [`docs/profiler.md`](../../../docs/profiler.md) | A | `pyqcu-testing-profiler` | `torch.profiler` 入口、Chrome trace 与 Perfetto 使用。 |
| [`docs/cuda_execution_topology.md`](../../../docs/cuda_execution_topology.md) | A | `pyqcu-cuda`, `pyqcu-testing`, `pyqcu-benchmark` | 一进程一 rank 一设备、MPI 跨设备分域、`MultiGpuMultigrid` 测试专用例外与检查清单。 |
| [`docs/qcu_quda_multigrid_algorithms.md`](../../../docs/qcu_quda_multigrid_algorithms.md) | B | `pyqcu-solver`, `pyqcu-cpp-qcu`, `pyqcu-cuda` | QCU legacy 与 QUDA/PyQUDA MG 的组成、默认算法、flow、版本边界和源码依据。 |
| [`docs/qcu_quda_multigrid_subalgorithms.md`](../../../docs/qcu_quda_multigrid_subalgorithms.md) | B | `pyqcu-solver` | Schur、null vector、transfer、smoother、Krylov、cycle、reliable update 的公式与优劣。 |
| [`docs/qcu_quda_multigrid_coarse_operator.md`](../../../docs/qcu_quda_multigrid_coarse_operator.md) | B | `pyqcu-solver`, `pyqcu-tools`, `pyqcu-cpp-qcu` | legacy 33-tensor 与 Strict/QUDA `X/Y/Yhat` 粗算子的表示差异；禁止把有效粗量解释成物理 gauge/clover。 |
| [`docs/progress_kupl_cuda_mg_20261021.md`](../../../docs/progress_kupl_cuda_mg_20261021.md) | C/D | `pyqcu-benchmark`, `pyqcu-kupl`, `pyqcu-cuda` | `dev99..1640349d` 的静态阶段归档；只作历史证据，未持久化测试明确视为未验证。 |
| [`docs/progress_until_20261022_worktree.md`](../../../docs/progress_until_20261022_worktree.md) | C | `pyqcu-benchmark`, `pyqcu-kupl`, `pyqcu-cuda` | 当前工作区优化、性能链、撤回方向和风险；状态为 `userinterlude`，不得写成已打 tag 结论。 |
| [`docs/three_way_benchmark_20261022.md`](../../../docs/three_way_benchmark_20261022.md) | C | `pyqcu-benchmark`, `pyqcu-kupl`, `pyqcu-cuda` | CUDA/KUPL/QUDA 同窗测量、正确性门、复现脚本、原始数据入口和限制。 |
| [`docs/profiler_solver_performance_history.md`](../../../docs/profiler_solver_performance_history.md) | C/D | `pyqcu-benchmark`, `pyqcu-testing-profiler` | `test9..dev103` 的 409 个唯一性能/失败记录索引；用于追历史，不用于重新合并跨口径数值。 |
| [`docs/form_pyqcu_20261020.md`](../../../docs/form_pyqcu_20261020.md) | D | `pyqcu-form` | form 托管块、历史例外与标签交付门；不再作为当前操作命令的唯一来源。 |
| [`docs/form_pyqcu_20261022.md`](../../../docs/form_pyqcu_20261022.md) | D | `pyqcu-form` | 标签名路径整改、目录索引、验证证据和恢复命令；当前规则回到根 `AGENTS.md` 与 `pyqcu-form`。 |

## 非提取项

- `docs/.gitignore` 是目录元数据，不进入技能知识。
- 大体积 PDF、Office 解包资产、图片和运行数据保留在原目录；技能只记录入口与读取条件。
- `logs/**`、`data/**` 中由 `docs/` 报告引用的实体不复制到技能；使用报告中给出的路径、
  SHA256 或 `git show <tag>:<path>` 回到证据。

## 维护规则

1. 新增 `docs/` 顶层文档时，同时更新 `docs/AGENTS.md` 和本表。
2. 文件改名后检查所有技能中的相对链接；历史 tag 路径不改。
3. 文件状态从 C 升为 A/B 前，必须有当前源码或复跑证据支撑。
4. 域技能已有细节时，本表只保留主责关系，不重复算法正文。
