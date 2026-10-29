# PyQCU `docs/**` 来源映射

本表覆盖当前 `docs/` 顶层的 35 份 Markdown 文档。`docs/AGENTS.md` 是文件索引；
需要判断“原文应保留在哪里”时先读该文件和 `docs/ORGANIZATION.md`。

## 权威级别

| 级别 | 含义 | 在技能中的使用方式 |
|---|---|---|
| A 规范 | 当前项目约定或操作指南 | 可提炼为硬约束与执行命令 |
| B 技术参考 | 当前实现、算法、布局的系统说明 | 提炼不变量，并保留源码回读入口 |
| P 设计提案 | 面向未来但尚未实现的接口、架构或迁移方案 | 只提炼目标与约束，引用时必须标注“尚未实现” |
| C 时点证据 | 有明确日期、窗口和状态的测试/性能结果 | 连同口径与状态引用，不作永久承诺 |
| D 历史记录 | 已完成的整理或阶段归档 | 仅用于追溯决策与恢复方法 |

每份文档只有一个 `Primary` 技术裁决技能；`Consumers` 只建立最小交叉指针，不复制完整规则。
跨域证据的 Primary 由最终解释权决定，而不是由参与生产的后端数量决定。

## 文件映射

| 文档 | 级别 | Primary | Consumers | 提取内容与边界 |
|---|---|---|---|---|
| [`docs/AGENTS.md`](../../../docs/AGENTS.md) | A | `pyqcu-docs` | `pyqcu-form` | 文件索引、正式文档边界、tag 生命周期；新增/迁移文档时先更新此索引。 |
| [`docs/ORGANIZATION.md`](../../../docs/ORGANIZATION.md) | A | `pyqcu-form` | — | `docs/data/logs/testing` 的规范位置、跨目录提升、tag 报告两提交生命周期和语义例外。 |
| [`docs/dims.md`](../../../docs/dims.md) | A | `pyqcu-tools` | `pyqcu-lattice`, `pyqcu-dslash` | `s/c/d/p/x/y/z/t` 轴含义、规范场/费米子/奇偶布局；空间时间始终是最后四轴。 |
| [`docs/env.md`](../../../docs/env.md) | A | `pyqcu-tools` | `pyqcu-testing` | Python 依赖、`PYTHONPATH` 与最小环境入口；永久环境仍需以 `env.sh` 和安装文档为准。 |
| [`docs/install.md`](../../../docs/install.md) | A | `pyqcu-cuda` | `pyqcu-kupl`, `pyqcu-testing-dcu`, `pyqcu-testing-pyquda` | CUDA/DCU/KUPL `install.sh` 选路、KUPL CMake/CTest、`PYQCU_KUPL_LIB`、QUDA/PyQUDA 安装选择与 ABI 对齐。 |
| [`docs/dcu_build_backend.md`](../../../docs/dcu_build_backend.md) | A | `pyqcu-cuda` | `pyqcu-testing-dcu`, `pyqcu-docs` | DCU 使用 `PYQCU_BUILD_BACKEND=dcu`/`install.sh dcu`、DTK compiler 预检、ROCm/HIP 验证、CUDA-compatible 运行时命名和当前占位边界。 |
| [`docs/kupl_only_deployment.md`](../../../docs/kupl_only_deployment.md) | A | `pyqcu-kupl` | `pyqcu-testing-cpu`, `pyqcu-benchmark`, `pyqcu-tools` | 纯 KUPL 无 CUDA 构建、旧版 PyTorch dtype、CTest/Python 测试、生产求解、NUMA 放置、Multigrid、升级和排错。 |
| [`docs/examples.md`](../../../docs/examples.md) | A | `pyqcu-testing` | — | 旧 `examples/` 到 `pyqcu/testing/` 的迁移映射、目录分类和常用入口。 |
| [`docs/kupl_kunpeng_guide.md`](../../../docs/kupl_kunpeng_guide.md) | A | `pyqcu-kupl` | `pyqcu-testing-cpu`, `pyqcu-benchmark`, `pyqcu-docs` | KUPL 通用 C++17 兼容边界、鲲鹏 HMPI/MPI/OpenMP 放置、NUMA/片上内存/大页策略、运行时规划器、专项测试和性能证据模板。 |
| [`docs/profiler.md`](../../../docs/profiler.md) | A | `pyqcu-testing-profiler` | — | `torch.profiler` 入口、Chrome trace 与 Perfetto 使用。 |
| [`docs/cuda_execution_topology.md`](../../../docs/cuda_execution_topology.md) | A | `pyqcu-cuda` | `pyqcu-testing`, `pyqcu-benchmark` | 一进程一 rank 一设备、MPI 跨设备分域、`MultiGpuMultigrid` 测试专用例外与检查清单。 |
| [`docs/torch_backend_layout.md`](../../../docs/torch_backend_layout.md) | A | `pyqcu-solver` | `pyqcu-dslash`, `pyqcu-smear`, `pyqcu-tools` | Torch 实现位于 `pyqcu/torch`；公共 lattice/tools/selector 与兼容门面的边界。 |
| [`docs/api_contract.md`](../../../docs/api_contract.md) | A | `pyqcu-api` | `pyqcu-solver`, `pyqcu-cuda`, `pyqcu-kupl` | P0 数据模型/Protocol 与 P1 `LinearOperator`、统一 `Krylov`、Torch 迭代和 CUDA/KUPL 原生整次 solve 语义。 |
| [`docs/qcu_unified_backend_strategy_20261023.md`](../../../docs/qcu_unified_backend_strategy_20261023.md) | P | `pyqcu-api` | `pyqcu-solver`, `pyqcu-cuda`, `pyqcu-kupl` | 六实现比较属于现状；P0 契约已落地，完整 backend factory 与组件迁移仍属于设计提案。 |
| [`docs/multigrid_setup_cpp_migration.md`](../../../docs/multigrid_setup_cpp_migration.md) | B/C | `pyqcu-solver` | `pyqcu-cuda`, `pyqcu-kupl`, `pyqcu-testing` | Legacy setup 的 C++ 迁移边界、ABI、已验证阶段与 10 分钟/cache-hit 验收门；DCU 调度未完成项不得视为已证实。 |
| [`docs/qcu_quda_multigrid_algorithms.md`](../../../docs/qcu_quda_multigrid_algorithms.md) | B | `pyqcu-solver` | `pyqcu-cpp-qcu`, `pyqcu-cuda` | QCU legacy 与 QUDA/PyQUDA MG 的组成、默认算法、flow、版本边界和源码依据。 |
| [`docs/qcu_quda_multigrid_subalgorithms.md`](../../../docs/qcu_quda_multigrid_subalgorithms.md) | B | `pyqcu-solver` | — | Schur、null vector、transfer、smoother、Krylov、cycle、reliable update 的公式与优劣。 |
| [`docs/qcu_quda_multigrid_coarse_operator.md`](../../../docs/qcu_quda_multigrid_coarse_operator.md) | B | `pyqcu-solver` | `pyqcu-tools`, `pyqcu-cpp-qcu` | legacy 33-tensor 与 Strict/QUDA `X/Y/Yhat` 粗算子的表示差异；禁止把有效粗量解释成物理 gauge/clover。 |
| [`docs/mg_cache_sizes_24x24x24x64_20261023.md`](../../../docs/mg_cache_sizes_24x24x24x64_20261023.md) | B | `pyqcu-solver` | `pyqcu-benchmark`, `pyqcu-cuda` | L24 c64 两级 `nvec=12` 下 Legacy 33-tensor、Strict schema-v2 cache 与 QUDA null-vector 的公式、逻辑 payload 和文件边界。 |
| [`docs/progress_kupl_cuda_mg_20261021.md`](../../../docs/progress_kupl_cuda_mg_20261021.md) | C/D | `pyqcu-benchmark` | `pyqcu-kupl`, `pyqcu-cuda` | `dev99..1640349d` 的静态阶段归档；只作历史证据，未持久化测试明确视为未验证。 |
| [`docs/progress_until_20261022_worktree.md`](../../../docs/progress_until_20261022_worktree.md) | C | `pyqcu-benchmark` | `pyqcu-cuda`, `pyqcu-kupl` | 已提交的时点工作区快照；状态为 `userinterlude`，不得写成 tag 后永久结论。 |
| [`docs/three_way_benchmark_20261022.md`](../../../docs/three_way_benchmark_20261022.md) | C | `pyqcu-benchmark` | `pyqcu-kupl`, `pyqcu-cuda` | CUDA/KUPL/QUDA 同窗测量、正确性门、复现脚本、原始数据入口和限制。 |
| [`docs/qcu_six_way_benchmark_20261023.md`](../../../docs/qcu_six_way_benchmark_20261023.md) | C | `pyqcu-benchmark` | `pyqcu-solver`, `pyqcu-cuda`, `pyqcu-kupl` | 六版本与 QUDA 的真实 LIME 前缀同窗结果；严格区分 fine-only、实际 correction/V-cycle、CPU/GPU 和不可比边界。 |
| [`docs/qcu_mg_benchmark_decisions_20261023.md`](../../../docs/qcu_mg_benchmark_decisions_20261023.md) | C | `pyqcu-benchmark` | `pyqcu-solver`, `pyqcu-cuda`, `pyqcu-kupl` | 大格 Strict setup 不可接受的 benchmark skip 决策；只说明未完成下界和项目范围，不写成数值 FAIL。 |
| [`docs/qcu_mg_refactor_validation_20261023.md`](../../../docs/qcu_mg_refactor_validation_20261023.md) | B/C | `pyqcu-benchmark` | `pyqcu-solver`, `pyqcu-cuda`, `pyqcu-kupl`, `pyqcu-api` | 16⁴ 六项矩阵、QUDA/BiCG 对照、跨实现解差、L24 单/多 rank 2/3/4 层回归、非 root 边界与已知未验证项。 |
| [`docs/qcu_mg_unified_interfaces_and_reuse_20261029.md`](../../../docs/qcu_mg_unified_interfaces_and_reuse_20261029.md) | C | `pyqcu-benchmark` | `pyqcu-api`, `pyqcu-solver`, `pyqcu-cuda`, `pyqcu-kupl`, `pyqcu-report` | 当前工作树统一接口、Legacy odd-Schur/Strict v2 cache、identity-safe reuse、L24 smoke 与 L32 正式入口、QUDA 构建经验、BiCG/MG 分栏及作图规则；未打 tag。 |
| [`docs/qcu_mg_mpi_cache_builder_20261029.md`](../../../docs/qcu_mg_mpi_cache_builder_20261029.md) | C | `pyqcu-benchmark` | `pyqcu-kupl`, `pyqcu-cuda`, `pyqcu-solver` | 多 MPI Legacy cache builder 的分布式 null-vector、四张量 HDF5 MPI-IO、private singleton communicator、GPU patch、1/2/4-rank slab 验证、KUPL 直用与 L48 主机 OOM 边界；未打 tag。 |
| [`docs/qcu_mg_performance_ledger_20261029.md`](../../../docs/qcu_mg_performance_ledger_20261029.md) | C | `pyqcu-benchmark` | `pyqcu-cuda`, `pyqcu-kupl`, `pyqcu-solver`, `pyqcu-report` | 用户要求的实时性能账本；逐批记录完整配方、输入/库 identity、耗时、迭代、真残差、cache、Slurm 审计、失败和主动取消；未打 tag。 |
| [`docs/qcu_mg_performance_report_20261029.tex`](../../../docs/qcu_mg_performance_report_20261029.tex) | C | `pyqcu-report` | `pyqcu-benchmark`, `pyqcu-cuda`, `pyqcu-kupl`, `pyqcu-solver` | 当前工作的 LaTeX 主报告源码；包含结果表、失败边界和证据路径，尚未完成 PDF 编译；未打 tag。 |
| [`docs/legacy_vs_strict_mg_status_20261022.md`](../../../docs/legacy_vs_strict_mg_status_20261022.md) | C | `pyqcu-benchmark` | `pyqcu-solver`, `pyqcu-cuda`, `pyqcu-kupl` | Legacy/Strict CUDA 与 KUPL 现有 JSON/stdout 的完成状态；保留 V-cycle 计数、KUPL 缺失和不可比边界。 |
| [`docs/legacy_vs_strict_mg_memory_20261023.md`](../../../docs/legacy_vs_strict_mg_memory_20261023.md) | C | `pyqcu-solver` | `pyqcu-benchmark`, `pyqcu-cuda` | 同输入 c64 两级设备峰值；Legacy 更低，但 Legacy 未执行 coarse V-cycle，24³×64 Strict 缺失。 |
| [`docs/multigrid_legacy_strict_comparison_20261022.md`](../../../docs/multigrid_legacy_strict_comparison_20261022.md) | B/C | `pyqcu-solver` | `pyqcu-cuda`, `pyqcu-kupl`, `pyqcu-testing` | 算法、接口与生命周期属于当前参考；第 5 节性能状态属于时点证据。 |
| [`docs/multigrid_deprecation_audit_20261022.md`](../../../docs/multigrid_deprecation_audit_20261022.md) | C | `pyqcu-solver` | `pyqcu-form`, `pyqcu-cuda`, `pyqcu-kupl`, `pyqcu-testing` | 稳定 HEAD 上的 MultiGrid 弃用、历史入口、占位目录与兼容别名审计；重构后需重审。 |
| [`docs/profiler_solver_performance_history.md`](../../../docs/profiler_solver_performance_history.md) | C/D | `pyqcu-benchmark` | `pyqcu-testing-profiler` | `test9..dev103` 的 409 个唯一性能/失败记录索引；用于追历史，不用于重新合并跨口径数值。 |
| [`docs/form_pyqcu_20261020.md`](../../../docs/form_pyqcu_20261020.md) | D | `pyqcu-form` | — | form 托管块、历史例外与标签交付门；不再作为当前操作命令的唯一来源。 |
| [`docs/form_pyqcu_20261022.md`](../../../docs/form_pyqcu_20261022.md) | D | `pyqcu-form` | — | 标签名路径整改、目录索引、验证证据和恢复命令；当前规则回到根 `AGENTS.md` 与 `pyqcu-form`。 |

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
5. 变更后运行 `python -B skills/pyqcu-docs/scripts/audit-docs-skills.py`，并按错误码修复。
