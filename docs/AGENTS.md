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
| `report_multigrid_quda_pyqcu_20260916.tex/.pdf` | 本轮 PyQCU/QUDA MultiGrid 分层 trace、三层优化与正式对照报告 |
| `report_multigrid_quda_pyqcu_20260917.tex/.pdf` | dev87 早期迭代：旧加速比撤回、double MG 修复、colored Galerkin 与正式协议复核 |
| `report_multigrid_distributed_20260918.tex/.pdf` | 分布式 Strict-MG：fine/compact/full halo、R/P coarse halo、distributed fused FGMRES、分布式 Galerkin setup 的实现、正确性与 V100 c64 性能矩阵；MG-1/c128/P100 的缺口逐条记录 |
| `report_multigrid_quda_pyqcu_20260929.tex/.pdf` | 基于 test27 原始结果的绝对耗时修订版：分组秒数与时间比并列、真实耗时图、QUDA 最粗层显式展示、阶段图重排与图例置底 |
| `report_multigrid_quda_pyqcu_20260930.tex/.pdf` | 基于 test27 的线性轴修订版：时间和显存图统一线性轴、测试元数据入图题、图 3/8 去重叠、图 6 四页重排 |
| `report_multigrid_quda_pyqcu_20261001.tex/.pdf` | test27 原始 combined JSON 的派生图表复核版：输出目录迁至 `data/report_multigrid_comprehensive_20261001/`，逐单元复核 20260930 数值 |
| `report_multigrid_quda_pyqcu_20261002.tex/.pdf` | 源码级架构、算法、并行、内存与接口多维对照版：分别剖析 PyQCU/QUDA，重算 test27 的迭代/单迭代成本分解并给出后续优化路线 |
| `plans/2026-09-16-multigrid-profiling-benchmark.md` | 本轮任务的实现与验证计划 |
| `plans/2026-10-02-multigrid-quda-comparison-report.md` | 20261002 报告的审计、派生图表、编译和全页验证计划 |
| `ORGANIZATION.md` | `docs/`、`data/`、`logs/`、`pyqcu/testing/` 的统一分类规则、垃圾清理与构建树例外 |
| `form-audit-20260928.md` | 2026-09-28 全库 form 审计：冲突表、本地例外、验证证据和恢复方法 |

## 目录分类

- `docs/` 顶层只保存正式文档：`.pdf`、`.tex`、`.md`；
  `张鑫 508应用测试报告-PyQCU/` 是解包 Office 资源，按文档资产保留。
- `docs/data/` 保存从 `data/**` 提升出的 `.md`/`.pdf`/`.tex`；tag 日志报告
  保留在 `logs/<tag>/`，不保留旧路径软链接。
- LaTeX 辅助文件、`.report-build-*`、逐页渲染图、agent/codex 会话和临时缓存
  均属于可再生产物，直接删除，不纳入归档。
- Office 文件及非文档资源不在 `docs/` 顶层保存；现有 `.docx`/`.xlsx` 的实体
  位于 `data/docs/`，调用方直接引用实体路径。
