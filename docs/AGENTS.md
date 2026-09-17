# AGENTS.md — docs

PyQCU 参考文档。

## 文件

| 文件 | 内容 |
|---|---|
| `dims.md` | 维度命名方案（`s`=spin、`c`=color、`d`=direction、`p`=parity、`x/y/z/t`=时空）。`ccdxyzt`、`scxyzt`、`psctzyx` 等约定 |
| `env.md` | Python 环境设置 — 必需变量（`QUDA_PATH`、`LD_LIBRARY_PATH`、`PYTHONPATH`） |
| `install.md` | 安装指南 — build.sh + install.sh 工作流 |
| `examples.md` | 示例使用指南 — 运行测试与解读输出 |
| `profiler.md` | 剖析指南 — torch.profiler 与 Perfetto |
| `report_multigrid_quda_pyqcu_20260916.tex/.pdf` | 本轮 PyQCU/QUDA MultiGrid 分层 trace、三层优化与正式对照报告 |
| `report_multigrid_quda_pyqcu_20260917.tex/.pdf` | dev87 早期迭代：旧加速比撤回、double MG 修复、colored Galerkin 与正式协议复核 |
| `report_multigrid_distributed_20260918.tex/.pdf` | 分布式 Strict-MG：fine/compact/full halo、R/P coarse halo、distributed fused FGMRES、分布式 Galerkin setup 的实现、正确性与 V100 c64 性能矩阵；MG-1/c128/P100 的缺口逐条记录 |
| `plans/2026-09-16-multigrid-profiling-benchmark.md` | 本轮任务的实现与验证计划 |
