# PyQCU/QUDA MultiGrid 20261002 报告实现计划

**目标:** 在保持 `git tag test27` 原始数据不变的前提下，生成
`docs/report_multigrid_quda_pyqcu_20261002.tex/.pdf`，并建立可复现的
`data/report_multigrid_comprehensive_20261002/report/` 派生数据与图表。

**架构:** 报告分为 test27 口径、PyQCU 路径、QUDA 路径、跨维度源码对比、
定量性能归因、双方优劣、优化路线和复现附录。架构比较使用 TeX/TikZ，
性能比较使用从 test27 `combined/*.json` 确定性重算的 CSV/JSON/PDF/SVG。

**技术栈:** Python 标准库、matplotlib、LaTeX/XeLaTeX、TikZ、pytest。

**规格:** 用户 2026-10-02 请求；`git tag test27` commit
`04382801c003353ecd1bc6b9bc10f6db750bf74d`；PyQCU 源码和 vendored
QUDA 1.1.0 源码。

**全局约束:**

- 66 个 combined 精确单元、132 条 side record、1 cold / 2 warmup /
  5 steady 全部保持不变。
- 不重跑、不替换、不覆盖 test27 原始数据。
- 每个性能摘要由原始 JSON 重算，并与 20261001 派生表逐单元核对。
- 每个架构结论给真实源码路径和行号，无法由源码确证的结论标为推断或未验证。
- `Overfull=0`、`Float too large=0`，并逐页检查 PDF。

### Task 1: test27 派生分析与图表

**文件:**

- 新建: `pyqcu/testing/qcu/strict/quda_comparison/plot_mg_architecture_report.py`
- 测试: `pyqcu/testing/qcu/strict/quda_comparison/test_plot_mg_architecture_report.py`
- 产出: `data/report_multigrid_comprehensive_20261002/report/`

**接口:**

- 消费: `data/report_multigrid_comprehensive_20260928/final_protocol/combined/*.json`
- 产出: `unit_analysis.csv`、`group_analysis.csv`、`algorithm_profiles.csv`、
  `summary.json`、`source_hashes.json` 和五个 PDF/SVG/PNG 性能图。

- [x] **Step 1: 写失败测试**
      断言原始矩阵为 66/132、每个单元 1+2+5、20261001 与重算中位秒数一致、
      `R = iteration_ratio * per_iteration_cost_ratio`、输出文件非空。
- [x] **Step 2: 运行确认失败**
      运行: `pytest -q pyqcu/testing/qcu/strict/quda_comparison/test_plot_mg_architecture_report.py`
      预期: 因生成器不存在而失败。
- [x] **Step 3: 最小实现**
      实现确定性 JSON 核验、统计、CSV/JSON、matplotlib PDF/SVG。
- [x] **Step 4: 运行确认通过**
      运行: `python -B .../plot_mg_architecture_report.py --check`
      再运行: `pytest -q .../test_plot_mg_architecture_report.py`
      预期: 全部通过。
- [x] **Step 5: 交付检查**
      检查所有生成文件存在、非空、哈希稳定；原始 test27 文件哈希未变化。

### Task 2: 20261002 主报告

**文件:**

- 新建: `docs/report_multigrid_quda_pyqcu_20261002.tex`
- 产出: `docs/report_multigrid_quda_pyqcu_20261002.pdf`

**接口:**

- 消费: Task 1 的 CSV/JSON/图，20261001 的原始派生表和源码。
- 产出: 一份可在无会话上下文中独立阅读的正式中文报告。

- [x] **Step 1: 内容失败检查**
      对照用户四项要求扫描章节，缺项即视为失败。
- [x] **Step 2: 编写 TeX**
      加入算法、数据、并行、内存、接口五维比较，性能分解、双方优劣、
      超越原因、优化路线、证据索引和复现命令。
- [x] **Step 3: 两遍编译**
      XeLaTeX 两遍，要求无 TeX 错误、`Overfull=0`、`Float too large=0`。
- [x] **Step 4: PDF 全页验证**
      记录 `pages_expected=pages_actual=pages_rendered=pages_checked`，
      检查遮挡、裁切、安全区和页脚。
- [x] **Step 5: 内容一致性检查**
      验证标题、核心数字、test27 commit、66/132 和 source index 可检索。

### Task 3: 报告系列与规范收尾

**文件:**

- 修改: `docs/AGENTS.md`
- 检查: `docs/report_multigrid_quda_pyqcu_20261002.{tex,pdf}`
- 检查: `data/report_multigrid_comprehensive_20261002/report/**`

**接口:**

- 消费: Task 2 的最终产物。
- 产出: 文档索引、验证记录和干净 diff。

- [x] **Step 1: 更新文档索引**
      在 `docs/AGENTS.md` 登记 20261002 报告和 20261001 的后续关系。
- [x] **Step 2: 仓库规范检查**
      运行 `git diff --check`、相关 pytest、`bash skills/form/form-audit.sh`。
- [x] **Step 3: 定向复查**
      核对新增文件、原始 test27 文件无 diff、无临时构建垃圾。
- [x] **Step 4: 最终总结**
      报告产物、数值一致性、源码证据、版式闸门、未验证边界和下一步。
