# AGENTS.md — pyqcu/testing/qcu/strict/quda_comparison

QUDA/PyQCU Strict-MultiGrid 对照工作区：算子约定锚定、分布式正确性探针、
性能矩阵编排与报告生成。所有脚本默认以仓库根为工作目录、产物写入
`data/`；QUDA 相关运行需先 `source pyqcu/testing/qcu/strict/quda_comparison/quda_env.sh`。

## 正确性探针（GPU，可 mpirun）

| 脚本 | 用途 |
|---|---|
| `strict_mpi_primitive_probe.py` | 粗层 full/compact 算子与单 rank 全局参考逐分量对照；`--dtype c64|c128`、`--grid` 支持 1/2/4 rank |
| `strict_mpi_setup_probe.py` | 分布式 Galerkin setup：按 rank 切片全局 gauge/null vector 建层级，与 1 rank 全局参考比较内部点与 rank 边界点 |
| `strict_mpi_solve_probe.py` | 端到端分布式求解 + 独立真残差；`--galerkin-mode` 可选 column/site-batch/colored/auto |
| `test_strict_mpi_preflight.py` | collective 一致性、几何与 capability 门禁（CPU） |

## 性能矩阵与报告

| 脚本 | 用途 |
|---|---|
| `bench_strict_vs_quda.py` | 单单元收集器：`--device {auto,v100,p100}`、`--mpi-ranks`、`--process-grid`、`--phases cold,warmup,steady`、`--levels 1..5`；输出含 `mg_levels`（六类 phase 耗时/迭代数）与 `iteration_semantics` |
| `bench_mg_matrix_full.py` | 完整笛卡尔矩阵编排（144 side-case / 1152 phase record）：`--list/--dry-run/--execute/--resume/--summarize`，逐单元解析 gauge/nullvec/QIO 资产并隔离 strict cache |
| `assemble_final_matrix.py` | 合并最终 PyQCU/QUDA side 文档并 fail-closed 审计 config/input/warmup/fair；导出 `units.csv`、`stages.csv`、`references.csv` 与 72 个 combined JSON |
| `build_mg_report.py` | JSON/trace → `mg_matrix.csv`、`mg_levels.csv`、SVG/PDF 图、`tables.tex`、`summary_analysis.json` |
| `plot_mg_absolute_times.py` | combined JSON → 绝对耗时散点/分组图、逐层阶段分页图、最粗层表与图例置底残差图；保留原始数据只读 |
| `convert_full_nullvec_to_quda_qio.py` | canonical full null vector → QUDA QIO + v1 manifest（含 byte-exact round-trip 校验） |
| `p100_env.sh` | P100 运行环境（torch cu118 site-packages、`CUDA_VISIBLE_DEVICES=0,1`）；详见 `P100_NOTES.md` |

## 口径与约束

- 最细层“总迭代次数”= 外层右预条件 FGMRES/GCR 的 `total_iter`；smoother/restriction/prolongation/coarse 迭代单独记录，不得累加进总迭代。
- trace v3 在 `stage/residual` 之外新增 `iteration_count`（逐外层迭代、逐层增量）；解析需接受 v1/v2/v3。
- 多 rank 运行要求每层同一 process grid、aggregate 不跨 rank、local shape 各维为偶数；不满足时 fail-closed。
- 分布式 setup 的 face 交换目前经 CPU staging，只用于 setup 期；小格点探针会受往返延迟限制。

## 2026-09-28 全矩阵复测

- `bench_mg_matrix_full.py --pyqcu-cache-expect {any,miss,hit}` 显式控制
  PyQCU runtime cache 期望：cold-miss 批次用 `miss`，trace-on 复用
  trace-off 资产用 `hit`，初始生成/探索可用 `any`。
- 冻结矩阵器固定传 `--reference-warmups 2`，保证 MG-1 BiCGStab 参考记录
  始终满足正式的 1 cold / 2 warmup / 5 steady 契约。
- `build_mg_report.py` 现在导出 device-wide peak memory、未计时 sampler、
  workspace 与 allocator 峰值，并只把 trace-on MG-2/3 计入 residual curve
  覆盖分母；BiCGStab MG-1 继续单独记录。
- 2026-09-28 复测的精确结果、容量替代和报告位于
  `data/report_multigrid_comprehensive_20260928/` 与
  `docs/report_multigrid_quda_pyqcu_20260928.{tex,pdf}`。
  双 P100 c64 `16x32x32x48` 因显存/时限门限替代为 `16^4`；替代不得混入
  原格点公平加速比。

## 2026-09-29 绝对耗时修订

- `plot_mg_absolute_times.py` 仅读取 test27 的 `combined/*.json`，在
  `data/report_multigrid_comprehensive_20260929/report/` 生成分组秒数、
  逐层阶段、最粗层耗时和可读图，不改写 2026-09-28 原始结果。
- QUDA 的全零尾层只作占位，绘制时删除；最后一个有效层显式标为
  `/coarsest`，最粗层 `coarse_solver_seconds` 同时进入独立汇总表。
- 正式修订报告为
  `docs/report_multigrid_quda_pyqcu_20260929.{tex,pdf}`。
