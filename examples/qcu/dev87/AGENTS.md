# AGENTS.md — examples/qcu/dev87

QUDA/PyQCU Strict-MultiGrid 对照工作区：算子约定锚定、分布式正确性探针、
性能矩阵编排与报告生成。所有脚本默认以仓库根为工作目录、产物写入
`data/`；QUDA 相关运行需先 `source examples/qcu/dev87/quda_env.sh`。

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
| `build_mg_report.py` | JSON/trace → `mg_matrix.csv`、`mg_levels.csv`、SVG/PDF 图、`tables.tex`、`summary_analysis.json` |
| `convert_full_nullvec_to_quda_qio.py` | canonical full null vector → QUDA QIO + v1 manifest（含 byte-exact round-trip 校验） |
| `p100_env.sh` | P100 运行环境（torch cu118 site-packages、`CUDA_VISIBLE_DEVICES=0,1`）；详见 `P100_NOTES.md` |

## 口径与约束

- 最细层“总迭代次数”= 外层右预条件 FGMRES/GCR 的 `total_iter`；smoother/restriction/prolongation/coarse 迭代单独记录，不得累加进总迭代。
- trace v3 在 `stage/residual` 之外新增 `iteration_count`（逐外层迭代、逐层增量）；解析需接受 v1/v2/v3。
- 多 rank 运行要求每层同一 process grid、aggregate 不跨 rank、local shape 各维为偶数；不满足时 fail-closed。
- 分布式 setup 的 face 交换目前经 CPU staging，只用于 setup 期；小格点探针会受往返延迟限制。
