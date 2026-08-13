# AGENTS.md — examples/qcu

C++ CUDA 后端测试（经 Cython 桥）。从 Python 驱动 `libqcu.so`。

## 测试文件

| 文件 | 测试内容 |
|---|---|
| `conftest.cuda.py` | CUDA 可用性与上下文 |
| `conftest.mpi.py` | MPI 网格设置与 halo 交换 |
| `conftest.wilson.bistabcg.py` | C++ 后端 Wilson BiStabCG |
| `conftest.wilson.bistabcg.dslash.py` | Wilson BiStabCG dslash 内核 |
| `conftest.wilson.cg.py` | Wilson CG 求解器 |
| `conftest.clover.py` | Clover 项构造 |
| `conftest.clover.bistabcg.py` | Clover BiStabCG 求解器 |
| `conftest.clover.bistabcg.dslash.py` | Clover BiStabCG dslash |
| `conftest.clover.multigrid.py` | Clover multigrid V-cycle 求解器 |

## 用法

```bash
mpirun -np 1 python examples/qcu/conftest.clover.multigrid.py
```

输出：收敛日志 → `logs/clover_multigrid.log`，性能报告 → `logs/clover_multigrid_report.log`

## Dev73_5 Multigrid Benchmark 套件

`mg_dev73_5_*.py` 系列脚本对 `applyCloverMultigridQcu` vs Clover BiStabCG 参考（`applyCloverBistabCgQcu`）做精度/格点/求解器参数扫描，产出 `logs/dev73_5.*`（报告、LaTeX 表、PNG 图）。脚本清单见 examples/qcu 目录。

## Dev74 进阶套件（大格子 + 资源统计 + 多线程构建）

`mg_dev74_*.py` 系列在 dev73_5 协议上扩展：

| 脚本 | 功能 |
|---|---|
| `mg_dev74_dslash.py` | `CudaSchurOp`：封装 `applyCloverBistabCgDslashQcu`（C++ Schur 奇偶算子，输入/输出 `[12,X,Y,Z,T/2]`），每实例独立 params 副本 + set_index 槽位，多线程安全 |
| `mg_dev74_layout_test.py` | C++ dslash 输入布局对照实验（vs Python `matvec_parity`） |
| `mg_dev74_stencil_mt.py` | 多线程 stencil build（探测点写集不相交，线程安全）+ 对照验证 |
| `mg_dev74_budget.py` | 显存/内存/磁盘预算模型（cold 53KB/V、warm 27KB/V 实测校准；`--fit`） |
| `mg_dev74_bench.py` | 本地（默认）/集群（`--cluster`）bench + 资源统计（cold/warm 显存、RSS、磁盘） |
| `mg_dev74_clean.py` | 干净测量（独立进程交叉计时）+ 资源统计 |
| `mg_dev74_verify.py` | 正确性验证（gauge/解/null_vecs + CudaSchurOp 对照） |
| `mg_dev74_collect.py` | 汇总 → `logs/dev74_results.json` |
| `mg_dev74_mktable.py` / `mg_dev74_plots.py` | LaTeX 表 / PNG 图 |
| `mg_dev74_cluster.sh` | 集群大格子运行（dry-run 默认，`RUN=1` 执行；16x32x32x32 单卡可行，16x32x32x64 需分阶段构建，24x32x32x64 需多卡） |

注意：`CudaSchurOp` 依赖 C++ 端 `applyCloverBistabCgDslashQcu`（已移除首尾全局 `cudaDeviceSynchronize`，见 `cpp/cuda/qcu/src/apply_clover_bistabcg_dslash.cu`）；多线程构建在单卡无收益（GPU 瓶颈），面向多卡/多节点集群。
