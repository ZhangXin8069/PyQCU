# AGENTS.md — cpp.cuda.qcu.include

CUDA 后端的 C++ 头文件：26 个模板化头文件，CUDA 内核内联于头中。

## 关键头

| 头文件 | 用途 |
|---|---|
| `define.h` | 参数索引常量、块大小 — **必须镜像 `pyqcu/cuda/define.py`** |
| `lattice_complex.h` | GPU 复数运算（`operator*=` 已修复覆盖写 bug） |
| `lattice_set.h` | 格点几何、网格布局、site 索引（网格维用 ceiling 除法） |
| `lattice_cuda.h` | CUDA stream 管理、设备工具 |
| `lattice_mpi.h` | MPI halo 交换（阻塞 `MPI_Sendrecv`） |
| `qcu.h` | 顶层 include 聚合器 |
| `dslash.h` / `wilson_dslash.h` / `clover_dslash.h` | dslash 分发与内核 |
| `lattice_wilson_dslash.h` / `lattice_clover_dslash.h` | dslash 实现 |
| `bistabcg.h` / `cg.h` | BiCGStab / CG 算法内核 |
| `lattice_wilson_bistabcg.h` / `lattice_wilson_cg.h` / `lattice_clover_bistabcg.h` | 求解器封装 |
| `lattice_clover_multigrid.h` | Clover 多重网格 V-cycle（~1100 行，5-stream） |
| `lattice_multigrid.h` / `multigrid.h` | MG restrict/prolong/粗 dslash |
| `laplacian.h` / `lattice_laplacian.h` | Laplacian |
| `gauss_gauge.h` | 高斯规范场生成 |

头文件对应 `../src/` 中的 `.cu` 文件（#include 并实例化模板）。
