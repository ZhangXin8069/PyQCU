# AGENTS.md — cpp/cuda/qcu/include

CUDA 后端的 C++ 头文件。26 个模板化头，CUDA 内核实现内联于头文件中。

## 关键头

| 头文件 | 用途 |
|---|---|
| `define.h` | 参数索引常量、block size — 必须镜像 `pyqcu/cuda/define.py` |
| `lattice_complex.h` | GPU 复数运算（operator*= 已修复覆盖 bug） |
| `lattice_set.h` | 格点几何、网格布局、格点索引（网格维度用向上取整除法） |
| `lattice_cuda.h` | CUDA 流管理、设备工具 |
| `lattice_mpi.h` | MPI halo 交换（阻塞 `MPI_Sendrecv`） |
| `qcu.h` | 顶层 include 聚合器 |
| `dslash.h` / `wilson_dslash.h` / `clover_dslash.h` | Dslash 分发与内核 |
| `lattice_wilson_dslash.h` / `lattice_clover_dslash.h` | Wilson/Clover dslash 实现 |
| `bistabcg.h` / `cg.h` | BiCGStab 与 CG 算法内核 |
| `lattice_wilson_bistabcg.h` / `lattice_wilson_cg.h` | Wilson 求解器封装 |
| `lattice_clover_bistabcg.h` | Clover BiStabCG 求解器 |
| `lattice_clover_multigrid.h` | Clover multigrid V-cycle（~1100 行，5-stream 架构） |
| `multigrid.h` / `lattice_multigrid.h` | MG restrict/prolong/粗网格 dslash |
| `laplacian.h` / `lattice_laplacian.h` | Laplacian 算子 |
| `gauss_gauge.h` | 高斯规范场生成 |

头文件与 `../src/` 中 #include 它们并实例化模板的 .cu 文件一一对应。
