# AGENTS.md — cpp.cuda.qcu.src

CUDA 内核源文件。每个 `.cu` `#include` `../include/` 的对应头，提供模板实例化与内核启动封装。

## 文件

| 文件 | 用途 |
|---|---|
| `apply_init.cu` / `apply_end.cu` | 内存分配/释放生命周期 |
| `apply_dslash.cu` | dslash 分发（按 plan 选 Wilson/Clover） |
| `wilson_dslash.cu` | Wilson dslash 内核 |
| `clover_dslash_single.cu` / `clover_dslash_multi.cu` / `clover_dslash_comm.cu` | Clover dslash：单卡/多卡/halo 交换 |
| `apply_wilson_bistabcg.cu` / `apply_wilson_bistabcg_dslash.cu` | Wilson BiStabCG + 其 dslash |
| `apply_wilson_cg.cu` / `apply_wilson_cg_dslash.cu` | Wilson CG + 其 dslash |
| `apply_clover_bistabcg.cu` / `apply_clover_bistabcg_dslash.cu` | Clover BiStabCG + 其 dslash |
| `apply_multigrid.cu` | MG restrict/prolong/粗 dslash |
| `apply_clover_multigrid.cu` | Clover 多重网格求解器入口（C API 桥） |
| `lattice_mpi.cu` | MPI halo 交换助手 |
| `lattice_cuda.cu` | CUDA 工具函数 |
