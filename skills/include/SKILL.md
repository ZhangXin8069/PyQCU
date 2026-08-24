---
name: include
description: cpp/cuda/qcu/include 目录的完整生成 skill：26 个模板化 CUDA 头文件（内核内联），define.h 须镜像 pyqcu/cuda/define.py。
---
# CLAUDE.md — cpp/cuda/qcu/include

C++ header files for the CUDA backend. 26 templated headers containing CUDA kernel implementations (kernels are inline in headers).

## Key Headers

| Header | Purpose |
|--------|---------|
| `define.h` | Parameter index constants, block size — must mirror `pyqcu/cuda/define.py` |
| `lattice_complex.h` | Complex number arithmetic on GPU |
| `lattice_set.h` | Lattice geometry, grid layout, site indexing (use ceiling division for grid dims) |
| `lattice_cuda.h` | CUDA stream management, device utilities |
| `lattice_mpi.h` | MPI halo exchange (blocking `MPI_Sendrecv`) |
| `qcu.h` | Top-level include aggregator |
| `dslash.h` | Dslash dispatch (Wilson vs Clover) |
| `wilson_dslash.h` | Wilson dslash kernel |
| `clover_dslash.h` | Clover dslash dispatch |
| `lattice_wilson_dslash.h` | Wilson dslash implementation |
| `lattice_clover_dslash.h` | Clover dslash implementation |
| `bistabcg.h` / `cg.h` | BiCGStab and CG algorithm kernels |
| `lattice_wilson_bistabcg.h` / `lattice_wilson_cg.h` | Wilson solver wrappers |
| `lattice_clover_bistabcg.h` | Clover BiStabCG solver |
| `lattice_clover_multigrid.h` | Clover multigrid V-cycle (~1100 lines, 5-stream architecture) |
| `lattice_multigrid.h` / `multigrid.h` | Multigrid restrict/prolong/coarse-dslash |
| `laplacian.h` / `lattice_laplacian.h` | Laplacian operator |
| `gauss_gauge.h` | Gaussian gauge field generation |

Headers correspond to `.cu` source files in `../src/` that `#include` them and instantiate the templates.

### lattice_clover_multigrid.h 性能机制（2026-08-25，dev84）

- CUDA Graph 段回放（8 迭代/段）——WSL2 ~300µs/内核执行税的对症解（成本单位是内核个数）
- 零拷贝映射内存标量传递 + 守卫标量内核（mg_give_1beta_rp / mg_give_1alpha /
  mg_give_1omega / mg_give_rx）、单块点积归约
- 粗解热启动 + r0_ref 锚定、SYNC DIET（削减冗余同步）、PROF 剖析计数开关
- `define.h` 同步项：`_MG_USE_DEFLATE_` / `_MG_MU_PRE_` / `_PARAMS_SIZE_=57`

实测收益（V100，`examples/qcu/dev84/dev84_report.md`）：粗解向量开销 3246→4ms（~800×）、
V-cycle 156→60ms（2.6×）。
