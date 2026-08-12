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
