# CLAUDE.md — cpp/cuda/qcu/src

CUDA kernel source files. Each `.cu` file `#include`s the corresponding header from `../include/` and provides template instantiations and kernel launch wrappers.

## Files

| File | Purpose |
|------|---------|
| `apply_init.cu` / `apply_end.cu` | Memory allocation/free lifecycle |
| `apply_dslash.cu` | Dslash dispatch (Wilson or Clover based on plan) |
| `wilson_dslash.cu` | Wilson dslash kernel |
| `clover_dslash_single.cu` / `clover_dslash_multi.cu` / `clover_dslash_comm.cu` | Clover dslash: single-GPU, multi-GPU, halo exchange |
| `apply_wilson_bistabcg.cu` / `apply_wilson_bistabcg_dslash.cu` | Wilson BiStabCG solver + its dslash |
| `apply_wilson_cg.cu` / `apply_wilson_cg_dslash.cu` | Wilson CG solver + its dslash |
| `apply_clover_bistabcg.cu` / `apply_clover_bistabcg_dslash.cu` | Clover BiStabCG solver + its dslash |
| `apply_multigrid.cu` | MG restrict/prolong/coarse-dslash |
| `apply_clover_multigrid.cu` | Clover multigrid solver entry (C API bridge) |
| `lattice_mpi.cu` | MPI halo exchange helpers |
| `lattice_cuda.cu` | CUDA utility functions |
