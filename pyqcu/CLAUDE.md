# CLAUDE.md — pyqcu

Python package for QCU: CUDA-accelerated lattice QCD library. Implements Wilson/Clover Dirac operators, BiStabCG and multigrid solvers, stout smearing, and gauge field generation — all MPI-distributed across a 4D process grid.

## Two-Layer Architecture

1. **Pure Python** (`dslash/`, `solver/`, `smear/`) — PyTorch-based implementations for CPU, CUDA GPU, or Ascend NPU (via `pyqcu.cann`).
2. **C++ CUDA backend** (`cuda/` → `cpp/cuda/qcu/`) — Hand-tuned CUDA kernels with MPI halo exchange, accessed through a Cython bridge.

## Subpackages

| Package | Purpose |
|---------|---------|
| `lattice/` | Gamma matrices, Gell-Mann matrices, SU(3) checks, gauge field generation |
| `dslash/` | Wilson & Clover Dirac operators, hopping/sitting decomposition, even-odd preconditioning |
| `solver/` | BiStabCG and multigrid (AMG) solvers |
| `smear/` | Stout gauge field smearing |
| `tools/` | MPI grid helpers, HDF5 I/O, einsum (TileLang), linear algebra, multigrid prolong/restrict |
| `testing/` | Integration tests for all components |
| `cuda/` | Cython bridge to libqcu.so + parameter constants |
| `cann/` | Torch compatibility layer for Ascend NPU (complex ops decomposition) |
| `dtk/` | Placeholder for DCU/ROCm backend (no implementation yet) |
| `maca/` | Placeholder for Maca backend (no implementation yet) |

## Key Convention

All code imports `pyqcu.cann as _torch` instead of `torch` directly. On CUDA/CPU it delegates to torch; on NPU it decomposes complex ops into real/imaginary parts since Ascend NPU doesn't natively support complex tensors.
