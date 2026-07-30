# CLAUDE.md — cpp

C++ backend implementations for PyQCU. Each subdirectory targets a different GPU architecture.

## Backends

| Directory | Architecture | Status |
|-----------|-------------|--------|
| `cuda/qcu/` | NVIDIA CUDA | **Active** — primary production backend |
| `cann/qcu/` | Huawei Ascend CANN | Placeholder stub |
| `dtk/qcu/` | AMD DCU / ROCm (HIP) | Placeholder stub |
| `maca/qcu/` | Maca | Placeholder stub |

## Active Backend: cpp/cuda/qcu

The CUDA backend implements hand-tuned kernels for Wilson/Clover dslash, BiStabCG/CG solvers, multigrid V-cycle, and gauge field generation — all with MPI halo exchange across a 4D process grid. Accessed from Python through the Cython bridge in `pyqcu/cuda/qcu/`.

## Build

Each backend should have its own `env.sh` for compiler/linker paths and a `make.sh` or CMake-based build script. The active CUDA backend uses `CMakeLists-nv.txt` (symlinked to `CMakeLists.txt`) with cmake + make chaining.
