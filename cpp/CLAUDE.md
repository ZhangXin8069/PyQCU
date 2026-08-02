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

---

## Complete Skills (Agent-Produced Subdirectories)

The content of each subdirectory below was produced with Claude Code assistance. Per repo convention, the complete skill that generates that content is reproduced verbatim below (source: the subdirectory's own `CLAUDE.md`), so the full knowledge is available directly at this level.

### Complete Skill: `cuda/` (source: `cuda/CLAUDE.md`)

# CLAUDE.md — cpp/cuda

CUDA backend container directory. The actual implementation lives in `qcu/`.

This directory exists to mirror the multi-backend structure (`cann/`, `dtk/`, `maca/`) and may contain shared CUDA utilities or a top-level CMakeLists.txt in the future.

### Complete Skill: `cann/` (source: `cann/CLAUDE.md`)

# CLAUDE.md — cpp/cann

Ascend CANN backend container directory. The actual (stub) implementation lives in `qcu/`.

Currently a placeholder — no active CANN C++ backend exists.

### Complete Skill: `dtk/` (source: `dtk/CLAUDE.md`)

# CLAUDE.md — cpp/dtk

DCU/ROCm backend container directory. The actual (stub) implementation lives in `qcu/`.

Currently a placeholder — no active DTK C++ backend exists.

### Complete Skill: `maca/` (source: `maca/CLAUDE.md`)

# CLAUDE.md — cpp/maca

Maca backend container directory. The actual (stub) implementation lives in `qcu/`.

Currently a placeholder — no active Maca C++ backend exists.

