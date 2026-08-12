---
name: qcu
description: cpp/cuda/qcu 目录的完整生成 skill：PyQCU 主 C++ CUDA 后端（hand-tuned CUDA 内核 + MPI halo 交换），含构建流程、参数协议、5-stream 架构与关键不变量。
---
# CLAUDE.md — cpp/cuda/qcu

Primary C++ CUDA backend for PyQCU. Hand-tuned CUDA kernels with MPI halo exchange for Wilson/Clover Dirac operators, BiStabCG/CG solvers, multigrid, and gauge field generation.

## Build

```bash
source ./env.sh       # CUDA toolkit paths, MPI, etc.
bash ./make.sh        # symlinks CMakeLists-nv.txt → CMakeLists.txt, then cmake + make
```

Output: `libqcu.so` — dynamically linked library loaded by the Cython bridge.

## Source Organization

```
include/          — 26 header files (templated C++ with CUDA kernels inline)
├── define.h      — Parameter index constants (must mirror pyqcu/cuda/define.py)
├── lattice_complex.h   — Complex number arithmetic (operator*= was fixed for overwrite bug)
├── lattice_set.h       — Lattice geometry, grid layout, site indexing
├── lattice_cuda.h      — CUDA utility functions (stream management, etc.)
├── lattice_mpi.h       — MPI halo exchange helpers (blocking Sendrecv)
├── qcu.h               — Top-level include aggregator
├── dslash.h            — Dslash dispatch
├── wilson_dslash.h     — Wilson dslash kernel
├── clover_dslash.h     — Clover dslash entry
├── lattice_wilson_dslash.h   — Wilson dslash implementation
├── lattice_clover_dslash.h   — Clover dslash implementation
├── bistabcg.h          — BiCGStab algorithm (GPU kernels)
├── cg.h                — Conjugate gradient algorithm
├── lattice_wilson_bistabcg.h — Wilson BiStabCG solver
├── lattice_wilson_cg.h        — Wilson CG solver
├── lattice_clover_bistabcg.h  — Clover BiStabCG solver
├── multigrid.h                — MG restrict/prolong/coarse-dslash
├── lattice_multigrid.h        — MG implementation
├── lattice_clover_multigrid.h — Clover multigrid solver (~1100 lines)
├── laplacian.h / lattice_laplacian.h — Laplacian operator
└── gauss_gauge.h              — Gaussian gauge field generation

src/              — .cu files that #include the headers and instantiate kernels
python/
└── pyqcu.h       — C API declarations (extern "C"); must match pyqcu/cuda/qcu/qcu.pxd
```

## Parameter Protocol

Parameters are passed from Python as flat arrays. Index constants in `include/define.h` must stay in sync with `pyqcu/cuda/define.py`:

- **`params`** (int32[54]): lattice dims, grid sizes, data types, iteration counts, plan selection, MG level configs
- **`argv`** (float[7]): mass, atol, sigma, MG tolerances
- **`set_ptrs`** (int64[100]): scratch buffer pointers

`_SET_PLAN_` (params[16]) selects the kernel plan:
- `-2` = Laplacian, `-1` = Gauss gauge, `0` = Wilson dslash, `1` = BiStabCG/CG, `2` = Clover dslash

## Clover Multigrid Stream Architecture (5 streams)

```
main (strm):   dslash operations (fine_dslash_op / coarse_dslash_op)
_a_:           dot(r_tilde,r) → give_1beta → give_p → give_s → give_r
_b_:           give_1rho_prev → give_x_o
_c_:           dot(t,s), convergence-check dot(r,r)
_d_:           dot(r_tilde,v) → give_1alpha → dot(t,t) → give_1omega
```

## Critical Invariants (from bug fixes)

1. **Scalars live only in `device_vals`** — no host→device scalar memcpy inside iteration loops
2. **Full stream sync at bottom of each iteration** — sync ALL 5 streams before next iteration
3. **`_send_tmp_` scratch for dot products** — cublasDot → scratch slot 7 → MPI_Allreduce → copy to target (never write cublasDot directly to target)
4. **`mpi_real_type<T>()` template** — dispatches `MPI_FLOAT`/`MPI_DOUBLE` per template type
5. **`run_mpi` uses blocking `MPI_Sendrecv`** — no `MPI_Wait` needed (only `run_mpi_non_block` requires it)

## Block Size

`_BLOCK_SIZE_` in `define.h`: use 8/16 for testing small lattices, 128 for NVIDIA production, 256 for AMD DCU production.

---

## Complete Skills (Agent-Produced Subdirectories)

The content of each subdirectory below was produced with Claude Code assistance. Per repo convention, the complete skill that generates that content is reproduced verbatim below (source: the subdirectory's own `CLAUDE.md`), so the full knowledge is available directly at this level.

### Complete Skill: `include/` (source: `include/CLAUDE.md`)

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

### Complete Skill: `src/` (source: `src/CLAUDE.md`)

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

### Complete Skill: `python/` (source: `python/CLAUDE.md`)

# CLAUDE.md — cpp/cuda/qcu/python

Python-facing C API declarations. This is the interface boundary between the C++ CUDA backend and the Python Cython bridge.

## Files

| File | Purpose |
|------|---------|
| `pyqcu.h` | C API header — 22 `extern "C"` functions taking raw pointers as `long long` |

This header must stay in exact sync with `pyqcu/cuda/qcu/qcu.pxd` (the Cython declaration file). Any mismatch causes silent memory corruption.

All functions take three parameter arrays:
- `set_ptrs` (int64[100]): scratch buffer pointers managed by C++ runtime
- `params` (int32[54]): lattice dims, grid sizes, data types, iteration counts, plan selection
- `argv` (float64[7]): mass, atol, sigma, MG tolerances

C++→Python data pointers are cast to `long long` from `tensor.contiguous().data_ptr()`.

### Complete Skill: `logs/` (source: `logs/CLAUDE.md`)

# CLAUDE.md — cpp/cuda/qcu/logs

Runtime output directory for the C++ CUDA backend (`cpp/cuda/qcu`). Holds generated log files produced by building, testing, and benchmarking the C++ backend.

## Contents

Currently empty. Logs written here may include:

- Build output from `bash ./make.sh` (compiler messages, linker output)
- Test output from running the C++ backend tests (e.g., `examples/qcu/conftest.clover.multigrid.py`)
- Performance / convergence reports

## Notes

- This is a local runtime directory. `cpp/cuda/qcu/logs/` is not tracked in git.
- The canonical location for development reports and test outputs is the repo-root `logs/` directory (see `logs/CLAUDE.md` for its file patterns). Only backend-local artifacts belong here.

