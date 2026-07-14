# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

PyQCU is a Python/Cython wrapper for QCU, a CUDA-accelerated lattice Quantum Chromodynamics (QCD) library. It implements Wilson and Clover Dirac operators, BiStabCG and multigrid solvers, stout smearing, and gauge field generation — all MPI-distributed across a 4D process grid.

**Dependencies:** Python ≥ 3.10, PyTorch, Cython, mpi4py, h5py, TileLang, numpy, CUDA toolkit.

## Build & Run

```bash
# Setup environment (LD_LIBRARY_PATH, PYTHONPATH)
source ./env.sh

# Build C++ CUDA backend → libqcu.so
bash ./build.sh

# Build Cython extension (pyqcu.cuda.qcu) in-place
bash ./install.sh

# Run examples/tests
cd examples && pytest .

# Run a single test file
mpirun -np 4 python examples/pyqcu/conftest.py

# Profile with perfetto
cd examples/profiler && mpirun -np 1 python -u conftest.py
# Load resulting trace_*.json into https://ui.perfetto.dev
```

The C++ CUDA backend lives at `cpp/cuda/qcu/` (source in `src/`, headers in `include/`). Building it also compiles `cpp/cuda/qcu/python/pyqcu.h` into `libqcu.so`. To switch between NVIDIA (`CMakeLists-nv.txt`) and DCU (`CMakeLists-dcu.txt`), edit the symlink in `make.sh`.

## Architecture

```
pyqcu/
├── lattice/     — Gamma matrices, Gell-Mann matrices, SU(3) checks, gauge field generation
├── dslash/      — Wilson & Clover Dirac operators, even-odd preconditioning, coarse-grid operators
├── solver/      — BiStabCG, multigrid (AMG) solver
├── smear/       — Stout gauge field smearing
├── tools/       — MPI grid helpers, I/O (HDF5+MPIO), einsum (TileLang JIT), linalg, multigrid prolong/restrict
├── testing/     — Integration tests exercising all components
├── cuda/        — Cython bridge (qcu.pyx/.pxd) + parameter constants (define.py) for the C++ CUDA backend
├── cann/        — Torch compatibility layer for Ascend NPU (handles complex ops not natively supported on NPU)
├── dtk/         — stub
└── maca/        — stub
cpp/
├── cuda/qcu/    — Primary backend: CUDA kernels, MPI halo exchange, solvers
├── cann/qcu/    — Stub for Ascend CANN backend
├── dtk/qcu/     — Stub for DCU (ROCm/HIP) backend
└── maca/qcu/    — Stub for Maca backend
```

Reference docs live in `docs/` — `dims.md` (dimension naming), `env.md` (Python environment setup), `install.md`, `examples.md`, `profiler.md`.

### How tests work

Tests are defined as Python functions in `pyqcu/testing/__init__.py` (e.g. `test_lattice`, `test_dslash_wilson`, `test_solver`, `test_smear_stout`). Each `examples/*/conftest.py` acts as a pytest entry point that imports from `pyqcu.testing` and calls specific test functions. The `conftest.py` files are manually edited to uncomment the test(s) to run. Run them with:

```bash
cd examples && pytest .                         # all conftest.py files
mpirun -np 4 python examples/pyqcu/conftest.py  # single file with MPI
```

Example subdirectories by backend:
- `examples/pyqcu/` — pure-Python operator/solver tests
- `examples/qcu/` — C++ CUDA backend tests via Cython bridge
- `examples/cpu/` — CPU-only tests
- `examples/npu/` — Ascend NPU tests
- `examples/dcu/` — DCU (ROCm/HIP) tests
- `examples/profiler/` — perfetto tracing
- `examples/benchmark/` — performance benchmarks
- `examples/tilelang/` — TileLang kernel tests

### Key design patterns

**Hardware abstraction via `pyqcu.cann`:** All code imports `pyqcu.cann as _torch` instead of using `torch` directly. For CUDA/CPU devices, `pyqcu.cann` delegates to `torch`. For NPU (`device.type == 'npu'`), it decomposes complex operations into real/imaginary parts because Ascend NPU doesn't natively support complex tensor ops. Use `pyqcu.cann` functions (`_torch.einsum`, `_torch.norm`, `_torch.roll`, etc.) anywhere complex tensors might run on NPU. Set `pyqcu.cann.force_use_npu = True` to test NPU code paths on CPU without NPU hardware.

**Parity (even-odd) preconditioning:** `tools.oooxyzt2poooxyzt` converts a standard layout tensor to parity-split `[p=2, ...]`. `tools.poooxyzt2oooxyzt` reverses it. The "p" prefix on dimension order strings means "parity-split". The `dslash.operator` class provides both full (`matvec`) and parity-preconditioned (`matvec_parity`, `matvec_eeo`, `matvec_oeo`) interfaces.

**MPI grid:** The 4D process grid is auto-factored from `MPI.COMM_WORLD` size via prime factorization (`tools.give_grid_size()`). Neighbor ranks are computed by `tools.give_rank_plus`/`give_rank_minus`. Halo exchange uses `MPI.Sendrecv` with contiguous CPU buffers. HDF5 I/O (`tools._io`) supports both MPI parallel I/O (`h5py` with `driver='mpio'`) and serial gather/scatter fallback, auto-detected via `tools.HAS_MPI_SUPPORT`.

**Coarse-grid operator construction:** `dslash.operator.__init__` with `fine_hopping`/`fine_sitting`/`local_ortho_null_vecs` builds the coarse-grid operator by explicitly applying the fine operator to each basis vector of the null space and restricting the result (Galerkin projection). The multigrid solver (`solver.multigrid`) supports:
- An adaptive level-back mechanism that drops to the coarsest level when convergence stalls
- Optional CUDA acceleration at the finest level via the C++ backend (`with_cuda_qcu=True` when `clover_ee_inv` and `clover_oo_inv` are provided)
- Configurable degrees of freedom, data types, and devices per level

### Data layout conventions

| Tensor | Shape | Notes |
|--------|-------|-------|
| Gauge field (U) | `[3, 3, 4, Lx, Ly, Lz, Lt]` | `[color, color, direction, x, y, z, t]` |
| Fermion field | `[4, 3, Lx, Ly, Lz, Lt]` | `[spin, color, x, y, z, t]` |
| Clover term | `[4, 3, 4, 3, Lx, Ly, Lz, Lt]` | `[spin, color, spin, color, x, y, z, t]` |
| Parity-split (prefix `p`) | `[2, ...original...]` | `p=0` is even sites, `p=1` is odd |
| Even-odd clover | `[12, 12, Lx, Ly, Lz, Lt]` | Flattened spin×color index |

The dimension order convention uses letters: `s`=spin, `c`=color, `d`=direction, `p`=parity, `x/y/z/t`=spacetime. See `docs/dims.md` for the full naming scheme.

### TileLang integration

`pyqcu/tools/_einsum.py` contains JIT-compiled TileLang kernels for specific einsum patterns (e.g., `Eexyzt_exyzt2Exyzt`). These are compiled for CUDA at import time with `@jit(target="cuda")`. The `_matul.py` module provides TileLang-based matrix multiply kernels for both GPU and CPU. TileLang kernels use `warp_size` (128) from `tools._define` for GPU launch configuration.

### C++ backend interface

The Cython bridge (`pyqcu/cuda/qcu/qcu.pyx`) exposes C++ functions that take raw data pointers. All arguments are passed as `long long` (pointer cast from `tensor.contiguous().data_ptr()`). Parameters are packed into two flat tensors:
- `params` (int32, size 54): lattice dimensions, grid sizes, data types, iteration counts
- `argv` (float, size 8): physical parameters (mass/tolerance/sigma)
- `set_ptrs` (int64): scratch pointers for the C++ runtime

The `pyqcu/cuda/define.py` file mirrors the C++ header `cpp/cuda/qcu/include/define.h` — indices into `params` and `argv` must stay in sync.

**Plan system:** The C++ backend uses `_SET_PLAN_` to select which kernel plan to execute:
- `-2`: laplacian
- `-1`: gauss gauge generation
- `0`: Wilson dslash
- `1`: BiStabCG / CG (and their dslash)
- `2`: Clover dslash

Each plan manages its own scratch buffers; `applyInitQcu` / `applyEndQcu` allocate/free them. Increment `_SET_INDEX_` between calls to avoid buffer reuse conflicts.
