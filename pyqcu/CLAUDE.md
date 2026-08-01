# CLAUDE.md — pyqcu

Top-level Python package for QCU: CUDA-accelerated lattice QCD library. Implements Wilson/Clover Dirac operators, BiStabCG and multigrid solvers, stout smearing, and gauge field generation — all MPI-distributed across a 4D process grid.

## Two-Layer Architecture

1. **Pure Python** (`dslash/`, `solver/`, `smear/`) — PyTorch-based implementations for CPU, CUDA GPU, or Ascend NPU (via `pyqcu.cann`).
2. **C++ CUDA backend** (`cuda/` → `cpp/cuda/qcu/`) — Hand-tuned CUDA kernels with MPI halo exchange, accessed through a Cython bridge (`pyqcu.cuda.qcu`).

The multigrid solver can mix both layers: finest-level smoothing via the C++ backend (`with_cuda_qcu=True`) and coarser levels in pure Python.

## Subpackages

| Package | Purpose |
|---------|---------|
| `lattice/` | Gamma matrices, Gell-Mann matrices, SU(3) checks, gauge field generation |
| `dslash/` | Wilson & Clover Dirac operators, hopping/sitting decomposition, even-odd preconditioning, coarse-grid Galerkin projection |
| `solver/` | BiCGStab(l) solver, adaptive multigrid (AMG) V-cycle solver, GMRES stub |
| `smear/` | Stout gauge field smearing (iterative, MPI-capable) |
| `tools/` | MPI grid helpers, HDF5 I/O (parallel + serial), einsum (TileLang JIT), linear algebra, multigrid restrict/prolong/null-vectors |
| `testing/` | Integration tests for all components |
| `cuda/` | Cython bridge to `libqcu.so` + parameter constants (`define.py`) |
| `cann/` | Torch compatibility layer for Ascend NPU (complex ops decomposition) |
| `dtk/` | Placeholder for DCU/ROCm backend (no implementation yet) |
| `maca/` | Placeholder for Maca backend (no implementation yet) |

## Key Convention

All code imports `pyqcu.cann as _torch` instead of `torch` directly. On CUDA/CPU it delegates to torch; on NPU it decomposes complex ops into real/imaginary parts (Ascend NPU doesn't natively support complex tensors).

## Data Layout Conventions

| Tensor | Shape | Notes |
|--------|-------|-------|
| Gauge field (U) | `[3, 3, 4, Lx, Ly, Lz, Lt]` | `[color, color, direction, x, y, z, t]` |
| Fermion field | `[4, 3, Lx, Ly, Lz, Lt]` | `[spin, color, x, y, z, t]` |
| Clover term | `[4, 3, 4, 3, Lx, Ly, Lz, Lt]` | `[spin, color, spin, color, x, y, z, t]` |
| Parity-split (prefix `p`) | `[2, ...original...]` | `p=0` is even sites, `p=1` is odd (prepended dim) |
| Flattened spin×color | `[12, ...]` or `[E, ...]` | E = degrees of freedom per site |

Spacetime dimensions are always the last four axes (`...xyzt` layout). Ward indices use negative indexing (`wards['x'] = -4`, `wards['t'] = -1`) to be robust against arbitrary prefix dimensions.

## Build & Run

```bash
source ./env.sh                # LD_LIBRARY_PATH, PYTHONPATH, MPI flags
bash ./build.sh                # build libqcu.so (C++ CUDA backend)
bash ./install.sh              # build Cython extension in-place

# Tests
cd examples && pytest .
mpirun -np 4 python examples/pyqcu/conftest.py
```

## Logging Convention

All modules use: `PYQCU::MODULE::SUBMODULE:\n message`
