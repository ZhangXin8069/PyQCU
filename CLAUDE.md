# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

PyQCU is a Python/Cython wrapper for QCU, a CUDA-accelerated lattice Quantum Chromodynamics (QCD) library. It implements Wilson and Clover Dirac operators, BiStabCG and multigrid solvers, stout smearing, and gauge field generation — all MPI-distributed across a 4D process grid.

**Dependencies:** Python ≥ 3.10, PyTorch, Cython, mpi4py, h5py, numpy, CUDA toolkit. TileLang is optional (runtime try/except import; not in `setup.py`'s `install_requires`).

## Two-Layer Architecture

PyQCU has two execution layers that share the same algorithms but target different hardware:

1. **Pure Python** (`pyqcu/dslash/`, `pyqcu/solver/`, `pyqcu/smear/`) — PyTorch-based implementations that run on CPU, CUDA GPU, or Ascend NPU (via `pyqcu.cann`). Used for development, testing, and NPU deployment.

2. **C++ CUDA backend** (`cpp/cuda/qcu/`) — Hand-tuned CUDA kernels with MPI halo exchange. Accessed through a Cython bridge (`pyqcu/cuda/qcu/qcu.pyx`) that passes raw data pointers as `long long`. This is the production path for NVIDIA GPUs.

The multigrid solver can mix both layers: finest-level smoothing via the C++ backend (`with_cuda_qcu=True`) and coarser levels in pure Python.

## Build & Run

```bash
# Setup environment (LD_LIBRARY_PATH, PYTHONPATH)
source ./env.sh
# Also sets MPI_ALLOW_RUN_AS_ROOT=1, OMPI_ALLOW_RUN_AS_ROOT=1,
# OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 — needed for containerized/dev environments

# Build C++ CUDA backend → libqcu.so
bash ./build.sh
# (internally: cd cpp/cuda/qcu && source ./env.sh && bash ./make.sh)

# Build Cython extension (pyqcu.cuda.qcu) in-place
bash ./install.sh
# (internally: python setup.py build_ext --inplace && rm -rf ./build)

# Run examples/tests
cd examples && pytest .

# Run a single test file with MPI
mpirun -np 4 python examples/pyqcu/conftest.py

# Clover multigrid solver test
mpirun -np 1 python examples/qcu/conftest.clover.multigrid.py
```

**C++ backend build details:** `cpp/cuda/qcu/make.sh` uses `set -e` for error detection, symlinks `CMakeLists-nv.txt` (NVIDIA) or `CMakeLists-dcu.txt` (DCU/ROCm) to `CMakeLists.txt`, then runs cmake + make with `&&` chaining. The resulting `libqcu.so` exposes C functions declared in `cpp/cuda/qcu/python/pyqcu.h`. The C++ build sources its own `env.sh` for CUDA toolkit paths.

**Cython build details:** `setup.py` defines a single extension `pyqcu.cuda.qcu` from `pyqcu/cuda/qcu/qcu.pyx`, linking against `libqcu.so`. It also runs `build.sh` as a pre-build step via the custom `CMakeBuild` command class. Key setup.py details:
- `python_requires=">=3.8"` (PyTorch 2.x hard requirement)
- `find_packages(exclude=["pyqcu.testing.*", "pyqcu.testing"])` — correct pattern matching
- `package_data={"pyqcu.cuda.qcu": ["*.pyi"]}` — includes type stubs in wheel
- Imports only `Extension` from distutils, `setup` from setuptools (avoiding the distutils `setup` → setuptools `setup` override bug)

## Architecture

```
pyqcu/
├── lattice/     — Gamma matrices, Gell-Mann matrices, SU(3) checks, gauge field generation
├── dslash/      — Wilson & Clover Dirac operators, even-odd preconditioning, coarse-grid operators
├── solver/      — BiStabCG, multigrid (AMG) solver
├── smear/       — Stout gauge field smearing
├── tools/       — MPI grid helpers, I/O (HDF5+MPIO), einsum (TileLang JIT), linalg, multigrid prolong/restrict
├── testing/     — Integration tests exercising all components
├── cuda/        — Cython bridge (qcu.pyx/.pxd/.pyi) + parameter constants (define.py) + package init (__init__.py) for the C++ CUDA backend
├── cann/        — Torch compatibility layer for Ascend NPU (handles complex ops not natively supported on NPU)
├── dtk/         — stub for DCU/ROCm (no implementation yet)
└── maca/        — stub for Maca (no implementation yet)
cpp/
├── cuda/qcu/    — Primary backend: CUDA kernels, MPI halo exchange, solvers
├── cann/qcu/    — Stub for Ascend CANN backend (no implementation yet)
├── dtk/qcu/     — Stub for DCU (ROCm/HIP) backend (no implementation yet)
└── maca/qcu/    — Stub for Maca backend (no implementation yet)
examples/
├── pyqcu/       — Pure-Python operator/solver tests (main test suite)
├── qcu/         — C++ CUDA backend tests via Cython bridge
├── cpu/         — CPU-only tests
├── npu/         — Ascend NPU tests
├── dcu/         — DCU (ROCm/HIP) tests
├── profiler/    — Perfetto tracing
├── benchmark/   — Performance benchmarks
├── tilelang/    — TileLang kernel tests
└── data/        — Reference HDF5 files used for validation when with_data=True
```

Reference docs live in `docs/` — `dims.md` (dimension naming), `env.md` (Python environment setup), `install.md`, `examples.md`, `profiler.md`.

### Module-level code in `pyqcu/lattice/__init__.py`

The lattice module runs initialization code at import time (not lazy): it defines gamma matrices (γ₀…γ₃, γ₅ = γ₀γ₁γ₂γ₃), the six γ_μ γ_ν products stored in `gamma_gamma`, Gell-Mann matrices (λ₁…λ₈, SU(3) generators), and `ward` index mappings (`wards`, `ward_keys`, `ward_wards`). Ward indices use negative indexing (e.g., `wards['x'] = -4`) — see the convention note below. These are plain module-level tensors on CPU. The `check_su3()` function verifies unitarity, det=1, and minor identities. `generate_gauge_field()` creates random SU(3) gauge links via exponential map of random Gell-Mann combinations.

### `pyqcu.cann` — NPU compatibility layer

All Python code imports `pyqcu.cann as _torch` instead of using `torch` directly. This module wraps torch operations that fail on Ascend NPU (which doesn't natively support complex tensors):

- **CUDA/CPU path:** delegates straight to `torch.*`
- **NPU path** (`device.type == 'npu'` or `force_use_npu=True`): decomposes complex ops into real/imaginary parts

Set `pyqcu.cann.force_use_npu = True` to test NPU code paths on CPU without NPU hardware. Note: some modules (`dslash/_wilson.py`, `tools/_define.py`, `tools/_multigrid.py`, `smear/_stout.py`) also have their own `force_use_npu` module-level variable for per-module NPU debugging; the global `pyqcu.cann.force_use_npu` controls the shared `cann` layer only.

Functions provided (always use these instead of raw torch calls anywhere complex tensors might run on NPU):
- **Math:** `abs`, `vdot`, `norm`, `sqrt`, `matmul`
- **Reduction/shape:** `roll`, `allclose`, `einsum`
- **Creation:** `zeros`, `zeros_like`, `randn`, `randn_like`, `eye`
- **Linear algebra:** `linalg_qr` (falls back to CPU on NPU for complex inputs)

### `dslash.operator` — assembled Dirac operator

The `dslash.operator` class (`pyqcu/dslash/_operator.py`) composes two sub-operators:

- **`hopping`** — the Wilson hopping term D_w (spatial derivative). On init, precomputes M_plus/M_minus matrices for each of 4 directions by calling `dslash.give_hopping_plus`/`give_hopping_minus`. If `support_parity=True`, also splits them into even/odd sub-blocks (M_e_plus, M_o_plus, etc.) via `tools.oooxyzt2poooxyzt`. Performs MPI halo exchange for gauge field boundaries on init and for fermion field boundaries on each matvec.

- **`sitting`** — the Clover term (chromo-magnetic field strength contribution). Computes `M = I + clover_term` via `dslash.add_I`, optionally splitting into even/odd and precomputing inverses for preconditioning.

The `operator.matvec()` returns `hopping.matvec() + sitting.matvec()`. The `matvec_eo()`/`matvec_oe()` methods handle even-odd preconditioned application with explicit MPI Sendrecv for halo exchange.

### Coarse-grid operator construction (Galerkin projection)

When `fine_hopping`, `fine_sitting`, and `local_ortho_null_vecs` are all provided to `dslash.operator.__init__`, it builds a coarse-grid operator. For each null-space basis vector `e` and each direction:
1. Prolong a delta-source from coarse to fine grid
2. Apply the fine hopping operator (plus and minus directions)
3. Restrict the result back to the coarse grid
4. Also project the fine sitting operator

This Galerkin projection `P^T D_fine P` yields the coarse-grid hopping and sitting matrices. See `_operator.py` lines 153–217 for the full construction.

### C++ backend: Plan system and parameter protocol

The C++ backend uses `_SET_PLAN_` (params index 16) to select which kernel plan to execute:

| Plan | Value | Purpose |
|------|-------|---------|
| `_SET_PLAN_N_2_` | -2 | Laplacian |
| `_SET_PLAN_N_1_` | -1 | Gauss gauge generation |
| `_SET_PLAN0_` | 0 | Wilson dslash |
| `_SET_PLAN1_` | 1 | BiStabCG / CG (and their dslash) |
| `_SET_PLAN2_` | 2 | Clover dslash |

**Critical:** The C++ backend call lifecycle is:

```python
qcu.applyInitQcu(set_ptrs, params, argv)          # allocate buffers
# ... perform operations ...
params[define._SET_INDEX_] += 1                     # MUST increment between calls
qcu.applyEndQcu(set_ptrs, params)                   # free buffers
```

Failing to increment `_SET_INDEX_` between successive C++ calls causes scratch buffer reuse conflicts that produce wrong results. `applyInitQcu` allocates buffers; `applyEndQcu` frees them.

Parameters are passed as three flat tensors whose Python-side indices are defined in `pyqcu/cuda/define.py` (must stay in sync with `cpp/cuda/qcu/include/define.h`):
- **`params`** (int32, size 54): lattice dims, grid sizes, data types, iteration counts, plan selection, multigrid level configs
- **`argv`** (float, size 7): physical parameters — mass (idx 0), atol (1), sigma (2), per-level MG tolerances (3–6)
- **`set_ptrs`** (int64, size 100): scratch pointers managed by the C++ runtime

All C functions take raw pointers cast to `long long` from `tensor.contiguous().data_ptr()`. See `pyqcu.h` for the full C API surface. A type stub file `pyqcu/cuda/qcu.pyi` provides full type annotations, docstrings with tensor layout conventions, and default parameter values — useful for IDE support.

**Data type mapping:** `pyqcu/cuda/define.py` provides `dtype(_data_type_)` to convert between QCU's internal data type constants (`_LAT_C64_`, `_LAT_R32_`, etc.) and PyTorch dtypes, and `epytd(torch_dtype)` for the reverse mapping.

### Python-level C API (Cython bridge)

The Cython module `pyqcu.cuda.qcu` exposes these functions (each takes raw tensor pointers):

| Function | Purpose |
|----------|---------|
| `applyInitQcu` / `applyEndQcu` | Allocate / free scratch buffers |
| `applyWilsonDslashQcu` | Wilson dslash (plan 0) |
| `applyCloverDslashQcu` | Clover dslash (plan 2) |
| `applyWilsonBistabCgQcu` / `applyWilsonBistabCgDslashQcu` | Wilson BiStabCG solver + its dslash (plan 1) |
| `applyWilsonCgQcu` / `applyWilsonCgDslashQcu` | Wilson CG solver + its dslash (plan 1) |
| `applyCloverBistabCgQcu` / `applyCloverBistabCgDslashQcu` | Clover BiStabCG (needs clover_ee/oo and their inverses) |
| `applyCloverQcu` / `applyCloversQcu` | Build Clover term (and its inverse) |
| `applyDslashQcu` | Combined Wilson+Clover dslash in one call |
| `applyLaplacianQcu` | Laplacian operator (plan -2) |
| `applyGaussGaugeQcu` | Gaussian gauge field generation (plan -1) |
| `applyMultigridRestrictQcu` / `applyMultigridProLongQcu` | MG restrict/prolong with null vectors |
| `applyMultigridCoarseDslashQcu` | Coarse-grid dslash (hopping + sitting) |
| `applyCloverMultigridQcu` | Full Clover multigrid V-cycle solver |

### C++ backend internal structure

The CUDA backend (`cpp/cuda/qcu/src/`) is organized by operator type:
- `apply_init.cu` / `apply_end.cu` — memory allocation/free lifecycle
- `apply_dslash.cu` — dispatches to Wilson or Clover dslash based on plan
- `wilson_dslash.cu` — Wilson dslash kernel
- `clover_dslash_*.cu` — Clover dslash: `single` (single-GPU), `multi` (multi-GPU), `comm` (halo exchange)
- `apply_wilson_bistabcg.cu` / `apply_wilson_bistabcg_dslash.cu` — Wilson BiStabCG solver + its dslash
- `apply_clover_bistabcg.cu` / `apply_clover_bistabcg_dslash.cu` — Clover BiStabCG solver + its dslash
- `apply_wilson_cg.cu` / `apply_wilson_cg_dslash.cu` — Wilson CG solver + its dslash
- `apply_multigrid.cu` — multigrid restrict/prolong/coarse-dslash
- `apply_clover_multigrid.cu` — Clover multigrid solver entry point (C API bridge)
- `lattice_mpi.cu` — MPI halo exchange helpers
- `lattice_cuda.cu` — CUDA utility functions (stream management, etc.)

### Parity (even-odd) preconditioning

`tools.oooxyzt2poooxyzt` converts a standard layout tensor to parity-split `[p=2, ...]`. `tools.poooxyzt2oooxyzt` reverses it. The "p" prefix on dimension order strings means "parity-split". The `dslash.operator` class provides both full (`matvec`) and parity-preconditioned (`matvec_parity`, `matvec_eeo`, `matvec_oeo`) interfaces.

### MPI grid

The 4D process grid is auto-factored from `MPI.COMM_WORLD` size via prime factorization (`tools.give_grid_size()`). Neighbor ranks are computed by `tools.give_rank_plus`/`give_rank_minus`. Halo exchange uses `MPI.Sendrecv` with contiguous CPU buffers. HDF5 I/O (`tools._io`) supports both MPI parallel I/O (`h5py` with `driver='mpio'`) and serial gather/scatter fallback, auto-detected via `tools.HAS_MPI_SUPPORT`.

### Multigrid solver

The multigrid solver (`solver.multigrid`) supports:
- Adaptive level-back mechanism that drops to the coarsest level when convergence stalls
- Optional CUDA acceleration at the finest level via `with_cuda_qcu=True` (enabled automatically when `clover_ee_inv` and `clover_oo_inv` are provided)
- Configurable degrees of freedom, data types, and devices per level
- Null vector generation via inverse iteration (`tools.give_null_vecs`)
- Local orthogonalization of null vectors (`tools.local_orthogonalize`)
- Coarse-grid restrict/prolong with optional CUDA acceleration (`applyMultigridRestrictQcu`/`applyMultigridProLongQcu`)

Note: `pyqcu/solver/_gmres.py` is a placeholder stub — the GMRES solver is planned but not yet implemented. The current solver suite consists of BiStabCG and multigrid only.

### Data layout conventions

| Tensor | Shape | Notes |
|--------|-------|-------|
| Gauge field (U) | `[3, 3, 4, Lx, Ly, Lz, Lt]` | `[color, color, direction, x, y, z, t]` |
| Fermion field | `[4, 3, Lx, Ly, Lz, Lt]` | `[spin, color, x, y, z, t]` |
| Clover term | `[4, 3, 4, 3, Lx, Ly, Lz, Lt]` | `[spin, color, spin, color, x, y, z, t]` |
| Parity-split (prefix `p`) | `[2, ...original...]` | `p=0` is even sites, `p=1` is odd |
| Even-odd clover | `[12, 12, Lx, Ly, Lz, Lt]` | Flattened spin×color index |

HDF5 I/O uses dimension order `zyxt` (fastest to slowest: t, z, y, x) internally. Conversion functions `ccdxyzt2ccdptzyx` and `scxyzt2psctzyx` handle the reordering between the tensor layout and the file layout. See `docs/dims.md` for the full naming scheme.

The dimension order convention uses letters: `s`=spin, `c`=color, `d`=direction, `p`=parity, `x/y/z/t`=spacetime.

**Negative ward index convention:** All tensors in PyQCU follow the `...xyzt` layout — spacetime dimensions are always the last four axes. Ward indices use negative integers (e.g., `wards['x'] = -4`) so they correctly index the spacetime axes regardless of how many prefix dimensions (spin, color, parity, etc.) the tensor has. This is by design, not a bug.

### TileLang integration

`pyqcu/tools/_einsum.py` contains JIT-compiled TileLang kernels for specific einsum patterns (e.g., `Eexyzt_exyzt2Exyzt`). These are compiled for CUDA at import time with `@jit(target="cuda")`. The `_matul.py` module provides TileLang-based matrix multiply kernels for both GPU and CPU. TileLang kernels use `warp_size` (128) from `tools._define` for GPU launch configuration.

Note: TileLang import is optional — the try/except in `tools/__init__.py` silently degrades if TileLang is unavailable.

## How tests work

Tests are defined as Python functions in `pyqcu/testing/__init__.py`:
- `test_lattice` — SU(3) gauge generation and validation
- `test_dslash_wilson` — Wilson Dirac operator (supports `with_data=True` to validate against reference HDF5 data from `examples/data/`)
- `test_dslash_parity` — Parity-preconditioned Wilson+Clover operator with MPI
- `test_dslash_clover` — Clover term construction
- `test_solver` — BiStabCG (`method='bistabcg'`) and multigrid (`method='multigrid'`) solver correctness
- `test_matmul` — TileLang vs PyTorch matmul benchmark
- `test_smear_stout` — Stout smearing

Each `examples/*/conftest.py` acts as a pytest entry point that imports from `pyqcu.testing` and calls specific test functions. The `conftest.py` files are manually edited to uncomment the test(s) to run. Run them with:

```bash
cd examples && pytest .                         # all conftest.py files
mpirun -np 4 python examples/pyqcu/conftest.py  # single file with MPI
```

Example subdirectories by backend:
- `examples/pyqcu/` — pure-Python operator/solver tests (main test suite)
- `examples/qcu/` — C++ CUDA backend tests via Cython bridge
- `examples/cpu/` — CPU-only tests
- `examples/npu/` — Ascend NPU tests
- `examples/dcu/` — DCU (ROCm/HIP) tests
- `examples/profiler/` — perfetto tracing
- `examples/benchmark/` — performance benchmarks
- `examples/tilelang/` — TileLang kernel tests
- `examples/data/` — reference HDF5 files (gauge fields, sources, expected results) used for validation when `with_data=True`

### Profiling

The profiler wraps operations with `torch.profiler.profile(...)` using `record_shapes=True`, `with_modules=True`, `with_flops=True`, and exports Chrome trace format:

```bash
cd examples/profiler && mpirun -np 1 python -u conftest.py
# Load resulting trace_*.json into https://ui.perfetto.dev
```

## Code conventions

**`Namespace.__module__` pattern:** Multiple `__init__.py` files set `Namespace.__module__` to their package name (e.g., `Namespace.__module__ = "pyqcu.dslash"`). This ensures that when test functions construct `argparse.Namespace` objects, they carry the correct module attribution for logging/tracing.

**Logging convention:** All modules use the pattern `PYQCU::MODULE::SUBMODULE:\n message` for print-based logging, controlled by `verbose` flags on most functions and classes.

**Device/dtype flexibility:** Most classes accept and preserve the device/dtype of their input tensors. When internal precomputed matrices are on a different device, explicit `.to()` casts are used.

## Clover Multigrid Solver (C++ CUDA)

The C++ backend now includes a full Clover multigrid solver (`cpp/cuda/qcu/include/lattice_clover_multigrid.h`, ~1100 lines). It performs V-cycle iterations with BiStabCG smoothing at each level, matching the Python MG algorithm in `pyqcu/solver/_multigrid.py` but using CUDA streams for parallelism.

**Stream architecture (5 streams):**

```
main (strm):   dslash operations (fine_dslash_op / coarse_dslash_op)
_a_:           dot(r_tilde,r) → give_1beta → give_p → give_s → give_r
_b_:           give_1rho_prev → give_x_o
_c_:           dot(t,s), convergence-check dot(r,r)
_d_:           dot(r_tilde,v) → give_1alpha → dot(t,t) → give_1omega
```

**Critical invariants** (failures here caused NaN and segfault bugs during development):

1. **Scalars live only in `device_vals`.** No host→device scalar memcpy inside the iteration loop. All scalar updates are done by GPU kernels writing directly to `device_vals`. This eliminated a race condition where host-side writes interleaved with kernel reads on other streams.

2. **Full stream sync at bottom of each iteration.** Sync ALL 5 streams before starting the next iteration. Missing syncs caused stale reads of `device_vals` slots and residual oscillation.

3. **`_send_tmp_` scratch pattern for dot products.** cublasDot writes to `_send_tmp_` (scratch slot index 7), then MPI_Allreduce, then copy to target slot. Never write cublasDot directly to the target slot — another stream could read the unreduced value during the MPI window.

4. **`mpi_real_type<T>()` template.** Dispatches `MPI_FLOAT` for float, `MPI_DOUBLE` for double. Hardcoding `MPI_FLOAT` breaks double-precision runs.

5. **`run_mpi` uses blocking `MPI_Sendrecv` — no `MPI_Wait` needed.** Only `run_mpi_non_block` (using `MPI_Isend`) requires `MPI_Wait`. Adding `MPI_Wait` to the blocking path causes segfault on uninitialized request handles.

**C API:** `applyCloverMultigridQcu` in `pyqcu.h` takes fermion in/out, gauge, clover_ee, clover_oo, clover_ee_inv, clover_oo_inv, set_ptrs, params. The Python entry point is `qcu.applyCloverMultigridQcu(...)` via Cython.

**Test:** `examples/qcu/conftest.clover.multigrid.py` — single-GPU correctness test at 8×8×8×16 lattice, outputs convergence log to `logs/clover_multigrid.log` and performance report to `logs/clover_multigrid_report.log`.

## Type Stub (pyi)

`pyqcu/cuda/qcu/qcu.pyi` (155 lines) provides full type annotations with docstrings for all 22 Cython bridge functions. New in R3 — enables IDE autocomplete and type checking for the CUDA backend. The stub covers: init/end, Wilson/Clover dslash, Wilson/Clover BiStabCG, Wilson CG, Laplacian, Gauss gauge, multigrid restrict/prolong/coarse-dslash, and Clover multigrid. `setup.py` includes it via `package_data`.

## Logging & Reports

The `logs/` directory contains development reports and debug artifacts. When adding new reports, place them here with date-based naming:

| Pattern | Purpose |
|---------|---------|
| `logs/dev<N>.md` / `.pdf` / `.tex` | Development milestone reports (between stab tags) |
| `logs/bug<N>.md` | Bug discovery & code review reports |
| `logs/review-*.md` | Code review findings |
| `logs/fix-report-*.md` | Bug fix summaries |
| `logs/debug/fix-log*.md` | Per-round fix logs |
| `logs/results/*.md` | Final/remaining fix reports |
| `logs/clover_multigrid.log` | C++ solver convergence output |
| `logs/*.png` | Performance charts, convergence plots |

**Tagging convention:** The project uses `cctag` (~/configure/bin/cctag) with three independent counters: `stab<N>` (stable milestones), `dev<N>` (development snapshots), `bug<N>` (bug fix markers). Tags are annotated with changelogs. The current baseline is `stab23`.

## Known Fixed Bugs (Anti-Patterns)

These bugs were found and fixed during the R1–R3 reviews. They represent patterns to avoid:

| Pattern | Example | Consequence |
|---------|---------|-------------|
| Complex `operator*=` overwriting `_data.x` before using it in `_data.y` | `lattice_complex.h` | All complex multiplication wrong |
| `cudaMallocAsync` buffer size mismatch with kernel write size | `gauss_gauge.cu` | OOB write |
| GPU buffer re-allocation overwriting existing pointers without `cudaFreeAsync` | `lattice_wilson_cg.h` | Memory leak |
| Integer division where ceiling division is needed for grid dimensions | `lattice_set.h` | Sites skipped |
| Bare `except:` swallowing `KeyboardInterrupt` | `_define.py`, `testing/__init__.py` | Unstoppable processes |
| `torch.linalg.inv` called in Python for-loop instead of batched | `_clover.py` | 10-50x slower |
| `self.sitting` (an object) used as truthy check instead of `self.sitting.clover_term is not None` | `_operator.py` | Always-true condition |
| Gathering MPI tuples in `(t,z,y,x)` order but unpacking as `(x,y,z,t)` | `_io.py` | Data written to wrong location |
| Stout smearing `nstep>1` loop not updating `U` between steps | `_stout.py` | nstep degraded to 1 |
| `python_requires=">=3.6"` when PyTorch 2.x needs ≥3.8 | `setup.py` | pip install fails on Py3.7 |

---

## Complete Skills (Agent-Produced Subdirectories)

The content of each subdirectory below was produced with Claude Code assistance. Per repo convention, the complete skill that generates that content is reproduced verbatim below (source: the subdirectory's own `CLAUDE.md`), so the full knowledge is available directly at this level.

### Complete Skill: `pyqcu/` (source: `pyqcu/CLAUDE.md`)

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

---

## Complete Skills (Agent-Produced Subdirectories)

The content of each subdirectory below was produced with Claude Code assistance. Per repo convention, the complete skill that generates that content is reproduced verbatim below (source: the subdirectory's own `CLAUDE.md`), so the full knowledge is available directly at this level.

### Complete Skill: `lattice/` (source: `lattice/CLAUDE.md`)

# CLAUDE.md — pyqcu.lattice

Lattice QCD fundamentals: gamma matrices, Gell-Mann matrices, SU(3) group utilities, and gauge field generation.

## Module-level Data (computed at import time, on CPU, complex64)

- **`gamma`** — 4×4×4 gamma matrices γ₀, γ₁, γ₂, γ₃ in the Dirac-Pauli representation (γ₀ anti-hermitian, γ_i hermitian). Shape `[4, 4, 4]`.
- **`gamma_5`** — γ₅ = γ₀γ₁γ₂γ₃. Shape `[4, 4]`.
- **`gamma_gamma`** — six γ_μ γ_ν products: [γ_x,γ_y], [γ_x,γ_z], [γ_x,γ_t], [γ_y,γ_z], [γ_y,γ_t], [γ_z,γ_t]. Shape `[6, 4, 4]`. Used as σ_{μν} matrices in the clover term.
- **`I`** — 4×4 identity matrix (complex64)
- **`minus_I`** — −I (precomputed)
- **`gell_mann`** — eight Gell-Mann matrices λ₁…λ₈ (SU(3) generators, traceless hermitian). Shape `[8, 3, 3]`. λ₁,λ₄,λ₆ are real; λ₂,λ₅,λ₇ are i×real.

## Ward Index Convention

Ward indices use **negative indexing** because spacetime dimensions are always the last four axes (`...xyzt` layout):

```python
wards['x'] = -4    # last 4th axis
wards['y'] = -3    # last 3rd axis
wards['z'] = -2    # last 2nd axis
wards['t'] = -1    # last axis
wards['t_p'] = -1  # parity-split temporal (same index as t)
```

This makes indexing robust regardless of prefix dimensions (spin, color, parity, etc.).

### Ward key lists
- **`ward_keys`** = `['x', 'y', 'z', 't']` — standard 4D directions
- **`ward_p_keys`** = `['x', 'y', 'z', 't_p']` — parity-aware (t_p for temporal with even/odd mask)
- **`ward_ward_keys`** = `['xy', 'xz', 'xt', 'yz', 'yt', 'zt']` — 6 plane directions for clover

### Ward mapping for gamma_gamma indexing
```python
ward_wards['xy'] = {'mu': -4, 'nu': -3, 'ward': -6}  # gamma_gamma index 0
ward_wards['xz'] = {'mu': -4, 'nu': -2, 'ward': -5}  # gamma_gamma index 1
# ... etc.
```

## Exported Functions

### `check_su3(U, tol=1e-3, verbose=True) → bool`

Verifies SU(3) properties of a gauge field:
1. **Unitarity:** U^H U ≈ I (uses `_torch.allclose` with `atol=tol`)
2. **Determinant:** det(U) ≈ 1 (uses raw `torch.linalg.det` — no NPU equivalent needed)
3. **Minor identities:** Each column is the cross product of the other two (with conjugation)

Returns `True` only if all three checks pass.

### `generate_gauge_field(U, sigma=0.1, seed=None, verbose=False) → torch.Tensor`

Generates random SU(3) gauge links via exponential map:
1. Sample 8 random Gaussian coefficients per site per direction
2. Form Hermitian matrix H = Σ_a c_a λ_a
3. Compute U = exp(i · σ · H) via `torch.matrix_exp`
4. Rearrange to `[3, 3, 4, Lx, Ly, Lz, Lt]` layout

Writes result in-place into `U`. Returns `U`.

### `give_support_multi() → bool`

Returns `True` if `MPI.COMM_WORLD.size > 1` (multi-process run).

## Data Layout

Gauge field `U`: shape `[3, 3, 4, Lx, Ly, Lz, Lt]` = `[color, color, direction, x, y, z, t]`

## Other Module-Level Data

In addition to the matrix data above, the module imports `mpi4py.MPI`, `pyqcu.cann as _torch`, and raw `torch` (for `torch.linalg.det` and `torch.matrix_exp` which have no NPU wrappers).

### Complete Skill: `dslash/` (source: `dslash/CLAUDE.md`)

# CLAUDE.md — pyqcu.dslash

Wilson and Clover Dirac operators — the core linear operators of lattice QCD.

## Files

| File | Purpose |
|------|---------|
| `_wilson.py` | Wilson hopping term D_w: spatial derivative with γ_μ matrices and parallel transport. Has per-module `force_use_npu` flag. |
| `_clover.py` | Clover term (chromo-magnetic field strength F_{μν} contribution). Four-plaquette clover construction with MPI boundary exchange. |
| `_operator.py` | Composed Dirac operator = hopping + sitting, with even-odd preconditioning, coarse-grid Galerkin projection, and MPI halo exchange. |

## Exported API

### Wilson (`_wilson.py`)

| Function | Purpose |
|----------|---------|
| `give_wilson(src, U, kappa, u_0, with_I, verbose)` | Full Wilson operator D_w = I − κ/u_0 · Σ_μ [(1−γ_μ)U_μ δ_{x+μ,y} + (1+γ_μ)U^†_{x−μ,μ} δ_{x−μ,y}] |
| `give_wilson_eo(src_o, U_eo, kappa, u_0, verbose)` | Even-odd Wilson (even dest from odd src) |
| `give_wilson_oe(src_e, U_eo, kappa, u_0, verbose)` | Odd-even Wilson (odd dest from even src) |
| `give_hopping_plus(ward, U, kappa, u_0, verbose)` | Directional hopping matrix M_μ^+ = −κ/u_0 · (I−γ_μ) ⊗ U_μ, shape `[12, 12, Lx, Ly, Lz, Lt]` |
| `give_hopping_minus(ward, U, U_head, kappa, u_0, verbose)` | Directional hopping matrix M_μ^- = −κ/u_0 · (I+γ_μ) ⊗ U^†_{x−μ,μ}, with halo exchange via `U_head` |
| `give_wilson_plus(ward, src, hopping, src_tail, parity, verbose)` | Apply hopping_plus to src: einsum("Eexyzt,exyzt→Exyzt", M_plus, rolled_src). Handles MPI `src_tail` boundary and parity masking. |
| `give_wilson_minus(ward, src, hopping, src_head, parity, verbose)` | Apply hopping_minus to src: einsum("Eexyzt,exyzt→Exyzt", M_minus, rolled_src). Handles MPI `src_head` boundary and parity masking. |

The eo/oe variants use `ward_p_keys` (x, y, z, t_p) — the `t_p` direction handles parity-split temporal hopping with even/odd masks.

### Clover (`_clover.py`)

| Function | Purpose |
|----------|---------|
| `make_clover(U, kappa, u_0, support_parallel, verbose)` | Build clover term from four-plaquette F_{μν} construction with 12 shifted gauge links per μν pair. Returns `[4,3,4,3,Lx,Ly,Lz,Lt]`. |
| `add_I(clover_term, verbose)` | Add identity: M = I + clover_term. Reshapes to `[12,12,N]`, adds I, reshapes back. |
| `cut_I(clover_term, verbose)` | Remove identity: M = clover_term − I |
| `inverse(clover_term, verbose)` | Batched 12×12 matrix inversion via `torch.linalg.inv` (NOT per-site loop!) |
| `give_clover(src, clover_term, verbose)` | Apply clover term: einsum("SCscxyzt,scxyzt→SCxyzt", clover, src) |
| `give_clover_ee(src_e, clover_e)` | Even-even clover application (delegates to `give_clover`) |
| `give_clover_oo(src_o, clover_o)` | Odd-odd clover application (delegates to `give_clover`) |

**Clover coefficient note:** Uses `_clover_factor = −0.125 · κ/u_0`. Standard convention with c_sw=1 gives −κ/(16·u_0). The factor of 2 may be due to the anti-hermitian part convention. Cross-validate against QUDA/Chroma before changing.

When `support_parallel=True`, `make_clover` performs MPI halo exchange for all 4 corners and edges of each μν plaquette (head, tail, head-tail, head-head, tail-tail).

### Operator (`_operator.py`)

Three classes compose the Dirac operator:

#### `hopping` class
- **Init:** Precomputes `M_plus_list[4]` and `M_minus_list[4]` via `give_hopping_plus`/`give_hopping_minus`. Performs MPI halo exchange for gauge field boundaries at init time. If `support_parity=True`, also splits into even/odd sub-blocks (`M_e_plus_list`, `M_o_plus_list`, etc.).
- **`matvec_plus(ward, src)` / `matvec_minus(ward, src)`:** Apply directional hopping with MPI halo exchange for fermion boundaries (send head to minus rank, receive tail from plus rank, and vice versa).
- **`matvec(src)`:** Sum over all 4 directions: Σ_μ (matvec_plus(μ) + matvec_minus(μ))

#### `sitting` class
- **Init:** Takes `clover_term` (can be None for pure Wilson). Adds I to get M = I + T. If `support_parity=True`, splits M into even/odd (`M_e`, `M_o`) and optionally precomputes inverses (`M_e_inv`, `M_o_inv`) unless provided externally.
- **`matvec(src)`:** Apply sitting term. If `clover_term is None`, returns `src` unchanged (identity).

#### `operator` class
- **Init:** Creates `hopping` and `sitting` instances. If `fine_hopping`, `fine_sitting`, and `local_ortho_null_vecs` are provided, builds a coarse-grid operator via Galerkin projection P^T D_fine P.
- **`matvec(src)`:** hopping.matvec + sitting.matvec. Auto-detects `[4,3,...]` vs `[12,...]` layout.
- **`matvec_eo(src_o)`:** Even-dest from odd-src hopping (used in preconditioned solves)
- **`matvec_oe(src_e)`:** Odd-dest from even-src hopping
- **`matvec_ee(src_e)` / `matvec_oo(src_o)`:** Even/odd sitting application
- **`matvec_ee_inv(src_e)` / `matvec_oo_inv(src_o)`:** Even/odd sitting inverse
- **`matvec_parity(src_o)`:** Parity-preconditioned operator: M_oo − M_oe · M_ee^{-1} · M_eo
- **`matvec_parity4fermion(fermion_in_o)`:** Same but auto-reshapes `[4,3,...]` ↔ `[12,...]`
- **`give_b_parity(b_e, b_o)`:** Preconditioned RHS: −M_oe · M_ee^{-1} · b_e + b_o
- **`give_x_e(b_e, x_o)`:** Recover even solution: M_ee^{-1} · (b_e − M_eo · x_o)
- **`matvec_eeo(src_e, src_o)` / `matvec_oeo(src_e, src_o)`:** Combined e→e+o→e and e→o+o→o
- **`matvec_all(src)`:** Full operator via parity-split/recombine: split src, apply eeo+oeo, recombine

**MPI halo exchange** in `matvec_eo`/`matvec_oe` is guarded by `grid_size[ward] != 1` — no MPI overhead for single-process directions.

## Coarse-Grid Operator (Galerkin Projection)

When `fine_hopping`, `fine_sitting`, and `local_ortho_null_vecs` are all provided, the operator builds a coarse-grid operator via P^T D_fine P:

1. For each null-space basis vector `e` and each direction `ward`:
   - Prolong a delta-source from coarse to fine grid
   - Apply the fine hopping operator (plus and minus directions)
   - Restrict the result back to coarse grid
   - Even/odd separation uses step=2 along the current direction
2. Also project the fine sitting operator: prolong → sitting.matvec → restrict

## Anti-Patterns

- **Never** use `self.sitting` (an object) as a truthy check; use `self.sitting.clover_term is not None`
- **Never** loop `torch.linalg.inv` per-site; use batched inversion (permute to batch dim, invert all at once, permute back)
- **Never** add `MPI.Barrier()` before/after blocking `Sendrecv` — it's redundant and slows down execution

### Complete Skill: `solver/` (source: `solver/CLAUDE.md`)

# CLAUDE.md — pyqcu.solver

Iterative solvers for the Dirac equation D ψ = η.

## Files

| File | Purpose |
|------|---------|
| `_bistabcg.py` | BiCGStab(l) solver (Bi-stabilized Conjugate Gradient) |
| `_multigrid.py` | Adaptive multigrid (AMG) V-cycle solver with CUDA acceleration at finest level |
| `_gmres.py` | GMRES solver — **placeholder stub, not yet implemented** |

## Exported API

### `bistabcg(b, matvec, tol=1e-6, max_iter=1000, x0=None, if_rtol=False, verbose=True) → torch.Tensor`

Standard BiCGStab solver. Takes a callable `matvec(src) → dest`.

**Breakdown detection** (added R2): raises `RuntimeError` on:
- `rho ≈ 0` (r_tilde orthogonal to r — loss of orthogonality)
- `vdot(r_tilde, v) ≈ 0` (pivot breakdown)
- `vdot(t, t) ≈ 0` (t is zero/near-zero)

**Tolerance:** `if_rtol=True` uses `tol * ||b||`; otherwise uses absolute `tol`.

### `multigrid` class

Adaptive multigrid V-cycle solver with configurable multi-level hierarchy.

**Constructor parameters:**
- `dtype_list`, `device_list` — per-level data types and devices
- `U`, `clover_term`, `kappa`, `u_0` — physical parameters
- `clover_ee_inv`, `clover_oo_inv` — if both provided, enables `with_cuda_qcu=True` (C++ backend at finest level)
- `min_size=4` — minimum lattice size per direction before coarsening stops
- `max_level=4` — maximum number of MG levels
- `mg_grid_size=[2,2,2,2]` — coarsening factor per direction
- `dof_list=[12,24,24,...]` — degrees of freedom per level
- `tol`, `max_iter`, `num_restart=5` — convergence parameters
- `num_convergence_sample=50` — window for adaptive level-back detection
- `support_parity=False` — use even-odd preconditioning

**Key methods:**

| Method | Purpose |
|--------|---------|
| `init()` | Build null-space vectors via inverse iteration, local-orthogonalize, construct coarse-grid operators (Galerkin) |
| `solve(b, x0)` | Solve D x = b. Returns `[4, 3, Lx, Ly, Lz, Lt]` tensor. |
| `cycle(level)` | Recursive V-cycle: BiCGStab smoothing → restrict residual → recurse → prolong correction → continue smoothing |
| `adaptive(iter)` | Level-back: drops to coarsest level if convergence stalls (≥3 counts in sample window) |
| `levels_back()` | Reset adaptive state |
| `plot(save_path)` | Plot convergence history (matplotlib, only on root rank) |

**Execution layers:**
- **Level 0 (finest):** Can use C++ CUDA backend (`with_cuda_qcu=True`) for BiCGStab smoothing
- **Level 1 (first coarse):** Can use C++ CUDA backend for coarse dslash via `_coarse_dslash_cuda()`
- **Levels 2+:** Pure Python einsum-based operators

**C++ backend integration (level 0):**
- `applyInitQcu`/`applyEndQcu` manage scratch buffer lifecycle
- `applyCloverBistabCgQcu` performs the full BiCGStab solve
- `applyCloverBistabCgDslashQcu` performs a single parity-preconditioned dslash
- `_SET_INDEX_` must be incremented between successive calls

**C++ backend integration (level 1):**
- `applyMultigridRestrictQcu`/`applyMultigridProLongQcu` for inter-grid transfers
- `applyMultigridCoarseDslashQcu` for coarse-grid operator application
- Hopping matrices packed as `[2, 4, E, E, Xc, Yc, Zc, Tc]` (pm dir Eout Ein XYZT)

**BiCGStab state reset after coarse-grid correction (R3 fix):** After a coarse-grid correction `x = x + e_fine`, the residual `r` changes, so all BiCGStab state must be reinitialized: `r_tilde = r.clone()`, reset `p/v/s/t` to zero, reset `rho_prev/alpha/omega` to 1.0. Without this, `rho = vdot(r_tilde_old, r_new)` gives meaningless results.

**Convergence tracking:** Records `r_norm` twice per iteration (before and after coarse-grid correction). Plot shows both.

**Debug helper:** `_verify_coarse_dslash(level, tol)` compares CUDA coarse dslash against Python einsum reference.

### Complete Skill: `smear/` (source: `smear/CLAUDE.md`)

# CLAUDE.md — pyqcu.smear

Gauge field smearing — spatial smoothing of gauge links to reduce UV noise.

## Files

| File | Purpose |
|------|---------|
| `_stout.py` | Stout smearing algorithm (copied/adapted from EasyDistillation's elemental generator) |

## Exported API

### `stout_smear(U, nstep=1, rho=0.12, support_parallel=False) → torch.Tensor`

Apply nstep iterations of stout smearing with parameter rho.

**Algorithm (per step):**

1. **Compute Q_μ = staple sum** for each direction μ: sum over ν≠μ of two 3-link staples (U_ν U_μ U^†_ν forward + U^†_ν U_μ U_ν backward)
2. **Project to su(3) algebra:** Q ← ρ · Q · U^†, then anti-hermitize: Q ← i/2 · (Q^† − Q) − (1/3) Tr(Q) · I
3. **Compute SU(3) projection coefficients f₀, f₁, f₂** via the Morningstar-Peardon method:
   - c₀ = Re(Tr(Q³))/3, c₁ = Re(Tr(Q²))/2
   - θ = arccos(c₀ / (2(c₁/3)^(3/2)))
   - u = √(c₁/3) · cos(θ/3), w = √c₁ · sin(θ/3)
   - f₀, f₁, f₂ expressed in terms of e^{iu}, e^{2iu}, cos(w), sinc(w)
4. **Parity handling** (when c₀ < 0): f₀ → f₀^*, f₁ → −f₁^*, f₂ → f₂^* (standard path); NPU path uses real/imag decomposition
5. **Update U:** U_new = (f₀·I + f₁·Q + f₂·Q²) · U

**Numerical stability:**
- c₁ clamped to min 1e-15 (prevents c₀_max = 0)
- ratio clamped to [−1+1e-15, 1−1e-15] for arccos domain
- sinc(w) uses Taylor expansion for |w| ≤ 0.05, sin(w)/w otherwise
- Denominator 9u² − w² has 1e-15 epsilon to prevent division by zero

**MPI support:** When `support_parallel=True`, MPI boundary data (U_head, U_tail, U_head_tail) is recomputed each step since U changes with each smearing step.

## Key Anti-Pattern (Fixed)

The `nstep>1` loop previously did not update `U` between steps — the loop variable was properly rebound but the MPI boundary data was computed outside the loop. Fixed by moving MPI exchange inside the step loop.

## Data Layout

Gauge field: `[3, 3, 4, Lx, Ly, Lz, Lt]` = `[color, color, direction, x, y, z, t]`

Returned tensor has the same shape.

## NPU Support

Has per-module `force_use_npu` flag. On NPU, the parity sign convention for f₀/f₁/f₂ uses explicit real/imag decomposition:
- f₀: imag = −imag (conj)
- f₁: real = −real, imag unchanged (conj + leading minus cancel)
- f₂: real = −real, imag unchanged (same as f₁)

### Complete Skill: `tools/` (source: `tools/CLAUDE.md`)

# CLAUDE.md — pyqcu.tools

Utility modules for MPI grid management, HDF5 I/O, linear algebra, tensor operations, multigrid transfers, and TileLang JIT kernels.

## Files

| File | Purpose |
|------|---------|
| `_define.py` | MPI grid size factorization, rank neighbors, parity splitting (`oooxyzt2poooxyzt`/`poooxyzt2oooxyzt`), dimension reordering (ccdxyzt↔ccdptzyx, scxyzt↔psctzyx), dtype conversion tables, device setup, slice helpers, prime factorization |
| `_io.py` | HDF5 I/O with MPI parallel I/O (`driver='mpio'`, `h5py`) and serial gather/scatter fallback (`comm.gather` + `comm.scatter`) |
| `_linalg.py` | Vector dot product (`vdot`) and norm (`norm`) via `_torch` |
| `_einsum.py` | TileLang JIT-compiled einsum kernels — currently `Eexyzt_exyzt2Exyzt` (optional, try/except import) |
| `_matul.py` | TileLang-based matrix multiply kernels: `matmul_gpu(M,N,K,...)` and `matmul_cpu(M,N,K,...)` (optional) |
| `_multigrid.py` | Null vector generation (`give_null_vecs`), local orthogonalization (`local_orthogonalize`), restrict/prolong operators — all with NPU-compatible fallback paths |
| `_roll.py` | Tensor rolling utilities |

## Exported API

### MPI Grid (`_define.py`)

| Function | Purpose |
|----------|---------|
| `give_grid_size()` | Auto-factor MPI communicator size into 4D grid `[gx, gy, gz, gt]` via prime factorization (sorted ascending) |
| `give_grid_index(rank)` | Convert flat rank to 4D grid index `[ix, iy, iz, it]` |
| `give_rank_plus(ward, rank)` | Neighbor rank in +direction |
| `give_rank_minus(ward, rank)` | Neighbor rank in −direction |
| `give_rank_plus_plus(ward_a, ward_b, rank)` | Diagonal neighbor (+a, +b) |
| `give_rank_plus_minus(ward_a, ward_b, rank)` | Diagonal neighbor (+a, −b) |
| `give_rank_minus_minus(ward_a, ward_b, rank)` | Diagonal neighbor (−a, −b) |
| `give_rank_minus_plus(ward_a, ward_b, rank)` | Diagonal neighbor (−a, +b) |
| `set_device(device, verbose)` | Set CUDA/NPU device based on MPI rank (round-robin assignment) |

### Parity Splitting (`_define.py`)

- **`oooxyzt2poooxyzt(input_array, verbose) → [2, ..., t, z, y, x//2]`** — Standard layout → parity-split. Separates even/odd sites based on (x+y+z+t) % 2. Splits along the fastest-varying (x) dimension.
- **`poooxyzt2oooxyzt(input_array, verbose) → [..., t, z, y, x]`** — Reverse: parity-split → standard layout. Recombines even/odd halves.

Both support NPU via explicit real/imaginary handling.

### Even-Odd Mask (`_define.py`)

- **`give_eo_mask(oootzy_t_p, eo, verbose)`** — Returns boolean mask for even (`eo=0`) or odd (`eo=1`) sites. Uses `(x+y+z) % 2` checkerboard. Results cached by shape+device+eo key.

### Dimension Reordering (`_define.py`)

HDF5 I/O uses dimension order `zyxt` (fastest to slowest: t, z, y, x):

- **`ccdxyzt2ccdptzyx(ccdxyzt) → [c,c,d,p,t,z,y,x]`** — Gauge field to file layout
- **`ccdptzyx2ccdxyzt(ccdptzyx) → [c,c,d,x,y,z,t]`** — File layout to gauge field
- **`scxyzt2psctzyx(scxyzt) → [p,s,c,t,z,y,x]`** — Fermion field to file layout
- **`psctzyx2scxyzt(psctzyx) → [s,c,x,y,z,t]`** — File layout to fermion field

### MPI Gather/Scatter (`_define.py`)

- **`local_xyzt2whole_xyzt(local_array, root) → Tensor | None`** — Gather distributed tensor chunks into a full global tensor on root rank. Uses `comm.Gather`.
- **`whole_xyzt2local_xyzt(dtype, device, whole_shape, whole_array, root) → Tensor`** — Scatter a global tensor (or shape template) to all ranks. Uses `comm.Scatter`. Each rank gets its grid block.

### Slice Helpers (`_define.py`)

- **`slice_dim(dims_num, ward, start, stop, step, point)`** — Build Python slice tuple for indexing along a specific ward dimension (using negative indexing). For `point`, returns integer index.
- **`slice_dim_dim(dims_num, ward_a, ..., ward_b, ...)`** — Two-dimension slice
- **`slice_dim_none_dim(dims_num, ward, ..., ward_none)`** — Slice with one skipped dimension

### Memory Helpers (`_define.py`)

- **`to_contiguous_real(tensor, channel, *shape)`** — Extract real/imag channel from complex tensor and return a truly stride-1 contiguous real tensor. Uses `empty + copy_` pattern instead of `.contiguous()` for correctness on single-element tensors.

### HDF5 I/O (`_io.py`)

- **`gridoooxyzt2hdf5oooxyzt(input_tensor, file_name, lat_size, verbose)`** — Write distributed tensor to HDF5. MPI path uses `h5py.File(..., driver='mpio')`; serial path uses `comm.gather` to root.
- **`hdf5oooxyzt2gridoooxyzt(file_name, lat_size, device, verbose)`** — Read HDF5 into distributed tensor. MPI path uses `h5py.File(..., driver='mpio')`; serial path uses root-read + `comm.scatter`.

**MPI support detection:** `HAS_MPI_SUPPORT = check_mpi_support()` at module import time. Tests h5py config and tries creating a test file with `driver='mpio'`. Can be manually overridden.

**Serial fallback note:** `comm.scatter` uses pickle serialization; may hit 2GB limit for very large lattices (>64⁴ float32). MPI I/O path preferred for production.

### Linear Algebra (`_linalg.py`)

- **`norm(input, p='fro', dim=None, keepdim=False)`** — Frobenius/vector norm via `_torch.norm`
- **`vdot(input, other)`** — Complex inner product `Σ conj(a_i) * b_i` via `_torch.vdot`

### Multigrid Utilities (`_multigrid.py`)

- **`give_null_vecs(null_vecs, matvec, bistabcg, normalize, ortho_r, ortho_null_vecs, verbose)`** — Generate near-null-space vectors via inverse iteration: v_i = v_i − A^{-1} A v_i. Optionally orthogonalizes against previous vectors. `null_vecs` parameter is used as shape/dtype/device template only; values are overwritten with random init.
- **`local_orthogonalize(null_vecs, coarse_lat_size, normalize, verbose)`** — Block-local Gram-Schmidt orthogonalization via batched QR decomposition. Splits null vectors into coarse-grid blocks, applies QR per block. NPU path avoids >8-dim tensors.
- **`restrict(local_ortho_null_vecs, fine_vec)`** — P^T v_fine = Σ v_fine · null_vec^†. Standard path uses 10-dim einsum; NPU path reshapes to ≤8 dims.
- **`prolong(local_ortho_null_vecs, coarse_vec)`** — P v_coarse = Σ null_vec · v_coarse. Standard path uses 10-dim einsum; NPU path reshapes to ≤8 dims.

**NPU compatibility:** NPU limits tensors to ≤8 dimensions, so restrict/prolong/orthogonalize all have `_npu` variants that use reshape/permute chains to stay within this limit. Cross-validated against standard path (max diff ~1e-7 for float32).

### TileLang Integration (`_einsum.py`, `_matul.py`)

Optional — try/except import at package level; silently degrades if TileLang unavailable.

- **`Eexyzt_exyzt2Exyzt(Eexyzt, exyzt)`** — JIT-compiled TileLang kernel for specific einsum pattern used in Wilson dslash (disabled by default; `tools_Eexyzt_exyzt2Exyzt = False`)
- **`matmul_gpu(M, N, K, block_M, block_N, block_K)`** / **`matmul_cpu(M, N, K, ...)`** — TileLang kernel definitions for matrix multiply benchmarking

Kernels use `warp_size = 128` from `_define`.

### Dtype Conversion Tables (`_define.py`)

- **`np2torch_dtype`**, **`torch2np_dtype`** — bidirectional NumPy ↔ PyTorch dtype maps
- **`torch2tl_dtype`** — PyTorch → TileLang dtype map (float16/32/64 only)

## Logging Convention

`PYQCU::TOOLS::<SUBMODULE>::\n message`

### Complete Skill: `testing/` (source: `testing/CLAUDE.md`)

# CLAUDE.md — pyqcu.testing

Integration tests for all PyQCU components. Tests are Python functions imported by `examples/*/conftest.py` entry points.

## Architecture

All test functions live in `pyqcu/testing/__init__.py`. They import from all PyQCU subpackages (`lattice`, `solver`, `dslash`, `tools`, `smear`). Each `examples/*/conftest.py` acts as a pytest entry point that imports specific test functions and calls them. The conftest files are manually edited to uncomment the test(s) to run.

The module imports `tilelang` at module level (with try/except fallback) for `test_matmul`.

## Test Functions

### `test_lattice(lat_size, dtype, device)`
Tests SU(3) gauge generation + gamma matrix algebra.
- Generates random gauge field, runs `check_su3`
- Verifies γ_μ² = I for all 4 gamma matrices
- **Assertion:** `check_su3` must return True

### `test_dslash_wilson(kappa, lat_size, dtype, device, with_data, support_parallel)`
Tests Wilson Dirac operator.
- `with_data=False`: Generates random gauge field + source, applies full Wilson operator and eo/oe preconditioned variants
- `with_data=True`: Loads reference HDF5 data (`refer.wilson.*.L32K0_125.*.h5`), validates operator.matvec against known result
- **Assertion:** Relative difference < 1e-4

### `test_dslash_parity(lat_size, kappa, dtype, device)`
Tests parity-preconditioned Wilson+Clover operator with MPI.
- Distributes gauge field across MPI grid
- Root rank computes full operator result as reference
- All ranks compare local parity-preconditioned operator against reference
- Tests both `matvec_all` and `matvec_eeo`/`matvec_oeo` paths

### `test_dslash_clover(device, with_data, dtype)`
Tests Clover term construction.
- `with_data=True`: Loads reference data, validates clover term and inverse against known results
- `with_data=False`: Tests parallel vs serial clover construction across MPI grid

### `test_solver(kind, method, kappa, lat_size, dtype, device, with_data, max_level, num_restart, support_parity)`
Tests BiStabCG and multigrid solvers.
- `method='bistabcg'`: Standard or parity-preconditioned BiCGStab
- `method='multigrid'`: Full multigrid V-cycle with `init()` + `solve()` + `plot()`
- `with_data=True`: Validates against reference Wilson data
- **Assertion:** Relative error < 1e-3

### `test_matmul()`
Benchmarks TileLang JIT-compiled matrix multiply vs PyTorch (cuBLAS/MKL).
- GPU: 4096×4096 matmul, TileLang vs cuBLAS
- CPU: 1024×1024 matmul, TileLang (LLVM or C backend) vs MKL/OneDNN
- Prints TFLOPS comparison table

### `test_smear_stout(lat_size, device, dtype)`
Tests stout smearing across MPI grid.
- Distributes gauge field, root computes whole-grid reference
- All ranks compare local parallel smear against reference
- Verifies SU(3) before and after smearing

## Running Tests

```bash
cd examples && pytest .                              # all conftest.py files
mpirun -np 4 python examples/pyqcu/conftest.py       # single file with MPI
```

## Logging Convention

All test output uses: `PYQCU::TESTING::<MODULE>::\n message`

## Important Notes

- Tests use `tools.local_xyzt2whole_xyzt` / `tools.whole_xyzt2local_xyzt` for MPI reference comparison
- Reference HDF5 data lives in `examples/data/`
- The `path` variable in tests is computed from `pyqcu.__file__` to locate data files
- **R3 fix:** Tests now include `assert` statements so pytest can detect failures

### Complete Skill: `cuda/` (source: `cuda/CLAUDE.md`)

# CLAUDE.md — pyqcu.cuda

Cython bridge package for the C++ CUDA backend (`libqcu.so`).

## Files

| File | Purpose |
|------|---------|
| `__init__.py` | Makes `pyqcu.cuda` a proper Python package (added 2026-07-28 R3; was missing, causing `pip install` failures) |
| `qcu/qcu.pyx` | Cython extension source — wraps C functions from `pyqcu.h` as `applyInitQcu`, `applyWilsonDslashQcu`, etc. |
| `qcu/qcu.pxd` | Cython declaration file — `cdef extern` block matching `pyqcu.h` |
| `qcu/qcu.pyi` | Type stub (155 lines) — full type annotations, docstrings, and default values for IDE support |
| `define.py` | Parameter constants (`_LAT_X_`, `_SET_PLAN_`, etc.) and dtype conversion helpers (`dtype()`, `epytd()`) |

## Public API

```python
from pyqcu.cuda import qcu      # Cython bridge to libqcu.so
from pyqcu.cuda import define   # Parameter constants, dtype helpers, pre-built params/argv/set_ptrs tensors
```

## Cython Extension — C Functions Exposed

| Function | Purpose | Plan |
|----------|---------|------|
| `applyInitQcu` / `applyEndQcu` | Allocate / free scratch buffers | — |
| `applyWilsonDslashQcu` | Wilson dslash | 0 |
| `applyCloverDslashQcu` | Clover dslash | 2 |
| `applyWilsonBistabCgQcu` / `applyWilsonBistabCgDslashQcu` | Wilson BiStabCG solver + its dslash | 1 |
| `applyWilsonCgQcu` / `applyWilsonCgDslashQcu` | Wilson CG solver + its dslash | 1 |
| `applyCloverBistabCgQcu` / `applyCloverBistabCgDslashQcu` | Clover BiStabCG (needs clover_ee/oo + inverses) | 1 |
| `applyCloverQcu` / `applyCloversQcu` | Build Clover term (and its inverse) | 2 |
| `applyDslashQcu` | Combined Wilson+Clover dslash | 0+2 |
| `applyLaplacianQcu` | Laplacian operator | -2 |
| `applyGaussGaugeQcu` | Gaussian gauge field generation | -1 |
| `applyMultigridRestrictQcu` / `applyMultigridProLongQcu` | MG restrict/prolong with null vectors | MG |
| `applyMultigridCoarseDslashQcu` | Coarse-grid dslash (hopping + sitting) | MG |
| `applyCloverMultigridQcu` | Full Clover multigrid V-cycle solver | MG |

All functions take raw pointers cast to `long long` from `tensor.contiguous().data_ptr()`.

## Parameter Protocol

Three flat tensors bridge Python ↔ C++:

- **`params`** (int32, size 54) — lattice dims (`_LAT_X_`…`_LAT_XYZT_`), grid sizes (`_GRID_X_`…), data types (`_DATA_TYPE_`), iteration counts (`_MAX_ITER_`), plan selection (`_SET_PLAN_`), verbosity (`_VERBOSE_`), parity (`_PARITY_`), multigrid level configs (`_MG_LEVEL1_X_`…, `_MG_NUM_LEVEL_`)
- **`argv`** (float, size 7) — physical parameters: `_MASS_` (idx 0), `_ATOL_` (1), `_SIGMA_` (2), per-level MG tolerances (3–6)
- **`set_ptrs`** (int64, size 100) — scratch pointers managed by the C++ runtime

Index constants in `define.py` MUST stay in sync with `cpp/cuda/qcu/include/define.h`.

`define.py` also provides pre-built tensors `params`, `argv`, and `set_ptrs` for convenience. They are modified in-place by the solver code.

## Critical: `_SET_INDEX_` Increment

Between successive C++ calls within the same `applyInitQcu`/`applyEndQcu` lifecycle, you MUST increment `params[define._SET_INDEX_]` by 1. Failing to do so causes scratch buffer reuse conflicts that produce wrong results.

Exception: coarse-grid dslash resets `_SET_INDEX_` to 0 (different MG level, no overlap with fine-level ops).

## Data Type Mapping

- `define.dtype(data_type)` — QCU internal constant (`_LAT_C64_`, `_LAT_R32_`, etc.) → PyTorch dtype
- `define.epytd(torch_dtype)` — PyTorch dtype → QCU internal constant
- `define.lat_shape(params)` — extract `[Lt, Lz, Ly, Lx]` from params tensor

## Plan Selection

| Plan Constant | Value | Purpose |
|---------------|-------|---------|
| `_SET_PLAN_N_2_` | -2 | Laplacian |
| `_SET_PLAN_N_1_` | -1 | Gauss gauge generation |
| `_SET_PLAN0_` | 0 | Wilson dslash |
| `_SET_PLAN1_` | 1 | BiStabCG / CG (and their dslash) |
| `_SET_PLAN2_` | 2 | Clover dslash |

## Call Lifecycle

```python
qcu.applyInitQcu(set_ptrs, params, argv)          # allocate
# ... operations with _SET_INDEX_ += 1 between calls ...
qcu.applyEndQcu(set_ptrs, params)                  # free
```

---

## Complete Skills (Agent-Produced Subdirectories)

The content of each subdirectory below was produced with Claude Code assistance. Per repo convention, the complete skill that generates that content is reproduced verbatim below (source: the subdirectory's own `CLAUDE.md`), so the full knowledge is available directly at this level.

### Complete Skill: `qcu/` (source: `qcu/CLAUDE.md`)

# CLAUDE.md — pyqcu.cuda.qcu

Cython extension module — bridges Python to the C++ CUDA backend `libqcu.so`.

## Files

| File | Purpose |
|------|---------|
| `qcu.pyx` | Cython source: thin wrappers around C functions from `pyqcu.h` |
| `qcu.pxd` | Cython declarations: `cdef extern` block (must match `pyqcu.h` exactly) |
| `qcu.pyi` | Python type stub for IDE autocomplete |

## C API Surface

All 22 C functions are exposed. Each takes raw tensor data pointers as `long long`:

| Function | Purpose |
|----------|---------|
| `applyInitQcu` / `applyEndQcu` | Allocate / free scratch buffers |
| `applyWilsonDslashQcu` | Wilson dslash |
| `applyCloverDslashQcu` | Clover dslash |
| `applyWilsonBistabCgQcu` / `applyWilsonBistabCgDslashQcu` | Wilson BiStabCG solver + dslash |
| `applyWilsonCgQcu` / `applyWilsonCgDslashQcu` | Wilson CG solver + dslash |
| `applyCloverBistabCgQcu` / `applyCloverBistabCgDslashQcu` | Clover BiStabCG (requires clover_ee/oo + inverses) |
| `applyCloverQcu` / `applyCloversQcu` | Build Clover term (and inverse) |
| `applyDslashQcu` | Combined Wilson+Clover dslash |
| `applyLaplacianQcu` | Laplacian operator |
| `applyGaussGaugeQcu` | Gaussian gauge field generation |
| `applyMultigridRestrictQcu` / `applyMultigridProLongQcu` | MG restrict/prolong with null vectors |
| `applyMultigridCoarseDslashQcu` | Coarse-grid dslash |
| `applyCloverMultigridQcu` | Full Clover multigrid V-cycle |

## Call Lifecycle

```python
qcu.applyInitQcu(set_ptrs, params, argv)   # allocate buffers
# ... perform operations ...
params[define._SET_INDEX_] += 1              # MUST increment between calls
qcu.applyEndQcu(set_ptrs, params)            # free buffers
```

## Synchronization

The `.pxd` file must exactly match the C declarations in `cpp/cuda/qcu/python/pyqcu.h`. Any mismatch causes silent memory corruption.

### Complete Skill: `cann/` (source: `cann/CLAUDE.md`)

# CLAUDE.md — pyqcu.cann

Torch compatibility layer for Ascend NPU. All Python code in PyQCU imports `pyqcu.cann as _torch` instead of using `torch` directly.

## Problem

Ascend NPU does not natively support complex tensors. This module wraps torch operations, decomposing complex ops into real/imaginary parts on NPU while passing through directly on CUDA/CPU.

## Behavior

- **CUDA/CPU path** (`device.type != 'npu'` and `force_use_npu=False`): delegates to `torch.*` unchanged
- **NPU path** (`device.type == 'npu'` or `force_use_npu=True`): decomposes complex ops into real/imaginary parts

## Global Flag

`pyqcu.cann.force_use_npu = True` — force NPU code paths on CPU for testing without NPU hardware. This affects only the `cann` layer; some modules (`dslash/_wilson.py`, `tools/_define.py`, `tools/_multigrid.py`, `smear/_stout.py`) also have their own per-module `force_use_npu` flag for deeper NPU workarounds (e.g., tensor dimension limits).

## Functions Provided

Always use these instead of raw torch calls anywhere complex tensors might run on NPU:

| Category | Functions | Notes |
|----------|-----------|-------|
| Math | `abs`, `vdot`, `norm`, `sqrt`, `matmul` | `vdot` → conj-flatten-sum; `norm` → abs-then-norm; `sqrt` → CPU fallback |
| Reduction/shape | `roll`, `allclose`, `einsum` | `roll` → roll real/imag separately; `allclose` → check real + imag separately |
| Creation | `zeros`, `zeros_like`, `randn`, `randn_like`, `eye` | Creates real parts then combines to complex |
| Linear algebra | `linalg_qr` | Falls back to CPU on NPU for complex inputs |

### Uses of raw `torch`

- `torch.linalg.det` — used in `lattice.check_su3()` for SU(3) determinant check. No equivalent in `_torch`; works on NPU for real matrices.
- `torch.matrix_exp` — used in `lattice.generate_gauge_field()` for exponential map.

## Einsum on NPU

General N-operand complex einsum uses a combinatorial approach. For Z = Π(a_k + i·b_k):

- Iterates all 2ⁿ sign combinations (bitmask k → real/imag selection for each operand)
- Even number of imaginary parts → contributes to real part with sign = (-1)^(n_imag/2)
- Odd number of imaginary parts → contributes to imaginary part with sign = (-1)^((n_imag-1)/2)
- 2-operand special case: explicit ac-bd + i(ad+bc) formula (faster)

## Key Implementation Details

- `eye(n, m, ...)` — creates real identity then casts to complex dtype on NPU
- `zeros(*args, ...)` / `randn(*args, ...)` — creates separate real + imag tensors and combines
- `sqrt(input)` — sends complex input to CPU, computes sqrt, sends back (NPU doesn't support complex sqrt)
- `matmul(input, other)` — uses explicit (ac-bd) + i(ad+bc) decomposition

## Subdirectory

`qcu/` — placeholder stub (empty `PASS` file), no implementation yet.

---

## Complete Skills (Agent-Produced Subdirectories)

The content of each subdirectory below was produced with Claude Code assistance. Per repo convention, the complete skill that generates that content is reproduced verbatim below (source: the subdirectory's own `CLAUDE.md`), so the full knowledge is available directly at this level.

### Complete Skill: `qcu/` (source: `qcu/CLAUDE.md`)

# CLAUDE.md — pyqcu.cann.qcu

Placeholder for the Ascend NPU C++ Cython bridge. No implementation yet.

Contains only an empty `PASS` file as a directory placeholder.

### Complete Skill: `dtk/` (source: `dtk/CLAUDE.md`)

# CLAUDE.md — pyqcu.dtk

Placeholder for DCU/ROCm (AMD GPU) backend. No implementation yet.

Contains only an empty `PASS` file as a directory placeholder.

### Complete Skill: `maca/` (source: `maca/CLAUDE.md`)

# CLAUDE.md — pyqcu.maca

Placeholder for Maca backend. No implementation yet.

Contains only an empty `PASS` file as a directory placeholder.

### Complete Skill: `cpp/` (source: `cpp/CLAUDE.md`)

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

---

## Complete Skills (Agent-Produced Subdirectories)

The content of each subdirectory below was produced with Claude Code assistance. Per repo convention, the complete skill that generates that content is reproduced verbatim below (source: the subdirectory's own `CLAUDE.md`), so the full knowledge is available directly at this level.

### Complete Skill: `qcu/` (source: `qcu/CLAUDE.md`)

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

### Complete Skill: `cann/` (source: `cann/CLAUDE.md`)

# CLAUDE.md — cpp/cann

Ascend CANN backend container directory. The actual (stub) implementation lives in `qcu/`.

Currently a placeholder — no active CANN C++ backend exists.

---

## Complete Skills (Agent-Produced Subdirectories)

The content of each subdirectory below was produced with Claude Code assistance. Per repo convention, the complete skill that generates that content is reproduced verbatim below (source: the subdirectory's own `CLAUDE.md`), so the full knowledge is available directly at this level.

### Complete Skill: `qcu/` (source: `qcu/CLAUDE.md`)

# CLAUDE.md — cpp/cann/qcu

Placeholder for Huawei Ascend CANN (NPU) C++ backend. No implementation yet.

Contains only an empty `PASS` file as a directory placeholder.

### Complete Skill: `dtk/` (source: `dtk/CLAUDE.md`)

# CLAUDE.md — cpp/dtk

DCU/ROCm backend container directory. The actual (stub) implementation lives in `qcu/`.

Currently a placeholder — no active DTK C++ backend exists.

---

## Complete Skills (Agent-Produced Subdirectories)

The content of each subdirectory below was produced with Claude Code assistance. Per repo convention, the complete skill that generates that content is reproduced verbatim below (source: the subdirectory's own `CLAUDE.md`), so the full knowledge is available directly at this level.

### Complete Skill: `qcu/` (source: `qcu/CLAUDE.md`)

# CLAUDE.md — cpp/dtk/qcu

Placeholder for AMD DCU / ROCm (HIP) C++ backend. No implementation yet.

Contains only an empty `PASS` file as a directory placeholder.

### Complete Skill: `maca/` (source: `maca/CLAUDE.md`)

# CLAUDE.md — cpp/maca

Maca backend container directory. The actual (stub) implementation lives in `qcu/`.

Currently a placeholder — no active Maca C++ backend exists.

---

## Complete Skills (Agent-Produced Subdirectories)

The content of each subdirectory below was produced with Claude Code assistance. Per repo convention, the complete skill that generates that content is reproduced verbatim below (source: the subdirectory's own `CLAUDE.md`), so the full knowledge is available directly at this level.

### Complete Skill: `qcu/` (source: `qcu/CLAUDE.md`)

# CLAUDE.md — cpp/maca/qcu

Placeholder for Maca C++ backend. No implementation yet.

Contains only an empty `PASS` file as a directory placeholder.

### Complete Skill: `examples/` (source: `examples/CLAUDE.md`)

# CLAUDE.md — examples

Test examples and benchmarks for PyQCU, organized by backend target.

## Directory Map

| Directory | Target | Description |
|-----------|--------|-------------|
| `pyqcu/` | CPU/CUDA/NPU | Pure-Python operator/solver tests (main test suite) |
| `qcu/` | NVIDIA CUDA | C++ CUDA backend tests via Cython bridge |
| `cpu/` | CPU | CPU-only tests (BiStabCG, MPI) |
| `npu/` | Ascend NPU | NPU-specific tests |
| `dcu/` | AMD DCU | DCU/ROCm tests |
| `profiler/` | All | Perfetto tracing with `torch.profiler` |
| `benchmark/` | All | Performance benchmarks |
| `tilelang/` | CUDA | TileLang kernel tests |
| `gpu/` | GPU | Empty — GPU test placeholder |
| `data/` | — | Reference HDF5 files for validation (`with_data=True`) |

## Running Tests

```bash
cd examples && pytest .                              # all conftest.py files
mpirun -np 4 python examples/pyqcu/conftest.py       # single file with MPI
```

Each subdirectory has its own `conftest.py` that imports test functions from `pyqcu.testing` and calls them. Conftest files are manually edited to uncomment desired tests.

## Reference Data

`examples/data/` contains HDF5 files with precomputed gauge fields, sources, and expected results used for validation when tests are run with `with_data=True`.

---

## Complete Skills (Agent-Produced Subdirectories)

The content of each subdirectory below was produced with Claude Code assistance. Per repo convention, the complete skill that generates that content is reproduced verbatim below (source: the subdirectory's own `CLAUDE.md`), so the full knowledge is available directly at this level.

### Complete Skill: `pyqcu/` (source: `pyqcu/CLAUDE.md`)

# CLAUDE.md — examples/pyqcu

Main test suite: pure-Python operator and solver tests. These run on CPU, CUDA GPU, or Ascend NPU (via `pyqcu.cann`).

## Test Files

| File | What it tests |
|------|---------------|
| `conftest.py` | Entry point — imports from `pyqcu.testing`, uncomment desired tests |
| `conftest.bistabcg.py` | BiStabCG solver (various lattice sizes, dtype, parity modes) |
| `conftest.clover.bistabcg.py` | Clover BiStabCG solver |
| `conftest.multigrid.py` | Wilson multigrid solver (various lattice sizes, max_level, num_restart) |
| `conftest.clover.multigrid.py` | Clover multigrid solver |

## Usage

Edit the conftest file to uncomment the desired test(s), then run:

```bash
mpirun -np 4 python examples/pyqcu/conftest.py
```

### Complete Skill: `qcu/` (source: `qcu/CLAUDE.md`)

# CLAUDE.md — examples/qcu

C++ CUDA backend tests via the Cython bridge. These exercise `libqcu.so` from Python.

## Test Files

| File | What it tests |
|------|---------------|
| `conftest.cuda.py` | Basic CUDA availability and context |
| `conftest.mpi.py` | MPI grid setup and halo exchange |
| `conftest.wilson.bistabcg.py` | Wilson BiStabCG via C++ backend |
| `conftest.wilson.bistabcg.dslash.py` | Wilson BiStabCG dslash kernel |
| `conftest.wilson.cg.py` | Wilson CG solver |
| `conftest.clover.py` | Clover term construction |
| `conftest.clover.bistabcg.py` | Clover BiStabCG solver |
| `conftest.clover.bistabcg.dslash.py` | Clover BiStabCG dslash |
| `conftest.clover.multigrid.py` | Clover multigrid V-cycle solver |

## Usage

```bash
mpirun -np 1 python examples/qcu/conftest.clover.multigrid.py
```

Output: convergence log → `logs/clover_multigrid.log`, performance report → `logs/clover_multigrid_report.log`

## Dev73_5 Multigrid Benchmark Suite

Development scripts for the dev73_5 multigrid performance milestone. They benchmark `applyCloverMultigridQcu` against the Clover BiStabCG reference (`applyCloverBistabCgQcu`) across precision / lattice / solver-parameter sweeps, and feed `logs/dev73_5.*` (report, LaTeX tables, PNG figures).

| File | Purpose |
|------|---------|
| `mg_dev73_5_clean.py` | Clean, isolated-process timing of a single config (ref/mg interleaved, min+median speedup) |
| `mg_dev73_5_bench.py` | Extended performance benchmark — precision / lattice / solver-parameter sweeps vs BiStabCG |
| `mg_dev73_5_verify.py` | Correctness checks — SU(3) gauge, solution error, null-vector zero-mode/orthogonality, C++ vs Python coarse dslash |
| `mg_dev73_5_collect.py` | Aggregate clean/bench/verify JSON into `logs/dev73_5_results.json` |
| `mg_dev73_5_mktable.py` | Emit LaTeX table snippets (`logs/dev73_5_tbl_*.tex`) for `dev73_5.tex` |
| `mg_dev73_5_plots.py` | Generate convergence / hotspot / speedup / time PNG figures into `logs/` |

### Complete Skill: `cpu/` (source: `cpu/CLAUDE.md`)

# CLAUDE.md — examples/cpu

CPU-only tests for the pure-Python backend.

## Test Files

| File | What it tests |
|------|---------------|
| `conftest.py` | Main entry — imports from `pyqcu.testing` |
| `conftest.bistabcg.py` | BiStabCG solver on CPU |
| `conftest.mpi.py` | MPI-distributed solver on CPU |

## Usage

```bash
mpirun -np 4 python examples/cpu/conftest.py
```

### Complete Skill: `npu/` (source: `npu/CLAUDE.md`)

# CLAUDE.md — examples/npu

Ascend NPU tests. Use `pyqcu.cann.force_use_npu = True` to test NPU code paths on CPU without NPU hardware.

## Test Files

| File | What it tests |
|------|---------------|
| `conftest.py` | Main NPU test entry |
| `conftest_1.py` | Additional NPU test variant |

## Usage

```bash
python examples/npu/conftest.py
```

### Complete Skill: `dcu/` (source: `dcu/CLAUDE.md`)

# CLAUDE.md — examples/dcu

AMD DCU / ROCm (HIP) tests.

## Test Files

| File | What it tests |
|------|---------------|
| `conftest.py` | Main DCU test entry |
| `conftest_1.py` | DCU test variant 1 |
| `conftest_2.py` | DCU test variant 2 |

## Usage

```bash
python examples/dcu/conftest.py
```

### Complete Skill: `profiler/` (source: `profiler/CLAUDE.md`)

# CLAUDE.md — examples/profiler

Performance profiling with `torch.profiler`. Exports Chrome trace format for visualization in Perfetto.

## Test Files

| File | What it profiles |
|------|-----------------|
| `conftest.py` | Main profiler entry |
| `conftest.cpu.py` | CPU profiling |
| `conftest.cuda.py` | CUDA GPU profiling |
| `conftest.npu.py` | NPU profiling |

## Profiler Configuration

Uses `torch.profiler.profile(...)` with `record_shapes=True`, `with_modules=True`, `with_flops=True`.

## Usage

```bash
cd examples/profiler && mpirun -np 1 python -u conftest.py
# Load resulting trace_*.json into https://ui.perfetto.dev
```

### Complete Skill: `benchmark/` (source: `benchmark/CLAUDE.md`)

# CLAUDE.md — examples/benchmark

Performance benchmarks comparing PyTorch, TileLang, and C++ CUDA backend implementations.

## Files

| File | Purpose |
|------|---------|
| `conftest.py` | Benchmark entry point |
| `env.py` | Benchmark environment configuration |

## Usage

```bash
python examples/benchmark/conftest.py
```

### Complete Skill: `tilelang/` (source: `tilelang/CLAUDE.md`)

# CLAUDE.md — examples/tilelang

TileLang JIT-compiled kernel tests for CUDA. Exercises the TileLang integration in `pyqcu/tools/_einsum.py` and `pyqcu/tools/_matul.py`.

## Files

| File | Purpose |
|------|---------|
| `conftest.py` | TileLang test entry |

## Usage

```bash
python examples/tilelang/conftest.py
```

Requires TileLang to be installed (optional dependency, silently degrades if unavailable).

### Complete Skill: `gpu/` (source: `gpu/CLAUDE.md`)

# CLAUDE.md — examples/gpu

GPU test placeholder. Currently empty — no test files.

Intended for generic GPU tests that don't fit into the more specific `qcu/` (NVIDIA CUDA), `dcu/` (AMD ROCm), or `npu/` (Ascend NPU) directories.

### Complete Skill: `data/` (source: `data/CLAUDE.md`)

# CLAUDE.md — examples/data

Reference HDF5 files for test validation. Used when tests are run with `with_data=True` to validate against precomputed expected results.

Contains gauge fields, fermion sources, and expected operator/solver outputs for various lattice sizes.

### Complete Skill: `docs/` (source: `docs/CLAUDE.md`)

# CLAUDE.md — docs

Reference documentation for PyQCU.

## Files

| File | Content |
|------|---------|
| `dims.md` | Dimension naming scheme (`s`=spin, `c`=color, `d`=direction, `p`=parity, `x/y/z/t`=spacetime). Documents conventions for `ccdxyzt`, `scxyzt`, `psctzyx` etc. |
| `env.md` | Python environment setup — required variables (`QUDA_PATH`, `LD_LIBRARY_PATH`, `PYTHONPATH`) |
| `install.md` | Installation guide — build.sh + install.sh workflow |
| `examples.md` | Examples usage guide — how to run tests and interpret output |
| `profiler.md` | Profiling guide — using torch.profiler and Perfetto |

### Complete Skill: `refer/` (source: `refer/CLAUDE.md`)

# CLAUDE.md — refer

Reference documents for development history.

## Contents

| File | Description |
|------|-------------|
| `dev71.md` | Development milestone 71 markdown report |
| `dev71.pdf` | Development milestone 71 PDF report |
| `dev71.tex` | Development milestone 71 LaTeX source |

These are historical reference documents tracking development progress and design decisions.

### Complete Skill: `logs/` (source: `logs/CLAUDE.md`)

# CLAUDE.md — logs

Development logs, review reports, bug fix summaries, and solver output. Milestone reports (`dev*.md`/`.tex`/`.pdf`), fix reports, and their figures are versioned; scratch output (`*.log`, `*.json`, `*.aux`) is gitignored.

## File Patterns

| Pattern | Purpose |
|---------|---------|
| `dev<N>.md` / `.tex` / `.pdf` | Development milestone reports (e.g., `dev73_5.md`, plus generated tables `dev73_5_tbl_*.tex` and figures `dev73_5_*.png`) |
| `bug<N>.md` | Bug discovery & code review reports |
| `review-*.md` | Code review findings (e.g., `review-2026-07-28.md`) |
| `fix-report-*.md` | Bug fix summaries |
| `mg-*-report-*.md` / `.tex` / `.pdf` | Multigrid development reports (e.g., `mg-v4-report-2026-08-02.*`) |
| `multigrid_report.md` | MG solver performance reports |
| `clover_multigrid.log` | C++ solver convergence output |
| `*.png` | Performance charts, convergence plots |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `debug/` | Per-round fix logs (`fix-log*.md`) |
| `results/` | Final/remaining fix reports |

---

## Complete Skills (Agent-Produced Subdirectories)

The content of each subdirectory below was produced with Claude Code assistance. Per repo convention, the complete skill that generates that content is reproduced verbatim below (source: the subdirectory's own `CLAUDE.md`), so the full knowledge is available directly at this level.

### Complete Skill: `debug/` (source: `debug/CLAUDE.md`)

# CLAUDE.md — logs/debug

Per-round debug and fix logs generated during development and bug-fixing sessions.

## File Pattern

`fix-log*.md` — per-round fix logs documenting individual bug fixes, root cause analysis, and verification results.

These are temporary/working files — final summaries are promoted to `logs/fix-report-*.md`.

### Complete Skill: `results/` (source: `results/CLAUDE.md`)

# CLAUDE.md — logs/results

Final and remaining fix reports. These are the polished, summary versions of fix reports after debug resolution.

## Purpose

When a bug-fixing session completes, the final summary report is written here. These are the authoritative record of what was fixed, what remains, and what was skipped.

### Complete Skill: `.claude/` (source: `.claude/CLAUDE.md`)

# CLAUDE.md — .claude

Claude Code configuration directory for the PyQCU repository. Holds agent skills and machine-local settings.

## Contents

| Path | Purpose |
|------|---------|
| `settings.local.json` | Machine-local Claude Code settings (untracked — do not commit) |
| `skills/` | Agent skill directory — reusable markdown knowledge files loaded on demand by Claude Code |

## Skills (`skills/`)

`skills/` is an agent-generated skill directory. Each skill is a markdown file with YAML frontmatter (`name`, `description`). Skills are surfaced to Claude Code sessions automatically; invoking a skill loads its full content into context.

Current skills:

| File | Skill | Description |
|------|-------|-------------|
| `skills/past-work.md` | `past-work` | Past work history of PyQCU — what was built, optimized, and remains TODO |

## Complete Skills (Agent-Produced Subdirectories)

The content of each subdirectory below was produced with Claude Code assistance. Per repo convention, the complete skill that generates that content is reproduced verbatim below (source: the subdirectory's own `CLAUDE.md`), so the full knowledge is available directly at this level.

### Complete Skill: `skills/` (source: `skills/CLAUDE.md`)

# CLAUDE.md — .claude/skills

Agent skills for the PyQCU repository. Each skill is a markdown file with YAML frontmatter. Skills are loaded on demand by Claude Code when their `description` matches the current task — they capture reusable knowledge so it does not have to be re-derived.

## Skill Format

Every skill file must begin with YAML frontmatter:

```yaml
---
name: <kebab-case-slug>          # must match the file name
description: <one-line summary>  # used to decide when the skill applies
---
```

Follow the frontmatter with the skill body: the knowledge, procedures, and conventions it is meant to encapsulate.

## Skills

| File | Skill | Description |
|------|-------|-------------|
| `past-work.md` | `past-work` | Past work history of PyQCU — what was built, optimized, and remains TODO (project phases, current state, known gaps) |

## Adding / Editing a Skill

1. Create or edit `skills/<name>.md`.
2. Keep the frontmatter `name` identical to the file name, and write a `description` specific enough to trigger appropriately.
3. Keep the body focused on one capability.
4. Update the table above so this CLAUDE.md stays accurate.

## Notes

- The full content of the `past-work` skill is reproduced in `../CLAUDE.md` (see "Complete Skill: `past-work`") so it is also available at the parent `.claude/` level.
- This directory is tracked in git (`skills/past-work.md`). Do not put machine-local or ephemeral content here.

## Complete Skill: `past-work`

Per repo convention, the complete skill of the agent skill directory `skills/` is reproduced below (verbatim source: `skills/past-work.md`) so it is available directly at this level.

```markdown
---
name: past-work
description: Past work history of PyQCU - what was built, optimized, and remains TODO
---

# Past Work

PyQCU began in April 2026 as a Python/Cython GPU lattice QCD library. The git history spans 2026-04-27 to 2026-07-20 (~3 months, 60 commits).

## Phase 1: Foundation (2026-04-27 to 2026-04-28)

**What was built:**
- Project scaffolding: `setup.py`, `env.sh`, `build.sh`, `install.sh`, directory structure
- `pyqcu/lattice/` — Gamma matrices, Gell-Mann matrices, SU(3) checks, gauge field generation
- `pyqcu/dslash/` — Wilson Dirac operator (`_wilson.py`), hopping/sitting/operator classes (`_operator.py`)
- `pyqcu/solver/` — BiStabCG (`_bistabcg.py`), GMRES stub
- `pyqcu/tools/` — MPI grid helpers, HDF5 I/O, linalg, parity conversion, einsum stubs
- `pyqcu/cann/` — NPU compatibility layer (complex op decomposition for Ascend)
- `pyqcu/testing/` — Integration tests
- C++ CUDA backend skeleton: `cpp/cuda/qcu/` with CMake build, all kernel files, Cython bridge (`qcu.pyx`, `define.py`, `pyqcu.h`)
- Test infrastructure: `examples/pyqcu/`, `examples/qcu/`, `examples/cpu/`, `examples/npu/`, `examples/dcu/`, `examples/profiler/`

**Key design decisions made in this phase:**
- Two-layer architecture: pure Python (PyTorch) for dev/testing + C++ CUDA for production
- `pyqcu.cann` as _torch import throughout (NPU compatibility)
- Flat parameter tensor protocol (`params`, `argv`, `set_ptrs`) for Cython↔C++ bridge
- Plan system (`_SET_PLAN_`) for kernel dispatch
- MPI 4D process grid auto-factorization

## Phase 2: Core Feature Completion (2026-05-04 to 2026-05-05)

**What was built:**
- Clover term construction (`_clover.py`) — field strength F_μν from plaquettes, sigma matrix products, MPI halo exchange for 12 gauge link patterns
- Clover dslash in C++: `clover_dslash_single.cu`, `clover_dslash_multi.cu`, `clover_dslash_comm.cu`
- Wilson BiStabCG, Wilson CG, Clover BiStabCG — both Python (`_bistabcg.py` with parity preconditioning) and C++ (`apply_wilson_bistabcg.cu`, `apply_clover_bistabcg.cu`, `apply_wilson_cg.cu`)
- Clover bistabcg dslash parity preconditioning in C++ (`apply_clover_bistabcg_dslash.cu`)
- Multigrid solver (`solver/_multigrid.py`) — level hierarchy construction, null vector generation via BiStabCG inverse iteration, local orthogonalization, Galerkin coarse-grid projection, V-cycle with adaptive level-back
- Multigrid tools (`tools/_multigrid.py`) — `give_null_vecs`, `local_orthogonalize`, `restrict`, `prolong`
- Batch clover inversion (`apply_clovers.cu`) — Clover term + its inverse computed in C++
- Stout smearing (`smear/_stout.py`) with MPI halo exchange support
- MPI correctness fixes in `lattice_set.h` and dslash operators
- Major C++ refactoring: consolidated Wilson/Clover BiStabCG/CG dslash templates in header files

**Key achievements:**
- Full Clover fermion operator (Wilson + clover term) working on both Python and C++ backends
- Multigrid solver with configurable levels, data types, and devices
- All solvers verified against reference HDF5 data

## Phase 3: Optimization (2026-07-05 to 2026-07-08)

**12 optimizations applied across 6 files** (documented in `log/stab23.log`):

| # | Optimization | File | Impact |
|---|-------------|------|--------|
| 1 | Batch matrix inversion | `pyqcu/dslash/_clover.py` | ~10-50x for clover inverse (N loops → 1 batch) |
| 2 | Tensor device/type caching | `pyqcu/dslash/_wilson.py` | Eliminated per-direction `.to()`/`.type()` |
| 3 | I±γ matrix precomputation | `pyqcu/dslash/_wilson.py` | 4 subtractions → dict lookup |
| 4 | Sigma matrix precomputation | `pyqcu/dslash/_clover.py` | 6 `.to()`/`.type()` → dict lookup |
| 5 | Clover coefficient precompute | `pyqcu/dslash/_clover.py` | Eliminated 6 redundant float casts |
| 6 | Remove unnecessary `.clone()` | `pyqcu/dslash/_clover.py` | 3 deep copies eliminated |
| 7 | Cache `give_eo_mask` | `pyqcu/tools/_define.py` | Avoid repeated meshgrid creation |
| 8 | Store `tools.norm(b)` | `pyqcu/solver/_bistabcg.py` | 1 redundant MPI Allreduce removed |
| 9 | Conditional perf_counter | `pyqcu/solver/_bistabcg.py` | Skip timer in silent mode |
| 10 | Remove duplicate import | `pyqcu/solver/_multigrid.py` | Code cleanup |
| 11 | Fix redundant `.flatten()` | `pyqcu/tools/_linalg.py` | Double flatten eliminated |
| 12 | Fix cut_I log message | `pyqcu/dslash/_clover.py` | Correctness fix |

**Reference document:** `refer/dev71.md` — 861-line design doc for CUDA C++ MultiGrid implementation (July 2026). Comprehensive analysis of Python multigrid implementation, C++ backend infrastructure, and optimization strategy for moving multigrid to CUDA. Contains detailed code snippets, architecture diagrams, and implementation roadmap.

**Environment fix:** `env.sh` updated with MPI root permissions and proper library paths.

## Phase 4: CUDA Multigrid Acceleration + Polish (2026-07-14 to 2026-07-20)

**What was built:**
- CUDA multigrid kernels (`cpp/cuda/qcu/src/multigrid.cu`) — 229 lines: restrict, prolong, and coarse dslash CUDA kernels with multi-GPU support
- Cython bridge expanded: `applyMultigridRestrictQcu`, `applyMultigridProLongQcu`, `applyMultigridCoarseDslashQcu`
- Python multigrid CUDA integration (`pyqcu/solver/_multigrid.py`) — `_restrict_cuda`, `_prolong_cuda`, `_coarse_dslash_cuda` methods that pack/unpack data for the C++ backend, with per-level caching
- CLAUDE.md created and iteratively improved (3 revisions covering architecture, build, and conventions)
- `.gitignore` added

## Current State (2026-07-20)

### Working ✅
- Wilson and Clover Dirac operators (Python + C++ CUDA)
- BiStabCG and CG solvers (Python + C++ CUDA)
- Wilson and Clover parity-preconditioned solvers (C++ CUDA)
- Multigrid solver (Python, with optional CUDA finest-level smoothing)
- Stout gauge smearing (Python)
- Gauge field generation and SU(3) validation
- MPI-distributed 4D process grid with halo exchange
- HDF5 I/O (MPI parallel + serial fallback)
- NPU compatibility layer (Ascend)
- TileLang JIT kernels for specific einsum patterns
- Perfetto profiling support
- CUDA-accelerated restrict/prolong/coarse-dslash in multigrid

### Stubs / Placeholders
- `cpp/cann/qcu/`, `cpp/dtk/qcu/`, `cpp/maca/qcu/` — PASS stubs only
- `pyqcu/solver/_gmres.py` — PASS stub
- `pyqcu/dtk/`, `pyqcu/maca/` — PASS stubs

### Known Gaps / TODO
- **CUDA multigrid main loop** — `multigrid.cu` has restrict/prolong/coarse-dslash kernels but no full V-cycle loop in C++. The V-cycle logic lives in `pyqcu/solver/_multigrid.py`.
- **GMRES solver** — stub only
- **DCU/CANN/MACA backends** — no implementation beyond stubs
- **CUDA coarse-grid operator construction** — Galerkin projection is done in Python, not on GPU

### Key Reference Files
| File | Content |
|------|---------|
| `refer/dev71.md` | CUDA C++ MultiGrid design document (861 lines) |
| `refer/dev71.pdf` | PDF version of the design doc |
| `refer/dev71.tex` | LaTeX source for the design doc |
| `log/stab23.log` | Optimization report (12 optimizations, July 2026) |
| `examples/data/` | Reference HDF5 files for Wilson and Clover validation |
```
