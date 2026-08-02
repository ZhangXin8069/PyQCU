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

Development logs, review reports, bug fix summaries, and solver output. This directory is gitignored — contents are not versioned.

## File Patterns

| Pattern | Purpose |
|---------|---------|
| `dev<N>.md` | Development milestone reports |
| `bug<N>.md` | Bug discovery & code review reports |
| `review-*.md` | Code review findings (e.g., `review-2026-07-28.md`) |
| `fix-report-*.md` | Bug fix summaries |
| `multigrid_report.md` | MG solver performance reports |
| `clover_multigrid.log` | C++ solver convergence output |
| `*.png` | Performance charts, convergence plots |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `debug/` | Per-round fix logs (`fix-log*.md`) |
| `results/` | Final/remaining fix reports |

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

