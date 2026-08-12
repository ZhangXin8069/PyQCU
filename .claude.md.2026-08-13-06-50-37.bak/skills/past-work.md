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
