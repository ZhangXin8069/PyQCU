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

