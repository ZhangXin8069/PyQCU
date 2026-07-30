# CLAUDE.md — pyqcu.cuda

Cython bridge package for the C++ CUDA backend (`libqcu.so`).

## Files

| File | Purpose |
|------|---------|
| `qcu/qcu.pyx` | Cython extension source — wraps C functions from `pyqcu.h` |
| `qcu/qcu.pxd` | Cython declaration file — `cdef extern` block matching `pyqcu.h` |
| `qcu/qcu.pyi` | Type stub (155 lines) — full type annotations, docstrings, and default values for IDE support |
| `define.py` | Parameter constants (`_LAT_X_`, `_SET_PLAN_`, etc.) and dtype conversion helpers |

## Public API

```python
from pyqcu.cuda import qcu      # Cython bridge to libqcu.so
from pyqcu.cuda import define   # Parameter constants and dtype helpers
```

## Parameter Protocol

Three flat tensors bridge Python ↔ C++:
- **`params`** (int32, size 54) — lattice dims, grid sizes, data types, iteration counts, plan selection, MG level configs
- **`argv`** (float64, size 7) — physical parameters: mass (idx 0), atol (1), sigma (2), per-level MG tolerances (3–6)
- **`set_ptrs`** (int64, size 100) — scratch pointers managed by the C++ runtime

Index constants in `define.py` MUST stay in sync with `cpp/cuda/qcu/include/define.h`.

## Cython Extension Details

Built via `install.sh` → `python setup.py build_ext --inplace`. The extension links against `libqcu.so` (built by `build.sh`). All C functions take raw pointers cast to `long long` from `tensor.contiguous().data_ptr()`.

## Critical: `_SET_INDEX_` Increment

Between successive C++ calls, you MUST increment `params[define._SET_INDEX_]` by 1. Failing to do so causes scratch buffer reuse conflicts that produce wrong results.

## Data Type Mapping

- `define.dtype(data_type)` — QCU internal constant → PyTorch dtype
- `define.epytd(torch_dtype)` — PyTorch dtype → QCU internal constant
