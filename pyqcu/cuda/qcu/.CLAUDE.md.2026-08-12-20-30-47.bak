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
