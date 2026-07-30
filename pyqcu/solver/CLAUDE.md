# CLAUDE.md — pyqcu.solver

Iterative solvers for the Dirac equation D ψ = η.

## Files

| File | Purpose |
|------|---------|
| `_bistabcg.py` | BiCGStab(l) solver (Bi-stabilized Conjugate Gradient) |
| `_multigrid.py` | Adaptive multigrid (AMG) V-cycle solver |
| `_gmres.py` | GMRES solver — **placeholder stub, not yet implemented** |

## BiStabCG (`_bistabcg.py`)

- Implements BiCGStab with configurable stabilization parameter L
- Supports both full operator and even-odd preconditioned modes
- Provides the smoother for the multigrid solver at each level

## Multigrid (`_multigrid.py`)

The multigrid solver supports:
- **Adaptive level-back mechanism** — drops to coarsest level when convergence stalls
- **Optional CUDA acceleration at finest level** — via `with_cuda_qcu=True` (enabled automatically when `clover_ee_inv` and `clover_oo_inv` are provided)
- **Configurable degrees of freedom, data types, and devices per level**
- **Configurable max_level, num_restart, smoother iterations**

The multigrid can mix both execution layers: finest-level smoothing via the C++ backend and coarser levels in pure Python.

Null vectors are generated via inverse iteration (`tools.give_null_vecs`) and locally orthogonalized (`tools.local_orthogonalize`).

## Solver Interface

Both solvers follow the same calling convention:
```python
solver.bistabcg(src, U, kappa, max_iter, atol, ...)
solver.multigrid(src, U, kappa, max_iter, atol, max_level, null_vecs, ...)
```
