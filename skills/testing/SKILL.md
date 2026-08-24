---
name: testing
description: pyqcu.testing 目录的完整生成 skill：全组件集成测试（函数名 test_*，由 examples/*/conftest.py 引用），含参考 HDF5 数据验证与日志约定。
---
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

### `test_smear_wuppertal()` (test16, 2026-08-24)
Wuppertal Gaussian smearing with triple invariants (cpu+cuda both PASS):
- `nstep>=1` guard; U=I fixed point (<1e-4); white-noise contraction ratio < 1.0.
- Golden criterion: np=2/4 constant-source rel≈5e-08.

### `verify_nullvecs()` — block structure required
Null-vector quality diagnostic requires the explicit 10-dim block structure argument
(documented in dev85); non-block layouts fail fast instead of being auto-corrected.

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
