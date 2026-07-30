# CLAUDE.md — pyqcu.testing

Integration tests for all PyQCU components. Tests are Python functions imported by `examples/*/conftest.py` entry points.

## Test Functions

| Function | What it tests |
|----------|---------------|
| `test_lattice` | SU(3) gauge generation and validation + gamma matrix algebra |
| `test_dslash_wilson` | Wilson Dirac operator; `with_data=True` validates against reference HDF5 data from `examples/data/` |
| `test_dslash_parity` | Parity-preconditioned Wilson+Clover operator with MPI |
| `test_dslash_clover` | Clover term construction |
| `test_solver` | BiStabCG (`method='bistabcg'`) and multigrid (`method='multigrid'`) solver correctness |
| `test_matmul` | TileLang vs PyTorch matmul benchmark |
| `test_smear_stout` | Stout smearing correctness |

## Running Tests

```bash
cd examples && pytest .                              # all conftest.py files
mpirun -np 4 python examples/pyqcu/conftest.py       # single file with MPI
```

Each `examples/*/conftest.py` imports from `pyqcu.testing` and calls specific test functions. The conftest files are manually edited to uncomment the test(s) to run.

## Logging Convention

All test output uses: `PYQCU::TESTING::<MODULE>::\n message`
