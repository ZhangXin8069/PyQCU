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
