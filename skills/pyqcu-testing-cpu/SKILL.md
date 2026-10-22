---
name: pyqcu-testing-cpu
description: 当任务涉及 pyqcu/testing/cpu 或纯 Python CPU 测试时使用。
---
# pyqcu/testing/cpu

CPU-only tests for the pure-Python backend.

## Test Files

| File | What it tests |
|------|---------------|
| `conftest.py` | Main entry — imports from `pyqcu.testing` |
| `conftest.bistabcg.py` | BiStabCG solver on CPU |
| `conftest.mpi.py` | MPI-distributed solver on CPU |

## Usage

```bash
mpirun -np 4 python pyqcu/testing/cpu/conftest.py
```
