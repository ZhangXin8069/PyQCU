---
name: cpu
description: examples/cpu 目录的完整生成 skill：纯 Python 后端的 CPU 专用测试。
---
# CLAUDE.md — examples/cpu

CPU-only tests for the pure-Python backend.

## Test Files

| File | What it tests |
|------|---------------|
| `conftest.py` | Main entry — imports from `pyqcu.testing` |
| `conftest.bistabcg.py` | BiStabCG solver on CPU |
| `conftest.mpi.py` | MPI-distributed solver on CPU |

## Usage

```bash
mpirun -np 4 python examples/cpu/conftest.py
```
