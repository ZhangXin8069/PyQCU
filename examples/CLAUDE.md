# CLAUDE.md — examples

Test examples and benchmarks for PyQCU, organized by backend target.

## Directory Map

| Directory | Target | Description |
|-----------|--------|-------------|
| `pyqcu/` | CPU/CUDA/NPU | Pure-Python operator/solver tests (main test suite) |
| `qcu/` | NVIDIA CUDA | C++ CUDA backend tests via Cython bridge |
| `cpu/` | CPU | CPU-only tests (BiStabCG, MPI) |
| `npu/` | Ascend NPU | NPU-specific tests |
| `dcu/` | AMD DCU | DCU/ROCm tests |
| `profiler/` | All | Perfetto tracing with `torch.profiler` |
| `benchmark/` | All | Performance benchmarks |
| `tilelang/` | CUDA | TileLang kernel tests |
| `gpu/` | GPU | Empty — GPU test placeholder |
| `data/` | — | Reference HDF5 files for validation (`with_data=True`) |

## Running Tests

```bash
cd examples && pytest .                              # all conftest.py files
mpirun -np 4 python examples/pyqcu/conftest.py       # single file with MPI
```

Each subdirectory has its own `conftest.py` that imports test functions from `pyqcu.testing` and calls them. Conftest files are manually edited to uncomment desired tests.

## Reference Data

`examples/data/` contains HDF5 files with precomputed gauge fields, sources, and expected results used for validation when tests are run with `with_data=True`.
