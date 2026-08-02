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

---

## Complete Skills (Agent-Produced Subdirectories)

The content of each subdirectory below was produced with Claude Code assistance. Per repo convention, the complete skill that generates that content is reproduced verbatim below (source: the subdirectory's own `CLAUDE.md`), so the full knowledge is available directly at this level.

### Complete Skill: `pyqcu/` (source: `pyqcu/CLAUDE.md`)

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

### Complete Skill: `qcu/` (source: `qcu/CLAUDE.md`)

# CLAUDE.md — examples/qcu

C++ CUDA backend tests via the Cython bridge. These exercise `libqcu.so` from Python.

## Test Files

| File | What it tests |
|------|---------------|
| `conftest.cuda.py` | Basic CUDA availability and context |
| `conftest.mpi.py` | MPI grid setup and halo exchange |
| `conftest.wilson.bistabcg.py` | Wilson BiStabCG via C++ backend |
| `conftest.wilson.bistabcg.dslash.py` | Wilson BiStabCG dslash kernel |
| `conftest.wilson.cg.py` | Wilson CG solver |
| `conftest.clover.py` | Clover term construction |
| `conftest.clover.bistabcg.py` | Clover BiStabCG solver |
| `conftest.clover.bistabcg.dslash.py` | Clover BiStabCG dslash |
| `conftest.clover.multigrid.py` | Clover multigrid V-cycle solver |

## Usage

```bash
mpirun -np 1 python examples/qcu/conftest.clover.multigrid.py
```

Output: convergence log → `logs/clover_multigrid.log`, performance report → `logs/clover_multigrid_report.log`

### Complete Skill: `cpu/` (source: `cpu/CLAUDE.md`)

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

### Complete Skill: `npu/` (source: `npu/CLAUDE.md`)

# CLAUDE.md — examples/npu

Ascend NPU tests. Use `pyqcu.cann.force_use_npu = True` to test NPU code paths on CPU without NPU hardware.

## Test Files

| File | What it tests |
|------|---------------|
| `conftest.py` | Main NPU test entry |
| `conftest_1.py` | Additional NPU test variant |

## Usage

```bash
python examples/npu/conftest.py
```

### Complete Skill: `dcu/` (source: `dcu/CLAUDE.md`)

# CLAUDE.md — examples/dcu

AMD DCU / ROCm (HIP) tests.

## Test Files

| File | What it tests |
|------|---------------|
| `conftest.py` | Main DCU test entry |
| `conftest_1.py` | DCU test variant 1 |
| `conftest_2.py` | DCU test variant 2 |

## Usage

```bash
python examples/dcu/conftest.py
```

### Complete Skill: `profiler/` (source: `profiler/CLAUDE.md`)

# CLAUDE.md — examples/profiler

Performance profiling with `torch.profiler`. Exports Chrome trace format for visualization in Perfetto.

## Test Files

| File | What it profiles |
|------|-----------------|
| `conftest.py` | Main profiler entry |
| `conftest.cpu.py` | CPU profiling |
| `conftest.cuda.py` | CUDA GPU profiling |
| `conftest.npu.py` | NPU profiling |

## Profiler Configuration

Uses `torch.profiler.profile(...)` with `record_shapes=True`, `with_modules=True`, `with_flops=True`.

## Usage

```bash
cd examples/profiler && mpirun -np 1 python -u conftest.py
# Load resulting trace_*.json into https://ui.perfetto.dev
```

### Complete Skill: `benchmark/` (source: `benchmark/CLAUDE.md`)

# CLAUDE.md — examples/benchmark

Performance benchmarks comparing PyTorch, TileLang, and C++ CUDA backend implementations.

## Files

| File | Purpose |
|------|---------|
| `conftest.py` | Benchmark entry point |
| `env.py` | Benchmark environment configuration |

## Usage

```bash
python examples/benchmark/conftest.py
```

### Complete Skill: `tilelang/` (source: `tilelang/CLAUDE.md`)

# CLAUDE.md — examples/tilelang

TileLang JIT-compiled kernel tests for CUDA. Exercises the TileLang integration in `pyqcu/tools/_einsum.py` and `pyqcu/tools/_matul.py`.

## Files

| File | Purpose |
|------|---------|
| `conftest.py` | TileLang test entry |

## Usage

```bash
python examples/tilelang/conftest.py
```

Requires TileLang to be installed (optional dependency, silently degrades if unavailable).

### Complete Skill: `gpu/` (source: `gpu/CLAUDE.md`)

# CLAUDE.md — examples/gpu

GPU test placeholder. Currently empty — no test files.

Intended for generic GPU tests that don't fit into the more specific `qcu/` (NVIDIA CUDA), `dcu/` (AMD ROCm), or `npu/` (Ascend NPU) directories.

### Complete Skill: `data/` (source: `data/CLAUDE.md`)

# CLAUDE.md — examples/data

Reference HDF5 files for test validation. Used when tests are run with `with_data=True` to validate against precomputed expected results.

Contains gauge fields, fermion sources, and expected operator/solver outputs for various lattice sizes.

