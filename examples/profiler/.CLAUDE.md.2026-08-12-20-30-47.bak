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
