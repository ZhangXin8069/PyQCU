---
name: pyqcu-testing-profiler
description: pyqcu/testing/profiler 目录的完整生成 skill：torch.profiler 性能剖析，导出 Chrome trace 供 Perfetto 可视化。
---
# pyqcu/testing/profiler

Performance profiling with `torch.profiler`. Exports Chrome trace format for visualization in Perfetto.

GPU/DCU/GPGPU profiling follows one Python process = one MPI rank = one
device. Do not profile a single-process multi-device production worker as the
default topology; `MultiGpuMultigrid` profiling is test-only. See
[`pyqcu-cuda/SKILL.md`](../pyqcu-cuda/SKILL.md#cuda-production-topology).

正式 profiler 工作流和历史性能索引的使用边界见
[`pyqcu-docs/references/extracted-guidance.md`](../pyqcu-docs/references/extracted-guidance.md#6-profiler)。

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
cd pyqcu/testing/profiler && mpirun -np 1 python -u conftest.py
# Load resulting trace_*.json into https://ui.perfetto.dev
```
