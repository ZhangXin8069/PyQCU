# AGENTS.md — examples/profiler

用 `torch.profiler` 做性能剖析，导出 Chrome trace 供 Perfetto 可视化。

## 测试文件

| 文件 | 剖析对象 |
|---|---|
| `conftest.py` | 主剖析入口 |
| `conftest.cpu.py` | CPU 剖析 |
| `conftest.cuda.py` | CUDA GPU 剖析 |
| `conftest.npu.py` | NPU 剖析 |

## 剖析配置

`torch.profiler.profile(...)`，`record_shapes=True`、`with_modules=True`、`with_flops=True`。

## 用法

```bash
cd examples/profiler && mpirun -np 1 python -u conftest.py
# 将 trace_*.json 载入 https://ui.perfetto.dev
```
