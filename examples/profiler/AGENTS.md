# AGENTS.md — examples.profiler

`torch.profiler` 性能剖析，导出 Chrome trace 格式供 Perfetto 可视化。

## 测试文件

| 文件 | 剖析对象 |
|---|---|
| `conftest.py` | 主剖析入口 |
| `conftest.cpu.py` | CPU 剖析 |
| `conftest.cuda.py` | CUDA GPU 剖析 |
| `conftest.npu.py` | NPU 剖析 |

## 配置

`torch.profiler.profile(...)`，`record_shapes=True`、`with_modules=True`、`with_flops=True`。

## 运行

```bash
cd examples/profiler && mpirun -np 1 python -u conftest.py
# 将生成的 trace_*.json 载入 https://ui.perfetto.dev
```
