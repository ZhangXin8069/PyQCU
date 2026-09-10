# AGENTS.md — examples

PyQCU 测试示例与基准，按后端目标组织。

## 目录映射

| 目录 | 目标 | 说明 |
|---|---|---|
| `pyqcu/` | CPU/CUDA/NPU | 纯 Python 算子/求解器测试（主测试套件） |
| `qcu/` | NVIDIA CUDA | C++ CUDA 后端测试（经 Cython 桥） |
| `cpu/` | CPU | 纯 CPU 测试（BiStabCG、MPI） |
| `npu/` | 昇腾 NPU | NPU 专属测试 |
| `dcu/` | AMD DCU | DCU/ROCm 测试 |
| `profiler/` | 全 | torch.profiler Perfetto 追踪 |
| `benchmark/` | 全 | 性能基准 |
| `tilelang/` | CUDA | TileLang 内核测试 |
| `gpu/` | GPU | 空 — GPU 测试占位 |
| `data/` | — | 校验用参考 HDF5（`with_data=True`） |

## 运行测试

```bash
cd examples && pytest .                              # 全部 conftest.py
mpirun -np 4 python examples/pyqcu/conftest.py       # 单文件 MPI
```

每个子目录有自己的 `conftest.py`，从 `pyqcu.testing` 导入测试函数并调用。Conftest 文件手动编辑取消注释所需测试。

## 参考数据

`examples/data/` 存放预计算规范场、源与期望结果的 HDF5，用于 `with_data=True` 校验。
