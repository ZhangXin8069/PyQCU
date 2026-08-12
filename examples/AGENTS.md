# AGENTS.md — examples

按后端目标组织的 PyQCU 测试示例与基准。

## 目录

| 目录 | 目标 | 说明 |
|---|---|---|
| `pyqcu/` | CPU/CUDA/NPU | 纯 Python 算子/求解器测试（主测试套件） |
| `qcu/` | NVIDIA CUDA | 经 Cython 桥测 C++ CUDA 后端 |
| `cpu/` | CPU | CPU 专用测试（BiStabCG、MPI） |
| `npu/` | 昇腾 NPU | NPU 专用测试 |
| `dcu/` | AMD DCU | DCU/ROCm 测试 |
| `profiler/` | 全 | `torch.profiler` Perfetto 追踪 |
| `benchmark/` | 全 | 性能基准 |
| `tilelang/` | CUDA | TileLang 内核测试 |
| `gpu/` | GPU | 空 — GPU 测试占位 |
| `data/` | — | 参考 HDF5 文件（`with_data=True` 验证用） |

## 运行

```bash
cd examples && pytest .                              # 全部 conftest.py
mpirun -np 4 python examples/pyqcu/conftest.py       # 单文件 + MPI
```

每个子目录自带 `conftest.py`，从 `pyqcu.testing` 导入测试函数并调用；conftest 通过手动注释/取消注释选择测试。

## 参考数据

`examples/data/` 存放预计算规范场、源与期望结果的 HDF5 文件，`with_data=True` 时用于验证。

## Skills

`skills/` 下子目录的完整生成 skill（原 CLAUDE.md 内容）：`pyqcu`、`qcu`、`cpu`、`npu`、`dcu`、`profiler`、`benchmark`、`tilelang`、`gpu`、`data`。
