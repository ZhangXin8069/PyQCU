# AGENTS.md — examples.npu

昇腾 NPU 测试。用 `pyqcu.cann.force_use_npu = True` 可在无 NPU 硬件时于 CPU 上测试 NPU 代码路径。

## 测试文件

| 文件 | 覆盖 |
|---|---|
| `conftest.py` | 主 NPU 测试入口 |
| `conftest_1.py` | NPU 测试变体 |

## 运行

```bash
python examples/npu/conftest.py
```
