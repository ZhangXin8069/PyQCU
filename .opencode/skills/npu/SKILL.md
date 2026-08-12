---
name: npu
description: examples/npu 目录的完整生成 skill：昇腾 NPU 测试（可用 force_use_npu 在 CPU 上测 NPU 路径）。
---
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
