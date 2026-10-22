---
name: pyqcu-testing-npu
description: 当任务涉及 pyqcu/testing/npu、Ascend NPU 或 force_use_npu 测试时使用。
---
# pyqcu/testing/npu

Ascend NPU tests. Use `pyqcu.cann.force_use_npu = True` to test NPU code paths on CPU without NPU hardware.

## Test Files

| File | What it tests |
|------|---------------|
| `conftest.py` | Main NPU test entry |
| `conftest_1.py` | Additional NPU test variant |

## Usage

```bash
python pyqcu/testing/npu/conftest.py
```
