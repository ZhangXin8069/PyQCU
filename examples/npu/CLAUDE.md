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
