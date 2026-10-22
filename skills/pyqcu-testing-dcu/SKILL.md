---
name: pyqcu-testing-dcu
description: 当任务涉及 pyqcu/testing/dcu、AMD DCU 或 ROCm/HIP 测试时使用。
---
# pyqcu/testing/dcu

AMD DCU / ROCm (HIP) tests.

DCU 与其他 GPGPU 使用同一拓扑：一 Python 进程 = 一 MPI rank = 一设备；
跨设备并行经 MPI。单进程多设备驱动只能是显式测试例外。规则见
[`pyqcu-cuda/SKILL.md`](../pyqcu-cuda/SKILL.md#cuda-production-topology)。

## Test Files

| File | What it tests |
|------|---------------|
| `conftest.py` | Main DCU test entry |
| `conftest_1.py` | DCU test variant 1 |
| `conftest_2.py` | DCU test variant 2 |

## Usage

```bash
python pyqcu/testing/dcu/conftest.py
```
