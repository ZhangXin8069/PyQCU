---
name: pyqcu-testing-dcu
description: 当任务涉及 pyqcu/testing/dcu、AMD DCU 或 ROCm/HIP 测试时使用。
---
# pyqcu/testing/dcu

AMD DCU / ROCm (HIP) tests.

DCU build selection is `PYQCU_BUILD_BACKEND=dcu` or `bash install.sh dcu`. Native
build uses `cpp/cuda/qcu/make-dcu.sh` + `CMakeLists-dcu.txt`; `cpp/dtk` and
`pyqcu/dtk` remain placeholders. Probe the compiler without a DCU device:

```bash
QCU_DCU_CUDA_COMPILER=/path/to/dtk/nvcc \
  bash cpp/cuda/qcu/make-dcu.sh --check
python -B pyqcu/testing/dcu/test_install_selection.py
```

Runtime still uses ROCm PyTorch's CUDA-compatible API and `backend="cuda"`. Do
not claim a `dcu` runtime selector exists. Hardware tests must show non-empty
`torch.version.hip`; actual numerical smoke and performance remain unverified
until run on DCU. See `docs/dcu_build_backend.md`.

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
