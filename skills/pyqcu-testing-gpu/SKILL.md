---
name: pyqcu-testing-gpu
description: 当任务涉及 pyqcu/testing/gpu 的通用 GPU 测试或占位入口时使用。
---
# pyqcu/testing/gpu

GPU test placeholder. Currently empty — no test files.

新增 CUDA/GPGPU 用例时保持一 Python 进程 = 一 MPI rank = 一设备；任何
`MultiGpuMultigrid` 用例必须显式标注测试专用。规则见
[`pyqcu-cuda/SKILL.md`](../pyqcu-cuda/SKILL.md#cuda-production-topology)。

目录内只有两件东西，核对只需一条命令：

```bash
ls pyqcu/testing/gpu/    # → AGENTS.md  __init__.py
```

`__init__.py` 全文只有一行 docstring `"""GPU-specific test entry points."""`——因此
`pytest pyqcu/testing/gpu` 可收集但收集不到任何用例，这是**预期状态**，不是漏文件。

Intended for generic GPU tests that don't fit into the more specific `qcu/` (NVIDIA CUDA),
`dcu/` (AMD ROCm), or `npu/` (Ascend NPU) directories. 新增用例时按
`test_<backend>_<function>.py` 命名（见 `pyqcu/testing/AGENTS.md` 的强制整理约定），
并在 `pyqcu/testing/AGENTS.md` 与 `skills/AGENTS.md` 同步「当前为空」的表述。
