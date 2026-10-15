---
name: pyqcu-testing-gpu
description: pyqcu/testing/gpu 目录的完整生成 skill：通用 GPU 测试占位（当前为空）。
---
# pyqcu/testing/gpu

GPU test placeholder. Currently empty — no test files.

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
