# CLAUDE.md — pyqcu.cann

Torch compatibility layer for Ascend NPU. All Python code in PyQCU imports `pyqcu.cann as _torch` instead of using `torch` directly.

## Problem

Ascend NPU does not natively support complex tensors. This module wraps torch operations, decomposing complex ops into real/imaginary parts on NPU while passing through directly on CUDA/CPU.

## Behavior

- **CUDA/CPU path** (`device.type != 'npu'`): delegates to `torch.*` unchanged
- **NPU path** (`device.type == 'npu'` or `force_use_npu=True`): decomposes complex ops into real/imaginary parts

## Global Flag

`pyqcu.cann.force_use_npu = True` — force NPU code paths on CPU for testing without NPU hardware.

## Functions Provided

Always use these instead of raw torch calls anywhere complex tensors might run on NPU:

| Category | Functions |
|----------|-----------|
| Math | `abs`, `vdot`, `norm`, `sqrt`, `matmul` |
| Reduction/shape | `roll`, `allclose`, `einsum` |
| Creation | `zeros`, `zeros_like`, `randn`, `randn_like`, `eye` |
| Linear algebra | `linalg_qr` (falls back to CPU on NPU for complex inputs) |

## Einsum on NPU

General N-operand complex einsum uses a combinatorial approach: for Z = Π(a_k + i·b_k), iterates all 2ⁿ sign combinations with correct i^n factors:
- Even number of imaginary parts → contributes to real part
- Odd number of imaginary parts → contributes to imaginary part

## Subdirectory

`qcu/` — placeholder stub (empty `PASS` file), no implementation yet.
