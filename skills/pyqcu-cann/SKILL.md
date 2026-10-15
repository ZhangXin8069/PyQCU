---
name: pyqcu-cann
description: pyqcu.cann 目录的完整生成 skill：Ascend NPU 的 torch 兼容层，将复数算子分解为实/虚部；含 force_use_npu 全局标志与 einsum 组合分解算法。
---
# pyqcu.cann

Torch compatibility layer for Ascend NPU. All Python code in PyQCU imports `pyqcu.cann as _torch` instead of using `torch` directly.

## Problem

Ascend NPU does not natively support complex tensors. This module wraps torch operations, decomposing complex ops into real/imaginary parts on NPU while passing through directly on CUDA/CPU.

## Behavior

- **CUDA/CPU path** (`device.type != 'npu'` and `force_use_npu=False`): delegates to `torch.*` unchanged
- **NPU path** (`device.type == 'npu'` or `force_use_npu=True`): decomposes complex ops into real/imaginary parts

## Global Flag

`pyqcu.cann.force_use_npu = True` — force NPU code paths on CPU for testing without NPU hardware. This affects only the `cann` layer; some modules (`dslash/_wilson.py`, `tools/_define.py`, `tools/_multigrid.py`, `smear/_stout.py`) also have their own per-module `force_use_npu` flag for deeper NPU workarounds (e.g., tensor dimension limits).

## Functions Provided

Always use these instead of raw torch calls anywhere complex tensors might run on NPU:

| Category | Functions | Notes |
|----------|-----------|-------|
| Math | `abs`, `vdot`, `norm`, `sqrt`, `matmul` | `vdot` → conj-flatten-sum; `norm` → abs-then-norm; `sqrt` → CPU fallback |
| Reduction/shape | `roll`, `allclose`, `einsum` | `roll` → roll real/imag separately; `allclose` → check real + imag separately |
| Creation | `zeros`, `zeros_like`, `randn`, `randn_like`, `eye` | Creates real parts then combines to complex |
| Linear algebra | `linalg_qr` | Falls back to CPU on NPU for complex inputs |

### Uses of raw `torch`

- `torch.linalg.det` — used in `lattice.check_su3()` for SU(3) determinant check. No equivalent in `_torch`; works on NPU for real matrices.
- `torch.matrix_exp` — used in `lattice.generate_gauge_field()` for exponential map.

## Einsum on NPU

General N-operand complex einsum uses a combinatorial approach. For Z = Π(a_k + i·b_k):

- Iterates all 2ⁿ sign combinations (bitmask k → real/imag selection for each operand)
- Even number of imaginary parts → contributes to real part with sign = (-1)^(n_imag/2)
- Odd number of imaginary parts → contributes to imaginary part with sign = (-1)^((n_imag-1)/2)
- 2-operand special case: explicit ac-bd + i(ad+bc) formula (faster)

## Key Implementation Details

- `eye(n, m, ...)` — creates real identity then casts to complex dtype on NPU
- `zeros(*args, ...)` / `randn(*args, ...)` — creates separate real + imag tensors and combines
- `sqrt(input)` — sends complex input to CPU, computes sqrt, sends back (NPU doesn't support complex sqrt)
- `matmul(input, other)` — uses explicit (ac-bd) + i(ad+bc) decomposition

## Subdirectory

`qcu/` — placeholder stub（空 `PASS` 文件），无实现。

占位后端目录的镜像约定（`cpp/{cann,dtk,maca}` 三处同构，均只有一个 `qcu/PASS`）见
`skills/pyqcu-cpp-containers/SKILL.md`。本文件此前内嵌的「Complete Skill: `qcu/`」全文副本已于
2026-10-15 删除——其声称的源文件 `qcu/CLAUDE.md` 在仓库中并不存在，且同一事实上方一行已给出。

