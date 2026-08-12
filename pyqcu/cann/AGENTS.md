# AGENTS.md — pyqcu.cann

昇腾 NPU 的 torch 兼容层。PyQCU 全部 Python 代码用 `import pyqcu.cann as _torch` 代替直接使用 `torch`。

## 行为

- **CUDA/CPU 路径**（`device.type != 'npu'` 且 `force_use_npu=False`）：原样委托 `torch.*`。
- **NPU 路径**：复数算子分解为实/虚部分别执行。
- `pyqcu.cann.force_use_npu = True`：无 NPU 硬件时在 CPU 上强制 NPU 路径（测试用）。部分模块（`dslash/_wilson.py`、`tools/_define.py`、`tools/_multigrid.py`、`smear/_stout.py`）有各自 `force_use_npu` 标志。

## 提供函数（NPU 场景一律用这些，勿裸调 torch）

| 类别 | 函数 |
|---|---|
| 数学 | `abs`、`vdot`、`norm`、`sqrt`（CPU 回退）、`matmul` |
| 归约/形状 | `roll`、`allclose`（实虚分别比）、`einsum` |
| 创建 | `zeros`、`zeros_like`、`randn`、`randn_like`、`eye` |
| 线性代数 | `linalg_qr`（NPU 复数回退 CPU） |

裸 `torch` 例外：`torch.linalg.det`（`lattice.check_su3`）、`torch.matrix_exp`（规范场生成）。

## Einsum 组合分解

N 操作数复数 einsum：遍历 2ⁿ 个符号组合（bitmask 选择各操作数实/虚）；偶数个虚部 → 实部，符号 (−1)^(n_imag/2)；奇数个虚部 → 虚部，符号 (−1)^((n_imag−1)/2)。2 操作数特例用显式 (ac−bd) + i(ad+bc)。

## 子目录

`qcu/` — 昇腾 NPU C++ Cython 桥占位（仅空 `PASS` 文件，未实现）。
