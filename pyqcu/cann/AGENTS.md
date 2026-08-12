# AGENTS.md — pyqcu.cann

昇腾 NPU 的 Torch 兼容层。PyQCU 所有 Python 代码 `import pyqcu.cann as _torch` 而非直接 `import torch`。

## 问题

昇腾 NPU 不原生支持复数张量。本模块包装 torch 运算，NPU 上把复数运算分解为实/虚部，CUDA/CPU 上直接透传。

## 行为

- **CUDA/CPU 路径**（`device.type != 'npu'` 且 `force_use_npu=False`）：原样委托 `torch.*`
- **NPU 路径**（`device.type == 'npu'` 或 `force_use_npu=True`）：复数运算分解为实/虚部

## 全局标志

`pyqcu.cann.force_use_npu = True` — 无 NPU 硬件时在 CPU 上强制走 NPU 路径。只影响 cann 层；部分模块（`dslash/_wilson.py`、`tools/_define.py`、`tools/_multigrid.py`、`smear/_stout.py`）另有每模块 `force_use_npu` 标志做更深层 NPU 工作区（如张量维度限制）。

## 提供的函数

任何可能跑 NPU 的复数张量操作，一律用这些而非裸 torch：

| 类别 | 函数 | 说明 |
|---|---|---|
| 数学 | `abs`、`vdot`、`norm`、`sqrt`、`matmul` | `vdot` → conj-flatten-sum；`norm` → abs 再 norm；`sqrt` → CPU 回退 |
| 归约/形状 | `roll`、`allclose`、`einsum` | `roll` → 实虚分开 roll；`allclose` → 实虚分别检查 |
| 创建 | `zeros`、`zeros_like`、`randn`、`randn_like`、`eye` | 先建实部再组合成复数 |
| 线性代数 | `linalg_qr` | NPU 复数输入回退 CPU |

### 裸 torch 使用
- `torch.linalg.det` — `lattice.check_su3()` 的 SU(3) 行列式检查；`_torch` 无等价，NPU 实矩阵可用
- `torch.matrix_exp` — `lattice.generate_gauge_field()` 指数映射

## NPU 上的 Einsum

通用 N 操作数复数 einsum 用组合法。Z = Π(a_k + i·b_k)：
- 遍历全部 2ⁿ 符号组合（bitmask k → 每个操作数的实/虚选择）
- 偶数个虚部 → 贡献实部，符号 (-1)^(n_imag/2)
- 奇数个虚部 → 贡献虚部，符号 (-1)^((n_imag-1)/2)
- 2 操作数特例：显式 ac-bd + i(ad+bc) 公式（更快）

## 关键实现细节

- `eye(n, m, ...)` — 先建实单位阵再在 NPU 上转复数 dtype
- `zeros`/`randn` — 分别建实虚张量再组合
- `sqrt(input)` — 复数输入送 CPU 算 sqrt 再送回（NPU 不支持复数 sqrt）
- `matmul(input, other)` — 显式 (ac-bd) + i(ad+bc) 分解

## 子目录

`qcu/` — 占位 stub（空 `PASS` 文件），暂无实现。
