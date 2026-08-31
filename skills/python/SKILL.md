---
name: python
description: cpp/cuda/qcu/python 目录的 C API 维护 skill：同步 pyqcu.h、qcu_api.pxd、qcu.pyx/qcu.pyi，以及 Strict MultiGrid ABI。
---
# cpp/cuda/qcu/python

`cpp/cuda/qcu/python` 是 C++ CUDA 后端与 Python/Cython 之间的 ABI 边界。

## Files

| File | Purpose |
|------|---------|
| `pyqcu.h` | 当前 `extern "C"` 声明；字段指针统一以 `long long` 传递，包含 legacy 与 Strict 入口 |
| `qcu_api.pxd` | Cython 的别名声明；每个 C 调用必须带 `nogil`，并与 `pyqcu.h` 逐项同步 |
| `qcu.pyx` / `qcu.pyi` | Python 包装与类型存根；包装层负责连续性/形状门禁和结果转换 |

`pyqcu.h` 必须与 `pyqcu/cuda/qcu/qcu_api.pxd` 完全同步；声明、参数顺序或返回类型
不一致可能导致静默内存破坏。

## Shared flat-array ABI

所有需要运行时控制的入口共享以下 CPU 控制数组：

- `set_ptrs`: `int64[100]`，低槽位为 `LatticeSet`/scratch 句柄；Strict transition
  `t` 使用 `60+4*t+0..3` 存放 `V`、可选 raw `Y`、`Yhat`、`(X,X^-1)`，槽位 `80`
  存放持久 Strict hierarchy 句柄。
- `params`: `int32[58]`，索引常量由 `pyqcu/cuda/define.py` 与
  `cpp/cuda/qcu/include/define.h` 镜像；54、55、56、57 分别是
  `_MG_USE_GCR_`、`_MG_USE_DEFLATE_`、`_MG_MU_PRE_`、`_MG_USE_INIT_GUESS_`。
  `params[57]` 只能为 0/1：0 表示冷启动，1 表示从调用方预填的 `fermion_out`
  奇半场读取 x0，不改变数组长度。
- `argv`: 7 元实数张量：mass、atol、sigma、四个 MG 容差；c64 使用
  `float32`，c128 使用 `float64`。

数据张量必须在取 `data_ptr()` 前满足所需 dtype、device、shape 和 contiguous
约束；Cython 在 `with nogil` 段调用 C API。

## Strict C API surface

Strict 入口分为：

- full-coarse `X/Yhat` 应用与 compact MATPC；
- full↔target-parity 的 prepare/reconstruct，以及 compact fine ↔ full coarse 的 R/P；
- 持久层级的 init、recursive V-cycle、end；
- 单 rank fused restarted right-FGMRES。

Strict primitive 返回 `int` 状态；init/V-cycle/fused FGMRES 另通过
`unsigned long long*` 返回其自有 arena 的精确字节数，fused 入口还返回迭代次数、
收敛标志和最终 true residual。Cython 声明必须保持 `nogil`，且 Strict 生命周期
内复用同一 `params[_SET_INDEX_]` 槽位；legacy 连续操作才执行 `_SET_INDEX_ += 1`。
