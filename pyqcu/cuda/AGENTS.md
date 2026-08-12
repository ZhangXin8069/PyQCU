# AGENTS.md — pyqcu.cuda

C++ CUDA 后端（`libqcu.so`）的 Cython 桥接包。

## 文件

| 文件 | 用途 |
|---|---|
| `__init__.py` | 包标记（2026-07-28 R3 添加，缺失曾导致 `pip install` 失败） |
| `qcu/qcu.pyx` | Cython 扩展源 — 封装 `pyqcu.h` 的 C 函数 |
| `qcu/qcu.pxd` | `cdef extern` 声明（必须与 `pyqcu.h` 完全一致） |
| `qcu/qcu.pyi` | 类型桩（155 行，IDE 支持） |
| `define.py` | 参数常量与 dtype 转换助手 |

## 参数协议（三个扁平张量桥接 Python ↔ C++）

- **`params`**（int32, 54）— 格点维度、网格、数据类型、迭代数、plan 选择、奇偶、MG 层配置
- **`argv`**（float, 7）— `_MASS_`/`_ATOL_`/`_SIGMA_`/MG 各层容差
- **`set_ptrs`**（int64, 100）— C++ 运行时的 scratch 指针

`define.py` 的索引常量必须与 `cpp/cuda/qcu/include/define.h` 同步；`define.py` 提供预建 `params`/`argv`/`set_ptrs` 张量（求解器原地修改）。

## 关键约束

- 同一 `applyInitQcu`/`applyEndQcu` 生命周期内，**每次调用后必须 `params[define._SET_INDEX_] += 1`**，否则 scratch 复用冲突产生错误结果。例外：粗网格 dslash 将 `_SET_INDEX_` 重置为 0。
- 所有函数以 `tensor.contiguous().data_ptr()` 强转 `long long` 传裸指针。

## Plan 选择

| 常量 | 值 | 用途 |
|---|---|---|
| `_SET_PLAN_N_2_` | -2 | Laplacian |
| `_SET_PLAN_N_1_` | -1 | Gauss 规范场生成 |
| `_SET_PLAN0_` | 0 | Wilson dslash |
| `_SET_PLAN1_` | 1 | BiStabCG/CG |
| `_SET_PLAN2_` | 2 | Clover dslash |

## 数据映射

`define.dtype(data_type)` / `define.epytd(torch_dtype)` 双向转换；`define.lat_shape(params)` 提取 `[Lt, Lz, Ly, Lx]`。

## 生命周期

```python
qcu.applyInitQcu(set_ptrs, params, argv)  # 分配
# 操作，调用之间 _SET_INDEX_ += 1
qcu.applyEndQcu(set_ptrs, params)         # 释放
```
