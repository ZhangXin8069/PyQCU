# AGENTS.md — pyqcu.cuda.qcu

Cython 扩展模块 — 桥接 Python 与 C++ CUDA 后端 `libqcu.so`。

## 文件

| 文件 | 用途 |
|---|---|
| `qcu.pyx` | Cython 源：`pyqcu.h` C 函数的薄封装 |
| `qcu.pxd` | `cdef extern` 声明（必须与 `pyqcu.h` 完全一致） |
| `qcu.pyi` | Python 类型桩 |

## C API 面（22 个函数，均以 `long long` 传裸指针）

`applyInitQcu`/`applyEndQcu`（scratch 分配/释放）、`applyWilsonDslashQcu`、`applyCloverDslashQcu`、`applyWilsonBistabCgQcu`(+Dslash)、`applyWilsonCgQcu`(+Dslash)、`applyCloverBistabCgQcu`(+Dslash)、`applyCloverQcu`/`applyCloversQcu`、`applyDslashQcu`、`applyLaplacianQcu`、`applyGaussGaugeQcu`、`applyMultigridRestrictQcu`/`ProLongQcu`、`applyMultigridCoarseDslashQcu`、`applyCloverMultigridQcu`。

## 调用生命周期

```python
qcu.applyInitQcu(set_ptrs, params, argv)   # 分配
# ... 调用之间必须 params[define._SET_INDEX_] += 1 ...
qcu.applyEndQcu(set_ptrs, params)          # 释放
```

## 同步

`.pxd` 必须与 `cpp/cuda/qcu/python/pyqcu.h` 的 C 声明完全一致，任何不匹配都导致静默内存损坏。
