# AGENTS.md — pyqcu.cuda.qcu

Cython 扩展模块 — 桥接 Python 与 C++ CUDA 后端 `libqcu.so`。

## 文件

| 文件 | 用途 |
|---|---|
| `qcu.pyx` | Cython 源码：`pyqcu.h` C 函数的薄封装 |
| `qcu.pxd` | Cython 声明：`cdef extern` 块（必须与 `pyqcu.h` 完全一致） |
| `qcu.pyi` | Python 类型 stub（IDE 自动补全） |

## C API 表面

全部 22 个 C 函数暴露。每个接收 `long long` 的裸张量指针：`applyInitQcu`/`applyEndQcu`、`applyWilsonDslashQcu`、`applyCloverDslashQcu`、`applyWilsonBistabCgQcu`(+Dslash)、`applyWilsonCgQcu`(+Dslash)、`applyCloverBistabCgQcu`(+Dslash)、`applyCloverQcu`/`applyCloversQcu`、`applyDslashQcu`、`applyLaplacianQcu`、`applyGaussGaugeQcu`、`applyMultigridRestrictQcu`/`applyMultigridProLongQcu`、`applyMultigridCoarseDslashQcu`、`applyCloverMultigridQcu`。

## 调用生命周期

```python
qcu.applyInitQcu(set_ptrs, params, argv)   # 分配缓冲
# ... 执行操作 ...
params[define._SET_INDEX_] += 1              # 调用间必须递增
qcu.applyEndQcu(set_ptrs, params)            # 释放缓冲
```

## 同步

`.pxd` 必须与 `cpp/cuda/qcu/python/pyqcu.h` 的 C 声明精确匹配。任何不匹配导致静默内存损坏。
