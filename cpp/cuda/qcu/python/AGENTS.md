# AGENTS.md — cpp.cuda.qcu.python

面向 Python 的 C API 声明 — C++ CUDA 后端与 Python Cython 桥的接口边界。

## 文件

| 文件 | 用途 |
|---|---|
| `pyqcu.h` | C API 头 — 22 个 `extern "C"` 函数，裸指针以 `long long` 传递 |

**必须与 `pyqcu/cuda/qcu/qcu.pxd` 完全同步**，任何不匹配都会导致静默内存损坏。

所有函数取三个参数数组：
- `set_ptrs` (int64[100])：C++ 运行时管理的 scratch 指针
- `params` (int32[54])：格点维度、网格、数据类型、迭代数、plan 选择
- `argv` (float64[7])：mass、atol、sigma、MG 容差

C++→Python 数据指针来自 `tensor.contiguous().data_ptr()` 强转 `long long`。
