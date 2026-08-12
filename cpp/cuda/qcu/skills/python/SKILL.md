---
name: python
description: cpp/cuda/qcu/python 目录的完整生成 skill：pyqcu.h C API 声明（22 个 extern C 函数），必须与 qcu.pxd 完全同步。
---
# CLAUDE.md — cpp/cuda/qcu/python

Python-facing C API declarations. This is the interface boundary between the C++ CUDA backend and the Python Cython bridge.

## Files

| File | Purpose |
|------|---------|
| `pyqcu.h` | C API header — 22 `extern "C"` functions taking raw pointers as `long long` |

This header must stay in exact sync with `pyqcu/cuda/qcu/qcu.pxd` (the Cython declaration file). Any mismatch causes silent memory corruption.

All functions take three parameter arrays:
- `set_ptrs` (int64[100]): scratch buffer pointers managed by C++ runtime
- `params` (int32[54]): lattice dims, grid sizes, data types, iteration counts, plan selection
- `argv` (float64[7]): mass, atol, sigma, MG tolerances

C++→Python data pointers are cast to `long long` from `tensor.contiguous().data_ptr()`.
