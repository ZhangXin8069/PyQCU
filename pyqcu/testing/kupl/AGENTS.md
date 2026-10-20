# pyqcu/testing/kupl

KUPL CPU 后端的 Python 黑盒回归。

- `test_kupl_python.py` 经真实 `ctypes` 库校验 Clover、dslash、
  BiStabCG、Gaussian gauge、槽位生命周期和 canonical PyQCU ABI。
- 缺少 `cpp/kupl/qcu/build/libqcu.so` 时测试明确 skip，不伪造通过。
- C++ 1/2/4-rank 与 sanitizer 回归属于 `cpp/kupl/qcu/tests/`。
