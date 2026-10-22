# pyqcu/testing/kupl

KUPL CPU 后端的 Python 黑盒回归。

- `test_kupl_python.py` 经真实 `ctypes` 库校验 Clover、dslash、
  BiStabCG、Gaussian gauge、槽位生命周期和 canonical PyQCU ABI。
- `test_krylov_api.py` 经真实 KUPL 库验证统一 `Krylov` facade 的 Wilson
  BiCGStab/normal-CG 整次 solve 与独立 full residual。
- `test_strict.py` 覆盖 strict coarse/MATPC/R-P、持久 V-cycle/FGMRES；
  `test_legacy_mg.py` 覆盖递归 Clover-MG、deflate 开关、CUDA-only 模式拒绝与
  verify；`check_abi.py` 同时比对 KUPL/CUDA C 声明和动态导出。
- 缺少 `cpp/kupl/qcu/build/libqcu.so` 时测试明确 skip，不伪造通过。
- C++ 1/2/4-rank 与 sanitizer 回归属于 `cpp/kupl/qcu/tests/`。
