# AGENTS.md — cpp/kupl/qcu/tests

KUPL 独立 C++/Python ABI、canonical 布局、失败路径和 MultiGrid 的 CMake 测试。

- `test_kupl.cpp`：Wilson/Clover、求解器、halo 与精度回归。
- `test_new_abi.py`、`test_canonical_layout.py`：Python/ctypes 差分、ABI 与公开张量布局。
- `test_multigrid.py`：restrict/prolong、粗算子与 MultiGrid 原语。
- `test_failure_modes.py`：非收敛、空指针和不支持模式必须显式失败。
- 从 `cpp/kupl/qcu` 运行 `ctest --test-dir build --output-on-failure`。这些测试属于
  独立 CMake 工程，不迁入 `pyqcu/testing/`；涉及 MPI 的用例应按 CTest 注册的 rank 数执行。
- 不削弱 fail-closed 断言，不用单 rank 结果替代 MPI/OpenMP 回归，也不伪造缺失库时的 PASS。
