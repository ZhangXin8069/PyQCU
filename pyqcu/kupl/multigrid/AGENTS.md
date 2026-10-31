# AGENTS.md — pyqcu.kupl.multigrid

KUPL Legacy/Strict MultiGrid 的 Python 生命周期包装层；具体数值实现位于
`cpp/kupl/qcu/src/lattice_clover_multigrid.cpp` 与 `lattice_strict.cpp`。

- Legacy 是默认路径，Strict 仅在调用方显式选择时启用；两者保持统一的
  `setup()/solve()/close()` 契约。
- 所有控制/数据张量必须是 CPU contiguous 张量；c64/c128 精度必须与 gauge 一致。
- Legacy cache 布局通过 `set_ptrs[99]` 显式选择；Strict hierarchy 资产仍由
  `_SET_PTRS_COARSE_BASE_` 之后的协议槽绑定，不能当普通 scratch 覆盖。
- 保留失败清理与独立真残差检查，不把未收敛或非有限结果返回给调用方。
- 修改包装契约后运行 `python -m pytest -q pyqcu/testing/python/test_multigrid_api.py`；
  真实 KUPL 回归按父规则选择 `pyqcu/testing/kupl` 中受影响的测试。
