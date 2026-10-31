# AGENTS.md — pyqcu.torch.solver.multigrid

Torch 后端具体 MultiGrid 层级的公共命名空间；三后端选择器位于
`pyqcu/solver/multigrid/`，不要把选择逻辑复制到这里。

- `legacy_multigrid.py` 暴露 33-stencil Legacy 实现；`strict_multigrid.py` 暴露
  Strict transfer、coarse/MATPC 与层级算子。
- 默认路线是 Legacy；Strict 必须显式构造。两个实现都保持
  `setup()/solve()/close()` 生命周期。
- 分布式调用必须传入明确的 `process_grid`/`comm` 并走现有 halo 上下文；
  缺少分布式上下文时 fail closed，不能退化为各 rank 独立周期边界。
- 新 torch 操作统一经 `pyqcu.cann as _torch`。受影响的最小回归为
  `python -m pytest -q pyqcu/testing/python/test_multigrid_api.py` 及相关小格代数测试。
