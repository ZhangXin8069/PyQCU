# AGENTS.md — pyqcu.solver.multigrid

Legacy/Strict MultiGrid 与统一 cache 的选择器层。

- `Multigrid` 默认 `variant="legacy"`；Strict 必须显式选择。
- `MultigridCache` 根据 `variant` 选择 canonical Legacy 33-tensor schema 或
  Strict v2 `V/Yhat/(X,X^-1)` schema，两者不可互载。
- cache 命中前必须验证 identity、geometry、dtype、manifest 与逐 tensor digest；
  命中后跳过 null-space/Galerkin 重建，并单独记录验证/加载时间。
- 本目录只做选择和稳定门面，具体求解器留在 `pyqcu/torch/solver/`，后端绑定留在
  `pyqcu/cuda/` 与 `pyqcu/kupl/`。
- 修改后运行 `python -m pytest -q pyqcu/testing/python/test_multigrid_api.py
  pyqcu/testing/python/test_mg_cache_reuse.py`。
