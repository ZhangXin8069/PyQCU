# AGENTS.md — pyqcu.solver

公共 solver API 与三后端 multigrid 选择器。

- Torch 数值实现位于 `pyqcu/torch/solver/`。
- 本目录 `__init__.py` 只提供稳定公开门面，兼容旧的 `pyqcu.solver` 导入。
- `multigrid/` 只负责按 `variant` 与 `backend` 选择 strict/legacy 层级；具体实现不得放回本目录。
- `Multigrid` 默认 `variant="legacy"`；Legacy-MultiGrid 是默认开发/调试/优化对象，Strict 仅显式启用。
- `krylov/` 只负责按 `backend` 与 `method` 选择 Torch 迭代或 CUDA/KUPL 原生整次 solve。
- `_native_qcu.py` 是 CUDA/KUPL 共享的 compact parity-Schur `LinearOperator` 适配器。
