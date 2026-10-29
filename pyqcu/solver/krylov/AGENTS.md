# AGENTS.md — pyqcu.solver.krylov

后端无关 Krylov 选择器。

- `Krylov` 是兼容入口，`Solver` 是统一高层名称；二者必须返回同一
  `SolveResult` 语义。
- CUDA/KUPL 只接受后端绑定的 compact-Schur `LinearOperator`，每次 `apply()`
  调用一次原生整次 solve，禁止在 Python 中重写其迭代循环。
- `bistabcg` 是 `bicgstab` 的兼容别名；公共结果中的 `method` 统一为
  `bicgstab`。
- CUDA/KUPL CG 只接受 `CGOperator.NORMAL`，不得把普通 Wilson 或默认 HPD
  算子隐式转成正规方程。
- 修改选择器或返回对象后运行
  `python -m pytest -q pyqcu/testing/python/test_krylov_api.py`。
