# AGENTS.md — pyqcu.cuda.multigrid

CUDA Legacy/Strict MultiGrid 的 canonical Python 命名空间。

- `legacy_multigrid.py` 包装 Clover-MG C ABI，是默认开发与调试路径；
  `strict_multigrid.py` 只用于显式 Strict 对照或特殊备选。
- 保持 `setup() -> solve() -> close()` 与上下文管理语义。Legacy 的普通
  `applyInitQcu/applyEndQcu` 生命周期由包装层拥有；`_SET_INDEX_` 只能在当前对象 end 后推进。
- `params/argv/set_ptrs` 保持 CPU 张量，数值资产保持同一 CUDA 设备且 contiguous；
  warm start 使用 `_MG_USE_INIT_GUESS_` 与调用方 `x0`。
- 新代码通过 `pyqcu.cann as _torch` 访问 torch 能力，不直接 `import torch`。
- 修改统一构造/生命周期契约后运行
  `python -m pytest -q pyqcu/testing/python/test_multigrid_api.py`，数值路径按受影响的
  CUDA 显式测试入口另行选择。
