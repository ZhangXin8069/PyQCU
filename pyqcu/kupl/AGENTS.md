# pyqcu.kupl

独立 KUPL CPU 后端的 Python `ctypes` 绑定。

- `qcu.py` 通过 `RTLD_LOCAL` 加载 `PYQCU_KUPL_LIB` 或
  `cpp/kupl/qcu/build/libqcu.so`，避免与 CUDA `libqcu.so` 符号冲突；
  37 个 CUDA QCU ABI 名称均有 Python 包装和 CPU 实现。
- `multigrid/` 提供统一命名的 `LegacyMultigrid` 与 `StrictMultigrid`，共同使用
  `setup()/solve()/close()`；粗层资产仍通过 `set_ptrs` 按各自协议预绑定。
- `define.py` 复用 canonical `params/argv/set_ptrs` 常量；`argv[7]` 为 `csw`。
- 新代码优先配合 `pyqcu.cuda.QcuRuntime` 管理 init/end 和空闲槽位。
- 后端能力与限制见 `../../cpp/kupl/qcu/README.md` 和
  `../../skills/pyqcu-kupl/SKILL.md`。
