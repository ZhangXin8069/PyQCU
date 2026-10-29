# AGENTS.md — pyqcu.torch.tools

Torch/CANN 专用数值工具。

- `_linalg.py`：`norm` / `vdot`。
- `_multigrid.py`：null vectors、restrict/prolong、33-tensor stencil。
- `_strict_galerkin.py`、`_distributed_setup.py`、`_mpi_roll.py`：strict Torch hierarchy 的分布式 setup。
- `_einsum.py`、`_matul.py`：可选 TileLang 内核；缺依赖时 CPU 的 einsum 回退
  `torch.einsum`，TileLang-only program factory 抛 `TileLangUnavailable`。
- `_hierarchical.py`：VRAM/RAM/DISK 分层存储。

这些模块只服务 Torch 实现。公共 MPI/HDF5/env/budget 工具留在 `pyqcu.tools`。
