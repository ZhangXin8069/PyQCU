# AGENTS.md — pyqcu.torch

Torch/CANN 执行后端。

- `dslash/`：Wilson/Clover 算子实现。
- `smear/`：stout 与 Wuppertal smearing。
- `solver/`：BiCGStab、GMRES、MR、CA-CG、Lanczos 与 strict/legacy multigrid 实现。
- `tools/`：仅服务 Torch 实现的 eigensum、批量 multigrid、strict Galerkin、分层存储等工具。

公共物理与布局定义仍在 `pyqcu.lattice`；公共 MPI/HDF5/env/budget 工具仍在
`pyqcu.tools`；三后端 `Multigrid` 选择器仍在 `pyqcu.solver.multigrid`。
新 Torch 实现代码必须放在本目录，顶层同名包只作为兼容门面。
