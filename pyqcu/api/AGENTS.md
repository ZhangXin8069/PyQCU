# AGENTS.md — pyqcu.api

公共 QCU 控制面与数据契约。此目录只能定义后端无关的配置、结果、能力、资产清单和
Protocol，不得放入 CUDA、KUPL、Torch kernel、缓存实体或第四套算法实现。

- `models.py`：几何、残差口径、求解结果、能力和资产 provenance。
- `config.py`：Krylov、cycle、coarsening 与 Legacy/Strict MG 配置。
- `protocols.py`：Operator、Transfer、Smoother、Krylov、MG 与 backend 契约。
- `operator.py`：`CallableLinearOperator` 与显式 `A†A` `NormalOperator`。
- `native.py`：原生整次 solve 的 `NativeSolve` 与 `WholeSolveKrylovDriver`。
- `_compat.py`：仅把现有后端已有字段映射到 `SolveResult`；未知字段必须保持 `None`。

新增字段必须保持向后兼容，并明确区分 `full`、`matpc`、`schur` 残差；不得把 restart 当作
cycle/correction 计数，也不得伪造 backend 未暴露的性能或资源指标。
