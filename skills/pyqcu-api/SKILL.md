---
name: pyqcu-api
description: 当任务涉及 pyqcu.api、后端无关配置/结果/能力/资产 manifest、公共 Protocol 或 Multigrid.solve_result 兼容层时使用。
---
# pyqcu-api

后端无关的 QCU 控制面。`pyqcu.api` 只定义经过校验的数据模型、配置、结果、
能力声明、资产 provenance 和 Protocol；不包含 CUDA/KUPL/Torch kernel、cache 实体
或第四套算法实现。

## 执行前置

先读取 `pyqcu/api/AGENTS.md` 和 [`docs/api_contract.md`](../../docs/api_contract.md)。
涉及具体后端适配时，再读取对应 `pyqcu-cuda`、`pyqcu-kupl` 或 `pyqcu-solver` 技能。

## 文件

| 文件 | 职责 |
|---|---|
| `pyqcu/api/models.py` | `LatticeGeometry`、`SolveResult`、`BackendCapabilities`、`AssetManifest`、残差/状态枚举 |
| `pyqcu/api/config.py` | Krylov、CG、coarsening、MR、cycle、periodic correction 与 Legacy/Strict 配置 |
| `pyqcu/api/protocols.py` | `LinearOperator`、`TransferOperator`、`CoarseOperator`、`Smoother`、`KrylovDriver`、`MultigridHierarchy`、`PyQCUBackend` |
| `pyqcu/api/operator.py` | `CallableLinearOperator` 与显式 `A†A` `NormalOperator` |
| `pyqcu/api/native.py` | 原生整次 solve 的 `NativeSolve` 与 `WholeSolveKrylovDriver` |
| `pyqcu/api/_compat.py` | 仅把现有后端已暴露字段映射到 `SolveResult`，未知字段保持 `None` |
| `pyqcu/api/__init__.py` | 公共导入面；不得从此处导出后端实现 |
| `pyqcu/testing/python/test_api_contract.py` | 数据模型、manifest identity、Protocol 与兼容映射回归 |

## 当前实现边界

已经实现：

- `LatticeGeometry` 与 full/parity 几何校验。
- `SolverControl`、`BiCGStabControl`、`FGMRESControl`、`CGControl`。
- `CoarsenConfig`、`MRConfig`、`CycleConfig`、Legacy/Strict MG 组合配置。
- `SolveResult`、`SolveStatus`、`ResidualScope`。
- `BackendCapabilities.require()` 的 fail-closed 能力检查。
- `AssetManifest` 的规范化 JSON SHA256 identity。
- runtime-checkable Operator/Krylov/MG/backend Protocol。
- `Multigrid.solve_result()` 对现有后端字段的保守兼容映射。
- `CallableLinearOperator`/`NormalOperator` 与统一 `Krylov` 选择器。
- Torch BiCGStab/CG driver；CUDA/KUPL compact-Schur Operator 与原生整次 solve driver。

尚未实现：

- `pyqcu.api.create_backend()`。
- 各 MG 后端显式返回原生 `SolveResult` 的完整适配器。
- 现有 Legacy 路径未暴露的 correction/cycle/full-residual 字段。
- CUDA/KUPL 的 compact-Schur adjoint、diagonal、normalized MATPC 与 Clover CG ABI。

## 结果口径

- `status=converged` 与 `converged=True` 必须一致；状态未知时不得写成成功。
- `restart_count` 只记录实际 Krylov restart 次数，restart 长度不是次数。
- `correction_count` 只统计被接受并改变迭代状态的 Legacy correction；
  proposed/rejected 分列。
- `cycle_count` 只统计实际递归 cycle，不能根据配置推断。
- `ResidualScope` 必须标明 `full`、`matpc`、`schur`、`normal`、`preconditioned` 或 `unknown`。
- `NORMAL` 表示显式正规方程 `A†A x=A†b`，不得与 full Wilson 残差混用。
- `residual_scope=matpc` 时，未独立计算的 `full_true_residual` 必须为 `None`；
  不得把 MATPC/Schur 残差复制成 full Wilson/Clover 真残差。
- 未知迭代数、耗时、workspace、cache 或诊断字段保持 `None`，不得按配置猜测。

## 配置与能力

- `SolverControl` 使用 `rtol*norm(rhs)` 与 `atol` 的组合阈值；二者不能同时为 0。
- `FGMRESControl.restart` 必须满足 `1 <= restart <= max_iter`。
- CG 必须显式声明 `CGOperator.HPD`、`NORMAL` 或 `SCHUR`；普通 Wilson 算子不得直接冒充 HPD CG。
- `CoarsenConfig` 至少两级，`block_size` 必须是四个正整数。
- `BackendCapabilities.require(name)` 对未声明能力 fail-closed；不得在运行时静默切换路线。
- `AssetManifest.identity` 对 canonical JSON 计算 SHA256，metadata 键顺序不影响结果；
  Legacy 33-stencil 与 Strict `X/Y/Yhat` 必须使用不同 `kind`，manifest 可验证不等于 payload 可互载。

## Protocol 边界

Protocol 只约束调用语义。CUDA/KUPL 生产热循环仍由 C++/CUDA/OpenMP 实现，不得为了满足
Protocol 在每次迭代穿过 Python。后端拥有 stream、halo、MPI communicator、cache payload
和持久 workspace 的生命周期。

## 兼容层

`Multigrid.solve()` 的返回类型保持不变；`solve_result()` 返回 `SolveResult`。兼容适配器只读取
后端已有属性，不拥有第二套停机、计数或残差逻辑。完整原生结果必须由后端显式提供后，
再移除属性探测。

`Multigrid` 默认 `variant="legacy"`。Legacy-MultiGrid 是 QCU 默认开发、调试和优化对象；
Strict-MultiGrid 只有显式 `variant="strict"` 或直接构造 `StrictMultigrid` 时才启用。

统一 Krylov 入口为 `Krylov(backend=..., method=..., operator=..., control=...)`。Torch
可以在 Python 中迭代；CUDA/KUPL 必须调用一次原生 C++ solve，不能把 `operator.apply()`
放进 Python Krylov 循环。CUDA/KUPL `apply` 是 compact parity-Schur，不是完整 Dirac 算子。

## 验证

```bash
source ./env.sh
python -m pytest -q pyqcu/testing/python/test_api_contract.py
python -m pytest -q pyqcu/testing/python/test_krylov_api.py
python -m pytest -q pyqcu/testing/kupl/test_krylov_api.py
python -m pytest -q pyqcu/testing/python/test_multigrid_api.py \
  pyqcu/testing/python/test_multigrid.py
```

验收要求：非法配置 fail-closed；manifest identity 可复现；`solve()` 返回类型不变；
`solve_result()` 不伪造未知字段；三后端 MG 选择器回归通过。

## 反模式

- 在 `pyqcu.api` 中 import Torch/CuPy/Cython 后端或定义 kernel。
- 用 `restart=20` 推导 20 次 cycle，或把配置值写成实际计数。
- 把 MATPC/Schur residual 标成 full true residual。
- 为缺失字段填 0、配置值或另一后端的估计值。
- 在兼容适配器中复制后端求解循环或改变停机语义。
