# PyQCU 统一 API 契约（P0/P1）

状态：2026-10-23 已实现 P0 数据模型与结构协议，以及 P1 的
`LinearOperator`、Torch BiCGStab/CG 和 CUDA/KUPL 原生整次 solve tracer。
尚未实现 `pyqcu.api.create_backend()` 的统一 backend factory。

## 目标与边界

P0/P1 冻结统一控制面，而不把三个后端改写成同一套 kernel。公共层负责：

- 校验几何、容差、cycle、coarsening 和 correction 配置；
- 区分 full、MATPC、Schur、normal 与预条件残差；
- 统一 solve 状态、计数器、workspace、cache provenance 和诊断；
- 为 LinearOperator、TransferOperator、Smoother、Krylov 和 MG 定义 Protocol。

CUDA stream、KUPL OpenMP/MPI、Torch eager 张量、halo、cache payload 和持久 workspace
仍由各后端拥有。

## 公共导入

```python
from pyqcu.api import (
    AssetManifest,
    BackendCapabilities,
    BiCGStabControl,
    CGControl,
    CGOperator,
    CallableLinearOperator,
    CoarsenConfig,
    CycleConfig,
    FGMRESControl,
    LatticeGeometry,
    LegacyMultigridConfig,
    MRConfig,
    NativeSolve,
    NormalOperator,
    PeriodicCorrectionConfig,
    ResidualScope,
    SolveResult,
    SolveStatus,
    StrictMultigridConfig,
)
from pyqcu.solver import Krylov
```

`pyqcu.api` 不包含算法实现。现有 mg 兼容入口仍为：

```python
from pyqcu.solver.multigrid import Multigrid

mg = Multigrid(backend="cuda", **legacy_options)  # 默认 Legacy-MultiGrid
mg.setup()
raw = mg.solve(rhs, x0=None)             # 原有返回：后端数组
result = mg.solve_result(rhs, x0=None)  # 新入口：SolveResult

# Strict-MultiGrid 仅用于显式对照、算法参考或特殊备选
mg_strict = Multigrid(variant="strict", backend="cuda", **strict_options)
```

`Legacy-MultiGrid` 是 QCU 默认 multigrid 求解器，也是日常开发、调试与优化对象；
`Strict-MultiGrid` 只有显式传入 `variant="strict"` 或直接构造 `StrictMultigrid` 时启用。

`solve_result()` 保留 `result.x` 为原有后端数组；未知的迭代、残差或资源字段保持 `None`，
不得根据配置或另一口径的残差补齐。

P1 的统一 Krylov 入口为：

```python
krylov = Krylov(
    backend="torch" | "cuda" | "kupl",
    method="bicgstab" | "bistabcg" | "cg",
    operator=operator,
    control=control,
)
result = krylov.solve_result(rhs, x0=None)
```

`bistabcg` 是 `bicgstab` 的兼容别名。Torch driver 在 Python 中执行迭代；CUDA/KUPL
driver 必须只调用一次原生整次 solve，禁止通过 `operator.apply()` 在 Python 中构造 Krylov 循环。
新 driver 在 `x0=None` 时使用零初值；旧 `pyqcu.solver.bistabcg()` 的随机初值兼容行为不变。

## 几何

`LatticeGeometry` 保存：

- `local_shape`：本 rank 的四维完整局部格点尺寸；
- `process_grid`：四维 MPI rank 网格；
- `parity`：`None` 表示 full view，`0/1` 表示单一 parity view；
- `dtype` 与 `layout`。

`volume` 是完整局部格点体积，`local_sites` 在 parity view 下为体积的一半。parity geometry
要求局部时间维为偶数。

## 求解结果

`SolveResult` 至少包含：

```text
x, backend, variant, method, status, converged
krylov_iterations, restart_count
correction_count, proposed_correction_count, rejected_correction_count
cycle_count, coarse_iterations
final_residual, residual_scope
matpc_true_residual, full_true_residual, relative_true_residual
setup_seconds, solve_seconds, workspace_bytes, cache_status, diagnostics
```

关键口径：

- `residual_scope=matpc` 时，`matpc_true_residual` 可等于 `final_residual`，但
  `full_true_residual` 在未独立计算时必须为 `None`。
- `restart_count` 记录实际 Krylov restart 次数；Krylov restart 长度不进入该字段。
- `correction_count` 只统计被接受并改变迭代状态的 Legacy 周期校正；proposed/rejected 分列。
- `cycle_count` 只统计实际递归 cycle，不根据配置推断。
- `status=converged` 与 `converged=True` 必须一致；未知状态不得写成成功。
- 任何 non-null 残差都必须带 `residual_scope`；后端未声明口径时，兼容层保留 `None`。

## 配置

`SolverControl` 统一 `rtol`、`atol`、`max_iter`、`min_iter`、真残差刷新周期和目标残差口径。
`BiCGStabControl`、`FGMRESControl`、`CGControl` 只增加各自特有字段。

CG 必须通过 `CGOperator` 明确声明：

- `HPD`：算子本身 Hermitian positive definite；
- `NORMAL`：显式正规算子或 `A†A`；
- `SCHUR`：已证明 HPD 的 Schur 子问题。

`CoarsenConfig` 统一 `nvec/block_size/levels`。Strict 与 Legacy 分别组合为：

```text
StrictMultigridConfig = coarsen + MR + V/W/F/K cycle + right-FGMRES
LegacyMultigridConfig = coarsen + periodic correction + outer Krylov
```

两种资产和 correction 语义仍不互转。

## 能力与资产

`BackendCapabilities.require(name)` 采用 fail-closed：未声明能力不会被推断为支持。
能力包括 full/compact coarse、parity、MATPC、persistent hierarchy/workspace、overlap、
混合精度、CUDA Graph、NVSHMEM、native/python Krylov、HPD/normal/Schur CG 和 cache schema。

`AssetManifest` 记录类型化资产的 backend、variant、precision、geometry、block、level、
算法版本、逻辑字节、逐 tensor SHA256 和 JSON metadata。`identity` 对 canonical JSON 计算
SHA256，metadata 键顺序不影响结果。

Legacy `Legacy33Stencil` 与 Strict `StrictCoarseLinks(X/Y/Yhat)` 应使用不同 `kind`；
manifest 只共享 provenance 协议，不表示 payload 可互载。

## 协议

`pyqcu.api` 提供 runtime-checkable Protocol：

- `LinearOperator`：geometry、capabilities、apply/adjoint/diagonal/normal/MATPC；
- `TransferOperator`：build/restrict/prolong/restrict_adjoint；
- `CoarseOperator`：粗层 LinearOperator；
- `Smoother`：固定步数 step；
- `KrylovDriver`：control 与 solve_result；
- `MultigridHierarchy`：backend/variant/capabilities/setup/solve_result/close；
- `PyQCUBackend`：dslash/solver/multigrid 的长期 canonical 工厂协议。

`CallableLinearOperator` 用显式 callable 适配已有算子；`NormalOperator(A)` 显式构造
`A†A`，标记 `hpd_cg/normal_cg=True`。普通非 Hermitian Wilson 算子不得直接作为 HPD CG。

Torch `TorchLinearOperator` 提供 full Wilson/Clover apply 与 `γ5`-Hermitian adjoint。
CUDA/KUPL `CudaLinearOperator`/`KuplLinearOperator` 表示 compact parity-Schur：

- `apply` 对应 Wilson/Clover Schur dslash；
- `apply_normal` 仅 Wilson 提供 `S†S` dslash；
- adjoint、diagonal、normalized MATPC 未暴露 ABI，必须抛 `NotImplementedError`；
- BiCGStab 可接受 Wilson 或 Clover Schur；CG 仅接受 Wilson normal Schur 且
  `CGControl(operator_kind=CGOperator.NORMAL)`。

Protocol 只约束调用语义。CUDA/KUPL 生产热循环不得为了满足 Protocol 而逐迭代穿过 Python。

## 当前兼容映射

`Multigrid.solve_result()` 只映射后端已暴露的字段：

- CUDA/KUPL/Torch Strict 的最终残差标记为 `matpc`；
- CUDA Strict 映射 outer iterations、converged 和 fused workspace；
- KUPL Strict 映射 outer iterations、converged、MATPC residual 和 allocated workspace；
- 当前 Legacy 未稳定暴露的 correction/cycle/full residual 保持 `None`。

原生 CUDA/KUPL Krylov driver 的约束：

- `atol != 0`、`min_iter != 0`、周期真残差刷新和 BiCGStab breakdown restart 暂不支持；
- 控制目标残差口径必须是 final full residual，原生 C++ 在返回前负责最终 full residual 门；
- 正常返回映射为 converged；ABI 不返回数值残差，因此 `final_residual` 保持 `None`；
- `diagnostics["tolerance_semantics"]` 记录后端实际容差公式；KUPL 当前为
  `max(atol,residual_floor)*max(rhs_norm,1)`，不得简写成精确的 `rtol*rhs_norm`；
- KUPL 可读取通用 native iteration getter；CUDA Wilson 当前无 getter，iteration 保持 `None`；
- Clover BiCGStab 在 CUDA/KUPL 均可读取 Clover iteration getter；Clover CG 不存在。

这不是最终 backend adapter。后续应让每个 MG 后端显式返回 `SolveResult`，逐步移除属性探测。

## 验证

```bash
python -m pytest -q pyqcu/testing/python/test_api_contract.py
python -m pytest -q pyqcu/testing/python/test_krylov_api.py
python -m pytest -q pyqcu/testing/kupl/test_krylov_api.py
python -m pytest -q pyqcu/testing/python/test_multigrid_api.py \
  pyqcu/testing/python/test_multigrid.py
```

验收要求：配置非法值 fail-closed；manifest identity 可复现；`solve()` 返回类型不变；
`solve_result()` 不伪造未知字段；Torch Krylov 与真实 KUPL 原生 solve 回归通过；
CUDA/KUPL driver 不得逐迭代穿过 Python。
