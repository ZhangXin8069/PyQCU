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

## 三后端统一接口（强制规范）

CUDA、KUPL、Torch 的同一 QCU 功能必须使用同一套公共 PyQCU 接口、参数名、返回类型和
生命周期；后端名称只作为 `backend="cuda"|"kupl"|"torch"` 的分派值。相同功能出现三套
不同入口、不同方法名或不同结果形状时，应先收敛到公共门面，再由薄适配器调用后端。

公共入口按职责固定为：

| 功能 | 唯一高层入口 | 统一结果/控制对象 |
|---|---|---|
| Krylov/BiCGStab/CG | `pyqcu.solver.Solver`（兼容名 `Krylov`） | `BiCGStabControl`、`CGControl`、`SolveResult` |
| Legacy/Strict MultiGrid | `pyqcu.solver.Multigrid(variant=..., backend=...)` | `SolveResult`、`BackendCapabilities` |
| Legacy/Strict MG cache | `pyqcu.solver.MultigridCache(variant=..., backend=...)` | `MultigridCacheLoadResult` |
| Operator | `pyqcu.api.LinearOperator` Protocol | `LatticeGeometry`、`ResidualScope`、capability mapping |

统一接口的最低要求：

1. 同一功能必须提供相同的方法名和调用顺序；求解器统一为 `solve()`、
   `solve_result()`、`capabilities`，层级统一为 `setup()`、`solve()`、`close()`，cache 统一为
   `path()`、`validate()`、`load()`、`save()`。
2. `bistabcg` 是 `bicgstab` 的兼容输入别名；公共结果中的 `method` 统一规范化为
   `"bicgstab"`，不得为不同后端返回不同拼写。
3. 公共高层函数只接受 backend-neutral 控制对象；后端私有整数数组、ctypes 指针、
   `set_ptrs` 槽位和厂商 stream 只存在于后端适配层。一个 Python 进程、一个 MPI rank、
   一张设备的约束不因统一接口而改变。
4. `solve_result()` 必须返回 `SolveResult`；未暴露的迭代数、耗时、workspace 或真残差保持
   `None`，不得由配置推断或复制其他后端字段。所有后端必须明确 `ResidualScope`。
5. 后端不支持某能力时由 `BackendCapabilities.require()` 或构造阶段 fail-closed，
   不得通过改名参数、静默降精度、切换算法或改变返回类型规避契约。
6. 高层字段/默认值只能定义一个权威实现。迁移完成后删除后端包中重复的选择器、转换器和
   兼容包装；保留的旧名只能是无算法逻辑的兼容别名，并指向唯一实现。

新增或修改三后端功能时，验收必须同时覆盖 `torch × cuda × kupl` 的签名一致性、控制对象
类型、返回对象、能力门禁和错误语义。至少运行
`pyqcu/testing/python/test_api_contract.py`、`test_krylov_api.py`、
`test_multigrid_api.py` 与对应后端回归；缺少跨后端契约测试时不得只凭单后端通过宣称接口统一。

## 资源复用

该 QCU 版本内存在 identity 可验证的 MG cache、已构建后端库、已准备输入或可增量续用的构建树时，
直接复用，不重复 setup、编译或生成物理输入。复用的前提是共享 identity 完全匹配并通过内容校验：
gauge/null-vector 物理摘要、lattice、dtype、mass/csw、levels、block/coarsening、dof、
process grid/rank shard、算法版本与 tensor digest 均须一致。

- Legacy cache 命中后跳过 null-space 与 33-tensor Galerkin 构建，只执行验证/加载和真实 solve。
- Strict cache 命中后跳过 transfer、`X/Y/Yhat` 构造，直接绑定 `fine V + 每层 Yhat/(X,X^-1)`。
- 只有资源缺失、identity 变化、摘要损坏或显式 `rebuild/miss` 请求时才重建。
- 报告分别记录 `cache_hit`、验证时间、加载时间、setup 时间和 solve 时间；cache 命中不是零成本，
  也不得把 setup 节省混入 solve speedup。

2026-10-29 起，Legacy 33-tensor cache 的跨后端复用已有真实数值证据：同一 gauge
的 CUDA cache 可由 KUPL 直接加载，`hnn/hdg/sit` 相对误差均低于 `3e-7`，1/2/4-rank
solve 的独立 true residual 均约 `2.02e-7`。cache miss 使用
`pyqcu.testing.simple.mpi.build_full_mg_cache` 的 native MPI builder；rank 0-only
构建不属于多 rank 证据。

## 兼容层

`Multigrid.solve()` 的返回类型保持不变；`solve_result()` 返回 `SolveResult`。兼容适配器只读取
后端已有属性，不拥有第二套停机、计数或残差逻辑。完整原生结果必须由后端显式提供后，
再移除属性探测。

`Multigrid` 默认 `variant="legacy"`。Legacy-MultiGrid 是 QCU 默认开发、调试和优化对象；
Strict-MultiGrid 只有显式 `variant="strict"` 或直接构造 `StrictMultigrid` 时才启用。

统一 Krylov 入口为 `Solver(backend=..., method=..., operator=..., control=...)`
（兼容名 `Krylov`）。Torch
可以在 Python 中迭代；CUDA/KUPL 必须调用一次原生 C++ solve，不能把 `operator.apply()`
放进 Python Krylov 循环。CUDA/KUPL `apply` 是 compact parity-Schur，不是完整 Dirac 算子。

## 验证

```bash
source ./env.sh
python -m pytest -q pyqcu/testing/python/test_api_contract.py
python -m pytest -q pyqcu/testing/python/test_krylov_api.py
python -m pytest -q pyqcu/testing/python/test_mg_cache_reuse.py
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
- 为 CUDA/KUPL/Torch 分别新增同名但参数、返回类型或生命周期不同的高层 API。
- 统一门面已经存在后继续在 backend package 中维护第二套选择器或转换实现。
- 已有兼容 cache/构建产物且 identity 校验通过时仍重复 setup 或全量重建。
