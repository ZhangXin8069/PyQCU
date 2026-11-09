# 求解器

PyQCU 有多个求解器层级。它们共享同一套格点约定，但职责不同：Torch 求解器在
Python 中迭代；CUDA/KUPL 的 Krylov 驱动只调用一次原生整次求解，不逐迭代穿过
Python。多重网格额外拥有层级、粗算子和可选 HDF5 资产。选择满足任务的最小层。

## 如何选择

| 需求 | 推荐入口 | 它负责什么 |
|---|---|---|
| 小规模求解、原型验证 | `pyqcu.torch.solver.bistabcg` | 显式 `matvec`，Python 迭代 |
| 生产 Clover、Torch | `pyqcu.solver.Multigrid(backend="torch", variant="legacy", ...)` | null 向量、33-tensor 粗算子、V-cycle |
| CUDA/KUPL 整次求解 | `pyqcu.solver.Krylov` + `CudaLinearOperator`/`KuplLinearOperator` | 一次原生 BiCGStab/CG 调用 |
| 原生多重网格 | `pyqcu.solver.Multigrid(backend="cuda"/"kupl", ...)` | 围绕已构建粗层级的生命周期 |
| Strict 对照 | `Multigrid(variant="strict", ...)` 或 `StrictMultigrid` | 显式 Strict 层级，不是默认 |
| 最低本征值 | `pyqcu.torch.solver.tr_lanczos` | Hermitian Lanczos |

Legacy-MultiGrid 是默认求解器，也是主要开发对象。Strict-MultiGrid 只在显式
`variant="strict"` 或直接构造 `StrictMultigrid` 时启用，不得当作隐式默认。

## 共同约定

* `kappa = 1 / (2 * mass + 8)`。Clover 系数 `csw` 单独存放；原生 ABI 中为
  `argv[7]`。
* 完整费米子场为 `[4, 3, X, Y, Z, T]`，规范场为
  `[3, 3, 4, X, Y, Z, T]`。奇偶分裂场压缩的是 **时间** 轴：
  `[2, 4, 3, X, Y, Z, T/2]`。
* `ResidualScope` 说明残差口径：`FULL`、`SCHUR`、`NORMAL` 或 `MATPC`。
  `SCHUR` 残差不是全格点残差。
* 对 complex64，发布驱动使用的验收线是
  `max(atol, 2048 * eps32) = 2.4414e-4`。把 `atol` 调得更小不会降低单精度地板。

## Torch 迭代求解器

| 函数 | 关键签名 | 用途 |
|---|---|---|
| `bistabcg` | `(b, matvec, tol=1e-6, max_iter=1000, x0=None, if_rtol=False, verbose=True, history=None)` | 一般非 Hermitian Wilson/Clover |
| `bistabcg_history` | `(b, matvec, tol=1e-6, max_iter=2000, if_rtol=False)` | 只返回残差历史 |
| `cacg` | `(b, matvec, tol=1e-6, max_iter=1000, n_krylov=8, ...)` | Hermitian 正定算子的 s-step CG |
| `fgmres` | `(b, matvec, tol=1e-6, max_iter=1000, restart=30, precond=None, ...)` | 灵活右预条件 GMRES |
| `mr` | `(b, matvec, tol=1e-6, max_iter=1000, matvec_dag=None, omega=1.0, ...)` | 极小残差平滑 |
| `multishift_cg` | `(b, matvec, shifts, tol=1e-6, max_iter=1000, ...)` | 一次求多个质量，HPD 算子 |
| `tr_lanczos` | `(matvec, v0, ncv=32, k=6, tol=1e-8, ...)` | 最低 Hermitian 本征对 |

一个最小的非 Hermitian 求解：

```python
import torch
from pyqcu import lattice
from pyqcu.torch import dslash, solver

lat = (4, 4, 4, 4)
U = torch.zeros((3, 3, 4, *lat), dtype=torch.complex64)
lattice.generate_gauge_field(U, seed=7, sigma=0.1, verbose=False)
kappa = torch.tensor([0.125])
clover = dslash.make_clover(U=U, kappa=kappa)
src = torch.randn((4, 3, *lat), dtype=torch.complex64)
operator = dslash.operator(U=U, clover_term=clover, kappa=kappa, verbose=False)

x = solver.bistabcg(
    b=src, matvec=operator.matvec, tol=1e-5, max_iter=200, verbose=False)
relative = float(
    torch.linalg.vector_norm(operator.matvec(x) - src)
    / torch.linalg.vector_norm(src))
assert relative < 1e-3, relative
```

## Torch Legacy-MultiGrid

推荐使用后端无关的选择器。它在进入时调用 `setup()`、离开时调用 `close()`，
并通过 `.implementation` 暴露具体后端对象。

```python
import torch
from pyqcu import lattice
from pyqcu.solver import Multigrid
from pyqcu.torch import dslash

lat = (4, 4, 4, 4)
U = torch.zeros((3, 3, 4, *lat), dtype=torch.complex64)
lattice.generate_gauge_field(U, seed=42, sigma=0.1, verbose=False)
kappa = torch.tensor([0.125])
clover = dslash.make_clover(U=U, kappa=kappa)
src = torch.randn((4, 3, *lat), dtype=torch.complex64)

with Multigrid(
    variant="legacy", backend="torch",
    dtype_list=[torch.complex64] * 2,
    device_list=[torch.device("cpu")] * 2,
    U=U, clover_term=clover, kappa=kappa,
    max_level=1, min_size=4, mg_grid_size=[2, 2, 2, 2],
    dof_list=[12, 24], tol=1e-5, max_iter=100, num_restart=2,
    verbose=False,
) as mg:
    x = mg.solve(src)

operator = dslash.operator(U=U, clover_term=clover, kappa=kappa, verbose=False)
relative = float(
    torch.linalg.vector_norm(operator.matvec(x) - src)
    / torch.linalg.vector_norm(src))
assert relative < 1e-3, relative
```

构造参数按组理解：

| 组 | 参数 | 含义 |
|---|---|---|
| 几何 | `max_level`, `min_size`, `mg_grid_size`, `dof_list` | 粗化深度、最小允许局部尺寸、块大小、逐层 null 向量数 |
| 求解 | `tol`, `max_iter`, `num_restart`, `coarse_tol`, `coarse_max_iter` | 外层停机容差与粗层求解预算 |
| 放置 | `dtype_list`, `device_list`, `process_grid` | 逐层 dtype/设备与 MPI rank 网格 |
| 资产 | `coarse_assets`, `clover_ee_inv`, `clover_oo_inv` | 复用 33-stencil cache 或 CUDA/KUPL Clover 逆 |
| 行为 | `support_parity`, `adaptive_correction`, `deflate`, `seed`, `verbose` | 奇偶布局、自适应校正、确定性 setup、日志 |

`4^4` 格点配 `max_level=1` 没有粗层转移，只适合验证生命周期，不是多重网格加速
演示。真正的粗化需要更大格点与多层。

### Torch Legacy 的 `solve_result`

`Multigrid.solve_result()` 返回 `SolveResult`，但 Torch Legacy 不会把所有字段
映射进兼容信封。当前代码中 `status` 为 `UNKNOWN`，`converged`、
`final_residual`、`residual_scope`、`krylov_iterations`、`cycle_count` 保持
`None`。不要根据配置推断它们。应使用后端报告或独立残差：

```python
# after a Legacy solve
report = mg.implementation.last_solve_report
history = mg.implementation.convergence_history
assert "solve_seconds" in report
assert len(history) > 0
```

## Strict-MultiGrid

Strict 是对照、算法参考或特殊备选路径，必须按名字启用：

```python
# docs-skip: requires a complete Strict hierarchy and backend assets
from pyqcu.solver import Multigrid

strict = Multigrid(
    variant="strict", backend="torch",
    # Strict constructor shares the Torch hierarchy arguments; see
    # StrictMultigrid and pyqcu.api.StrictMultigridConfig.
    **strict_options,
)
```

`StrictMultigridConfig` 组织 Strict 控制项：

| 对象 | 字段 | 含义 |
|---|---|---|
| `CoarsenConfig` | `nvec`, `block_size`, `levels` | 块粗化；`levels >= 2` |
| `MRConfig` | `steps` | 固定步数 MR 平滑 |
| `CycleConfig` | `kind` (`V`/`W`/`F`/`K`), `pre`, `post` | cycle 形状与平滑次数 |
| `FGMRESControl` | `restart`, `rtol`, `atol`, `max_iter` | 外层右 FGMRES |

Strict 的 setup 比 Legacy 昂贵。仓库生产驱动默认使用 Legacy，Strict 只用于显式
对照。

## 原生 CUDA/KUPL Krylov

原生路径只调用一次 C++ 整次求解；Python 侧只负责构造算子并检查返回证据。
KUPL 示例：

```python
import torch
from pyqcu import lattice, tools
from pyqcu.api import BiCGStabControl, ResidualScope
from pyqcu.cuda import QcuRuntime
from pyqcu.solver import Krylov
from pyqcu.torch import dslash

try:
    from pyqcu.kupl import qcu
    from pyqcu.kupl.linear_operator import KuplLinearOperator
except Exception as error:
    qcu = None
    print("KUPL backend unavailable; skipping native example:", error)

assert qcu is None or hasattr(qcu, "applyCloverBistabCgQcu")
if qcu is not None:
    lat = (4, 4, 4, 4)
    compact = (*lat[:3], lat[3] // 2)
    full_gauge = torch.empty((3, 3, 4, *lat), dtype=torch.complex64)
    lattice.generate_gauge_field(
        full_gauge, sigma=0.1, seed=59, verbose=False)
    gauge = tools.oooxyzt2poooxyzt(full_gauge).contiguous()
    generator = torch.Generator().manual_seed(61)
    source = torch.complex(
        torch.randn((2, 4, 3, *compact), generator=generator),
        torch.randn((2, 4, 3, *compact), generator=generator))
    runtime = QcuRuntime.from_lattice(
        lat, mass=0.17, atol=1e-7, max_iter=200,
        device="cpu", plan=1, grid=(1, 1, 1, 1), qcu_module=qcu)
    operator = KuplLinearOperator(
        runtime=runtime, gauge=gauge, parity=1)
    result = Krylov(
        backend="kupl", method="bicgstab", operator=operator,
        control=BiCGStabControl(rtol=1e-6, max_iter=200),
    ).solve_result(source)
    full_source = tools.poooxyzt2oooxyzt(source)
    full_solution = tools.poooxyzt2oooxyzt(result.x)
    residual = dslash.give_wilson(
        full_solution, full_gauge,
        torch.tensor([runtime.kappa]), with_I=True) - full_source
    relative = float(
        torch.linalg.vector_norm(residual)
        / torch.linalg.vector_norm(full_source))
    assert result.converged is True
    assert result.residual_scope is ResidualScope.SCHUR
    assert 0 < result.krylov_iterations < 200
    assert relative < 2e-4, relative
```

CUDA 侧把 `pyqcu.kupl.qcu` 换成 `pyqcu.cuda.qcu`，`KuplLinearOperator` 换成
`CudaLinearOperator`，`backend="kupl"` 换成 `backend="cuda"`。runtime 的
device 必须是 `cuda:N`，gauge、source 与 Clover 资产也必须在同一设备。

原生 Krylov 限制：

* 原生 ABI 不暴露所有 `SolveResult` 字段，未暴露的保持 `None`；
* CUDA/KUPL 驱动要求算子类型与 Krylov 方法匹配（`CG` 需要 normal/HPD Schur
  算子）；
* 不要在 Python 中包一层迭代循环调用原生整次求解，那会失去整次求解的意义。

## 原生多重网格包装器

`Multigrid(backend="cuda"|"kupl", variant="legacy", ...)` 包装的是粗层
`params/argv/set_ptrs` 资产已经存在的层级。包装器只负责 `setup()` 与
`close()`，不构建 null 向量或 Galerkin 算子。

```python
# docs-skip: requires pre-built coarse 33-stencil assets in params/argv/set_ptrs
from pyqcu.solver import Multigrid

mg = Multigrid(
    variant="legacy", backend="kupl",  # or backend="cuda"
    params=params, argv=argv, set_ptrs=set_ptrs,
    gauge=gauge,
    clover_even=clover_even, clover_odd=clover_odd,
    clover_even_inv=clover_even_inv, clover_odd_inv=clover_odd_inv,
)
mg.setup()
x = mg.solve(rhs)
mg.close()
```

生产多层级运行使用源码仓库参考驱动（不进入 PyPI 包）：

```bash
python pyqcu/testing/simple/mpi/solve_case.py \
  --backend kupl --lat 8 8 16 16 --ranks 4 --grid 1 2 2 1 \
  --levels 2 --restart 5 --omp-threads 4 \
  --gauge data/release_8x8x16x16/gauge_8x8x16x16_c64.h5
```

## 缓存

`LegacyMultigridCache` 与 `StrictMultigridCache` 负责校验与加载 HDF5 资产。
identity 必须精确匹配规范场、精度、格点、块大小与 variant；不匹配就是 cache
miss，不会静默回退。

```python
# docs-skip: identity and path must match an existing cache
from pyqcu.solver.multigrid import LegacyMultigridCache

cache = LegacyMultigridCache(backend="cuda")
loaded = cache.load(
    path, identity=identity, local_shape=(X, Y, Z, T))
assert loaded.variant == "legacy"
```

## MPI 与几何

* CUDA 生产拓扑是一个 Python 进程 = 一个 MPI rank = 一张设备。单进程内多设备
  只是测试专用驱动。
* `process_grid` 的乘积必须等于 MPI size；奇偶布局要求每个局部维度为偶数。
* KUPL 使用同一 rank 网格，rank 内用 OpenMP。固定 `OMP_NUM_THREADS` 与
  `KUPL_OMP_THREADS`，避免超订。
* 奇偶分裂场的局部时间维是 `T/2`；不要把完整场直接 reshape 成该布局。

## 验收与排错

| 现象 | 可能原因 | 检查 |
|---|---|---|
| `no Legacy levels` | `min_size` 大于局部尺寸 | 降低 `min_size` 或使用更大局部格点 |
| `process_grid ... does not match MPI size` | rank 网格错误 | 让 `prod(process_grid)` 等于 `MPI.COMM_WORLD.Get_size()` |
| `coarse_assets ... does not match` | cache identity/形状不匹配 | 用相同格点/块/精度重建 cache |
| `solve_result` 字段为 `None` | Torch Legacy 未暴露 | 用 `last_solve_report`、`convergence_history` 或独立残差 |
| 原生库缺失 | KUPL/CUDA 未构建 | `python -m pyqcu` 与安装文档 |
| 残差高于验收线 | 口径错误或单精度地板 | 检查 `residual_scope`；c64 地板为 `2.4414e-4` |

## 生产求解 checklist

从 notebook 迁移到生产运行时，按以下顺序固定决策：

1. **冻结几何与精度**：记录 `(X, Y, Z, T)`、dtype、`mass`/`csw`/`kappa`
   与 MPI 进程网格。`prod(process_grid)` 必须等于 MPI size；奇偶布局要求
   每个局部维度为偶数。
2. **准备 canonical 张量**：规范场 `[3, 3, 4, ...]`、完整费米子
   `[4, 3, ...]`、奇偶分裂费米子 `[2, 4, 3, ..., T/2]`；求解前运行
   `lattice.check_su3`。
3. **选最小层级**：原型用 Torch `bistabcg`，一次原生整次求解用
   `Krylov`，生产 Clover 用 `Multigrid(variant="legacy")`；Strict 只在
   显式指定时启用。
4. **构建或复用资产**：原生 Legacy-MG 需要预构建的粗层 33-stencil 层级。
   cache identity 覆盖规范场、精度、格点、块大小与 variant；不匹配就是
   miss，不会静默回退。
5. **只求解一次**：原生驱动内部已经完成迭代，不要在 Python 外再包循环。
6. **独立验证**：检查 `SolveResult.residual_scope`；`SCHUR` 残差不是全格点
   残差。验收必须计算全格点残差；c64 地板为 `2.4414e-4`。
7. **记录 provenance**：后端、原生库路径与 SHA256、MPI 网格、线程数、
   迭代数、残差与 wall time。

## 参见

* {doc}`usage`：算子、smearing 与 HDF5。
* {doc}`install`：原生构建变量。
* {doc}`performance`：已测求解与 provenance。
* {doc}`troubleshooting`：求解器与后端故障排查。
