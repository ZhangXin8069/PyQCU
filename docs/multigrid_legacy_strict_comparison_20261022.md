# PyQCU Multigrid：Legacy 与 Strict 对比摘要（2026-10-22）

状态：代码、小格正确性回归与既有运行记录的核对；本次未新增性能 benchmark。

## 1. 结论

PyQCU 现在把 multigrid 明确分成两条互不替代的层级路线：

- **LegacyMultigrid**：保留历史 33-stencil、细层 BiCGStab 与周期 coarse correction 语义。
- **StrictMultigrid**：保留 full coarse lattice、逐层 MATPC、`X/Y/Yhat` 与递归 V-cycle/FGMRES 语义。
- 上层 `Multigrid` 只负责按 `variant` 和 `backend` 选择实现；算法、容差、资产格式和生命周期边界仍由下级决定。
- Torch、CUDA、KUPL 都使用同名 `LegacyMultigrid` / `StrictMultigrid`，统一生命周期为 `setup()` → `solve()` → `close()`。

```python
from pyqcu.solver.multigrid import Multigrid

mg = Multigrid(variant="strict", backend="torch", **hierarchy_options)
mg.setup()
try:
    solution = mg.solve(rhs, x0=None)
finally:
    mg.close()
```

`backend` 当前可取 `torch`、`cuda`、`kupl`。后端特定参数通过 `**kwargs` 传给具体实现，不应伪造跨后端同义参数。

## 2. 算法与数据语义

| 维度 | LegacyMultigrid | StrictMultigrid |
|---|---|---|
| 粗化对象 | 历史扁平 coarse DOF；第一粗层采用 33-tensor `onsite + 8 nearest + 24 diagonal` stencil | 全场粗算子的 `R X^-1 D P`；粗格点保持 full lattice |
| 传输 | near-null/blocking `P/R`，资产按 Legacy 槽序绑定 | spin `4→2` 的 aggregate transfer，`R=P†` |
| 奇偶边界 | Legacy/CUDA 可采用 compact odd-Schur setup；不等于 Strict | 只有 fine MATPC、prepare、R/P 消费 compact parity；粗层不得 checkerboard |
| 粗算子资产 | CUDA/KUPL 使用 `V/hop_nn/hop_diag/sit` | `V/Yhat/(X,X^-1)`；raw `Y` 只是诊断资产 |
| 外层迭代 | 细层 BiCGStab，按 restart 周期触发 coarse correction；KUPL 另有真残差验收与撤销 | 右预条件 FGMRES；递归 MATPC V-cycle，MR 前后平滑 |
| 容差 | 历史路径通常使用绝对容差；求解器入口需逐后端核对 | 当前生产语义以相对真残差门为主 |
| 生命周期 | 传统 `init/end` 包装为 `setup/close` | `setup` 构造层级；`close` 释放层级/运行期资产 |

不能把 Legacy 的 33-tensor cache 改名为 Strict 的 `X/Y/Yhat` cache，也不能在两者之间自动转换槽序。二者只共享上层选择器和生命周期名称。

## 3. 后端映射

| 后端 | LegacyMultigrid | StrictMultigrid |
|---|---|---|
| Torch | `pyqcu.torch.solver.LegacyMultigrid`，Python/CUDA 混合自适应 V-cycle | `pyqcu.torch.solver.StrictMultigrid`，Python 参考层级与 Galerkin/MATPC 资产 |
| CUDA | `pyqcu.cuda.multigrid.LegacyMultigrid`，包装 `applyCloverMultigridQcu` monolithic ABI | `pyqcu.cuda.multigrid.StrictMultigrid`，包装 persistent strict hierarchy 与 fused right-FGMRES |
| KUPL | `pyqcu.kupl.multigrid.LegacyMultigrid`，包装 CPU Clover-MG ABI | `pyqcu.kupl.multigrid.StrictMultigrid`，包装 strict init/FGMRES/end ABI |

实现入口见
[`multigrid/__init__.py`](../pyqcu/solver/multigrid/__init__.py#L17)、
[`_legacy_multigrid.py`](../pyqcu/torch/solver/_legacy_multigrid.py#L10)、
[`_multigrid_core.py`](../pyqcu/torch/solver/_multigrid_core.py#L3836)、
[`cuda/multigrid/legacy_multigrid.py`](../pyqcu/cuda/multigrid/legacy_multigrid.py#L12)、
[`cuda/_strict_multigrid.py`](../pyqcu/cuda/_strict_multigrid.py#L95)、
[`kupl legacy`](../pyqcu/kupl/multigrid/legacy_multigrid.py#L12) 与
[`kupl strict`](../pyqcu/kupl/multigrid/strict_multigrid.py#L12)。

## 4. 后端实现差异

### Torch

- Legacy 使用 `pyqcu.torch.tools._multigrid` 的 null-vector、正交化、restrict/prolong 与 `pyqcu.torch.dslash.operator`。
- Strict 使用 `_multigrid_core.py` 中的 transfer、coarse operator、MATPC 和递归 hierarchy。
- 小格回归覆盖 transfer/Galerkin、`Yhat`、MATPC、V-cycle 与 matrix-free 路径；不构成性能结论。

### CUDA

- Legacy 入口是 [`apply_clover_multigrid.cu`](../cpp/cuda/qcu/src/apply_clover_multigrid.cu#L21)，核心为 [`LatticeCloverMultigrid`](../cpp/cuda/qcu/include/lattice_clover_multigrid.h#L657)。
- Strict 核心为 [`StrictCoarseHierarchy`](../cpp/cuda/qcu/src/apply_multigrid_strict.cu#L3160)，fused FGMRES 入口在 [`applyMultigridStrictFgmresQcu`](../cpp/cuda/qcu/src/apply_multigrid_strict.cu#L5022)。
- Legacy coarse 资产位于 `set_ptrs[30+4k]`；Strict 资产从 `set_ptrs[60]` 起，持久 hierarchy 句柄位于 `set_ptrs[80]`。
- 两者都保持一 Python 进程 = 一 MPI rank = 一设备的执行拓扑。
- 本次重命名后的 V100 小格回归中，strict fused-FGMRES 3 项与 strict transfer/V-cycle 11 项共 `14 passed`。

### KUPL

- Legacy 入口为 [`applyCloverMultigridQcu`](../cpp/kupl/qcu/src/apply_clover_multigrid.cpp#L8)，核心为 [`LegacyCloverMultigrid`](../cpp/kupl/qcu/src/lattice_clover_multigrid.cpp#L129)。
- Strict 核心为 [`StrictHostHierarchy`](../cpp/kupl/qcu/src/lattice_strict.cpp#L848)，FGMRES 包装在 [`applyMultigridStrictFgmresQcu`](../cpp/kupl/qcu/src/apply_multigrid_strict.cpp#L266)。
- 使用 MPI 四维分域、rank 内 OpenMP；Legacy 每次 solve 分配层内向量，Strict 可通过 `set_ptrs[80]` 持久化层级工作区。
- 当前只支持同精度 c64/c128；CUDA-only GCR/CA-GCR 或混合精度配置必须 fail-closed。
- 本次 1/2/4-rank strict/legacy MG CTest 共 8 项通过；统一 Python 生命周期包装另有 KUPL 回归覆盖。

## 5. 已有对比证据

### CUDA 状态

既有同输入 c64、`tol=1e-6`、2 次 warmup + 5 次 steady 的记录显示：

- Legacy build 与 solve 在 `8³×16`、`12³×24`、`16³×32`、`24³×64` 均为 PASS，真残差低于 `1e-6`。
- Strict null-vector 创建 4/4 成功；Strict solve 前三组 PASS，`24³×64` 只有空 stdout、无 JSON。
- Legacy stdout 的 `n_vcycles=0`，只能证明进入 Legacy solve 并完成 fine-level 迭代，不能证明实际发生 coarse V-cycle。
- Strict `restart=20` 不是 V-cycle 计数；现有 JSON 未提供 cycle counter。
- 已成功尺寸的 steady wall 为 Strict 慢于 Legacy，但因 Legacy 未实际执行 coarse correction、两者迭代语义不同，不能写成 MG 算法加速比。

详细数据和可比性边界见
[`legacy_vs_strict_mg_status_20261022.md`](legacy_vs_strict_mg_status_20261022.md#8-legacy-cuda-solve)。

### KUPL 状态

- Legacy 生产 c64 记录覆盖 `24³×64`、4 ranks × 4 threads，47 iterations、native relative residual `9.138e-7`、verifier=0。
- Strict 有 1/2/4-rank CPU 正确性测试，但尚未找到与 Legacy 同规模、同输入的正式性能 A/B。
- 因此目前只能确认两条 KUPL 路径可运行，不能声明二者性能优劣。

### Torch 状态

- 当前小格 CPU 测试验证两条层次的代数关系和接口隔离，不测量大格性能。
- 本次统一接口与 strict 参考回归合计 `41 passed`；与 KUPL Python 回归合并运行为 `51 passed`。

## 6. 统一命名后的硬边界

1. `multigrid` 是上级命名空间和选择器，不再指代旧 Python 类。
2. 具体路线只能叫 `LegacyMultigrid` 或 `StrictMultigrid`；不再使用 `quda_multigrid` 一类名字描述 PyQCU 本地实现。
3. 外部 QUDA/PyQUDA 对照目录可以继续保留 `quda`，因为它标识第三方实现，不是 PyQCU 的算法路线。
4. `setup/solve/close` 只是公共生命周期，不表示两者可以共享资产、容差或 cycle 语义。
5. CUDA/KUPL 的低阶 C ABI 暂时保持兼容；统一入口位于 Python `multigrid/` 命名空间。

## 7. 未验证项

- Strict CUDA `24³×64` 缺失记录及失败原因。
- Legacy CUDA 真正执行 coarse V-cycle 的同输入证据。
- KUPL Legacy/Strict 同规模性能 A/B。
- Torch 两条路径的大格收敛与性能比较。
- CUDA 多 rank、c128、异构 GPU 与长时间稳态资源行为。
- 所有后端跨精度、跨资产格式的自动转换；当前明确禁止。
