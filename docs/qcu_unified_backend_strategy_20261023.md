# PyQCU QCU 六实现统一架构建议（2026-10-23）

状态：基于当前工作区源码、已归档运行状态和已有 benchmark 文档的架构分析。本文区分
`确证`、`推断`、`未验证`；不把不同算法、不同层级或不同执行路径的耗时直接解释为
“MG 加速比”。

更新（2026-10-23）：P0 公共数据模型与 Protocol 已落地到 `pyqcu.api`，现有
`Multigrid.solve_result()` 提供兼容结果封装；具体字段与已知限制见
[`api_contract.md`](api_contract.md)。P1 已增加统一 `Krylov` 选择器、Torch
`LinearOperator`/BiCGStab/CG，以及 CUDA/KUPL compact-Schur LinearOperator 与原生整次
solve driver。`create_backend()`、统一 backend factory 和逐后端原生 MG 结果对象仍未实现。

## 1. 总结

Torch、CUDA、KUPL 各有 Legacy/Strict 两条实现，共六种小版本。当前没有任何一个小版本在所有维度
同时占优：

- **Strict 更适合作为通用语义骨架**：full-coarse、`X/Y/Yhat`、MATPC、明确的 V-cycle 与右预条件
  Krylov 容易拆成可复用组件，也最容易扩展到三层以上和新后端。
- **Legacy 更适合作为显式特化路径**：compact odd coarse、33-tensor stencil、周期加性 coarse
  correction 在资产较小、已有 cache 和特定硬件路径上仍有价值。
- **CUDA 提供了最成熟的性能工程**：Legacy 的五流/融合/Graph/自适应 correction gate，以及 Strict 的
  persistent hierarchy、fused FGMRES、overlap halo 和 cache 规范，应分别抽象后共享。
- **KUPL 提供了最清晰的 CPU 真残差保护**：coarse correction 的验收、撤销和后续关闭策略应提升为公共
  correction 语义，而不是留在单一后端。
- **Torch 提供了最好的代数参考与调试层**：统一的 transfer/coarse/MATPC/diagnostics 应作为其他后端的
  小格 oracle，但不能作为 CUDA/KUPL 的生产执行循环。

推荐的“最优秀 QCU”不是一个删掉其余实现的大类，而是：

```text
统一 Python 控制面
  + 公共物理/数值语义
  + 可组合的 MG/Krylov 组件
  + 三套后端数据面
  + Legacy/Strict 两种显式策略
```

## 2. 六个实现的对象映射

| 编号 | 小版本 | 当前实现 | 核心对象 | 生命周期 |
|---|---|---|---|---|
| TL | Torch Legacy | `pyqcu.torch.solver.LegacyMultigrid` | 33-point/粗 `dslash.operator`、Python BiCGStab V-cycle | Python 实例持有层级，`setup/close` |
| TS | Torch Strict | `pyqcu.torch.solver.StrictMultigrid` | transfer、Galerkin full coarse、MATPC、递归 V-cycle | Python 实例持有层级，`setup/close` |
| CL | CUDA Legacy | `applyCloverMultigridQcu` → `LatticeCloverMultigrid` | odd-Schur、33-tensor、周期 correction、可选多种 Krylov/smoother | 单次 ABI 调用内 `init/run/end`，资产外置 cache |
| CS | CUDA Strict | `applyMultigridStrict*Qcu` → `StrictCoarseHierarchy` | full coarse、`X/Y/Yhat`、MATPC、fused right-FGMRES | `slot 80` persistent hierarchy，独立 init/end |
| KL | KUPL Legacy | `applyCloverMultigridQcu` → `LegacyCloverMultigrid` | MPI full-site、33-tensor、周期 correction | 每次 solve 内分配层级向量 |
| KS | KUPL Strict | `applyMultigridStrict*Qcu` → `StrictHostHierarchy` | full coarse、MATPC、MR V-cycle、right-FGMRES | `slot 80` persistent hierarchy；FGMRES/halo 临时区每次 solve 分配 |

证据入口：

- TL：[`_legacy_multigrid.py`](../pyqcu/torch/solver/_legacy_multigrid.py#L11)
- TS：[`_multigrid_core.py`](../pyqcu/torch/solver/_multigrid_core.py#L2204)
- CL：[`lattice_clover_multigrid.h`](../cpp/cuda/qcu/include/lattice_clover_multigrid.h#L657)
- CS：[`apply_multigrid_strict.cu`](../cpp/cuda/qcu/src/apply_multigrid_strict.cu#L3160)
- KL：[`lattice_clover_multigrid.cpp`](../cpp/kupl/qcu/src/lattice_clover_multigrid.cpp#L129)
- KS：[`lattice_strict.cpp`](../cpp/kupl/qcu/src/lattice_strict.cpp#L848)

## 3. 六维对比

| 维度 | TL | TS | CL | CS | KL | KS |
|---|---|---|---|---|---|---|
| 粗化与几何 | 扁平 `E`，粗格直接缩小 | full/compact 可选，标准 Strict 为 full coarse | compact odd coarse | full coarse，粗层不 checkerboard | compact full-site/odd 路线 | full coarse |
| 粗算子 | 每层 `dslash.operator` 的 33-point | `R X^-1 D P`，`X/Y/Yhat` | 33-tensor onsite + 8 nearest + 24 diagonal | link hopping + onsite，MATPC 两次作用 | 33-tensor | link hopping + onsite，MATPC |
| 传输 | Python restrict/prolong | blocked aggregate transfer，`R=P†` | 33-tensor P/R | parity R/P + full coarse fields | 33-tensor P/R | parity R/P + full coarse fields |
| 平滑/粗解 | 每层 BiCGStab | MR，多档 setup 方法 | CG/MR/Chebyshev，粗层 guarded BiCGStab | MR，最粗层 BiCGStab | legacy coarse BiCGStab | fixed-step smooth + coarsest BiCGStab |
| 外层策略 | BiCGStab + 周期 correction | FGMRES + 递归 V-cycle | BiCGStab/BiCGStabL/FGMRES/CA-GCR | fused FGMRES | BiCGStab + 周期 correction | FGMRES |
| 并行/设备 | CPU/CUDA/NPU Python 路径 | Python + MPI setup | CUDA MPI、流、Graph、可选 NVSHMEM | CUDA MPI，c64 overlap halo | MPI + OpenMP | MPI + OpenMP |
| 持久化 | Python 对象内 | Python 对象内 | 资产 cache，但运行时层级单次创建 | persistent hierarchy + workspace | 每次 solve 分配 | persistent hierarchy；solve arena 非持久 |
| 主要优势 | 易读、易调试、可组合 | 代数清晰、诊断丰富 | 成熟融合内核、runtime 调优最多 | 数学结构一致、可扩展 | CPU 真残差保护、wide coarse | 结构清晰、代码较简洁 |
| 主要短板 | Python 开销、`x0` 尚未消费、无大格证据 | 大量 Python 张量与参数组合 | 单体复杂、mode bit 多、已有生产记录未跑 coarse correction | 内存/准备成本高、限制严格 | 非持久、性能证据有限 | 临时 halo、尚无同规模性能 A/B |

## 4. 各实现的可取与不足

### 4.1 Torch Legacy

**可取**

- 构建流程直接使用 Python `dslash.operator`、null vectors 和 local orthogonalization，便于逐层观察。
- `seed`、per-level dtype/device、自适应 level-back 等接口适合小格和 NPU/CUDA 混合调试。
- 作为“快速改写算法”的试验层成本最低。
- 当前工作区已修复 `init()` 提前返回导致层级未构建的回归；回归测试断言两层 `op_list`、`b_list` 和
  `lonv_list` 均实际生成。

**不足**

- `solve(x0=...)` 保存了初值，但 `cycle()` 仍从零解开始，热启动语义没有真正实现。
- 停止条件、迭代计数和错误状态没有统一返回对象；大格上 Python 开销和临时张量不可控。
- Legacy 的绝对容差与 Strict 的相对真残差不能直接互换。
- 目前直接依赖 Torch 张量和 Python 控制流；统一 API 应只暴露注入式 field runtime，不应要求 CUDA/KUPL
  生产热路径经过该层。

### 4.2 Torch Strict

**可取**

- `StrictTransfer`、`StrictCoarseOperator`、`StrictMatPCOperator` 将 transfer、Galerkin、MATPC 分离，
  是最适合作为跨后端 algebra oracle 的结构。
- 支持 materialized/matrix-free、compact/full、setup family、diagnostics 和多 rank setup。
- 能把“层级资产”和“Krylov 控制”分开测试。

**不足**

- 构造函数选项过多，默认 `setup_method="random"` 并不等于高质量的物理 null-space setup。
- Python 循环不适合作为 CUDA/KUPL 生产热路径；部分 API 仍以 `dict`/私有字段传递状态。
- 当前没有统一的 capabilities、result 和 asset schema 抽象。

### 4.3 CUDA Legacy

**可取**

- 五流结构、CUDA Graph、zero-copy scalar、guarded coarse BiCGStab、真残差重启和多 smoother 选择是成熟的
  性能工程资产。
- adaptive correction gate 能在粗校正无效时关闭 correction，避免粗层精度破坏 fine Krylov。
- 33-tensor cache、local stencil build 和混合层级支持适合受限显存和历史资产。
- 支持逐层 c64/c128 混合精度，以及 NVSHMEM、CUDA-aware MPI、pinned staging 等可替换 halo transport；
  这些能力应由统一 capabilities 显式声明，而不是藏在 mode bit 中。

**不足**

- 单体 `LatticeCloverMultigrid` 汇集 setup、资产绑定、solver 模式、cycle、诊断和日志，复用边界不清。
- `params` mode bit 重叠、`restart` 同时有不同语义，隐藏行为多，难以做跨后端统一。
- 现有生产 solve 记录大量出现 `n_vcycles=0`；不能据此宣称实际 coarse V-cycle 的性能。
- 层级对象和 33-tensor 资产的所有权跨 Python cache、Cython 槽位与 C++ 单次 solve，错误路径缺少统一的
  opaque handle；统一层必须禁止裸指针隐式续命。

### 4.4 CUDA Strict

**可取**

- 数学对象边界最清晰：fine MATPC、full coarse R/P、coarse `X/Y/Yhat`、recursive V-cycle、right-FGMRES。
- persistent hierarchy 和精确 workspace accounting 适合长 solve 和多层层次；cache v2 的逐 tensor SHA 是可复用
  的 provenance 设计。
- c64 non-blocking overlap、MPI rank-symmetric preflight、fused true residual 和 trace schema 应成为公共协议。
- `slot 60:79` 资产、`slot 80` runtime 和逆序关闭路径形成了可审计的资源所有权，优于 Legacy 的外部隐式生命周期。

**不足**

- 当前 full-coarse 资产、fine transfer 和 FGMRES arena 使端到端峰值显著高于 Legacy；已有可比数据中
  Strict/Legacy 峰值约为 `1.32x`、`1.52x`、`1.63x`。
- 当前 fused 入口固定 fine parity/start level，mixed precision 和部分 operator 支持明确拒绝。
- fine 层的 `nu_pre/nu_post` 独立，但递归粗层在 Python 配置时写成二者最大值，C++ 再对前后平滑共同
  使用 `kMg_MuPre`，见 [`_strict_multigrid.py`](../pyqcu/cuda/_strict_multigrid.py#L163)；因此不能把
  当前实现描述为所有层都支持非对称 pre/post smoothing。
- strict CUDA 的 setup 成本和大格 `24³×64` 完整证据仍不足。

### 4.5 KUPL Legacy

**可取**

- coarse correction 后比较 true residual；若不变、变差或非有限，撤销 correction 并关闭本次 solve 的后续
  correction，是比“只要算法名义上是 MG 就一定接受校正”更稳健的策略。
- wide coarse tile、canonical 转换合并和 paired reductions 显示了 CPU 数据布局优化的价值。
- 细层 `restart=0` 的 parity-Schur BiCGStab 当前仍有很强的 CPU 性能。
- 最终未收敛会明确抛出“未达到 full true residual”，避免把 fine-only 或残差递推值当成成功。

**不足**

- Legacy 每次 solve 分配层内向量，缺少 strict 式持久 hierarchy。
- 现有 `restart=5` coarse 路径和 `restart=0` fine-only 路径性能不可混写；前者本身也不是当前最快路径。
- `restart=0` 是 fine-only parity-Schur 路径，不是“零次重启的 V-cycle”；结果 schema 必须用单独字段记录
  实际 correction/cycle 次数。
- strict 大格性能、同输入 Legacy/Strict A/B 尚未建立。

### 4.6 KUPL Strict

**可取**

- `StrictHostHierarchy` 的层级、V-cycle、MATPC 和 FGMRES 结构清楚，适合作为 CUDA strict 的可读 CPU 对照。
- 持久 `slot 80`、偶数几何检查、同精度 fail-closed 和显式 workspace budget 是可移植的公共约束。
- 小格 1/2/4-rank CTest 已覆盖关键代数路径。
- 最粗层 BiCGStab 失败会显式报错，外层 FGMRES 每次 restart 后从 MATPC 方程重新构造残差。

**不足**

- hopping 期间仍有临时 halo/vector 分配，未达到 CUDA strict 的 persistent halo 与 overlap 水平。
- `final_true_residual` 是重构 full Wilson/Clover 解之前的 fine compact MATPC 残差；最终 `full_out` 虽已
  reconstruct，但没有再用 full operator 重算该返回值，见
  [`lattice_strict.cpp`](../cpp/kupl/qcu/src/lattice_strict.cpp#L1076) 和
  [`lattice_strict.cpp`](../cpp/kupl/qcu/src/lattice_strict.cpp#L1218)。接口必须把它命名为
  `matpc_true_residual`，另设 `full_true_residual` 或明确区分“未计算”。
- 每个 strict primitive 都会重新创建 Cartesian topology 并逐维阻塞 `Sendrecv`，各维还会分配 packed/send/recv
  buffer，见 [`lattice_strict.cpp`](../cpp/kupl/qcu/src/lattice_strict.cpp#L1413) 和
  [`lattice_strict.cpp`](../cpp/kupl/qcu/src/lattice_strict.cpp#L410)；持久 hierarchy 只持久粗层资产和
  level workspace，不持久 MPI communicator、halo 或 FGMRES arena。
- 每次 `run_fgmres` 都重新分配 `V/Z/r/w/x/coarse_rhs/coarse_out/smooth_image`；`allocated_bytes` 表示本次
  arena 预算/占用，不等于 Python 进程常驻总量，见
  [`lattice_strict.cpp`](../cpp/kupl/qcu/src/lattice_strict.cpp#L1014)。
- fine 层 `nu_pre/nu_post` 独立，但递归粗层统一使用 `kMg_MuPre` 作为前后平滑步数；统一配置不能把它误报为
  分层非对称平滑已经完全生效，见 [`lattice_strict.cpp`](../cpp/kupl/qcu/src/lattice_strict.cpp#L864) 和
  [`lattice_strict.cpp`](../cpp/kupl/qcu/src/lattice_strict.cpp#L1354)。
- 大规模专用 CPU benchmark 和大格 strict/legacy 公平 A/B 尚缺。
- FGMRES/outer solver 的组件化程度低于 CUDA strict，错误分类也不完全统一。

## 5. 相互借鉴矩阵

| 来源 | 接收方 | 应借鉴内容 | 保留边界 |
|---|---|---|---|
| CUDA Legacy | Strict | adaptive correction gate、true residual restart、多种 smoother/outer 策略 | correction gate 只能作用于 Legacy 周期加性校正，不能偷偷改变 Strict 右预条件语义 |
| CUDA Strict | Legacy | persistent hierarchy、capability preflight、cache SHA、精确 workspace budget、trace schema | Legacy 的 33-tensor 槽序不能改成 Strict `X/Y/Yhat` |
| KUPL Legacy | 全部 | correction 后真残差验收、撤销和关闭；wide coarse tile；计数反映实际 correction | 不能用撤销策略掩盖真正数值错误，失败必须进入 `SolveResult`/诊断 |
| KUPL Strict | 全部 | strict geometry validation、同精度 fail-closed、持久 host hierarchy | CPU 临时 halo 策略不应直接复制到 CUDA |
| Torch Strict | 全部 | transfer/coarse/MATPC 独立对象、materialized/matrix-free 对照、诊断 oracle | Torch Python 循环不作为生产热路径 |
| Torch Legacy | Legacy | 可组合 Python setup、显式 seed、level-back 调试 | `x0` 优先级、iteration/correction 计数必须重新定义并验证 |
| CUDA/KUPL Legacy | Legacy | 33-tensor cache 和 local stencil builder 的两条实现路径 | 只在验证 `P†DP` 与局部窗口一致时启用 local stencil |
| CUDA/KUPL Strict | Strict | full coarse、MATPC、MR、FGMRES 的同一数学契约 | 算子/cache 的物理 provenance 与精度必须由各后端自己校验 |

## 6. 统一架构

### 6.1 分层

```text
pyqcu/
  api/                      # 新增：公共协议、配置、结果、能力声明
  lattice/                  # 公共布局、gamma、源场
  tools/                    # 公共 MPI/HDF5/env/budget
  torch/                    # Torch/CANN 数据面与参考实现
  cuda/                     # CUDA 数据面与 Cython 桥
  kupl/                     # CPU 数据面与 ctypes 桥
  solver/
    multigrid/              # 公共 Multigrid 选择器
  dslash/                   # 公共 Operator 门面
```

`api/` 不是第四套算法实现，而是所有后端共享的契约层。现有
`pyqcu.torch.*`、`pyqcu.cuda.*`、`pyqcu.kupl.*` 继续拥有实际 kernel 和资产。

### 6.2 公共组件

| 组件 | 公共职责 | 后端特化 |
|---|---|---|
| `LatticeGeometry` | full/parity 几何、process grid、local extent、布局、dtype | halo 与本地索引实现 |
| `LinearOperator` | `apply(x)`, `apply_adjoint(x)`, `diagonal(parity)` | CUDA/KUPL kernel；Torch eager/reference |
| `TransferOperator` | build、restrict、prolong、adjoint 关系 | aggregate/block/tile 布局 |
| `CoarseOperator` | full/compact/matrix-free policy，apply/preconditioned_apply | 33-tensor 或 `X/Y/Yhat` 资产 |
| `Smoother` | fixed-step CG/MR/Chebyshev 的统一控制 | 设备端/OpenMP 更新与同步 |
| `Cycle` | V/W/F/K 和 pre/post smoother 顺序 | 递归调用与 workspace |
| `KrylovDriver` | BiCGStab/BiCGStabL/FGMRES/GCR/CA-GCR 的语义与结果 | fused loop、reduction 顺序 |
| `ResidualMonitor` | relative true residual、刷新周期、breakdown/restart | MPI reduction 与日志 |
| `CorrectionPolicy` | Legacy 周期加性 correction 与 gate/rollback | correction kernel 与真残差 probe |
| `AssetStore` | manifest、算法版本、precision、geometry、哈希 | backend asset schema |
| `Capabilities` | 支持/不支持模式与预算上限 | 平台能力探测 |

### 6.3 两种 MG 策略只是不同组合

```text
LegacyMultigrid =
  KrylovDriver(BiCGStab/BiCGStabL/FGMRES/GCR)
  + PeriodicAdditiveCorrection(CoarseOperator=Blocked33)
  + CorrectionGate/TrueResidualRollback

StrictMultigrid =
  KrylovDriver(FGMRES, default)
  + RightPreconditioner(RecursiveCycle)
  + CoarseOperator=FullGalerkin(X/Y/Yhat)
  + MATPC/RestrictionProlongation
```

两者共享 `LinearOperator`、`Smoother`、`ResidualMonitor`、`AssetStore` 和 `SolveResult`，但
`CoarseOperator`/`TransferOperator` 资产不互转，也不以“同名”掩盖差异。

### 6.4 调用契约与兼容策略

公共控制面只固定语义和顺序，不要求三个后端共享同一种存储：

```text
backend = create_backend(name, device/process_grid, precision)
operator = backend.dslash(...)                 # LinearOperator
plan = backend.plan(operator, config)          # preflight + asset identity
hierarchy = backend.multigrid(plan, variant=...) # legacy | strict
hierarchy.setup()
result = hierarchy.solve_result(rhs, x0=x0, control=...)  # SolveResult
hierarchy.close()
```

- `setup()` 负责能力检查、collective preflight、资产校验和 workspace 预留；不支持的模式必须在 collective
  操作前 fail-closed。
- `solve_result()` 是完整结果入口；兼容层 `solve()` 只返回 `result.x`，不得拥有第二套停机或计数逻辑。
- 后端对象拥有 hierarchy、halo、communicator 和 workspace 的生命周期；调用方不得通过 `set_ptrs` 或裸指针
  隐式延长资产寿命。
- `pyqcu.api.create_backend()` 是扩展到 dslash/Krylov 后的目标 canonical 入口；现有
  `pyqcu.solver.multigrid.Multigrid(variant=..., backend=...)` 继续作为兼容选择器，并委托给同一后端实现。
- 公共协议不规定 kernel、stream、OpenMP、MPI transport 或 cache payload；这些由 `Capabilities` 和
  `AssetStore` 声明，避免为统一接口牺牲平台性能。

## 7. 目标 Python 接口

以下为建议的 canonical API；当前尚未实现，不应把示例当成现有功能。

```python
from pyqcu import create_backend
from pyqcu.api import (
    BiCGStabControl,
    CoarsenConfig,
    CycleConfig,
    FGMRESControl,
    MRConfig,
    PeriodicCorrectionConfig,
    StrictMultigridConfig,
)

backend = create_backend(
    "cuda",
    device="cuda:0",
    precision="c64",
    process_grid=(1, 1, 1, 1),
)

D = backend.dslash(
    U=U,
    clover=A,
    kappa=kappa,
    mode="full",       # full / parity / matpc / adjoint
)

bicg = backend.solver(
    "bicgstab", operator=D, control=BiCGStabControl(rtol=1e-8)
)
result = bicg.solve(b, x0=None)       # -> SolveResult

mg = backend.multigrid(
    variant="strict",
    operator=D,
    coarsen=CoarsenConfig(nvec=12, block_size=(2, 2, 2, 2), levels=3),
    smoother=MRConfig(steps=2),
    cycle=CycleConfig(kind="V", pre=2, post=2),
    outer=FGMRESControl(rtol=1e-8, restart=20, max_iter=200),
)
mg_result = mg.solve(b, x0=None)
```

Legacy 使用同一 `backend/operator/solve/close`，但配置类型明确写成：

```python
mg_legacy = backend.multigrid(
    variant="legacy",
    operator=D,
    coarsen=CoarsenConfig(nvec=12, block_size=(2, 2, 2, 2), levels=2),
    corrector=PeriodicCorrectionConfig(
        interval=5, gate=True, true_residual_rollback=True
    ),
)
```

### 7.1 `SolveResult`

```text
x
backend
variant
method
converged
status
krylov_iterations
restart_count
correction_count
proposed_correction_count
rejected_correction_count
cycle_count
coarse_iterations
final_residual
residual_scope              # matpc / schur / full
matpc_true_residual
full_true_residual          # 未计算时必须为 null，而不是复制另一口径
relative_true_residual
setup_seconds
solve_seconds
workspace_bytes
cache_status
diagnostics
```

`restart` 只能出现在 Krylov 配置中；`restart_count`、`correction_count`/`cycle_count` 必须记录实际发生
的事件。`correction_count` 只统计被接受并改变迭代状态的 correction，拒绝和回滚分别进入另外两个计数。
任何残差都必须带 `residual_scope`，禁止把 KUPL compact MATPC 残差标成 full Wilson/Clover 真残差。

### 7.2 `BackendCapabilities`

```text
full_coarse
compact_coarse
parity
matpc
adjoint
persistent_hierarchy
overlap_halo
mixed_precision
supported_precisions
cuda_graph
nvshmem
max_levels
cache_schema
warm_start
```

`backend.multigrid()` 在构建前用 capabilities 做 fail-closed 校验；不在运行时静默切换到另一条层级路线。

## 8. 扩展到 dslash、BiCGStab、CG

### 8.1 Dslash

公共 `LinearOperator` 应至少提供：

```text
apply(x, parity="full"|0|1, adjoint=False)
apply_diagonal(x, parity)
to_parity(parity)
to_matpc(parity)
capabilities()
```

- Torch 用 `pyqcu.torch.dslash.operator` 实现。
- CUDA/KUPL 用各自 kernel，但输入输出布局、dtype 和异常语义相同。
- `NormalOperator(D)` 是对 Wilson 非厄米算子做 CG 的显式包装，不得把普通 Wilson CG 伪装成 HPD CG。

### 8.2 BiCGStab

公共 driver 负责 Krylov 轨迹、breakdown policy、relative/absolute tolerance、true residual probe 和
`SolveResult`；后端只提供 `operator.apply`、`dot/norm/axpy` 等原语。CUDA/KUPL 可以保留 fused iteration，
但必须输出同一组计数器与结果。

### 8.3 CG

- CG 只在算子满足 HPD 时启用；Wilson full operator 需显式构造 `D†D` 或 Schur/Hermitian 子问题。
- CG driver 与 BiCGStab 共享 `LinearOperator`、`ResidualMonitor`、`SolveResult`。
- 不把“名字里有 CG”当作数学前提；后端应在 capabilities 中标明 `hpd_cg`、`normal_cg`、`schur_cg`。

## 9. 平台特化边界

| 平台 | 必须保留的特化 |
|---|---|
| Torch | Python eager/reference、CPU/CUDA/NPU 多设备语义、diagnostics、matrix-free 和高层可组合性 |
| CUDA | stream/stream ordering、Graph、fused reductions、persistent hierarchy、overlap halo、精确 workspace |
| KUPL | MPI+OpenMP、NUMA/affinity、canonical 转换、wide coarse tile、CPU 真残差 rollback |

统一接口不能要求 CUDA/KUPL 走 Python kernel；也不应为了“看起来统一”把 CUDA 的持久 workspace 强行塞进
Torch 的 tensor hierarchy。公共的是控制面与数学契约，特化的是数据面和同步策略。

## 10. 推荐默认

1. **通用生产默认使用 Strict**：full-coarse、MATPC、MR smoother、right-preconditioned FGMRES，便于验证和扩展。
2. **Legacy 作为显式兼容/特化策略**：只有在 compact 资产、历史 cache 或经过公平 A/B 证明更优的工作负载上选用。
3. **所有变体统一以相对真残差为主停机判据**；Legacy 的绝对容差只作为显式兼容选项。
4. **Legacy correction 默认启用 gate + true-residual rollback**，并记录实际 correction/cycle 次数。
5. **Strict 默认 MR smoother**；CG 只用于已证明 HPD 的 Schur/normal 子问题。
6. **禁止隐式 fallback**：Strict 不得静默退回 Legacy，Legacy 不得把 fine-only 路径报告成 coarse V-cycle。

## 11. 实施顺序

### P0：契约冻结

- 新增 `pyqcu.api` 的 `Capabilities`、`SolverControl`、`SolveResult`、`AssetManifest`。
- 定义 `LinearOperator`、`TransferOperator`、`CoarseOperator`、`Smoother`、`KrylovDriver` 协议。
- 保持当前 `pyqcu.solver.multigrid.Multigrid` 作为兼容入口。

### P1：Dslash 与 Krylov 统一

- 为三后端提供同一个 `LinearOperator` 适配器。
- 用 Torch 作为参考，统一 BiCGStab 的 breakdown、warm start、true residual 和 `SolveResult`。
- 新增显式 `NormalOperator`，再统一 CG；禁止把普通 Wilson 算子直接交给 CG。

### P2：Strict MG 组件化

- 以 CUDA strict 为生产语义基线，以 Torch strict 为 algebra oracle，以 KUPL strict 为 CPU 对照。
- 抽出 `Transfer/Coarse/MATPC/MR/Cycle/KrylovDriver` 接口。
- 将 persistent hierarchy、workspace budget、cache manifest 和 counters 提升为公共契约。

### P3：Legacy MG 组件化

- 保留 33-tensor 资产和 compact geometry；将 correction gate、真残差验收/撤销、local stencil guard 提升为公共策略。
- 先统一设置/关闭、错误分类、计数器和结果，再考虑是否共享 Krylov driver。

### P4：退出路径

- 旧 `apply*Qcu` 保留为低层/兼容 ABI，不作为新 PyQCU 用户的首选入口。
- 所有新代码只从 `pyqcu.api` / `backend.multigrid` / `backend.solver` 进入。

## 12. 验收门

### 数值与代数

- 六种小版本在同一 gauge/source 上完成 transfer adjoint、Galerkin、MATPC、V-cycle 线性性测试。
- Strict 的 full-coarse `R(X^-1 D)P` 与独立参考一致；Legacy 的局部 stencil 与全局 `P†DP` 在周期/非周期边界一致。
- `x0` 在所有 driver 中真正参与热启动，冷/热结果和迭代数差异有记录。
- 相对真残差是统一主判据；允许绝对容差时必须显式标注兼容模式。

### 并发与生命周期

- 1/2/4 rank 使用对称元数据、halo 和 reduction；不支持的模式在 collective 前 fail-closed。
- `setup()`/`close()` 幂等，异常路径逆序释放；Strict slot 索引不被错误递增。
- 记录实际 `krylov_iterations`、`correction_count`、`cycle_count` 和 `coarse_iterations`。

### 资源与性能

- 分别记录 setup、cache 验证/加载、首次 workspace、warmup、steady solve 和 close。
- Legacy 必须有至少一个实际 `correction_count > 0` 的正确性回归，才能报告 Legacy V-cycle 性能。
- 跨后端比较至少包含同一 RHS、同一精度、同一层数/块大小、同一真实停机阈值和独立真残差。
- 不以现有 CUDA 小格结果、KUPL fine-only 结果或 Torch 小格结果推导通用性能结论。

### 扩展性

- 新增一个后端只需实现 `LinearOperator`、基础 field ops、halo/reduction、hierarchy/cache adapter，不重写公共 API。
- 新增 Krylov 方法只实现 driver 和 capability 声明，不改变 Dslash/MG 的用户接口。

## 13. 反模式

- 不要把 Legacy 33-tensor 与 Strict `X/Y/Yhat` 自动互转。
- 不要用 `restart` 代替 cycle/correction 计数。
- 不要在 CUDA/KUPL 生产热循环中经过 Python `LinearOperator` 调用。
- 不要让 Strict 在几何/精度不支持时静默退回 Legacy。
- 不要用 fine-only 计时冒充 coarse MG 计时。
- 不要把 Torch reference 的大范围 Python 分支复制到 C++；只共享数学契约和测试 oracle。
- 不要把多设备线程 worker 作为生产拓扑；生产仍是 MPI rank = process = device。

## 14. 证据与未验证项

**已确证**

- 当前统一生命周期和 strict/legacy 命名已存在，但构造参数、资产格式和结果对象尚未统一。
- CUDA Legacy 记录覆盖 4 个小格尺寸的 build/solve，当前记录未触发 coarse correction。
- CUDA Strict 记录覆盖 3 个小格尺寸的 solve；`24³×64` 缺失。
- CUDA 现有可比容量记录显示 Strict 峰值高于 Legacy，约 `1.32x`–`1.63x`。
- KUPL strict/legacy 有 1/2/4-rank 正确性路径，但没有同规模公平性能 A/B。
- Torch 两条路线主要由小格代数测试覆盖，不构成生产性能结论。
- KUPL Strict 返回的是重构前的 compact MATPC 残差；其 recursive coarse smoother 目前只使用
  `kMg_MuPre`，FGMRES arena、halo buffer 和 Cartesian topology 也未跨 solve 持久化。
- CUDA Strict 同样只在 fine 层保持独立 `nu_pre/nu_post`；recursive coarse layer 使用二者最大值。
- 当前工作区已把 Torch 实现迁入 `pyqcu/torch/`，并由 `pyqcu.solver.multigrid.Multigrid` 统一选择
  `torch/cuda/kupl × legacy/strict`；这不是本文建议的完整 `pyqcu.api`。
- Torch Legacy `init()` 提前返回的回归已在当前工作区修复，并有“层级实际构建”的 CPU 回归测试。

**未验证**

- Legacy coarse correction 与 Strict recursive V-cycle 在同一大格、同精度、同预算下的公平性能。
- KUPL 大格 Strict 与 Legacy 的峰值内存和长期稳态。
- CUDA/KUPL/Torch 三后端在 c128 和真实 LIME 输入上的统一 API 回归。
- `SolveResult`、counters、公共 Protocol 和资产 manifest 的 P0 schema 已实现；最终仍需由
  各后端原生适配并完成全后端回归。
- `pyqcu.api.create_backend()` 和完整 backend factory 尚未实现；在实现前，不能把本文示例
  中的 factories 当作可运行 API。
