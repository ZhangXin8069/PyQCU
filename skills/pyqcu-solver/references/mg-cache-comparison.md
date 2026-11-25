# 33-tensor cache 与 Strict cache 对比

本文记录 PyQCU 两条 MultiGrid 粗化路径的 cache 语义。用户询问 cache 内容、选择依据、
HDF5 布局、`set_ptrs` 槽序、加载边界或性能计时口径时读取本文。深层 transition 的
构造（递归、无 coarse gauge）、多 MPI 构建的耗时陷阱与两个已知构建缺陷也在本文。

## 结论

两者不是同一种 cache 的两种序列化格式：

- **33-tensor cache** 面向 legacy/compact-Schur MG，直接缓存粗算子
  `A_c=P^\dagger S P`。
- **Strict cache** 面向 QUDA-style strict MG，缓存预条件层级资产
  `V/Yhat/(X,X^-1)`，运行时以 full-coarse onsite + 8-link 形式作用。

两者可以描述相关的 Galerkin 物理，但 geometry、槽序、C++ kernel 和生命周期均不同，
不能直接互换。

## 核心对比

| 维度 | 33-tensor cache | Strict cache |
|---|---|---|
| 数学对象 | 显式 `A_c=P^\dagger S P` | `D=X+H`，使用 `Yhat=X^-1Y` |
| 粗格 geometry | compact odd coarse lattice | full coarse lattice |
| parity | 粗层本身为 odd/compact | 仅 fine R/P、MATPC 边界使用 parity |
| 位移类别 | 1 onsite + 8 nearest + 24 diagonal | onsite + 8 个正负方向 link |
| 主要数据 | `lonv/hnn/hdg/sit` | `fine_blocked_v`、`Yhat`、`(X,X^-1)`、递归 `V` |
| raw link | 不适用 | 可选；普通 solve 不驻留 |
| cache 内容 | “算子结果” | “预条件算子因子” |
| 运行期槽位 | `[lonv, hop_nn, hop_diag, sitting]` | `[V, raw Y(optional), Yhat, (X,X^-1)]` |

## 33-tensor cache

每个 transition 保存：

```text
lonv: near-null vectors
sit:  [E,E,Xc,Yc,Zc,Tc]
hnn:  [2,4,E,E,Xc,Yc,Zc,Tc]       # 8 个最近邻
hdg:  [2,2,6,E,E,Xc,Yc,Zc,Tc]     # 24 个双轴对角项
```

`hnn/hdg/sit` 是三个数组，但逻辑上包含 33 个位移 block。矩阵主体大小为：

```text
33 * E^2 * Vc_odd
```

另外保存 `lonv`。当前 simple builder 固定两级、即一条 transition；多层 MG 必须为每条
transition 各保存一套。HDF5 写入只包含 `lonv/hnn/hdg/sit`，构建报告在外部 JSON 中记录
gauge SHA256 和物理参数，调用方必须按身份隔离路径。

实现锚点：

- `pyqcu/torch/tools/_multigrid.py:739`：Python stencil 构建与形状。
- `cpp/cuda/qcu/src/multigrid.cu:858`：wide-stencil CUDA kernel。
- `pyqcu/testing/simple/kupl_multigrid.py:172`：`lonv/hnn/hdg/sit` 读取。
- `pyqcu/testing/simple/kupl_multigrid.py:211`：四 dataset 写入。

## 深层 transition：递归，而非 coarse gauge

33-tensor 层级是纯代数递归，与 gauge 无关（`logs/v20261123.txt` §177）：

```text
op(0) = fine odd-Schur Clover operator
op(L) = R_L · op(L-1) · P_L
```

`op(L-1)` 本身可以是一个 33 点 stencil，因此 level 2 的算子是 level 1 stencil 的
Galerkin 投影。全树搜索 `coarse_gauge` / `block_average` / `coarsen_gauge` /
`coarse_clover` 只命中 QUDA 散文，没有实现、也不需要实现。torch/CUDA 链就是这么
写的：`pyqcu/cuda/_multi_gpu.py:84-160` 逐层构建，`:229`
`S = lambda v: apply_stencil(hnn, hdg, sit, v)` 就是全部交接。

### 探针在原 stencil 父算子上有效，且已数值验证

两个构建原语对 `(E, e, dims)` 通用，父算子只需是一个 matvec：

| 原语 | 位置 | 作用 |
|---|---|---|
| `give_null_vecs` | `pyqcu/torch/tools/_multigrid.py:105-166` | 逆迭代产生候选零空间 |
| `local_orthogonalize` | `:168-241` | 分块正交化 |
| `build_stencil` | `:873-953`（`_probe_point` `:873-918`、`build_stencil` `:920`） | 探 33 个 offset 得到 `P†(parent)P` |
| `apply_stencil` | `:1018-1060` | 下一代父 matvec |

2026-11-23 在合成父 stencil 上验证（§177）：

```text
isotropic block 2, E=(2,2)      : max ||A2 v - R A1 P v||/||.|| = 4.89e-16
A2·delta 的支撑                  : 恰好 33 个 offset
anisotropic (2,2,2,1), E=(3,3)  : 4.26e-16，同为 33 个 offset
```

33 点父算子只能产生 `{0, ±e_mu, ±e_mu±e_nu}` 的耦合，所以该构造归纳到任意深度都
成立，不需要新的物理输入。多层所需的“每条 transition 一套资产”已在上一节说明。

不可做的一件事：把 coarse gauge + coarse Clover 交给
`applyMultigridBuildNullVectorsQcu` / `applyMultigridBuildStencilLocalQcu`——两者
的入口都只收细层 gauge 与 Clover（`cpp/kupl/qcu/src/apply_multigrid.cpp:232,281`）。
Wilson-Clover 是 onsite + 1 hop，33 点 stencil 有 2 hop 耦合，没有任何 gauge 能
复现它，两个 builder 会静默构建**另一个**算子的层级（§177）。level 2 的构建因此
必须留在 Python/torch，那里 matvec 是参数。

### 建链并跑三层的可复现配方（2026-11-25 实测通过）

三个工具职责不同，**不要混用**：

| 工具 | 职责 | 是否随包发行 |
|---|---|---|
| `pyqcu.test --mg-levels N` | N 层**求解**；深层根只消费、不构建 | 是 |
| `pyqcu.testing.simple.mg_transition` | 构建 transition ≥ 2（33-tensor 探针） | 否（checkout-only） |
| `pyqcu.testing.simple.kupl_multigrid.py` | 只构建 transition 1；它的 `--levels` 是"要建的层级数"，2 即上限 | 否 |

在计算节点上（launcher 三要素见 `pyqcu-testing/references/remote-l48-experience.md`，
`-nl "!host"` 排除 inode 耗尽的节点见 `logs/v20261123.txt` §181）：

```bash
# 1) transition 1：C++ 构建器。E_1 必须取 12（见下）。
python -B pyqcu/testing/simple/kupl_multigrid.py \
  --lat 16 16 16 32 --grid 2 2 2 1 --mass 0.1 --csw 1.160920226 --dtype c64 \
  --levels 2 --dof 12 12 --mg-grid 2 2 2 2 --nv-iters 2 --restart 5 \
  --write-coarse-cache <L1> --gauge <gauge.h5> --source <source.h5>
# 实测：status=PASS，builder 186 s（stencil 171 s），lonv [12,12,8,2,8,2,8,2,8,2]

# 2) transition 2：Python 探针，父层已是 33-tensor
python -B -m pyqcu.testing.simple.mg_transition \
  --parent <L1> --output <L2> --parent-lattice 16 16 16 32 \
  --mg-grid 2 2 2 2 --dof 12 2 --dtype c64 --samples 3 --seed 42
# 实测：Galerkin 3.833e-07 (<=2e-5) | Gram 1.073e-06 (<=1e-4) | 33 offsets
#       identity lattice [8,8,8,16]（父层粗盒的两倍奇时间），dof [12,2]

# 3) 求解
python -B -m pyqcu.test --mode solve --native mg --only kupl/native/mg \
  --dtype c64 --lattice-size 16 16 16 32 --lattice-grid 2 2 2 1 --ranks 8 \
  --mass 0.1 --atol 1e-5 --max-iter 2000 \
  --mg-levels 3 --mg-dof 12 12 2 --mg-grid 2 2 2 2 --restart 5 \
  --cache-dir <L1> --coarse-cache-level <L2> --outdir <out>
```

判读：只有 `PASS` + `residual_verified: true` 才算数；`MG_LEVEL_ITERS` 的
`level2` 必须 `> 0`，这是"第三层被真正进入"的唯一证据。实测三层
`level0=59 level1=12 level2=12 accepted_corrections=12`，
2.3566 s→3.7684 s（mg-2→mg-3），残差 4.28e-06→3.19e-06（`logs/v20261123.txt` §184）。
该行是 2026-11-25 删除前的记录：当时还打印 `fallback_fine=`，且
`accepted_corrections` 计被接受的阻尼修正；现在 `MG_LEVEL_ITERS` 只打印
`level0/level1/...`、`total=` 与 `accepted_corrections=`，后者计**预条件应用次数**。

两个必须同时成立的条件，别当成两个限制：

- **`E_1 = 12`**：`legacy_mg_cache_shapes` 的 `fine_dof = 12` 与 ABI 包装
  `apply_multigrid.cpp:149,167` 的 `e = kVectorSize` 都是硬编码，取 `E_1 = 12`
  时二者同时为真。求解器内部不用改——`restrict(parent)` 传 `p.E` 作 `e`、`c.E`
  作 `E`（`lattice_clover_multigrid.cpp:986-1001`），所以 transition 2 的 `e`
  就是 level 1 的 `E`；ABI 那两处只影响**独立**的 restrict/prolong 入口。
- **`kappa < kappa_c`**：`mass=+0.1` → `kappa=0.121951 < 0.125`。用 `mass=-0.277`
  时同一块 gauge 上 BiCGStab 残差涨到 1.29、第 229 次变 NaN（§180），任何构建或
  收敛结论都会变成测量伪影。

## Strict cache

每条 transition 保存：

```text
V:      blocked near-null vectors
Yhat:   [2,4,E,E,Xc,Yc,Zc,Tc]
X/X^-1: [2,E,E,Xc,Yc,Zc,Tc]
```

Strict 的粗层不做 checkerboard crop，`P/R` 只在 fine parity 与 full coarse field
之间转换。其粗化关系为：

```text
D_{l+1} = R (X_l^-1 D_l) P
```

普通 solve 只需要 `Yhat` 和 `(X,X^-1)`；raw `Y` 用于 setup/诊断，默认不驻留。若层级有
`T` 条 transition，runtime cache 通常有：

```text
1 fine V + 2T Yhat/onsite + (T-1) recursive V = 3T logical tensors
```

`T=1` 是 3 个逻辑 tensor，`T=2` 是 6 个。这里的 logical tensor 与 33 个 block 使用不同
计数口径。

实现锚点：

- `pyqcu/torch/solver/_multigrid_core.py:1506`：`X/Y/Yhat` packed ABI。
- `pyqcu/torch/solver/_multigrid_core.py:3589`：strict transition 四槽导出。
- `pyqcu/cuda/_strict_cache.py:1`：schema v2 cache 与不可变发布。
- `pyqcu/cuda/_strict_cache.py:1074`：完整验证后加载。
- `cpp/cuda/qcu/src/apply_multigrid_strict.cu:374`：strict coarse apply kernel。

## 存储量

33-path 每 transition 的矩阵 block 数为：

```text
33 * E^2 * Vc_odd
```

Strict runtime 每 transition 为：

```text
(8 Yhat + 2 onsite) * E^2 * Vc_full + 对应 V
```

不能只比较 `33` 与 `10`。标准每轴 block=2 时，通常：

```text
Vc_full ~= 2 * Vc_odd
```

因此 33 block 与 10 full-coarse block 的实际比例要同时考虑 geometry。若 Strict 额外保留
raw `Y`，还会增加 8 个 link block。

## 运行显存峰值（2026-10-23）

同一 gauge/RHS、c64、两级、`nvec=12` 的独立进程 benchmark 中，Legacy-MG
的设备级运行时峰值低于 Strict-MG：

| 格点 | Legacy solve | Strict solve | Strict / Legacy |
|---|---:|---:|---:|
| `8x8x8x16` | 639,631,360 B | 847,249,408 B | 1.3246 |
| `12x12x12x24` | 1,646,264,320 B | 2,497,708,032 B | 1.5172 |
| `16x16x16x32` | 4,416,602,112 B | 7,210,008,576 B | 1.6325 |

该数值是设备级 `cudaMemGetInfo` 峰值相对进程基线的增量，包含 C++ 原生分配，
不是 `memory_report()` 的单一所有权账本。Strict 更高来自 full-coarse
`Yhat/X`、fine transfer 与首次 FGMRES arena；例如 `16x16x16x32` 的 Strict
runtime-owned 为 972,822,540 B，Legacy cache 为 231,215,104 B，其中 fused
FGMRES arena 单独为 286,261,248 B。

边界：Legacy 三个 solve 记录都显示 `n_vcycles=0`，所以这是当前实际 fine-only
入口的峰值，不是已执行 coarse V-cycle 的算法级公平 A/B；`24x24x24x64` 没有
Strict solve JSON，不能比较。完整证据见
[`docs/legacy_vs_strict_mg_memory_20261023.md`](../../../docs/legacy_vs_strict_mg_memory_20261023.md)。

## Cache 完整性

当前 legacy simple cache 的契约是四个 dataset 必须存在；物理身份依赖构建报告、gauge
SHA256、目录隔离和调用方约定。

Strict cache schema v2 内建：

- 不可变原子发布；
- identity、metadata、stats、manifest；
- 每个 logical tensor 的流式 SHA256；
- dataset attrs、shape、dtype、nbytes 和内容摘要交叉验证；
- 完整验证后再执行 device transfer；
- 损坏或不匹配时返回明确 miss/fail-closed。

Strict cache hit 会读取一次内容做校验、再读取一次做传输，不能把这段验证 I/O 混入纯 solve
时间。

### 2026-10-29 跨后端复用补充

Legacy 33-tensor payload 已在同一 gauge 上完成 CUDA -> KUPL 直接复用。此前观察到的
约 8.9% Galerkin 失配和 L8 `hnn` 差异，根因不是 KUPL `[pm,direction]` 槽序，而是
CUDA raw lifecycle 在同一个 `applyInitQcu` slot 内修改 parity 后未同步 native/device
params，导致 even/odd Clover 都按 parity 0 生成。CUDA `sync_parity()` 修复后，
同 gauge、同 `lonv` 的 KUPL-vs-CUDA 误差为：

```text
hnn  2.480209e-7
hdg  7.340791e-8
sit  2.524953e-7
```

因此不得对 Legacy cache payload 增加后端私有转置或重排来“修复”该历史现象。跨后端
命中仍必须验证 manifest、物理 identity、shape/dtype 和逐 tensor SHA。

### 多 MPI cache 构建

`pyqcu.testing.simple.mpi.build_full_mg_cache` 是规范入口。每个 rank 生成自己的
null-vector，并只计算自己的 coarse slab；周期 overlap patch 只承担 halo。`lonv` 的
coarse 轴是 `(2,4,6,8)`，时间 stride 是 `2*mg_grid[t]`。各 rank 通过 HDF5 MPI-IO
写 owned slab，并直接从暂存 HDF5 读取周期 patch；多 GiB tensor 不得经 root
`gather/scatter`。最终 HDF5 由所有 rank 回读各自 hyperslab 后原子发布。

CUDA patch LatticeSet 在 serial geometry 下持有 private singleton communicator，
Clover/coarse halo 使用 `set_ptr->comm`，因此多 rank patch stencil 在各自 GPU 执行，
报告要求 `stencil_device=cuda:*`。`8x8x8x16` 4-rank E2 实测总 setup `3.040447 s`，
其中 `lonv` MPI-IO `0.354144 s`、GPU stencil `0.137035 s`、其余三张量
MPI-IO `0.884016 s`，4/4 slab 验签通过；同一 cache
由 KUPL 直接加载得到 `verify_status=0`、独立真残差 `4.110297e-7`。

构建耗时先看线程，再怀疑算法：`OMP_NUM_THREADS` 被承认，`KUPL_OMP_THREADS` 覆盖
它，未设时钳到 affinity 集（`cpp/kupl/qcu/src/lattice_cpu.cpp:419-431`
`configured_threads`）。同一 `[16,16,16,32]` 单进程构建，单线程跑满 `29:33`
（91% CPU、0 个 slab 发布），16-32 线程把 null-vector 阶段压到 30-60 s
（`logs/v20261123.txt` §176）。

`--max-iter` 会漏进分布式 null-vector 求解：`kupl_multigrid.py:558-561` 的
`applyMultigridBuildNullVectorsQcu` 复用 driver 的 `params`（`params[13]` 就是
`--max-iter`），而串行路径 `:1002` 硬编码 `params[13] = 1000`。因此 L48 配方里的
`--max-iter 50` 给多 rank builder 少 20 倍 Krylov 预算；修好只是把失败变成慢 20 倍
的失败，不是让它通过（§176）。

### 两个已知缺陷（2026-11-23 记录，勿再撞）

两个都不是新改动引入，各自值得单独开工单（`logs/v20261123.txt` §171）：

1. `--lat 4 4 4 4 --mg-grid 2 2 2 2`（粗格 `[4,4,4,4]`）在
   `applyMultigridBuildStencilLocalQcu` 内以 glibc `corrupted size vs. prev_size`
   直接 abort——是堆损坏，不是可捕获错误。最小可建格点是 `[8,8,8,16]`；geometry
   闸门应拒绝 `[4,4,4,4]`，而不是让它踩坏堆。
2. cache 文件 `X.h5` 的 sidecar manifest 是 `X.json`，与 `--report X.json` 撞名：
   一次探针里运行报告静默覆盖了 cache manifest。报告路径不得允许等于 cache
   manifest 路径。

## 生命周期与 ABI

33-path 使用 legacy coarse slots：

```text
[lonv, hop_nn, hop_diag, sitting]
```

Strict 每 transition 使用：

```text
[V, raw Y(optional), Yhat, (X,X^-1)]
```

Strict 在私有 `LatticeSet` slot 中完成 init、VCycle/FGMRES 和 end，整个生命周期保持
`_SET_INDEX_` 不变。legacy 继续遵循普通事务式生命周期。不可把一个路径的槽序直接交给另一
路径。

## 选择建议

- 需要 compact-Schur、一次 wide kernel 表达完整两跳粗作用：使用 33-tensor cache。
- 需要 QUDA-style full-coarse、strict FGMRES 和更强 cache 完整性：使用 Strict cache。
- 做 QUDA 公平对照：Strict cache 是匹配的层级表示；不要把 33-tensor cache 当成 QUDA
  的 `vec_load`。
- cache-hit 性能应分别报告验证/加载、setup、首次 workspace 分配、steady solve 和真残差。

## 反模式

- 将 `lonv/hnn/hdg/sit` 改名或重排后冒充 strict `V/Yhat/onsite`。
- 对 Strict coarse asset 再做 checkerboard，或把 compact odd geometry 传给 strict。
- 把 `_SET_PTRS_STRICT_*` 槽序用于 legacy 33-tensor kernel，或反向使用。
- 只报 cache-hit solve 而隐藏 33 setup 或 Strict schema-v2 验签/加载开销。
- 在没有 gauge/null-vector/physics identity 的情况下跨配置复用 cache。

## 验证

```bash
python -m pytest -q pyqcu/testing/qcu/strict/quda_comparison/test_strict_runtime_cache.py
python -m pytest -q pyqcu/testing/simple/test_coarse_cache.py
python -m pytest -q pyqcu/testing/python/test_multigrid.py \
  -k 'qcu_stencil_pack_matches_all_33_periodic_slots or qcu_stencil_degenerate_extent_and_strict_support_guard'
```

2026-10-21 基线：分别为 `23 passed`、`2 passed`、`2 passed, 32 deselected`。
