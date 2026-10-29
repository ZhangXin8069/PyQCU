# AGENTS.md — cpp/cuda/qcu

PyQCU 主 C++ CUDA 后端。手写 CUDA 内核 + MPI halo 交换：Wilson/Clover Dirac 算子、BiStabCG/CG 求解器、multigrid、规范场生成。

生产执行拓扑固定为一个 Python 进程 = 一个 MPI rank = 一张 GPU/DCU/GPGPU。
C++ halo 与全局归约负责跨 rank 协同；单进程多线程多 GPU 仅限 Python 层显式
调用的测试驱动 `pyqcu.cuda._multi_gpu.MultiGpuMultigrid`，不是后端默认语义。

## 构建

```bash
source ./env.sh       # CUDA toolkit 路径、MPI 等
bash ./make-nv.sh     # symlink CMakeLists-nv.txt → CMakeLists.txt, 然后 cmake + make
```

产物：`libqcu.so` — 由 Cython 桥加载的动态库。

DCU/DTK 构建从仓库根使用 `bash install.sh dcu`，或先运行
`bash cpp/cuda/qcu/make-dcu.sh`。`make-dcu.sh` 要求显式可定位的 DTK
nvcc-compatible compiler，使用 `CMakeLists-dcu.txt`，不得调用 NVIDIA 专用架构探测；
`--check` 只验证编译器选路，不启动 CMake。

## 源码组织

- `include/` — 26 个模板头（CUDA 内核内联）；`define.h` 必须镜像 `pyqcu/cuda/define.py`
- `src/` — .cu 文件：#include 头 + 模板实例化 + 内核启动封装
- `python/pyqcu.h` — C API 声明（extern "C"），必须与 `pyqcu/cuda/qcu/qcu.pxd` 精确同步

## 参数协议

- `params` (int32[58])：格点维度、网格大小、数据类型、迭代次数、计划选择、MG 层级配置、`_MG_USE_INIT_GUESS_`(57, x0 热启动)
- `_MG_USE_GCR_`(54) 保持 0/1 兼容，同时按位支持 `_MG_MODE_GCR_`(1) 与
  `_MG_MODE_MR_SMOOTHER_`(2)；后者启用 QUDA 风格 MR 粗层平滑，不扩展协议长度。
- `argv` (float[8])：mass、atol、sigma、四级 MG 容差、csw
- `set_ptrs` (int64[100])：scratch 缓冲指针

`_SET_PLAN_` (params[16])：-2 Laplacian、-1 Gauss gauge、0 Wilson dslash、1 BiStabCG/CG、2 Clover dslash。

Clover onsite 系数为 `-kappa*csw/8`，其中 `kappa=1/(2*mass+8)`；
`_CSW_` 固定为 argv 槽 7。默认 `csw=1` 保持旧数值，但 `csw=0` 必须被保留为
合法的纯 Wilson 配置，不能用 0 表示“未设置”。

## Clover Multigrid 5-stream 架构

```
main (strm):   dslash 操作 (fine/coarse_dslash_op)
_a_:           dot(r_tilde,r) → give_1beta → give_p → give_s → give_r
_b_:           give_1rho_prev → give_x_o
_c_:           dot(t,s), 收敛检查 dot(r,r)
_d_:           dot(r_tilde,v) → give_1alpha → dot(t,t) → give_1omega
```

## 关键不变量（来自 bug 修复）

1. 标量只存在于 `device_vals` — 迭代循环内禁止 host→device 标量 memcpy
2. 每轮迭代底部全流同步 — 同步全部 5 个流
3. 点积用 `_send_tmp_` scratch — cublasDot → scratch slot 7 → MPI_Allreduce → 复制到目标（绝不直接写目标）
4. `mpi_real_type<T>()` 模板 — 按模板类型分发 `MPI_FLOAT`/`MPI_DOUBLE`
5. `run_mpi` 用阻塞 `MPI_Sendrecv` — 无需 `MPI_Wait`（仅 `run_mpi_non_block` 需要）

## 分布式 strict MultiGrid（2026-09-18）

`src/apply_multigrid_strict.cu` 现为真正的多 rank 实现：

1. fine 层默认走 `LatticeWilsonDslash::run_mpi` 的 face halo（`PYQCU_STRICT_GLOBAL_DSLASH=1` 可切回 allgather 诊断路径）。
2. 粗层用 `StrictVectorHalo`（任意分量数的 face）与 `StrictAxisLinkHalo`（只交换 $\mu$ 方向 backward link face）。向量 slot 以 **最大 face** 为 stride；link 数组按全格坐标索引，packing 不得使用 compact 解码。
3. backward 修正 kernel 的符号必须与 base kernel 一致：`out = H in` 用 `+correction`，`out = base - H in` 用 `-correction`（由 `base == nullptr` 判定）。
4. 外层 FGMRES 的 `dot`/`dot_pair`/`dot_many` 全部做全局归约；`dot_many` 归约后必须把全局系数写回 device（正交化 kernel 消费 device 缓冲）。
5. coarsest cooperative fused kernel 无 MPI 集合操作，多 rank 时必须禁用并回退到带全局归约的 host BiCGStab。
6. 多 rank 下 `strict_validate_mpi_backend()` 检查 `params` 的 GRID/NODE 与 `MPI_COMM_WORLD` 一致；几何不满足（层间 grid 不一致、aggregate 跨 rank、local extent 非偶数）时 fail-closed。

## Block Size

`_BLOCK_SIZE_`（define.h）：小格点测试用 8/16，NVIDIA 生产 128，AMD DCU 生产 256。

## 2026-10-21 legacy Clover BiCGStab 成对归约

单 rank sync-light Clover BiCGStab 的 `t^H s` 与 `t^H t` 现在由
`bistabcg_dot_pair_complete` 一次完成：每个 block 同时累加两个点积，最后一个
block 按固定 partial 顺序汇总并写回 `device_vals`。实现使用两级树形归约和单次
`cudaMemsetAsync` 清零的 completion counter，不使用非确定性浮点 atomicAdd；
scratch 仅为 2 KiB complex 数组加一个 counter。multi-rank 五流路径保持原样。

同热态 ABBA、真实 `24^3x64` c64、每轮 10 稳态样本：Clover BiCGStab
`0.161293 -> 0.157776 s`（2.18%，1.0223x），均 18 iterations，独立相对真残差
`6.6952e-7 -> 6.6950e-7`。上层二层 MG 后端中位均为 `0.1746 s`，生命周期
`0.417192 -> 0.414053 s`，20 iterations、真残差 `4.1247e-7`，即本改动主要服务
null-vector/BiCG 构建，不能宣称改变了 MG V-cycle 本身。strict tier0 20 项、
cuda-strict 11 项、fused-FGMRES 3 项及 2 项非平凡 Clover 测试均通过。

## 2026-10-21 Clover 6x6 结构门控

legacy Clover-MG 在每次 `LatticeCloverMultigrid::init` 时扫描本轮实际使用的
`clover_ee_inv` 与 `clover_oo` 的两个 off-block 区域。只有所有 off-block 元素严格
为零时才启用 6x6 专用 multiply；任意非零、NaN 或外部通用矩阵都会自动回退完整
12x12 内核。检测使用设备端 flag 与一次 D2H，不改变公共 ABI，也不假定
`make_clover` 的稀疏结构。

真实 `24^3x64` c64、cache-hit、两轮 ABBA × 10 steady：MG backend 中位
`0.175150 -> 0.147050 s`（16.04%，1.1911x），lifecycle
`0.414565 -> 0.386002 s`（6.89%），均 20 iterations、真残差 `4.1247345e-7`；
两库最终解逐元素完全相同。人为加入 `1e-20` 的 off-block 元素后，检测自动走通用
路径，3 次中位 `0.179510 s`，残差不变，证明回退真实生效。strict tier1
20+11+3 项全部通过。

## 2026-10-21 legacy Clover BiCGStab 结构门控

同一 `clover_schur_matrices_block_diagonal` 探测现在也用于 legacy Clover
BiCGStab 的完整 solve `init`，但不在单 matvec wrapper 中执行，避免 stencil
build 的 36 次调用重复扫描。物理矩阵选择 6x6 内核；任意 off-block 非零仍自动
回退完整 12x12。真实 `24^3x64` c64 ABBA、20 steady：
`0.159155 -> 0.132760 s`（16.58%，1.1988x），lifecycle
`0.197254 -> 0.169155 s`（14.25%），均 18 iterations、真残差 `6.6950e-7`。
两版 BiCG 最终解逐元素完全相同。当前 CUDA wall 已比 QUDA `0.137520 s` 快
约 3.59%，但每迭代 `7.376 ms` 对 `7.238 ms` 仍慢约 1.90%；优势来自 18 对
19 次外层迭代，不能改写成单次迭代更快。

融合 `r/x` 更新与 `r^H r` 归约的候选也已测试：首个版本因归约 block 数小于
site block 数导致漏更新，修正为 block-stride 遍历后，两轮中位仅比 Stage18
改善 `0.23%`，且第二样本回到 `0.1333 s`，低于噪声阈值，已完整撤销；不要以
该候选的第一轮 `0.1290 s` 单点作为保留依据。

## 2026-10-21 单 rank 收敛同步分块

legacy Clover BiCGStab 的单 rank main-stream 路径现在每 2 次迭代执行一次
`r^H r`、D2H 和 host synchronization。中间迭代仍完整执行全部 scalar 更新、
dslash 与 vector 更新，只省略不用于控制流的中间范数。真实 `24^3x64` c64、
两轮 A/B × 10 steady：`0.127732 -> 0.125572 s`（1.69%，1.0172x），均 18
iterations、真残差 `6.694990e-7`；最终解逐元素相同。QUDA 同热态
`0.134009 s`，CUDA wall 快 6.72%，每迭代 `6.976 ms` 对 `7.053 ms`，CUDA
快 1.09%。若某问题在第 2k+1 次才达到阈值，本策略最多多执行一次迭代；报告必须
使用实际执行次数，不能声称每次都与逐迭代检查完全相同。

## 2026-10-21 legacy MG fine 成对归约

legacy Clover-MG 的单 rank fine Schur fast path 将每轮末尾的 `<t,s>` 与 `<t,t>`
合并为一次 `bistabcg_dot_pair_complete`。该 kernel 同时累加两个点积，并由最后
完成的 block 按固定 partial 顺序汇总；completion counter 在每次 launch 前异步
清零，不使用非确定性浮点 atomicAdd。multi-rank 五流路径不变。

真实 `24^3x64` c64、cache-hit、ABBA 两轮各 10 steady、共 20 样本：MG backend
中位 `0.14125 -> 0.13700 s`（降低 `3.01%`），lifecycle wall
`0.372507 -> 0.368274 s`（降低 `1.14%`）；均 20 iterations 且 `pair_checks=20`。
真残差由归约顺序变化从 `4.1247345e-7` 变为 `4.1255681e-7`，均满足门禁。
strict tier1 20+11+3、直接 c64/c128 Clover 8 项、ABI、MG breakdown 与 c128
pair smoke 全部通过。不得把该收益外推到多 rank 或真实 coarse V-cycle。

### legacy MG fine 单点积 fast path

进一步将 fine Schur fast path 的 `rho=<r_tilde,r>`、`<r_tilde,v>` 和检查范数
统一到 `bistabcg_dot_complete`。该 kernel 与成对归约共用 block 数、partials 与
completion counter，仍由最后完成 block 按固定 partial 顺序汇总；cuBLAS 不再位于
此单 rank 快路径的关键迭代链中。

以 Stage21 成对归约库为直接基线、ABBA 两轮各 10 steady：MG backend
`0.13760 -> 0.13295 s`（降低 `3.38%`），lifecycle
`0.362613 -> 0.356707 s`（降低 `1.63%`）。相对 Stage20 原始基线
`0.14125/0.372507 s`，累计降低 `5.88%/4.24%`。均为 20 iterations、
`single_checks=45`、`pair_checks=20`，真残差 `4.1257200e-7`。strict tier1、
直接 Clover 8 项、ABI、c64 breakdown 与 c128 single/pair smoke 全部通过。
multi-rank 与 coarse V-cycle 仍不得引用此收益。

### 6x6 register specialization rejected

尝试把 runtime block-diagonal Clover multiply 改为 `true/false` 模板特化，并在
block-diagonal 分支顺序处理两个 6x6 子块。资源检查显示 c64 block-diagonal kernel
仍为 88 registers，未降低占用；三轮 4 组 ABBA 共 30 steady 仅改善 MG backend
`0.19%`，lifecycle 反而慢 `0.09%`，低于噪声阈值。该候选已完整撤销，恢复
runtime-gated 12-row row-streaming 实现。不要重复把“6 个累加器”当成寄存器收益，
kernel 的寄存器峰值仍由同一实例的通用路径和其余 site product 状态决定。

### E12 coarse V-cycle production screen

固定生产 LIME、c64、cache-hit、单 V100 的同一进程筛选：

| mode | lifecycle median | backend / residual |
|---|---:|---:|
| fine-only baseline | `0.365086 s` | `0.13245 s`, `4.125720e-7` |
| BiCG + one deflate V-cycle | `0.401633 s` | `0.17246 s`, `2.666153e-7` |
| FGMRES + MG, CG smoother | `1.160491 s` | `0.93003 s`, `7.181618e-7` |
| FGMRES + MG, MR smoother | `0.832060 s` | `0.60251 s`, `4.818365e-7` |
| FGMRES + MG, Chebyshev | `1.366174 s` | `1.12978 s`, `4.743437e-7` |
| CA-GCR + MG, MR | `1.911062 s` | `1.68461 s`, `8.093534e-7` |

deflate/MR/Chebyshev 均仍执行 20 fine iterations；单次 V-cycle 本身约 `40 ms`，
其中 coarse residual check 约 `30 ms`。当前 E12、单个 null vector 的粗空间不足
以抵消校正成本。E24 的 33-tensor cache 按现有精确形状约 `5.22 GB`，QUDA
nvec24 对应的 E48 约 `18.86 GB`，且当前 E12 local-stencil setup 已出现约
`29.75 GB` peak allocation；在 32 GiB V100 上不能直接扩大 E 后重测。下一步若继续
真实 MG，应先完成 compressed/matrix-free coarse operator 或降低 setup workspace，
而不是继续给 E12 V-cycle 做微调。

### Standalone Clover BiCGStab single-dot path

Stage22 的 `bistabcg_dot_complete` 现已同时用于 standalone Clover BiCGStab 单
rank fast path 的 `rho`、`rtv` 与每两次迭代检查范数；末尾 `(t,s)/(t,t)` 继续使用
成对归约。ABBA 两轮各 10 steady，以 Stage22 库为直接基线：
`0.123997709 -> 0.120461542 s`（降低 `2.85%`）；均 18 iterations，独立真残差
`6.694990e-7 -> 6.694679e-7`。相对 QUDA BiCGStab `0.134009 s`，CUDA wall
快 `10.11%`，每迭代 `6.692 ms` 对 `7.053 ms`，快 `5.12%`。
strict tier1、直接 c64/c128 Clover 8 项与 ABI 全绿。multi-rank 五流路径不变。

无 cache 冷构建补充 A/B：总 setup `356.179 -> 353.635 s`（降低 `0.71%`），但
null-space 段 `26.524 -> 26.677 s` 在噪声内，33-tensor stencil 仍占绝对主导。
因此只能报告 BiCG 主循环的 2.85% 收益，不能声称 null-vector 构建已显著提速；
冷构建下降也不能归因于单一 kernel。
