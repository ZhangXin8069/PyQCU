# KUPL-QCU 与 CUDA-QCU 算法差异（2026-11-14）

> 证据等级：B 级（源码逐行核对 + 既有跨后端数值门），未新增远端数值实验。
> 本文只描述**算法与实现选择**，不重复两后端的构建、绑定与性能数字；性能证据见
> [`kupl_l24_perf_20261113.md`](kupl_l24_perf_20261113.md) 与
> [`kupl_mpi_scaling_20261114.md`](kupl_mpi_scaling_20261114.md)。
> 路径前缀：KUPL 为 `cpp/kupl/qcu/`，CUDA 为 `cpp/cuda/qcu/`；行号以本文写作时工作区为准。

## 0. 结论摘要

1. **数学内核等价，工程分解不同**。两者都求解 Wilson/Clover Dirac 算子的
   parity-Schur（MATPC）系统，Clover onsite 项都取
   `A = 1 − (κ·c_sw/8)·σ_{μν}F_{μν}`；差异集中在**数据布局、向量化轴、halo 协议、
   点积/停机策略、粗层算子装配和多级算法菜单**。
2. **KUPL 面向“高带宽、低单核算力”的鲲鹏 CPU**：把并行度放在 **site-lane SIMD**
   上（一次处理 8/16 个相邻 site），用 gamma 预投影把每方向 36 次复数乘降到
   18 次，用共享内存 halo 取代 MPI 延迟，用整格 512-site tile 预解码压低 TLB miss。
3. **CUDA 面向“高算力、强延迟隐藏”的 GPU**：一个线程一个 site，gamma 手写展开、
   规范链路只存 6 个复数并用叉积重建第三行；用 5 条流 + 内点/8 个 halo 核分工，
   把通信压到阻塞 `MPI_Sendrecv` 的主机段，同时用 device-aware/pinned 双路径传输。
4. **求解器层面**：KUPL 是“双容差 + 强制真残差复查”的保守 BiCGStab；CUDA 是
   “递推残差停机 + 每 2 迭代检查”的高吞吐 BiCGStab，两者的初值策略和
   CG 目标函数也不同（KUPL CG 解正规方程 `A A†`，CUDA CG 直接解 Schur 系统）。
5. **多级菜单**：KUPL 只保留 Legacy 两层 V-cycle（BiCGStab 平滑 + 阻尼搜索）与
   Strict full-field/MATPC + 右预条件 FGMRES，显式拒绝 GCR/CA-GCR/BiCGStabL/
   W-F-K/混合精度；CUDA 全部保留，另有 mixed restrict/prolong、融合粗层求解、
   deflate 初值与预条件子自适应关闭。

## 1. 硬件约束如何决定实现（先看物理）

| 维度 | 鲲鹏 920（LX2 计算节点） | GPU（V100/P100/DCU） | 直接后果 |
|---|---|---|---|
| 单核算力 | 低；每核每周期 2–4 个 FP64/FP32 FMA 级 | 数千 CUDA core，SIMT 高吞吐 | CPU 必须**减少每 site 的算术量**并靠 SIMD 摊薄；GPU 可以“多算少想” |
| 内存带宽 | 节点级约 600 GB/s（STREAM triad 探针），32 rank 聚合 | 单卡数百 GB/s～1 TB/s，但延迟高 | CPU 是带宽/延迟受限，GPU 是占用率/延迟隐藏受限 |
| 同步代价 | `MPI_Allreduce` 单次约 0.45 ms（32 rank，ACTIVE） | 同一 MPI 归约 + D2H/H2D，单次可达 ms 级 | CPU 尽量把多个归约**融合成一次**；GPU 尽量让归约与计算重叠 |
| 线程模型 | MPI rank + OpenMP 线程，NUMA 显式绑定 | 一线程一 site，数千线程 | CPU 用 site-lane 跨 site 向量化；GPU 用 thread-per-site |
| 缓存 | 每 NUMA 38 CPU 共享 L3，访存局部性关键 | L2 命中 + 合并访存 | CPU 需要 block/tile 复用 gauge 与 spinor；GPU 依赖 coalescing |

这解释了本文后面几乎所有“看起来奇怪”的差异：**不是算法不同，而是同一个算法在
两种 roofline 位置上的最优工程形态不同**。

## 2. 数据布局与并行映射

### 2.1 费米子场

| 项 | KUPL | CUDA |
|---|---|---|
| 分量布局 | **component-major**：`field[component*half_sites + site]`（`src/lattice_geometry.cpp:216-224`） | 每 site 连续 12 个 `LatticeComplex`（SoA per site） |
| parity 压缩 | 完整 4D 棋盘压缩 + **t 轴折半**：`t = 2*t2 + ((x+y+z)&1)^parity`（`include/kupl/lattice_geometry.h:76-84,144-146`） | 半 T 压缩：`host_params[_LAT_T_] /= 2`（`include/lattice_set.h:77`），用 `move_*_t` 的奇偶门控实现 t 邻居（`include/define.h:317-324`） |
| 外部索引 | `(parity*12+component)*half_sites + site` | 由 Cython 侧按 `[2,4,3,X,Y,Z,T/2]` 直接映射 |
| 向量化轴 | 连续 site（t2 或 z）共 `64/sizeof(Real)` 个 = 16(c64)/8(c128) | 线程线性 id → (x,y,z,t) |

两者**都**是 parity-compressed，因此 Schur/MATPC 的向量长度一致；区别在
“压缩后的最内层连续轴”是什么：KUPL 明确把 t2 当作最内层连续轴（见 §3.2），
CUDA 由线程索引自然展开。

### 2.2 规范场

| 项 | KUPL | CUDA |
|---|---|---|
| 存储 | `[parity][row][col][direction][x][y][z][t2]`，9 个复数（`src/lattice_geometry.cpp:226-239`） | 每方向 **6 个复数**；第三行由叉积重建（`include/define.h:325-333`） |
| 访问 | `local_gauge` 返回带 `matrix_stride = kDim*half_sites` 的 view | `get_u` 内联重建 3×3 |
| halo | face + **corner（2D 对角 rank）** 缓存（`src/lattice_clover_dslash.cpp:144-192`） | 10 类 link halo：4 轴向 + 6 对角（`src/clover_dslash_comm.cu:172-520`） |

**为什么 CPU 存 9 个而 GPU 存 6 个**：CPU 的 SU(3) 乘是标量循环，重建第三行要额外
9 次乘加，且在 site-lane 循环里会被重复执行；直接存 9 个复数只多 50% 带宽，
换来更少的算术。GPU 的线程内重建几乎免费（寄存器、无额外访存），因此省带宽更重要。

## 3. Wilson dslash

### 3.1 gamma 代数

- **KUPL**：先把 4-spinor 投影成 2×3 复向量（每个方向 2 个线性组合），做两次 3×3
  复乘，再按 `±1/±i` 重组 4 个分量。注释明确写“18 MAC 而不是 36 MAC”
  （`src/lattice_wilson_dslash.cpp:704-708`，实现 `:101-146`、`:274-397`）。
- **CUDA**：每个方向在 thread 内直接展开 4 个 Dirac 分量的 `multi_i/conj` 组合，
  无预投影阶段（`src/wilson_dslash.cu:57-109` 的 x 方向为模板）。

**差异根源**：CPU 的算术吞吐低，预投影是“用 2 次线性组合换掉一半复数乘”；
GPU 的算术吞吐高，多做的乘法与寄存器压力相比可忽略，反而省掉一遍中间数组。

### 3.2 内核分解

| 路径 | KUPL | CUDA |
|---|---|---|
| 单 rank | 标量 site 核（`src/lattice_wilson_dslash.cpp:2068-2110`） | **融合全量核** `wilson_dslash<T,ApplyClover>`（`src/wilson_dslash.cu:9-302`），1 次 launch |
| 内点 | `compute_interior_blocks`：`omp parallel for collapse(3)` + t2 方向 8/16-lane（`:492-552`） | `wilson_dslash_inside`：线程级 site 循环，跨边界 hop 用上界门控跳过（`src/wilson_dslash.cu:303-576`） |
| 边界 | 标量/`boundary_lanes` 两套（`:840-965`、`:1367-1531`） | 8 个方向 send/recv 核（`src/wilson_dslash.cu:577-1383`） |
| 周期面 | 仅 `grid[d]==1` 的维走 block 周期面（`:554-702`） | 由 `move_backward/forward` 的模回绕直接处理，无单独核 |
| 特化 | t2 lane（`:1532-1810`）、t2 z-block（`:1824-2066`）、split-boundary（`:2397-2423`） | 无；单 rank 融合核 + 多 rank 分段核 |
| 启动门限 | `half_sites >= KUPL_BLOCK_DSLASH_MIN_HALF_SITES`（默认 65536，`:36-51`） | 无（线程数随格点自动变化） |

**关键点**：KUPL 的 `t2 lane` 与 `split-boundary` 只在
`grid=(1,1,1,N)`、`local[3]==2`（或 `local[3]==4` 且显式开启）时启用
（`:64-73`、`:1414-1416`）。这意味着 **L24 在 32 rank（t=2）才命中专门路径**；
rank 数变化会同时改变“每 rank 计算量”和“命中的内核”，这是扩展性分析中必须
分离的两个变量（见 `kupl_mpi_scaling_20261114.md`）。

### 3.3 halo 协议

| 项 | KUPL | CUDA |
|---|---|---|
| 粒度为 face | 是（fermion 只有 face，无 corner；`include/kupl/lattice_set.h`） | 是（4 方向 send/recv 核） |
| 打包量 | 默认 12 分量；`KUPL_HALO_PACK6` 只发 2 spin×3 color（`:969-1039`） | send 核压 6 个 spinor 分量（`_LAT_HALF_SC_`，`src/wilson_dslash.cu:630-635`） |
| 去重 | `KUPL_HALO_DEDUP_PACK`：`ht==1` 时两面字节相同，一次打包两次发送（`:1187-1195,1242-1252`） | 无 |
| 单节点 | **默认共享内存窗口**：`MPI_Win_allocate_shared` + publish/consume 序号 + 2 buffer（`:1075-1173`） | 无；始终 MPI（device-aware 或 pinned staging） |
| 多节点 | `MPI_Irecv/Isend` + `MPI_Waitall`（`:1220-1270`） | 生产路径阻塞 `MPI_Sendrecv` 逐方向（`include/lattice_wilson_dslash.h:489-498`）；非阻塞版存在但被注释（`:691`） |
| 重叠 | split-boundary：先 post，再算 part1（含本地 t），wait 后算 part2（`:2397-2423`） | inside 核异步 + 主机在 `MPI_Sendrecv` 阻塞；5 条流（`include/lattice_set.h:380-382`） |
| 超时/失败 | 自旋 30 s 后抛错（`:20-34`） | 无显式超时；MPI 错误码由 `checkCudaErrors` 以外路径处理 |

**为什么 CPU 用共享内存 halo**：32 rank 单节点时，MPI 往返的固定延迟（实测
`wait≈0.78 ms/次`）占 dslash 墙钟约 40%；`MPI_Win_allocate_shared` 直接 `memcpy`
把 wait 降到 `0.16 ms/次`，与“高带宽”特性一致。GPU 的同类优化是 NVSHMEM
（代码中有 `multigrid_nvshmem_put_halo`，`src/multigrid.cu:1146`），但 Wilson
生产路径没有启用。

## 4. Clover 项

### 4.1 构造

| 项 | KUPL | CUDA |
|---|---|---|
| 执行位置 | host（OpenMP 并行 per site） | device kernel |
| 公式 | `coefficient=-0.125*κ*csw`，6 个平面 `σ_{μν}⊗F_{μν}` 累加，`A=I+...`（`src/lattice_clover_dslash.cpp:399-498`） | `clover *= -κ*csw*0.125`，对角 `+=1`（`src/clover_dslash_single.cu:845-854`） |
| 求逆 | **两个 6×6** 分块，带部分主元 Gauss-Jordan，奇异即 throw（`:500-654`） | **12×12** 就地 `[A|I]` Gauss-Jordan，**无主元**（`include/define.h:538-569`） |
| 6×6 结构 | 本 gamma 基下天然分块对角，显式清零非对角块（`:615-633`） | 运行时 GPU 扫描 `clover_ee_inv/clover_oo` 的 off-block 是否全零，一票决定是否用 6×6 快路径（`src/clover_dslash_single.cu:875-907`、`include/lattice_clover_dslash.h:730-752`） |
| parity | 只算 `target_parity()` 的半场（`src/backend.cpp:136-139`） | 同样按 parity 分半存储 |

### 4.2 应用与缓存

- **KUPL**：`apply_clover_onsite` 固定走两个 6×6 块，跳过 72 个结构零
  （`src/lattice_linalg.cpp:118-166`）。缓存有两层：
  1. gauge halo cache（face+corner，指针相同即命中，`src/lattice_clover_dslash.cpp:91-99`）；
  2. 目标 parity 的 inverse cache（`cached_gauge_[parity] == gauge`，`:656-678`）。
- **CUDA**：`block_diagonal_` 是**运行时探测结果**，而非编译期假设；若外部 Clover
  含跨块项则自动回退完整 12×12（`include/lattice_clover_dslash.h:730-752`）。
  CUDA 侧没有等价的“inverse cache”描述；每次 solve 由调用方传入
  `clover_ee/oo/_inv` 张量。

**为什么 CPU 敢固定 6×6**：KUPL 的 gamma 基与 onsite 构造保证 A 分块对角；
但代码仍保留通用入口 `supplied_clover_dslash` 走 12×12（`src/backend.cpp:157-169`）。
CUDA 的 `make_clover` 与外部矩阵共用入口，因此必须运行时探测。

## 5. 求解器

### 5.1 系统形式

两者都解奇偶 Schur（MATPC）：

```text
S = A_pp − κ² · D_pq · A_qq^{-1} · D_qp
```

- KUPL：`apply_schur` 严格按上式实现；t2 快路用
  `bare_dslash_t2_scaled(..., −κ²)` 合并第二步（`src/lattice_wilson_dslash.cpp:2476-2542`）。
- CUDA：`run_eo`（含 `A_ee^{-1}`）→ `run_oe` → `clover_dslash_oo.give_with_subtract(κ²)`
  （`include/lattice_clover_bistabcg.h:159-184`）；c64 单 rank 把 `A_ee^{-1}` 融进
  Wilson 核（`:167-174`），c128 不融合。

### 5.2 BiCGStab 细节

| 项 | KUPL | CUDA |
|---|---|---|
| shadow 残差 | 固定 `r̂ = r₀`（`src/lattice_solver.cpp:261-270`） | 固定 `r_tilde = r`（同标准形式） |
| 初值 | 冷启动置零；warm start 保留 `fermion_out`（`:598-603`） | 冷启动 **随机** `seed=23333`（`include/lattice_clover_bistabcg.h:256-258`）；warm start 保留 x0 |
| 停机 | 每轮 `global_norm_and_dot(r̂,r)`，递推容差 `max(tol,4ε)‖b‖`（`:242-245,280-285`） | 单 rank 每 **2** 迭代一次 `dot(r,r)`（`convergence_chunk=2`，`:511-547`）；多 rank 每轮但用**上一轮**的 `‖r‖²`（`:596` 先算 `r,r` 再更新 r） |
| 容差 | 相对：`stop=max(atol,floor)‖b‖`；Clover 的 c128 用 `0.1*atol`，c64 用 `KUPL_CLOVER_TOL_FACTOR`（`:717-727`） | 相对：`tolerance=max(atol,4ε)`，`stop²=tolerance²‖b‖²`（`:452-456`） |
| 真残差 | 循环后**必做**完整 12×12+dslash 复查，超容差 throw（`:740-766`） | 求解器内**不做**；MG 驱动另算 `_diff2`（`include/lattice_clover_bistabcg.h:380-399`） |
| 点积 | 3-real 融合：`norm+dot` 一次、`(t,t)+(t,s)` 一次（`:161-199,332-340`） | device 端 `bistabcg_dot_complete` / `dot_pair_complete`（atomicInc 票据 + 最后 block 汇总，`src/bistabcg.cu:199-326`）；多 rank 再 D2H+Allreduce+H2D（`include/lattice_clover_bistabcg.h:312-375`） |
| 强度削减 | 融合 `r=s−ωt` 与 `x+=αp+ωs`（`:346-357`） | `bistabcg_give_r_x` 单核双更新（`src/bistabcg.cu:164-183`） |
| 随机初值的代价 | 无（置零） | 需要一次 `dslash(r,x)` 得真实 r（`:260-268`） |

**注**：CUDA 多 rank 路径“用上一轮残差判停”是一个**已知的一迭代滞后**；单 rank
快路径则显式每 2 迭代检查一次（`:544-546`）。这两者在结果上都不影响最终真残差门
（由调用方/MG verify 单独做），但会影响迭代数统计的口径，比较时不可混用。

### 5.3 CG

| 项 | KUPL | CUDA |
|---|---|---|
| 目标系统 | **正规方程** `A A†`（Wilson）：`apply_wilson_cg_dslash = A†(A x)`（`src/lattice_wilson_dslash.cpp:2561-2584`） | 直接解 Schur 的 CG（`src/cg.cu` + `include/lattice_wilson_cg.h:243-327`） |
| 右端 | 先 `A†b`（`src/lattice_solver.cpp:658-659`） | 无 |
| 容差 | 相对 `0.01*atol`（`:665`） | **绝对** `atol²`（`include/lattice_wilson_cg.h:314`） |
| 点积 | 复用融合归约 | cuBLAS `cublasDotcEx`（`include/lattice_wilson_cg.h:266,290`） |

**为什么 KUPL 解正规方程**：CPU 侧 CG 只作为对照/兼容入口，正规方程条件数平方，
但实现简单且不需要额外的 Schur 反向 dslash；生产路径是 BiCGStab，不受此影响。
GPU 侧保留直接 Schur CG 以维持与 QUDA 的迭代口径一致。

## 6. Legacy MultiGrid

| 项 | KUPL | CUDA |
|---|---|---|
| 默认层数 | 2（`levels` 只支持到 2 的正式路径；`src/lattice_clover_multigrid.cpp:276-302`） | 最多 5 层（参数平铺） |
| 外层 | odd-preconditioned BiCGStab + 每 `num_restart` 步一次粗校正（`:220-223`） | 同样 BiCGStab + 粗校正；另有 GCR/FGMRES、CA-GCR、BiCGStabL（`include/lattice_clover_multigrid.h:4755-4769`） |
| 平滑器 | **整 2 次 BiCGStab 迭代**（`:933`、`:951`） | MR/Chebyshev 可选，默认 BiCGStab 系 |
| 最粗层 | BiCGStab 到收敛，不收敛 throw（`:911-931`） | 单核融合 BiCGStab（cooperative `grid.sync()`），体积阈值 262144 且多 rank 禁用（`include/lattice_clover_multigrid.h:3838-3847`） |
| 粗校正接受 | 阻尼 `{1,0.5,0.25,0.125}` 逐个试，取第一个使真残差下降者，否则回滚（`:984-1013`） | deflate 一次性初值 + 下降则接受（`:4795-4844`） |
| 粗算子 | 33 点 stencil（sitting + 8 近邻 + 24 对角）；宽版按 512-site tile 预解码 32 邻居（`src/lattice_multigrid.cpp:508-708`） | 同样 33 点；setup 用 batched local stencil（逐 coarse 中心、逐 probe 调完整 Wilson 核，`src/multigrid.cu:2576-2818`） |
| 混合精度 | **拒绝**（`:263-266`） | legacy 支持 float↔double restrict/prolong（`src/multigrid.cu:2107-2130`） |
| 其它模式 | 显式拒绝 GCR/MR/Chebyshev/W-F-K（`:198-201,1104-1107`） | 全部保留 |
| null vector | 逆迭代 + 5 次确定性重试 + CGS2（`src/lattice_solver.cpp:822-956`、`src/apply_multigrid.cpp:24-140`） | 随机 → `M M†` 平滑 → 归一化 → CGS2（`src/multigrid.cu:2447-2573`） |

**为什么 KUPL 砍掉一半菜单**：CPU 每次迭代的固定开销（归约、线程唤醒）高，
W/F/K、GCR、CA-GCR 的额外 dslash 调用在低算力平台上通常不划算；KUPL 的选择是
“少量更强的前提条件 + 严格真残差门”。CUDA 的算力足以把额外 Krylov 空间换成更少
的外层迭代，因此保留全套。

## 7. Strict MultiGrid

| 项 | KUPL | CUDA |
|---|---|---|
| coarse 几何 | full-field（`volume`），E 分量 | 同 |
| 细层 | `H_{qp} → A_q^{-1} → H_{pq} → A_p^{-1}`（`src/lattice_strict.cpp:35-74`） | 同（`strict_fine_matpc`，`src/apply_multigrid_strict.cu:2780-2823`） |
| R/P | 只扫 parity 匹配的 fine 点（`:1591-1665`） | 同（`strict_restrict_parity_kernel`，`:1499`） |
| V-cycle | 持久 hierarchy；平滑器是 **1 维最优步长 Richardson**（`:1392-1409`） | `mr_smooth`（`:4635-4644`） |
| FGMRES | 右预条件 = V-cycle + fine smooth；每 restart 重算真残差（`:1296-1306`） | 同语义；Arnoldi 批量内积 + Givens，估计残差早退（`:4182-4266`） |
| halo | 逐维**阻塞** `MPI_Sendrecv`（`:413-452`） | 非阻塞 Isend/Irecv + event 与计算流解耦（`:1797-1864`） |
| 归约 | `MPI_COMM_WORLD`（`:823`） | `MPI_COMM_WORLD`（`:2524-2565`） |
| trace | 无 | `PYQCU_STRICT_TRACE_FILE` 逐迭代 trace v3（`:31-150`） |
| 混合精度 | 拒绝（`:965-968`） | 拒绝（`:3195-3200`） |

**为什么 Strict 的 halo 两边都变“保守”**：Strict 的粗层向量短、消息多，
CPU 侧用阻塞 Sendrecv 避免多缓冲状态机；GPU 侧则用 event 把 pack 与计算解耦，
因为 GPU 的 kernel launch 与 stream 同步成本更高，必须显式重叠。

## 8. Laplacian 与 Gauss gauge

| 项 | KUPL | CUDA |
|---|---|---|
| Laplacian 几何 | 独立 3D：`local[3]=1`、`grid[3]=1`、gauge 无 parity 维（`src/lattice_geometry.cpp:47-50`、`src/lattice_laplacian.cpp:15-25`） | 同样 `_LAT_T_=1`、gauge 不分半（`src/laplacian.cu:4-6,51`） |
| Laplacian 算子 | `6ψ − Σ_± U ψ`，跨 rank 项延迟到 boundary pass（`:142-255`） | 同，收尾核 `6src − dest`（`src/laplacian.cu:707-721`） |
| halo | source face + gauge face 两类消息（`:66-140`） | 6 个 send/recv 核（`:161-660`） |
| Gauss gauge 随机数 | `mix64` + Box-Muller 的**确定性正态**，按全局坐标播种（`:20-43,158-171`） | `curand_uniform`（**均匀**，`src/gauss_gauge.cu:18-20`） |
| SU(3) 指数 | scaling & squaring（‖·‖∞≤0.25）+ 14 阶 Taylor（`:58-87`） | 20 阶 Taylor（`src/gauss_gauge.cu:76-99`） |
| 可复现性 | 任意 rank 分解下逐位可复现 | 依赖 curand 的 seed/offset，跨 rank 需固定调用顺序 |

**这是最容易被忽略的差异**：KUPL 的 Gauss gauge 用**正态**分布且按全局坐标播种，
CUDA 用**均匀**分布。两者生成的规范场分布不同；跨后端比较 gauge 依赖的物理量时，
必须重新生成或显式转换，不能假定“同 seed 同场”。

## 9. 精度与 fail-closed

| 项 | KUPL | CUDA |
|---|---|---|
| 支持精度 | c64/c128，模板实例化（`src/lattice_cpu.cpp:847-860`） | c64/c128（`include/define.h:58-67` 中其余类型未实现） |
| 混合精度 | 无；跨层不一致即 throw | legacy MG 有 mixed restrict/prolong；Strict 拒绝 |
| 半精度 | 无 | 无 |
| 数值失败 | 真残差超容差 / Krylov breakdown / 6×6 奇异 → throw | CUDA 错误检查 + 调用方 verify；求解器内不强制真残差 |
| 枚举保留 | GCR/MR/Chebyshev/W-F-K 等位显式拒绝 | 全部实现 |

## 10. 已有一致性证据（跨后端）

- Clover even/odd 项跨后端误差约 `4.9e-8`（`skills/pyqcu-cuda/SKILL.md:411-422`）。
- Legacy `hnn/hdg/sit` 跨后端误差低于 `3e-7`（同上）。
- Strict public wrapper 与 raw ctypes ABI 同输入 L32 解相对差约 `8.3e-8`
  （`skills/pyqcu-kupl/SKILL.md:79-80`）。
- 这些是**数值等价性**证据，不是性能等价性证据；两者的迭代数口径不同（§5.2），
  不可直接比较“谁迭代更少”。

## 11. 选型与调优建议

1. **鲲鹏 CPU 上优先 KUPL**：共享内存 halo + site-lane SIMD + 真残差门在当前节点
   是最稳的组合；`KUPL_SHM_HALO=0` 只用于跨节点回退或多节点对照。
2. **不要在 CPU 上照搬 CUDA 的 Krylov 菜单**：GCR/CA-GCR/W-F-K 的额外 dslash
   在低算力平台上通常净负收益；若必须尝试，先用 `verify_status=0` 与独立真残差
   做单变量 A/B。
3. **不要在 GPU 上照搬 KUPL 的保守停机**：GPU 的每迭代成本低，递推残差 + 定期
   真残差更划算；KUPL 的双容差是为了防止 CPU 上 fp32 递推漂移。
4. **跨后端比较必须固定**：gauge 生成方式（正态 vs 均匀）、源、atol、parity、
   rank/grid/thread、迭代口径（是否含真残差复查）。
5. **多级求解的默认路线不同**：KUPL 默认 `Legacy-MultiGrid`（`levels=2`），
   但当前生产最优是 `levels=1, restart=0` 的 parity-Schur BiCGStab；
   CUDA 的 Legacy/Strict 选择器见 `pyqcu-solver` 技能。

## 12. 待补证据

- `KUPL_HALO_PACK6` 与 CUDA 的 6-component send 打包在**同一 gauge**上的逐位对照
  （目前只有 KUPL 内部的 MPI/shm 两条路径逐位一致）。
- Strict 粗层在 2/4 rank 的 KUPL vs CUDA 迭代数与真残差同表对照。
- Gauss gauge 正态 vs 均匀的物理量差异量化（建议用 plaquette 分布与 topological
  charge 的样本统计，而不是单点值）。
