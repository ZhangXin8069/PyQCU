# KUPL 与 CUDA 的实现取向差异（鲲鹏 = 高带宽 / 低算力）

本文是 [`docs/kupl_vs_cuda_algorithms_20261114.md`](../../../docs/kupl_vs_cuda_algorithms_20261114.md)
的**技能速查版**。只记录“为什么这样写”和“什么时候不能照搬”，完整算法对照与
`文件:行号` 证据见正式文档。

## 1. 一条总原则

鲲鹏节点是 **带宽相对充足、单核算力低、同步延迟高**；GPU 是 **算力高、延迟隐藏强**。
因此同一个 Dirac 算子，CPU 侧优化目标是“少算 + 多复用 + 少同步”，GPU 侧是
“多线程 + 多流 + 让硬件隐藏延迟”。看到两侧实现不对称时，先按这条原则判断是不是
有意为之，再决定是否统一。

## 2. 不能照搬的差异（按影响排序）

| 主题 | KUPL（CPU） | CUDA | 照搬的后果 |
|---|---|---|---|
| dslash 并行轴 | site-lane SIMD，一次 8(c128)/16(c64) 个相邻 site；t2 或 z 为最内层 | thread-per-site | 在 CPU 上按 site 写标量循环会丢 2–4× 向量收益 |
| gamma 代数 | 先投影成 2×3，再两次 3×3 复乘（18 MAC） | 每方向内联手写展开 | CPU 上不预投影会多一倍复数乘 |
| 规范场存储 | 9 复数（不重建） | 6 复数 + 叉积重建 | 在 CPU 上重建会重复算 9 次乘加；在 GPU 上存 9 会多 50% 带宽 |
| 单节点 halo | 默认 `MPI_Win_allocate_shared` + 序号握手（`KUPL_SHM_HALO=0` 回退） | 无；始终 MPI（device-aware 或 pinned） | 在 CPU 上用 MPI 会在 32 rank 时多付约 0.6 ms/次的 wait |
| halo 打包 | 12 分量；`KUPL_HALO_PACK6` 只发 2 spin×3 color；`KUPL_HALO_DEDUP_PACK` 在 `ht==1` 复用 | send 核压 6 分量 | 在 CPU 上按 GPU 的 6 分量打包会增加解包算术，不划算 |
| 点积 | 3-real 融合 Allreduce（`norm+dot`、`(t,t)+(t,s)`） | device 端票据归约，单 rank 每 2 迭代检查 | CPU 上逐点积 Allreduce 会把归约次数翻倍 |
| 停机 | 双容差 + 循环后**强制真残差** | 递推残差；每 2 迭代检查；真残差由外层 | 在 CPU 上照搬会放宽精度；在 GPU 上照搬会多一次全算 |
| 初值 | 冷启动置零 | 冷启动随机（seed 23333） | 跨后端比较迭代数时口径不一致 |
| CG | 正规方程 `A A†`，相对 `0.01*atol` | 直接 Schur CG，绝对 `atol²` | 迭代数不可直接比较 |
| Legacy MG 菜单 | 只有两层 V-cycle；平滑 = 整 2 次 BiCGStab；阻尼搜索 {1,.5,.25,.125}；拒绝 GCR/CA-GCR/W-F-K/mixed | 全套（GCR、CA-GCR、BiCGStabL、W/F/K、mixed、deflate） | 在 CPU 上开额外 Krylov 通常是净负收益 |
| Strict halo | 阻塞 `MPI_Sendrecv` | 非阻塞 + event 解耦 | 在 CPU 上用 event 模型没有对应物；在 GPU 上改阻塞会丢重叠 |
| Gauss gauge | 确定性**正态**（mix64+Box-Muller） | `curand_uniform` **均匀** | 同 seed 不同场；跨后端先统一生成方式 |

## 3. 内核路径门限（调参前先查）

- `use_block_dslash`：`half_sites >= KUPL_BLOCK_DSLASH_MIN_HALF_SITES`（默认 65536），
  且未设 `KUPL_DISABLE_BLOCK_DSLASH`。**固定全局格点时，rank 数增加会让
  `half_sites` 跌破门限，block SIMD 自动关闭**——这是“多 rank 性能断崖”的
  候选机制之一。
- t2 lane：`grid=(1,1,1,N)`、`local[3]==2`、`KUPL_ENABLE_T2_LANES`、
  `KUPL_T_LOCAL_IN_PART1`。
- split-boundary：`grid[0..2]==1`、`!block_dslash`、`local[3]==2`
  （或 `local[3]==4` 且 `KUPL_ENABLE_HT2_TBOUNDARY_SPLIT`）。
- boundary lanes：`KUPL_ENABLE_BOUNDARY_LANES`；关掉在 L24 上曾慢 22%。
- 这些路径都改变**每 site 的算术/访存结构**；跨配置比较必须同时记录
  `half_sites`、`grid`、`local[3]` 与命中的内核。

## 3.1 线程上限与静默超订阅（2026-11-14 实测节点谱）

- 节点：592 可用 CPU、16 个 CPU NUMA、每 NUMA 38 CPU；`ppr:2:numa` 时
  调度器给每 rank 的实测 cpuset 是 **18 CPU**。
- 判据：`TH_max = min(floor(NUMA_width / ranks_per_numa), cpuset_size)`。
  32 rank → 18；ppr=1、16 rank → 38。
- `TH=19..36` 由库内守卫**失败关闭**
  （`KUPL_OMP_THREADS exceeds the NUMA-local CPU set`），不是静默超订阅。
- 只有脱离调度器 cpuset、手工整节点 `mpirun` 时 cpuset 才可能是 38，
  此时 `TH=20..38` 会按 `cpus[thread % 38]` 别名绑定；必须用
  `KUPL_TOPOLOGY_REPORT=1` 核对 `threads == bound_threads` 且 cpuset 不重叠。
- 固定 L24：t 方向最多 32 rank（`local[3]` 必须为偶数）；再拆分必须切
  空间维，同时失去 t2 lane 与 split-boundary，并引入 gauge corner halo。

## 3.2 SU(3) 重构：计算换带宽的实测（2026-11-14）

背景：CUDA 的规范场只存 6 个复数，第三行由
`conj(cross(row0,row1))` 重建；KUPL 存完整 9 个复数。这是“省带宽 vs 省算力”
的取舍，不能凭直觉移植。

微基准（`~/sh/su3_probe.c`，1M site × 8 dir、c64、单线程同条件、取 3 次最优）：

| 变体 | 时间 (s) | 有效带宽 | 相对 9 分量 |
|---|---:|---:|---:|
| 读 9 + 3×3 matvec | 0.1935 | 3.12 GB/s | 1.00 |
| 读 6 + 叉积 + 3×3 matvec | 0.3013 | 1.34 GB/s | **慢 1.56×** |
| 只读 9 | 0.0847 | 7.13 GB/s | 1.00 |
| 只读 6 + 叉积 | 0.1740 | 2.31 GB/s | **慢 2.05×** |

解释：

- 省：gauge 9→6 少 33% 字节；gauge 占 dslash 流量约 75%，总流量约 -25%。
- 费：每方向每 site 多 6 次复乘（叉积 3 个分量 × 2 乘 + 1 减），
  相对 18 次 MAC 约 +33% 算术。
- 在鲲鹏低单核算力上，叉积的算术代价压过带宽收益：**9 分量布局是有意选择，
  不是遗漏**。

结论与做法：

- **不要**把 CUDA 的 6 分量方案直接移植到 KUPL；除非先在生产内核
  （site-lane 或 t2 lane）上做 kernel 级 A/B，并以生产 bench/真残差门为准。
- 微基准是标量循环、未 lane 向量化，绝对值不可外推；探针的 16 线程未复现
  扩展（疑似 cgroup 配额），只采用同条件相对值。
- 若未来在更高算力/更低带宽的平台移植，重跑这个探针即可判定方向。


## 4. 跨后端比较的最小口径

必须同时固定并记录：gauge 生成方式与身份、源类型与坐标、`atol`、`kappa/csw`、
dtype、rank/grid/thread、`half_sites`、`levels/restart`、是否 warm start、
停机口径（递推 vs 真残差）、是否启用 `KUPL_SKIP_FINAL_RESIDUAL`。

## 5. 已验证的等价性

- Clover even/odd：跨后端约 `4.9e-8`；Legacy `hnn/hdg/sit` < `3e-7`。
- Strict public wrapper vs raw ctypes ABI：L32 相对差约 `8.3e-8`。
- 这些只证明数值等价，不证明性能或迭代数可比。
