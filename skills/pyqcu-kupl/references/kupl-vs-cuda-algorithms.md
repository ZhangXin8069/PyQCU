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
  每 rank 物理配额 = 19 CPU。
- `TH_max = min(floor(NUMA_width / ranks_per_numa), floor(usable_cpus / ranks))`。
  32 rank → 18（cgroup 约束）；ppr=1、16 rank → 38。
- 守卫只在 `threads > 本 rank cpuset.size()` 时抛错；cpuset 是整个 NUMA
  （38 CPU），所以 `TH=20..38` **不会报错**，而是让两个 rank 的线程按
  `cpus[thread % 38]` 重叠绑定。必须用 `KUPL_TOPOLOGY_REPORT=1` 核对
  `threads == bound_threads` 且 cpuset 不重叠。
- 固定 L24：t 方向最多 32 rank（`local[3]` 必须为偶数）；再拆分必须切
  空间维，同时失去 t2 lane 与 split-boundary，并引入 gauge corner halo。


## 4. 跨后端比较的最小口径

必须同时固定并记录：gauge 生成方式与身份、源类型与坐标、`atol`、`kappa/csw`、
dtype、rank/grid/thread、`half_sites`、`levels/restart`、是否 warm start、
停机口径（递推 vs 真残差）、是否启用 `KUPL_SKIP_FINAL_RESIDUAL`。

## 5. 已验证的等价性

- Clover even/odd：跨后端约 `4.9e-8`；Legacy `hnn/hdg/sit` < `3e-7`。
- Strict public wrapper vs raw ctypes ABI：L32 相对差约 `8.3e-8`。
- 这些只证明数值等价，不证明性能或迭代数可比。
