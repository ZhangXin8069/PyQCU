# KUPL Legacy-MG 两节点性能断崖排查（2026-11-23）

> 证据等级：C（远端作业 + 源码追踪）。本文只记录**已测到的**事实与**待验证**假设，
> 不把未复现的机制写成根因。

## 1. 已测事实（L48 c64，两节点，`pyqcu.test` native `kupl/native/mg`）

| 配置 | ranks / grid | threads/rank | fine local | solve (`row.seconds`) | case | 层级迭代 |
|---|---:|---:|---|---:|---:|---|
| 16r | 16 / `2 2 2 2` | 36 | `24×24×24×48` | **2.9444 s** | 147.46 s | 无（旧构建） |
| 32r | 32 / `2 2 2 4` | 18 | `24×24×24×24` | **102.4651 s** | 342.07 s | `level0=18 level1=4 total=18 accepted=4 fallback=0` |
| 64r | 64 / `2 2 4 4` | 18 | `24×24×12×24` | **92.8595 s** | 396.17 s | 同上 |

- 三个配置的 `case`（含输入读取/构建）差异远小于 solve 差异，说明断崖在 **solve 内部**。
- 32r/64r 的层级迭代与 coarse correction 接受数**完全相同**（18/4/4，无 fallback）；
  因此断崖**不是**迭代数增加，也**不是** fine-BiCG 回退。
- 32r 与 16r 使用了**相同节点数**与**相同总核数**（每节点 8×36 = 16×18 = 288 核），
  所以差异来自 **rank 数**，不是核数或节点数。
- `MG_LEVEL_ITERS` 由 `cpp/kupl/qcu/src/lattice_clover_multigrid.cpp` 的
  `bicgstab_chunk` 累加、rank0 输出；每一层都统计实际 BiCGStab 迭代。

## 2. 源码层候选机制（按可能性排序）

1. **粗层 halo 每次 apply 都重发、且按最大面尺寸发送全部 32 个方向**
   （`cpp/kupl/qcu/src/lattice_multigrid.cpp` 的 `ensure_coarse_halo` /
   `exchange_coarse_halo` / `pack_coarse_direction`）：
   每次 coarse apply 是 `2×32 = 64` 条消息、`E × max_face` 字节/条。对角方向实际
   只需 `1/12` 数据却按整面发送。32r 每次 coarse apply 约 10.1 MB 上线，
   每次 solve 约 19 次 coarse apply。
2. **MG 的全局归约走 `MPI_COMM_WORLD`**（`mg_dot` / `mg_norm2` / `mg_dot_pair`），
   而不是 `geometry_.comm`；每个 BiCG 迭代 2 次归约，fine+coarse 合计上百次。
   归约延迟若异常（见 §3），这部分会按 rank 数放大。
3. **每次 `time_samples` 的 `solve_once` 都包含 `applyInitQcu` 的 NUMA/拓扑初始化**
   （`bind_logical_numa` 内的多次 `MPI_Allreduce`/`Allgather`）与
   `verifyCloverMultigridQcu` 的完整校验（又一次 MG 资产构造与 6 次 apply）。
   这部分对 16r/32r 都同样存在，单独不足以解释 35×，但会放大 fabric 抖动。

三个候选**都还没有 profile 确认**：`data/l48_prof16`、`data/l48_prof32`、
`data/l48_mg2_profile32`（`PYQCU_KUPL_MG_PROFILE=1`）三次 profile 作业都落在故障
窗口内、全部 `rc=255`，没有一条有效 `MG_PROFILE` 行。站点已恢复，必须先重跑这三条
再决定改哪个候选。

## 3. 本轮 fabric 状态（关键约束，2026-11-23 收尾）

站点故障是**真实且间歇**的，不是一直全挂：20:46–21:47 之间，跨节点与跨 NUMA 的
MPI 启动在**三个不同节点对**上都以 `rc=137`（超时）失败，而单节点 2 rank 一直
可用。22:18 起逐步转好，**22:54 恢复**：作业 `2236344`（`cn[12044,12048]`）一次
通过四级能力阶梯，全部 `rc=0`。

| 阶梯 | 配置 | `2236344` 结果 |
|---|---|---|
| L1 ×3 | 单节点 2 rank，同 NUMA | `rc=0, 0, 0` |
| L2 | 2 节点，每节点 1 rank | `rc=0` |
| L3 | 2 节点，每节点 2 rank（`0-17` / `38-55`） | `rc=0`，smoke `size=4` |
| L4 | 2 节点，每节点 8 rank（共 16 rank） | `rc=0`，smoke `size=16`，`{cn12044: 8, cn12048: 8}` |

故障期最硬的证据是 PyQCU 自己的作业，而不是探针：`2230693` / `2230694` /
`2230474` 都在 PyQCU 的集体操作里报 UCX `ud_ep.c:287 unhandled timeout`
（`rc=255`）；同一窗口 2 rank Allreduce 正常。

### 3.1 更正：早先所有 L4 `rc=255` 都不是 fabric 失败

21:04 之后每一次 L4 `rc=255` 都是**请求超配**，不是网络：

- 探针请求 `drun -rpn 8 -R core=18` → 8 rank/节点 × 18 core = **144 core/节点**；
- 实际分配是 `core=36` × 2 replica = **72 core/节点**（cpuset `0-36,38-72`）；
- `drun` 直接拒绝该请求，MPI 从未启动；改成 `core=9`（8 × 9 = 72）后即 `rc=0`。

**教训：负向 gate 结果在归因给网络之前，必须先做资源配置自检**——
`ranks_per_node × cores_per_rank` 不得超过授予的每节点核数；granted cpuset 见
`/proc/self/status` 的 `Cpus_allowed_list` 与 `CCS_ALLOC_FILE`。

同一类错误还有更早的探针 bug：`sh/mpi_probe.py` 的 `Sendrecv` 用同一个 peer
同时当 dest 和 source，`size >= 3` 结构性死锁（只有 `size == 2` 恰好成立）；
正确写法是 `dest=(rank+1)%size`、`source=(rank-1)%size`。所以“2 rank 正常、
≥4 rank 挂起” 这类由该探针得出的 fabric 判读一律作废。

结论：故障期不能做 A/B，也不能把任何新数字当作断崖修复证据；阶梯恢复后按 §4
拿到了 profile 分项并完成 A/B。

## 4. 结论（2026-11-23 23:20 已确证）：断崖不存在，差异在启动环境

`PYQCU_KUPL_MG_PROFILE=1` 的分项已拿到，并有同形状 A/B 对照（唯一变量是
NUMA/hugepage 绑定）：

| 作业 | 绑定 | `total` | `fine_apply` | 每站点 |
|---|---|---:|---:|---:|
| `2236385` 16r | 无 | 186.08 s | 93.46 s | 2.93 µs |
| `2236485` 16r | 有 | **2.06 s** | **1.52 s** | **0.048 µs** |
| `2236451` 16r | 无（对照） | 196.00 s | 101.06 s | 3.04 µs |
| `2236383` 32r | 无 | 102.52 s | 49.92 s | 3.13 µs |
| `2236599` 32r | 有 | **3.04 s** | — | — |

- 两个未绑定运行（16r 与 32r）**每站点代价相同**（2.93 vs 3.13 µs），
  18 次 fine 迭代、4 次 coarse correction 完全一致。**没有 rank 数断崖。**
- 绑定后同一形状从 196.00 s 降到 2.06 s，**95x**；这正是历史基线
  `2.9444 s` 的量级。32r 复测同样从 102.47 s 降到 **3.04 s**（34x），
  且 `residual_verified=true`、`relative_residual=7.942e-06`。
- 修正后的两点比较是 **16r 2.06 s / 32r 3.04 s**：32 rank 慢 1.5x，这是
  「本地体积减半、rank 数加倍」的正常代价，不是断崖。
- 历史 16r 基线来自 `sh/l48_2n16r_kupl_mg2.sh`，它导出
  `KUPL_NUMA_BIND=1 KUPL_MEMORY_NODE=local+16 KUPL_HUGEPAGE=1`；
  而所有 32r/64r 与 `pyqcu.test` 正式作业来自
  `pyqcu/testing/kupl/submit_pyqcu_test.sh`，此前**不导出这三个变量**。
  断崖 = 两个作业族的启动环境差异。

机制：每个 rank 的 fine 算子要流式读取 gauge slab 与四个 Clover 项（12×12 复数），
没有 `MADV_HUGEPAGE` 时页表足迹巨大，没有 NUMA 绑定则 OpenMP 线程与
first-touch 页跨 NUMA 散布，两者相乘，在内存流式内核上就是 30–95x。

### 4.1 已修复（改启动器，不改求解器）

- `pyqcu/testing/kupl/submit_pyqcu_test.sh` 默认导出
  `KUPL_NUMA_BIND=1 KUPL_MEMORY_NODE=local+16 KUPL_HUGEPAGE=1`，
  并新增 `--no-numa-bind` 作为显式无绑定对照。
- `pyqcu/test/native.py` 在每个报告里记录 `binding_environment`
  （三个变量），使绑定运行与未绑定运行不可能再被当成等价比较。
- 求解器代码**未改动**：§2 的三个候选（粗层 halo、`MPI_COMM_WORLD` 归约、
  每次采样含 init/verify）都被这组数据排除，不应据此改内核。

### 4.2 硬性约定：真实鲲鹏环境必须 NUMA 绑定

绑定不是可选优化，而是测量配置的一部分。因此 `pyqcu/test/native.py`
在 **KUPL 且 `mpi_size > 1`** 时直接拒绝未设置 `KUPL_NUMA_BIND` 的运行，
错误信息给出 196.00 s vs 2.06 s 的依据与修复方法。单 rank 与非 KUPL 后端
不受影响（tier-0 门仍可在普通硬件上跑）。

唯一例外是刻意的绑定/不绑定 A/B：`PYQCU_ALLOW_UNBOUND_KUPL=1`，且此时
行内会写入「比绑定运行慢约 95x、不可比较」的说明。

### 4.3 后续

1. 所有历史 32r/64r 计时都是未绑定测量，**引用前必须用修复后的启动器重测**。
2. L64 矩阵仍被输入缺陷阻塞（见 `logs/v20261123.txt` §48：L64 gauge 实际是
   complex128、source 是 complex64，与文件名和 manifest 不符）。
