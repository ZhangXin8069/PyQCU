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

## 3. 本轮 fabric 状态（关键约束）

2026-11-23 当天，**连昨天已经 PASS 的 16r/2n 配置也无法复现**：

| 作业 | 配置 | 结果 |
|---|---|---|
| `2230693` | 16r / `2 2 2 2` / 36 threads / MG2 | UCX `ud_ep.c:287 unhandled timeout`，`rc=255` |
| `2230694` | 32r / `2 2 2 4` / 18 threads / MG1 | 同上 |
| `2230474` | 32r 单节点 / 18 threads / MG2 | 同上 |
| `2231089` | 纯 MPI 探针，2 rank，1 节点 | **`PROBE_OK`**：`ar8=10 µs`、`ar1024=31 µs`、`ar65536=136 µs`、`ring4096=45 µs` |
| `2231110` | 同探针，2→4→8→16→32 rank | 2 rank `PROBE_OK`；**4 rank 起挂住**，未返回 |

结论：**断点位于 4 rank 的纯 MPI 集体/点对点操作，早于任何 PyQCU 代码**。
2 rank 的 Allreduce 延迟（10–136 µs）完全正常，说明不是带宽或算法问题；4 rank
起直接挂起，是本轮站点 fabric/UCX 状态。在这种状态下无法用 A/B 区分 §2 的三个
候选，也不能把任何新数字当作断崖修复证据。

## 4. 下一步（必须按顺序）

1. 等 fabric 恢复后，重跑 16r/2n 与 32r/2n 的**同节点数**对照，确认 2.94 s 基线
   可复现；不可复现则断崖结论作废。
2. 用 `PYQCU_KUPL_MG_PROFILE=1`（`MG_PROFILE` 行）拿到
   `fine_apply/coarse_apply/restrict/prolong/coarse_correction/total` 分项。
   - `total − Σparts` 大 → 归约（候选 2）；
   - `coarse_apply` 大 → 粗层 halo（候选 1）；
   - `total` 正常但 `row.seconds` 大 → 初始化/verify（候选 3）。
3. 只有分项定位后才改代码；候选 1 的最小改动是**按方向真实面尺寸分配/发送**，
   候选 2 的最小改动是把 MG 归约换到 `geometry_.comm`。
