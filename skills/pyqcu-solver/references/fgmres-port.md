# KUPL Legacy-MG: the single FGMRES outer solver

The port, the two same-configuration three-arm measurements, the verdict, and the
list of what the port deleted. Split out of `SKILL.md` to keep the entry point under
the 500-line gate; the content is unchanged.


2026-11-25 **整段删除**了 Legacy-MG 的 BiCGStab 修正器，换成 CUDA 的结构。原因是
CUDA-vs-KUPL 对比在两边跑不同方法时，任何差异都无法归因到硬件：

- **外层**：右预条件 FGMRES(m = 10)。`z_j = M⁻¹v_j`、`w = A z_j`、Arnoldi、
  `x += Z y`；restart 时按**真残差**重算。
- **`M⁻¹`**：`nu_pre = 4` 步 MR 平滑（作用在进来的向量上）→ restrict → 一次
  coarse cycle → prolong → 权重 1 加回；**没有 post-smoothing**（CUDA 也不用，
  剩余部分由外层 Krylov 吸收）。
- **粗层**：`cycle()` 就是 CUDA 的 V-cycle 体（2 步 BiCGStab 前平滑、restrict、
  递归、prolong、2 步后平滑）；**两级时入口层就是最粗层，该体不运行**——这是深度
  的属性，两后端相同。
- **门**：6 步窗口的残差比几何平均；连续两个窗口 ≥ 0.9 就把预条件子关到本次 solve
  结束，复现 CUDA 的 adaptive preconditioner gate。
- `_MG_USE_DEFLATE_ == 0` 或 `--mg-levels 1` 表示**没有粗空间**，直接走
  `set_.clover_bistabcg`；`restart` 已不是"修正间隔"，只剩 `simple/mpi` launcher
  用它（`levels>1 and restart!=0`）决定是否启用粗空间。

删除项（再当成现行机制写、照它复跑都是错的）：BiCGStab 修正器循环、修正间隔调度与
`KUPL_MG_CORRECTION_INTERVAL`、平滑前缀与 `KUPL_MG_NU_PRE`、解析最优阻尼
`t = <r,Ac>/‖Ac‖²` 作为活机制、`diagnose_correction` 与其 `MG_CORRECTION_DIAG` 行
（`best_damping`/`best_ratio`/`overlap`/`capture_bound` 等打印）、"无改善就关掉修正
回落细层 BiCGStab"、`KUPL_MG_OUTER`，以及 `MG_LEVEL_ITERS` 的 `fallback_fine=`
token。`MG_LEVEL_ITERS` 仍打印 `level0/level1/...`、`total=` 与
`accepted_corrections=`，后者现在计**预条件应用次数**。

代价只有删除：`cpp/kupl/qcu/src/lattice_clover_multigrid.cpp` 在 git 里最大的历史版本是
**1910 行**，删除旧修正器后 **1701 行**（`628231ee`），加入 FGMRES 外层与守门后 HEAD **1834 行**；
库已在计算节点重编译通过、符号中含 `fgmres`。**FGMRES 的三臂数字已有**（下节表格，作业 `2260302`
为修正前、`2260654` 为修正后），旧路径的 A/B 按设计不再存在。完整记录见
`logs/v20261126.txt`。

**两次同配置三臂测量（8 rank、`grid 2 2 2 1`、`mass=+0.1`、`atol 1e-5`、
`--repeat 3 --warmup 1`；全部 `PASS` 且 `residual_verified=true`）：**

| 作业 | 位置 | `level0`（base/mg-2/mg-3） | 秒 | 相对残差 |
|---|---|---|---|---|
| **2260302** | **修正前**（Givens 非酉、门控 6/6） | 28 / 14 / 13 | 1.2008 / 1.6921 / 2.6399 | 4.41e-06 / 3.83e-06 / 4.68e-06 |
| **2260654** | **修正后**（`e3d561f7`） | — | 1.2020 / 1.6769 / 2.6349 | 4.41e-06 / **3.10e-06** / 4.33e-06 |

**判据与结论**：MG 把细层迭代 28→14（正好一半），但每次迭代贵 **2.82×**
（0.1209 vs 0.0429 s），而 `2.82 > 2`，所以 `MG/plain = 1.41×` **更慢**。
盈亏平衡要求每次迭代成本比 <2.0 或迭代 ≤10。`mg-3` 只省 1 次迭代却把成本推到
4.73×。注意 `1.41×/2.82×` 是 **2260302（修正前）** 的数字；对 2260654 同法算是
**1.40× / 2.79×**，差异 1–3%，属修正带来的残差改善，不改变结论。五个参数臂
（`nu_pre` 4/2/1 × 粗容差 1e4/1e2/10）墙体时间 1.7220–1.7862 s，**平坦到 3.7%**，
说明成本在预条件内部的 fine apply 与 restrict/prolong，不在平滑步数与粗解容差。
