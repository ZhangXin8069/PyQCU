# KUPL 鲲鹏真实环境进展汇报（截至 2026-11-10）

本文汇总 2026-11-05 至 2026-11-10 期间 KUPL CPU 后端在**真实鲲鹏环境**上的
性能数据、代码改动与硬件环境信息。

> 文档状态：**C 级时点证据，未打 tag**。本文所有耗时、倍率与迭代数均为特定
> 窗口、特定作业号的实测值，不是永久性能承诺，也不得跨窗口拼接为新基线。
> 证据实体未全部入库，见第 7 节。

## 0. 记号与参数速查

本文所有缩写在此定义；完整版（含 ABI 槽位与源码行号）见
[`qcu_l24_solver_timing_20261111.md`](qcu_l24_solver_timing_20261111.md) §0。

| 记号 | 展开 | 含义 | 证据 |
|---|---|---|---|
| `c64` / `c128` | complex64 / complex128 | 复数量子场精度（实部 float32 / float64） | `pyqcu/kupl/define.py:47-48` |
| `cc` | `coarse_calls` | MG 粗层算子调用次数 | `lattice_clover_multigrid.cpp:1039` |
| `fc` | `fine_calls` | MG 细层算子调用次数 | 同上 |
| `屏` | screen hardcopy | GNU `screen` 终端快照文件 `logs/screen-*.txt` | 文件名模式 |
| `yNN` | 远端结果目录 | `data/kupl_hbm_opt_20261106/yNN/`，**不是作业号** | 屏记录 `job_y76.sh` 内 `WORK_DIR` |
| `2049xxx` | dsub 作业号 | 六位数字，来自 `djob -ll` | 屏记录 `Submit job <NNNNNNN>` |
| `r10`/`r20`/`r50` | `restart=` | MG 粗修正间隔；仅此语境 | CLI `--restart` |
| `r1`/`r2`（`dev99`） | MPI ranks | 1/2 ranks；与 `restart` 无关 | `profiler_solver_performance_history.md` §5.1 |
| `t24`/`t36`/`t576` | 线程数 | 单域 24/36 线程、全节点 576 线程 | STREAM 报告 |
| `n` | 样本数 | `n=1` 为单样本，不是中位数 | — |
| `MAD` | 中位绝对偏差 | `median(abs(x_i − median(x)))` | — |

**参数默认值**（`kupl_multigrid.py:607-648`、`_common.py:63-68`）：`atol=1e-6`、
`max_iter=1000`、`levels=2`、`mg_grid=(2,2,2,2)`、`restart=5`、
`coarse_max_iter=200`、`coarse_tol_factor=1e4`、`nv_iters=2`、`seed=42`、
`mass=0.05`、`csw=1.160920226`。

**收敛判据**（`lattice_solver.cpp:157-165`）：
递推退出 `‖r_k‖ ≤ max(atol, 4ε)·‖b‖`；
真残差验收 `‖r_true‖ ≤ max(atol, ε·M)·‖b‖`，其中
`M = 2048`（c64）或 `128`（c128）。c64 下 `ε·M = 2.44140625e-4`，即本文
反复出现的验收门。

**相对 vs 绝对、单奇偶 vs 完整场**（详见
[`qcu_l24_solver_timing_20261111.md`](qcu_l24_solver_timing_20261111.md) §0.3.1–0.3.2）：

- 本文全部 `atol` / `residual` 都是**相对值**；但 C++ 内部把
  `tolerance = max(atol, ε·M)·‖b‖` 算成绝对值再比，两者等价。
  **例外**：Clover-BiCGStab 用 `max(‖b‖,1)` 作分母（`lattice_solver.cpp:636`）。
- `独立真残差` 一律是 **FULL（两个 parity 全覆盖）**，由
  `kupl_residual.py:170-208` 对 `target in range(2)` 累加后开方。
- C++ 的 `native` 残差在 `compact_fine_`（odd-Schur cache）模式下是
  **Schur 单奇偶**，不是 FULL。因此 native 残差与独立残差**不能直接比大小**。
- `--skip-independent-residual` 时报告写入的 `acceptance` 就是 `args.atol`
  本身（`kupl_multigrid.py:885-890`），不是计算值。

**粗层容差**（`kupl_multigrid.py:188-189`）：
`coarse_atol(k) = atol × coarse_tol_factor × 3^(k−1)`。

## 1. 范围与证据等级

### 1.1 收录口径

| 类别 | 判定 | 本文处理 |
|---|---|---|
| **真实且稳定的计算环境** | `dsub` 提交到计算节点 `cn10579`、有 `report.json`、独立真残差与 `status=PASS` | **主证据**，第 5 节 |
| 真实计算节点但初始化异常/超时 | 已调度到 `cn10579`，无 `report.json` | 记为 BLOCKED，不入性能表 |
| 鲲鹏登录环境（`kunpeng920`） | `login02`/`login03`，共享机、无 HBM 域放置 | **辅助参考**，第 6 节，显式标注 |
| 早期调用 HBM 失败 | `set_mempolicy` 返回 `EINVAL`，HBM 掩码为空 | 第 4.1 节，作为失败证据收录 |
| 后期调用 HBM 成功 | `memory_policy=bind` + `memory_bound=true` | 第 4.2 节，含逐 rank 拓扑 |

### 1.2 三条不可混算的边界

1. **登录节点与计算节点是不同硬件**：登录节点 `Kunpeng-920`（无 SVE、4 个
   NUMA、无片上内存域），计算节点为 SVE2 + 16 个 CPU NUMA + 16 个片上内存
   NUMA。两者倍数只能作方向参考。
2. **`levels=1, restart=0` 不是 multigrid**：该路径是 parity-Schur
   BiCGStab，只作对照。正式 MG 必须 `levels>=2`、`restart>0`、`coarse_calls>0`
   且无 fine-BiCG fallback。
3. **`wall_seconds` 只包住求解调用**：不含输入预绑定、HDF5 读取、独立真残差
   重建与作业调度时间。setup 与 solve 必须分栏报告。

## 2. 计算节点硬件环境（实测）

### 2.1 节点身份与拓扑

| 项 | 实测值 | 来源 |
|---|---|---|
| 主机名 | `cn10579` | `djob -ll` 的 `execNode` 字段 |
| 调度器 | `dsub`/`djob`/`djob -ll`，队列 `q_hpcapp`，MPI 类型 `hmpi` | 作业脚本 `#DSUB` 头 |
| CPU NUMA 域 | **16** | `PYQCU::KUPL::TOPOLOGY` 的 `numa_nodes=16` |
| 片上内存（HBM）NUMA 域 | **16**，编号 = CPU 节点 `+16` | 绑定成功记录的 `cpu_node` / `memory_node` 配对 |
| 实测 CPU 编号上界 | **577**（`cpuset` 最高 `570-577`） | 全部拓扑记录的最大 cpuset |
| 每 NUMA 域 CPU 步长 | **38**（`0-7`、`38-45`、`76-83`、`114-121`、`152-159`、`190-197`、`228-235`、`266-273`、`304-311`、`342-349`、`380-387`、`418-425`、`456-463`、`494-501`、`532-539`、`570-577`） | 16 个域起点逐一核对 |
| SVE | SVE2（VL=64B，由 STREAM 探测程序口径确认） | `refer/kunpeng/lx2-stream-source/report.html` |

> 注意：`docs/kupl_kunpeng_guide.md` 第 10 节引用的“608 核、每域 38 核、16 DDR
> NUMA、16 片上 NUMA”来自鲲鹏 920 专业版开发手册的**示例机型参数**。本文的
> `numa_nodes=16`、`cpuset 0-577`、域步长 38 已由现场拓扑日志与 `cpuset` 逐条
> 核对，属于实测值；未直接取用手册数字。

### 2.2 实测 topology 记录样例

全部日志中共 **784 条** `PYQCU::KUPL::TOPOLOGY` 记录，按绑定状态统计如下：

| 类别 | 条数 | 判读 |
|---|---:|---|
| `numa_nodes=16` + `memory_policy=bind` + `memory_bound=true` | **166** | 真实启用了片上内存绑定 |
| `numa_nodes=16` + `memory_policy=default` + `memory_bound=false`（`bound_threads>0`） | **58** | 只做 CPU 绑核，**不能**计为 HBM 运行 |
| `numa_nodes=4`（登录节点）`cpu_node=-1` `cpuset=none` `memory_node=-1` | **560** | 无 NUMA/片上内存放置 |

16 rank × 8 thread、`KUPL_MEMORY_POLICY=bind`、`KUPL_MEMORY_NODE=local+16`：


```text
local_rank=0  local_size=16 numa_nodes=16 cpu_node=0  cpuset=0-7   memory_node=16 memory_policy=bind memory_bound=true threads=8 bound_threads=8
local_rank=4  local_size=16 numa_nodes=16 cpu_node=4  cpuset=152-159 memory_node=20 memory_policy=bind memory_bound=true threads=8 bound_threads=8
local_rank=12 local_size=16 numa_nodes=16 cpu_node=12 cpuset=456-463 memory_node=28 memory_policy=bind memory_bound=true threads=8 bound_threads=8
local_rank=14 local_size=16 numa_nodes=16 cpu_node=14 cpuset=532-539 memory_node=30 memory_policy=bind memory_bound=true threads=8 bound_threads=8
```

4 rank × 32 thread（每 rank 独占一个完整 NUMA 域）：

```text
local_rank=0 cpuset=0-31   memory_node=16   threads=32 bound_threads=32
local_rank=1 cpuset=38-69  memory_node=17   threads=32 bound_threads=32
local_rank=2 cpuset=76-107 memory_node=18   threads=32 bound_threads=32
local_rank=3 cpuset=114-145 memory_node=19  threads=32 bound_threads=32
```

### 2.3 硬件带宽实测（LX2 STREAM，FP64）

来源：`refer/kunpeng/lx2-stream-source/report.html`。口径为自定义 STREAM-style
固定窗口（每配置每操作 5 轮，共同计时窗口 0.15 s），逻辑读写量 ÷ 窗口时间，
**不是**内存控制器实际传输字节；Triad 使用 FMA，每线程独立数组并显式绑核。

| 频率 | 配置 | 总线程 | Copy | Scale | Add | **Triad** | 最大轮内 CV |
|---|---|---:|---:|---:|---:|---:|---:|
| 1.55 GHz | 全节点 HBM | 576 | 5391.36–5577.78 | 5391.04–5579.33 | 5453.54–5474.28 | **5455.05–5475.92** | 1.23% |
| 1.55 GHz | 全节点 DDR | 576 | 588.45–590.14 | 588.66–589.51 | 615.43–616.19 | **615.74–616.48** | 0.23% |
| 2.00 GHz | 全节点 HBM | 576 | 6915.37–6932.22 | 6914.63–6932.50 | 6788.25–6809.24 | **6790.91–6811.21** | 0.10% |
| 2.00 GHz | 全节点 DDR | 576 | 588.39–589.23 | 588.38–589.71 | 615.30–615.83 | **615.30–616.35** | 0.16% |
| 1.55 GHz | 单域 HBM | 24 | 301.33 | 301.44 | 278.65 | **278.88** | 0.06% |
| 1.55 GHz | 单域 HBM | 36 | 327.63–360.43 | 356.52–360.64 | 338.67–343.28 | **338.71–343.37** | 4.10% |
| 2.00 GHz | 单域 HBM | 24 | 345.75–367.31 | 346.27–367.32 | 319.14–336.93 | **319.06–336.95** | 0.10% |
| 2.00 GHz | 单域 HBM | 36 | 434.94–435.80 | 434.94–435.82 | 427.32–427.99 | **427.36–427.97** | 0.03% |

单位 GB/s。下界—上界为“各样本先取 5 轮中位数、再取这些中位数的最小–最大值”，
不是置信区间；两组频率来自不同样本，差异不能全部归因于频率。

**384 线程上限为推断值**：`4.46 TB/s @1.55 GHz`、`5.11–5.40 TB/s @2.0 GHz`，
由 16 × 单域 `t24` 结果推算，**不是**同时运行 384 线程的实测聚合样本。

> **口径更正**：`data/kupl_roofline_20261106.json` 的
> `hardware_stream_fp64.single_domain_t24_hbm_triad_gbps` 记为
> `278.65`（1.55 GHz）与 `[319.23, 337.31]`（2.0 GHz）。核对源 HTML 后确认，
> 这两个数字分别取自对应行的 **Add** 列；该配置的 **Triad** 应为
> `278.88` 与 `319.06–336.95`。全节点 576 线程的 Triad 字段与源 HTML 一致，
> 无需更正。

理论 FP32 SVE 峰值（假设每核 2 条 FMA 流水、512-bit SVE 每向量 16 个 FP32
lane、每 FMA 2 FLOP）：

| 线程数 | 1.55 GHz | 2.00 GHz |
|---|---:|---:|
| 384 | 38.09 TFLOP/s | 49.15 TFLOP/s |
| 608 | 60.31 TFLOP/s | 77.82 TFLOP/s |

这是**理论推算**，不是实测吞吐。

## 3. 登录节点硬件环境（辅助参考）

| 项 | 值 | 来源 |
|---|---|---|
| 主机名 | `login02` / `login03` | 会话提示符与 `hostname` |
| CPU 型号 | `Kunpeng-920` | 历史参考文档 |
| 核数 | 128 核、2 socket、128 MB L3 | 历史参考文档 |
| SVE | **无**（NEON 路径） | 历史参考文档 + 实际走 NEON |
| NUMA 域 | **4**（实测 `numa_nodes=4`） | `PYQCU::KUPL::TOPOLOGY` |
| 片上内存域 | 无 | 全部 560 条 4-NUMA 记录均为 `memory_node=-1`、`memory_bound=false` |
| 共享状态 | 审计时 `load average ≈ 91/48 CPUs`、约 106 用户 | 历史记录 |

> **口径说明（存在未解冲突）**：历史参考文档写“2 socket”，
> `docs/qcu_mg_performance_ledger_20261029.md` 同一台机器又写“2 NUMA”，
> 而 2026-11-06 的实测 `PYQCU::KUPL::TOPOLOGY` 明确给出 `numa_nodes=4`
> （560 条记录，`cpu_node=-1`、`cpuset=none`）。三者不能同时成立。
> 本文以最新实测 **4 个 NUMA 域** 为准，并保留“2 socket”作为历史描述；
> 差异可能来自不同登录节点（`login02` 与 `login03`）或不同日期的节点分组
> 调整，**未验证**，不得把 4 NUMA 当作所有登录节点的统一事实。

登录节点无法使用 SVE 与片上内存绑定，因此**不能**承担 HBM 验收；其数字只能
用于算法/参数/分格的相对筛选。

## 3.1 图表集

上述数值的可视化在 [`kupl_kunpeng_figures_20261110.pdf`](kupl_kunpeng_figures_20261110.pdf)
（源文件 [`kupl_kunpeng_figures_20261110.tex`](kupl_kunpeng_figures_20261110.tex)，9 页、
13 图 2 表）。图集中的每根柱都带真实数值标签与样本数，图注给出配置、窗口与证据等级；
作图硬规则见第 12 节。

| 图 | 内容 | 关键判读 |
|---|---|---|
| 图 1 | 全部求解点的绝对耗时（对数轴） | y76 `0.4346 s` 与目标 `0.1 s` 的 4.35× 差距 |
| 图 2 | 相对 89.13 s 基线的加速比 | 计算节点 205×；目标需 891× |
| 图 3 | 12 个计算节点作业的墙钟台账 | 只有 2 个作业真正完成求解 |
| 图 4 | 三个计算节点求解点的几何外推对比 | 213.4/164.7 s 高出弱扩展预期 123×/47× |
| 图 5 | y61 的 dslash 分项（左）与 roofline 利用率（右） | boundary 29.96% + wait 22.17%；带宽仅用 2.8–3.4% |
| 图 6 | LX2 STREAM FP64 Triad 八种配置 | 全节点 HBM 5465–6801 GB/s；单域 279–428 GB/s |
| 图 7 | 784 条拓扑记录的绑定统计 | 166 条真绑定、58 条仅绑核、560 条无放置 |
| 图 8 | 16 个 CPU NUMA 域到片上内存域的 cpuset 范围 | 步长恒为 38，上界 577 |
| 图 9 | 去重后 4 个同窗 A/B 窗口 | 比值 0.927→1.745，单调改善但窗口独立 |
| 图 10 | L24 true-MG 十种合法分格 | 最优 `1 1 1 16`/16×8 = 1.686 s |
| 图 11 | L8/L16 上 MG 对 BiCGStab | L8 1.86×；L16 4 rank 4.15×、32 rank 1.002× |
| 图 12 | 串行/并行 builder 与计算节点初始化 | builder 12.9–31×；HBM register 升至 44 s |
| 图 13 | L24 时间块 cache/builder 与 z-lane A-B-B-A | 时间块越大 solve 越慢；z-lane 慢 4.8% |

## 4. HBM 调用：早期失败与后期成功

### 4.1 早期失败（2026-11-05 之前，已修复）

**症状**：`KUPL_MEMORY_NODE=local+16` + `KUPL_MEMORY_POLICY=bind` 下，
片上内存掩码为空，`set_mempolicy()` 返回 `EINVAL`，HBM 完全不生效，
作业退回默认 DDR 分配但仍继续运行（静默降级）。

**根因**：`set_mempolicy()` 的 `maxnode` 参数必须按 64 位字对齐，原始实现传入
`node + 1`，内核按字解析时把有效位清零。

**修复**（提交 `acb550fa`）：

```cpp
// 修复前
if (syscall(SYS_set_mempolicy, mode, mask.data(), node + 1) != 0) { ... }
// 修复后
if (syscall(SYS_set_mempolicy, mode, mask.data(),
            static_cast<unsigned long>(words * bits)) != 0) { ... }
```

同一提交还修复了第二个静默失败：OpenMP 线程原本全部绑定到同一个 CPU，导致
32 线程纯计算循环 `1.00 s → 0.031 s`（约 32 倍）；修复后每线程绑定不同物理 CPU。

### 4.2 后期成功（2026-11-06 起）

修复后拓扑日志出现 `memory_policy=bind` 且 `memory_bound=true`。全部日志中共
**166 条**绑定成功记录，覆盖 16 个 CPU 域到 16 个片上内存域的完整配对：

```text
cpu_node 0 → memory_node 16      cpu_node 4  → memory_node 20
cpu_node 1 → memory_node 17      cpu_node 8  → memory_node 24
cpu_node 2 → memory_node 18      cpu_node 12 → memory_node 28
cpu_node 3 → memory_node 19      cpu_node 15 → memory_node 31
```

同时存在 **58 条**“策略为 default、`memory_bound=false`、但线程仍绑定”
（`bound_threads>0`）记录 —— 这类运行只做了 CPU 绑定，未做片上内存绑定，
不能计为 HBM 运行。

**验收边界**：`memory_bound=true` 只证明 `set_mempolicy()` 调用成功，**不等于**
页实际落在片上内存。严格验收还需在 solve 运行期间用
`numastat -p <rank_pid>` 与 `/proc/<rank_pid>/numa_maps` 检查实际页分布。
本轮 live 页分布证据未采集，因此本文不宣称任何一次运行完成端到端 HBM 页验证。

## 5. 计算节点性能数据（主证据）

### 5.1 生产配置

`24^3×64`、`complex64`、`mass=-0.277`、`kappa=0.13430029546065003`、
`csw=1.160920226`、`atol=1e-4`、`max_iter=50`。
构建：`build-native-lto`（`-march=native` + LTO）。

### 5.2 BiCGStab（`levels=1, restart=0`）对照基线

| 作业 | ranks × threads | grid | wall_seconds | iters | 独立真残差 | 验收门 | 状态 |
|---|---|---:|---:|---:|---:|---:|---|
| y61（剖析轮） | 32 × 12 | `1 1 1 32` | 0.4877（剖析总窗口） | — | — | — | PASS |
| **y76** | 32 × 12 | `1 1 1 32` | **0.4345627510** | 16 | `3.5412238955e-6` | `2.44140625e-4` | PASS |
| 2049318（4-rank 弱扩展点） | 4 × 32 | `1 1 1 4` | 164.7499241 | 16 | `1.237e-5`（**仅 native**） | 未做独立残差 | PASS（仅 native 门） |
| y91（手工几何 + slab） | 16 × 8 | `1 1 1 16` | 213.44090568200045 | 16 | `3.5410827126526993e-6` | `2.44140625e-4` | PASS |

`y76` 是本轮**计算节点最低可信值**，配置为 `mpirun -np 32 --bind-to none
--map-by ppr:2:numa`、`KUPL_MEMORY_POLICY=bind`、`KUPL_MEMORY_NODE=local+16`、
`KUPL_HUGEPAGE=1`、`KUPL_TOPOLOGY_REPORT=1`、rank-local HDF5 slab。

y91 的 `213.44 s` 与 2049318 的 `164.75 s` 都是**同代码、不同几何**
（16 × 8 与 4 × 32）的结果，与 y76 的 `0.4346 s` **不可比**。按每 rank 局部体积
线性外推，理想弱扩展应分别约 `1.74 s` 与 `3.48 s`，实测高出 **123×** 与
**47×**；该差距来自分格/线程布局相关的标量边界工作量、集合通信规模与
16 CPU/rank 的性能悬崖，**当前未分解**，不能归因于 HBM 是否生效。

### 5.3 安静节点重复与噪声带

| 作业 | wall_seconds | 备注 |
|---|---:|---|
| 历史最低 | 0.475 | 32×12 + HUGEPAGE |
| y61 | 0.4877 | 同轮剖析 |
| y50 | 0.5093 | 重复测量 |
| y47 | 0.5126 | 重复测量 |
| 争用时段 | 0.85–0.93 | 仅作参考，不可用于判定 |

同二进制同配置噪声带约 **±8%**（0.492–0.576 s）。因此**小于 10% 的差异一律
视为噪声**，任何改动必须在同一时段连续三次取中位数，或只看剖析分项计时。

### 5.4 相对 89.13 s 原始基线

原始基线为 4 ranks × 32 threads、无 NUMA 映射、HBM 未生效、线程挤单核：
`89.13 s`。相对该基线，`y76` 的 `0.4346 s` 约 **205×**，安静带
`0.475–0.513 s` 约 **174–188×**。

> 该倍率混合了四项修复（mempolicy 对齐、逐线程绑核、NUMA 映射、rank-local
> slab）的合计效果，不是单一优化项的收益。

### 5.5 Roofline 与瓶颈归属

按 y76 的 `0.43456 s / 16 it` 折算 34 次 Schur apply，每 rank 每次 apply
建模 `59,716,608 B`，全局建模流量 `60.51 GiB`、算术量 `86.63 GFLOP`：

| 指标 | 当前 | 384 线程保守上限 | 利用率 |
|---|---:|---:|---:|
| 有效 HBM 带宽 | 149.5 GB/s | 4.46–5.40 TB/s | 2.8–3.4% |
| 建模 FP32 算术 | 199.4 GFLOP/s | 38.09–49.15 TFLOP/s | 0.4–0.5% |

`y61` 剖析归属：scalar boundary `29.96%`、MPI wait `22.17%`、halo packing
`7.48%`、interior `1.89%`、dslash 之外 `38.51%`。

**结论**：当前求解器既不是带宽受限也不是 FP32 算力受限，而是 scalar
`ht=1` 边界内核与小消息全局同步受限。`0.1 s` 目标在同一悲观模型下只需约
`650 GB/s` 与 `0.87 TFLOP/s`，低于实测带宽与理论算力上限，因此
“物理不可达”的说法不成立；可达性取决于向量化与同步开销能否下降。

32 rank 时 t 方向 local extent 为 2、压缩后 `ht=1`，生产快路径每个站点都走
scalar boundary 内核。现有 block-dslash 在该 local volume 被门控关闭，且其
向量 lane 轴假设 `ht>=3`，因此**仅调低 `KUPL_BLOCK_DSLASH_MIN_HALF_SITES`
无法覆盖该几何**。

### 5.6 计算节点初始化剖析

作业 `2050850`：16 rank，`END rc=0`，总墙钟 `124 s`。

| 阶段 | 值 |
|---|---|
| `applyInit`/`applyEnd`（15 个 rank） | 6.02–7.43 s |
| `applyInit`/`applyEnd`（rank 2，离群） | 34.81 s |
| 最慢 rank 的 `register` | 19.48 s |
| `erase` | 0.002–0.237 s |
| `MPI_Comm_dup` | ≈1.55 s |

真实 HBM 求解（y91，`KUPL_HUGEPAGE=1`）中 `register` 升至 **41.75–46.30 s**，
一个 `erase` 为 `7.80 s`。**HBM 绑定 + HugePage advice 使首次触页显著变贵**，
该成本在 `wall_seconds` 之外，必须与求解内核分开优化与报告。

### 5.7 计算节点作业台账与几何不可比性

| 作业 | 类型 | ranks×thr | 墙钟 (s) | 结局 | 证据等级 |
|---|---|---:|---:|---|---|
| 2048748 | 1-rank 探针 | 1×1 | — | 被接受并调度到 `cn10579`，声明的 `y79.out` 未生成 | A |
| 2048816 | 32-rank A/B | 32×12 | 4894 | host pre-hook 后无 case stdout，同队列 kill，exit 143 | A |
| 2048893 | 16-rank 提交 | 16 | — | 被接受但未出现在活动作业查询 | A |
| 2049318 | 4-rank A/B | 4×32 | 850 | BiCGStab `164.75 s` PASS（仅 native 残差）；配对 MG 达 10 min 门后 kill，exit 143 | A |
| 2049394 | block 路径屏 | 16×8 | 1370 | 仅拓扑输出（16 rank cpuset 唯一、`bind`、HBM 节点正确），exit 143 | A |
| 2049477 | slab 路径屏 | 16×8 | 1160 | 仅拓扑输出，slab 文件存在，exit 143 | A |
| 2050223 | t-only 屏 | 32×4 | 540 | 无 report | A |
| 2050424 | init 剖析 v1 | 16×8 | 56 | `END rc=0` | A |
| 2050689 | init 剖析 | 16×8 | 240 | 内部超时 `rc=124`，停在 `topology-report`（HMPI Cartesian 挂起） | A |
| 2050850 | init 剖析 | 16×8 | 124 | `END rc=0`，全部 16 rank 完成 | A |
| 2050917 | MG 正式 | 16×8 | 787 | 无 `report.json`，TERM kill-job `2051058`，exit 143 | A |
| 2051058 | kill-job | 1×1 | — | 同队列本账号 TERM，成功 | A |

**几何不可比性的定量**：以 y76（32×12，每 rank 局部体积最小）为基准按局部体积
线性外推，16×8 与 4×32 的理想弱扩展预测应约为 `1.74 s` 与 `3.48 s`；实测为
`213.44 s` 与 `164.75 s`，分别高出 **123×** 与 **47×**。该差距不可能由 HBM 是否
生效解释，只能来自几何相关路径（每 rank 标量边界站点数、集合通信规模、以及
4×32 时 16 CPU/rank 的性能悬崖）。**当前未分解**，三项也**未**同窗重复。

## 6. 登录节点性能数据（辅助筛选）

全部为共享登录节点同窗口筛选值，**不是计算节点结论**。

### 6.1 L24 true-MG 对 BiCGStab（同窗 A/B）

配置：`24^3×64` c64、16 ranks × 8 threads、`grid=1 1 1 16`、同 gauge/RHS、
`atol=1e-4`、`max_iter=50`、E4 cache、`mg-grid=2 2 2 1`。

| restart | BiCGStab 中位 | true-MG 中位 | BiCG/MG 比值 | MG 证据 |
|---:|---:|---:|---:|---|
| 1（阻尼 gate，单 rank） | — | 7.52818155894056 | — | `fine_calls=49`、`coarse_calls=168` |
| 10 | 1.3972427972 | 0.7897557600 | 1.7692× | `fine=32`、`coarse=19` |
| 20 | 1.2307832352 | 0.7051456925 | 1.7454× | `fine=29`、`coarse=9` |
| 50 | 1.0918660681 | 0.7042407769 | 1.5504× | `fine=29`、`coarse=9` |

对应的 full-residual 伴随运行：

| 配置 | wall_seconds | 独立真残差 | 验收门 | 状态 |
|---|---:|---:|---:|---|
| r10 | 1.0437915269 | `6.8887275716e-5` | `2.44140625e-4` | PASS |
| r20 | 1.316182 | `5.5308248713e-5` | `2.44140625e-4` | PASS |
| **r50** | **0.9473317482** | `5.5308248713e-5` | `2.44140625e-4` | PASS |

**`restart=50` 只执行一次 coarse correction**，应描述为“initial deflation +
fine BiCGStab”，**不是**周期 V-cycle；它是登录节点最低绝对耗时，仍距
`0.1 s` 约 9.5 倍。

早期同窗 A/B（`restart` 未优化）曾得到 BiCGStab 中位 `1.2545952125 s`、
true-MG 中位 `1.3541084125 s`、比值 `0.926510`，即 MG 慢 7.9%。该负向结果与
后续正向结果**不是同一配置**，不得合并。

#### 6.1.1 窗口重建后的实际样本数（含一次更正）

> **更正（2026-11-11）**：本节原先写"只有 4 个独立窗口"，该结论基于错误的切窗
> 方法——把 `screen-127` 第二窗与 `screen-128` 第一窗当作同一运行。
> 以 `bicgstab_1` 为窗口起点重解后实为 **6 个完整窗口**，其中 w2/w4 未进入阶段
> 日志。决定性证据：`screen-128` 的 `bicgstab_1 = 1.2080937`，而 `screen-127`
> 的 `bicgstab_1 = 1.5286182`——同一运行不可能有两个不同值。
> 完整 6 窗表见
> [`qcu_l24_solver_timing_20261111.md`](qcu_l24_solver_timing_20261111.md) §3.1。

从屏记录逐条重建后，`24^3×64` c64、16 ranks × 8 threads、`grid=1 1 1 16`
同窗 A/B 共有 **6 个窗口**，其中 4 个（w1/w3/w5/w6）与阶段日志的
`early/r10/r20/r50` 精确对应：

| 窗口 | 屏 | `cc`/`fc` | BiCG 中位 | MG 中位 | 比值 | 对应阶段日志 |
|---|---|---|---:|---:|---:|---|
| w1 | 123 | 29/37 | 1.2545952125 | 1.3541084125 | 0.9265 | early |
| w2 | 126 | 19/34 | 1.2958426615 | 0.9988198766 | 1.2974 | — |
| w3 | 127 | 19/32 | 1.3972427972 | 0.7897557600 | 1.7692 | r10 |
| w4 | 128 | 19/32 | 1.1482496969 | 0.9025444695 | 1.2722 | — |
| w5 | 191 | 9/29 | 1.2307832352 | 0.7051456925 | 1.7454 | r20 |
| w6 | 192 | 9/29 | 1.0918660681 | 0.7042407769 | 1.5504 | r50 |

切窗依据（均来自 `logs/screen-73056-ss-20261106-*.txt`，等级 A）：

- `screen-123`（8 条）= w1 完整。
- `screen-126`（8 条）= w2 完整。
- `screen-127`（14 条）= w2 尾部 8 条 + **w3 完整 8 条**（hardcopy 重叠）。
  判据：`mg_1` 的 `fine_calls` 由 34（w2）变为 32（w3）。
- `screen-128`（14 条）= w3 尾部 8 条 + **w4 完整 8 条**。
  判据：`bicgstab_1` 出现 `1.2080937`，与 `screen-127` 的 `1.5286182` 不同。
- `screen-191`（8 条）= w5 完整。
- `screen-192`（14 条）= w5 尾部 8 条 + **w6 完整 8 条**。
  判据：`bicgstab_1` 出现 `1.3008456`，与 `screen-191` 的 `1.6146743` 不同。

**先前错误**：曾把 `screen-127` 第二窗与 `screen-128` 第一窗当成同一运行而合并，
得出"只有 4 个窗口"。该方法错误地把 w3 尾与 w4 头混合，产生了一个不存在的中位
（BiCG 1.2370 / MG 0.8683）。正确值为 w3（1.3972 / 0.7898）与
w4（1.1482 / 0.9025）。

`restart` 与 `cc`/`fc` 的对应关系只能由 `cc`/`fc` 间接判读：`cc=29,fc=37` 对应
早期的 `restart=5` 参数族，`cc=19,fc=34` 与 `cc=19,fc=32` 对应 r8/r10 族，
`cc=9,fc=29` 对应 r20/r50 族。屏记录本身不含 `restart` 字段，因此**不能**
逐窗口唯一确定 `restart` 值；上表主表（第 6.1 节）的 r10/r20/r50 归属来自阶段
日志的显式标注，两者不是同一证据层级。

### 6.2 分格筛选

| grid | ranks × threads | 结论 |
|---|---|---|
| `1 1 1 16` | 16 × 8 | 唯一保留正 MG 信号的 L24 分格 |
| `1 1 2 8` | 16 × 8 | BiCG `1.844325`、MG `1.840493`，MG/BiCG `1.0021×`，无收益 |
| `1 2 2 4` | 16 × 8 | BiCG `1.084889`、MG `1.250455`，MG 负收益，否决 |
| `1 1 1 8` | 8 × 16 | BiCG `1.877938`、MG `1.586531`，劣于 t-only 16-rank |
| 十种合法分格 | 多种 | 全部通过 fail-closed 门；最佳 `1.686342053115368 s`（16×8, `1 1 1 16`） |

16 CPU/rank 会稳定触发 3.2–3.6 s 性能悬崖（疑为 cpuset 跨 NUMA），
未提升为生产配置。

### 6.3 小格点与 setup 成本

| 格点 | 路径 | 值 |
|---|---|---:|
| L8 | BiCGStab（13 it） | 0.04161 s |
| L8 | MG `restart=1` | 0.02242 s |
| L16 | MG（16 ranks × 8, `1 1 2 8`） | 0.13384 s |
| L16 | BiCGStab（32 ranks × 4, `1 2 2 8`） | 0.16635 s |
| L16 | MG（32 ranks × 4） | 0.16597 s |

L16 上 4 rank 时 MG 约快 `4.15×`；32 rank 时 MG 与 BiCGStab 基本持平
（`0.1660` vs `0.1663 s`），粗层工作与小消息同步抵消了迭代优势。
**BiCG 与 MG 必须分格调优。**

并行 local-stencil setup（`KUPL_ENABLE_MG_STENCIL_PARALLEL=1`）实测：

| 格点 | 串行 builder | 并行 builder | 加速 | 数值检查 |
|---|---:|---:|---:|---|
| `8^3×16` | 8.617 s | 0.669 s | 12.9× | gram/Galerkin 一致 |
| `16^3×32` | 186.52 s | 10.62 s | 17.6× | gram/Galerkin 一致 |
| `24^3×64` | 外推约 42 min | 80.70 s | >31× | gram `1.55e-6`、Galerkin `3.58e-7` |

该路径默认关闭，仍是 opt-in，尚需计算节点验收。

### 6.4 L24 true-MG 网格矩阵（登录节点）

| mg-grid | ranks × threads | grid | MG wall (s) |
|---|---:|---|---:|
| `2 2 2 1` | 16 × 8 | `1 1 1 16` | 1.77820 |
| `2 2 2 1` | 32 × 4 | `1 1 1 32` | 1.51917 |
| `2 2 2 1` | 32 × 4 | `1 1 2 16` | 1.75694 |
| `2 2 2 4` | 16 × 8 | `1 1 2 8` | 2.06250 |
| `2 2 2 8` | 32 × 4 | `2 2 2 4` | 1.64822 |

> 该矩阵中 `1.60861 s` 与 `1.51917 s` 两组**不得作为 true-MG 证据**：
> `PYQCU_KUPL_MG_PROFILE=1` 显示 `coarse_calls=0` 并跟随
> `fallback=fine_bicgstab`，它们是 fine-BiCG 回退耗时。

时间块筛选（L24）：

| time block | cache 字节 | builder (s) | 合法最大 `grid_t` |
|---:|---:|---:|---:|
| 1 | 143,335,424 | 119.57 | 32 |
| 4 | 99,540,992 | 69.33 | 8 |
| 8 | 92,241,920 | 63.59 | 4 |

三者在 solve 阶段均慢于 time block 1；build 节省不抵粗层同步成本增加。

## 7. 代码改动说明

### 7.1 提交谱系

| 提交 | 日期 | 范围 | 标签关系 |
|---|---|---|---|
| `acb550fa` | 2026-11-05 12:40 | HBM 掩码对齐、逐线程绑核、rank-local slab、`KUPL_SKIP_INPUT_PREBIND` | dev107 之后 |
| `8eaef80f` | 2026-11-05 21:07 | matvec 展开、prefetch、t 全拆分段、halo 重叠 | 同上 |
| `82cc633a` | 2026-11-05 21:42 | 性能参考文档归档 | 同上 |
| `e8335068` | 2026-11-05 22:32 | rank-local slab 下 MG cache identity 与 coarse gauge 传递 | 同上 |
| `f8c2b238` | 2026-11-05 21:46 | dev108 日志归档 | **dev108** |
| `a51d32a0` | 2026-11-06 09:43 | true-MG fail-closed、阻尼线搜索、z-lane、并行 stencil、cache C-order 摘要 | **dev109** 的基础 |
| `136033f9` | 2026-11-06 16:03 | 周期手工几何、初始化剖析、ht2 boundary split、slab 输入 | **dev109** 指向此提交 |
| `8847f84e` | 2026-11-09 02:25 | 内容与 `136033f9` 完全等价（`cpp/kupl` 与 `pyqcu/testing/simple` 逐字节相同） | **在 `main` 祖先链上** |

> **谱系说明**：`dev109` 标签指向 `136033f9`，该提交**不是** `main` 的祖先。
> 但 `main` 上的 `8847f84e` 与它内容等价，因此 dev109 的源码改动已实际进入
> `main`。引用 dev109 的内容时应以 `8847f84e` 为可达锚点。

`dev107..HEAD` 区间共 51 个提交，其中 KUPL/测试/文档相关 5 个源码提交；
`git diff --stat dev108..HEAD` 在 `cpp/kupl`、`pyqcu/testing/kupl`、
`pyqcu/testing/simple`、`docs/kupl_kunpeng_guide.md` 上合计
**21 文件、+1008/−63**。

### 7.2 逐项改动

#### 7.2.1 HBM 掩码与线程绑核（`acb550fa`）

`cpp/kupl/qcu/src/lattice_cpu.cpp`：

- `set_mempolicy` 的 `maxnode` 由 `node + 1` 改为 `words * bits`（64 位字对齐），
  修复 HBM 掩码为空导致的 `EINVAL`。
- OpenMP 绑定从“设置一个共享 cpuset”改为“每个线程 `CPU_SET` 到
  `state.cpus[thread % size]` 的单一物理 CPU”，修复全部线程挤在单核。

`pyqcu/testing/simple/kupl_multigrid.py`：

- 新增 `_read_local_slab()`，按 `(Ellipsis,) + slice×4` 只读本 rank 的 HDF5
  hyperslab，不再全量读入后切片。
- `KUPL_SKIP_INPUT_PREBIND=1` 可跳过输入预绑定（每作业省约 15 s，已验证 PASS）。

效果：rank-local slab 把整作业从约 7 min 降到约 3 min。

#### 7.2.2 内核实验（`8eaef80f`）

`lattice_wilson_dslash.cpp`（+170 行）、`lattice_set.h`：手写 12 分量累加展开、
`#pragma omp simd`、`t` 全拆时边界双段 halo 重叠快路径。

**必须保留的负面结论**：这些实验在后续同窗三次中位复测中被判定为**无可靠
收益甚至更慢**（见第 8 节否决清单）。恢复性能实验前必须核对实际 blob，
不得假定当前 HEAD 优于 8eaef80 之前版本。

#### 7.2.3 MG cache identity（`e8335068`）

`kupl_multigrid.py`：rank-local slab 下 `gauge_full` 为 `None` 时，用
`gauge_file_digest` 替代 `logical_array_sha256(gauge_full)` 计算
`gauge_canonical_sha256`，并把 identity 计算改用 `gauge_local`，
修复 `UnboundLocalError` 与布局错误。

#### 7.2.4 true-MG fail-closed 与阻尼修正（`a51d32a0`）

`cpp/kupl/qcu/src/lattice_clover_multigrid.cpp`（+167 行）：

- 新增 `KUPL_REQUIRE_MG_CORRECTION=1`：`levels<=1`/`restart=0`、fine-BiCG
  fallback、或成功结束但无任何 accepted coarse correction 时**直接失败**。
- 新增 coarse correction 阻尼线搜索：在**同一个** coarse solve 结果上依次尝试
  `1, 0.5, 0.25, 0.125`，只接受使 fine true residual 下降的步长；全部失败则
  恢复原解与残差。避免重复 coarse solve。

背景：修复前 full-step additive correction 会把 fine residual 从 `3723.37`
推到约 `3666.03` 后再次增大，随后被回退门拒绝；因此**此前记为 MG 的
fallback 结果作废**。

#### 7.2.5 并行 local-stencil setup（`a51d32a0`）

`lattice_multigrid.cpp`（+91 行）：

- `KUPL_ENABLE_MG_STENCIL_PARALLEL=1` 为 coarse-site 循环启用
  `#pragma omp parallel for schedule(dynamic, 1)`，每个 worker 独立
  `source/even/schur` 缓冲。
- `KUPL_MG_STENCIL_THREADS` 可显式限制 worker 数，非法值直接报错。
- 通过 `std::atomic<int64_t> completed_sites` 与 `omp critical` 输出进度，
  避免长时间无输出。
- `CMakeLists.txt` 为 1/2/4-rank multigrid 与 legacy-MG 测试显式设置
  `KUPL_ENABLE_MG_STENCIL_PARALLEL=1;KUPL_MG_STENCIL_THREADS=2`。

#### 7.2.6 ht=1 z-lane 向量化候选（`a51d32a0`）

`lattice_wilson_dslash.cpp`（+248 行）：新增
`KUPL_ENABLE_T2_ZBLOCK_DSLASH=1`，实现 `compute_dslash_t2_spatial_zblocks()`
与 `compute_dslash_t2_time_zblocks()`，沿空间连续 z 轴向量化并显式 gather
交错的 local/remote 时间邻居 lane。

**状态：opt-in，非默认**。实测在登录节点 NEON 构建上 A-B-B-A
（32 ranks × 4 threads，`24^3×64`，仅 Schur）中位 **慢 4.8%**：

| 轮次 | 模式 | 秒/20 |
|---|---|---:|
| A | scalar | 0.46023 |
| B | z-lane | 0.49062 |
| B | z-lane | 0.49989 |
| A | scalar | 0.48491 |

scalar 中位约 `0.47257 s`，z-lane 约 `0.49525 s`。编译器同时报告循环向量化
失败。首个看起来更快的单样本被判定为噪声。

#### 7.2.7 cache 流式摘要 C-order 修复（`a51d32a0`）

`pyqcu/tools/_legacy_mg_cache.py`（76 行改动）：

- 新增 `_logical_chunk_selections(shape, target_elements)`，**固定前缀坐标、
  只拆分单一轴、保留完整尾轴**，按严格 C-order 生成有界 chunk。
- 根因：原实现把多个前导轴拆成 Cartesian 网格，组合出的块在 C-order 中互不
  相邻，导致 manifest 整数组摘要与流式摘要不一致。

验证：远端 E4 L24 cache 的 `lonv`/`hnn`/`hdg`/`sit` 四个张量在
`manifest_eq_streaming` 与 `streaming_eq_whole` 上全部 `true`。
`pyqcu/testing/simple/mpi/test_mpi_contract.py` 新增 blocked `lonv` 形状回归。

#### 7.2.8 周期手工几何（`8847f84e` / `136033f9`）

`cpp/kupl/qcu/src/lattice_geometry.cpp`（+49 行）、
`include/kupl/lattice_geometry.h`（+8 行）：

- **移除** `MPI_Cart_create` / `MPI_Cart_shift`，改用 `MPI_Comm_dup` 加显式
  行主序坐标与邻居 rank 计算：

```text
coord[dim]        = (rank / stride) % grid[dim]
neighbor_minus    = rank + ((coord+grid-1) % grid - coord) * stride
neighbor_plus     = rank + ((coord+1) % grid   - coord) * stride
```

- 进程网格为**全周期**边界，边缘邻居必须回绕；误用 `MPI_PROC_NULL` 会破坏
  2/4-rank Wilson 与 Laplacian halo。
- `LatticeGeometry::cartesian_rank()` 统一 diagonal Clover 与 coarse-MG peer
  的 rank 计算，禁止在非拓扑 duplicate communicator 上调用 `MPI_Cart_rank`。

**根因**：HMPI 的 `MPI_Cart_create` 路径在计算节点停在 topology 阶段。
profile 作业 `2050689` 在 240 s 内部超时（`END rc=124`），stderr 只有
`INIT_PROFILE` 到 `topology-report`，没有 `GEOMETRY_PROFILE` 或
`LATTICE_SET_PROFILE`。

修复后作业 `2050850` 完成全部 16 rank，`END rc=0`，124 s。
本地定向回归 `18/18` PASS（1/2/4-rank Wilson/Clover、2-rank Laplacian、
MG、Strict-MG、Legacy-MG）。

#### 7.2.9 初始化剖析（`8847f84e` / `136033f9`）

`KUPL_PROFILE_INIT` 与 `PYQCU_KUPL_PROFILE_INIT` 统一为同一开关，新增阶段：

| 文件 | 新增阶段 |
|---|---|
| `lattice_geometry.cpp` | `validated`、`base-comm`、`comm-duplicate`、`coordinates`、`metadata` |
| `lattice_cpu.cpp` | `read-numa`、`gather`、`process-affinity`、`omp-team` |
| `apply_init.cpp` | `register` |
| `apply_end.cpp` | `erase` |

输出格式 `PYQCU::KUPL::GEOMETRY_PROFILE: stage=... elapsed=...`，
与既有 `INIT_PROFILE` / `SCRATCH_PROFILE` / `LATTICE_SET_PROFILE` 一致。

#### 7.2.10 ht=2 boundary split 候选（`8847f84e`）

`KUPL_ENABLE_HT2_TBOUNDARY_SPLIT=1`（**默认关闭**）把 `local[3]==4` 的几何
纳入边界分段快路径，并新增 `use_ht2_tboundary_split()` 与
`!use_block_dslash(...)` 前置条件，避免与 block-dslash 路径冲突。

同窗 A/B：BiCG 中位 `1.099630 → 1.037594 s`（略正向），但 true-MG
`0.902746 → 1.041128 s`（约慢 15%）。**保持 opt-in，不设为生产默认。**

#### 7.2.11 slab 输入与 MPIO fail-closed（`8847f84e`）

- 新增 `pyqcu/testing/simple/prepare_mpi_slab.py`（164 行）与
  `test_prepare_mpi_slab.py`（34 行）：按固定 `(lattice, grid)` 写出连续
  `rank_XXXX.h5` slab 与 manifest。L24 `grid=1 1 2 8` 下每个 gauge slab 为
  `15,931,608` 字节、source slab 为 `5,314,776` 字节。
- `kupl_multigrid.py` 新增 `--gauge-slab-dir` / `--source-slab-dir`，并在
  slab 尾四轴形状不符时直接报错。
- 新增 `KUPL_HDF5_MPIO_READ=1`，要求所有 rank 取值一致且 h5py 支持 MPI；
  远端 `h5py 3.7.0` 的 `get_config().mpi == False`，因此该路径 fail-closed，
  **未在远端验证**。
- slab cache 必须绑定固定 `(lattice, grid)`，禁止跨 rank/grid 复用。

## 8. 已否决路线（不得重复）

以下均在同一时段三次中位复测中被判定无收益或更差，已完整撤销：

| 路线 | 实测结论 |
|---|---|
| block/face dslash（t 全拆 `ht=1`） | 相关分支直接返回，face 路径更慢 |
| `KUPL_FUSED_BICG_DOTS=1` | 相邻对 1.05%/3.41%，落在噪声带 |
| `OMP_WAIT_POLICY=ACTIVE/PASSIVE` | 无稳定收益 |
| `OMP_PROC_BIND=close/spread` | 库主动拒绝（与库内绑定互斥）；历史曾把 L24 从 7.2–7.4 s 恶化到 65–67 s |
| `OMP_DYNAMIC` | 无收益 |
| x 轴拆分 | 未通过筛选 |
| z 向 16-lane SIMD（登录节点 NEON） | 慢 4.8% |
| halo 与计算重叠（`y39`–`y44`） | dslash 与 dagger 两侧三次中位均无收益 |
| 软件预取（`y52`/`y53`） | boundary 仅降 2.3%，噪声 |
| matvec 数组化（`y60`） | boundary `+52%`，已回退 |
| matvec 完全展开（`y62`） | 与标量版持平或略差 |
| numactl 外层绑定 | 与库内绑定冲突并导致挂起 |
| 48 ranks（`ht=2`） | MPI abort（exit 2） |
| 64 ranks | 始终排队 |
| 16 CPU/rank | 稳定触发 3.2–3.6 s 悬崖 |
| `levels=2` 两级 setup | 单 rank 构造超 10 min，多 rank 导出被代码禁止 |
| 128 OpenMP 线程跨 NUMA 建 cache | 短并行区严重回退 |

## 9. 失败、阻塞与未验证项

### 9.1 已解除的阻塞

| 阻塞 | 根因 | 解除方式 |
|---|---|---|
| HBM 绑定静默失效 | `set_mempolicy` `maxnode` 未按 64 位字对齐 | `words*bits` 修复（`acb550fa`） |
| 线程挤单核 | OpenMP 绑定共享 cpuset | 逐线程单一 CPU（`acb550fa`） |
| HMPI topology 挂起 | `MPI_Cart_create`/`MPI_Cart_shift` | `MPI_Comm_dup` + 显式坐标（`8847f84e`） |
| cache 摘要不一致 | 多前导轴 Cartesian 拆分破坏 C-order | 单轴拆分流式摘要（`a51d32a0`） |
| 假 MG 结果 | full-step correction 被回退门拒绝 | `KUPL_REQUIRE_MG_CORRECTION` + 阻尼线搜索 |

### 9.2 仍阻塞

| 项 | 状态 | 证据 |
|---|---|---|
| 计算节点 true-MG 正式 A/B | **BLOCKED** | 作业 `2050917` 运行 787 s 无 `report.json`，经同队列 TERM kill-job `2051058` 终止，`jobExitCode=143`；stderr 停在初始化 profile |
| 32 ranks × 12 threads 手工几何复验 | **BLOCKED** | 作业 `2048816` 在 host pre-hook 后 4894 s 无 case stdout，已终止 |
| 多次 16/32-rank 求解作业 | **BLOCKED** | 均达 10 分钟门禁无 report，`jobExitCode=143` |
| L32 true-MG / BiCG / 加速比 | **BLOCKED** | 1-rank E1/tblock8 cache build 与 32-rank BiCG 均超 10 分钟门禁终止 |
| 远端 HDF5 MPIO | **BLOCKED** | `h5py 3.7.0` 无 MPI 支持，fail-closed |
| live HBM 页分布 | **未采集** | 未在 solve 期间执行 `numastat -p` / `numa_maps` |

### 9.3 未验证项

- 计算节点 `restart=20/50` 与 BiCGStab 的同窗 A/B。
- 计算节点 z-lane、ht2 boundary split、并行 stencil 的正式 A/B。
- 计算节点标准 STREAM 与算力复验（`job_hw_stream_20261106.sh` 与
  `job_hw_stream_mpi_20261106.sh` 已准备，结果未取回）。
- 远端 `set_mempolicy` 严格模式下页实际落点。
- 0.1 s 目标未达成；最低可信值为 y76 的 `0.4346 s`。

### 9.4 作业审计规则（本轮执行）

- 运行满 5 分钟必须查日志、stderr、调度器状态与进度。
- 满 10 分钟且不能证明健康则终止并全面审查。
- 不覆盖历史输出；每次实验使用新 `yNN` 目录与新 `out`/`err`。
- 未绕过调度器 hook、未绕过 host-key 校验、未操作其他用户进程。
- 终止只经同队列本账号 kill-job 路径。

## 10. 证据清单与位置

### 10.1 已入库（可 Git 复现）

下表按 `git ls-files` 实测状态填写，**不是**按设计意图填写。

| 路径 | Git 状态 | 内容 |
|---|---|---|
| `logs/kupl_qcu_mg_performance_reference_20261105.txt` | **已跟踪** | 登录/计算节点性能参考、配置与噪声带 |
| `logs/kupl_qcu_mg_instructions_20261105.txt` | **已跟踪** | 原始任务约束、89.13 s 基线、否决路线、屏交互教训 |
| `logs/resume-20261105T080514.gnF1tz.txt` | **已跟踪** | 跨平台 resume 引导：根因、断点、SHA-256 清单 |
| `logs/v2026110521.txt` | 已跟踪 | dev108 归档日志 |
| `logs/v20261106091427.txt` | 已跟踪 | cache 摘要修复与 true-MG fail-closed 检查点 |
| `logs/v202611061035.txt` | 已跟踪 | 阻尼修正、分格/参数扫描、L32 尝试 |
| `logs/v20261106104704.txt` | 已跟踪 | L24 `restart=10` 同窗 A/B 与 CTest 35/35 |
| `logs/v20261106114436.txt` | 已跟踪 | `restart=20/50` 与最低绝对耗时 |
| `logs/v202611061446.txt` | 已跟踪 | 阶段性能简报（HBM/roofline/计算节点诊断） |
| `logs/v202611061550.txt` | 已跟踪 | HMPI 拓扑修复与计算节点 BiCGStab 基线 |
| `refer/kunpeng/lx2-stream-source/` | 已跟踪 | LX2 STREAM 源码、计划、报告 HTML 与 SHA256SUMS |
| `logs/screen-73056-ss-20261106-*.txt` | 已跟踪（约 500 个） | 屏幕 hardcopy，A 级原始记录来源 |
| `data/kupl_roofline_20261106.json` | **未跟踪**（`data/.gitignore` 忽略） | Roofline 机读数据；单域 Triad 字段有第 2.3 节所述偏差 |
| `logs/kupl_hbm_opt_20261106.md` | **未跟踪**（`logs/.gitignore` 忽略 `*`） | 本轮最完整的阶段总文档，含计算节点全部 `yNN` 结果与失败边界 |
| `logs/v20261109.txt` | 已跟踪 | PyPI 0.1.0–0.1.13 发布记录（与本主题无关，仅说明 `logs/` 混合内容） |

> **重要缺口**：`logs/kupl_hbm_opt_20261106.md` 与 `data/kupl_roofline_20261106.json`
> 是本次汇报的两个主要机读/文字来源，但**都未被 Git 跟踪**。`logs/.gitignore`
> 的第 1 行 `*` 使所有非 `.txt` 日志被忽略；`data/.gitignore` 同样以 `*` 开头。
> 因此 y76、L8/L16 表、builder 时间等 **B 级证据只存在于工作区**，tag 恢复不到。
> 这是本报告必须说明的证据完整性缺口，不是笔误。

### 10.2 仅存于远端（未入库）

以下实体位于远端运行树 `~/PyQCU-main/`，本地工作区不存在，需经已认证会话读取：

- `data/kupl_hbm_opt_20261106/`：`y76`、`y91`（`bicgstab_1/report.json`、
  `mg_1/stderr.txt`）、`l24_true_mg_vs_bicg_r{10,20,50}_v1/results.json`、
  `l24_true_mg_r{10,50}_full_v1/report.json`、`mg_damping_gate_L24_v2/`、
  `mg_true_grid_sweep_L24_v1/`、`mg_true_param_sweep_L24_v2/`、
  `mg_e4_tblock1_L24_v1/cache_E4_mg2x2x2x1.h5`、`y91_{gauge,source}_slabs_t16/`。
- `data/production_L24/`：`gauge_24x24x24x64_c64.h5`、
  `source_24x24x24x64_c64.h5`。
- `data/production_L32_v1/`：由
  `beta6.20_mu-0.2770_ms-0.2400_L32x64_cfg_11000.lime` 转换的
  `32×32×32×64` c64 gauge/source。
- `data/kunpeng_stream_20261106/`、`data/kunpeng_stream_mpi_20261106/`：
  STREAM 作业输出目录，结果未取回。
- `build-native-lto/`、`build-t2zblock-generic/`：实际构建目录与 `libqcu.so`。

### 10.3 远端环境事实

- 运行树 `~/PyQCU-main/` **无 `.git` 元数据**，不能从远程直接核对提交身份。
- 输入在 `~/data`，作业脚本在 `~/sh`，stdout/stderr 在 `~/out`/`~/err`，
  辅助代码在 `~/py`。
- 批量 SSH 公钥认证被拒；经已认证的 `74049.lingsheng` screen 会话作为审计
  控制路径。直接 SSH `cn10579` 被 host-key 校验拒绝，**未绕过**。

## 11. 结论与下一步

### 11.1 结论

1. **HBM 已能真正启用**：HBM 掩码对齐与逐线程绑核修复后，计算节点稳定输出
   `memory_policy=bind` + `memory_bound=true`，16 个 CPU 域到 16 个片上内存域
   配对完整。但**端到端页分布验收尚未完成**。
2. **计算节点最低可信求解为 `0.4346 s`**（y76，32 ranks × 12 threads，
   16 iterations，独立真残差 `3.54e-6`，PASS）。相对 89.13 s 原始基线约
   205 倍，但距 0.1 s 目标仍有约 4.3 倍。
3. **瓶颈不是 HBM 带宽**：利用率仅约 2.8–3.4% 带宽、0.4–0.5% FP32 SVE 峰值；
   主因是 scalar `ht=1` 边界内核（约 30%）、MPI wait（约 22%）与 dslash 之外
   的 38.5%。
4. **true-MG 在登录节点出现正向信号**（r10/r20/r50 约 1.55–1.77 倍中位加速，
   full-residual PASS），但**计算节点正式 A/B 仍 BLOCKED**，不能升级为生产结论。
5. **代码侧已完成五项关键修复**：HBM 掩码、逐线程绑核、rank-local slab、
   HMPI 周期几何、cache C-order 摘要；并新增 true-MG fail-closed、阻尼线搜索、
   并行 stencil（12.9–31 倍 setup 加速）与多项 opt-in 内核。

### 11.2 按优先级的下一步

1. **恢复计算节点执行健康**：先用最小、短时、可写输出的 `dsub` 探针验证
   host pre-hook 与输出路径，再启动任何求解作业。
2. 用**同一手工周期几何**在 `32 ranks × 12 threads`、`grid=1 1 1 32` 上复跑
   BiCGStab，与 y76 的 `0.4346 s` 做同配置对照。
3. 优化 HBM 首次触页：`register` 的 `41.75–46.30 s` 必须从 setup 侧解决，
   不能计入 solve 比较。
4. 在同一 32×12 配置上做 true-MG `restart=10/20` 正式 A/B，硬门为
   `coarse_calls>0`、无 fallback、独立真残差 PASS。
5. 在 solve 运行期间采集 `numastat -p` 与 `numa_maps`，完成端到端 HBM 页验收。
6. 降低 L32 cache builder 成本或改用已验证的分布式 builder，再补 L32 三项目标。

## 11.4 记号可解释性规则

本轮修正暴露出一类系统性问题：归档文档大量使用未定义的缩写，读者无法判断数据
是否可信。此后新增或修改归档数据时必须遵守：

1. **每个记号在首次出现处或统一 §0 定义一次**，并给出**源码行号或文件路径**作为
   证据。仅靠上下文暗示不算定义。
2. **同一字母不得携带两种语义**：
   - `E1–E7` 专用于环境代号，故 4 ranks×32 threads 一律写 `4r×32t`，
     禁用旧写法 `E4x32`（本轮的 tex 已改正）；
   - `r*` 必须区分 `restart=`（如 `r10`）与 MPI ranks（如 `r1/r2`），
     跨文档引用时写出完整词；
   - `A1–A11` 是行号，与"A/B 测试"无关，正文出现时加编号前缀（如 `A100-A1`）。
3. **`cc`/`fc` 必须展开为 `coarse_calls`/`fine_calls`**，且不得把
   `coarse_calls>0` 解释为"修正被接受"——后者需要 `KUPL_REQUIRE_MG_CORRECTION=1`
   放行才成立。
4. **`yNN` 是远端目录名，不是作业号**。引用时必须同时给出目录名与（若已知）
   dsub 作业号；两者无映射记录时写"未记录"，不得省略。
5. **单样本、仅 native 残差、跨几何并列**三类风险必须在同一行内标注，
   不能只写在表注里。
6. **收敛判据必须写公式而非名字**：`‖r‖ ≤ max(atol, ε·M)·‖b‖`，并把
   `M=2048(c64)/128(c128)` 与 `ε·M=2.44140625e-4` 一起给出。

## 11.3 图表与作图规则

本轮补充的图表集（第 3.1 节）固化了五条作图规则，后续同类汇报必须复用：

1. **对数轴柱标签**：pgfplots 在 `ymode=log` 下默认把 `nodes near coords` 输出成
   log10 值（`0.435` 会显示成 `-0.4`）。必须使用
   `point meta=explicit symbolic` 并给出 `meta` 列，才能打印真实数值。
2. **误差条语义**：只允许表示样本内 min–max 的半幅，并在图注写明 $n$ 与轮数；
   不得把 min–max 当作置信区间，也不得用单样本画“误差条”。
3. **跨硬件禁止混算**：计算节点 `cn10579`（SVE2、16 CPU NUMA + 16 片上 NUMA）
   与登录节点（Kunpeng-920、无 SVE、4 NUMA、共享机）必须分栏；同一张图内若
   同时出现，必须用不同填充与显式口径警告。
4. **几何不可比性**：当 ranks × threads × 每 rank 局部体积不同时，必须给出弱扩展
   外推基准（本例 `1.74 s` / `3.48 s`）与实测倍差（`123×` / `47×`），
   不得只并列绝对值。
5. **列名与取列**：从外部报告抄数必须核对列名。本轮发现
   `data/kupl_roofline_20261106.json` 把源表 Triad 列的 `278.88` 写成了
   Add 列的 `278.65`、把 `319.06–336.95` 写成了 `319.23–337.31`；
   引用 STREAM 数字时以 `refer/kunpeng/lx2-stream-source/report.html` 原表为准。

编译验收：两遍 `xelatex` 后必须满足 `Overfull=0`、`Float too large=0`、
`Missing character=0`，并逐页目检裁切与遮挡。

## 12. 相关文档

- [`kupl_kunpeng_guide.md`](kupl_kunpeng_guide.md)：通用鲲鹏构建、MPI/OpenMP
  拓扑、NUMA/片上内存策略与测试入口。
- [`kupl_only_deployment.md`](kupl_only_deployment.md)：纯 KUPL 安装、生产
  BiStabCG/Legacy-MG 与 HBM 放置。
- [`three_way_benchmark_20261022.md`](three_way_benchmark_20261022.md)：
  CUDA/KUPL/QUDA 三方同窗测评（x86 Xeon，非鲲鹏）。
- [`qcu_mg_performance_ledger_20261029.md`](qcu_mg_performance_ledger_20261029.md)：
  A100/DCU 性能账本（GPU 路径，与本文 CPU 路径不可混算）。
- `skills/pyqcu-kupl/references/kupl-history-2026.md`：KUPL 历史证据索引。
