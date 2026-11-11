# PyQCU L24（`24³×64`）全配置耗时台账

本文回答一个具体问题：**PyQCU 求解 `24³×64` 需要多久**。由于该格点涉及 6 个后端/
硬件类别、4 种精度与十余种求解路径，不存在单一答案。本文把所有**已记录**的
`24³×64` 求解耗时按后端与硬件分类归档，并逐行给出配置、迭代、残差、证据等级与
限制。

> 文档状态：**C 级时点证据汇总，未打 tag**。本文不产生新测量，只对既有
> `docs/**` 与 `logs/**` 记录做分层归档。任何单值都必须在同一表格行内连同后端、
> 硬件、精度、路径与统计口径一起引用，禁止跨行取最小值作为"PyQCU 的 L24 性能"。

## 0. 记号、精度与参数约定

本文使用大量缩写。**所有记号在此定义一次**，正文不再重复；凡本文未定义的记号，
视为无来源，不得引用。

### 0.1 记号表

| 记号 | 展开 | 含义 | 证据来源 |
|---|---|---|---|
| `c64` / `c128` | complex64 / complex128 | 复数量子场精度；实部为 float32 / float64 | `pyqcu/kupl/define.py:47-48`（`_LAT_C64_=2`、`_LAT_C128_=3`） |
| `cc` | `coarse_calls` | MG 粗层算子（restrict/coarse solve/prolong 组合）被调用次数。**不是**"coarse correction 次数" | `cpp/kupl/qcu/src/lattice_clover_multigrid.cpp:1039` 的 profile 输出字段 |
| `fc` | `fine_calls` | 细层算子（Schur dslash）被调用次数 | 同上 |
| `屏` | screen hardcopy | GNU `screen` 的 `hardcopy` 落盘文件，内容为终端快照（含 `(base) user@login02:~$` 提示符与折行） | 文件名 `logs/screen-73056-ss-20261106-NNN.txt` |
| `yNN`（`y76`/`y75`/`y61`…） | 远端结果目录 | `/home/share/nsls_yangyb/PyQCU/data/kupl_hbm_opt_20261106/yNN/`。**是目录名，不是作业号** | `logs/screen-73056-ss-20261106-146.txt`（显示 `job_y76.sh` 内 `WORK_DIR="$CASE_DIR/y76"`） |
| `K1`–`K9` | 本文行号 | 仅用于本文 §2 表格索引，无外部含义 | — |
| `V1`–`V8` | 本文行号 | §7 V100 行索引 | — |
| `X1`–`X7` | 本文行号 | §4 Xeon 行索引 | — |
| `A1`–`A11` | 本文行号 | §8 A100 行索引。**与"A/B 测试"无关** | — |
| `C1`–`C3` | 本文行号 | §9 DCU 行索引 | — |
| `D1`–`D13` | 本文行号 | §6 `dev99` 期行索引 | — |
| `w1`–`w6` | 窗口编号 | §3.1 的 6 个同窗 A/B 窗口 | 本文 §3.1 |
| `r*`（`r10`/`r20`/`r50`） | `restart=` 参数值 | **仅在此语境**表示 MG 粗修正间隔 | CLI `--restart` |
| `r1`/`r2`（`dev99` 表） | MPI ranks 数 | **仅此类**表示 1/2 个 MPI rank，与 `restart` 无关 | `docs/profiler_solver_performance_history.md` §5.1 原注 |
| `E1`–`E7` | 环境代号 | §2–§8 的 7 个计算环境 | `qcu_l24_environment_bandwidth_20261111.md` §1 |
| `4r×32t` / `16r×8t` | ranks × threads | 4 ranks × 32 threads / 16 ranks × 8 threads。**旧写法 `E4x32`/`E16x8` 已废弃**，因与 `E1–E7` 环境代号撞字母 | — |
| `n` | 样本数 | 该行统计用的独立重复次数；`n=1` 表示单样本，**不是中位数** | — |
| `MAD` | median absolute deviation | 中位绝对偏差：`median(x_i − median(x))` 的绝对值，即 `median(abs(x_i − median(x)))` | — |

### 0.2 精度与质量参数

| 参数 | 本文字段 | 定义 | 证据 |
|---|---|---|---|
| 精度 | c64 / c128 | 见 §0.1 | — |
| `mass` | 质量参数 | 直接写入 `argv[0]` | `include/kupl/define.h:73`（`kMass=0`） |
| `kappa` | `1/(2·mass + 8)` | Clover-Wilson 跳距系数。本文 `mass=-0.277` → `kappa=1/7.446=0.13430029546065003` | PyQCU 公共约定；与本文数值一致 |
| `csw` | Sheikholeslami-Wohlert 系数 | 写入 `argv[7]` | `define.h:80`（`kCsw=7`） |
| `atol` | 相对残差目标 | 写入 `argv[1]` | `define.h:74`（`kAtol=1`） |
| `max_iter` | 最大外层迭代 | 写入 `params[13]` | `define.h:26`（`kMaxIter=13`） |

### 0.3 收敛判据（两套阈值，此前文档未区分）

源码 `cpp/kupl/qcu/src/lattice_solver.cpp:157-165`：

```cpp
solve_tolerance      = atol                    // 相对目标
rhs_norm             = sqrt(global_norm2(rhs))
recurrence_tolerance = max(atol, 4ε) · ‖b‖     // 迭代内退出判据
acceptance_tolerance = max(atol, ε·M) · ‖b‖    // 收敛后的真残差验收门
```

其中 `ε` 为该实精度机器 epsilon，`M` 由 `residual_floor()` 给出
（`cpp/kupl/qcu/src/backend.cpp:473-477`）：

```cpp
M = (sizeof(Real) == sizeof(float)) ? 2048 : 128
```

**代入验算**：

| 精度 | `ε` | `M` | `ε·M` | 本文出现的验收门 |
|---|---|---:|---|---|
| c64 | `1.1920929e-07` | 2048 | **`2.44140625e-4`** | 是的，即此值 |
| c128 | `2.2204460e-16` | 128 | `2.84217094e-14` | — |

即**本文反复出现的 `2.44140625e-4` 不是经验常数，而是 `eps(float32) × 2048`**。

三个残差量必须区分：

| 名称 | 定义 | 本文如何标注 |
|---|---|---|
| 递推残差 | `‖r_k‖`，迭代内由递推公式更新 | 未单列 |
| native 真残差 | 用 `v = A·x` 重算的 `‖b − A·x‖`，C++ 内计算 | 标注"仅 native" |
| **独立真残差** | Python/NumPy 用完整 Wilson-Clover 算子重建并做 MPI halo，是唯一可称"独立验证"的量 | 标注"独立真残差" |

`residual_source` 字段直接写入报告，取值见
`pyqcu/testing/simple/kupl_multigrid.py:890-905`（三种：跳过 / 分布式 NumPy / 单 rank NumPy）。

### 0.3.1 相对值还是绝对值？——三重口径

**本文所有 `atol`、`tolerance`、`residual` 都是相对值**，但"相对"的分母有三个
版本，必须按行判断：

| 口径 | 公式 | 出现位置 | 分母是什么 |
|---|---|---|---|
| **A. 相对 `‖b‖`**（标准） | `rel = ‖b − A·x‖ / ‖b‖` | 本文所有表格、`report.json` 的 `final_true_residual` | 该求解 RHS 的 2-范数 |
| **B. 相对 `max(‖b‖, 1)`** | `rel = ‖b − A·x‖ / max(‖b‖, 1)` | C++ 内部日志 `full_true_residual=... relative=...`（`lattice_solver.cpp:489-491`、`:637-641`） | 同上但**下限为 1**；当 `‖b‖ < 1` 时该口径会**系统性地偏小** |
| **C. 绝对量** | `‖b − A·x‖`，不除任何东西 | C++ 日志的 `full_true_residual=` 与 `recomputed_residual=` 字段；验收比较 `residual > tolerance` 两边都是**绝对值** | — |

**为什么容易混淆**：C++ 把 `tolerance` 算成 `max(atol, ε·M) · ‖b‖`（绝对值），
再与同样绝对值的 `residual` 相比 —— 这等价于 A 口径的相对判据。因此
"`atol` 是相对目标"与"比较用绝对值"**同时成立**，不矛盾。

**必须注意的例外**：`Clover-BiCGStab` 的真实比较式是
`max(atol, ε·M) · max(‖b‖, 1)`（`lattice_solver.cpp:636`），即 B 口径。
因此凡本文引用 `CLOVER_BISTABCG_SCHUR` 的 `tolerance=` 字段，都与表格里的
A 口径相对残差**不完全可比**；当 `‖b‖ ≥ 1` 时两者等价。

**`--skip-independent-residual` 的影响**：
`pyqcu/testing/simple/kupl_multigrid.py:885-890` 直接把
`acceptance = float(args.atol)` 写进报告，`final_true_residual` 置为 `null`。
所以这类行的"验收门"是**配置值**，不是算出来的；本文所有标"仅 native"
的行都属于此类。

### 0.3.2 单奇偶还是完整场？——三种残差作用域

| 作用域 | 枚举值 | 覆盖的格点 | 代码位置 |
|---|---|---|---|
| **Schur（单奇偶）** | `ResidualScope.SCHUR` | 只覆盖一个 parity（`params[kParity]`），是奇偶预条件后的 Schur 补方程 | `pyqcu/api/models.py:40`；`lattice_solver.cpp:150-300` |
| **MATPC** | `ResidualScope.MATPC` | 单奇偶的 `M†M` 正规方程 | `models.py:39` |
| **FULL（完整场）** | `ResidualScope.FULL` | 两个 parity 都覆盖，即完整 4 分量旋量场 | `models.py:38`；`lattice_solver.cpp:473-496` |

本文表格中的 `独立真残差` **一律是 FULL 口径**：它由
`pyqcu/testing/simple/kupl_residual.py:170-208` 的
`distributed_relative_residual_streaming()` 计算，内部对 `target in range(2)`
两个 parity 分别累加 `‖r‖²` 与 `‖b‖²`，最后取 `sqrt(Σ分子 / Σ分母)`：

```text
rel_full = sqrt( Σ_p ‖b_p − A_p·x‖² / Σ_p ‖b_p‖² ),  p ∈ {even, odd}
```

**C++ 内部的 native 残差不是 FULL**：`applyCloverMultigridQcu` 在
`compact_fine_` 模式下走 `clover_bistabcg_dslash`，其残差是 **Schur 单奇偶**；
只有 `full_to_canonical` 路径才覆盖整场。因此：

- 报告 `residual_source = "KUPL native fail-closed solve"` → **Schur**
- 报告 `residual_source = "independent ... full Wilson-Clover reconstruction"`
  → **FULL**

**这解释了本文一处关键差异**：`K9`（作业 2049318）的 native 残差
`1.237e-5` 是 Schur 口径，而 `K8`（作业 2050917）的
`3.5411e-6` 是 FULL 口径 —— 两者**作用域不同**，不能因为
`1.237e-5 > 3.5411e-6` 就判断 K9 收敛更差。

### 0.3.3 三个残差量再对照

| 名称 | 作用域 | 计算者 | 分母 | 本文标注 |
|---|---|---|---|---|
| 递推残差 | Schur | C++ 递推公式 | `‖b‖`（A 口径） | 未单列 |
| native 真残差 | **Schur**（`compact_fine_`）或 FULL（full 路径） | C++ 重算 | `max(‖b‖,1)`（B 口径） | 标注"仅 native" |
| **独立真残差** | **FULL** | Python/NumPy，两个 parity 全累加 | `‖b‖`（A 口径） | 标注"独立真残差" |

### 0.4 可调参数默认值与 ABI 槽位

CLI 定义：`pyqcu/testing/simple/kupl_multigrid.py:607-648`；
默认常量：`pyqcu/testing/simple/_common.py:63-68`。

| 参数 | CLI | **默认值** | ABI 槽位 |
|---|---|---|---|
| 格点 | `--lat` | `8 8 8 16` | `params[0:4]` |
| MPI 分格 | `--grid` | 自动（x 上一 rank） | `params[5:9]` |
| 质量 | `--mass` | `0.05` | `argv[0]` |
| csw | `--csw` | `1.160920226` | `argv[7]` |
| atol | `--atol` | **`1e-6`** | `argv[1]` |
| max_iter | `--max-iter` | `1000` | `params[13]` |
| 精度 | `--dtype` | `c64` | `params[14]` |
| MG 层数 | `--levels` | `2` | `params[17]` |
| dof | `--dof` | `12 2 2 2 2` | `params[19+8k]` |
| mg_grid（block） | `--mg-grid` | `2 2 2 2` | `params[19+8k]` 同段 |
| **restart** | `--restart` | **`5`**（`0` = 退化为 parity-Schur BiCGStab） | `params[26+8k]` |
| coarse_max_iter | `--coarse-max-iter` | `200` | `params[24+8k]` |
| coarse_tol_factor | `--coarse-tol-factor` | **`1e4`** | 合成进 `argv[3+k]`，见 §0.5 |
| nv_iters | `--nv-iters` | `2` | null-vector 构建迭代数 |
| deflate | `--no-deflate` | 开启 | `params[55]` |
| seed | `--seed` | `42` | `params[52]` |
| 粗层 cache | `--coarse-cache` | 无（按需重建） | `set_ptrs[30+4k]` |

> `--restart 0` 是完全不同的算法：跳过粗层，直接做 parity-Schur BiCGStab。
> 本文中所有 `levels=1, restart=0` 的行都属于这条路径，**不是 multigrid**。

### 0.5 粗层容差的合成公式

`pyqcu/testing/simple/kupl_multigrid.py:188-189`：

```python
argv[ARGV_MG_LEVEL_BASE + level - 1] = float(atol * coarse_tol_factor * (3 ** (level - 1)))
```

即第 `k` 个粗层的相对容差为：

```text
coarse_atol(k) = atol × coarse_tol_factor × 3^(k-1)
```

默认值代入：第 1 粗层 = `1e-6 × 1e4 = 1e-2`，第 2 粗层 = `3e-2`。
**本文提到的 `coarse-tol-factor=10`、`1e6` 等都必须按此式换算成实际 `coarse_atol`**
才能与其它结果比较。

### 0.6 MG 粗修正的阻尼线搜索

`cpp/kupl/qcu/src/lattice_clover_multigrid.cpp:984-985`：

```cpp
static constexpr Real damping_steps[] = {1, 0.5, 0.25, 0.125};
```

每 `restart` 步构造一次粗修正后，按 `1 → 0.5 → 0.25 → 0.125` 依次尝试，
**只接受使细层真残差下降的步长**；四步全失败则回滚解并恢复原残差
（`:1008-1014`）。这解释了本文两处容易误读的现象：

- `restart=50` 在 16 次迭代内只触发 **1 次**修正 → 应表述为"initial deflation +
  fine BiCGStab"，**不是**周期 V-cycle；
- `coarse_calls>0` 只证明粗层算子被调用过，**不证明**修正被接受；
  必须同时看 `KUPL_REQUIRE_MG_CORRECTION=1` 是否放行。

### 0.6.1 `restart` 与 `coarse_calls`/`fine_calls` 的真实关系

读表时最容易误判的三组关系，逐一给出代码依据：

**① `restart` 是"每隔多少次细层迭代尝试一次粗修正"**
（`lattice_clover_multigrid.cpp:220-222`）：

```cpp
levels_[0].correction_interval = params_[kMgLevel1NumRestart] > 0
    ? params_[kMgLevel1NumRestart] : <默认>;   // 来自 --restart
```

- `restart=0` → 不进入 MG 路径，直接走 optimized parity-Schur BiCGStab；
- `restart=N>0` → 每 N 次细层迭代检查一次是否做粗修正。

**② `fine_calls` 是细层算子调用总数，不是迭代次数**
（`lattice_clover_multigrid.cpp:1039`）。BiCGStab 每迭代固定调用 2 次
Schur 算子（`A·p` 与 `A·s`），所以经验关系是：

```text
fine_calls ≈ 2 × 外层迭代数 + 常数
```

代入本文数据验证（迭代数取自同窗口的 BiCGStab 对照行）：

| 窗口 | `cc`/`fc` | 2×迭代 = | `fc` | 偏差 |
|---|---|---:|---:|---:|
| wIV（`restart=20/50` 族） | 9/29 | 32 | 29 | −3 |
| wIII（`restart=8/10` 族） | 19/32 | 32 | 32 | 0 |
| wI（早期 `restart=5` 族） | 29/37 | 36 | 37 | +1 |

偏差 ±3 来自起手/收尾的额外算子调用与不同停止点，因此该关系只能作**量级校验**，
不能当作精确公式。
**因此 `fc` 与"迭代数"列不能互相替代，也不能用 `fc/2` 反推迭代数。**

**③ `coarse_calls` 是粗层算子调用次数，不等于被接受的修正数**
`lattice_clover_multigrid.cpp:982-1014` 的阻尼线搜索每轮最多调用粗层
4 次（`1, 0.5, 0.25, 0.125`），且可能全部失败后回滚。所以：

```text
coarse_calls ≥ 被接受的修正数 × 每轮粗层调用次数
```

**判据**：要证明"真的做了 MG"，必须同时满足
`coarse_calls > 0` **且** `KUPL_REQUIRE_MG_CORRECTION=1` 未抛错
（`:469-472`）。只看 `coarse_calls > 0` 会把 fine-BiCG fallback 误判为 MG。

### 0.7 计时边界

| 字段 | 包住的区间 | 不含 |
|---|---|---|
| `wall_seconds` | 仅 `applyCloverMultigridQcu` 调用 | 输入预绑定、HDF5 读取、独立残差重建 |
| `input_prebind_seconds` | 读输入前的 NUMA/HBM 预绑定 | solver |
| `builder_seconds` | MG cache 构建 | solve |
| `residual_seconds` | 独立真残差重建 | solve |
| `process_seconds` | 整个子进程 | — |

跨后端的绝对时间只在**同一字段**内可比。

### 0.8 内存与显存峰值：字段、口径与缺口

本文原先只引用耗时与残差，**内存/显存数据基本缺失**。本节说明可用字段、口径，
以及仓库内确实没有的部分。

#### 0.8.1 字段定义（CPU 侧）

`pyqcu/testing/simple/kupl_bistabcg.py:76-91` 的 `cpu_memory_snapshot()`：

| 字段 | 单位 | 含义 | 是否峰值 |
|---|---|---|---|
| `memory.max_rss_kib` | KiB | `getrusage(RUSAGE_SELF).ru_maxrss` | **是**：进程生命周期内 RSS 最大值 |
| `memory.resident_kib` | KiB | 报告写入时刻 `/proc/self/status` 的 `VmRSS` | **否**：瞬时值 |
| `memory.omp_threads` | 字符串 | 环境变量 `KUPL_OMP_THREADS` | — |
| `memory.available` | bool | 快照是否成功 | — |

**关键区别**：`max_rss_kib` 是**进程终身峰值**，包含解释器、NumPy、MPI 运行时；
`resident_kib` 是**写入时刻**的驻留。两者不能互换，也不能相减求"库占用"。

#### 0.8.2 字段定义（设备侧）

只有 CUDA/DCU 路径有设备显存：

| 字段 | 定义 | 来源 |
|---|---|---|
| `device.peak_delta_over_baseline_bytes` | `cudaMemGetInfo` 采样峰值相对进程基线的**增量** | `docs/legacy_vs_strict_mg_memory_20261023.md:20-23` |
| `device.peak_bytes` / `baseline_bytes` | 峰值与基线绝对值 | 同上 |

该增量**包含** CUDA context、PyTorch allocator 与 C++ `cudaMallocAsync` 分配，
不只是 PyTorch allocator 统计。

#### 0.8.3 本文已记录的内存数据（CPU 侧，全部等级 A）

来自 `logs/screen-*.txt` 的 `report.json` 原文，`memory.omp_threads` 均为 8
（个别 16）：

| 场景 | `max_rss_kib` | `resident_kib` | 线程 | 对应行 |
|---|---:|---:|---:|---|
| 登录节点 r50 full-residual（`0.9473 s`） | 586,284 | 394,636 | 8 | §3.2 |
| 登录节点 r20 full-residual（`1.3162 s`） | 571,608 | 478,892 | 8 | §3.2 |
| 登录节点 r10 full-residual（`1.0438 s`） | 589,468 | 397,788 | 8 | §3.2 |
| **计算节点 K8（`213.44 s`）** | **579,256** | **486,140** | 8 | §2.1 |
| 登录节点 wI MG 样本（`1.4172 s`） | 574,096 | 481,184 | 8 | §3.1 |
| 登录节点 wIV MG 样本（`0.7051 s`） | 572,844 | 480,228 | 8 | §3.1 |
| 登录节点 `1 1 2 8` MG（`1.8405 s`） | 578,476 | 542,460 | 8 | §3.3 |
| 登录节点 8 rank×16 thr BiCG（`1.8779 s`） | 725,216 | 621,756 | 16 | §3.3 |
| 登录节点 8 rank×16 thr MG（`1.5865 s`） | 821,940 | 676,016 | 16 | §3.3 |

**可读出的量级结论**：

- 单 rank、8 线程下，`24³×64` c64 的**进程终身峰值约 540–590 MiB**；
- 16 线程配置升至 **725–822 MiB**，增量主要来自 OpenMP 线程栈与每线程 scratch；
- 8 线程 vs 16 线程的 `resident` 差约 150–230 MiB，与线程数近似线性。

**不能从这些数得出的结论**：

- 不能拆出"KUPL 库本身占用"——`max_rss` 含 Python、NumPy、mpi4py；
- 不能外推到 32 rank ——只有 8/16 线程的样本；
- 不能当作显存数据——这是 CPU RSS。

#### 0.8.4 仓库内确实缺失的内存/显存数据

下表逐项列出**本文涉及的运行中，哪些有内存数据、哪些没有**，以及缺失原因：

| 环境 / 行 | 内存或显存数据 | 状态 |
|---|---|---|
| E1 计算节点 K1（`y76`，`0.4346 s`） | **无** | `y76` 的 `report.json` **不在本地**（`logs/kupl_hbm_opt_20261106.md` 为转述，且该文件被 gitignore）。仓库内无该次运行的 `memory` 字段 |
| E1 计算节点 K2–K7 | **无** | 同上，`y75`/`y74`/`y61`/`y50`/`y47`/`y27` 的 report 均未入库 |
| E1 计算节点 K8（`213.44 s`） | **有**：579,256 / 486,140 KiB | 屏记录含完整 report |
| E1 计算节点 K9（`164.75 s`） | **有**：1,282,116 / 1,114,496 KiB | `screen-…-253.txt`（4 rank × 32 线程，故显著更大） |
| E1 计算节点 true-MG | **无** | 无 `report.json`（作业被杀） |
| E2 登录节点全部行 | **有**（见 §0.8.3） | 屏记录 |
| E3 鲲鹏控制节点 `25.3325/14.9281/9.8310 s` | **无** | `qcu_mg_performance_ledger_20261029.md:478` 只给墙钟与残差 |
| E4 Xeon 全部行 | **无** | `three_way_benchmark_20261022.md` 原始 JSON 在 gitignore 的 `data/**` |
| E5 V100 全部行 | **无** | 同上；且 `legacy_vs_strict_mg_memory` 明确"benchmark JSON 没有 GPU 型号字段" |
| E6 A100 全部行 | **无设备峰值** | `qcu_mg_performance_ledger_20261029.md` 记录的是 OOM 时刻 `MaxRSS`（Slurm 字段，非设备显存峰值），见 §0.8.5 |
| E7 DCU 全部行 | **无** | `qcu_quda_solver_dcu_l24_20261029.md` 未记 `max_rss`；只有硬件总显存 `17,163,091,968 B` |

#### 0.8.5 A100 / DCU 的"容量墙"数字不是峰值

`qcu_mg_performance_ledger_20261029.md` 里出现的 `35,000,128 KiB`、
`29,410,048 KiB`、`52.68 GiB` 等是 **Slurm `MaxRSS`（主机内存）在 OOM 时刻的值**，
用于描述 cgroup 撞墙，**不是**设备显存峰值，**也不是**正常运行的显存占用。
引用时不得写成"显存峰值"。

DCU 侧只有卡的总显存 `17,163,091,968 B`（`16,368 MiB`）与 L48 cache 逻辑
payload `18,728,091,648 B` 的对比，属于**容量可行性**判断，不是运行峰值。

#### 0.8.6 补齐这些数据需要什么

要在仓库内补齐，需要从远端取回并入库（当前均不在本地）：

1. `data/kupl_hbm_opt_20261106/y76/report.json`（含 `memory` 段）
2. `data/kupl_hbm_opt_20261106/y7{4,5}/report.json`
3. `data/kupl_hbm_opt_20261106/y6{1}/report.json`、`y5{0}/`、`y4{7}/`
4. `data/` 下 `three_way_benchmark_20261022` 的原始 JSON（含 `memory`）
5. A100/DCU 运行若要显存峰值，需在 worker 里加 `peak_delta_over_baseline_bytes`
   采样；当前 DCU L24 报告没有该字段

**在这些文件入库前，本文任何"内存/显存"表述都只能引用 §0.8.3 的 9 行实测值，
其余一律标"无数据"。**

### 0.9 求解测试使用的源场（`src`）是什么

本文所有求解数据的右端项（RHS / source）**不是点源，也不是物理源**，而是
**固定种子的伪随机复数场**。具体生成方式按代码路径分三类，必须逐行区分。

#### 0.9.1 三类源场

| 类型 | 生成方式 | 代码位置 | 用于 |
|---|---|---|---|
| **A. PCG64 正态随机（NumPy，float64→c64）** | `rng = np.random.default_rng(seed)`；`rng.normal(size=(4,3,*lat)) + 1j*rng.normal(...)` 后 `astype(complex64)` | `prepare_case.py:98-100`；`kupl_bistabcg.py:209-212` | 本文**全部** L24 表格行（K/V/X/A/C/D 系列） |
| **B. Torch 正态随机（c64）** | `torch.Generator().manual_seed(seed)`；`torch.complex(randn, randn)` | `_selftest.py:86-93` | 仅 CLI `--selftest` 的 torch/KUPL 层 |
| **C. LIME 文件内源（若存在）** | 直接读 LIME/HDF5 中的 source dataset | `kupl_bistabcg.py:213-218` | 仅当显式传 `--source <file>` 且文件含该 dataset |

**没有点源、wall source、one-end-trick 或 Z2 噪声源**参与本文任何数值。

#### 0.9.2 生产运行的源是哪一种

本文的 K 系列（`cn10579`）与 E 系列作业都在命令行显式传入
`--source .../source_24x24x24x64_c64.h5`：

```bash
# logs/job_y89.sh:52-53
--gauge  "$case_dir/gauge_24x24x24x64_c64.h5"
--source "$case_dir/source_24x24x24x64_c64.h5"
```

这个 `source_24x24x24x64_c64.h5` 由 `prepare_case.py` 生成，即 **A 类**：

```python
# prepare_case.py:98-100
rng = np.random.default_rng(seed)                    # 默认 seed=42
array = (rng.normal(size=(4,3,*target)) +
         1j * rng.normal(size=(4,3,*target))).astype(np.complex64)
```

HDF5 元数据里写死了来源声明（`prepare_case.py:101-105`）：

```json
{"seed": 42,
 "generator": "numpy PCG64 normal float64, cast to complex64",
 "layout": "[spin,color,x,y,z,t]"}
```

**因此：本文所有 L24 求解的 `src` = PCG64 正态随机复数场，seed=42，
实部虚部各自独立采样自标准正态分布，转 complex64。**

#### 0.9.3 种子在本仓库各处的实际取值

| 位置 | seed | 说明 |
|---|---|---|
| `prepare_case.py` 默认 | **42** | 生产 case 的源 |
| `kupl_multigrid.py --seed` 默认 | **42** | 与生产一致 |
| `_selftest.py` torch 层 | 61 | 仅自检 |
| `_selftest.py` KUPL 层 | 61 | 仅自检 |
| `_selftest.py` gauge（torch 层） | 7 | 仅自检 |
| `_selftest.py` gauge（KUPL/CUDA 层） | 59 | 仅自检 |
| `compare_kupl_bistabcg.py` | `args.seed + 1` | 对比脚本，故意错开 |
| `compare_kupl_multigrid.py` | `args.seed + 1` | 同上 |
| `check_clover.py` | 7 | 仅 6×6 块测试 |

**注意 `args.seed + 1`**：对比脚本用 `seed+1` 生成源，是为了避免与同目录下
用 `seed` 生成的 gauge 或其他产物撞种子；这不影响同一脚本内 gauge/source
的一致性。

#### 0.9.4 为什么不用点源

| 考量 | 说明 |
|---|---|
| **残差门敏感性** | 点源只有单个非零分量，`‖b‖` 很小，Krylov 求解退化为近似单模式；无法反映真实多模式收敛行为 |
| **跨后端可比性** | 随机源在 CUDA/KUPL/QUDA 上给出相同的 `‖b‖` 与相近的迭代轨迹，便于同窗 A/B |
| **独立真残差** | 检查的是 `‖b − A·x‖ / ‖b‖`；随机源的 `‖b‖` 有统计意义，点源的 `‖b‖` 只依赖 κ 与单点值 |
| **可复现性** | PCG64 + 固定 seed 保证逐位可复现，且 HDF5 里记录了 generator 与 seed |

**代价**：随机源的收敛不代表物理传播子（propagator）行为；本文所有迭代数
**不能**作为物理计算的迭代数预期。这一点在引用 K/V/A/C 系列迭代数时必须说明。

#### 0.9.5 必须保留的口径警告

1. **源是随机的，不是物理的**。本文的任何迭代数、残差、耗时都不代表用实际
   物理源（如 wall source、点源、或规范场上的物理关联函数）时的结果。
2. **`seed=42` 是约定，不是"正确"**。换种子会改变迭代数与耗时（同分布，
   均值相近但单次不同）；本文所有数据固定 seed=42，跨文献比较时必须确认对方
   的种子与生成器一致。
3. **`source` 与 `gauge` 是独立资产**。gauge 来自真实 LIME 组态
   （`beta6.20_mu-0.2770_ms-0.2310_L24x64_cfg_9950.lime`），source 是自造随机场；
   两者没有物理关联，只是"在真实背景场里解一个随机右端项"。
4. **点名源与文件源的差别**：`--source` 未指定时 `load_source` 会**临时**生成
   一个同种子的 A 类源（`kupl_bistabcg.py:209-212`），不落盘。因此判断一次
   运行用的是"文件源"还是"临时源"，要看命令行是否传了 `--source`。

### 0.10 `restart` 到底是什么

`restart` 是我见过最容易被误读的参数名。它**不是**"求解器重启次数"，也
**不是**"重启迭代"，而是 **MG 粗修正的检查间隔（coarse-correction interval）**。

#### 0.10.1 定义

`cpp/kupl/qcu/src/lattice_clover_multigrid.cpp:220-223`：

```cpp
levels_[0].correction_interval =
    params_[kMgLevel1NumRestart] > 0
        ? params_[kMgLevel1NumRestart]
        : 3;                                  // 未配置时的默认
```

即 `restart = N` 的语义是：**每跑 N 次细层 BiCGStab 迭代，就检查一次是否
插入粗层修正**。它不是"重启"，而是"分段（chunk）"。

#### 0.10.2 为什么叫 restart

名称来自 QUDA 的 `num_restart`（`pyqcu/cuda/_multi_gpu.py:343`、
`pyqcu/torch/solver/_legacy_multigrid.py:72` 的 `num_restart: int = 5`）。
在 QUDA 语境里它表示"Krylov 求解器每多少次迭代重启一次（重置搜索方向）"。
PyQCU 沿用了这个名字，但**在 Legacy-MG 里它实际控制的是粗修正节奏**，
不是方向重置。

**这是命名遗留，不是语义一致**。读本文任何 `restart=` 都必须按 §0.10.1 理解。

#### 0.10.3 三种取值的行为

| 取值 | 行为 | 代码依据 |
|---|---|---|
| `restart = 0` | **完全不进入 MG**，直接调用 optimized parity-Schur BiCGStab | `lattice_clover_multigrid.cpp:1115-1124`：`if (num_levels <= 1 \|\| kMgLevel1NumRestart == 0) { clover_bistabcg(...); return; }` |
| `restart = N > 0` | 每 N 次细层迭代插一次粗修正 | `:386-434` 的主循环：`chunk = min(剩余, N − since_correction)` |
| `restart = N > max_iter` | 只会在开头做 1 次 deflation，之后全是细层 BiCGStab | 主循环在最开始有一次 `:358-385` 的 deflation 分支 |

**因此 `restart=50` 配 `max_iter=50` 时只发生 1 次修正**（本文 §3.2 的
`0.9473 s` 就是这种情况），必须写成"initial deflation + fine BiCGStab"，
**不能**写成周期 V-cycle。

#### 0.10.4 与 `levels` 的联合判据

`restart` 只在 `levels > 1` 时才有意义。C++ 的分派条件是两个一起判断的
（`:1115`）：

```cpp
if (params[kMgNumLevel] <= 1 || params[kMgLevel1NumRestart] == 0) {
    // 退化为 parity-Schur BiCGStab
}
```

所以 **`levels=1` 与 `restart=0` 等价**，都表示"不走 MG"。本文表格里凡写
`levels=1, restart=0` 的行，都是同一条 BiCGStab 路径。

#### 0.10.5 与 `coarse_max_iter` 的区别

两者常被混淆：

| 参数 | 控制什么 | 默认 | ABI 槽位 |
|---|---|---|---|
| `restart` | **多久插一次**粗修正（细层迭代间隔） | 5 | `params[26 + 8k]` |
| `coarse_max_iter` | **每次粗层求解允许多少迭代** | 200 | `params[24 + 8k]` |

一个管"节奏"，一个管"每次粗解多努力"。本文 §6.4 的 `mg-grid` 扫描与
`coarse_max_iter` 扫描是两件独立的事。

#### 0.10.6 本文数据中 `restart` 的实际取值分布

| 取值 | 出现处 | 含义 |
|---|---|---|
| `restart=0` | K1–K9 全部（`levels=1`） | parity-Schur BiCGStab，非 MG |
| `restart=5` | 早期登录节点窗口 wI（`cc=29`） | 每 5 次迭代一次修正 |
| `restart=8` | `r8_tol10_*` 系列 | 每 8 次 |
| `restart=10` | wIII（`cc=19`）、`r10_tol10_t16` | 每 10 次 |
| `restart=20` | wIV（`cc=9`，`r20` 族） | 每 20 次 |
| `restart=50` | wIV（`r50` full-residual，`0.9473 s`） | 只 1 次 deflation |
| `restart=1/2` | `dev99` 期 L8 筛（`0.02242/0.02249 s`） | 最频繁修正 |

**注意 `restart=20` 与 `restart=50` 的 `cc` 都是 9**：因为 `max_iter=50`
时两者都只够做 1–2 次修正，粗层调用次数相同。这解释了为何 `cc` 相同但
`restart` 不同——**`cc` 反映的是实际发生的修正，不是配置值**。

#### 0.10.7 引用时必须写全的四个字段

只写 `restart=10` 是不完整的。可复现的最小集合是：

```text
levels=<N>  restart=<N>  coarse_max_iter=<N>  coarse_tol_factor=<N>
```

因为 `restart` 的实际效果同时取决于 `levels`（是否启用 MG）与
`coarse_max_iter`（每次粗解质量）。本文所有 MG 行已按此四元组标注；
若发现某行只写了 `restart`，该行的重现已不可保证。

### 0.11 `cc` 到底统计什么：多层级时是全层合并

你指出的问题成立：**`cc` 不是"第一粗层调用次数"，而是"所有 level ≥ 1 的算子调用
总次数"**。当 `levels > 2` 时，这一列会把多个粗层加在一起，因此单独看 `cc` 无法
判断修正在第几层发生。

#### 0.11.1 计数代码

`cpp/kupl/qcu/src/lattice_clover_multigrid.cpp:714-718`：

```cpp
void apply_level(int level, std::vector<Complex<Real>> &out,
                 const std::vector<Complex<Real>> &in) {
    MgProfileTimer timer(profile_, level == 0 ? prof_fine_apply_seconds_
                                              : prof_coarse_apply_seconds_);
    ++(level == 0 ? prof_fine_apply_calls_ : prof_coarse_apply_calls_);
    ...
}
```

`level == 0` 计 `fine_calls`；**所有 `level >= 1` 一律累加到同一个
`prof_coarse_apply_calls_`**。没有按层拆分的计数器。

#### 0.11.2 多层级时的实际后果

设 `levels = 3`，一次 V-cycle 从 level 0 递归到 level 2：

| 调用 | `apply_level` 参数 | 计入 |
|---|---|---|
| 细层 | `level=0` | `fine_calls` +1 |
| 第 1 粗层 | `level=1` | `coarse_calls` +1 |
| 第 2 粗层 | `level=2` | **`coarse_calls` +1（同一计数器）** |

更准确地说，`cc` 的增量按 `bicgstab_chunk` 内的 `apply_level` 调用次数累计
（`lattice_clover_multigrid.cpp:817`、`:849` 各一次/迭代），**每个 level ≥ 1 都各自
贡献**。所以：

```text
cc(levels=2) = Σ_{level=1}  该层 apply_level 次数
cc(levels=3) = Σ_{level=1..2} 各层 apply_level 次数   → 更大
```

**同一份 `cc` 数值在 `levels=2` 与 `levels=3` 下含义不同，不能跨层数比较。**

**实测边界**：本地归档的 L24 数据里 `levels` 只有 `1`（32 条）与 `2`（54 条），
没有 `levels≥3` 的记录。因此本文所有 `cc` 都只来自**单一粗层**，
`cc == restrict_calls == prolong_calls` 恒成立（见 §0.11.3 的实测配对表）。
多层合并的风险**在原理上成立，但不影响已归档数据**。

#### 0.11.3 实测佐证：`cc` 与 `restrict_calls` 的关系

本文所有数据都是 `levels=2`，所以 `cc == restrict_calls == prolong_calls`：

| `coarse_calls` | `restrict_calls` | `prolong_calls` | `coarse_correction_calls` | 样本数 |
|---:|---:|---:|---:|---:|
| 9 | 1 | 1 | **1** | 34 |
| 19 | 2 | 2 | **2** | 18 |

如果 `levels=3`，这三个数会分叉：`restrict_calls` 每层各 +1（一次 V-cycle 加 2），
而 `coarse_calls` 也会加 2 —— **但 `coarse_correction_calls` 仍按"修正事件"计数**，
不会因为层数而翻倍。

#### 0.11.4 五个计数器各自统计什么

`lattice_clover_multigrid.cpp:1080-1084` 定义了 5 个独立计数器：

| 字段 | 计数器 | 统计对象 | 递增位置 |
|---|---|---|---|
| `fine_calls` | `prof_fine_apply_calls_` | **level 0** 的算子应用次数 | `:718`（`level==0`） |
| `coarse_calls` | `prof_coarse_apply_calls_` | **level ≥ 1 全部层合并**的算子应用次数 | `:718`（`level!=0`） |
| `restrict_calls` | `prof_restrict_calls_` | 限制算子调用次数（每层各一次） | `:744` |
| `prolong_calls` | `prof_prolong_calls_` | 延拓算子调用次数（每层各一次） | `:756` |
| `coarse_correction_calls` | `prof_coarse_correction_calls_` | **修正事件**次数（与层数无关） | `:958` |

**关键区别**：

- `coarse_calls` 是**算子调用次数**（层数越多越大）；
- `coarse_correction_calls` 是**修正尝试次数**（每次尝试算 1，不论内部跑几层）。

#### 0.11.5 为什么本文的 `cc` 看起来"大"

以 wI 窗口 `cc=29, fc=37` 为例：

```text
fc = 37   细层算子调用 37 次
cc = 29   粗层算子调用 29 次（levels=2，所以等于 restrict_calls=29）
correction_calls 未记录在该行
```

29 次粗调用对应"每 5 次细层迭代做一次修正"（`restart=5`）+ 阻尼搜索的重复尝试。
`fc ≈ 2×迭代` 的关系在 §0.6.1 已给；`cc` 与迭代数没有简单线性关系。

#### 0.11.6 引用 `cc` 时必须同时给出的字段

单写 `cc=19` 无法解读。最小完整集：

```text
levels=<N>  cc=<N>  fc=<N>  restrict_calls=<N>  coarse_correction_calls=<N>
```

其中 `levels` 决定 `cc` 是否被多层合并。**本文全表都是 `levels=2`**，因此
`cc == restrict_calls == prolong_calls` 恒成立；若将来出现 `levels≥3` 的数据，
必须显式标注，不能与本文 `cc` 直接比较。

#### 0.11.7 本文数据的一个限制

本地屏记录中**只保存了 5 个计数器里的 4 个**（`coarse_correction_calls` 在部分
hardcopy 中被折行截断）。因此本文表格只能引用 `cc`/`fc`，无法给出
"每次修正在第几层发生"。要完整还原多层级行为，需要在后续运行中：

1. 打开 `PYQCU_KUPL_MG_PROFILE=1`；
2. 保存**完整**的 `MG_PROFILE` 行（不截断）；
3. 对 `levels≥3` 单独标注。

## 1. 为什么没有单一数值

| 维度 | 取值 | 对耗时的影响 |
|---|---|---|
| 后端 | Torch / CUDA QCU / KUPL QCU / QUDA | GPU 与 CPU 差 1–2 个数量级 |
| 硬件 | V100 / A100 / DCU gfx906 / Xeon x86 / 鲲鹏 920 | 同类内也差数倍 |
| 精度 | c64 / c128 | c128 约为 c64 的 2–3 倍 |
| 求解路径 | BiCGStab / Legacy-MG / Strict-MG | MG 不必然更快 |
| 并行 | 1–32 ranks × 1–32 threads | 强/弱扩展均非单调 |
| 分格 | 10+ 种合法 `grid` | 同一 rank 数可差 2 倍 |
| 计时边界 | solve-only / 含 setup / 含 I/O | 可差数小时（cache builder） |

**`24³×64` 在本项目被定为 smoke 格点**，正式大格下限是 `32³×64`
（`docs/qcu_mg_performance_ledger_20261029.md` §1.1）。因此本文所有数值
**不得**用于正式大格加速比。

## 2. 鲲鹏计算节点 `cn10579`（主证据）

硬件：SVE2，16 CPU NUMA + 16 片上 NUMA，cpuset 上界 577，每域 38 核。
详见 [`kupl_kunpeng_progress_20261110.md`](kupl_kunpeng_progress_20261110.md)。

物理参数：c64、mass `-0.277`、kappa `0.13430029546065003`、csw `1.160920226`、
`atol=1e-4`、`max_iter=50`。构建 `build-native-lto`。

### 2.1 BiCGStab（`levels=1, restart=0`）

> 等级：K1–K7 为 **B**（阶段日志转述）；K8、K9 为 **A**（屏记录含完整 report 原文）。

| # | ranks × thr | grid | solve (s) | 迭代 | 独立真残差 | 验收门 | 样本 | 作业 | 等级 |
|---|---|---:|---:|---:|---|---:|---:|---|---|
| K1 | **32 × 12** | `1 1 1 32` | **0.4345627510** | 16 | `3.5412238955e-6`（FULL） | `2.44140625e-4` | 1 | y76 | B |
| K2 | 32 × 12 | `1 1 1 32` | 0.4472 | 16 | `3.54e-6` | 同上 | 1 | y75 | B |
| K3 | 32 × 12 | `1 1 1 32` | 0.5719 | 18 | `7.39e-7` | 同上 | 1 | y74 | B |
| K4 | 32 × 12 | `1 1 1 32` | 0.4877 | — | — | — | 1 | y61 剖析轮 | B |
| K5 | 32 × 12 | `1 1 1 32` | 0.5093 | — | — | — | 1 | y50 | B |
| K6 | 32 × 12 | `1 1 1 32` | 0.5126 | — | — | — | 1 | y47 | B |
| K7 | 历史最低（安静节点） | `1 1 1 32` | 0.475 | — | — | — | 1 | y27 | B |
| K8 | 16 × 8 | `1 1 1 16` | **213.44090568200045** | 16 | `3.5410827126526993e-6`（FULL） | `2.44140625e-4` | 1 | 2050917 | A |
| K9 | 4 × 32 | `1 1 1 4` | **164.7499241** | 16 | `1.237e-5`（**仅 native，Schur 作用域**） | 未做独立残差 | 1 | 2049318 | **A** |

**K9 的 grid 与 rank/thread 直接证据**：`logs/job_y89.sh` 第 12–13 行
`#DSUB -N 4` / `#DSUB -rpn 4`、第 40 行 `--grid 1 1 1 4`、第 37 行
`export OMP_NUM_THREADS=32`；report 原文见 `logs/screen-73056-ss-20261106-253.txt`
（`"omp_threads": "32"`、`"residual_source": "KUPL native fail-closed solve;
independent check skipped"`）。**K8 的 grid 同样有直接证据**：
`job_y91_manualgeom.sh` 第 53 行 `--grid 1 1 1 16`、`#DSUB -N 16`、8 线程。

> **残差口径警告**：K9 的 `1.237e-5` 是 **Schur 单奇偶** native 残差，
> K8 的 `3.5411e-6` 是 **FULL 完整场**独立残差。两者作用域不同（见 §0.3.2），
> 不能直接比较大小来判断收敛质量。

**内存数据**（`memory` 段，见 §0.8）：

| 行 | `max_rss_kib` | `resident_kib` | 说明 |
|---|---:|---:|---|
| K1（`y76`） | **无** | **无** | `y76/report.json` 未入库 |
| K2–K7 | **无** | **无** | 同上 |
| K8 | **579,256**（565.7 MiB） | 486,140（474.7 MiB） | 16 rank × 8 线程，屏记录原文 |
| K9 | **1,282,116**（1,252.1 MiB） | 1,114,496（1,088.4 MiB） | 4 rank × 32 线程，线程多故更大 |

K9 的 `max_rss` 约为 K8 的 **2.2 倍**，与线程数 32 vs 8 的方向一致，但 K9 只有
4 个 rank，故不能简单按线程线性外推。

**K1–K7 属同一 32×12 几何**，彼此可比；**K8/K9 是不同几何，不可与 K1 相减或相除
后归因于 HBM**。按每 rank 局部体积线性外推，K8/K9 的理想弱扩展预测约为
`1.74 s` 与 `3.48 s`，实测高出 **123×** 与 **47×**；该差距来自几何相关路径
（标量边界站点数、集合通信规模、16 CPU/rank 性能悬崖），**尚未分解**。

同二进制同配置噪声带 **±8%**（0.492–0.576 s）；争用时段可升至 0.85–0.93 s。
小于 10% 的差异一律视为噪声。

### 2.2 true-MG（`levels=2, restart>0`）

| ranks × thr | grid | solve (s) | 结论 | 等级 |
|---|---|---:|---|---|
| 16 × 8 | `1 1 1 16` | **BLOCKED** | 作业 2050917 达 787 s 无 `report.json`，TERM kill-job 2051058，`jobExitCode=143` | A |

**计算节点 true-MG 至今没有任何有效数值。** 同 job 的 BiCGStab（K8）成功，
MG case 在 5 分钟审计时 CPU 利用率约 83%（仍在计算），10 分钟门禁内无输出。

### 2.3 相对基线

原始基线 `89.13 s` 为 4 ranks×32 threads、无 NUMA 映射、HBM 未生效、OpenMP
线程挤单核的历史运行（**等级 C，本地无 report.json**）：

| 配置 | 相对 89.13 s | 距 0.1 s 目标 |
|---|---:|---:|
| K1 `0.4346 s` | **205.1×** | 4.35× |
| K7 `0.475 s` | 187.6× | 4.75× |
| K4 `0.4877 s` | 182.8× | 4.88× |
| K5 `0.5093 s` | 175.0× | 5.09× |
| K6 `0.5126 s` | 173.9× | 5.13× |
| K3 `0.5719 s` | 155.8× | 5.72× |

该倍率混合了**四项修复的合计效果**（`set_mempolicy` 掩码对齐、逐线程绑核、
NUMA 映射、rank-local slab），不是单项优化收益。

## 3. 鲲鹏登录节点（辅助筛选，非生产结论）

硬件：Kunpeng-920，无 SVE，4 NUMA（实测 `numa_nodes=4`，560 条记录），
共享机（`load average≈91/48`、约 106 用户），**无片上内存域**。
同一 `24³×64` c64、`atol=1e-4`、`max_iter=50`。

### 3.1 同窗 A/B（6 个独立窗口，每窗 n=4）

> 等级 **A**（本次从 `logs/screen-73056-ss-20261106-*.txt` 重算）。

**切窗规则**：以每个 `bicgstab_1` 标签为窗口起点，收满 4 个 BiCGStab + 4 个 MG
样本即闭窗。按此规则得到 **6 个完整窗口**，且与阶段日志声明的
`r10 / r20 / r50 / early` 四组中位**精确对应（差值 <1e-3）**。

| 窗口 | 屏 | `cc`/`fc` | BiCG 4 样本 (s) | MG 4 样本 (s) | BiCG 中位 | MG 中位 | 比值 | MAD (BiCG/MG) | 对应阶段日志 |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| w1 | 123 | 29/37 | 2.2581948, 1.3411769, 1.1680136, 1.1376181 | 1.4172308, 1.3237246, 1.2846084, 1.3844923 | 1.2545952125 | 1.3541084125 | 0.9265 | 0.10178 / 0.04675 | **early**（MG 慢 7.9%） |
| w2 | 126 | 19/34 | 1.3857920, 1.3902571, 1.2005569, 1.2058933 | 0.8379837, 1.1596560, 0.7961195, 1.3948515 | 1.2958426615 | 0.9988198766 | 1.2974 | 0.09218 / 0.18177 | —（未进入阶段日志） |
| w3 | 127 | 19/32 | 1.5286182, 1.2658674, 1.1714726, 1.5592493 | 0.7352905, 1.0648872, 0.7651099, 0.8144017 | **1.3972427972** | **0.7897557600** | **1.7692** | 0.14669 / 0.03956 | **r10** |
| w4 | 128 | 19/32 | 1.2080937, 1.3080575, 1.0484091, 1.0884057 | 0.9222270, 0.8828619, 1.0413504, 0.7337528 | 1.1482496969 | 0.9025444695 | 1.2722 | 0.07984 / 0.07924 | —（未进入阶段日志） |
| w5 | 191 | 9/29 | 1.6146743, 1.2736822, 1.1525702, 1.1878842 | 0.6422034, 0.6819733, 0.8924069, 0.7283181 | **1.2307832352** | **0.7051456925** | **1.7454** | 0.06056 / 0.04306 | **r20** |
| w6 | 192 | 9/29 | 1.3008456, 1.0947032, 1.0890289, 1.0303652 | 0.7345825, 0.7153401, 0.6931414, 0.6466229 | **1.0918660681** | **0.7042407769** | **1.5504** | 0.03217 / 0.02072 | **r50** |

配置均为 `24³×64` c64、16 ranks×8 threads、`grid=1 1 1 16`、同 gauge/RHS、
`atol=1e-4`、`max_iter=50`；全部 `verify_status=0`。

#### 3.1.1 窗口归属校正（本轮核验）

先前版本写"只有 4 个去重窗口"，并把 `screen-127` 第二窗与 `screen-128` 第一窗
合并。逐条重解后确认该合并是**切窗方法的错误**，不是样本重复：

| 屏 | 记录数 | BiCG | MG | 实际包含的窗口 |
|---|---:|---:|---:|---|
| 123 | 8 | 4 | 4 | w1（完整） |
| 126 | 8 | 4 | 4 | w2（完整） |
| **127** | **14** | **7** | **7** | w2 尾（4+4）+ **w3 全 8 个** |
| **128** | **14** | **7** | **7** | w3 尾（4+4）+ **w4 全 8 个** |
| 191 | 8 | 4 | 4 | w5（完整） |
| **192** | **14** | **7** | **7** | w5 尾（4+4）+ **w6 全 8 个** |

`127`、`128`、`192` 各含**两个相邻窗口的尾部/头部**，hardcopy 部分重叠但窗口
本身互不相同。决定性证据：`screen-127` 的 `mg_1` 的 `fine_calls` 从 34（w2）
变为 32（w3）；`screen-128` 的 `bicgstab_1` 出现 `1.2080937`，而 `screen-127`
的 `bicgstab_1` 是 `1.5286182`——同一运行不可能有两个不同值。

**结论改为：6 个窗口，其中 w2 与 w4 未进入阶段日志。** 先前"4 个窗口"作废。
先前表格里记为 wIII 的那一行（BiCG 1.2370 / MG 0.8683）是把 w3 尾与 w4 头
混合算出的**错误中位**，正确值应为 w3（1.3972 / 0.7898）或 w4（1.1482 / 0.9025）。

### 3.2 full-residual 伴随运行

| 配置 | solve (s) | 迭代 | 独立真残差 | 验收门 | 状态 |
|---|---:|---:|---:|---:|---|
| `restart=50` | **0.9473317482** | — | `5.5308248713e-5` | `2.44140625e-4` | PASS |
| `restart=20` | 1.316182 | — | `5.5308248713e-5` | 同上 | PASS |
| `restart=10` | 1.0437915269 | — | `6.8887275716e-5` | 同上 | PASS |
| 阻尼 gate（单 rank） | 7.52818155894056 | — | 残差未完成（Python 超 180 s） | — | 部分 |

**`restart=50` 只执行 1 次 coarse correction**，应描述为 "initial deflation +
fine BiCGStab"，**不是**周期 V-cycle。

**口径更正（本轮核验）**：`0.9473 s` 不是"登录节点最低绝对耗时"——§3.1 的 wIV
窗口中 MG 单样本已有 `0.6422 / 0.6820 / 0.7283 / 0.8924 s`（中位 `0.7051 s`）。
`0.9473 s` 的正确表述是**"登录节点最低的、带独立真残差验证的 full-residual
伴随运行耗时"**（`verify_status=0`、残差 `5.53e-5` PASS）。两者差一个验证层级：
A/B 样本只有 native `verify_status=0`，没有独立全算子残差。原始文档
`logs/kupl_hbm_opt_20261106.md:504` 写 "lowest observed absolute" 是该
时点（r50 窗口）的表述，在汇总全部窗口后不再成立。

### 3.3 分格扫描（每点 n=1）

| grid | ranks × thr | solve (s) | 判读 |
|---|---|---:|---|
| `1 1 1 16` | 16 × 8 | **1.68634** | 最优 |
| `1 1 2 8` | 16 × 8 | 1.86214 | — |
| `1 1 2 16` | 32 × 4 | 1.94527 | — |
| `1 1 4 8` | 32 × 4 | 2.01389 | — |
| `1 2 2 8` | 32 × 4 | 2.03263 | — |
| `1 1 1 8` | 8 × 16 | 2.07968 | — |
| `1 1 1 32` | 32 × 4 | 2.10423 | — |
| `2 2 2 4` | 32 × 4 | 2.12254 | — |
| `1 2 2 4` | 16 × 8 | 2.37107 | — |
| `1 1 1 4` | 4 × 32 | 3.27704 † | 最慢 |

全部通过 `KUPL_REQUIRE_MG_CORRECTION=1` fail-closed 门且 `verify_status=0`。
单样本，柱间差异同时含共享负载漂移。

† **`3.27704`（`mg-grid 2 2 2 8`、4×32、`grid 1 1 1 4`）只有文档来源
（`logs/kupl_hbm_opt_20261106.md:334`），在全部 `logs/screen-*.txt` 中
搜索不到该数值**。屏记录里 4-rank 的 MG 只有 `2.26409`（`e2_tblock8`）与
`3.42403`/`2.42426`（L8/L16 的 `r4_t`），无 `3.27704`。该行等级应视为 **C**，
其余九行有屏记录支撑（**A**）。

### 3.4 时间块与 cache

| 时间块 | cache 字节 | builder (s) | 合法最大 `grid_t` | solve 比较 |
|---:|---:|---:|---:|---|
| 1 | 143,335,424 | 119.57 | 32 | 最快 |
| 4 | 99,540,992 | 69.33 | 8 | 慢于块 1 |
| 8 | 92,241,920 | 63.59 | 4 | 慢于块 1 |

build 节省不抵粗层同步成本增加。

### 3.5 L24 true-MG 网格矩阵（`mg-grid` 维度）

| mg-grid | ranks × thr | grid | solve (s) |
|---|---|---:|---:|
| `2 2 2 1` | 16 × 8 | `1 1 1 16` | 1.77820 |
| `2 2 2 1` | 32 × 4 | `1 1 1 32` | 1.51917 † |
| `2 2 2 1` | 32 × 4 | `1 1 2 16` | 1.75694 |
| `2 2 2 4` | 16 × 8 | `1 1 2 8` | 2.06250 |
| `2 2 2 8` | 32 × 4 | `2 2 2 4` | 1.64822 |

† 该值与同表的 `1.60861 s` **不得作为 true-MG 证据**：
`PYQCU_KUPL_MG_PROFILE=1` 显示 `coarse_calls=0` 并跟随
`fallback=fine_bicgstab`，它们是 fine-BiCG 回退耗时。

## 4. x86 Intel Xeon E5-2696 v3（18 核 / 1 socket / 1 NUMA）

`24³×64` c64、19 迭代、独立真残差 `4.01e-7`。构建 `-O3 -march=native`（另有 LTO）。

### 4.1 强扩展（`restart=0`）

| # | ranks × thr | grid | solve (s) | 相对 1×4 | 效率 | 等级 |
|---|---|---:|---:|---:|---:|---|
| X1 | 1 × 4 | `1×1×1×1` | 19.229 | 1.000× | — | C |
| X2 | 1 × 16 | `1×1×1×1` | 9.397 | 2.046× | — | C |
| X3 | 2 × 4 | `1×2×1×1` | 11.490 | 1.674× | — | C |
| X4 | 4 × 4 | `1×4×1×1` | 7.842 | 2.452× | 61.3% | C |
| X5 | **8 × 2** | `1×4×2×1` | **7.532** | 2.553× | 63.8% | C |
| X6 | 16 × 1 | `1×4×4×1` | 8.299 | 2.317× | 57.9% | C |
| X7 | 16 × 1 | `1×4×2×2` | 8.406 | 2.288× | — | C |

同窗口 GPU 对照（**不可与 CPU 混算**）：CUDA `0.147801 s`（18 it）、
QUDA `0.156680 s`（19 it）、CUDA fine-only MG `0.173372 s`（20 it）。

### 4.2 非 2 幂 rank 与分格

| ranks × thr | grid | solve (s) | 使用核数 |
|---|---|---:|---:|
| 6 × 3 | `1×2×3×1` | 8.249 | 18 |
| 8 × 2 | `1×4×2×1` | 7.532 | 16 |
| 9 × 2 | `1×3×3×1` | 10.640 | 18 |
| 12 × 1 | `1×3×2×2` | 7.830 | 12 |
| 18 × 1 | `1×3×3×2` | 11.276 | 18 |
| 12 | `1×4×3×1` | 7.796 | 18 |
| 12 | `1×6×2×1` | 8.776 | 18 |
| 16 | `1×2×2×4` | 8.702 | 18 |

### 4.3 线程绑定与构建

| 配置 | 并行参数 | solve (s) | 结论 |
|---|---|---|---|
| 默认 `--bind-to none` | 8×2 | 7.2–7.4 | 生产默认 |
| `OMP_PROC_BIND=spread/close` + `OMP_PLACES=cores` | 8×2 | **65–67** | 约 **8.5× 恶化** |
| `OMP_WAIT_POLICY=active` | 8×2 | 8.9–9.9 | 无收益 |
| 默认库 `-O3 -march=native` | 8×2 | 7.556 | 5 次交替 median |
| **LTO** `-flto -fno-semantic-interposition` | 8×2 | **7.196** | 快 4.8%，解逐位一致 |

## 5. 鲲鹏控制节点（2 NUMA，非 HMPI/片上内存）

> 等级 **C**（仅 `qcu_mg_performance_ledger_20261029.md` 转述，本地无 raw JSON）。

`24³×64` c64、`restart=0`、独立真残差约 `4.018e-7`。**该路径是 fine-only
parity-Schur BiCGStab（`coarse_source="unused: restart=0"`），不是 MultiGrid**，
不得进入 MG 加速比。

| ranks | solve (s) | 迭代 | 独立真残差 |
|---:|---:|---|---|
| 1 | 25.3325 | **未记录** | `4.0185e-7` |
| 2 | 14.9281 | **未记录** | `4.0174e-7` |
| 4 | 9.8310 | **未记录** | `4.0178e-7` |

> **口径更正（本轮核验）**：迭代数在 `qcu_mg_performance_ledger_20261029.md:478`
> 中**没有给出**，先前填的 `20` 是无来源推测，已改为"未记录"。

该批 `restart=5` true-MG **无有效数值**：KUPL Galerkin verifier 报告
`projection=2.11372e-07`、`galerkin=0.0886153` > `tolerance=0.002`，作业不可作为
PASS 或性能数据。

## 6. `dev99`–`dev101` 时期（硬件未完整记录）

`24³×64` c64/c128、5 次稳态 median、质量 `-0.277`、csw `1.160920226`：

| # | 精度 | 配置 | solve (s) | 迭代 | 独立真残差 |
|---|---|---|---:|---:|---|
| D1 | c64 | r1 baseline | 9.858334 | 20 | `4.125306e-7` |
| D2 | c64 | r1 spin6 | 7.830318 | 20 | `4.131546e-7` |
| D3 | c64 | r1 spin6 + inv6 | 7.769662 | 20 | `4.121851e-7` |
| D4 | c64 | r2 baseline | 7.960636 | 20 | `4.120797e-7` |
| D5 | c64 | r2 spin6 | 7.452968 | 20 | `4.128289e-7` |
| D6 | c64 | r2 spin6 + grid-x | 8.108518 | 20 | `4.124767e-7` |
| D7 | c64 | r2 spin6 + inv6 | 6.351769 | 20 | `4.126025e-7` |
| D8 | c128 | r1 baseline | 20.196165 | 23 | `6.962408e-8` |
| D9 | c128 | r1 spin6 | 16.116355 | 23 | `6.962408e-8` |
| D10 | c128 | r1 spin6 + inv6 | 15.592787 | 23 | `6.962408e-8` |
| D11 | c128 | r2 baseline | 14.449964 | 23 | `6.962408e-8` |
| D12 | c128 | r2 spin6 | 13.106006 | 23 | `6.962408e-8` |
| D13 | c128 | r2 spin6 + inv6 | 11.472949 | 23 | `6.962408e-8` |

c128 的 `restart=0` 是 parity-Schur BiCGStab 路径，**不是** c128 coarse V-cycle。

## 7. V100（WSL2，sm70）

> 等级 **C**（`three_way_benchmark_20261022.md` 与 `profiler_solver_performance_history.md` 转述；原始 JSON 在 gitignore 的 `data/**`）。

`24³×64`、P01 参数集（真实 LIME `cfg_9950`）、c64、冷启动：

| # | 路径 | solve (s) | 迭代 | 独立真残差 | 相对 QUDA |
|---|---|---:|---:|---|---:|
| V1 | CUDA QCU `applyCloverBistabCgQcu` | **0.147801** | 18 | `6.709e-7` | 1.060× |
| V2 | QUDA Clover BiCGStab | 0.156680 | 19 | `4.207e-7` | 1.000× |
| V3 | CUDA fine-only Legacy MG | 0.173372 | 20 | `4.149e-7` | 0.904× |
| V4 | CUDA，sm60+PTX 强制 | 0.146966 | 18 | `6.709e-7` | — |
| V5 | CUDA，原生 sm70 | 0.147531 | 18 | — | — |
| V6 | CUDA 早期低负载窗口 | 0.121856 | 18 | `6.709e-7` | — |
| V7 | QUDA 早期低负载窗口 | 0.137938 | 19 | `4.180e-7` | — |
| V8 | CUDA 05:19 窗口 | 0.149015 | 18 | `6.709e-7` | — |

V4/V5 差异仅 `0.38%`，小于样本散布，**不得宣称原生 SASS 有可测收益**。
V6–V8 是不同窗口，不可与本表主体混算。

`24³×64` 的 Legacy 深度与 MPI（c64、`nvec=12`、block `2⁴`）：

| ranks / levels | 迭代 | solve (s) | native 残差 | 独立残差 | 状态 |
|---|---:|---:|---|---|---|
| 1 / 2 | 20 | **0.2151** | `2.336648e-7` | `2.604745e-7` | PASS |
| 1 / 3 | 20 | 0.9324 | `2.335826e-7` | `2.605036e-7` | PASS |
| 1 / 4 | 20 | 0.9222 | `2.336648e-7` | `2.604566e-7` | PASS |
| 2 / 2 | 20 | 1.0505 | `2.342344e-7` | `2.453464e-7` | PASS |
| 2 / 3 | 20 | 6.9275 | `2.342240e-7` | `2.453817e-7` | PASS |
| 2 / 4 | 20 | 8.0346 | `2.342462e-7` | `2.453665e-7` | PASS |

3/4 层修复了正确性但**没有迭代收益**；`MultigridDepthWarning` 对 L24 3L/4L
明确告警。Legacy 生产默认仍为 2 层。

CUDA Legacy-MG 优化阶段（同一真实 LIME，c64，20 迭代，`n_vcycles=0`）：

| 阶段 | 优化 | solve (s) | 相对上一步 |
|---|---|---:|---:|
| S1 | 自适应门控 | 0.3903 | — |
| S2 | `cudaMemcpyAsync` | 0.3099 | −20.6% |
| S3 | `mg_give_rx` 合并访存 | 0.2423 | −21.7% |
| S4 | lazy workspace | 0.1954 | −19.4% |
| S5 | Clover I/O 融合 | 0.1837 | −6.0% |
| S6 | Clover subtract 融合 | 0.1802 | −1.9% |
| S7 | Wilson+Clover 融合 | 0.1721 | −4.5% |
| S10 | 最终合并 10 样本 | **0.1720** | — |
| S13 | gather/W=8 + coarse tile | 0.1636 | — |
| S17 | 6×6 Clover 结构门控 | 0.148100 | — |
| S21 | fine 成对归约 | 0.13700 | — |
| S22 | fine 单点积 | **0.13295** | — |
| S24 | fine-only baseline | 0.132445 | — |
| S24-DEFLATE | BiCG + 一次 deflate | 0.172463 | 残差更低但慢 30% |

## 8. A100-PCIE-40GB

> 等级 **C**（`qcu_mg_performance_ledger_20261029.md` 转述）。

`24³×64` c64、`applyCloverMultigridQcu`、levels 2、`dof=[12,12]`、block `2⁴`、
`restart=20`、`atol=1e-6`：

| # | ranks / grid | 样本类型 | solve (s) | 迭代 | 残差 | 状态 |
|---|---|---|---:|---:|---|---|
| A1 | 1 / `1×1×1×1` | smoke 单样本 | 0.1587396 | — | `2.8791103e-7` | PASS |
| A2 | 1 / `1×1×1×1` | smoke 单样本 | 0.1158422 | — | `2.8791103e-7` | PASS |
| A3 | 2 / `2×1×1×1` | baseline 单样本 | 0.1757574 | — | `2.8837007e-7` | PASS |
| A4 | 2 / `2×1×1×1` | candidate 单样本 | 0.1469283 | — | `2.8837007e-7` | PASS |
| A5 | 2 / `2×1×1×1` | candidate 单样本 | 0.1459943 | — | `2.8837007e-7` | PASS |
| A6 | 2 / `2×1×1×1` | baseline 单样本 | 0.1498923 | — | `2.8837007e-7` | PASS |
| A7 | 4 / `2×2×1×1` | baseline 单样本 | 0.1683383 | — | `2.8844304e-7` | PASS |
| A8 | 4 / `2×2×1×1` | candidate 单样本 | 0.1660481 | — | `2.8844304e-7` | PASS |
| A9 | 4 / `2×2×1×1` | candidate 单样本 | 0.1665799 | — | `2.8844304e-7` | PASS |
| A10 | 4 / `2×2×1×1` | **ABBA baseline (n=6)** | **0.1554899**（MAD 0.0103） | — | — | **INVALID** |
| A11 | 4 / `2×2×1×1` | **ABBA candidate (n=5)** | **0.1682335**（MAD 0.0155） | — | — | **INVALID** |

A1–A9 为 `warmup=0, repeat=1` 的**单样本 smoke**，不是统计性能。
A10/A11 虽然样本数更多，但该阶段一次 candidate 运行在约 4 秒 segfault、
MPI 运行时随之崩溃，**正式结论必须标记 INVALID**。

## 9. DCU（Pre-Wukong D160A1，gfx906，16 GB）

> 等级 **C**（`qcu_quda_solver_dcu_l24_20261029.md` 转述；硬件审计原文件不在工作树）。

`24³×64` c64、CUDA QCU Legacy-MG、20 迭代、`n_vcycles=1`、`coarse_iters=8`、
1 warmup + 5 timed cold solves：

| # | ranks / grid | median (s) | MAD (s) | 样本 | 强扩展 | native 残差 |
|---|---|---|---:|---|---:|---|
| C1 | 1 / `1×1×1×1` | **0.301689520** | 0.002222990 | 5 | 1.000× | `2.582429e-7` |
| C2 | 2 / `2×1×1×1` | 0.461907550 | 0.001445631 | 5 | 0.653×（32.66%） | `2.582687e-7` |
| C3 | 4 / `2×2×1×1` | 1.024669448 | 0.004110831 | 5 | 0.294×（7.36%） | `2.581189e-7` |

**只有 native full-operator 残差**，按任务要求跳过独立真残差。
`24³×64` 单次 solve 仅 0.3–1.0 s，通信与固定开销占主导，**不得据此推断更大格点
的并行效率**。QUDA 对照 `BLOCKED-RUNTIME`（HIP `hipMemcpy` SIGSEGV）。

硬件：`gfx906`、64 CU、4 SIMD/CU、wavefront 64、最高 1.7 GHz、
`17,163,091,968 B` 显存；节点 Hygon C86 7185 32-core / 4 NUMA / 126,500 MiB。

### 9.1 与各环境硬件配置的对照

各环境的完整硬件字段（CPU 型号、核数、NUMA 域、MPI/编译器版本、显存、构建选项）
与带宽实测口径见
[`qcu_l24_environment_bandwidth_20261111.md`](qcu_l24_environment_bandwidth_20261111.md)。

**只有 E1 `cn10579` 有实测带宽分母**（LX2 STREAM FP64 Triad）。把本文的 solve 时间
按同一流量模型（`64.97 GB` / 34 次 Schur apply）折算为有效带宽：

| 环境 | 配置 | solve (s) | 有效带宽 (GB/s) |
|---|---|---:|---:|
| E1 cn10579 | 32r × 12t（K1） | **0.4346** | **149.51** |
| E1 cn10579 | 16r × 8t（K8） | 213.4409 | 0.30 |
| E1 cn10579 | 4r × 32t（K9） | 164.7499 | 0.39 |
| E2 登录节点 | 16r × 8t（§3.2 r50） | 0.9473 | 68.58 |
| E4 Xeon | 8r × 2t（X5） | 7.532 | 8.63 |
| E5 V100 | CUDA BiCG（V1） | 0.1478 | 439.59 |
| E6 A100 | 1r 最佳 smoke（A2） | 0.1158 | 560.86 |
| E7 DCU | 1r（C1） | 0.3017 | 215.36 |

> 图表文件 `kupl_kunpeng_figures_20261110.tex` 曾把 x 轴坐标写作
> `E4x32` / `E16x8`（意为 4 ranks×32 threads / 16 ranks×8 threads），
> 与环境代号 `E1–E7` 撞字母。该写法已在本轮全部替换为 `r4t32` / `r16t8`，
> PDF 已重编译（`Overfull=0`/`Float=0`/`Missing=0`），因此**当前 PDF 的 x 轴
> 标签已是 `r4t32`/`r16t8`**；只有引用本轮之前的旧版图表时才会看到 `E4x32`/
> `E16x8`，且须记住那**不是**环境代号。

| 分母（仅 E1 有实测） | 值 (GB/s) | K1 占该分母 |
|---|---:|---:|
| 16 × t24 @1.55（**384 线程乐观上限**，非下界） | 4462 | 3.35% |
| 384 × 9.489 @1.55（用 576 的每线程效率） | 3644 | 4.10% |
| 576 线程实测 @1.55 | 5465 | 2.74% |
| 576 线程实测 @2.0 | 6801 | 2.20% |
| **单节点 DDR 实测 @1.55** | **616** | **24.3%** |

> 利用率随分母选择在 **2.20%–4.10%** 间变化，均远低于带宽墙；
> `16 × t24` 用于 384 线程估计时是**乐观上限**（分母偏大、利用率偏低），
> 只有对"全节点容量"才是下界。详见
> [`qcu_l24_environment_bandwidth_20261111.md`](qcu_l24_environment_bandwidth_20261111.md) §9.5。

**结论**：K1 的有效带宽只有本机 **DDR** 实测带宽的 1/4，更只有 HBM 的 2–3%。
瓶颈不是带宽，而是 scalar `ht=1` 边界内核与小消息全局同步。K8/K9 的
`0.30/0.39 GB/s` 与带宽无关，是几何相关路径开销（见 §2.1 末段）。

> 其余环境（E2–E7）在仓库内**没有 STREAM 实测**，因此不能计算"带宽利用率"，
> 上表的 GB/s 只用于同后端内的量级比较。

## 10. 精度对比（同后端同窗口）

`dev99` 期 x86 主机、`restart=0`、5 次稳态 median：

| 精度 | 1 rank | 2 ranks | c128 / c64 |
|---|---:|---:|---:|
| c64（inv6 变体） | 7.769662 | 6.351769 | — |
| c128（inv6 变体） | 15.592787 | 11.472949 | **2.01× / 1.81×** |

迭代数不同（c64 20 次、c128 23 次），因此倍率同时包含迭代与每迭代成本差异。

## 11. 缺失与不可用

| 后端 | L24 状态 |
|---|---|
| Torch Legacy / Strict | **无 L24 记录**；六版本基准只做到 `8³×16` |
| CUDA Strict-MG | **无 L24 记录**；`strict_solve_24x24x24x64.stdout` 为 0 字节且无 JSON |
| KUPL Strict-MG（`8³×16` 前缀） | 1/2/4 rank 为 `1.222731/1.101605/1.658620 s`，但格点是 `8³×16`，**不是 L24** |
| L24 Strict 显存 | Legacy 有记录、Strict 无 solve JSON，**不可比较** |
| 计算节点 true-MG | BLOCKED（作业 2050917，exit 143） |
| QUDA MG on DCU | BLOCKED-RUNTIME |
| L24 QUDA MG（任意平台） | 无同窗可信记录；`nvec24` 与 QCU `dof12` 粗空间不等价，禁止计算倍率 |

## 12. 一句话答案

若问题是"**最新版 PyQCU 在真实鲲鹏计算节点求解 `24³×64` 的耗时**"，唯一可信单值是
**`0.4346 s`**（K1：32 ranks × 12 threads，`grid=1 1 1 32`，BiCGStab，
16 迭代，独立真残差 `3.54e-6` PASS，n=1，同二进制噪声带 ±8%）。

引用时必须同时说明：**该值不含 cache 构建与输入读取；计算节点 true-MG 仍
BLOCKED；`24³×64` 是本项目 smoke 格点，不得进入正式大格加速比。**

## 13. 证据等级与来源

| 等级 | 含义 | 本文行 |
|---|---|---|
| **A** | 本地可复现原始记录（屏幕 hardcopy / 调度器状态 / JSON 原文） | K8、K9（`screen-…-250..253.txt` 有完整 report 原文）、§2.2 |
| **B** | 阶段日志转述（本地无该次运行的 raw JSON） | K1–K7、§3、§6 |
| **C** | 文档转述，本地无任何原文 | §4 全部（`profiler_solver_performance_history.md`）、§5、§7、§8、§9、89.13 s 基线 |

> **本轮修正（依据核验）**：
> - **K9 由 B 升为 A**：`164.7499241` 的完整 report 原文存在于
>   `logs/screen-73056-ss-20261106-250.txt`（及 `-251/252/253` 偏移副本），
>   含 `"omp_threads": "32"`、`"input_prebind_seconds": 12.889966779999668`、
>   `"residual_source": "KUPL native fail-closed solve; independent check skipped"`。
> - **§7 V 系列由隐含 B 改为 C**：其数值来自 `three_way_benchmark_20261022.md`
>   与 `profiler_solver_performance_history.md`，本地**无**对应 raw JSON
>   （`three_way` 原始 JSON 在 gitignore 的 `data/**`），属文档转述。
> - **§8 A100、§9 DCU 同属 C**：`logs/qcu_solver_test34_20261029/` 不在工作树，
>   全部为文档二次转引。
> - **§5 控制节点为 C**：仅 `qcu_mg_performance_ledger_20261029.md` 转述。

**重要缺口**：`logs/kupl_hbm_opt_20261106.md` 与 `data/kupl_roofline_20261106.json`
是本文 K 系列与 §3 的主要来源，但**均被 `.gitignore` 忽略**（`logs/.gitignore`
第 1 行 `*`、`data/.gitignore` 第 1 行 `*`），因此这些 B 级证据只存在于工作区，
tag 恢复不到。

## 14. 相关文档

- [`kupl_kunpeng_progress_20261110.md`](kupl_kunpeng_progress_20261110.md)：鲲鹏真实环境
  进展全文（硬件、HBM 两阶段、代码改动、否决路线、阻塞项）。
- [`kupl_kunpeng_figures_20261110.pdf`](kupl_kunpeng_figures_20261110.pdf)：同源 9 页图表集。
- [`kupl_kunpeng_guide.md`](kupl_kunpeng_guide.md)：鲲鹏构建、MPI/OpenMP 拓扑与
  NUMA/片上内存策略。
- [`three_way_benchmark_20261022.md`](three_way_benchmark_20261022.md)：V100 x86
  三方同窗测评（§4、§7 的 V 系列来源）。
- [`qcu_mg_performance_ledger_20261029.md`](qcu_mg_performance_ledger_20261029.md)：
  A100/DCU 性能账本（§5–§8 的 A/C 系列来源）。
- [`qcu_quda_solver_dcu_l24_20261029.md`](qcu_quda_solver_dcu_l24_20261029.md)：
  DCU L24 正式采样与硬件审计。
- [`qcu_mg_refactor_validation_20261023.md`](qcu_mg_refactor_validation_20261023.md)：
  L24 Legacy 深度与 MPI 矩阵（§7 末表来源）。
- [`profiler_solver_performance_history.md`](profiler_solver_performance_history.md)：
  `test9..dev103` 全量历史索引（§4、§6 的来源）。
- [`mg_cache_sizes_24x24x24x64_20261023.md`](mg_cache_sizes_24x24x24x64_20261023.md)：
  L24 MG cache 尺寸推导。
