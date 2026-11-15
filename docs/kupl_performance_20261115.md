# KUPL 性能汇总：dev143 对照 stab64（2026-11-15）

> 证据等级：**B/C 混合**。数值主体转抄自已提交文档与 `logs/**`；L24/L32 生产
> `report.json`/`meta.txt` 位于远端共享目录，本机未挂载，未逐份复核原始 JSON。
> 本文只做只读汇总，**未运行新算例、未改动代码或既有文件**。
>
> 基线：`stab64` = `2c07cf94b6da68b4e5afa3d66ac6b5192ff3e51b`（2026-11-12 08:59，
> release 0.1.15）。目标：`dev143` = `4f60e3487ca145f0cf64930f38a4518e5f705c00`
> （2026-11-14 20:53）。两者相距 47 个提交。
>
> 范围：**KUPL CPU 后端**；主格点 L24（`24³×64`，c64）；L32（`32³×64`）与
> `32³×12` 只有 dev143 数据。c128 见 §11：鲲鹏侧没有 c128 性能数据，只有
> 历史 x86 值与 dev143 正确性门。本文不比较 CUDA/QUDA/DCU，也不把 CPU 与
> GPU 合并成单一加速比。

## 0. 结论摘要

1. **L24 BiCGStab 的当前最低中位是 dev143 的 `0.154883 s`**（cn19699、32 rank×18
   thread、grid `1 1 1 32`、共享内存 halo、c64、atol `1e-4`、12 it、真残差
   `5.3848659457849185e-05`、n=8）。stab64 可引用的同节点最低单样本是
   `0.5193 s`（32r×12t、y76 原配方、16 it、真残差 `3.5412238955e-6`、n=1）。
   两者数值比为 `3.35×`，但**线程、源、迭代/停止点与剖析开关不同，不是受控
   A/B**。
2. **dev143 内唯一的受控 halo A/B 是共享内存对 MPI：中位 `0.154883` 对
   `0.181733 s`，改善 `14.77%`**，8+8 样本区间不重叠（shm 最大 `0.157304` <
   MPI 最小 `0.180094`）。机制为 wait 中位 `0.164` 对 `0.775 ms`（约 `4.7×`
   更低），dslash 中位 `0.067483` 对 `0.093351 s`。
3. **dev143 首次打通多 rank Legacy-MG 的 patch cache 构建**：32 rank 下 L24
   `24.476 s`、L32 `53.023 s`、`32³×12` `11.368 s`，三组 `verify_status=0`。
   1-rank 的 L24 cache 仍需 `493.63817879895214 s`（`143,335,424 B`），距
   `30 s` 目标约 `16.5×`。
4. **真实 MG 的 `atol=1e-6` 求解**：L24 32 rank（grid `2×2×2×4`）`0.519831 s`、
   真残差 `4.536e-07`；L32 `1.175139 s`、`4.543e-07`；`32³×12` `0.266999 s`、
   `4.461e-07`。同格点/grid 4-rank `8³×16` 对照中，MG `0.037935 s` 比 BiCGStab
   `0.031577 s` **慢 20%**。
5. **`atol=1e-4` 的“MG”数字 `0.135579 s` 不是真实 MG**：默认
   `coarse_tol_factor=1e4` 使粗层目标 `atol×factor=1.0`，粗层首行即“收敛”、
   修正恒为零并触发 `fallback_to_fine`，最终走 fine BiCGStab。该组与历史
   “MG 快 26%”结论均**作废**，不得作为 MG 加速证据。
6. **dslash 已进入带宽受限区**：按解析流量 `1.10 GB / 0.067483 s` 得
   `16.3 GB/s·rank`、聚合 `522 GB/s`，约为节点 STREAM triad 峰值
   （`18.3–20.1 GB/s·rank × 32 ≈ 585–643 GB/s`）的 **87%**。`0.1 s` 目标不能
   靠继续加 rank 或调线程达成。
7. **可比性红线**：两侧的 `wall_seconds` 都只含 solve 调用，不含 Python 导入／
   输入读取／cache 构建；但源类型、迭代数、halo 路径、剖析开关和频率／
   节点身份未全部对齐。除本文标注的受控 A/B 外，**不得把任意两行相除当成
   代码加速比**。

## 1. 口径、字段与判读规则

### 1.1 字段定义

| 字段 | 定义 | 备注 |
|---|---|---|
| `wall_seconds` | 从进入求解器调用到返回的墙钟 | 不含 Python 导入/初始化、gauge/source 读取、独立真残差计算；MG 也不含 cache 构建 |
| `builder_seconds` | 从开始读入 gauge/source 到 cache 构建/回读结束的时间；多 rank 取 `MPI.MAX` | 不是纯 stencil 时间；含 patch 读入、33-stencil、coarse export/写盘 |
| `iterations` | BiCGStab 外迭代数 | 真实 MG（`levels>1` 且 `restart≠0`）为 `None`，不得按迭代数比较 |
| `atol` | 相对停机容差，`‖r‖/‖b‖` | 驱动 `target = tolerance × rhs_norm`（`lattice_clover_multigrid.cpp:350`） |
| `final_true_residual` | 独立重建的 FULL Wilson-Clover 相对真残差 | `relative_full_residual` / `distributed_relative_residual`；与 `atol` 可直接比较 |
| `verify_status` | cache 回读/native 校验状态 | `0` 为通过；非 0 一律 FAIL |
| `rank × thread` | MPI rank 数 × 每 rank OpenMP 线程 | 鲲鹏 `ppr:2:numa` 下每 rank cpuset 为 18 CPU |
| `grid` | 4D process grid（`x y z t`） | `1 1 1 32` 为纯时间分解；MG patch 路径要求均衡 4D 分解 |

### 1.2 可比性规则

1. **同一行内**的 wall、迭代、残差、dtype、atol 才能互相解释；跨行比较必须
   同时对齐后端、格点、精度、rank×thread、grid、halo、源、停止点与硬件身份。
2. 同节点运行抖动约 **±2%**；全 1 源的 L24 3 样本极差 **±3.3%**、L32 **±4.1%**。
   单样本差异 `<5%` 不得判为加速或回退。
3. 跨节点差异可达约 **20%**（cn19650 对 cn19699 的 PASSIVE 基线
   `0.319600` 对 `0.266416 s`）；跨节点数字只能作量级参考。
4. 1.55 GHz 是当前运行态、2.0 GHz 是硬件上限；若跨节点频率不同，CPU 受限路径
   的预期差上界为 `2.0/1.55 = 1.29×`，必须按节点分别标注。
5. **不同源不得互算加速比**：点源（seed 42）与全 1 源在 L24 上分别是 12 it
   与 11 it，真残差 `5.384866e-05` 与 `6.756443e-05`；性能测试必须记录源类型、
   源坐标与 gauge 身份（`AGENTS.md` 约定）。
6. **MG 与 BiCGStab 不得按迭代数比较**：MG 的 `iterations=None`，且
   `atol=1e-4` 的默认参数会回退到 fine BiCGStab。
7. **`atol=1e-4` 与 `atol=1e-6` 不是同一判据**：本文第 3–4 节的 dev143 数据全部是
   `atol=1e-4` **非 strict**，实际门槛被 `max(atol, eps(float32)×2048) = 2.44140625e-4`
   抬高；2026-11-15 矩阵批次的 37 条 solve 全部带 `--strict-acceptance`，门槛就是
   `1e-6`（c64）/`1e-9`（c128），**有效容差差 244.1×**。因此矩阵批次的
   `0.2405 s`（L24 c64 t-finest 32×18 shm1，18 it）与 `0.3497 s`（balanced 同条件 shm0）
   **不能**与本文件的 `0.171331–0.182659 s`（中位 `0.173362`，11 it）直接相除：
   迭代 18/11、容差 244.1×、grid、shm 四项同时不同。矩阵侧的分解与红线见
   [`kupl_kunpeng_matrix_20261115.md`](kupl_kunpeng_matrix_20261115.md) §1.8、§10.3。

### 1.3 证据等级

| 等级 | 含义 | 本文典型行 |
|---|---|---|
| A | 本机可复现原始 JSON/屏幕 hardcopy | 仅本地 `data/kupl_opt_20261113/real8/*.json`（`8³×16×16` 小格，不用于 L24 结论） |
| B | 已提交阶段日志/文档逐项转抄，含作业号或 `report.json` 字段 | `logs/v202611141855.txt`、`docs/kupl_l24_perf_20261113.md` |
| C | 文档二次转述，远端 raw JSON 未在本机复核 | L24/L32 生产 `report.json`、`data/kupl_scaling_20261114/**` |

## 2. 硬件与环境身份

| 项 | 取值 | 来源 |
|---|---|---|
| 集群/队列 | LX2，`q_hpcapp`；计算节点 `cn19699`、`cn19650`，历史节点 `cn10579` | `docs/kupl_l24_perf_20261113.md` §2；`docs/qcu_l24_solver_timing_20261111.md` §12 |
| CPU | aarch64 鲲鹏；`nproc=592`；16 个 CPU NUMA，每 NUMA `cpulist` 宽 38；16 个无 CPU 的内存 NUMA | `docs/kupl_kunpeng_node_survey_20261114.md` §2–§3 |
| 内存 | 约 `565.0 GiB`；`KUPL_MEMORY_NODE=local+16` 指向片上内存域（实测 `memory_node=16–31`） | 同上；`docs/kupl_l24_perf_20261113.md` §2 |
| 频率 | `scaling_cur_freq≈1.55 GHz`；`cpuinfo_max_freq=2.0 GHz`；governor `userspace` | 同上 |
| 内核 | `5.10.0-285.6.v2403.ky10p.aarch64`；`BogoMIPS=200.00` | 同上 |
| MPI | HMPI（HPCKit 25.2.1）；`mpirun -np 32 --bind-to none --map-by ppr:2:numa --allow-run-as-root` | `docs/kupl_l24_perf_20261113.md` §2 |
| 调度 | `dsub -q q_hpcapp -T 900 -N 32 -rpn 32 -R "cpu=18,mem=8G"` | `docs/kupl_mpi_scaling_20261114.md` §5.3 |
| 每 rank cpuset | 18 CPU（`ppr:2:numa` + 32 rank；`KUPL_OMP_THREADS=18` 为上限，`TH=36` 失败关闭） | `docs/kupl_mpi_scaling_20261114.md` §3 |
| 构建 | KUPL native 构建；部分批次为 native+LTO（`build-native-lto`）；逐用例构建身份未在证据中逐条记录；`-ffast-math` 仅作 A/B，非默认 | `docs/kupl_l24_perf_20261113.md` §3、§12.4；`logs/v202611141215.txt` |
| 未记录 | 具体 Kunpeng SKU、编译器版本、KUPL 库 SHA256、完整编译命令、内存带宽档位 | 待补（第 8 节） |

## 3. stab64 基线（截至 2026-11-12 08:59）

> 该标签早于 2026-11-13 的共享 halo、t2 lane、PACK6、dedup、boundary-lane
> 等优化；因此 stab64 的 KUPL 数字是**优化前基线**。

### 3.1 L24 BiCGStab（`24³×64`，c64，`levels=1`、`restart=0`）

固定参数（除注明外）：

```text
--lat 24 24 24 64 --mass -0.277 --csw 1.160920226 --atol 1e-4 \
--max-iter 50 --dtype c64 --levels 1 --restart 0
```

`levels=1, restart=0` 是 parity-Schur BiCGStab，不是 multigrid。

| 用例 | 节点 | ranks × thr | grid | wall (s) | 迭代 | 真残差 | 状态 | n | 备注 |
|---|---|---|---:|---:|---:|---|---|---:|---|
| y76 历史 K1 | cn10579 | 32 × 12 | `1 1 1 32` | **0.4346** | 16 | `3.5412238955e-6` | PASS | 1 | B 级；同二进制噪声带 ±8% |
| y76 原配方复现 | cn19699 | 32 × 12 | `1 1 1 32` | **0.5193** | 16 | `3.5412238955e-6` | PASS | 1 | 同配方跨节点 1.20× |
| x_case_dbg | cn19699 | 32 × 12 | `1 1 1 32` | 0.8410 | 16 | `3.54117e-6` | PASS | 1 | 含 profiling/fused/oversubscribe，相对 y76 配方 1.62× |
| x_case_dbg | cn19699 | 32 × 18 | `1 1 1 32` | 1.5979 | 16 | `3.54122e-6` | PASS | 1 | 18 thread 更慢 |
| x_case_dbg | cn19699 | 16 × 8 | `1 1 1 16` | 1.1442 | 16 | `3.54128e-6` | PASS | 1 | — |
| hbm8c | cn19699 | 8 × 16 | `1 1 1 8` | 2.0595 | 16 | `3.54132e-6` | PASS | 1 | — |
| hbm16d | cn19698 | 16 × 8 | `1 1 1 16` | 1.0946 | 16 | `3.54128e-6` | PASS | 1 | — |
| hbm4 | cn19698 | 4 × 8 | `1 1 1 4` | 4.7462 | 16 | `3.54113e-6` | PASS | 1 | — |

读法：

- `0.4346` 是 stab64 文档对外给出的唯一可信单值（`qcu_l24_solver_timing_20261111.md`
  §12），但它在 cn10579 上、n=1、B 级，且已被 dev143 的 L24 数据取代；
  引用时必须注明节点与配方。
- `0.5193` 是同一 y76 配方在 cn19699 上的复现；与 `0.4346` 的 `1.20×` 是
  **节点差异**，不是代码差异。
- `0.8410` 与 `0.5193` 的 `1.62×` 是 profiling/fused/oversubscribe 与包装开销，
  不能当作纯节点性能。
- 全部 16 it、真残差约 `3.54e-6`；这是 stab64 的停止点，**与 dev143 的
  11–12 it、`5.38e-05`–`6.76e-05` 不是同一收敛路径**。

### 3.2 L24 true-MG（登录节点同窗 A/B）

配置：`24³×64` c64、16 rank × 8 thread、grid `1 1 1 16`、同 gauge/RHS、
`atol=1e-4`、`max_iter=50`、E4 cache、`mg-grid=2 2 2 1`、`restart` 10/20/50。

| restart | BiCGStab 中位 (s) | true-MG 中位 (s) | BiCG/MG | MG 证据 |
|---:|---:|---:|---:|---|
| 10 | 1.397243 | 0.789756 | 1.7692× | `fine=32`、`coarse=19` |
| 20 | 1.230783 | 0.705146 | 1.7454× | `fine=29`、`coarse=9` |
| 50 | 1.091866 | 0.704241 | 1.5504× | `fine=29`、`coarse=9` |

对应的 full-residual 伴随运行（同一登录节点批次，独立真残差门
`2.44140625e-4`）：

| 配置 | wall (s) | 独立真残差 | 状态 |
|---|---:|---:|---|
| r10 | 1.0437915269 | `6.8887275716e-5` | PASS |
| r20 | 1.316182 | `5.5308248713e-5` | PASS |
| **r50** | **0.9473317482** | `5.5308248713e-5` | PASS |

读法：

- 同窗 A/B 的中位（`0.789756/0.705146/0.704241 s`）与 full-residual 伴随运行
  （`1.0437915269/1.316182/0.9473317482 s`）**不是同一组样本**；后者的 wall 更高，
  说明是否启用库内/伴随真残差检查会改变耗时，不能混用。
- 文档把 r50 的 `0.9473317482 s` 称为“登录节点最低绝对耗时”，同时把
  “十种合法分格”筛选的最优记为 **`1.686342053115368 s`**（16 × 8、
  grid `1 1 1 16`）。两者是不同批次/不同扫描口径，**不能合并成一个“最优”**；
  本文分别记录，不替文档裁决。
- `restart=50` 只执行一次 coarse correction，应描述为“initial deflation +
  fine BiCGStab”，不是周期 V-cycle。
- **计算节点 true-MG 正式 A/B 当时为 BLOCKED**（作业 `2050917` 运行 787 s
  无 `report.json`，经同队列 TERM kill `2051058` 终止，`jobExitCode=143`）。
  L32 true-MG/BiCG 当时也为 **BLOCKED**（1-rank cache 与 32-rank BiCG 均超
  10 分钟门禁）。

> **不得误引**：stab64 文档明确标注 `1.60861 s` 与 `1.51917 s` 两组
> `coarse_calls=0`、`fallback=fine_bicgstab`，不是 true-MG 证据。

### 3.3 MG cache 构建（stab64，登录节点筛选）

> 以下 builder 时间为登录节点 `§6.3 小格点与 setup 成本` 的筛选值，
> 不是计算节点结论；计算节点 true-MG 当时为 BLOCKED。

| cache / time block | cache 字节 | builder (s) | 合法最大 `grid_t` | 备注 |
|---|---:|---:|---:|---|
| time block 1 | `143,335,424` | **119.57** | 32 | 与 dev143 的 L24 cache 同为 143,335,424 B |
| time block 4 | `99,540,992` | 69.33 | 8 | solve 更慢，build 节省不抵粗层同步 |
| time block 8 | `92,241,920` | 63.59 | 4 | 同上 |
| 并行 local-stencil L24 | — | **80.70** | — | 串行外推约 42 min；并行 stencil 当时为 opt-in |

> 这些 builder 数值与 dev143 的 `builder_seconds` **口径不同**（并行度、
> 是否含输入读入/导出、cache identity 未逐项对齐），只能并列展示，
> **不能据此计算 dev143 相对 stab64 的 cache 构建加速比**。

### 3.4 带宽（stab64，solver 级）

| 指标 | 取值 | 口径 |
|---|---:|---|
| 模型流量/算术 | `64.97 GB` / `86.63 GFLOP` | 按 y76 的 34 次 Schur apply 折算 |
| y76 `0.4346 s` 有效带宽 | **149.51 GB/s** | 模型流量 ÷ 实测墙钟 |
| y76 `0.5193 s` 有效带宽（派生） | 约 `125.1 GB/s` | 同一模型；本文派生，非原文数值 |
| 每线程有效带宽 | `0.389 GB/s` | y76，32r×12t |
| STREAM FP64 Triad 全节点 HBM | `5465–6801 GB/s` | LX2 STREAM，八种配置 |
| STREAM 单域 | `279–428 GB/s` | 同上 |
| 单节点 DDR 实测 | `616 GB/s` | 同上 |
| y76 占 HBM 分母 | `2.74%`（576 线程实测 @1.55）至 `4.10%` | 随分母选择变化；结论“不是带宽瓶颈”不变 |
| y76 占 DDR 分母 | `24.3%` | 同上 |

> 这是 **solve 级有效带宽**，不是 dslash 单算子带宽；与 dev143 的
> `522 GB/s` 聚合 dslash 带宽不是同一口径，不能相除。

## 4. dev143 数据

### 4.1 L24 BiCGStab（dev141/143，cn19699，c64）

固定参数（除注明外）：

```text
# 程序参数
--lat 24 24 24 64 --mass -0.277 --csw 1.160920226 --atol 1e-4 \
--dtype c64 --levels 1 --restart 0 --grid 1 1 1 32

# 并行与绑定
32 rank × 18 thread；ppr:2:numa；--bind-to none
KUPL_NUMA_BIND=1
KUPL_MEMORY_POLICY=preferred
KUPL_MEMORY_NODE=local+16
KUPL_HUGEPAGE=1
OMP_WAIT_POLICY=ACTIVE
KUPL_ENABLE_T2_LANES=1
KUPL_HALO_PACK6=1
KUPL_HALO_DEDUP_PACK=1
KUPL_T_LOCAL_IN_PART1=1
KUPL_ENABLE_BOUNDARY_LANES=1
```

`max_iter` 未在 dev143 证据中逐条抄录；驱动默认 `1000`，11–12 it 远低于上限，
停机由 `atol` 触发。`seed=42`；源类型按行注明。

| 用例 | 并行 | halo | 源 | 迭代 | 真残差 | wall 中位 (s) | 样本范围 (s) | n | 状态 |
|---|---|---|---:|---:|---:|---:|---|---:|---|
| shm 8 样本 | 32 × 18 | 共享内存（默认） | 点源（seed 42） | 12 | `5.3848659457849185e-05` | **0.154883** | 0.150240–0.157304 | 8 | PASS |
| MPI 8 样本 | 32 × 18 | MPI（`KUPL_SHM_HALO=0`） | 点源（seed 42） | 12 | `5.3848659457849185e-05` | 0.181733 | 0.180094–0.187894 | 8 | PASS |
| shm 3+3 | 32 × 18 | 共享内存 | 点源（seed 42） | 12 | 同 8 样本 | 0.154389 | 0.152981–0.155377 | 3 | PASS |
| MPI 3+3 | 32 × 18 | MPI | 点源（seed 42） | 12 | 同 8 样本 | 0.180999 | 0.180750–0.184409 | 3 | PASS |
| 全 1 源 L24 | 32 × 18 | 共享内存 | 全 1 源（所有点实部=1） | **11** | `6.756443e-05` | **0.173362** | 0.171331–0.182659 | 3 | PASS |
| 全 1 源 L32 | 32 × 18 | 共享内存 | 全 1 源 | **12** | `6.831831e-05` | **0.348549** | 0.340658–0.369371 | 3 | PASS |

读法：

- 8+8 样本的中位改善 `14.77%`，区间完全不重叠；均值 `0.154132` 对 `0.183074`，
  sd `0.002503` 对 `0.002797`。
- 3+3 与 8+8 不是同一批；`0.154389` 是 3 个 shm 样本的中位，不是新的最优。
- 全 1 源的 L24 `0.173362 s` 与点源的 `0.154883 s` **不可比较**：源、迭代数
  与真残差都不同。
- 首次全 1 源样本使用 P1 之前的构建，后两次使用 P1 构建（零拷贝默认关，仅多
  profile 计时，量级约 10 µs）；三次迭代数与真残差逐位一致。
- L32 只有 dev143 的这 3 个样本，没有 stab64 的可用 L32 基线。

### 4.2 线程与 rank 扫描（dev143）

线程扫描（L24 全 1 源，32 rank，其余配置同上，单样本）：

| TH | L24 wall (s) | 相对 TH18 | L32 wall (s) | 相对 TH18 |
|---:|---:|---:|---:|---:|
| 9 | 0.215964 | +24.6% | 0.460573 | +32.1% |
| 16 | 0.182221 | +5.1% | 0.352243 | +1.06% |
| 18 | **0.173362**（中位） | — | **0.348549**（中位） | — |
| 36 | — | 失败关闭 | — | 失败关闭 |

`TH=36` 的失败是库内守卫 `KUPL_OMP_THREADS exceeds the NUMA-local CPU set`，
不是性能点；当前每 rank cpuset 为 18 CPU。`TH_max = min(floor(W/P), C)`，
本节点 `W=38, P=2, C=18 → 18`。

rank 扫描（共享 halo 之前，MPI halo，L24，c64，32r×18t 基线；每点 3 样本）：

| ranks (t) | 样本 (s) | 说明 |
|---:|---|---|
| 8 | `0.437795` / `0.449878` / `1.731601` | 每 rank 计算量为 32 rank 的 4 倍；不稳定 |
| 16 | `3.054642` / `0.479433` / `3.533757` | 方差极大 |
| 32 | `0.183248` / `0.176644` / `0.185366` | 同批最优 |

32 rank 是 L24 的 t 方向上限（`T=64`、`local[3]` 必须为偶数，最多 32 rank）；
64 rank 需要 2D/4D 分解，且 `grid[0..2]==1` 不再成立，t2 lane 与
split-boundary 两条生产路径同时失效。

### 4.3 分项剖析与带宽（dev143）

`KUPL_PROFILE_OPT=1`，ACTIVE，cn19699，rank 17（较早的 prof4 批次，
MPI halo，32r×18t）：

| 项目 | 次数 | 时间 (s) | 字节 (MB) | 每 rank 带宽 | 聚合带宽 |
|---|---:|---:|---:|---:|---:|
| dslash | 52 | 0.104099 | 828.113 | 7.96 GB/s | 254.56 GB/s |
| clover | 52 | 0.013720 | 552.075 | 40.24 GB/s | 1287.63 GB/s |
| reduce | 51 | 0.023190 | — | 0.45 ms/次 | — |

phases：`prepare=0.003573 s`、`solve=0.156956 s`、`reconstruct=0.002327 s`；
solve 之外 setup 约 `44 ms`。

共享 halo 3+3 A/B 的 rank 中位（作业 `2119917` 对 `2119919`）：

| 配置 | dslash 中位 (s) | solve 中位 (s) | wait 中位 (ms) | arrival 中位 (ms) |
|---|---:|---:|---:|---:|
| 共享 halo | **0.067483** | 0.110958 | **0.164** | 0.945 |
| MPI halo | 0.093351 | 0.133611 | 0.775 | 0.899 |

dslash 解析流量（c64，每 site `768 B`）：

```text
gauge: 8 dir × 9 complex × 8 B = 576 B
src  : 12 complex × 8 B        =  96 B
dst  : 12 complex × 8 B        =  96 B
合计                            = 768 B/site
```

32 rank 每 dslash 每 rank `27,648 site × 768 B = 21.2 MB`；52 次 → `1.10 GB`。

| 口径 | 值 | 说明 |
|---|---:|---|
| dslash 解析流量带宽 | **16.3 GB/s·rank** | `1.10 GB / 0.067483 s`；聚合 **522 GB/s** |
| 占 STREAM triad 峰值 | 约 **87%** | 分母 `18.3–20.1 GB/s·rank × 32 ≈ 585–643 GB/s` |
| dslash profiler 口径 | 12.3 GB/s·rank / 393 GB/s | `828 MB / 0.067483 s`；字节数取自较早 prof4 批次（`0.104099 s`），与 shm 墙钟是跨批次组合，仅作量级核对 |
| clover | 40.24 GB/s·rank / 1287.63 GB/s | 缓存复用；不能据此宣称超过硬件峰值 |
| reduce | 0.45 ms/次 × 51 ≈ 23 ms | 约占 `0.155 s` 的 15% |
| 共享 halo 内存 | `stride=165888`、`slot_bytes=21,233,920`（约 20.25 MiB/rank） | 32 rank 合计约 648 MiB |
| STREAM triad | `18.3–20.1 GB/s·rank`；32 rank 约 `585–643 GB/s` | LX2 节点实测 |

### 4.4 算子级强扩展（dev143）

全局 L24 固定、每 rank 18 线程、`applyWilsonDslashQcu`、单 parity、50 次迭代、
c64；默认内核路径（**未开** t2 lane / PACK6 / dedup / T_LOCAL；split-boundary
由几何触发）。作业 `2123210`，cn19699，`-R cpu=576`。

| ranks | 总线程 | 50 it (s) | 每迭代 (ms) | 相对 4 rank | 每线程 Gflop/s |
|---:|---:|---:|---:|---:|---:|
| 4 | 72 | 0.626466 | 12.529 | 1.00 | 1.13 |
| 8 | 144 | 0.445490 | 8.910 | 1.41 | 0.79 |
| 16 | 288 | 0.283023 | 5.660 | 2.21 | 0.63 |
| 32 | 576 | 0.238596 | 4.772 | **2.63** | **0.37** |

并行效率（相对 4 rank）：8 rank `70.5%`、16 rank `55.3%`、32 rank `32.9%`；
边际收益 `1.41× → 1.57× → 1.19×`，16→32 只有 `+19%`。每线程吞吐从
`1.13` 降到 `0.37 Gflop/s`（−67%）。

同作业线程扫描（NP=32，local t=2，同一默认路径）：

| TH | 50 it (s) | 每迭代 (ms) | 相对 TH=1 |
|---:|---:|---:|---:|
| 1 | 1.543505 | 30.870 | 1.00 |
| 4 | 0.499424 | 9.988 | 3.09 |
| 9 | 0.311865 | 6.237 | 4.95 |
| 18 | 0.238596 | 4.772 | **6.47** |
| 36 | — | — | 失败关闭 |

> 该曲线是**默认内核路径**；同几何的生产配置（t2 lane + PACK6 + dedup +
> T_LOCAL + boundary lanes）在 NP=32 的 dslash 为 `1.30 ms/次`（52 次
> `0.067483 s`），比基准的 `4.77 ms/次` 快 `3.68×`。不能把这条曲线当作
> 生产路径的扩展性。

### 4.5 Legacy-MG cache 构建（dev143，patch 算法，32 rank）

1-rank L24 cache（`data/kupl_scaling_20261114/cache_l24_parallel.h5`）：

| 项 | 值 |
|---|---|
| 文件大小 | `143,335,424 B`（约 136.7 MiB） |
| `setup_seconds`（= `builder_seconds`） | **`493.63817879895214`** |
| `status` | PASS |
| `setup_method` | `cxx-null-vector+local-33-stencil` |
| `setup_backend` | `kupl-qcu-cxx` |
| `mg_grid` | `[2,2,2,1]` |
| `nv_iters` | `2` |
| lattice | `[24,24,24,64]` |
| 构建并行 | 1 rank × 18 thread；并行 stencil 默认开 |

32-rank patch 构建（全部 `verify_status=0`）：

| 格点 | grid | `builder_seconds` (s) | `coarse_export` (s) | `stencil` (s) | `gauge patch` (s) | `lonv patch` (s) | 真残差 |
|---|---|---:|---:|---:|---:|---:|---:|
| `24³×64` | `2×2×2×4` | **24.476** | 22.751 | 14.118 | 3.455 | 2.346 | `4.535e-07` |
| `32³×64` | `2×2×2×4` | **53.023** | 48.943 | 29.879 | 10.275 | 2.874 | `4.543e-07` |
| `32³×12` | `4×2×2×2` | **11.368** | 10.650 | 6.394 | 1.164 | 2.054 | `4.460e-07` |

读法：

- `builder_seconds` 是 `MPI.MAX`，含 gauge/source 读入、patch 构造、33-stencil
  与 coarse export；`coarse_export` 是其中最大单项（L24 `22.751/24.476`）。
- 32-rank 构建的线程数与节点均未在 §9.3/日志中逐条记录；同批调度配方为
  18 CPU/rank，报告中标为“未逐条记录”，不以推断值替代。
- 三组均 `verify_status=0`；L24/L32 的真残差是 cache 回读后的 solve 残差，
  不是 cache 构建本身的收敛量。
- **t-only（`grid 1 1 1 32`）在 patch 算法下不可用**：x/y 轴因 `grid=1`
  退化为整轴（24），t 轴也退化为整轴（64），每 rank 构建整格（约 493 s），
  32 rank 并行仍 `rc=124`（≈355 s）。多 rank MG 一律使用均衡 4D 分解
  （`2×2×2×4`、`4×2×2×2` 等）。

### 4.6 真实 MG 求解（dev143，`atol=1e-6`）

`atol=1e-6` + 默认 `coarse_tol_factor=1e4` → `coarse_atol=1e-2`，粗层真正工作；
以下为真实 MG，不是回退。

| 用例 | ranks × thr | grid | 格点 | wall (s) | 迭代 | 真残差 | verify | 状态 | n |
|---|---|---|---:|---:|---|---:|---|---|---:|
| MG（cache 回读） | 4 × 未记录 | `2×2×1×1` | `8³×16` | **0.037935** | `None` | `3.409e-07` | 0 | PASS | 1 |
| BiCGStab 对照（不走 cache） | 4 × 未记录 | `2×2×1×1` | `8³×16` | **0.031577** | 未记录 | `5.421e-07` | — | PASS | 1 |
| MG（cache 回读） | 32 × 未记录 | `2×2×2×4` | `24³×64` | **0.519831** | `None` | `4.536e-07` | 0 | PASS | 1 |
| MG（cache 回读） | 32 × 未记录 | `2×2×2×4` | `32³×64` | **1.175139** | `None` | `4.543e-07` | 0 | PASS | 1 |
| MG（cache 回读） | 32 × 未记录 | `4×2×2×2` | `32³×12` | **0.266999** | `None` | `4.461e-07` | 0 | PASS | 1 |

读法：

- 4-rank `8³×16` 是同格点、同 grid、同 `atol=1e-6` 的对照（MG 侧走 cache
  回读，BiCGStab 侧不走 cache）：MG 比 BiCGStab **慢 20%**，不是快 2.93×；
  两者的真残差不同（`3.409e-07` 对 `5.421e-07`），因为算法路径与停机点不同。
- 32-rank 三组都只给一个数值（原文 §9.5 未声明重复数），且
  `iterations=None`；不能按迭代数比较 MG 与 BiCGStab，也不能把三组之间的
  墙钟差归因于格点体积（rank/grid 同时变化）。
- 32-rank MG 的线程数未在 §9.5/日志中逐条记录；同批生产配方为 18 thread，
  报告中标为“未记录”。
- `mg_grid`/`nv_iters` 在 §9.5 未重述；cache identity（§4.5）为
  `mg_grid=[2,2,2,1]`、`nv_iters=2`。1e-6 回读是否复用同一 identity 未在
  证据中逐字写明，标为待核（cache readback 会校验 identity，若不一致应失败）。
- 4-rank 对照的 BiCGStab 迭代数未记录；`wall_seconds` 之外的时间（cache 回读、
  输入读取、独立残差）未计入。
- 表中 `n=1` 表示证据只列出一个数值；原文只在 §6.4.5 明确写“单样本”，
  §9.5 的 1e-6 回读未声明重复数。

### 4.7 已作废：`atol=1e-4` 的“MG”数字

| 用例 | restart | `coarse_max_iter` | `coarse_tol_factor` | wall (s) | 真残差 | 状态 | 判定 |
|---|---:|---:|---:|---:|---:|---|---|
| r3_m200_t4 | 3 | 200 | `1e4` | 0.136258 | `6.756443214052956e-05` | PASS | **fine-BiCGStab 回退，不是 MG** |
| r1_m200_t4 | 1 | 200 | `1e4` | 0.135579 | `6.756443214052956e-05` | PASS | 同上 |
| r6_m200_t4 | 6 | 200 | `1e4` | 0.136228 | `6.756443214052956e-05` | PASS | 同上 |

机制（`pyqcu/testing/simple/kupl_multigrid.py:215`）：

```python
argv[ARGV_MG_LEVEL_BASE + level - 1] = atol * coarse_tol_factor * (3 ** (level - 1))
# atol=1e-4, coarse_tol_factor=1e4, level=1 → coarse_atol = 1.0
```

粗层 BiCGStab 的初始残差等于 `‖rhs‖`，目标 `1.0 × ‖rhs‖`，首行判据立即成立，
解保持 0；修正恒为 0 → damping 搜索全部回滚 → `fallback_to_fine=true` →
最终走 fine BiCGStab。因此：

- 这组 `0.135579–0.136258 s` 只能标为“MG 入口 + fine 回退”，**不得**写成
  “MG 比 BiCGStab 快 21.8%”；
- 历史 `0.128289 s`（`restart=1, cmax=200, tol=1e5`）同样是回退，
  “MG 快 26.0%”作废；
- 真实 MG 的对照必须使用 `atol=1e-6`（§4.6）。

### 4.8 调优开关与负结果（dev143）

| 变量 | 候选 | 中位/结果 | 判断 |
|---|---|---|---|
| `OMP_WAIT_POLICY` | `ACTIVE` 对 `PASSIVE` | `0.249223` 对 `0.319600 s`（cn19650，同 lane） | **1.28×**，必要 |
| `KUPL_HALO_PACK6` | `0` | dslash `0.213639` 对 `0.096633 s`；solve `0.266636` 对 `0.139138 s` | 关闭慢 **2.21× / 1.92×**，默认开 |
| `KUPL_ENABLE_BOUNDARY_LANES` | `0` | `0.225429` 对 `0.185412 s` | 慢 **22%**，默认开 |
| `KUPL_T_LOCAL_IN_PART1` | `0` | `0.184149` 对 `0.185412`（5+5） | 0.68%，顺序未随机化，不可归因 |
| `KUPL_SHM_HALO_ZEROCOPY` | `1` | 3 样本中位 `0.172885` 对 `0.173362`；单次 profile 配对 `0.180983 → 0.172885` | 中位差 0.28%，在 ±4% 抖动内；**默认关** |
| `KUPL_ENABLE_T2_ZBLOCK_DSLASH` | `1` | `0.183248` 对 `0.184108 s`（3+3） | 0.47%，噪声；保持 opt-in |
| `GOMP_SPINCOUNT` | `1000` / `30000` | `0.183795` / `0.185177` | 噪声 |
| `KUPL_HALO_DEDUP_PACK` | `0` | `0.186914` | 略差，噪声 |
| `KUPL_MEMORY_POLICY` | `bind` | `0.187453` 对 `0.186068` | 略差，噪声；`bind` 曾在 cn19650 触发 `SIGKILL`（RC=137） |
| `-ffast-math` | 构建 | `0.204288` 对 `0.207077` | 1.4%，噪声；非默认 |
| block-dslash 阈值 | `8192` | `0.244864` 对 `0.249223` | 1.8%，噪声 |
| `--levels`/`restart` | 32 rank MG | 见 §4.7 | 1e-4 下回退；真实 MG 见 §4.6 |

## 5. dev143 对 stab64 对照

### 5.1 最接近的可比对

| 对照 | stab64 | dev143 | 数值比 | 是否受控 | 结论 |
|---|---|---|---:|---|---|
| cn19699、L24、c64、32 rank、BiCGStab、atol `1e-4` | `0.5193 s`（32×12，y76 配方，16 it，`3.5412e-6`，n=1） | `0.154883 s`（32×18，shm，12 it，`5.3849e-5`，n=8） | **3.35×** | **否**：thread、halo、源、迭代/停止点不同 | 只能写“当前最低中位 vs 历史最低单样本”，不能写受控加速 |
| 同节点、同 32×18、同 grid | `1.5979 s`（x_case_dbg，16 it，`3.54122e-6`，n=1） | `0.173362 s`（全 1 源，11 it，`6.7564e-5`，n=3） | **9.22×** | **否**：stab64 含 profiling/fused/oversubscribe；源不同 | 不能把 9.22× 全归因于代码 |
| dev143 内部受控 A/B（同节点/L24/32 rank/同源） | — | shm `0.154883` 对 MPI `0.181733` | **1.173×**（14.77%） | **是**（8+8 交替） | dev143 内唯一可引用的受控 halo A/B |

不存在同源、同迭代、同 halo 的第三方交叉点；因此无法把 `3.35×` 分解成
代码、硬件与口径各自的贡献。

> **建议对外表述**：
> “在 cn19699、L24、c64、32 rank、BiCGStab、atol=1e-4 范围内，dev143 的
> 当前最低中位为 `0.154883 s`（n=8，共享 halo）；stab64 的可引用最低单样本为
> `0.5193 s`（32×12，y76 原配方）。两者相差 3.35×，但线程、halo、源与停止点
> 不同，不是受控加速比。dev143 内受控的共享 halo 对 MPI 改善为 14.77%。”

### 5.2 MG 对照

| 对照 | stab64 | dev143 | 数值比 | 可比性 |
|---|---|---|---:|---|
| L24 MG（r50，单次 coarse correction）登录节点最低绝对耗时 | `0.9473317482 s`（16×8，grid `1 1 1 16`，atol `1e-4`，max_iter 50，E4 cache，mg_grid `2 2 2 1`，真残差 `5.5308248713e-5`）；计算节点 true-MG 当时 **BLOCKED** | `0.519831 s`（32 rank，grid `2×2×2×4`，atol `1e-6`，cache 回读） | **1.82×** | 节点/rank/grid/atol 全变；**不是受控** |
| L24 true-MG 十种分格扫描最优 | `1.686342053115368 s`（16×8，grid `1 1 1 16`，atol `1e-4`） | 同上 | **3.24×** | 与上一行不是同一批次；**不能与 r50 合并** |
| L24 true-MG 登录节点同窗 A/B | MG `0.789756/0.705146/0.704241 s` 对 BiCG `1.397243/1.230783/1.091866 s`（16×8，restart 10/20/50） | 同格点/grid 4-rank `8³×16`：MG `0.037935` 对 BiCG `0.031577 s`（atol `1e-6`） | — | 格点/环境/容差/cache 路径不同；只能说明“MG 不总是更快” |
| L24 MG cache 构建 | `119.57 s`（登录节点 time block 1，143,335,424 B）/ `80.70 s`（登录节点并行 local-stencil L24） | `24.476 s`（32 rank patch，含 export；节点未逐条记录） | — | 并行度、节点与计时字段不同；**不得相除** |
| L32 MG | **BLOCKED**（1-rank cache 与 32-rank BiCG 均超 10 min） | `1.175139 s`（32 rank，atol `1e-6`）+ `53.023 s` cache 构建 | — | dev143 首次有 L32 可引用 MG 数字，但为单样本 |

### 5.3 cache 构建对照

| 项 | stab64 | dev143 | 说明 |
|---|---|---|---|
| L24 cache 文件 | `143,335,424 B`（time block 1） | `143,335,424 B`（`cache_l24_parallel.h5`） | 大小相同，identity/构建路径不同 |
| L24 builder | `119.57 s`（登录节点 time block 1）；`80.70 s`（登录节点并行 stencil，opt-in） | `493.638 s`（1 rank × 18 thread，含输入/export）；`24.476 s`（32 rank patch，含 export） | 1-rank 与 32-rank、不同字段，不能相除 |
| L32 cache | 无可用（BLOCKED） | `53.023 s`（32 rank patch） | dev143 首次打通 |
| `32³×12` cache | 未记录 | `11.368 s`（32 rank，grid `4×2×2×2`） | 新增 |
| 并行 stencil | opt-in | 默认开 | dev143 的 1-rank 构建仍 493.6 s，说明并行 stencil 本身没有把 1-rank 构建带到 30 s 量级 |

### 5.4 带宽对照（不同口径）

| 指标 | stab64 | dev143 | 可否相除 |
|---|---:|---:|---|
| solve 级有效带宽 | `149.51 GB/s`（y76 `0.4346 s`，模型 64.97 GB） | 未给 solve 级模型；dslash 级 `522 GB/s` 聚合 | **否**：模型与范围不同 |
| dslash 级有效带宽 | 未给 GB/s（只有 halo/interior/wait/boundary 分项，70 calls，总 `0.6013 s`） | `16.3 GB/s·rank` 解析 / `12.3 GB/s·rank` profiler | **否**：一侧无同一口径的带宽值；且批次不同 |
| STREAM triad | HBM `5465–6801 GB/s`；单域 `279–428 GB/s`；DDR `616 GB/s` | `18.3–20.1 GB/s·rank × 32 ≈ 585–643 GB/s` | 可作分母对照，但需注明 HBM/DDR 与线程配置 |
| 占峰值 | HBM 的 `2.74%–4.10%`；DDR 的 `24.3%` | dslash 约占 triad 峰值 `87%` | **否**：solver 级 vs dslash 级 |

### 5.5 不应做的比较

1. **不要把 `0.154883` 与 `0.4346` 直接写成 2.81× 代码加速**：节点不同
   （cn19699 对 cn10579）、rank×thread 不同（32×18 对 32×12）、源/迭代/停止点
   不同，且 `0.4346` 是 B 级 n=1。
2. **不要把 `0.154883` 与 `0.173362` 当作同一基准的重复测量**：前者是点源
   （seed 42）、12 it、`5.3849e-5`；后者是全 1 源、11 it、`6.7564e-5`。
3. **不要把 `0.135579` 当作 MG 加速**：它是 fine-BiCGStab 回退（§4.7）。
4. **不要把 `0.135579` 与 dev143 的 `0.519831` 当作“MG 变慢 3.8×”**：
   前者是回退、atol `1e-4`、t-only grid；后者是真 MG、atol `1e-6`、均衡 4D grid。
5. **不要把 dslash 的 `522 GB/s` 与 solve 的 `149.51 GB/s` 相除**：前者是
   单算子解析流量，后者是整个 solve 的模型流量。
6. **不要把 CUDA/QUDA 的 L32 `0.214–0.219 s` 当成 KUPL L32 基线**：那是
   A100 上的 CUDA Legacy-MG，不是 CPU 数据。

## 6. 参数总表（按用例）

### 6.1 stab64（优化前）

| 用例 | 格点 | dtype | ranks × thr | grid | 算法 | atol | 迭代 | 真残差 | wall (s) | n |
|---|---|---|---|---:|---|---:|---:|---|---:|---:|
| y76 K1 | `24³×64` | c64 | 32 × 12 | `1 1 1 32` | BiCGStab `levels=1/restart=0` | `1e-4` | 16 | `3.5412238955e-6` | 0.4346 | 1 |
| y76 复现 | `24³×64` | c64 | 32 × 12 | `1 1 1 32` | 同上 | `1e-4` | 16 | `3.5412238955e-6` | 0.5193 | 1 |
| x_case_dbg | `24³×64` | c64 | 32 × 18 | `1 1 1 32` | 同上 + profiling | `1e-4` | 16 | `3.54122e-6` | 1.5979 | 1 |
| L24 true-MG（登录节点，十种分格扫描） | `24³×64` | c64 | 16 × 8 | `1 1 1 16` | Legacy-MG（配方见 §3.2） | `1e-4` | `None` | PASS（fail-closed；数值未抄录） | 1.686342 | 1 |
| L24 MG（登录节点，r50 full-residual；单次 coarse correction） | `24³×64` | c64 | 16 × 8 | `1 1 1 16` | Legacy-MG `mg_grid=2 2 2 1` | `1e-4` | `None` | `5.5308248713e-5` | 0.9473317482 | 1 |

### 6.2 dev143（优化后）

| 用例 | 格点 | dtype | ranks × thr | grid | 算法 | atol | 迭代 | 真残差 | wall (s) | n |
|---|---|---|---|---:|---|---:|---:|---|---:|---:|
| L24 shm | `24³×64` | c64 | 32 × 18 | `1 1 1 32` | BiCGStab + shm halo | `1e-4` | 12 | `5.3849e-5` | 0.154883 | 8 |
| L24 MPI | `24³×64` | c64 | 32 × 18 | `1 1 1 32` | BiCGStab + MPI halo | `1e-4` | 12 | `5.3849e-5` | 0.181733 | 8 |
| L24 全 1 | `24³×64` | c64 | 32 × 18 | `1 1 1 32` | BiCGStab + shm | `1e-4` | 11 | `6.7564e-5` | 0.173362 | 3 |
| L32 全 1 | `32³×64` | c64 | 32 × 18 | `1 1 1 32` | BiCGStab + shm | `1e-4` | 12 | `6.8318e-5` | 0.348549 | 3 |
| L24 MG 1e-6 | `24³×64` | c64 | 32 × 未记录 | `2×2×2×4` | Legacy-MG cache 回读 | `1e-6` | `None` | `4.536e-07` | 0.519831 | 1 |
| L32 MG 1e-6 | `32³×64` | c64 | 32 × 未记录 | `2×2×2×4` | 同上 | `1e-6` | `None` | `4.543e-07` | 1.175139 | 1 |
| `32³×12` MG 1e-6 | `32³×12` | c64 | 32 × 未记录 | `4×2×2×2` | 同上 | `1e-6` | `None` | `4.461e-07` | 0.266999 | 1 |
| `8³×16` MG 1e-6 | `8³×16` | c64 | 4 × 未记录 | `2×2×1×1` | 同上 | `1e-6` | `None` | `3.409e-07` | 0.037935 | 1 |
| `8³×16` BiCG 1e-6 | `8³×16` | c64 | 4 × 未记录 | `2×2×1×1` | BiCGStab | `1e-6` | 未记录 | `5.421e-07` | 0.031577 | 1 |
| L24 MG 1e-4（回退） | `24³×64` | c64 | 32 × 18 | `1 1 1 32` | MG 入口 → fine 回退 | `1e-4` | `None` | `6.7564e-5` | 0.135579 | 1 |

> 表中“未记录”不是“默认值”：线程/迭代/`max_iter` 等字段若未在证据中逐条
> 出现，本文不补齐，避免把配置值冒充实测值。

## 7. 数据来源与可追溯性

### 7.1 Git 标签与提交

| 标签 | 提交 | 日期 | 说明 |
|---|---|---|---|
| `stab64` | `2c07cf94b6da68b4e5afa3d66ac6b5192ff3e51b` | 2026-11-12 08:59 | release 0.1.15；KUPL 优化前基线 |
| `dev143` | `4f60e3487ca145f0cf64930f38a4518e5f705c00` | 2026-11-14 20:53 | 多 rank patch cache、MG 容差语义修正 |

### 7.2 文档 URL（示例基址）

```text
https://gitlab.itp.ac.cn/ZhangXin8069/PyQCU/-/blob/<tag>/<path>
```

| 内容 | dev143 路径 | stab64 路径 |
|---|---|---|
| L24 单节点实测（dev141） | `docs/kupl_l24_perf_20261113.md` | — |
| 多 rank/多线程扩展性、MG patch | `docs/kupl_mpi_scaling_20261114.md` | — |
| L24 全配置耗时台账 | `docs/qcu_l24_solver_timing_20261111.md` | `docs/qcu_l24_solver_timing_20261111.md` |
| 鲲鹏带宽实测 | `docs/qcu_l24_environment_bandwidth_20261111.md` | `docs/qcu_l24_environment_bandwidth_20261111.md` |
| 节点核验 | `docs/kupl_kunpeng_node_survey_20261114.md` | — |
| 鲲鹏进展/true-MG | — | `docs/kupl_kunpeng_progress_20261110.md` |
| KUPL 技能摘要 | `skills/pyqcu-kupl/SKILL.md` | `skills/pyqcu-kupl/SKILL.md` |
| S8 阶段日志（patch 构建） | `logs/v202611141855.txt` | — |
| S7 阶段日志（cache 1e-4 扫描） | `logs/v202611141551.txt` | — |
| S5 阶段日志（全 1 源/TH） | `logs/v202611141241.txt` | — |

用户给出的示例链接（与本文同一来源体系）：

```text
https://gitlab.itp.ac.cn/ZhangXin8069/PyQCU/-/blob/dev143/docs/kupl_l24_perf_20261113.md
```

### 7.3 日志与远端产物

| 证据 | 本地/远端 | 覆盖内容 |
|---|---|---|
| `logs/v202611141855.txt` | 本地已跟踪 | S8：根因、32-rank patch 构建表、1e-6 对照、t-only 禁用 |
| `logs/v202611141551.txt` | 本地已跟踪 | S7：1-rank cache、1e-4 扫描、BiCGStab 对照 |
| `logs/v202611141241.txt` | 本地已跟踪 | S5：全 1 源 3 样本、TH 扫描、LIME 校验 |
| `logs/v202611141329.txt` | 本地已跟踪 | S6：零拷贝 3 样本、L32 TH、两节点 SSH 阻塞 |
| `data/kupl_opt_20261113/real8/*.json` | **本地**，仅 `8³×16×16` 小格 | 本地 A 级，不用于 L24 结论 |
| `data/kupl_scaling_20261114/**` | **远端**，本机不存在 | MG cache 身份、1e-4 扫描、1e-6 回读 |
| `/home/share/nsls_yangyb/PyQCU/data/kupl_opt_20261113/remote/**` | **远端**，本机未挂载 | L24/L32 生产 `report.json`/`meta.txt` |
| `/home/share/nsls_yangyb/PyQCU/data/node_survey_20261114/**` | **远端**，本机未挂载 | 18 节点频率/拓扑原始输出 |

### 7.4 本机复核状态

- 本文的数值与 `docs/kupl_l24_perf_20261113.md`、
  `docs/kupl_mpi_scaling_20261114.md`、`logs/v202611141855.txt` 逐项对照，
  未发现转抄冲突；但**没有**重新打开远端 `report.json` 做字段级复核。
- `data/kupl_scaling_20261114/**` 在本机不存在，文中已按文档值标注为 C 级。
- `docs/qcu_l24_solver_timing_20261111.md` 的 `0.4346 s` 是 B 级 n=1，
  其 raw JSON 不在工作树；本文保留该限制。
- 本文没有重新运行任何算例；因此“当前机器上的可复现性”未验证。

## 8. 缺失项、风险与下一步

### 8.1 必须补齐才能做受控对照的字段

1. **gauge 身份**：文件路径/SHA256、生成参数或 seed。除少数运行外，两侧证据
   都未逐条记录；`kupl_l24_perf_20261113.md` §9 已把 gauge 配置身份列为待补。
2. **源身份**：源类型、源坐标、文件 SHA256。点源与全 1 源不可互换。
3. **构建 provenance**：KUPL 库 SHA256、编译器版本、完整编译命令、LTO/
   `-ffast-math` 开关。
4. **样本统计**：dev143 的 MG 三组、4-rank 对照、TH 扫描均为单样本；应补
   ≥5 样本或交替配对。
5. **线程数**：32-rank patch 构建与 1e-6 MG 回读的线程数未在产物中逐条记录。
6. **`max_iter`**：dev143 的运行未逐条抄录；虽 11–12 it 远低于上限，仍应记录。
7. **cache identity 的完整字段**：`mg_grid`/`nv_iters`/`dof`/seed/dtype 在
   §9.5 未重述，需从远端 cache JSON 复核。
8. **频率与负载**：跨节点比较前必须记录 `scaling_cur_freq`、作业号、采样
   时刻；本文的 1.55 GHz 是 2026-11-14 09:00 的时点采样。

### 8.2 风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| 把 `0.154883` 当“生产 atol 下的最优” | 生产 c64 目标为 `1e-6`，该值是 `1e-4` | 引用时必须写 `atol=1e-4` 与 12 it |
| 把 `0.135579` 当 MG 加速 | 实际是 fine-BiCG 回退 | 与 §4.6 的 `atol=1e-6` 数字分开引用 |
| 把 `0.5193` 与 `0.154883` 相除当代码加速 | thread/halo/源/停止点不同 | 用 §5.1 的限定表述 |
| 把 dslash `522 GB/s` 当 solve 带宽 | 口径不同 | 只与 STREAM 分母比较，不跨层相除 |
| 把 stab64 的 `119.57 s` 与 dev143 的 `24.476 s` 相除 | 并行度/字段不同 | 只并列，不算加速比 |
| 跨节点直接比较 | 节点差可达 20%，频率差可达 1.29× | 记录节点、频率、作业号；同节点 A/B 优先 |

### 8.3 建议的下一步

1. 在**同一节点、同一时段**用同一源/同一 gauge 补一组
   `stab64` 与 `dev143` 的 32r×18t 对照（至少 5 样本），把“历史最低单样本 vs
   当前最低中位”替换成受控 A/B。
2. 对 32-rank MG 三组补 ≥5 样本，并记录线程数、`mg_grid`、`nv_iters`、
   `dof`、cache identity。
3. 复跑 `atol=1e-6` 的 L24 MG 与 BiCGStab 同格点/grid 对照，确认 20% 的负收益在
   多样本下是否稳定。
4. 把 `builder_seconds` 拆成“读入 / patch / stencil / export”四段并逐项记录，
   避免 1-rank 与 32-rank、含/不含 export 的数字混比。
5. 补两节点 hostfile/launcher 后验证跨节点 MPI 回退；当前跨节点仅有代码分支
   审查，未做运行时验证。

## 9. 复现配方（证据中给出的部分）

### 9.1 BiCGStab（L24/L32，dev143）

```bash
dsub -q q_hpcapp -T 900 -n <tag> \
     -N 32 -rpn 32 -R "cpu=18,mem=8G" \
     env OMP_WAIT_POLICY=ACTIVE \
         KUPL_NUMA_BIND=1 \
         KUPL_MEMORY_POLICY=preferred \
         KUPL_MEMORY_NODE=local+16 \
         KUPL_HUGEPAGE=1 \
         KUPL_ENABLE_T2_LANES=1 \
         KUPL_HALO_PACK6=1 \
         KUPL_HALO_DEDUP_PACK=1 \
         KUPL_T_LOCAL_IN_PART1=1 \
         KUPL_ENABLE_BOUNDARY_LANES=1 \
     bash ~/sh/x_case_l24_v7.sh <tag> 32 18 2 600
```

作业内：

```bash
source /work_ssd/software/HPCKit/25.2.1/setvars.sh
export PATH=/work_ssd/software/HPCKit/25.2.1/hmpi/bisheng/release/hmpi/bin:/work_ssd/software/soft/tool/anaconda3/2023.3/bin:$PATH
export OMPI_MCA_rmaps_base_oversubscribe=1
mpirun -np 32 --bind-to none --map-by ppr:2:numa --allow-run-as-root ...
```

程序参数（按证据）：

```text
--lat 24 24 24 64 --mass -0.277 --csw 1.160920226 --atol 1e-4 \
--dtype c64 --levels 1 --restart 0 --grid 1 1 1 32
```

> `max_iter` 未在 dev143 证据中逐条抄录；stab64 配方为 `--max-iter 50`。

### 9.2 MG cache 构建与回读（dev143）

```text
# 1-rank cache（L24）
--lat 24 24 24 64 --mass -0.277 --csw 1.160920226 --dtype c64 \
--levels 2 --mg-grid 2 2 2 1 --nv-iters 2 --restart 1 \
--coarse-max-iter 200 --coarse-tol-factor 1e4 \
--write-coarse-cache data/kupl_scaling_20261114/cache_l24_parallel.h5

# 32-rank patch 构建（均衡 4D）
--lat 24 24 24 64 --grid 2 2 2 4 --levels 2 --mg-grid 2 2 2 1 --nv-iters 2

# 1e-6 真实 MG 回读
--atol 1e-6 --coarse-tol-factor 1e4 --coarse-cache <cache.h5> ...
```

> `--coarse-tol-factor 1e4` 与 `atol=1e-4` 组合会得到 `coarse_atol=1.0` 并
> 触发 fine 回退；真实 MG 必须用 `atol=1e-6`（或显式降低 factor）。

### 9.3 环境入口的已知坑

- `--mpi hmpi` 不等于 MPI 进入 `PATH`；作业脚本内必须 source HPCKit 的
  `setvars.sh`。
- 2026-11-14 10:40 起，HMPI 的 `mpirun` 曾对所有选项报 `unknown option`；
  `OPAL_PREFIX` 必须在作业/驱动内显式设置。该问题不是 KUPL 代码问题。
- `-N 16 -rpn 1` 只执行 1 个 task（`BATCH_TASK_INDEX=0`），不能用于一次
  采集多节点。
- `-N 1 -R cpu=576` 与 `-N 32 -rpn 32 -R "cpu=18"` 的 cgroup 行为不同；
  后者是历史可复现配方。

## 10. 引用模板

对外材料建议使用以下限定句，避免把不同口径拼成单一加速比：

```text
KUPL CPU 后端，L24（24^3×64），complex64，32 MPI rank × 18 OpenMP thread，
grid 1 1 1 32，单节点共享内存 halo，OMP_WAIT_POLICY=ACTIVE，
mass=-0.277，csw=1.160920226，atol=1e-4，BiCGStab（levels=1, restart=0），
12 次迭代，独立 FULL Wilson-Clover 真残差 5.3849e-05，solve 墙钟中位
0.154883 s（n=8，范围 0.150240–0.157304），节点 cn19699，1.55 GHz 现态。
该值不含 Python 导入/初始化、输入读取与 cache 构建；与 stab64 的 0.5193 s
（32×12、16 it、3.5412e-6）不是受控 A/B。
```

```text
KUPL Legacy-MG，L24（24^3×64），complex64，32 rank，均衡 grid 2×2×2×4，
atol=1e-6，cache 回读，solve 墙钟 0.519831 s，真残差 4.536e-07，
verify_status=0，单样本；cache 构建 24.476 s（含 coarse export），
不计入 solve 墙钟。真实 MG 在同格点/grid 4-rank 8^3×16 上比 BiCGStab 慢 20%
（0.037935 对 0.031577 s），不得宣称 MG 普遍更快。
```

## 11. complex128（c128）数据与缺口

### 11.1 结论

- **dev143 与 stab64 的鲲鹏 L24 生产性能台账没有 c128 solve 墙钟。**
  dev143 的 c128 只出现在正确性/容差门（共享 halo、Legacy-MG levels 1/2、
  deflate、`restart=3`、ABI 差分、UBSAN/ASAN），没有可引用的 c128 时间。
- **2026-11-15 矩阵批次补了一个鲲鹏 c128 点，但只有 n=1、且口径不同**：
  L24 `24³×64`、c128、32×18、t-finest `1 1 1 32`、shm1、`atol=1e-9`（`--strict-acceptance`）、
  31 it、真残差 `5.3257e-11`、`wall=0.475555 s`（`d2ce9be32b406b0f`，cn19699，单样本）；
  同格 c64 是 `0.240473 s`/18 it。比值 `1.978×` 里迭代数已占 `31/18=1.72×`，
  **不能**当作 c128 单次算子成本，也不能与本文的 c64 dev143 数据合并成一条曲线。
  出处与同批 3 条 c128（`f19ba1fe`、`f212adf8`、`749b3b02`）见
  [`kupl_kunpeng_matrix_20261115.md`](kupl_kunpeng_matrix_20261115.md) §6.4。
- 可引用的 c128 L24 墙钟除此以外全部来自 **dev99–dev101 的 x86 主机**
  （`ENV-X86-18C` 或“硬件未完整记录”），不是鲲鹏、也不是 dev143/stab64
  同窗数据；**不能**与 c64 的 `0.154883 s` 相除。
- 历史 c128 数字全部是 **parity-Schur BiCGStab 类路径**（D99-027 明确
  `restart=0`；D99-028 为 `applyCloverBistabCgQcu`），不是 c128 coarse
  V-cycle；`10.282 s / 21 it` 也是全格点 BiCGStab 路径。

### 11.2 历史 x86 c128 L24 数据（dev99–dev101）

`24³×64`、c128、mass `-0.277`、csw `1.160920226`、5 次稳态中位
（除注明外）；r1/r2 的线程数按同批记录为 24 thread/rank，硬件未完整记录。

| 配置 | ranks × thr | wall (s) | 迭代 | 独立真残差 | 备注 |
|---|---|---:|---:|---:|---|
| r1 baseline | 1 × 24 | 20.196165 | 23 | `6.962408e-8` | 未优化 Clover onsite |
| r1 spin6 | 1 × 24 | 16.116355 | 23 | `6.962408e-8` | 6×6 块 |
| r1 spin6 + inv6 | 1 × 24 | 15.592787 | 23 | `6.962408e-8` | 优化后 1-rank |
| r2 baseline | 2 × 24 | 14.449964 | 23 | `6.962408e-8` | 2-rank |
| r2 spin6 | 2 × 24 | 13.106006 | 23 | `6.962408e-8` | 6×6 块 |
| r2 spin6 + inv6 | 2 × 24 | 11.472949 | 23 | `6.962408e-8` | 优化后 2-rank |
| D99-027：`restart=0` parity-Schur | 4 × 4 | 10.282 | 21 | `6.079e-8` | `ENV-X86-18C`，dev101；非 V-cycle |
| D99-028：`applyCloverBistabCgQcu` | 4 × 4 | 11.311 | 23 | `6.96241e-8` | 同 LIME；独立残差 |

c128 相对 c64（同窗口、同 `inv6` 变体）：

| 精度 | 1 rank (s) | 2 ranks (s) | 迭代 |
|---|---:|---:|---:|
| c64 `inv6` | 7.769662 | 6.351769 | 20 |
| c128 `inv6` | 15.592787 | 11.472949 | 23 |
| c128 / c64 | **2.01×** | **1.81×** | — |

读法：

- 倍率同时包含迭代数差（20 对 23）与每迭代成本差，不能全部归因于 c128 的
  单次算子成本。
- 这些数字是 **x86 主机**、1/2/4 rank 的低并行配置，不是鲲鹏 32 rank；
  与 dev143 的 `0.154883 s` 之间隔着硬件、rank、精度、算法四层差异。
- `10.282 s` 与 `11.311 s` 都是 `restart=0` 的全格点 BiCGStab；不要写成
  “c128 MG 10.282 s”。

### 11.3 dev143 的 c128 正确性覆盖（无墙钟）

| 门 | 配置 | c128 结果 | 有墙钟？ |
|---|---|---|---|
| 共享 halo ABI 差分 | `test_new_abi.py`，`2 2 2 4`、`--grid 2 1 1 1`、2 线程 | PASS | 否 |
| block dslash/Schur/solve | `test_dslash_block_mpi.py --lattice 4 4 8 8 --grid 2 1 1 1 --threads 2 --solve` | dslash `1.65e-16`、schur `1.68e-16`、solve `8.67e-14` | 否 |
| Legacy-MG 共享 halo | levels 1/2、deflate、`restart=3` | PASS、`verify=0` | 否 |
| t-only MG 合成资产 | 2/8/32 rank、`local_t=2`、coarse `[4,4,4,1]` | PASS、`verify=0`；deflate 误差约 `1.98e-12`（c64 为 `9.811e-09`） | 否 |
| UBSAN/ASAN | 2/4 rank 与 `24 24 24 2` | 0 报告、PASS | 否 |

对照 c64 同门：dslash `8.78e-08`、schur `9.21e-08`、solve `1.15e-06`。
这些是数值一致性证据，不是性能数据。

### 11.4 为什么 dev143 没有 c128 性能

1. dev143 的 L24/L32 性能战役（32r×18t、共享 halo、MG cache/回读）全部用 c64；
   1-rank cache、32-rank patch cache、`atol=1e-6` MG 也全部是 c64。
2. c128 的场与 scratch 约为 c64 的 2 倍；历史同配置墙钟约 `1.8–2.0×`。
   32 rank 下的 c128 内存与 cache 体积未做过容量验证。
3. 生产容差不同：c64 是 `1e-6`、c128 是 `1e-9`（相对 norm）；把 c64 的
   `1e-6` 结果直接当 c128 基线会低估迭代与墙钟。
4. 因此本文第 3–6 节的所有 c64 对照**不包含** c128；c128 不能从这些数字外推。

### 11.5 需要补的 c128 数据

1. L24 `24³×64`、32r×18t、共享 halo、`levels=1/restart=0`、`atol=1e-9`、
   3–5 样本，记录迭代、真残差、RSS/峰值内存。
   **现状**：2026-11-15 矩阵批次只有 1 个样本（`0.475555 s`/31 it，见 §11.1），
   离"3–5 样本"还差；balanced + c128 在 L24 上仍是 0 行。
2. c128 MG cache：1-rank 构建时间/文件大小 + 32-rank patch 构建
   （`2×2×2×4`、`4×2×2×2`），并做 `atol=1e-9` 回读。
   **现状**：矩阵批次的 `cache-3b6e1e37b3af0b9c` 是本轮唯一 L32 c128 cache
   （`build_mg_cache_seconds=130.371`、`verify_status=0`、notes 为 `rc=124` 非干净构建）；
   L24 c128 cache 是 `cache-9b7a383e23a633be`（`48.191 s`）。两者都是 global，非 patch 对照。
3. L32 c128 的容量与 cache 可行性；当前没有任何 c128 L32 **solve** 数据（快照内 2 条全 FAIL）。
   快照后 `82ba09ec3a8e0d2d` 是 L32 c128 `mg2` 的单条 PASS（残差 `4.03e-10`），
   但 n=1 且用 point 源，只能证明"可算"，不能当性能数据。
4. 若要在报告中给出 c128 相对 c64 的倍率，必须在**同节点、同 rank×thread、
   同源、同 gauge、同停止点**下重测；不能用本文的历史 x86 数字，也不能用
   `0.475555/0.240473=1.978×`（迭代数已占 `1.72×`）。

### 11.6 不能做的比较

- 不能把 c128 `15.592787 s`（x86、1 rank×24 thread）与 c64 `0.154883 s`
  （鲲鹏、32×18、共享 halo）相除。
- 不能把 `10.282 s / 21 it` 写成“c128 MG”；它是 `restart=0` 的
  parity-Schur BiCGStab。
- 不能把 c64 的 `1e-6` 与 c128 的 `1e-9` 当作同一收敛精度。
- 不能把 dev143 c128 的 PASS 结果当成性能通过；它们只证明数值一致性。

## 12. 交付边界与审计状态

- 本文是**新增**的 `docs/kupl_performance_20261115.md`；创建时按任务约束，
  **未修改**任何既有文档、代码、技能或索引，也未运行新算例。
- **状态更新（2026-11-15 收口后）**：本文已在同批文档登记中进入
  `docs/AGENTS.md`、`skills/pyqcu-docs/references/project-docs-map.md` 与
  `extracted-guidance.md`；`docs` 计数现为 **53**（同批新增
  `kupl_l24_c64_mg2_shm1_boundary_20261115.md`）。登记完成后的
  `python -B skills/pyqcu-docs/scripts/audit-docs-skills.py` 为 `PASS docs=53 skills=36`。
  因此下面"未登记/审计未通过"的段落是**创建时**的历史记录，不再是现状。
- 创建时的记录（保留作为过程证据）：审计曾报 `docs=49`、计数 48≠49、
  `DOCS-INDEX`/`DOCS-MAP`/`DOCS-GUIDANCE` 未登记；另有 dev143 既有的 3 份文档
  未进入 `extracted-guidance`（`kupl_kunpeng_node_survey_20261114.md`、
  `kupl_mpi_scaling_20261114.md`、`kupl_vs_cuda_algorithms_20261114.md`），以及
  `skills/pyqcu-docs/SKILL.md` 的计数 45≠48。
- 登记动作（已完成）：`docs/AGENTS.md` 增加本文职责行并更新计数；
  `project-docs-map.md` 登记本文并更新计数；`extracted-guidance.md` 登记本文
  并补齐上述 3 份既有文档的结论覆盖行；重跑 `audit-docs-skills.py` 清零 `DOCS-*`。
- 2026-11-15 的 docs 补全批次对本文追加了 §1.2 第 7 条（`atol` 口径与 244.1×）、
  §11.1/§11.5 的鲲鹏 c128 现状，以及本节状态更新；**没有**改任何数值。
