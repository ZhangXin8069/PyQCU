# KUPL 鲲鹏 HBM 高性能配置（2026-11-12）

> 证据等级：C 级时点证据，未打 tag，单样本为主；本文数值仅来自
> [`skills/pyqcu-kupl/references/kupl-hbm-scan-20261112-evidence.md`](../skills/pyqcu-kupl/references/kupl-hbm-scan-20261112-evidence.md)，
> 该文件未出现的数值一律标为“待补”，不作推算。

本文记录 2026-11-12 会话在 LX2 集群上验证过的 KUPL HBM 绑定配置、MPI 映射、
调度器用法、超时协议和实测结果。通用的构建、变量语义和规划器用法见
[`kupl_kunpeng_guide.md`](kupl_kunpeng_guide.md)，本文不重复。

## 1. 硬件与拓扑

拓扑探测在 `cn19701` 上完成；性能样本在 `cn19699`/`cn19698` 上采集，
两者不是同一节点，不能混用。

| 项目 | 实测值 |
|---|---|
| sysfs NUMA 节点数 | 32 |
| `node0–15` | 每节点约 32 GB |
| `node16–31` | 每节点 4 GB，片上 HBM |
| KUPL 报告 `numa_nodes` | 16 |
| KUPL 报告 `memory_node` | `16..31` |
| 每 rank `cpuset` | 8–12 个 CPU |
| `memory_policy` | `bind` |
| `bound_threads` | `threads` |
| 拓扑来源 | `source=sysfs` |
| `dnode` 显示的 `cn196xx` | CPU=608，内存约 578–592 GB |

KUPL 把前 16 个节点计为 CPU NUMA 域（`numa_nodes=16`），并将 `local+16`
映射到 `16–31` 的片上内存节点。这里的 `16` 是本次现场探测得到的 offset，
不是所有鲲鹏平台的常量；更换机型、BIOS 或节点配置后必须重新探测
（`docs/kupl_kunpeng_guide.md:176-178`）。

## 2. 绑定变量

本次所有用例使用同一套绑定变量：

| 变量 | 本次取值 | 语义与约束 | 依据 |
|---|---|---|---|
| `KUPL_NUMA_BIND` | `1` | 由库按 sysfs cpulist 选择 NUMA-local CPU 并尝试内存策略；必须配合 MPI `--bind-to none`，且不得设置 `OMP_PROC_BIND`/`OMP_PLACES`/`GOMP_CPU_AFFINITY`/`KMP_AFFINITY` | `docs/kupl_kunpeng_guide.md:78,90-95` |
| `KUPL_MEMORY_POLICY` | `bind` | 严格绑定本地节点，容量不足或节点不存在时失败关闭；`preferred` 会溢出到 DDR | `docs/kupl_kunpeng_guide.md:81` |
| `KUPL_MEMORY_NODE` | `local+16` | 指定后续库分配的 NUMA 内存节点；`16` 为本次现场 offset | `docs/kupl_kunpeng_guide.md:80,195-220` |
| `KUPL_REQUIRE_MEMORY_BIND` | `1` | `set_mempolicy` 失败从告警改为 fatal，不得静默退回默认分配 | `docs/kupl_kunpeng_guide.md:82,97-101` |
| `KUPL_HUGEPAGE` | `1` | 只对大型内部 scratch 发送 `MADV_HUGEPAGE` advice，不修改系统配置，也不保证实际使用大页 | `docs/kupl_kunpeng_guide.md:86,237-242` |
| `KUPL_SKIP_INPUT_PREBIND` | `1` | 跳过 `kupl_multigrid.py` 读取 gauge/source 前的临时 prebind lifecycle；该 lifecycle 原本只在 `KUPL_NUMA_BIND=1` 时执行 | `pyqcu/testing/simple/kupl_multigrid.py:682-687` |
| `KUPL_TOPOLOGY_REPORT` | `1` | 每 rank 输出实际 NUMA、cpuset、线程数和内存策略，作为绑定证据 | `docs/kupl_kunpeng_guide.md:85` |

本次测量还固定了下列变量：

```bash
export OMP_NUM_THREADS=TH
export KUPL_OMP_THREADS=TH
export KUPL_PROFILE_INIT=1
export KUPL_FUSED_BICG_DOTS=1
export KUPL_PROFILE_DSLASH=1
export KUPL_ALLOW_OVERSUBSCRIBE=1
export KUPL_BLOCK_DSLASH_MIN_HALF_SITES=16384
export KUPL_ENABLE_T2_ZBLOCK_DSLASH=0
```

`KUPL_FUSED_BICG_DOTS=1` 与库默认一致（`docs/kupl_kunpeng_guide.md:87`）；
`KUPL_ALLOW_OVERSUBSCRIBE=1` 显式允许线程数大于 cpuset CPU 数
（`docs/kupl_kunpeng_guide.md:84`）。`KUPL_PROFILE_INIT`、`KUPL_PROFILE_DSLASH`
用于产生分项与 DSLASH 计时；`KUPL_ENABLE_T2_ZBLOCK_DSLASH` 的语义在证据文件和
现有技能文档中均未说明，语义待补。

## 3. MPI 映射

复现命令的 MPI 部分：

```bash
mpirun --oversubscribe -np NP \
  --bind-to none \
  --map-by ppr:PPR:numa \
  --allow-run-as-root \
  <program> ...
```

- `--bind-to none`：`KUPL_NUMA_BIND=1` 时必需，CPU/内存绑定由 KUPL 在
  `applyInitQcu` 内完成；同时必须取消 OpenMP 侧的绑定变量
  （`docs/kupl_kunpeng_guide.md:90-95`）。
- `--map-by ppr:PPR:numa`：每个 CPU NUMA 域放置 `PPR` 个 rank。本次 32 秩
  用例均为 `ppr=2`，16/8/4 秩用例均为 `ppr=1`；因此 32 秩占满 16 个 CPU
  NUMA 域，16 秩也占满 16 个域，8/4 秩只占部分域。
- `--allow-run-as-root`：本次以 root 运行所需。
- `--oversubscribe`：与 `KUPL_ALLOW_OVERSUBSCRIBE=1` 配套，允许每 rank
  线程数超过其 cpuset 的 CPU 数。

## 4. 调度器用法

- 命令：`dsub` 提交，`djob`/`djob -ll` 查询，`dkill` 终止；队列 `q_hpcapp`；
  独占节点使用 `-x job`。
- `#DSUB` 头中的 `-N/-rpn/-R/-nl` 可能被忽略；调度参数必须写在 `dsub`
  命令行。
- 提交后用 `djob -ll <id>` 核对 `execNodes` 是否等于目标节点。
- 排除节点：`cn12038`、`cn10579`，清单见
  `data/kupl_hbm_opt_20261106/excluded_nodes.txt`。
- 账号上限：64 task 作业被 `testusers_rebuild` 上限拒绝
  （`mainStateReasonMsg: The limit <testusers_rebuild> has been exceeded`）；
  32 秩作业需在命令行显式写 `-N 32 -rpn 32`。
- `-nl` 指定的节点偏好可能被忽略：目标节点无资源时作业会挂起并报
  `No node is available to meet node preferences`，需确认节点状态或在空闲
  节点重提。

## 5. 超时与测量协议

- 逐用例 95 s 超时过短：16 秩端到端约 108–117 s（其中 Python 导入和初始化
  约 100 s），32 秩约 233 s。
- `RC=124` 不能直接判定为“卡死”；应把超时提高到 400–600 s 后再判。
- 本文结果表中的 `wall_seconds` 是 solve 墙钟；108–117 s / 233 s 是端到端
  时间，两者不可混用。
- 每次运行都检查真残差。本次固定对照为真残差约 `3.5412e-6`，验收阈值
  `2.44140625e-4`，状态 `PASS`。

复现用例参数（除注明外，本次所有用例一致）：

```bash
--lat 24 24 24 64 --mass -0.277 --csw 1.160920226 --atol 1e-4 \
  --max-iter 50 --dtype c64 --levels 1 --restart 0
```

`levels=1,restart=0` 是 parity-Schur BiCGStab，不是 multigrid；16 次迭代收敛。

## 6. 实测结果

节点独占（`-x job`），`wall_seconds` 为 solve 墙钟：

| 用例 | 节点 | 规模 | grid | ppr | wall (s) | 迭代 | 真残差 | 状态 |
|---|---|---|---:|---:|---:|---:|---|---|
| r32t12b | cn19699 | 32r×12t | 1 1 1 32 | 2 | **0.8410** | 16 | 3.54117e-6 | PASS |
| r32t18b | cn19699 | 32r×18t | 1 1 1 32 | 2 | 1.5979 | 16 | 3.54122e-6 | PASS |
| r16t8b | cn19699 | 16r×8t | 1 1 1 16 | 1 | 1.1442 | 16 | 3.54128e-6 | PASS |
| hbm8c | cn19699 | 8r×16t | 1 1 1 8 | 1 | 2.0595 | 16 | 3.54132e-6 | PASS |
| hbm16d | cn19698 | 16r×8t | 1 1 1 16 | 1 | 1.0946 | 16 | 3.54128e-6 | PASS |
| hbm4 | cn19698 | 4r×8t | 1 1 1 4 | 1 | 4.7462 | 16 | 3.54113e-6 | PASS |
| y76r32t12 | cn19699 | 32r×12t | 1 1 1 32 | 2 | **0.5193** | 16 | 3.54122e-6 | PASS（y76 原配方） |

同节点（`cn19699`，同配方组）排序：32r×12t `0.8410 s` ＜ 16r×8t `1.1442 s`
＜ 32r×18t `1.5979 s` ＜ 8r×16t `2.0595 s`。32 秩时 12 线程明显优于
18 线程。

### 6.0 y76 原配方复现（标志开销分离）

`y76r32t12` 在 `cn19699` 上使用 y76 原配方（无 `KUPL_PROFILE_*`、无
`KUPL_FUSED_BICG_DOTS`、无 `stdbuf`/`--oversubscribe`）：

- `wall_seconds=0.5193 s`，16 迭代，真残差 `3.5412238955e-6`，`PASS`。
- 与 x_case_dbg 配方 `0.8410 s` 相比为 **1.62×**，说明 profiling/fused/
  oversubscribe 标志与包装开销显著，不能把 `0.8410 s` 当作纯节点性能。
- 与历史 y76（`cn10579`）`0.4346 s` 相比为 **1.20×**，这才是同配方跨节点
  的可比比值。
- `input_prebind_seconds=15.006 s`（y76 配方未设
  `KUPL_SKIP_INPUT_PREBIND=1`），属初始化，不计入 `wall_seconds`。

### 6.1 分项耗时

| 用例 | gauge_read (s) | source_read (s) | residual (s) | max_rss (KiB) |
|---|---:|---:|---:|---:|
| r32t12b | 14.684 | 1.920 | 0.2475 | 417448 |
| r32t18b | 18.080 | 5.434 | 0.2425 | 394084 |
| r16t8b | 11.470 | 2.930 | 0.4442 | 594488 |
| hbm8c | 4.516 | 0.759 | 0.8777 | 980792 |
| hbm16d | 6.793 | 1.041 | 0.4552 | 595144 |
| hbm4 | 2.680 | 0.351 | 1.5970 | 1626492 |

### 6.2 DSLASH profile

`KUPL_PROFILE_DSLASH=1`，70 calls：

| 用例 | halo | interior | wait | boundary | total |
|---|---:|---:|---:|---:|---:|
| r32t12b | 0.0675 | 0.0095 | 0.2111 | 0.3132 | 0.6013 |
| hbm16d | 0.0306 | 0.000265 | 0.1627 | 0.6164 | 0.8100 |

### 6.3 优化补丁 A/B（新库，16 秩）

2026-11-12 晚应用并重建的优化补丁（详见
[`kupl_opt_handover_20261112.md`](kupl_opt_handover_20261112.md)）在 `cn19650`
（共享节点、preferred 内存策略）上测得：

| 用例 | 门控 | wall (s) | 迭代 | 真残差 | 状态 |
|---|---|---|---:|---:|---|
| opt16p_base | 默认 | 0.6814 | 12 | 5.385e-5 | PASS |
| opt16p_halo | SKIP_FINAL_RESIDUAL + PACK6 + DEDUP + T_LOCAL | 0.5483 | 12 | 5.385e-5 | PASS |
| opt16p_all | 上述 + BOUNDARY_LANES | **0.3900** | 12 | 5.3848e-5 | PASS |

`BOUNDARY_LANES` 单独为 1.406×；全门控相对 base 1.747×；单样本，需复测中位数。
旧库同规模 `r16t8b` 为 1.1442 s（不同节点/策略，仅作量级参考）。

## 7. 关键经验

1. **95 s 的逐用例超时过短**：16 秩端到端约 108–117 s（Python 导入和初始化
   约 100 s），32 秩约 233 s。`RC=124` 不能直接判定为“卡死”；需 400–600 s
   超时后再判。
2. **调试打印不得替换初始化行**：远端 `pyqcu/testing/simple/kupl_multigrid.py:847`
   曾被改成 `print("BEFORE_INIT")`，删除了 `verify_status = None`，导致
   `levels=1`（`use_coarse_assets=False`）必然 `UnboundLocalError`。已恢复；
   远端 md5 `c7756171a3619cb93456639dab9d3239` 与本地一致；备份
   `kupl_multigrid.py.2026-11-12-20-10-00.bak`（46552 B，含调试行）。
3. **节点差异与标志开销已分离**：x_case_dbg 配方在 `cn19699` 为 `0.8410 s`；
   y76 原配方在 `cn19699` 复现为 `0.5193 s`（1.62× 来自标志/包装）；同配方
   对照历史 y76（`cn10579`）`0.4346 s`，节点差异约 1.20×。跨节点或跨配方
   比较必须同时注明节点与标志。
4. **HBM 绑定有效**：所有 rank 的 `memory_node` 落在 `16–31`，
   `memory_bound=true`，`bound_threads` 等于设定线程数；HBM 节点上
   16 秩×8 线程比 8 秩×16 线程更快。
5. **64 task 作业被账号上限阻塞**；32 秩作业需在命令行显式
   `-N 32 -rpn 32`。
6. **作业节点偏好可被忽略**：`-nl` 指定的节点可能无资源而被挂起
   （`No node is available to meet node preferences`），需要确认或在空闲
   节点重提。

## 8. 参考基线与 y76 原配方

历史基线（B 级，非本次实测）：`cn10579`，32r×12t，grid `1 1 1 32`，
`levels=1,restart=0`，`wall_seconds=0.4346 s`，16 迭代，真残差
`3.5412238955e-6`；来源 `docs/qcu_l24_solver_timing_20261111.md`。

y76 实际启动片段：

```bash
mpirun -np 32 --bind-to none --map-by ppr:2:numa --allow-run-as-root ...
```

该配方**不带** `--oversubscribe`、`stdbuf` 或 profiling 变量，使用 `-T 600`、
`--seed 42 --output solution.h5`。完整程序参数未在证据文件中给出，待补。

同配方复现已完成：`cn19699` 上 `y76r32t12` 为 `0.5193 s`，与历史 `cn10579`
的 `0.4346 s` 相差约 **1.20×**，这才是可比的节点差异；而 x_case_dbg 配方的
`0.8410 s` 还包含 profiling/fused/oversubscribe 与包装开销（相对 y76 配方
1.62×）。跨节点比较必须同时注明节点和标志差异。

## 9. 待补项

- grid 变体（如 `1 1 2 16`、`1 2 2 8`）在 `cn19699` 的对比。
- 32r×12t 的多样本统计（当前每用例 1 个样本）。
- `KUPL_ENABLE_T2_ZBLOCK_DSLASH` 的语义（本次仅取值 `0`）。

## 10. 证据来源对照

| 内容 | 证据位置 |
|---|---|
| 远端环境、调度器、排除节点、64 task 上限 | `skills/pyqcu-kupl/references/kupl-hbm-scan-20261112-evidence.md:6-21` |
| NUMA/HBM 拓扑与 `dnode` 字段 | `skills/pyqcu-kupl/references/kupl-hbm-scan-20261112-evidence.md:17-21` |
| 用例参数、环境变量、MPI 命令、固定残差 | `skills/pyqcu-kupl/references/kupl-hbm-scan-20261112-evidence.md:23-36` |
| 主结果表、同节点排序、y76 复现 | `skills/pyqcu-kupl/references/kupl-hbm-scan-20261112-evidence.md:38-58` |
| 分项耗时表 | `skills/pyqcu-kupl/references/kupl-hbm-scan-20261112-evidence.md:60-69` |
| DSLASH profile 表 | `skills/pyqcu-kupl/references/kupl-hbm-scan-20261112-evidence.md:71-76` |
| 关键经验 1–6 | `skills/pyqcu-kupl/references/kupl-hbm-scan-20261112-evidence.md:78-95` |
| y76 基线与原配方 | `skills/pyqcu-kupl/references/kupl-hbm-scan-20261112-evidence.md:97-104` |
| 待补项 | `skills/pyqcu-kupl/references/kupl-hbm-scan-20261112-evidence.md:106-110` |
