# KUPL L24 HBM 扫描证据（2026-11-12 会话）

本文件是 2026-11-12 20:00 之后恢复/补测得到的**已验证**证据摘要，供文档与技能编写使用。
只能引用此处出现的数值；未在此列出的数据一律标记为“待补”，不得推算或虚构。

## 环境

- 远端：LX2 集群，登录节点 `login06`（`ssh 10.44.9.4`，用户 `nsls_yangyb`）。
- 远端工作树：`/home/share/nsls_yangyb/PyQCU` → `PyQCU-main`；用例目录
  `data/kupl_hbm_opt_20261106/scan8_20261112/`；脚本目录 `~/sh`。
- 调度器：`dsub` / `djob` / `djob -ll` / `dkill`，队列 `q_hpcapp`，独占用 `-x job`。
  `#DSUB` 头中的 `-N/-rpn/-R/-nl` 可能被忽略，调度参数必须写在 `dsub` 命令行；
  提交后用 `djob -ll <id>` 核对 `execNodes` 是否等于目标节点。
- 排除节点：`cn12038`、`cn10579`（`data/kupl_hbm_opt_20261106/excluded_nodes.txt`）。
- 账号限制：64 task 作业被 `testusers_rebuild` 上限拒绝（`mainStateReasonMsg:
  The limit <testusers_rebuild> has been exceeded`）。
- 节点 `cn19701` sysfs 探测：32 个 NUMA 节点；`node0–15` 约 32 GB/节点，
  `node16–31` 为 4 GB/节点（片上 HBM）；KUPL 拓扑报告 `numa_nodes=16`，
  `memory_node=16..31`，每 rank `cpuset` 为 8–12 个 CPU，`memory_policy=bind`，
  `bound_threads=threads`，`source=sysfs`。
- `dnode` 显示 `cn196xx` 节点 CPU=608、内存约 578–592 GB。

## 复现配置（与本次所有用例一致，除注明外）

- 用例：`--lat 24 24 24 64 --mass -0.277 --csw 1.160920226 --atol 1e-4
  --max-iter 50 --dtype c64 --levels 1 --restart 0`；`levels=1,restart=0`
  为 parity-Schur BiCGStab，不是 multigrid；16 次迭代收敛。
- 环境变量：`OMP_NUM_THREADS=KUPL_OMP_THREADS=TH`、`KUPL_NUMA_BIND=1`、
  `KUPL_MEMORY_POLICY=bind`、`KUPL_MEMORY_NODE=local+16`、
  `KUPL_REQUIRE_MEMORY_BIND=1`、`KUPL_HUGEPAGE=1`、`KUPL_SKIP_INPUT_PREBIND=1`、
  `KUPL_TOPOLOGY_REPORT=1`、`KUPL_PROFILE_INIT=1`、`KUPL_FUSED_BICG_DOTS=1`、
  `KUPL_PROFILE_DSLASH=1`、`KUPL_ALLOW_OVERSUBSCRIBE=1`、
  `KUPL_BLOCK_DSLASH_MIN_HALF_SITES=16384`、`KUPL_ENABLE_T2_ZBLOCK_DSLASH=0`。
- MPI：`mpirun --oversubscribe -np NP --bind-to none --map-by ppr:PPR:numa
  --allow-run-as-root`。
- 固定对照：真残差 `≈3.5412e-6`，验收阈值 `2.44140625e-4`，`PASS`。

## 实测结果（节点独占 `-x job`，solve 墙钟 `wall_seconds`）

| 用例 | 节点 | 规模 | grid | ppr | wall (s) | 迭代 | 真残差 | 状态 |
|---|---|---|---:|---:|---:|---:|---|---|
| r32t12b | cn19699 | 32r×12t | 1 1 1 32 | 2 | **0.8410** | 16 | 3.54117e-6 | PASS |
| r32t18b | cn19699 | 32r×18t | 1 1 1 32 | 2 | 1.5979 | 16 | 3.54122e-6 | PASS |
| r16t8b | cn19699 | 16r×8t | 1 1 1 16 | 1 | 1.1442 | 16 | 3.54128e-6 | PASS |
| hbm8c | cn19699 | 8r×16t | 1 1 1 8 | 1 | 2.0595 | 16 | 3.54132e-6 | PASS |
| hbm16d | cn19698 | 16r×8t | 1 1 1 16 | 1 | 1.0946 | 16 | 3.54128e-6 | PASS |
| hbm4 | cn19698 | 4r×8t | 1 1 1 4 | 1 | 4.7462 | 16 | 3.54113e-6 | PASS |
| y76r32t12 | cn19699 | 32r×12t | 1 1 1 32 | 2 | **0.5193** | 16 | 3.54122e-6 | PASS（y76 原配方） |

同节点（cn19699）排序（同配方组）：32r×12t 0.8410 s ＜ 16r×8t 1.1442 s
＜ 32r×18t 1.5979 s ＜ 8r×16t 2.0595 s。32 秩时 12 线程明显优于 18 线程。

**y76 原配方复现（20:53:03 完成）**：`y76r32t12` 在 cn19699 上
`wall_seconds=0.5193 s`，16 it，真残差 `3.5412238955e-6`，PASS；与 x_case_dbg
配方（0.8410 s）相差 1.62×，说明 profiling/fused/oversubscribe 标志开销显著。
同配方跨节点：cn19699 0.5193 s vs 历史 cn10579 y76 0.4346 s，仅约 1.20×。
`input_prebind_seconds=15.006 s`（y76 配方未设 `KUPL_SKIP_INPUT_PREBIND=1`），
属初始化，不计入 `wall_seconds`。

### 分项耗时（同两次运行）

| 用例 | gauge_read (s) | source_read (s) | residual (s) | max_rss (KiB) |
|---|---:|---:|---:|---:|
| r32t12b | 14.684 | 1.920 | 0.2475 | 417448 |
| r32t18b | 18.080 | 5.434 | 0.2425 | 394084 |
| r16t8b | 11.470 | 2.930 | 0.4442 | 594488 |
| hbm8c | 4.516 | 0.759 | 0.8777 | 980792 |
| hbm16d | 6.793 | 1.041 | 0.4552 | 595144 |
| hbm4 | 2.680 | 0.351 | 1.5970 | 1626492 |

### DSLASH profile（`KUPL_PROFILE_DSLASH=1`，70 calls）

| 用例 | halo | interior | wait | boundary | total |
|---|---:|---:|---:|---:|---:|
| r32t12b | 0.0675 | 0.0095 | 0.2111 | 0.3132 | 0.6013 |
| hbm16d | 0.0306 | 0.000265 | 0.1627 | 0.6164 | 0.8100 |

## 关键经验

1. **95 s 的逐用例超时过短**：16 秩端到端约 108–117 s（Python 导入 + 初始化约 100 s），
   32 秩约 233 s。RC=124 不能直接判定为“卡死”；需 400–600 s 超时后再判。
2. **调试打印不得替换初始化行**：远端 `pyqcu/testing/simple/kupl_multigrid.py:847`
   曾被改成 `print("BEFORE_INIT")`，删除了 `verify_status = None`，导致
   `levels=1`（`use_coarse_assets=False`）必然 `UnboundLocalError`。已恢复；
   远端 md5 `c7756171a3619cb93456639dab9d3239` 与本地一致；备份
   `kupl_multigrid.py.2026-11-12-20-10-00.bak`（46552 B，含调试行）。
3. **节点差异与标志开销需分离**：x_case_dbg 配方（含 profiling/fused）在 cn19699
   为 0.8410 s；y76 原配方在 cn19699 复现为 0.5193 s（1.62× 差异来自标志/包装）；
   同配方对照历史 y76（cn10579）0.4346 s，节点差异仅约 1.20×。跨节点或跨配方
   比较必须同时注明节点与标志。
4. **HBM 绑定有效**：所有 rank 的 `memory_node` 落在 16–31，`memory_bound=true`，
   `bound_threads` 等于设定线程数；HBM 节点上 16 秩×8 线程相比 8 秩×16 线程更快。
5. **64 task 作业被账号上限阻塞**；32 秩作业需在命令行显式 `-N 32 -rpn 32`。
6. **作业节点偏好可被忽略**：`-nl` 指定的节点可能无资源而被挂起
   （`No node is available to meet node preferences`），需要确认或在空闲节点重提。

## 参考基线（历史，非本次实测）

- K1 / y76：`cn10579`，32r×12t，grid `1 1 1 32`，`levels=1,restart=0`，
  `wall_seconds=0.4346 s`，16 迭代，真残差 `3.5412238955e-6`。
  来源：`docs/qcu_l24_solver_timing_20261111.md`（证据等级 B）。
- y76 实际启动命令（用于对照）：`mpirun -np 32 --bind-to none
  --map-by ppr:2:numa --allow-run-as-root`，**不带** `--oversubscribe`、
  `stdbuf` 或 profiling 变量，`-T 600`，`--seed 42 --output solution.h5`。

## 待补（未验证，不得写成结论）

- grid 变体（如 `1 1 2 16`、`1 2 2 8`）在 cn19699 的对比。
- 32r×12t 的多样本统计（当前每用例 1 个样本）。
- `KUPL_ENABLE_T2_ZBLOCK_DSLASH` 的语义（本次仅取值 `0`）。

## 2026-11-12 优化 A/B（新库，16 秩）

环境：cn19650（共享节点，preferred 内存策略，非独占），16r×8t，grid `1 1 1 16`，
y76 原配方，每点 1 样本；新库为应用本轮全部补丁后重建的 `build-native-lto`。

| 用例 | 门控 | wall (s) | 迭代 | 真残差 | 状态 | 相对 base |
|---|---|---|---:|---:|---|---:|
| opt16p_base | 默认 | 0.6814 | 12 | 5.385e-5 | PASS | — |
| opt16p_halo | +KUPL_SKIP_FINAL_RESIDUAL +PACK6 +DEDUP_PACK +T_LOCAL_IN_PART1 | 0.5483 | 12 | 5.385e-5 | PASS | 1.243× |
| opt16p_all | 上述 +KUPL_ENABLE_BOUNDARY_LANES | **0.3900** | 12 | 5.3848e-5 | PASS | **1.747×** |

- BOUNDARY_LANES 单独贡献 `0.5483 → 0.3900`（1.406×）。
- 容差放宽（Clover `atol`）使迭代 16 → 12；真残差仍低于 `2.4414e-4`。
- 旧库同规模 `r16t8b` 为 1.1442 s（cn19699、bind、独占），跨节点/策略仅作量级参考。
- 单样本，正式结论需补 2 次取中位数。
- 非独占 + `KUPL_MEMORY_POLICY=bind` + `KUPL_MEMORY_NODE=local+16` 会出现 RC=137
  SIGKILL（约 87–110 s，OOM）；改 `preferred` 或独占节点可避免。
