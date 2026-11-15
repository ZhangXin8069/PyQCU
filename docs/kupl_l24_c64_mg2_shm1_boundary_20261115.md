# L24 c64 MG2 + shm1 的 balanced / t-finest 边界测量（2026-11-15）

> 证据等级：**B**。所有数值来自远端 `/home/share/nsls_yangyb/PyQCU-main/data/kunpeng_all_20261115*/`
> 的 `rows/*.json`、`reports/*.json`、`logs/*.err|out`、`kupl_kunpeng_l24_c64_mg_shm1_*.csv` 字段级读取，
> 没有从别的报告转抄。本次**没有修改任何代码**（驱动/库/提交脚本原样），只新增作业脚本、
> 新 CSV、新 outdir 与本文。
>
> 范围：KUPL CPU 后端，L24 `24³×64`、c64、`mg2`（levels=2）、`shm=1`、`source=ones`、
> `gauge=real`、`seed=42`、`threads=18`、`atol=1e-6`、global cache
> （`data/kupl_scaling_20261114/cache_l24_parallel.h5`）。**每格 1 个样本（n=1）**，
> 唯一例外是被复测的 32×18 balanced r5（n=3）。

## 1. 结论

1. **8 格矩阵 7 PASS / 1 FAIL**。唯一 FAIL 是 `32×18 balanced (2,2,2,4) r5 shm1`
   （`b3267714a641ea60`，cn19699，rc=2，`KUPL shared halo handshake timeout: publish`，
   `elapsed=271.437 s`，无 `report.json`）。
2. **该 FAIL 不是稳定失败**：同一 config（同一 `run_id`）在两个独立 outdir 各复测一次，
   **2/2 PASS**（cn12104，`elapsed` 203.832 s / 193.509 s，`verify=0`，
   真残差 `7.923186435088777e-07`，与 shm0 的 `b2715635518ec65b` 完全一致）。
   当前统计 **1 FAIL / 2 PASS**。
3. **因此本次不写 `data/kunpeng_all_20261115/known_skips.json`**（条件"稳定失败"不满足）。
   候选条目写在 `data/kunpeng_all_20261115/known_skips.candidate.json`，**默认不启用**，
   并明确禁止"静默回退 shm0"。
4. `balanced + MG2 + shm1` **不能整体判死**：16×18 balanced r5/r20 **2/2 PASS**，
   32×18 balanced r20 **PASS**。把 `balanced` 或 `shm1` 任一维整体 ban 都会错误丢掉
   已通过的覆盖。
5. 16×18 与 32×18 的 t-finest r5/r20 **4/4 PASS**（其中 32×18 t-finest r20 是本次新测）。
6. 物理一致性：同 restart 下 shm1 与 shm0 的真残差**逐位相同**
   （32×18 r5 `7.923186435088777e-07`、32×18 r20 `6.6286024494418e-07`、
   16×18 r5 `7.924649867572881e-07`）。本次没有"shm 通了但算错"的行。

## 2. 测量口径

| 项 | 取值 |
|---|---|
| 驱动 | `pyqcu/testing/kupl/test_kunpeng_all.py --mode solve`（`--expand --resume --skip-failed --hard-timeout 600`） |
| 驱动版本（本次关键约束） | FAIL 行由 **≤14:06 的版本**产生（`rows/b3267714a641ea60.json` 落盘 14:06:42；md5 未记录，14:20 版为 64,642 B）；复测行由 **16:30:50 版**（85,625 B，`md5=7a000ca97b758d45b5e0edfd3494ab8b`）产生 |
| 启动模型 | 单 replica（`#DSUB -N 1 -rpn 1 -R "cpu=<ranks*18>;mem=..."`，`-x job`），`mpirun` 由 Python 驱动内部拉起；与产生 FAIL 行的 `kall_l24_c64_full.sh` 完全同型 |
| 作业墙钟 | 16 秩 `-T 1800`、32 秩 `-T 2400`；**单例硬超时 600 s**（驱动默认，未超） |
| 节点 | 由调度器分配，未 pin `-nl`：cn12067 / cn19650 / cn19699 / cn12109 / cn12113 / cn12104 |
| cache | 全部 `cache_kind=global`；`balanced` 行另带 `ranked cache unavailable; using global cache`（h5py 无 MPI，ranked builder 不可用） |
| 数据落盘 | `data/kunpeng_all_20261115/`（主台账）+ `data/kunpeng_all_20261115_boundary{,2}/`（复测隔离 outdir，避免覆盖原 FAIL 行） |

作业 ID：`2133279`（16r balanced）、`2133280`（16r t-finest）、`2133281`（32r balanced）、
`2133282`（32r t-finest）、`2133332`/`2133458`（32r balanced r5 复测 1/2）。
所有作业都是本账号本次提交，**没有 kill 任何作业**。

> **口径陷阱（实测）**：`--resume` 判重读的是 `outdir/rows/<run_id>.json`，**不是** `--csv`。
> 因此"换一个新 CSV 重跑同一 config"不会重跑——作业 `2133280` 正是因此返回
> `{"done": 2}` 却一条新行都没写（16×18 t-finest r5/r20 早已 PASS）。
> 要重测必须用新 `outdir`（本次做法）或先让旧行失效。

## 3. 行表（8 格，n=1）

| # | 单元 | grid | restart | shm | run_id | node | status | verify | wall (s) | elapsed (s) | final_true_residual | cache |
|---|---|---|---:|---:|---|---|---|---:|---:|---|---|
| 1 | 16×18 balanced | `2 2 2 2` | 5 | 1 | `e9324698db606356` | cn12067 | PASS | 0 | 0.681115 | 89.003 | `7.924649867572881e-07` | global |
| 2 | 16×18 balanced | `2 2 2 2` | 20 | 1 | `abf08639db210120` | cn12067 | PASS | 0 | 0.367069 | 86.974 | `6.628968909675872e-07` | global |
| 3 | 16×18 t-finest | `1 1 1 16` | 5 | 1 | `cf5659638b00b61e` | cn19650 | PASS | 0 | 0.566035 | 102.279 | `7.921932287461337e-07` | global |
| 4 | 16×18 t-finest | `1 1 1 16` | 20 | 1 | `9c122f681fa3ffe1` | cn19650 | PASS | 0 | 0.312759 | 101.491 | `6.628977307549227e-07` | global |
| 5 | 32×18 balanced | `2 2 2 4` | 5 | 1 | `b3267714a641ea60` | cn19699 | **FAIL** | – | – | 271.437 | – | global |
| 6 | 32×18 balanced | `2 2 2 4` | 20 | 1 | `8a9404d32fe9928e` | cn12109 | PASS | 0 | 0.299501 | 187.280 | `6.6286024494418e-07` | global |
| 7 | 32×18 t-finest | `1 1 1 32` | 5 | 1 | `c98060824b55ebf7` | cn19699 | PASS | 0 | 0.394854 | 207.836 | `7.936243078683469e-07` | global |
| 8 | 32×18 t-finest | `1 1 1 32` | 20 | 1 | `abbce785703da235` | cn12113 | PASS | 0 | 0.206687 | 182.693 | `6.632567997965678e-07` | global |

复测（config 与 #5 完全相同，独立 outdir，`--resume` 不参与）：

| # | outdir | run_id | node | status | verify | wall (s) | elapsed (s) | final_true_residual |
|---|---|---|---|---|---:|---:|---|---|
| R1 | `kunpeng_all_20261115_boundary` | `b3267714a641ea60` | cn12104 | PASS | 0 | 0.500034 | 203.832 | `7.923186435088777e-07` |
| R2 | `kunpeng_all_20261115_boundary2` | `b3267714a641ea60` | cn12104 | PASS | 0 | 0.530118 | 193.510 | `7.923186435088777e-07` |

读法：

- `wall_seconds` 只包住一次底层求解调用；`elapsed_seconds` 是整行墙钟（含 MPI 启动、
  HDF5 读、独立真残差复算）。8 行 `elapsed` 89–271 s，与 `wall` 差两个数量级，
  不能把 `wall` 当"端到端"。
- MG 的 `iterations` **不存在这个键**（不是 null），与
  [`kupl_kunpeng_matrix_20261115.md`](kupl_kunpeng_matrix_20261115.md) §4.2-3 一致。
- 同一 restart 下 `final_true_residual` 只由物理/容差决定，与 grid、shm、rank 数无关；
  本次所有 PASS 行都满足 `atol=1e-6`，且 shm1 与 shm0 逐位一致。

## 4. 通过 / 失败矩阵

| 单元 | restart | shm | 结果 | 样本 |
|---|---:|---:|---|---:|
| 16×18 balanced `2 2 2 2` | 5 | 1 | **PASS** | 1 |
| 16×18 balanced `2 2 2 2` | 20 | 1 | **PASS** | 1 |
| 16×18 t-finest `1 1 1 16` | 5 | 1 | **PASS** | 1 |
| 16×18 t-finest `1 1 1 16` | 20 | 1 | **PASS** | 1 |
| 32×18 balanced `2 2 2 4` | 5 | 1 | **FAIL 1 / PASS 2 → 不稳定** | 3 |
| 32×18 balanced `2 2 2 4` | 20 | 1 | **PASS** | 1 |
| 32×18 t-finest `1 1 1 32` | 5 | 1 | **PASS** | 1 |
| 32×18 t-finest `1 1 1 32` | 20 | 1 | **PASS** | 1 |

同批里 `shm=0` 的对照（既有台账）全部 PASS：
32×18 balanced r5 `b2715635518ec65b`（211.357 s，真残差 `7.923186435088777e-07`）、
32×18 balanced r20 `cb136a9593aa2021`（200.345 s）、
32×18 t-finest r20 `4d187eadc00a7feb`（201.162 s）、
16×18 balanced r5 `716b1f6a3fd49353`（100.820 s）、
16×18 t-finest r5 `7668f066600ef451`（103.351 s）、16×18 t-finest r20 `06ad0f1f0accf025`（106.743 s）。
所以 **shm1 只在 32×18 balanced r5 这一格出现过 1 次异常，其余 13 条 shm1/shm0 行全 PASS**。

## 5. 失败模式：`KUPL shared halo handshake timeout: publish`

### 5.1 现象（`b3267714a641ea60`）

| 项 | 值 |
|---|---|
| 单元 | L24 c64 `mg2`，`32×18`，`grid=[2,2,2,4]`（balanced），`restart=5`，`shm=1` |
| 启动脚本 | `data/kunpeng_all_20261115/jobs/kall_l24_c64_full.sh`（单 replica，`cpu=576`） |
| 节点 | cn19699 |
| `returncode` | 2（`MPI_ABORT ... errorcode 2`，`rank 30`） |
| `timeout` / `review` | `False` / `False`（**不是** 600 s 硬超时，也不是 300 s 审查阈值） |
| `elapsed_seconds` | 271.437（≈ 240 s 正常阶段 + 30 s spin 超时 + abort） |
| `wall_seconds` / `final_true_residual` / `verify_status` | 均缺失（无 `report.json`） |
| `logs/<id>.out` | 0 B |
| `logs/<id>.err` | 9,317 B：`KUPL shared halo handshake timeout: publish` ×10，`consume` 超时 ×0，随后 `MPI_ABORT was invoked on rank 30` |
| 台账 `error` | `null`（签名只在 `.err` 里） |
| 台账 `notes` | `['ranked cache unavailable; using global cache', 'no report.json']`——**没有** `shared halo handshake timeout (MG+shm asymmetric)` 这条 note |

> 可观测性缺口：驱动在 `solve_row` 里有把 `shared halo handshake timeout` 写进 `notes`
> 的分支，但该行没有触发（`.err` 里确实有签名）。所以**只查台账会把这条失败读成
> "no report.json"**，必须回到 `logs/*.err` 才看得见根因签名。

### 5.2 机制（读代码，不改代码）

`cpp/kupl/qcu/src/lattice_wilson_dslash.cpp`：

```cpp
// 20-33 行
inline void shm_halo_spin(const uint64_t *address, uint64_t target,
                          bool at_least, const char *label) {
  const auto deadline = steady_clock::now() + seconds(30);   // 硬编码 30 s
  ...
      throw std::runtime_error("KUPL shared halo handshake timeout: " + label);
}
```

- 调用点 1：`1172` 行 `shm_halo_spin(published, seq, false, "publish")`，在
  `wait_fermion_halo()` 里等对端把 face 的 `published` 计数推到本轮 `seq`。
- 调用点 2：`1233` 行 `shm_halo_spin(shm_consumed(...), need, true, "consume")`，
  在 `post_fermion_halo()` 里等对端消费掉两轮前（`seq-2`）的 slot，即双缓冲复用保护。
- 代码注释（17-19 行）写明语义："A mismatch means the neighbour is not following the
  same dslash call sequence, so fail loudly instead of hanging the whole job."
  → 超时 = **邻 rank 的 dslash 调用序列与本 rank 不一致**，不是网络慢。
- 本次只命中 `publish`（10 次），没有 `consume` → 是"等对端发布"的方向，不是"slot 复用"。

驱动侧的相关环境（`base_env`）：`KUPL_SHM_HALO=1`、`KUPL_HALO_PACK6=1`、
`KUPL_HALO_DEDUP_PACK=1`、`KUPL_T_LOCAL_IN_PART1=1`、`KUPL_ENABLE_BOUNDARY_LANES=1`、
`KUPL_ENABLE_T2_LANES=1`。这些是按局部几何选内核的开关，和"邻 rank 走不同分支"的
假设相容；**本次没有做 A/B，不能把根因归到其中任何一个**。

### 5.3 触发面（本次实测）

| 几何 | 变动的维数（node-local peer 数） | 结果 |
|---|---|---|
| 16×18 balanced `2 2 2 2` | 4 维（x,y,z,t） | PASS ×2 |
| 16×18 t-finest `1 1 1 16` | 1 维（t） | PASS ×2 |
| 32×18 balanced `2 2 2 4` | 4 维（x,y,z,t） | **FAIL ×1 / PASS ×2** |
| 32×18 t-finest `1 1 1 32` | 1 维（t） | PASS ×2 |

"4 维 node-local" 不足以解释（16×18 balanced 也是 4 维且 PASS）；更可能的是
**rank 数 / 每 NUMA rank 数（32 秩 `ppr:2:numa` = 4 rank/NUMA，16 秩 = 2 rank/NUMA）
放大了竞争窗口**，属于竞态而非确定性缺陷。**这是假设，不是结论**。

### 5.4 复测：不稳定

| 轮次 | 驱动 | 节点 | 结果 |
|---|---|---|---|
| 原始 | ≤14:06 版（md5 未记录；14:20 版 64,642 B） | cn19699 | FAIL（handshake publish） |
| R1 | 16:30:50 版（`md5 7a000ca9…`） | cn12104 | PASS |
| R2 | 同上 | cn12104 | PASS |

**1 FAIL / 2 PASS**。两种解释都不能排除：

1. **竞态**：同一二进制、同一 config，失败概率非零（30 s 窗口内没等到 publish）。
2. **驱动版本差异**：FAIL 行由 16:30:50 之前的驱动产生，复测用新驱动。新驱动
   85,625 B（旧 64,642 B，+33%），`base_env`/命令行可能变了；行 JSON **不记录驱动
   md5**，所以无法从台账上区分这两个解释。

若要彻底区分，需要把 ≤14:06 版驱动留存并重跑同一 config（该副本已不存在，本次未做）。

## 6. skip 建议

### 6.1 不写 `known_skips.json`

任务条件是"balanced+MG2+shm1 **稳定**失败"。实测相反：

- `32×18 balanced r5 shm1`：1 FAIL / 2 PASS（复测全通，真残差正确）；
- `32×18 balanced r20 shm1`：PASS；
- `16×18 balanced r5/r20 shm1`：PASS ×2；
- `32×18 / 16×18 t-finest r5/r20 shm1`：PASS ×4。

写 `known_skips.json` 会让下游**静默跳过**一个 2/3 会通过的格，损失真实覆盖，
因此本次**不创建**该文件（也不覆盖任何已有文件；远端原本就不存在）。

### 6.2 候选条目（默认关闭）

已写 `data/kunpeng_all_20261115/known_skips.candidate.json`（注意文件名不是
`known_skips.json`，不会被当作权威 skip 表）。把它提升为 `known_skips.json` 的
**前置条件**：同一 config 再观察到 ≥2 次 FAIL（建议跑 5–10 次，每次 ~200 s）。

条目内容（摘要；完整字段、`not_tested`、`rejected_skips` 见文件）：

```json
{
  "id": "mg2-l24-c64-balanced-32x18-shm1-r5",
  "enabled": false,
  "confidence": "flaky: 1 FAIL / 2 PASS",
  "config": {
    "solver": "mg2", "lattice": [24, 24, 24, 64], "dtype": "c64", "atol": 1e-6,
    "ranks": 32, "threads": 18, "rank_x_thread": "32x18",
    "grid_mode": "balanced", "grid": [2, 2, 2, 4],
    "mg_restart": 5, "shm": 1, "source_kind": "ones", "gauge_kind": "real", "seed": 42
  },
  "failure": {
    "signature": "KUPL shared halo handshake timeout: publish",
    "returncode": 2, "mpi_abort_rank": 30, "report": null,
    "elapsed_seconds": 271.4371476210654,
    "run_id": "b3267714a641ea60",
    "err_path": "data/kunpeng_all_20261115/logs/b3267714a641ea60.err"
  },
  "reason": "node-local shared-halo publish 握手 30 s 超时后 MPI_ABORT；复测 2/2 PASS，未稳定复现",
  "minimal_repro": "见 §7",
  "failing_combinations": [
    "32x18 balanced [2,2,2,4] mg2 shm1 r5 (cn19699, driver <=14:06, run b3267714a641ea60)"
  ],
  "passing_combinations": [
    "32x18 balanced [2,2,2,4] mg2 shm1 r5 (cn12104, 16:30:50 driver, boundary/boundary2)",
    "32x18 balanced [2,2,2,4] mg2 shm1 r20 (cn12109, run 8a9404d32fe9928e)",
    "16x18 balanced [2,2,2,2] mg2 shm1 r5/r20 (cn12067, runs e9324698db606356/abf08639db210120)",
    "16x18 t-finest [1,1,1,16] mg2 shm1 r5/r20 (cn19650, runs cf5659638b00b61e/9c122f681fa3ffe1)",
    "32x18 t-finest [1,1,1,32] mg2 shm1 r5/r20 (cn19699/cn12113, runs c98060824b55ebf7/abbce785703da235)"
  ],
  "fallback_policy": "no silent shm0 fallback: 调用方必须显式选择 shm=0 并在台账标注",
  "recommended_action": "先用 shm=0 作为对照（32x18 balanced r5 shm0 已 PASS），不要全局禁用 balanced+shm1"
}
```

### 6.3 如果一定要设边界

建议按**最窄格**设，而不是按维度设：

| 建议 | 内容 | 理由 |
|---|---|---|
| ✅ 可接受 | 只在 `ranks=32 + grid_mode=balanced + mg2 + shm=1` 上加"跑 3 次取多数 / 失败自动重跑 1 次" | 1/3 失败率，重跑可覆盖 |
| ⚠️ 谨慎 | 只 skip `32×18 balanced r5 shm1`（保留 r20） | r20 1/1 PASS，skip 它会丢覆盖 |
| ❌ 不要 | skip `balanced + mg2 + shm1` 整体 | 16×18 balanced 2/2 PASS |
| ❌ 不要 | 静默回退 `shm=0` | 会掩盖问题，且改变被测配置 |

## 7. 最小复现

```bash
# 远端 /home/share/nsls_yangyb/PyQCU-main
python3 -B pyqcu/testing/kupl/test_kunpeng_all.py --mode solve \
  --outdir data/kunpeng_all_20261115_boundary \
  --ranks 32 --threads 18 --shard-index 0 --shard-count 1 --expand \
  --lattices L24 --dtypes c64 --solvers mg2 \
  --grid-modes balanced --ranks-list 32 --threads-list 18 \
  --shm-list 1 --sources ones --gauge-kinds real --seeds 42 \
  --mg-restarts 5 --limit 1 --hard-timeout 600 \
  --csv data/kunpeng_all_20261115_boundary/kupl_kunpeng_l24_c64_mg_shm1_b32_bal_r5_repro.csv
```

作业化（与产生 FAIL 行同型的单 replica 启动器）：

```bash
#DSUB -q q_hpcapp
#DSUB -x job
#DSUB -nn 1
#DSUB -N 1
#DSUB -rpn 1
#DSUB -R "cpu=576;mem=500G"
#DSUB -T 2400
dsub -s data/kunpeng_all_20261115/jobs/kall_mg_shm1_b32_bal_r5_repro.sh
```

注意：
- **必须用新的 `--outdir`**，否则 `--resume` 会按 `rows/<run_id>.json` 直接跳过
  （`--skip-failed` 还会跳过已 FAIL 的行）。
- `--hard-timeout 600` 保证单例 ≤ 600 s；本次最坏 271 s。
- 复现的 2 次都 PASS，说明**用这条命令单跑一次不能判定"会不会失败"**。

## 8. 证据路径

| 内容 | 路径 |
|---|---|
| 主台账（8 格行） | `data/kunpeng_all_20261115/rows/{e9324698db606356,abf08639db210120,cf5659638b00b61e,9c122f681fa3ffe1,b3267714a641ea60,8a9404d32fe9928e,c98060824b55ebf7,abbce785703da235}.json` |
| FAIL 日志 | `data/kunpeng_all_20261115/logs/b3267714a641ea60.err`（9,317 B）/ `.out`（0 B） |
| 复测 outdir | `data/kunpeng_all_20261115_boundary/`（R1）、`data/kunpeng_all_20261115_boundary2/`（R2） |
| 本次 CSV | `kupl_kunpeng_l24_c64_mg_shm1_b16_bal.csv`、`..._b32_bal.csv`、`..._b32_tf.csv`、`_boundary*/...repro.csv` |
| 作业脚本 | `data/kunpeng_all_20261115/jobs/kall_mg_shm1_{b16_bal,b16_tf,b32_bal,b32_tf,b32_bal_r5_repro,b32_bal_r5_repro2}.sh` |
| 候选 skip | `data/kunpeng_all_20261115/known_skips.candidate.json` |
| 相关旧文档 | [`kupl_kunpeng_matrix_20261115.md`](kupl_kunpeng_matrix_20261115.md) §6.5（MG2 表）、§7 第 3 条（同一 run_id 的首次记录） |

驱动版本对照：

| 时间 | 大小 | md5 | 覆盖的行 |
|---|---:|---|---|
| 2026-11-15 14:20 | 64,642 B | 未记录 | 16×18 balanced r5/r20（16:26 启动）、32×18 balanced r20（16:29:47 启动） |
| 2026-11-15 ≤14:06 | 未记录 | 未记录 | FAIL 行 `b3267714a641ea60`（行 JSON 落盘 14:06:42，源 `kall_l24_c64_full.sh`） |
| 2026-11-15 16:30:50 | 85,625 B | `7a000ca97b758d45b5e0edfd3494ab8b` | 32×18 t-finest r20（16:32:20 启动）、R1、R2 |

> 驱动在本轮测量期间被同一台机器上的其它工作改写（14:20 → 16:30:50），
> 这是本次证据最大的不确定源。**行 JSON 不记录驱动 md5 / libqcu 版本**。

## 9. 局限与下一步

1. **n 太小**：8 格各 1 样本，唯一有 3 样本的格是 32×18 balanced r5。要判定边界，
   至少对 32×18 balanced r5/r20 各跑 5–10 次，记录 min/median/失败率。
2. **驱动版本夹杂**：见 §8。建议在 `rows/*.json` 里加 `driver_md5`、`libqcu_md5`、
   `git_describe`（本次未改代码，只提建议）。
3. **签名没进台账**：`b3267714a641ea60` 的 `notes` 缺 `shared halo handshake timeout`
   标签，只查 CSV 会误读成 "no report.json"。建议把 `.err` 的已知签名映射进
   `error`/`notes`。
4. **节点未固定**：16 秩落在 cn12067/cn19650，32 秩落在 cn19699/cn12109/cn12113/cn12104。
   已知同配置跨节点有 1.4× 级差异，所以 `wall` 只能做同格比较，不能做扩展性曲线。
5. **根因未定位，也没改代码**：若要继续，最小侵入的做法是在
   `shm_halo_spin` 超时分支里 dump `rank/face/seq/peer/` 计数器值（需要改库），
   或按 §6.3 用"重试 1 次"的调度策略绕开偶发失败。
