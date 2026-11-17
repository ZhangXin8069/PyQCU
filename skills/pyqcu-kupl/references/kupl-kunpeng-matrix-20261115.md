# 鲲鹏 KUPL 全配置矩阵：入口、CSV 与 2026-11-15 台账口径

本文件是 [`../SKILL.md`](../SKILL.md) 的按需参考：只在运行/解读
`pyqcu/testing/kupl/test_kunpeng_all.py` 的鲲鹏矩阵、审计其 CSV 台账，或复用
矩阵级 runbook 时读取。算子、ABI、绑定与收敛判据仍以 `SKILL.md` 为准。

证据时点：2026-11-15（c64/c128、L24/L32、`atol` 1e-6/1e-9、`--strict-acceptance`）。
数据根 `data/kunpeng_all_20261115/`（`rows/`、`reports/`、`solutions/`、`logs/`、
`caches/`、`cache_index.json`、`*.csv`）在计算节点侧；
本机台账只读副本 `data/kunpeng_all_20261115_audit/`（`ledger.csv`、`report.md`、
`scripts/check_*.py`）。数值汇总文档见 `docs/kupl_performance_20261115.md`。

## 1. 入口与模式

```bash
# 单条用例（所有可调参数走命令行，便于按节点分组独立提交）
python -B pyqcu/testing/kupl/test_kunpeng_all.py --mode run \
  --lat 24 24 24 64 --mass -0.277 --csw 1.160920226 --dtype c64

# 小尺寸冒烟（2 rank、8x8x8x16、bistabcg + 1-rank cache + mg2，并自检 CSV 列）
python -B pyqcu/testing/kupl/test_kunpeng_all.py --mode validate

# 矩阵展开 / 分片 / 合并
python -B pyqcu/testing/kupl/test_kunpeng_all.py --mode list --expand --limit 5
python -B pyqcu/testing/kupl/test_kunpeng_all.py --mode solve --expand \
  --shard-index 0 --shard-count 4 --resume
python -B pyqcu/testing/kupl/test_kunpeng_all.py --mode merge

# 提交一个节点组（脚本自己生成 #DSUB 并调用 dsub -s）
pyqcu/testing/kupl/submit_kunpeng_all.sh --mode solve --tag kall_l24_a \
  --ranks 32 --threads 18 --node cn19699 --shard-index 0 --shard-count 2 \
  --extra "--expand --lattices L24 --solvers bistabcg,mg2"
```

- 模式：`list`（JSONL 展开）、`prepare`（建/复用 gauge、source、Legacy-MG cache）、
  `prewarm`（生成并预热页缓存）、`solve`（跑一个分片并追加 CSV）、`merge`（`rows/*.json`
  → CSV）、`run`（单条用例）、`validate`（小格冒烟）。
- 驱动**只包装**既有规范入口 `pyqcu/testing/simple/kupl_multigrid.py` 与
  `pyqcu/testing/simple/mpi/build_full_mg_cache.py`（源码不改）；自己启动 `mpirun`，
  因此外层必须是**单任务**（见 §3）。
- `--lat/--mass/--csw/--dtype` 是全部参数入口：同一份代码按不同参数分组提交到不同
  节点，互不共享状态，只共享 `--outdir` 下的 `rows/`（`flock` 追加 CSV）。
- 每次 solve 传 `--output solutions/<run_id>.h5` 与 `--report reports/<run_id>.json`，
  并硬编码 `--strict-acceptance`（判据 = `atol`，不叠 dtype 地板）。

## 2. 矩阵维度

`--expand` 的笛卡尔积（默认值即全矩阵）：

| 轴 | 默认 | 说明 |
|---|---|---|
| `--lattices` | `L24,L32` | `24³×64`、`32³×64` |
| `--dtypes` | `c64,c128` | `atol` 默认 c64 `1e-6`、c128 `1e-9` |
| `--solvers` | `bistabcg,mg1,mg2` | `mg1` 只有 `restart=0`，`mg_restart` 仅 `mg2` 生效 |
| `--mg-restarts` | `5,10,20,50` | 只作用于 `mg2`；**50 已放弃**（0 行，不新增），10 保留待补 |
| `--grid-modes` | `balanced,t-finest` | `t-finest` = `1 1 1 ranks`；`balanced` 取偶数局部格的均衡 4D 分解 |
| `--ranks-list` | `16,32` | 只测 `ranks*threads <= 608`（`--max-total-threads`）的组合 |
| `--threads-list` | `9,18,36` | 36 在 `ppr:2:numa` 下超出 cpuset，由库守卫失败关闭；**9 已放弃**（不扩展、不重测），36 保留 |
| `--shm-list` | `0,1` | `KUPL_SHM_HALO`；`shm=1 + mg2 restart=5` 曾出现 1 次握手超时，同 config 复测 2/2 PASS → **candidate 而非 known skip**（见 §5.1） |
| `--sources` | `ones,real1,point,random` | `ones`/`real1` 都映射全 1 墙源文件；**`point`、`random` 已放弃**（不新增不重测） |
| `--gauge-kinds` | `real,random` | `real` 用生产组态；`random` 用 NumPy QR 生成，**已放弃** |
| `--seeds` | `42,24` | 源的全局随机种子；**24 已放弃**（现有 2 行保留，不扩展） |
| MG 参数 | `--mg-grid 2 2 2 1`、`--dof 12 2 2 2 2`、`--nv-iters 2`、`--coarse-max-iter 200`、`--coarse-tol-factor 1e4`、`--no-deflate` | 逐行写入 CSV |

> **本轮范围裁定（2026-11-15）**：thread 9、seed24、random gauge、random 源、point 源、
> `mg-restart 50` 六项**放弃**——耗时代价过大，已有部分数据保留，不重测、不新增。保留
> 16×36（已有 3 行）、MG1、r10、L32/c128、MG+shm。完整裁定与判据见
> [`docs/kupl_matrix_scope_20261115.md`](../../../docs/kupl_matrix_scope_20261115.md)。

`run_id` = 配置元组（solver/dtype/atol/lattice/ranks/threads/grid/shm/source/seed/
gauge/`mg_*`）的 SHA1 前 16 位；`--run-label` 追加后缀，用于把同一配置的 cold/warm 或
复测样本分开。**同名 `run_id` 在多份 CSV 里出现即同一配置的重复测量，不是不同用例。**

## 3. 作业模型与调度

- `#DSUB -N 1 -rpn 1 -R "cpu=<ranks*threads>;mem=400G" -x job`：单任务独占整节点
  （32r×18t → `cpu=576`，16r×18t → `cpu=288`），驱动在任务内自行 `mpirun -np R`。
  这是本批 24/40 个作业脚本用的模型，`-x job` 保证整节点不被他人共享。
- `#DSUB -N R -rpn R -R "cpu=T;mem=8G"`（每 rank 一个 task）也会被生成；本批 16 个
  脚本用它，对应行的 `input_prebind_seconds` 明显偏高（`13.15 s` 对单任务行
  `2.8–3.7 s`），按受扰样本处理，性能结论只用单任务模型复测值。
- `-x job` 要求整节点空闲：节点被部分占用时作业会保持 `PENDING`，多节点作业还可能
  因放置失败长期 `PENDING`（`logs/v20261114005301.txt`）。矩阵因此**一个作业一个
  节点**、按 `--shard-index/--shard-count` 切片，而不是一个大作业跨节点。
- `#DSUB -T` 的单位是**秒**（脚本默认 900）；写 `10` 会被调度器拒绝，不要按分钟填。
  同批脚本还固定 `-nn 1`、`--mpi hmpi`，task 数由 `-N/-rpn` 决定。
- 账号上限：64 task 的作业会被 **`testusers_rebuild`** 上限拒绝（`dsub` 只回
  `job was refused by hook`，`djob -ll` 的 `mainStateReasonMsg` 才是
  `The limit <testusers_rebuild> has been exceeded`）——先减 task 数或拆作业，
  不要当脚本语法错误排查。
- 取消：`dkill <job-id>` 是历史上使用的终止入口（作业 2112385 与 `v7_r16t38/36`
  都被 `dkill` 停掉）；本轮 `djob -c` 与 `dsub -c` 均报 `invalid option`，pending
  作业只能由用户经调度器 UI 取消。
- `#DSUB` 资源块只有 `dsub -s <job>` 才解析；`submit_kunpeng_all.sh` 用
  `bash -lc "dsub -s ..."` 调用，`--dry-run` 先打印脚本。
- 作业内必须 `source /work_ssd/software/HPCKit/25.2.1/setvars.sh` 并把 HMPI
  `bin`/`lib` 与 anaconda3 `2023.3` 放前面；`OPAL_PREFIX` 未设置时 HMPI 的 `mpirun`
  会把所有选项报成 `unknown option`。`-np` 是正确写法，`-n` 不被 HMPI 接受。
- `mpirun --bind-to none --map-by ppr:2:numa --allow-run-as-root`；
  `KUPL_NUMA_BIND=1`、`KUPL_MEMORY_NODE=local+16`、`KUPL_MEMORY_POLICY=preferred`、
  `KUPL_HUGEPAGE=1`、`OMP_WAIT_POLICY=ACTIVE`、`OMP_DYNAMIC=FALSE`；登录节点自动降级为
  `KUPL_NUMA_BIND=0` 且不设 `KUPL_MEMORY_NODE`（登录节点没有 HBM）。
- 每行记录节点身份：`node`、`freq_khz`（`scaling_cur_freq`，实测 `1 549 990–1 550 057`）、
  `freq_max_khz`（`cpuinfo_max_freq` = `2 000 000`）、`governor`（`userspace`）。

## 4. CSV 列（94 列，按组读）

| 组 | 列 |
|---|---|
| 标识 | `run_id`、`phase`（solve/prepare/prewarm）、`solver`、`status`、`node`、`returncode` |
| 物理参数 | `dtype`、`atol`、`lattice`、`lat_x..lat_t`、`mass`、`csw`、`kappa`、`seed` |
| 并行 | `ranks`、`threads`、`rank_x_thread`、`grid`、`grid_mode`、`shm` |
| 精度 | `iterations`、`final_true_residual`、`relative_full_residual`、`distributed_relative_residual`、`residual_source`、`converged`、`verify_status` |
| 计时 | `wall_seconds`（只含求解调用）、`builder_seconds`（本次 cache 载入）、`build_mg_cache_seconds`（一次性构建）、`elapsed_seconds`（整行墙钟）、`residual_seconds` |
| 输入 | `source_kind/path/file_dtype/sha256`、`gauge_kind/path/file_dtype/sha256`、`source_read_mode`、`gauge_read_mode`、`*_read_seconds`、`*_generate_seconds`、`*_prewarm_seconds`、`input_prebind_seconds`、`hdf5_mpio_read` |
| cache | `cache_kind`（global/ranked）、`cache_path`、`cache_sha256`、`cache_identity_sha256`、`cache_manifest_path/sha256` |
| MG | `mg_levels`、`mg_restart`、`mg_grid`、`mg_dof`、`nv_iters`、`coarse_max_iter`、`coarse_tol_factor`、`deflate`、`max_iter` |
| 剖析 | `profile_rank`、`dslash_*`、`clover_*`、`reduce_*`、`halo_pack/sync/copy_seconds`、`phase_prepare/solve/reconstruct_seconds` |
| 节点 | `freq_khz`、`freq_max_khz`、`governor` |
| 产物 | `report_path/sha256`、`solution_path/sha256`、`out_path`、`err_path` |
| 判读 | `review`（超过 `--timeout` 复核阈值）、`timeout`（rc 124/137）、`error`、`notes` |

- `notes` 是判读入口：`h5py lacks MPI; HDF5 parallel read disabled`、
  `builder_seconds is cache load/build time`、`ranked cache unavailable; using global
  cache`、`shared halo handshake timeout (MG+shm asymmetric)`、`mpirun_seconds=...`。
  CSV 里 `notes` 是 Python repr（单引号），`rows/*.json` 是 JSON——下游解析要区分。
- `append_csv` 只在文件为空时写表头，旧表头文件（本批 3 个 93 列 CSV 缺
  `hdf5_mpio_read`）再追加会整体错列；`--mode merge` 用当前列重写可修表头。
- `save_tensor`/CSV 更新走 `tmp + os.replace`，行 JSON 与 CSV 各写一次；
  同一 `--outdir` 多作业并行安全依赖 `flock`，不要手工并发写同一 CSV。

## 5. 2026-11-15 结果口径

### L24 c64（`24³×64`、ones 源 seed 42、真实组态、18 it、真残差 ~`7.12e-07`、PASS）

| ranks×thr | grid | shm | wall (s) | run_id |
|---|---|---|---:|---|
| 32×18 | `1 1 1 32` | 0 | 0.247571 | `a466a8207fe379e1` |
| 32×18 | `1 1 1 32` | 1 | 0.240473 | `232d6b859b93d000` |
| 32×18 | `2 2 2 4` | 0 | 0.349715 | `5c6cdbe48901c409` |
| 32×18 | `2 2 2 4` | 1 | 0.313275 | `72e04aa67eb8241c` |
| 16×18 | `1 1 1 16` | 0 | 0.350344 | `dfdc9bf54c935e5d` |
| 16×18 | `1 1 1 16` | 1 | 0.326054 | `b2493fd05d6fc276` |
| 16×18 | `2 2 2 2` | 0 | 0.486863 | `8a6e3367f25f9f59` |
| 16×18 | `2 2 2 2` | 1 | 0.451844 | `10b4206e57ac132c` |

- 均衡 4D vs t-only：32 秩 `1.4126×`（shm0）/`1.3027×`（shm1），16 秩 `1.3896×`。
  机制是带宽局部性：同 76 次 dslash，balanced 的 profiler 字节更少
  （`857.3 MB` 对 `1210.3 MB`）但每 rank 只有 `4.06 GB/s`，t-only 为 `8.32 GB/s`。
- 共享 halo 收益在本批很小：t-only 只省 `2.87%`（32 秩）/`6.9%`（16 秩），
  balanced 省 `10.4%`；与 2026-11-13 的 `14.77%`（32r×18t、MPI 对 shm、8 样本）
  不是同一批，不能混引。
- **16 秩与 32 秩不可当 scaling 曲线**：16 秩 20 行里 17 行在 `cn19650`（3 行
  `cn19699`）、32 秩 17 行里 15 行在 `cn19699`（2 行 `cn19650`），节点与 `grid_mode`
  共线且混杂，单样本跨节点同配置差可达 `1.45×`。

### L24 c128（`atol=1e-9`、31 it、真残差 `5.325718e-11`、PASS）

| ranks×thr | grid | shm | wall (s) | run_id | node |
|---|---|---|---:|---|---|
| 32×18 | `1 1 1 32` | 0 | 0.567444 | `f19ba1fef0ccff32` | cn19699 |
| 32×18 | `1 1 1 32` | 1 | 0.475555 | `d2ce9be32b406b0f` | cn19699 |
| 16×18 | `1 1 1 16` | 1 | 0.822888 | `f212adf893cb23e0` | cn19650 |
| 32×18 | `2 2 2 4` | 0 | 0.510789 | `a7e19f7350604c4a` | cn12237 |

- 同节点同配置倍率（32 秩、ones、cn19699）：shm0 `0.567444/0.247571 = 2.29×`，
  shm1 `0.475555/0.240473 = 1.98×`；其中迭代数 `31/18 = 1.72×`，单次迭代
  `1.15–1.33×`。**先减掉迭代数差再谈 c128 的每步代价**。
- c128 的 profiler 字节不按 2 倍缩放（`4013.162 MB` 对 c64 `1210.319 MB`），
  不能靠 dtype 倍率推算带宽。

### MG（`levels=2`、cache 回读、`atol=1e-6`、无可靠 `iterations`）

| ranks×thr | grid | shm | restart | wall (s) | run_id |
|---|---|---|---:|---:|---|
| 32×18 | `2 2 2 4` | 0 | 5 | 0.725579 | `b2715635518ec65b` |
| 32×18 | `2 2 2 4` | 0 | 20 | 0.300639 | `cb136a9593aa2021` |
| 32×18 | `1 1 1 32` | 0 | 20 | 0.252589 | `4d187eadc00a7feb` |
| 32×18 | `1 1 1 32` | 1 | 5 | 0.394854 | `c98060824b55ebf7` |
| 16×18 | `1 1 1 16` | 0 | 5 / 20 | 0.625900 / 0.342719 | `7668f066600ef451` / `06ad0f1f0accf025` |
| 16×18 | `1 1 1 16` | 1 | 5 / 20 | 0.566035 / 0.312759 | `cf5659638b00b61e` / `9c122f681fa3ffe1` |
| 16×18 | `2 2 2 2` | 0 | 5 | 0.888075 | `716b1f6a3fd49353` |
| 16×18 | `1 1 1 16` | 1 | — | 1.445706（c128） | `749b3b028b3248c9` |
| 32×18 | `2 2 2 4` | 1 | 5 | **FAIL** | `b3267714a641ea60` |

- **`restart` 是最大单项**：同配置 `restart=5` → `20` 省 `1.8–2.4×`
  （32 秩 balanced `2.41×`、16 秩 t-only `1.83×`、16 秩 t-only shm1 `1.81×`）。
  CPU 上每次粗层修正都要一次分布式 coarse solve，`restart=5` 在 18 次细迭代里做
  3–4 次。
- MG 对 BiCGStab **没有净收益**：32 秩 t-only shm0 `0.2526` 对 `0.2476`（+2.0%），
  16 秩 t-only shm0 `0.3427` 对 `0.3503`（−2.2%）；而 `builder_seconds`
  （载入 cache）`4.35–6.07 s` 不计入 `wall_seconds`，端到端 `+4.4~6.1 s`
  比单次 BiCGStab 慢 `10–19×`。
- `builder_seconds` 与 `build_mg_cache_seconds` 是两个量：本批复用
  `data/kupl_scaling_20261114/cache_l24_parallel.h5`（历史 `setup_seconds=493.638`，
  `143,335,424 B`）的 6 行 `build_mg_cache_seconds` 完全相同；本批新建的 global cache
  记录为 `48.19 s`（c128 ones）/`100.58 s`（c64 random 源）/`130.37 s`
  （c128 random 源）。相差 10× 且构建口径未逐项对齐，**不要用它们算 cache 加速比**。
- 同一 `run_id` 的复测差值可达 36×（`06ad0f1f0accf025`：`builder` 72.1/78.5/5.64 s，
  `wall` 12.36/12.53/0.34 s）——单样本行一律标 `n=1`，复测优先。

#### MG2 + shm1 边界：1 FAIL / 2 PASS，不稳定（2026-11-15 专项）

`docs/kupl_l24_c64_mg2_shm1_boundary_20261115.md` 用同一驱动补测 8 格
（L24 c64、`mg2`、`shm=1`、`atol=1e-6`、global cache、`threads=18`）：

| 单元 | restart | 结果 |
|---|---:|---|
| 16×18 balanced `2 2 2 2` | 5 / 20 | PASS ×2 |
| 16×18 t-finest `1 1 1 16` | 5 / 20 | PASS ×2 |
| 32×18 balanced `2 2 2 4` | 5 | **FAIL ×1 / PASS ×2（不稳定）** |
| 32×18 balanced `2 2 2 4` | 20 | PASS |
| 32×18 t-finest `1 1 1 32` | 5 / 20 | PASS ×2 |

- FAIL 行 `b3267714a641ea60`：`rc=2`、`MPI_ABORT` rank 30、
  `KUPL shared halo handshake timeout: publish` ×10（`.err` 里才有，台账 `notes`
  只有 `ranked cache unavailable` / `no report.json`）、`elapsed=271.437 s`、
  `timeout/review=False`（不是 600 s 硬超时）。
- 同 config 在独立 outdir 复测 **2/2 PASS**（`cn12104`、`elapsed` 203.832/193.510 s、
  `verify=0`、真残差 `7.923186435088777e-07` 与 shm0 的 `b2715635518ec65b` **逐位相同**）。
- 机制（读代码）：`shm_halo_spin` 的 30 s 窗口内没等到对端 `publish` 计数即抛错，语义是
  "邻 rank 的 dslash 调用序列不一致"；4 维 node-local 不足以解释（16×18 balanced 也 4 维
  且 PASS），更可能是 32 秩 4 rank/NUMA 放大了竞态窗口——**假设，未做 A/B**。
- 处置：**不写 `known_skips.json`**（条件"稳定失败"不成立），只留
  `known_skips.candidate.json`（`enabled=false`，提升门槛为再观察 ≥2 次 FAIL）；不得整体
  禁 `balanced + mg2 + shm1`，不得静默回退 `shm=0`；允许的最窄做法是
  "跑 3 次取多数/失败重跑 1 次"。
- **`--resume` 判重读 `outdir/rows/<run_id>.json`，不是 `--csv`**：换新 CSV 重跑同一
  config 不会执行（作业 `2133280` 返回 `{"done": 2}` 却无新行），重测必须换 outdir。

### L32（c64/c128，6 条全部 FAIL —— 缺一次有效运行，不是重测）

| run_id | dtype | ranks×thr | grid | shm | 结论 |
|---|---|---|---|---:|---|
| `327f9aa531167d55` | c64 | 32×18 | `1 1 1 32` | 0 | FAIL（无 report） |
| `c9bfd6bfb733b052` | c64 | 32×18 | `2 2 2 4` | 0 | FAIL（无 report） |
| `23daf98e9c477bdd` | c128 | 32×18 | `1 1 1 32` | 0 | FAIL（无 report） |
| `35548a2b06ff0989` | c128 | 32×18 | `1 1 1 32` | 1 | FAIL（无 report） |
| `07c80fc8` | — | — | — | — | FAIL（快照后，本地无 row） |
| `44faded2` | — | — | — | — | FAIL（快照后，本地无 row） |

根因**不是算法**：32 个 rank 都在 `kupl_multigrid.py::_read_local_slab` 的
`h5py.File(..., "r")` 处抛 `OSError: Unable to open file (bad object header version
number)`，且 traceback 行是 `source_local = _read_local_slab(...)`——失败在**源文件**
而不是 gauge。6 条用的都是坏 ones（`source_sha256=a9c2acd8…`）。

修复与判定（2026-11-15 16:28）：

- L32 `ones` 与 L24 `point` 已换成可读文件；坏字节保留为 `*.corrupt-20261115`。
- `ones.h5` 与 `ones_v2.h5` 的 `file_sha256` 同为 `9cdcfc82…f55a`、dataset 全 `1+0i`、
  **逐字节相同 → 无需重新生成**。
- `source_32x32x32x64_c64.h5` 是 **random seed42**，不能当 ones 用。
- **6 条 FAIL 需要一次"首次有效运行"，不是"重测"**（它们没有产生过有效性能数据）。
- L24 point `c3bd73d2` 按范围裁定**放弃**。

详见 [`pyqcu-testing-data/SKILL.md`](../../pyqcu-testing-data/SKILL.md) 与
[`docs/kupl_matrix_scope_20261115.md`](../../../docs/kupl_matrix_scope_20261115.md) §3。

## 6. 判读红线

1. `wall_seconds` 只含一次 `applyCloverMultigridQcu` 调用：不含 Python 导入、
   `mpirun` 启动、gauge/source 读取、cache 构建、独立真残差。
2. **先看 `gauge_read_seconds`/`source_read_seconds`/`*_read_mode` 再看 `wall`**：
   warm 读 `2.4–5.2 s`，cold 行 `29.9–35.6 s`，对应 `wall` 被抬高
   `33–60×`（`11.44/10.93/20.86 s` 对 `0.24–0.35 s`），`dslash_gbps_rank` 掉到
   `0.19–0.44`。cold 行不得进入性能表。
3. `--strict-acceptance` 让判据等于 `atol`；旧批次不带该开关时判据被 dtype 地板
   抬高（c64 `max(atol, eps(float32)*2048) = 2.441e-04`），有效门槛差 `244×`，
   迭代数 11 → 18。比 `atol` 之前先比 `residual_acceptance`。
4. 节点是混杂因子：本批 16 秩与 32 秩落在不同节点，跨节点同配置单样本差可达
   `1.45×`；轴对齐比较必须锁定同一 `node`。
5. `mg2` 行没有 `iterations`（native MG 不暴露）与 `dslash/clover/reduce` 剖析
   （只有 `KUPL_PROFILE_OPT` 的 BiCGStab 行输出）；这不是失败，但必须显式标注
   `N/A`，否则空值会被当成"更快"。
6. 随机组态/随机源行与真实组态不可并列；`real1`、`point`、`random` 是不同源，
   迭代数与真残差都不同。
7. **超时只留 partial 输出**：`rc=124/137`（600 s 硬超时）的 cache 行可能已落盘并
   判 `PASS`（`cache-2139960e885a0627`/`cache-3b6e1e37b3af0b9c` 的
   `elapsed=602.2/602.1 s`）——这只证明"部分输出可用"，不能当"L32 可跑"或 cache
   构建成功的证据。solve 行没有 `reports/<run_id>.json` 一律 FAIL，而
   `--resume --skip-failed` 会让这些 FAIL 单元永久跳过。
8. **输入资产只读**：不改写、不修补 `data/production_*/**`；损坏文件只能改名重生成
   并保留 `*.corrupt-<date>` 备份，回写一次会让整批 rank 在求解前全部退出。

## 7. 启动摊销与 loop 原型（`data/kupl_loop_proto_20261115/`）

- 计算节点上**每个 `mpirun` 的固定成本约 120 s**：import 总计 `80.8 s`
  （numpy `7.8` + mpi4py `34.9` + driver `38.1`），MPI 起跑线约 `39 s`；登录节点
  import 只有 `1.3–2.1 s`。
- 因此一行 CSV 的 `elapsed_seconds` 为 `95–211 s`（16 秩 ≈100 s、32 秩
  `183–211 s`），而 `wall_seconds` 只有 `0.21–0.57 s`：**启动占 99% 以上**。
  矩阵吞吐由启动主导，不以 `wall` 排序作业成本。
- loop 原型（`loop_proto.py`，throw-away，不入库）在**同一个 `mpirun` /
  `COMM_WORLD` 内**反复调用 `kupl_multigrid.main(argv)`：3 个用例时进程总计
  `107.4 s`（其中会话 `26.6 s`，单例 `4.2–12.3 s`），对比 3×`~120 s` 的一次一进程。
  FD 数稳定在 154，RSS 从 228 MB 缓慢涨到 258 MB。
- 故障注入（`fault.target=applyCloverMultigridQcu`）与 `SystemExit(2)`（非法 grid）
  只让**该用例**失败，后续用例仍 PASS：驱动的 `try/finally` teardown 足以支持
  进程内复用。前提是用例之间不得共享 `_SET_PLAN_` 残留与 cache 句柄；
  2 rank 下 `loop_mixed` 的 RSS 从 589 MB 涨到 925 MB、`wall` 波动到 19.5 s，
  说明复用有代价，正式批次仍以"一例一进程"为准。

## 8. h5py MPI 与 cache 策略

- **只做接口检查，不重新编译/安装 MPI h5py**。本批计算节点 conda `2023.3` 实测
  h5py `3.7.0` / HDF5 `1.10.6`、`h5py.get_config().mpi=False`、`mpio` driver 不存在，
  因此 `h5py_has_mpi()`（`test_kunpeng_all.py:1090-1096`）恒为 `False`，驱动在
  `718-721` 的自动分支**永远不会**设置 `KUPL_HDF5_MPIO_READ=1`（该分支只在
  `h5py_has_mpi()` 为真时写，否则 `env.pop(...)`）。
- 输入读的集体 MPI-IO 是**显式 opt-in**：`KUPL_HDF5_MPIO_READ=1`，且
  `kupl_multigrid._hdf5_mpio_read_enabled()`（`kupl_multigrid.py:95-112`）会在所有 rank
  上 `allgather` 投票，不一致或 h5py 无 MPI 时直接抛错关闭。本轮请求值为 0，校验不进。
  因此**没有任何一行记录 `hdf5_mpio_read=true`**：28 行显式 `false` 并带 note
  `h5py lacks MPI; HDF5 parallel read disabled`（`test_kunpeng_all.py:1442-1443`），
  其余 14 行（5 条 prepare + 9 条 solve，旧脚本版本）缺该键；读路径回退为**每 rank
  全量 open + 本地 slab**（`_read_local_slab`，`kupl_multigrid.py:1210`）。
- **ranked cache 因 MPI h5py 不可用而 blocked，不列入后续计划**：`prepare_cache_rows`
  在 `kind == "ranked" and not h5py_has_mpi()` 时直接产出 `status=SKIP`
  （`test_kunpeng_all.py:1123-1131`，本轮落在 `cache-042c63a815c8d0b3`）。**不要**把
  "编译/安装 MPI h5py"或"启用 ranked cache"写成下一步；`bg-3` 的离线构建与 `bg-23` 的
  ranked cache 尝试都不作为本轮交付，其产物未验证、不得引用。
- 保留的只有：① 每批开跑前记 `h5py.__version__`、`hdf5_version`、`get_config().mpi`
  与 `mpio` 是否存在；② capability-aware fallback（无 mpio 时不设该变量、不排 ranked
  cache、I/O 预算按"文件字节 × rank 数"估算，`prewarm` 是唯一低成本重复读）。
- cache 两类：`global`（1 rank 构建，任何 rank/grid 都能读，identity 不含 grid）与
  `ranked`（32 rank 按 grid 构建，需要 h5py MPI + 均衡 4D 分解）。`prepare` 先查
  `cache_index.json` 与 manifest `identity`，命中即 `REUSE`；`solve` 找不到 ranked
  时回退 global 并写 note。
- identity 字段：`lattice`、`dtype`、`mass`、`csw`、`mg_grid`、`nv_iters`、`dof`、
  `seed`、`gauge_kind`、`ranks`、`grid`；**不含** `atol`/`coarse_tol_factor`/
  `coarse_max_iter`/`restart`/`deflate`（设计正确：这些不改变 cache 内容）。
  `change identity` 就会重建，禁止按文件名或文件大小复用。
- `--mode prepare` 只处理 `mg2` 行；`bistabcg` 不需要 cache。
- **L48（2026-11-17）**：只有 **16 rank** 的 Legacy-MG cache（`E=2`、`1.276 GiB`）
  可用，且只能被**同 identity 的 16 rank 行**复用；`32 rank`/两节点**没有 identity
  匹配的 cache → 暂缓（DEFERRED），不重建**。判据与本节 identity 字段完全一致：
  不按文件名/路径/大小复用，`verify_status` 必须为 `0`。权威记录见
  [`docs/pyqcu_test_l48_remote.md`](../../../docs/pyqcu_test_l48_remote.md) §0.1。
- **L48 的 CG 组已退役**：没有 CG harness，不再排 CG 行；不要为了“补齐矩阵”新增
  L48 CG 用例。L48 的 c64 BiCGStab 32 rank `1.6646 s`（`2152757`）与 16 rank
  共享节点 `183.0 s`（病态，不作基线）见 `kupl-history-2026.md` 同日条目。

## 9. 复现与审计入口

```bash
# 台账审计脚本组（只读；输入是 data/kunpeng_all_20261115 的快照）
cd data/kunpeng_all_20261115_audit/scripts
python3 check_headers.py      # CSV 表头 vs CSV_COLUMNS
python3 check_align.py        # 每行字段数
python3 check_csv.py          # run_id 唯一 / 跨 CSV 重复 / CSV-vs-JSON
python3 check_consistency.py  # 残差分流 / MG 标注 / builder 口径 / 矩阵覆盖
python3 check_hashes.py       # report_sha256 完整性
python3 analyze_diff.py       # 2x2 受控矩阵与旧参考的乘性分解
python3 render.py             # ledger.md + ledger.csv
```

判读顺序：`status`/`returncode` → `*_read_mode` 与 `*_read_seconds` →
`atol`/`residual_acceptance` → `node`/`grid_mode`/`shm` → 才看 `wall_seconds`。

## 10. 范围裁定、known skips 与 cache 重签（2026-11-15）

权威版与判据见 [`docs/kupl_matrix_scope_20261115.md`](../../../docs/kupl_matrix_scope_20261115.md)；
机器可读版见 `data/kunpeng_all_20261115/known_skips.json` 与同目录 `quarantine.json`。
`MG2 + shm1` 只有 `known_skips.candidate.json`（`enabled=false`）——复测 2/2 PASS，
**当前没有生效中的 known skip**。

### 10.1 放弃轴（耗时代价过大，部分数据保留，不重测/不新增）

| 轴 | 已保留 | 处置 |
|---|---|---|
| thread 9（16×9、32×9、32×9 shm1 复核） | 3 行（`b614ad9349f5d4e6`、`baf92e2b99937b2b`、`861ac912ac3c7c42`） | 不扩展；`861ac912` 的 `11.437897 s` 不再复核，引用必须剔除 |
| seed24 | 2 行（`0041182f2d8e1c50`、`9cc71c5d52dbbf22`） | 不扩展 |
| random gauge | 1 行（`4c5cd8a65e492c1e`） | 不重跑 |
| random 源 | 2 行（`9cc71c5d52dbbf22`、`e1574f172b525c02`） | 不新增 |
| point 源 | 1 行（`c3bd73d2999000f0`，FAIL） | 源已修复，但 point 性能不跑 |
| mg-restart 50 | **0 行** | 不新增 |

保留继续：16×36（3 行）、MG1、r10、L32/c128、MG+shm。

### 10.2 cache 重签（不重跑数值）

| cache | identity 问题 | 处置 |
|---|---|---|
| L32 c64（`2139960e885a0627`） | `gauge_file_sha256=b2a57594…` 指到 1.2 GB c128 文件 | 重签为 `production_L32_c64/gauge_32x32x32x64_c64.h5`（604 MB，`70bbe621…`）；canonical `9ac7c15c…` 不变 → 不重跑 |
| L32 c128（`3b6e1e37b3af0b9c`，679 MB） | 无（`b2a57594…`/`52f3c07a…` 与实际文件一致） | **不重建** |
| L24 c64（`b708cefe4583941a`） | `gauge_file_sha256=4777e795…` 指到 509 MB c128 文件 | 建议重签为 `production_L24/gauge_global_c64.h5`（`8a482d6a…`） |
| 所有 MG cache | identity 不含 source 哈希 | source 修复**不失效** |

### 10.3 重测判据

同口径才不重测：source 语义/hash、gauge dtype/canonical、cache identity、`atol`、
`grid`/`ranks`/`threads`/`shm`/`solver`/`mg_restart`。任一不同必须重跑。按此规则
**31 条 PASS / L24 数据不重跑**；**L32 6 条 FAIL 是首次有效运行，不是重测**；L24 point
`c3bd73d2` 放弃。

### 10.4 队列状态

`bg-26`（thread9）、`bg-27`（seed24/random）、`bg-28`（point）、`bg-32`（r50）**已 kill**，
但对应 pending 作业未能在 CLI 取消：`djob -c` 与 `dsub -c` 均报 `invalid option`，队列中
仍可能 `PENDING`，可由用户经调度器 UI 取消。**保留** `kall_l32_*`、`kall_l24_c1*`、
`kall_mg_shm*`、`kall_mg1_r5*`、`kall_mgr10*`、`k9_16x36*`；不 kill 保留作业、不删数据。
