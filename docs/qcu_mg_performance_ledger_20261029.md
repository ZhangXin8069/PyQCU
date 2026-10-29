# QCU、KUPL、Torch 与 QUDA 实时性能账本（2026-10-29）

状态：`active`，活动工作树记录，未打 tag。本文按用户要求实时记录性能测试，
不是永久性能承诺。每个批次完成、失败或被取消后，必须在同一工作日追加记录并保留
机器可读证据路径。

## 1. 固定口径

### 1.1 仓库与硬件

| 项目 | 当前值 |
|---|---|
| Git 基线 | `dev106=2c8fccc8`，工作树有未提交改动 |
| GPU | NVIDIA A100-PCIE-40GB，单节点 |
| CPU 对照平台 | Kunpeng-920 控制节点，2 NUMA；不是 HMPI/片上内存鲲鹏 |
| CUDA 架构 | `sm_80` |
| 正式大格下限 | `32x32x32x64` |
| smoke 格点 | `8x8x8x16`、`24x24x24x64`，不得进入正式大格加速比 |

### 1.2 L32 输入

| 项目 | 值 |
|---|---|
| Gauge | `beta6.20_mu-0.2770_ms-0.2400_L32x64_cfg_11000.lime` |
| Gauge SHA256 | `daa59a53542eade112ee5d5ebce7e1166cdcd57525502a29cef1d7deb04fd4c6` |
| RHS | PCG64 float64 正态流采样后 cast c64，seed 42 |
| RHS SHA256 | `8c00a7341b7abb2b3d0e26e477731c3528c86c337888cfe1e539b0e47673ddb6` |
| mass | `-0.277` |
| kappa | `0.13430029546065003` |
| csw | `1.160920226` |
| 停机 | `atol=1e-6`，`max_iter=1000` |

## 2. L32 native BiCGStab 同算法矩阵

入口与参数：

| 项目 | CUDA | QUDA |
|---|---|---|
| 公共入口 | CUDA worker，调用 `applyCloverBistabCgQcu` | PyQUDA `Dirac.invert` |
| native solver | QCU native BiCGStab | `QUDA_BICGSTAB_INVERTER` |
| solve type | Clover odd-Schur native solve | `QUDA_DIRECT_PC_SOLVE` |
| normalization | QCU mass/kappa | `QUDA_KAPPA_NORMALIZATION` |
| warmup/repeat | `1/5` | `1/5` |
| tuning | PyQCU default | `QUDA_ENABLE_TUNING=0` |

结果：

| backend | ranks/grid | iterations | steady wall (s) | independent true residual | PyQCU/QUDA wall ratio |
|---|---:|---:|---:|---:|---:|
| CUDA | `1 / 1x1x1x1` | 18 | 0.1224217 | `6.1676169e-7` | 1.065x |
| CUDA | `2 / 2x1x1x1` | 20 | 0.1154108 | `2.9952355e-7` | 0.833x |
| CUDA | `4 / 2x2x1x1` | 19 | 0.1090329 | `4.8919664e-7` | 1.045x |
| QUDA | `1 / 1x1x1x1` | 17 | 0.1303580 | `8.8903185e-7` | 1.000x |
| QUDA | `2 / 2x1x1x1` | 17 | 0.0960886 | `8.6055751e-7` | 1.000x |
| QUDA | `4 / 2x2x1x1` | 17 | 0.1139240 | `8.6239248e-7` | 1.000x |

结论：CUDA 在 1/4-rank 分别约快 `6.5%/4.5%`，但 2-rank 约慢 `20%`，且 CUDA
迭代数多 `1-3` 次。当前不能宣称 CUDA native BiCGStab 在所有要求 rank 上稳定
快于 QUDA。

证据：

- `logs/qcu_sync_resume_20261029/large-volume/qcu-bicg-r{1-v2,2,4}/report.json`
- `logs/qcu_sync_resume_20261029/large-volume/quda-bicg-r{1,2,4}/report.json`
- `data/qcu_sync_resume_20261029/large_volume_plots/l32_bicgstab_summary.{json,csv}`

该批 CUDA 库 SHA256 为 `d24dd3ae3a5a13a803a9d6ae4fa94e3ec398c60599458bee9dd7b0f5e1cedf24`，
QUDA 库 SHA256 为 `3e43627f16adc0384416ca3081cd27a6b475052558c84688c9870bab8ecf730a`。

## 3. L32 CUDA Legacy-MultiGrid

参数：

- 公共入口：`applyCloverMultigridQcu`
- variant：Legacy-MultiGrid
- levels：2
- dof：`[12,12]`
- `mg_grid=[2,2,2,2]`
- `restart=20`
- `coarse_max_iter=200`
- `coarse_tol_factor=1e4`
- `atol=1e-6`，`max_iter=1000`
- warmup/repeat：`1/5`
- cache：`data/qcu_mg_sweep_20261029/cache_L32_E12_12.h5`
- cache SHA256：`ec11ebf8bf6ea12259838b45df944110618699fddd57f1d5bcf242dc984703ba`
- cache identity SHA256：`a79e0f60ea8e4dac6acb0e6065615936e4036d6bb5d2924fbc9d4d4b5b33f369`

结果：

| ranks/grid | solve wall (s) | independent true residual | status |
|---|---:|---:|---|
| `1 / 1x1x1x1` | 0.2144558 | `3.7442830e-7` | PASS |
| `2 / 2x1x1x1` | 0.2195061 | `3.7473052e-7` | PASS |
| `4 / 2x2x1x1` | 0.2183433 | `3.7475920e-7` | PASS |

实际执行 `n_vcycles=1`、`coarse_iters=8`，不是 fine-only 包装。多 rank 没有强扩展
收益。同输入、同 1/2/4-rank 的 QUDA GCR-MG 对照现已补齐：

| ranks/grid | QUDA GCR-MG (s) | QUDA iters | CUDA Legacy-MG (s) | QUDA/CUDA | CUDA true residual |
|---:|---:|---:|---:|---:|---:|
| `1 / 1x1x1x1` | 0.482425556 | 4 | 0.214455826 | 2.2495x | `3.744283e-7` |
| `2 / 2x1x1x1` | 0.314393339 | 4 | 0.219506067 | 1.4323x | `3.747305e-7` |
| `4 / 2x2x1x1` | 0.287465227 | 4 | 0.218343299 | 1.3166x | `3.747592e-7` |

QUDA 的 independent full-operator residual 分别为
`1.240173728e-7/1.243527887e-7/1.247108894e-7`。QUDA 入口为 GCR-MG，
2 levels、block `2^4`、`nvec=12`、no-MMA、内部 null-vector setup；CUDA 入口为
`applyCloverMultigridQcu`，restart `20`、dof `[12,12]`、外部 E12 cache。该表证明
同格点、同 rank 的 MG 入口比较中 CUDA Legacy 更快，但因循环和 cache 所有权不同，
不能外推为任意算法配置的普适结论。

证据：`logs/qcu_sync_resume_20261029/large-volume/qcu-mg-r{1,2,4}/report.json`。

## 4. CUDA Strict-MG 正式小格门

早期 smoke 参数：

- 输入：随机 Wilson-Clover 测试场，不是 LIME，不是正式大格
- 格点：`8x8x8x16`
- 精度：c64
- mass：`0.1`
- csw：`1.0`
- 入口：`strict_mpi_solve_probe.py`，Torch hierarchy 负责 setup，CUDA fused
  `applyMultigridStrictFgmresQcu` 负责求解
- levels：2
- block：`[2,2,2,2]`
- nvec：`2`
- coarse dof：4
- `target_parity=1`
- `restart=20`，`atol=1e-6`，`max_iter=30`
- warmup/repeat：未设置；每项单次 smoke

结果：

| ranks/grid | candidate iterations | candidate true residual | baseline iterations | baseline true residual |
|---|---:|---:|---:|---:|
| `1 / 1x1x1x1` | 25 | `7.8125409e-7` | 未取得 | 未取得 |
| `2 / 2x1x1x1` | 26 | `7.6480e-7` | 29 | `6.56e-7` 至 `6.89e-7` |
| `4 / 2x2x1x1` | 27 | `6.3880e-7` | 29 | `7.5685e-7` |

早期结论：正确性与 1/2/4-rank MPI 路径 PASS；原始记录仅保存进程级时间，不能作为
solve-only wall。该批不是正式性能结果。candidate 库 SHA256 为
`d24dd3ae3a5a13a803a9d6ae4fa94e3ec398c60599458bee9dd7b0f5e1cedf24`，baseline
库 SHA256 为 `fb2264bcc392aebc9225ae65448db5dfe536ed487367bcc6935b08f3ba7e5c60`。

证据：`logs/qcu_sync_resume_20261029/a100-validation-20261029/strict/01..10-*`。

### 4.1 2026-10-29 当前 Strict-MG 1/2/4-rank

输入为统一 L24 前缀 `8x8x8x16`，c64，mass `-0.277`、kappa
`0.13430029546065003`、csw `1.160920226`、2 levels、block `2x2x2x2`、
`nvec=12`、`coarse_spin=2`、`restart=16`、`coarse_max_iter=200`、
`coarse_tol=0.003`、`nu_pre=nu_post=1`、`atol=1e-6`，2 warmup + 5 steady。

| ranks/grid | median (s) | MAD (s) | FGMRES iters | independent residual | status |
|---:|---:|---:|---:|---:|---|
| `1 / 1x1x1x1` | 0.0270776 | 0.0000483 | 9 | `2.730702e-7` | PASS |
| `2 / 1x2x1x1` | 0.0321998 | 0.0001737 | 10 | `6.257694e-7` | PASS |
| `4 / 2x2x1x1` | 0.0419196 | 0.0001740 | 10 | `5.189006e-7` | PASS |

每 rank 的 trace 均执行 coarse solver；4-rank 的 level-0 restriction 为 0，
coarse solver 调用 10 次、20 次 coarse iterations。证据：
`data/qcu_strict_perf_dispatch_20261029/cuda/runs/8x8x8x16/r{1,2,4}/measure.json`。
1-rank 数值已在 trace counter 修复后用同 cache、同 2 warmup + 5 steady 重跑；
新 trace 正确记录 9 次 restriction/prolongation，独立残差不变。原 1-rank 原始
JSON 被该同配置回归覆盖，正式表采用最新可审计文件；2/4-rank 仅未重跑计数器展示，
求解 kernel 与数值路径未改。

与 Legacy case 使用同一 gauge/source 的 QUDA GCR-MG 1/2/4-rank 中位数为
`0.0173493/0.0244426/0.0297798 s`、8 iterations，independent residual
`5.136697e-7/5.139111e-7/5.140204e-7`。因此，小型 `8x8x8x16` 上 Strict 的
固定开销仍高于 QUDA；大格生产默认继续保留 Legacy，Strict 作为显式对照/备选。

### 4.2 同一 \(8^3\times16\) 输入上的 CUDA Legacy-MG

参数：`applyCloverMultigridQcu`，c64，mass `-0.277`，kappa
`0.13430029546065003`，csw `1.160920226`，2 levels，`dof=[12,2]`，
`mg_grid=2^4`，`restart=20`，`coarse_max_iter=200`，
`coarse_tol_factor=3000`，`nv_iters=1`，2 warmup + 5 steady，加载对应 rank 的
CUDA E2 cache。1/2/4 rank 均执行 16 iterations、1 个 V-cycle/correction attempt，
1-rank accepted，2/4-rank correction 被 residual guard 拒绝但后续 fine solve 收敛。

| ranks/grid | Legacy median (s) | QUDA aligned median (s) | QUDA/Legacy | independent residual | status |
|---:|---:|---:|---:|---:|---|
| `1 / 1x1x1x1` | 0.0388045 | 0.0173493 | 0.4471x | `2.282105e-7` | PASS |
| `2 / 1x2x1x1` | 0.1128736 | 0.0244426 | 0.2165x | `2.071858e-7` | PASS |
| `4 / 2x2x1x1` | 0.1670967 | 0.0297798 | 0.1782x | `2.063892e-7` | PASS |

该小格点的 QUDA MG 固定开销和粗层并行路径明显更优；不能把 L32 上 Legacy 的
入口级胜出外推到 `8^3\times16`。L32 是生产性能点，`8^3\times16` 是统一正确性门。
同源图表与机器可读汇总：
`data/qcu_strict_perf_dispatch_20261029/mg_8x8x8x16_summary.{json,csv,png,svg}`
和 `data/qcu_strict_perf_dispatch_20261029/mg_l32_legacy_vs_quda.{png,svg}`。

### 4.3 L32 CUDA Strict-MG

记录时间：2026-10-29 09:47。输入与 Legacy/QUDA 对照相同：
`32x32x32x64`、c64、mass `-0.277`、kappa `0.13430029546065003`、
csw `1.160920226`；两层、block `[2,2,2,2]`、`nvec=12`、`coarse_spin=2`、
`restart=16`、`coarse_max_iter=200`、`coarse_tol=0.003`、
`nu_pre=nu_post=1`、`atol=1e-6`、`max_iter=1000`，2 warmups + 5 steady。
公共入口为 `pyqcu.solver.Multigrid(variant='strict', backend='cuda')`，计时 native
入口为 `applyMultigridStrictFgmresQcu`。

没有重建 null-space。canonical full basis 从已有 L32 Legacy E12 cache 的 `lonv`
逐向量重构并在 GPU 上验签：

| 资产 | shape | SHA256 |
|---|---|---|
| canonical null | `[12,4,3,32,32,32,64]` c64 | 文件 `7940376ab...f468df`，逻辑 dataset `12deb1d84...c446cf` |
| parity gauge | `[2,3,3,4,32,32,32,32]` c64 | 文件 `21becd92e...1d06a` |
| parity RHS | `[2,4,3,32,32,32,32]` c64 | 文件 `795ffd412...c92e87` |

1-rank/1-GPU 在 `StrictTransfer._block_orthogonalise` 的正交化阶段 OOM：A100-40GB
总容量 `39.39 GiB`，当时已用 `39.37 GiB`，新请求仅 `96 MiB` 即失败。没有生成
runtime cache，也没有求解或性能样本，因此不得记作数值 FAIL。按一进程/rank/设备
改为 2-rank 后，每卡 40GB 可以容纳本地 full basis；cold setup 成功。

2-rank 作业 Slurm `1603843`：`16 CPU / 2 GPU / 27488 MiB`、10 分钟上限；实际
`4m17s` 完成。cold setup `201.073175 s`，cache save `5.486443 s`，首次 solve
`0.572468 s`，每 rank runtime cache `5,435,835,264 B`。首次 solve 只用于验证，
不进入 steady 性能。

| ranks/grid | cold setup (s) | cache/rank (B) | steady median (s) | MAD (s) | FGMRES iters | independent true residual |
|---:|---:|---:|---:|---:|---:|---:|
| `1 / 1x1x1x1` | OOM，未进入 solve | n/a | n/a | n/a | n/a | n/a |
| `2 / 2x1x1x1` | 201.073175 | 5,435,835,264 | rank0 `0.474707417`; sync max `0.476122238` | `0.000610`–`0.001461` | 11 | `6.253315328e-7` |
| `4 / 2x2x1x1` | 111.872657 | 2,717,926,272 | rank0 `0.351840196`; sync max `0.356964148` | `0.001039`–`0.001580` | 11 | `6.320174e-7` |

cache-hit measure 作业 Slurm `1603858` 用时 `2m34s`，其中 solve-only steady 为
rank0 `[0.476386124, 0.473246205, 0.474707417, 0.474060473, 0.478756777] s`，
rank1同步最大值中位数 `0.476122238 s`，两 rank 均值 `0.475414827 s`。trace 显示
level-1 `coarse_solver_calls=11`、`coarse_solver_iterations=25`，证明执行了粗层
求解；顶层 `restriction_calls` 字段却为 0，与 `fine_restriction` stage 冲突，后续
报告以 stage、coarse calls 和 coarse iterations 三者交叉核验，不能只读一个字段。
4-rank cold 作业 Slurm `1603871` 用时 `2m50s`，cache-hit measure 作业 Slurm
`1603881` 用时 `1m42s`。4-rank steady 为 rank0
`[0.358812084, 0.359446023, 0.351388168, 0.351840196, 0.350658788] s`，
4-rank 同步最大值中位数 `0.356964148 s`，四 rank 均值 `0.353600223 s`。trace
同样显示每 rank 11 次 coarse solver 调用、26 次 coarse iterations。2→4 rank
wall 加速为 `0.476122238/0.356964148=1.334x`，相对并行效率为 `66.7%`。
4-rank 每卡设备级峰值 `7,859,470,336 B`，runtime resident
`1,509,949,440 B`，明显低于 2-rank 的冷启动峰值。

同格点、同 rank 的 Legacy-MG 为 `0.219506067/0.218343299 s`，QUDA GCR-MG 为
`0.314393339/0.287465227 s`。因此 2/4-rank 的 Strict/QUDA 分别为
`1.514x/1.242x`，Strict/Legacy 分别为 `2.169x/1.635x`；当前 L32 Strict 均比
Legacy 和 QUDA 慢，不应切换生产默认。2-rank Strict runtime-owned resident 为
`3,019,898,880 B`，coarse workspace `534,790,156 B`，设备级峰值
`14,681,505,792 B`。`restart=16` 的 fused FGMRES 实际 workspace 为
`1,887,436,800 B`；512 MiB 上限会主动 fail-closed，本次正式运行使用 2 GiB 上限。
2-rank native 字段 `0.0046466533`、4-rank `0.0046474584` 与独立相对真残差
`6.253315e-7/6.320174e-7` 不是同一归一化口径，正式验收只采用独立
full-operator residual。

证据：`data/qcu_strict_perf_dispatch_20261029/cuda/runs/32x32x32x64/r2/{cold,measure}.json`
、`.../r4/{cold,measure}.json` 以及对应 `logs/`、`trace/`。1-rank OOM 证据为
`.../r1/logs/cold.stderr.txt`。统一图为
`data/qcu_strict_perf_dispatch_20261029/mg_l32_mg_comparison.{png,svg}`，机器汇总为
同目录 `mg_l32_summary.{json,csv}`；图中 1-rank Strict 明确标记 OOM，不填伪造值。

## 5. CUDA Legacy 优化候选隔离

测试口径：

- 格点：`24x24x24x64`，c64
- mass：`-0.277`
- kappa：`0.13430029546065003`
- csw：`1.160920226`
- 入口：`applyCloverMultigridQcu`
- levels：2
- dof：`[12,12]`
- block：`[2,2,2,2]`
- `restart=20`，`coarse_max_iter=200`，`coarse_tol_factor=1e4`
- `atol=1e-6`，`max_iter=1000`
- 4-rank，grid `[2,2,1,1]`
- 每项 10 个交错 A/B 样本

结果：

| 候选 | baseline median (s) | candidate median (s) | ratio | baseline MAD (s) | candidate MAD (s) |
|---|---:|---:|---:|---:|---:|
| dot-pair only | 0.1700721 | 0.1603017 | 1.0610x | 0.00778 | 0.00839 |
| barrier-only | 0.1687275 | 0.1695029 | 0.9954x | 0.01150 | 0.01725 |

结论：dot-pair 有候选信号但方差过大；删除 barrier 无可观测收益。组合候选此前在
4-rank 约慢 `11%`，不能保留为稳定性能优化。该批次属于 smoke/调参，不进入正式 L32
性能表。

证据：

- `logs/qcu_sync_resume_20261029/a100-validation-20261029/interaction/summary.json`
- `logs/qcu_sync_resume_20261029/a100-validation-20261029/legacy-perf/summary.json`
- Slurm `1603556`，完成时 `7m27s`，ExitCode `0:0`

### 5.1 CUDA final candidate smoke 矩阵

记录时间：2026-10-29 06:55，Slurm `1603584` 的 `final-matrix` 阶段。

- 输入：真实 L24 LIME，`24x24x24x64`，c64
- mass：`-0.277`，kappa：`0.13430029546065003`，csw：`1.160920226`
- 入口：`applyCloverMultigridQcu`
- levels：2，dof：`[12,12]`，block：`[2,2,2,2]`
- restart：20，coarse max iter：200，coarse tolerance factor：`1e4`
- atol：`1e-6`，max_iter：1000
- 每个 variant 每 rank 单次运行，warmup=0，repeat=1；本表是 smoke，不是正式统计

| ranks/grid | variant | solve wall (s) | independent true residual | status |
|---|---:|---:|---:|---|
| `1 / 1x1x1x1` | candidate | 0.1587396 | `2.8791103e-7` | PASS |
| `1 / 1x1x1x1` | candidate | 0.1158422 | `2.8791103e-7` | PASS |
| `2 / 2x1x1x1` | baseline | 0.1757574 | `2.8837007e-7` | PASS |
| `2 / 2x1x1x1` | candidate | 0.1469283 | `2.8837007e-7` | PASS |
| `2 / 2x1x1x1` | candidate | 0.1459943 | `2.8837007e-7` | PASS |
| `2 / 2x1x1x1` | baseline | 0.1498923 | `2.8837007e-7` | PASS |
| `4 / 2x2x1x1` | baseline | 0.1683383 | `2.8844304e-7` | PASS |
| `4 / 2x2x1x1` | candidate | 0.1660481 | `2.8844304e-7` | PASS |
| `4 / 2x2x1x1` | candidate | 0.1665799 | `2.8844304e-7` | PASS |
| `4 / 2x2x1x1` | baseline | 本轮截止时待完成 | 待完成 | 待完成 |

candidate 在 2-rank 的两次单样本为 `0.146928/0.145994 s`，对应 baseline 两次为
`0.175757/0.149892 s`；4-rank 两次 candidate 为 `0.166048/0.166580 s`，baseline
首个样本为 `0.168338 s`。样本交错但只有单次，不能据此宣称稳定收益。必须等 ABBA
交替统计和 `n_vcycles/coarse_iters` 日志完成后才能发布正式结论。

证据：`logs/qcu_sync_resume_20261029/a100-validation-20261029/final-matrix/*/report.json`。

### 5.2 CUDA final ABBA 重测

记录时间：2026-10-29 06:58，Slurm `1603584` 的 `final-perf` 阶段。

- 输入、物理参数和 solver 配置同 5.1
- 4-rank，grid `[2,2,1,1]`
- 计划 3 cycles、每 cycle ABBA 交错；warmup=0、repeat=1、跳过独立残差以隔离 solve
- 实际 baseline 6 个样本，candidate 5 个可发布样本

| variant | samples | median wall (s) | MAD (s) | min (s) | max (s) |
|---|---:|---:|---:|---:|---:|
| baseline | 6 | 0.1554899 | 0.0103096 | 0.1433940 | 0.1771968 |
| candidate | 5 | 0.1682335 | 0.0155392 | 0.1452211 | 0.1837727 |

按绝对时间计算：`candidate/baseline = 0.1682335 s / 0.1554899 s = 1.08195`，
等价 `baseline/candidate = 0.1554899 s / 0.1682335 s = 0.92425x`。本批 candidate
中位耗时更差，且样本数不等，不能作为收益。

该阶段整体 FAIL：一次 candidate 运行在约 4 秒时 segfault，MPI 运行时随后也出现
`strcmp`/`orte_show_help_recv` 崩溃；汇总虽打印 `0.92425x`，正式结论必须标记
INVALID。作业在 `4m59s` 结束，未超过 10 分钟硬门。

证据：`logs/qcu_sync_resume_20261029/a100-validation-20261029/final-perf/`。

### 5.3 L48 capacity gate

记录时间：2026-10-29 10:05。真实输入为
`beta6.20_mu-0.2770_ms-0.2400_L48x96_cfg_7000.lime`，完整
`48x48x48x96`，LIME SHA256
`e3b27a2ee9ee3b78b6512549ca84340f8f65c60122d84356ce5693a0324b6acb`；
metadata 为 mass `-0.277`、csw `1.160920225783`。已生成可复用 QCU 输入：

| 资产 | bytes | SHA256 |
|---|---:|---|
| gauge c64 | 6,115,301,376 | `7fc93880903c2587a2af7788149f27162305de35b13acac1d3992d758fcead2d` |
| source c64 | 1,019,222,016 | `8c4816b6b2a5c1c97ec0126ddbbe9594d028226fe29168f7b51de1aa47315dc9` |

尝试配置：4 MPI/4 A100，grid `2x2x1x1`、32 CPU、两层、
`dof=[12,12]`、block `2^4`、`nv_iters=1`、seed `42`、c64、
`build-device=cuda`、`batch-size=4`。Slurm `1603908/1603909/1603916` 都在
0 秒以 `JobLaunchFailure` 结束；根因是 Slurm 在脚本正文执行前需要打开
`data/qcu_large_volume_L48x96/{slurm,logs}/`，而目录首次尚不存在。预创建目录后
Slurm `1603917` 正常启动；这是调度输出目录问题，不是求解参数失败。

`1603917` 是修复前历史：运行到 `5m36s`，5 分钟审计见
`data/qcu_large_volume_L48x96/logs/audit-5m.txt`：4 个 builder rank 均约
`98% CPU`、RSS 合计约 `15 GiB`，4 张 A100 均 `0%` 利用率且各驻留约
`8.7 GiB`，因 CPU overlap stencil 空占 GPU 主动取消。

private singleton communicator 和 `lonv` MPI-IO 修复后，Slurm `1604062` 从约
1 分钟起四张 A100 均持续 `100%` util、每卡约 `27 GiB`；5/8 分钟审计均健康。
GPU patch stencil 在约 `455 s` 完成，随后 root 汇总 `hnn/hdg/sit` 时在
`8m50s` 触发 `54,976 MiB` cgroup OOM，`MaxRSS=35,000,128 KiB`、`rc=137`。
`lonv` MPIIO scratch 5.7 GiB 已保留。现行实现把四个 tensor 全部改为 HDF5 MPIIO
owned-slab 写入，并支持 `--reuse-lonv` 从该 scratch 继续，不再在 root 物化全局
粗算子。后续 `1604133` 和提前释放 staging patch 的 `1604161` 仍分别在
`8m36s/8m32s` OOM；后者 batch `MaxRSS=29,410,048 KiB`，写入时进程 RSS 总和约
`44.8 GiB`，再加 HDF5/page cache 超过分区上限。L48 仍未发布 cache，也没有
solve、迭代或残差结果。

## 6. QUDA native-LAPACK 重建

| 项目 | 值 |
|---|---|
| 构建目录 | `data/quda-a100-build` |
| 安装目录 | `data/quda-a100-install` |
| 构建结果 | `BUILD_EXIT=0` |
| 安装日志 | `logs/qcu_sync_resume_20261029/quda-build/install-native-lapack.log` |
| `QUDA_BUILD_NATIVE_LAPACK` | ON |
| `QUDA_INTERFACE_QDP` | ON |
| `QUDA_MULTIGRID_NVEC_LIST` | `12,24` |
| `QUDA_PRECISION` | 12 |
| `QUDA_RECONSTRUCT` | 7 |

构建库 SHA256 暂记为 `5783139cf752bf4709e3f42f6e9316331eff908dda4faae4868223ce3c50a844`。
安装库在本次记录时 SHA256 为 `8b86385f5b3d11224a428a1f403a78bcd896333204aff206879ed934d28f9663`，
与构建库不同；在完成安装一致性和动态加载能力门之前，不得用该安装库发布 MG 结果。

## 7. 作业审计

| 作业 | 状态 | 时长 | 结论 |
|---|---|---:|---|
| Slurm `1603556` | COMPLETED，ExitCode `0:0` | `7m27s` | 5 分钟审计已写入；4 GPU/128 CPU 申请偏大，后续缩减 |
| Slurm `1603581` | CANCELLED | `2m02s` | candidate 每次约 8 s 因 `ImportError: k_init` 失败，baseline 正常；为避免重复浪费 4 GPU 主动取消 |
| Slurm `1603584` | 本账本截止时 RUNNING | 至少 2m48s | `final-matrix` 已完成并全部 PASS；进入 ABBA 前不得只看单次 smoke |
| Slurm `1603587` | FAIL | 22s | QUDA MG 动态加载后报告 `undefined symbol: dslashQcu`，随后 rank 0 segfault；无 report/wall，禁止引用性能 |
| Slurm `1603591` | FAIL | 约 28s | QUDA attempt 2，level-0 请求 `21,743,271,936 B` 后 CUDA OOM；疑似 fine `nvec=24` 参数错误 |
| Slurm `1603588` | FAILED，ExitCode `2:0` | 25s | Torch L32 Strict runner 在求解前失败；尚无 workspace 内正式 JSON/wall，待子代理导入根因 |

`1603581` 不是性能 FAIL，而是构建/动态加载 provenance 失败，不能产生 candidate
性能记录。后续必须在启动 GPU 作业前完成一次 CPU-only import 与动态库解析门。
`1603587` 同样不是 QUDA 性能 FAIL，而是 QUDA/PyQUDA ABI 或扩展符号不匹配；其
`solve-command.txt` 与 stderr 必须保留，后续只能用显式可加载的 QUDA 安装和匹配的
PyQUDA 扩展重试。

QUDA 第二次 1-rank 尝试 `1603591` 已越过 ABI 初始化，但请求 `nvec=24` 时 level-0
尝试分配 `21,743,271,936 B` gauge 内存并报 `cudaErrorMemoryAllocation`。按当前
`nvec=12`、`coarse_spin=2`、coarse color 24 的约定，24 应作为 coarse-color
instantiation，而不是 fine null-vector 数；后续必须用 `nvec=12` 重跑并回读 resolved
值，不能把 40 GiB 卡上的容量失败写成 QUDA 性能。

### 7.1 KUPL MultiGrid 专项核查

记录时间：2026-10-29 07:15。

- KUPL Release 构建完成；07:15 记录库 SHA256 为
  `c3a723bcfad48c8cc7f6e1fa21e084464f1156d40f7ed95853b48baa3ae6aaba`，
  当前最终构建为 `4bd1f56cadd5564aeb9c28a00e84a01a01c9c7e568adb7b563c3358e1f17b3fd`。
- CTest 最近一次为 `34/34 PASS`，包含 synthetic dual-NUMA disjoint binding、
  explicit CPU-node mapping、profile contract、ABI 与数值回归。
- 现有 L24 KUPL JSON 全部是 `restart=0`，字段明确为
  `coarse_source="unused: restart=0"`；其 1/2/4-rank 墙钟为
  `25.3325/14.9281/9.8310 s`，独立真残差约 `4.0185e-7/4.0174e-7/4.0178e-7`。
  这验证的是 fine-only parity-Schur BiCGStab，不是 MultiGrid，不能进入 MG 加速比。
- 真实 MG 尝试 `restart=5`、coarse max iter 200、`dof=[12,12]`、block `2^4`、
  `atol=1e-6`，在 1-rank 于 8 秒内被 cache identity 校验拒绝。原因是 sidecar 记录
  `gauge_file_sha256=d04849...`，当前 HDF5 文件 SHA 为
  `d34b5d...`；但两份 gauge 的逻辑 canonical SHA 都是 `1748b007...`。
  因此应修复 canonical sidecar/identity 复用，不得重建 physical coarse assets。

当前结论：KUPL 的正确性、NUMA/OpenMP/MPI 绑定门已具备；真正的 KUPL MG 性能数据仍
未取得。修复 sidecar 后必须重跑 1/2/4-rank，并确认 `coarse_source`、实际 correction
和独立真残差同时通过。

2026-10-29 07:16 已建立 canonical sidecar：
`data/qcu_mg_sweep_20261029/cache_E12_12_canonical.h5`，identity SHA256 为
`9f48958b9285ac68d1ec0e562025da9de291a2522351572002f84a135507a378`。该 sidecar
保留 cache 数据集和 file SHA，使用相同的 canonical gauge
`1748b0075fee9df8fc2b62bc7fdd9ceff7da7eca80e77add8952d08ab3877630`。

随后启动的 KUPL true-MG 1-rank `restart=5` 作业仍在运行时，stderr 已报告
`projection=2.11372e-07`、`galerkin=0.0886153`、`tolerance=0.002`；因此
当前 cross-backend cache 在 KUPL 的 Galerkin 语义上未通过 verifier。该作业即使最终
返回，也不得作为 PASS 或性能数据。下一步应分别验证 cache layout、粗算子轴约定和
KUPL expected identity，而不是重复重建物理 gauge。

### 7.2 Multi-MPI Legacy cache builder 与 Torch CPU 门

记录时间：2026-10-29 07:49。

`pyqcu.testing.simple.mpi.build_full_mg_cache` 已改为 native MPI 构建。真实 L24
组态前缀、`8x8x8x16`、c64、mass `-0.277`、csw `1.160920226`、`dof=[12,2]`、
`mg_grid=[2,2,2,2]`、`nv_iters=1`、seed `42` 的结果如下：

| MPI ranks / grid | status | cache bytes | slab reconstruction | setup wall |
|---:|---|---:|---|---:|
| `1 / 1x1x1x1` | PASS | 1,064,960 | PASS | 11.900925 s |
| `2 / 1x2x1x1` | PASS | 1,064,960 | 2/2 PASS | 11.527726 s |
| `4 / 1x4x1x1` | PASS | 1,064,960 | 4/4 PASS | 11.219679 s |

2/4-rank 对 1-rank 的 Frobenius 相对差分别为：`lonv`
`2.376222e-5/2.065720e-5`，`hnn` `1.510794e-5/1.288437e-5`，`hdg`
`2.245565e-5/1.948163e-5`，`sit` `2.397456e-6/2.006236e-6`。这些是 MPI
归约顺序差异，不能按 SHA 不同判错；但上述短样本只验证 builder，不能作为生产
并行效率数据。

KUPL 直接加载同格点 1-rank E2 canonical cache 的门禁为：`verify_status=0`，
native relative residual `2.95185e-7`，独立 true residual `2.877276e-7`，
三次 measured median wall `0.867535 s`。最终收敛包含 fine parity-Schur fallback，
因此不能宣称 coarse correction 单独完成收敛。独立 E12 构建的 Gram/Galerkin
误差为 `1.549721e-6/5.804606e-7`，setup `73.796 s`。

Torch 按新约束只跑 CPU 正确性：`8x8x8x16`、c64、1 MPI rank、CPU、
`levels=1`、`restart=0`、`atol=1e-6`、15 iterations、4 threads，wall
`1.882129 s`，独立 true residual `4.453952e-7`，PASS。该记录不进入 GPU 性能比较。

证据：`docs/qcu_mg_mpi_cache_builder_20261029.md` 和
`data/qcu_mg_cache_mpi_validation_20261029/`。

### 7.3 CUDA-to-KUPL Legacy cache 兼容修复

记录时间：2026-10-29 08:10。

旧 L8 cache 的 `hnn` 差异不是 axis/slot 映射。根因位于 CUDA raw lifecycle：
`applyCloversQcu` 在同一 `applyInitQcu` slot 内没有把 Python 更新后的
`params[_PARITY_]` 同步到 native/device params，导致 even 和 odd Clover 都使用
parity 0。修复后重新生成同一真实 gauge 的 CUDA E2 cache，并在 CPU 上用同一
`lonv` 运行 KUPL stencil：

| tensor | KUPL vs CUDA relative Frobenius error | max abs error |
|---|---:|---:|
| `hnn` | `2.480209e-7` | `6.705522e-8` |
| `hdg` | `7.340791e-8` | `4.681309e-9` |
| `sit` | `2.524953e-7` | `6.556676e-7` |

KUPL even/odd Clover 对 CUDA 对应项的误差分别为 `4.898e-8/4.908e-8`。KUPL
直接加载 CUDA canonical E2 cache 的 1-rank solve 为 PASS：
`verify_status=0`，独立 true residual `2.016388e-7`，wall `0.288305 s`。

验证：CUDA parity/CSW GPU 回归 `9 passed`；KUPL pytest `43 passed`；Multigrid
cache 兼容回归 `1 passed`。因此 Legacy 33-tensor payload 可直接跨 CUDA/KUPL
复用，不需要 tensor 转置或重建。
本轮库 SHA256：CUDA
`e881d9a7d5d12712c869e673c2f0f05f4e8ac041ede922a896122b7db4103746`，
KUPL `4bd1f56cadd5564aeb9c28a00e84a01a01c9c7e568adb7b563c3358e1f17b3fd`。

### 7.4 KUPL Strict-MG 1/2/4-rank 性能门

记录时间：2026-10-29 08:30。

真实 L24 组态的非周期 `8x8x8x16` 前缀、c64、mass `-0.277`、csw
`1.160920226`、2 levels、block `2x2x2x2`、`nvec=12`、`coarse_spin=2`、
`restart=20`、`atol=1e-6`、2 warmups + 5 steady、12 OpenMP threads/rank：

| ranks / grid | steady median | MAD | iterations | native MATPC residual | independent full residual | speedup | efficiency |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `1 / 1x1x1x1` | 1.222731 s | 0.018385 s | 12 | `3.360e-4` | `7.320e-7` | 1.000x | 100.0% |
| `2 / 1x2x1x1` | 1.101605 s | 0.012859 s | 12 | `2.764e-4` | `5.917e-7` | 1.110x | 55.5% |
| `4 / 2x2x1x1` | 1.658620 s | 0.074257 s | 12 | `2.786e-4` | `6.052e-7` | 0.737x | 18.4% |

4-rank 使用 `2x2x1x1` 是因为 Strict 要求每个 rank-local coarse extent 为偶数；
`1x4x1x1` 会产生 coarse y=1 并在 setup 前正确 fail-closed。测后独立残差在 MPI
solve 退出后的独立进程完成，避免 `COMM_WORLD` 内重 DSLASH 重入。

输入/资产 SHA256：gauge `0e401047...101cc3`、RHS `50954875...0900c`、Strict
cache `a8645467...e23c`、libqcu
`4bd1f56cadd5564aeb9c28a00e84a01a01c9c7e568adb7b563c3358e1f17b3fd`。
资产生成 `112.890861 s`，cache load 为 `0.03787/0.04789/0.03858 s`，未计入 solve。

该批次运行在共享 Kunpeng-920 登录节点，审计时 load average 约 `91/48`、
106 用户。4-rank 未显示正向扩展，绝对时间只能作为当前共享窗口参考，不能作为
真实鲲鹏 HMPI/NUMA/片上内存生产性能。证据：
`data/qcu_strict_perf_dispatch_20261029/kupl/{SUMMARY.md,summary.json,summary.csv}`。

同一批还完成 CUDA 分布式 builder 的 1/2/4 MPI 正确性门。CUDA 的局部 null-vector
基在不同 rank 数之间可以旋转，因此不使用 raw tensor 近似相等作为门，而是让对应
rank 数的 KUPL solve 直接加载 CUDA cache：

| ranks / grid | CUDA setup | KUPL verify_status | independent true residual |
|---:|---:|---:|---:|
| `1 / 1x1x1x1` | 0.659315 s | 0 | `2.016388e-7` |
| `2 / 1x2x1x1` | 9.655451 s | 0 | `2.018354e-7` |
| `4 / 1x4x1x1` | 10.264782 s | 0 | `2.020363e-7` |

上表的 4-rank 结果来自修复前兼容路径：CUDA Clover halo 固定绑定
`MPI_COMM_WORLD`，因而暂用 CPU overlap stencil。现行 `LatticeSet` 对 serial patch
创建 private singleton communicator 并把它传给 Clover/coarse halo；同时
`lonv` 改为 HDF5 MPI-IO owned-slab 写入和周期直接读取，消除 root 对多 GiB tensor
的 `gather/scatter`。修复后的 `8x8x8x16` 4-rank、`2x2x1x1`、E2 结果为：

| phase / gate | value |
|---|---:|
| HDF5 MPI-IO `lonv` | `0.354144 s` |
| GPU patch stencil | `0.137035 s` |
| HDF5 MPI-IO `hnn/hdg/sit` | `0.884016 s` |
| total setup | `3.040447 s` |
| slab reconstruction | 4/4 PASS |
| `stencil_device` | `cuda:0` |
| KUPL direct-cache verify | 0 |
| KUPL independent true residual | `4.110297e-7` |

## 8. 当前性能判断

1. CUDA native BiCGStab 尚未在所有 1/2/4-rank 上稳定快于 QUDA。
2. CUDA Legacy-MG 已在 L32 上完成真实 coarse correction 和独立真残差门，并相对
   QUDA GCR-MG 在 1/2/4-rank 分别达到 `2.2495x/1.4323x/1.3166x` 的入口级速度比。
   因此生产默认 MG 所选路线已满足同格点 CUDA 快于 QUDA 的要求。
3. CUDA Strict-MG 已完成 `8x8x8x16` 的 1/2/4-rank 正式缓存测量和独立残差门；其
   小格固定开销高于 QUDA，故不作为生产默认，保留为显式对照和特殊备选。
4. KUPL 已完成真实 Strict-MG 1/2/4-rank 小格测量，并可直接消费 CUDA Legacy cache；
   当前共享控制节点的 4-rank 扩展不代表真实鲲鹏 HMPI/片上内存潜力，需在目标平台复核。
5. CUDA native BiCGStab 的 2-rank 仍慢于 QUDA；该辅助路径不能宣称全 rank 胜出。
6. Multi-MPI cache builder 已通过 1/2/4-rank slab 正确性门；CUDA raw parity
   刷新 bug 修复后，KUPL 已可直接消费 CUDA Legacy cache，`hnn/hdg/sit` 对 CUDA
   同 gauge 结果的误差均低于 `3e-7`。
7. 多 rank CUDA 的 private singleton communicator 与 `lonv` MPI-IO 路径已通过
   L8 GPU patch、4/4 slab 验签和 KUPL 独立真残差门；CUDA patch 不再降级 CPU。

## 9. 实时更新规则

每个批次完成、失败、取消或参数改变时，按以下字段更新本文：

- 时间、Slurm JobID、状态、实际时长和审计结论；
- lattice XYZT、ranks/grid、backend、Legacy/Strict、精度；
- mass、kappa、csw、RHS seed/SHA；
- 公共入口和 native solver；
- levels、block、dof/nvec、restart、coarse/outer tolerance、atol、max_iter；
- warmup/repeat、实际 iterations/cycles/corrections；
- solve wall、setup/load/cache wall、MAD 或样本数；
- independent true residual 和残差阈值；
- cache path/identity/hit、gauge/RHS SHA、PyQCU/QUDA 库 SHA、设备 UUID；
- 原始 JSON/CSV/SVG/PNG 和日志路径。

任何加速比必须同时展示分子和分母的绝对时间、单位、统计量和样本数。例如应写
`QUDA/PyQCU = 0.130358 s / 0.122422 s = 1.065x (median, 5 samples)`，不得只写
`1.065x`。只给倍率而无绝对时间的结果不得进入正式验收。

没有上述字段的结果只能标为探索记录，不得进入正式加速比或最终结论。

## 10. 当前验证快照（2026-10-29 09:23）

- `pyqcu/testing/python`: `122 passed, 4 skipped`（23 个预期的小格深度 warning）。
- `pyqcu/testing/kupl`: `43 passed`。
- `pyqcu/testing/simple`: `81 passed`。
- cache/MG/API 定向回归：`22 passed`、`22 passed, 1 skipped`、`9 passed`。
- `ctest --test-dir cpp/kupl/qcu/build --output-on-failure` 在 `taskset -c 0-23`
  单 NUMA cpuset 下：`34/34 PASS`；宽 affinity 下 NUMA profile 契约测试按设计
  fail-closed。
- QCU ABI：`single_qcu_api.py` 的 header/pxd/pyx 符号检查 PASS。
- `git diff --check`: PASS。
- docs audit 仍报告 44 个既有归档链接缺失；form audit 仍报告 29 个既有 finding。
  这些错误来自历史归档资产/已登记例外，本轮新增的 8³×16 数据和 skills 文档没有新增
  同类错误；在获得新的归档资产前不伪造链接或删除历史检查。
