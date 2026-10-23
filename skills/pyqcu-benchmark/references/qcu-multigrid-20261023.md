# QCU MultiGrid 基准与调试基线（2026-10-23）

本文供后续 agent 复现 PyQCU/QCU MultiGrid 的快速矩阵、分布式验收和故障归因。
执行前先区分 `事实`、`项目决策`、`未解决`；不得把未完成初始化、fine-only 路径或
native residual 单独包装成 coarse-correctness 或性能结论。

## 事实

以下结论均有本轮运行记录或当前入口契约支持。

- MPI benchmark 的有效 rank 数为 1/2/4/8 等 2 的正整数次幂。`simple/mpi` 的
  flat example wrapper 会在启动 `solve_case.py` 前调用 rank validator；3/6/12 等
  rank 数应 fail-closed，不能进入 solver 后再解释结果。
- CUDA 生产拓扑仍是一 Python 进程 = 一 MPI rank = 一设备。2-rank CUDA 可以使用
  两张 P100；P100 运行环境必须兼容 `sm_60`，原生库集合必须同时覆盖 `sm_60`/`sm_70`。
  V100+P100 异构组合会混淆性能归因，不可作为正式同窗结果。
- `24×24×24×64` c64、`nvec=12`、block `2^4` 的单 rank CUDA Strict 两层初始化从
  `03:07:22` 运行到 `03:41:12`，约 `2030 s`（33m50s），仍未写出 stdout 或 side
  JSON。该值是初始化耗时下界，不是完成的 setup 时间，也不是数值 PASS/FAIL。
- 旧 2-rank 三层/1-rank 四层失败已定位为 CUDA Legacy 递归 V-cycle 返回父层时错误恢复
  最细层 volume。修复后 `24×24×24×64` 单 rank 3L/4L 与 2-rank 3L/4L 的独立 full
  residual 均约 `2.45e-7–2.61e-7` 并 PASS；层数增加没有迭代收益，3L/4L 每次 solve
  约 `0.89/0.92 s`（单 rank），仍不得当作默认生产深度。
- 多 rank CUDA Legacy 必须显式设置 `params[_MG_USE_DEFLATE_]`。未设置时 coarse
  correction 未请求，`n_vcycles=0`，结果只是 fine-only；不能写成 coarse MG 性能。
- 配置的 `levels`、`restart`、cycle 类型都不是实际 coarse 执行证据。必须读取
  `n_vcycles`、`coarse_iters`/`coarse_iterations`、cycle 或 correction 计数，并确认
  实际到达了哪一层。
- `MultigridDepthWarning` 同时考虑层级深度、process grid 后的 rank-local geometry、
  parity-compressed T、coarsening block 整除性、最粗层体量以及是否还能构造下一层。
  警告是可行性提示，不是正确性门；没有警告也不等于真残差会通过。
- Legacy correction 使用确定性的 post-residual 回滚：校正变差时撤销并关闭本 solve
  后续 correction。CUDA/Torch 在 near-tolerance 带内逐迭代执行 full gate；KUPL 在
  correction 被拒绝后显式 fallback 到已验证的 parity-Schur BiCGStab 并报告总迭代。
- 长任务可能长时间没有阶段性 stdout。本轮容器为 `ptrace_scope=1`，`py-spy`/`gdb`
  无法附加到运行进程；因此持续 100% GPU 利用率只能说明设备忙碌，不能证明 setup
  或 solver 健康。

## 项目决策

- MPI benchmark 只接受 2 的正整数次幂个 rank。每个新启动器必须在启动 MPI/solver
  前完成 rank preflight，并以非零状态 fail-closed。
- CUDA 生产路径继续采用一进程、一 rank、一设备。2-rank 的 P100 验证使用独立的
  `sm_60` 兼容 Python 环境与包含 `sm_60`/`sm_70` 的原生库集合；V100+P100 混合拓扑
  只允许作为排障记录，不进入性能结论。不得通过重复同一个可见 GPU UUID 伪造多设备。
- `Legacy-MultiGrid` 是默认开发、调试和优化对象；`Strict-MultiGrid` 必须通过
  `hierarchy_mode="strict"` 或直接使用 `StrictMultigrid` 显式启用。
- 六项快速矩阵固定为 Torch/CUDA/KUPL × Legacy/Strict，即 `TL/TS/CL/CS/KL/KS` 六行。
  合格的批次还必须包含同一输入下的 QUDA aligned MG 对照，以及 Torch/CUDA/KUPL/QUDA
  各侧 BiCGStab 边界对照。BiCGStab 不属于 MG 加速比分子或分母。
- 快速矩阵只在 `16×16×16×16`（`16^4`）运行；该格点只用于六项 QCU-MG 快速矩阵。
  当前编排器若仍以历史默认格点启动，必须显式覆盖，不能把历史默认值当成本决策。
- 快速矩阵的计时只使用 `trace-off`。trace-on 只允许作为单侧诊断，不能进入正式
  speedup 或 matrix median。
- QUDA 全程设置 `QUDA_ENABLE_TUNING=0`，并在进程启动前设置环境变量。运行时能回读
  tune 状态时应同时验证；不能只靠命令行意图。
- 大格 `24×24×24×64` 和 `32×32×32×64` 不再测试 Strict 行。Strict 行标记为项目
  skip，而不是失败、容量 PASS 或性能结果；显式 Strict 正确性/ABI 调试不受此 skip
  限制。
- Legacy 验证深度为 2，Strict 验证深度为 3。超过验证深度可以用于显式诊断，但不得
  自动升级为生产支持路径；分布式 Legacy 两层以上必须先通过独立真残差门。
- 每个长 benchmark 阶段必须输出时间戳/阶段进度并设置阶段超时或总超时。超过阶段预算
  且无阶段进展时应停止并保存诊断，而不是仅凭 GPU util 继续等待。
- 单个测试组的总耗时达到 `1000 s` 时，可以停止并标记
  `SKIPPED: elapsed_time_unacceptable`。必须记录 timeout、退出码、最后完成阶段以及是否
  发布计时样本；不得把 setup 超时解释为数值失败或完成性能结果。
- `pyqcu/testing/simple/mpi` 的示例保持扁平文件布局，不创建 `examples/` 子目录。
  示例必须预检 2 的幂 rank 数，并支持 `DRY_RUN=1` 只打印解析后的命令。

## 快速矩阵与对照

六项主矩阵及其职责：

| ID | Backend | Variant | 用途 |
|---|---|---|---|
| TL | Torch | Legacy | Python/reference Legacy 路径 |
| TS | Torch | Strict | Python/reference Strict 路径 |
| CL | CUDA | Legacy | CUDA 默认开发路径 |
| CS | CUDA | Strict | CUDA 显式 full-coarse 对照 |
| KL | KUPL | Legacy | CPU MPI/OpenMP Legacy 路径 |
| KS | KUPL | Strict | CPU MPI/OpenMP Strict 路径 |

主矩阵之外至少保留以下边界记录，且都使用同 gauge、source、precision、tolerance、
max-iter、levels、block 和 nvec 中适用于该 side 的字段：

- QUDA aligned MG：作为 MG 算法对照，不把不同 smoother/coarse budget 混成一个倍率。
- Torch、CUDA、KUPL、QUDA BiCGStab：说明 MG 是否真的优于普通 Krylov，不参与
  MG-vs-MG speedup 计算。
- CUDA Legacy fine-only：只作为同一输入的算法消融；`n_vcycles=0` 时必须明确标为
  fine-only，不能补成第七项 coarse MG。

只有所有主行和必要对照均完成、独立真残差通过、实际 coarse 执行计数非零且配置一致时，
才能称该批次为完整快速矩阵。缺少 BiCGStab 对照的批次只能报告 MG 巡检，不能宣称完整的
solver 快慢结论。

### 2026-10-23 `16^4` 正式批次

输入为 c64、`mass=0.05`、`csw=1.0`、tol `1e-6`、2 levels、block `2^4`、
`nvec=12`，全部 trace-off 且 `QUDA_ENABLE_TUNING=0`。计时为 1 warmup + 3 steady
的中位数；CPU 行为参考值。

| ID | iterations | steady median | independent full residual | status |
|---|---:|---:|---:|---|
| Torch Legacy GPU | 79 | `2.5032 s` | `7.406e-7` | PASS |
| CUDA Legacy | 87 | `0.0564 s` | `3.051e-7` | PASS |
| KUPL Legacy | 73（5 MG + 68 fallback） | `4.0008 s`（MAD 较高） | `3.362e-7` | PASS |
| Torch Strict GPU | 23 | `12.4956 s` | `6.577e-7` | PASS |
| CUDA Strict | 23 | `0.1950 s` | `6.543e-7` | PASS |
| KUPL Strict | 23 | `13.9101 s` | `6.535e-7` | PASS |
| QUDA aligned MG | 12 | `0.4041 s` | `4.744e-7`（独立检查另有记录） | PASS |

Legacy 解向量相对 Torch 参考：CUDA `8.07e-6`、KUPL `8.06e-6`；Strict 解差：
CUDA `2.82e-7`、KUPL `3.35e-7`。因此 Legacy 迭代计数虽因 c64 Krylov 舍入路径相差
最多 14 次，解仍处于同一 `1e-6` 残差邻域。正式 parity 门采用 Legacy 相对迭代差
`<=20%`（另至少允许 2 次）加解向量 `relative L2 <=1e-4`；Strict 采用绝对差 `<=2`。
该门来自实测数值可复现性，不代替正确性门。

BiCGStab 边界对照（同输入、trace-off）：Torch `67 it / 1.595 s / 7.245e-7`，
CUDA `68 it / 0.0441 s / 4.175e-7`，KUPL 4-rank `68 it / 1.731 s / 3.276e-7`，
QUDA `70 it / 0.1059 s / 4.726e-7`。它们不进入 MG 加速比。

### 2026-10-23 L32 Legacy 正式批次

输入为 `beta6.20_mu-0.2770_ms-0.2400_L32x64_cfg_11000`、`32x32x32x64` c64、
`mass=-0.277`、`csw=1.160920226`、tol `1e-6`、2 levels、block `2^4`、
`nvec=12`。gauge/source SHA256 为 `5054e60d...29a4f3` /
`e1e86130...ac100`。全部 trace-off，QUDA tuning 关闭。

CUDA/Torch 使用同一 odd-Schur cache，SHA256
`ec17e56c...d82fd`，coarse shape `[16,16,16,16]`，大小 `3,699,380,224 B`。
builder 使用 `local_stencil_batch=1`、`null_batch_chunk=1`，耗时 `1202.07 s`。
KUPL 使用独立 full-time cache，SHA256 `86173c8f...7cef`；不得互换或按文件大小推断兼容。

| backend | ranks/device | iterations | solve seconds | independent full residual | status |
|---|---:|---:|---:|---:|---|
| Torch Legacy | 1 / V100 | 19 | `17.4902` | `3.863e-7` | PASS |
| CUDA Legacy | 1 / V100 | 20 | `0.4592` | `2.808e-7` | PASS |
| CUDA Legacy | 2 / 2x P100 | n/a | `1.7760` | `2.668e-7` | PASS |
| KUPL Legacy | 1 / CPU | n/a | `36.2029` | `3.424e-7` | PASS |
| KUPL Legacy | 2 / CPU | n/a | `36.8231` | `3.422e-7` | PASS |

同侧 BiCGStab：Torch `18 it / 14.085 s / 7.172e-7`、CUDA
`18 it / 0.2767 s / 6.183e-7`、KUPL 2-rank
`19 it / 17.738 s / 3.424e-7`、QUDA `17 it / 0.2696 s / 8.644e-7`。
本批次 MG 均未优于同侧 BiCGStab。Torch MG 的设备峰值达到完整 `32 GiB`，属于容量边界。

大格 Strict 跳过。QUDA MG `nvec=12` 因 sm70 构建的 coarse `Nc=24` Yhat 内核不受支持；
`nvec=24 -> Nc=48` 在 `1000 s` 处超时，未产生计时行。两个 QUDA MG 尝试都只能记为 skip，
不能与 `nvec=12` 计算公平倍率。

## 正确性门

Legacy 多级结果不能只检查 C++ native residual。求解返回后必须独立重建 full
Wilson/Clover true residual，并在多 rank 情形下执行全局范数归约。建议至少同时记录：

```text
native_true_residual
independent_true_residual
residual_acceptance
levels / process_grid / local_lattice
n_vcycles / coarse_iterations / correction_count
```

修复后 `24×24×24×64` c64 Legacy 证据：

| 拓扑 / 层数 | native residual | independent true residual | 状态 |
|---|---:|---:|---|
| 2 rank / 2 level | `2.342344e-7` | `2.453464e-7` | PASS |
| 2 rank / 3 level | `2.342240e-7` | `2.453817e-7` | PASS |
| 2 rank / 4 level | `2.342462e-7` | `2.453665e-7` | PASS |
| 1 rank / 2 level | `2.336648e-7` | `2.604745e-7` | PASS |
| 1 rank / 3 level | `2.335826e-7` | `2.605036e-7` | PASS |
| 1 rank / 4 level | `2.336648e-7` | `2.604566e-7` | PASS |

## 未解决

- 大格 CUDA Strict 两层初始化在约 33m50s 后仍未完成，根因尚未形成完成态证据。当前
  决策只是停止大格 Strict benchmark；不能推断为数值失败、永久不可运行或性能倍率。
- 本次无输出长任务无法用 `py-spy`/`gdb` 附加到进程。后续应先补阶段 heartbeat、
  per-phase timeout 和可导出的 C++ 进度的可观测性，再讨论更长的 Strict 尝试。

## 证据索引

- 决策文档：[`docs/qcu_mg_benchmark_decisions_20261023.md`](../../../docs/qcu_mg_benchmark_decisions_20261023.md)
- Strict skip：[`strict-skip.json`](../../../logs/qcu_l24x64_mpi_20261023/strict-skip.json)
- 2-rank 2-level PASS：[`cuda-legacy-2l-deflate.json`](../../../data/logs/qcu_l24x64_mpi_20261023/multi2/cuda-legacy-2l-deflate.json)
- 修复后单 rank 2L/3L/4L：[`qcu_mg_final_20261023`](../../../logs/qcu_mg_final_20261023)
- 修复后 2-rank 2L/3L/4L：[`qcu_mg_final_20261023`](../../../data/logs/qcu_mg_final_20261023)
- 16^4 完整矩阵：[`summary.json`](../../../logs/qcu_mg_matrix_16cubed_20261023/summary.json)
- 跨实现 parity：[`summary.json`](../../../logs/qcu_16cubed_quick_20261023/parity/summary.json)
- BiCGStab 对照：[`qcu_bicgstab_final_20261023`](../../../logs/qcu_bicgstab_final_20261023)
- L32 汇总：[`summary.json`](../../../logs/qcu_l32x64_mpi_20261023/summary.json)
- L32 odd cache build：[`legacy-cache-odd-schur-2l-build.json`](../../../logs/qcu_l32x64_mpi_20261023/legacy-cache-odd-schur-2l-build.json)
- L32 BiCGStab：[`summary.json`](../../../logs/qcu_l32x64_mpi_20261023/bicgstab_three_way/summary.json)
- QUDA MG status：[`quda-multigrid-status.json`](../../../logs/qcu_l32x64_mpi_20261023/quda-multigrid-status.json)
- rank validator：[`mpi_ranks_power_of_two.py`](../../../pyqcu/testing/simple/mpi/mpi_ranks_power_of_two.py)

引用 JSON 时只读取上述字段和对应 side record；不要复制大段原始 JSON，也不要把
工作区暂存报告当作长期规范归档。
