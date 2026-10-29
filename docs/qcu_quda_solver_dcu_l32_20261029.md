# QCU Legacy-MultiGrid 与 QUDA DCU L32 全尺寸重测（2026-10-29）

> 状态：`BLOCKED-CACHE`；16-batch、resume-32、recovery-64 和 16-rank 均超时，
> recovery-128 因 native stencil status 1 失败；32-rank 恢复作业 `126271962`
> 已由用户终止，L32 构建暂停；最后更新 2026-10-30 02:24 CST。  
> 范围：`32x32x32x64`、complex64、真实 LIME；cache 恢复覆盖 1--32 ranks，
> 目标性能测试为 1/2/4 ranks；partition 为 `kshdnormal`。  
> 原则：旧 L32 cache 和性能数据全部失效；只有本轮 `build_full_mg_cache` cache 通过
> 认证后，性能数据才进入正式表。未做独立 residual 的结果只报告 native full residual。

## 1. 输入与 canonical case

从真实 LIME 全格转换完成，`sub_volume=false`；cache builder 不直接读取 LIME。

| 项目 | 结果 |
|---|---|
| LIME | `/public/home/sunpeng_1/data/beta6.20_mu-0.2770_ms-0.2400_L32x64_cfg_11000.lime` |
| lattice | `32x32x32x64` |
| mass / csw | `-0.277` / `1.160920226` |
| gauge | `data/qcu_solver_test34_20261029/L32/case/gauge_32x32x32x64_c64.h5`，shape `(3,3,4,32,32,32,64)`，`1,207,965,696 B`，SHA256 `0e1d2de249fef3124b5a3985446abac63f1bf33217e274abb9b5376ca143f433` |
| source | `data/qcu_solver_test34_20261029/L32/case/source_32x32x32x64_c64.h5`，shape `(4,3,32,32,32,64)`，`201,332,736 B`，SHA256 `3d9fefe243f9ab7c8880f28156fa257a803e924d950b68925e20903f3351beca` |
| manifest | `data/qcu_solver_test34_20261029/L32/case/case_manifest.json` |
| 准备耗时 | wall `36.21 s`，peak RSS `3,050,828 KiB` |

## 2. 构建 provenance

| 项目 | 结果 |
|---|---|
| `libqcu.so` | SHA256 `b513fbcd41e48caa5e9bc3acd2f23693c52b177ddd26efe65e0cb6f82ed068bf`，2026-10-29 22:25:41 CST |
| Cython `qcu*.so` | SHA256 `df204a1c5dcce7be6720e988fbc1534be5bfd1f2d5d92f11916b49f589aa9386`，2026-10-29 22:29:40 CST |
| QCU runtime | DTK 23.10；`PYQCU_MPI_DEVICE_AWARE=0` |
| QUDA runtime | DTK 25.04.4、`quda-mpi-gfx906`，独立子进程，不与 QCU 混载 |
| tuning / threading | `QUDA_ENABLE_TUNING=0`，`OMP_NUM_THREADS=1` |

## 3. DCU 硬件与运行环境

硬件证据由只读作业 `126271539` 在 `b16r1n16` 采集，完整输出见
`logs/qcu_solver_test34_20261029/dcu-hardware-audit.txt`。L32 16-rank 作业运行于
`e04r3n[15-18]`；Slurm 对这些节点统一报告 `dcu:Hygon:4`。

| 项目 | 实测值 |
|---|---|
| 加速卡 | Pre-Wukong DCU，Card SKU `D160A1`，rocminfo 名称 `ZIFANG`，Chip ID `0x66a1` |
| GPU ISA | `amdgcn-amd-amdhsa--gfx906:sramecc-:xnack-` |
| 计算资源 | 64 CU，每 CU 4 SIMD，wavefront 64，最高 1.7 GHz |
| 显存 | `17,163,091,968 B`；rocminfo `16,760,832 KiB`；rocm-smi `16,368 MiB` |
| VBIOS | `113-D160A1SG-F42` |
| 节点 CPU | Hygon C86 7185 32-core，4 NUMA node，最高 2.0 GHz，126,500 MiB |
| 节点拓扑 | 32 CPU、4 DCU；一个 Python 进程 = 一个 MPI rank = 一张 DCU |
| 系统/软件 | CentOS 7.6、Linux `3.10.0-957.el7.x86_64`；QCU DTK 23.10、HIP `5.4.23453`、HPC-X MPI 2.11.0 |
| QUDA 辅助栈 | DTK 25.04.4、QUDA 1.1.0 `d615172-dirty-gfx906`，独立进程加载 |

`rocm-smi` 的 `Driver version` 字段返回内核版本，不是 DTK 用户态驱动版本。硬件型号
只采用工具实际返回的 `Pre-Wukong DCU / D160A1 / ZIFANG`，不推测等价 AMD SKU。

## 4. Distributed cache

Slurm job `126250045`（`test34-l32-cache`）于 22:46 CST 启动：1 node、4 DCU、32 CPU、
10 min，4-rank `2x2x1x1`，命令为规定的 native MPI `build_full_mg_cache` 路径。

- 目标：`data/qcu_solver_test34_20261029/L32/cache_L32_E12_12_mg2x2x2x2.h5`
- report：`data/qcu_solver_test34_20261029/L32/cache_L32_E12_12_mg2x2x2x2.json`
- 脚本：`logs/qcu_solver_test34_20261029/L32/run_cache.sh`
- stdout/stderr/audit：`cache.stdout.txt`、`cache.stderr.txt`、`cache.audit.txt`

首次作业 `126250045` 于 22:55:49 以 `rc=124` 被 570 s 内部 watchdog 终止，未发布
`cache.h5` 或 report，故明确判为 `TIMEOUT`，不得使用。终止前 4 个 rank 的 C++ stencil
均报告 `gram=0`、`galerkin=0`，累计 stencil `431.4--431.9 s`；临时 HDF5 已完成
`lonv` 和 `hnn`，但缺少 `hdg/sit`、manifest 和 slab reconstruction，不能据此认证。

在相同 canonical gauge、相同 4-rank `2x2x1x1` builder 与相同几何下，提交作业
`126250791`（`test34-l32-cache-reuse`）。它使用 builder 官方
`--reuse-lonv` 与 `--reuse-existing-assets` 路径，从首次作业的
`.cache_L32_E12_12_mg2x2x2x2.32700-ffbc265740ce.partial.h5` 复用已完成的
`lonv/hnn`，重新执行 stencil、写齐 `hdg/sit`、manifest、4-rank slab 回读和 report。
该重跑在 5 min 审计时四卡均 100% 利用、约 7.1 GiB VRAM，但 stencil 本身为
`458.5--459.0 s`；加上前置 staging 与 hnn slab 校验后仍在 590 s 前未能写出
`hdg/sit`，于 22:57--23:07 以 `rc=124` 终止。证据前缀为 `cache-reuse.*`。

为保持在单作业 10 min 内完成，提交第三次、也是最后一次恢复作业 `126251765`
（`test34-l32-cache-resume32`）：继续复用同一 distributed `lonv` 与已完成 `hnn`，
仅将 local-stencil 调度批量从 16 调为 32（算法不变，改变内核批调度），并以
`--trust-existing-asset hnn` 避免重复写回；最终 4/4 rank slab reconstruction 会将已有
`hnn` 与本次重算的 local slab 做 SHA256 对比。该 `batch-size=32` 是硬截止恢复参数，
不是基线配方；正式性能仍使用同一 `mg-grid=2^4, levels=2` cache，且只有 report PASS
后才启动。脚本为 `run_cache_resume32.sh`，证据前缀为 `cache-resume32.*`。

若第三次仍未发布完整 PASS report，则性能全部记为 `BLOCKED-CACHE`，不回退旧 cache。

在用户将作业上限更新为 15 分钟后，追加最后一次恢复作业 `126253443`
（`test34-l32-cache-recovery64`）：继续复用同一 `lonv/hnn` partial，将
`batch-size` 从 32 提高到 64，外部时限 15 分钟、内部 watchdog 890 s。该作业在
`14m57s` 被 `rc=124` 终止；5 分钟审计时四卡均 `100%`、约 `8.5 GiB` 显存，
随后 3 个 rank 在约 `400 s` 完成 stencil，但第 4 个 rank 进入不可中断 `D` 状态并阻塞，
未写出 `hdg/sit`、manifest、report 或最终 cache。因此 L32 正式状态为
`BLOCKED-CACHE/TIMEOUT`。

保留的 partial 仅包含已完成的 `lonv/hnn` 数据，供未来恢复；由于缺少完整 cache 与
slab reconstruction，这两个 tensor 仍不能单独作为正式 cache 认证结果。

最后一次参数尝试为 `126265750`（`batch-size=128`、4 DCU，避开此前 stuck 节点）。
作业在 `2m48s` 后以 `rc=1` 失败，四个 rank 均报
`applyMultigridBuildStencilLocalQcu failed with status 1`，未发布 cache/report。
根因是 `[cpp/cuda/qcu/src/multigrid.cu:2589](../cpp/cuda/qcu/src/multigrid.cu#L2589)`
显式要求 `batch_size <= 64`，否则 `launch_multigrid_build_stencil_local` 在启动
kernel 前直接返回 1。该结果是参数门拒绝，不是数值发散、显存不足或 HDF5 损坏；
不再重复 `batch-size=128`，L32 性能保持 `BLOCKED-CACHE`。

按用户要求固定 `batch_size=64` 后，作业 `126269535` 将恢复扩展到 4 节点、16 MPI
rank、16 DCU 和 `grid=2x2x2x2`。单 rank patch 为 552,960 点，较 4-rank 方案的
921,600 点降低 40%。16 个 CUDA stencil 均在 `86.436--87.855 s` 完成，5 分钟审计时
每卡利用率为 0%、显存约 `4,704 MiB`，说明 stencil 已结束而进程主要阻塞在后续共享
HDF5 切片哈希/写入的 `D` 状态。该作业在 `15m02s` 被外部硬限终止，退出码 `124`。
两个 partial 文件名指向同一 inode，当前只含 `lonv/hnn`，没有 `hdg/sit`、manifest、
正式 cache 或 report。当前 partial 为 `3,623,852,032 B`，SHA256
`9759d559701f9c66847713b79dce6fd6fd24fb12c3c5dbb336a749be9374c66d`，仍只是恢复输入。

为避免原样重试，运行了 `126271962`：8 节点、32 MPI rank、32 DCU、
`grid=2x2x2x4`，继续固定 `batch_size=64`，cache identity 不变，单 rank patch 再降至
276,480 点。32 个 stencil 在 `58.197--59.729 s` 完成；5 分钟审计时各 DCU
利用率为 0%、显存约 `3,232 MiB`，进程仍处于共享 HDF5 切片处理路径。用户于运行
`6m09s` 时要求终止，Slurm 终态为 `CANCELLED by 31874`，正式 cache/report 均未发布。
脚本为 `run_cache_recovery32.sh`；L32 构建暂停并继续保持 `BLOCKED-CACHE`。

认证继续执行以下硬门：
`status=PASS`、全部 `rank_slab_reconstruction=PASS`、manifest identity/shape 一致、四个
tensor 逐项 SHA256 及 global output SHA256 完整。任一门失败均不启动性能作业。

## 5. 性能与正确性结果

未启动。缓存认证失败，所有 QCU/QUDA MG 与 BiCGStab 行均为
`BLOCKED-CACHE`；没有旧值回填，也不发布历史或不同 cache 的性能。

## 6. 证据边界

- 本轮仅回答真实 L32 c64 的 Legacy-MG 与 native BiCGStab 对照，不外推到 Strict 或 c128。
- 按任务要求 QCU MG 使用 `--skip-independent-residual`，因此只可声明 native full residual；
  不得将其表述为 independent residual PASS。
- `levels=3`、旧 cache、历史 L32 时间及不同 mg-grid 结果不得进入本轮正式表。
- 若 deadline 前未完成其它 mg-grid 构建，则调优扩展明确记为 `BLOCKED-TIME`。
