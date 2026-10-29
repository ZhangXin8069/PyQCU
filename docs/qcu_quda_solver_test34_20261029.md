# QCU Legacy-MultiGrid 与 QUDA 全尺寸对照测试（test34 续测）

> 状态：CUDA QCU MG L24 正式采样 PASS；L32 的 16-rank 恢复作业超时，32-rank
> 恢复作业已由用户终止，L32 构建暂停；QUDA 辅助对照停止，L48
> `BLOCKED-BY-USER`；最后更新 2026-10-30 02:48 CST。  
> 截止：本轮测试已收尾，其后未保留活动作业。  
> 范围：c64；L24 单节点 1/2/4 MPI；L32 分布式 cache 恢复与 1/2/4-rank
> 性能门；L48 暂缓。  
> 原则：默认不修改核心代码；旧 cache 与旧 L32 性能数据仅作历史基线，正式结果必须来自本轮重建的 QCU 与新 `build_full_mg_cache` cache。

## 执行摘要

- **主目标 CUDA QCU Legacy-MG**：L24 的 1/2/4-rank 正式采样全部 PASS；L32 因分布式 cache 未能在作业时限内完成而保持 `BLOCKED-CACHE`，没有发布性能数字。
- **L24 正式结果**：1/2/4-rank median 分别为 `0.301689520 s`、`0.461907550 s`、`1.024669448 s`；均为 20 iterations，native full residual 约 `2.58e-7`。
- **L24 扩展性**：相对 1 rank，2 rank 为 `0.653x`、4 rank 为 `0.294x`；并行效率分别为 `32.66%` 和 `7.36%`。这组数据不显示强扩展收益，原因是 `24^3x64` 本地子格较小且求解仅约 0.3--1.0 s，通信占比很高。
- **L32 阻塞点**：16-rank 与 32-rank 的 CUDA stencil 都只需约 58--88 s，但随后共享 HDF5 切片的哈希/写入使进程长期处于 `D` 状态。32-rank 作业在 `6m09s` 时由用户终止，正式 cache/report 均不存在。
- **QUDA 对照**：已按要求停止为辅助路线；L24 QUDA 在 PyQUDA/QUDA HIP 内存路径中触发 SIGSEGV，标记 `BLOCKED-RUNTIME`，未产生可用于比较的性能结果。
- **L48**：按用户要求暂停，状态为 `BLOCKED-BY-USER`，没有提交本轮 canonical/cache/性能作业。
- **资源收尾**：没有活动 test34 作业，没有 core dump；公共文件系统为 `40G/50G`，可用约 `11G`。L24 认证 cache、L32 partial、硬件审计和全部正式日志均保留。

## 1. 验收标准

1. QCU 原生库与 Cython 桥必须由当前源码重建，动态符号和 Python 导入通过。
2. MG cache 必须由 `pyqcu.testing.simple.mpi.build_full_mg_cache` 的 native MPI 分布式路径生成。
3. 每个正式 cache 必须具备 `status=PASS`、完整 identity/shape、逐 tensor SHA256、全局 HDF5 SHA256 和每个 rank 的 slab reconstruction PASS。
4. 正式性能只接受同一 canonical gauge/source、同一精度和已认证 cache 下的结果。
5. 1/2/4 rank 必须保持一 Python 进程 = 一 MPI rank = 一 DCU；不得在单进程内复制整题到多卡。
6. 用户于 2026-10-29 23:00 将 GPU 作业上限更新为 15 分钟；运行超过 5 分钟必须至少有一次进程、DCU 利用率、显存和输出增长审计；超过 15 分钟必须终止并审查。已按原 10 分钟上限启动的作业保留其更严格内部门限。
7. L32 的旧性能表全部失效，必须在新 cache 上重跑。
8. 无法完成的尺寸、rank、层次或调优点必须标记 `BLOCKED`/`UNVERIFIED`，不得填伪值或将异架构历史结果当作本轮性能。

## 2. 本轮构建 provenance

| 项目 | 结果 |
|---|---|
| QCU 原生库 | `cpp/cuda/qcu/libqcu.so` |
| 原生库 SHA256 | `b513fbcd41e48caa5e9bc3acd2f23693c52b177ddd26efe65e0cb6f82ed068bf` |
| 原生库构建完成 | 2026-10-29 22:25:41 CST |
| Cython 扩展 | `pyqcu/cuda/qcu.cpython-38-x86_64-linux-gnu.so` |
| Cython SHA256 | `df204a1c5dcce7be6720e988fbc1534be5bfd1f2d5d92f11916b49f589aa9386` |
| Cython 构建完成 | 2026-10-29 22:29:40 CST |
| 动态符号 | `applyCloverMultigridQcu`、`applyCloverBistabCgQcu`、`applyInitQcu`、`applyEndQcu` 均存在 |
| Python 导入 | `pyqcu.cuda.qcu` 导入成功，MG/BiStabCG API 可见 |
| build_full_mg_cache CLI | `--help` 成功，包含 `--grid`、`--ranks`、`--batch-size`、`--reuse-lonv`、`--reuse-existing-assets` |

### 测试工具修改

| 文件 | 修改 | 验证边界 |
|---|---|---|
| `pyqcu/testing/simple/mpi/worker.py` | 执行此前仅解析但未运行的 `--warmup/--repeat`；每个 timed sample 都是同一 source 的 cold solve，x0 清零和设备同步在计时区外；报告逐次样本、median、MAD 与每 rank 时间 | 已由三组正式 L24 报告和独立认证 JSON 验证 |
| `pyqcu/testing/simple/README.md` | 更新 warmup/repeat 的实际语义 | 文档同步 |
| `pyqcu/testing/quda/benchmark_quda_lime_mpi.py` | 将 PyQuda even/odd 转换后的字段恢复为请求的 c64/c128 dtype，并同步 `gauge_param.cpu_prec` | 纯 Python helper 直接测试通过；实机 QUDA 仍被 HIP runtime SIGSEGV 阻塞 |
| `pyqcu/testing/quda/test_benchmark_quda_lime_mpi.py` | 增加字段 dtype 和 pinned PyQuda pointer 回归 | helper 直接调用 PASS；环境缺少 pytest，未执行 pytest runner |

## 3. DCU 硬件与运行环境

硬件证据由独立只读作业 `126271539` 在 `b16r1n16` 采集，完整输出保存在
`logs/qcu_solver_test34_20261029/dcu-hardware-audit.txt`。Slurm 对测试节点统一报告的
资源类型为 `dcu:Hygon:4`。

| 项目 | 实测值 |
|---|---|
| 加速卡 | Pre-Wukong DCU，Card SKU `D160A1`，rocminfo 名称 `ZIFANG`，Chip ID `0x66a1` |
| GPU ISA | `amdgcn-amd-amdhsa--gfx906:sramecc-:xnack-` |
| 计算资源 | 64 CU，每 CU 4 SIMD，wavefront 64，最高 1.7 GHz |
| 显存 | `17,163,091,968 B`（rocminfo `16,760,832 KiB`；rocm-smi `16,368 MiB`） |
| VBIOS | `113-D160A1SG-F42` |
| 节点 CPU | Hygon C86 7185 32-core，4 NUMA node，最高 2.0 GHz |
| 节点内存/拓扑 | 126,500 MiB；32 CPU 与 4 DCU，执行拓扑为一个 Python 进程、一个 MPI rank、一张 DCU |
| 本轮资源 | L24 正式采样为 1/2/4 rank 单节点；L32 16-rank 为 4 节点、16 DCU；32-rank 恢复为 8 节点、32 DCU |
| 系统 | CentOS 7.6、Linux `3.10.0-957.el7.x86_64`，`amdgpu` 已加载 |
| QCU 软件栈 | DTK 23.10、HIP `5.4.23453`、HPC-X MPI 2.11.0，`PYQCU_MPI_DEVICE_AWARE=0` |
| QUDA 软件栈 | DTK 25.04.4、QUDA 1.1.0 `d615172-dirty-gfx906`，独立进程加载 |

`rocm-smi` 的 `Driver version` 字段在上述环境返回内核版本
`3.10.0-957.el7.x86_64`，因此报告不把它误写为 DCU 用户态驱动版本。商品型号只写
硬件工具实际返回的 `Pre-Wukong DCU / D160A1 / ZIFANG`，不推测等价 AMD SKU。

## 4. 已发现构建与环境 bug

### B1. `setup.py` 在 DCU 上硬编码 NVIDIA 构建

`setup.py::CMakeBuild.run` 无条件调用 `build-nv.sh`。在 DTK/ROCm 环境执行 `bash ./install.sh` 时，Cython 化成功，但随后按 NVIDIA `sm_52` 路径重建，并因 `__hipPushCallConfiguration` 未声明失败。  
本轮未修改核心代码，使用已完成的原生 DCU 构建并执行：

```bash
PYQCU_SKIP_NATIVE_BUILD=1 python setup.py build_ext --inplace
```

该问题应修复为根据 `PYQCU_CUDA_BUILD_BACKEND=dcu|nv` 选择构建脚本。

### B2. 默认环境 Python 版本低于项目声明

`~/env.sh` 激活 Python 3.8.13，而项目要求 Python >=3.10。`pyqcu/testing/_config.py` 使用运行时求值的 `tuple[int, ...]`，导致部分测试入口在导入时失败：

```text
TypeError: 'type' object is not subscriptable
```

该问题影响 `pyqcu/testing/qcu/single_qcu_api.py` 等测试工具；分布式 cache builder 与 simple MPI 主入口仍可启动，但这使完整 pytest/API 回归不满足项目声明的运行时基线。

### B3. PyQUDA 固定提交缺少版本文件

仓库固定的 PyQUDA 提交 `138d8ad6e4d6e0805ecc7ec0a59a45febe0482ce` 不在 tag 上，浅克隆下 `setup.py egg_info` 无法生成 `pyquda_utils/_version.py`。本轮仅在该子模块中生成未提交占位版本文件以完成运行，未修改 QCU 核心。

### B4. QUDA 与默认 DTK 运行时不匹配

当前 `libquda.so` 依赖 DTK 25.04 的 hipBLAS/hipFFT，而 `~/env.sh` 优先加载 DTK 23.10，直接导入 PyQUDA 出现：

```text
libhipfft.so.0: undefined symbol: rocfft_field_create
```

QUDA 子进程必须显式将 DTK 25.04 置于 `LD_LIBRARY_PATH` 前部；QCU 子进程继续使用 DTK 23.10，二者不得在同一进程混载。

## 5. 输入与资产

### L24

| 项目 | 路径 |
|---|---|
| LIME | `/public/home/sunpeng_1/data/beta6.20_mu-0.2770_ms-0.2310_L24x64_cfg_9950.lime` |
| canonical gauge | `/public/home/sunpeng_1/PyQCU/logs/dcu_qcu_quda_20261027/data/qcu-mg-c64-24x24x24x64/gauge_global_c64.h5` |
| source | `/public/home/sunpeng_1/PyQCU/logs/dcu_qcu_quda_20261027/data/qcu-mg-c64-24x24x24x64/source_global_c64.h5` |

旧 L24 cache 不作为本轮认证输入；本轮由新 distributed builder 重建。

### L32

LIME 为 `beta6.20_mu-0.2770_ms-0.2400_L32x64_cfg_11000.lime`。本轮已生成 canonical c64 输入，但未生成认证 cache：

| 项目 | 结果 |
|---|---|
| canonical gauge | `data/qcu_solver_test34_20261029/L32/case/gauge_32x32x32x64_c64.h5`，1,207,965,696 B，SHA256 `0e1d2de249fef3124b5a3985446abac63f1bf33217e274abb9b5376ca143f433` |
| canonical source | `data/qcu_solver_test34_20261029/L32/case/source_32x32x32x64_c64.h5`，201,332,736 B，SHA256 `3d9fefe243f9ab7c8880f28156fa257a803e924d950b68925e20903f3351beca` |
| case manifest | `data/qcu_solver_test34_20261029/L32/case/case_manifest.json` |
| 准备耗时 | wall `36.21 s`，peak RSS `3,050,828 KiB` |

### L48

LIME 为 `beta6.20_mu-0.2770_ms-0.2400_L48x96_cfg_7000.lime`。历史 A100 cache 的元数据存在，但当前工作树没有对应 18.7 GB cache 实体；本轮必须用新方案重新构建，未发布前不得运行或引用性能。

## 6. 并行分片

| 分片 | 子代理运行句柄 | 输出 |
|---|---:|---|
| L24 新 cache + 1/2/4-rank MG/QUDA | `19694` | `docs/qcu_quda_solver_test34_L24.md`、`logs/qcu_solver_test34_20261029/L24/` |
| L32 新 cache 认证 + 全部性能重跑 | `94958` | `docs/qcu_quda_solver_test34_L32.md`、`logs/qcu_solver_test34_20261029/L32/` |
| L48 新 cache/容量门 + 条件性能 | `76130` | `docs/qcu_quda_solver_test34_L48.md`、`logs/qcu_solver_test34_20261029/L48/` |

三个分片只写各自目录，不修改 `pyqcu/`、`cpp/`、`setup.py` 或既有报告。

## 7. 当前结果

L24 已产生可发布的单次 QCU native solve 数据；其余结果严格按以下状态记录，
任何历史结果不得替代：

| 数据 | QCU MG | QUDA MG | QCU BiCGStab | QUDA BiCGStab | cache 认证 |
|---|---|---|---|---|---|
| L24 1/2/4 rank | PASS | BLOCKED-RUNTIME | OPTIONAL | OPTIONAL | **PASS** |
| L32 1/2/4 rank | BLOCKED-CACHE | BLOCKED-RUNTIME | OPTIONAL | OPTIONAL | **未发布/用户终止** |
| L48 2/4 rank | BLOCKED-BY-USER | BLOCKED-BY-USER | OPTIONAL | OPTIONAL | BLOCKED-BY-USER |

### L24 新 cache

Slurm `126249969` 使用 4 DCU、32 CPU、`2x2x1x1` process grid 运行新 native MPI builder，10 分钟上限，实际 `297 s` 完成，退出码 `0`。未达到 5 分钟审计阈值。

| 项目 | 结果 |
|---|---|
| builder | `mpi-distributed-null-vector+overlap-local-stencil` |
| setup | `205.90119095589034 s` |
| file | `data/qcu_solver_test34_20261029/L24/cache_L24_E12_12_mg2x2x2x2.h5` |
| size | `1,560,688,640 B` |
| SHA256 | `92d0cd7d858fdb3658cf926509cb99703ee4358754c1fa0febf1f051df4fec9f` |
| identity SHA256 | `5defd59004535e3a125137cddb75cd062caa860efa5e4ee04b1b86079e02d779` |
| rank slabs | `4/4 PASS` |
| tensor digests | `lonv/hnn/hdg/sit` 均存在，global digest 已写入 manifest |

该 cache 满足进入 L24 solve 的资产门。

#### L24 QCU Legacy-MG 结果

Slurm `126250621/126250623/126250620` 的首轮编排因外层 `mpirun` 与 wrapper 内层 launcher 叠加而 fail-closed；修正为直接调用 wrapper 后，三档均产出 PASS。

| ranks/grid | solve wall (s) | iterations | V-cycles | coarse iters | native full residual | status |
|---:|---:|---:|---:|---:|---:|---|
| `1 / 1x1x1x1` | `0.418668830` | 20 | 1 | 8 | `2.582429e-7` | PASS |
| `2 / 2x1x1x1` | `0.714287481` | 20 | 1 | 8 | `2.582687e-7` | PASS |
| `4 / 2x2x1x1` | `1.427416675` | 20 | 1 | 8 | `2.581189e-7` | PASS |

上表是早期单次 native solve 数据。simple CUDA worker 的
`--warmup/--repeat` 执行路径已修复：现在每个样本都重置 x0 并执行 cold solve，
零化与同步位于计时区外，报告逐次样本、median 和 MAD。修复后的正式结果：

| ranks/grid | warmup (s) | steady samples (s) | median (s) | MAD (s) | iterations | native residual | status |
|---:|---:|---|---:|---:|---:|---:|---|
| `1 / 1x1x1x1` | `1.389318520` | `0.299408800, 0.305628530, 0.301689520, 0.302894160, 0.299466530` | `0.301689520` | `0.002222990` | 20 | `2.582429e-7` | PASS |
| `2 / 2x1x1x1` | `1.776467187` | `0.489892693, 0.460595469, 0.459965060, 0.461907550, 0.463353181` | `0.461907550` | `0.001445631` | 20 | `2.582687e-7` | PASS |
| `4 / 2x2x1x1` | `1.235514439` | `1.054773713, 1.033934803, 1.024669448, 1.024618929, 1.020558617` | `1.024669448` | `0.004110831` | 20 | `2.581189e-7` | PASS |

对应作业为 `126264615`（1 rank）、`126264880`（2 ranks）、`126264881`（4 ranks）。
三个作业均以 `MODE=qcu` 运行，未启动 QUDA。三档每轮均为 `n_vcycles=1`、
`coarse_iters=8`；独立 full residual 按任务要求显式跳过，因此这里只声明 native
full-operator residual PASS。

独立校验结果见
`logs/qcu_solver_test34_20261029/L24/qcu-sample5-authentication.json`：三档均满足
1 warmup、5 timed samples、20 iterations、报告 median/MAD 复算一致，三个 solution
HDF5 均已记录 SHA256。

以 1-rank median 为基线的同任务强扩展结果如下。该表只用于本轮 L24 计时口径，
不把短 solve 的低效率外推到更大格点：

| ranks | median (s) | speedup vs 1 rank | parallel efficiency |
|---:|---:|---:|---:|
| 1 | `0.301689520` | `1.000x` | `100.00%` |
| 2 | `0.461907550` | `0.653x` | `32.66%` |
| 4 | `1.024669448` | `0.294x` | `7.36%` |

QUDA 首轮将 canonical HDF5 传给仅接受 LIME 的 QUDA benchmark，按预期 fail-closed。后续从 canonical gauge 生成同源 LIME 后，torch 与 cupy 两种数组后端仍在
`loadClover -> quda::GaugeField::copy -> quda::qudaMemcpy_ -> hipMemcpy` 路径触发 SIGSEGV
（`rc=139`）。attempt 5 的 fault-handler 栈保存在
`logs/qcu_solver_test34_20261029/L24/solve/r1/attempt5/`；因此 L24 QUDA MG 标记为
`BLOCKED-RUNTIME`，不使用历史结果替代。

### L32 新 cache

本轮 L32 cache 的主要作业与终止原因如下。表中的“超时”均表示没有发布完整
cache/report，不允许进入性能阶段：

| Slurm job | ranks/grid | batch | elapsed | 终态 | 结论 |
|---|---:|---:|---:|---|---|
| `126250045` | `4 / 2x2x1x1` | 16 | `9m43s` | `rc=124` | 初次构建超时，仅完成 `lonv/hnn` |
| `126250791` | `4 / 2x2x1x1` | 16 | `9m57s` | `rc=124` | 复用 finalize 超时 |
| `126251765` | `4 / 2x2x1x1` | 32 | `9m57s` | `rc=124` | 提高 batch 后仍超时 |
| `126253443` | `4 / 2x2x1x1` | 64 | `14m57s` | `rc=124` | stencil 后共享写入阻塞 |
| `126255388` | `1 / 1x1x1x1` | 64 | `14m44s` | `rc=124` | 单 rank 容量/时间均不足 |
| `126265750` | `4 / 2x2x1x1` | 128 | `2m48s` | `rc=1` | 参数上限校验拒绝，非数值失败 |
| `126269535` | `16 / 2x2x2x2` | 64 | `15m02s` | `rc=124` | stencil 约 87 s，后续共享 I/O 超时 |
| `126271962` | `32 / 2x2x2x4` | 64 | `6m09s` | `CANCELLED by 31874` | stencil 约 59 s；用户终止并暂停 L32 |

`batch_size=128` 的失败根因已经定位到 native 参数门：
`[cpp/cuda/qcu/src/multigrid.cu:2589](../cpp/cuda/qcu/src/multigrid.cu#L2589)`
明确要求 `batch_size <= 64`，否则 `launch_multigrid_build_stencil_local`
直接返回 1。因此四个 rank 的 `applyMultigridBuildStencilLocalQcu status 1`
是 CUDA kernel 启动前的参数拒绝，不是数值发散、显存不足或 HDF5 损坏。

Slurm `126250045` 使用 4 DCU、32 CPU、`2x2x1x1` process grid。5 分钟审计时四张 DCU 均为 `100%` 利用率、各约 `9.35 GiB` 显存，stdout 持续增长，未发现停滞；该作业在旧 10 分钟内部门限下于 `9m43s` 终止，已算出的 `lonv` partial 保留。随后提交 `126250791`，使用同一 canonical gauge、相同 identity 和 `--reuse-lonv/--reuse-existing-assets` finalize 其余 slab；该作业也在 `9m57s` 因 `rc=124` 终止，没有形成完整 cache。

最终 4-rank 恢复作业 `126253443` 使用 `batch-size=64`、4 DCU 和 15 分钟上限；作业在
`14m57s` 以 `rc=124` 终止。3 个 rank 在约 `400 s` 完成 stencil，但第 4 个 rank
进入 `D` 状态并阻塞，最终未发布 `hdg/sit`、manifest、report 或 cache。L32 cache
因此为 `TIMEOUT`，所有 L32 性能项为 `BLOCKED-CACHE`。

最后的 `batch-size=128` 尝试 `126265750` 在另一节点运行，`2m48s` 后四个 rank 均以
`applyMultigridBuildStencilLocalQcu status 1` 失败，未发布 cache/report。该错误来自
native 参数门对 `batch_size > 64` 的显式拒绝，不代表 128 批次下的数值失败，也不再
重复同构尝试。

按用户要求固定 `batch_size=64` 后，进一步提交 16-rank 恢复作业 `126269535`：
4 节点、16 DCU、`grid=2x2x2x2`，单 rank patch 从 4-rank 方案的 921,600 点降至
552,960 点。16 个 rank 的 CUDA stencil 均在 `86.436--87.855 s` 完成，5 分钟审计时
每卡显存约 `4,704 MiB`。作业随后因共享文件切片哈希/写入长期处于 `D` 状态，在
`15m02s` 被 15 分钟硬限终止，退出码 `124`；两路名称指向同一个 partial，仍只包含
`lonv/hnn`，没有 `hdg/sit`、manifest、正式 cache 或 report。因此该次结果仍是
`BLOCKED-CACHE/TIMEOUT`，不能进入性能阶段。

当前 partial 大小为 `3,623,852,032 B`，SHA256 为
`9759d559701f9c66847713b79dce6fd6fd24fb12c3c5dbb336a749be9374c66d`；它仅作为恢复
输入，不是认证 cache。为继续降低单 rank 哈希/写入工作量，曾运行作业 `126271962`：
8 节点、32 DCU、`grid=2x2x2x4`、`batch_size=64`，cache identity 不变，时限
15 分钟，并设置至少 5 GiB 磁盘余量门。32 个 stencil 在 `58.197--59.729 s`
完成；5 分钟审计时各卡利用率为 0%、显存约 `3,232 MiB`，进程仍处于共享 HDF5
切片处理路径。用户于运行 `6m09s` 时明确要求终止，Slurm 终态为
`CANCELLED by 31874`；正式 cache/report 均不存在。L32 构建暂停，继续保持
`BLOCKED-CACHE`。

L32 的直接瓶颈位于
`[pyqcu/testing/simple/mpi/_distributed_mg_cache.py:878](../pyqcu/testing/simple/mpi/_distributed_mg_cache.py#L878)`
之后的串行 tensor 循环：每个 rank 先对 patch 计算 chunked SHA256，再参与共享
HDF5 slab 写入。增大 rank 数显著缩短 stencil（4-rank 约 431--459 s、16-rank 约
87 s、32-rank 约 59 s），但未消除该阶段的 `D` 状态。因此后续恢复不应继续只调
`batch_size`，而应先把大 tensor 改为可恢复的分阶段写入/独立 rank 文件后合并，或
提高该阶段可用的本地暂存与共享 I/O 预算。

### L48 canonical 输入

`prepare_case.py` 在登录节点转换 `48x48x48x96` 时约 `24.62 s` 后被信号 9 终止，尚未生成 canonical gauge/cache。当前没有进入 4-DCU cache 作业，也没有任何可发布性能。代理正在区分登录节点内存限制、文件系统/配额和转换实现容量问题；在该阻塞解除前，L48 cache 与 solve 均标记 `BLOCKED`。

2026-10-29 23:35 CST 用户明确要求暂缓 L48 测试并控制 50GB 总磁盘占用。L48 当前统一标记为
`BLOCKED-BY-USER`，不再提交 canonical/cache/性能作业；失败转换留下的 `254,831,528 B` partial 已清理。

## 8. 结论使用边界

- 没有新 cache 的 `rank_slab_reconstruction` 全 PASS，不得把 solve 时间列入正式比较。
- 没有同输入的 QCU/QUDA 两条记录，不得计算加速比。
- 没有独立或明确的 native full residual 证据，不得宣称数值正确性。
- L8 仅作为正确性/输入尺寸选项之一，本轮不把它作为正式性能目标，也不再为 L8 扩展消耗测试资源。
- L48 若在 10 分钟内未发布 cache，只能给出容量/超时证据，不能引用历史 A100 solve 作为本轮结果。

## 9. 磁盘清理

用户确认本库所在文件系统总限额为 50GB，并要求实时清理 `data/` 与 `logs/` 中无用文件；
L48 工作同时暂停。2026-10-29 23:28--23:36 CST 已执行以下清理：

| 对象 | 释放空间 | 保留边界 |
|---|---:|---|
| L24 各 solve attempt 的 `work/` scratch | `1,613,833,954 B` | 保留 `logs/.../report.json`、stdout/stderr/audit |
| 旧 `24x24x24x64-batch16` 的 H5 cache/gauge/source 副本 | `1,374,896,224 B` | 保留 JSON/report 和主 E6 cache |
| 旧 `data/logs/.../qcu-clover-c64.h5` | `1,019,222,016 B` | 保留 `qcu-clover-c64.json` |
| 旧 trial 中未被 `final_summary.json` 引用的 solution HDF5 | `1,445,585,440 B` | 保留最终 7 组 QCU、3 组 QUDA 及 baseline 结果 |
| L48 失败转换 partial | `254,831,528 B` | L48 继续 `BLOCKED-BY-USER` |
| L24 QUDA round-trip LIME | `509,609,448 B` | canonical HDF5 与 SHA/回环记录保留，可再生成 |

后续 CUDA QCU 正式采样与故障测试又清理了以下仅用于复现的临时文件：

| 对象 | 释放空间 | 保留边界 |
|---|---:|---|
| 5 个 GPU 失败 core dump | `10,870,423,552 B` | 完整 stderr/fault-handler/backtrace 日志保留 |
| L24 `qcu-sample5` gauge/source scratch | `1,019,253,024 B` | solution HDF5、report、manifest、stdout/audit 保留 |
| L24 QUDA round-trip LIME 再次生成副本 | `509,609,448 B` | canonical HDF5、精确 round-trip JSON 与 SHA 保留 |

清理后 `/public` 使用量由 `45G/50G` 降至 `40G/50G`，可用空间由约 `5G` 增至 `11G`。
认证 cache、当前 L32 恢复源、正式 JSON/文本日志及最终 trial solution 均未删除。

本轮最终目录占用为 `data/` 约 `8.2G`、`logs/` 约 `3.1G`。L32 的三个 partial
文件名是同一 inode 的三个硬链接，总大小 `3,623,852,032 B`，没有按文件名重复占用磁盘。
收尾检查未发现 `core`/`core.*`，也未发现活动 test34 作业。

## 10. 最终验证

| 检查 | 结果 |
|---|---|
| L24 cache 独立认证 | PASS；4/4 rank reconstruction、identity、shape、四个 tensor SHA256 和 global output SHA256 全部通过 |
| L24 solve 独立认证 | PASS；三档均为 1 warmup、5 timed samples、20 iterations，median/MAD 可复算 |
| L24 原始报告 | 三份 `qcu.report.json` 与认证 JSON 的 median、MAD、residual、solution SHA256 一致 |
| L32 正式输出 | cache/report 均不存在；保持 `BLOCKED-CACHE` |
| L32 partial | 仅含 `lonv/hnn`；没有 `hdg/sit`、manifest 或 slab reconstruction，不认证 |
| Slurm 收尾 | 无活动 test34 作业；`126271962` 为 `CANCELLED by 31874` |
| 文件检查 | `git diff --check` PASS；恢复脚本 `bash -n` PASS |
| Python 检查 | `worker.py`、QUDA benchmark 及测试文件 `py_compile` PASS；QUDA helper 直接测试 PASS |
| pytest | 未执行；当前 Python 环境没有 `pytest` 模块，不能把该项写成 PASS |
| form 审计 | FAIL，但为仓库既有 162 项未登记 finding；本轮新增报告、硬件审计和恢复脚本未引入新的 finding |

直接执行的 QUDA helper 检查包括 UUID JSON 安全转换、LIME slab/global RHS 布局、
MG CLI 合同、c64/c128 字段 dtype 恢复和 pinned PyQuda pointer。它们证明测试辅助代码的
局部合同，不证明 QUDA 实机求解通过；后者仍为 `BLOCKED-RUNTIME`。

## 11. 最终结论与恢复建议

1. **可交付结果**：只发布 L24 CUDA QCU Legacy-MG 的 1/2/4-rank 数据。正确性表述限定为
   native full-operator residual，不得写成 independent residual PASS。
2. **L24 工程判断**：当前 `24^3x64` 规模下 MPI 扩展被短 solve 的通信和固定开销主导；
   该结果不是大格点并行效率结论。后续应先在更大本地子格或减少全局同步的条件下测试扩展性。
3. **L32 工程判断**：`batch_size=64` 已是 native 上限；继续增加 rank 只能缩短 stencil，
   不能解决共享 HDF5 哈希/写入的 `D` 状态。恢复前应先改造 cache 持久化阶段，而不是再次提交
   同构长作业。
4. **L32 恢复点**：保留的 partial 只能作为 `lonv/hnn` 输入；必须先检查 tensor digest，
   再生成 `hdg/sit`、manifest，最后通过全部 rank slab reconstruction 才可运行性能。
5. **暂停项**：L48 继续 `BLOCKED-BY-USER`。恢复需用户明确解除暂停并重新确认 50GB 容量门。
6. **辅助对照**：QUDA 当前被 DTK/PyQUDA HIP 内存路径阻塞。除非先修复运行库，不应继续消耗
   DCU 时间；不能用历史 QUDA 数字与本轮 QCU 结果拼成加速比。

## 12. 关键产物索引

| 内容 | 路径 |
|---|---|
| 总报告 | `docs/qcu_quda_solver_test34_20261029.md` |
| L24 子报告 | `docs/qcu_quda_solver_test34_L24.md` |
| L32 子报告 | `docs/qcu_quda_solver_test34_L32.md` |
| L48 子报告 | `docs/qcu_quda_solver_test34_L48.md` |
| L24 cache 认证 | `logs/qcu_solver_test34_20261029/L24/cache-authentication.json` |
| L24 正式采样认证 | `logs/qcu_solver_test34_20261029/L24/qcu-sample5-authentication.json` |
| L24 三份原始 solve 报告 | `logs/qcu_solver_test34_20261029/L24/solve/r{1,2,4}/qcu-sample5/qcu.report.json` |
| L32 16-rank 审计 | `logs/qcu_solver_test34_20261029/L32/cache-recovery16.audit.txt` |
| L32 32-rank 审计 | `logs/qcu_solver_test34_20261029/L32/cache-recovery32.audit.txt` |
| DCU 硬件审计 | `logs/qcu_solver_test34_20261029/dcu-hardware-audit.txt` |
| 32-rank 恢复脚本 | `logs/qcu_solver_test34_20261029/L32/run_cache_recovery32.sh` |
