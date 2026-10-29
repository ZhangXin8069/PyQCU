# PyQCU DCU L24 QCU/QUDA 求解器实测（2026-10-29）

## 状态

- 开始时间：2026-10-29 22:42:56 CST
- 输入：真实 LIME 配置对应的 canonical c64 gauge/source，格点 `24x24x24x64`
- 二进制：`libqcu.so` SHA256 `b513fbcd41e48caa5e9bc3acd2f23693c52b177ddd26efe65e0cb6f82ed068bf`
- Cython：`qcu.cpython-38-x86_64-linux-gnu.so` SHA256 `df204a1c5dcce7be6720e988fbc1534be5bfd1f2d5d92f11916b49f589aa9386`
- 本次测试状态（2026-10-29 至 2026-10-30）：CUDA QCU 目标完成；cache 已认证，
  1/2/4-rank CUDA QCU MG 已完成 1 warmup + 5 timed cold solves。QUDA 辅助对照已停止
  且标记 `BLOCKED-RUNTIME`。
- 报告限制：未独立计算 full residual 的 case 只记录 native residual，不能宣称独立真残差 PASS。

## DCU 硬件与运行环境

硬件证据由只读作业 `126271539` 在 `b16r1n16` 采集；完整 `rocminfo/rocm-smi/hipconfig`
输出见 `logs/qcu_solver_test34_20261029/dcu-hardware-audit.txt`。Slurm 对测试节点统一
报告资源类型 `dcu:Hygon:4`。

| 项目 | 实测值 |
|---|---|
| 加速卡 | Pre-Wukong DCU，Card SKU `D160A1`，rocminfo 名称 `ZIFANG`，Chip ID `0x66a1` |
| ISA | `gfx906`（`sramecc-`、`xnack-`），64 CU，4 SIMD/CU，wavefront 64，最高 1.7 GHz |
| 显存 | `17,163,091,968 B`（rocm-smi 显示 `16,368 MiB`） |
| VBIOS | `113-D160A1SG-F42` |
| 节点 | Hygon C86 7185 32-core，4 NUMA node，126,500 MiB，32 CPU 与 4 DCU |
| 执行拓扑 | 一个 Python 进程 = 一个 MPI rank = 一张 DCU |
| 运行栈 | QCU：DTK 23.10、HIP `5.4.23453`、HPC-X MPI 2.11.0；QUDA 对照：DTK 25.04.4 |

正式 L24 采样作业分别落在 `b16r1n08`、`e14r2n18`、`e10r1n13`；调度器对三类节点
报告相同的 `dcu:Hygon:4` 资源类型。`rocm-smi` 的 driver 字段返回的是内核版本
`3.10.0-957.el7.x86_64`，不将其误记为 DTK 用户态驱动版本。

## 执行门

1. 新 cache 必须满足 `status=PASS`、4/4 rank slab reconstruction PASS，并核验 manifest identity、shape、逐 tensor SHA256 与全局 output SHA256。
2. cache 认证前不运行、不发布 QCU 性能。
3. QCU 与 QUDA 在隔离进程和隔离环境变量下运行，不混载 ROCm 运行库。
4. 每个 Slurm 作业单节点、时限不超过 10 分钟；300 秒执行进程/DCU/输出尾部审计，570 秒内部终止。

## 实时记录

### 2026-10-29 22:43 CST

- 已确认 `libqcu.so` 与 Cython 扩展 SHA256 和任务指定值一致。
- 已确认 canonical gauge/source 存在：gauge 244 MiB，source 82 MiB。
- 已确认三个目标入口可由 `--help` 解析。
- 已创建 `run_cache.sh`，资源为 `kshdnormal`、1 node、4 DCU、32 CPU、10 分钟硬限。

### 2026-10-29 22:50 CST

- cache 作业 `126249969` 在节点 `e03r1n16` 完成，耗时 297 秒，返回码 0。
- builder 报告 `status=PASS`，4/4 rank slab reconstruction PASS。
- cache 路径：`data/qcu_solver_test34_20261029/L24/cache_L24_E12_12_mg2x2x2x2.h5`。
- cache 大小：1,560,688,640 bytes；全局 SHA256：`92d0cd7d858fdb3658cf926509cb99703ee4358754c1fa0febf1f051df4fec9f`。
- 独立认证重新计算 canonical gauge SHA256 `1748b0075fee9df8fc2b62bc7fdd9ceff7da7eca80e77add8952d08ab3877630` 与 gauge 文件 SHA256 `18d3a134c288a700baa5ed92a7d405dc04c40b74cd579e4cd863ccb46fde823f`，manifest identity SHA256 为 `5defd59004535e3a125137cddb75cd062caa860efa5e4ee04b1b86079e02d779`。
- 四个 tensor SHA256：`lonv=80d5bcdadfd384f14de899d3f02c218af395dfefe2e536a16e09724db1615409`、`hnn=0f5deb8b89a6e10c9904684d737c1f352ba16092e36a6ca237ab53e6deabd2b9`、`hdg=d7cd445419313dd06704cb73c37c140805bc749ae6b934f61bfca83d0872b422`、`sit=e892670139152281f1b4bb652c14be40f18f43c68d81c54077d73e6c65aa2e4a`。
- cache 认证门全绿，可用于性能测试；builder `setup_seconds=205.90119095589034`。
- 已提交 solve 作业：`126250621`（1 rank）、`126250623`（2 ranks）、`126250620`（4 ranks）。每个作业单节点、10 分钟硬限，并按指定 DCU/CPU 数申请。

### 2026-10-29 22:55 CST

- 首轮 solve 作业均在 72--103 秒内 fail-closed：MPI 版 QCU/QUDA wrapper 本身是单进程 launcher，脚本外层再次调用 `mpirun`，触发 `mpirun does not support recursive calls`。
- 失败作业没有任何 solve report，未产生可比较性能数据；cache 及其认证不受影响。
- 原始 stdout/stderr/audit 保留在 `logs/qcu_solver_test34_20261029/L24/solve/r{1,2,4}/`。修正后的 attempt 2 改为直接调用 wrapper，由其内部执行唯一一次 `mpirun`。
- 修正版作业：`126250750`（1 rank）、`126250752`（2 ranks）、`126250751`（4 ranks）。

### 2026-10-29 22:59 CST

- QCU attempt 2 三档均 PASS：1/2/4-rank wall time 分别为 `0.41866882995236665`、`0.7142874810378999`、`1.4274166750255972` 秒；均为 20 次外层迭代、`n_vcycles=1`、`coarse_iters=8`。
- QCU native full relative residual 分别为 `2.582429e-7`、`2.582687e-7`、`2.581189e-7`。按任务要求使用 `--skip-independent-residual`，因此独立真残差未验证。
- QUDA attempt 2 三档均 fail-closed：`benchmark_quda_lime_mpi` 的 gauge reader 仅接受 LIME，canonical HDF5 触发 `not a LIME file at byte 0`，没有发布性能数据。
- 已从 canonical c64 gauge 生成 `gauge_global_c64_from_canonical.lime`；该 LIME 经 QDP back-conversion 与原 canonical HDF5 逐元素完全一致。原 complex128 LIME 到 canonical c64 的最大绝对差为 `4.2132016269772676e-8`。
- QUDA attempt 3 已提交：`126251399`（1 rank）、`126251400`（2 ranks）、`126251396`（4 ranks），仅运行 QUDA，gauge 使用 canonical c64 的精确 LIME round-trip，source 仍为 canonical source HDF5。

### 2026-10-29 23:07 CST

- QUDA attempt 3（1/2/4-rank）均失败且未发布 report。默认 PyTorch-HIP 数组后端在 DTK 25.04.4 上进入 `torch.Tensor.clone` 后于 `hipMemcpyAsync`/`libgalaxyhip.so.5` 段错误；这是运行库兼容失败，不能记为 QUDA 数值或性能结果。
- attempt 4 固定 `PYQCU_QUDA_ARRAY_BACKEND=cupy`，避免加载 PyTorch-HIP；先仅运行 1-rank，通过后才提交其余 rank。

### 2026-10-29 23:35 CST

- attempt 5 在 Slurm `126253707` 中以 1-rank、`PYQCU_QUDA_ARRAY_BACKEND=cupy` 和
  `PYTHONFAULTHANDLER=1` 重试，耗时 177 s 后再次以 `rc=139` 退出。
- fault handler 定位到 `pyquda.dirac.general.loadClover` 调用链：
  `quda::GaugeField::copy -> quda::qudaMemcpy_ -> quda::QudaMem::apply -> hipMemcpy`，
  最终在 HIP 内存路径触发 SIGSEGV；不是 gauge 格式、MPI rank 或输入缺失错误。
- attempt 3 的 torch 后端同样为 `rc=139`，attempt 2 已证明 HDF5 在
  `read_lime_metadata` 门失败；因此本轮不继续消耗 GPU 时间，QUDA MG 结果标记为
  `BLOCKED-RUNTIME`，不发布数值或性能值。
- 用于 QUDA 的 509,609,448 B canonical round-trip LIME 可从现有 canonical HDF5 再生成，
  已从 `data/qcu_solver_test34_20261029/L24/` 清理。

### 2026-10-30 01:25 CST

simple CUDA worker 已修复 `--warmup/--repeat` 被解析但未执行的问题。每个样本先清零 x0，
在计时区外完成同步，再执行 cold solve；报告记录 warmup、逐次 steady 样本、median 和 MAD。

| ranks/grid | warmup (s) | steady samples (s) | median (s) | MAD (s) | iterations | native residual | status |
|---:|---:|---|---:|---:|---:|---:|---|
| `1 / 1x1x1x1` | `1.389318520` | `0.299408800, 0.305628530, 0.301689520, 0.302894160, 0.299466530` | `0.301689520` | `0.002222990` | 20 | `2.582429e-7` | PASS |
| `2 / 2x1x1x1` | `1.776467187` | `0.489892693, 0.460595469, 0.459965060, 0.461907550, 0.463353181` | `0.461907550` | `0.001445631` | 20 | `2.582687e-7` | PASS |
| `4 / 2x2x1x1` | `1.235514439` | `1.054773713, 1.033934803, 1.024669448, 1.024618929, 1.020558617` | `1.024669448` | `0.004110831` | 20 | `2.581189e-7` | PASS |

同任务强扩展只作本轮 L24 计时口径：2 rank 相对 1 rank 为 `0.653x`、效率
`32.66%`；4 rank 为 `0.294x`、效率 `7.36%`。`24^3x64` 上单次 solve 仅约
0.3--1.0 s，通信和固定开销占主导，因此不得据此推断更大格点的并行效率。

作业：`126264615`、`126264880`、`126264881`，均设置 `MODE=qcu`，未启动 QUDA。
每轮均为 20 iterations、`n_vcycles=1`、`coarse_iters=8`。按任务要求跳过独立真残差，
因此只声明 native full-operator residual PASS。

独立认证文件：`logs/qcu_solver_test34_20261029/L24/qcu-sample5-authentication.json`。
三档 solution SHA256 分别为
`e1f6bb6c35963c266b4a7b951eb2658cd9c521be95963ca56b7abf360dcd6ab6`、
`b662ccdb511fc2cb9f652a2830c16fa80f309d23ce68e4cb19d124032fe32499`、
`e86b88fe94829b4658e118f2653944e81022c0735baa5b4419488df5d511792d`。
