# AGENTS.md — docs

PyQCU 参考文档。

> 2026-09-27 迁移：顶层 `examples/` 已迁入 `pyqcu/testing/`。本目录旧报告中若
> 出现 `examples/...`，应映射为 `pyqcu/testing/...`；现行文档已统一使用新路径。

当前 `docs/` 顶层共有 **59 份 Markdown 文档（含本索引）**。下表只维护文件职责；
权威级别、唯一主责技能和提取边界见
[`skills/pyqcu-docs/references/project-docs-map.md`](../skills/pyqcu-docs/references/project-docs-map.md)。

## Read the Docs 源文

`docs/en/` 与 `docs/zh/` 是 PyQCU Read the Docs 双语站点的唯一源文，各含
`.readthedocs.yaml`、`conf.py`、`index.rst`、`requirements.txt` 与
`overview.md`、`usage.md`、`solvers.md`、`performance.md`、`install.md`、
`troubleshooting.md` 六页。沿用 PyQCD 的
约定：站点根不设 `.readthedocs.yaml`，两份配置分别位于 `docs/en/` 与 `docs/zh/`，
在 RTD 网页按语言指定；项目 slug 固定为 `pyqcu`。站点视角是已安装的
`pip install pyqcu` 包，不引用仓库内 `pyqcu/testing` 路径；确需源码仓库的段落必须
显式标注。`docs/` 根目录的 52 份研究报告与 `.tex` 不在站点源文内。

## 核心操作规范

| 文件 | 内容 |
|---|---|
| `ORGANIZATION.md` | `docs/`、`data/`、`logs/`、正式 `pyqcu/test/` 与非正式 `pyqcu/testing/` 的统一分类规则、垃圾清理与构建树例外 |
| `dims.md` | 维度命名方案（`s`=spin、`c`=color、`d`=direction、`p`=parity、`x/y/z/t`=时空）。`ccdxyzt`、`scxyzt`、`psctzyx` 等约定 |
| `env.md` | Python 环境设置；`env.sh` 导出当前仓库优先的 `PYTHONPATH`、CUDA/DCU/KUPL 库路径与保守 KUPL 默认值，并配置可用 QUDA 安装树 |
| `install.md` | 安装指南 — CUDA/DCU/KUPL 后端选择和 install.sh 工作流 |
| `dcu_build_backend.md` | DCU/DTK 构建选路、ROCm/HIP 验证、测试、生产限制和故障排查 |
| `kupl_only_deployment.md` | 纯 KUPL 无 CUDA 安装、旧版 PyTorch 兼容、环境检测、生产 BiStabCG/真实 Legacy-MG、HBM、升级和故障排查手册 |
| `examples.md` | 测试与示例迁移映射、运行入口与结果解读 |
| `kupl_kunpeng_guide.md` | KUPL CPU 后端在鲲鹏上的构建、MPI/OpenMP 绑定、NUMA/HBM 策略、环境检测、生产算法选路、测试与性能证据规范 |
| `kupl_kunpeng_progress_20261110.md` | 2026-11-05 至 11-10 鲲鹏真实环境进展：计算节点/登录节点硬件实测、HBM 早期失败与后期成功、计算节点与登录节点性能数据、dev108 以来代码改动、已否决路线与阻塞项（C 级时点证据，未打 tag） |
| `kupl_kunpeng_hpc_config_20261112.md` | 2026-11-12 鲲鹏 HBM 高性能配置实测：拓扑与绑定变量、MPI 映射、调度器用法、超时协议、cn19699/cn19698 实测表与待补项（C 级时点证据，未打 tag） |
| `kupl_opt_handover_20261112.md` | KUPL QCU 极致优化接手文档：远端/本地环境、已应用补丁与 md5、门控变量、审计结论、基准现状、已知坑与复现命令、待办（C 级工作文档，未打 tag） |
| `kupl_kunpeng_matrix_20261115.md` | 2026-11-15 KUPL 鲲鹏全矩阵驱动（`pyqcu/testing/kupl/test_kunpeng_all.py` + `submit_kunpeng_all.sh`）：7,680 行展开与 5,760 个去重输入组合的口径、`list/prepare/prewarm/solve/merge/run/validate` 七模、`--resume/--skip-failed/--shared/--task-model single`、94 列 CSV 字段与判读（含 `gauge_file_dtype` 按文件名误标的量化）、`dsub -s`/`-T`/单任务内层 `mpirun`/`-x job`/`testusers_rebuild` 账号 cap 边界、42 行快照（37 solve，31 PASS/6 FAIL；L24 c64 32×18 t-finest+shm `0.2405 s`、balanced `0.3497 s`、c128 `0.4756 s`、快照内 L32 0 PASS 与快照后 `82ba09ec` 的单条旁证）与失败清单、旧 `0.17 s`（`1e-4`/11 it、有效容差差 244.1×）的可比性红线；§12 输入不可变/超时语义/写原子性（只读审计结论与 R1–R7 规则、复核命令）、§13 启动成本与 loop 原型（A 39.0 s + B 91.7 s、C 113.9 s）。末节合并原可选 runbook（环境/构建/提交/常见错误）。C 级时点证据，未打 tag |
| `kupl_hdf5_source_gauge_rootcause_20261115.md` | 2026-11-15 生产 HDF5 输入不可读根因：L32 `source_..._ones.h5`（201,375,008 B/11:20:03）与 L24 `source_..._point.h5`（84,959,592 B/11:07:12）在 `h5py` open 阶段报 `bad object header version number`、26 s 后的 `ones_v2` 与 2 分 57 秒后的 L24 `ones` 可读；尺寸按 `[4,3,x,y,z,t]` 复算证明"全尺寸≠写完"；复用只按存在性放大损害；写侧 `temp+fsync+rename`+shape/dtype 回读与读侧 `h5_readable` 闸门的修复与四道验收门（门 3 现为"部分达成"：快照后 `82ba09ec` 满足 32×18 一半）；L24/L32 两份 `real` gauge 磁盘上都是 `complex128`（509,614,080 B / 1,207,965,696 B）；脚本内无 `.corrupt-*`、8 文件 `os.fsync` 计数为 0、`real` gauge 会静默降级为 identity gauge 的审计补强。C 级时点证据 |
| `kupl_matrix_scope_20261115.md` | 2026-11-15 KUPL 鲲鹏矩阵本轮范围裁定：MPI h5py 只做接口检查（`h5py 3.7.0`/`HDF5 1.10.6`、`get_config().mpi=False`、无 `mpio`、无一行 `hdf5_mpio_read=true`、ranked cache 不列入计划、不重编译/安装）、6 个放弃轴（thread 9/seed24/random gauge/random 源/point 源/r50）与保留项（16×36 三行、MG1/r10、L32/c128、MG+shm）、L32 source 语义（`source_..._c64.h5` 是 random seed42；`ones`/`ones_v2` 同为 `9cdcfc82…f55a` 全 `1+0i`；`source_..._c64_point.h5` 是 `dc427e47…` 的 point 源）、6 条 L32 FAIL 属"首次有效运行"（快照后 `82ba09ec` 的 point+c128+mg2 PASS 不能抵）、全输入身份表、cache identity 全字段（`cache_sha256`/`cache_identity_sha256`/L24 c128 `48.191 s`）与重签（L32 c64→`70bbe621…`、L24 c64→`8a482d6a…`）、31 条 PASS 不重跑判据、`bg-26/27/28/32` kill 与 pending 未能 CLI 取消的队列状态。C 级时点证据，未打 tag |
| `kupl_l24_c64_mg2_shm1_boundary_20261115.md` | 2026-11-15 L24 c64 `mg2` + `shm=1` 的 balanced/t-finest 边界测量：8 格矩阵 7 PASS / 1 FAIL（`32×18 balanced 2 2 2 4 r5 shm1` 的 `KUPL shared halo handshake timeout: publish` → `MPI_ABORT` rank 30、`elapsed=271.437 s`、无 report）；同 config 在独立 outdir 复测 **2/2 PASS**、同 restart 下 shm1 与 shm0 真残差逐位相同 → **1 FAIL / 2 PASS 属不稳定失败**，`known_skips.json` 不写、只留 `known_skips.candidate.json`（默认 disabled）；不得整体禁 `balanced`/`shm1`、不得静默回退 `shm=0`；`--resume` 判重读 `outdir/rows/<run_id>.json` 而非 `--csv`，重测必须换新 outdir。B 级时点证据，未打 tag |
| `kupl_performance_20261115.md` | 2026-11-15 KUPL 性能对照汇总（stab64 `2c07cf94` 对 dev143 `4f60e348`，相距 47 个提交）：字段口径与可比性红线（`atol=1e-4` 非 strict 的有效门槛 `2.44140625e-4` 与矩阵批次 `1e-6` 相差 244.1×，旧 `0.17 s`/11 it 不可与 `0.2405`/`0.3497` 直接相除）、L24/L32 的 BiCGStab 与真实 MG、cache 构建、带宽对照、c128 缺口（历史 x86 值不能相除；2026-11-15 矩阵批次仅补 L24 c128 `0.4756 s`/31 it/n=1 一个点，L32 c128 只有快照后 1 条 PASS）与"不得做的比较"清单、审计状态更新（现 `PASS docs=53 skills=36`）；数值为已提交文档与 `logs/**` 的只读转抄，B/C 混合证据，未打 tag |
| `kupl_l24_perf_20261113.md` | 2026-11-13 KUPL L24（`24³×64`、c64）单节点实测：cn19699/cn19650 的 8 条用例（最优中位 `0.154883 s`，8 样本，32r×18t + 共享 halo）、`OMP_WAIT_POLICY` 同节点 A/B 的 1.28×、dslash/clover/reduce 分项与 7.96 GB/s·rank、STREAM triad 18.3–20.1 GB/s·rank 峰值口径、节点差异与 MG cache 阻塞（C 级时点证据，dev141） |
| `kupl_vs_cuda_algorithms_20261114.md` | KUPL-QCU 与 CUDA-QCU 的算法级差异：布局（component-major vs 半 T 压缩）、gamma 预投影 18 MAC vs 手写展开、site-lane vs thread-per-site、shm halo vs MPI、双容差真残差 vs 递推停机、Legacy/Strict MG 菜单、Gauss gauge 正态 vs 均匀，以及“高带宽/低算力”如何决定这些选择（B 级源码核对） |
| `kupl_mpi_scaling_20261114.md` | KUPL 多 rank/多线程扩展性与断崖巡查：L24 的 halo/计算比、带宽 roofline（dslash 已近 600 GB/s 量级）、每迭代 2 次 Allreduce 的固定成本、`TH_max` 公式与静默超订阅、t 方向 32 rank 几何上限与路径门限；§6.4 记录 2026-11-14 L24 Legacy-MG cache（1 rank 构建 `493.63817879895214 s`、`143,335,424 B`）与 32 rank 复用扫描（`restart=1` `0.135579 s`，单样本） |
| `kupl_kunpeng_node_survey_20261114.md` | 2026-11-14 18 个鲲鹏计算节点轻量核验：592 CPU、16 CPU NUMA + 16 内存 NUMA、565 GiB、1.55 GHz 当前频率 / 2.0 GHz 上限、userspace governor；逐节点表与 1.55↔2.0 判定规则 |
| `qcu_l24_solver_timing_20261111.md` | L24（`24³×64`）全配置耗时台账：7 个环境 × BiCGStab/Legacy-MG/Strict-MG × c64/c128 的 solve 时间、迭代、残差、样本数与证据等级；含弱扩展外推与相对 89.13 s 基线的加速比（C 级时点证据，未打 tag） |
| `qcu_l24_environment_bandwidth_20261111.md` | 上述环境（`cn10579`/login/控制节点/Xeon/V100/A100/DCU）的硬件字段与带宽实测对照：LX2 STREAM 全表、DDR↔HBM 实测比、每线程饱和点、聚合效率、3 处 Add/Triad 抄录更正，以及 `16×t24` 的**用途相关**语义（对全节点容量是下界，对 384 线程可实现带宽是乐观上限） |
| `kupl_kunpeng_figures_20261110.tex` / `.pdf` | 上述进展的同源图表集（9 页、13 图 2 表）：绝对耗时与加速比、作业台账与几何不可比性、瓶颈归属与 roofline、STREAM 带宽、HBM 绑定统计、NUMA→片上内存映射、登录节点 A/B 与分格扫描、setup/初始化成本、证据台账。两遍 `xelatex` 通过 `Overfull=0`、`Float too large=0`、`Missing character=0` |
| `profiler.md` | 剖析指南 — torch.profiler 与 Perfetto |
| `cuda_execution_topology.md` | CUDA 生产拓扑：一 Python 进程 = 一 MPI rank = 一 GPU/DCU/GPGPU；`MultiGpuMultigrid` 测试专用例外 |
| `torch_backend_layout.md` | Torch/CANN 实现目录、顶层公共层、兼容门面与 multigrid 选择器边界 |
| `api_contract.md` | `pyqcu.api` 的 P0 数据模型/Protocol 与 P1 LinearOperator、统一 Krylov、Torch 迭代和 CUDA/KUPL 原生整次 solve 边界 |

## MultiGrid 技术参考

| 文件 | 内容 |
|---|---|
| `qcu_unified_backend_strategy_20261023.md` | Torch/CUDA/KUPL 的 Legacy/Strict 六实现对比、统一组件架构及扩展 dslash/BiCGStab/CG 方案 |
| `multigrid_setup_cpp_migration.md` | Legacy MG 的 null-vector、局部正交化、33-tensor stencil 从 Python 迁移到 CUDA/KUPL C++ 的已完成内核边界、混合编排与验收门 |
| `qcu_quda_multigrid_algorithms.md` | QCU legacy MG 与 QUDA/PyQUDA MG 的算法组成、流程图、默认参数和并排对照 |
| `qcu_quda_multigrid_subalgorithms.md` | QCU/QUDA MG 子算法的独立流程图、数学摘要、默认配置与同类算法优劣 |
| `qcu_quda_multigrid_coarse_operator.md` | QCU legacy 33-tensor 与 Strict/QUDA `X/Y/Yhat` 粗层 dslash 表示辨析及源码依据 |
| `mg_cache_sizes_24x24x24x64_20261023.md` | 真实 L24、c64、两级、`nvec=12` 下 Legacy 33-tensor、Strict schema-v2 cache 与 QUDA null-vector 的逐项尺寸推导 |
| `multigrid_legacy_strict_comparison_20261022.md` | Torch/CUDA/KUPL 的 Legacy/Strict 算法、接口、生命周期与既有证据简要对比（2026-10-22） |

## 性能、状态与风险证据

| 文件 | 内容 |
|---|---|
| `three_way_benchmark_20261022.md` | CUDA QCU / KUPL QCU / QUDA 三方性能测评与并行扩展报告（2026-10-22） |
| `qcu_six_way_benchmark_20261023.md` | Torch/CUDA/KUPL 六版本、QUDA 对照、真实 LIME 前缀、coarse 执行证据与同窗口性能结果（2026-10-23） |
| `qcu_mg_benchmark_decisions_20261023.md` | 大格 Strict 初始化的 benchmark SKIP 决策、适用范围与证据边界（2026-10-23） |
| `qcu_mg_refactor_validation_20261023.md` | 历史 16⁴ 六项矩阵及 L24 深度回归；该格点已被当前 `8x8x8x16` 正确性门替代，仅保留时点证据（2026-10-23） |
| `qcu_mg_unified_interfaces_and_reuse_20261029.md` | 当前工作树的三后端统一接口、Legacy/Strict cache 复用、L32 正式入口、QUDA 构建经验与图表规范（2026-10-29，未打 tag） |
| `qcu_mg_mpi_cache_builder_20261029.md` | Legacy-MG 多 MPI cache builder：分布式 null-vector、overlap stencil、1/2/4-rank slab 验证、Torch CPU 小格门和 KUPL 跨 cache 边界（2026-10-29，未打 tag） |
| `qcu_mg_performance_ledger_20261029.md` | 用户要求的实时性能账本：按批次记录求解器、完整参数、输入/库 SHA、耗时、迭代、真残差、cache、作业审计与失败记录（2026-10-29，未打 tag） |
| `qcu_mg_performance_report_20261029.tex` | 当前工作的 LaTeX 主报告：统一接口、CUDA/KUPL/QUDA 数据、失败边界和证据路径（2026-10-29，未打 tag；当前环境缺少 XeLaTeX） |
| `qcu_quda_solver_dcu_20261029.md` | DCU 上 QCU/QUDA L24/L32/L48 总报告：构建 provenance、L24 Real Legacy-MG 正式采样、L32 cache 阻塞、L48 用户暂停与恢复边界 |
| `qcu_quda_solver_dcu_l24_20261029.md` | L24 CUDA QCU Legacy-MG 的 1/2/4-rank 正式采样、cache 认证、QUDA runtime 阻塞和 native residual 边界 |
| `qcu_quda_solver_dcu_l32_20261029.md` | L32 distributed cache 的 1--32-rank 恢复尝试、超时/参数门/共享 I/O 阻塞及无性能结果的状态记录 |
| `qcu_quda_solver_dcu_l48_20261029.md` | L48 容量门、真实 Legacy cache shape/payload 预算、用户暂停状态和恢复条件 |
| `pyqcu_test_l48_remote.md` | 2026-11-17 `pyqcu.test` L48 双节点正式端口设计（P）：目标/非目标、分层、输入契约、执行模型、`driver.run_row` 契约、验收门与开放问题；§0 状态表区分已验证/TODO，§0.1 首批结果（c64 BiCGStab 32 rank 独占节点 `1.6646 s`/11 it/`7.126e-5`（`2152757`）、16 rank 共享节点 `183.0 s` 属病态不作基线、c64 Legacy-MG 16 rank wall `110.35 s`+builder `1741.88 s`/cache `1.276 GiB`、c128 `2153005` 与两节点 `2153008` PENDING、32r dslash/clover UNVERIFIED、CG 组退役、Strict-MG 排除、cache 只认同 identity 且 32r 不重建），§2.4 远端基线（`login02`、非 git 的 0.1.15 快照 + 0.1.17 Python 覆盖、`--check` 18/18、HMPI 单 rank、L48 转换 41 s、`dsub` 两节点 PENDING）；设计未实现，未打 tag |
| `kupl_l48_mg_cliff_20261123.md` | 2026-11-23 KUPL Legacy-MG 两节点性能断崖排查（C）：16r `2.9444 s` / 32r `102.4651 s` / 64r `92.8595 s`、层级迭代同为 `level0=18 level1=4 accepted=4 fallback=0`（迭代数与 fallback 已排除）、三个源码候选（粗层 32 方向整面 halo、MG 归约走 `MPI_COMM_WORLD`、每样本 init+verify）、以及当天 4 rank 起纯 MPI 探针即挂住（`PROBE_OK` 仅 2 rank）证明是站点 fabric 失效而非 PyQCU 路径；断崖未修复，无回归证据 |
| `pyqcu_test_l48_two_node_matrix.md` | 2026-11-17 L48 两节点 tier-1 矩阵（P，只有定义）：5 个正式 case（`kupl/l48/bicgstab/{c64,c128}`、`kupl/l48/legacy-mg/{c64,c128}`、`kupl/l48/io-residual/c64`）+ 2 个 dslash/clover harness 入口，`2x2x2x4` / 32 ranks / 2 nodes，全部 `evidence=UNVERIFIED`；CG 退役、Strict-MG 排除；`launch_plan()` 只返回数据结构，不启动进程 |
| `pyqcu_test_audit_risk_20261117.md` | 2026-11-17 `pyqcu.test`/L48 远端 P0/P1 风险登记：无文件驱动原生 solve、无 `driver.run_row`、无跨节点 launcher、CUDA rank-grid 仅 SKIP、MPI abort/死锁与 `_SET_INDEX_` 游标风险、KUPL 驱动无 L48、`submit_kunpeng_all.sh` 写死 `-nn 1`/`mem=8G`、CSV/Excel parity 未验证；全部标“未修复”，不声称 L48 solve 或两节点 PASS；C 级时点证据，未打 tag |
| `kupl_remote_l48_experience_20261117.md` | 2026-11-17 L48/LX2 远端会话经验（C）：访问/运行时（`login02`、0.1.15 非 git 快照 + 0.1.17 Python 覆盖、`--check` 远端 18/18 与本机 20/20（17:55；17:47 为 19/19）、HPCKit 26.1.RC1、Python 3.10.9）、3.10 语法门（PEP 646 `source[*tuple]` 与 `ast.parse(feature_version)` 的缺口）、L48 LIME→HDF5 41 s 与 c64 dtype 误标/bit-exact cast/SHA、HDF5 384 B 基本读与 3.56× 放大、`dtfs` 33–36 s 停顿、32r `1.6646 s` vs 16r 病态 `183 s`、16r “MG” 实为 fine BiCGStab fallback、cache identity/约 24 次 break-even、两节点 request/rankfile/`CCS_DRUN_RANK_FILE`、发布红线（0.1.17/test37、十轮、`bug45` 历史例外）；未验证项显式标注；未打 tag |
| `pyqcu_test_l48_review.md` | 2026-11-17 L48/远端发布前十轮终审清单（C，不是发布记录）：R1–R10 的目标/命令/验收、四条硬性红线（0.1.17/test37 不可覆盖、`bug45` 历史例外、四个缺口、十轮全过才发布）、当前快照 R1/R2/R6/R7/R9 FAIL、R3/R4/R5 部分 PASS、R8 的 docs 计数已修复（PASS）、R10 UNVERIFIED；不执行发布、不改版本/标签 |
| `legacy_vs_strict_mg_status_20261022.md` | Legacy/Strict CUDA 与 KUPL MG 现有 JSON/stdout 的完成状态、数值边界和未验证项（2026-10-22） |
| `legacy_vs_strict_mg_memory_20261023.md` | Legacy/Strict CUDA MG 的设备级运行时显存峰值、测量边界、差异来源和未验证项（2026-10-23） |
| `progress_until_20261022_worktree.md` | 当前工作区续接进度总结（2026-10-22） |
| `profiler_solver_performance_history.md` | PyQCU / QCU / QUDA 求解器历史性能、失败与证据索引（2025-08-29 至 2026-10-22） |

## 历史与治理归档

| 文件 | 内容 |
|---|---|
| `progress_kupl_cuda_mg_20261021.md` | KUPL/CUDA 多卡与多重网格进度总结（2026-10-21） |
| `multigrid_deprecation_audit_20261022.md` | MultiGrid 弃用、历史 CUDA 入口、占位目录和兼容别名的证据审计（2026-10-22） |
| `form_pyqcu_20261020.md` | 2026-10-20 form 整理记录 |
| `form_pyqcu_20261022.md` | 2026-10-22 form 整理记录（tag 名路径整改与目录索引补齐） |

已归档的 tag 阶段报告以 tag 对象为准，通过 `git show <tag>:<path>` 或临时 worktree 读取。
当前任务的活跃草稿、未提交阶段结果和逐轮进度写入 `logs/**`。已经提交、具有稳定
status/date/provenance 的跨项目报告与时点证据可以保留在 `docs/**`，但必须明确
“未打 tag/特定工作树”属性，不能升级为永久结论；`docs/**` 不保存活跃工作草稿。
任务计划不作为 `docs/` 当前文件保留。

## 质量门

- 新增、重命名或删除文档时，同一批次更新本索引、
  `skills/pyqcu-docs/references/project-docs-map.md` 和
  `skills/pyqcu-docs/references/extracted-guidance.md` 的结论覆盖行。
- 文档/skills 改动提交前运行
  `python -B skills/pyqcu-docs/scripts/audit-docs-skills.py`，并完成本地链接、代码围栏、
  `git diff --check` 与 form 审计。

## 目录分类

- `docs/` 顶层只保存正式文档：`.pdf`、`.tex`、`.md`；
  `张鑫 508应用测试报告-PyQCU/` 是解包 Office 资源，按文档资产保留。
- `docs/data/` 保存从 `data/**` 提升出的 `.md`/`.pdf`/`.tex`。tag 阶段报告
  先进入待 tag 提交，tag 创建后在后续清理提交中删除，不保留
  `logs/<tag>/` 工作区副本或旧路径软链接。
- LaTeX 辅助文件、`.report-build-*`、逐页渲染图、agent/codex 会话和临时缓存
  均属于可再生产物，直接删除，不纳入归档。
- Office 文件及非文档资源不在 `docs/` 顶层保存；现有 `.docx`/`.xlsx` 的实体
  位于 `data/docs/`，调用方直接引用实体路径。
