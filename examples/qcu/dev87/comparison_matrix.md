# dev87 对照单测矩阵 — PyQCU CUDA_C++ 多线程 MultiGrid vs quda / PyQUDA

任务：~auto-all（2026-08-25）。对象=PyQCU MG 链路；目标a=quda 对照（高优）；目标b=PyQUDA 辅助。
状态标记：`[ ] 待测 [x] 已测(结论见 dev87_report) [~] 部分/受阻 [-] 不适用(范围外)`。

## 双方版本锚点

| 侧 | 版本 | 说明 |
|---|---|---|
| quda 快照 | 上游 develop（NEWS 1.1.0 之后，含 MADWF/CCCL v3.1.4），源码零改动 | git-rep 内共享单仓 @9e5ed03 |
| PyQUDA 快照 | 0.10.54（pyquda_core/utils/io/plugins 四包） | pip 已装的是旧 0.3.2 egg（缺 libquda.so），需以 QUDA_PATH 重装新版 |
| PyQCU | main@9e5ed03（stab30 后） | libqcu.so 于 cpp/cuda/qcu/ |

## G1 规范场生成

| # | 功能 | quda 侧 | PyQCU 侧 | 状态 |
|---|---|---|---|---|
| 1.1 | 高斯规范场(种子) | gaussGaugeQuda(seed,sigma) interface_quda.cpp:5055 | applyGaussGaugeQcu(plan-1, argv[_SIGMA_], params[_SEED_]) | [~] 真实运行并做统计对照；同一 σ 的参数化语义不同 |
| 1.2 | 单位规范场 | constructIdentityGaugeField tests/utils/gauge_utils.cpp:319 (--unit-gauge) | 无独立入口（torch 单位阵可自构） | [~] QUDA 单位场算子锚点真实运行；PyQCU 用 torch 构造，非同一生成入口 |
| 1.3 | 随机 SU(3)(确定性) | constructRandomSU3GaugeField gauge_utils.cpp:419 | lattice.generate_gauge_field(seed,sigma)（Python torch） | [~] 双方真实生成并统计检查；生成分布不等价 |

## G2 Wilson dslash 与完整 M

| # | 功能 | quda 侧 | PyQCU 侧 | 状态 |
|---|---|---|---|---|
| 2.1 | Dslash 单奇偶 | dslashQuda(parity) | applyWilsonDslashQcu(plan0) | [x] 双方真实运行；按 parity/布局完成算子级验证 |
| 2.2 | Mat/MatDagMat | MatQuda/MatDagMatQuda | applyDslashQcu(Wilson+Clover 组合)；Wilson 全格点 fine_full_dslash_op 内部 | [x] M_q=(m+4)·M_p，cos=1 |
| 2.3 | CPU 参考一致性 | dslash_test.cpp vs host 参考 | testWilsonDslashQcu(run_test 路径) | [~] GPU/矩阵锚点已测，独立 CPU 全套尚未闭环 |

## G3 Clover 项与 Clover dslash

| # | 功能 | quda 侧 | PyQCU 侧 | 状态 |
|---|---|---|---|---|
| 3.1 | Clover 项构建(+逆) | createCloverQuda / cloverQuda(inverse=true) | applyCloverQcu / applyCloversQcu | [x] 差分法 cos=1.000000，scale=4.05 |
| 3.2 | Clover dslash 奇偶 | cloverQuda / dslash_test --test MatPC | applyCloverDslashQcu(plan2) | [x] 双方真实运行并完成差分/布局锚定 |

## G4 求解器族（MG 的直接依赖）

| # | 功能 | quda 侧 | PyQCU 侧 | 状态 |
|---|---|---|---|---|
| 4.1 | BiCGStab(Schur 奇偶预条件) | invert_test --inv-type bicgstab --matpc even-even | applyCloverBistabCgQcu（同语义基线） | [x] PyQCU/QUDA 双方真实求解；以 m+4=4.05 归一化后解差约 4.1e-7 |
| 4.2 | CG(MdagM) | invert_test --inv-type cg | applyWilsonCgQcu；python solver CG | [ ] |
| 4.3 | MR 平滑器 | inv_mr_quda.cpp（MG 默认平滑器） | pyqcu/solver/_mr.py（quda 思想）；C++ MG 支持设备端 MR 定步更新 | [x] 2L 普通 V-cycle 与 FGMRES/MG 预条件真实运行；大格 full-op 真残差 6.59e-7 |
| 4.4 | GCR 外层 | Solver::create GCR(+MG 前置) solver.cpp:67 | run_gcr()=FGMRES(10)⊕V-cycle（_MG_USE_GCR_） | [x] 1L/2L/3L 均真实运行；正确但本机比普通 MG 慢 |
| 4.5 | CA-CG/CA-GCR | inv_ca_cg/inv_ca_gcr | Python `_cacg.py`（CA-CG）；C++ `run_ca_gcr()`（4 阶块、双遍 MGS、Gram 小系统、奇异回退 FGMRES） | [x] CA-GCR 2L 真实运行；`atol=1e-6` 真残差 7.58e-7，严格 `atol=1e-8` 在 400 次块步后为 1.43e-7（fp32 平台） |
| 4.6 | 多移位 CG | invertMultiShiftQuda | python _multishift_cg.py（updateAlphaZeta 同源） | [ ] |
| 4.7 | deflate(eigcg/IRAM 低模) | newDeflationQuda/deflated_invert_test；CG 内 deflate 钩子 | _MG_USE_DEFLATE_=一次 V-cycle 校正作初值（弱对应）；python tr_lanczos 未接 C++ | [~] PyQCU 一次 V-cycle 初值真实运行；不是 QUDA eigcg/IRAM 同实现 |

## G5 MG setup（null 向量与粗空间）

| # | 功能 | quda 侧 | PyQCU 侧 | 状态 |
|---|---|---|---|---|
| 5.1 | null vec 生成(求解器逆迭代) | generateNullVectors multigrid.cpp:1275（setup-inv 可选族+自举） | give_null_vecs_mt tools/_multigrid.py:355（逆迭代/DDalphaAMG 配方） | [x] 质量口径：本 gauge ‖Sv‖/‖v‖=0.31-0.46（谱连续,非实现缺陷；quda 侧接口不外露同层诊断）|
| 5.2 | 多轮 setup+全局正交化 | :1365-1416 | 单轮+local_orthogonalize（块局部 GS） | [ ] |
| 5.3 | 特征向量 nullvec | generateEigenVectors :1646(IRAM/TRLM) | tr_lanczos 存在未接入 | [ ] |
| 5.4 | 自由场解析 nullvec | buildFreeVectors :1477 | 无 | [ ] |
| 5.5 | nullvec 持久化 | VectorIO <file>_level_N_nvec_M；dumpMultigridQuda | h5 缓存 logs/nullvec_cache + data/*.h5（lonv/hnn/hdg/sit） | [~] PyQCU 缓存读写与多层加载真实运行；QUDA VectorIO 未独立对照 |
| 5.6 | 块正交化 | BlockOrthogonalize(two-pass 二次 GS) | local_orthogonalize（批量 QR） | [x] Gram: 非对角≤2.4e-7, 对角=1±2.4e-7 |

## G6 Transfer P/R

| # | 功能 | quda 侧 | PyQCU 侧 | 状态 |
|---|---|---|---|---|
| 6.1 | Restrict R | Transfer::R transfer.cpp:292→Restrict 核 | applyMultigridRestrictQcu（细层 e=12 硬编码 FIX） | [x] 组件级真实运行；L2 相对误差 2.09e-7 |
| 6.2 | Prolong P | Transfer::P :260→Prolongate 核 | applyMultigridProLongQcu（同上硬编码） | [x] 组件级真实运行；L2 相对误差 6.40e-8 |
| 6.3 | 站点/spin 映射 | createGeoMap/createSpinMap CPU 表 | Python 层块结构约定 10 维 | [x] R·P=I + PᵀP 正交（经 verify_nullvecs）|

## G7 粗格算子

| # | 功能 | quda 侧 | PyQCU 侧 | 状态 |
|---|---|---|---|---|
| 7.1 | 精确 Galerkin 粗化 Y,X | CoarseOp coarse_op.cuh calculateY:974(GPU/CPU 双路) | torch 探测 build_stencil(_mt) 33-tensor，经 set_ptrs 槽 30+ 传入 | [x] 重复回归约 7.5–9.5e-7（A_c≈PᵀSP）|
| 7.2 | 二次粗化 | CoarseCoarseOp | build_stencil_mt lvl≥2（CudaCoarseSchurOp 续探） | [ ] |
| 7.3 | 预条件粗算子 Ŷ,X⁻¹ | calculateYhat + DiracCoarsePC(eo) | 宽版粗 Schur 形式直接存奇子格算子（结构不同、作用等价性需验证 A_c≈RSP） | [ ] |
| 7.4 | 粗 dslash 核 | ApplyCoarse dslash_coarse.cuh(dagger/parity/clover 全组合) | multigrid_coarse_dslash[_wide].cu（窄/宽两版） | [x] 窄/宽核真实运行；相对误差 2.65e-7/5.01e-7 |
| 7.5 | 粗核基准计时 | multigrid_benchmark_test | bench 脚本 coarse 计时（dev84 check_ms 剖析） | [x] 组件基准真实运行（窄约 0.775 ms，宽约 1.881 ms） |

## G8 MG 求解循环

| # | 功能 | quda 侧 | PyQCU 侧 | 状态 |
|---|---|---|---|---|
| 8.1 | V-cycle 递归 | MG::operator() multigrid.cpp:1131(pre→R→coarse→P→post) | v_cycle lattice_clover_multigrid.h:1520(热启动+图回放+守卫) | [x] 1L/2L/3L 真实端到端运行 |
| 8.2 | W/F-cycle | cycle_type 经 RECURSIVE 粗解器包装 | `v_cycle()` 递归 `coarse_correction()`，W=两次递归 W，F=递归 F+V | [x] 3L W/F 真实运行；无 NaN/非法访问，full-op 真残差均约 1.36e-6 |
| 8.3 | 平滑器族 | MR 默认；Chebyshev/CA-GCR/BiCGStabL 可选；nu_pre/post 分离 | 粗层支持 CG/MR/Chebyshev；`run_ca_gcr()` 提供 CA-GCR 外层；外层 `run_bicgstab_l()` 固定 L=2 | [x] CG/MR/Chebyshev/CA-GCR/BiCGStabL 均已闭环；BiCGStabL 当前固定 L=2 |
| 8.4 | 粗解器 | PreconditionedSolver 包（coarse_solver[level] 可选族+粗格 deflation 注入） | bistabcg_iter_coarse/coarse_solve_fused(cooperative) | [x] 2L/3L 粗解路径真实运行 |
| 8.5 | K-cycle | coarse_solver=RECURSIVE | `k_cycle_correction()`：递归 K-cycle 作为短重启 FGMRES(m=2) 右预条件器 | [x] 3L K-cycle 真实运行；`atol=1e-5` 下 full-op 真残差 5.97e-6 |
| 8.6 | verify 自检五项 | MG::verify :745(P·R 正交/Galerkin/厄米等) | verify_nullvecs(Python 测试层四重诊断)+run_test 全残差 | [~] PyQCU 四项组件诊断+full-op 真残差已测，尚非 C++ 五项同接口 |
| 8.7 | 收敛判据/可靠更新 | reliable_delta/pipeline/累加器流水线 | r0_ref 锚定+自适应门控；dev87 加周期真残差刷新(每50迭代)+相对停机 | [x] 对齐验证 |

## G9 驱动生命周期与工程面

| # | 功能 | quda 侧 | PyQCU 侧 | 状态 |
|---|---|---|---|---|
| 9.1 | 建/销毁层次 | newMultigridQuda/destroyMultigridQuda | applyInitQcu→applyCloverMultigridQcu→applyEndQcu(set_ptrs 槽位生命周期) | [x] 双方真实建/销毁并重复运行 |
| 9.2 | gauge 更新后薄/全刷新 | updateMultigridQuda(thin/full) :2946 | 无（重建全部） | [ ] |
| 9.3 | 逐层混合精度 | 全字段 per-level precision/sloppy | `_MG_LEVELn_DATA_TYPE_` 逐层解析；c64/c128 擦除存储；restrict/prolong 显式 cast kernel | [x] 2L/3L 单 rank 与 2-rank `c64→c128` 真实运行 |
| 9.4 | 外部初值热启动 | use_init_guess | params[_MG_USE_INIT_GUESS_](57)：跳过 x_o 清零/随机化，r 真算；双求解器类支持 | [x] 大格 WARM 0.198s vs COLD 1.412s，解一致 3.7e-6 |
| 9.5 | 多右端项 | invertMultiSrcQuda | 无 C++ 批量（tools/_bistabcg_batch 为 Python 层） | [ ] |
| 9.6 | MPI 分布 | 分布式粗格+ghost 打包 | rank-local 粗格/粗算子；33 点 stencil 的 32 邻居 host-staging halo；粗层与 fine 层点积全局 Allreduce | [x] 2-rank 粗算子等价性与 MG/BiCGStabL 冒烟通过；阻塞通信，未声明 overlap/NVSHMEM 性能 |
| 9.7 | tensor-core MMA | *_use_mma[level] 全链路 | 无（平台 sm_70 SIMT mma 变体存在但未实现） | [ ] |
| 9.8 | setup 位置可编程 | setup_location/location per-level CUDA/CPU | 固定 GPU | [ ] |

## G10 端到端（目标b：PyQUDA 辅助）

| # | 功能 | quda/PyQUDA 侧 | PyQCU 侧 | 状态 |
|---|---|---|---|---|
| 10.1 | 同一 gauge 文件端到端 MG solve | pyquda.init+loadGauge+getClover(multigrid=..)+invert | run_qcu_mg(data/*.h5 缓存 stencil) | [x] 双方收敛 |
| 10.2 | 输出数值一致性 | 解向量/残差历史 | 解向量/残差历史 | [x] 缩放(m+4)后 rel=8.63e-6（两侧容差不同） |
| 10.3 | 性能对照 | quda 计时(V100) | PyQCU 计时(V100) | [x] 见报告§十九（含缓存/容差/精度口径注记） |

## 本轮真实运行证据（2026-08-27）

以下结果来自实际 CUDA 进程，不是静态代码检查；`true_residual_rel` 是
Python full Wilson+Clover 算子重新作用后的相对残差，`rel_diff_vs_bistabcg`
是与同一 PyQCU BiCGStab 参考解的差异。大格统一为
`16×32×32×48, m=0.05, atol=1e-6, mg_grid=[2,2,2,2]`，粗算子为
`data/L16x32x32x48_lv1_E12_nvi1_t1e-2.h5`。

| PyQCU 配置 | 墙钟(s) | 参考解差异 | full-op 真残差 | 结果 |
|---|---:|---:|---:|---|
| 1L | 1.378 | 6.55e-6 | 3.92e-7 | 通过 |
| 2L, E=12 | 1.412 | 8.57e-6 | 6.59e-7 | 通过 |
| 2L + MR, E=12 | 1.445 | 8.57e-6 | 6.59e-7 | 通过（MR） |
| 2L + deflate | 1.350 | 6.77e-6 | 4.83e-7 | 通过 |
| 2L + warm（cold→warm） | cold 1.412 / warm 0.198 | warm 3.66e-6 | warm 4.41e-7 | 通过 |
| 2L + GCR/FGMRES | 5.015 | 1.27e-6 | 6.86e-7 | 通过 |

同一批次的 warm 约为 cold 的 `7.1×`；2L 相对 1L 未获得额外加速
（`0.976×`），GCR 正确但约为普通 2L 的 `0.28×`。3L 的缓存链也在
`8×8×8×16, E=24/E=24` 上真实运行，真残差为 `5.94e-7`；目前只有
小格 3L 缓存验证，不能据此宣称大格 3L 性能收益。

本轮新增 MR 验证：同一大格配置下，MR 为 `1.445 s`，相邻 CG 基线为
`1.420 s`（单次运行约慢 `1.7%`，计时波动下不作为性能结论）；两者均为
68 次外层迭代、2 次 V-cycle，且输出真残差与解差相同。小格
`4×4×4×8` 上普通 MR 与 FGMRES/MR 也分别真实进入粗层，耗时
`0.145/0.250 s`，真残差为 `5.46e-7/6.58e-7`。

组件级 `restrict/prolong/coarse dslash` 的 L2 相对误差分别为
`2.09e-7/6.40e-8/2.65e-7（窄）/5.01e-7（宽）`，窄/宽粗核中位耗时
约 `0.775/1.881 ms`；最新回归的 Galerkin 误差为 `9.47e-7`，块正交 Gram 非对角
最大值为 `2.38e-7`。

QUDA/PyQUDA 对照使用独立进程，统一大格的直接 Clover 解在
`m+4=4.05` 缩放后相对差为 `3.91e-7`；最新 MG 端到端解差为
`8.63e-6`。当前 QUDA MG 实测 `setup=270.43 s, solve=82.12 s,
9 次迭代`，而 PyQCU 使用离线 HDF5 粗算子，且两侧停止容差/精度策略
不同，因此该耗时不能作为无条件的算法优劣结论。多卡 `P100×2` 与同型号
单卡的一致性全部通过，但并行比为 `0.970×`，本平台没有观测到加速。

结果文件：`out/qcu_mg_matrix_*.json`、`out/component_cuda.json`、
`out/multigpu.json`、`out/quda_clover_{solve,mg}.json`。

## 后续功能候选清单（按价值/可行性排序）

| 序 | 缺口 | 依据 | 备注 |
|---|---|---|---|
| S1 | 动态 thin update（规范场变化后） | G9.2 | **待做**：当前策略为安全地重建全部层次 |
| S2 | setup 多轮+全局再正交 | G5.2 | **缓**：本 gauge 质量受谱限制；现有单轮 local setup 已完成闭环 |
| S3 | C++ 版完整 `verify()` 五项接口 | G8.6 | **待做**：当前已有 Python 四项组件诊断与 full-op 真残差 |
| 范围外 | MMA/MADWF/NVSHMEM | 平台与体量 | 当前未实现；分布式粗格已落地，但采用阻塞 host-staging halo |

## 构建与运行资产

- quda 构建：/tmp/opencode/quda-build-sm70（RELEASE, sm_70, MPI ON, MULTIGRID ON,
  DW/staggered/twisted/laplace OFF, NVEC_LIST=12,24,48），安装前缀 /tmp/opencode/quda-install
- PyQUDA 安装：pip install ./pyquda_core . （QUDA_PATH=<install>，rpath 烧入）
- 数据：data/gauge_16x32x32x48_m0.05_seed42_c64.h5（统一格子）及 L16x32x32x48_lv1_E12_nvi*_t1e-2.h5 nullvec 缓存
- GPU 规约：单卡 V100-32GB(sm_70)；多卡 P100×2(sm_60，如需另建 build-sm60)
