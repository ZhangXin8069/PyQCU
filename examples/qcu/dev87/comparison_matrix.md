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
| 1.1 | 高斯规范场(种子) | gaussGaugeQuda(seed,sigma) interface_quda.cpp:5055 | applyGaussGaugeQcu(plan-1, argv[_SIGMA_], params[_SEED_]) | [ ] |
| 1.2 | 单位规范场 | constructIdentityGaugeField tests/utils/gauge_utils.cpp:319 (--unit-gauge) | 无独立入口（torch 单位阵可自构） | [ ] |
| 1.3 | 随机 SU(3)(确定性) | constructRandomSU3GaugeField gauge_utils.cpp:419 | lattice.generate_gauge_field(seed,sigma)（Python torch） | [ ] |

## G2 Wilson dslash 与完整 M

| # | 功能 | quda 侧 | PyQCU 侧 | 状态 |
|---|---|---|---|---|
| 2.1 | Dslash 单奇偶 | dslashQuda(parity) | applyWilsonDslashQcu(plan0) | [~] 经 Mat 级验证 |
| 2.2 | Mat/MatDagMat | MatQuda/MatDagMatQuda | applyDslashQcu(Wilson+Clover 组合)；Wilson 全格点 fine_full_dslash_op 内部 | [x] M_q=(m+4)·M_p，cos=1 |
| 2.3 | CPU 参考一致性 | dslash_test.cpp vs host 参考 | testWilsonDslashQcu(run_test 路径) | [ ] |

## G3 Clover 项与 Clover dslash

| # | 功能 | quda 侧 | PyQCU 侧 | 状态 |
|---|---|---|---|---|
| 3.1 | Clover 项构建(+逆) | createCloverQuda / cloverQuda(inverse=true) | applyCloverQcu / applyCloversQcu | [x] 差分法 cos=1.000000，scale=4.05 |
| 3.2 | Clover dslash 奇偶 | cloverQuda / dslash_test --test MatPC | applyCloverDslashQcu(plan2) | [ ] |

## G4 求解器族（MG 的直接依赖）

| # | 功能 | quda 侧 | PyQCU 侧 | 状态 |
|---|---|---|---|---|
| 4.1 | BiCGStab(Schur 奇偶预条件) | invert_test --inv-type bicgstab --matpc even-even | applyCloverBistabCgQcu（同语义基线） | [~] 受阻：libquda 仅编 sm_70，P100(sm60) 静默挂死/V100 报 CC 不匹配（QUDA_DEVICE 路由未达 V100）；解锁=重编 libquda 含 sm60 或修设备选择 |
| 4.2 | CG(MdagM) | invert_test --inv-type cg | applyWilsonCgQcu；python solver CG | [ ] |
| 4.3 | MR 平滑器 | inv_mr_quda.cpp（MG 默认平滑器） | pyqcu/solver/_mr.py（quda 思想）；C++ MG 用 BiCGStab/CG 定步替代 | [ ] |
| 4.4 | GCR 外层 | Solver::create GCR(+MG 前置) solver.cpp:67 | run_gcr()=FGMRES(10)⊕V-cycle（_MG_USE_GCR_） | [ ] |
| 4.5 | CA-CG/CA-GCR | inv_ca_cg/inv_ca_gcr | python _cacg.py；C++ 无 | [ ] |
| 4.6 | 多移位 CG | invertMultiShiftQuda | python _multishift_cg.py（updateAlphaZeta 同源） | [ ] |
| 4.7 | deflate(eigcg/IRAM 低模) | newDeflationQuda/deflated_invert_test；CG 内 deflate 钩子 | _MG_USE_DEFLATE_=一次 V-cycle 校正作初值（弱对应）；python tr_lanczos 未接 C++ | [ ] |

## G5 MG setup（null 向量与粗空间）

| # | 功能 | quda 侧 | PyQCU 侧 | 状态 |
|---|---|---|---|---|
| 5.1 | null vec 生成(求解器逆迭代) | generateNullVectors multigrid.cpp:1275（--mg-setup-inv bicgstab/gcr/cacg/bicgstabl，MdagM 分支） | give_null_vecs_mt tools/_multigrid.py:355（逆迭代/DDalphaAMG 配方，相对容差） | [ ] |
| 5.2 | 多轮 setup+全局正交化 | :1365-1416 | 单轮+local_orthogonalize（块局部 GS） | [ ] |
| 5.3 | 特征向量 nullvec | generateEigenVectors :1646(IRAM/TRLM) | tr_lanczos 存在未接入 | [ ] |
| 5.4 | 自由场解析 nullvec | buildFreeVectors :1477 | 无 | [ ] |
| 5.5 | nullvec 持久化 | VectorIO <file>_level_N_nvec_M；dumpMultigridQuda | h5 缓存 logs/nullvec_cache + data/*.h5（lonv/hnn/hdg/sit） | [ ] |
| 5.6 | 块正交化 | BlockOrthogonalize(two-pass 二次 GS) block_orthogonalize.in.cu:215 | local_orthogonalize（批量 QR） | [ ] |

## G6 Transfer P/R

| # | 功能 | quda 侧 | PyQCU 侧 | 状态 |
|---|---|---|---|---|
| 6.1 | Restrict R | Transfer::R transfer.cpp:292→Restrict 核 | applyMultigridRestrictQcu（细层 e=12 硬编码 FIX） | [ ] |
| 6.2 | Prolong P | Transfer::P :260→Prolongate 核 | applyMultigridProLongQcu（同上硬编码） | [ ] |
| 6.3 | 站点/spin 映射 | createGeoMap/createSpinMap CPU 表 | Python 层块结构约定 [E,e,X,x,...] 10 维 | [ ] |

## G7 粗格算子

| # | 功能 | quda 侧 | PyQCU 侧 | 状态 |
|---|---|---|---|---|
| 7.1 | 精确 Galerkin 粗化 Y,X | CoarseOp coarse_op.cuh calculateY:974(GPU/CPU 双路) | torch 探测 build_stencil(_mt) 33-tensor(sit/hop_nn/hop_diag)，经 set_ptrs 槽 30+ 传入 | [ ] |
| 7.2 | 二次粗化 | CoarseCoarseOp | build_stencil_mt lvl≥2（CudaCoarseSchurOp 续探） | [ ] |
| 7.3 | 预条件粗算子 Ŷ,X⁻¹ | calculateYhat + DiracCoarsePC(eo) | 宽版粗 Schur 形式直接存奇子格算子（结构不同、作用等价性需验证 A_c≈RSP） | [ ] |
| 7.4 | 粗 dslash 核 | ApplyCoarse dslash_coarse.cuh(dagger/parity/clover 全组合) | multigrid_coarse_dslash[_wide].cu（窄/宽两版） | [ ] |
| 7.5 | 粗核基准计时 | multigrid_benchmark_test | bench 脚本 coarse 计时（dev84 check_ms 剖析） | [ ] |

## G8 MG 求解循环

| # | 功能 | quda 侧 | PyQCU 侧 | 状态 |
|---|---|---|---|---|
| 8.1 | V-cycle 递归 | MG::operator() multigrid.cpp:1131(pre→R→coarse→P→post) | v_cycle lattice_clover_multigrid.h:1520(热启动+图回放+守卫) | [ ] |
| 8.2 | W/F-cycle | cycle_type 经 RECURSIVE 粗解器包装 | 无（仅 V） | [ ] |
| 8.3 | 平滑器族 | MR 默认；Chebyshev/CA-GCR/BiCGStabL 可选；nu_pre/post 分离 | level0 内嵌 BiCGStab 定步；FGMRES 路径 μ_pre 步 CG；无 Chebyshev | [ ] |
| 8.4 | 粗解器 | PreconditionedSolver 包（coarse_solver[level] 可选族+粗格 deflation 注入） | bistabcg_iter_coarse/coarse_solve_fused(cooperative) | [ ] |
| 8.5 | K-cycle | coarse_solver=RECURSIVE | 无 | [ ] |
| 8.6 | verify 自检五项 | MG::verify :745(P·R 正交/Galerkin/厄米等) | verify_nullvecs(Python 测试层四重诊断)+run_test 全残差 | [ ] |
| 8.7 | 收敛判据/可靠更新 | reliable_delta/pipeline/累加器流水线 | r0_ref 锚定+自适应门控停用校正；无可靠更新 | [ ] |

## G9 驱动生命周期与工程面

| # | 功能 | quda 侧 | PyQCU 侧 | 状态 |
|---|---|---|---|---|
| 9.1 | 建/销毁层次 | newMultigridQuda/destroyMultigridQuda | applyInitQcu→applyCloverMultigridQcu→applyEndQcu(set_ptrs 槽位生命周期) | [ ] |
| 9.2 | gauge 更新后薄/全刷新 | updateMultigridQuda(thin/full) :2946 | 无（重建全部） | [ ] |
| 9.3 | 逐层混合精度 | 全字段 per-level precision/sloppy | _MG_LEVELn_DATA_TYPE_ 槽存在但 parse_params 不读（声明未实现） | [ ] |
| 9.4 | 外部初值热启动 | use_init_guess | C++ init 强制 x_o=0 | [ ] |
| 9.5 | 多右端项 | invertMultiSrcQuda | 无 C++ 批量（tools/_bistabcg_batch 为 Python 层） | [ ] |
| 9.6 | MPI 分布 | 分布式粗格+ghost 打包 | 冗余全局粗格+Allreduce 点积；单 rank 多线程多卡 | [ ] |
| 9.7 | tensor-core MMA | *_use_mma[level] 全链路 | 无（平台 sm_70 SIMT mma 变体存在但未实现） | [ ] |
| 9.8 | setup 位置可编程 | setup_location/location per-level CUDA/CPU | 固定 GPU | [ ] |

## G10 端到端（目标b：PyQUDA 辅助）

| # | 功能 | quda/PyQUDA 侧 | PyQCU 侧 | 状态 |
|---|---|---|---|---|
| 10.1 | 同一 gauge 文件端到端 MG solve | pyquda.init+loadGauge+getClover(multigrid=..)+invert | MultiGpuMultigrid.solve(data/*.h5) | [ ] |
| 10.2 | 输出数值一致性 | 解向量/残差历史 | 解向量/残差历史 | [ ] |
| 10.3 | 性能对照 | quda 计时(V100) | PyQCU 计时(V100) | [ ] |

## 功能补充候选清单（P5，按价值/可行性排序）

| 序 | 缺口 | 依据 | 备注 |
|---|---|---|---|
| S1 | per-level 混合精度接线 | G9.3 声明未实现 | 近乎 bug 修复；quda 全字段对照 |
| S2 | Chebyshev 平滑器 | G8.3 缺失 | quda inv_chebyshev；MG 标配 |
| S3 | x0 热启动通道 | G9.4 | use_init_guess 对标 |
| S4 | thin update(gauge 变化后) | G9.2 | updateMultigridQuda thin 档对标 |
| S5 | setup 多轮+全局再正交 | G5.2 | 提升粗空间质量 |
| S6 | verify() 五项自检入 C++ | G8.6 | Galerkin 一致性闭环 |
| 范围外 | MMA/K-cycle/MADWF/NVSHMEM/分布式粗格 | 平台与体量 | 如实报告不做，非静默跳过 |

## 构建与运行资产

- quda 构建：/tmp/opencode/quda-build-sm70（RELEASE, sm_70, MPI ON, MULTIGRID ON,
  DW/staggered/twisted/laplace OFF, NVEC_LIST=12,24,48），安装前缀 /tmp/opencode/quda-install
- PyQUDA 安装：pip install ./pyquda_core . （QUDA_PATH=<install>，rpath 烧入）
- 数据：data/gauge_16x32x32x48_m0.05_seed42_c64.h5（统一格子）及 L16x32x32x48_lv1_E12_nvi*_t1e-2.h5 nullvec 缓存
- GPU 规约：单卡 V100-32GB(sm_70)；多卡 P100×2(sm_60，如需另建 build-sm60)
