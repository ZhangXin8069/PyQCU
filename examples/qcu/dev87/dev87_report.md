# dev87 阶段报告 1 — 对照基线搭建与算子约定锚定（~analy）

日期：2026-08-25　任务：~auto-all 对照单测（对象=PyQCU CUDA_C++ 多线程 MultiGrid；
基准=quda 快照 + PyQUDA 0.10.54）。工作区：`examples/qcu/dev87/`。

## 一、资产与构建

| 资产 | 状态 |
|---|---|
| quda（/tmp 副本构建，refer/ 零改动） | 356/356 编译，RELEASE sm_70，MPI/MULTIGRID ON，DW/stag/twisted/laplace OFF，NVEC_LIST=12,24,48；安装于 /tmp/opencode/quda-install |
| 构建排障 | cmake3.22→3.31.6(user)；快照缺 c_interface_test.c/trove/generics → 上游稀疏克隆补入 /tmp 副本 |
| PyQUDA | 源码安装 0.10.54；修 InvalidVersion 'tab30.post2'；`quda_env.sh` 以 LD_LIBRARY_PATH 前缀压过全局旧 libquda |
| **平台级发现 F1** | 本机(WSL2)映射内存 GPU→host 原子写不可见 ⇒ quda 归约引擎无限自旋（reduce_helper.h:162）。已在 /tmp 副本打补丁：有界自旋→cudaDeviceSynchronize→设备别名 D2H 回拷。与 dev84 平台结论同源 |
| **工程约束 F2** | 同进程加载 libqcu.so 与 libquda.so 会因 cudart 符号/上下文冲突使 quda driver 路径报 INVALID_CONTEXT ⇒ 对照脚本一律双进程阶段隔离 |

## 二、约定锚定结论（G2/G3/G4.1）

统一条件：随机规范 seed42/σ0.1、m=0.05、κ=1/(2m+8)、周期 T 边界、双精度（quda 侧）。

1. **Wilson+Clover 全算子逐元素同构**：
   - 单位规范 MatQuda vs PyQCU give_wilson(+clover)：线性回归 y_q = 4.0485·y_p + ε，
     而 m+4 = 1/(2κ) = 4.05 —— 仅整体归一化差。
   - Clover 差分隔离（MatQuda_{csw=1} − getWilson）：**cosine = 1.000000**，
     最小二乘 scale = 4.050000，残差 2.0e-7（fp32 底）。
   - 布局转换链（PyQCU eo[t列压缩] → 全格点 xyzt → QDP(t,z,y,x)/行主色、eo[x压缩]、
     tzyxsc 字段）全部验证正确。
2. **解对照（G4.1）**：同一 b 下 x_qcu ≈ (m+4)·x_quda；缩放后
   `rel_diff = 2.65e-2`，且 `rel_res(M_p, (m+4)·x_quda) = 9.9e-8`
   ⇒ **quda 解即 PyQCU 全算子的精确解；PyQCU C++ 解偏离全算子真解 ~2.6e-2**。

## 三、新发现（转入 P4 debug 清单）

- **F3（重要）PyQCU C++ BiCGStab/MG 的收敛判据语义**：内部报告残差 8.8e-10 时，
  全算子相对残差实际 ~2.5e-2（dev84 配方实测）。内部度量是 Schur 奇偶子系统/
  b__o 变换后的量，非全 M 残差 ⇒ "atol=1e-6" 的物理含义与 quda 不对齐。
  这同时解释 MG-vs-BiCGStab 解差 2.65e-2 与 n_vcycles=0 的门控行为观察。
- F4：opcmp 三项基回归在大格子 fp32 下受共线性干扰（小格差分法为准）。

## 四、产物清单

- comparison_matrix.md（G1-G10 状态登记）
- run_qcu_ops.py / run_qcu_mg.py / run_quda_py.py（solve/mg/opcmp）
- cmp_anchor.py / cmp_operator.py / cmp_clover_vec.py / cmp_clover_field.py(解码搁置)
- cmp_matrix.py 聚合器、quda_env.sh
- out/*.json|npz（基线与对照数据）

## 五、下一步

1. P4-F3：核对 C++ run()/bistabcg 停机判据源码，给出全算子残差停机选项或修正文档口径；
2. G8/G10：MG 端到端对照（quda multigrid=[[2,2,2,2],[24]] vs PyQCU E12/E48）+ 性能表；
3. G5-G7 组件级对照推进；S1-S6 补充候选评估。

## 六、阶段 2 增补（2026-08-25 下午）：F3 根因修复

### 根因链（全部实测）
1. 运行器缺陷：`main()` 解包 make_clover_tensors 把 cei/coo 对调 →
   求解器拿到奇偶互换的 Clover 逆（已修正 run_qcu_ops/run_qcu_mg）。
2. 库级缺陷（真实）：lattice_clover_multigrid.h 主循环 fp32 递推残差漂移 ——
   跟踪 rn 降到 9.2e-7 时真 Schur 残差 62.9（16³×48 实测），
   绝对判据 rn²<atol² 在大 ‖b‖ 下系统性早停。
   （quda 同场景靠 reliable updates 规避 —— 即矩阵 G8.7 缺口。）

### 修复（cpp/cuda/qcu/include/lattice_clover_multigrid.h）
- 周期真残差刷新：单 rank 每 50 次迭代 compute_full_residual() 覆写 st.r
  并 reset_bistabcg_state_l0()，映射收敛量同步刷新；
- 停机改相对：rn² < atol²·‖b__o‖²（与 run_gcr 的 r/b 语义对齐）；
  V-cycle 门控阈值同步改相对。

### 回归数字（16×32×32×48, m=0.05, atol=1e-6）
| 指标 | 修复前 | 修复后 |
|---|---|---|
| 全算子真相对残差 | 2.48e-2 | **3.72e-7** |
| MG vs BiCGStab 解差 | 2.65e-2 | **3.50e-7** |
| G4.1 vs quda(缩放 m+4) | 0.753(未缩放口径) | **3.85e-7** |
| solve 用时(V100) | 2.54 s | 2.09 s |

遗留：n_vycles=0 为该 gauge 谱不可压缩下的门控正确行为（非回归）；
多 rank 刷新路径未启用（需全局归约，后续接 MPI Allreduce）。

## 七、阶段 3 增补：G8/G10 MG 端到端对照与性能表

### 归约补丁终态
WSL2 映射原子**间歇性可见**：有界自旋路径会偶发采到陈旧归约值并静默毒化求解
（quda MG 首跑自称收敛 9.8e-9 而真残差 >100%）。`DEV87_REDUCE_SYNC=1` 强制
同步式回拷为正确基线；quda_env.sh 已固化。

### 性能表（16×32×32×48, m=0.05, V100-SXM2-32GB, 单卡）
| 侧 | 配置 | setup | solve | iters | 真全算子相对残差 |
|---|---|---|---|---|---|
| PyQCU | MG 2L(E12, stencil h5 缓存) | ~0(离线缓存) | **2.41 s** | ~140(BiCGStab 主循环) | 1.55e-7 |
| PyQCU | Clover BiCGStab(参考) | — | 2.09 s | ~140 | 3.7e-7 |
| quda | MG 1L(block[4,4,4,4], nvec12, tol1e-8, double 外层) | 291.9 s | **84.0 s** | 9(GCR) | 9.9e-8(×4.05 后) |
| quda | BiCGStab(double, tol1e-8) | — | 1.39 s | 123 | — |

### 口径注记（诚实声明）
1. nullvec 策略不同：PyQCU 离线缓存 vs quda 在线生成——setup 分列已公平化，
   但 PyQCU 的离线成本未计入（一次性资产）。
2. quda 数值含 WSL2 归约兜底补丁惩罚（未单独量化；健康平台应显著更快）。
3. 停机口径已对齐为相对残差（dev87/bug42），tol 取值两侧不同(1e-6 vs 1e-8)。
4. quda 外层为 double、MG 内部混合精度；PyQCU 全程 c64(fp32)。

### 结论
- 正确性：G10 解一致性 rel=2.24e-7 —— 两库 MG 在同一物理问题上的解等价。
- 性能：本机口径下 PyQCU MG solve 快 ~35×，但含上述三项有利于 PyQCU 的口径差；
  其中第 2 条(补丁惩罚)是平台强加的，第 1/3 条是策略差异。绝对性能结论需在
  健康 CUDA 平台复测后方可对外声明。

## 八、阶段 4 增补：G5-G7 组件级诊断（PyQCU 侧标准验收）

`component_diag.py` → out/component_diag.json（16³×48, E12, nvi1 缓存 stencil）：

| 指标 | 数值 | 判读 |
|---|---|---|
| Galerkin 一致性 ‖A_c e − RSP e‖/‖·‖ | **7.9e-7** | 粗算子构建正确（与 dev85 的 5.6e-7 同量级） |
| Gram 正交性（块内） | 非对角 ≤2.4e-7；对角 1±2.4e-7 | local_orthogonalize 达标 |
| 近零性 ‖Sv‖/‖v‖（抽样4） | 0.31 / 0.34 / 0.46 / 0.44 | 非近零空间——谱连续所致（dev84 ρ_V 结论），非管线缺陷 |
| 幂迭代谱半径 S_λmax | 1.169 | — |

结论：组件实现正确；MG 门控停用校正源于物理谱而非代码。quda 侧不外露同层
null 向量诊断接口，其质量只能间接由 MG 外层收敛行为体现（9 次 GCR @tol1e-8）。

## 九、P5 补充落地与 P6 回归收束

- S6-lite 已落地：`applyCloverMultigridQcu` 结束时自报
  `FINAL TRUE residual (full-op) = 1.48e-6 (relative 6.1e-10)`
  （与外部 harness 口径一致，quda verify() 精神的最小实现）。
- S1/S2/S4/S5 缓办、S3 待做（理由见矩阵补充清单更新）。
- 回归：conftest.multi_gpu 三场景全 PASS（一致性 tol=1e-5）；
  conftest.clover.bistabcg.dslash 首迭代差 1.9e-6 ✓
  （该脚本尾部除零为其自身槽位复用遗留问题，先于本任务存在，另行处理）。

## 十、终局状态

对照矩阵 G1-G10：G2/G3/G4.1/G5(部分)/G6/G7.1/G8.7/G10 全部 [x] 实测闭环；
G1 统计口径、G8.2/8.5 等结构性差异项以注记形式存档。库级产出：
停机语义修复(bug42)、真残差刷新与相对判据、S6-lite 日志。
