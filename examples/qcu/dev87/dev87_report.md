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

## 十一、阶段 5 增补：G1 统计对照 + S3 热启动功能落地

### G1 规范场生成统计口径（σ=0.1, 8³×16, seeds{43,44,45}）
| 侧 | 单位性缺陷 max‖U†U−I‖ | plaq 均值范围 | plaq std 范围 |
|---|---|---|---|
| PyQCU(c64) | 1.7e-7 | +0.0588..+0.0596 | 0.0420..0.0427 |
| quda(fp64) | 5.5e-13 | −0.0012..+0.0005 | 0.0959..0.0968 |

结论：两侧生成器各自自洽且稳定，但 **σ 的分布参数化语义不同**（同 σ 下
plaq_std 差 ~2.3×，均值符号/幅度不同）。使用跨库规范场时须各自标定，
不构成实现缺陷。PyQCU 缓存大格实测 0.0589/0.0424 与小格一致。

### S3：x0 热启动通道（对标 quda use_init_guess）
- 协议扩展：params int32[57]→**[58]**，新增 `_MG_USE_INIT_GUESS_=57`
  （define.py ⇔ define.h 同步；LatticeSet 按 _PARAMS_SIZE_ 拷贝自动扩展）。
- 语义：≠0 时两求解器类跳过 x_o 初始化（清零/固定种子随机），保留调用方
  预填在 fermion_out 奇半的解，初始残差 r=b__o−S·x₀ 真实计算。
- 实测（16³×48, atol=1e-6）：COLD 1.94s → **WARM 0.346s**（5.6×），
  真残差 1.65e-7，与冷启解一致 3.5e-7。多轮物理参数扫描/演化场景收益显著。

## 十二、阶段 6 增补：多 rank 正确性加固与 MPI 冒烟结论

- 加固 1（缺陷预防）：多 rank 下 ‖b__o‖² 改走 dot_mpi 全局归约
  （此前相对停机阈值在分布式奇子格上会用到本地范数——bug42 引入的
  相对判据在 MPI 场景的潜在错误，本次主动修正）。
- 加固 2：mg_multi 时禁用 fused cooperative 粗解（np=2 实测 illegal access，
  与 dev84 记录的 WSL2 cooperative+同步脆弱性同源），回退普通迭代路径。
- MPI 冒烟（np=2, 8³×16）：仍被**既有分布式布局约束**阻断（粗层几何/
  通信路径为单 rank 假设，先于本任务存在）。单 rank 全链路回归保持绿。
- 结论：多 rank MG 的完整对照属独立后续任务；本阶段保证不因 dev87 改动
  恶化其现状，并消除两处潜在/实际崩溃点。

## 十三、终局：一键回归闸门

`run_all.py [--with-quda]`：串联 G4.1 真残差 / G8 MG-vs-参考 / G5-G7 组件 /
quda 缩放对照四项断言，产物 out/regression.json。

终局实测（2026-08-25，V100，缓存全热）：
```
[PASS] clover_solve_true_res: rel=3.720e-07 wall=2.10s
[PASS] mg_vs_ref:             rel=3.499e-07 wall=2.42s
[PASS] component_quality:     galerkin=7.90e-07 ortho_offdiag=2.38e-07
[PASS] quda_solve_scaled_agreement: rel=3.842e-07 iters=120
=== regression GREEN (4/4) in 18.3s ===
```
后续任何触及 `lattice_clover_multigrid.h`/`lattice_clover_bistabcg.h`/
协议层的改动，跑本闸门即可在分钟级复验本轮全部结论。

## 十四、热点剖析与稳定性浸泡（收束补充）

### 热点（torch.profiler, CUDA runtime 视角；CUPTI 内核级本机不可用）
applyCloverBistabCgQcu 单次求解（atol=1e-6, 16³×48, V100）：

| 运行时 API | 次数 | CPU 自耗时 | 占比 |
|---|---|---|---|
| cudaStreamSynchronize | 3097 | **1.896 s** | ~92% |
| cudaMemcpyAsync | 1995 | 0.194 s | 8.9% |
| cudaLaunchKernel | 8573 | 0.069 s | 3.2% |

结论：剩余瓶颈是**细粒度迭代内的流同步次数**（均值 612µs/次，WSL2 thunk），
内核发射本身仅 ~69ms。后续优化方向明确：把 dev84 的"图段回放/SYNC DIET"
思想延伸到普通 BiCGStab 路径（如收敛检查映射读降频、多迭代合段），
理论上限可再削 ~1.5s/2.0s。本轮不做（保持已验证稳定态，改动留待下阶段）。

### 稳定性浸泡
run_all.py --with-quda 连续 3 次：GREEN 4/4 ×3（18.1/18.3/18.2s，
零失败零抖动）。刷新/热启动/相对停机新路径在重复运行下确定且稳定。

### §14.1 SYNC-DIET 延伸实验（本轮）
实现：细层快路径加 `host_sync` 形参，run() 每 CHECK_STRIDE=4 迭代一次主机同步
（收敛检查/门控/刷新全部对齐检查网格；刷新周期 50→48；门控窗按迭代计）。
结果：闸门 GREEN，真残差 3.72e-7 不变，但**本机墙钟无收益（2.04s）**。

根因修正认知：profiler 计数表明成本=每次 CUDA API 调用的 WSL2 thunk 税
（~13.6k 次/解 × ~150µs），同步只是其中一类。仅减同步次数不动总量，
收益趋零。**有效路径只剩图段回放把 K 次迭代封装为 1 次发射**（发射数÷K）。
留待下阶段；本改动在健康平台（同步 ~5µs）仍直接兑现 4× 同步缩减，予以保留。

## 十五、图段回放批量迭代：实验记录与平台结论

- 实现（已存档 `out/fine_graph_experiment.patch`，未入主线）：细层迭代体抽取 +
  K=32 图段捕获/发射，含 cublas 工作区预绑定(64MiB)、最小 Dot 金丝雀探针、
  全路径异常熔断。
- 金丝雀判定：**本平台(WSL2/551.78)不支持 cublas-Dot 进入 stream capture**
  （canary 失败→优雅回退 legacy 正确性保持）。
- 强行多次捕获触发硬中止后，驱动上下文出现跨进程持续性劣化
  （后续普通 cudaMemsetAsync 偶发 InvalidArgument；component/clover 路径不受影响）。
  恢复手段为宿主侧 `wsl --shutdown`（超出本会话权限）。
- 处置：头文件已回退至 stab31 等价绿点（d2f0198），主线不含该实验代码；
  后续在健康 CUDA 平台重启此优化时，从存档补丁起步，并将 cublasDot 替换为
  自研 dot 内核以规避 cublas-capture 限制。

### 当前回归状态声明（诚实口径）
clover_solve / component / quda 对照三项持续 GREEN；
MG 端到端在本机当前驱动状态下间歇失败（init memset InvalidArgument），
代码与最后绿点零差异——属平台态而非代码回归，待 WSL 复位后复验。
