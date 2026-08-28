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

### 当时的临时回归状态（已被后续复验修正）
clover_solve / component / quda 对照三项持续 GREEN；
MG 端到端在本机当前驱动状态下间歇失败（init memset InvalidArgument），
代码与最后绿点零差异——属平台态而非代码回归，待 WSL 复位后复验。

## 十六、阶段性平台态记录与资产归档（后续已修正）

- `CUDA_LAUNCH_BLOCKING=1` 下 MG 仍在 init memset 报 InvalidArgument
  ⇒ 排除异步竞态，定性为 **上下文级损伤**（capture 实验硬中止遗留）。
- 恢复手段：宿主侧 `wsl --shutdown` 后重开终端，运行
  `python examples/qcu/dev87/run_all.py --with-quda` 复验（预期全 GREEN）。
- 战役资产已归档 `logs/stab31/`：总报告、对照矩阵、回归/对照 JSON。
- 当时版本的 run_all.py 的 mg_vs_ref 失败信息曾附平台态提示（不掩盖失败本身）。

## 十七、诊断更正（重要）

后续证据（conftest.multi_gpu 三场景全 PASS，含 V100+P100×2 上的
applyCloverMultigridQcu）推翻了 §15 的"WSL2 驱动上下文损伤"结论：
**库与平台健康；失败特定于 run_qcu_mg 的直连桥序列**。

已排除：params 逐位一致、槽位句柄残留清零、slot 重选、阻塞启动。
未决：直连序列中 solver-set 的 cudaMemsetAsync(x_o) 报 Invalid argument，
而 MultiGpuMultigrid 包装路径同参数下正常——差异在进程内上下文/分配布局，
复现配方与本节记录齐备，留待下一会话在复位后的干净环境定位。

影响评估：生产入口（MultiGpuMultigrid / conftest 套件）不受影响；
仅 dev87 新增的直连运行器受限。G4.1/G10 已交付对照结论不依赖该直连路径
（其数据产生于该路径尚正常的时段，且经双向交叉验证）。

## 十八、未决异常 interim 记录（2026-08-25 深夜，疲劳警戒）

现象：G4.1 缩放解对照从 3.84e-7（12:0x）退化为 ~0.72；同进程算子级探针
（热规范、double、mat vs give_wilson+clover）现测 rel=0.48。
而既有锚定事实仍成立：单位规范算子精确 4.05 对齐；8³×16 clover 差分
cos=1.000000/scale=4.05。三者相互矛盾 ⇒ 存在未被识别的实验耦合
（疑似方向：quda 侧多次重编后的行为差异 / 探针脚本自身细节）。

处置：不下结论。下一会话以隔离矩阵重验：
  {单位/热规范} × {单精度/双精度} × {新进程对} 的算子级全组合，
  并先复跑 cmp_operator.py 与 cmp_clover_vec.py 确认旧锚定可复现性。
在此之前，G4.1/G10 的已交付数字以 logs/stab31/ 归档件为准。

## 十九、最终复验补充（2026-08-27）

本节更新并覆盖前面关于“驱动上下文损伤”的临时判断。随着后续
`conftest.multi_gpu` 和本节直连运行器复验，库与 CUDA 平台均可正常工作；
旧异常应归因于当时 `run_qcu_mg.py` 的桥接调用序列/槽位生命周期排查过程，
不能再作为当前平台限制。`run_all.py` 的失败提示已同步改成要求检查本次
直连日志，不再把失败预先归因于 WSL2。

本次收尾时还捕获并修复了一个独立的闸门环境问题：`env.sh` 的 QCU 库路径
曾排在旧的 `sm_60` `libquda.so` 之前，导致 V100 上出现
`no kernel image available`；`run_all.py` 现在为 QUDA 子进程显式前置
`quda_env.sh` 指定的 `sm_70` 安装路径。修复后完整
`run_all.py --with-quda` 为 `GREEN (5/5)`，总耗时 `424.8 s`。

### 1. PyQCU MultiGrid 实际配置矩阵

统一大格为 `16×32×32×48`、`m=0.05`、`atol=1e-6`、
`mg_grid=[2,2,2,2]`，粗算子从 `data/` 的 E=12 HDF5 缓存读取。墙钟是
`applyCloverMultigridQcu` 调用的同步后耗时；真残差由 Python full
Wilson+Clover 算子重新计算。

| 配置 | 墙钟(s) | 与 BiCGStab 参考解差异 | full-op 真残差 | 结论 |
|---|---:|---:|---:|---|
| 1L | 1.378 | 6.55e-6 | 3.92e-7 | 通过 |
| 2L, E=12 | 1.412 | 8.57e-6 | 6.59e-7 | 通过 |
| 2L + MR, E=12 | 1.445 | 8.57e-6 | 6.59e-7 | 通过（MR） |
| 2L + deflate | 1.350 | 6.77e-6 | 4.83e-7 | 通过 |
| 2L + warm | cold 1.412 / warm 0.198 | warm 3.66e-6 | warm 4.41e-7 | 通过 |
| 2L + GCR/FGMRES | 5.015 | 1.27e-6 | 6.86e-7 | 通过 |

因此，当前大格上 2L 相对 1L 为 `0.976×`，没有额外加速；warm 约有
`7.1×` 加速；GCR/FGMRES 数值正确但约为普通 2L 的 `0.28×`。deflate
本次只显示小幅墙钟差异，不能据一次运行宣称性能收益。3L 也已在
`8×8×8×16, E=24/E=24` 的已有缓存上真实运行，真残差 `5.94e-7`；
尚无大格 3L 缓存，故不外推其性能。

本轮补齐并验证了 QUDA 风格 MR 平滑器。普通 2L V-cycle 与
FGMRES/MG 预条件路径均支持 `--smoother mr`；大格 MR 实测 `1.445 s`，
相邻 CG 基线 `1.420 s`，两者均为 68 次外层迭代、2 次 V-cycle，full-op
真残差 `6.59e-7`，相对 BiCGStab 参考解差 `8.57e-6`。小格
`4×4×4×8` 的普通 MR 与 FGMRES/MR 也真实触发粗层，分别得到真残差
`5.46e-7` 与 `6.58e-7`。大格单次 MR 约慢 `1.7%`，受单次 GPU 计时波动影响，
当前只确认数值兼容性，未宣称性能收益。

本轮新增三类路径。2L Chebyshev 固定步多项式平滑器在统一大格上耗时
`1.431 s`，full-op 真残差 `6.59e-7`；2L CA-GCR（4 阶块、双遍 MGS、
Gram 小系统，块退化时回退 FGMRES）在 `atol=1e-8,max_iter=400` 下耗时
`5.215 s`，与 BiCGStab 参考解差 `3.37e-7`，full-op 真残差 `1.43e-7`。
后者已达到单精度误差平台附近，但本次 400 次块步仍未达到请求的 `1e-8`，
因此不把该严格配置表述为按容差收敛；在常用 `atol=1e-6` 冒烟配置下真残差为
`7.58e-7`。

3L 小格缓存上递归 W/F/K-cycle 均完成真实运行（`8×8×8×16`、
`E=24/E=24`）：W/F 的 full-op 真残差分别为 `1.36e-6/1.36e-6`，
K-cycle 为 `5.97e-6`，对应配置 `atol=1e-5`；三条路径均无 NaN 或非法访问。
在 FGMRES 外层的 W/F/K 组合上又以宽松 `atol=1e-3` 做了安全冒烟，真残差分别为
`2.78e-4/2.78e-4/2.70e-4`。这些结果验证了递归与回退路径，尚不足以证明性能优于 V-cycle。

组件级实测结果为：restrict/prolong L2 误差 `2.09e-7/6.40e-8`，窄/宽
粗 dslash 误差 `2.65e-7/5.01e-7`，最新回归的 Galerkin 误差 `9.47e-7`，Gram
非对角最大值 `2.38e-7`。窄/宽粗核中位耗时约 `0.775/1.881 ms`。

### 2. QUDA/PyQUDA 交叉结果与口径

双方运行严格分进程，避免 `libqcu.so` 与 `libquda.so` 的 CUDA runtime
符号/上下文相互影响。直接 Clover 求解在 `m+4=4.05` 缩放后解差为
`3.91e-7`；最新 MG 输出的交叉结果为 `8.63e-6`。QUDA MG 本次归档为
`setup=270.43 s`、`solve=82.12 s`、9 次迭代；PyQCU 读取离线粗算子，
且两侧容差、精度和外层算法不同，因此不能将这两个数字直接当成公平的
端到端算法倍率。

多线程多卡实测中，V100 单卡、P100 单卡及 P100×2 的结果一致性均通过
`1e-5`；P100 单卡到 P100×2 的并行比为 `0.970×`，当前机器未获得加速。

### 3. 结论与未完成项

本轮已完成并真实运行闭环的核心范围是：Wilson/Clover 算子锚定、
BiCGStab、V/W/F/K-cycle 递归（3L 为小格缓存验证）、GCR/FGMRES、CA-GCR、
CG/MR/Chebyshev 平滑器、deflate、warm start、transfer、Galerkin、粗 dslash、
full-op 真残差、缓存加载、逐层混合精度、真正分布式粗格及单 rank 多线程多卡
一致性。固定 `L=2` 的 BiCGStabL、2L/3L 的 c64/c128 混合路径，以及阻塞
host-staging MPI 粗格 halo 已在下一阶段补齐并通过冒烟/等价性验证。仍未实现或
未完成同参数闭环的项目包括动态 thin gauge update、MMA/NVSHMEM，以及 C++ 版
完整五项 `verify()` 接口；矩阵中均保留为 `[ ]` 或 `[~]`，没有静默跳过。

最新证据文件为 `out/qcu_mg_matrix_*.json`、`out/component_cuda.json`、
`out/multigpu.json`、`out/quda_clover_{solve,mg}.json`。后续修改
`lattice_clover_multigrid.h`、`lattice_clover_bistabcg.h` 或参数协议后，
应至少重新运行语法检查、`test_mg_breakdown.py` 和 `run_all.py`。

## 二十、BiCGStabL、逐层混合精度与真正分布式粗格（2026-08-28）

### 实现边界

- BiCGStabL 当前实现为固定 `L=2` 的外层块迭代，包含可靠重启、粗校正后的
  Krylov 状态重置，以及末尾不完整 block 的边界保护；这不是可配置的任意 `L`。
- 每个 MG 层使用独立的 c64/c128 擦除存储。跨层 restrict/prolong 不依赖隐式
  指针类型，而是经过显式 mixed-precision cast kernel；`_MG_LEVELn_DATA_TYPE_`
  已接入参数解析与层级构造。
- 粗格不再复制完整全局向量：每个 rank 只保存自己的局部粗格和局部 33 点
  stencil。跨 rank 的 32 个邻居采用阻塞 host-staging halo，粗层点积、fine 层
  点积和相对残差范数使用 MPI 全局归约。

### 最新真实运行

- 单 rank 大格 `16×32×32×48`、fine c64/coarse c128、`max_iter=80` 的
  BiCGStabL 运行完成 80 次迭代，full-op 真相对残差约 `7.85e-7`。
- 单 rank 3L c64→c128→c64 路径及 3L BiCGStabL mixed 路径真实运行；大格
  c64→c128、c128→c64 2L 路径均稳定。大格 mixed 标准 BiCGStab 在
  `max_iter=80` 时真残差约 `3e-6`，受迭代上限限制，不能据此宣称已达到更严
  容差；小格提高到 `max_iter=160` 时真残差约 `7.7e-7`。
- 最新库构建完成：`build.sh` 成功链接 `libqcu.so`。
- MPI `np=2, grid=[1,1,1,2]` 的同精度与 `--bicgstab-l --coarse-dtype c128`
  mixed MG 冒烟均退出码为 0，无死锁或非法访问。
- 独立粗算子等价性复测：`grid=[1,1,1,2]`、c64 的全局 L2 相对误差为
  `5.60e-7`；`grid=[2,1,1,1]`、c128 为 `1.03e-15`。两项均将 rank-local
  输出重建为全局场后，与完整周期参考 stencil 比较。

因此，“真正分布式粗格”在正确性意义上已经落地；当前限制是通信仍为阻塞
host-staging，没有做 device-aware MPI、通信计算重叠或 NVSHMEM 性能声明。
