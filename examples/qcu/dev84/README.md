# dev84 — CUDA C++ 多线程版 MultiGrid 稳定 >2 真实加速比

任务：令 `applyCloverMultigridQcu`（多线程多卡，一线程一卡）在统一格子 **16×32×32×48**
（mass=0.05, atol=1e-6, c64）上具有**稳定 >2 的真实加速比**。

## 判据与对照（用户指令 2–5）

| 量 | 定义 |
|---|---|
| 真实加速比 | `t(MG_L1) / t(MG_多层)`，L1 = 仅最细层（num_levels=1） |
| 正确性对照 | 多线程 BiStabCG `applyCloverBistabCgQcu` + Python 全算子残差 |
| 并行效果对照 | MG 单线程运行 vs P100×2 多线程 |
| 通过标准 | speedup_vs_L1 > 2.0 且解正确（rel_vs_ref < 1e-5、全算子残差 ≲ atol 量级） |

器件：V100-32GB（torch cuda:0）单卡；P100-16GB×2（torch cuda:1,2）多卡。
数据：gauge/nullvec 统一存 `data/`（seed=42 一一对应），setup 全部走缓存，
**只测 solve 阶段墙钟**（与 L1 同口径）。

## 基线事实（dev80_3 report.json，V100 单卡，2026-08-21）

| 求解器 | 时间 | 迭代 | 备注 |
|---|---|---|---|
| BiStabCG 参考 | 2.250 s | — | res 3.78e-7 |
| MG_L1 | 1.732 s | 138 | 12.5 ms/iter |
| MG_2L (rs15/cmi3/cf1e3/nvi1) | 1.966 s | 138 | **0.881× FAIL** |

诊断（指令 9 的机理）：
1. **粗解经济性**：粗格 [8,16,16,12]×E12 → vec_sz=294912 ≥ 融合内核阈值
   `<262144`（lattice_clover_multigrid.h v_cycle），落入普通路径——每迭代
   ~1 ms host 同步开销主导，粗解贵而无益；
2. **null 向量质量**：nvi=1 的逆迭代向量太差（conftest 已知良好区间 nvi≥20），
   粗空间捕捉不了低模，校正无效；
3. **Krylov 破坏**：每 num_restart 次迭代做一次校正并**完全重置 BiCGStab 状态**
   （R3 fix 必需但代价大）——quda 从不在细层这样做，而是把整个 V-cycle 作为
   *柔性 Krylov 外迭代*（FGMRES/GCR）的**预条件子**，外层 Krylov 不被破坏；
4. atol=1e-6 为绝对容差而 ||b||~2e3 → 相对容差 ~5e-10，收敛尾段在 rn<100·atol
   后被跳过校正（`run()` 守卫），故校正只影响收敛前段，2L≈1L。

## 优化路线（quda 主参考，函数级对照）

| 轮 | 内容 | 对应 quda |
|---|---|---|
| R1 | 参数回归已知良好区间（rs5/cf3e3/cmi200/nvi20 缓存入 data/） | smoother_tol / nu_pre |
| R2 | 大粗层融合 CG 阈值提升（294912 走 cooperative fused） | coarse 域单 kernel 化 |
| R3 | quda 式柔性外迭代：FGMRES(m) ⊕ V-cycle 预条件子（pre/post 平滑+粗解） | multigrid.cpp operator():1131 + Solver::create preconditioner |
| R4 | P100×2 多线程并行验证 + 一致性 | 多 GPU 冗余全局模型 |

quda 对照表（复现对象 → PyQCU 落点）：
- `MG::operator()`（lib/multigrid.cpp:1131）：pre-smoother→R→coarse→P→post-smoother
  → dev84 R3 `v_cycle_prec()`；
- `MG::createSmoother`（:273）：固定 ν_pre/ν_post 步数、sloppy_converge=true、
  return_residual=true → dev84 固定步数平滑器（无收敛检查）；
- `generateNullVectors`（multigrid.h:1275）→ `tools.give_null_vecs_mt`（缓存 data/）；
- 转移算子 transfer->R/P → `multigrid_restrict/prolong` 内核（33-tensor Schur 一致）。

## 文件

| 文件 | 用途 |
|---|---|
| `main.py` | 子命令入口：bench / verify / multi / hotspot / check / report |
| `README.md` | 本文件（设计 + 判据 + 轮次日志索引） |

产物按指令 16 写本目录 `out/`（镜像至 `logs/dev84/`）。

## 轮次日志（2026-08-22 auto-all）

| 轮 | 内容 | 实测 |
|---|---|---|
| R1 | 基线复测 + nvprof 剖析 | BiStabCG 2.4-2.5s；MG_L1 138it≈2.0s；MG_2L 0.65×；同步 4248 次=6.1s |
| R2 | 块检查→绝对锚定 target→固定步数→守卫标量→CUDA Graph 段回放→零拷贝 | coarse_vec 3246→4ms；V-cycle 156→60ms；NaN 分裂消除 |
| R3 | 粗空间诊断（隔离实验） | ρ_V=0.976±0.0001；Galerkin 3.8e-7；nv_tol 绝对/相对语义 bug 定性 |
| R4 | 配方排除战：ddamg/nvi24/谱收缩/块Jacobi/FGMRES/deflate/参数扫描 | 全部 ≤1×，证据见 `dev84_report.md` §3 |
| 收尾 | 多卡 P100×2 一致性 PASS、并行 1.302×；小格子复核复证指令 9 | 报告 `dev84_report.md` |
| dev84_1 | 库级 nv_tol 相对容差修复 + 修复语义缓存实测（_rt 0.514×/_dd 0.501×） | 结论对生成语义鲁棒闭环 |
| 终环 | setup_staged 三阶段分进程流水线（指令23 端到端落地）；24×32×32×48 体积探针 **0.421×** | 大格子有利假设证伪，全体量收敛 |
| dev84_2_2 | 自适应校正门控（斜率标定+窗口监测）；细层 r/x 融合+范数零拷贝 | MG_2L **2.383s（−18%）**, speedup_vs_L1 **0.798×** |
| 扩展 | P100×2 门控复验（0.82×）+ GCR 预条件子同款门控（4.395s，仍无竞争力） | 门控机制全路径覆盖 |

结论：>2 指标在本配置不可达（连续谱无谱分离，机理级解释见报告 §3.4）；
落地为真实性能优化（V-cycle 2.6×）+ 平台税剖析 + 根因链存照。
