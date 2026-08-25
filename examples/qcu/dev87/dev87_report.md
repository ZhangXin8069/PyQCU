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
