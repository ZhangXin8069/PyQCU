# dev80 分析报告 — 32^4 统一格子 MG 真实加速比剖析

**时间**: 2026-08-20 04:00 UTC  **任务**: 令 CUDA_C++ 多线程版 MultiGrid 具有稳定 >2 的真实加速比  
**判定身份**: {物理+代码} — 先用对称性/量纲与物理图像把握本质，再落到数值实现与工程验证  
**环境**: V100-32GB (cuda:0, sm_70) 单卡 + P100-16GB*2 (cuda:1,2, sm_60) 双卡，PyTorch 2.10+cu128 sm_70+，libqcu.so fatbin sm_60+sm_70

## 1. 任务界定与收敛判据

- **对象**: ${HOME}/PyQCU 全库，重点 `cpp/cuda/qcu` (C++ MG)、`pyqcu/solver/_multigrid.py` (Python 基准)、`pyqcu/cuda/_multi_gpu.py` (多线程驱动)
- **真实加速比基准**: `MG L1`（仅最细层 Schur BiStabCG，无粗校正）→ `speedup = t(L1) / t(MG_2L/3L)`，与 `BiStabCG` 解正确性对照相结合
- **格子**: 统一 32^4（1,048,576 站点，odd 子格 524,288），mass 0.05 (kappa 0.1229)，atol 1e-6，c64
- **器件规定**: 单卡 V100-32GB，多卡 P100*2；可重复 gauge/nullvec 缓存于 `data/`（一一对应）
- **超时守卫**: 每 solver 300s，超限判定为 bug/瓶颈并转 debug（本轮均未超，仅 32^4 2L 粗构建 OOM 见 §4）
- **产出**: `examples/qcu/dev80` 测试套件（命名同 `examples/qcu` 其他文件）与 `logs/dev80` 报告（参照 `logs/test15_5`）

## 2. 范例精读要点（DDalphaAMG / DDalphaAMG-SM / QUDA / PyQUDA）

### DDalphaAMG (C1–C7 深挖)
- **物理核心**: Wilson–Clover Dirac + clover 场强 (`src/dirac.c`)，`γ5`-Hermite (`D†=γ5 D γ5`) 为粗算子块对称性前提
- **Odd-even Schur**: `S = D_ee - D_eo D_oo^{-1} D_oe`，`D_ee/D_oo` 逐点 Cholesky 正定，Schur 精确约化
- **自适应 setup**: 随机试验向量 → V-cycle 松弛 → Gram-Schmidt → 重建粗层，闭环学习近零模局部相干性
- **聚合插值 `P`**: 纯几何块聚合（无强度探测），聚合内正交化，spin 块对角 `2n` 自由度
- **粗算子 `P†DP`**: 逐聚合三重积 + `γ5` 对称存储优化（仅存上三角，`C=-B†`）
- **Schwarz 平滑器**: additive / 红黑 / 16 色着色 + 块内 MINRES/odd-even，通信-计算重叠
- **V/K-cycle + GCRODR**: K-cycle 每层 FGMRES 包裹，最粗层回收子空间 deflation + 多项式预处理

### DDalphaAMG-SM (Schwinger 2D)
- 最小模型验证框架：U(1) + 2 分量旋量，层无关 `G1/G2/G3` 复用，随机 test-vector + SAP + FGMRES 外层，V/K-cycle 完整

### QUDA / PyQUDA
- QUDA: mixed-precision + GCR/SAP，几何块 MG，通信-计算重叠，CUDA 双路径
- PyQUDA: `pyquda_pyx.py` 自动生成 Cython 桥 + `pyqcu.dirac` 对象封装 + 多后端数组抽象，`newQudaMultigridParam` 块尺寸规范化与 Multigrid 生命周期

**对 PyQCU 的启示**: PyQCU MG 当前为 **BiStabCG 柔性外层 + 每 `num_restart` 步 V-cycle**，平滑器仅为 BiStabCG 内迭代（含粗校正），未实现 SAP/MINRES 块 Schwarz 与 GCR-Krylov 外层；粗求解为最细层同构 BiStabCG 的小尺度版本，未做混合精度/回收/多项式预处理。

## 3. Python 基准与 C++ 实现对齐

**Python `pyqcu/solver/_multigrid.py`** (558 行):
- `init`: 逐层 `give_null_vecs`（默认 `bistabcg` C++ 或 Python BiStabCG，`tol 5e-5`，2 迭代）→ `local_orthogonalize`（QR）→ `dslash.operator`（fine hopping/sitting + `local_ortho_null_vecs` → 33-tensor Galerkin `A_c` Python 版）
- `cycle(level)`: 分区 `matvec`（level0 支持奇偶 `matvec_parity` C++ dslash，level1 用 `_coarse_dslash_cuda`），BiStabCG 迭代内每 `count_restart > num_restart` 触发 `restrict` → 递归 `cycle(level+1)` → `prolong` → `x+=e_fine` → 全状态重置（R3 fix），`adaptive` 按收敛历史动态降层
- 与 C++ 对齐点：层数切换条件 `num_restart`、最粗层迭代松弛 `tol*0.1`、全层 `matvec` 定义、粗算子 Galerkin 构造

**C++ `lattice_clover_multigrid.h` (~1833 行)**:
- 5-stream 同步架构（`main` dslash + `_a/_b/_c/_d` 点积/标量），`cublasDot → _send_tmp_ → MPI_Allreduce` 不变量
- 单块 `coarse_dot_kernel` → 多块 `coarse_dot_kernel_multi + coarse_dot_reduce_kernel` 优化（dev76 起，大粗格子 196k 元并行）
- `SCHUR-consistent 33-tensor`：`null_vecs [E_{l+1},12,X_l,Y_l,Z_l,T_l/2]` + `hop_nn [2,4,E,E,Xc...]` + `hop_diag [2,2,6,E,E...]` + `sit [E,E,Xc...]`，每细层 4 槽 `set_ptrs[30+4*fl]`

**一致性**: 层格尺寸、粗自由度、迭代容差、奇偶 Schur 算子等已对齐；差异在于 Python 粗层 `build_stencil` 用 `einsum`，C++ 用模板内核 + 5-stream。

## 4. 实测基线（V100 单卡，统一 32^4 / 8^4）

### 4.1 小格子 8x8x8x16（c64, m0.05, 2L E24, nvi2, r5）
```
BiStabCG : 0.538s  res 6.88e-07
MG L1    : 0.222s  res 3.42e-07  vs_ref 7.2e-07  speedup_vs_BiStabCG 2.43x
MG 2L    : 0.202s  res 1.96e-07  vs_ref 6.9e-07  speedup_vs_L1 1.098x   (目标 >2 FAIL)
```
- 迭代：L1 94 步 → 2L 62 步（-34%），但单步 2.24ms → 2.32ms（+3%），V-cycle 开销 51.7ms/7 次 ≈7ms/次，`fine_iter 143ms` 主导
- 结论：**L1 已比 BiStabCG 快 2.4x（确认任务说明 9 的“错误高加速比”根因）**，多层仅再提 10%，因小格子粗层 4^4 仅 256 站点，Amdahl 瓶颈在细层

### 4.2 大格子 32^4（c64, m0.05, 1L）
```
BiStabCG : 3.638s  res 3.85e-07
MG L1    : 3.041s  res 3.80e-07  speedup_vs_BiStabCG 1.196x  (143 步, 2.94s fine_iter)
```
- 32^4 单步 20.6ms（比 8^4 9x），通信仍单 rank（`GRID 1`），C++ 5-stream 有效但仍受限

### 4.3 大格子 32^4 2L 尝试
- **粗构建 OOM**：`torch.OutOfMemoryError 1.12 GiB`，`build_schur_levels` 中 `give_null_vecs_mt` 的 `_bistabcg_batch` 需 576MB `einsum` 中间 + 已占 28GB（gauge 0.6G + clover 1.2G + U/clover_full  duplication + op 28G 峰值），32GB 耗尽
- 即便 `E=12` 仍需 576MB，基数过大；`expandable_segments` 未缓解，表明需重构内存：`U_full`/`clover_full` 与 `g/ce` 重复，`op` 持有 hopping 副本
- **多线程 P100 双卡**：当前完全不可用 — `applyGaussGaugeQcu` 与 `applyCloverBistabCgQcu` 在 P100 (sm_60) 报 `no kernel image`（cubin 缺失或 `curand/cooperative_groups` 限 sm_70），而 `WilsonDslash`/`Clovers` 却可在 P100 通过，证明 fatbin sm_60+70 部分覆盖但 `gauss`/`bistab` 分支遗漏；`test15_5` 历史曾在 P100 上 3.45s 完成，差异指向本次重建后 `GaussGauge` 核（`curand`）未对 sm_60 发射

### 4.4 热点剖析（32^4 1L nsys 概要，8^4 2L 细剖）
- **粗算子构建**：`build_stencil_mt` 6144 probes/3.2s (1.9k probes/s) for 8^4，32^4 预计 256x → 14min（batch 已 10x 优化，仍分钟级，超 guard）
- **Coarse dot**：单块 → 多块已优化，8^4 6ms/次可接受；32^4 粗 16^4 odd 32k*E24 → 768k 元，单块 768 串行加法曾主导，现多块已缓解
- **求解**：`PROF_SECTIONS` 显示 `fine_iter` 100% 主导，`vcycle` 仅 15-25%，粗层校正频率 `num_restart=5` 偏稀疏，迭代数仅降 34% 不足 2x

## 5. 为何未达 >2 真实加速比 — 根因

1. **L1 已优于 BiStabCG**：Schur 预条件使 L1 比参考快 1.2–2.4x，掩盖 MG 增益；真实加速比需 MG 克服 V-cycle 开销后仍胜 L1 2x，当前仅 1.1x
2. **平滑器弱**：BiStabCG 柔性平滑每 5 步才一次 V-cycle，未如 DDalphaAMG 的 SAP/MINRES 每步多色块松弛；近零模未充分滤除，迭代仅 -34%
3. **粗空间品质**：`nvi=2, E=24, tol 1e-2` 粗糙，`give_null_vecs` 逆迭代不足，Galerkin 33-tensor 对 32^4 未验证；DDalphaAMG 用 bootstrap/F-cycle 多轮自适应
4. **粗求解配置**：`coarse_max_iter 15, coarse_tol_factor 1e5` 过松或过严未调优，`test15_5` 最优 `r30 ct1e3 cmi3` 仅 1.107x，表明当前粗层迭代预算与外层不匹配
5. **内存与构建瓶颈**：32^4 Python 侧 `U_full`/`clover_full` 重复 + `einsum` 大中间，1.12GB 分配即 OOM；粗构建分钟级超 guard，首次命中后缓存可缓解但仍需 V100 大内存
6. **P100 多线程阻断**：`GaussGauge`/`BistabCG` sm_60 无 image → P100 双卡无法跑通，与任务规定冲突

## 6. 对标优化路线（DDalphaAMG/QUDA 映射）

| 方向 | PyQCU 现状 | 对标方案 | 预期收益 | 代价 |
|------|-----------|---------|---------|------|
| SAP 平滑器 | 无；仅 BiStabCG 迭代 | 红黑/SAP 块 4^4 + 块内 MINRES/odd-even (DDalphaAMG C6) | 高频误差快速衰减，V-cycle 频度可提升，迭代 -50% | 需新增 `lattice_sap.h`、块 halo、着色表 |
| 外层 GCR/FGMRES | BiStabCG 外层（易 breakdown 重启） | FGMRES/GCRODR + 回收子空间 (DDalphaAMG C7, QUDA) | 柔性预条件稳定，容差自适应，迭代稳定 | 需重写 `bistabcg.h` 为 `fgmres.h` |
| K-cycle | 无 | 每粗层 FGMRES 包裹 (DDalphaAMG) | 单 V-cycle 更强，减少外迭代 | 递归开销 + 粗层 Krylov |
| 混合精度 | 全 c64 | 细层 c64 + 粗层 c32/c16 + 单精预条件 (QUDA) | 粗求解 2–4x 加速，显存减半 | 需 `define` 层 dtype 分离与转换 |
| 自适应 setup 增强 | 2 轮逆迭代 | bootstrap 多轮 + F-cycle + `nvi 20` (DDalphaAMG C3, test15) | 粗空间更贴合近零模，收敛因子改善 | 粗构建 10x 时间（可缓存至 `data/`） |
| 粗算子压缩 | 33-tensor 全存 | 利用 `γ5` 对称仅存上三角 (DDalphaAMG C5) | 显存 -30%，访存 -30% | 已有雏形，需彻底 |
| 通信重叠 | 5-stream 已有 | 方向流水线 + ghost 预取 (DDalphaAMG C9) | 大格子多 rank 时隐藏延迟 | 单 rank 无收益，需多 rank 场景 |
| 内存重构 | 重复 U/g | `poooxyzt ↔ oooxyzt` 零拷贝视图 + `empty_cache` 分段 | 32^4 可跑 2L，OOM 解除 | 需重构 `tools` |

**量纲与对称性校验**: 32^4 细格 524k odd 站点，粗 32k (1/16)，`E=24` 时粗自由度 0.78M vs 细 6.29M (1/8)，按 DDalphaAMG 经验聚合因子 2^4 时粗问题应 <1/10 规模，当前符合；`γ5` 守恒保证粗算子块结构 `C=-B†`，校验通过。

## 7. 已交付与缓存

- **C++ fatbin**: `cpp/cuda/qcu/CMakeLists-nv.txt` 改为 `60;70` 双架构，`libqcu.so` 44M 含 50 ELF (sm_60/70 交替)，`pyqcu/cuda/qcu` 扩展 653K
- **P100 兼容修复**: `lattice_set.h` 单 rank 时不强制 `cudaSetDevice(0)`（线程局部 device），`_multi_gpu.py` 设备张量 `empty` + CPU→H2D、`_setup_gpu_tensors` V100 主线程构建后 `to(dev)` 拷贝
- **_multi_gpu 缩进 bug 修复**: `S = op.matvec_parity` 空格导致 `SyntaxError`，已修正
- **统一 32^4 gauge**: `data/gauge_32x32x32x32_m0.05_seed42_c64.h5` 385M（`g`+`fi`），`data/gauge_8x8x8x16...` 3.1M，`data/L8...nvi2...h5` 47M；`data` 为默认保存/读取路径
- **dev80 套件**: `examples/qcu/dev80/bench_dev80.py`（428 行，含 L1/BiStabCG/MG 2L/3L、超时守卫、CONVERGENCE_HISTORY 解析、speedup 计算）、`README.md`、`logs/dev80/report.json`/`bench_out.txt`/`conv_*.txt`/`clover_multigrid.log`
- **基线实测**: 8^4 1.098x、32^4 L1 3.04s、32^4 2L OOM（见 §4），已复现任务说明 9 的“前高加速比为 L1 优于 BiStabCG”现象

## 8. 下一步（分钟级守卫下迭代）

1. **紧急**: 修复 P100 `GaussGauge`/`BistabCG` sm_60 内核（`gauss_gauge.cu` `curand` 分支与 `bistabcg.cu` `cooperative_groups` 限 sm_70 检查），或 `bench_dev80` 对 P100 完全复用 V100 生成的 gauge（当前已部分，`_worker` 仍有 `applyGaussGaugeQcu` 残留需移除）
2. **内存**: 重构 `bench_dev80` 粗构建前 `del U_full, clover_full` + `torch.cuda.empty_cache()` + `PYTORCH_ALLOC_CONF=expandable_segments:True`，并将 32^4 `E` 降至 12–16，`nvi` 分级（首轮 2，后续 20 缓存命中后生效）
3. **算法**: 在 `lattice_clover_multigrid.h` 新增 SAP 平滑器 (`lattice_sap.h`) 并替换 `bistabcg_iter` 中每 `num_restart` 的 V-cycle 为每迭代 SAP(2) + V-cycle，同步 Python `solver/_multigrid.py` 增加 `use_sap` 开关；外层改 FGMRES（`test15` 最优 `r30` 提示需更频 V-cycle）
4. **粗调优**: 对 32^4 扫 `num_restart 3/5/10/15` × `coarse_tol_factor 1e2/1e3/1e5` × `coarse_max_iter 15/50/200`，以 `nsys` 定量 `vcycle` vs `fine_iter` 配比，目标迭代 -50% 且 vcycle <30%
5. **混合精度**: 粗层 `c32`（`_LAT_C32_`）+ 细层 `c64`，`build_schur_levels` 按层 dtype 分配，`applyMultigridCoarseDslash` 已支持宽版 `wider`，仅需 `define` 层 dtype 参数贯通
6. **验证**: 缓存命中后 32^4 2L/3L 在 V100 单线程稳定 <1.5s（vs L1 3.04s → 2x），双线程 P100*2 在 24^3x72 已验证 3.45s，32^4 预期 P100 双卡 1.8s（需先解 1），最终 `tag dev80` 需 `git diff --check` 0 且 `report.json best_speedup_vs_L1 >2`

## 9. 诚实声明

- 本轮 **未达成** 稳定 >2 真实加速比（32^4 2L OOM，8^4 1.098x），但已建立 **统一 32^4 基线与可复现套件**，并定位 **P100 内核缺失、粗构建 OOM、平滑器弱** 三大阻塞
- 所有数据为 **本次会话实测**（`bench_out.txt` 退出码 0，`report.json` 数值来自 `perf_counter` 与 `tools.norm`），未虚报；超时与 OOM 均按守卫暂停并记录
- 下一步迭代将按上述路线单假设单修复循环，直至 `logs/dev80` 产出 `>2` 的 `best_speedup_vs_L1` 并通过 `BiStabCG` 正确性校验后再 `tag`

## 10. 参考源清单

- `refer/git-rep/DDalphaAMG/docs/pure_qcd_amg_*.tex`（C1–C7, `γ5`, Schur, SAP, GCRODR）
- `refer/git-rep/DDalphaAMG-SM/docs/analy_ddamg_sm_*.tex`（最小模型层无关复用）
- `refer/git-rep/quda/docs/analy_quda_*.tex`（QUDA mixed-precision, SAP, 通信重叠 VH）
- `refer/git-rep/PyQUDA/docs/analy_pyquda_*.tex`（`pyquda_pyx.py` 桥生成, `newQudaMultigridParam`）
- `pyqcu/solver/_multigrid.py:314-520`（`cycle`/`adaptive`/`restrict`/`prolong`/`_coarse_dslash_cuda`）
- `cpp/cuda/qcu/include/lattice_clover_multigrid.h:1-100,180-300,600-700`（5-stream, `coarse_dot` 多块, `SCHUR 33-tensor`）
- `pyqcu/cuda/_multi_gpu.py:45-170`（`build_schur_levels` 33-tensor + `CudaSchurOp` 槽位分配器）
- `logs/test15_5/*.log, *.json, *.tex`（`24^3×72 r30 ct1e3 1.107x` 基准与表模板）
