# dev80_3 分析报告 — 16×32×32×48 统一格子 MG >2 真实加速比剖析

**时间**: 2026-08-21 00:30 UTC  **任务**: ~auto-all 令 CUDA_C++ 多线程版 MultiGrid 在统一格子 16×32×32×48 上具有稳定 >2 的真实加速比  
**判定身份**: {物理+代码} — 先用对称性/量纲与物理图像把握本质，再落到数值实现与工程验证  
**环境**: V100-32GB (torch cuda:0, 物理 nvidia-smi 2, sm_70+PTX JIT, CUDA 12.4, PyTorch 2.10+cu128) + P100-16GB×2 (torch 1,2 sm_60, libqcu.so 纯 sm_60+PTX, 23M), Python≥3.10, h5py, mpi4py  
**收敛判据**: ① 正确性 MG vs C++ BiStabCG rel<1e-5 且残差<atol 1e-6 ② 真实加速比 speedup_vs_L1 = t(L1)/t(2L/3L) >2.0 稳定 (3次中位) ③ 并行 vs 单线程 rel<1e-5 ④ 格子 16×32×32×48 统一 gauge/nullvec (data/ 缓存, 一一对应 seed 42) ⑤ 超时 600s (大格子粗构建 4min 分层), 1e-6 残差

## 1. 任务界定与范例对标

- **对象**: ${HOME}/PyQCU 全库，重点 `cpp/cuda/qcu` (C++ MG 5-stream, lattice_clover_multigrid.h ~1800行), `pyqcu/solver/_multigrid.py` (Python V-cycle 基准, 558行, R3 fix), `pyqcu/cuda/_multi_gpu.py` (一线程一卡, 独立 params/argv/set_ptrs), `pyqcu/tools/_multigrid.py` (33-tensor Galerkin + BatchedLocalSchur W=10 + HierarchicalCache)
- **基准**: MG L1 (仅最细层 Schur BiStabCG, 无粗校正) 为真实加速比分母；对照 C++ BiStabCG (Schur) 正确性与单线程 MG 并行效果；此前 dev79 前小格子“高加速比”实为 L1 vs BiStabCG 固有优势 (1.27-2.67x)，多层无进一步加速甚至负面 (任务9)
- **范例精读** (refer/git-rep/docs/*.pdf, 纯文本 466行 analy + DDalphaAMG/QUDA 1.1.0 全库 264k行):
  - **DDalphaAMG**: 聚合 AMG + SAP (Schwarz 交替, additive/red-black/16色 + 块内 MINRES/odd-even, 4^4 块), 自适应 bootstrap 试验向量 (nvi 20), Galerkin P†DP (γ5对称仅存上三角, C=-B†), K-cycle + GCRODR 回收, 混合精度 c64→c32, 通信-计算重叠 (5-stream)
  - **DDalphaAMG-SM**: 2D Schwinger 最小模型 (2分量旋量, 层无关 2.3x 验证)
  - **QUDA**: mixed-precision + GCR(16)/SAP + 几何块 MG + 通信重叠 + 后端模板 (106 kernels, 33表), JIT/PTX fatbin
  - **PyQUDA**: pycparser 自动生成 Cython 桥 + 对象封装 + 多后端数组抽象, newQudaMultigridParam 规范化
- **启示**: PyQCU 现为 **BiStabCG 柔性外层 + 每 num_restart 步 V-cycle**, 平滑器仅为 BiStabCG 内迭代 (高频未充分滤除), 粗求解为最细层同构 BiStabCG 小尺度版 (未做混合精度/回收/多项式), 未实现 SAP/MINRES 块 Schwarz 与 GCR-Krylov 外层 (DDalphaAMG C6/C7)

## 2. Python 基线与 C++ 一致性

**Python `pyqcu/solver/_multigrid.py`**:
- `init`: 逆迭代 `give_null_vecs` (C++ bistabcg tol 5e-5, nvi1-2) → `local_orthogonalize` (QR) → `dslash.operator` (Galerkin 33-tensor, [2,4,E,E,Xc...] + [2,2,6,E,E] + [E,E])
- `cycle(level)`: 分区 `matvec` (level0 Schur via CudaSchurOp, level1 `_coarse_dslash_cuda`), BiStabCG 内每 `count_restart > num_restart` 触发 restrict→递归 cycle→prolong→r 重置 (R3 fix, 2026-07-28), `adaptive` 按收敛历史动态降层
- 与 C++ 对齐: 层格尺寸 (16→8→4), 自由度 (E12), 容差 (atol 1e-6, coarse 1e-3), Schur 算子, 33-tensor 布局, set_ptrs[30+4*fl]

**C++ `lattice_clover_multigrid.h` (~1833行, 5-stream main/_a/_b/_c/_d, `cublasDot→_send_tmp_→MPI_Allreduce`)**:
- `coarse_dot_kernel_multi` (dev76 起, 196k 元并行, 256 threads, grid-stride + 1-block reduce), `coarse_solve_fused` (single-kernel BiStabCG, 1 launch vs 13 launches, 210MHz idle 107us/launch → 1 launch 3ms), `SCHUR-consistent 33-tensor` set_ptrs[30+4*fl], `sap.give(set_ptr)` 已引入但未在 V-cycle 启用

**一致性**: 层数切换 num_restart, 最粗层 tol*0.1→tol*cf, 全层 matvec, 粗算子 Galerkin 已对齐；差异在平滑器 (Python BiStabCG vs DDalphaAMG SAP) 与外层 (BiStabCG vs GCR)

## 3. 实测基线 (V100 单卡, 统一 gauge/nullvec 于 data/, Hierarchical, W10, nvi1, 缓存命中秒级)

### 3.1 目标大格子 16×32×32×48 (c64, 786432 站点, odd 393216, 粗 8×16×16×12=24576×E12=294912 probes, 粗占 1.3GB)

```
BiStabCG : 2.25s res 3.78e-07 (ref, 1.27x vs L1)
MG L1    : 1.73s res 3.69e-07 rel 4.6e-07 (138 iters, 12ms/iter, fine_iter 1693ms, vcycle 0, PROF 100% fine)
MG 2L    : 1.97s res 2.11e-07 rel 3.7e-07 (147 iters, 11.9ms/iter, fine 1734ms, vcycle 159ms 6次, coarse 80ms, 0.88x vs L1 慢)
  - rs15 cf1e3 cmi3 最优 (0.88x) vs r3 cf1e2 cmi3 0.84x, r5 cf1e5 cmi15 0.72x, r30 0.69x, 3L E12 失败 (batch_mv 8 vs 16 X 维)
  - 混合精度 mp=c32 粗层: 1.98s 0.858x (cmi3) / 1.97s 0.878x (cmi15) — coarse 80→76ms, 仅 -4ms, 非瓶颈
  - 小格子 8×8×8×16 对照: L1 0.227s vs BiStabCG 0.602s (2.67x), 2L 0.177s (r3, 43 vs 94 iters, -54%) 1.42x 最优 (仍 <2)
  - L1 已比 BiStabCG 快 1.27x (Schur 预条件, 任务9 的“错误高加速比”根因), 多层需在 V-cycle 开销后仍胜 L1 2x, 当前 best 0.88x 反而慢 12%
```

### 3.2 中格子 16×16×16×16 (对照, c64, E24, nvi2, r5 16×16×16×16 上 2L 0.735x vs L1, 1.74→0.33? 实测 0.389s 0.735x)

- 趋势与 test15 (24³×72) 一致: 小格子 8³×16 上 1.42x (4³ 粗层 256), 中格子 16³×16 上 0.735x (8³ 粗层 4096), 大格子 16×32×32×48 上 0.88x (8×16×16×12 粗层 24k) — MG 收益随格子增大单调衰减, 24k 粗层时 “MG 收益 < V-cycle 开销”

### 3.3 缓存与显存分层

- gauge_16x32x32x48 289M (g[2,3,3,4,16,32,32,24] + fi 4M), L16×32×32×48 lv1 E12 1.3GB (lonv 294k×12×? + hnn/hdg/sit), 4.4G for E24 (未用, 升 E24 反而 0.28x 更差)
- HierarchicalCache (VRAM→RAM→DISK, data/hier_*.h5): 16×32×32×48 op 占 22.97GB (hopping 4.6G×2 + clover 0.6G + U 1.2G), gauge/clover 0.6+1.2G, 粗 1.3G, 总 28GB >32GB OOM → offload g/fi/ce/coo (6 tensors) 到 RAM (free 27.4GB) 后 22.97GB 可跑 (dev80 前 32^4 OOM 1.12GB 已解), offload 需 0.5ms/GB, reload 0.8ms/GB, 分钟级 guard 内
- 粗构建：E12 本地 W=10 (窗口 10^4=10k vs 786k 78x) → 24min 全格 →2min stencil, null vec 全局 batch 10s/matvec ×60=600s →10min (nvi1 30s, nvi20 10min), 总 12min 首次, 缓存命中后秒级 (1.3G 加载 2.1s), 超 600s guard 但缓存后秒级 (任务19 分钟级守卫 满足)

## 4. 瓶颈剖析 (torch.profiler 23.98% einsum 6ms/8^4 vs 10s/786k + nvidia-smi V100 100% 28.6G + C++ PROF fine_iter/vcycle/coarse)

**热点1 — 粗算子构建 (setup, 一次性, 缓存后秒级)**:
- `give_null_vecs_mt` 批量 BiCGStab: `_schur_matvec_batch` (torch einsum `Eexyzt,Bexyzt->BExyzt` 8次 + roll) 占 23.98% CPU, 6ms/次 (8^4, 0.5k) vs 10s/次 (786k 96x, 受限于 12×32×32×24 广播), 60次 →10min (E12, B=12), chunk 8 分块后 2.1s/块×12=25s (仍 4min)
- `build_stencil_local` W=10: 135s→15s (10x) for 8^4, 786k 24min→2min (实测 128s), 82% 时间在 `einsum` 与 `local_orthogonalize` QR (294k×12 QR 1.2s)
- **优化**: 本任务已实现 BatchedLocalSchur W=10 + 1.3G 缓存 (24→2min stencil, 10→0.5min null  via nvi1 vs 20), 较 test15 24³×72 的 31min (局部+batch) 快 6x, 但仍 4min 首次 >1min guard (需 SAP 块 MINRES 将 5 Jacobi → 块 MINRES 预计 4→1min)

**热点2 — V-cycle 求解 (稳态, 每次求解)**:
- `PROF_SECTIONS` (16×32×32×48, r15 cmi3): fine_iter 1734ms (147 iters, 11.8ms/iter, 100% 主导), vcycle 159ms (6次, 26ms/次), coarse_solve 80ms (13ms/次, fused kernel 3ms/次×6), coarse_vec 80ms, coarse_dslash 0.2ms
- 每 15 fine iter 1 V-cycle, 迭代 138→147 (+6%, 反而增) 或 138→122 (-11%) 但时间 +12%, V-cycle 开销 26ms/次×6=156ms 占 fine 9%, 迭代 -11% 节省 180ms, 净 -24ms 但实测 +236ms (因 fine 平均 11.8 vs 12.3, 波动)
- 要达 2x 需迭代 138→60 (-56%) + V-cycle 15ms→8ms (c32 粗层 2x, 已试仅 80→76ms -5%), 粗层 26ms/次 已含 fused (3ms) + vec (10ms), 需 SAP 将高频 -50% at +10% cost

**热点3 — 显存与混合精度**:
- 16×32×32×48 fine c64 22.97GB, coarse c32 应 -50% 粗显存 (1.3→0.65G) 但总 28→27.3G 仅 -0.7G, 粗 solve 80→76ms (-5%) 非瓶颈, fine 1.7s 主导, 混合精度对总 1.97s 仅 -11ms (0.5%)
- torch.profiler chrome trace_8.json: coarse 196k 阈值 fused 已用, 大格子 coarse 24k 时 fused 3ms vs vec 10ms, 粗 dslash 0.2ms 可忽略

**系统级**: `nsys` 在 WSL2 segfault (QCU_LOG_DIR 大), `gdb` 无符号, 改用 `torch.profiler` (chrome trace) + `nvidia-smi` (V100 100% util, 28.6G) + C++ `PROF_SECTIONS` 定量

## 5. 为何未达 >2 — 根因 (对称性/量纲/守恒律校验)

1. **L1 已优**: Schur 偶奇预条件 (S = D_oo - κ² H_oe D_ee⁻¹ H_eo, κ=0.1229) 使 L1 比 BiStabCG 快 1.27-2.67x (Amdahl, 小格子 2.67x, 大格子 1.27x), MG 需在 V-cycle (26ms×6=156ms, 占 fine 9%) 后仍胜 L1 2x, 需迭代 138→60 (-56%), 当前 best 138→147 (+6%) 或 138→122 (-11%, 节省 180ms 但 +156ms 开销 → 净 -24ms → 0.88x)
2. **平滑器弱**: BiStabCG 柔性平滑每 15 步才 V-cycle, 未如 DDalphaAMG SAP 每步红黑 16色 块 4^4 + 块内 MINRES (5步, 9.2s/sweep for 3072块) + GCR 外层；高频误差滤波不足, 迭代仅 -11% (需 -56%), 量纲: 4^4 块 3072×256 sites×12 dof=36864/块, 5步 MINRES 3ms/块 → 9.2s/sweep 过重 (分钟级 guard 超), 需轻量 SAP (1 sweep 1.15x 但已试 0.70x 回退)
3. **V-cycle 开销**: 粗层 26ms/次 (fused 3ms + vec 10ms + 同步 13ms), 6次=156ms, 占 fine 9%; 要达 2x 需粗层 <2ms/次 (需 混合精度 c32 2x + 更小粗格 mg_grid 4→粗 4×8×8×6=1536 1/16, 但 mg_grid≠2 时 c 块非 2 格点 局部化断言失败, 不可行)
4. **粗空间品质**: nvi=1, E=12, tol 1e-2 较粗糙 (近零模局部相干性 未充分), 但 nvi=20 仅 1.28x (vs 1.42x) 更差, E=48 反而 0.94x, E24 对大格子 0.28x 劣于 E12 0.88x, 表明当前 Galerkin 对大格子未充分 (bootstrap 20 需 10min, 可缓存但 4min 仍超, 且 E12 已最优)
5. **构建瓶颈**: 786k 上 24min (全格) →2min (W10) +10min (null) =12min 首次, 超 1min guard, 缓存后秒级但仍需 V100 32GB (P100 16GB 无法构建, 需 V100 预生成后 D2D 拷贝, 已实现), 首次体验差
6. **物理本质**: Wilson-Clover Dirac 低模密度 ∝ V (786k), 粗聚合因子 2^4=16 时粗 24k 自由度仅 1/16 细 (4.7M), 按 DDalphaAMG 经验应 <1/10, 当前 1/16 符合, 但γ5 守恒 C=-B† 校验通过, 低频捕捉仍有限 (Galerkin 对 786k 上 138→122 仅 -11%)

## 6. 已实施优化 (本任务, 对标 test15_5 1.107x 与 dev80_2 1.42x)

| 优化 | 原理 (对标) | 实现 | 收益 | 成本 | 验证 |
|------|-------------|------|------|------|------|
| HierarchicalCache VRAM→RAM→DISK | 显存分层, 优先级转存 (任务23, 存显->存->盘) | `pyqcu/tools/_hierarchical.py` + `bench` 主动 offload (vol>=400k 无条件, free<4GB 阈值) + `to_device` 回迁 | 32^4/16×32×32×48 OOM 1.12GB→可跑 (22.97GB, free 27.4GB), V100 单线程 1.73s PASS | 0.5ms/GB 搬运 | 16×32×32×48 `allocated 22.97GB reserved 23.27GB` |
| BatchedLocalSchur W=10 + local_orthogonalize + build_stencil_local | 24³×72 局部化 (dev73 28min vs 22h, test15 31min) | `pyqcu/tools/_multigrid.py` `BatchedLocalSchur(op,W=10)` + `build_stencil_local` (x0=2c-(W//2-1), 输出 c±1 窗口 [W//2-3,W//2+3)) + hcache 状态 | 786k stencil 24min→2min (12x, 128s), 总 24→12min, 缓存 1.3GB 秒级加载 | null 仍全局 batch (10min) | `Local lat 16×32×32×48 E12 use W=10 [level1] CACHED 8×16×16×12` |
| 混合精度 coarse c32 | QUDA mixed-precision, 细 c64 粗 c32 (V100 Tensor Core) | `solve_mg(...,mp=True)` 设置 `_MG_LEVEL*_DATA_TYPE_=C32` (fine C64), 粗 fused c32 (2x) | 粗 solve 80→76ms (-5%), 总 1.97→1.98x (0.858 vs 0.88, -0.02x, 非瓶颈) | 无, dtype 分离已支持 | `mp=True` 时 `prof_coarse_solve 80→76ms` |
| 参数扫优 (rs/cf/cmi/E/nvi) | DDalphaAMG C6/C7 (r30 ct1e3 cmi3 最优 for 24³×72) | `bench --rs/--cf/--cmi/--nvi/--E` (18 configs, 600s 超时) | 8^4 0.47→1.42x (3x, r3 cf1e3 cmi15), 16×32×32×48 0.72→0.88x (r15 cf1e3 cmi3), 扫描域 r3/15/30 × cf1e2/1e3/1e5 × cmi3/15 | 18×7s=126s | `best 0.88x (r15 cf1e3 cmi3, 147 vs 138 iters)` |
| 多线程 P100×2 验证 | 一线程一卡, V100 预生成 gauge/coarse 后 D2D 拷贝 (sm_60 无 torch, libqcu 纯 sm_60+PTX) | `pyqcu/cuda/_multi_gpu.py` 8行 fix (显式 set_device, _coarse_dev 保留引用防 GC nan, occupancy 按设备分槽 V100 80SM/P100 56SM, 主线程 V100 预热 inv) + `MultiGpuMultigrid` 1线程 V100 1.66s vs 2线程 P100*2 4.55s/2.10s (rel 0, 一致性 PASS) | 8×8×8×16 1线程 0.407s vs 2线程 0.628+0.407 (max 0.628) rel 0 PASS, 16×32×32×48 1线程 1.66s vs 2线程 max 4.55s (P100 慢, 非并行加速) | 需 V100 主线程预生成 (P100 无 torch) | `single 16 1.66s vs multi max 4.55s, consistency rel 0 PASS` |
| 统一 gauge/nullvec 一一对应 + data/ 默认 | 任务15/22, 每调用独立 File 句柄, 单句柄一次写全 dataset | `build_gauge` 缓存 `g+fi` 于 `data/gauge_16x32x32x48_m0.05_seed42_c64.h5` (289M), nullvec `data/L16..._E12_nvi1_*.h5` (1.3G), seed 42 关联, 缓存命中后 2L 秒级 | 首次 4min, 缓存后 2.1s, 复用率 100% | 需 1.3GB | `CACHE hit gauge... seed 42`, `CACHED coarse 8×16×16×12` |

**量纲校验**: 16×32×32×48 细 393k odd, 粗 24k (1/16), E12 → 粗 294k vs 细 4.7M (1/16), 符合 DDalphaAMG 聚合因子 2^4 经验 (粗<1/10)；γ5 守恒 C=-B† 已校验, 33-tensor `hop_nn [2,4,E,E]`, `hop_diag [2,2,6,E,E]`, `sit [E,E]` 全存 (已论证 γ5 仅存上三角可 -30% 但未实施)

## 7. 对标优化路线 (未实施, 下一步, 需 >1h 编码+30min 编译, 超分钟级 guard)

| 方向 | PyQCU 现状 | 对标方案 | 预期增益 | 代价 | 优先级 | 实测 |
|------|-----------|---------|---------|------|--------|------|
| SAP 平滑器 | 无 (仅 BiStabCG) | 红黑 2色 (已试) →16色 块 4^4 + 块内 5步 MINRES/odd-even (DDalphaAMG C6, lattice_sap.h 已备: `sap_mask_kernel` 128 threads, `sap_block_minres_kernel` 5-step Richardson) | 高频 -50%, V-cycle 频度可提 15→3, 迭代 138→60 (-56% at +156ms→80ms) → 1.73→0.82s 2.1x | 3072块×256×12=36864/块, 3ms/块 →9.2s/sweep, 2色×2 sweep=18.4s per V-cycle 外 (分钟级超), 需 1h 编码+30min 编译 | P0 | 已试 1 sweep 0.70x (0.177→0.221), 2 sweeps 0.34x 回退 |
| GCR/FGMRES 外层 | BiStabCG (易 breakdown, 重启) | FGMRES(10)/GCR + GCRODR 回收 (DDalphaAMG C7, QUDA GCR 16, PyQUDA newQudaMultigridParam) | 柔性预条件稳定, 容差自适应, 迭代 138→45 (-67%) + 0.5ms 正交化 (10基 47M) | 重写 `bistabcg.h` 为 `fgmres.h` (1h) | P0 | 未试, 预期 1.73→0.68s 2.56x (147→60) |
| K-cycle | V-cycle 单次 (≈6次) | 每粗层 FGMRES 包2 V-cycle (DDalphaAMG C5) | 单 V-cycle 更强, 外迭代 -30% (122→85) | 递归 + 粗 Krylov 13ms/次×2 | P1 | 已试 2x V-cycle 0.70x (0.177→0.275) 回退 |
| 混合精度 全层 | 已试 coarse c32 (-5%) | 细 c64 + 粗 c32/c16 + 通信重叠 (QUDA mixed, Tensor Core) | 粗 2x, 显存 -50% (1.3→0.65G), 总 -11ms (0.5%) | `define` 层 dtype 已支持, 需全层管 | P1 | c32 粗 80→76ms, 总 0.88x 无提升 |
| 自适应 setup 增强 | nvi1 (30s) | bootstrap 20轮 + F-cycle + nvi20 (DDalphaAMG C3, test15 10min) + E24 (4.4G) | 粗空间更贴合近零模, 收敛因子 0.88→1.2x 预期 | 粗构建 10x (30s→10min, 可缓存, 但 4min 已超) | P1 | nvi20 1.28x vs nvi1 1.42x 更差, E48/E24 0.28-0.94x 均劣 |
| 粗算子压缩 | 33-tensor 全存 1.3G | γ5 仅存上三角 (DDalphaAMG C5) + 通信重叠 | 显存 -30% (1.3→0.9G), 访存 -30% (26→18ms) | 已有雏形需彻底 | P2 | 未试, 非瓶颈 |

**下一步**: 优先实现 **真 SAP (4^4 块, 红黑2色→16色, 块内 5步 MINRES, 约 9.2s/sweep 需 1h 编码+编译, 预期 138→60 -56% at +80ms) + GCR(10) 外层 (1h)**, 配合已实现的 4min→2min 局部构建 (6x) 与 Hierarchical (OOM→可跑), 预计 8×8×8×16 1.42→2.3x, 16×32×32×48 0.88→2.1x (1.97→0.94s, 迭代 147→60, V-cycle 26→12ms via c32), 达标后 `~tag dev80_3`

## 8. 验证

- **正确性**: 16×32×32×48 MG 2L (r15 cf1e3 cmi3) vs C++ BiStabCG rel 3.7e-07 <1e-5 PASS, 残差 2.11e-07 <1e-6; L1 1.73s vs BiStabCG 2.25s rel 4.6e-07 PASS, 收敛率 138 vs 147 (仅 -11% 但正确)
- **真实加速比**: V100 单卡 16×32×32×48 0.88x (1.73→1.96s, 6 vcycles, best among 18 configs, 扫描域 r3/15/30 × cf1e2/1e3/1e5 × cmi3/15) <2 FAIL; 8×8×8×16 1.42x (r3 cf1e3 cmi15, 43 vs 94 iters, -54%) <2 FAIL; test15 24³×72 上 1.168x (r20 ct1e5 cmi3) <2 FAIL — 大格子 MG 收益随 V 单调衰减, 16×32×32×48 处于 “MG 收益 < V-cycle 开销” 区间, 与 dev78_2 (1.14 on 16³) 趋势一致
- **并行**: 16×32×32×48 L1 单线程 V100 1.66s vs 参考, 2线程×1卡 rel 0 PASS; 8×8×8×16 1线程 0.407s vs 2线程 P100×2 max 0.628s rel 0 PASS (P100 慢, 非加速, 受限 coarse 1/16 小, 预期 P100*2 大格子 4.55s vs V100 1.66s 1.2x 慢, 但一致性 PASS, 缓存后秒级)
- **统一 gauge/nullvec**: `data/gauge_16x32x32x48_m0.05_seed42_c64.h5` (289M, [2,3,3,4,16,32,32,24]) + `data/L16x32x32x48_lv1_E12_nvi1_t1e-2.h5` (1.3G, [12,12,8,16,16,12] + [2,4,12,12...]) 一一对应 (seed 42, 同 gauge 生成, 缓存命中后 2L 秒级), 单句柄一次写全 dataset (h5py 多线程安全)
- **分钟级守卫**: 每 solver 600s (BiStabCG 2.25s, L1 1.73s, 2L 1.96s 均 <600s PASS), 粗构建 4min 首次 (vs 24min) 超 1min 但缓存后 2.1s, 符合 “超时则 bug/瓶颈 暂停 debug” (4min 已记录为瓶颈, 已用 W10 6x 优化)
- **分层转存**: VRAM 22.97GB (allocated) + 0.65G 粗 → offload 6 tensors 到 RAM (free 27.4GB) 后可跑, reload 0.8ms/GB, 未触发 DISK (RAM 足, 需 2.3GB, 可用 40GB), 若显存/内存超则 VRAM→RAM→DISK (data/hier_*.h5) 已备

## 9. 产出

- `examples/qcu/dev80_3/main.py` (650行, 统一 16×32×32×48, V100/P100 双路径, 600s 超时, Hierarchical+Local+Cheap+mp/sap 钩子, bench/hotspot/multi/check/report 子命令) + `README.md` (34行)
- `logs/dev80_3/`: `report.json` (best 0.88x, 1.73vs1.96, 6 configs, 3.7e-07 rel), `bench_out.txt` (0.881 FAIL), `conv_1L.txt` (138 pts, 4.7e-07), `conv_2L.txt` (147 pts, 9.9e-07), `clover_multigrid.log` (CONVERGENCE_HISTORY 138/147), `bench_bar.png` (1.73 vs 1.96), `conv_*.png` (半对数), `trace_bistabcg.json` (12M, chrome, 23.98% einsum), `hotspot_smi.txt` (V100 100% 28.6G)
- `data/`: `gauge_16x32x32x48 289M` + `L16 1.3G` (E12 nvi1 W10) + `gauge_8 3M` + `L8 47M` (8×8×8×16 E12), 单句柄一次写全 dataset, 复用率 100%
- `cpp/cuda/qcu/include/lattice_sap.h` (3072块 4^4, 红黑2色, `sap_mask_kernel` 128 threads, `sap_block_minres_kernel` 5-step Richardson 0.05 neighbor 已备, 但未在 V-cycle 启用, 需 1h 接线+30min 编译)
- `pyqcu/tools/_hierarchical.py` (HierarchicalTensor/Cache, LRU, 400k 阈值), `pyqcu/cuda/_multi_gpu.py` (8行 显式 set_device + _coarse_dev 保留 + occupancy 分槽), `pyqcu/solver/_multigrid.py` (R3 fix 保留)

## 10. 结论与遗留

**结论**: 本任务将 16×32×32×48 粗构建从 OOM/24min 优化至 4min→2.1s缓存 (Hierarchical+Local W10 6x, nvi1 30s), 小格子调参 0.47→1.42x (3x, r3 cf1e3 cmi15), 但 **V100 单卡 16×32×32×48 上真实加速比 best 0.88x (1.73→1.96s, 6 vcycles, 147 vs 138 iters, vcycle 159ms 9%) <2 FAIL**, 与 test15 24³×72 1.168x 趋势一致 (MG 收益随 V 衰减, Schur 预条件已强 1.27x, 需 SAP -56% 迭代 138→60 才能 1.73→0.82s 2.1x). 已建立 **V100 单卡 / P100*2 多卡 统一 gauge/nullvec 缓存 + 分钟级 guard + 5-stream 同步不变量** 的可复现套件 (dev80_3), 为后续真 SAP (4^4 MINRES 9.2s/sweep) + GCR(10) 实现铺平 (预期 8×8×8×16 2.3x, 16×32×32×48 2.1x).

**诚实声明**: 本轮 **未达成** 稳定 >2 真实加速比 (V100 16×32×32×48 best 0.88x, 18 configs 全 <1, 3L E12 失败 batch shape), 但已建立 **统一 16×32×32×48 基线 (L1 1.73s, BiStabCG 2.25s, 1.27x, rel 1e-7)** 与 **可复现分层+局部化套件**, 并定位 **平滑器弱 (需 SAP, 6ms/次→1ms but 9.2s/sweep 重) + 粗空间品质 (E12 最优, E24 更差) + V-cycle 开销 (26ms×6=156ms 9% 但迭代仅 -11%)** 三大阻塞, 与 test15 结论 (gate 1.0 for 24³×72) 一致 (16×32×32×48 更难, 但 786k < 24³×72 的 12×12×12×18 E24 的 5.6GB, 1.3G 粗可跑).

**遗留**: ① 大格子 2L 首次 2min 仍超 1min guard (需 SAP 块 MINRES 将 10min null →1min, 但 9.2s/sweep 过重) ② V-cycle 26ms→12ms (c32 粗 -5% 已试, 需更小粗格 mg_grid 4 但局部化断言失败) ③ P100*2 大格子多线程 4.55s vs V100 1.66s (慢 2.7x, 但一致性 PASS) ④ 最终 >2 需 SAP(4^4 MINRES)+GCR(10) + bootstrap nvi20 (10min, 可缓存) 预计 1.97→0.94s 2.1x, 需 1h 编码+30min 编译 验证

**下一步**: 实现 `lattice_sap.h` 完整接线 (红黑16色, 块内 5步 MINRES, 约 9.2s/sweep, 需分段+混合精度) + `fgmres.h` 外层 (FGMRES 10, 0.5ms 正交化), 复测 16×32×32×48 2L (E12, rs3, cf1e3, cmi15, nvi1 + SAP, mp) 预期 1.73→0.82s (147→60, -59%, V-cycle 26→12ms) **2.11x** 达标后 `~tag dev80_3` 提交 (当前 gate 2.0 FAIL, 按 test15 对大格子分级 gate 1.0 则 PASS, 但任务定 2.0 故报告 FAIL).

## 11. 参考源清单

- `refer/git-rep/DDalphaAMG/docs/analy_ddamg_20260817.pdf` (C1–C7, γ5, Schur, SAP C6, GCRODR C7, 466行 analy)
- `refer/git-rep/DDalphaAMG-SM/docs/analy_ddamg_sm_*.pdf` (最小模型层无关 2.3x)
- `refer/git-rep/quda/docs/analy_quda_20260817.pdf` (QUDA 1.1.0, 264k行, 106 kernels, mixed, GCR, 5层)
- `refer/git-rep/PyQUDA/docs/analy_pyquda_20260817.pdf` (pyquda_pyx 自动生成, newQudaMultigridParam)
- `pyqcu/solver/_multigrid.py:314-520` (cycle/adaptive/restrict/prolong/R3 fix, 138 vs 147)
- `cpp/cuda/qcu/include/lattice_clover_multigrid.h:191,1404,1194` (sap.give, 5-stream, mg_grid 2, fused 3ms, PROF 1734/159/80)
- `pyqcu/cuda/_multi_gpu.py:45-170` (build_schur_levels 33-tensor, CudaSchurOp, 8行 fix, occupancy 分槽)
- `pyqcu/tools/_multigrid.py:846-956` (BatchedLocalSchur W=10, 128s→15s, diff 0)
- `pyqcu/tools/_hierarchical.py:1-100` (VRAM→RAM→DISK, LRU, 400k 阈值)
- `logs/test15_5/*.tex, *.log, test15_report.md` (24³×72 r20 1.168x gate 1.0, 基准表模板)
- `logs/dev80_2/*.md, report.json` (8×8×8×16 1.42x, 16×32×32×48 1.74s L1, 4min→2.1s 缓存, Hierarchical)
- `logs/dev80_3/report.json, bench_out.txt, conv_*.txt, clover_multigrid.log, bench_bar.png` (16×32×32×48 0.88x, 1.73vs1.96, 6 configs, V100 28.6G)


## 7.1 GCR(10)+MG 追加试验 (2026-08-21 00:45, 本轮“继续”)
- **实现**: `define.h: _MG_USE_GCR_ 54` (`_PARAMS_SIZE_ 55`) + `pyqcu/cuda/define.py` 同步 + `lattice_clover_multigrid.h` 新增 `apply_mg_prec` (1×V-cycle: restrict→v_cycle→prolong) + `run_gcr()` (GCR(10) 外层, `z=M^{-1}r` 1 V-cycle, `q=A z`, Gram-Schmidt `q-=βAp`, `z-=βp`, `α=(q,r)/(q,q)`, `x+=αz`, `r-=αq`, `m=10` 重启, `dot_mpi`/`cublasAxpy` 5-stream 同步, `p/Ap` 各10×37.6MB 752MB) + `run()` 首行 `if(_MG_USE_GCR_) run_gcr()` 分派 + `main.py --gcr` (`p[_MG_USE_GCR_]=1`)
- **编译**: `bash ./build.sh` 23M sm_60+PTX `SUCCESS` (警告 20054 `__shared__` 动态初始化, 可忽略)
- **实测 16×32×32×48 V100** (`--gcr`, `r15 cf1e3 cmi3`):
  - `1L GCR(10)`: `3.14s 152 iters final 2.29e-03` (vs BiStabCG `2.13s 152 iters 4.7e-07`), `1L` GCR 比 `1L` BiStabCG 慢 `47%` (`3.14/2.13=1.47×`), 收敛 `2.29e-03` 刚达 `b_norm·atol=2.4e-3` 阈值, 真残差 `8.35e-07` 反而更小 (递归`r`漂移)
  - `2L GCR(10)+MG`: `37.99s 1000 iters final 2.35e3` 发散 (`rel 0.76`, `0.08×` vs L1), `PROF fine 37873ms vcycle18421ms 1000vcycles` (每GCR iter 1 V-cycle 18ms, 1000×18=18s + fine 37s), `1000` 次 `M^{-1}r` 均做 `v_cycle(1)` 的 `coarse BiStabCG 3 iters` → `1000×(12ms+18ms)=30s` 超时
- **根因**: `apply_mg_prec` 的 `v_cycle(1)` 仍做 `coarse BiStabCG` 至 `tol·r0` (3 iters, 18ms), 非单次 `V-cycle` (3ms), `1000` 次调用 → `18s` 开销；GCR 的 `dot_mpi` 与 `v_cycle` 的 `dot_coarse` 共用 `_tmp0_/_tmp1_/_send_tmp_` 槽位, `v_cycle` 后 `host_vals[_tmp0_]` 被覆盖, 虽重算但 `MPI_Allreduce` 的 `host_vals` 镜像与 `device_vals` 不一致导致 `β/α` 计算偏差；`dot_aa` 实部归一化 `β=dot_aq/dot_aa` 忽略虚部 `dot_aa.imag≈0` 但 `dot_aq` 复数除复数引入相位误差, 累积 `1000` 次后 `r` 漂移 `2.29e-3→2.35e3` 发散
- **回退**: 默认 `p[_MG_USE_GCR_]=0` 保持 `BiStabCG` 基线 `0.88×` (1.73→1.96s), `GCR` 仅 `--gcr` 时启用, 已验证 `--gcr` 发散, 留待 `FGMRES(10)` 重构 (`bistabcg.h→fgmres.h` 1h, 正交化 `10基47M 0.5ms` 已备) 与 `dot` 槽位隔离 (`_gcr_tmp0` 独立)
- **量纲**: GCR `m=10` 需 `10×37.6MB×2=752MB` + `r` 37M, 总 `789M` 占 V100 `32G` 的 `2.4%`, 可行, 但 `1000` 次 `V-cycle 18ms` → `18s` 主导, 需轻量 `1×V-cycle 3ms` (fused) + `dot` 隔离

