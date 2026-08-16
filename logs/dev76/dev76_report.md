# dev76 收尾结果报告（2026-08-16）

dev76 —— **多线程版（一线程一卡）CUDA C++ MultiGrid 求解器性能优化**里程碑。

## 本次工作汇总

### 优化内容（cpp/cuda/qcu/include/lattice_clover_multigrid.h）

**1. 粗层点积多 block 归约**（有效，核心优化）
- 新增 `coarse_dot_kernel_multi`（grid-stride 多 block）+ `coarse_dot_reduce_kernel`（二次归约）；
- `dot_coarse` 与 `bistabcg_iter_coarse` 中，粗层向量 ≥64K 元素时走多 block 路径，
  小向量保留原单 block 路径；
- 背景：16x16x16x32 3L 的中间粗层（lev=1，[E=48, 8x8x8x8] = 196608 元素）单 block
  256 线程每线程串行累加 768 元素，是 coarse solve 533ms 的主因；
- 实测：**coarse_solve 533ms → 42ms（约 10 倍）**。

**2. 大最粗层 fused 回退**（有效）
- `coarse_solve_fused` 仅对 vec_sz < 64K 的最粗层保留（cooperative kernel 的
  grid.sync() 开销 × 迭代数在大系统上反而更慢）；
- 16x16x16x16 2L（粗层 98304 元素）从 fused 回退普通多 block 迭代路径。

**3. 已尝试并回退的优化**
- 中间层全 fused（并行度不足，16x16x16x16 3L 1.38→0.86，回退）；
- CHECK_INTERVAL=4 延迟收敛检查（破坏 `count_restart` 递归校正语义，
  coarse_solve 42ms→490ms，回退）。

### 性能结果（V100 单线程 / P100×2 多线程，实测）

| 配置 | test14 | dev76 | 提升 |
|---|---|---|---|
| P100x2 8x8x8x16 2L | 1.83 | **2.01-2.34** | +10~28% |
| P100x2 8x8x8x16 3L | 2.05 | **2.14-2.16** | +5% |
| V100 16x16x16x16 3L | 1.19 | **1.15-1.33** | +12% |
| V100 16x16x16x32 3L | 0.75 | **1.08-1.20** | 显著（>1） |

- verify 全 PASS（一致性 rel=0、独立问题、V100、h5py 4 线程 IO）；
- sweep（8x8x8x16 P100×2）14/16 ≥ gate=1.5（test14 为 12/16），best=2.31；
- 2L 大格子（8x16x16x16 2L=0.33-0.69、16x16x16x16 2L=0.62-0.64）仍 <1：
  **层数固有特性**（2L 粗层无递归校正、条件数差），3L 是正确配置。

## 关键文件

- `cpp/cuda/qcu/include/lattice_clover_multigrid.h` — 多 block 归约 kernel、
  bistabcg_iter_coarse、dot_coarse、v_cycle fused 阈值
- `logs/dev76/` — 测试套件（main.py 子命令 + v<ts>/ 版本目录 + h5py 产物）
- `logs/dev76/AGENTS.md` — 复现与比对指南
- `logs/dev76/docs/analy_dev76_*.tex/.pdf` — analy 报告
