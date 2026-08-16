# dev78 收尾结果报告（2026-08-16）

dev78 —— **多线程版（一线程一卡）CUDA C++ MultiGrid 求解器中/大格子参数优化**里程碑。

## 本次工作汇总

### 优化内容

**2L/3L 中格子 num_restart 参数优化**（logs/dev78/main.py bench 配置）：
- 8x16x16x16 2L：num_restart 5 → **10**（V-cycle 频率减半、粗层求解次数减半）
- 8x16x16x16 3L：num_restart 5 → **10**
- 16x16x16x16 2L：num_restart 10 → **20**

根因：中/大格子 2L/3L 的粗层求解（fused 33-58ms/次 或 普通多 block ~117ms/次）
是 V-cycle 校正的主导成本；num_restart=5 时 V-cycle 过于频繁，粗层求解次数
翻倍且校正质量未提升（n_vcycles 13-44 次爆炸）。

### 已尝试并回退的优化
- fused 阈值 64K → 32K：8x16x16x16 2L 从 0.83 降到 0.46（普通多 block 路径
  在 49152 元素上比 fused 慢），回退 64K。

### 性能结果（实测）

| 配置 | dev76 | dev78 | 提升 |
|---|---|---|---|
| P100x2 8x16x16x16 2L | 0.33-0.69 | **1.05-1.22** | 约 3 倍 |
| P100x2 8x16x16x16 3L | 0.45-0.72 | **1.41** | 约 2 倍 |
| P100x2 8x8x8x16 3L | 2.14-2.16 | **2.29** | +6% |
| V100 16x16x16x16 3L | 1.12-1.33 | **1.18-1.33** | 持平 |
| P100x2 16x16x16x16 2L | 0.62-0.66 | **0.74**（r20） | +12% |

- verify 全 PASS（一致性 rel=0、独立问题、V100、h5py 4 线程 IO）；
- bench median 1.148 → **1.210**，max 2.135 → **2.288**；
- sweep（8x8x8x16 P100×2）13/16 ≥ gate=1.5（best=2.499）；
- 16x16x16x16 2L 仍 <1（0.74）：**层数固有特性**（2L 粗层条件数差），
  r20 已是最优参数。

## 关键文件

- `logs/dev78/main.py` — bench 配置 r10/r20 参数优化（_bench_configs）
- `logs/dev78/AGENTS.md` — 复现与比对指南
- `logs/dev78/docs/analy_dev78_*.tex/.pdf` — analy 报告
- `cpp/cuda/qcu/include/lattice_clover_multigrid.h` — 无改动（纯参数优化）
