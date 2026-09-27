# 20260906 Strict MultiGrid 逐迭代对照摘要

本文件由 `analyze_strict_vs_quda_detailed.py` 生成。性能时间来自无 trace 正式 benchmark；
逐迭代残差来自开启日志的独立 trace。两者通过 `config_hash` 与 `bundle_hash` 校验。

- 格点：`[16, 32, 32, 48]`；mass=`0.05`；nvec=`12`；coarse_spin=`2`；target_parity=`1`。
- comparison：`{'status': 'pass', 'profile': 'formal', 'fair': True, 'reasons': [], 'speedup_pyqcu_over_quda': 1.026051084423885, 'pyqcu_median_seconds': 2.041653319989564, 'quda_median_seconds': 2.0948406029929174}`。

| 侧 | steady 迭代数 | median solve (s) | MAD (s) | median ms/外层迭代 | full-op 真残差 |
|---|---:|---:|---:|---:|---:|
| PyQCU Strict | [11, 11, 11, 11, 11] | 2.041653 | 0.026462 | 185.605 | `3.6013e-07, 3.6013e-07, 3.6013e-07, 3.6013e-07, 3.6013e-07` |
| QUDA | [37, 37, 37, 37, 37] | 2.094841 | 0.012283 | 56.617 | `7.3030e-07, 7.3030e-07, 7.3030e-07, 7.3030e-07, 7.3030e-07` |

逐外层迭代数据、阶段事件和 aggregate profile 分别见同目录 CSV；图表见同目录 SVG。
PyQCU 曲线是 Arnoldi estimate，
QUDA 曲线是 GCR iterated residual，不能把两条曲线的每个浮点值当成同一递推量；
最终收敛判据由 full Wilson/Clover operator 的 true residual 给出。

## PyQCU 阶段诊断（仅用于归因）

阶段计时通过每个区间前后同步同一 CUDA stream 获得，包含同步开销；外层总区间包住内部区间，因此百分比不能求和。QUDA 本次接口没有提供逐 GCR 迭代或逐 V-cycle 层的回调，不能从聚合 profile 反推出同口径阶段时间。

| level | 阶段 | count | median ms | MAD ms | median/outer |
|---:|---|---:|---:|---:|---:|
| 0 | `arnoldi` | 55 | 1.319 | 0.105 | 0.72% |
| 0 | `fgmres_solution_update` | 15 | 0.630 | 0.025 | 0.35% |
| 0 | `fine_correction_residual` | 55 | 2.644 | 0.011 | 1.44% |
| 0 | `fine_matpc` | 55 | 2.476 | 0.005 | 1.36% |
| 0 | `fine_post_smoother` | 55 | 3.104 | 0.027 | 1.69% |
| 0 | `fine_pre_smoother` | 55 | 3.257 | 0.034 | 1.80% |
| 0 | `fine_prolongation` | 55 | 2.726 | 0.007 | 1.48% |
| 0 | `fine_restriction` | 55 | 3.307 | 0.005 | 1.80% |
| 0 | `outer_iteration` | 55 | 185.852 | 5.876 | 100.00% |
| 0 | `true_residual_refresh` | 15 | 3.090 | 0.051 | 1.70% |
| 1 | `coarse_vcycle` | 55 | 164.769 | 6.526 | 88.89% |
| 1 | `coarsest_bicgstab` | 55 | 157.980 | 6.507 | 85.10% |
| 1 | `level_prepare` | 55 | 3.247 | 0.014 | 1.77% |
| 1 | `level_reconstruct` | 55 | 2.923 | 0.013 | 1.60% |

QUDA 阶段时间状态：`unavailable via current public invertQuda/TimeProfile interface`; QUDA 的总 solve/steady wall time仍用于正式对照。
