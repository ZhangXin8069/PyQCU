# Strict MultiGrid 逐迭代对照摘要

本文件由 `analyze_strict_vs_quda_detailed.py` 生成。性能时间来自无 trace 正式 benchmark；
逐迭代残差来自开启日志的独立 trace。两者通过 `config_hash` 与 `bundle_hash` 校验。

- 格点：`[16, 32, 32, 48]`；mass=`0.05`；nvec=`12`；coarse_spin=`2`；target_parity=`1`。
- comparison：`{'status': 'pass', 'profile': 'formal', 'fair': True, 'reasons': [], 'speedup_pyqcu_over_quda': 1.0650651550816639, 'pyqcu_median_seconds': 1.9667946510016918, 'quda_median_seconds': 2.094764449982904}`。

| 侧 | steady 迭代数 | median solve (s) | MAD (s) | median ms/外层迭代 | full-op 真残差 |
|---|---:|---:|---:|---:|---:|
| PyQCU Strict | [11, 11, 11, 11, 11] | 1.966795 | 0.000993 | 178.800 | `3.6013e-07, 3.6013e-07, 3.6013e-07, 3.6013e-07, 3.6013e-07` |
| QUDA | [37, 37, 37, 37, 37] | 2.094764 | 0.011047 | 56.615 | `7.3030e-07, 7.3030e-07, 7.3030e-07, 7.3030e-07, 7.3030e-07` |

逐外层迭代数据、阶段事件和 aggregate profile 分别见同目录 CSV；图表见同目录 SVG。
PyQCU 曲线是 Arnoldi estimate，
QUDA 曲线是 GCR iterated residual，不能把两条曲线的每个浮点值当成同一递推量；
最终收敛判据由 full Wilson/Clover operator 的 true residual 给出。

## PyQCU 阶段诊断（仅用于归因）

阶段计时通过每个区间前后同步同一 CUDA stream 获得，包含同步开销；外层总区间包住内部区间，因此百分比不能求和。QUDA 本次接口没有提供逐 GCR 迭代或逐 V-cycle 层的回调，不能从聚合 profile 反推出同口径阶段时间。

| level | 阶段 | count | median ms | MAD ms | median/outer |
|---:|---|---:|---:|---:|---:|
| 0 | `arnoldi` | 55 | 1.375 | 0.101 | 0.78% |
| 0 | `fgmres_solution_update` | 15 | 0.629 | 0.004 | 0.35% |
| 0 | `fine_correction_residual` | 55 | 2.643 | 0.011 | 1.48% |
| 0 | `fine_matpc` | 55 | 2.480 | 0.008 | 1.39% |
| 0 | `fine_post_smoother` | 55 | 2.829 | 0.008 | 1.59% |
| 0 | `fine_pre_smoother` | 55 | 2.974 | 0.009 | 1.67% |
| 0 | `fine_prolongation` | 55 | 2.750 | 0.008 | 1.55% |
| 0 | `fine_restriction` | 55 | 3.307 | 0.006 | 1.86% |
| 0 | `outer_iteration` | 55 | 178.437 | 5.397 | 100.00% |
| 0 | `true_residual_refresh` | 15 | 3.175 | 0.036 | 1.76% |
| 1 | `coarse_vcycle` | 55 | 158.184 | 5.638 | 88.78% |
| 1 | `coarsest_bicgstab` | 55 | 151.301 | 5.779 | 85.07% |
| 1 | `level_prepare` | 55 | 3.268 | 0.015 | 1.83% |
| 1 | `level_reconstruct` | 55 | 2.958 | 0.026 | 1.65% |

QUDA 阶段时间状态：`unavailable via current public invertQuda/TimeProfile interface`; QUDA 的总 solve/steady wall time仍用于正式对照。
