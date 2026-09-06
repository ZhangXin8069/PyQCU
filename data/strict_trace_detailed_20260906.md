# 20260906 Strict MultiGrid 逐迭代对照摘要

本文件由 `analyze_strict_vs_quda_detailed.py` 生成。性能时间来自无 trace 正式 benchmark；
逐迭代残差来自开启日志的独立 trace。两者通过 `config_hash` 与 `bundle_hash` 校验。

- 格点：`[16, 32, 32, 48]`；mass=`0.05`；nvec=`12`；coarse_spin=`2`；target_parity=`1`。
- comparison：`{'status': 'pass', 'profile': 'formal', 'fair': True, 'reasons': [], 'speedup_pyqcu_over_quda': 1.026051084423885, 'pyqcu_median_seconds': 2.041653319989564, 'quda_median_seconds': 2.0948406029929174}`。

| 侧 | steady 迭代数 | median solve (s) | MAD (s) | median ms/外层迭代 | full-op 真残差 |
|---|---:|---:|---:|---:|---:|
| PyQCU Strict | [11, 11, 11, 11, 11] | 2.041653 | 0.026462 | 185.605 | `3.6013e-07, 3.6013e-07, 3.6013e-07, 3.6013e-07, 3.6013e-07` |
| QUDA | [37, 37, 37, 37, 37] | 2.094841 | 0.012283 | 56.617 | `7.3030e-07, 7.3030e-07, 7.3030e-07, 7.3030e-07, 7.3030e-07` |

逐外层迭代数据见同目录 CSV；图表见同目录 SVG。PyQCU 曲线是 Arnoldi estimate，
QUDA 曲线是 GCR iterated residual，不能把两条曲线的每个浮点值当成同一递推量；
最终收敛判据由 full Wilson/Clover operator 的 true residual 给出。
