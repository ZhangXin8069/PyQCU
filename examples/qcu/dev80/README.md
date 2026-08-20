# dev80 — 32^4 统一格子 MG >2 真实加速比验证套件

- 格子固定 32^4，mass 0.05，atol 1e-6，单卡 V100-32GB / 双卡 P100-16GB*2
- 基准：MG L1（仅最细层 Schur BiStabCG）vs 多层 MG（2L/3L）真实加速比 >2
- 对照：C++ BiStabCG 正确性（rel <1e-5），单线程 MG 并行效果对照
- 统一 gauge/nullvec 缓存于 /root/PyQCU/data（gauge_32...h5 + nullvec_<tag>.h5）
- 产物：logs/dev80/*.log + h5 + report.json + plots

运行：
  source ./env.sh
  # V100 单卡基准
  python examples/qcu/dev80/bench_dev80.py --lat 32,32,32,32 --device 0 --levels 1,2,3
  # P100 双卡并行（待 GaussGauge sm_60 内核修复后启用）
  # python examples/qcu/dev80/bench_dev80.py --lat 32,32,32,32 --device 1,2 --nthreads 2
