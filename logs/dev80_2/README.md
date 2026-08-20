# logs/dev80_2 — 16×32×32×48 统一格子 MG >2 真实加速比任务报告

- 任务：令 CUDA_C++ 多线程版 MultiGrid 具有稳定 >2 的真实加速比（vs L1）
- 格子：16×32×32×48 统一，mass 0.05, atol 1e-6, c64
- 器件：单卡 V100-32GB (torch cuda:0, sm_70), 双卡 P100-16GB*2 (torch 1,2 sm_60, libqcu.so sm_60+PTX)
- 基准：MG L1 (Schur BiStabCG 单层) vs MG 2L/3L (V-cycle)，正确性 vs C++ BiStabCG (rel<1e-5)
- 产出：bench_dev80_2.py (主基准) + bench_multi_gpu.py (多卡) 在 examples/qcu/dev80_2

见 dev80_2_analy.md (分析) 与 dev80_2_diff.md (改动)
