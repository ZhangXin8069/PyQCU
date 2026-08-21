# dev80_3 — 16×32×32×48 统一格子 MG >2 真实加速比验证套件

- 格子固定 16×32×32×48（786,432 站点，odd 子格 393,216），mass 0.05，atol 1e-6，c64
- 基准：MG L1（仅最细层 Schur BiStabCG，无粗校正）→ 真实加速比 `L1 / MG_2L(3L)` 目标 >2.0 稳定 (3次中位)
- 对照：C++ Clover BiStabCG 正确性（rel <1e-5 且残差<atol），单线程 MG vs P100*2 多线程并行效果
- 统一 gauge / nullvec 缓存于 `data/`（gauge_16x32x32x48_m0.05_seed42_c64.h5 + L16x32x32x48_lv*_E*_nvi*.h5，一一对应，按 gauge seed 关联）
- 器件：单卡 V100-32GB (torch cuda:0, 物理 nvidia-smi 2, sm_70)，双卡 P100-16GB*2 (torch 1,2，物理 0,1, sm_60；torch 无 sm_60 kernel→V100 生成后 D2D 拷贝，C++ libqcu.so 纯 sm_60 PTX 通用)
- 优化：Hierarchical VRAM→RAM→DISK + BatchedLocalSchur W=10 + Cheap-Jacobi 5-step + 混合精度 c32 粗层 + SAP 4^4 块 MINRES + GCR 外层 (参考 DDalphaAMG/QUDA)
- 产物：`logs/dev80_3/report.json, bench_out.txt, conv_*.txt, *.png, *.tex` + `data/*.h5` 缓存；`examples/qcu/dev80_3/main.py` 单文件多子命令入口

运行（V100 单测，P100 双测需 libqcu.so sm60 PTX）：
  source ./env.sh
  # V100 单卡基准（L1 + 2L/3L，600s 超时，Hierarchical 分层）
  python examples/qcu/dev80_3/main.py bench --lat 16,32,32,48 --device 0 --levels 1,2 --verbose
  python examples/qcu/dev80_3/main.py bench --lat 16,32,32,48 --device 0 --levels 1,2,3 --rs 5 --cf 1e5 --cmi 15 --nvi 2
  # 参数扫描（rs/ct/cmi/E 最优: r3 cf1e3 cmi15 E12 预期）
  for rs in 3 5 15 30; do for cf in 1e3 1e5; do python examples/qcu/dev80_3/main.py bench --lat 16,32,32,48 --device 0 --levels 1,2 --rs $rs --cf $cf --cmi 15; done; done
  # 热点剖析
  python examples/qcu/dev80_3/main.py hotspot --lat 16,32,32,48
  # P100 双卡并行对照
  python examples/qcu/dev80_3/main.py multi --lat 16,32,32,48 --levels 2
  # 报告
  python examples/qcu/dev80_3/main.py report --lat 16,32,32,48

超时守卫：每 solver 600s（大格子粗构建分钟级，Hierarchical offload 需 4min）+ 分级 gate (小格子2.0, 16x32x32x48 暂1.0, 目标>2.0)
