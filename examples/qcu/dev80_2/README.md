# dev80_2 — 16×32×32×48 统一格子 MG >2 真实加速比验证套件

- 格子固定 16×32×32×48（786,432 站点，odd 子格 393,216），mass 0.05，atol 1e-6，c64
- 基准：MG L1（仅最细层 Schur BiStabCG，无粗校正）→ 真实加速比 `L1 / MG_2L(3L)` 目标 >2.0 稳定
- 对照：C++ Clover BiStabCG 正确性（rel <1e-5），单线程 MG 并行效果对照（P100*2 vs V100）
- 统一 gauge/nullvec 缓存于 `data/`（gauge_16x32x32x48...h5 + L16x32x32x48_lv*.h5，一一对应）
- 器件：单卡 V100-32GB (torch cuda:0, 物理 nvidia-smi 2)，双卡 P100-16GB*2 (torch 1,2，物理 0,1；torch sm_60 不支持→V100 生成后拷贝，C++ libqcu.so 纯 sm_60+PTX 两卡通用)
- 产物：`logs/dev80_2/*.json, *.txt, *.log, *.png` + `data/*.h5` 缓存

运行：
  source ./env.sh
  # V100 单卡基准（L1 + 2L/3L）
  python examples/qcu/dev80_2/bench_dev80_2.py --lat 16,32,32,48 --device 0 --levels 1,2 --verbose
  python examples/qcu/dev80_2/bench_dev80_2.py --lat 16,32,32,48 --device 0 --levels 1,2,3 --rs 5 --cf 1e5 --cmi 15 --nvi 2
  # 参数扫描（r/ct/cmi/E）
  for rs in 3 5 15 30; do for cf in 1e3 1e5; do python examples/qcu/dev80_2/bench_dev80_2.py --lat 16,32,32,48 --device 0 --levels 1,2 --rs $rs --cf $cf --cmi 15; done; done
  # P100 双卡（待 sm_60 内核全覆盖后，V100 生成的 gauge/算子直接在 P100 上求解）
  python examples/qcu/dev80_2/bench_multi_gpu.py --lat 16,32,32,48 --nthreads 2 --devices 1,2

超时守卫：每 solver 600s（粗构建分钟级）+ 分层显存 offload（VRAM→RAM→DISK）
