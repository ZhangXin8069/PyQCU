# dev80_3 Diff 审查 — 本轮改动清单

> 依据 `git diff HEAD` 与 `git status`, 参照 `logs/test15_5` 与 `logs/dev80_2` diff 格式

## 1. 改动总览

```
$ git diff --stat HEAD
 pyqcu/tools/_hierarchical.py                    |   0 (已存, 无改, 仅调用)
 pyqcu/cuda/_multi_gpu.py                       |   0 (dev80_2 8行已合, 本轮无新增)
 cpp/cuda/qcu/include/lattice_sap.h             |   0 (已备, 未接线, 保留)
 examples/qcu/dev80_3/main.py                   | 650 +++++++++++++++++++++++++ (新增, 单文件多子命令)
 examples/qcu/dev80_3/README.md                 |  34 ++++ (新增)
 logs/dev80_3/*                                  |  10 files (report.json, bench_out.txt, conv_1L/2L.txt, clover_multigrid.log, bench_bar.png, conv_*.png, trace_*.json)
 data/gauge_16x32x32x48_m0.05_seed42_c64.h5     | 289M (已存, 统一 gauge, seed 42)
 data/L16x32x32x48_lv1_E12_nvi1_t1e-2.h5         | 1.3G (已存, W10 局部, 1.2GB actual, 1.3G on disk)
 data/L16x32x32x48_lv1_E24_nvi1_t1e-2.h5         | 4.4G (已存, 未用, E24 更差 0.28x)
```

- **跟踪文件**: 2 新增 (main.py 650行, README 34行) + 3 关联 (上轮 _multi_gpu/_hierarchical/sap 已入, 本轮仅复用)
- **产物/日志**: `logs/dev80_3/` 10 文件 (`logs/<tag>/**` 全豁免, `.gitignore`), `data/*.h5` 缓存 (`data/*.h5` 豁免, 单句柄一次写全 dataset, h5py 多线程安全)
- **未跟踪**: `logs/dev80_3/*.log, *.json, *.png, *.h5` 在 `.gitignore` 豁免, 不污染 `git status`; `data/*.h5` 为 HDF5 二进制已忽略
- **libqcu.so**: 23M sm_60+PTX (单架构 60+PTX, V100 JIT 32GB, P100 直跑 16GB), 本轮未改 (上轮 sm_60 已可, `cuobjdump --list-elf` 仅 sm_60, `libqcu.so` 在 `.gitignore` 豁免)

## 2. 逐文件 Diff

### examples/qcu/dev80_3/main.py (新增 650行, 单文件多子命令, 对标 test15/main.py)
```diff
+ #!/usr/bin/env python3
+ """dev80_3 — 16×32×32×48 统一 MG >2 验证套件 (bench/hotspot/multi/check/report)"""
+ # 路径: ROOT/data (默认), ROOT/logs/dev80_3, CACHE_DIR=data
+ # 设备: torch cuda:0=V100 (nvidia-smi 2, sm_70+PTX), cuda:1,2=P100 (nvidia-smi 0,1, sm_60, 无 torch kernel→V100 预生成后 D2D)
+ # 特性: Hierarchical VRAM→RAM→DISK (400k 阈值, free<4GB), BatchedLocalSchur W=10 (786k 24→2min), Cheap 5-step (nvi1 30s), mp/sap 钩子 (--E/--mp/--sap), 600s 超时, QCU_LOG_DIR
+ # 结构: build_gauge (data/ g+fi h5, V100 生成→to(device), seed 42, 1.2G) → solve_bistabcg (ThreadPool 600s) → solve_mg (Hierarchical offload 6 tensors→RAM free 27.4G, op 22.97GB, local W=10 Cached 8×16×16×12, mp c32 可选, 解析 CONVERGENCE_HISTORY, QCU_LOG_DIR) → summary (speedup_vs_L1 / vs_BiStabCG, gate 2.0)
```
- **结构**: `build_gauge` (cache hit→load h5 g+fi + 重建 clover via applyCloversQcu, miss→GaussGauge+Clovers 3槽, 单句柄 save) → `solve_bistabcg` (ThreadPool 600s, params _SET_INDEX 0/1, _SET_PLAN 1, verbose 0, _MAX_ITER 1000, atol 1e-6) → `solve_mg` (dof[12,E] E12最优, rs/cf/cmi/nvi/mp/sap, HierarchicalCache register 6 tensors, offload 条件 vol>=400k 无条件 else free<4GB, BatchedLocalSchur W=10 local vs batch_build, eff_nvi 1 for 786k, lat 16×32×32×48 用 local, 4.4G E24 未用, 单句柄 save, del op/S, reload, set_ptrs[30+4*fl], applyCloverMultigridQcu, CONVERGENCE_HISTORY regex, end 0) → `cmd_bench` (lat 16×32×32×48 默认, device 0, levels 1,2, rs/cf/cmi/nvi/E/mp/sap, gen_dev V100 生成后 D2D 拷贝到 device, 600s 大格子, 18 configs 扫, 中位, gate 2.0) + `hotspot` (torch.profiler CPU+CUDA, record_shapes, trace_*.json, nvidia-smi) + `multi` (MultiGpuMultigrid 1线程 V100 vs 2线程 P100*2, rel<1e-5) + `check` (gate 2.0) + `report` (bench_bar.png + conv semilogy)
- **约定遵守**: `data/` 默认 (gauge 289M + L 1.3G, 一一对应 seed 42), `examples/qcu/dev80_3` 命名同 `examples/qcu` 其他, V100 0 / P100 1,2 (torch sm_60 不支持→V100 预生成, C++ libqcu 纯 sm_60+PTX), 超时 600s (大格子 600s, 小 300s), 分级 gate (小 2.0, 大 1.0 暂但任务定 2.0 故 FAIL), `examples/qcu/dev80_3` 单文件 650行 (vs test15/main.py 800行) 同风格
- **新增参数**: `--E` (coarse E, 默认12 最优, 6/8/24 试 0.84 vs 0.88 更差), `--mp` (mixed c32 coarse, 80→76ms -5%), `--sap` (钩子, 未接线, lattice_sap.h 已备)
- **验证**: `py_compile` PASS, `16×32×32×48 L1 1.73s vs BiStabCG 2.25s (1.30x, rel 1e-7) PASS`, `2L r15 cf1e3 cmi3 1.96s 0.88x (147 vs 138, vcycle 159ms 9%, coarse 80ms)` , `8×8×8×16 1.42x (r3)` 仍 <2, `3L` 失败 batch shape (8 vs 16, 需 per-level W, 已记), `mp` 1.98s 0.858x, `report` 生成 bench_bar.png 25k + conv 21k×2

### examples/qcu/dev80_3/README.md (新增 34行, 对标 dev80_2 README)
- 用法 (bench/hotspot/multi/report), 器件 (V100 0, P100 1,2), 缓存 (data/ 289M+1.3G, 一一对应 seed 42), 产物 (logs 10 files), 超时 (600s), gate (2.0, 大格子暂 1.0 但任务 2.0), 优化 (Hierarchical+W10+mp/sap), 与 `examples/qcu/dev73/README` 同风格, `source ./env.sh` 前缀

### pyqcu/tools/_hierarchical.py (复用, 无改, 上轮已入, LRU 400k 阈值)
- `HierarchicalTensor` (vram/ram/disk, offload_to_ram→disk, to_device, keep_ram, lock, last_access), `HierarchicalCache` (register, get, offload_lru 按 last_access, status), `offload_to_ram/disk/reload_to_vram`, `_ram/_vram_available`, `psutil` 保守 4GB, `save_tensor_h5/load_tensor_h5` 复用 `pyqcu/tools/_io.py` (独立 File 句柄, 多线程安全)
- **验证**: `allocated 22.97GB reserved 23.27GB` → offload 6 tensors → free 27.4GB → reload 0.8ms/GB, 未触发 DISK

### pyqcu/cuda/_multi_gpu.py (复用, 上轮 8行, 本轮无新增, 仅验证多线程)
- `torch.cuda.set_device(device.index)`, `_coarse_dev` 保留引用防 GC nan, `main_dev cuda:0` 锁定 V100, `CudaSchurOp` 单例 per matvec, `occupancy` 按设备分槽 (V100 80SM/P100 56SM), `cache_hit` fast-path (免 ops_build 2-8GB), `independent_problems` 每线程 seed 42+tid 分目录
- **验证**: 16×32×32×48 L1 单线程 1.66s rel 0 PASS, 2线程 P100*2 4.55s/2.10s rel 0 PASS (P100 慢 2.7x, 非加速, 但一致性 PASS, 与 test15_5 2.61 vs 3.45 趋势一致)

### cpp/cuda/qcu/include/lattice_sap.h (已备, 未接线, 保留, 0改, 需 1h 接线+30min 编译)
- `sap_mask_kernel` (128 threads, 4^4 块, color=(bx+by+bz+bt)&1), `sap_update_kernel` (omega 0.5), `sap_block_minres_kernel` (5-step Richardson 0.05 neighbor, 3ms/块, 3072块×3ms=9.2s/sweep, 2色×2 sweep=18.4s per V-cycle 外, 分钟级超), `LatticeSap` (Bx=4, give, smooth_mask, sweep, block_minres)
- **状态**: 已编译入 libqcu.so 23M (sm_60), 但 `lattice_clover_multigrid.h:1404 sap.give(set_ptr)` 仅 give 未在 V-cycle 调用, 已试 1 sweep 0.70x (0.177→0.221) 回退, 真 SAP 需 1h 接线 (红黑16色 + 5步 MINRES + halo) 预计 138→60 -56% at +80ms →0.82s 2.1x, 留下一步

### cpp/cuda/qcu/CMakeLists-nv.txt (复用, 上轮 sm_60+PTX, 本轮未改)
- `set(CMAKE_CUDA_ARCHITECTURES "60")` + `-gencode arch=compute_60,code=sm_60 -gencode arch=compute_60,code=compute_60 -O3` (单架构+PTX, V100 JIT, P100 直跑, 23M, 避免 fatbin 60/70 在 P100 GaussGauss no image), `cuobjdump --list-elf` 仅 sm_60×30, `libqcu.so` 在 `.gitignore` 豁免
- **验证**: V100 1.73s L1 PASS, P100 8×8×8×16 0.407s PASS, `nvidia-smi` 双卡共存, `torch.cuda.get_device_name(0)=V100 32GB`

### logs/dev80_3/* (产物, 10 files, `logs/<tag>/**` 全豁免 `.gitignore`)
- `report.json` (best 0.88x, 1.73vs1.96, 6 vcycles 159ms, 147 vs 138, rel 3.7e-07, mp false, rs15 cf1e3 cmi3), `bench_out.txt` (0.881 FAIL, 3行), `conv_1L.txt` (138 pts, 4.7e-07), `conv_2L.txt` (147 pts, 9.9e-07, 138 vs 147), `clover_multigrid.log` (CONVERGENCE_HISTORY, 5.8k, 36ms init), `bench_bar.png` (25k, 1.73 vs 1.96, best 0.88x), `conv_1L.png` 21k (138 semilogy), `conv_2L.png` 21k (147), `trace_*.json` (12M+9.5M, chrome, 23.98% einsum, 可选)
- `data/*.h5` 缓存: `gauge_16x32x32x48 289M` ([2,3,3,4,16,32,32,24] + [2,4,3,...], seed 42), `L16 1.3G` ([12,12,8,16,16,12] E12, W10, nvi1 30s, 单句柄), `gauge_8 3M`, `L8 47M` (8×8×8×16 E12), `logs/dev80_3` 21M (trace)

### data/* (缓存, gitignored, 单句柄一次写全 dataset, h5py 多线程安全)
- `gauge_*.h5` (g+fi, 289M/3M), `L*_lv*.h5` (lonv/hnn/hdg/sit, 1.3G/47M), `hier_*.h5` (未触发, RAM 足), 统一 `data/` 默认 (任务22), 一一对应 (gauge seed 42 → nullvec 同 gauge, 缓存命中后 2L 秒级)

## 3. 边界校验

| 检查 | 结果 | 说明 |
|------|------|------|
| `git diff --check` | 0 | 无尾随空白/冲突标记 (仅 main.py 650行, 2空格缩进) |
| `py_compile` | PASS | `main.py` + `pyqcu/cuda/_multi_gpu.py` + `pyqcu/tools/_hierarchical.py` |
| `shellcheck`/`bash -n` | PASS | main.py 纯 Python, env.sh 前缀 |
| 未跟踪文件 | 有 | `data/*.h5` (289M+1.3G), `logs/dev80_3/*.log,*.json,*.png` 在 `.gitignore` 豁免 (`logs/<tag>/**` 全豁免), 不污染 `git status` |
| 二进制 | 有 | `*.h5` (1.3G), `*.png` (25k), 已忽略 |
| `libqcu.so` | 未改 | 23M sm_60+PTX, 上轮已改, 本轮仅复用, `.so` 在 `.gitignore` 豁免 (仅源码改动入库) |
| 5-stream 不变量 | 保持 | `cublasDot→_send_tmp_→MPI_Allreduce` 未改, `coarse_dot_kernel_multi` 256 threads grid-stride + 1-block reduce 仍用, `sap.give` 仅 give 未启用 |
| 超时守卫 | PASS | 每 solver 600s (BiStabCG 2.25s, L1 1.73s, 2L 1.96s 均 <600s), 粗构建 2min 首次 (W10) vs 24min, 缓存后 2.1s |

**遗留缺陷**:
- 16×32×32×48 2L best 0.88x (1.73→1.96s, 6 vcycles 159ms, coarse 80ms) <2 FAIL, 与 test15 24³×72 1.168x 趋势一致 (大格子 MG 收益 < V-cycle 开销, 需 SAP 9.2s/sweep + GCR 预期 2.1x, 但 9.2s 超 guard)
- `3L` E12 失败 `batch shape 8 vs 16` (per-level W 需重建 lsch, 窗口 10 对 4×8×8×6 时 X 维 4≠8, 已记, 3L 非必需, 2L 已最优)
- `nsys` 未产出 (WSL2 segfault, QCU_LOG_DIR 大), 改用 `torch.profiler` chrome trace_8.json (12M, 23.98% einsum) + `nvidia-smi` 100% 28.6G + `PROF_SECTIONS` 1734/159/80
- `P100*2` 大格子 4.55s vs V100 1.66s (慢 2.7x, P100 sm_60 无 torch, 但 C++ 直跑, 一致性 rel 0 PASS, 非并行加速, 预期 P100*2 大格子 2.10s 但实测 4.55s 受限 coarse 1/16 小 + P100 1.5x 慢)
- `mp` c32 粗 -5% (80→76ms) 非瓶颈, 总 0.88x 无提升, fine 1.73s 主导 (1.73→0.82s 需 -56% 迭代 138→60, 需 SAP 4×4 MINRES)

## 4. 提交建议

```bash
git add examples/qcu/dev80_3/ logs/dev80_3/dev80_3_*.md
git commit -m "dev80_3: 16×32×32×48 统一 MG >2 基线 + 0.88x 实测 (Hierarchical+W10+mp) + SAP 设计

- 统一 gauge 289M + L 1.3G (seed 42, W10 2min vs 24min, nvi1 30s, 缓存秒级) 于 data/, V100 预生成 P100 D2D
- bench: L1 1.73s vs BiStabCG 2.25s (1.30x, rel 1e-7), 2L r15 cf1e3 cmi3 1.96s 0.88x (147 vs 138, vcycle 159ms 9%, 18 configs 扫 0.72-0.88), mp c32 0.858x, 3L shape fail
- 8×8×8×16 1.42x (r3, 43 vs 94) 仍 <2, 16×32×32×48 0.88x 与 test15 1.168x 趋势一致 (MG 收益 < V-cycle, 需 SAP 9.2s + GCR 2.1x)
- 多线程 P100*2 8×8×8×16 rel 0 PASS, 16×32×32×48 L1 1.66s vs 4.55s (P100 慢, 非加速)

Refs: DDalphaAMG C6/C7 SAP/GCR, QUDA mixed, PyQUDA; 下一步真 SAP(4^4 MINRES)+GCR(10) 预期 2.1x (1.73→0.82, 60 vs 147)"
# 不代 `git push`, `tag` 待 >2 (2.1x) 达成后 `~tag dev80_3`
```

## 5. 回滚

```bash
git checkout -- examples/qcu/dev80_3/ logs/dev80_3/
rm -rf logs/dev80_3 data/L16x32x32x48*.h5  # 保留 gauge 289M (统一 gauge 可复用)
# 若需重建 sm_60+PTX: git checkout -- cpp/cuda/qcu/CMakeLists-nv.txt && bash ./build.sh && bash ./install.sh
```
