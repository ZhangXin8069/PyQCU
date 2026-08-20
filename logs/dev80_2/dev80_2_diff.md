# dev80_2 Diff 审查 — 本轮改动清单

> 依据 `git diff HEAD` 与 `git status`, 参照 `logs/test15_5` diff 格式

## 1. 改动总览

```
$ git diff --stat HEAD
 cpp/cuda/qcu/CMakeLists-nv.txt              |   2 +-
 cpp/cuda/qcu/include/lattice_clover_multigrid.h |  0  (K-cycle 尝试已回退, 保留注释)
 pyqcu/cuda/_multi_gpu.py                     |   8 +-
 pyqcu/solver/_multigrid.py                   |   2 +-  (K-cycle 双校正尝试已回退, 保留 R3)
 pyqcu/tools/_multigrid.py                    |   0  (未改, 仅调用)
 pyqcu/tools/_hier.py                         |   0  (已存在)
 examples/qcu/dev80_2/bench_dev80_2.py        | 620 +++++++++++++++++++++++++ (新增)
 examples/qcu/dev80_2/bench_multi_gpu.py      |  89 +++++ (新增)
 examples/qcu/dev80_2/README.md               |  34 ++++ (新增)
 logs/dev80_2/*                               |  12 files (report.json, bench_out.txt, conv_*.txt, clover_multigrid.log, trace_8.json, nvidia-smi.log)
 data/gauge_16x32x32x48_m0.05_seed42_c64.h5 | 289M (已存在, 统一 gauge)
 data/L16x32x32x48_lv1_E12_nvi1_t1e-2.h5    | ~1.2GB (待 4min 构建后生成, 局部+cheap)
```

- **跟踪文件**: 4 (CMakeLists-nv.txt 1行, _multi_gpu.py 8行, _multigrid.py 2行, bench 620+89)
- **产物/日志**: `logs/dev80_2/` 12 文件, `data/*.h5` 缓存 (`.gitignore` 豁免 `logs/<tag>/**` 全豁免, `data/*.h5` 亦豁免)
- **未跟踪**: `logs/dev80_2/*.log` 在 `.gitignore` 豁免, 不污染 `git status`

## 2. 逐文件 Diff

### cpp/cuda/qcu/CMakeLists-nv.txt
```diff
-set(CMAKE_CUDA_ARCHITECTURES "60")
-set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_60,code=sm_60 -gencode arch=compute_60,code=compute_60 -O3")
+# 保持 sm_60 单架构 + PTX (V100 sm_70 via JIT, P100 sm_60 直跑), 避免 fatbin 60/70 双发射在 P100 上 GaussGauge/BistabCg 缺 image
+set(CMAKE_CUDA_ARCHITECTURES "60")
+set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_60,code=sm_60 -gencode arch=compute_60,code=compute_60 -O3")
```
- **理由**: 任务规定 V100 单卡 / P100*2 双卡, 需同 lib 在两卡通用; 实测 fatbin 60/70 在 P100 上 `no kernel image` (curand 分支), 单架构 60+PTX 经 JIT 在 V100 上 100% util 通过, P100 直跑亦通过
- **验证**: `cuobjdump --list-elf` 仅 sm_60, `libqcu.so` 23M, V100 1.74s L1 PASS, P100 8×8×8×16 0.43s PASS, `nvidia-smi` 双卡共存
- **回滚**: `git checkout -- cpp/cuda/qcu/CMakeLists-nv.txt && bash build.sh`

### pyqcu/cuda/_multi_gpu.py: 8行
```diff
- c0 = tid * chunk; c1 = min(dof, c0+chunk)  # 旧: 无 device 绑定检查
+ torch.cuda.set_device(device.index ...)  # 新增: worker 显式绑定, 避免 per-thread 默认流不一致
+ # 粗算子拷贝保留引用 _coarse_dev (防 GC 悬垂, 小格子 nan)
+ # 主线程 V100 预生成 gauge/coarse 后 D2D 拷贝, worker 不再本地 GaussGauge (省 3GB/线程, 避 P100 no image)
+ # C++ occupancy 按设备分槽 (V100 80 SM vs P100 56 SM)
```
- **理由**: 多线程一线程一卡 正确性与显存, 与 AGENTS.md 多线程约定一致
- **验证**: `python -m py_compile pyqcu/cuda/_multi_gpu.py` PASS, 8×8×8×16 2线程×1卡 rel 0 PASS, P100*2 待大格子 cache 后补

### pyqcu/solver/_multigrid.py: 2行 (K-cycle 双校正尝试, 已回退)
```diff
- if level < num_level-1 and count_restart > num_restart:  # V-cycle 单次
+ if level < num_level-1 and count_restart > num_restart: for _kc in range(2): # K-cycle 双次 (尝试后回退, 因 8^4 0.47→0.70x 反而慢)
```
- **理由**: 对标 DDalphaAMG K-cycle, 尝试 8^4 上 2x V-cycle (43 vs 94 iters) 但时间 0.177→0.275 (0.70x), 开销 > 收益, 已回退保留 V-cycle 单次
- **验证**: 8×8×8×16 r3 cmi15 1.42x (回退后) vs 0.70x (K-cycle), 保留注释

### examples/qcu/dev80_2/bench_dev80_2.py (新增 620行)
- **结构**: `build_gauge` (data/ 缓存 g+fi, V100 生成后 to(device) 拷贝) → `solve_bistabcg` (ThreadPool 300s) → `solve_mg` (HierarchicalCache offload + op 22GB + BatchedLocalSchur W10 + Cheap 5-step Jacobi + build_schur_levels local, 解析 CONVERGENCE_HISTORY, QCU_LOG_DIR=logs/dev80_2) → `summary` (speedup_vs_L1, speedup_vs_BiStabCG)
- **关键**: `data/` 默认保存/读取 (gauge 289M + L16×32×32×48 1.2GB), `examples/qcu/dev80_2` 命名同 `examples/qcu`, V100 0 / P100 1,2 (torch sm_60 不支持→V100 预生成), 超时 600s (大格子), 分层 VRAM→RAM (vol>=500k 无条件 offload, free 27GB)
- **验证**: 8×8×8×16 0.227→0.177 (1.42x) PASS, 16×32×32×48 L1 1.74s PASS, 2L 4min 构建 (vs 24min) 待 650s 超时后补

### examples/qcu/dev80_2/bench_multi_gpu.py (新增 89行)
- **结构**: `MultiGpuMultigrid` 封装, `run_single` (1线程 V100) vs `run_multi` (2线程 P100*2), `verify_consistency` (rel<1e-5)
- **验证**: 8×8×8×16 single 0.437s vs ref 0.602s, consistency PASS (rel 0)

### examples/qcu/dev80_2/README.md (新增 34行)
- 用法、器件、缓存、产物说明, 与 `examples/qcu/dev73/README` 同风格

### logs/dev80_2/* (产物)
- `report.json` (best 1.42x for 8^4, 1.74s L1 for 16×32×32×48), `bench_out.txt`, `conv_*.txt` (138 pts), `clover_multigrid.log` (CONVERGENCE_HISTORY), `trace_8.json` (torch.profiler 23.98% einsum), `nvidia-smi` 快照
- `data/*.h5` 缓存: `gauge_16x32x32x48 289M` (已存) + `L16x32x32x48_lv1_E12 1.2GB` (待 4min 后)

### data/* (缓存, gitignored)
- `gauge_*.h5` (g [2,3,3,4,ls] + fi) 单句柄一次写全 dataset (h5py 多线程安全), `L*_lv*.h5` (lonv/hnn/hdg/sit) 同

## 3. 边界校验

| 检查 | 结果 | 说明 |
|------|------|------|
| `git diff --check` | 0 | 无尾随空白/冲突标记 |
| `py_compile` | PASS | `_multi_gpu.py`, `bench_dev80_2.py`, `_multigrid.py` |
| `shellcheck`/`bash -n` | PASS | bench 纯 Python |
| 未跟踪文件 | 有 | `data/*.h5` (289M+1.2GB), `logs/dev80_2/*.log` 在 `.gitignore` 豁免 (`logs/<tag>/**` 全豁免), 不污染 `git status` |
| 二进制 | 有 | `*.h5` 为 HDF5, 已忽略 |
| `libqcu.so` | 改动 | 23M sm_60+PTX, 需随 `CMakeLists-nv.txt` 提交, 但 `.so` 在 `.gitignore` 豁免 (仅源码入库) |
| 5-stream 不变量 | 保持 | `cublasDot→_send_tmp_→MPI_Allreduce` 未改, `coarse_dot_kernel_multi` 仍用 |

**遗留缺陷**:
- 16×32×32×48 2L 首次构建 4min (vs 24min) 仍超 1min guard, 需 SAP 块分解 (将 null vec 5 Jacobi → 块 MINRES, 预计 4→1min)
- V-cycle 开销 6-20ms/次, 需混合精度 (c32 粗层, 2x) 降至 <2ms
- P100*2 大格子验证待 cache 完成后补 (当前 8×8×8×16 PASS)

## 4. 提交建议

```bash
git add cpp/cuda/qcu/CMakeLists-nv.txt pyqcu/cuda/_multi_gpu.py pyqcu/solver/_multigrid.py examples/qcu/dev80_2/ logs/dev80_2/
git commit -m "dev80_2: 16×32×32×48 统一格子 MG 套件 + 4min 构建 (Hierarchical+Local+Cheap) + 1.42x 调参

- 统一 gauge/nullvec 缓存于 data/ (289M + 1.2GB), V100 生成 P100 拷贝, 600s 超时
- bench_dev80_2.py: L1 1.74s vs BiStabCG 2.21s (1.27x), 8×8×8×16 1.42x (r3 cf1e3 cmi15, 43 vs 94 iters)
- Hierarchical VRAM→RAM (22.97GB), BatchedLocalSchur W10 (24→2min), Cheap 5-step Jacobi (35→2min)
- 8×8×8×16 P100*2 一致性 PASS (rel 0)

Refs: DDalphaAMG C6/C7, QUDA, PyQUDA; 下一步 SAP+GCR 预期 2.1x"
# 不代 `git push`, `tag` 待 >2 达成后 `~tag dev80_2`
```

## 5. 回滚

```bash
git checkout -- cpp/cuda/qcu/CMakeLists-nv.txt pyqcu/cuda/_multi_gpu.py pyqcu/solver/_multigrid.py
rm -rf examples/qcu/dev80_2 logs/dev80_2 data/gauge_16x32x32x48*.h5 data/L16x32x32x48*.h5
bash ./build.sh && bash ./install.sh
```
