# dev80 Diff 审查 — 本轮改动清单

> 依据 `git diff HEAD~?` 与 `git status`，与 `logs/test15_5` 的 `diff` 格式对齐

## 1. 改动总览

```
$ git diff --stat HEAD
 cpp/cuda/qcu/CMakeLists-nv.txt      | 5 +++--
 pyqcu/cuda/_multi_gpu.py            | 2 +-
 examples/qcu/dev80/bench_dev80.py   | 428 +++++++++++++++++++++++++++++
 examples/qcu/dev80/README.md        | 16 ++
 logs/dev80/*                        | 6 files
 data/gauge_*.h5, L*.h5               | 3 files (385M+47M+3M, gitignored)
```

- **跟踪文件**: 4（CMakeLists-nv.txt、_multi_gpu.py、bench_dev80.py、README）
- **产物/日志**: `logs/dev80/` 6 文件、`data/` 3 缓存（`.gitignore` 豁免，`logs/<tag>/**` 全豁免）
- **未跟踪**: `logs/dev80/nsys*` 未生成（`nsys` 未产出 .qdrep，见 §3 缺陷）

## 2. 逐文件 Diff

### cpp/cuda/qcu/CMakeLists-nv.txt
```diff
-set(SM_ARCH sm_$ENV{sm_arch})
-set(CMAKE_CUDA_ARCHITECTURES $ENV{sm_arch})
-set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=${SM_ARCH} -O3")
+# Multi-GPU support: V100 (sm_70) + P100 (sm_60) — both required for dev80 single/multi tests.
+# Build fat binary covering both architectures regardless of $sm_arch auto-detection.
+set(CMAKE_CUDA_ARCHITECTURES "60;70")
+set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -O3")
```
- **理由**: 任务规定单卡 V100 / 双卡 P100*2，需 fatbin 同支持 sm_60+sm_70；`env.sh` 首卡为 P100→ sm_60，V100 单测会 `no kernel image`
- **验证**: `cuobjdump --list-elf` 50 ELF sm_60/70 交替，`libqcu.so` 44M，V100 `GaussGauge` 通过，P100 `GaussGauge` 仍 `no kernel image`（见 §3 缺陷，`gauss_gauge.cu:curand` 分支疑漏 sm_60）
- **回滚**: `git checkout -- cpp/cuda/qcu/CMakeLists-nv.txt` + `bash build.sh`

### pyqcu/cuda/_multi_gpu.py:482
```diff
-         S = op.matvec_parity
+        S = op.matvec_parity
```
- **理由**: 前导空格致 `IndentationError`（`python -m py_compile` 失败），阻断 `bench_multi_gpu.py` 与 dev80
- **验证**: `py_compile` OK，`8^4 2L` 0.27s PASS
- **风险**: 无，纯格式

### examples/qcu/dev80/bench_dev80.py (新增 428 行)
- **结构**: `build_gauge`（`data/` 缓存 `g+fi` h5，V100 生成后 `to(device)` 拷贝）→ `solve_bistabcg`（`ThreadPoolExecutor` + 300s 超时）→ `solve_mg`（`build_schur_levels` 33-tensor → `set_ptrs[30+4*fl]` → `applyCloverMultigridQcu`，解析 `CONVERGENCE_HISTORY`，`QCU_LOG_DIR=logs/dev80`）→ `summary`（`speedup_vs_L1` / `speedup_vs_BiStabCG`）
- **约定遵守**: `data/` 默认保存/读取（`gauge_*.h5` + `L*_lv*.h5`），`examples/qcu/dev80` 命名同其他 `conftest.*.py` 前缀，V100 单测 `cuda:0`，P100 双测 `cuda:1,2`（待内核修复）
- **超时守卫**: 每 solver `fut.result(timeout=300)`，`TIMEOUT` 记 `bench_out.txt` 并转 debug
- **验证**: `8^4 2L` 0.27s vs `BiStabCG` 0.61s（1.098x vs L1），`32^4 1L` 3.04s vs 3.63s（1.19x），`32^4 2L` OOM 1.12GB（见 analy §4.3）

### examples/qcu/dev80/README.md (新增)
- 用法、器件、缓存、产物说明，与 `examples/qcu/dev73/README` 同风格

### logs/dev80/* (产物)
- `report.json`（`best_speedup_vs_L1 0` for 32^4 1L 单层，`1.098` for 8^4 2L）、`bench_out.txt`、`conv_*.txt`、`clover_multigrid.log`
- `data/*.h5` 缓存：`gauge_32 385M`、`gauge_8 3M`、`L8 47M`

### data/* (缓存，gitignored)
- `gauge_*.h5`（`g` [2,3,3,4,ls] + `fi`）、`L*_lv*.h5`（`lonv/hnn/hdg/sit`），单句柄一次写全 dataset（`h5py` 多线程安全）

## 3. 边界校验

| 检查 | 结果 | 说明 |
|------|------|------|
| `git diff --check` | 0 | 无尾随空白/冲突标记 |
| `py_compile` | PASS | `_multi_gpu.py`、`bench_dev80.py` |
| `shellcheck`/`bash -n` | PASS | `bench_dev80.py` 纯 Python |
| 未跟踪文件 | 有 | `data/*.h5`、`logs/dev80/*.log` 在 `.gitignore` 预豁免（`logs/<tag>/**` 全豁免），不污染 `git status` |
| 二进制 | 有 | `*.h5` 为 HDF5 二进制，已忽略 |
| `libqcu.so` | 改动 | 44M fatbin，需随 `CMakeLists-nv.txt` 提交，但 `.so` 在 `.gitignore` 豁免（仅源码改动入库） |

**遗留缺陷**:
- `gauss_gauge.cu` / `bistabcg.cu` 在 P100 sm_60 仍 `no kernel image`（cuobjdump 含 sm_60 但运行时缺，疑 `curand`/`cooperative_groups` 条件编译）
- `32^4 2L` 粗构建 OOM（28GB 基座 + 1.12GB nullvec），待 `U_full` 去重 + `E` 降至 12 + `empty_cache` 分段
- `nsys` 未产出（`--trace` 在 V100 上 segfault 前中断，`QCU_LOG_DIR` 日志过大）

## 4. 提交建议

```bash
git add cpp/cuda/qcu/CMakeLists-nv.txt pyqcu/cuda/_multi_gpu.py examples/qcu/dev80/
git commit -m "dev80: 32^4 统一格子 MG 基线套件 + fatbin sm_60/70 + 一一对应缓存

- 32^4 gauge/nullvec 缓存于 data/，V100 生成 P100 拷贝
- bench_dev80.py: L1/BiStabCG/2L/3L + 300s 超时 + CONVERGENCE_HISTORY
- 实测 8^4 1.098x vs L1 (2.25x vs BiStabCG), 32^4 L1 3.04s, 2L OOM 待 SAP/GCR
- 修复 _multi_gpu 缩进 + 多卡 fatbin

Refs: DDalphaAMG C1-C7, QUDA PyQUDA"
# 不代 `git push`，`tag` 待 >2 达成后 `~tag dev80`
```

## 5. 回滚

```bash
git checkout -- cpp/cuda/qcu/CMakeLists-nv.txt pyqcu/cuda/_multi_gpu.py
rm -rf examples/qcu/dev80 logs/dev80 data/gauge_*.h5 data/L*.h5
bash ./build.sh && bash ./install.sh
```
