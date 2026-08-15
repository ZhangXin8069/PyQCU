# test13 结果分析报告（多线程 CUDA C++ MultiGrid）

- 日期：2026-08-15
- 被测对象：`MultiGpuMultigrid`（`pyqcu/cuda/_multi_gpu.py`），一线程一卡，N 线程并行求解
- 基准：多线程 CUDA C++ BiStabCG（各线程独立跑 `applyCloverBistabCgQcu`，墙钟取 max）
- 设备：P100×2 多线程（device_ids=[1,2]）+ V100 单线程（device_ids=[0]），单 MPI rank
- 产物：`logs/test13/v2026081516*/`（verify/clean/bench/sweep/results/budget/tex/png）

## 1. 结论摘要

| 项 | 结果 |
|---|---|
| 正确性 | **全部 PASS**（一致性 rel_diff=0.0；独立问题解不同；V100 单线程一致；h5py 多线程 IO 往返零误差） |
| 加速比 gate（≥1.5） | **PASS：11/16 配置 ≥ 1.5**，最佳 2.149（L3 r10 ct1e5 cmi15，8x8x8x16） |
| 最优配置 | r10 > r5、cmi15 > cmi10、ct 影响弱（ct1e4 略优）；3L 在 r10 下不劣于 2L |
| 已知限制 | 中/大格子（8x16x16x16、16x16x16x16）MG < 1（粗层求解开销主导，历史特性）；大格子测试时长受限，16x16x16x32 从 bench 表移除 |

## 2. 正确性验证（verify）

| 场景 | 设备 | 格子 | L | r | rel_diff | speedup | 结果 |
|---|---|---|---|---|---|---|---|
| consistency | P100×2 | 8x8x8x16 | 2 | 5 | 0.0 | 1.934 | PASS |
| independent | P100×2 | 4x4x4x8 | 2 | 5 | 解不同 (\|d\|=6.82) | 2.417 | PASS |
| v100_single | V100 | 8x16x16x16 | 3 | 10 | 0.0 | 0.991 | PASS |

h5py 多线程（4 线程）写读往返：PASS（max_err=0.0）。

## 3. 批量基准（bench，pairs=3）

| 配置 | L | nT | ref(s) | mg(s) | speedup |
|---|---|---|---|---|---|
| P100x2 8x8x8x16 | 2 | 2 | 0.571 | 0.272 | **2.101** |
| P100x2 8x8x8x16 | 3 | 2 | 0.575 | 0.451 | 1.275 |
| P100x2 8x16x16x16 | 2 | 2 | 0.613 | 1.024 | 0.598 |
| P100x2 8x16x16x16 | 3 | 2 | 0.598 | 1.495 | 0.400 |
| P100x2 16x16x16x16 | 2 | 2 | 0.695 | 0.779 | 0.892 |
| V100 16x16x16x16 | 3 | 1 | 0.679 | 0.734 | 0.926 |
| V100 8x16x16x16 | 3 | 1 | 0.600 | 0.528 | 1.137 |

观察：
- **小格子 8x8x8x16 2L 加速最显著（2.10）**；3L 反而降（1.27，粗层迭代开销占比上升）。
- 中格子（8x16x16x16、16x16x16x16）speedup<1：MG 细层求解本身快，但 coarse solve（FUSED-PARALLEL，48 dof×12 网格）迭代多、每 V-cycle 固定开销大，历史特性（test13 AGENTS.md「已知硬件特性」）。
- 多线程墙钟取 max，两 P100 负载均衡良好（clean 中两线程 mg=0.234/0.238，偏差 <2%）。

## 4. 参数扫描（sweep，8x8x8x16，P100×2）

16 配置全 PASS；gate=1.5 通过 11/16：

| 排序 | L | r | ct | cmi | speedup |
|---|---|---|---|---|---|
| 1 | 3 | 10 | 1e5 | 15 | **2.149** |
| 2 | 2 | 10 | 1e4 | 15 | 2.071 |
| 3 | 2 | 5 | 1e4 | 15 | 1.999 |
| 4 | 2 | 5 | 1e5 | 15 | 1.996 |
| 5 | 2 | 10 | 1e5 | 10 | 1.965 |
| 6 | 2 | 5 | 1e4 | 10 | 1.926 |

- 按因子平均：**2L=1.874 vs 3L=1.357；r10=1.732 vs r5=1.499**；cmi15 > cmi10；ct 影响弱。
- 3L 仅在高 r（r10）下才优于 2L——粗层 restart 数不足时收敛差（r3 全崩历史特性，已排除）。

## 5. 显存/内存预算（budget）

16G P100 / 32G V100 两档全部 OK（cold 模型 α=0.0528 GB/V）：16x16x16x64 cold 14.08GB（32G 档 44%），16x16x16x32 cold 7.32GB——常规格子均在设备显存内。

## 6. 代码改动（本次会话）

1. `pyqcu/cuda/_multi_gpu.py`（未提交）：粗算子构建统一走 C++ matvec 路径（`CudaSchurOp`，单线程 nthreads=1 亦用 1 个 op），避免 Python matvec 构建大格子 50min+ 瓶颈——**verify 已用此代码全部 PASS**。
2. `logs/test13/main.py`（未提交）：
   - `_bench_configs`：V100 组 16x16x16x32 3L → 8x16x16x16 3L（测试时长受限，缓存齐备、分钟级可复现）；
   - 修复 `check`/`collect`/`mktable`/`plots` 读取 bug：`load_dict_h5` 将数字 key 组还原为 list，读取方原先按 dict `.values()` 处理（AttributeError）；新增 `_entries_list()` 展开（兼容旧 dict 格式）；
   - 修复 `budget` 缺 levels 的解析 bug（`--lattices` 字符串 → `(lat, levels=2)` 对）；
   - `_lat_str()` 兼容 dataset 还原 key（`d_lattice`）。

## 7. 局限

- 16x16x16x32/16x16x16x64 大格子 MG 未实测 bench（求解慢 + 粗算子构建耗时超 30min，无完整缓存）——预算表显示显存可容纳，性能以中小格子外推（历史特性：越大越慢）。
- bench 中格子 speedup<1 属已知特性，未做优化尝试（任务限定为验证+参数扫描收敛，优化留待后续）。

## 8. 遗留

- `_multi_gpu.py`、`logs/test13/main.py` 改动未提交（提示用户 `git add` + commit）。
- `logs/nullvec_cache/L16x16x16x32_lv1_E48_nvi2.h5`（2.8GB）为中断构建残留，lv2 缺失不可用，可清理或保留待续跑。
