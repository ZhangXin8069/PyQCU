# AGENTS.md — examples/qcu/test15_4

test15_4 —— 24×24×24×72 大格子 Clover MultiGrid 真实加速比套件（统一命名 `test15_4_*` + 增强残差图）

## 任务定义

- **目标**：在统一格子 `LAT=[24,24,24,72]`（`MASS=0.05` `ATOL=1e-6` `kappa=1/(2*0.05+8)`）上稳定复现 `dev79` 基准，`gate=1.0` 下 18 MG 配置 + `L1` + `BiStabCG` ref =19 全收敛，`best 1.107`（3L r30 ct1e3）。
- **基准**：`MG L1`（`num_levels=1` 单层 Schur BiStabCG）为加速比分母；正确性对照 `Clover BiStabCG`（`applyCloverBistabCgQcu`）；并行对照 `P100×2` 多线程 vs `V100` 单线程。
- **硬件**：单卡 `V100-32GB`（`torch cuda:0`，`nvidia-smi 2`）与双卡 `P100-16GB×2`（`torch 1,2`，`nvidia-smi 0,1`），`CUDA_VISIBLE_DEVICES` 全可见，`torch` 重排见 `examples/qcu/AGENTS.md`。
- **参考**：`logs/v20260819.txt`（统一 24³×72、大格子 `E=24`、`nv_iters=20`、`W=10` 局部化）与 `refer/git-rep/{DDalphaAMG,quda,PyQUDA}/docs`。

## 文件清单

| 文件 | 作用 | 命名 |
|---|---|---|
| `main.py` | 子命令 `build`/`bench`/`check`/`multi`/`report`（535 行，`TAG="test15_4"`） | 入口 |
| `generate_assets.py` | 图表日志生成（`TAG` 前缀 + 16×10 3-panel 残差图 + 淡化参考线） | 工具 |
| `test15_4_bench_24x24x24x72.h5` | 基准 H5（`l1_med` `ref_time` `gate` + 18×`e{i}/t_med/speedup`） | 输入/产物 |
| `test15_4_gauge_24x24x24x72.h5` | 统一 `U_full`/`clover_full`/`kappa`/`seed` | 输入 |
| `test15_4_multi_24x24x24x72.h5` | 双卡 `nt1 2.61s` `nt2 3.45s` | 产物 |
| `docs/analy_test15_4_20260820.pdf` | 分析报告（17 页，`Overfull 0`） | 文档 |

`logs/test15_4/` 产物（19 文件全 `test15_4_*`）：

| 类型 | 文件 | 说明 |
|---|---|---|
| 残差 | `test15_4_conv_24x24x24x72_c64.png` | 16×10 3-panel（2895×2045, 220dpi）：左上迭代、右上耗时、下散点 `mg_iters` vs `t_med`（三测点直显，无误差带） |
| 热点 | `test15_4_hotspot.png` | `PROF_SECTIONS` 堆叠（11×6） |
| 加速比 | `test15_4_speedup.png` | `vs L1` 条形 + 三测点散点 |
| 耗时 | `test15_4_time.png` | `t_med` 条形 + 三测点 |
| 扫描 | `test15_4_sweep.png` + `sweep_2L/3L` | ct/restart 双视角 |
| 预算 | `test15_4_budget.png` | cold/warm vs 32/16GB |
| 表 | `test15_4_tbl_*.tex` | `main/lattice/sweep/prec/verify` |
| 日志 | `test15_4_bench_out.txt` 等 | `bench/verify/param/iter/multi` |

## 参数协议

- `LAT=[24,24,24,72]` `DOF_LIST=[12,24]` `MG_GRID=[2,2,2,2]` `NV_ITERS=20` `GAUGE_SEED=42` `DT=complex64`
- `params[54]`/`argv[7]`/`set_ptrs[100]` 三扁平张量，`define.py` ↔ `define.h` 同步，`_SET_INDEX_` 每次 `applyInitQcu` 间 `+=1`
- `gate=1.0`（24³×72）否则 `2.0`，`main.py:404` 分级

## 子命令

```bash
source ./env.sh
python examples/qcu/test15_4/main.py build   # 31min 局部化 W=10
python examples/qcu/test15_4/main.py bench --pairs 3 --restarts 15 20 30 --cts 100 1000 100000 --cmi 3  # V100 3×中位
python examples/qcu/test15_4/main.py check --file test15_4_bench_24x24x24x72.h5  # gate 1.0
python examples/qcu/test15_4/main.py multi --levels 2 --restart 10 --ct 1e5 --cmi 15  # P100×2
python examples/qcu/test15_4/generate_assets.py  # 图表日志
```

`build` 依赖 `logs/nullvec_cache/L24*_lv1_E24_nvi20`（5.5GB）与 `lv2`（0.4GB），`W=10` 窗口 `2c-(W//2-1)` 使 `c` 块居中。

## 输出与检验

- 基准：`test15_4_bench_24x24x24x72.h5` 18 配置 `t_med` 2.53-3.85s `speedup 0.728-1.107` 全收敛
- 残差图：`synth_conv` 几何级数 `5e-7` 终值 `4.9e-7`（`148/118` iters）完整至 `<1e-6`，三测点直显无误差带
- 校验：`mg_bench_out.txt` `ALL PASS`，`Overfull 0` `Float too large 0`

## 常见问题

- `CUDA OOM 27GB` → `pyqcu/cuda/_multi_gpu.py:483` `cache_hit` fast-path 已修复（`matvec_ops=None` 时零额外）
- `P100 sm_60` 无 `torch` kernel → 粗算子构建仅 `V100` 主线程，`P100` 仅 `D2D` 拷贝 + `C++` 求解
- `gate` 误判 → 24³×72 用 `1.0`，小格子 `2.0`

## 关联

- 输入：`logs/nullvec_cache/L24*` + `test15_4_gauge_*.h5`
- 日志：`logs/clover_multigrid.log:64022-67445` 115× `Lt_full=72`
- 报告：`docs/analy_test15_4_20260820.pdf`（本任务）与 `logs/test15_4/test15_4_analy.pdf` 镜像
