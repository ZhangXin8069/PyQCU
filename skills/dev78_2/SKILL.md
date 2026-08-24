---
name: dev78_2
description: logs/dev78_2 目录的完整生成 skill：多线程版（一线程一卡）CUDA C++ MultiGrid 求解器测试套件 — 单文件 main.py 子命令入口 + 版本化产物目录 v<ts>/ + 全部 h5py 持久化 + 全求解器迭代残差图（conv/conv_plots 子命令：MG 实测 CONVERGENCE_HISTORY + 参考 BiStabCG Python 复现，逐配置/逐格子/汇总全采集）。测试 MultiGpuMultigrid（pyqcu/cuda/_multi_gpu.py）相对多线程 BiStabCG 的正确性与加速比（P100×2 多线程 + V100 单线程大格子）。
---
# Skill: dev78_2

# dev78_2 — 多线程 MultiGrid 求解器测试套件（全求解器迭代残差图补全）

生成 `logs/dev78_2/` 工作目录的完整指南。与 dev78_1 同构（main.py 子命令 +
版本化产物目录 + h5py 持久化），**核心新增：全求解器迭代残差图**——
`conv` 子命令逐配置采集（MG 实测 + ref 复现），`conv_plots` 子命令成图
（逐配置 21 + 逐格子多参数 7 + 全配置汇总 1 + LOG 提取 1 = 29 张）。

## 核心设计

| 方面 | dev78_1 | dev78_2 |
|---|---|---|
| 被测对象 | MultiGpuMultigrid（N 线程×卡并行） | 同（无变化） |
| 加速比基准 | 多线程 CUDA C++ BiStabCG | 同 |
| 残差图 | 仅 1 张 conv_bench.png（LOG 收敛点） | **29 张**：conv 子命令逐配置采集（21 配置 × MG 实测 + ref 复现）+ 逐格子组图 + 汇总 |
| 数据持久化 | h5py | 同 + `dev78_2_conv.h5`（vlen 变长残差数组） |
| 报告 | dev78_1_report.md + docs/analy tex/pdf | 更详细：残差分析专章（21 配置收敛表 + 收敛行为解读） |

## 目录结构

```
logs/dev78_2/
├── main.py           全部测试代码（11 子命令，--outdir 公共参数）
├── run-local.sh      本地运行脚本（verify→clean→bench→sweep→check→collect→budget→conv→归档）
├── AGENTS.md         复现与比对指南
├── dev78_2_report.md  收尾结果报告（含残差分析专章）
├── docs/             analy 报告（analy_dev78_2_20260816.tex/.pdf，Overfull=0 交付）
└── v<YYYYMMDDHHMM>/  每次运行生成的版本目录（同分钟重跑加 -<SS>）
    ├── run-local-<ts>.log + run-local-continue-*.log  完整终端输出（tee 归档）
    ├── env.h5 / dev78_2_verify.h5 / dev78_2_clean_*.h5
    ├── dev78_2_bench.h5 / dev78_2_sweep.h5 / dev78_2_results.h5
    ├── dev78_2_conv.h5               迭代残差数据（entries: label/lattice/levels/restart/
    │                                 ct/cmi/nthreads/device_ids/ref_hist/conv_mg/speedup/...）
    ├── dev78_2_budget_*g.h5 / dev78_2_tbl_*.tex
    ├── dev78_2_speedup.png / time.png / sweep.png / hotspot.png
    ├── dev78_2_conv_*.png            29 张迭代残差图（本套件核心新增）
    ├── dev78_2_vram_*g.png / prof.png / conv_bench.png
    └── clover_multigrid.log          C++ 收敛日志归档副本
```

## 快速开始

```bash
cd /root/PyQCU
source ./env.sh
bash logs/dev78_2/run-local.sh                # 实际执行；--dry-run 只打印命令
```

环境前提：libqcu.so + Cython 扩展已构建；单 MPI rank（勿 mpirun 多进程）。

## main.py 子命令

```bash
python logs/dev78_2/main.py <subcommand> [options] [--outdir <dir>]

verify | clean | bench | sweep | check | budget | collect | mktable | plots   # 同 dev78_1
conv        # 全求解器迭代残差历史采集 → dev78_2_conv.h5（21 去重配置：bench7+sweep16+verify3）
conv_plots  # 迭代残差图 → dev78_2_conv_*.png（逐配置 21 + 逐格子 7 + 汇总 1）
```

## 关键约定（同 dev78_1）

- 约定参数：mass=0.05, atol=1e-6, gauge_seed=42, κ=1/(2m+8), E=48,
  NV_ITERS=2, MG_GRID=[2,2,2,2]；nullvec 缓存共享 logs/nullvec_cache。
- 加速比语义：multi_ref_wall = max(各线程 ref_time)，speedup = ref/mg；
  计时仅求解阶段。
- h5py 约定：save_dict_h5/load_dict_h5（独立 File 句柄多线程安全）；
  **变长残差数组**（MG 每线程一条、长度不同）用 h5py vlen dataset
  （`_h5_write` 的 list-of-list 分支，读回为 object ndarray，消费方经
  `_get_entry_list()` 展开）。
- 设备分配：P100×2 多线程（device_ids=[1,2]，torch 视角 1/2=P100、
  0=V100）+ V100 单线程大格子（device_ids=[0]）。注意 nvidia-smi index
  顺序与 CUDA 运行时视角不同（smi: 0/1=P100、2=V100），以 torch
  `get_device_properties` 实测为准。

## conv 子命令实现要点（核心）

- **MG 残差（C++ 实测）**：C++ 端每次 applyCloverMultigridQcu 求解无条件
  写 `CONVERGENCE_HISTORY: [r0,r1,...]` 到 REPO/logs/clover_multigrid.log
  （lattice_clover_multigrid.h:1673，非 verbose 依赖）——**无需修改 C++**。
  采集：测量前记录日志文件字节偏移（os.path.getsize），测量后
  `_parse_conv_histories(offset)` 增量解析（正则 `CONVERGENCE_HISTORY:\s*\[([^\]]*)\]`）；
  多线程时每线程一条，均保留。
- **ref 残差（Python 复现，dev73_5 已验证）**：C++ stdout 只输出收敛点
  （PRINT_MULTI_GPU_CLOVER_BISTABCG 默认关闭）；用同一 Schur 算子 +
  同一 BiStabCG 算法在 torch 上复现（`_bistabcg_history` +
  `_ref_conv_history`：`op.give_b_parity` + `op.matvec_parity`）。
- **每格子 op 缓存**：`ref_cache[latk] = (op, b_full)`，4 个格子各建一次
  （V100 主线程，`_setup_gpu_tensors` + dslash.operator）。
- 配置集合：`_conv_configs()` —— bench 7 + sweep 16 + verify 3 按
  (lattice, levels, restart, ct, cmi) 去重 → 21。
- 2026-08-16 实测：21/21 配置 MG/ref 残差全收敛 < 1e-6；多线程 2 曲线重合。

## 迭代残差图（conv_plots）

| 类型 | 数量 | 文件 | 内容 |
|---|---|---|---|
| 逐配置 | 21 | `dev78_2_conv_<lat>_L<n>_r<n>_ct<n>_cmi<n>.png` | MG 各线程曲线 + ref + atol=1e-6 虚线 + speedup 标注 |
| 逐格子逐 L | 7 | `dev78_2_conv_<lat>_L<n>.png` | 同组多参数 MG 对比 + ref（dev74_1 风格） |
| 汇总 | 1 | `dev78_2_conv_all.png` | 4 格子子图全部配置 |
| LOG 提取 | 1 | `dev78_2_conv_bench.png` | make_extra_plots（全部运行收敛点） |

图标题用英文（DejaVu Sans 无中文字形）；x 轴迭代号 0 起。

## 实测参考（2026-08-16）

- verify 全 PASS（consistency rel=[0,0]、independent |d|=5.60、
  V100 3L rel=[0]、h5py 4 线程 IO）。
- clean：8x8x8x16 2L P100x2 speedup=2.300（ref 0.579s / mg 0.252s）。
- bench（pairs=3）：中位 1.411、最大 2.303（8x8x8x16 2L）；
  8x16x16x16 3L P100x2=1.748、V100 16³ 3L=1.141、16³ 2L r20=0.700(<1 已知)。
- sweep：13/16 ≥ 1.5、best 2.413（L3 r10 ct1e5 cmi15）。
- conv：21 配置残差全收敛；MG 迭代 30-120 vs ref 81-90（8x8x8x16）；
  ct1e4 校正弱 → 迭代膨胀（120 次），ct1e5+cmi15 → 45 次 / speedup 2.43。
- 演进：bench 中位 dev76 1.148 → dev78 1.210 → dev78_1 1.422 → dev78_2 1.411。

## 生成方式（反向撰写要点）

dev78_2 由 dev78_1 演进而来：main.py 新增 conv/conv_plots 子命令
（残差采集 + 成图），run-local.sh 增加 Step 8 conv；报告增加残差分析
专章（21 配置收敛表 + 收敛行为解读 + 残差图清单）。`_h5_write` 增加
vlen 变长列表支持（向后兼容）。AGENTS.md 为复现指南（见
`logs/dev78_2/AGENTS.md` 权威版本，本 skill 保持同步）。
