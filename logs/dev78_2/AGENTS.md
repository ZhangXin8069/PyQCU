# AGENTS.md — logs/dev78_2

dev78_2 —— **多线程版（一线程一卡）CUDA C++ MultiGrid 求解器**测试套件，
形式参考 test12（单文件 main.py 子命令入口 + 版本化产物目录 v<ts>/）。

## 与 test12 的核心差异

| 方面 | test12 | dev78_2 |
|---|---|---|
| 被测对象 | 单线程 `applyCloverMultigridQcu`（细层 C++ + 主线程驱动） | **`MultiGpuMultigrid`（pyqcu/cuda/_multi_gpu.py）**：N 线程并行、每线程绑定一张 GPU（一线程一卡），每线程完整 C++ 后端流程（参考 BiStabCG + Clover MG） |
| 加速比基准 | 单线程 Clover BiStabCG | **多线程 CUDA C++ BiStabCG**（solve() 内各线程并行跑 `applyCloverBistabCgQcu`；多线程墙钟 = max(各线程 ref_time)） |
| 设备 | 单卡 | **P100×2 多线程**（device_ids=[1,2]）+ **V100 单线程大格子**（device_ids=[0]）；三卡并行不测（任务约束） |
| 数据持久化 | json（test12_results.json / env.json 等） | **全部 h5py**（env.h5 / dev78_2_*.h5；save_dict_h5/load_dict_h5，独立 File 句柄多线程安全，参考 pyqcu/tools/_io.py） |
| 真多线程 | 无（单线程） | 求解热点全部在 worker 线程各自卡上并行（qcu.pyx with nogil）；测试脚本只编排收集（测试用途允许）；粗算子构建为 setup 阶段（h5 缓存命中秒级）在主线程 V100 完成（P100 sm_60 无 torch kernel image） |

## 目录结构

```
logs/dev78_2/
├── main.py           全部测试代码（子命令入口，--outdir 公共参数）
├── run-local.sh      本地运行脚本（P100×2 多线程 + V100 单线程大格子）
├── AGENTS.md         本文件（复现与比对指南）
├── dev78_2_report.md  收尾结果报告（2026-08-15；本次工作汇总）
├── docs/             analy 报告（analy_dev78_2_*.tex/.pdf）
└── v<YYYYMMDDHHMM>/  每次运行生成的版本目录（同分钟重跑加 -<SS>）
    ├── run-local-<ts>.log              完整终端输出（tee 归档）
    ├── env.h5                          环境快照（h5py；GPU/软件/git/命令）
    ├── dev78_2_verify.h5                正确性验证（一致性/独立问题/V100/h5py IO）
    ├── dev78_2_clean_L*.h5              干净测量（P100×2 多线程交叉计时 + RSS）
    ├── dev78_2_bench.h5                 批量基准（P100×2 多线程组 + V100 单线程组）
    ├── dev78_2_sweep.h5                 参数扫描（r/ct/cmi/levels × speedup）
    ├── dev78_2_results.h5               collect 汇总（表/图输入）
    ├── dev78_2_budget_*g.h5             显存/内存预算表（16G P100 / 32G V100 档）
    ├── dev78_2_tbl_*.tex                LaTeX 表（bench/sweep）
    ├── dev78_2_*.png                    图（加速比/耗时/参数扫描/热点）
    └── clover_multigrid.log            C++ 收敛日志归档副本
```

代码文件（main.py / run-local.sh / AGENTS.md）位于根目录**不入版本目录**；
运行产物全部进版本目录。

## 快速开始

```bash
cd /root/PyQCU
source ./env.sh
bash logs/dev78_2/run-local.sh                # 实际执行；--dry-run 只打印命令
```

环境前提：C++ CUDA 后端与 Cython 扩展已构建（`bash ./build.sh && bash ./install.sh`）；
单 MPI rank（MultiGpuMultigrid 要求；直接 python 运行，勿 mpirun 多进程）。

## main.py 子命令

```bash
python logs/dev78_2/main.py <subcommand> [options] [--outdir <dir>]

verify  [--lattice 8 8 8 16] [--h5threads 4]      # 一致性 P100×2 + 独立问题 + V100 单线程 + h5py IO
clean   --lattice 8 8 8 16 --levels 2 --restart 5 --ct 1e5 --cmi 15 --nthreads 2 [--devices 1 2]
bench   [--pairs 3] [--only 前缀...]              # P100×2 多线程组 + V100 单线程大格子组
sweep   --lattice 8 8 8 16                        # r∈{3,5,10}×ct∈{1e4,1e5}×cmi∈{10,15}×L∈{2,3}（P100×2）
check   --gate 1.5 --file dev78_2_sweep.h5         # 加速比断言（exit 0/1）
budget  --vram 16|32 [--lattices LxLyLzLt ...]    # 显存/内存预算表（默认 16G P100 档，32G V100 档）
collect | mktable | plots                         # 汇总 h5 / LaTeX 表 / PNG 图
```

`--outdir` 公共参数（子命令前后皆可）；未指定读 `TEST78_2_OUTDIR` 环境变量，
再默认 `logs/dev78_2/`。每次调用自动在输出目录写 `env.h5`。

## 关键约定

- **约定参数**（与 test12 一致）：mass=0.05, atol=1e-6, gauge_seed=42,
  kappa=1/(2m+8), E=48, NV_ITERS=2, MG_GRID=[2,2,2,2]。
- **加速比语义**：multi_ref_wall = max(各线程 ref_time)（多线程 BiStabCG 墙钟），
  multi_mg_wall = max(各线程 mg_time)（多线程 MG 墙钟），
  speedup = multi_ref_wall / multi_mg_wall。计时仅求解阶段（不含 setup/粗算子构建）。
- **h5py 约定**：数据持久化只用 h5py——每调用独立 File 句柄（with 语句）多线程
  安全；多 dataset 单句柄一次写完；结果 dict → attrs+datasets（save_dict_h5）。
  PNG/TeX 为图表展示产物（matplotlib/文本渲染，非数据持久化）。
- **真多线程**：求解热点（BiStabCG/MG 内核）全部在 worker 线程各自卡上并行
  （qcu.pyx with nogil 真并行）；禁止把热点计算放主线程。粗算子构建（setup，
  h5 缓存命中秒级）在主线程 V100 完成——P100 sm_60 无 torch kernel image，
  属非计算热点 setup 例外。
- **nullvec 缓存**：共享 `logs/nullvec_cache`（`PYQCU_NULLVEC_CACHE` 可覆盖），
  跨 tag 复用粗算子；缓存 key 不区分设备，V100/P100 共享。
- **收敛日志**：C++ 端写死 `REPO/logs/clover_multigrid.log`，运行脚本结束归档副本。
- **entries 读取**（2026-08-15 修复）：save_dict_h5 将 dict 列表写为数字 key 子组、
  读取还原为 list——消费方统一经 `_entries_list()` 展开（main.py:557），
  勿按 dict `.values()` 处理；dataset 还原键为 `d_lattice`（用 `_lat_str()` 取格子串，
  main.py:622）。
- **bench 配置**（2026-08-15 调整）：V100 组为 16x16x16x16 3L + 8x16x16x16 3L
  （16x16x16x32 求解偏慢，移除；大格子预算见 budget 子命令）。
- **粗算子构建加速**（2026-08-15 dev78_2 优化，pyqcu/tools/_multigrid.py +
  pyqcu/cuda/_multi_gpu.py）：
  * nv_tol=1e-2（默认）：null 向量 BiCGStab 容差从 5e-5 放宽 —— 粗层大系统
    （16x16x16x32 lv2，196608 未知数）5e-5 迭代爆炸（>34min 未完成）→ 分钟级；
    小格子 8x8x8x16 质量等价（rel_diff=0，sweep 16/16≥1.5 vs test13 11/16）。
  * 批量 stencil 探测（_probe_point_batch + _schur_matvec_batch +
    _stencil_matvec_batch）：固定 c_idx 一次批量全部 E 探针（torch einsum，
    单位向量 prolong 切片化 + restrict 邻域块局部化）—— 8x8x8x16 lv1
    12288 probes 135.6s → 3.3s（21 倍）；16x16x16x32 lv1（196608 probes）
    ~36min → ~3min。
  * 批量 BiCGStab（_bistabcg_batch）：null 向量 dof 个右端一次批量迭代
    （标量按批独立 + 复数安全除法）—— 16x16x16x32 lv2 从 40min+ → 83s。
  * 实测 16x16x16x32 3L 完整构建 86s（原 1h+ 未完成）；求解正确
    （consistency=True），speedup=0.75（大格子 MG coarse solve 开销，
    历史特性，正确性优先）。
  * 缓存 key 含 `_t{nv_tol}` 后缀（旧 5e-5 缓存自动失效重建）。

- **中/大格子参数优化**（2026-08-16 dev78，logs/dev78/main.py bench 配置）：
  * 8x16x16x16 2L/3L num_restart 5→10、16x16x16x16 2L 10→20：V-cycle 频率
    减半、粗层求解次数减半（中格子粗层求解是校正主导成本，r5 时 n_vcycles
    13-44 次爆炸）。
  * 实测：8x16x16x16 2L 0.33-0.69→1.05-1.22（约 3 倍）、3L 0.45-0.72→1.41
    （约 2 倍）；bench median 1.148→1.210、max 2.135→2.288。
  * 16x16x16x16 2L r20=0.74 仍 <1（层数固有特性）；fused 阈值 32K 试验
    更差已回退 64K。

- **dev78_1 完整复测**（2026-08-16，参考 logs/dev74 输出形式）：
  * 全套图表：speedup/time/sweep/hotspot（main.py plots）+ conv_bench/vram/prof
    （make_extra_plots.py 补充）；
  * 全部日志归档：verify/clean/bench/sweep 每阶段独立 log + C++ 收敛日志；
  * 实测：bench median 1.422（dev76 1.148 → dev78 1.210 → 持续提升）、
    max 2.271、sweep 14/16≥1.5 best 2.480、verify 全 PASS。

## 跨环境比对

每次运行产生独立 `v<ts>/`，产物同名同构；`env.h5` 提供 GPU/软件/git 基准；
`diff` 两个版本的 `dev78_2_results.h5`（h5diff 或 load_dict_h5）直查差异。

## 已知硬件特性（勿误读）

- 本地 P100（sm_60，Pascal）：torch 无 kernel image，仅 libqcu.so（sm_60 SASS）
  可用——多线程测试的 torch 运算必须在主线程 V100 完成（setup 例外）。
- 大格子（16³×16 以上）MG 本身偏慢（coarse solve 开销，历史特性），
  加速比以中小格子（8x8x8x16 / 8x16x16x16）为准；大格子用 r10。
- 性能以本机 P100×2 / V100 实测为准；参数相对行为（3L>2L、r10>r5）可迁移。
