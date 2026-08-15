---
name: test13
description: logs/test13 目录的完整生成 skill：多线程版（一线程一卡）CUDA C++ MultiGrid 求解器测试套件 — 单文件 main.py 子命令入口 + 版本化产物目录 v<ts>/ + 全部 h5py 持久化，测试 MultiGpuMultigrid（pyqcu/cuda/_multi_gpu.py）相对多线程 BiStabCG 的正确性与加速比（P100×2 多线程 + V100 单线程大格子）。
---
# Skill: test13

# test13 — 多线程版（一线程一卡）CUDA C++ MultiGrid 求解器测试套件

生成 `logs/test13/` 工作目录的完整指南。目标：测试**多线程版**
（`MultiGpuMultigrid`，pyqcu/cuda/_multi_gpu.py）CUDA C++ MultiGrid 求解器的
正确性与性能。与 test12 的核心差异：被测对象从单线程 `applyCloverMultigridQcu`
升级为 **N 线程并行、每线程绑定一张 GPU（一线程一卡）**；加速比基准从单线程
BiStabCG 升级为**多线程 BiStabCG**（墙钟 = max(各线程时间)）；数据持久化全部
h5py（多线程安全）。

## 核心设计

| 方面 | test12 | test13 |
|---|---|---|
| 被测对象 | 单线程 applyCloverMultigridQcu | **MultiGpuMultigrid**（N 线程×卡并行，每线程完整 C++ 后端流程） |
| 加速比基准 | 单线程 Clover BiStabCG | **多线程 CUDA C++ BiStabCG**（solve() 内各线程并行；墙钟=max） |
| 设备 | 单卡 | P100×2 多线程（device_ids=[1,2]）+ V100 单线程（device_ids=[0]）；三卡并行不测（任务约束） |
| 数据持久化 | json | **全部 h5py**（env.h5 / test13_*.h5；save_dict_h5/load_dict_h5，独立 File 句柄） |
| 真多线程 | 无 | 求解热点在 worker 线程各自卡上并行（qcu.pyx with nogil）；粗算子构建为 setup（主线程 V100，缓存命中秒级） |

## 目录结构

```
logs/test13/
├── main.py           全部测试代码（子命令入口，--outdir 公共参数）
├── run-local.sh      本地运行脚本（P100×2 多线程 + V100 单线程大格子）
├── AGENTS.md         复现与比对指南
├── test13_report.md  收尾结果报告
├── docs/             analy 报告（analy_test13_*.tex/.pdf）
└── v<YYYYMMDDHHMM>/  每次运行生成的版本目录（同分钟重跑加 -<SS>）
    ├── run-local-<ts>.log              完整终端输出（tee 归档）
    ├── env.h5                          环境快照（h5py；GPU/软件/git/命令）
    ├── test13_verify.h5                正确性验证（一致性/独立问题/V100/h5py IO）
    ├── test13_clean_L*.h5              干净测量（P100×2 多线程交叉计时 + RSS）
    ├── test13_bench.h5                 批量基准（P100×2 多线程组 + V100 单线程组）
    ├── test13_sweep.h5                 参数扫描（r/ct/cmi/levels × speedup）
    ├── test13_results.h5               collect 汇总（表/图输入）
    ├── test13_budget_*g.h5             显存/内存预算表（16G P100 / 32G V100 档）
    ├── test13_tbl_*.tex                LaTeX 表（bench/sweep）
    ├── test13_*.png                    图（加速比/耗时/参数扫描/热点）
    └── clover_multigrid.log            C++ 收敛日志归档副本
```

## 快速开始

```bash
cd /root/PyQCU
source ./env.sh
bash logs/test13/run-local.sh                # 实际执行；--dry-run 只打印命令
```

环境前提：C++ CUDA 后端与 Cython 扩展已构建（`bash ./build.sh && bash ./install.sh`）；
**单 MPI rank**（MultiGpuMultigrid 要求；直接 python 运行，勿 mpirun 多进程）。

## main.py 子命令

```bash
python logs/test13/main.py <subcommand> [options] [--outdir <dir>]

verify  [--lattice 8 8 8 16] [--h5threads 4]      # 一致性 P100×2 + 独立问题 + V100 单线程 + h5py IO
clean   --lattice 8 8 8 16 --levels 2 --restart 5 --ct 1e5 --cmi 15 --nthreads 2 [--devices 1 2]
bench   [--pairs 3] [--only 前缀...]              # P100×2 多线程组 + V100 单线程组
sweep   --lattice 8 8 8 16                        # r∈{5,10}×ct∈{1e4,1e5}×cmi∈{10,15}×L∈{2,3}（P100×2）
check   --gate 1.5 --file test13_sweep.h5         # 加速比断言（exit 0/1）
budget  --vram 16|32 [--lattices LxLyLzLt ...]    # 显存/内存预算表（16G P100 / 32G V100 档）
collect | mktable | plots                         # 汇总 h5 / LaTeX 表 / PNG 图
```

`--outdir` 公共参数；未指定读 `TEST13_OUTDIR` 环境变量，再默认 `logs/test13/`。
每次调用自动在输出目录写 `env.h5`。

## 关键约定

- **约定参数**（与 test12 一致）：mass=0.05, atol=1e-6, gauge_seed=42,
  kappa=1/(2m+8), E=48, NV_ITERS=2, MG_GRID=[2,2,2,2]。
- **加速比语义**：multi_ref_wall = max(各线程 ref_time)，multi_mg_wall =
  max(各线程 mg_time)，speedup = multi_ref_wall / multi_mg_wall。
  计时仅求解阶段（不含 setup/粗算子构建）。
- **h5py 约定**：数据持久化只用 h5py——每调用独立 File 句柄（with 语句）多线程
  安全；结果 dict → attrs+datasets（save_dict_h5）。PNG/TeX 为展示产物。
- **真多线程**：求解热点全部在 worker 线程各自卡上并行（with nogil 真并行）；
  禁止把热点计算放主线程。粗算子构建（setup，h5 缓存命中秒级）在主线程 V100
  完成——P100 sm_60 无 torch kernel image，属 setup 例外。
- **entries 读取**（2026-08-15 修复）：save_dict_h5 将 dict 列表写为数字 key 子组、
  读取还原为 list——消费方统一经 `_entries_list()` 展开，勿按 dict `.values()`
  处理；dataset 还原键为 `d_lattice`（用 `_lat_str()` 取格子串）。
- **bench 配置**（2026-08-15 调整）：V100 组为 16x16x16x16 3L + 8x16x16x16 3L
  （16x16x16x32 求解偏慢且无完整缓存，实测超 30min，已移除）。
- **nullvec 缓存**：共享 `logs/nullvec_cache`（`PYQCU_NULLVEC_CACHE` 可覆盖），
  跨 tag 复用粗算子；缓存 key 不区分设备，V100/P100 共享。
- **粗算子构建**（pyqcu/cuda/_multi_gpu.py，2026-08-15 统一）：一律走 C++ matvec
  路径（每线程一个 CudaSchurOp；单线程 nthreads=1 也用 1 个），避免 Python matvec
  构建大格子 50min+ 瓶颈。
- **收敛日志**：C++ 端写死 `REPO/logs/clover_multigrid.log`，运行脚本结束归档副本。

## 跨环境比对

每次运行产生独立 `v<ts>/`，产物同名同构；`env.h5` 提供 GPU/软件/git 基准；
`diff` 两个版本的 `test13_results.h5`（h5diff 或 load_dict_h5）直查差异。

## 已知硬件特性（勿误读）

- 本地 P100（sm_60，Pascal）：torch 无 kernel image，仅 libqcu.so（sm_60 SASS）
  可用——多线程测试的 torch 运算必须在主线程 V100 完成（setup 例外）。
- 大格子（16³×16 以上）MG 本身偏慢（coarse solve 开销，历史特性），
  加速比以中小格子（8x8x8x16 / 8x16x16x16）为准；大格子用 r10。
- 性能以本机 P100×2 / V100 实测为准；参数相对行为（3L>2L、r10>r5）可迁移。

## 实测参考（2026-08-15 收尾轮）

- verify 全 PASS（consistency/independent/V100/h5py IO，rel_diff=0.0）。
- bench（pairs=3）最佳：P100×2 8x8x8x16 2L speedup=2.101。
- sweep gate=1.5：11/16 PASS，最佳 2.149（8x8x8x16, L3 r10 ct1e5 cmi15）；
  因子均值 2L=1.874 vs 3L=1.357，r10=1.732 vs r5=1.499。
- 中/大格子 speedup<1（8x16x16x16、16x16x16x16）为已知 coarse solve 开销特性。

## 生成方式（反向撰写要点）

本套件由 test12 演进而来：main.py 为单文件入口（被测对象换为
MultiGpuMultigrid，持久化换 h5py），run-local.sh 按设备组织流程并自动创建
版本目录；AGENTS.md 为复现指南（见 `logs/test13/AGENTS.md` 权威版本，
本 skill 保持同步）。
