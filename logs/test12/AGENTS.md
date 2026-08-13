# AGENTS.md — logs/test12

test12 —— dev74*（dev74 + dev74_1）整合测试套件工作目录，**test11_1 的优化版**，
目标不变：测试 CUDA C++ 版 MultiGrid 求解器性能（正确性 / 干净测量 / 参数扫描 /
大格子预算 / 加速比图表）。测试代码与运行脚本自包含（main.py 不 import 任何
dev73*/dev74*/test11 模块，仅内联照抄）。

## 与 test11 的核心差异（版本化产物目录）

| 方面 | test11 | test12 |
|---|---|---|
| 产物位置 | 固定 `logs/test11/` | **每次运行一个版本目录** `logs/test12/v<ts>/` |
| 二次运行 | 同名文件覆盖 | 新版本目录，天然隔离 |
| 跨环境比对 | 手动整理历史产物 | 各环境一次运行一个 `v<ts>/`，同名产物直接 diff/叠图 |
| 环境快照 | 无 | **`env.json`**（GPU 型号/显存/驱动、torch、git HEAD、命令）自动写入 |
| 输出重定向 | 不支持 | `--outdir` 参数 / `TEST12_OUTDIR` 环境变量（优先级前者） |
| 版本目录创建 | — | 运行脚本自动创建 + 自动 export（sweep 子进程继承同一目录） |

## 目录结构

```
logs/test12/
├── main.py           全部测试代码（子命令入口，--outdir 公共参数）
├── run-local.sh      本地运行脚本（RTX 4060 8GB 小卡验证）
├── run-local-v20260814.sh
│                     本地实测服务器 >1.2 加速比配置的对照脚本
│                     （8x8x8x16 2L/3L、8x16x16x16 3L；Step 5 输出
│                     本地实测 vs dev73_5 V100 服务器参考对照表）
├── run-snsc.sh       服务器运行脚本（默认 16GB 档，VRAM=32 预留 32GB 档）
├── run-snsc-v20260814.sh
│                     服务器 A100-40GB 对照脚本（默认 40GB 档，VRAM=16/32 可覆盖；
│                     Step 4 最大格子 20x32x32x64，warm 34.4G ≈86% 可行）
├── AGENTS.md         本文件（复现与比对指南）
└── v<YYYYMMDDHHMM>/  每次运行生成的版本目录（如 v202608140624；同分钟重跑加 -<SS>）
    ├── run-local-<ts>.log / run-snsc-<ts>.log   完整终端输出（tee 归档）
    ├── env.json                                 环境快照（比对基准）
    ├── test12_clean_L*.json                     干净测量（独立进程交叉计时+资源统计）
    ├── test12_bench.json                        批量基准汇总（--mode local/server）
    ├── test12_sweep.json                        参数扫描（r/ct/cmi/levels × speedup）
    ├── test12_verify_*.json                     正确性验证（gauge/解/null_vecs/CudaSchurOp 对照）
    ├── test12_results.json                      collect 汇总（表/图输入）
    ├── test12_budget_server_*g.json             显存/内存/磁盘预算表（16G/32G/40G 档）
    ├── test12_tbl_*.tex                         LaTeX 表（性能/资源/预算）
    ├── test12_*.png / test12_1_*.png            图（dev74 风格 / dev74_1 风格，范围同 dev73_5）
    └── clover_multigrid.log                     C++ 收敛日志归档副本
```

代码文件（main.py / run-*.sh / AGENTS.md）位于根目录**不入版本目录**，
保证多次运行共享同一份代码；运行产物全部进版本目录。

## 快速开始

```bash
cd /root/PyQCU
source ./env.sh

# 本地（小格子验证）——自动创建 logs/test12/v<ts>/
bash logs/test12/run-local.sh                # 实际执行；--dry-run 只打印命令
# 本地实测「服务器 >1.2 加速比配置」对照（8x8x8x16 2L/3L、8x16x16x16 3L）
bash logs/test12/run-local-v20260814.sh      # 实际执行；--dry-run 只打印命令

# 服务器（16GB 显存默认档）
bash logs/test12/run-snsc.sh                 # 实际执行
VRAM=32 bash logs/test12/run-snsc.sh         # 预留 32GB 档（暂不启用）

# 服务器 A100-40GB（40GB 显存默认档；16/32 档可覆盖）
bash logs/test12/run-snsc-v20260814.sh       # 实际执行；--dry-run 只打印命令
```

环境前提：C++ CUDA 后端与 Cython 扩展已构建（`bash ./build.sh && bash ./install.sh`）。

## main.py 子命令

```bash
python logs/test12/main.py <subcommand> [options] [--outdir <dir>]

clean   --lattice 8 8 8 16 --prec c64 --levels 2 --restart 10 --ct 1e5 --cmi 15 --pairs 3 [--build py|cpp]
bench   --mode local|server --vram 16 [--only 前缀...] [--build py|cpp]
verify  --lattice 8 8 8 16 --prec c64
sweep   --lattice 8 8 8 16 --pairs 3 --timeout 1800     # 子进程=main.py clean，透传输出
check   --gate 1.5 --file test12_sweep.json             # 可选工具（流程不再强制 gate）
budget  --mode server --vram 16|32|40 [--fit]          # 默认 16G 档，32/40 预留
collect | mktable --mode server --vram 16 | plots --vram 16 | plots1 [--file ...]
layout_test | stencil_mt --threads 4
```

`--outdir` 为公共参数（所有子命令均可，位置在子命令前后皆可）；
未指定时读 `TEST12_OUTDIR` 环境变量，再默认 `logs/test12/`。
每次调用自动在输出目录写 `env.json`（含命令与硬件/软件快照）。

## 版本目录约定

- **命名**：`v<YYYYMMDDHHMM>`（如 `v202608140624`）；同分钟重复运行自动追加 `-<SS>` 防覆盖。
- **创建**：run-*.sh 开头 `mkdir -p` + `export TEST12_OUTDIR=$VDIR`；主流程命令不带
  `--outdir`，经环境变量生效；`sweep` 派生的 clean 子进程自动继承同一版本目录。
- **env.json**：每次子命令调用刷新（sweep 子进程覆盖为 clean 命令记录，其余字段一致），
  作为该版本目录的环境基准；比对时先读它确认两目录硬件/软件差异。
- **C++ 收敛日志**：C++ 端写死 `REPO/logs/clover_multigrid.log`（相对 REPO），
  运行脚本结束时归档副本到版本目录。
- **清理**：版本目录不再需要时可整体删除（不影响代码与后续运行）。

## 跨环境比对流程

```bash
# 环境 A（本地 4060）
bash logs/test12/run-local.sh
# 环境 B（服务器 16G）
bash logs/test12/run-snsc.sh
# 环境 C（服务器 32G）
VRAM=32 bash logs/test12/run-snsc.sh
# 环境 D（服务器 A100-40GB）
bash logs/test12/run-snsc-v20260814.sh
```

→ 每次运行产生独立 `v<ts>/`，产物**同名同构**（test12_results.json / test12_*.png /
test12_tbl_*.tex 命名一致）：
- `diff logs/test12/vA/test12_results.json logs/test12/vB/test12_results.json` 直查差异
- 图件同名可直接叠图；`env.json` 提供各环境 GPU/驱动/torch/git 基准
- 加速比以服务器为准；本地 4060 上 MG 恒慢（speedup<1）是硬件特性，勿用本地
  结果推断服务器；参数相对行为（3L>2L、r20>r10）跨 GPU 一致

## 关键约定（沿用 test11）

- **显存档**：`--vram 16`（默认）对应服务器 16GB 卡；`--vram 32` / `VRAM=32`
  为预留 32GB 档；`--vram 40` / `VRAM=40` 为 A100-40GB 档（run-snsc-v20260814.sh
  默认）。16G 档大格子：8x32x32x32（cold 13.3G / warm 6.8G 全流程可行）、
  16x32x32x32（cold 26.5G 超档，warm 13.5G 需外部缓存）。40G 档最大格子：
  20x32x32x64（warm 34.4G ≈86% 可行，cold 69.5G 超档需外部缓存）。
- **nullvec 缓存**：共享 `logs/nullvec_cache`（`PYQCU_NULLVEC_CACHE` 可覆盖），
  跨 tag 复用粗算子，避免重复构建（8x32x32x32 首次构建 ~1-2h，缓存命中后秒级）。
- **CudaSchurOp 同步**：`matvec` 内 `torch.cuda.synchronize()`（test11 BUGFIX，
  修复 C++ 私有流异步 + 读取竞态 —— dev74 遗留的非确定结果问题）。
- **收敛日志**：见上「版本目录约定」。

## 产物对接

`collect` 汇总 → `test12_results.json`；`mktable`/`plots` 读它生成 LaTeX 表与
PNG；`plots1` 读 `test12_sweep.json`（作图范围与 dev73_5 一致：收敛历史/热点/
加速比/耗时/参数扫描，亮色调色板）。报告引用版本目录内产物即可；版本目录
名（`v<ts>`）与 `env.json` 即为报告的环境来源标注。
