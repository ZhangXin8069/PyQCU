# AGENTS.md — logs/test11

test11 —— dev74*（dev74 + dev74_1）整合测试套件的工作目录。所有测试产物
（json / png / tex / 运行日志）均存放于本目录；测试代码与运行脚本自包含
（main.py 不 import 任何 dev73*/dev74* 模块，仅内联照抄）。

## 文件清单

| 文件 | 用途 |
|---|---|
| `main.py` | 全部测试代码（子命令入口，见下） |
| `run-local.sh` | 本地运行脚本（RTX 4060 8GB 小格子验证） |
| `run-snsc.sh` | 服务器运行脚本（默认 16GB 显存档，`VRAM=32` 预留 32GB 档） |
| `AGENTS.md` | 本文件（复现指南） |
| `test11_clean_L*.json` | 干净测量（独立进程交叉计时 + 资源统计） |
| `test11_bench.json` | 批量基准汇总（--mode local/server） |
| `test11_sweep.json` | 参数扫描（r/ct/cmi/levels × speedup） |
| `test11_verify_*.json` | 正确性验证（gauge/解/null_vecs/CudaSchurOp 对照） |
| `test11_results.json` | collect 汇总（表/图输入） |
| `test11_budget_server_*g.json` | 显存/内存/磁盘预算表（16G/32G 档） |
| `test11_tbl_*.tex` | LaTeX 表（性能 / 资源 / 预算） |
| `test11_*.png` / `test11_1_*.png` | 图（dev74 风格 / dev74_1 风格，作图范围同 dev73_5） |
| `run-local-<ts>.log` / `run-snsc-<ts>.log` | 运行完整输出（tee 归档） |
| `clover_multigrid.log` | C++ 收敛日志归档副本（C++ 端写死 REPO/logs/ 下） |

## 快速开始

```bash
cd /root/PyQCU
source ./env.sh

# 本地（小格子验证）
bash logs/test11/run-local.sh                # 实际执行；--dry-run 只打印命令

# 服务器（16GB 显存默认档）
bash logs/test11/run-snsc.sh                 # 实际执行
VRAM=32 bash logs/test11/run-snsc.sh         # 预留 32GB 档（暂不启用）
```

环境前提：C++ CUDA 后端与 Cython 扩展已构建（`bash ./build.sh && bash ./install.sh`）。

## main.py 子命令

```bash
python logs/test11/main.py <subcommand> [options]

clean   --lattice 8 8 8 16 --prec c64 --levels 2 --restart 10 --ct 1e5 --cmi 15 --pairs 3 [--build py|cpp]
bench   --mode local|server --vram 16 [--only 前缀...] [--build py|cpp]
verify  --lattice 8 8 8 16 --prec c64
sweep   --lattice 8 8 8 16 --pairs 3 --timeout 1800     # 子进程=main.py clean，透传输出
check   --gate 1.5 --file test11_sweep.json             # 可选工具（流程不再强制 gate）
budget  --mode server --vram 16|32 [--fit]              # 默认 16G 档，32G 预留
collect | mktable --mode server --vram 16 | plots --vram 16 | plots1 --file ...
layout_test | stencil_mt --threads 4
```

sweep 的 `--timeout` 为每配置子进程超时（防卡壳）；运行脚本每步也带 timeout，
单步失败仅记录继续（16G 档大格子 cold OOM 属预期，见 run-snsc.sh Step 4 说明）。

## 关键约定

- **显存档**：`--vram 16`（默认）对应服务器 16GB 卡；`--vram 32` / `VRAM=32`
  为预留 32GB 档（暂不启用）。16G 档大格子：8x32x32x32（cold 13.3G / warm 6.8G
  全流程可行）、16x32x32x32（cold 26.5G 超档，warm 13.5G 需外部缓存）。
- **nullvec 缓存**：共享 `logs/nullvec_cache`（`PYQCU_NULLVEC_CACHE` 可覆盖），
  跨 tag 复用粗算子，避免重复构建（8x32x32x32 首次构建 ~1-2h，缓存命中后秒级）。
- **C++ 收敛日志**：C++ 端写死 `REPO/logs/clover_multigrid.log`（相对 REPO），
  main.py 从同路径解析（parse_mg_log）；运行脚本结束时归档副本到本目录。
- **CudaSchurOp 同步**：`matvec` 内 `torch.cuda.synchronize()`（test11 BUGFIX，
  修复 C++ 私有流异步 + 读取竞态 —— dev74 遗留的非确定结果问题）。
- **本地小卡**：RTX 4060 上 MG 恒慢（speedup<1）是硬件特性，勿用本地结果
  推断服务器；参数相对行为（3L>2L、r20>r10）跨 GPU 一致。

## 产物对接

`collect` 汇总 → `test11_results.json`；`mktable`/`plots` 读它生成 LaTeX 表与
PNG；`plots1` 读 `test11_sweep.json`（作图范围与 dev73_5 一致：收敛历史/热点/
加速比/耗时/参数扫描，亮色调色板）。报告引用本目录产物即可。
