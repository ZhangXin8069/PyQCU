# PyQCU dev99 至 1640349d 进度总结

状态：`finished`

## 总结范围

范围为上一标签 `dev99`（peeled `1a3e31c3`）到 `HEAD 1640349d`，共 3 个提交：

| 提交 | 日期 | 主要内容 |
|---|---|---|
| `57f40fa1` | 2026-10-21 02:39 | KUPL/CUDA 多卡优化、LIME benchmark 与 Slurm 工具 |
| `3d0ce433` | 2026-10-21 05:11 | KUPL/CUDA MG 优化、QCU 残差门、QUDA MG 接口、参考库整理 |
| `1640349d` | 2026-10-21 05:55 | 完整 MG cache 构建、QCU 独立真残差修正及最终验证 |

本文档是 `dev99..1640349d` 的静态进度归档，只复述已有代码、日志、JSON 报告和 Slurm 状态。归档时不修改代码，也不补跑测试。

## 改动内容

### KUPL 与 CUDA 多卡

- KUPL 删除 OpenMP 线程数强制向下取 2 的幂的逻辑，显式请求的线程数现在精确生效，例如 24 核不再被压缩到 16。
- 新增 `kupl_qcu_explicit_threads` CTest，验证请求 3 线程时实际得到 3。
- CUDA `MultiGpuMultigrid` 新增 `run_reference`。默认保留 BiStabCG 参考解用于正确性验证；纯 MG 吞吐模式可关闭冗余参考求解和第二次 `LatticeSet` 初始化。
- worker 的 `applyEndQcu` 移入 `finally`，初始化失败时也能正确释放槽位。
- `bench_multi_gpu.py` 增加 warmup/repeat、median、`--no-reference`、外部 LIME/HDF5 规范场、固定源、独立残差验证和粗网格 cache 参数。
- 新增 `slurm_multi_gpu_smoke.sh`，覆盖 ABI、Wilson dslash、Clover BiStabCG、双卡一致性和 LIME 回归。

### KUPL/CUDA 多重网格

- KUPL 将 onsite Wilson-Clover 矩阵识别为两个 6x6 spin block，替换通用 12x12 路径，并用于 full operator、Schur prepare/reconstruct 和 Clover inverse。
- CUDA `build_schur_levels` 增加局部窗口 stencil 构建、外部规范场/源、cache 命中日志、构建进度输出和残差验证。
- 新增 `BatchedLocalWilsonClover`，支持 full Wilson-Clover 算子的局部批量探测。
- QCU benchmark 增加 stage 日志、MPI device-aware 记录、Clover setup 和 solve 分阶段计时。
- 新增 QCU 全格真残差验证路径。
- QUDA benchmark 增加 `--solver mg`、`--levels`、`--nvec` 和 `--mg-block` 接口。

### 完整 MG cache 与残差门

- 新增 `build_full_mg_cache.py`，可从 CPU 或 CUDA 构建完整 33-tensor Wilson-Clover MG cache，并写出包含 SHA256、格子尺寸、mass、csw、dof、MG grid、setup 时间和大小的 JSON 报告。
- 新增 `write_coarse_assets()`，统一写出 `lonv/hnn/hdg/sit` 四个 HDF5 dataset。
- 新增 `test_coarse_cache.py`，覆盖完整 cache 往返和缺失 dataset 拒绝。
- QCU 独立真残差改用求解器实际使用的 even/odd Clover 项，并通过 `cut_I` 去掉 `A=I+T` 中的单位项。
- `_gather_local_blocks()` 和 `_independent_true_residual()` 支持显式 communicator，最终残差由 root broadcast 给所有 MPI rank。
- 修正多 rank 下所有 rank 必须参与 `tools.norm` MPI 归约的语义。

### 参考库整理

- `refer/git-rep` 中 4 个 vendored 参考库改为固定上游提交的 submodule。
- 34 份 PyQCU 自产生报告迁移到 `refer/git-rep/docs/`。
- 更新引用路径、只读边界和 shadow patch 规则。

## 完整测试结果

### 非 Slurm 验证

| 检查 | 结果 |
|---|---|
| KUPL Release CTest | `28/28 PASS`，0 failed，26.42 s |
| KUPL 显式线程回归 | 请求 3 线程，修复前实际 2 线程 FAIL，修复后 3 线程 PASS |
| Python `py_compile` | PASS |
| `bash -n slurm_multi_gpu_smoke.sh` | PASS |
| `git diff --check` | PASS |
| `bash skills/pyqcu-form/form-audit.sh` | PASS，33 个技能目录一致，既有资产豁免保持 |
| 参考库结构检查 | `top_level_gitlinks=5 reports=34 report_pairs=complete upstream_worktrees=clean` |
| TeX/PDF | 未验证，环境缺少 `xelatex/latexmk`，3 份修改交叉引用的 PDF 未重生成 |

新增或修改的 Python 单测没有找到对应的持久化 pytest 输出，因此不能宣称已经通过：

- `pyqcu/testing/simple/test_coarse_cache.py`
- `pyqcu/testing/qcu/test_benchmark_qcu_lime_mpi.py`
- `pyqcu/testing/kupl/test_simple_mg_geometry.py`
- `pyqcu/testing/qcu/strict/quda_comparison/test_stencil_batch_local.py`
- `pyqcu/testing/quda/test_benchmark_quda_lime_mpi.py`

### 全部 Slurm 作业结果

以下覆盖 `dev99..1640349d` 中找到的全部 37 个作业，包括失败和取消项。

| 作业 | Slurm 状态 | 结果与原因 |
|---|---|---|
| `1582499` | FAILED `1:0` | Slurm 脚本路径错误，尝试执行不存在的 `/var/env.sh`，测试未启动 |
| `1582500` | FAILED `1:0` | ABI 检查 PASS；Wilson dslash 因扩展未重建，缺少 `applyWilsonDslashQcu` 而失败 |
| `1582504` | FAILED `2:0` | 多卡 benchmark 已生成 JSON，但后续 LIME 调用错误传入不支持的 `--dtype c64` |
| `1582511` | COMPLETED `0:0` | ABI、Wilson、Clover BiStabCG、双卡一致性和真实 LIME 全链 PASS |
| `1582514` | FAILED `1:0` | 参考解 JSON 已生成；`--no-reference` 路径对 `None` 调用 `round()` 失败 |
| `1582525` | COMPLETED `0:0` | 双 A100 参考解和 no-reference benchmark 均 PASS |
| `1582543` | COMPLETED `0:0` | 配对 benchmark PASS；LIME 回归被跳过 |
| `1582546` | COMPLETED `0:0` | 重复配对 benchmark PASS；LIME 回归被跳过 |
| `1582641` | CANCELLED | QCU 和 QUDA 单 rank smoke PASS；未完成完整矩阵 |
| `1582692` | FAILED `1:0` | QCU 单 rank PASS；后续在 `prepare_global_rhs` 阶段失败 |
| `1582694` | FAILED `1:0` | QCU 单 rank PASS；后续在 `construct_clover_bicgstab` 阶段失败 |
| `1582697` | COMPLETED `0:0` | QUDA c64 单 rank BiCGStab PASS，19 次迭代，真残差 `4.521737e-7` |
| `1582698` | CANCELLED | QCU 单 rank PASS；双 rank 运行约 20 分钟后无报告并取消 |
| `1582702` | FAILED `124:0` | QCU 双 rank 超时 |
| `1582706` | FAILED `124:0` | QCU 双 rank 超时 |
| `1582707` | FAILED `124:0` | 加入 stage 日志后双 rank solve 仍超时 |
| `1582708` | FAILED `124:0` | 10 次迭代限制下仍超时 |
| `1582709` | FAILED `124:0` | debug 作业超时 |
| `1582710` | FAILED `124:0` | wrapper/alloc/prepare 阶段正常进入，随后超时 |
| `1582711` | CANCELLED | 已进入 prepare 和 dslash，主动取消 |
| `1582712` | FAILED `1:0` | host-staged MPI 跑通，10 次迭代；独立真残差 `0.745531`，FAIL |
| `1582713` | FAILED `1:0` | 31 次迭代原生收敛，但独立真残差仍为 `0.745520`，确认不是单纯收敛不足 |
| `1582715` | FAILED `1:0` | 单 rank 同样得到 `0.745520`，将问题定位到残差公式，而非 MPI |
| `1582716` | FAILED `2:0` | benchmark 不接受当时传入的 `--dof` 和 `--mg-grid` |
| `1582718` | CANCELLED | 运行约 31 分钟后在 MG 粗算子构建期间取消 |
| `1582721` | FAILED `1:0` | QUDA MG `Unsupported nVec 12` |
| `1582723` | CANCELLED | QUDA MG 构建约 35 分钟后仍未完成 |
| `1582726` | CANCELLED | KUPL MG cache 构建约 36 分钟后仍未完成 |
| `1582732` | FAILED `1:0` | cache 构建和 20 次 MG 迭代成功；后续残差验证因 CUDA/CPU tensor 混用失败 |
| `1582735` | COMPLETED `0:0` | QCU 单卡 cache-hit MG PASS，独立真残差 `4.125240e-7` |
| `1582739` | COMPLETED `0:0` | 单卡 MG 复验 PASS，独立真残差 `4.125240e-7` |
| `1582740` | COMPLETED `0:0` | 双 A100 双线程 MG PASS，两个线程残差均为 `4.125240e-7` |
| `1582742` | FAILED `1:0` | PyQUDA 扩展报 `undefined symbol: dslashQcu`，且未暴露 `QudaMultigridParam` |
| `1582746` | FAILED `1:0` | QUDA MG GCR breakdown |
| `1582750` | FAILED `1:0` | 完整 cache 首次构建因 `make_clover` 的 CUDA/CPU device 不一致失败 |
| `1582751` | COMPLETED `0:0` | 完整 33-tensor MG cache 构建成功 |
| `1582754` | CANCELLED | c64 单 rank 真残差 PASS；双 rank 因 `visible=1`、local rank 1 无私有 GPU 而失败 |

多数作业同时出现非致命的 `No module named 'tilelang'` 和 OpenMPI OpenIB 警告，这两项没有直接导致上述失败。

### 关键正确性结果

| 项目 | 结果 |
|---|---|
| A100 Wilson dslash 相对误差 | `6.125111440269393e-8` |
| A100 Clover BiStabCG 真残差 | `2.6243412776238983e-7` |
| 双 A100 MG 一致性 | `all_pass=true`，两个 device 的 `rel_max_diff=0` |
| QCU 真实 LIME c64，旧独立残差路径 | 31 次迭代，`0.171282 s`，真残差 `2.428980e-7` |
| QCU 真实 LIME c64，formal | 31 次迭代，`0.153413 s`，真残差 `2.417295e-7` |
| QCU 最终残差门 | 31 次迭代，`0.153048 s`，真残差 `2.377387e-7`，PASS |
| QUDA 真实 LIME c64 smoke | 19 次迭代，solver `4.419682 s`，真残差 `4.521737e-7` |
| QUDA 真实 LIME c64 formal gate | 19 次迭代，solver `0.140328 s`，真残差 `4.521737e-7` |

QCU 中 `independent_true_residual_verified=false` 的早期 smoke 报告只验证了原生收敛，没有独立全算子残差，不能与最终残差门结果混为一谈。

## 性能测试结果

### KUPL 小格性能

格子为 `16x16x16x32`，每项为 `median ms / estimated GFLOP/s`。GFLOP/s 是操作数模型，不是硬件计数器。

| 配置 | c64 Wilson | c64 Clover | c64 Schur | c128 Wilson | c128 Clover | c128 Schur |
|---|---:|---:|---:|---:|---:|---:|
| 8 threads | 242.823 / 6.218 | 382.996 / 5.914 | 786.976 / 5.756 | 417.857 / 3.614 | 693.715 / 3.265 | 1414.873 / 3.202 |
| 16 threads | 137.268 / 11.000 | 260.523 / 8.694 | 554.278 / 8.173 | 228.177 / 6.617 | 430.035 / 5.267 | 903.012 / 5.016 |
| 请求 24，旧逻辑实际 16 | 136.834 / 11.035 | 266.615 / 8.495 | 544.734 / 8.316 | 227.700 / 6.631 | 432.051 / 5.242 | 940.253 / 4.818 |
| scalar 16 threads | 228.989 / 6.594 | 358.143 / 6.324 | 759.787 / 5.962 | 318.725 / 4.737 | 518.556 / 4.368 | 1100.391 / 4.117 |
| 精确 24 threads | 101.194 / 14.921 | 183.055 / 12.373 | 360.392 / 12.569 | 170.845 / 8.838 | 310.747 / 7.289 | 645.743 / 7.015 |

相对 16 threads，小格 c64 约提升 26% 至 35%，c128 约提升 25% 至 29%。

### KUPL 大格性能

格子为 `32x32x32x64`，每项为 `median ms / estimated GFLOP/s`。

| 配置 | c64 Wilson | c64 Clover | c64 Schur | c128 Wilson | c128 Clover | c128 Schur |
|---|---:|---:|---:|---:|---:|---:|
| 16 threads | 569.919 / 12.717 | 929.919 / 11.691 | 1955.753 / 11.118 | 1000.394 / 7.245 | 1801.575 / 6.035 | 3483.500 / 6.242 |
| 精确 24 threads | 447.039 / 16.213 | 750.267 / 14.490 | 1528.468 / 14.226 | 763.944 / 9.487 | 1348.367 / 8.063 | 2904.192 / 7.487 |

大格 Wilson 吞吐分别提升：c64 `27.4%`，c128 `31.0%`。

### KUPL 真实 LIME 矩阵

真实组态为 `24^3 x 64`，mass `-0.277`，csw `1.160920226`，每项为 5 次稳态运行的中位数。

| 文件配置 | Solve median | 迭代 | 独立真残差 | 状态 |
|---|---:|---:|---:|---|
| c64 r1 baseline | 9.858334 s | 20 | `4.125306e-7` | PASS |
| c64 r1 spin6 | 7.830318 s | 20 | `4.131546e-7` | PASS |
| c64 r1 spin6 + inv6 | 7.769662 s | 20 | `4.121851e-7` | PASS |
| c64 r2 baseline | 7.960636 s | 20 | `4.120797e-7` | PASS |
| c64 r2 spin6 | 7.452968 s | 20 | `4.128289e-7` | PASS |
| c64 r2 spin6 + grid-x | 8.108518 s | 20 | `4.124767e-7` | PASS |
| c64 r2 spin6 + inv6 | 6.351769 s | 20 | `4.126025e-7` | PASS |
| c128 r1 baseline | 20.196165 s | 23 | `6.962408e-8` | PASS |
| c128 r1 spin6 | 16.116355 s | 23 | `6.962408e-8` | PASS |
| c128 r1 spin6 + inv6 | 15.592787 s | 23 | `6.962408e-8` | PASS |
| c128 r2 baseline | 14.449964 s | 23 | `6.962408e-8` | PASS |
| c128 r2 spin6 | 13.106006 s | 23 | `6.962408e-8` | PASS |
| c128 r2 spin6 + inv6 | 11.472949 s | 23 | `6.962408e-8` | PASS |

最终 `inv6` 相对 baseline：c64 单 rank 从 `9.858334 s` 降至 `7.769662 s`，双 rank 从 `7.960636 s` 降至 `6.351769 s`；c128 单 rank 从 `20.196165 s` 降至 `15.592787 s`，双 rank 从 `14.449964 s` 降至 `11.472949 s`。

### CUDA 双 A100 纯 MG 配对

格子为 `8x8x8x16`，表中为 `total_solve_median`。

| 作业 | 参考解模式 | no-reference | 降幅 |
|---|---:|---:|---:|
| `1582525` | 0.267100 s | 0.209200 s | 21.7% |
| `1582543` | 0.255085 s | 0.232044 s | 9.0% |
| `1582546` | 0.258240 s | 0.219773 s | 14.9% |

三组降幅中位数为 `14.9%`。逐线程 MG 时间基本不变，收益来自取消冗余参考求解、第二次 `LatticeSet` 初始化和结果路径。

### CUDA QCU 单卡真实 LIME

| 报告 | Warmups / repeats | Solve median | 迭代 | estimated GFLOP/s | 独立真残差 |
|---|---:|---:|---:|---:|---:|
| triad smoke | 1 / 2 | 0.163096 s | 31 | 1162.347 | 未验证 |
| backend smoke | 0 / 1 | 0.154417 s | 31 | 1227.675 | 未验证 |
| dot 优化探测 | 1 / 5 | 0.159336 s | 31 | 1189.774 | 未验证 |
| formal | 1 / 5 | 0.153413 s | 31 | 1235.714 | `2.417295e-7` |
| 最终残差门 | 0 / 1 | 0.153048 s | 31 | 1238.660 | `2.377387e-7` |

最终残差门的 warmup 为 0、repeat 为 1，因此它适合作为正确性门，不是稳定的统计性能样本。

### QUDA 对照

| 配置 | Solver time | Wall time | 迭代 | 真残差 | estimated GFLOP/s |
|---|---:|---:|---:|---:|---:|
| c64 smoke | 4.419682 s | 6.728750 s | 19 | `4.521737e-7` | 80.433 |
| c64 formal gate | 0.140328 s | 0.455391 s | 19 | `4.521737e-7` | 80.433 |

QCU 与 QUDA 的 GFLOP/s 模型和计时边界不完全相同，不建议直接用于跨后端加速比结论。

### 完整 MG cache

| 作业 | 内容 | 结果 |
|---|---|---|
| `1582732` | 局部 stencil cache | stencil 构建 `365.599 s`，663552 probes；随后 20 次 MG solve `0.143280 s`，真残差 `4.325137e-7`；后续验证因 device 混用失败 |
| `1582751` | 完整 33-tensor cache | setup `316.347 s`；stencil `297.4 s`，`2231 probes/s` |
| `1582751` 产物 | `full_wilson_clover_L24x24x24x64_E12_nv1.h5` | `3,121,352,704 bytes`，SHA256 `b78c68f4...343f52d` |
| `1582735` / `1582739` | QCU 1 GPU cache-hit MG | MG `0.1436 s`，总 solve median `1.761294 s`，残差 `4.125240e-7` |
| `1582740` | QCU 2 GPU/2 threads cache-hit MG | 最慢线程 MG `0.1822 s`，总 solve median `1.999177 s`，两线程残差 `4.125240e-7` |

`1582732` 的 `_perf` 为 GPU util `92.31%`、gmem util `69.00%`、gmem BW util `71.77%`、平均功耗 `229.64 W`。`1582751` 为 GPU util `37.36%`、gmem util `43.33%`、gmem BW util `22.18%`、平均功耗 `145.43 W`。其他部分作业的 `_perf` 明确显示 GPU mapping 不可用，因此没有可报告的利用率数据。

## 计划但未完成

1. 双 rank QCU 独立真残差门尚未闭环。最新作业只申请了 1 张 GPU，但启动 2 个 rank，local rank 1 被 `visible=1` 拒绝。需要为每 rank 提供独立可见 GPU，或改用正确的多进程 GPU 设备协议。
2. 分布式 QCU 多 rank 稳定性尚未解决。多个双 rank 调试作业以 exit code 124 超时，较早作业曾出现 CUDA-aware MPI halo 卡住。
3. 完整后端矩阵尚未完成。设计目标包含 QCU/QUDA、c64/c128、1/2/4 rank，目前 QUDA 只有 c64 单 rank BiCGStab 正式通过。
4. QUDA MG 对照尚未完成。当前依次遇到 `Unsupported nVec 12`、`QudaMultigridParam` 未暴露、`undefined symbol: dslashQcu` 和 GCR breakdown。
5. 生产级 `16x32x32x48` 双 A100 cache-hit 吞吐和真残差门尚未运行。当前真实 LIME 性能数据为 `24^3 x 64`。
6. 完整 3.12 GB cache 尚未固化为标准正式 benchmark/Slurm 流程，也尚未完成基于现有 cache 的 cross-backend 复用验证。
7. 新增 Python 单测没有持久化执行记录，需要在后续任务运行并保存结果。
8. 最后一批 `1640349d` 没有 final 任务日志索引条目，也没有对应归档 tag；阶段结果尚未完成正式归档。
9. 参考库迁移中修改交叉引用的 TeX 尚未重新生成 PDF，因为环境缺少 `xelatex/latexmk`。

## 证据入口

- `logs/v2026102100.txt`
- `logs/v2026102105.txt`
- `logs/data/lime_qcu_residual_gate_20261021/qcu/c64/r1/report.json`
- `logs/data/lime_backend_formal_20261021/mg/qcu/c64/r1_nvec6_local/report.json`
- `logs/data/lime_backend_formal_20261021/mg/qcu/c64/r2_nvec6_local/report.json`
- `data/logs/lime_mg_cache_20261021/full_wilson_clover_L24x24x24x64_E12_nv1.json`
- `logs/AGENTS.md`

## 归档说明

本总结来自当时已有的提交、Slurm 日志、JSON 报告和任务最终日志。新增 Python 单测没有可追溯执行记录的部分已明确标为未验证，没有用未执行结果补齐通过状态。
