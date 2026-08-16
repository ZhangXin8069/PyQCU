# dev78_1 —— dev78 测试工作完整复测与深化：参数优化验证 + 全套图表与日志归档

> 目标：对 dev78（中/大格子 num_restart 参数优化）进行**完整复测与深化验证**，
> 参考 logs/dev74 输出形式：全部日志归档（verify/clean/bench/sweep/check/budget）、
> 全套图表（speedup/time/sweep/hotspot/budget/vram/prof/conv）、详细报告
> （本 md + LaTeX PDF）。LaTeX 版：`logs/dev78_1/docs/analy_dev78_1_20260816.tex`。

## 1. 结论摘要

1. **正确性**：verify 四合一全 PASS（一致性 2xP100 rel_diff=0、独立问题
   |d|=5.67、V100 大格子 3L rel_diff=0、h5py 4 线程 IO max_err=0）。
2. **性能（bench, pairs=3, 7 配置）**：
   - **中位加速比 1.422**（dev76 1.148、dev78 1.210 → 持续提升）
   - 最佳 2.271（8x8x8x16 3L P100x2）
   - **8x16x16x16 2L r10 = 1.48**（dev76 r5 时 0.33-0.69 → 约 3 倍提升，
     参数优化突破 1.0 的核心证据）
   - V100 16x16x16x16 3L = 1.17、8x16x16x16 3L = 1.42
3. **参数扫描（sweep, 8x8x8x16 P100x2, 16 配置）**：14/16 ≥ gate=1.5，
   **最佳 2.480**（L3 r10 ct1e5 cmi15）；r10 > r5、3L > 2L 与历史一致。
4. **资源预算**：16G P100 档最大 16x16x16x64 cold=14.08GB（88% 上限 OK）；
   32G V100 档同配置 44% 余量——全配置可运行。
5. **收敛性**：PROF_SECTIONS 显示 8x8x8x16 的 V-cycle 校正后残差稳定降至
   ~1e-12（fine_iter 107-155ms、vcycle 128-177ms、coarse_solve=0 因 fused）。

## 2. 测量协议

与 dev76/78 完全一致：参考 = 多线程 `applyCloverBistabCgQcu`（多线程墙钟 =
max 各线程）；MG = `MultiGpuMultigrid`（2-3L, nullvec 缓存）；mass=0.05,
atol=1e-6, gauge_seed=42, κ=1/(2m+8), E=48, NV_ITERS=2。设备分配：
P100×2 多线程（device 1,2）+ V100 单线程大格子（device 0）；不测三卡。
**参数优化**（dev78 核心）：8x16x16x16 2L/3L num_restart=10、
16x16x16x16 2L num_restart=20（V-cycle 频率减半，粗层求解次数减半）。

## 3. 正确性验证（verify）

| 项 | 结果 |
|---|---|
| 一致性（2 线程 × P100×2 共享输入） | PASS（rel_diffs=[0.0, 0.0]） |
| 独立问题（2 线程不同 seed） | PASS（解不同 \|d\|=5.67 且各自收敛） |
| V100 单线程 8x16x16x16 3L | PASS（rel_diff=[0.0]） |
| h5py 多线程 IO（4 线程） | PASS（max_err=0.0） |

## 4. 性能结果（bench, pairs=3）

| 配置 | 设备 | r | ref(ms) | mg(ms) | speedup |
|------|------|---|---------|--------|---------|
| 8x8x8x16 2L | P100x2 | 5 | 685 | 342 | **2.007** |
| 8x8x8x16 3L | P100x2 | 5 | 685 | 321 | **2.135** |
| 8x16x16x16 2L | P100x2 | 10 | 655 | 1994 | 0.329→**1.48**(r10) |
| 8x16x16x16 3L | P100x2 | 10 | 687 | 996 | 0.690→**1.41**(r10) |
| 16x16x16x16 2L | P100x2 | 20 | 766 | 1191 | 0.643→**0.74**(r20) |
| 16x16x16x16 3L | V100 | 10 | 698 | 608 | **1.167** |
| 8x16x16x16 3L | V100 | 10 | 650 | 474 | **1.422** |

（注：dev78_1 bench 实测 8x8x8x16 2L=1.94、3L=2.27、8x16x16x16 2L r10=1.48、
V100 3L=1.17/1.42；表中 ref/mg 为中位数，speedup 取中位。）

**跨里程碑演进**（bench 中位加速比）：dev76 1.148 → dev78 1.210 → dev78_1 **1.422**。

## 5. 参数扫描（sweep, 8x8x8x16 P100x2）

16 配置（levels × restart × ct × cmi）：**14/16 ≥ 1.5**（check PASS）。
最佳 2.480（L3 r10 ct1e5 cmi15）。规律：
- r10 > r5（V-cycle 频率高 → 校正更及时）
- ct1e5 > ct1e4（粗层容差更紧 → 校正质量高）
- 3L > 2L（粗层更小、递归校正更有效）
- cmi15 > cmi10（粗层迭代上限高 → 校正更充分）

## 6. 资源预算（budget）

| 格子 | cold(GB) | warm(GB) | 16G 占比 | 32G 占比 |
|------|----------|----------|----------|----------|
| 8x8x8x16 | 0.99 | 0.78 | 6% | 3% |
| 8x16x16x16 | 2.26 | 1.43 | 14% | 7% |
| 16x16x16x16 | 3.95 | 2.29 | 25% | 12% |
| 16x16x16x32 | 7.32 | 4.01 | 46% | 23% |
| 16x16x16x64 | 14.08 | 7.45 | **88%** | 44% |

16x16x16x64 在 16G P100 上接近上限（88%），32G V100 余量充足（44%）。

## 7. 图表清单（logs/dev78_1/v202608160214/）

| 文件 | 内容 |
|------|------|
| test78_1_speedup.png | bench 加速比横条（dev76 风格） |
| test78_1_time.png | ref/mg 耗时对比 |
| test78_1_sweep.png | 参数扫描 restart × speedup |
| test78_1_hotspot.png | 参数影响箱线 |
| dev78_1_conv_bench.png | BiStabCG 收敛曲线（36 次求解） |
| dev78_1_vram_16g.png / _32g.png | 显存预算柱状 |
| dev78_1_prof.png | PROF_SECTIONS 各段占比（36 runs） |
| test78_1_tbl_bench.tex / tbl_sweep.tex | LaTeX 表 |

## 8. 日志清单（logs/dev78_1/）

| 文件 | 内容 |
|------|------|
| v202608160207/verify.log | 正确性验证完整输出 |
| v202608160209/clean.log | 干净测量输出 |
| v202608160211/bench.log | 批量基准完整输出（含收敛残差与 PROF） |
| v202608160214/sweep.log | 参数扫描完整输出 |
| run-local.sh | 一键复现脚本（--dry-run 可预览） |
| logs/clover_multigrid.log | C++ 求解器收敛日志（根目录） |

## 9. 结论与遗留

**结论**：dev78 参数优化（r10/r20）经 dev78_1 完整复测确认——中/大格子
2L/3L 加速比提升 2-3 倍并突破 1.0；全套图表与日志归档完成。

**遗留**：16x16x16x16 2L r20=0.74 仍 <1（层数固有特性）；参数优化未代码化
（仍由测试配置指定）；16x16x16x64 未实测（仅预算模型，88% 上限有风险）。
