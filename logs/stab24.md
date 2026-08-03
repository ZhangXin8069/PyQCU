# stab24 —— PyQCU CUDA-C++ Clover MultiGrid 系列开发报告（git tag dev73 → dev73_5）

> 对象：`git tag dev73*` 全部六个版本标签
> 数据来源：`logs/`（开发/调试/优化/结果报告）与 `examples/qcu/`（测试/基准/剖析脚本）
> LaTeX 版：`logs/stab24.tex` → `logs/stab24.pdf`（21 页，16 张数据图表，5 张 LaTeX 表格）

---

## 摘要

CUDA-C++ Clover MultiGrid 求解器（`applyCloverMultigridQcu`）从"正确但无性能优势"
到**最高 2.43× 加速**的完整演进。正确性主线：审查修 bug（dev73）→ 全算子一致化
（dev73_2）→ Schur 一致粗空间（dev73_3）→ 公平计时 + 缓存验证（dev73_4）。性能主线：
同步开销根因定位（dev73_4，WSL2 上 `cudaStreamSynchronize` ~177 µs/次）→ 5 项同步极简
优化 → 干净测量协议 + 全参数扩展（dev73_5）。全系列解与参考 BiStabCG 一致
（`vs_ref~3×10⁻⁷` c64 / `4×10⁻⁸` c128）。

## 版本演进一览

| tag | 提交 | 日期 | 主题 | 净变更（vs 前一 tag） |
|-----|------|------|------|----------------------|
| `dev73` | bd81d31 | 07-28 | 首个完整 C++ CUDA Clover MG（953 行），3 轮审查修 33 bug | — |
| `dev73_1` | cfbd753 | 08-01 | 全库 CLAUDE.md 文档体系（无功能改动） | 43 文件 +1247 |
| `dev73_2` | d296612 | 08-02 | MG 系统调试：10 个根因 bug + 全算子 level-0 | 45 文件 +5491/−451 |
| `dev73_3` | a1b9a95 | 08-02 | Schur 一致粗空间 + 33 张量粗算子 A_c=P^T S P | 40 文件 +8917/−367 |
| `dev73_4` | a724357 | 08-02 | 同步开销根因 + 5 项同步极简优化 | 18 文件 +1430/−35 |
| `dev73_5` | 1955795 | 08-03 | 干净测量协议 + 精度/格子/参数扩展扫描 | 24 文件 +2012 |

累计 dev73 → dev73_5：**150 文件，+18,790 / −550 行**。硬件在 dev73_4 由
RTX 4060 Laptop（SM 8.9）切换为 Tesla V100-SXM2-32GB（SM 7.0）。

## 计算结果分析（正确性）

- **Gauge SU(3)**：`check_su3` 全部通过；幺正性 `max|U†U−I|`~3×10⁻⁷（c64）/~10⁻¹⁶（c128）。
- **解误差**：`vs_ref` 从 dev73 的 6.23×10⁻⁷ 到 dev73_5 的 ~3.2×10⁻⁷（c64）；
  dev73_2 全配置 `vs_ref∈[10⁻¹¹,10⁻⁶]`；c128 提高两个数量级（~4×10⁻⁸）。
- **null_vecs 四重检查**：零模质量 ~10⁻¹~10⁰（捕获 S 低模）、块内正交 ~3.6×10⁻⁷
  （c64）/4.4×10⁻¹⁶（c128）、C++ restrict/prolong 与 Python 一致 ~10⁻⁷、
  33 张量粗 dslash 与 A_c 一致 ~10⁻⁷。
- **迭代次数**：dev73 ~104；dev73_2 190–434（全算子更慢）；dev73_3 63–68（**减 30%**）；
  dev73_4 65–90；dev73_5 64–119（与参考相当或更少）。

## 计算性能分析

### 性能演进（关键数字）

| 版本 | 硬件 | 格子 | 配置 | 加速比 | MG/ref(ms) |
|------|------|------|------|--------|-----------|
| dev73 | RTX 4060 | 8³×16 | 1L | 0.87–1.28× | 1742–3636 |
| dev73_2 | RTX 4060 | 8³×16 | 2L | 0.21× | 648/135 |
| dev73_3 | RTX 4060 | 8×16³ | 2L | 0.170× | 1137/194 |
| dev73_4 | V100 | 8×16³（默认） | 2L r=12 | **1.26×** | 395/498 |
| dev73_4 | V100 | 8³×16 | 2L r=10 | **2.10×** | 228/478 |
| dev73_5 | V100 | 8×16³（默认） | 2L r=10 | 1.16× | 399/464 |
| dev73_5 | V100 | 8³×16 | 2L r=10 c64 | **2.43×** | 192/465 |

### 根因与 5 项优化（dev73_4）

WSL2 虚拟化使 `cudaStreamSynchronize` 达 ~177 µs/次；参考 BiStabCG 每迭代 20–30 次同步
→ ~6 ms/迭代（计算 <5%）。5 项同步极简优化：

1. 单进程 dslash 快速路径（去掉 ~9 次中间同步）：Schur dslash 4.5→0.7 ms；
2. Clover give 去冗余同步；
3. 同步极简细层 BiStabCG（每迭代 1 次同步）：细迭代 ~6→2.6 ms；
4. cooperative-groups 融合粗解（整个粗 BiStabCG 一个 kernel，grid.sync()）：
   粗解 ~30→10 ms/次；
5. 参数扫描：ct=10⁵（粗容差 0.1）、maxiter=15、r=10~12。

### 规模效应（dev73_5，干净 min）

加速比随格子增大而**下降且非单调**：
8×8×8×16 **2.43×** → 默认 {8,16,16,16} 1.16× → {8,16,16,32} 1.11× →
{16,16,16,16} 0.81×。大格子上粗层 V-cycle 修正开销增长（{16,16,16,16} 的 V-cycle
383 ms 占 59%）且未显著降迭代（108 vs 90）。MG 优势窗口 = 细层迭代成本高于粗层修正
成本的配置（小格子、单精度）。

### 精度 / 参数扫描（dev73_5）

- **c128**：解精度 +2 个数量级，但默认格子上粗层双精度求解开销主导（0.73×）；小格子
  上达标（1.97×）。
- **V-cycle 频率 r**：r=5 过频繁（0.87×）、r=10~12 最优（1.16–1.18×）、r=20 1.26×。
- **最粗层容差 ct**：10⁴~10⁵ 最优（1.12–1.16×），与 Python 参考 0.1·‖r‖ 一致。
- **最粗层迭代 cmi**：15 即可（1.16×）。
- **层数**：默认格子上 3L 1.32× 略优于 2L 1.16×；v4 连续负载下 3L 1.06×。

## 对照讨论（以 dev73_5 为范例）

- **正确性贯穿始终**：dev73 6.23×10⁻⁷ → dev73_5 3.2×10⁻⁷；正确性是性能优化的前提，
  dev73_2 的"性能让位"换来数值彻底正确。
- **迭代优势 ≠ 墙钟优势**：dev73_3 迭代减 30% 但被 RTX 4060 启动延迟（210 MHz 空闲）
  抵消；dev73_4 在 V100 消除同步开销后，同一算法结构立即转为 1.26–2.43×。
  **瓶颈是执行开销而非算法**。
- **测量协议决定结论**：dev73_5 干净协议（独立进程 + 交叉计时 + min of 5 对）排除
  GPU 时钟噪声（绝对耗时 ±10% 波动）；**迭代数与解精度是稳健指标**。
- **与 v4 一致性**：8×8×8×16 2.43×（v4 2.07–2.10×）、默认格子 MG 399 ms（v4 394 ms）、
  细层迭代 89 vs 86（v4 88 vs 86）、`vs_ref` 一致。

## 从 dev73 到 dev73_5：改动、原由与最终效果

| 阶段 | 改动 | 原由 | 最终效果 |
|------|------|------|----------|
| dev73 | 首个 C++ Clover MG + 3 轮审查修 33 bug + 12 项优化 | 正确性是一切前提；复数乘法/越界写等缺陷会静默污染结果 | 正确性 6.23×10⁻⁷，测试 13/13；性能无优势（0.87–1.28×） |
| dev73_1 | 全库 CLAUDE.md 文档体系 | 多目录工程需可维护的文档约定 | 无功能改动 |
| dev73_2 | 修 10 个根因 bug；level-0 改全算子；受保护修正 | V-cycle 残差爆炸/3 层 NaN/coarse dslash 差 97%；对齐 Python 参考 | 全配置数值正确（vs_ref 10⁻¹¹–10⁻⁶）；性能退回 0.15–0.41× |
| dev73_3 | Schur 一致粗空间 + 33 张量 stencil | 全算子 D 零模与 Schur S 低模不匹配，V-cycle 失效 | 迭代 ~90→63–68（−30%），首次满足"迭代更少"；4060 启动延迟致墙钟仍慢 |
| dev73_4 | 同步根因定位 + 5 项同步极简优化 + null_vecs 缓存 | WSL2 同步 ~177 µs/次，MG 迭代优势被吞掉；测量不公平 | 默认格子 1.26×、8³×16 2.07×（c128 1.92×），**关键一跃** |
| dev73_5 | 干净测量协议 + 精度/格子/参数扫描 + 脚本族 | 绝对耗时对 GPU 时钟敏感，需稳健结论 | 8³×16 复现 2.43×/1.97×；揭示规模效应（2.43→0.81×） |

## 结论

1. **正确性**：六版本全部与参考解一致；gauge SU(3)、null_vecs 四重检查通过。
2. **性能**：0.87–1.28× → 2.43×（小格子 c64）/1.16×（默认格子）。关键在 dev73_4
   消除同步开销；dev73_3 的 Schur 一致粗空间（迭代 −30%）是必要前提。
3. **规模效应**：加速比随格子增大下降且非单调；c128 默认格子 <1。最优窗口 = 小格子、
   单精度、细层迭代成本主导。
4. **工程**：干净测量 + 脚本族/JSON/LaTeX 表格流水线保证结论稳健、数字可复现。

## 数据与图表清单

**图表（16 张，全部内嵌于 stab24.pdf）**：dev73（`multigrid_result.png`、
`multigrid_performance.png`）、dev73_2（`mg_convergence.png`、
`multigrid_result_L1/L2.png`）、dev73_3（`schur_mg_convergence.png`）、
dev73_5（`dev73_5_conv_*.png`×6、`dev73_5_hotspot/speedup/time/sweep.png`）。

**LaTeX 表格（5 张 \input）**：`dev73_5_tbl_{main,prec,lattice,sweep,verify}.tex`。

**数据文件**：`multigrid_report.json`、`mg_dev_results.json`、`schur_mg_results.json`、
`dev73_5_results.json`、`dev73_5_clean_*.json`、`dev73_5_bench.json`、
`dev73_5_verify_*.json`。

**原始日志**：`clover_multigrid.log`、`mg_bench_out.txt`、`mg_iter_sweep_out.txt`、
`mg_param_sweep_out.txt`、`schur_mg_test_stdout.txt`、`pyref_target_*.log`。

**脚本（examples/qcu，约 55 个）**：测试（`conftest.clover.multigrid.py`、
`conftest.schur.multigrid.py`、`quick_test.py`、`verify_coarse.py`、`diag_coarse.py`）、
dev73_2 调试（`mg_dev_*`、`v3_*`）、dev73_3 概念/诊断（`mg_schur_concept.py`、
`mg_pyref_expt.py`、`mg_strategy.py`、`mg_stencil_build.py`、`mg_iter_sweep.py`、
`mg_param_sweep.py`、`mg_micro_bench.py`、`mg_verify_level2.py`）、dev73_4
（`mg_nullvec_cache.py`、`mg_v4_{bench,sweep,verify_nv,diag1,build1616,verbose_log}.py`）、
dev73_5 测量族（`mg_dev73_5_{clean,bench,verify,collet,mktable,plots}.py`）。

---

*报告生成：2026-08-03 | 从 dev73（2026-07-28）到 dev73_5（2026-08-03）*
*LaTeX 编译：`logs/stab24.tex` → `logs/stab24.pdf`（21 页）*
