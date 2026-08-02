# PyQCU CUDA-C++ Clover MultiGrid — V4 调试与优化报告

**日期**: 2026-08-02
**目标**: 使 `applyCloverMultigridQcu` 在默认格子 `{8,16,16,16}` 上加速比 ≥ 1.2，
功能与 `applyCloverBistabCgDslashQcu` 一致，多层子迭代，算法对齐
`pyqcu/solver/_multigrid.py`，附 BiStabCG 对照、null_vecs 正确性检查与缓存。

---

## 1. 根因定位

在 V100（sm_70）上独立微基准测得：
- 平凡内核启动：3.5 µs
- **内核 + cudaStreamSynchronize：177 µs**（WSL2 虚拟化开销）
- 8 字节 D2H + sync：157 µs

参考 BiStabCG 每迭代 ~20-30 次同步 → ~6 ms/迭代（计算 <5%）。
MultiGrid 细层继承同一结构；粗层每迭代 ~14 次启动 + 1 次同步 → ~30 ms/次 V-cycle。
原基准 MG 用 VERBOSE=1（每次迭代写日志 ~1ms），参考用 VERBOSE=0，不公平。

## 2. 修复的 Bug

| # | Bug | 位置 | 修复 |
|---|-----|------|------|
| 1 | Cython 扩展过期，`applyCloverMultigridQcu` 缺失 | `pyqcu/cuda/qcu/*.so` | `bash install.sh` 重编译 |
| 2 | `_WILSON_AND_LAPLACIAN_TEST_SINGLE_IN_MULTI_=1` 强制单进程走 MPI halo | `include/define.h` | 改为 0 |
| 3 | 基准 MG 用 VERBOSE=1 计时（日志开销 ~1ms/迭代） | `conftest.schur.multigrid.py` | 改为 VERBOSE=0 |
| 4 | 单 block 融合粗求解器慢且 NaN | `multigrid.cu` | 多 block cooperative-groups 版本 |

## 3. 优化措施（同步极简化）

1. **单进程 dslash 快速路径**（`run_mpi`）：grid=1 时 send/inside/recv 全部放主
   stream，去掉 ~9 次中间同步。Schur dslash 4.5 ms → 0.7 ms。
2. **Clover give 去冗余同步**：单进程时去掉前后各一次同步。
3. **同步极简细层 BiStabCG**（`bistabcg_iter_fine_fast`）：点积直接写 device_vals
   （cublasH 绑定主 stream），每迭代只同步一次。细迭代 ~6 ms → ~2.6 ms。
4. **Cooperative-groups 并行融合粗层求解**（`multigrid_coarse_solve_cg`）：整个
   BiStabCG 求解融合进一个 kernel，grid.sync() 跨 block 归约。粗求解 ~30 ms → ~10 ms。
5. **参数扫描**：ct=1e5（粗容差 0.1）、maxiter=15、r=10~12。

## 4. 性能结果

| 格子 | 配置 | BiStabCG | MG | 加速比 |
|------|------|----------|-----|--------|
| 8×8×8×16 | 2L E=48 r=10 ct=1e5 | 478 ms | 228 ms | 2.10× |
| 8×16×16×16（默认） | 2L E=48 r=12 ct=1e5 | 498 ms（中位数） | 395 ms（中位数） | **1.26×** |
| 8×16×16×16（默认） | 2L E=48 r=10 ct=1e5 | 549 ms | 394 ms | 1.40× |
| 8×16×16×16（默认） | 3L E=[12,48,48] r=10 ct=1e5 | 474 ms | 446 ms | 1.06× |
| 8×8×8×16 | 2L E=48 r=10 ct=1e5 **c128** | 619 ms | 323 ms | 1.92× |

`vs_ref ≈ 3×10⁻⁷`（c64）/ `6.6×10⁻⁸`（c128），解与 BiStabCG 一致。加速比主要
来自细层单迭代成本（~2.6--3.4 ms vs 参考 ~5.8 ms，~0.6×）与更少的迭代次数
（8×8×8×16 上 65 vs 86）。默认格子 2 层中位数加速比 1.26× ≥ 1.2；3 层配置正确但
在该格子规模上粗层开销大于收益。默认格子 MG 时间范围 370--469 ms（GPU 时钟波动）。

## 5. null_vecs 缓存与正确性

- 缓存：`examples/qcu/mg_nullvec_cache.py`，默认开启，按
  (lattice, level, E, nv_iters) 存 `logs/nullvec_cache/*.pt`。
- 验证：`examples/qcu/mg_v4_verify_nv.py` —— null 向量质量、块内正交、
  C++ restrict/prolong/粗 dslash 与 Python 参考一致（~10⁻⁷）。

## 6. 剖析

`nvprof/ncu/nsys` 在本机（混合 sm_60/sm_70 GPU + WSL2）均不可用
（nsys: No GPU associated to the given UUID；ncu: Unknown Error）。
改用 C++ 段计时（PROF_SECTIONS/PROF_COARSE）+ 独立 CUDA 微基准完成热点分析。

## 7. 文件修改

```
cpp/cuda/qcu/include/define.h                     — TEST_SINGLE_IN_MULTI=0
cpp/cuda/qcu/include/lattice_wilson_dslash.h      — run_mpi 单进程快速路径 + skip_final_sync
cpp/cuda/qcu/include/lattice_clover_dslash.h      — give 单进程去同步
cpp/cuda/qcu/include/lattice_clover_multigrid.h   — 快速细迭代 + 融合粗求解 + 段计时
cpp/cuda/qcu/include/multigrid.h                  — cooperative kernel 声明
cpp/cuda/qcu/src/multigrid.cu                     — multigrid_coarse_solve_cg 实现
examples/qcu/mg_nullvec_cache.py                  — null_vecs 缓存（默认开启）
examples/qcu/mg_v4_bench.py                       — 最终基准
examples/qcu/mg_v4_verify_nv.py                   — null_vecs 正确性验证
examples/qcu/mg_v4_sweep.py                       — 参数扫描
```
