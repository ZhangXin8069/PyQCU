---
name: pyqcu-kupl
description: |
  当维护或验证 cpp/kupl/qcu、pyqcu.kupl、pyqcu/testing/kupl，处理鲲鹏/NUMA
  CPU 版 QCU，MPI+OpenMP Wilson/Clover dslash、Clover BiStabCG、csw、
  Wilson BiStabCG/CG、3D Laplacian、canonical flat ABI、MPI halo、
  rank/NUMA 绑定或 KUPL 性能基准时使用。
metadata:
  openclaw:
    emoji: 🧵
---

# pyqcu-kupl — 独立 MPI+OpenMP CPU 后端

## 执行前置

先读取工作目录 `AGENTS.md`、`cpp/kupl/qcu/AGENTS.md` 与本次触及目录的
`AGENTS.md`。该后端独立实现，不 include 或 link `cpp/cuda/qcu`。

## 文件

| 路径 | 用途 |
|---|---|
| `cpp/kupl/qcu/include/kupl/` | 扁平 ABI、`LatticeSet`、几何、CPU 运行时与算子声明 |
| `cpp/kupl/qcu/src/apply_*.cpp` | 与 CUDA QCU 对应的单入口族 C ABI 薄封装 |
| `cpp/kupl/qcu/src/backend.cpp` | `LatticeSet<Real>` 门面、参数校验与 scratch 所有权 |
| `cpp/kupl/qcu/src/lattice_*.cpp` | geometry、halo、Wilson/Clover、solver、Laplacian、Gauss、MG 分域实现 |
| `cpp/kupl/qcu/qcu.map` | 仅导出 37 个 CUDA 兼容 `*Qcu` 符号 |
| `cpp/kupl/qcu/tests/test_kupl.cpp` | c64/c128、1/2/4-rank、canonical layout 与真残差 |
| `cpp/kupl/qcu/tests/test_new_abi.py` | 独立 NumPy/ctypes 差分，覆盖新增 Wilson/Laplacian 入口 |
| `cpp/kupl/qcu/tests/test_multigrid.py` | 1/2-rank Gaussian、restrict/prolong、窄/宽 coarse halo 差分 |
| `pyqcu/testing/kupl/test_dslash_block_mpi.py` | block dslash/Schur/adjoint 的 2-rank 直接全局参考门 |
| `pyqcu/testing/kupl/test_strict.py` | strict coarse/fine MATPC、R-P、持久 V-cycle、FGMRES，1/2/4-rank 与 c64/c128 |
| `pyqcu/testing/kupl/test_legacy_mg.py` | 1/2/4-rank 递归 Clover-MG、deflate 开关、warm/cold 路径与 verify |
| `cpp/kupl/qcu/tests/test_failure_modes.py` | 非收敛与 fail-closed 入口的显式失败门 |
| `cpp/kupl/qcu/tests/benchmark_kupl.cpp` | Wilson/Clover Dslash 吞吐与 rank affinity 证据 |
| `pyqcu/testing/kupl/check_abi.py` | KUPL/CUDA C 声明与动态库 37 项导出符号一致性门 |
| `pyqcu/kupl/qcu.py` | `ctypes.CDLL(..., RTLD_LOCAL)` Python 绑定；37 个 ABI 名称均有 Python 包装 |
| `pyqcu/kupl/linear_operator.py` | `KuplLinearOperator` — compact Wilson/Clover Schur apply 与 native whole-solve adapter |
| `pyqcu/kupl/krylov.py` | `KuplBiCGStabDriver` / `KuplCGDriver` — 每次请求只调用一次 C++ solve |
| `pyqcu/kupl/multigrid/` | 统一 KUPL multigrid 入口：`LegacyMultigrid` / `StrictMultigrid`，生命周期为 `setup/solve/close` |
| `pyqcu/testing/kupl/test_kupl_python.py` | PyQCU 参考、Clover、dslash、solver 跨语言门 |
| `pyqcu/testing/simple/kupl_bistabcg.py` | 独立 NumPy 布局转换、KUPL ctypes 求解与真残差 |
| `pyqcu/testing/simple/kupl_multigrid.py` | full-site Galerkin、MG verify、独立真残差 |
| `pyqcu/testing/simple/compare_kupl_*.py` | 同 gauge/RHS 的 KUPL↔QUDA 解对照 |
| `pyqcu/testing/simple/benchmark_kupl.py` | 多次运行 C++ 算子 benchmark 并汇总中位数/MAD |

## ABI 与布局

- `params=int32[58]`、`argv=real[8]`、`set_ptrs=int64[100]`。
- `argv[7]=csw`，`kappa=1/(2*mass+8)`，Clover onsite 项
  `T=-0.125*kappa*csw*sum(sigma*F)`。
- `applyCloverQcu`/`applyCloversQcu` 返回 `A=I+T`；后者另返回 `A^-1`。
  `csw=0` 必须精确得到 identity。
- Gauge `[2,3,3,4,X,Y,Z,T/2]`、fermion `[2,4,3,X,Y,Z,T/2]`、
  Clover `[4,3,4,3,X,Y,Z,T/2]` 均为 C-order canonical 布局。
- 外部 fermion 索引必须为 `(parity*12+component)*half_sites+site`；
  外部 Clover 索引必须为 `(row*12+column)*half_sites+site`。
- MPI halo 临时缓冲可 site-major，但进入计算前必须 gather 为局部 `C[12]`，
  结果必须 scatter 回 canonical 外部布局。
- Strict public wrapper 与 raw ctypes ABI 已对齐；回归应交叉比较两条路径，
  当前同输入 L32 解相对差约 `8.3e-8`。若只验证 raw ABI，不得外推为所有公共路径可用。
- KUPL Legacy 原生 cache 是 full-time 布局；CUDA/Torch Legacy 的公共 cache 是
  odd-Schur、parity-compressed-time 布局。两者 coarse shape 不同，禁止按路径或文件大小
  互换。KUPL 收到外部 cache 时仍须校验 E、coarse geometry 与 ABI。

## 并行与绑定

- MPI 负责四维 rank 分解与 face/corner halo；rank 内只使用 OpenMP。
- `KUPL_OMP_THREADS` 显式指定正数时精确采用，不再向 2 的幂取整；未指定时
  将 OpenMP 最大值裁剪到进程初始 CPU affinity 大小。
- `KUPL_NUMA_BIND=1` 由库分配连续 CPU 子集；此时 OpenMPI 必须使用
  `--bind-to none` 且不得设置 `OMP_PROC_BIND`，否则运行时会在库绑定前压缩
  可见 cpuset。
- 调度器已提供 NUMA-local cpuset 时保持 `KUPL_NUMA_BIND=0`，只设置
  `KUPL_OMP_THREADS`/`OMP_NUM_THREADS`。
- 同节点多 rank 应在 Python/NumPy 输入分配前完成 MPI 或调度器绑定；库内
  `KUPL_NUMA_BIND=1` 发生在 `applyInitQcu`，不能迁移调用方已前触到错误 NUMA
  节点的页。OpenMPI 可用 `--bind-to numa --map-by ppr:1:numa`。
- 全局点积/范数使用 `MPI_Allreduce`，所有 rank 必须走对称集合操作。
- 大 local-volume dslash 默认启用跨 site SIMD 内点/周期面内核；local
  `half_sites < 65536` 自动回退 scalar。`KUPL_DISABLE_BLOCK_DSLASH=1`
  可做 A/B，`KUPL_BLOCK_DSLASH_MIN_HALF_SITES` 仅用于矩形/阈值回归。
- Wilson 求解使用 `S_W=I-kappa^2 D_pq D_qp`；BiStabCG 解 `S_W`，
  CG 解 `S_W^dagger S_W`，二者迭代轨迹不同。
- 3D Laplacian 使用独立 `gauge[3,3,3,X,Y,Z]`/`field[3,X,Y,Z]` 布局与
  三方向 MPI halo，不走 parity-compressed 四维 Wilson 几何。
- legacy Clover-MG 内部 full-site 向量在进入 `clover_full_apply` 前必须转换为
  parity-canonical，返回后再转回；direct full-site 指针会造成非平凡 κ 的静默
  错误解，近单位测试无法发现。

## 构建与验证

```bash
cmake -S cpp/kupl/qcu -B cpp/kupl/qcu/build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/kupl/qcu/build -j
ctest --test-dir cpp/kupl/qcu/build --output-on-failure
python -m pytest -q pyqcu/testing/kupl/test_kupl_python.py
```

UBSAN 全 MPI 回归建议使用单 OpenMP 线程：

```bash
cmake -S cpp/kupl/qcu -B cpp/kupl/qcu/build-ubsan \
  -DCMAKE_BUILD_TYPE=Debug -DKUPL_ENABLE_NATIVE=OFF \
  -DKUPL_TEST_OMP_THREADS=1 \
  -DCMAKE_CXX_FLAGS='-fsanitize=undefined -fno-omit-frame-pointer' \
  -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=undefined' \
  -DCMAKE_SHARED_LINKER_FLAGS='-fsanitize=undefined'
cmake --build cpp/kupl/qcu/build-ubsan -j
UBSAN_OPTIONS=halt_on_error=1 ctest \
  --test-dir cpp/kupl/qcu/build-ubsan --output-on-failure
```

ASAN 与 OpenMPI 的 signal handler 可能递归输出 `DEADLYSIGNAL`；先以
`KUPL_TEST_OMP_THREADS=1` 和 `ASAN_OPTIONS=...:handle_segv=0` 做单 rank 最小
复现。没有可解释栈时，不能把 ASAN 超时写成通过。

## 性能基准

同窗 KUPL 路径选择、LTO 结果、跨后端比较边界与当前工作区证据状态见
[`pyqcu-docs/references/extracted-guidance.md`](../pyqcu-docs/references/extracted-guidance.md#5-性能与证据解释)。

```bash
KUPL_OMP_THREADS=16 \
  cpp/kupl/qcu/build/kupl_qcu_bench \
  --lattice 16 16 16 32 --iterations 10

KUPL_BENCH_SHOW_AFFINITY=1 KUPL_OMP_THREADS=8 KUPL_NUMA_BIND=1 \
  mpirun --bind-to none --oversubscribe -np 2 \
  cpp/kupl/qcu/build/kupl_qcu_bench \
  --lattice 16 16 16 32 --grid 2 1 1 1 --iterations 10
```

不要在短 OpenMP 并行区设置 `OMP_PROC_BIND=close/spread` 和 `OMP_PLACES=cores`。
同题 A/B 曾把 `24^3x64` 从 `7.2-7.4 s` 恶化到 `65-67 s`，迭代数和残差不变；
生产绑定按本文件的 `KUPL_NUMA_BIND` / MPI 规则执行。

benchmark 的 `estimated_gflops` 分别使用 Wilson `4*12*12*8` 和
Clover 再加 onsite `12*12*8` flop/site/application 模型，不是硬件计数器。

## 当前边界

已实现 Wilson/Clover dslash、Wilson BiStabCG/CG、Clover 构造/逆、
supplied Clover dslash、Clover Schur prepare/reconstruct、Clover BiStabCG、
3D Laplacian、Gaussian gauge、legacy 递归 Clover-MG/verify、strict
coarse/MATPC/R-P、持久 V-cycle 与右预条件 FGMRES。当前限制为同精度
c64/c128；CUDA 专用 mixed-precision、GCR/CA-GCR/BiCGStabL 模式必须显式
拒绝，不得静默降级。LIME/HDF5 I/O 与 GPU 执行仍不属于 KUPL。

`restart=0` 是全格点 parity-Schur BiCGStab，不调用 coarse hierarchy。
当前 c128 全格点 `24^3x64` 的 `10.282 s / 21 it` 就是该路径，不能写成
c128 coarse V-cycle 性能；同规模 c128 coarse 性能仍未验证。

## 反模式

- 不要用 site-major 索引读写 PyQCU 的 fermion/Clover 外部张量。
- 不要把 `applyCloverQcu` 实现为裸 `T`。
- 不要在 `KUPL_NUMA_BIND=1` 时同时让 MPI 或 OpenMP 做 CPU binding。
- 不要用进程当前 cpuset 覆盖显式 `KUPL_OMP_THREADS`；OpenMP runtime 可能已把
  master thread 临时绑定到单核。
- 不要让 MPI 集合调用依赖 rank 的提前返回。

## 2026-10-21 Clover 6×6 块与 MG 几何

KUPL 的 Clover onsite 矩阵及逆矩阵在当前 gamma 基下按自旋 `(0,1)` 与
`(2,3)` 分成两个稠密 `6×6` 块，不是 `(0,3)/(1,2)`。求解热路径使用
`apply_clover_onsite`，逆矩阵也按两个 `6×6` 块计算，但外部 ABI 仍写回完整
`12×12` 矩阵并显式清零另外两块。29 项 CTest 必须全绿后才能保留该优化。

真实 `24³×64`、24 线程/rank、5 次稳态中位数：

| 精度 | MPI | solve 基线 | solve 优化后 | 真残差 |
|---|---:|---:|---:|---:|
| c64 | 1 | 9.858 s | 7.770 s | `4.12e-7` |
| c64 | 2 | 7.961 s | 6.352 s | `4.13e-7` |
| c128 | 1 | 20.196 s | 15.593 s | `6.96e-8` |
| c128 | 2 | 14.450 s | 11.473 s | `6.96e-8` |

双 MPI 相对单 MPI 仅约 `1.22x/1.36x`，该 workload 已受 NUMA/内存带宽和集合
归约限制，不能把增加 rank 数等同于线性加速。两 rank 分域优先切最长时间轴
`(1,1,1,2)`；x 轴二分实测慢约 9%。

`simple/kupl_multigrid.py` 的旧 builder 曾把 `mg_grid` 误作粗格尺寸，导致
`prolong` reshape 失败；正确值恒为 `lat[i] // mg_grid[i]`。非退化回归见
`pyqcu/testing/kupl/test_simple_mg_geometry.py`，不可只用 `4⁴/mg_grid=2`
这种粗格与 block 恰好相等的退化工况验证。

## 2026-10-21 legacy-MG coarse tile 与 rank 扩展

先区分当前选路和历史优化：dev103 同窗 `24^3x64` c64 的当前最佳开发路径是
`restart=0`、8 ranks × 2 threads、`1x4x2x1`，中位 `7.532 s / 19 it`，
并有独立全算子 MPI 真残差 `4.011e-7`。下面 `restart=5` 的 coarse 优化记录
说明该实现如何演进，但不是当前最快路径，也不应把其 native residual 写成独立
真残差。

KUPL LTO 的单点 A/B 为 `7.556 -> 7.196 s`（`4.8%`），解逐位一致、CTest
`29/29`。该收益只覆盖 8 ranks × 2 threads 的 `24^3x64` c64 点，LTO 需通过
`--kupl-lib lto` 或 `PYQCU_KUPL_LIB` 显式选择，默认构建未切换。

`apply_multigrid_coarse_dslash_wide` 以 512-site tile 预解析 32 个邻接流，
随后按 site-major coefficient stream 连续累加；canonical/full 转换改为
half-site 遍历并一次写两个时间 parity，coarse correction 复用刚形成的
true residual。生产 `24³×64` c64、`restart=5`、deflate、4 threads/rank：

| ranks | 中位 | speedup | efficiency |
|---:|---:|---:|---:|
| 1 | 85.204 s | 1.00x | 100% |
| 2 | 46.752 s | 1.822x | 91.1% |
| 4 | 29.942 s | 2.846x | 71.1% |

4-rank 相对优化前 `94.108 s` 降低 68.18%，60 iterations，native relative
residual `9.70132e-7`，verifier=0。跨 rank 仍必须使用对称集合操作；该结果只覆盖
单节点 y-only 分域，不外推到跨节点 InfiniBand。

coarse correction 现在采用 fail-closed 验收：每次 correction 后比较 true residual；
若不变、变差或非有限，则撤销 correction 并关闭本 solve 的后续 correction，剩余
BiCGStab 在单个长 chunk 中完成。相同生产 `restart=5, deflate=true` 参数下，
solve 中位进一步降至 `16.771 s`，47 iterations，relative residual
`9.13844e-7`，verifier=0；相对最初的 `94.108 s` 降低 82.18%。

### 2026-10-21 BiCGStab 成对归约

`bicgstab_chunk` 将相邻的 `(v,v)` 与 `(v,s)` 点积合并为一次四实数组
`MPI_Allreduce`，并在同一 OpenMP 遍历中计算两个局部和。这样每轮全流程的集合
归约次数由 6 降到 5，同时保持更新公式、actual iteration counting、停机判据和
adaptive correction 不变。生产 `24³×64` c64、4 ranks x 4 threads、`restart=5`、
deflate：中位 `18.501 -> 17.725 s`（4.19%，1.0438x），均 47 iterations，
native relative residual `9.138e-7`，verifier=0。最终 CTest `29/29` 通过；
独立 NumPy 真残差未在该多 rank 脚本中启用，不能把 native residual 写成独立验证。

## 2026-10-22 2/4-rank efficiency decomposition

固定生产 c64、4 threads/rank 的 profile 分解显示，4-rank 的 fine
Wilson-Clover rank-local apply 占 solve `73.2%`；Allreduce `7.7%`、canonical
conversion `6.4%`、coarse dslash/halo `1.3%`、vector/other `11.4%`。相对
1-rank 完美扩展的额外耗时中，fine apply 占 `52.5%`，Allreduce `16.1%`，
canonical `9.7%`，other `21.1%`，coarse 仅 `0.6%`；4-rank fine apply 内 halo
仅 `7.55%`。因此 2/4-rank efficiency 损失首先是 rank 内计算和内存带宽问题。

把每轮 5 次 Allreduce 降到 4 次的候选在 4-rank 三轮 A/B 中为
`15.072466 -> 16.681220 s`，慢 `10.67%`；虽然均 47 iterations、verifier=0、
native residual 约 `9.138e-7`，候选已撤销。减少集合调用次数不保证降低关键路径。

## 2026-10-22 fine dslash hotspot and rejected face SIMD

固定 local `24×6×24×64`、4 ranks × 4 threads、c64：dslash 内点+周期面占
`53.22%`，标量边界占 `37.85%`，halo `3.29%`，MPI wait `5.65%`。SIMD block
对 Wilson/Clover/Schur 比 scalar 快 `13.98%/6.95%/2.04%`；Clover onsite 占
Clover 算子 `39.36%`，canonical 往返约为每迭代墙钟 `1.73%`。

y-face 连续时间 lane SIMD 候选使 4-rank 中位 `15.446281 -> 16.157466 s`
（慢 `4.60%`），MAD 从 `0.061435` 扩到 `0.715516 s`；六轮均 47 iterations、
verifier=0、PASS，已撤销。下一步应分析 scalar boundary 的 rank-face 分布和页
局部性，不能只因标量边界占比大就盲目扩大 SIMD lane。
