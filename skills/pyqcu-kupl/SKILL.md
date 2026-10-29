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
| `cpp/kupl/qcu/qcu.map` | 仅导出 40 个 CUDA 兼容 `*Qcu` 符号 |
| `cpp/kupl/qcu/tests/test_kupl.cpp` | c64/c128、1/2/4-rank、canonical layout 与真残差 |
| `cpp/kupl/qcu/tests/test_new_abi.py` | 独立 NumPy/ctypes 差分，覆盖新增 Wilson/Laplacian 入口 |
| `cpp/kupl/qcu/tests/test_multigrid.py` | 1/2-rank Gaussian、restrict/prolong、窄/宽 coarse halo 差分 |
| `pyqcu/testing/kupl/test_dslash_block_mpi.py` | block dslash/Schur/adjoint 的 2-rank 直接全局参考门 |
| `pyqcu/testing/kupl/test_strict.py` | strict coarse/fine MATPC、R-P、持久 V-cycle、FGMRES，1/2/4-rank 与 c64/c128 |
| `pyqcu/testing/kupl/test_legacy_mg.py` | 1/2/4-rank 递归 Clover-MG、deflate 开关、warm/cold 路径与 verify |
| `pyqcu/testing/kupl/kunpeng_runtime.py` | 实际 affinity/sysfs NUMA 探测与 HMPI/OpenMPI、CPU/内存绑定规划 |
| `pyqcu/testing/kupl/test_kunpeng_runtime.py` | 合成双 NUMA、片上内存 offset、线程容量与 CLI JSON 契约 |
| `pyqcu/testing/kupl/test_kunpeng_mpi_binding.py` | 双 rank 实际 cpuset 非空/不重叠门；硬件不足返回 skip 77 |
| `pyqcu/testing/kupl/test_kunpeng_profile_contract.py` | 非鲲鹏主机拒绝鲲鹏 profile，rank 间 profile 不一致快速失败 |
| `pyqcu/testing/kupl/benchmark_runtime.py` | 分离 init、首次 Clover 与 RSS 的库级初始化 A/B |
| `cpp/kupl/qcu/tests/test_failure_modes.py` | 非收敛与 fail-closed 入口的显式失败门 |
| `cpp/kupl/qcu/tests/benchmark_kupl.cpp` | Wilson/Clover Dslash 吞吐与 rank affinity 证据 |
| `pyqcu/testing/kupl/check_abi.py` | KUPL/CUDA C 声明与动态库 40 项导出符号一致性门 |
| `pyqcu/kupl/qcu.py` | `ctypes.CDLL(..., RTLD_LOCAL)` Python 绑定；40 个 ABI 名称均有 Python 包装 |
| `pyqcu/kupl/linear_operator.py` | `KuplLinearOperator` — compact Wilson/Clover Schur apply 与 native whole-solve adapter |
| `pyqcu/kupl/krylov.py` | `KuplBiCGStabDriver` / `KuplCGDriver` — 每次请求只调用一次 C++ solve |
| `pyqcu/kupl/multigrid/` | 统一 KUPL multigrid 入口：`LegacyMultigrid` / `StrictMultigrid`，生命周期为 `setup/solve/close` |
| `pyqcu/testing/kupl/test_kupl_python.py` | PyQCU 参考、Clover、dslash、solver 跨语言门 |
| `pyqcu/testing/simple/kupl_bistabcg.py` | 独立 NumPy 布局转换、KUPL ctypes 求解与真残差 |
| `pyqcu/testing/simple/kupl_multigrid.py` | full-site Galerkin、MG verify、独立真残差 |
| `pyqcu/testing/simple/compare_kupl_*.py` | 同 gauge/RHS 的 KUPL↔QUDA 解对照 |
| `pyqcu/testing/simple/benchmark_kupl.py` | 多次运行 C++ 算子 benchmark 并汇总中位数/MAD |

## ABI 与布局

跨 CUDA/KUPL/Torch 的公共 PyQCU 接口规范以
[`pyqcu-api/SKILL.md`](../pyqcu-api/SKILL.md#三后端统一接口强制规范) 为唯一权威源。
`KuplLinearOperator`、`KuplBiCGStabDriver`、KUPL `Multigrid` 和 `ctypes` ABI 属于后端
适配层；高层调用统一走 `Solver`/`Krylov`、`Multigrid`、`MultigridCache`。新增 KUPL
功能时保持公共参数名、控制对象和返回类型一致；不支持的能力在构造或能力检查阶段失败关闭。

同一 QCU 版本内 identity、gauge、MPI coarse geometry、dtype 和 setup provenance 一致的
Legacy/Strict cache 直接复用，包括 CUDA/Torch 生成的 canonical odd-Schur Legacy cache。
复用前执行统一 manifest 校验，命中后跳过 null-space/Galerkin setup；只有 miss、损坏或
identity 不一致时才重建。

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
- `KUPL_NUMA_BIND=1` 按 sysfs cpulist 分配完整 NUMA-local CPU 子集，并尝试
  `KUPL_MEMORY_POLICY=preferred|bind`；此时 MPI 必须使用 `--bind-to none` 且
  不得设置 `OMP_PROC_BIND`/`OMP_PLACES`/`GOMP_CPU_AFFINITY`/`KMP_AFFINITY`，
  否则运行时会在库绑定前压缩可见 cpuset。库会绑定实际 OpenMP team 的每个线程。
- `KUPL_MEMORY_NODE=local+N` 用于独立编号的片上内存 node；
  `KUPL_REQUIRE_MEMORY_BIND=1` 把 `set_mempolicy` 失败改为 fatal，
  `KUPL_TOPOLOGY_REPORT=1` 输出每 rank 的最终 cpuset/node/thread 证据。
- `PYQCU_KUPL_PROFILE=kunpeng|kunpeng-2numa` 会校验 Linux ARM64 与
  Kunpeng/HiSilicon CPU 标识；profile/topology/oversubscribe 开关在 rank 间必须
  一致，不一致必须在任何提前返回之前由集合归约统一失败。
- HMPI 首选调度器 `--map-by numa --bind-to numa`。HMPI 在 `--bind-to none`
  下可能让部分拓扑集合算法回退，因此库内绑定只作无调度器回退。
- 默认不设置 `OMP_PROC_BIND=close/spread`；只在稳定独占 cpuset 上做同题 A/B。
  短 OpenMP 并行区的历史回退证据禁止把该设置写成通用默认。
- `KUPL_FUSED_BICG_DOTS=0` 可回退 BiCGStab `(t,t)`/`(t,s)` 成对归约；
  `KUPL_HUGEPAGE=1` 只对大型内部 scratch 发 `MADV_HUGEPAGE` advice。
- Clover cache 默认要求 gauge 内容在 set 生命周期内不可变。原地改写同一地址时
  必须 `KUPL_DISABLE_CLOVER_CACHE=1`，强制刷新 gauge halo 和目标 parity inverse。
- 调度器已提供 NUMA-local cpuset 时保持 `KUPL_NUMA_BIND=0`，只设置
  `KUPL_OMP_THREADS`/`OMP_NUM_THREADS`。
- 同节点多 rank 应在 Python/NumPy 输入分配前完成 MPI 或调度器绑定；库内
  `KUPL_NUMA_BIND=1` 发生在 `applyInitQcu`，不能迁移调用方已前触到错误 NUMA
  节点的页。仓库 LIME benchmark/MG driver 会先执行一次私有 prebind lifecycle；
  自定义调用方仍须自行保证顺序。OpenMPI 可用
  `--bind-to numa --map-by ppr:1:numa`。
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

目标节点性能构建使用 `-DKUPL_ENABLE_LTO=ON`；跨型号部署使用
`-DKUPL_ENABLE_NATIVE=OFF`。完整鲲鹏 HMPI、NUMA、大页、测试和证据模板见
[`docs/kupl_kunpeng_guide.md`](../../docs/kupl_kunpeng_guide.md)。

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

## 2026-10-29 MPI coarse-cache builder

Legacy 33-tensor cache 的规范 MPI 入口是
`pyqcu.testing.simple.mpi.build_full_mg_cache`。它不能退回父目录的 rank-0
串行脚本。实现边界如下：

- 每个 rank 用全局 process grid 生成自己的 fine gauge/Clover 和 null-vector slab；
- rank 0 汇总 blocked `lonv=[E,12,X,bx,Y,by,Z,bz,T/2,bt]`，overlap patch 必须按
  `(2,4,6,8)` 四个 coarse 轴切片，不能把 `lonv` 当成普通 lattice-axes-last-4 张量；
- KUPL local-stencil API 在全局 MPI 作业内需要 `1x1x1x1` 私有 context；C++ geometry
  以 `params[_NODE_SIZE_]=1` 显式创建单 rank communicator，不能强制用
  `MPI_COMM_WORLD` 解释 patch grid；
- 每个 rank 只输出自己的 coarse slab，最终四张量由 rank 0 组装、manifest 验签，
  再由所有 rank 回读各自 hyperslab 与本地 slab 比对；
- `--build-device cuda` 在 1 rank 使用 CUDA patch stencil；多 rank 时 CUDA Clover
  halo 固定绑定 `MPI_COMM_WORLD`，不能安全承载 serial overlap patch，因此
  null-vector 仍在 GPU 分布式生成，patch stencil 切到 CPU，并在报告中写
  `stencil_device=cpu`。不得把该混合 setup 时间当作纯 GPU 性能；
- 1/2/4-rank 不要求 SHA 位级相同，但同一物理输入的功能资产应报告相对误差。
  2026-10-29 `8x8x8x16` c64 E2 实测 `lonv/hnn/hdg/sit` 的 2/4-rank 对 1-rank
  Frobenius 相对差约 `1.3e-5–2.4e-5`，符合 MPI 归约顺序差异。

禁止把 overlap patch 做成每 rank 重建完整 hierarchy；它是仅覆盖
`2*mg_grid` coarse block 的 halo 工作区。

Legacy CUDA-to-KUPL cache 复用已在 2026-10-29 打通。此前约 `8.9%` 的
Galerkin 和 L8 `hnn=1.35e-2` 失配不是 KUPL slot 顺序，而是 CUDA raw lifecycle
未同步更新后的 parity，导致 CUDA even/odd Clover 相同。修复 CUDA
`sync_parity()` 后，同一 gauge 与同一 `lonv` 下 `hnn/hdg/sit` 相对误差为
`2.48e-7/7.34e-8/2.52e-7`；KUPL 直接加载 CUDA canonical E2 cache 的
`verify_status=0`，独立 true residual `2.016388e-7`。因此不得再对 Legacy payload
做方向转置或私有重排来绕过该问题。

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

KUPL LTO 的单点 A/B 为 `7.556 -> 7.196 s`（`4.8%`），解逐位一致、当时 CTest
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
native relative residual `9.138e-7`，verifier=0。当时 CTest `29/29` 通过；
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

## 2026-10-27 鲲鹏运行时与初始化收敛

- KUPL 现从 Linux sysfs 读取 CPU NUMA cpulist，`KUPL_NUMA_BIND=1` 不再按连续 CPU
  编号猜测节点；支持 `KUPL_CPU_NODE`、`KUPL_MEMORY_NODE=local|N|local+N` 与
  preferred/bind memory policy。`PYQCU_KUPL_PROFILE=kunpeng-2numa` 和
  `KUPL_REQUIRE_TOPOLOGY=1` 对跨 NUMA/拓扑缺失 fail closed；鲲鹏 profile 还会
  拒绝非鲲鹏主机，rank 间不一致会立即失败而不是挂起。
- 合成双 NUMA CTest 实际启动 2 rank，得到互斥 cpusets 和不同 cpu_node。
  `test_kunpeng_mpi_binding.py --synthetic-two-numa` 是通用 CI 的拓扑门；真实鲲鹏
  仍须用 `KUPL_TOPOLOGY_REPORT=1` 和 `mpirun --report-bindings` 复核。
- Clover cache 先收敛到目标 parity 惰性 `term + inverse`，本轮再把 term 原位求逆，
  每个 parity 常驻从 `2B` 降到 `B`（`B=144*half_sites*sizeof(complex)`）。缓存版
  standalone Clover dslash 同时改走已有 6×6 spin-block onsite，而外部 supplied
  12×12 ABI 仍保留通用路径。相邻 A/B 中 c64 `1.074 -> 0.957 s`（约 10.9%），
  c128 `1.881 -> 1.451 s`（约 22.9%）；只适用于 standalone `operator=clover`，
  求解器原本已使用 6×6 helper。
- 求解器 9 块全场级 scratch 与 3 块 parity scratch 改为首次求解惰性分配。
  `24³×64`、4 threads、plan=Clover 的新口径中，c64 init `0.547 -> 0.099 s`，
  c128 `1.066 -> 0.180 s`；由于 runtime benchmark 同时改为 init 后再分配输入，
  新报告的 `after_inputs` 必须与旧报告分离。组合 tile 后的最终 c64/c128 首次
  Clover 后 RSS 为 `2.079/3.359 GB`，首次 Clover 为 `3.757/4.690 s`。
- `compute_clover` 改为 site-tile 计算和 component 连续写回。两组相邻 A/B 中 c64
  首次 Clover 分别快 18.1%/23.2%，c128 快 8.4%/8.2%；该收益只属于 Clover setup，
  不得写成 steady solve 加速。
- 6×6 求逆也改为 site-tile 连续加载/写回；相对仅构造 tile 的基线，两组 c64 分别
  快 12.6%/17.4%，c128 快 12.1%/10.8%，同样只归因于 setup。
- face/corner halo 现在只为实际分解维度分配。1-rank plan=Clover 的 init 从
  c64/c128 `0.140/0.236 s` 降到 `0.023/0.021 s`，init RSS delta 约
  `207/411 MB -> 2.3 MB`；2/4-rank x/y/z/t 与对角路径仍须全测。
- `KUPL_FUSED_BICG_DOTS=1` 将 `(t,t)`/`(t,s)` 合并为一次遍历/归约。两个相邻
  `24³×64` c64、2 ranks × 4 threads 进程对分别得到 `19.039 -> 17.434 s` 和
  `19.509 -> 19.143 s`；最终交付二进制两次相邻复核为 `17.968 -> 17.778 s`
  和 `24.551 -> 23.713 s`（1.05% / 3.41%）。均 20 iterations、独立真残差约
  `4.1e-7`。共享机存在大幅漂移，只可引用相邻对，稳定承诺写约 1%–3.4%，
  不可跨窗口拼接或选最大收益。
- 默认 Release、`KUPL_ENABLE_NATIVE=OFF`、`KUPL_ENABLE_LTO=ON` 和 UBSAN 构建
  均完成 33/33 CTest；本轮新增 profile contract 后当前并行构建为 34/34。
  LTO 选项只是把历史 `4.8%` 构建方式产品化，不把共享机短样本升级为新倍率。
- boundary 预过滤与“全部 Irecv 提前于 face pack”两个候选在 2/4-rank 相邻复测
  中收益不一致，已完整撤销。不能仅凭访存或通信直觉保留。
- 完整指南：`docs/kupl_kunpeng_guide.md`；当前阶段证据：
  `logs/kupl_kunpeng_opt_20261027.md`。当前开发机为单 NUMA x86，容器禁止
  `set_mempolicy`，因此不能据此宣称鲲鹏 NUMA/片上内存性能。

  Strict-MG `8x8x8x16` c64, 2 levels, block `2^4`, `nvec=12`,
  `coarse_spin=2`, `restart=20`, 12 threads/rank currently measures
  `1.222731/1.101605/1.658620 s` at 1/2/4 ranks (2 warmups + 5 steady), with
  independent residuals below `1e-6`. The 4-rank point must use `2x2x1x1`; the
  shared login-node load average makes the negative 4-rank scaling unsuitable as a
  Kunpeng production claim. Record NUMA/cpuset, MPI ranks/grid, cache load and actual
  correction counts with every result.
