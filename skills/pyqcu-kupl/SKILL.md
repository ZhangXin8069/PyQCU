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

## 并行与绑定

- MPI 负责四维 rank 分解与 face/corner halo；rank 内只使用 OpenMP。
- OpenMP 线程数向下取 2 的幂，`KUPL_OMP_THREADS` 显式指定时优先。
- `KUPL_NUMA_BIND=1` 由库分配连续 CPU 子集；此时 OpenMPI 必须使用
  `--bind-to none` 且不得设置 `OMP_PROC_BIND`，否则运行时会在库绑定前压缩
  可见 cpuset。
- 调度器已提供 NUMA-local cpuset 时保持 `KUPL_NUMA_BIND=0`，只设置
  `KUPL_OMP_THREADS`/`OMP_NUM_THREADS`。
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

```bash
KUPL_OMP_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores \
  cpp/kupl/qcu/build/kupl_qcu_bench \
  --lattice 16 16 16 32 --iterations 10

KUPL_BENCH_SHOW_AFFINITY=1 KUPL_OMP_THREADS=8 KUPL_NUMA_BIND=1 \
  mpirun --bind-to none --oversubscribe -np 2 \
  cpp/kupl/qcu/build/kupl_qcu_bench \
  --lattice 16 16 16 32 --grid 2 1 1 1 --iterations 10
```

benchmark 的 `estimated_gflops` 分别使用 Wilson `4*12*12*8` 和
Clover 再加 onsite `12*12*8` flop/site/application 模型，不是硬件计数器。

## 当前边界

已实现 Wilson/Clover dslash、Wilson BiStabCG/CG、Clover 构造/逆、
supplied Clover dslash、Clover Schur prepare/reconstruct、Clover BiStabCG、
3D Laplacian、Gaussian gauge、legacy 递归 Clover-MG/verify、strict
coarse/MATPC/R-P、持久 V-cycle 与右预条件 FGMRES。当前限制为同精度
c64/c128；CUDA 专用 mixed-precision、GCR/CA-GCR/BiCGStabL 模式必须显式
拒绝，不得静默降级。LIME/HDF5 I/O 与 GPU 执行仍不属于 KUPL。

## 反模式

- 不要用 site-major 索引读写 PyQCU 的 fermion/Clover 外部张量。
- 不要把 `applyCloverQcu` 实现为裸 `T`。
- 不要在 `KUPL_NUMA_BIND=1` 时同时让 MPI 或 OpenMP 做 CPU binding。
- 不要用进程当前 cpuset 覆盖显式 `KUPL_OMP_THREADS`；OpenMP runtime 可能已把
  master thread 临时绑定到单核。
- 不要让 MPI 集合调用依赖 rank 的提前返回。
