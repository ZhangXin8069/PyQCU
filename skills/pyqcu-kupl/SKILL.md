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
| `pyqcu/testing/kupl/kupl_env.py` | MPI-free 环境报告：CPU/NUMA/cgroup、HBM 候选、KUPL 库/变量冲突、真实 MG/BiStabCG 生产命令 |
| `pyqcu/testing/kupl/test_kupl_env.py` | HBM offset/歧义/无 sysfs fail-safe、报告命名/原子写和 CLI 不导入 MPI 的契约 |
| `pyqcu/testing/kupl/test_kunpeng_mpi_binding.py` | 双 rank 实际 cpuset 非空/不重叠门；硬件不足返回 skip 77 |
| `pyqcu/testing/kupl/test_kunpeng_profile_contract.py` | 非鲲鹏主机拒绝鲲鹏 profile，rank 间 profile 不一致快速失败 |
| `pyqcu/testing/kupl/benchmark_runtime.py` | 分离 init、首次 Clover 与 RSS 的库级初始化 A/B |
| `pyqcu/testing/kupl/test_kunpeng_all.py` | 鲲鹏全配置矩阵驱动：`list/prepare/prewarm/solve/merge/run/validate` 与 94 列 CSV 台账 |
| `pyqcu/testing/kupl/submit_kunpeng_all.sh` | 按节点分组的 `#DSUB` 提交器：单任务独占整节点、`--shard-index/count`、`--dry-run` |
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
- 公共 Legacy cache 使用 canonical odd-Schur、parity-compressed-time 布局，KUPL 通过
  `SET_PTRS_LEGACY_LAYOUT=1` 可直接消费 CUDA/Torch 生成的兼容 cache。KUPL 也保留
  历史 full-time cache 路径；两种布局 coarse shape 不同，必须按 manifest 校验
  identity、dtype、coarse geometry 与 layout，禁止按路径或文件大小猜测。

## 并行与绑定

- MPI 负责四维 rank 分解与 face/corner halo；rank 内只使用 OpenMP。
- `KUPL_OMP_THREADS` 显式指定正数时精确采用，不再向 2 的幂取整；未指定时
  将 OpenMP 最大值裁剪到进程初始 CPU affinity 大小。
- `KUPL_NUMA_BIND=1` 按 sysfs cpulist 分配完整 NUMA-local CPU 子集，并尝试
  `KUPL_MEMORY_POLICY=preferred|bind`；此时 MPI 必须使用 `--bind-to none` 且
  不得设置 `OMP_PROC_BIND`/`OMP_PLACES`/`GOMP_CPU_AFFINITY`/`KMP_AFFINITY`，
  否则运行时会在库绑定前压缩可见 cpuset。库会绑定实际 OpenMP team 的每个线程。
- 绑定是配置项不是调优旋钮：Clover/gauge 顺序 streaming，未绑定时页按 4K 散在多个
  NUMA 域、大页未生效、OpenMP 线程不固定；16 rank 未绑定 `196.00 s` / 绑定 `2.06 s`、
  32 rank `102.47 s` / `3.04 s`（`2236451`/`2236485`、`2236599`），迭代与残差逐位同。
  未绑定多 rank 由 `_require_kunpeng_binding` 拒绝，例外 `PYQCU_ALLOW_UNBOUND_KUPL=1`。
- `KUPL_MEMORY_NODE=local+N` 用于独立编号的片上内存 node；
  `KUPL_REQUIRE_MEMORY_BIND=1` 把 `set_mempolicy` 失败改为 fatal，
  `KUPL_TOPOLOGY_REPORT=1` 输出每 rank 的最终 cpuset/node/thread 证据。
- 生产前用 `python -B pyqcu/testing/kupl/kupl_env.py --ranks ... --threads-per-rank ...`
  生成 `logs/kupl_env-vYYYY-MM-DD-HH-MM-SS.txt`。只有 sysfs 中无 CPU 且有内存的
  NUMA 节点对每个 CPU 节点按 `distance` 唯一映射、且 offset 完全一致时才建议
  `local+N`；无候选或不唯一时只建议 `local`。`bind + require=1` 是严格 HBM
  配置，`preferred` 会回退 DDR；最终用 `numastat -p` 或 `/proc/<pid>/numa_maps`
  验证实际页分布。
- 从仓库根 `source ./env.sh` 会导出当前树优先的 `PYTHONPATH`、`PYQCU_ROOT`、
  `PYQCU_KUPL_BUILD_DIR`/`PYQCU_KUPL_LIB` 和保守 KUPL 默认值；显式调用者值优先。
  `cpp/kupl/qcu/libqcu.so` 是构建生成的稳定软链接，指向当前 build 目录中的实际
  `libqcu.so`；Python ctypes 与生产启动器都通过该路径，不直接写死 build 目录。
  批处理命令应在作业内 source，随后使用 `--kupl-lib default`，不要在外层单引号中
  直接展开 `$PYQCU_KUPL_LIB`。
- `PYQCU_KUPL_PROFILE=kunpeng|kunpeng-2numa` 会校验 Linux ARM64 与
  Kunpeng/HiSilicon CPU 标识；profile/topology/oversubscribe 开关在 rank 间必须
  一致，不一致必须在任何提前返回之前由集合归约统一失败。
- HMPI 首选调度器 `--map-by numa --bind-to numa`。HMPI 在 `--bind-to none`
  下可能让部分拓扑集合算法回退，因此库内绑定只作无调度器回退。
- 默认不设置 `OMP_PROC_BIND=close/spread`；只在稳定独占 cpuset 上做同题 A/B。
  短 OpenMP 并行区的历史回退证据禁止把该设置写成通用默认。
- Kunpeng 计算节点上显式设 `OMP_WAIT_POLICY=ACTIVE`。同节点同 lane A/B
  （`cn19650`、32r×12t、t2 lane、`preferred`、L24 c64、12 it、均 `PASS`）：
  PASSIVE `0.319600 s` → ACTIVE `0.249223 s`（`1.28×`）。原因是全局归约与
  向量更新这类小并行区的线程唤醒开销：reduce 单次约 `1.02 ms`（PASSIVE 批次
  rank 26，`0.052268 s/51`）降到约 `0.45 ms`（ACTIVE 批次 rank 17，
  `0.023190 s/51`）。每分支各 1 个样本，未做多样本中位数。
- `KUPL_FUSED_BICG_DOTS=0` 可回退 BiCGStab `(t,t)`/`(t,s)` 成对归约；
  `KUPL_HUGEPAGE=1` 只对大型内部 scratch 发 `MADV_HUGEPAGE` advice。
- Clover cache 默认要求 gauge 内容在 set 生命周期内不可变。原地改写同一地址时
  必须 `KUPL_DISABLE_CLOVER_CACHE=1`，强制刷新 gauge halo 和目标 parity inverse。
- 调度器已给 NUMA-local cpuset 时保持 `KUPL_NUMA_BIND=0`，只设线程数；多 rank 绑定须
  在 Python/NumPy 分配前完成：库内绑定在 `applyInitQcu`，不能迁移已前触页（LIME
  benchmark/MG driver 内含私有 prebind lifecycle，自定义调用方须自行保证顺序）。
  OpenMPI 可用 `--bind-to numa --map-by ppr:1:numa`。
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

## 收敛判据（2026-11-14）

- **生产求解**：c64 相对 **norm** 收敛条件 `1e-6`；c128 为 `1e-9`。
  判据是 `||r|| / ||b||`（代码内部 `sqrt(norm2)`），**不是 norm²**。
- 驱动按 dtype 设默认 `ATOL`（c64 `1e-6`、c128 `1e-9`），并传
  `--strict-acceptance`：Python 报告的 PASS/FAIL 只按 `atol`，不再叠加
  dtype 地板。
- 测试/单元用例仍保留 dtype 地板（c64 `2048ε`、c128 `128ε`），用于探测
  浮点可达极限；不得把测试地板的宽松值当成生产判据。
- C++ 递推停机用 `max(atol, 4ε)·||b||`，收尾真残差门用
  `max(atol, floor)·||b||`；生产严格门由 `--strict-acceptance` 在 Python
  侧执行。若 c64 在目标格点上到不了 `1e-6`，报告必须如实标 FAIL/WARN，
  不得用地板放行。

## 求解产物协议（2026-11-14）

- **每次 solve 必须保存 `dest`**：驱动应传 `--output "$W/solution_${DTYPE}.h5"`；
  这是独立正确性/精度校验的输入，不得只写 `report.json`。
- **`src` 若原本就是被读取的文件，则跳过重复保存**；但必须把
  `SRC=<path>`、`DEST=<path>` 写入 `meta.txt`，保证后续可用
  gauge + src + dest 复算残差。
- 校验入口：把 `dest` 代入原方程 `A x = b`，用与求解器相对独立的实现
  （Torch-QCU 或 NumPy 参考）计算 `||b - A x|| / ||b||`，并与
  `report.json` 的 `final_true_residual` 对照。

## 硬件取向与 CUDA 差异（2026-11-14）

鲲鹏节点是**高带宽、低单核算力、高同步延迟**；GPU 是**高算力、强延迟隐藏**。
两侧实现不对称是有意选择，不是待统一的缺陷。改动前先读
[`references/kupl-vs-cuda-algorithms.md`](references/kupl-vs-cuda-algorithms.md)，
完整算法对照见 [`docs/kupl_vs_cuda_algorithms_20261114.md`](../../docs/kupl_vs_cuda_algorithms_20261114.md)。

- CPU 侧并行轴是 **site-lane**（c64 16 site、c128 8 site，128 B tile），
  不是 thread-per-site；gamma 先投影成 2×3 再两次 3×3（18 MAC）。
- 单节点 halo 默认共享内存窗口（`KUPL_SHM_HALO=0` 回退 MPI）；CPU 上照搬 CUDA 的
  阻塞 MPI 会多付约 0.6 ms/次 wait。
- 求解器是双容差 + 强制真残差；冷启动置零。CUDA 冷启动随机、递推残差停机，
  迭代数口径不可直接比较。
- Legacy MG 只保留两层 V-cycle 与 BiCGStab 平滑，显式拒绝 GCR/CA-GCR/W-F-K/
  混合精度；不要因为 CUDA 有就默认在 CPU 上开。
- `half_sites` 跌破 65536 时 block SIMD 自动关闭；固定全局格点增加 rank 会触发，
  是多 rank 性能断崖的候选机制之一。调参必须同时记录
  `half_sites`/`grid`/`local[3]`/命中内核。
- Gauss gauge：KUPL 用确定性正态，CUDA 用 curand 均匀；同 seed 不同场。

## 构建与验证

```bash
bash cpp/kupl/qcu/make.sh --check
KUPL_ENABLE_NATIVE=OFF KUPL_ENABLE_LTO=OFF bash ./install.sh kupl
ctest --test-dir cpp/kupl/qcu/build --output-on-failure
python -m pytest -q pyqcu/testing/kupl/test_kupl_python.py
```

在 `cpp/kupl/qcu` 内可直接运行 `bash make.sh` 只构建 native 库和测试程序；
`make.sh --check` 不写文件，仅显示 build dir、native/LTO/tests/jobs 选择。

目标节点性能构建使用 `KUPL_ENABLE_NATIVE=ON KUPL_ENABLE_LTO=ON`；跨型号部署使用
`KUPL_ENABLE_NATIVE=OFF`。完整鲲鹏 HMPI、NUMA、大页、测试和证据模板见
[`docs/kupl_kunpeng_guide.md`](../../docs/kupl_kunpeng_guide.md)。

纯 KUPL 无 CUDA 模式必须使用 `bash install.sh kupl` 或
`PYQCU_BUILD_BACKEND=kupl`。该模式不调用 `build-nv.sh`、不要求 `nvcc`，也不创建
`pyqcu.cuda.qcu` Cython 扩展。旧版 PyTorch 的复数/实数 dtype 转换统一使用
`pyqcu.cann.real_dtype()`，禁止调用可能不存在的 `torch.dtype.to_real()`。
完整安装、测试、生产、升级和排错流程见
[`docs/kupl_only_deployment.md`](../../docs/kupl_only_deployment.md)。

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

### 生产算法选路

- 真正 Legacy-MG：`levels=2`、`restart>0`、身份匹配的全局 coarse cache、
  `coarse_source!=unused: restart=0`、`verify_status=0`。推荐入口
  `pyqcu/testing/simple/mpi/solve_case.py`，它自动构建/校验 cache 并计算独立真残差。
- 高效 BiStabCG：`levels=1`、`restart=0`、无 coarse hierarchy/cache；多 rank 生产
  也走 `simple/mpi/solve_case.py`，`kupl_bistabcg.py` 只作单 rank ABI/数值示例。
- BiStabCG 原生核心是 `lattice_solver.cpp::solve_compact_bistabcg()`；Clover
  `solve_clover()` 完成 Schur prepare/solve/reconstruct/full residual。
  `applyCloverBistabCgQcu` 是直接 ABI；`applyCloverMultigridQcu` 在
  `restart=0` 时转发到同一个 `clover_bistabcg()`。当前缺独立多 rank CLI，不缺算法。
- MG 稳态必须复用同一 cache，并与同 gauge/RHS、dtype、atol、rank/grid/thread 的
  BiStabCG A/B；MG 负收益时保留 BiStabCG，不得把 `restart=0` 写成 MG。
- `simple/solve_case.py` 和 `simple/mpi/solve_case.py` 都在内部启动 `mpirun`；
  外层再次运行 `mpirun` 会形成递归 MPI，直接失败。

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

### 鲲鹏全配置矩阵（2026-11-15）

入口 `pyqcu/testing/kupl/test_kunpeng_all.py` + `submit_kunpeng_all.sh`。矩阵为
`dtype{c64,c128} × solver{bistabcg,mg1,mg2} × lat{L24,L32} × grid{balanced,t-finest}
× ranks{16,32} × threads{9,18,36}(≤608 总线程) × shm{0,1} ×
source{ones,real1,point,random} × gauge{real,random} × seed{42,24} ×
mg-restart{5,10,20,50}`；每行写 94 列 CSV（物理参数、rank×thread、grid、cache
identity、三层残差、`*_read_mode` 与读时间、`dslash/clover GB/s`、节点频率、
`wall/builder/build_mg_cache/elapsed`）。单条用例走
`--mode run --lat 24 24 24 64 --mass -0.277 --csw 1.160920226 --dtype c64`，便于按
节点分组独立提交。列定义、结果表与 runbook 见
[`references/kupl-kunpeng-matrix-20261115.md`](references/kupl-kunpeng-matrix-20261115.md)。

- 模式 `list/prepare/prewarm/solve/merge/run/validate`；`solve --expand` 配
  `--shard-index/--shard-count` 分片。`--resume --skip-failed` 会**永久跳过**已有
  `status=FAIL` 的单元，且判重只看 `outdir/rows/*.json`（不看 `--csv`）：重跑必须
  删行、换 `--run-label` 或换 `--outdir`。
- CSV 94 列由 `CSV_COLUMNS` 决定；`append_csv` 只在空文件写表头，旧 93 列 CSV 再
  追加会整体错列（`--mode merge` 重写可修）。`rows/*.json` 走 `atomic_write_json`。
- 驱动自启 `mpirun`，故一个 `dsub` 作业必须是**一个任务独占整节点**（`-N 1 -rpn 1
  -R "cpu=ranks*threads;mem=400G" -x job`）。`-x job` 在节点被部分占用时保持
  `PENDING`，因此一作业一节点；`#DSUB -T` 单位是秒（默认 900），资源块只在
  `dsub -s` 时解析。调度细节见 `pyqcu-testing-qcu`。
- 口径：L24 c64（ones、18 it、真残差 `7.12e-07`）32r×18t t-only `0.247571 s`
  （shm1 `0.240473`）、balanced `0.349715`；L24 c128（31 it、`5.33e-11`）同配置
  `0.567444/0.475555 s` = c64 的 `1.98–2.29×`（迭代数占 `1.72×`）；L32 6 行全
  FAIL（缺一次有效运行）。16 秩主要落 `cn19650`（17/20 行）、32 秩主要落 `cn19699`
  （15/17 行），节点与 grid 共线，**不能当 scaling 曲线**（16×18 的 8 行见矩阵参考 §5）。
- L32 是**源文件**故障而非算法：32 rank 都在 `_read_local_slab` 的 `h5py.File` 抛
  `bad object header version number`。16:28 已用 `ones`/`ones_v2`（同 `9cdcfc82…f55a`、
  全 `1+0i`）换掉坏 ones；`source_..._c64.h5` 是 **random seed42**，不可当 ones 用。
- MG + `shm=1` 边界：8 格 7 PASS / 1 FAIL，唯一 FAIL 是 32 秩 balanced `2 2 2 4`
  + `restart=5` 的 `shared halo handshake timeout: publish`；同 config 独立 outdir 复测
  2/2 PASS → **1 FAIL / 2 PASS，属不稳定而非稳定失败**：不写 `known_skips.json`（只留
  `known_skips.candidate.json`，default disabled），不得整体禁 `balanced` 或 `shm1`、
  不得静默回退 `shm=0`。签名只在 `logs/*.err`，见
  [`docs/kupl_l24_c64_mg2_shm1_boundary_20261115.md`](../../docs/kupl_l24_c64_mg2_shm1_boundary_20261115.md)。
- 启动摊销：计算节点单次启动 = `MPI boot 38.993 s + import 91.716 s ≈ 130.7 s`
  （16 rank、`cn12090`；矩阵行内 import 记账 `80.8 s`）；登录节点 import 仅
  `1.0–1.3 s`。一例一进程时 `elapsed_seconds` 16 秩 ≈`100 s`、32 秩 `183–211 s`，
  而 `wall` 仅 `0.21–0.57 s`。loop 原型在同一 `mpirun` 内重复调用驱动：计算节点
  3 例同进程 `C=113.9 s`（对逐例 `3×130.7 s` 摊销 ~261 s），登录节点 3 例 `107.4 s`
  但 import 只有 1 s、摊销不可见；故障注入/`SystemExit` 不污染后续用例，RSS
  589→925 MB、wall 抖动，正式批次仍一例一进程。
- `KUPL_HDF5_MPIO_READ=1` 是集体 opt-in（rank 间不一致或 h5py 无 MPI 立即抛错）。
  本轮只做**接口检查**：conda h5py `3.7.0`/HDF5 `1.10.6`、`get_config().mpi=False`、
  无 `mpio` → per-rank 读、全批 `hdf5_mpio_read=false`（0 行 `true`）、ranked cache
  `SKIP`；**不重编译/不重装 MPI h5py，ranked cache 不列入后续计划**。cache 策略与
  identity 字段见
  [`references/kupl-kunpeng-matrix-20261115.md`](references/kupl-kunpeng-matrix-20261115.md) §8。

### 矩阵范围裁定与重测判定（2026-11-15）

- **不重测判据**：source 语义与 hash、gauge dtype 与 `gauge_canonical_sha256`、cache
  identity、`atol`、`grid/ranks/threads/shm/solver/mg_restart` 全部同口径才不重测；
  任一不同必须重跑并落新 `run_id`。据此 **31 条 PASS / L24 数据保留、不重跑**；
  L32 的 6 条 FAIL 是**首次有效运行**，不是重测。
- **放弃轴**（耗时代价过大，已有部分数据保留，**不重测/不新增**）：thread 9、seed24、
  random gauge、random 源、point 源、`mg-restart 50`；其单样本（如
  `861ac912ac3c7c42` 的 `11.437897 s`）不得进入任何性能表。保留 16×36、MG1、r10、
  L32/c128、MG+shm。
- **cache**：`--mode prepare` 只处理 `mg2`；命中看 `cache_index.json` + manifest
  identity。identity 不含 source 哈希（source 修复不失效）与 `atol/restart/deflate`；
  L32 c64 与 L24 c64 的 `gauge_file_sha256` 指到 c128 大文件，只需重签（canonical
  不变、不重跑数值），L32 c128 不重建。
- **超时只留 partial 输出**：600 s 硬超时（rc 124/137）的 cache 行可能已落盘并判
  `PASS`（`elapsed=602.2 s`），不得当作"L32 可跑"证据；solve 行无 `reports/*.json`
  一律 FAIL。输入资产按**只读**处理：不回写、不修补损坏文件，只能换名重生成并保留
  `*.corrupt-*` 备份。

### L24 单节点最优点与带宽口径（2026-11-13）

固定用例（除注明外）：L24 `24^3x64`、c64、单点源 `seed 42`、`mass=-0.277`、
`csw=1.160920226`、`atol=1e-4`、`levels=1`、`restart=0`（parity-Schur BiCGStab，
不是 multigrid）、grid `1 1 1 32`、`-R cpu=TH` 决定每 rank 线程数、
`KUPL_NUMA_BIND=1`、`preferred`、`mpirun --bind-to none --map-by ppr:2:numa`，另加
PACK6/DEDUP_PACK/`T_LOCAL_IN_PART1`/BOUNDARY_LANES/`KUPL_SKIP_FINAL_RESIDUAL`/
`OMP_WAIT_POLICY=ACTIVE`。

- 8+8 样本：`t18` + 共享 halo（单节点默认）中位 `0.154883 s` 对 `KUPL_SHM_HALO=0`
  `0.181733 s`，改善 `14.77%`、区间不重叠；机制是 wait `0.775 → 0.164 ms/次`、
  dslash 中位 `0.067483` 对 `0.093351 s`。两组均 12 it、真残差 `5.384866e-05`。
  跨节点共享窗口自动回退 MPI；最优中位距 `0.1 s` 目标约 `1.55×`。
- 全格实部 1 源 3 样本：L24 中位 `0.173362 s`（11 it、`6.756e-05`）、L32 中位
  `0.348549 s`（12 it、`6.832e-05`）；TH9/16/18 扫描两尺寸均 TH18 最优、TH36 失败
  关闭。**源不同，不得与 `0.154883 s` 算比值**。
- `KUPL_PROFILE_OPT=1`（默认关、仅计时）：rank 17 的 dslash `0.104099 s/52`、
  `828.113 MB`、`7.96 GB/s·rank`（聚合 `254.56`）；clover `0.013720 s/52`、
  `552.075 MB`、`40.24 GB/s·rank`；reduce `0.023190 s/51`；phases
  `prepare/solve/reconstruct = 0.003573/0.156956/0.002327 s`。
- 同节点 STREAM triad 探针 `18.3–20.1 GB/s·rank`（32 rank `585–643 GB/s`）：
  dslash ≈ 峰值 `40%`（计算/latency 受限），clover 更高来自 cache 复用。profiler
  字节、解析流量、STREAM、solve 级四种带宽口径不可互除（见 `pyqcu-benchmark`）。

逐用例（`t18+shm`/`t18`/`t16`/`prof4`/`base`/`c50_act`/`c50_base`）、点源重测与
TH 扫描、数据来源（远端 `data/kupl_opt_20261113/remote/<tag>/`，C 级时点证据、
未提交）见 [`references/kupl-history-2026.md`](references/kupl-history-2026.md)。

### L24 Legacy-MG cache（2026-11-14）

1-rank 构建、32 rank 复用已打通（仅限**已有预置 cache** 的场景）。cache
`data/kupl_scaling_20261114/cache_l24_parallel.h5`（+ 同名 `.json`）：
`setup_seconds=493.63817879895214`（1 rank × 18 线程、并行 stencil 默认开）、
`file_size_bytes=143,335,424`（约 `136.7 MiB`）、`status=PASS`、
`mg_grid=[2,2,2,1]`、`nv_iters=2`、lattice `[24,24,24,64]`。构建期修复（删
per-probe 全量 fill、第一步 dslash 稀疏到 `block+1-hop`、时间窗口
`min(fine,3*block+3)`）把 1-rank 的 `rc=124`（>600 s）降到 493.6 s，仍比 `30 s`
目标慢约 `16.5×`。

**容差语义**：`coarse_atol = atol × coarse_tol_factor × 3^(levels−1)`；`atol=1e-4`
× 默认 `1e4` = `1.0` → 粗层首行即“收敛”、修正全拒、`fallback_to_fine=true`。
因此 `atol=1e-4` 的 `0.135579–0.136258 s` 与历史“MG 快 21.8%/26%”**一律作废**
（实为 fine BiCGStab）。真实 MG 必须 `atol=1e-6`：32 秩 t-only `0.252589` 对
BiCGStab `0.247571 s`（±2%），`restart=5→20` 省 `1.8–2.4×`；`builder_seconds`
（cache 载入 `4.4–6.1 s`）不在 `wall_seconds` 内，MG 的 `iterations` 为 `None`，
不得按迭代数比较 MG 与 BiCGStab。第二批 `atol=1e-6` 扫描与全矩阵见
[`references/kupl-kunpeng-matrix-20261115.md`](references/kupl-kunpeng-matrix-20261115.md)。

### 多 rank Legacy-MG patch 构建（2026-11-14）

**旧路径根因**：`_periodic_window(gauge_local, …)` 把 rank 局部 slab 周期平铺成
patch；4-rank `8³×16` 实测 **75%（6144/8192）站点 gauge 错误**，导致
`hnn=0.493 / hdg=0.787 / sit=0.039`。两次逐位实验（生产窗口→复现失败 cache；
真实全局窗口→复现 1-rank 参考）闭合因果。**不是 C++ slot/halo 语义问题。**

**正确算法（照抄 CUDA-QCU patch 语义）**：
- patch 每轴 `extent = local + 4*mg`（`halo=2*mg`）；
- `start = (coord*local - 2*mg) % global`；
- 若 `local + 4*mg > global` 且 `global % mg == 0`，退化为整轴（`start=0`）；
- patch 用 **serial-local `1×1×1×1`**（`node_rank=0,node_size=1`）调用单 rank
  builder，截 `coarse_center = 2*mg/step`（t 轴 `1`）的中心 block；
- patch gauge 必须来自**全局** HDF5/全局数组/slab gather，禁止局部平铺；
  保持 `patch_galerkin=True` 作为 halo 门。

**32-rank 实测（全部 `verify_status=0`）**：

| 格点 | grid | `builder_seconds` | 真残差 |
|---|---|---:|---:|
| `24³×64` | `2×2×2×4` | **24.476 s** | 4.535e-07 |
| `32³×64` | `2×2×2×4` | **53.023 s** | 4.543e-07 |
| `32³×12` | `4×2×2×2` | **11.368 s** | 4.460e-07 |

**t-only 禁用**：`grid 1 1 1 32` 时 x/y patch 退化为整轴，每 rank 建全格
（≈493 s），32 rank 仍 `rc=124`；多 rank MG 一律均衡 4D 分解。

**MG 容差语义**：`coarse_atol = atol × coarse_tol_factor × 3^(levels−1)`。
`atol=1e-4` + 默认 `factor=1e4` → `coarse_atol=1.0` → 粗层解恒 0 → 修正全拒 →
回退纯 BiCGStab；**历史 `1e-4` 的 `0.128 s` 不是 MG**。同网格 1e-6 对照：
BiCGStab `0.0316 s` vs MG `0.0379 s`（MG 慢 20%，非 2.93×）。
32-rank MG 1e-6 回读：L24 `0.519831 s`（`2×2×2×4`）、L32 `1.175139 s`
（`2×2×2×4`）、`32³×12` `0.266999 s`（`4×2×2×2`），均 `verify_status=0`、
真残差约 `4.5e-07`。

**旧分布式路径**：`StencilFrame` 的 push/pull slot 混叠、pull cell 索引、
压缩时间 hop `±1`、3 维 corner 缺失均未修；生产禁用
`KUPL_MG_STENCIL_FORCE_DISTRIBUTED=1`。

## 鲲鹏 HBM 高性能配置（2026-11-12）

L24 独占 HBM 节点实测配方：所有 rank 报告 `memory_bound=true`，`memory_node`
落在 16–31；逐用例超时用 400–600 s（16 秩端到端约 108–117 s，32 秩约 233 s，
`RC=124` 不等于死锁）。本地实测数据、DSLASH 分解与跨节点注意事项见
[`references/kupl-history-2026.md`](references/kupl-history-2026.md)，逐项数值与来源
对照见 [`references/kupl-hbm-scan-20261112-evidence.md`](references/kupl-hbm-scan-20261112-evidence.md)。

| 配置项 | 推荐值/说明 |
|---|---|
| `KUPL_NUMA_BIND=1` | 按 sysfs cpulist 绑定；MPI 必须 `--bind-to none`，不得再设 `OMP_PROC_BIND`/`OMP_PLACES` |
| `KUPL_MEMORY_POLICY=bind` | 严格 HBM 策略；`preferred` 会回退 DDR |
| `KUPL_MEMORY_NODE=local+16` | 仅当 sysfs 片上内存节点对每个 CPU 节点唯一映射时使用 |
| `KUPL_REQUIRE_MEMORY_BIND=1` | `set_mempolicy` 失败即 fatal，避免静默回退 |
| `KUPL_HUGEPAGE=1` / `KUPL_TOPOLOGY_REPORT=1` | 前者对大型内部 scratch 发 `MADV_HUGEPAGE`，后者输出每 rank cpuset/node/thread 证据 |

MPI 使用 `--map-by ppr:P:numa` 与 `--bind-to none`；`dsub` 资源参数写在命令行，或
用 `dsub -s <job>` 让 `#DSUB` 资源块生效（直接以位置参数提交时 `#DSUB` 头可能被
忽略），提交后用 `djob -ll <id>` 核对 `execNodes`。

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
- 不要把 clover 的 `40.24 GB/s·rank` 与 STREAM triad 探针峰值直接比较：clover 有
  cache 复用、探针不命中缓存；也不要跨层相除（见 `pyqcu-benchmark` 带宽四级口径）。
- 不要把旧 `KUPL_PROFILE_DSLASH` 的 halo/wait/boundary 分解当作生产 split-boundary
  路径的分解（旧实现用 `!profile` 跳过该路径）。生产路径就地计时：作业 `2114720`
  rank 0 为 halo `0.007918`、interior `0.018824`、wait `0.022757`、
  boundary `0.008081 s`（28 次），wait 仍占约 40%。
- 不要用单一节点的绝对值跨节点外推：同为 PASSIVE 基线，`cn19650` `0.319600 s` 对
  `cn19699` `0.266416 s`（约 `20%`），硬件/负载/cpuset 未逐项对齐，只能作上界。
- 不要据单次样本宣称 `-ffast-math` 或 block-dslash 加速：cn19699 同 lane 上
  `0.204288 s` 对 `0.207077 s`（约 1.4%）、cn19650 `0.244864 s` 对
  `0.249223 s`（约 1.8%）均在噪声级。
- 不要用 `wrapped_site = site` 处理多 block 的 +z 周期；z-block 累加必须给
  内点/上界 wrap 传目标 lane 偏移；时间 backward gauge 在 backward 远端时
  取 halo。
- 真实 MG（`levels=2`）多 rank 仍必须预置全局 `--coarse-cache`；`kupl_multigrid.py`
  的多 rank 导出仍以 `multi-rank coarse-cache export is not implemented` 拒绝。
  1-rank 构建的 L24 cache 已被 32 rank 复用（`setup_seconds=493.63817879895214`、
  `143,335,424 B`），但 cache 构建仍 `493.6 s`（>30 s 目标）且 L32 因源文件不可读
  没有可复用 cache。早前作业 2112385 的 25 分钟 `dkill` 属旧构建器失败，不是算法
  结论。

## 历史性能与运行时证据

2026-10 至 2026-11 的 cache builder、Clover 6×6、rank scaling、鲲鹏运行时、HMPI 记录见
[`references/kupl-history-2026.md`](references/kupl-history-2026.md)；与 CUDA 的算法取向
对照与“不可照搬”清单见
[`references/kupl-vs-cuda-algorithms.md`](references/kupl-vs-cuda-algorithms.md)；节点
频率/拓扑核验记录见 [`docs/kupl_kunpeng_node_survey_20261114.md`](../../docs/kupl_kunpeng_node_survey_20261114.md)。
