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

### L24 单节点最优点与带宽口径（2026-11-13）

固定用例：L24 `24^3x64`、c64、点源 `seed 42`、`mass=-0.277`、
`csw=1.160920226`、`atol=1e-4`、`levels=1`、`restart=0`（parity-Schur
BiCGStab，不是 multigrid）、grid `1 1 1 32`、每 rank 线程数由作业申请的
`-R cpu=TH` 决定（32r×12t 的 report `memory.omp_threads` 为 `12`）、
`KUPL_NUMA_BIND=1`、`KUPL_MEMORY_POLICY=preferred`、
`KUPL_REQUIRE_MEMORY_BIND=0`、
`mpirun --bind-to none --map-by ppr:2:numa`；全部另加 `KUPL_HALO_PACK6=1`、
`KUPL_HALO_DEDUP_PACK=1`、`KUPL_T_LOCAL_IN_PART1=1`、
`KUPL_ENABLE_BOUNDARY_LANES=1`、`KUPL_SKIP_FINAL_RESIDUAL=1`（求解器内残差
复查被跳过，外层独立分布式 FULL Wilson-Clover 真残差门仍在跑）。所有数字为
单样本 `wall_seconds`。

| 用例 | 节点 | 规模 | 关键配置 | wall (s) | 迭代 | 状态 |
|---|---|---|---:|---:|---:|---|
| t18 | cn19699 | 32r×18t（`-R cpu=18`） | ACTIVE + t2 lane | **中位 0.186230**（5 次：0.180422–0.193310） | 12 | PASS（真残差 5.384866e-05） |
| t16 | cn19699 | 32r×16t（`-R cpu=16`） | ACTIVE + t2 lane | 0.188526 | 12 | PASS |
| prof4 | cn19699 | 32r×12t | ACTIVE + t2 lane | 0.207077 | 12 | PASS |
| base | cn19699 | 32r×12t | PASSIVE，无 t2 | 0.281166 | 12 | PASS |
| c50_act | cn19650 | 32r×12t | ACTIVE + t2 lane | 0.249223 | 12 | PASS |
| c50_base | cn19650 | 32r×12t | PASSIVE + t2 lane | 0.319600 | 12 | PASS |

`t18` 的独立 FULL 真残差为 `5.384866e-05`、`t16` 为 `5.384853e-05`，其余为
`5.38e-05` 量级。当前最优为 `t18` 的同配置 5 次重复：中位 `0.186230 s`、最小
`0.180422 s`、最大 `0.193310 s`、极差 `6.9%`（全部 12 it、PASS）；单点
`0.183044 s` 比中位低 `1.74%`，只可当作样本区间内的一点。其次是 `t16` 的
`0.188526 s`。t16/t18 与 `prof4` 同时改变了 `-R cpu=TH` 资源申请，不是纯线程数
单变量实验。当前最优点仍高于 `0.1 s` 目标约 `1.86×`。

`KUPL_PROFILE_OPT=1` 是逐 rank 诊断变量（默认关闭，仅计时，不改变算法与停机）。
每个 rank 输出 `dslash`/`clover`/`reduce` 的秒数与调用次数、`dslash`/`clover`
的字节数与 `GB/s`（rank 与全 rank 聚合），以及 `prepare`/`solve`/`reconstruct`
三阶段。ACTIVE、cn19699、rank 17 实测：

| 项 | 次数 | 时间 (s) | 字节 (MB) | 每 rank | 聚合 |
|---|---:|---:|---:|---:|---:|
| dslash | 52 | 0.104099 | 828.113 | 7.96 GB/s | 254.56 GB/s |
| clover | 52 | 0.013720 | 552.075 | 40.24 GB/s | 1287.63 GB/s |
| reduce | 51 | 0.023190 | — | 0.45 ms/次 | — |

phases：`prepare=0.003573`、`solve=0.156956`、`reconstruct=0.002327`；solve 外的
setup 约 `44 ms`（scratch/gauge halo）。

诊断作业 `2114373`/`2114720`（cn19699、32r×18t、ACTIVE + t2 lane、L24 c64）
的 `KUPL_PROFILE_OPT=1`（与 dslash 路径无关）给出 rank 0 的生产 dslash 约
`1.83–1.95 ms/次`（`0.094981`/52 与 `0.101344`/52，各 `828.113 MB`）；
逐 rank `8.2–9.2 GB/s·rank` 为运行时派生值。

带宽口径：同节点 STREAM triad 探针（8M double × 3 数组、5 次取最优、
12 线程/rank、`--bind-to numa --map-by ppr:2:numa`）为 `18.3–20.1 GB/s·rank`，
32 rank 聚合约 `600 GB/s`。据此 dslash 的 `7.96 GB/s·rank` 约为探针峰值的
`40%`，属计算/latency 受限；clover 的 `40.24 GB/s·rank` 高于探针值，来自
clover cache 的复用命中，不是超过硬件峰值。

数据来源为远端 `data/kupl_opt_20261113/remote/<tag>/report.json` 与
`meta.txt`（C 级时点证据、未提交）；逐用例残差、派生比值口径与复现命令见
[`references/kupl-history-2026.md#2026-11-13-l24-单节点性能与带宽剖析`](references/kupl-history-2026.md#2026-11-13-l24-单节点性能与带宽剖析)。

## 鲲鹏 HBM 高性能配置（2026-11-12）

L24 独占 HBM 节点实测配方：所有 rank 报告 `memory_bound=true`，`memory_node`
落在 16–31；逐用例超时用 400–600 s（16 秩端到端约 108–117 s，32 秩约 233 s，
`RC=124` 不等于死锁）。本地实测数据、DSLASH 分解与跨节点注意事项见
[`references/kupl-history-2026.md#2026-11-12-l24-hbm-node-scan-recovery-and-high-performance-configuration`](references/kupl-history-2026.md#2026-11-12-l24-hbm-node-scan-recovery-and-high-performance-configuration)；
逐项数值与来源对照见
[`references/kupl-hbm-scan-20261112-evidence.md`](references/kupl-hbm-scan-20261112-evidence.md)。

| 配置项 | 推荐值/说明 |
|---|---|
| `KUPL_NUMA_BIND=1` | 按 sysfs cpulist 绑定；MPI 必须 `--bind-to none`，不得再设 `OMP_PROC_BIND`/`OMP_PLACES` |
| `KUPL_MEMORY_POLICY=bind` | 严格 HBM 策略；`preferred` 会回退 DDR |
| `KUPL_MEMORY_NODE=local+16` | 仅当 sysfs 片上内存节点对每个 CPU 节点唯一映射时使用 |
| `KUPL_REQUIRE_MEMORY_BIND=1` | `set_mempolicy` 失败即 fatal，避免静默回退 |
| `KUPL_HUGEPAGE=1` | 对大型内部 scratch 发 `MADV_HUGEPAGE` advice |
| `KUPL_TOPOLOGY_REPORT=1` | 输出每 rank cpuset/node/thread 证据 |

MPI 使用 `--map-by ppr:P:numa` 与 `--bind-to none`；`dsub` 调度参数必须写在
命令行（`#DSUB` 头可能被忽略），提交后用 `djob -ll <id>` 核对 `execNodes`。

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
- 不要把 clover 的 `40.24 GB/s·rank` 与 STREAM triad 探针峰值直接比较；clover
  有 cache 复用，探针不命中缓存，两者口径不同。
- 不要把旧 `KUPL_PROFILE_DSLASH` 的 halo/wait/boundary 分解当作生产
  split-boundary 路径的分解；旧实现用 `!profile` 跳过该路径。现已在
  split-boundary 与 t2 z-block 路径就地计时：作业 `2114720` 的 rank 0 为
  halo `0.007918`、interior `0.018824`、wait `0.022757`、
  boundary `0.008081 s`（28 次），wait 仍占约 40%。
- 不要用单一节点的绝对值跨节点外推。同为 PASSIVE 基线，`cn19650`
  `0.319600 s` 对 `cn19699` `0.266416 s`，约 `20%` 差异，硬件/负载/cpuset
  身份未逐项对齐，只能作上界。
- 不要据单次样本宣称 `-ffast-math` 或 block-dslash 加速：cn19699 同 lane 上
  `0.204288 s` 对 `0.207077 s`（约 1.4%）、cn19650 `0.244864 s` 对
  `0.249223 s`（约 1.8%）均在噪声级。
- 真实 MG（`levels=2`）多 rank 目前无法运行：需要预置全局 `--coarse-cache`，
  而 `kupl_multigrid.py` 的多 rank 导出会以
  `multi-rank coarse-cache export is not implemented` 拒绝。1-rank L24 缓存
  构建（作业 2112385）运行约 25 分钟未产出 `cache_l24_v1.h5`，已 `dkill`
  终止；这是工具链阻塞，不是算法结论。

## 历史性能与运行时证据

2026-10 至 2026-11 的 cache builder、Clover 6×6、rank scaling、鲲鹏运行时、HMPI 记录见
[`references/kupl-history-2026.md`](references/kupl-history-2026.md)。
