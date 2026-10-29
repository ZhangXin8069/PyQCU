# KUPL CPU 后端鲲鹏运行时与性能指南

本文是 `cpp/kupl/qcu` 在鲲鹏服务器上的构建、MPI/OpenMP 拓扑、NUMA/内存绑定、
测试和性能证据指南。通用 Linux、x86_64 与其他 ARM64 平台仍使用同一套 C++17 ABI。
纯 KUPL、不编译 CUDA 的端到端安装和运维流程见
[`kupl_only_deployment.md`](kupl_only_deployment.md)。

> 名称说明：本文的 KUPL 指 PyQCU 的独立 CPU backend。它不是 Kunpeng HPCKit 中
> 同名的 KUPL 并行库；PyQCU 不链接 HPCKit KUPL/KML，也不要求安装 HPCKit。

## 1. 兼容边界

保底运行条件：

- C++17、MPI 3.1 或更高版本、OpenMP；
- Linux、x86_64 或其他 ARM64；
- 仅使用标准 C++ 容器、MPI、OpenMP 和 Linux 可选运行时接口；
- 不强制依赖 hwloc、libnuma、numactl、HMPI、UCX、KML 或 HPCKit。

鲲鹏增强路径是兼容性之上的显式 opt-in：

- Linux sysfs 可用时读取真实 NUMA cpulist；
- 按完整 NUMA 域而不是连续 CPU 编号切分 rank；
- 可选设置 CPU affinity 和单节点内存策略；
- 可选对大 scratch `madvise(MADV_HUGEPAGE)`；
- 无 sysfs、memory-policy 权限或非 Linux 时保留 MPI/OpenMP 基准路径。

严格 profile 可由 `PYQCU_KUPL_PROFILE=kunpeng`/`kunpeng-2numa` 或
`KUPL_REQUIRE_TOPOLOGY=1` 启用。鲲鹏 profile 会校验 Linux ARM64 主机身份及
Kunpeng/HiSilicon CPU 标识；x86 或普通 ARM64 直接入口会失败。拓扑缺失、rank
跨 NUMA、profile/topology 开关在 rank 间不一致或线程超订时必须失败，不得静默
降级为“看起来成功”的性能运行。

## 2. 构建

通用可部署构建关闭本机 ISA 特化：

```bash
PYQCU_KUPL_BUILD_DIR="$PWD/cpp/kupl/qcu/build-generic" \
KUPL_ENABLE_NATIVE=OFF KUPL_ENABLE_LTO=OFF \
bash cpp/kupl/qcu/make.sh
ctest --test-dir cpp/kupl/qcu/build-generic --output-on-failure
```

需要同时验证 Python ctypes 时使用：

```bash
PYQCU_KUPL_BUILD_DIR="$PWD/cpp/kupl/qcu/build-generic" \
KUPL_ENABLE_NATIVE=OFF KUPL_ENABLE_LTO=OFF \
bash ./install.sh kupl
ctest --test-dir cpp/kupl/qcu/build-generic --output-on-failure
```

目标鲲鹏节点的性能构建：

```bash
PYQCU_KUPL_BUILD_DIR="$PWD/cpp/kupl/qcu/build-kunpeng" \
KUPL_ENABLE_NATIVE=ON KUPL_ENABLE_LTO=ON \
bash ./install.sh kupl
ctest --test-dir cpp/kupl/qcu/build-kunpeng --output-on-failure
```

`KUPL_ENABLE_NATIVE=ON` 生成的二进制只保证与构建节点兼容；登录节点和计算节点 CPU
型号不同时应使用通用构建，或分别构建。`KUPL_ENABLE_LTO=ON` 启用 CMake IPO，并在
编译器支持时加入 `-fno-semantic-interposition`。不要启用 `-ffast-math`，它会破坏
Clover 相位和复数计算所需的数值语义。

HPCKit 环境若使用 Bisheng/HMPI/UCX，编译器、MPI、UCX 和运行时应来自同一套
HPCKit 安装树。禁止把系统 UCX 与 HPCKit UCX、或两套 HMPI 启停脚本混用。

## 3. 运行时参数

| 环境变量 | 取值 | 默认 | 作用 |
|---|---|---|---|
| `KUPL_OMP_THREADS` | 正整数 | 初始 affinity 大小 | rank 内 OpenMP 线程数，显式值精确采用 |
| `OMP_NUM_THREADS` | 正整数 | 与 rank cpuset 有关 | 建议与 `KUPL_OMP_THREADS` 相同 |
| `OMP_DYNAMIC` | `FALSE` | 库强制 `FALSE` | 生产运行固定线程数 |
| `KUPL_NUMA_BIND` | `0/1` | `0` | `1` 由库选择 NUMA-local CPU 并尝试内存策略 |
| `KUPL_CPU_NODE` | 非负节点号 | 自动轮转 | 强制 rank 使用指定 CPU NUMA 节点 |
| `KUPL_MEMORY_NODE` | `local`、`N`、`local+N` | CPU 节点 | 指定后续库分配的 NUMA 内存节点 |
| `KUPL_MEMORY_POLICY` | `preferred/bind/default` | policy 启用时 `preferred` | preferred 可溢出，bind 严格本地 |
| `KUPL_REQUIRE_MEMORY_BIND` | `0/1` | `0` | memory policy 失败时从告警改为失败 |
| `KUPL_REQUIRE_TOPOLOGY` | `0/1` | 由 profile 决定 | 要求每 rank 恰好绑定一个 NUMA 域 |
| `KUPL_ALLOW_OVERSUBSCRIBE` | `0/1` | `0` | 显式允许线程数大于 cpuset CPU 数 |
| `KUPL_TOPOLOGY_REPORT` | `0/1` | `0` | 每个 rank 输出实际 NUMA、cpuset、线程和内存策略 |
| `KUPL_HUGEPAGE` | `0/1` | `0` | 对大型内部 scratch 请求 THP advice |
| `KUPL_FUSED_BICG_DOTS` | `0/1` | `1` | BiCGStab `(t,t)` 与 `(t,s)` 成对归约；`0` 为 A/B 回退 |
| `KUPL_DISABLE_CLOVER_CACHE` | `0/1` | `0` | 每次刷新 gauge halo/Clover inverse；用于原地修改 gauge |

`KUPL_NUMA_BIND=1` 与 MPI/OpenMP 自身绑定互斥：

- 必须使用 `mpirun --bind-to none`；
- 必须取消 `OMP_PROC_BIND`、`OMP_PLACES`、`GOMP_CPU_AFFINITY` 和
  `KMP_AFFINITY`；
- 调度器已提供 NUMA-local cpuset 时保持 `KUPL_NUMA_BIND=0`。

`solve_case.py` 和 benchmark 启动器在用户显式指定 `KUPL_MEMORY_POLICY` 时自动设置
`KUPL_REQUIRE_MEMORY_BIND=1`，避免容器 seccomp 或系统权限让策略静默退化为默认分配。
若确实允许退化，应直接运行库并显式覆盖该变量，同时保留
`KUPL_TOPOLOGY_REPORT=1` 的 stderr 证据，其中 `memory_bound=false`。结构化
solve JSON 的 `ranks[]` 记录请求环境，不能替代该实际运行证据。

`KUPL_CPU_NODE` 允许不同 local rank 指定不同节点，但同节点共享通信域内必须全部
设置或全部留空；库会按选择同一节点的 rank 数再次平分该节点 CPU，而不是按全局
local rank 简单切片。

库内绑定发生在 `applyInitQcu`。它不能迁移 Python、NumPy、LIME 或 HDF5 在调用前
已经 first-touch 的页。KUPL 的低层 C++ benchmark 已先 init 再分配基准场；
`benchmark_kupl_lime_mpi.py` 与 `kupl_multigrid.py` 在 `KUPL_NUMA_BIND=1` 时会在
读取 gauge/source 和分配 coarse 资产前执行一次临时 prebind lifecycle，并在报告中
写入 `input_prebind_seconds`。自定义调用方仍须遵守同一顺序。

默认 Clover cache 以 gauge 指针身份判定复用，因此同一 `set_ptrs` 生命周期内 gauge
内容必须保持不变。若调用方原地更新 gauge 或复用了同一地址，必须设置
`KUPL_DISABLE_CLOVER_CACHE=1`；该模式会牺牲热缓存性能换取正确性。

## 4. 推荐放置模式

### 4.1 调度器/HMPI 先绑定

这是生产首选，尤其适合 Hyper MPI。每个 rank 在 Python import 和大张量分配前获得
一个 NUMA-local cpuset：

```bash
unset OMP_PROC_BIND OMP_PLACES GOMP_CPU_AFFINITY KMP_AFFINITY
export KUPL_NUMA_BIND=0
export PYQCU_KUPL_PROFILE=kunpeng-2numa
export KUPL_OMP_THREADS=24
export OMP_NUM_THREADS=24
export OMP_DYNAMIC=FALSE

mpirun --report-bindings --map-by numa --bind-to numa -np 2 \
  python -B pyqcu/testing/simple/mpi/benchmark_kupl_lime_mpi.py ...
```

每 NUMA 多个 rank 时不能用 `--bind-to numa`，否则每个 rank 都得到整个 NUMA
cpuset。必须提供精确 rankfile：

```bash
mpirun --report-bindings --rankfile kunpeng.rankfile -np 4 ...
```

规划器的 scheduler 模式在 `ranks_per_numa>1` 且未给 `--rankfile` 时直接失败；
不会生成看似可用但 cpuset 重叠的计划。

`--report-bindings` 只是 MPI 声明，不是最终证据；正式运行还要记录
`KUPL_TOPOLOGY_REPORT=1` 给出的实际 cpuset。

默认不要在短 OpenMP 并行区设置 `OMP_PROC_BIND=close/spread`。只有在固定 cpuset、
独占节点和同题 A/B 下才测试显式 OpenMP binding；仓库曾记录该设置在 x86 短并行区
造成约 8.5 倍回退。

### 4.2 库内绑定回退

调度器未绑定 CPU 时：

```bash
export KUPL_NUMA_BIND=1
export KUPL_MEMORY_POLICY=preferred
export KUPL_TOPOLOGY_REPORT=1
export KUPL_OMP_THREADS=24
export OMP_NUM_THREADS=24
unset OMP_PROC_BIND OMP_PLACES GOMP_CPU_AFFINITY KMP_AFFINITY

mpirun --bind-to none -np 2 \
  cpp/kupl/qcu/build-kunpeng/kupl_qcu_bench \
  --lattice 16 16 16 32 --grid 2 1 1 1 --iterations 10
```

在 HMPI 上，`--bind-to none` 可能使部分集合通信拓扑算法回退。因此该模式是兼容
回退，不应优先于调度器绑定。

### 4.3 显式高带宽内存节点

鲲鹏 920 专业版的 Flat on-chip memory 模式会把 DDR 与片上内存暴露成不同 NUMA
节点。具体节点号随机型、BIOS 和节点配置变化，必须用 `numactl --hardware` 或
sysfs 现场确认。不能把示例中的 offset `16` 当作所有鲲鹏平台的常量。

确认 `本地CPU节点 N` 对应高带宽节点 `N+16` 后：

```bash
export KUPL_NUMA_BIND=1
export KUPL_MEMORY_NODE=local+16
export KUPL_MEMORY_POLICY=preferred
export KUPL_REQUIRE_MEMORY_BIND=1
```

`preferred` 在片上内存耗尽后允许溢出；`bind` 严格限制在该节点，容量不足会失败。
若容器 seccomp 禁用 `set_mempolicy`，日志会明确报告失败，不能用 `memory_bound=true`
伪装成功。

Python/NumPy 输入数组通常大于片上内存，且可能在 `applyInitQcu` 前完成分配。此时
应优先用调度器/`numactl -m/--preferred` 完成进程级放置，或改造调用方在绑定后
分配；仅设置 `KUPL_MEMORY_NODE` 不会迁移现有页。

### 4.4 大页

专用节点可按厂商系统手册配置 2 MiB/1 GiB 大页，并在性能 A/B 中确认
`/proc/meminfo`、`numastat -m` 和 TLB 指标。PyQCU 的 `KUPL_HUGEPAGE=1` 只对
内部大型 scratch 发送 `MADV_HUGEPAGE` advice，不修改系统配置，也不保证实际使用
大页。通用构建默认关闭。

### 4.5 Clover cache 与容量估算

单个 Clover parity 的 12×12 onsite 矩阵大小为：

```text
B_clover = 12 * 12 * half_sites * sizeof(complex<Real>)
```

其中 `half_sites = X*Y*Z*(T/2)`。当前 lazy cache 在只使用一个 parity 时驻留
`B_clover`（原位构造并求逆的目标 parity cache）；两个 parity 都使用后最多
`2*B_clover`。修改前实现会在 init 时一次分配 `4*B_clover`，即两个 term 加两个
inverse。

例如 24³×64、`half_sites=442368`：

| 精度 | 单块 | 单 parity 当前 | 两个 parity 当前 | 旧 init 分配 |
|---|---:|---:|---:|---:|
| c64 | 486 MiB | 486 MiB | 972 MiB | 1.90 GiB |
| c128 | 972 MiB | 972 MiB | 1.90 GiB | 3.80 GiB |

求解器使用的全场级 scratch 也改为按首次求解惰性分配。Wilson/Clover 单算子 lifecycle
的 init 不再预留 9 块全场级向量和 3 块 parity 向量；从 dslash plan 动态切换到
solver plan 时会在首次求解调用中幂等补齐。因此容量报告必须把 init scratch、
首次求解新增 scratch 和稳态 scratch 分列。

以上只含内部 Clover cache，不含 gauge、调用方 RHS/解、halo、solver 向量和
coarse-MG 资产。容量规划必须再加 Python/NumPy 输入与每个 rank 的 scratch。

## 5. 拓扑规划器

不需要导入 MPI 或 PyTorch 即可检查当前主机：

```bash
python -B pyqcu/testing/kupl/kunpeng_runtime.py inspect --json
```

生成双 NUMA、每 rank 24 线程的调度器绑定计划：

```bash
python -B pyqcu/testing/kupl/kunpeng_runtime.py plan \
  --ranks 2 --threads-per-rank 24 \
  --binding scheduler --mpi hmpi \
  --mpirun /path/to/hmpi/mpirun \
  --program 'python -B pyqcu/testing/simple/mpi/benchmark_kupl_lime_mpi.py ...'
```

生成库内绑定和片上内存计划：

```bash
python -B pyqcu/testing/kupl/kunpeng_runtime.py plan \
  --ranks 2 --threads-per-rank 24 \
  --binding library --mpi openmpi \
  --memory-policy preferred --memory-node-offset 16 \
  --program cpp/kupl/qcu/build-kunpeng/kupl_qcu_bench
```

规划器只打印经过 `shlex` 安全转义的命令和环境，不自动执行。非鲲鹏主机会标记
`profile=generic`；显式请求鲲鹏 profile 会失败。它会先清理继承的 KUPL/OpenMP
放置变量，再显式输出 `KUPL_FUSED_BICG_DOTS=1` 和
`KUPL_DISABLE_CLOVER_CACHE=0` 等默认值；可用
`--no-fused-bicg-dots`/`--disable-clover-cache` 做 A/B。

`--mpi` 不再只是元数据：规划器执行 `--mpirun --version` 探测 launcher，并记录
`mpi_detected`/`mpi_executable`。请求 HMPI 但实际为 OpenMPI 时默认失败；只有
显式 `--allow-launcher-mismatch` 才允许生成差异可审计的 dry-run。
`--allow-oversubscribe` 会同时写入 `KUPL_ALLOW_OVERSUBSCRIBE=1` 和 MPI
`--oversubscribe`。

## 6. MPI 与网络参数

单节点只需 CPU/NUMA 放置，不应设置 UCX 网络变量。多节点运行先执行
`ucx_info -d`，再显式选择计算网卡：

```bash
export UCX_NET_DEVICES=<compute-hca>:1
export UCX_TLS=sm,ud
```

RoCE/IB transport 名称必须以现场探测结果为准。默认不要强制
`UCG_PLANC_UCX_ALLREDUCE_ATTR`；若测试集合算法，固定算法 1/6 分别建立 A/B，
算法 8 的非固定归约顺序不适合作为数值可重复性默认。启动异常时可测试
`--mca coll ^ucg` 或 `--mca osc ^ucx --mca pml ucx`，但均应作为故障定位，
不得写入无条件生产模板。

隔离核节点上，MPI 顺序绑核可能占用 housekeeping CPU。此时使用 rankfile 或
调度器 cpuset 明确列出每 rank 的计算核，并保留现场核隔离配置作为证据。
规划器可直接传 `--rankfile <path>`；此时不再生成 `--map-by`，由 rankfile 的槽位
决定节点和精确 CPU 范围。

## 7. 性能测试

短算子 A/B：

```bash
KUPL_TOPOLOGY_REPORT=1 KUPL_OMP_THREADS=24 \
OMP_NUM_THREADS=24 OMP_DYNAMIC=FALSE \
cpp/kupl/qcu/build-kunpeng/kupl_qcu_bench \
  --lattice 16 16 16 32 --iterations 10
```

Python 基准默认不设置 `OMP_PROC_BIND`：

```bash
python -B pyqcu/testing/simple/benchmark_kupl.py \
  --lat 16 16 16 32 --iterations 10 --repeats 5 --threads 24 \
  --openmp-bind none --numa-bind library --memory-policy preferred \
  --output data/kunpeng_smoke.json
```

若需要把初始化与首次 Clover 延迟从 solver 迭代中剥离，并记录 RSS/HWM：

```bash
PYQCU_KUPL_LIB=/absolute/path/libqcu.so \
KUPL_OMP_THREADS=24 OMP_NUM_THREADS=24 \
python -B pyqcu/testing/kupl/benchmark_runtime.py \
  --lat 24 24 24 64 --dtype c64 --threads 24 \
  --output data/kunpeng_runtime.json
```

该工具先执行 `applyInitQcu`，再分配和首次触页 gauge/source；它把 init、输入准备
和首次 Clover dslash 分开计时。内存结果给出 `before_init`、`after_init`、
`after_inputs`、`after_first_clover` 四个 RSS 阶段、相对 baseline delta 和明确命名
的进程终身 `lifetime_hwm_bytes`。后者包含此前进程历史，不能写成 init 阶段峰值。

正式样本必须同时记录：

- commit、`libqcu.so` 路径与 SHA256、编译器和 MPI/UCX 版本；
- CPU 型号、NUMA cpulist、cgroup/cpuset、大页和 `numa_balancing`；
- `mpirun --report-bindings` 原文和 `KUPL_TOPOLOGY_REPORT=1` 实际 cpuset；
- 格点、grid、rank/thread、精度、mass/csw、atol、迭代和 warmup/repeat；
- Clover setup 与 solve 分开计时，并给 median/MAD；
- 独立 full operator 真残差；native residual 不能替代独立验证；
- 是否发生 NUMA memory policy 降级、HMPI collective fallback 或共享机争用。

同题比较必须使用同一 gauge/RHS SHA、精度、停止条件、rank/thread 和 cache 状态。
不要混用历史 x86、不同鲲鹏型号、不同 HMPI/UCX 版本或不同 NUMA 放置的绝对时间。

## 8. 测试入口

纯运行时规划测试不需要鲲鹏硬件：

```bash
python -B -m pytest -q pyqcu/testing/kupl/test_kunpeng_runtime.py
```

MPI cpuset 不重叠门：

```bash
python -B pyqcu/testing/kupl/test_kunpeng_mpi_binding.py \
  --binary cpp/kupl/qcu/build/kupl_qcu_bench \
  --mpirun mpirun --ranks 2
```

完整 KUPL CTest 中的相关项：

```bash
ctest --test-dir cpp/kupl/qcu/build \
  -R 'kupl_qcu_(kunpeng_runtime_plan|kunpeng_profile_contract|numa_disjoint_binding|numa_explicit_nodes)' \
  --output-on-failure
```

非 Linux、无 sysfs 或可用 CPU 不足时，硬件特化脚本返回 CTest skip code 77；
普通 MPI+OpenMP 数值 CTest 继续运行，不能用 skip 冒充性能 PASS。

## 9. 当前优化与证据边界

当前实现包括：

- 按真实 NUMA cpulist 分配 rank CPU 子集；
- Linux `set_mempolicy` 可选 preferred/bind；
- `KUPL_MEMORY_NODE=local+N` 片上内存映射；
- `MADV_HUGEPAGE` opt-in；
- 大型 Clover inverse/cache 延迟到目标 parity 首次使用，并原位构造/求逆；
- `compute_clover` 以 16 c64/8 c128 site tile 计算并按 component 连续写回，避免
  每 site 向 144 个远距流写出的 TLB/页局部性问题；
- 6×6 Clover 求逆同样按 site tile 连续加载/写回；两组 A/B 中 c64 快
  12.6%/17.4%，c128 快 12.1%/10.8%；
- 缓存版 Clover dslash 使用两个 6×6 自旋块 onsite，避免通用 12×12 结构零；
- 求解器全场级 scratch 按首次使用惰性分配；
- face/corner halo 只在实际 MPI 分解维度存在时分配；1-rank init RSS delta 可降到
  约 2.3 MB；
- BiCGStab `(t,t)` 与 `(t,s)` 成对点积，默认一次归约，可 A/B 回退；
- 可复现的 `KUPL_ENABLE_LTO` CMake 选项。

本仓库在 x86 共享开发机上验证了构建、35 项 CTest、MPI cpuset 不重叠和函数正确性，
但该主机只有一个 NUMA 节点，且容器禁止 `set_mempolicy`。因此它只能证明回退与
测试逻辑，不构成鲲鹏 NUMA、片上内存或 HMPI 性能承诺。历史双 NUMA 记录只保留
作时点证据；正式优化必须在目标鲲鹏节点按本文证据模板重跑。

## 10. 参考资料

- `refer/kunpeng/Kunpeng HPCKit 26.1.RC1 开发指南 01.pdf`：HMPI 映射/绑定、
  rankfile、UCX transport、集合算法和 OpenMP 参数。
- `refer/kunpeng/Kunpeng HPCKit 26.1.RC1 安装指南 01.pdf`：HPCKit 环境一致性、
  HMPI 类型和 OS 支持边界。
- `refer/kunpeng/Kunpeng HPCKit 26.1.RC1 迁移指南 01.pdf`：ARM64/KML 迁移与
  `-march` 示例；本项目不依赖 KML。
- `refer/kunpeng/鲲鹏920专业版系统开发手册.pdf`：NUMA、Flat on-chip memory、
  `numactl`、大页、核隔离与 STREAM 示例。

PDF 中的 608 核、16 DDR NUMA、16 on-chip NUMA、每 NUMA 38 核是示例机型参数，
不是 PyQCU 默认值。

## 11. Strict-MG benchmark 经验（2026-10-29）

- Strict 2-level 的 full-coarse extent 必须逐 rank 保持偶数。`8x8x8x16` 4-rank
  不能使用 `1x4x1x1`；改用 `2x2x1x1` 后 local fine `4x4x8x16`、local coarse
  `2x2x4x8` 才满足几何门。
- solve 结束后先退出 MPI solve 进程，再在独立进程做 full Wilson/Clover 残差。
  在仍持有同一 `COMM_WORLD` 的 rank 0 内重入 distributed dslash 会出现 busy-wait，
  该次不计入样本。
- cache load、wrapper setup、warmup、steady solve 和测后残差分开计时。当前真实
  `8x8x8x16` c64、12 threads/rank、2 warmups + 5 steady 的 1/2/4-rank median
  为 `1.222731/1.101605/1.658620 s`，12 次外层迭代，独立残差
  `7.320e-7/5.917e-7/6.052e-7`。
- 共享登录节点 `load average≈91/48 CPUs`、106 用户时，4-rank 的 `0.737x`
  扩展比只能说明该窗口资源拥塞，不能外推目标鲲鹏性能。正式 HMPI benchmark 必须
  在独占或可审计资源上重跑。
