# 纯 KUPL QCU 安装、测试与生产手册

本文面向只使用 PyQCU KUPL CPU 后端的 Kunpeng/Linux 环境。适用范围是 MPI +
OpenMP 的 Wilson/Clover 算子、Krylov 求解器、Legacy/Strict multigrid，以及
`pyqcu.kupl` 的 `ctypes` ABI。该模式不安装或编译 CUDA QCU，不要求 `nvcc`，
也不构建 `pyqcu.cuda.qcu` Cython 扩展。

更深入的 NUMA、片上内存、HugePage、性能和历史证据边界见
[`kupl_kunpeng_guide.md`](kupl_kunpeng_guide.md)。本文是纯 KUPL 的完整操作入口。

## 1. 运行前提

必需组件：

- Linux、Python 3.10 或更高版本；
- C++17 编译器、CMake 3.20 或更高版本；
- MPI 3.1 或更高版本；
- OpenMP；
- Python 包 `torch`、`mpi4py`、`numpy`、`h5py`；
- 测试另需 `pytest`。

可选组件：

- TileLang 只用于少数 Torch JIT 实验，KUPL CTest 和求解不依赖它；
- HMPI、UCX、NUMA 库、大页和片上内存是 Kunpeng 增强项，不是纯 KUPL 的硬依赖。

以下工具不是纯 KUPL 前提：

- `nvcc`、CUDA runtime、`CUDAToolkit_ROOT`；
- Cython 生成的 `pyqcu.cuda.qcu`；
- QUDA、PyQUDA 或 CUDA 设备。

## 2. 后端构建选择

`setup.py` 使用 `PYQCU_BUILD_BACKEND` 选择 Python 构建路径：

| 值 | 行为 |
|---|---|
| `kupl` | 不创建 CUDA Cython 扩展，不调用 `build-nv.sh` |
| `cuda` | 构建 CUDA Cython 扩展，默认值，保持历史行为 |

`install.sh` 提供对应入口：

```bash
bash install.sh kupl
bash install.sh cuda
```

纯 KUPL 场景必须使用 `kupl`。直接运行不带参数的 `bash install.sh` 仍等价于
`bash install.sh cuda`，因此没有 CUDA toolkit 时再次出现 `Failed to find nvcc`
不是 KUPL 构建失败，而是使用了 CUDA 安装入口。

`PYQCU_SKIP_NATIVE_BUILD=1` 只跳过 CUDA native build 脚本，不会切换到 KUPL，
也不会移除 Cython 扩展，不能作为纯 KUPL 安装方式。

## 3. 快速安装

从仓库根执行：

```bash
source ./env.sh

KUPL_ENABLE_NATIVE=OFF \
KUPL_ENABLE_LTO=OFF \
bash ./install.sh kupl
```

该命令执行：

1. 配置 `cpp/kupl/qcu`；
2. 构建 `kupl_qcu` 动态库以及默认 CTest/benchmark targets；
3. 以 `PYQCU_BUILD_BACKEND=kupl` 运行 `setup.py build_ext`；
4. 通过真实 `ctypes` 加载生成的 `libqcu.so` 做冒烟验证。

默认库路径为：

```text
cpp/kupl/qcu/libqcu.so -> build/libqcu.so
```

`make.sh` 在成功构建后原子更新该软链接；`PYQCU_KUPL_LIB`、Python ctypes 和生产启动器
都默认使用这个稳定入口，而不是直接依赖构建目录名。

目标 Kunpeng 节点可启用本机 ISA 和 LTO：

```bash
KUPL_ENABLE_NATIVE=ON \
KUPL_ENABLE_LTO=ON \
bash ./install.sh kupl
```

启用 `KUPL_ENABLE_NATIVE=ON` 的二进制只保证与当前构建节点 CPU 架构兼容。
需要跨 Kunpeng 型号部署时使用 `KUPL_ENABLE_NATIVE=OFF`。

### 3.1 自定义构建目录

```bash
PYQCU_KUPL_BUILD_DIR="$PWD/cpp/kupl/qcu/build-kunpeng" \
PYQCU_BUILD_JOBS=32 \
KUPL_ENABLE_NATIVE=ON \
KUPL_ENABLE_LTO=ON \
bash ./install.sh kupl
```

安装完成后显式设置库路径：

```bash
source ./env.sh
```

`make.sh` 会把 `cpp/kupl/qcu/libqcu.so` 更新为指向自定义构建目录中的实际库。

只部署运行库、不构建测试程序时可设置：

```bash
KUPL_BUILD_TESTS=OFF bash ./install.sh kupl
```

此模式没有 `kupl_qcu_tests` 和 benchmark executable，不能直接运行完整 CTest。

### 3.2 手工安装

不使用 `install.sh` 时，等价步骤为：

```bash
PYQCU_KUPL_BUILD_DIR="$PWD/cpp/kupl/qcu/build" \
KUPL_ENABLE_NATIVE=OFF KUPL_ENABLE_LTO=OFF KUPL_BUILD_TESTS=ON \
  bash cpp/kupl/qcu/make.sh

PYQCU_BUILD_BACKEND=kupl python setup.py build_ext --inplace
source ./env.sh
```

`make.sh --check` 只打印 build directory、稳定公开库路径和 native/LTO/tests/jobs
选择；`make.sh` 构建 native 库、更新公开软链接和默认测试程序，不创建或修改 Python
Cython 扩展。

需要 editable Python 安装时可使用：

```bash
PYQCU_BUILD_BACKEND=kupl python -m pip install -e . --no-deps
```

`--no-deps` 假设 MPI、PyTorch、NumPy、HDF5 等运行依赖已由站点环境提供。

### 3.3 安装变量

| 变量 | 默认值 | 用途 |
|---|---|---|
| `PYQCU_BUILD_BACKEND` | `cuda` | `kupl` 禁止 CUDA/Cython 扩展路径 |
| `PYQCU_KUPL_BUILD_DIR` | `cpp/kupl/qcu/build` | CMake 构建目录 |
| `PYQCU_BUILD_JOBS` | 当前在线 CPU 数 | 并行编译任务数 |
| `CMAKE_BUILD_TYPE` | `Release` | CMake 构建类型 |
| `KUPL_ENABLE_NATIVE` | `OFF` | 是否加入 `-march=native` |
| `KUPL_ENABLE_LTO` | `OFF` | 是否启用 LTO 和局部符号绑定 |
| `KUPL_BUILD_TESTS` | `ON` | 是否构建 CTest/benchmark；纯部署可设 `OFF` |
| `PYQCU_KUPL_LIB` | `cpp/kupl/qcu/libqcu.so` | 指向当前构建库的稳定软链接 |

## 4. 安装验证

确认 Python 选择的是预期动态库：

```bash
python -B -c \
  'from pyqcu.kupl import qcu; print(qcu.library_path())'
```

验证纯 KUPL Python 构建不会触发 CUDA：

```bash
PYQCU_BUILD_BACKEND=kupl python setup.py build_ext --inplace
```

输出应包含：

```text
PYQCU build backend: kupl
running build_ext
```

不得出现 `build-nv.sh`、`nvcc` 或 `project(qcu LANGUAGES ... CUDA)` 相关步骤。

验证拓扑规划器不会初始化 MPI：

```bash
python -B pyqcu/testing/kupl/kunpeng_runtime.py inspect --json
```

该命令是纯 sysfs/affinity/`mpirun --version` 查询，不应进入 `MPI_Init_thread`。
生成包含 HBM 候选、cgroup、工具链、动态库和推荐变量的完整环境报告：

```bash
python -B pyqcu/testing/kupl/kupl_env.py \
  --ranks 4 --threads-per-rank 32 \
  --lattice 24 24 24 64 --dtype c64 --grid 1 1 2 2
```

默认写入 `logs/kupl_env-vYYYY-MM-DD-HH-MM-SS.txt`。该脚本不导入 `mpi4py`、不启动
solve，也不会调用 `set_mempolicy`。报告中的 `detected_offset` 只表示 sysfs 的
memory-only NUMA 节点与 CPU 节点形成了唯一稳定 offset；要证明进程实际使用 HBM，
仍须使用严格 `bind` 并在运行中检查页分布。

## 5. 旧版 PyTorch 兼容

PyQCU 现在通过 `pyqcu.cann.real_dtype()` 把复数 dtype 映射到实部 dtype，不再依赖
`torch.dtype.to_real()`。该转换覆盖：

```text
torch.complex64  -> torch.float32
torch.complex128 -> torch.float64
real dtype        -> 原 dtype
```

如果环境中存在 `torch.complex32`，则映射到 `torch.float16`；不存在时不会在 KUPL
常量导入阶段访问该属性。

这条兼容规则适用于 `argv` 实数缓冲、Gaussian gauge 生成、Torch/CUDA 辅助路径和
KUPL runtime benchmark。旧版 PyTorch 报错：

```text
'torch.dtype' object has no attribute 'to_real'
```

说明调用到了未更新的代码或第三方脚本，不应通过安装 `torch_npu` 绕过。

缺少 TileLang 时可能看到一次 warning：

```text
RuntimeWarning: TileLang is unavailable; TileLang-specific acceleration and tests will be skipped
```

这是可选依赖的降级信息，不是错误。TileLang 专项测试会 skip，普通 PyTorch CPU
路径会回退；该信息不会影响 KUPL。

## 6. 测试

### 6.1 完整 KUPL CTest

```bash
ctest --test-dir cpp/kupl/qcu/build --output-on-failure
```

自定义构建目录时替换 `build`。完整测试覆盖：

- c64/c128 canonical ABI；
- 1/2/4-rank 四维 halo；
- Wilson/Clover dslash、Krylov 和 Laplacian；
- Legacy/Strict multigrid；
- Python `ctypes` 与 C++ 声明一致性；
- Kunpeng topology/profile/NUMA 契约；
- 纯 KUPL 构建和旧版 PyTorch dtype 契约。

### 6.2 快速定点测试

```bash
python -B pyqcu/testing/kupl/test_kunpeng_runtime.py
python -B pyqcu/testing/kupl/test_pure_kupl_install.py
```

运行 Python KUPL 回归：

```bash
source ./env.sh
python -m pytest -q pyqcu/testing/kupl
```

### 6.3 单独 CTest 过滤

```bash
ctest --test-dir cpp/kupl/qcu/build \
  -R 'kupl_qcu_(pure_kupl_contract|kunpeng_runtime_plan|runtime_benchmark_smoke)' \
  --output-on-failure
```

### 6.4 MPI 与 NUMA 测试

完整 CTest 已包含 1/2/4-rank 和 synthetic NUMA 路径。物理节点还需要记录：

```bash
numactl --hardware
mpirun --report-bindings -np 2 \
  python -c 'import os; print(sorted(os.sched_getaffinity(0)))'
```

测试通过只证明功能或布局契约，不能替代实际 Kunpeng 性能报告。

## 7. 生产运行

### 7.1 准备输入

```bash
python pyqcu/testing/simple/prepare_case.py \
  --lime /path/to/reference.lime \
  --lat 24 24 24 64 \
  --out-dir data/production_L24
```

输出包含 canonical HDF5 gauge、source 和带 SHA256 的 case manifest。

### 7.2 单次生产求解

```bash
source ./env.sh
test -f "$PYQCU_KUPL_LIB"

export KUPL_OMP_THREADS=24
export OMP_NUM_THREADS=24

python pyqcu/testing/simple/solve_case.py \
  --backend kupl \
  --gauge data/production_L24/gauge_24x24x24x64_c64.h5 \
  --source data/production_L24/source_24x24x24x64_c64.h5 \
  --lat 24 24 24 64 \
  --kupl-ranks 2 \
  --kupl-threads 24 \
  --kupl-grid 1 1 1 2 \
  --kupl-lib default \
  --out-dir data/production_L24
```

`env.sh` 默认把 `PYQCU_KUPL_LIB` 指向当前仓库的
`cpp/kupl/qcu/libqcu.so` 稳定软链接，并优先保持当前仓库位于 `PYTHONPATH` 首位。使用其他
构建目录时，在 `source` 前覆盖：

```bash
export PYQCU_KUPL_BUILD_DIR="$PWD/cpp/kupl/qcu/build-kunpeng"
source ./env.sh
```

批处理提交时必须在计算作业内重新 `source` 当前仓库环境；外层单引号不会在提交节点
展开变量，也容易让旧安装树继续占据 `PYTHONPATH`：

```bash
./ddub.sh 'source ~/PyQCU/env.sh && cd ~/PyQCU && \
python pyqcu/testing/simple/solve_case.py \
  --backend kupl \
  --gauge ~/PyQCU/data/production_L24/gauge_24x24x24x64_c64.h5 \
  --source ~/PyQCU/data/production_L24/source_24x24x24x64_c64.h5 \
  --lat 24 24 24 64 --kupl-ranks 4 --kupl-threads 32 \
  --kupl-grid 1 1 2 2 --kupl-lib default \
  --out-dir ~/PyQCU/data/production_L24'
```

不要在该命令中再次传 `--kupl-lib "$PYQCU_KUPL_LIB"`：`default` 会读取
`env.sh` 已导出的绝对值，减少空变量和嵌套引号问题。

`solve_case.py --backend kupl` 会在内部启动 `mpirun --bind-to none`，不要在外层
重复执行 `mpirun`。两个 rank 分域优先切最长时间轴 `t`，但最终必须满足每个 local
维度为偶数。无法得到合法偶数几何时显式传 `--kupl-grid`。

### 7.3 CPU 放置

调度器已经提供 NUMA-local cpuset 时：

```bash
export KUPL_NUMA_BIND=0
unset OMP_PROC_BIND OMP_PLACES GOMP_CPU_AFFINITY KMP_AFFINITY
```

没有调度器绑定时可由库内绑定：

```bash
export KUPL_NUMA_BIND=1
export KUPL_MEMORY_POLICY=bind
export KUPL_REQUIRE_MEMORY_BIND=1
export KUPL_TOPOLOGY_REPORT=1
```

禁止同时让 OpenMPI、OpenMP 和 KUPL 三方绑定同一进程。`KUPL_NUMA_BIND=1` 时 MPI
使用 `--bind-to none`，且不得设置 `OMP_PROC_BIND`、`OMP_PLACES`、
`GOMP_CPU_AFFINITY` 或 `KMP_AFFINITY`。

只有在现场确认 CPU 节点 `N` 对应独立片上内存节点 `N+offset` 后才设置：

```bash
: "${hbm_offset:?set this to kupl_env.py detected_offset}"
export KUPL_MEMORY_NODE="local+${hbm_offset}"
export KUPL_MEMORY_POLICY=bind
export KUPL_REQUIRE_MEMORY_BIND=1
```

`preferred` 在容量不足时会回退 DDR，不能满足“确保使用 HBM”。若本机只有
CPU+内存合一的 NUMA 节点，使用 `KUPL_MEMORY_NODE=local`，不能把 `local+16` 等
示例硬编码；不存在的节点会让 `set_mempolicy()` 返回 `EINVAL`，严格模式下进程应
失败关闭。

### 7.4 产物与验收

生产目录包含：

```text
solution_24x24x24x64_kupl_c64.h5
report_kupl.json
last_solve_kupl.json
```

只有后端报告 `PASS`、真残差满足 `--atol` 且 HDF5 完整写出时，`.partial` 文件才会
原子替换为正式解文件。验收至少记录：

- git commit、库路径和 SHA256；
- 编译器、MPI、UCX、Python、PyTorch 版本；
- CPU 型号、NUMA cpulist、cpuset、线程数、rank 数和四维 grid；
- lattice、mass、csw、atol、精度、迭代数和耗时；
- 独立 full-operator 真残差；
- cache identity、命中状态和 correction 次数（使用 MG 时）。

### 7.5 真正的 Legacy-MG 与高效 BiStabCG

#### 7.5.1 先选择算法路线

`restart` 不是性能档位，而是算法路线：

| 路线 | 正确参数 | coarse hierarchy | 生产用途 |
|---|---|---|---|
| Legacy-MG | `--levels 2 --restart 5` | 会构建/加载 33-tensor 粗算子并执行周期 coarse correction | 评估真实 MG 的迭代和墙钟收益 |
| BiStabCG | `--levels 1 --restart 0` | 不遍历 coarse hierarchy | 稳定的多 rank parity-Schur 基线；当前许多组态上最快 |

`--levels 2 --restart 0` 仍会在数据准备层保留 MG 参数，但实际求解不执行 coarse
correction；它不能作为完整 MG 性能样本。只有 `levels=2`、`restart>0` 且报告中的
`coarse_source` 不是 `unused: restart=0` 时，才是在运行真正的 Legacy-MG。
这里的“真正”表示实际执行 Legacy 粗层 correction，不表示 QUDA-style 或
Strict-MG 的持久 V-cycle/FGMRES。当前 `mpi/solve_case.py` 的 KUPL 生产路线是
Legacy-MG；Strict-MG 需要显式 `pyqcu.kupl.multigrid.StrictMultigrid` 集成或专用
资产/benchmark harness，不能把 Legacy 报告改名为 Strict。

推荐生产入口是 MPI 启动器，不要直接拼 worker 命令：

```bash
python -B pyqcu/testing/simple/mpi/solve_case.py --help
```

该入口会完成输入物化、全局 coarse-cache 构建或复用、`mpirun` 启动、每 rank
environment/affinity 记录、rank 0 汇总和独立 full-operator 真残差。它自己启动
`mpirun`，所以外层不得再次嵌套 `mpirun`。

#### 7.5.2 检测环境并确定绑定

```bash
python -B pyqcu/testing/kupl/kupl_env.py \
  --ranks 4 --threads-per-rank 32 \
  --lattice 24 24 24 64 --dtype c64 --grid 1 1 2 2
```

报告给出 rank/thread/grid、`KUPL_MEMORY_NODE`、是否检测到稳定 HBM offset，以及
MG/BiStabCG 两条可直接改路径执行的命令。推荐基线优先使用“一个 CPU NUMA 一个
rank”；rank 数超过 NUMA 数、Python/NumPy 输入在绑定前 first-touch、或三方同时
绑定都会降低有效带宽。

#### 7.5.3 运行真实 Legacy-MG

以下示例假定现场检测到唯一 HBM offset。先从 `kupl_env.py` 报告设置
`hbm_offset`；没有 HBM 时把库参数改为 `--kupl-memory-node local`。默认
`env.sh` 已设置当前仓库的稳定 `PYQCU_KUPL_LIB`；只有刻意选择其他构建时才显式覆盖。

```bash
source ./env.sh
: "${hbm_offset:?set this to kupl_env.py detected_offset}"

python -B pyqcu/testing/simple/mpi/solve_case.py \
  --backend kupl \
  --gauge data/production_L24/gauge_24x24x24x64_c64.h5 \
  --source data/production_L24/source_24x24x24x64_c64.h5 \
  --lat 24 24 24 64 --dtype c64 --atol 1e-6 \
  --ranks 4 --grid 1 1 2 2 --omp-threads 32 \
  --kupl-numa-bind library \
  --kupl-memory-policy bind \
  --kupl-memory-node "local+${hbm_offset}" \
  --kupl-profile kunpeng \
  --kupl-topology-report \
  --levels 2 --dof 12 12 --mg-grid 2 2 2 2 \
  --restart 5 --coarse-max-iter 200 \
  --coarse-tol-factor 1e4 --nv-iters 2 \
  --work-dir data/production_L24/legacy-mg
```

第一次运行包含 null-vector/Galerkin/cache setup；后续使用同一输入、mass、csw、
dtype、dof、mg-grid、nv-iters 和 seed 时应复用同一身份匹配的 cache。正式性能比较
应把 `builder_seconds` 与稳态 `wall_seconds` 分栏，先用
`--rebuild-coarse-cache` 做一次 cold setup，再在 cache-hit 条件比较 solver。

合格的真实 MG 报告至少应满足：

- `levels=2`、`restart>0`、`coarse_source` 指向实际 cache；
- `verify_status=0`；
- `final_true_residual <= residual_acceptance`，且未使用
  `--skip-independent-residual`；
- `coarse_source`、cache identity、dtype、MPI grid 和 coarse geometry 一致。

MG 不等于“必然更快”。必须与同 gauge/RHS、dtype、atol、ranks/grid/threads 的
BiStabCG 基线比较；若 coarse correction 增加墙钟或未减少有效迭代，应保留
BiStabCG 生产路线并在报告中记录 MG 负收益，而不是修改残差或把 `restart=0`
改写成 MG。

#### 7.5.4 运行高效多 rank BiStabCG

`kupl_bistabcg.py` 是单 MPI rank 的 ctypes 示例，不是多 NUMA 生产入口。多 rank
生产求解使用同一 MPI 启动器并显式关闭 coarse hierarchy：

```bash
: "${hbm_offset:?set this to kupl_env.py detected_offset}"
python -B pyqcu/testing/simple/mpi/solve_case.py \
  --backend kupl \
  --gauge data/production_L24/gauge_24x24x24x64_c64.h5 \
  --source data/production_L24/source_24x24x24x64_c64.h5 \
  --lat 24 24 24 64 --dtype c64 --atol 1e-6 \
  --ranks 4 --grid 1 1 2 2 --omp-threads 32 \
  --kupl-numa-bind library \
  --kupl-memory-policy bind \
  --kupl-memory-node "local+${hbm_offset}" \
  --kupl-profile kunpeng \
  --kupl-topology-report \
  --levels 1 --restart 0 \
  --work-dir data/production_L24/bistabcg
```

该路线执行优化的 parity-Schur BiStabCG，不构建 coarse cache。保持
`KUPL_FUSED_BICG_DOTS=1` 作为默认；只有同题 A/B 稳定改善时才改用 `0`。
`simple/solve_case.py --backend kupl` 也能运行同一多 rank 路线，但它同样在内部
启动 `mpirun`，外层不要再执行 `mpirun`。

实现层级需要与 CLI 名称区分：

- 独立 Clover BiStabCG 核心是 `cpp/kupl/qcu/src/lattice_solver.cpp` 中的
  `solve_compact_bistabcg()`；Clover 路径在 `solve_clover()` 中完成 Schur
  prepare、BiStabCG、解重构和 full-operator residual 验收。
- `applyCloverBistabCgQcu` 直接调用 `LatticeSet::clover_bistabcg()`，并可通过
  `kupl_bistabcg.py`、`KuplLinearOperator` 或统一 `Krylov` facade 使用。
- `applyCloverMultigridQcu` 是共享入口；当 `levels<=1` 或 `restart=0` 时，
  `clover_multigrid_solve()` 明确转发到同一个 `set.clover_bistabcg()`。因此
  `solver=applyCloverMultigridQcu` 记录的是入口 ABI，不代表实际执行了 coarse
  correction。
- 当前没有名为 `kupl_bistabcg_mpi.py` 的独立多 rank CLI；多 rank 生产复用
  `kupl_multigrid.py` 或 `simple/mpi/solve_case.py` 的 `levels=1,restart=0`。

#### 7.5.5 性能 A/B 与 HBM 证据

1. MG 与 BiStabCG 各自至少一次 warmup，再做同题 3 次以上 steady 采样。
2. MG 稳态复用同一 cache；禁止把 cold setup 时间与 BiStabCG solver-only 时间直接
   相乘或相除。
3. 扫描 threads 时至少测试满 NUMA 线程数及一半线程数，并记录 load average。
4. `KUPL_HUGEPAGE=1` 只在专用节点做 A/B，不能在没有 THP/大页证据时写成默认最优。
5. 严格 HBM 配置必须对每个 rank 检查 `numastat -p <pid>` 或
   `/proc/<pid>/numa_maps` 的 `N<node>=` 页数；只看 `memory_bound=true` 不够。
6. 输入文件、Python/NumPy 数组若在 `applyInitQcu` 前 first touch，后续 memory
   policy 不会自动迁移这些页；使用启动器或在绑定后完成分配。

公共 Legacy cache 统一使用 odd-Schur、parity-compressed-time 布局；身份匹配时 KUPL
可直接复用 CUDA/Torch 生成的该布局 cache，并通过 Legacy layout 标志选择对应数据路径。
KUPL 仍保留历史 full-time cache 支持，但两种布局的 coarse shape 不同；不得仅按文件名
或大小互换，必须校验 manifest identity、dtype、MPI coarse geometry 和 layout。

## 8. 升级、补充与维护

### 8.1 更新源码后重建

```bash
source ./env.sh
KUPL_ENABLE_NATIVE=ON KUPL_ENABLE_LTO=ON bash ./install.sh kupl
ctest --test-dir cpp/kupl/qcu/build --output-on-failure
```

已有构建目录会执行增量配置和编译。切换编译器、MPI 或 CMake 选项时，推荐使用新的
`PYQCU_KUPL_BUILD_DIR`，避免污染旧二进制证据。

### 8.2 多版本库并存

不同构建目录可并存，通过 `PYQCU_KUPL_LIB` 选择：

```bash
export PYQCU_KUPL_LIB=/absolute/path/to/libqcu.so
```

稳定软链接只跟随最近一次 `make.sh/install.sh kupl`；需要固定到某个历史构建时，直接
把 `PYQCU_KUPL_LIB` 设为该 build 目录中的 `libqcu.so`。

禁止仅依赖 `LD_LIBRARY_PATH` 中同名 `libqcu.so` 的搜索顺序。CUDA 与 KUPL 动态库
可能同名，生产报告必须写出实际解析路径。

### 8.3 ABI 与文档同步

修改 KUPL C API 后必须同时检查：

```bash
python pyqcu/testing/kupl/check_abi.py \
  --library cpp/kupl/qcu/libqcu.so \
  --header cpp/kupl/qcu/include/kupl/qcu.h \
  --cuda-header cpp/cuda/qcu/python/pyqcu.h
```

即使不编译 CUDA，本仓库仍保留 CUDA 兼容 C 声明作为 ABI 对照源；不得加载或运行
CUDA 动态库。

### 8.4 Sanitizer

UBSAN/ASAN 构建使用独立目录，并优先以单 OpenMP 线程复现：

```bash
cmake -S cpp/kupl/qcu -B cpp/kupl/qcu/build-ubsan \
  -DCMAKE_BUILD_TYPE=Debug \
  -DKUPL_ENABLE_NATIVE=OFF \
  -DKUPL_TEST_OMP_THREADS=1 \
  -DCMAKE_CXX_FLAGS='-fsanitize=undefined -fno-omit-frame-pointer' \
  -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=undefined' \
  -DCMAKE_SHARED_LINKER_FLAGS='-fsanitize=undefined'
cmake --build cpp/kupl/qcu/build-ubsan -j 16
UBSAN_OPTIONS=halt_on_error=1 \
  ctest --test-dir cpp/kupl/qcu/build-ubsan --output-on-failure
```

## 9. 故障排查

### 9.1 `Failed to find nvcc`

原因：执行了 CUDA 安装入口。

处理：

```bash
bash install.sh kupl
```

纯 KUPL 不应设置虚假的 `CUDAToolkit_ROOT`，也不需要安装 CUDA toolkit。

### 9.2 `'torch.dtype' object has no attribute 'to_real'`

原因：仍有脚本直接调用旧版 PyTorch 不存在的 dtype 方法。

处理：

```bash
rg -n 'to_real\(' pyqcu -g '*.py'
```

仓库活动路径应使用 `pyqcu.cann.real_dtype()`。若命中第三方或未纳入维护的历史脚本，
记录来源和调用命令，不修改虚拟环境中的 PyTorch。

### 9.3 `MPI_Init_thread` 与 `drun`/DONAU

拓扑规划器、`--version` 查询和纯 argparse CLI 不应初始化 MPI。本轮修复后，
`kunpeng_runtime.py` 不再通过 `pyqcu.tools` 间接导入 `mpi4py`。

若其他 Python 命令出现该错误，分别检查：

```bash
python -c 'import mpi4py.MPI as MPI; print(MPI.Get_library_version())'
env | grep -E '^(PMI|PMIX|OMPI|HYDRA|I_MPI|SLURM|PBS|LSF)'
```

HMPI、链接库和启动器必须来自同一套 HPCKit 安装树，不能混用系统 OpenMPI/UCX。

### 9.4 `KUPL QCU library not found`

```bash
export PYQCU_KUPL_LIB=/absolute/path/to/libqcu.so
test -f "$PYQCU_KUPL_LIB"
```

非默认构建目录必须显式设置。`solve_case.py` 也接受绝对路径的 `--kupl-lib`。

### 9.5 TileLang、ROCE 或 SDMA 日志

- TileLang 缺失是可选项降级；
- ROCE/SDMA 日志来自 MPI/网络运行时，只有对应功能测试失败时才作为错误；
- LLVM `loop not vectorized` 是 LTO 优化提示，不等于链接或数值失败。

### 9.6 rank/grid 不合法

KUPL 要求每个 local 维度为偶数。用拓扑规划器或 `--kupl-grid` 显式选择分解，不能把
同一个 cpuset 同时交给 MPI 和库内绑定。

## 10. 发布前检查表

- [ ] `PYQCU_BUILD_BACKEND=kupl`，日志中没有 `build-nv.sh` 或 `nvcc`。
- [ ] `PYQCU_KUPL_LIB` 指向预期 `libqcu.so`，并记录 SHA256。
- [ ] `ctest --test-dir <build> --output-on-failure` 全绿。
- [ ] `test_pure_kupl_install.py` 和 `test_kunpeng_runtime.py` 通过。
- [ ] MPI rank、grid、线程、NUMA/cpuset 和绑定责任明确。
- [ ] 生产求解使用 `--backend kupl`，报告 `PASS` 并具有独立真残差。
- [ ] MG/cache 运行记录 cache identity、布局和 correction 次数。
- [ ] 文档、ABI 头文件和动态库导出符号同步。

满足以上条件后，才可把该次结果作为纯 KUPL Kunpeng 安装或生产证据。
