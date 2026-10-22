# PyQCU Simple 完整用户指南

本文档说明 `pyqcu/testing/simple/` 的真实输入、输出、后端差异、数值约定、
OpenMPI 绑定、缓存、报告、状态和故障排查。命令参数速查见
[`COMMAND_REFERENCE.md`](COMMAND_REFERENCE.md)。

## 目录

1. 适用范围与运行模式
2. 环境、路径和默认值
3. 数据布局与文件格式
4. LIME / QIO 输入兼容
5. mass、csw、kappa 与方程归一化
6. 输入准备 `prepare_case.py`
7. 生产求解 `solve_case.py`
8. 直接后端示例
9. 数值正确性对照
10. 三方性能测量
11. Multigrid 与 33-tensor 缓存
12. 转换与检查工具
13. 输出文件和 JSON 报告
14. 状态、退出码和真残差门限
15. GPU、KUPL 与 OpenMPI 进程绑定
16. MPI 小格点入口
17. 推荐工作流
18. 故障排查
19. 已知限制
20. 验证清单

## 1. 适用范围与运行模式

`simple/` 包含三种容易混淆的运行模式。

| 模式 | 入口 | 目的 |
|---|---|---|
| 输入准备 | `prepare_case.py` | 从 LIME 生成小体积或全格 HDF5 gauge/source 与 manifest |
| 生产单次求解 | `solve_case.py` | 选择一个后端，做一次冷求解，PASS 后原子发布解文件 |
| 测量/对照 | `compare_*.py`、`benchmark_*.py` | 重复计时、同 RHS 对照或 MPI 扩展测量 |

不要用 `compare_three_way.py` 发布生产解。该脚本以性能测量为主，KUPL 计时样本
默认跳过昂贵的独立真残差；生产真残差应由 `solve_case.py` 或直接后端脚本计算。

## 2. 环境、路径和默认值

### 2.1 加载环境

从仓库根目录执行：

```bash
source ./env.sh
```

`env.sh` 主要设置：

- `PYTHONPATH`，使 `pyqcu` 与 `pyqcu.testing.simple` 可导入。
- `LD_LIBRARY_PATH`，使 `libqcu.so` 可加载。
- OpenMPI root 运行确认变量。

它不会自动为所有脚本选择正确的 GPU 架构库。`solve_case.py`、
`compare_three_way.py` 有架构选择逻辑；直接运行 `qcu_*.py` 时需要用户自行设置
`LD_LIBRARY_PATH`。

### 2.2 路径解析

共享函数 `_common.resolve_path()` 的规则为：

1. 绝对路径原样使用。
2. 以 `data/` 开头的相对路径按仓库根解析。
3. 当前工作目录下已存在的相对路径按当前工作目录解析。
4. 其他相对路径按 `DATA_DIR` 解析。

`DATA_DIR` 默认为仓库 `data/`，可由以下变量覆盖：

```bash
export PYQCU_DATA_DIR=/path/to/data
```

注意：`solve_case.py` 的默认生产输出目录固定使用仓库
`data/production_<lattice>/`，不会仅因 `PYQCU_DATA_DIR` 改变；显式传入
`--out-dir` 最稳妥。

### 2.3 默认 LIME

多个脚本默认使用：

```text
~/data/beta6.20_mu-0.2770_ms-0.2310_L24x64_cfg_9950.lime
```

共享 `_common.DEFAULT_LIME` 可由以下变量覆盖：

```bash
export PYQCU_REFERENCE_LIME=/absolute/path/to/reference.lime
```

`solve_case.py` 和 `compare_three_way.py` 自己定义默认 home 路径，因此推荐总是显式
传入 `--lime` 或 `--gauge`，避免不同入口的默认路径不一致。

### 2.4 默认数值参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `mass` | 优先 LIME `Mass`，否则 `0.05` | 传给 QCU/KUPL/QUDA |
| `csw` | 优先 LIME `clovCoeff`，否则 `1.160920226` | Clover 系数 |
| `atol` | `1e-6` | 后端目标容差 |
| `max_iter` | 通常 `1000` | 直接示例与生产入口 |
| `seed` | `42` | source 和随机 gauge 的确定性种子 |

### 2.5 环境变量参考

| 变量 | 作用 |
|---|---|
| `PYQCU_DATA_DIR` | 共享 helper 的默认 data 目录 |
| `PYQCU_REFERENCE_LIME` | `_common.DEFAULT_LIME` 默认参考组态 |
| `PYQCU_KUPL_LIB` | KUPL `.so` 绝对路径，优先于 build 默认 |
| `KUPL_OMP_THREADS` | KUPL 库内 OpenMP 精确线程数 |
| `KUPL_NUMA_BIND=1` | 由 KUPL 库做 NUMA 绑定；MPI 必须 `--bind-to none` |
| `OMP_NUM_THREADS` | OpenMP 线程数，也供启动脚本设置 |
| `OMP_PROC_BIND` / `OMP_PLACES` | 与 `--bind-to none`/KUPL 绑定冲突时清除 |
| `KUPL_DISABLE_BLOCK_DSLASH=1` | 禁用 KUPL block dslash，回退 scalar site kernel 做 A/B |
| `CUDA_VISIBLE_DEVICES` | 限制可见 GPU 和 logical ordinal 映射 |
| `CUDA_DEVICE_ORDER` | `PCI_BUS_ID` 时按 PCI 顺序映射 GPU |
| `QUDA_INSTALL` | PyQUDA/QUDA 安装根，优先于自动 variant |
| `QUDA_PATH` | 部分旧 PyQUDA API 使用的 QUDA 根路径 |
| `QUDA_RESOURCE_PATH` | QUDA tuning resource 目录 |
| `QUDA_BUILD_DIR` | 站点脚本使用的 QUDA build 目录 |
| `QUDA_ENABLE_TUNING` | 站点脚本设置 QUDA tuning 开关 |
| `QUDA_KAPPA_NORMALIZATION=1` | simple 对照统一 QCU 归一化 |
| `QUDA_VERBOSE` | QUDA verbose 级别 |
| `QUDA_USE_INIT_GUESS_NO` | 禁用 QUDA initial guess |
| `QUDA_BICGSTAB_INVERTER` | QUDA BiCGStab inverter 类型覆盖 |
| `QCU_LOG_DIR` | QCU MG native log 输出目录 |
| `QCU_GPU_COMPUTE_CAPABILITY` | runner 写入的诊断值 |
| `QCU_LIBCU_VARIANT` | 报告中的 native library variant |
| `QCU_KUPL_VARIANT` | 报告中的 KUPL library variant |
| `QCU_QUDA_VARIANT` | 报告中的 QUDA variant |
| `DEV87_REDUCE_SYNC` | QUDA comparison 的同步调试开关 |
| `PYQCU_MPI_DEVICE_ID` | CUDA MPI worker 内部设备选择；用户通常不直接设置 |

## 3. 数据布局与文件格式

### 3.1 canonical PyQCU full layout

| 对象 | HDF5 shape | 轴顺序 |
|---|---|---|
| gauge | `[3,3,4,X,Y,Z,T]` | `[color_out,color_in,direction,x,y,z,t]` |
| fermion | `[4,3,X,Y,Z,T]` | `[spin,color,x,y,z,t]` |
| clover | `[4,3,4,3,X,Y,Z,T]` | `[spin_out,color_out,spin_in,color_in,x,y,z,t]` |

HDF5 默认 dataset 名为 `data`。可用 `--dataset` 的检查工具读取其他 dataset。
文件 attrs 通常包含 `layout`、`mass`、`reference_mass`、`csw`、`seed`、`source`
等元数据。

### 3.2 QCU parity-compressed layout

C++ QCU 后端内部使用时间方向奇偶压缩：

| 对象 | shape |
|---|---|
| gauge | `[2,3,3,4,X,Y,Z,T/2]` |
| fermion | `[2,4,3,X,Y,Z,T/2]` 或等价的 `[2,12,X,Y,Z,T/2]` |
| clover parity block | `[4,3,4,3,X,Y,Z,T/2]` |

parity 的压缩方式不是简单把时间轴一分为二。转换必须使用 `_common.py` 中的
`qcu_gauge_to_kupl()`、`qcu_fermion_to_kupl()`、`kupl_*_to_qcu()`，不能直接
`reshape`。

### 3.3 QUDA / QDP layout

| 对象 | shape |
|---|---|
| gauge | `[dir,t,z,y,x,color_out,color_in]` |
| fermion | `[q,t,z,y,x/2,spin,color]` |
| packed clover | `[q,t,z,y,x/2,chirality,36]` |

QUDA fermion 沿 x 方向做 checkerboard 压缩，`q=(x+y+z+t)%2`。QCU 的 parity
压缩沿时间轴表达，二者必须经转换函数互换。

### 3.4 支持的文件容器

| 容器 | 读取 | 写入 | 说明 |
|---|---|---|---|
| HDF5 `.h5/.hdf5` | 是 | 是 | 默认 dataset `data`，带 attrs |
| NumPy `.npy` | 是 | 是 | 无 metadata attrs |
| NumPy `.npz` | 是 | 是 | dataset 名由 `dataset` 指定 |
| LIME `.lime` | 是 | gauge 可写 | 单文件 QIO/SciDAC gauge |

## 4. LIME / QIO 输入兼容

### 4.1 支持的记录

读取器支持常见 PyQUDA/Chroma 单文件 LIME 记录：

- `scidac-private-file-xml`
- `scidac-file-xml`
- `scidac-record-xml`
- `scidac-private-record-xml`
- `ildg-format`
- `ildg-binary-data`
- `scidac-checksum`

gauge metadata 从以下 XML 元素解析：

- `<dims>`：`X Y Z T`
- `<clovCoeff>`：csw
- `<Mass>`：reference mass
- `<precision>`：单精度 `F` 或双精度 `D`
- `<colors>`：必须为 3
- `<datacount>`：必须为 4

### 4.2 布局与 checksum

LIME 二进制按：

```text
[t,z,y,x,direction,color_out,color_in]
```

读取后转换为 QDP `[direction,t,z,y,x,color_out,color_in]`，再转换为 QCU
`[color_out,color_in,direction,x,y,z,t]`。

若存在 `scidac-checksum`，读取器会计算并校验 checksum。格式错误、payload 大小
错误或 checksum 不匹配会直接失败，不会静默继续。

### 4.3 子体积

当 `--lat` 小于 LIME 的 `[X,Y,Z,T]` 时，脚本取从原点开始的前缀块：

```text
[0:X, 0:Y, 0:Z, 0:T]
```

这不会重建周期边界。切口处的 link 仍来自原始组态，因此该子体积是一个受控数值
测试系统，不是独立周期规范组态。

## 5. mass、csw、kappa 与方程归一化

QCU/KUPL 使用：

```text
kappa = 1 / (2 * mass + 8)
```

Clover onsite 系数统一为：

```text
-kappa * csw / 8
```

QCU ABI 中：

- `argv[0]`：mass。
- `argv[1]`：atol。
- `argv[2]`：sigma。
- `argv[7]`：csw。

`applyCloverQcu` / `applyCloversQcu` 返回 `A=I+T`；其中 `T` 是 additive
Clover term。不要把返回值再次加 I。

QUDA simple 对照强制：

```bash
QUDA_KAPPA_NORMALIZATION=1
```

使 QUDA 与 QCU 的 `M=I-kappa D` 使用相同归一化。若使用 PyQUDA 默认
asymmetric normalization，解会额外相差 `1/(2*kappa)=mass+4`。

如果 CLI 显式 csw 与 LIME `clovCoeff` 不同：

- 求解仍可运行。
- 报告保留 `gauge_csw`。
- 设置 `comparison_valid=false`。
- 状态可能为 `WARN`，不会作为严格对照的 `PASS`。

## 6. 输入准备 `prepare_case.py`

标准命令：

```bash
python pyqcu/testing/simple/prepare_case.py \
  --lime ~/data/reference.lime \
  --lat 24 24 24 64 \
  --out-dir data/production_L24
```

行为：

1. 读取 LIME metadata，得到完整格点、mass 和 csw。
2. 检查目标格点不超过 LIME。
3. 切出 gauge 前缀块并转成 QCU HDF5。
4. 用 NumPy PCG64 生成共享 complex64 source。
5. 写 `case_manifest.json`，记录路径、shape、SHA256、大小和子体积说明。

缓存规则：

- gauge 文件存在时，检查 lattice、mass、csw 是否一致。
- source 文件存在时，检查 seed 是否一致。
- 不一致会报错，不会静默覆盖。

输出：

```text
<out-dir>/gauge_<lat>_c64.h5
<out-dir>/source_<lat>_c64.h5
<out-dir>/case_manifest.json
```

## 7. 生产求解 `solve_case.py`

### 7.1 输入优先级

- `--gauge` 优先于 `--lime`。
- `--source` 显式给出时直接使用，否则生成确定性 source。
- `--gauge` 为 HDF5 时，其 lattice 必须与 `--lat` 完全一致；生产入口不会对 HDF5
  gauge 自动再切片。
- `--gauge` 为 LIME 时，可按 `--lat` 切前缀子体积。

### 7.2 CUDA QCU

```bash
python pyqcu/testing/simple/solve_case.py \
  --backend cuda \
  --lime ~/data/reference.lime \
  --lat 24 24 24 64 \
  --device cuda:2 \
  --libqcu auto \
  --atol 1e-6 \
  --out-dir data/production_L24
```

`--device cuda:N` 在未设置 `CUDA_VISIBLE_DEVICES` 时被解释为物理 PCI 索引 N；
`cuda_environment()` 会设置 `CUDA_DEVICE_ORDER=PCI_BUS_ID` 和隔离的
`CUDA_VISIBLE_DEVICES=N`，子进程内使用逻辑 `cuda:0`。

`--libqcu`：

- `auto`：按目标 GPU compute capability 选择 `data/libqcu-sm60` 或
  `data/libqcu-sm70`。
- `sm60` / `sm70`：强制 A/B。
- `default`：保留调用者当前 `LD_LIBRARY_PATH`。

### 7.3 KUPL QCU

```bash
python pyqcu/testing/simple/solve_case.py \
  --backend kupl \
  --lime ~/data/reference.lime \
  --lat 24 24 24 64 \
  --kupl-ranks 8 \
  --kupl-threads 2 \
  --kupl-grid 1 4 2 1 \
  --kupl-lib lto \
  --out-dir data/production_L24
```

`--kupl-grid` 不指定时，`default_kupl_grid()` 按 y→z→t→x 的优先级寻找偶数 local
extent 的分解。若 rank 数无法分解，生产入口会报错；三方 benchmark 会回退到 1 rank。

`--kupl-lib`：

- `default`：`cpp/kupl/qcu/build/libqcu.so` 或 `PYQCU_KUPL_LIB`。
- `lto`：`cpp/kupl/qcu/build-lto/libqcu.so`。
- 绝对路径：直接指定 `.so`。

`solve_case.py` 的 KUPL 子进程使用 `mpirun --bind-to none`。CPU 放置由调度器
cpuset 或 `KUPL_NUMA_BIND=1` 负责。需要启动器直接管理 NUMA 绑定时，使用
`mpi/solve_case.py`。

### 7.4 QUDA / PyQUDA

```bash
python pyqcu/testing/simple/solve_case.py \
  --backend quda \
  --lime ~/data/reference.lime \
  --lat 24 24 24 64 \
  --device cuda:2 \
  --quda-variant auto \
  --out-dir data/production_L24
```

`--quda-variant`：

- `auto`：按 compute capability 选择 `data/quda-sm60-install` 或
  `data/quda-sm70-install`。
- `sm60` / `sm70`：强制架构安装。
- `default`：保留调用者 `QUDA_INSTALL`/`LD_LIBRARY_PATH`。

显式 `QUDA_INSTALL` 优先于脚本自动选择。

## 8. 直接后端示例

直接脚本适合调试 ABI、单算子或单一后端。它们不负责跨后端输入隔离。

### 8.1 CUDA QCU BiStabCG

```bash
python pyqcu/testing/simple/qcu_bistabcg.py \
  --gauge ~/data/reference.lime \
  --lat 8 8 16 16 \
  --mass -0.277 \
  --atol 1e-6 \
  --device cuda:0 \
  --warmup 0 --repeats 1 \
  --output data/qcu_solution.h5 \
  --report data/qcu_bistabcg.json
```

### 8.2 CUDA QCU legacy MG

```bash
python pyqcu/testing/simple/qcu_multigrid.py \
  --gauge ~/data/reference.lime \
  --lat 8 8 8 16 \
  --levels 2 \
  --dof 12 12 \
  --mg-grid 2 2 2 2 \
  --restart 5 \
  --device cuda:0 \
  --warmup 0 --repeats 1 \
  --report data/qcu_mg.json
```

`qcu_multigrid.py` 在 Python 侧用 full-site matvec 构建 33-tensor Galerkin 粗算子。
`--levels 1` 是 fine-only 路径，不等价于完整递归 MG。

### 8.3 KUPL BiStabCG

```bash
KUPL_OMP_THREADS=8 OMP_NUM_THREADS=8 \
python pyqcu/testing/simple/kupl_bistabcg.py \
  --gauge ~/data/reference.lime \
  --lat 8 8 16 16 \
  --mass -0.277 --csw 1.160920226 \
  --dtype c64 \
  --report data/kupl_bistabcg.json
```

该脚本是单 MPI rank 示例。多 rank 使用 `kupl_multigrid.py` 或
`mpi/solve_case.py`。

### 8.4 KUPL legacy MG

```bash
KUPL_OMP_THREADS=2 OMP_NUM_THREADS=2 \
mpirun --allow-run-as-root --oversubscribe --bind-to none -np 4 \
  python pyqcu/testing/simple/kupl_multigrid.py \
  --gauge data/production_L24/gauge_24x24x24x64_c64.h5 \
  --source data/production_L24/source_24x24x24x64_c64.h5 \
  --lat 24 24 24 64 --grid 1 4 1 1 \
  --mass -0.277 --csw 1.160920226 \
  --levels 2 --dof 12 12 --mg-grid 2 2 2 2 \
  --restart 5 \
  --coarse-cache data/mg_cache.h5 \
  --report data/kupl_mg_mpi.json
```

多 rank 且 `levels>1, restart!=0` 时必须提供全局 `--coarse-cache`。缓存需由
`build_full_mg_cache.py` 或单 rank 的 `kupl_multigrid.py --write-coarse-cache`
生成。

### 8.5 QUDA

```bash
python pyqcu/testing/simple/quda_bistabcg.py \
  --gauge ~/data/reference.lime \
  --lat 8 8 16 16 \
  --mass -0.277 --csw 1.160920226 \
  --tol 1e-6 \
  --report data/quda_bistabcg.json
```

QUDA 脚本兼容 `pyquda_utils` 和旧 `PyQuda-0.3.2` 原生 API。旧 API 缺少可捕获的
逐迭代日志时，`residual_history` 为空并记录来源。

## 9. 数值正确性对照

### 9.1 QCU vs QUDA BiStabCG

```bash
python pyqcu/testing/simple/compare_bistabcg.py \
  --gauge ~/data/reference.lime \
  --lat 8 8 16 16 \
  --mass -0.277 --csw 1.160920226 \
  --atol 1e-6 \
  --qcu-device cuda:0 \
  --solution-rtol 5e-3
```

脚本会：

1. 生成同一 complex64 source。
2. 在独立子进程运行 QCU 和 QUDA，避免两个库在同一进程冲突。
3. 检查 backend、lattice、mass、csw、mass normalization。
4. 检查两边都 `PASS`，不允许 `SKIP` 冒充成功。
5. 读取两边 HDF5 解并计算相对 L2 差。
6. 只有全部满足才输出总状态 `PASS`。

### 9.2 KUPL vs QUDA BiStabCG

```bash
python pyqcu/testing/simple/compare_kupl_bistabcg.py \
  --lat 4 4 4 4 \
  --mass 0.17 --csw 0.8 \
  --kupl-threads 2
```

不提供 `--gauge` 时，脚本生成确定性 NumPy SU(3) gauge；source 由 seed+1 生成。

### 9.3 KUPL MG vs QUDA BiStabCG

```bash
python pyqcu/testing/simple/compare_kupl_multigrid.py \
  --lat 4 4 4 4 \
  --mass 0.17 --csw 0.8 \
  --levels 2 --dof 12 2 --mg-grid 2 2 2 2 \
  --restart 5 \
  --kupl-threads 2
```

除解差外，还要求 KUPL `verify_status` 为 `0`。该检查验证 restrict/prolong 与
Galerkin 粗算子关系。

## 10. 三方性能测量

`compare_three_way.py` 默认测量：

| 项 | 实际路径 |
|---|---|
| CUDA QCU BiCG | standalone Clover BiStabCG |
| CUDA QCU MG | `qcu_multigrid.py --levels 1`，fine-only |
| KUPL QCU | `kupl_multigrid.py --restart 0`，optimized parity-Schur BiStabCG |
| QUDA | Clover BiStabCG |

因此 `summary.md` 适合回答“同一小/中等体积上各后端 BiStabCG 路径的墙钟和迭代
表现”，不适合证明完整多层级 MG 的算法等价或加速比。

计时规则：

- 可选 `--gpu-warmup-seconds 3` 提升 GPU 时钟。
- CUDA/QUDA 进程内 warmup + repeats。
- KUPL 每个样本启动独立 MPI 作业，取中位数。
- KUPL 计时样本默认 `--skip-independent-residual`。
- 真残差仍由各后端报告；需要严格生产验收时使用 `solve_case.py`。

可只测部分后端：

```bash
python pyqcu/testing/simple/compare_three_way.py \
  --backends cuda,kupl \
  --lat 8 8 16 16 \
  --device cuda:0 \
  --kupl-ranks 4 --kupl-threads 4
```

## 11. Multigrid 与 33-tensor 缓存

`build_full_mg_cache.py` 为两级 Clover MG 构建：

- `lonv`：null-vector basis。
- `hnn`：nearest-neighbor hopping tensors。
- `hdg`：diagonal hopping tensors。
- `sit`：sitting/onsite coarse operator。

标准命令：

```bash
python pyqcu/testing/simple/build_full_mg_cache.py \
  --gauge data/production_L24/gauge_24x24x24x64_c64.h5 \
  --lat 24 24 24 64 \
  --mass -0.277 --csw 1.160920226 \
  --dtype c64 \
  --dof 12 12 \
  --mg-grid 2 2 2 2 \
  --nv-iters 1 \
  --build-device cpu \
  --output data/mg_cache_L24_E12.h5 \
  --report data/mg_cache_L24_E12.json
```

约束：

- 脚本固定构建 `levels=2`。
- gauge 必须是目标格点的完整 canonical HDF5，不能直接传更大的 LIME。
- `--mg-grid` 必须整除格点。
- `--build-device cuda` 要求目标 CUDA/PyTorch 可执行对应 kernel。
- 多 rank worker 只读取全局缓存的 rank-local 超切片，不应每个 rank 重复构建。

## 12. 转换与检查工具

### 12.1 gauge 转换

```bash
python pyqcu/testing/simple/quda_gauge2qcu_gauge.py \
  --input reference.lime --output gauge_qcu.h5

python pyqcu/testing/simple/qcu_gauge2quda_gauge.py \
  --input gauge_qcu.h5 --output gauge_qdp.npy
```

`qcu_gauge2quda_gauge.py` 的 `--output` 若为 `.lime`，会写单文件 SciDAC/QIO gauge。
它不是 MPI/QIO part-file writer。

### 12.2 fermion 转换

```bash
python pyqcu/testing/simple/qcu_fermion2quda_fermion.py \
  --input source_qcu.h5 --output source_qdp.npy

python pyqcu/testing/simple/quda_fermion2qcu_fermion.py \
  --input source_qdp.npy --output source_qcu.h5
```

### 12.3 clover 转换

```bash
python pyqcu/testing/simple/qcu_clover2quda_clover.py \
  --input clover_qcu.h5 --output clover_qdp.npy

python pyqcu/testing/simple/quda_clover2qcu_clover.py \
  --input clover_qdp.npy --output clover_qcu.h5 \
  --dtype complex64
```

### 12.4 检查工具

无 `--input` 时运行内置 self-test：

```bash
python pyqcu/testing/simple/check_gauge.py
python pyqcu/testing/simple/check_fermion.py
python pyqcu/testing/simple/check_clover.py
```

有输入时检查有限性、往返布局误差；gauge 还检查 unitarity。

## 13. 输出文件和 JSON 报告

### 13.1 生产摘要

`last_solve_<backend>.json` 固定字段：

| 字段 | 含义 |
|---|---|
| `backend` | `cuda`、`quda` 或 `kupl` |
| `solution` | 正式解路径；失败时该文件可能不存在 |
| `report` | 后端原始 JSON 路径 |
| `status` | `PASS/WARN/FAIL/SKIP` |
| `iterations` | 后端原生迭代数，缺失时 `null` |
| `wall_seconds` | 后端操作墙钟 |
| `solver_seconds` | 后端独立 solver timer，若提供 |
| `true_residual` | 独立或后端真残差 |
| `residual_source` | 残差来源 |
| `mpi_size` / `grid` | KUPL MPI 几何 |
| `libqcu_variant` | CUDA native library 选择 |
| `quda_variant` | QUDA 安装选择 |
| `kupl_variant` | KUPL library 选择 |
| `error` | 失败摘要 |

### 13.2 后端原始报告

常见字段包括：

- `backend`、`solver`、`lattice`、`mass`、`kappa`、`csw`、`atol`。
- `iterations`、`wall_seconds`、`seconds_per_iteration`。
- `final_true_residual`、`residual_acceptance`、`residual_source`。
- `residual_history`、`outer_iteration_residual_history`。
- `vcycle_residual_history`、`profile_ms`。
- `memory`、`converged`、`status`。

QCU MG 的日志从 `QCU_LOG_DIR/clover_multigrid.log` 解析；直接脚本默认写入
`data/logs/`。

### 13.3 对照报告

`compare_bistabcg.py`、`compare_kupl_bistabcg.py`、
`compare_kupl_multigrid.py` 的报告包含：

- 总 `status`。
- `compatible` 与逐字段 `compatibility`。
- 两边进程 return code、stdout/stderr、原始报告。
- `solution_relative_difference`。
- 输入参数、source、两边解路径和错误列表。

`compare_three_way.py` 的报告包含：

- 输入 lattice、mass、csw、atol、seed。
- warmup/repeats、device、KUPL ranks/threads/library。
- `rows` 统一性能表。
- 每个后端的独立 JSON。

## 14. 状态、退出码和真残差门限

`convergence_status()` 使用：

```text
acceptance = max(atol, precision_floor)
```

默认门限：

- complex64：`max(atol, 2048*eps32)`；默认约 `2.4414e-4`。
- complex128：`max(atol, 128*eps64)`。

因此 c64 即使请求 `atol=1e-6`，Python 侧验收门限也约为 `2.44e-4`。后端自身
可能仍按 `argv[1]=1e-6` 停机；报告会同时记录 `final_true_residual` 和
`residual_acceptance`。

生产入口只发布 `PASS` 解。`WARN` 不会发布正式解。直接示例可能对 `WARN` 或
`SKIP` 返回 0，因此自动化必须同时检查 JSON `status`，不能只看进程退出码。

## 15. GPU、KUPL 与 OpenMPI 进程绑定

### 15.1 CUDA

- 生产约定固定为一个 Python 进程 = 一个 MPI rank = 一张 GPU/DCU/GPGPU；
  跨设备求解通过 MPI 格点分域，不使用单进程多设备 worker。
- 完整拓扑规则见
  [`docs/cuda_execution_topology.md`](../../../docs/cuda_execution_topology.md)。
- `--device cuda:N` 在未设置 `CUDA_VISIBLE_DEVICES` 时表示物理 PCI 索引 N。
- 多进程 GPU 运行优先设置 `CUDA_VISIBLE_DEVICES`，再让每个 rank 选择 logical
  device。
- `MultiGpuMultigrid` 是显式调用的测试专用例外，不属于生产入口。除非任务明确
  要求多设备一致性或吞吐测试，不得启用。
- P100 与 V100 混合节点应使用与所有目标 GPU 兼容的 library；仅含 sm_70 SASS
  的库不能直接用于 P100。
- 系统 PyTorch 若不支持 sm_60，可使用仓库提供的 P100 虚拟环境，但 native
  `libqcu.so` 仍需匹配 sm_60。

### 15.2 KUPL 单进程

`kupl_bistabcg.py` 和单 rank `kupl_multigrid.py` 使用：

```bash
KUPL_OMP_THREADS=8 OMP_NUM_THREADS=8 python ...
```

`KUPL_OMP_THREADS` 是库内 OpenMP 精确线程数；未设置时库会按进程初始 CPU affinity
裁剪。

### 15.3 OpenMPI + OpenMP 混合绑定

推荐的多 NUMA 模式：

```bash
export OMP_NUM_THREADS=8
export KUPL_OMP_THREADS=8
unset OMP_PROC_BIND OMP_PLACES
export KUPL_NUMA_BIND=0

mpirun \
  --bind-to numa \
  --map-by ppr:1:numa \
  -np 4 \
  python pyqcu/testing/simple/kupl_multigrid.py ...
```

含义：

- 每个 NUMA 域放一个 MPI rank。
- 每个 rank 的 OpenMP 线程数由 `OMP_NUM_THREADS` 固定。
- MPI 负责进程/CPU 放置，KUPL 库不得再次绑定，因此 `KUPL_NUMA_BIND=0`。
- 不要同时设置 `OMP_PROC_BIND` 或 `OMP_PLACES`。该后端由大量短 OpenMP 并行区
  组成，重复线程绑定可能造成显著性能回退。

OpenMPI 4.1 会拒绝：

```text
--bind-to numa --map-by ppr:1:numa:pe=8
```

因为 `pe=8` 指定每进程 8 个 processing elements，与 `bind-to numa` 冲突。可选
方案：

```bash
# 推荐：绑定整个 NUMA，线程数由 OMP_NUM_THREADS 控制
mpirun --bind-to numa --map-by ppr:1:numa -np 4 ...

# 若确实要求 OpenMPI 分配 8 个 PE，则绑定策略应与 PE 一致
mpirun --bind-to core --map-by ppr:1:numa:pe=8 -np 4 ...
```

若 rank 数超过本机 NUMA 域数，`ppr:1:numa` 无法映射。单 NUMA 开发机应使用：

```bash
mpirun --oversubscribe --bind-to none -np 4 ...
```

或由 Slurm/cgroup 为每个 rank 提供独立 cpuset。`--oversubscribe` 允许进程数超过
可用 slot，但不等于正确 NUMA 绑定。

验证 affinity：

```bash
mpirun --bind-to numa --map-by ppr:1:numa -np 4 \
  python -c 'import os; print(sorted(os.sched_getaffinity(0)))'
```

### 15.4 生产驱动与 MPI 快速入口的绑核责任

| 入口 | OpenMPI 绑定 | KUPL 库绑定 | 说明 |
|---|---|---|---|
| `simple/solve_case.py --backend kupl` | `--bind-to none` | 由 `KUPL_NUMA_BIND` 决定 | 适合调度器已分配 cpuset，或由库绑定 |
| `simple/mpi/solve_case.py --backend kupl` | 自动 `numa` 或 `none` | 强制 `KUPL_NUMA_BIND=0` | 启动器负责放置，避免双重绑定 |

## 16. MPI 小格点入口

`mpi/solve_case.py` 是独立入口，不复用生产 `solve_case.py` 的 KUPL 命令构造。
它直接接受 LIME，默认 `8x8x8x16`，自动准备 source 和粗算子缓存，并启动一次
`mpirun`。详细说明见 [`mpi/USER_GUIDE.md`](mpi/USER_GUIDE.md)。

## 17. 推荐工作流

### 17.1 新组态快速验收

```bash
source ./env.sh
python pyqcu/testing/simple/check_gauge.py --input reference.lime
python pyqcu/testing/simple/prepare_case.py \
  --lime reference.lime --lat 8 8 8 16 --out-dir data/smoke_case
python pyqcu/testing/simple/solve_case.py \
  --backend kupl --lat 8 8 8 16 \
  --kupl-ranks 4 --kupl-threads 2 \
  --out-dir data/smoke_case
```

### 17.2 GPU 正确性验收

```bash
python pyqcu/testing/simple/compare_bistabcg.py \
  --gauge reference.lime --lat 8 8 16 16 \
  --qcu-device cuda:0 --solution-rtol 5e-3
```

### 17.3 生产发布

1. `prepare_case.py` 固定输入和 SHA256。
2. 三个后端分别运行 `solve_case.py`。
3. 比较 HDF5 解和真残差。
4. 只保留 `PASS` 解；保存报告和 manifest。

### 17.4 性能测量

先确认所有后端在正确性对照中 `PASS`，再运行 `compare_three_way.py`。性能样本
不能替代正确性样本。

## 18. 故障排查

| 现象 | 常见原因 | 处理 |
|---|---|---|
| `FileNotFoundError: libqcu.so` | `LD_LIBRARY_PATH` 未设置或架构目录错误 | `source ./env.sh`；显式 `--libqcu sm60/sm70` |
| `no kernel image is available` | library 架构与 GPU 不匹配 | P100 用 sm60，V100 用 sm70，或使用含兼容 PTX 的 default 库 |
| PyTorch 不支持 P100 | 系统 PyTorch 最低架构为 sm70 | 使用兼容 P100 的 Python 环境 |
| `pyquda` import 失败 | PyQUDA/QUDA 安装不在当前环境 | 设置 `QUDA_INSTALL`、`PYTHONPATH`、`LD_LIBRARY_PATH` |
| QUDA 与 QCU 解差一个 `mass+4` | QUDA normalization 不同 | 保持 `QUDA_KAPPA_NORMALIZATION=1` |
| gauge lattice 与 `--lat` 不一致 | HDF5 gauge 不自动切片 | 使用 LIME，或先用 `prepare_case.py` 生成目标 HDF5 |
| csw mismatch / `WARN` | CLI csw 与 LIME metadata 不同 | 使用 metadata csw，或明确接受非严格对照 |
| KUPL rank 分解失败 | local extent 为奇数或格点不能整除 | 显式传合法的 `--kupl-grid` |
| KUPL 多 rank MG 报缺 cache | `levels>1, restart!=0` 需要全局粗算子 | 先构建 `--coarse-cache` |
| OpenMPI `ppr:1:numa` 进程过多 | rank 数超过 NUMA 域数 | `--bind-to none --oversubscribe` 或改 PPR |
| OpenMPI `numa + pe=N` 冲突 | 绑定策略不一致 | 用 `ppr:1:numa` 或 `--bind-to core` |
| KUPL 性能异常慢 | 同时设置了 MPI/OpenMP/KUPL 绑核 | MPI 绑定时设 `KUPL_NUMA_BIND=0`，清除 `OMP_PROC_BIND/PLACES` |
| `SKIP` 被自动化当成失败 | 聚合脚本不接受缺失依赖 | 安装依赖或只运行可用后端 |
| `.partial` 文件残留 | 后端失败或未写正式解 | 检查 report/error；正式解不会从不完整文件发布 |
| 解文件存在但返回非零 | 上次旧文件仍在，本次未 PASS | 检查 `last_solve_*.json` 时间戳和报告，不要只看文件存在性 |

## 19. 已知限制

1. `prepare_case.py` 只生成 complex64 production case。
2. HDF5 gauge 生产入口不自动切片；切片只对 LIME 自动进行。
3. 子体积切口不周期。
4. `compare_three_way.py` 的 CUDA MG 项是 fine-only，不是完整递归粗网格。
5. KUPL 单 rank 全格独立残差在 `24^3x64` 上可能远慢于求解；生产应使用多 rank
   或明确 `--skip-independent-residual`。
6. `benchmark_kupl_lime_mpi.py` 要求外部 `mpirun`；它不是自动 launcher。
7. `slurm_lime_backend_matrix.sh` 依赖站点特定的 BSCC 环境和 QUDA 安装，不可直接
   假定所有机器可用。
8. 旧 PyQUDA API 的逐迭代历史可能不可获取；空历史是明确限制，不是解析失败。

## 20. 验证清单

文档或脚本改动后至少执行：

```bash
source ./env.sh
python -m compileall -q pyqcu/testing/simple
python pyqcu/testing/simple/check_gauge.py
python pyqcu/testing/simple/check_fermion.py
python pyqcu/testing/simple/check_clover.py
python pyqcu/testing/simple/prepare_case.py --help
python pyqcu/testing/simple/solve_case.py --help
python -m pytest -q -p no:cacheprovider \
  pyqcu/testing/simple/test_coarse_cache.py \
  pyqcu/testing/simple/mpi/test_mpi_contract.py
git diff --check
bash skills/pyqcu-form/form-audit.sh
```

有对应设备时再运行一个 CUDA、一个 KUPL 和一个 QUDA 小格点 smoke；设备缺失时在
报告中明确写“未验证”，不要用其他后端结果代替。
