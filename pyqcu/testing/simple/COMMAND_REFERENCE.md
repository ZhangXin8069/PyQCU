# PyQCU Simple 命令参考

本文档逐项列出 `pyqcu/testing/simple/` 的主要用户命令。所有示例默认从仓库根目录
执行，并假定已运行：

```bash
source ./env.sh
```

每个 Python 脚本都支持：

```bash
python <script> --help
```

## 1. 输入准备

### `prepare_case.py`

用途：从 LIME 生成 production case；写 canonical HDF5 gauge/source 与 manifest。

```bash
python pyqcu/testing/simple/prepare_case.py \
  --lime ~/data/reference.lime \
  --lat 24 24 24 64 \
  --mass -0.277 \
  --csw 1.160920226 \
  --seed 42 \
  --out-dir data/production_L24
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--lime` | `~/data/beta6.20...cfg_9950.lime` | 输入 LIME |
| `--lat X Y Z T` | `24 24 24 64` | 目标格点，可为 LIME 前缀子体积 |
| `--mass` | LIME `Mass` 或 `0.05` | 写入 metadata 并用于后续检查 |
| `--csw` | LIME `clovCoeff` 或 `1.160920226` | Clover 系数 |
| `--seed` | `42` | NumPy PCG64 source seed |
| `--out-dir` | `data/production_<lat>` | 输出目录 |

输出：

- `gauge_<lat>_c64.h5`
- `source_<lat>_c64.h5`
- `case_manifest.json`

## 2. 生产求解

### `solve_case.py`

用途：选择 CUDA QCU、KUPL QCU 或 QUDA，执行一次冷求解。

```bash
python pyqcu/testing/simple/solve_case.py --backend <cuda|quda|kupl> [options]
```

通用参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--backend` | 必填 | `cuda`、`quda`、`kupl` |
| `--lime` | home 路径的 reference LIME | 输入 LIME |
| `--gauge` | 无 | LIME/HDF5 gauge；覆盖 `--lime` |
| `--source` | 自动生成 | canonical source HDF5 `[4,3,X,Y,Z,T]` |
| `--lat` | `24 24 24 64` | 目标格点 |
| `--mass` | LIME 或默认 | 质量 |
| `--csw` | LIME 或默认 | Clover 系数 |
| `--atol` | `1e-6` | 后端容差 |
| `--seed` | `42` | source seed |
| `--out-dir` | `data/production_<lat>` | 输出目录 |
| `--output` | `solution_<lat>_<backend>_c64.h5` | 解路径 |
| `--report` | `report_<backend>.json` | 后端报告路径 |

CUDA 参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--device` | `cuda:0` | 目标 GPU |
| `--libqcu` | `auto` | `auto/sm60/sm70/default` |

QUDA 参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--device` | `cuda:0` | 目标 GPU |
| `--quda-variant` | `auto` | `auto/sm60/sm70/default` |

KUPL 参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--kupl-ranks` | `4` | MPI rank 数 |
| `--kupl-threads` | `4` | 每 rank OpenMP 线程数 |
| `--kupl-grid` | 自动 | 4D process grid |
| `--kupl-lib` | `default` | `default/lto` 或 `.so` 路径 |
| `--skip-independent-residual` | 关闭 | 信任后端 fail-closed 真残差检查 |

行为：

- 先调用 `prepare_case` 逻辑准备输入。
- CUDA/QUDA 子进程通过隔离环境选择架构库。
- KUPL 通过 `mpirun --bind-to none` 启动。
- 解先写 `.partial`，仅 `PASS` 时发布正式文件。
- 总是写 `last_solve_<backend>.json` 摘要。
- 只有最终状态 `PASS` 返回 0。

## 3. 直接 CUDA QCU 示例

### `qcu_bistabcg.py`

```bash
python pyqcu/testing/simple/qcu_bistabcg.py [options]
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--lat` | `8 8 8 16` | 正确性格点 |
| `--mass` | LIME 或 `0.05` | 质量 |
| `--csw` | LIME 或 `1.160920226` | Clover 系数 |
| `--atol` | `1e-6` | 容差 |
| `--max-iter` | `1000` | 最大迭代 |
| `--device` | `cuda:0` | CUDA 设备 |
| `--gauge` | 无 | LIME/HDF5 gauge；无则生成 Gaussian gauge |
| `--source` | 自动生成 | full fermion HDF5 |
| `--seed` | `42` | 随机种子 |
| `--warmup` | `1` | 不计时求解次数 |
| `--repeats` | `3` | 计时次数，报告 median |
| `--output` | 无 | 可选解 HDF5 |
| `--report` | `performance_qcu_bistabcg.json` | 报告路径 |

### `qcu_multigrid.py`

```bash
python pyqcu/testing/simple/qcu_multigrid.py [options]
```

除 `qcu_bistabcg.py` 的通用参数外：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--levels` | `2` | hierarchy 层数 |
| `--dof` | `12 48 48` | 每层块自由度 |
| `--mg-grid` | `2 2 2 2` | 粗化 block size |
| `--restart` | `5` | coarse-correction interval |
| `--coarse-max-iter` | `200` | 粗层最大迭代 |
| `--coarse-tol-factor` | `1e4` | 粗层容差倍率 |
| `--nv-iters` | `2` | null-vector inverse iteration 次数 |
| `--no-cache` | 关闭 | 禁用 null-vector/coarse cache |

注意：该脚本的完整 coarse hierarchy 在 Python/CUDA 主进程中构建；`--levels 1`
只是 fine-only 路径。

## 4. 直接 KUPL 示例

### `kupl_bistabcg.py`

```bash
KUPL_OMP_THREADS=8 OMP_NUM_THREADS=8 \
python pyqcu/testing/simple/kupl_bistabcg.py [options]
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--lat` | `8 8 8 16` | 正确性格点 |
| `--mass` | 自动/LIME | 质量 |
| `--csw` | 自动/LIME | csw |
| `--atol` | `1e-6` | 容差 |
| `--max-iter` | `1000` | 最大迭代 |
| `--dtype` | `c64` | `c64/c128` |
| `--gauge` | 单位 gauge | LIME/HDF5 gauge |
| `--source` | 自动生成 | source HDF5 |
| `--seed` | `42` | source seed |
| `--output` | 无 | 解 HDF5 |
| `--report` | `performance_kupl_bistabcg.json` | 报告路径 |

该脚本为单 MPI rank 的 ABI/数值示例，不是多 NUMA 生产入口。生产多 rank
BiStabCG 使用 `mpi/solve_case.py --levels 1 --restart 0`；真正 Legacy-MG 使用
`--levels 2 --restart 5` 并复用身份匹配的全局 cache。

### `kupl_multigrid.py`

```bash
mpirun [OpenMPI options] -np <ranks> \
  python pyqcu/testing/simple/kupl_multigrid.py [options]
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--lat` | `8 8 8 16` | 正确性格点 |
| `--grid` | `(MPI size,1,1,1)` | process grid；乘积必须等于 MPI size |
| `--mass` / `--csw` | 自动/LIME | 物理参数 |
| `--atol` | `1e-6` | 容差 |
| `--max-iter` | `1000` | 最大迭代 |
| `--dtype` | `c64` | `c64/c128` |
| `--levels` | `2` | 只支持 1 或 2 |
| `--dof` | `12 2 2 2 2` | 每层 E |
| `--mg-grid` | `2 2 2 2` | 粗化 block |
| `--restart` | `5` | `0` 关闭 coarse correction |
| `--coarse-max-iter` | `200` | 粗层迭代上限 |
| `--coarse-tol-factor` | `1e4` | 粗层容差倍率 |
| `--nv-iters` | `2` | null-space iter |
| `--coarse-cache` | 无 | 全局 33-tensor 缓存 |
| `--write-coarse-cache` | 无 | 单 rank 导出缓存 |
| `--build-device` | `cpu` | `cpu/cuda` |
| `--no-deflate` | 关闭 | 禁用初始 coarse correction |
| `--verbose` | 关闭 | 运行时 verbose |
| `--verbose-build` | 关闭 | 粗算子构建 verbose |
| `--local-stencil` | 关闭 | local stencil builder |
| `--local-stencil-batch` | `4` | local builder batch |
| `--gauge` / `--source` | 自动 | 输入 |
| `--seed` | `42` | source/随机 gauge |
| `--skip-independent-residual` | 关闭 | 跳过 NumPy 真残差 |
| `--output` / `--report` | 解/报告路径 | 输出 |

多 rank 约束：

- `grid` 乘积等于 MPI rank 数。
- 每维 local extent 必须为偶数。
- `levels>1, restart!=0` 时多 rank 必须提供 `--coarse-cache`。
- 多 rank 不支持 `--write-coarse-cache`。

推荐生产编排使用 `mpi/solve_case.py`，它会自动构建/校验全局 cache、记录每 rank
NUMA/内存环境和独立真残差。`simple/solve_case.py` 与 `mpi/solve_case.py` 都在
内部启动 `mpirun`，外层不得再次嵌套。

环境检测与生产变量建议：

```bash
python -B pyqcu/testing/kupl/kupl_env.py \
  --ranks 4 --threads-per-rank 32 \
  --lattice 24 24 24 64 --dtype c64 --grid 1 1 2 2
```

默认输出 `logs/kupl_env-vYYYY-MM-DD-HH-MM-SS.txt`。没有唯一 HBM NUMA offset
证据时只建议 `KUPL_MEMORY_NODE=local`；`preferred` 不能作为实际使用 HBM 的保证。

## 5. 直接 QUDA 示例

### `quda_bistabcg.py`

```bash
python pyqcu/testing/simple/quda_bistabcg.py [options]
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--lat` | `8 8 8 16` | 正确性格点 |
| `--mass` | LIME 或默认 | 质量 |
| `--tol` | `1e-6` | QUDA tolerance |
| `--max-iter` | `1000` | 最大迭代 |
| `--csw` | LIME 或默认 | clover |
| `--gauge` | 无 | LIME/HDF5 |
| `--source` | 自动 | source |
| `--seed` | `42` | 随机种子 |
| `--warmup` | `1` | 不计时次数 |
| `--repeats` | `3` | 计时次数 |
| `--output` | 无 | QCU HDF5 解 |
| `--report` | `performance_quda_bistabcg.json` | 报告 |

相关环境变量：

- `QUDA_INSTALL`
- `QUDA_KAPPA_NORMALIZATION=1`
- `QUDA_VERBOSE`
- `QUDA_USE_INIT_GUESS_NO`
- `QUDA_BICGSTAB_INVERTER`

### `quda_multigrid.py`

| 参数 | 默认 | 说明 |
|---|---|---|
| `--lat` | `8 8 8 16` | 正确性格点 |
| `--mass` | 自动 | 质量 |
| `--tol` | `1e-6` | tolerance |
| `--max-iter` | `1000` | 最大迭代 |
| `--csw` | 自动 | csw |
| `--gauge` / `--source` | 自动 | 输入 |
| `--seed` | `42` | seed |
| `--output` | 无 | 解 |
| `--levels` | `2` | MG levels |
| `--nvec` | `24` | null-vector 数 |
| `--block` | `2 2 2 2` | geometric block |
| `--report` | `performance_quda_multigrid.json` | 报告 |

## 6. 双后端正确性对照

### `compare_bistabcg.py`

```bash
python pyqcu/testing/simple/compare_bistabcg.py \
  --gauge <lime-or-h5> --lat X Y Z T [options]
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--gauge` | 必填 | 共享 gauge |
| `--lat` | 必填 | 格点 |
| `--mass` / `--csw` | 自动 | 物理参数 |
| `--atol` | `1e-6` | 两侧容差 |
| `--max-iter` | `1000` | 两侧最大迭代 |
| `--qcu-device` | `cuda:0` | QCU GPU |
| `--seed` | `42` | source seed |
| `--timeout` | `3600` s | 每后端超时 |
| `--solution-rtol` | `5e-3` | 解相对 L2 差上限 |
| `--output` | `performance_simple_bistabcg_pair.json` | 汇总报告 |

输出目录还包含每次运行的 source、两侧报告和两侧解文件。

### `compare_kupl_bistabcg.py`

| 参数 | 默认 | 说明 |
|---|---|---|
| `--gauge` | 自动生成 | LIME/HDF5 gauge |
| `--lat` | `8 8 8 16` | 正确性格点 |
| `--mass` | `0.17` | 质量 |
| `--csw` | `0.8` | csw |
| `--atol` | `1e-6` | 容差 |
| `--max-iter` | `500` | 最大迭代 |
| `--seed` | `42` | gauge/source seed |
| `--kupl-threads` | `2` | KUPL OpenMP |
| `--timeout` | `3600` s | 后端超时 |
| `--solution-rtol` | `5e-3` | 解差上限 |
| `--output` | `performance_simple_kupl_quda_bistabcg.json` | 汇总 |

### `compare_kupl_multigrid.py`

在上一组参数基础上增加：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--levels` | `2` | 只支持 1/2 |
| `--dof` | `12 2` | level E |
| `--mg-grid` | `2 2 2 2` | block |
| `--restart` | `5` | coarse correction |
| `--coarse-max-iter` | `200` | coarse iterations |
| `--nv-iters` | `2` | null-space iteration |
| `--solution-rtol` | `5e-3` | 解差门限 |
| `--output` | `performance_simple_kupl_quda_multigrid.json` | 汇总 |

除解差外还要求 KUPL `verify_status=0`。

## 7. 三方性能测量

### `compare_three_way.py`

```bash
python pyqcu/testing/simple/compare_three_way.py [options]
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--lat` | `24 24 24 64` | 格点 |
| `--lime` | home reference LIME | 输入 |
| `--gauge` | 无 | HDF5 gauge，覆盖 LIME |
| `--source` | 自动 | source HDF5 |
| `--mass` / `--csw` | LIME/default | 参数 |
| `--atol` | `1e-6` | 容差 |
| `--seed` | `42` | source seed |
| `--warmup` | `1` | 不计时次数 |
| `--repeats` | `3` | 计时次数 |
| `--device` | `cuda:0` | CUDA/QUDA 设备 |
| `--libqcu` | `auto` | native库选择 |
| `--quda-variant` | `auto` | QUDA选择 |
| `--gpu-warmup-seconds` | `3.0` | GPU 时钟预热；0 关闭 |
| `--backends` | `cuda,quda,kupl` | 可传 `all` 或逗号子集 |
| `--kupl-ranks` | `4` | rank 数 |
| `--kupl-threads` | `4` | OpenMP |
| `--kupl-grid` | 自动 | process grid |
| `--kupl-lib` | `default` | KUPL library |
| `--out-dir` | `data/benchmark_three_way` | 输出 |

输出 `summary.json`、`summary.md` 和每后端独立 JSON。

## 8. KUPL benchmark 与缓存

### `benchmark_kupl.py`

用途：重复运行 C++ operator benchmark。

| 参数 | 默认 | 说明 |
|---|---|---|
| `--binary` | 自动查找 build binary | benchmark 可执行文件 |
| `--lat` | `16 16 16 32` | 格点 |
| `--grid` | `(1,1,1,1)` | process grid |
| `--iterations` | `10` | 每轮 kernel iteration |
| `--repeats` | `5` | 进程重复次数 |
| `--threads` | `16` | OpenMP threads |
| `--timeout` | 后端超时 | 秒 |
| `--output` | `performance_kupl_benchmark.json` | 汇总 |

### `benchmark_kupl_lime_mpi.py`

必须由外部 `mpirun` 启动：

```bash
mpirun --bind-to numa --map-by ppr:1:numa -np 4 \
  python pyqcu/testing/simple/benchmark_kupl_lime_mpi.py \
  --gauge reference.lime \
  --lat 24 24 24 64 \
  --grid 1 4 1 1 \
  --threads 8 \
  --warmups 1 --repeats 5
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--gauge` | 必填 | LIME/HDF5 |
| `--lat` | `24 24 24 64` | 全局格点 |
| `--grid` | `1 1 1 1` | 必须匹配 MPI size |
| `--mass` / `--csw` | 文件 metadata | 参数 |
| `--dtype` | `c64` | c64/c128 |
| `--atol` | `1e-6` | 容差 |
| `--max-iter` | `1000` | solver iterations |
| `--threads` | `1` | OpenMP |
| `--warmups` | `1` | warmup |
| `--repeats` | `5` | timed repeats |
| `--source` | 自动 | global source |
| `--seed` | `42` | seed |
| `--report` | `performance_kupl_lime_mpi.json` | 报告 |

### `build_full_mg_cache.py`

```bash
mpirun --allow-run-as-root --oversubscribe --bind-to none -np 4 \
  python -B -m pyqcu.testing.simple.mpi.build_full_mg_cache \
  --gauge gauge.h5 --lat X Y Z T \
  --mass M --csw C \
  --dof 12 12 --mg-grid 2 2 2 2 \
  --grid GX GY GZ GT \
  --output cache.h5
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--gauge` | 必填 | 目标格点 canonical HDF5 |
| `--lat` | 必填 | 四维格点 |
| `--mass` / `--csw` | 必填 | 物理参数 |
| `--dtype` | `c64` | c64/c128 |
| `--dof` | `12 12` | 两层 E |
| `--mg-grid` | `2 2 2 2` | coarse block |
| `--nv-iters` | `1` | null-space iteration |
| `--seed` | `42` | builder seed |
| `--build-device` | `cuda` | `cpu/cuda` |
| `--batch-size` | `4` | local stencil probe batch |
| `--grid` | 按 rank 自动 | 4D MPI process grid，乘积必须等于 rank 数 |
| `--ranks` | `MPI_COMM_WORLD` | 可选预期 rank 数校验 |
| `--output` | 必填 | cache HDF5 |
| `--report` | 无 | setup 和 SHA256 报告 |

MPI 入口固定构建 2-level 33-tensor cache。每个 rank 生成自己的 null-vector、
计算 coarse slab，并通过周期 overlap patch 完成 local stencil；rank 0 只负责
全局组装、manifest 和原子发布。父目录的 `build_full_mg_cache.py` 保留单进程
兼容行为。

## 9. Gaussian gauge 生成

### `give_gauss_gauge.py`

| 参数 | 默认 | 说明 |
|---|---|---|
| `--lat` | `8 8 8 16` | 四维必须为正偶数 |
| `--seed` | `42` | random seed |
| `--sigma` | `0.1` | Gaussian width |
| `--mass` | `0.05` | 写入 metadata/kappa |
| `--device` | `cpu` | torch device |
| `--backend` | `auto` | `auto/qcu/torch` |
| `--output` | `data/gauge_gauss_<lat>_seed<seed>_c64.h5` | 输出 |

需要 QCU 路径时应显式指定 `--backend qcu --device cuda:N`，避免 auto 在 CPU device 上
选择 QCU。

## 10. 格式转换与检查

### gauge

| 命令 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `quda_gauge2qcu_gauge.py` | LIME/QDP | QCU HDF5 | QDP `[dir,t,z,y,x,3,3]` -> QCU |
| `qcu_gauge2quda_gauge.py` | QCU HDF5 | QDP `.npy/.h5/.lime` | QCU -> QDP/QIO |

共同参数：`--input`、`--output`、`--dry-run`。

### fermion

| 命令 | 输入 | 输出 |
|---|---|---|
| `qcu_fermion2quda_fermion.py` | QCU `[4,3,x,y,z,t]` | QDP compressed `[2,t,z,y,x/2,4,3]` |
| `quda_fermion2qcu_fermion.py` | QDP compressed | QCU full |

共同参数：`--input`、`--output`、`--dry-run`。

### clover

| 命令 | 输入 | 输出 |
|---|---|---|
| `qcu_clover2quda_clover.py` | QCU `[4,3,4,3,x,y,z,t]` | packed `[2,t,z,y,x/2,2,36]` |
| `quda_clover2qcu_clover.py` | packed QUDA | QCU clover |

`quda_clover2qcu_clover.py` 额外支持 `--dtype complex64|complex128`。

### check 工具

| 命令 | 无输入时 | 有输入时 |
|---|---|---|
| `check_gauge.py` | 单位 gauge self-test | finite、roundtrip、unitarity |
| `check_fermion.py` | 零场 self-test | finite、roundtrip、norm |
| `check_clover.py` | Hermitian block self-test | finite、clover roundtrip |

共同参数：`--input`、`--backend auto|qcu|quda`、`--dataset data`。

## 11. 集群矩阵脚本

### `slurm_lime_backend_matrix.sh`

用途：在站点特定 Slurm/BSCC 环境中构建 QCU/QUDA 并运行 LIME rank/precision 矩阵。

常用环境变量：

| 变量 | 作用 |
|---|---|
| `PYQCU_ROOT` | 仓库根 |
| `PYQCU_REFERENCE_LIME` | 输入 LIME |
| `PYQCU_LIME_SCOPE=smoke|formal` | smoke 或正式矩阵 |
| `PYQCU_LIME_BACKENDS` | `qcu quda` |
| `PYQCU_LIME_PRECISIONS` | `c64 c128` |
| `PYQCU_LIME_RANKS` | `1 2 4` |
| `PYQCU_LIME_WARMUPS` | warmup 次数 |
| `PYQCU_LIME_REPEATS` | timed repeat 次数 |
| `PYQCU_LIME_MAX_ITER` | solver 最大迭代 |
| `PYQCU_LIME_TIMEOUT` | 每次 mpirun 超时 |
| `PYQCU_LIME_RESULT_ROOT` | 结果根目录 |
| `PYQCU_SKIP_BUILD=1` | 跳过 build/install |
| `PYQCU_MPI_DEVICE_AWARE` | CUDA-aware MPI 开关，默认 0 |

该脚本不是通用 local quickstart；先检查站点路径和 `BSCC_ENV`。

## 12. MPI 小格点入口

```bash
python pyqcu/testing/simple/mpi/solve_case.py --help
```

完整参数、示例和报告字段见 [`mpi/USER_GUIDE.md`](mpi/USER_GUIDE.md)。

## 13. 内部模块

以下文件不是日常用户 CLI：

| 文件 | 职责 |
|---|---|
| `_common.py` | 兼容门面、路径/默认值、HDF5/JSON、残差与状态共享函数 |
| `_lime.py` | 单文件 SciDAC/LIME 元数据与 gauge payload |
| `_layouts.py` | QCU/QDP/KUPL 与 Clover 布局转换 |
| `_runner.py` | GPU 隔离、library variant、KUPL 命令构造 |
| `kupl_residual.py` | KUPL/CUDA MPI 分布式独立真残差 |
| `test_coarse_cache.py` | coarse cache alias regression |
| `compare_three_way.py` | 三方重复计时与报告入口；旧 `bench_three_way.py` 转发壳已删除 |

## 14. 推荐帮助命令

```bash
python pyqcu/testing/simple/prepare_case.py --help
python pyqcu/testing/simple/solve_case.py --help
python pyqcu/testing/simple/qcu_bistabcg.py --help
python pyqcu/testing/simple/qcu_multigrid.py --help
python pyqcu/testing/simple/kupl_bistabcg.py --help
python pyqcu/testing/simple/kupl_multigrid.py --help
python pyqcu/testing/simple/quda_bistabcg.py --help
python pyqcu/testing/simple/quda_multigrid.py --help
python pyqcu/testing/simple/compare_three_way.py --help
python pyqcu/testing/simple/build_full_mg_cache.py --help
```
