# AGENTS.md — PyQCU

PyQCU：Lattice QCD 的 Python/Cython 库 —— CUDA 加速的 Wilson/Clover Dirac 算子、BiStabCG 与多重网格求解器、stout smearing、规范场生成，全部 MPI 分布于 4D 进程网格。

## 概要命令

- 环境：`source ./env.sh`（LD_LIBRARY_PATH、PYTHONPATH、MPI root 运行权限）
- 构建 C++ CUDA 后端：`bash ./build.sh`（→ `libqcu.so`，cd cpp/cuda/qcu && make.sh）
- 构建 Cython 扩展：`bash ./install.sh`（setup.py build_ext --inplace）
- 测试：`cd examples && pytest .`；MPI 单文件：`mpirun -np 4 python examples/pyqcu/conftest.py`
- Git 标签：`stab<N>`/`dev<N>`/`bug<N>` 独立编号 + 子版本（如 `stab15_1`），见 tag 技能
- Python ≥ 3.10，依赖 PyTorch、Cython、mpi4py、h5py、numpy、CUDA toolkit；TileLang 可选

## 架构：两个执行层级

1. **纯 Python**（`pyqcu/dslash/`、`pyqcu/solver/`、`pyqcu/smear/`）— PyTorch 实现，跑 CPU/CUDA/昇腾 NPU（经 `pyqcu.cann` 兼容层）。所有 Python 代码 `import pyqcu.cann as _torch`，不得直接 `import torch`（NPU 不支持复数张量，cann 层自动分解实虚部）。
2. **C++ CUDA 后端**（`cpp/cuda/qcu/`）— 手写 CUDA 内核 + MPI halo 交换，经 Cython 桥 `pyqcu/cuda/qcu/qcu.pyx` 暴露，生产路径。

## 关键惯例

- **参数协议**：`params`（int32[54]）、`argv`（float[7]）、`set_ptrs`（int64[100]）三个扁平张量桥接 Python↔C++；`pyqcu/cuda/define.py` 与 `cpp/cuda/qcu/include/define.h` 必须同步。
- **`_SET_PLAN_` 计划选择**：-2 Laplacian、-1 Gauss gauge、0 Wilson dslash、1 BiStabCG/CG、2 Clover dslash。
- **调用生命周期**：`applyInitQcu` → 操作 → **`params[define._SET_INDEX_] += 1`（每次调用间必须递增！）** → `applyEndQcu`。不递增导致 scratch 缓冲复用冲突、结果错误。
- **张量布局**：规范场 `[3,3,4,Lx,Ly,Lz,Lt]`、费米子场 `[4,3,Lx,Ly,Lz,Lt]`、Clover 项 `[4,3,4,3,Lx,Ly,Lz,Lt]`；时空维永远是最后 4 轴（`...xyzt`），ward 索引用负整数（`wards['x']=-4`）。HDF5 内部用 `zyxt` 序，经 `ccdxyzt2ccdptzyx`/`scxyzt2psctzyx` 转换。
- **日志约定**：`PYQCU::MODULE::SUBMODULE:\n message`，由 verbose 标志控制。
- **测试**：测试函数在 `pyqcu/testing/__init__.py`，`examples/*/conftest.py` 手动取消注释要运行的测试。
- **多线程多卡（一线程一卡）**：`pyqcu/cuda/_multi_gpu.py`（`MultiGpuMultigrid`）单进程内 N 线程 × 卡绑定并行；每线程独立 `params/argv/set_ptrs` 副本（`_SET_INDEX_` 各自从 0 计数）。Cython 桥（`qcu.pyx`）全部函数在 GIL 段取指针、`with nogil` 调 C++（真并行）；pxd 的 cdef extern 声明必须带 `nogil` 关键字，且 pxd 声明名不得与 pyx 内 def 同名（用 `qcu_api.pxd` 别名 cimport）。MultiGpuMultigrid 要求单 MPI rank（C++ LatticeSet 用 COMM_WORLD rank 覆盖 `_NODE_RANK_`）。
- **HDF5 持久化（h5py）**：所有保存/读取走 h5py；`pyqcu/tools/_io.py` 的 `save_tensor_h5`/`load_tensor_h5`（每调用独立 File 句柄，多线程安全）+ MPI mpio 路径（`gridoooxyzt2hdf5oooxyzt`）。null-vector/粗网格算子缓存 `.h5`（单句柄一次写全部 dataset，勿逐 dataset 覆盖重建）。

## 目录结构

| 路径 | 内容 |
|---|---|
| `pyqcu/` | 纯 Python 实现：`lattice/`（gamma/Gell-Mann 矩阵、SU(3)）、`dslash/`（Wilson/Clover 算子）、`solver/`（BiStabCG、multigrid；`_gmres.py` 为占位）、`smear/`（stout）、`tools/`（MPI 网格、HDF5 I/O、linalg、multigrid 工具含 33-tensor stencil build、TileLang JIT）、`testing/`（集成测试）、`cuda/`（Cython 桥 + `_schur_op.py` 多线程 Schur 算子 + `_multi_gpu.py` 多线程多卡 MG 驱动）、`cann/dtk/maca`（NPU 兼容层与占位） |
| `cpp/cuda/qcu/` | C++ CUDA 后端：`src/`（.cu 内核）、`include/`（26 个模板头）、`python/pyqcu.h`（C API，须与 qcu.pxd 同步）、`logs/` |
| `cpp/{cann,dtk,maca}/qcu/` | 占位 PASS，无实现 |
| `examples/` | 测试入口：`pyqcu/`（主套件）、`qcu/`（C++ 后端；dev 套件归档于 `qcu/dev73/`、`qcu/dev74/`，产物写 `logs/dev73/`、`logs/dev74/`）、`cpu/npu/dcu/gpu/tilelang/profiler/benchmark/`、`data/`（参考 HDF5） |
| `docs/` | dims.md、env.md、install.md、examples.md、profiler.md |
| `refer/` | 开发历史报告（dev71.*） |
| `logs/` | 按 tag 归档：`dev<N>/`、`stab<N>/`、`bug<N>/` 子目录（如 `dev73/`、`dev73/stab24/`、`dev74/`、`bug30/`；`logs/<tag>/**` 在 .gitignore 全豁免入库），根目录留 `fix-report-*.md`、`debug/`、`results/` 与共享缓存 `nullvec_cache/`；测试套件 `test11/`（历史版）与 `test12/`（当前版，产物归档 `test12/v<ts>/`） |

## 已知反模式（勿重复）

- 复数 `operator*=` 覆盖 `_data.x` 前先使用（lattice_complex.h）
- `cudaMallocAsync` 缓冲大小与内核写大小不匹配
- 裸 `except:` 吞掉 KeyboardInterrupt
- `torch.linalg.inv` 逐点 for 循环（须批量）
- 用对象（如 `self.sitting`）做 truthy 判断
- `nstep>1` 循环内不更新 U（stout）
- `python_requires` 低于 3.8
