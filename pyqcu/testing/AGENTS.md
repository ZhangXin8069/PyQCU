# AGENTS.md — pyqcu.testing

PyQCU 的集成测试、后端单功能测试、性能基准与历史回归资产统一位于本目录。

## 架构

跨组件测试函数位于 `pyqcu/testing/__init__.py`：公共 `pyqcu.lattice` / `pyqcu.tools`
与 Torch 实现 `pyqcu.torch.dslash|solver|smear` 分别导入。后端各自维护入口：

| 目录 | 分类与职责 |
|---|---|
| `python/` | 纯 Python/Torch 主测试：pytest 模块覆盖 API、Krylov、MG facade 与小格代数；`conftest*` 是手选参数的显式运行入口 |
| `qcu/` | C++ CUDA/Cython 后端；顶层 `test_*.py` 为定向 pytest，`single_qcu_*.py` 与 `conftest.*.py` 为手动/GPU/MPI 生命周期入口，Strict 对照在 `strict/quda_comparison/` |
| `kupl/` | 独立 MPI+OpenMP CPU 后端；pytest 模块覆盖 ctypes ABI、Krylov、严格/legacy MG 和 simple MG geometry，`test_strict.py` 等脚本也是直接 MPI 入口 |
| `simple/` | 生产冷求解与三方对照入口，以及 `test_coarse_cache.py`、`mpi/test_mpi_contract.py` 契约测试 |
| `quda/` | QUDA 单功能对照测试；公共适配在 `common.py` |
| `pyquda/` | 与 PyQUDA 的隔离进程对比套件 |
| `cpu/`、`npu/`、`dcu/`、`gpu/` | 后端专项或占位测试 |
| `benchmark/`、`profiler/` | 性能基准与 Profiler 入口 |
| `benchmark/mg_current/` | `data/mg-bench-current-20260830/` 的汇总与 GPU 监控维护脚本 |
| `data/` | `with_data=True` 使用的参考 HDF5 与缓存 |
| `regression/session-2026-08-24/` | bug31–37 跨组件无人值守回归脚本 |

迁移边界：顶层 `examples/` 已删除，原内容只按当前 backend 目录使用。旧的
`qcu/multigrid/{legacy,scaling}/`、`benchmarks/{volume32,volume1632}` 等套件也已删除，
不是工作区入口，不要在新文档或命令中恢复旧路径；可复用算法已进入现行实现，
历史报告只在 Git 标签中用 `git show <tag>:<path>` 读取。现役大/多卡 MG 入口仅为
`qcu/multigrid/volume24/final/`、`qcu/multigrid/benchmarks/{large_volume,multi_gpu}/`，
Strict/QUDA 对照仅为 `qcu/strict/quda_comparison/`。

## 入口选择

| 场景 | 使用方式 |
|---|---|
| 可重复的单元/契约测试 | 运行定义了 `test_*` 函数的文件并显式传入目录或模块；不会加载手工参数列表 |
| 手选格点、dtype、device 或 MPI 的旧式测试 | 编辑或直接运行 `conftest.py` / `conftest.<backend>.<function>.py`，不要把它当 pytest node |
| CUDA QCU 单接口诊断 | 运行 `qcu/single_qcu_<function>.py`；纯参考可用 `--pure-only`，真实 CUDA 生命周期需扩展和显式请求 |
| MPI 小格点验收 | 优先运行 `simple/mpi/solve_case.py`，由启动器组织 rank/grid；低层手工入口才直接使用 `mpirun` |
| 多卡一致性/吞吐 | 仅显式运行 `qcu/conftest.multi_gpu.py`、`qcu/bench_multi_gpu.py` 或 Slurm 脚本；它们不是生产拓扑 |

`test_*.py` 只有在文件内定义 pytest 测试函数时才适合直接收集；`kupl/test_strict.py`、
`kupl/test_legacy_mg.py` 和 `kupl/test_dslash_block_mpi.py` 是可直接执行的 MPI 脚本。
新增目标先运行 `pytest --collect-only -q <path>` 确认实际 node。

推荐的定向 pytest 命令（从仓库根执行）：

```bash
source ./env.sh
python -m pytest -q pyqcu/testing/python
python -m pytest -q pyqcu/testing/simple
python -m pytest -q pyqcu/testing/quda
python -m pytest -q pyqcu/testing/qcu/test_qcu_api_smoke.py
python -m pytest -q pyqcu/testing/kupl/test_kupl_python.py \
  pyqcu/testing/kupl/test_krylov_api.py \
  pyqcu/testing/kupl/test_simple_mg_geometry.py
```

不要把 `cd pyqcu/testing && pytest .` 当作默认门禁：全树混有可选 CUDA/QUDA、
手工脚本和当前仍在演进的 Strict 诊断模块。全树收集只用于盘点，失败时应回到
具体 backend 文件定位。

TileLang 当前没有独立测试目录或 pytest node；需要验证时显式调用
`pyqcu.testing.test_matmul()`，不可把空收集或退出码 0 视为通过。

显式脚本入口示例：

```bash
python pyqcu/testing/python/conftest.py
mpirun -np 4 python pyqcu/testing/python/conftest.py
python pyqcu/testing/qcu/single_qcu_api.py
mpirun -np 1 python pyqcu/testing/qcu/conftest.clover.multigrid.py
python pyqcu/testing/qcu/conftest.multi_gpu.py
```

## API/Krylov/新 facade 定位

| 测试 | 覆盖范围 |
|---|---|
| `python/test_api_contract.py` | `pyqcu.api` 的 geometry/config/control、capability、asset manifest、`SolveResult` 与兼容返回契约 |
| `python/test_krylov_api.py` | `LinearOperator`、Torch BiCGStab/CG driver、统一 `pyqcu.solver.Krylov`，以及 CUDA/KUPL adapter 的无硬件契约测试 |
| `python/test_multigrid_api.py` | 三后端统一 `Multigrid` 名称、选择器、生命周期和 ABI 委托契约 |
| `python/test_multigrid.py` | CPU 小格的 Strict/Legacy transfer、Galerkin、MATPC、FGMRES/V-cycle 代数回归 |
| `python/test_torch_namespace.py` | `pyqcu.torch` canonical 实现位置和顶层兼容门面 |
| `kupl/test_krylov_api.py` | 真实 KUPL ctypes 的 Wilson BiCGStab/normal-CG 整次 solve 与独立 full residual |
| `qcu/test_qcu_api_smoke.py` | CUDA Cython bridge 与 `pyqcu.h` 的符号集合；`QCU_API_REQUIRE_CUDA=1` 时追加最小 init/end |

纯 Python API/Krylov 快速回归：

```bash
python -m pytest -q \
  pyqcu/testing/python/test_api_contract.py \
  pyqcu/testing/python/test_krylov_api.py \
  pyqcu/testing/python/test_multigrid_api.py
```

真实 KUPL 回归需要先构建 `cpp/kupl/qcu/build/libqcu.so`；缺库时测试应明确 skip。

统一 `Multigrid` 的默认 `variant` 是 `legacy`。测试不得把 Strict 当作隐式默认；
Strict 用例必须显式写 `variant="strict"` 或直接构造 `StrictMultigrid`。

模块级 `import tilelang`（try/except 回退）供 `test_matmul` 使用。

## 测试函数

| 函数 | 测试内容 |
|---|---|
| `test_lattice(lat_size, dtype, device)` | SU(3) 规范生成 + gamma 代数（γ_μ²=I；`check_su3` 须 True） |
| `test_dslash_wilson(kappa, lat_size, dtype, device, with_data, support_parallel)` | Wilson 算子；`with_data=True` 对照 `data/refer.wilson.*.L16K0_125.*.h5`；相对差 < 1e-4 |
| `test_dslash_parity(lat_size, kappa, dtype, device)` | 奇偶预处理 Wilson+Clover + MPI；root 算全算子参考，各 rank 对比局部奇偶算子；测 `matvec_all` 与 `matvec_eeo`/`matvec_oeo` |
| `test_dslash_clover(device, with_data, dtype)` | Clover 项构造；`with_data=True` 对照参考数据校验 clover 项与逆 |
| `test_solver(kind, method, kappa, lat_size, dtype, device, with_data, max_level, num_restart, support_parity)` | BiStabCG 与 multigrid；`method='bistabcg'` 标准/奇偶预处理 BiCGStab；`method='multigrid'` 全 V-cycle `init()+solve()+plot()`；相对误差 < 1e-3 |
| `test_matmul()` | TileLang JIT 矩阵乘 vs PyTorch（GPU 4096² vs cuBLAS；CPU 1024² vs MKL/OneDNN）；打印 TFLOPS 对比表 |
| `test_smear_stout(lat_size, device, dtype)` | 跨 MPI 网格 stout smearing；各 rank 对比局部并行 smear vs 整网格参考；smearing 前后验证 SU(3) |
| `test_npu_emulation(lat_size, dtype, device)` | cann 层 NPU 复数分解路径模拟（force_use_npu=True，CPU 上同构真机行为）：lattice/wilson/stout/bistabcg 四组件；开关 try/finally 恢复 |
| `test_smear_wuppertal(lat_size, rho, nstep, device, dtype)` | Wuppertal 三重不变量回归（2026-08-24）：nstep≥1 防护、自由场(U=I)常数场不动点（wards 含 t 时发散，bug33）、默认参数白噪声范数收缩 |
| `test_h5py_multithread(nthreads, dtype, lat_size)` | h5py 多线程并发读写；由 `python/conftest.py` 的注释入口手工启用 |
| `test_multigrid_multithread(nthreads, lat_size, mass, tol)` | 单进程多线程 MG；由 `qcu/conftest.multigrid.mt.py` 手工运行 |
| `test_multi_gpu_multigrid(...)` | 测试专用单进程多设备 MG 一致性/吞吐；仅由 `qcu/conftest.multi_gpu.py` 或基准显式调用 |
| `verify_nullvecs(S, lonv, lat_fine, lat_coarse, n_sample=4, stencil=None, verbose=False)` | null 向量质量四重诊断：近零性 `||S v||/||v||`、幂迭代谱半径、块内正交性（Gram 矩阵）、可选 Galerkin 一致性 `A_c ≈ Pᵀ S P`；返回 dict，判据由调用方断言 |

## 日志约定

`PYQCU::TESTING::<MODULE>::\n message`

## 重要提示

- 测试用 `tools.local_xyzt2whole_xyzt` / `tools.whole_xyzt2local_xyzt` 做 MPI 参考对比
- 参考 HDF5 在 `pyqcu/testing/data/`
- 测试中的参考数据路径由 `pyqcu/testing/__init__.py` 自身位置计算
- **R3 fix：** 测试含 `assert` 语句，pytest 可检测失败

## 强制整理约定

独立项目测试完成后或公开接口变化后，必须复查本目录：修复失效单测、删除无关重复
脚本、仅保留必要入口。生成物由 `pyqcu/testing/.gitignore` 排除；新增文件名应使用
`test_<backend>_<function>.py`（含 pytest 测试函数）或
`conftest.<backend>.<function>.py`（显式脚本入口）。整理后至少运行
`git diff --check` 与受影响的最小测试集。
