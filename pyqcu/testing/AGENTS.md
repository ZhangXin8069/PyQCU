# AGENTS.md — pyqcu.testing

PyQCU 的集成测试、后端单功能测试、性能基准与历史回归资产统一位于本目录。
原先的顶层 `examples/` 已整体迁入此处，不再保留兼容副本。

## 架构

跨组件测试函数位于 `pyqcu/testing/__init__.py`：公共 `pyqcu.lattice` / `pyqcu.tools`
与 Torch 实现 `pyqcu.torch.dslash|solver|smear` 分别导入。后端各自维护入口：

| 目录 | 分类与职责 |
|---|---|
| `pyqcu/` | 纯 Python 算子/求解器主测试与对应运行入口 |
| `qcu/` | C++ CUDA/Cython 后端测试；`multigrid/` 现存 `volume24/final/` 与 `benchmarks/{large_volume,multi_gpu}`，Strict 对照在 `strict/quda_comparison/` |
| `kupl/` | 独立 MPI+OpenMP CPU 后端的 Python ctypes ABI、canonical layout、PyQCU 参考差分与 Clover BiStabCG 真残差测试 |
| `quda/` | QUDA 单功能对照测试；公共适配在 `common.py` |
| `pyquda/` | 与 PyQUDA 的隔离进程对比套件 |
| `cpu/`、`npu/`、`dcu/`、`gpu/` | 后端专项或占位测试 |
| `benchmark/`、`profiler/`、`tilelang/` | 性能基准、Profiler、TileLang 入口 |
| `benchmark/mg_current/` | `data/mg-bench-current-20260830/` 的汇总与 GPU 监控维护脚本 |
| `data/` | `with_data=True` 使用的参考 HDF5 与缓存 |
| `regression/session-2026-08-24/` | bug31–37 跨组件无人值守回归脚本 |

已被现行实现取代的 dev76/dev78/test11–14 与 dev73/dev74 套件
（`qcu/multigrid/{legacy,scaling}/`、`benchmarks/{volume32,volume1632}`）已于 2026-10-15 删除，
可复用算法全部收入库内；映射表见 `qcu/AGENTS.md`。`strict/quda_comparison/` 保留
Strict-MG 与 QUDA/PyQUDA 对照的现役套件（`run_strict_fast.py`、`bench_strict_vs_quda.py`、
`run_all.py`、`component_cuda.py` 及 `plot_mg_*.py` 报告作图）。

`conftest.py` 与 `conftest.<backend>.<function>.py` 是显式运行入口，通常需要
手动取消注释目标测试；`test_*.py` 是可由 pytest 直接收集的测试模块。

模块级 `import tilelang`（try/except 回退）供 `test_matmul` 使用。

## 测试函数

| 函数 | 测试内容 |
|---|---|
| `test_lattice(lat_size, dtype, device)` | SU(3) 规范生成 + gamma 代数（γ_μ²=I；`check_su3` 须 True） |
| `test_dslash_wilson(kappa, lat_size, dtype, device, with_data, support_parallel)` | Wilson 算子；`with_data=True` 对照参考 HDF5（`refer.wilson.*.L32K0_125.*.h5`）；相对差 < 1e-4 |
| `test_dslash_parity(lat_size, kappa, dtype, device)` | 奇偶预处理 Wilson+Clover + MPI；root 算全算子参考，各 rank 对比局部奇偶算子；测 `matvec_all` 与 `matvec_eeo`/`matvec_oeo` |
| `test_dslash_clover(device, with_data, dtype)` | Clover 项构造；`with_data=True` 对照参考数据校验 clover 项与逆 |
| `test_solver(kind, method, kappa, lat_size, dtype, device, with_data, max_level, num_restart, support_parity)` | BiStabCG 与 multigrid；`method='bistabcg'` 标准/奇偶预处理 BiCGStab；`method='multigrid'` 全 V-cycle `init()+solve()+plot()`；相对误差 < 1e-3 |
| `test_matmul()` | TileLang JIT 矩阵乘 vs PyTorch（GPU 4096² vs cuBLAS；CPU 1024² vs MKL/OneDNN）；打印 TFLOPS 对比表 |
| `test_smear_stout(lat_size, device, dtype)` | 跨 MPI 网格 stout smearing；各 rank 对比局部并行 smear vs 整网格参考；smearing 前后验证 SU(3) |
| `test_npu_emulation(lat_size, dtype, device)` | cann 层 NPU 复数分解路径模拟（force_use_npu=True，CPU 上同构真机行为）：lattice/wilson/stout/bistabcg 四组件；开关 try/finally 恢复 |
| `test_smear_wuppertal(lat_size, rho, nstep, device, dtype)` | Wuppertal 三重不变量回归（2026-08-24）：nstep≥1 防护、自由场(U=I)常数场不动点（wards 含 t 时发散，bug33）、默认参数白噪声范数收缩 |
| `verify_nullvecs(S, lonv, lat_fine, lat_coarse, n_sample=4, stencil=None, verbose=False)` | null 向量质量四重诊断（整合自 `pyqcu/testing/qcu/multigrid/legacy/`，该套件 2026-10-15 已删除，本函数为规范位置）：近零性 `||S v||/||v||`、幂迭代谱半径、块内正交性（Gram 矩阵）、可选 Galerkin 一致性 `A_c ≈ Pᵀ S P`（提供 33-tensor stencil 时）；返回 dict，判据由调用方断言 |

## 运行测试

```bash
cd pyqcu/testing && pytest .
mpirun -np 4 python pyqcu/testing/python/conftest.py
python pyqcu/testing/qcu/single_qcu_api.py
pytest pyqcu/testing/quda
```

历史套件按目录直接运行，例如：

```bash
python pyqcu/testing/qcu/strict/quda_comparison/run_strict_fast.py --list
```

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
`test_<backend>_<function>.py` 或 `conftest.<backend>.<function>.py`。整理后至少运行
`git diff --check` 与受影响的最小测试集。
