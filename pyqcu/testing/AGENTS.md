# AGENTS.md — pyqcu.testing

全组件集成测试。测试函数为 Python 函数，由 `examples/*/conftest.py` 入口导入。

## 架构

所有测试函数位于 `pyqcu/testing/__init__.py`，从各子包（`lattice`、`solver`、`dslash`、`tools`、`smear`）导入。每个 `examples/*/conftest.py` 是 pytest 入口，导入并调用特定测试函数。conftest 文件手动编辑取消注释要运行的测试。

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
| `verify_nullvecs(S, lonv, lat_fine, lat_coarse, n_sample=4, stencil=None, verbose=False)` | null 向量质量四重诊断（2026-08-22 整合自 logs/test11 与 examples/qcu/dev73）：近零性 `||S v||/||v||`、幂迭代谱半径、块内正交性（Gram 矩阵）、可选 Galerkin 一致性 `A_c ≈ Pᵀ S P`（提供 33-tensor stencil 时）；返回 dict，判据由调用方断言 |

## 运行测试

```bash
cd examples && pytest .
mpirun -np 4 python examples/pyqcu/conftest.py
```

## 日志约定

`PYQCU::TESTING::<MODULE>::\n message`

## 重要提示

- 测试用 `tools.local_xyzt2whole_xyzt` / `tools.whole_xyzt2local_xyzt` 做 MPI 参考对比
- 参考 HDF5 在 `examples/data/`
- 测试中的 `path` 变量由 `pyqcu.__file__` 计算定位数据文件
- **R3 fix：** 测试含 `assert` 语句，pytest 可检测失败
