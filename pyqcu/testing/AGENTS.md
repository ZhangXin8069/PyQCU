# AGENTS.md — pyqcu.testing

全组件集成测试。测试函数位于 `pyqcu/testing/__init__.py`，由 `examples/*/conftest.py` 引用（conftest 手动注释/取消注释决定运行哪些测试）。模块级 try/except import `tilelang`（`test_matmul` 用）。

## 测试函数

| 函数 | 覆盖 |
|---|---|
| `test_lattice(lat_size, dtype, device)` | SU(3) 生成 + γ²=I；断言 `check_su3` 为 True |
| `test_dslash_wilson(kappa, lat_size, dtype, device, with_data, support_parallel)` | 完整 Wilson 与 eo/oe 变体；`with_data=True` 对照参考 HDF5（`refer.wilson.*.L32K0_125.*.h5`）；相对差 < 1e-4 |
| `test_dslash_parity(lat_size, kappa, dtype, device)` | MPI 下奇偶预处理 Wilson+Clover；root 全格参考，各 rank 比对 `matvec_all`/`matvec_eeo`/`matvec_oeo` |
| `test_dslash_clover(device, with_data, dtype)` | Clover 项构造；parallel vs serial 对照 |
| `test_solver(kind, method, kappa, lat_size, dtype, device, with_data, max_level, num_restart, support_parity)` | BiCGStab / 多重网格（`init()`+`solve()`+`plot()`）；相对误差 < 1e-3 |
| `test_matmul()` | TileLang JIT vs PyTorch（cuBLAS/MKL）TFLOPS 对比 |
| `test_smear_stout(lat_size, device, dtype)` | MPI 下 stout smearing；SU(3) 前后校验 |

## 运行

```bash
cd examples && pytest .                        # 全部 conftest.py
mpirun -np 4 python examples/pyqcu/conftest.py # 单文件 + MPI
```

## 约定

- 输出日志前缀：`PYQCU::TESTING::<MODULE>::`；参考数据在 `examples/data/`；`path` 由 `pyqcu.__file__` 推算
- MPI 参考比对用 `tools.local_xyzt2whole_xyzt` / `whole_xyzt2local_xyzt`
- **R3 fix**：必须带 `assert`，pytest 才能检测失败
