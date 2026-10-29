# AGENTS.md — pyqcu/testing/python

主测试套件：Torch 后端算子与求解器测试。可跑 CPU、CUDA GPU 或昇腾 NPU（经 `pyqcu.cann`）。

## 测试文件

| 文件 | 测试内容 |
|---|---|
| `conftest.py` | 入口 — 从 `pyqcu.testing` 导入，取消注释所需测试 |
| `conftest.bistabcg.py` | BiStabCG 求解器（多格点尺寸、dtype、parity 模式） |
| `conftest.clover.bistabcg.py` | Clover BiStabCG 求解器 |
| `conftest.multigrid.py` | Wilson multigrid 求解器（多格点尺寸、max_level、num_restart） |
| `conftest.clover.multigrid.py` | Clover multigrid 求解器 |
| `test_torch_namespace.py` | `pyqcu.torch` canonical 实现位置与顶层兼容门面契约 |
| `test_api_contract.py` | P0 API 数据模型、能力、资产 manifest 与兼容结果契约 |
| `test_krylov_api.py` | LinearOperator、Torch Krylov driver、三后端选择器与原生 adapter 契约 |
| `test_tilelang_optional.py` | 无 TileLang 时单次 warning、CPU fallback 和 DCU 入口 skip 契约 |

## 用法

编辑 conftest 文件取消注释所需测试，然后运行：

```bash
mpirun -np 4 python pyqcu/testing/python/conftest.py
```
