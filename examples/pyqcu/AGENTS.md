# AGENTS.md — examples/pyqcu

主测试套件：纯 Python 算子与求解器测试。可跑 CPU、CUDA GPU 或昇腾 NPU（经 `pyqcu.cann`）。

## 测试文件

| 文件 | 测试内容 |
|---|---|
| `conftest.py` | 入口 — 从 `pyqcu.testing` 导入，取消注释所需测试 |
| `conftest.bistabcg.py` | BiStabCG 求解器（多格点尺寸、dtype、parity 模式） |
| `conftest.clover.bistabcg.py` | Clover BiStabCG 求解器 |
| `conftest.multigrid.py` | Wilson multigrid 求解器（多格点尺寸、max_level、num_restart） |
| `conftest.clover.multigrid.py` | Clover multigrid 求解器 |

## 用法

编辑 conftest 文件取消注释所需测试，然后运行：

```bash
mpirun -np 4 python examples/pyqcu/conftest.py
```
