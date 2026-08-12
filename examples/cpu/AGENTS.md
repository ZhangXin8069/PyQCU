# AGENTS.md — examples.cpu

纯 Python 后端的 CPU 专用测试。

## 测试文件

| 文件 | 覆盖 |
|---|---|
| `conftest.py` | 主入口 — 从 `pyqcu.testing` 导入 |
| `conftest.bistabcg.py` | CPU 上 BiStabCG 求解器 |
| `conftest.mpi.py` | CPU 上 MPI 分布式求解器 |

## 运行

```bash
mpirun -np 4 python examples/cpu/conftest.py
```
