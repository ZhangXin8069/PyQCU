# AGENTS.md — examples.pyqcu

主测试套件：纯 Python 算子与求解器测试。可在 CPU、CUDA GPU 或昇腾 NPU（经 `pyqcu.cann`）运行。

## 测试文件

| 文件 | 覆盖 |
|---|---|
| `conftest.py` | 入口 — 从 `pyqcu.testing` 导入，注释/取消注释选择测试 |
| `conftest.bistabcg.py` | BiStabCG 求解器（多格点规模、dtype、奇偶模式） |
| `conftest.clover.bistabcg.py` | Clover BiStabCG 求解器 |
| `conftest.multigrid.py` | Wilson 多重网格（多规模、max_level、num_restart） |
| `conftest.clover.multigrid.py` | Clover 多重网格求解器 |

## 运行

编辑 conftest 文件取消注释目标测试，然后：

```bash
mpirun -np 4 python examples/pyqcu/conftest.py
```
