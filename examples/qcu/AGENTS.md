# AGENTS.md — examples/qcu

C++ CUDA 后端测试（经 Cython 桥）。从 Python 驱动 `libqcu.so`。

## 测试文件

| 文件 | 测试内容 |
|---|---|
| `conftest.cuda.py` | CUDA 可用性与上下文 |
| `conftest.mpi.py` | MPI 网格设置与 halo 交换 |
| `conftest.wilson.bistabcg.py` | C++ 后端 Wilson BiStabCG |
| `conftest.wilson.bistabcg.dslash.py` | Wilson BiStabCG dslash 内核 |
| `conftest.wilson.cg.py` | Wilson CG 求解器 |
| `conftest.clover.py` | Clover 项构造 |
| `conftest.clover.bistabcg.py` | Clover BiStabCG 求解器 |
| `conftest.clover.bistabcg.dslash.py` | Clover BiStabCG dslash |
| `conftest.clover.multigrid.py` | Clover multigrid V-cycle 求解器 |

## 用法

```bash
mpirun -np 1 python examples/qcu/conftest.clover.multigrid.py
```

输出：收敛日志 → `logs/clover_multigrid.log`，性能报告 → `logs/clover_multigrid_report.log`

## Dev73_5 Multigrid Benchmark 套件

`mg_dev73_5_*.py` 系列脚本对 `applyCloverMultigridQcu` vs Clover BiStabCG 参考（`applyCloverBistabCgQcu`）做精度/格点/求解器参数扫描，产出 `logs/dev73_5.*`（报告、LaTeX 表、PNG 图）。脚本清单见 examples/qcu 目录。
