# AGENTS.md — examples.qcu

经 Cython 桥从 Python 调用 `libqcu.so` 的 C++ CUDA 后端测试。

## 测试文件

| 文件 | 覆盖 |
|---|---|
| `conftest.cuda.py` | CUDA 可用性与上下文基础 |
| `conftest.mpi.py` | MPI 网格设置与 halo 交换 |
| `conftest.wilson.bistabcg.py` | C++ 后端 Wilson BiStabCG |
| `conftest.wilson.bistabcg.dslash.py` | Wilson BiStabCG dslash 内核 |
| `conftest.wilson.cg.py` | Wilson CG 求解器 |
| `conftest.clover.py` | Clover 项构造 |
| `conftest.clover.bistabcg.py` | Clover BiStabCG 求解器 |
| `conftest.clover.bistabcg.dslash.py` | Clover BiStabCG dslash |
| `conftest.clover.multigrid.py` | Clover 多重网格 V-cycle 求解器 |

## 运行

```bash
mpirun -np 1 python examples/qcu/conftest.clover.multigrid.py
```

输出：收敛日志 → `logs/clover_multigrid.log`，性能报告 → `logs/clover_multigrid_report.log`。

## Dev73_5 多重网格基准套件

dev73_5 多重网格性能里程碑的开发脚本：对 `applyCloverMultigridQcu` 与参考 `applyCloverBistabCgQcu` 做精度/格点/求解器参数扫描，产出 `logs/dev73_5.*`（报告、LaTeX 表、PNG 图）。

| 文件 | 用途 |
|---|---|
| `mg_dev73_5_clean.py` | 单配置隔离进程计时（ref/mg 交错，min+median 加速比） |
| `mg_dev73_5_bench.py` | 扩展性能基准 — 精度/格点/参数扫描 vs BiStabCG |
| `mg_dev73_5_verify.py` | 正确性检查 — SU(3) 规范场、解误差、null 向量零模/正交性、C++ vs Python 粗 dslash |
| `mg_dev73_5_collect.py` | 聚合 clean/bench/verify 的 JSON 到 `logs/dev73_5_results.json` |
| `mg_dev73_5_mktable.py` | 为 `dev73_5.tex` 生成 LaTeX 表片段（`logs/dev73_5_tbl_*.tex`） |
| `mg_dev73_5_plots.py` | 生成收敛/热点/加速比/时间 PNG 图到 `logs/` |
