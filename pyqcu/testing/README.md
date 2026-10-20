# PyQCU 测试入口

`pyqcu/testing/` 是 PyQCU 的统一测试树。顶层 `examples/` 已删除，原路径与分类关系如下：

| 原路径 | 当前路径 | 分类 |
|---|---|---|
| `examples/pyqcu/` | `pyqcu/testing/python/` | 纯 Python 主测试 |
| `examples/qcu/` | `pyqcu/testing/qcu/` | CUDA/C++/Cython 后端 |
| CPU KUPL backend | `pyqcu/testing/kupl/` | 独立 MPI+OpenMP CPU 后端 ABI/参考差分 |
| `examples/quda/` | `pyqcu/testing/quda/` | QUDA 单功能对照 |
| `examples/pyquda/` | `pyqcu/testing/pyquda/` | PyQUDA 对照 |
| `examples/{cpu,npu,dcu,gpu}/` | `pyqcu/testing/{cpu,npu,dcu,gpu}/` | 后端专项 |
| `examples/{benchmark,profiler,tilelang}/` | `pyqcu/testing/{benchmark,profiler,tilelang}/` | 基准、剖析与 TileLang |
| `examples/data/` | `pyqcu/testing/data/` | 参考数据与缓存 |

日常回归优先运行纯 Python 测试、QCU 单功能测试与 QUDA 对照测试。

```bash
source ./env.sh
python pyqcu/testing/qcu/single_qcu_api.py
python -m pytest -q pyqcu/testing/kupl/test_kupl_python.py
python pyqcu/testing/qcu/single_qcu_wilson_dslash.py
pytest pyqcu/testing/quda
cd pyqcu/testing && pytest .
```

单功能入口默认使用 `16 16 16 16` 格点。脚本启动时会打印当前脚本说明和解析后的输入参数，随后以分步进度条显示阶段耗时，并在结束时汇总总耗时与分项耗时。常用参数如下：

```bash
python pyqcu/testing/qcu/single_qcu_wilson_dslash.py --lat 8 8 8 8 --mass 0.05
python pyqcu/testing/qcu/single_qcu_wilson_dslash.py --pure-only  # 只跑纯 PyTorch 参考
python pyqcu/testing/qcu/single_qcu_wilson_dslash.py --no-doc --no-progress
```

`pyqcu/testing/quda/test_*.py` 使用相同的 `--lat`、`--mass`、`--no-doc` 和 `--no-progress` 选项；在
pytest 中会自动忽略 pytest 自身的命令行参数。

历史项目已按职责重组，不使用版本标签作为目录名：

| 当前路径 | 内容 |
|---|---|
| `qcu/multigrid/volume24/final/` | 24³×72 规范基准套件（图表/LaTeX 表/日志生成） |
| `qcu/multigrid/benchmarks/large_volume/` | 16×32×32×48 大体积加速比研究（dev84，报告 `report.md`） |
| `qcu/multigrid/benchmarks/multi_gpu/` | 多线程多卡基准 |
| `qcu/strict/quda_comparison/` | Strict MultiGrid 与 QUDA/PyQUDA 对照 |

dev73（`multigrid/legacy/`）、dev74（`multigrid/scaling/`）与 `benchmarks/{volume32,volume1632}`
等已被现行实现取代的套件已于 2026-10-15 删除，其可复用算法全部收入库内
（映射表见 `pyqcu/testing/qcu/AGENTS.md`；代码可由 git 历史恢复）。

上述目录保留历史入口和数据 provenance，但新代码和新文档必须使用职责命名。
测试产生的日志、缓存和数据由 `pyqcu/testing/.gitignore` 排除；新增或接口变更后
必须按 `pyqcu/testing/AGENTS.md` 的强制整理约定复查。
