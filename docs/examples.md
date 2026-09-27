# PyQCU 测试与示例

> 迁移公告（2026-09-27）：原顶层 `examples/` 已删除，全部内容迁入
> `pyqcu/testing/`。历史文档、日志和数据记录中的 `examples/...` 路径均应按
> `pyqcu/testing/...` 解读；当前代码、脚本和可执行文档已使用新路径。

## 目录分类

| 分类 | 路径 | 内容 |
|---|---|---|
| 纯 Python | `pyqcu/testing/python/` | 算子、求解器与 MPI 集成入口 |
| QCU | `pyqcu/testing/qcu/` | C++ CUDA/Cython 单功能测试；MultiGrid 按 legacy/scaling/benchmark 分类，Strict 对照单独归档 |
| QUDA | `pyqcu/testing/quda/` | QUDA 单功能对照与布局/归一化锚定 |
| PyQUDA | `pyqcu/testing/pyquda/` | 独立进程对比、聚合与绘图 |
| 其他后端 | `pyqcu/testing/{cpu,npu,dcu,gpu}/` | CPU、NPU、DCU 模板与占位入口 |
| 工具类 | `pyqcu/testing/{benchmark,profiler,tilelang}/` | 基准、Profiler、TileLang |
| 数据 | `pyqcu/testing/data/` | `with_data=True` 参考 HDF5 与可再生缓存 |

## 常用命令

```bash
source ./env.sh
cd pyqcu/testing && pytest .
mpirun -np 4 python pyqcu/testing/python/conftest.py
python pyqcu/testing/qcu/single_qcu_api.py
pytest pyqcu/testing/quda
```

历史归档保留各自 README/AGENTS 与运行入口，但命令前缀统一改为
`pyqcu/testing/qcu/...`。
