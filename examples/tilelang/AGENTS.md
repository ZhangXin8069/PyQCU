# AGENTS.md — examples.tilelang

CUDA 的 TileLang JIT 编译内核测试。覆盖 `pyqcu/tools/_einsum.py` 与 `pyqcu/tools/_matul.py` 中的 TileLang 集成。

## 文件

| 文件 | 用途 |
|---|---|
| `conftest.py` | TileLang 测试入口 |

## 运行

```bash
python examples/tilelang/conftest.py
```

需要安装 TileLang（可选依赖，缺失时静默降级）。
