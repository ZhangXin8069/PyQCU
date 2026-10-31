# AGENTS.md — pyqcu/testing/qcu/multigrid

CUDA Legacy/Strict MultiGrid 的基准与大格点研究容器；正确性门仍统一使用
`8x8x8x16`，这里的更大格点只用于性能、容量或历史证据。

- `benchmarks/`：Legacy/Strict 对照、大体积与测试专用多卡基准。
- `volume24/`：24³×72 规范性能套件及报告生成。
- 各基准是显式入口，不把整个目录作为 pytest 默认门禁；先读对应子目录 `AGENTS.md`。
- Legacy 是默认路线，Strict 行必须明确标注；不得混用 L1、BiCGStab、setup 与
  solve-only 的计时口径。
- 纯 CPU 契约测试可直接运行
  `python -m pytest -q pyqcu/testing/qcu/multigrid/benchmarks/test_legacy_vs_strict.py`；
  GPU/MPI 长任务只在任务需要时显式运行。
