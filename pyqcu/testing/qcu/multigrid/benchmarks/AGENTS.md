# AGENTS.md — pyqcu/testing/qcu/multigrid/benchmarks

CUDA MultiGrid 正式对照与专用基准入口。

- `legacy_vs_strict.py` 与 `test_legacy_vs_strict.py`：比较口径、cache provenance、
  粗层执行证据和纯 CPU 回归。
- `large_volume/`：16×32×32×48 历史大体积研究，不得替代统一正确性格点。
- `multi_gpu/`：测试专用单进程多卡吞吐/一致性入口，不是生产拓扑。
- 基准输出必须记录 dtype、格点、rank/grid、线程、cache identity、迭代数、真残差和
  timing scope；不得把 reference 或 setup 时间混入 solve-only 指标。
- 单文件 CPU 回归使用
  `python -m pytest -q pyqcu/testing/qcu/multigrid/benchmarks/test_legacy_vs_strict.py`；
  其余入口按子目录规则显式启动。
