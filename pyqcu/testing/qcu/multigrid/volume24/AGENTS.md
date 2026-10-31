# AGENTS.md — pyqcu/testing/qcu/multigrid/volume24

24³×72 CUDA Clover MultiGrid 大格点性能套件的容器目录。

- 现役实现与完整参数、命令、产物说明在 `final/AGENTS.md`；先读该文件。
- 该体积仅用于性能、容量与历史回归，不替代统一的 `8x8x8x16` 正确性门。
- `final/` 的 Legacy 与测试专用多卡入口是显式基准，不作为 pytest 默认收集目标。
- 性能结果必须区分 setup、solve、reference、L1 与多层 MG；保留真残差、迭代数、
  dtype、rank/grid、cache identity 和采样口径。
