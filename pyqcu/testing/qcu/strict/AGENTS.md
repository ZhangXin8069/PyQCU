# AGENTS.md — pyqcu/testing/qcu/strict

CUDA Strict-MG 专项测试容器。当前维护内容位于 `quda_comparison/`，先读取其
`AGENTS.md` 后再选择脚本。

- Strict 是显式对照/特殊备选，不是 QCU 默认求解路线。
- GPU、QUDA、MPI 和大矩阵编排入口都是显式运行；不要把整个目录作为日常 pytest 门禁。
- 分布式多 rank 运行必须保持同一 process grid、偶数 local shape 与 fail-closed 几何校验。
- 报告、JSON、trace 与 cache 的 provenance 必须保留，不能跨配置复用身份不匹配的资产。
