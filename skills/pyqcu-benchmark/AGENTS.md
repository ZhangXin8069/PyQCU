# AGENTS.md — pyqcu-benchmark

PyQCU 性能基准 skill：覆盖 pyqcu/testing/benchmark，并规定 dev87 Strict MultiGrid
对 QUDA 的公平输入、正确性、计时与显存证据口径。

- 规范全文：`SKILL.md`（frontmatter description 为触发依据）
- 跨后端公平性、性能图、完整参数和实时账本：`references/qcu-benchmark-contract.md`
- GPU 基准拓扑：一进程一 rank 一设备；MultiGpu 仅作为显式测试证据。
- 维护约定：更新内容时同步本文件与库级 `../AGENTS.md` 技能表（先读后写、最小改动、不代提交）
