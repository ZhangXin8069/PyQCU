# AGENTS.md — pyqcu-cpp-qcu-python

cpp/cuda/qcu/python 的 C API 边界 skill：`pyqcu.h`（37 个入口）必须与
`pyqcu/cuda/qcu/qcu_api.pxd`、`qcu.pyx`/`qcu.pyi` 及 Strict 入口同步。

- 规范全文：`SKILL.md`（frontmatter description 为触发依据）
- CUDA 拓扑：ABI 绑定当前 rank/设备；单进程多设备仅限显式测试。
- 维护约定：更新内容时同步本文件与库级 `../AGENTS.md` 技能表（先读后写、最小改动、不代提交）
