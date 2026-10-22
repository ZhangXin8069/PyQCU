# AGENTS.md — pyqcu-cuda

pyqcu.cuda 目录的 Cython/libqcu.so 桥接 skill：当前 Strict MultiGrid 入口、
`params=int32[58]`、`argv[8]`（槽 7 为 `csw`）、`set_ptrs=int64[100]` 和显存生命周期。

- 规范全文：`SKILL.md`（frontmatter description 为触发依据）
- 执行拓扑：一 Python 进程 = 一 MPI rank = 一 GPU/DCU/GPGPU；`MultiGpuMultigrid`
  仅限显式调用的测试专用例外。
- 公开 multigrid 层级名统一为 `StrictMultigrid` 与 `LegacyMultigrid`，生命周期为 `setup/solve/close`。
- 维护约定：更新内容时同步本文件与库级 `../AGENTS.md` 技能表（先读后写、最小改动、不代提交）
