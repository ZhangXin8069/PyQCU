# AGENTS.md — pyqcu-testing

pyqcu.testing 集成测试 skill 的导航：非正式套件、正式 `pyqcu.test` 快速端口、Strict MG tier 0/1/2 闸门、MPI/缓存/QIO/显存证据协议。

- 规范全文与触发条件：`SKILL.md`（含「发布验证（release gate）」）。
- 按需参考：`references/l48-remote.md`、`references/remote-l48-experience.md`、
  `references/strict-mg-fast-gates.md`。
- 正式端口选项与 full-mode `--mode` 见 `pyqcu/test/README.md`；`--threads` 只在
  torch runner 应用，KUPL/CUDA runner 仅记录。
- GPGPU 拓扑：一进程一 rank 一设备；MultiGpu 测试必须显式标注。
- 维护：更新内容时同步本文件与库级 `../AGENTS.md` 技能表。
