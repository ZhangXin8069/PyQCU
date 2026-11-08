# AGENTS.md — pyqcu-testing

pyqcu.testing 集成测试 skill：全组件 `test_*` 入口，以及 Strict MultiGrid
tier 0/1/2 快速闸门、MPI/缓存/QIO/显存证据协议。

- 规范全文：`SKILL.md`（frontmatter description 为触发依据）
- 发布验证：`SKILL.md`「发布验证（release gate）」（release_check / docs_examples、8x8x16x16
  验收线、干净 venv 复验）；`pyqcu/testing/**` 不进 sdist，安装态验证只能在源码树外做。
- GPGPU 测试拓扑：一进程一 rank 一设备；MultiGpu 测试必须显式标注。
- 维护约定：更新内容时同步本文件与库级 `../AGENTS.md` 技能表（先读后写、最小改动、不代提交）
