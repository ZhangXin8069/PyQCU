# AGENTS.md — pyqcu-testing

pyqcu.testing 集成测试 skill：全组件 `test_*` 入口，以及 Strict MultiGrid
tier 0/1/2 快速闸门、MPI/缓存/QIO/显存证据协议。

- 规范全文：`SKILL.md`（frontmatter description 为触发依据）
- 发布验证：`SKILL.md`「发布验证（release gate）」；`pyqcu/testing/**` 不进 sdist，安装态验证用 `pyqcu.test` 或公开 API。
- 正式端口选项：`--lattice-size`/`--lattice-grid`/`--mg-grid`、`--gauge`/`--source`、
  `--mass`/`--kappa`/`--csw`、`--solver`/`--dslash`、`--device`、`--backend` 别名，
  以及 full-mode `--mode list/prepare/prewarm/solve/merge/run/validate`
  （`rows/reports/solutions/caches`、`--resume`/`--skip-failed`/`--run-label`、
  `--gauge-file`/`--source-file`）见 `pyqcu/test/README.md`；`--threads` 只在 torch
  runner 应用，KUPL/CUDA runner 仅记录。
- GPGPU 测试拓扑：一进程一 rank 一设备；MultiGpu 测试必须显式标注。
- 维护约定：更新内容时同步本文件与库级 `../AGENTS.md` 技能表（先读后写、最小改动、不代提交）
