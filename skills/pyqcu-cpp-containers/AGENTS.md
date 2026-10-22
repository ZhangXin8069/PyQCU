# pyqcu-cpp-containers

`cpp/` 四个后端容器目录（`cuda` 真实现 + `cann`/`dtk`/`maca` 占位）的技能：
镜像结构、各容器职责、cpp/cuda/qcu 构建入口与三数组 ABI 速查。

2026-10-15 由 `cann.v2`/`cuda.v2`/`dtk`/`maca` 四个占位技能合并而成（原四份内嵌的
qcu/include/src/python/logs 正文副本删除，改为指向对应独立技能）。

- 规范全文：`SKILL.md`（frontmatter description 为触发依据）
- 执行拓扑：GPGPU 后端按一进程一 rank 一设备设计；MultiGpu 仅限测试。
- 维护约定：更新内容时同步本文件与库级 `../AGENTS.md` 技能表（先读后写、最小改动、不代提交）
