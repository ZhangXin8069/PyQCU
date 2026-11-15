# AGENTS.md — pyqcu-tools

pyqcu.tools 公共 MPI/布局/I/O/env/budget 与 pyqcu.torch.tools 专用数值工具。

- 规范全文：`SKILL.md`（frontmatter description 为触发依据）
- 写路径：`save_tensor_h5`/`write_h5`/`write_json` 就地写；`save_dict_h5`/`atomic_write_json`
  走临时名 + 替换但无 `fsync`；生产输入只读，详见 `SKILL.md` §HDF5 原子写与并发。
- 设备工具：生产 GPGPU 一进程一 rank 一设备，不按线程创建多设备 worker。
- 维护约定：更新内容时同步本文件与库级 `../AGENTS.md` 技能表（先读后写、最小改动、不代提交）
