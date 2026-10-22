# AGENTS.md — pyqcu-qcu-single-tests

规范全文：`SKILL.md`；frontmatter description 只负责触发，执行细节以正文为准。
只修改单功能测试、其公共布局辅助和本技能文档；不把生成日志、HDF5、QIO 或 GPU 缓存写入版本库。
若 C API 发生变化，先核对 `pyqcu.h`、`qcu_api.pxd`、`qcu.pyx` 三者符号集合，再更新测试清单。
