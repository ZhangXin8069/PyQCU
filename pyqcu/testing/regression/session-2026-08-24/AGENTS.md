# AGENTS.md — pyqcu/testing/regression/session-2026-08-24

bug31–37 时期跨组件无人值守回归快照，保留历史脚本、输入假设和结果复现路径。

- `r1_baseline.py` 与 `c2_*`–`c14_*` 是顺序/专项历史入口，不是稳定 pytest 套件。
- 部分脚本直接使用 `torch`、较大格点或旧参数协议；不得将其复制为新代码模板。
- 当前正确性开发应以现行 backends、`pyqcu.cann` 兼容层和 `8x8x8x16` 统一门为准。
- 仅在明确复现该会话时运行；执行前核对 CUDA/KUPL/QUDA 依赖、数据和预期失败，
  结果写入当前任务的 `logs/**`，不要改写历史脚本冒充新验证。
