# PyQCU examples 测试入口

示例按后端分目录。日常回归优先运行 `pyqcu/` 的纯 PyTorch 测试、`qcu/` 的
`single_qcu_*.py` 单功能 CUDA 测试，以及 `quda/` 的 `test_*.py` 单功能对照测试。

```bash
source ./env.sh
python examples/qcu/single_qcu_api.py
python examples/qcu/single_qcu_wilson_dslash.py
pytest examples/quda
```

`qcu/dev73`、`qcu/dev74`、`qcu/dev80`、`qcu/dev84`、`qcu/dev87` 和 `qcu/test15*`
是按 git tag 归档的独立项目目录，保留其历史入口，不纳入日常整理。测试产生的日志、缓存和数据由
`examples/.gitignore` 排除；新增或接口变更后必须按 `examples/AGENTS.md` 的强制整理约定复查。
