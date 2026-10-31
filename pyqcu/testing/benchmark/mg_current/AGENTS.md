# AGENTS.md — pyqcu/testing/benchmark/mg_current

`data/mg-bench-current-20260830/` 的汇总与 GPU 监控维护脚本。

- `generate_summary.py` 读取既有 stdout/JSON/TSV，生成 `summary.csv`；不要改写原始证据。
- `run_with_gpu_monitor.sh` 的接口是 `OUTPUT_DIR COMMAND [ARG ...]`，依赖
  `nvidia-smi`，会生成 baseline、逐时采样、stdout/stderr 与 `memory.tsv`。
- 不得把缺失采样、NaN 或未达到 `atol` 的运行标成通过；保留来源路径和计时口径。
- 修改后至少运行 `bash -n run_with_gpu_monitor.sh`，并用已有输入复核
  `python generate_summary.py`；无 GPU 时不做长基准。
