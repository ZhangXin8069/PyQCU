# AGENTS.md — .../benchmarks/large_volume

CUDA C++ MultiGrid 的 16×32×32×48 大体积研究套件，属于历史性能与容量证据，
不是当前日常正确性门。

- `main.py` 的子命令和参数以本目录 `README.md` 为准；运行前先确认 `data/` 中
  对应的 gauge/null-vector/cache 实体与内存预算。
- 保留 `MG_L1`、BiStabCG、setup 与 solve-only 的原始口径；不得把多线程多卡
  `MultiGpuMultigrid` 解释为单次 solve 的 MPI 分域。
- README/report 已记录连续谱下粗空间收益有限与大体积假设证伪；新增结果必须有独立
  真残差、重复采样和 cache provenance，不能仅凭短样本宣称加速。
- 纯文档或解析逻辑改动至少运行 `git diff --check`；可快速执行的相关单测优先于 GPU 长跑。
