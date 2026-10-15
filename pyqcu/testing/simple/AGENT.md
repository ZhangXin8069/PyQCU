# AGENT.md — pyqcu/testing/simple

本目录是 QUDA/QCU 的最小对照示例，不是生产测试套件。修改时遵循：

1. 单文件只负责一个功能；共享逻辑只放在 `_common.py`。
2. 默认读写目录为仓库 `data/`，输入路径必须通过 `_common.resolve_path` 解析。
3. QCU 排布固定为 `[3,3,4,x,y,z,t]` / `[spin,color,x,y,z,t]`；QUDA/QDP
   排布及奇偶压缩必须显式转换，禁止直接 reshape 冒充转换。
4. QUDA clover 使用 `mass` 与 `csw`，QCU 使用 `mass -> kappa` 且由后端构造
   clover；QUDA multigrid 数组属性要整列写回，QCU 使用扁平 ABI。
5. QCU 调用必须保持 `applyInitQcu -> operation -> _SET_INDEX_ += 1 -> applyEndQcu`。
6. 缺少后端依赖时输出可解释的 `SKIP`；不要生成伪造的 `.lime`、日志或收敛结果。
7. 求解性能只记录后端原生指标；迭代残差历史缺失时留空并注明来源，估算 FLOPs
   必须明确标注模型，不得冒充硬件计数器。
8. 改动后至少运行 `python -m compileall`、相关脚本的 `--help` 和
   `git diff --check`。
