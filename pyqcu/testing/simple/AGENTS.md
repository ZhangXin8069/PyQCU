# AGENTS.md — pyqcu/testing/simple

本目录是 QUDA/QCU/KUPL 的最小对照示例，不是生产测试套件。修改时遵循：

1. 单文件只负责一个功能；共享逻辑只放在 `_common.py`。
2. 默认读写目录为仓库 `data/`，输入路径必须通过 `_common.resolve_path` 解析。
3. QCU/KUPL 的 full 输入排布固定为 `[3,3,4,x,y,z,t]` /
   `[spin,color,x,y,z,t]`；QUDA/QDP
   排布及奇偶压缩必须显式转换，禁止直接 reshape 冒充转换。
4. QUDA clover 使用 `mass` 与 `csw`；QCU 使用 `mass -> kappa`，并把
   `csw` 写入 `argv[define._CSW_]` 后由后端构造 clover。两者的
   onsite 系数均为 `-kappa*csw/8`；LIME 元数据中的 `clovCoeff` 作为
   CLI 未显式指定时的默认值。
5. QCU 普通单操作事务必须保持
   `applyInitQcu -> operation -> applyEndQcu -> _SET_INDEX_ += 1`；新示例优先使用
   `QcuRuntime.call()`/`operation()`。不能在 end 前递增，否则会泄漏当前 `LatticeSet`。
6. 缺少后端依赖时输出可解释的 `SKIP`；不要生成伪造的 `.lime`、日志或收敛结果。
7. 新增 QCU 调用优先使用 `pyqcu.cuda.QcuRuntime`，由它负责 init/end、
   `_SET_INDEX_` 槽位选择和异常清理；不要在新示例中重复样板生命周期。
8. 求解性能只记录后端原生指标；迭代残差历史缺失时留空并注明来源，估算 FLOPs
   必须明确标注模型，不得冒充硬件计数器。
9. 改动后至少运行 `python -m compileall`、相关脚本的 `--help` 和
   `git diff --check`。
10. PyQUDA 求解前必须选择匹配 GPU 架构的 `libquda.so`；simple 对照统一使用
    `QUDA_KAPPA_NORMALIZATION`，与 PyQCU 的 `I-kappa D` 保持同一方程。
11. KUPL 求解走 `pyqcu.kupl.qcu` 的真实 ctypes ABI；输入先经 `_common.py`
    显式转换为 parity-compressed canonical 布局。真残差必须用独立 NumPy
    重构；MG 对照还必须校验 `verifyCloverMultigridQcu` 和同 RHS 的 QUDA 解差。
