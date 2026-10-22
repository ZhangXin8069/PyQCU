# AGENTS.md — pyqcu/testing/simple

本目录包含 QUDA/QCU/KUPL 的最小对照示例与生产求解入口。修改时遵循：

## 入口选择

| 目标 | 入口 |
|---|---|
| pytest 契约测试 | `test_coarse_cache.py`、`mpi/test_mpi_contract.py`；直接传文件或目录 |
| 准备共享输入 | `prepare_case.py` |
| 单次生产冷求解 | `solve_case.py`，`--backend` 取 `cuda`、`quda` 或 `kupl` |
| 重复计时或后端对照 | `compare_*.py`、`compare_three_way.py`；不是 pytest |
| MPI 小格点验收 | `mpi/solve_case.py`；由脚本内部组织和启动 rank |
| 单后端人工示例/格式检查 | `qcu_*.py`、`kupl_*.py`、`quda_*.py`、`check_*.py` |

本目录没有需要手工取消注释的 `conftest*` 入口。带 `main()` 的脚本不是 pytest node，
不要通过 `pytest pyqcu/testing/simple` 期待执行生产求解。推荐命令：

```bash
source ./env.sh
python -m pytest -q pyqcu/testing/simple

python pyqcu/testing/simple/prepare_case.py \
  --lime /path/to/reference.lime --lat 24 24 24 64 --out-dir data/production_L24
python pyqcu/testing/simple/solve_case.py \
  --backend cuda --lat 24 24 24 64 --device cuda:0 --out-dir data/production_L24
python pyqcu/testing/simple/mpi/solve_case.py \
  --backend kupl --lime /path/to/reference.lime --lat 8 8 8 16 \
  --grid 1 4 1 1 --ranks 4 --omp-threads 8
```

需要定位具体 node 时先运行
`python -m pytest --collect-only -q pyqcu/testing/simple`。CUDA/QUDA/KUPL 依赖缺失时应
得到明确 `SKIP` 或后端错误，不得把未执行的生产脚本计为 pytest 通过。

1. 单文件只负责一个功能；共享层按职责拆分：
   `_common.py` 负责兼容门面、路径/默认值、HDF5/JSON 与求解状态；
   `_lime.py` 负责单文件 SciDAC/LIME；`_layouts.py` 负责 QCU/QDP/KUPL/Clover
   布局转换；`_runner.py` 负责子进程、GPU 架构选择与 KUPL 命令构造。
2. 生产与测量分离：`prepare_case.py` 准备输入，`solve_case.py` 做单次冷启动生产
   求解（`.partial` 仅在 PASS 后原子发布），`compare_three_way.py` 只做重复计时；
   已弃用的 `bench_three_way.py` 转发壳已删除，统一使用 `compare_three_way.py`。
3. 默认读写目录为仓库 `data/`，输入路径必须通过 `_common.resolve_path` 解析。
4. QCU/KUPL 的 full 输入排布固定为 `[3,3,4,x,y,z,t]` /
   `[spin,color,x,y,z,t]`；QUDA/QDP
   排布及奇偶压缩必须显式转换，禁止直接 reshape 冒充转换。
5. QUDA clover 使用 `mass` 与 `csw`；QCU 使用 `mass -> kappa`，并把
   `csw` 写入 `argv[define._CSW_]` 后由后端构造 clover。两者的
   onsite 系数均为 `-kappa*csw/8`；LIME 元数据中的 `clovCoeff` 作为
   CLI 未显式指定时的默认值。
6. QCU 普通单操作事务必须保持
   `applyInitQcu -> operation -> applyEndQcu -> _SET_INDEX_ += 1`；新示例优先使用
   `QcuRuntime.call()`/`operation()`。不能在 end 前递增，否则会泄漏当前 `LatticeSet`。
7. 缺少后端依赖时输出可解释的 `SKIP`；不要生成伪造的 `.lime`、日志或收敛结果。
8. 新增 QCU 调用优先使用 `pyqcu.cuda.QcuRuntime`，由它负责 init/end、
   `_SET_INDEX_` 槽位选择和异常清理；不要在新示例中重复样板生命周期。
9. 求解性能只记录后端原生指标；迭代残差历史缺失时留空并注明来源，估算 FLOPs
   必须明确标注模型，不得冒充硬件计数器。
10. 改动后至少运行 `python -m compileall`、相关脚本的 `--help` 和
   `git diff --check`。
11. PyQUDA 求解前必须选择匹配 GPU 架构的 `libquda.so`；simple 对照统一使用
    `QUDA_KAPPA_NORMALIZATION`，与 PyQCU 的 `I-kappa D` 保持同一方程。
12. KUPL 求解走 `pyqcu.kupl.qcu` 的真实 ctypes ABI；输入先经 `_common.py`
    显式转换为 parity-compressed canonical 布局。真残差必须用独立 NumPy
    重构；MG 对照还必须校验 `verifyCloverMultigridQcu` 和同 RHS 的 QUDA 解差。
13. `mpi/solve_case.py` 是小格点 MPI 快速入口：KUPL 使用四维 MPI 分域加
    OpenMP，CUDA 固定为一 Python 进程 = 一 MPI rank = 一 GPU/DCU/GPGPU；
    它接受 LIME 魔数并自动切子体积。`MultiGpuMultigrid` 仅限显式调用的测试/
    一致性/吞吐场景，不得接入生产入口。MPI 启动负责 CPU 绑定时，必须设置
    `KUPL_NUMA_BIND=0`，不得让库再次绑定。
14. 用户文档入口为 `README.md`；完整格式、状态、绑定和排错见
    `USER_GUIDE.md`，逐脚本参数见 `COMMAND_REFERENCE.md`。修改 CLI 或默认值时
    必须同步这三处，MPI 行为同时同步 `mpi/README.md` 与
    `mpi/USER_GUIDE.md`。
