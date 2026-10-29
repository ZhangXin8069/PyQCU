# AGENTS.md - pyqcu/testing/simple/mpi

本目录提供 `simple/` 的小格点 MPI 快速验收入口。修改时遵循：

1. 公共入口是 `solve_case.py`。它负责接受 LIME/HDF5、切出小体积、生成共享
   source、构建 33-tensor 粗算子和启动一次 `mpirun`。`worker.py` 是 MPI
   子进程内部入口，不作为日常用户接口。
2. KUPL 是真实四维 MPI 分域加 rank 内 OpenMP。MPI 负责绑定时必须设置
   `KUPL_NUMA_BIND=0`，并清除 `OMP_PROC_BIND`/`OMP_PLACES`，避免库内和
   OpenMPI 重复绑核。无调度器绑定时可显式选择 `--kupl-numa-bind library`，
   此时强制 MPI `--bind-to none`，由库按 sysfs NUMA cpulist 绑定；内存策略、
   片上内存节点和 profile 必须写入 dry-run/report 证据。
3. CUDA 生产路径固定为一 Python 进程 = 一 MPI rank = 一 GPU/DCU/GPGPU，
   并做 MPI 格点分域。不要把它与测试专用的
   `pyqcu.cuda._multi_gpu.MultiGpuMultigrid`“单进程、每卡复制整题求解”
   语义混写。后者只能在显式调用的测试/一致性/吞吐场景启用；文档和报告必须明确区分。
4. LIME 输入按魔数识别，不依赖扩展名大小写；必须继续支持
   `scidac-private-file-xml`/`scidac-file-xml`、
   `scidac-private-record-xml`/`scidac-record-xml`、`clovCoeff` 和 `Mass`。
5. 快速验收默认使用 `8x8x8x16` 子体积。LIME 子体积的边界缝不满足原始周期
   条件，只用于代码路径、MPI 通信和数值真残差验证，不作为物理结果。
6. 多 rank MG 必须使用全局粗算子缓存。`build_full_mg_cache.py` 由所有 rank
   协同构建：每 rank 生成 local null-vector、计算自己的 coarse slab，并汇总为
   一个原子发布的全局 HDF5；worker 只读取本 rank 粗格超切片，不得重复构建完整
   hierarchy。overlap patch 只能用于边界 halo，不能退化为每 rank 重建全题。
7. 每个新路径至少验证 `python -m compileall`、契约 pytest、一个 KUPL MPI
   小格点求解和一个 CUDA 小格点求解；设备缺失时记录未验证原因。
8. 用户文档入口为 `README.md`，完整参数、OpenMPI 绑定、报告字段和排错见
   `USER_GUIDE.md`。CLI、默认值、环境变量或报告 schema 变化时必须同步更新。
