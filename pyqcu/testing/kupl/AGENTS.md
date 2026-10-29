# pyqcu/testing/kupl

KUPL CPU 后端的 Python 黑盒回归。

- `test_kupl_python.py` 经真实 `ctypes` 库校验 Clover、dslash、
  BiStabCG、Gaussian gauge、槽位生命周期和 canonical PyQCU ABI。
- `test_krylov_api.py` 经真实 KUPL 库验证统一 `Krylov` facade 的 Wilson
  BiCGStab/normal-CG 整次 solve 与独立 full residual。
- `test_strict.py` 覆盖 strict coarse/MATPC/R-P、持久 V-cycle/FGMRES；
  `test_legacy_mg.py` 覆盖递归 Clover-MG、deflate 开关、CUDA-only 模式拒绝与
  verify；`check_abi.py` 同时比对 KUPL/CUDA C 声明和动态导出。
- `kunpeng_runtime.py` 读取实际 affinity/sysfs NUMA 拓扑并生成 HMPI/OpenMPI、
  CPU/内存绑定计划；`test_kunpeng_runtime.py` 用合成双 NUMA 数据验证参数契约。
- `kupl_env.py` 在无 MPI 初始化条件下汇总 CPU/NUMA/cgroup、HBM/片上内存候选、
  KUPL 动态库、当前冲突变量和生产 A/B 命令；默认原子写入
  `logs/kupl_env-vYYYY-MM-DD-HH-MM-SS.txt`。只有 memory-only NUMA 的
  `distance` 对全部 CPU 节点形成唯一且一致的 `local+N` offset 时才建议
  `KUPL_MEMORY_NODE=local+N`，否则回退 `local` 并明确“未证明使用 HBM”。
- `test_kupl_env.py` 覆盖无 HBM、稳定 offset、歧义映射、无 sysfs fail-safe、
  文件名冲突、原子写和 CLI 不导入 `mpi4py` 的契约。
- `test_pure_kupl_install.py` 验证纯 KUPL 构建不触发 CUDA/Cython、旧 PyTorch
  dtype 映射不依赖 `to_real()`，且拓扑 CLI 不初始化 MPI。
- `test_kunpeng_mpi_binding.py` 启动双 rank，验证 `KUPL_NUMA_BIND=1` 后每个 rank
  的 cpuset 非空且互不重叠；无 Linux/sysfs 或 CPU 不足时返回 CTest skip 77。
- `test_kunpeng_profile_contract.py` 验证非鲲鹏主机拒绝鲲鹏 profile，并保证
  rank 间 profile 不一致时快速失败而不是在集合点挂起。
- `benchmark_runtime.py` 先 `applyInitQcu`、再分配/first-touch 输入，分离记录
  init、输入准备和首次 Clover 的耗时；RSS 必须保留 `before_init`、
  `after_init`、`after_inputs`、`after_first_clover` 四阶段，并单列
  `lifetime_hwm_bytes`。不得把进程终身 HWM 当作阶段峰值。
- `simple/kupl_multigrid.py` 与 `simple/benchmark_kupl_lime_mpi.py` 在
  `KUPL_NUMA_BIND=1` 时须先执行私有 prebind lifecycle 再读输入；其
  `input_prebind_seconds` 不得混入 solver steady 时间。
- 缺少 `cpp/kupl/qcu/build/libqcu.so` 时测试明确 skip，不伪造通过。
- `cpp/kupl/qcu/make.sh` 是 native-only 构建入口；`make.sh --check` 验证变量和路径
  选择但不写文件。Python ctypes 安装仍使用根目录 `install.sh kupl`。
- C++ 1/2/4-rank 与 sanitizer 回归属于 `cpp/kupl/qcu/tests/`。
