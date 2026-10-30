# AGENTS.md — cpp/kupl

KUPL CPU 后端容器目录，活跃实现位于 `qcu/`。该后端是独立的 C++17
MPI+OpenMP 实现，不得包含、链接或依赖 `cpp/cuda/qcu`；也不等同于鲲鹏
HPCKit 中的 KUPL 并行库。

## 目录职责

| 路径 | 职责 |
|---|---|
| `qcu/include/kupl/` | 扁平 ABI、`LatticeSet`、几何、MPI halo、CPU 运行时与算法域声明 |
| `qcu/src/apply_*.cpp` | 40 个 CUDA 兼容 C ABI 入口的薄适配层 |
| `qcu/src/lattice_*.cpp` | Wilson/Clover、求解器、Laplacian、Gauss、multigrid 等数值实现 |
| `qcu/src/backend.cpp` | `LatticeSet<Real>` 门面、参数校验、注册表与 scratch 所有权 |
| `qcu/tests/` | C++ 正确性、失败路径、canonical layout 与 benchmark 测试 |
| `qcu/CMakeLists.txt`、`qcu/make.sh` | native 库、测试与 benchmark 的构建入口 |
| `qcu/qcu.map` | 仅导出 CUDA 兼容的 40 个 `*Qcu` 符号 |
| `qcu/libqcu.so` | 指向当前构建产物的稳定软链接，Python 与生产入口默认经此加载 |

Python ctypes 绑定位于 `pyqcu/kupl/`，跨语言和分布式测试位于
`pyqcu/testing/kupl/`。修改 C ABI 时必须在同一任务中同步 C/C++ 声明、导出表、
ctypes 包装与 ABI 检查测试。

## 强制边界

- 修改 `qcu/` 前先读取 `qcu/AGENTS.md`；触及头文件或实现时，再读取对应
  `include/AGENTS.md` 或 `src/AGENTS.md`。
- 保持扁平 ABI：`params int32[58]`、`argv real[8]`、`set_ptrs int64[100]`。
  `argv[7]` 是 `csw`，常量以 `qcu/include/kupl/define.h` 为准。
- 外部数据使用时空轴在后的 canonical C-order 布局：gauge
  `[2,3,3,4,X,Y,Z,T/2]`、fermion `[2,4,3,X,Y,Z,T/2]`、Clover
  `[4,3,4,3,X,Y,Z,T/2]`。不要用 site-major 索引直接读写公开张量。
- MPI 负责四维进程网格和 face/corner halo，rank 内只使用 OpenMP。全局点积和
  范数必须走对称的 `MPI_Allreduce`，任何 rank 都不得在集合操作前独立提前返回。
- `KUPL_NUMA_BIND=1` 时库接管 CPU 放置，MPI 必须使用 `--bind-to none`，并且不得
  同时设置 `OMP_PROC_BIND`、`OMP_PLACES`、`GOMP_CPU_AFFINITY` 或
  `KMP_AFFINITY`。调度器已提供 NUMA-local cpuset 时保持库内绑定关闭。
- `applyCloverQcu` 返回 `A=I+T`，不是裸 `T`；`csw=0` 必须精确得到 identity。
  Clover 求解器使用 parity-Schur 路径，Wilson BiStabCG 与 CG 的迭代轨迹不得混写。
- 新求解路径必须检查重构后的真残差；非收敛、无效几何、数值失败和不支持的模式必须
  显式失败，不能输出貌似有效的缓冲区。当前仅支持同精度 c64/c128。
- Clover cache 假定 gauge 在 set 生命周期内不变；原地修改 gauge 必须使用
  `KUPL_DISABLE_CLOVER_CACHE=1`，不能静默返回旧 halo 或旧逆矩阵。
- GPU 执行、LIME/HDF5 I/O 和 CUDA 专用混合精度不在 KUPL C++ 后端内实现；若需要
  扩展这些能力，先单独定义接口、资产校验和验证边界。

## 构建与验证

从 `cpp/kupl` 运行最小 native 验证：

```bash
bash qcu/make.sh --check
KUPL_ENABLE_NATIVE=OFF KUPL_ENABLE_LTO=OFF bash qcu/make.sh
ctest --test-dir qcu/build --output-on-failure
```

跨型号部署使用 `KUPL_ENABLE_NATIVE=OFF`；目标节点性能构建才启用 native/LTO。
纯 KUPL 安装从仓库根执行 `bash ./install.sh kupl`，不得调用 CUDA 构建或创建
Cython CUDA 扩展。Python 回归从仓库根执行：

```bash
python -m pytest -q pyqcu/testing/kupl
```

代码改动至少覆盖受影响精度的 CTest；涉及 ABI、布局、MPI halo、solver、multigrid
或 NUMA 时，必须同时运行对应 1/2/4-rank 与 Python 差分门。性能结论必须记录
dtype、rank/grid、threads/rank、输入规模、cache identity、迭代数、停止判据和独立
真残差，不能把 native residual、单 rank 或短样本外推为生产结论。纯文档改动至少执行
`git diff --check` 与 `bash skills/pyqcu-form/form-audit.sh`。

部署与鲲鹏运行时细节见 `../../docs/kupl_only_deployment.md` 和
`../../docs/kupl_kunpeng_guide.md`。
