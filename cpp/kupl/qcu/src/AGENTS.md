# AGENTS.md — cpp/kupl/qcu/src

KUPL QCU 实现层，按 CUDA QCU 的 ABI 与算法域对应拆分。

## ABI 层

- `apply_*.cpp`：只做指针解包、状态查找、异常边界和 `LatticeSetBase` 分派。
- `backend.cpp`：`LatticeSet<Real>` 构造、参数校验、公开 override 和 scratch 分配。
- `api_common.hpp`：C ABI 共用 helper，不承载数值算法。

## 数值域

| 文件 | 职责 |
|---|---|
| `lattice_cpu.cpp` | MPI 初始化、OpenMP/NUMA、注册表和精度工厂 |
| `lattice_geometry.cpp` | 四维/三维几何、canonical 索引与视图 |
| `lattice_linalg.cpp` | gauge matrix 加载和 12×12 onsite 乘法 |
| `lattice_clover_dslash.cpp` | gauge halo、plaquette、Clover `A/A^-1`、缓存 |
| `lattice_clover_multigrid.cpp` | full-site 递归 Clover-MG、33-stencil V-cycle、verify |
| `lattice_wilson_dslash.cpp` | fermion halo、Wilson/Dslash-dagger、Schur 算子 |
| `lattice_solver.cpp` | 全局归约、BiStabCG/CG、Wilson/Clover 求解 |
| `lattice_laplacian.cpp` | 3D Laplacian MPI halo 与局部作用 |
| `lattice_gauss_gauge.cpp` | 确定性 Gaussian SU(3) 生成 |
| `lattice_multigrid.cpp` | restrict/prolong 与窄/宽 33-stencil coarse dslash |
| `lattice_strict.cpp` | strict coarse/MATPC、R/P、持久 hierarchy、V-cycle/FGMRES |

内部函数和变量使用小写下划线；`LatticeSetBase` 的 override 保留现有接口命名。
每个模板实现文件显式实例化 `float` 和 `double`。禁止跨域复制 MPI 逻辑或改变
`post -> local/interior -> Waitall -> boundary` 顺序。
