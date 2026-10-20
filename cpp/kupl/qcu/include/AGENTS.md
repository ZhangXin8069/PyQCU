# AGENTS.md — cpp/kupl/qcu/include

KUPL QCU 的声明层，目录形式与 `cpp/cuda/qcu/include` 对齐。

- `kupl/define.h`：`params/argv/set_ptrs` ABI 常量与计划编码。
- `kupl/qcu.h`：37 个 `extern "C"` QCU 入口，函数名和签名必须与 CUDA 版一致。
- `kupl/lattice_set.h`：`LatticeSetBase`、`LatticeSet<Real>` 和注册生命周期；
  只声明接口，数值实现留在 `src/lattice_*.cpp`。
- `kupl/lattice_complex.h`：复数别名、自旋/颜色/向量维度常量。
- `kupl/lattice_geometry.h`：参数、四维 MPI 几何、canonical 视图与 halo 类型。
- `kupl/lattice_cpu.h`：MPI/OpenMP/NUMA 运行时和 OpenMP chunk 计算。
- `kupl/lattice_halo.h`：face/corner MPI tag 与偏移。
- `kupl/lattice_linalg.h`：gamma、色空间乘法和 onsite 线性代数内联模板。
- `kupl/lattice_clover_multigrid.h`：递归 Clover-MG 求解与 verify 域接口。
- `kupl/lattice_multigrid.h`：32 方向粗网格 stencil 辅助声明。
- `kupl/lattice_strict.h`：strict coarse/MATPC/R-P、层次与 FGMRES 域接口。

头文件名使用小写下划线；类型使用大驼峰；内部函数使用小写下划线。不要把
CUDA stream、device 指针或 GPU kernel 声明复制到 KUPL。
