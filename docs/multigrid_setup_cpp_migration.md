# MultiGrid Setup C++ Migration

日期：2026-10-27

## 目标

Legacy MultiGrid 的以下三个 setup 阶段必须由 CUDA-QCU 或 KUPL-QCU 的 C++
实现拥有，而不是由 Torch-QCU 求解路径承担：

1. near-null-vector 逆迭代；
2. 每个 coarse site 的局部正交化；
3. Galerkin 33-tensor stencil 探测与 cache 发布。

求解计时仍只属于 `applyCloverMultigridQcu`。Torch-QCU 和 KUPL-QCU 的
独立求解行不作为本迁移的实现替代。

## 当前状态

| 阶段 | CUDA-QCU | KUPL-QCU |
|---|---|---|
| 局部 CGS2 正交化 | 已完成，`applyMultigridLocalOrthogonalizeQcu` | 已完成，同名 C++/OpenMP ABI |
| near-null-vector 逆迭代 | 已完成 C++ CUDA 实现 | 已完成 C++ MPI/OpenMP 实现 |
| stencil 探测 | 已完成 C++ CUDA 33-tensor 实现 | 已完成 C++/OpenMP 33-tensor 实现 |
| cache 发布 | Python/h5py | Python/h5py |

已完成 ABI 的共同约束：

- 输入为 flat `[E,e,Xf,Yf,Zf,Tf]`；
- 输出为 blocked `[E,e,Xc,bx,Yc,by,Zc,bz,Tc,bt]`；
- 使用 two-pass modified Gram-Schmidt；
- 每个 coarse site 的 `E` 列必须满足 `Q^H Q = I`；
- 线性相关或非有限结果必须 fail closed。
- Legacy 的时间维已经 parity 压缩，生产配置允许 `bt=1`；局部窗口按
  `3*block+dim_padding` 计算，不得再次除以 2。
- 最近邻系数必须存放在目标点并使用相反位移编码；对角项按
  `[sign1,sign2,pair]` 写入。仅用粗空间维 2 的测试无法区分 `+1/-1`。

## 已验证证据

- CUDA，`24x24x24x32 -> 12x12x12x16`、E=6：27,648 blocks，
  5 次中位 `0.03358 s`，Gram 最大误差 `1.19e-6`。
- KUPL，相同几何、OMP=8：中位 `3.95 s`，Gram 最大误差
  `9.54e-7`。
- C++ ABI 符号集合检查通过：`pyqcu.h`、`qcu_api.pxd`、
  `qcu.pyx` 三者闭合。
- KUPL CTest 的非退化 `4^3x8` 粗格回归覆盖 `bt=1`、E=3；1-rank 与
  2-rank（x-only）全部通过。抽样基向量和随机向量的 Galerkin 相对误差门为
  `2e-5`，Gram 误差门为 `1e-4`。
- KUPL 集成 smoke（本地物理格 `8^3x16`、parity 格 `8^3x8`、粗格
  `4^3x8`、E=3）实测 null-vector `8.07 s`、stencil `10.36 s`、
  Galerkin 校验 `0.385 s`，Gram `7.15e-7`，Galerkin `3.28e-7`。
- CUDA 原生库与 Cython 扩展已用 DTK 23.10 重编译并导出 3 个新 C ABI；
  DCU 数值 smoke 仍在等待 `kshdnormal` 的 1 卡调度，不能把它写成已验证。

历史 `24^3x64` 日志中的 `orthogonalize done 0.208 s` 证明 601 秒超时并非停在
正交化；后续阶段是 Python local-stencil/HDF5 路径。新 C++ builder 必须分别记录
`null_vector_seconds`、`stencil_seconds`、`galerkin_validation_seconds`、
`host_transfer_seconds` 和上层 cache 发布耗时。不得只报告总 setup 时间。

## 未完成项的验收门

在迁移 remaining setup 前，不得把当前 `simple/mpi` cache builder 描述为
完整 C++ builder。后续每个阶段必须满足：

1. 不调用 `pyqcu.torch.solver` 或 Torch-QCU operator 生成 null vectors；
2. 不调用 Torch-QCU operator 构造 coarse stencil；
3. CUDA 与 KUPL 使用各自 C++ 算子，并保持相同 blocked cache schema；
4. 输出 cache 与现有已验证 cache 在同一 gauge/source identity 下通过
   Gram、Galerkin consistency、独立 full true residual 和 MG 收敛回归；
5. cold setup 单进程不超过 10 分钟；超过时终止并保留阶段日志；
6. cache-hit 的 1/2/4-rank solve 分别记录实际 `n_vcycles`、
   coarse iterations、独立真残差和 MPI geometry。

## 运行边界

在迁移完成前，允许使用 PyQCU helper 准备 cache，但必须把该阶段标记为
legacy setup，并与 C++ MG solve 分开计时。cache 构建可拆为独立作业，
避免 cache builder 与 solve 总墙钟被误算为单次 MG 求解时间。
