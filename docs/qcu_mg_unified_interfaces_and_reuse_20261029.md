# QCU MultiGrid 统一接口、缓存复用与大格验证（2026-10-29 工作区）

状态：`userinterlude`，未打 tag。本文记录当前工作树的已验证结果、已采用规则和
未完成项，不构成 tag 后的永久性能承诺。

## 1. 目标与证据边界

本阶段目标是把 CUDA-QCU、KUPL-QCU、Torch-QCU 的同一功能收敛到统一 PyQCU
接口，并让同一版本的 Legacy/Strict MG cache 可跨后端复用；性能验收同时包含
MultiGrid 与 native BiCGStab，以 QUDA 为对照。

`24^3x64` 仅用于参数筛选和 smoke，不承担最终性能结论。正式大格从
`32^3x64` 开始，容量允许时再加入 `48^3x96`。不同 LIME 文件按各自元数据记录
mass/CSW，不把相同文件名模式误当成相同物理参数。

## 2. 统一接口

高层入口固定如下：

| 功能 | 入口 | 统一结果/控制 |
|---|---|---|
| BiCGStab/CG | `pyqcu.solver.Solver`，兼容名 `Krylov` | `BiCGStabControl`、`CGControl`、`SolveResult` |
| Legacy/Strict MG | `pyqcu.solver.Multigrid` | `SolveResult`、`BackendCapabilities` |
| Legacy/Strict cache | `pyqcu.solver.MultigridCache` | `MultigridCacheLoadResult` |
| Operator | `pyqcu.api.LinearOperator` | `LatticeGeometry`、`ResidualScope` |

`bistabcg` 是 `bicgstab` 的兼容输入别名，公共结果统一返回
`method="bicgstab"`。后端私有 `params/argv/set_ptrs`、ctypes 指针和 cache
槽位只能存在于适配层。统一入口不改变“一 Python 进程 = 一 MPI rank = 一设备”
的 CUDA 生产拓扑。

## 3. Cache 规范

### 3.1 Legacy

Legacy 使用唯一 canonical odd-Schur 33-tensor schema：

- `lonv=[E,12,Xc,bx,Yc,by,Zc,bz,Tc,bt]`
- `hnn=[2,4,E,E,Xc,Yc,Zc,Tc]`
- `hdg=[2,2,6,E,E,Xc,Yc,Zc,Tc]`
- `sit=[E,E,Xc,Yc,Zc,Tc]`

CUDA/Torch 原本使用该布局；KUPL 通过 `set_ptrs[99]=1` 选择
`kLegacyLayoutOddSchur` 后读取同一布局。共享模块
`pyqcu/tools/_legacy_mg_cache.py` 统一 shape、rank slab、manifest 和内容摘要，
后端包不再各自实现一套 HDF5 轴切片。

复用 identity 至少包含 gauge canonical/file SHA256、lattice、dtype、mass、
csw、levels、dof、mg_grid、nv_iters、seed 和 setup schema。只有 identity、
shape、dtype、manifest 与 tensor digest 全部通过时才允许命中；旧 flat JSON
报告保留只读兼容，但不得绕过物理 identity。

### 3.2 Strict

Strict 保留 `pyqcu.strict-runtime-cache` v2 文件协议：

- `assets/fine_blocked_v`
- `assets/levels/<n>/preconditioned_links`（`Yhat`）
- `assets/levels/<n>/onsite_pair`（`X/X^-1`）
- transition > 0 的递归 `null_vectors`

共享 facade `pyqcu/tools/_strict_mg_cache.py` 将其映射为
`fine_null_vectors + levels`，供 CUDA/Torch/KUPL 绑定。Legacy 33-tensor 与
Strict full-coarse `X/Y/Yhat` 仍不可互载。

## 4. 资源复用规则

同一 QCU 版本内已有可验证资源时直接复用，不重复 setup 或全量构建：

- Legacy/Strict cache 命中后跳过 null-space、局部正交化和 Galerkin stencil 构建；
- QUDA/QCU 构建优先增量，不清理可复用对象；
- QUDA 重新构建失败时回退 `$HOME/lib/quda-develop/build/lib/libquda.so`，但必须重新记录
  SHA256、架构、精度、QDP 和 MG 能力；
- cache 命中仍执行 manifest/身份/tensor SHA 校验，并分别记录验证、加载、setup 和
  steady solve 时间；
- 只有 miss、损坏、identity 不匹配或显式 cold/miss 请求才重建。

### 4.1 多 MPI Legacy cache builder

规范入口为 `python -m pyqcu.testing.simple.mpi.build_full_mg_cache`。每个 rank
生成自己的 null-vector slab；所有 rank 通过 HDF5 MPI-IO 写入 blocked `lonv`
owned slab，再按周期 coarse 坐标直接读取自己的 overlap patch。每个 rank 只计算
自己的 `lonv/hnn/hdg/sit` 并写回 owned slab；rank 0 只完成属性和 manifest，
所有 rank 回读 HDF5 hyperslab 验签。`lonv` 的 coarse 轴是 `(2,4,6,8)`，不能按普通
lattice-axes-last-4 切片；其时间步长是 `2*mg_grid[t]`。不得恢复 root
`gather/scatter` 多 GiB `lonv` 的旧路径。

CUDA `--build-device cuda` 的多 rank patch stencil 已恢复为 GPU：当 patch 的
`params[NODE_SIZE]=1` 且 grid product 为 1 时，CUDA `LatticeSet` 创建并持有私有
singleton communicator，Clover/coarse halo 改用 `set_ptr->comm`，不再与
`MPI_COMM_WORLD` 的 Python collective 串序。修复后 `8x8x8x16` 4-rank、
grid `2x2x1x1` builder 的 lonv MPI-IO 为 `0.354144 s`、GPU patch stencil 为
`0.137035 s`、其余三张量 MPI-IO 为 `0.884016 s`、总 setup `3.040447 s`，
4/4 rank slab 验签通过并记录
`stencil_device=cuda:0`。同 cache 由 KUPL 直接加载后 `verify_status=0`，独立真残差
`4.110297e-7`。

### 4.2 跨后端直接复用验证

同一 gauge 的 CUDA E2 cache 已直接交由 KUPL 的 odd-Schur kernel 使用。修复 CUDA
raw lifecycle 未同步 odd Clover parity 的 bug 后，KUPL-vs-CUDA 的
`hnn/hdg/sit` 相对误差分别约 `2.48e-7/7.34e-8/2.52e-7`。CUDA builder 生成的
1/2/4-rank cache 均由对应 rank 数的 KUPL solve 加载，`verify_status=0`，独立
true residual 约 `2.016e-7/2.018e-7/2.020e-7`。

## 5. 当前实测配方

硬件：NVIDIA A100-PCIE-40GB，单节点；PyQCU 与 QUDA 均使用 `sm_80` 构建。

### 5.1 L24 smoke（不得作为最终性能）

- Lattice：`24x24x24x64`，c64
- Gauge：`beta6.20_mu-0.2770_ms-0.2310_L24x64_cfg_9950.lime`
- Gauge SHA256：`a49773fa3742d310f4ea162a932cf559253aeab708bac08a8dd1f6847df4294b`
- `mass=-0.277`，`kappa=0.13430029546065003`，`csw=1.160920226`
- CUDA MG：`applyCloverMultigridQcu`，2 levels，`dof=[12,12]`，
  `mg_grid=[2,2,2,2]`，`restart=20`，`coarse_max_iter=200`，
  `coarse_tol_factor=1e4`，`atol=1e-6`，`max_iter=1000`

L24 的 1/2/4-rank MG 均通过独立真残差门，残差分别约
`2.8791e-7`、`2.8837e-7`、`2.8844e-7`。40 个交错 A/B 样本显示候选相对基线
在 2-rank 仅约 `+0.54%`，4-rank 约 `-11.0%`，因此当前 fused multi-rank
候选不能保留为性能收益。L24 native BiCGStab 为 18 iterations，
`0.05145 s`，独立真残差 `6.6946e-7`；这是 smoke，不是大格结论。

### 5.2 L32 正式入口

- Lattice：`32x32x32x64`，c64
- Gauge：`beta6.20_mu-0.2770_ms-0.2400_L32x64_cfg_11000.lime`
- Gauge SHA256：`daa59a53542eade112ee5d5ebce7e1166cdcd57525502a29cef1d7deb04fd4c6`
- `mass=-0.277`，`kappa=0.13430029546065003`，`csw=1.160920226`
- RHS：PCG64 float64 正态流先采样再 cast c64，seed 42，
  全局 SHA256 `8c00a7341b7abb2b3d0e26e477731c3528c86c337888cfe1e539b0e47673ddb6`

QUDA native BiCGStab 已通过：1 rank，`1x1x1x1`，17 iterations，
steady wall `0.130358 s`，true residual `8.89032e-7`，1 warmup + 5 repeats。
该结果使用 `QUDA_BICGSTAB_INVERTER`、Direct-PC、Kappa normalization，
`tol=1e-6`、`max_iter=1000`。

CUDA L32 BiCGStab 的数值阶段在 17 iterations 完成；首次独立残差阶段因大数组
`comm.gather` 使用 pickle 路径触发 `MPI_ERR_ARG`。现改为显式 dtype/count/displacement
的 `MPI_Gatherv`，单 rank 直接返回；修复后回归必须重新发布完整 PASS 报告。

### 5.3 L32 Strict-MG 容量与多 MPI

现有 L32 Legacy E12 cache 的 `lonv` 已通过 `prepare_fair_nullvec.py` 重构为
canonical full basis `[12,4,3,32,32,32,64]`，没有重新生成 null-space。真实 L32、
c64、mass `-0.277`、csw `1.160920226`、两层、`nvec=12`、`coarse_spin=2`、
`restart=16` 的 `applyMultigridStrictFgmresQcu` 结果：

| ranks/grid | status | cold setup | steady sync max | FGMRES | independent residual |
|---:|---|---:|---:|---:|---:|
| `1/1^4` | OOM | n/a | n/a | n/a | n/a |
| `2/2x1x1x1` | PASS | `201.073 s` | `0.476122 s` | 11 | `6.253315e-7` |
| `4/2x2x1x1` | PASS | `111.873 s` | `0.356964 s` | 11 | `6.320174e-7` |

1-rank 在 `StrictTransfer._block_orthogonalise` 把 39.39 GiB A100 用到 39.37 GiB 后
无法再分配 96 MiB，说明 L32 full-coarse Strict 不能按小格点单卡容量外推。多 MPI
不只是并行优化，也是当前 L32 的必要容量路径。4-rank 相对 2-rank 加速 `1.334x`，
但 Strict 仍比同窗 Legacy/QUDA 慢；生产默认保持 Legacy。`restart=16` 的 fused
FGMRES workspace 实需 `1,887,436,800 B`，L32 runner 使用 2 GiB 上限并显式
fail-closed。native residual 字段为 absolute/aggregate 口径，验收只使用独立 full
operator relative residual；trace 的顶层 `restriction_calls=0` 也不能覆盖 level-1
`coarse_solver_calls=11`、`coarse_solver_iterations=25/26` 的粗层证据。

### 5.4 L48 容量门与 builder 边界

真实 L48 输入为完整 `48x48x48x96` LIME，SHA256
`e3b27a2ee9ee3b78b6512549ca84340f8f65c60122d84356ce5693a0324b6acb`；
已生成 c64 gauge/source 和 manifest。早期 `1603917` 曾进入 CPU overlap-stencil，
5 分钟时 4 张 A100 均 0%，已主动取消。private singleton communicator 修复后的
`1604062` 在整个 5/8 分钟窗口保持四卡 100%、每卡约 27 GiB；GPU stencil 完成后，
root 汇总 `hnn/hdg/sit` 在约 `530 s` 触发 `54,976 MiB` cgroup OOM，此时 max RSS
为 `35,000,128 KiB`。因此当前根因已从 GPU/communicator 转为主机全局粗算子峰值。

现行 builder 已把四个 tensor 全部改为 HDF5 MPI-IO owned-slab，并支持复用失败
作业留下的 `lonv` scratch。提前释放 staging patch 后，`1604161` 仍在 `8m32s`
因 54,976 MiB cgroup 内存上限 OOM；L48 目前没有已发布 cache 或 solve 数据；
这些 OOM 不得写为算法失败，也不能据此给出 L48 性能。新 Slurm 批次仍必须在提交前
创建 `slurm/` 和 `logs/`，否则会在 0 秒 `JobLaunchFailure`。

## 6. 求解器与参数记录

正式求解矩阵分别比较：

1. CUDA `applyCloverMultigridQcu` 对 QUDA recursive MG；
2. CUDA `applyCloverBistabCgQcu` 对 QUDA `QUDA_BICGSTAB_INVERTER`；
3. MG/BiCGStab 比只作为各后端内部上下文，不能替代同算法胜负。

每条结果必须记录 lattice、ranks/grid、设备 UUID、backend、Legacy/Strict、
精度、mass、kappa、csw、公共入口和 native solver、levels、block/dof/nvec、
restart、coarse budget/tolerance、atol、max_iter、warmup/repeat、实际迭代/cycle、
残差口径与独立真残差、cache identity、gauge/RHS SHA256、库 SHA256。

## 7. QUDA 构建经验

- `/tmp` 容量不足会使 `nvcc` 在无关源码编译阶段失败；使用仓库内 `TMPDIR` 并增量续建。
- PyQUDA `copyGauge` 需要 `QUDA_INTERFACE_QDP=ON`。
- QUDA MG `BatchInvertMatrix` 需要 `QUDA_BUILD_NATIVE_LAPACK=ON`；仅 BiCGStab smoke
  通过不能证明 MG 可用。
- torch CUDA UUID 是 `torch._C._CUuuid`，不能直接 MPI pickle；报告前必须转为字符串。
- QDP BiCGStab smoke 和 MG smoke 是不同能力门，不得互相替代。

## 8. 作图与报告

正式批次至少输出：总 solve wall、相对 QUDA 加速比、1/2/4-rank 并行效率、
迭代数、独立真残差、setup/cache 与 steady solve 分解。同一后端保持稳定颜色，
一个轴只使用一种单位，显示样本数/误差定义、`1.0x` 参考线和残差阈值。
优先 SVG，同时保存 PNG；原始 CSV/JSON、生成命令和 SHA256 必须随图归档。

L24/smoke 图必须与 L32/L48 正式图分开；不能把不同 lattice、rank、cache 状态或
trace 模式连成同一曲线。

## 9. 未完成项

- L32 Legacy E12 cache 与 1/2/4-rank MG/BiCG 已完成；同窗 QUDA GCR-MG 1/2/4-rank
  也已完成，CUDA Legacy 入口级速度比为 `2.2495x/1.4323x/1.3166x`，证据见
  `logs/qcu_sync_resume_20261029/quda-mg/summary.json`。
- L48x96 容量与 10 分钟硬门评估；只在 L32 多 MPI MG 全门通过后启动，并优先复用
  L32 已验证物理资产，避免重复 setup。
- CUDA Strict-MG 已完成 L32 的 2/4-rank solve、迭代、独立残差和容量记录；1-rank
  因 40 GiB 设备容量 OOM。当前数据不支持把 Strict 改成生产默认。
- KUPL 真实鲲鹏 HMPI/NUMA/片上内存验证；本 x86 参考机不能替代。
- Torch 只保留小格 CPU 正确性，不再纳入 GPU 性能选择。
- 全部正式图、CSV/JSON、阶段日志、定向回归、form/docs 审计和 git tag 交付。
