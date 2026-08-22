# AGENTS.md — pyqcu.solver

Dirac 方程 D ψ = η 的迭代求解器。

## 文件

| 文件 | 用途 |
|---|---|
| `_bistabcg.py` | BiCGStab(l) 求解器 + `bistabcg_history`（零初始解逐迭代残差历史复现） |
| `_multigrid.py` | 自适应多重网格 (AMG) V-cycle 求解器，最细层 CUDA 加速 |
| `_gmres.py` | `fgmres` — FGMRES(m) 右预条件求解器（flexible GMRES，移植自 DDalphaAMG-SM fgmres.cpp） |
| `_mr.py` | `mr` — MR 最小残差求解器（quda inv_mr 思想，非 Krylov 平滑器；正规方程方向 p=A†r + 步长 ω 阻尼） |

## 导出 API

### `bistabcg(b, matvec, tol=1e-6, max_iter=1000, x0=None, if_rtol=False, verbose=True, history=None)`

标准 BiCGStab。`matvec` 为可调用对象 `matvec(src) → dest`。`history` 传入 list 时逐迭代追加
`float(||r||)`（含初始 ||r0||，共 max_iter+1 条以内）——供收敛曲线绘制与 h5 持久化。

**Breakdown 检测（R2 增加）**：`rho ≈ 0`、`vdot(r_tilde, v) ≈ 0`（pivot breakdown）、`vdot(t, t) ≈ 0` 时抛 `RuntimeError`。

**容差**：`if_rtol=True` 用 `tol * ||b||`；否则绝对 `tol`。

### `bistabcg_history(b, matvec, tol=1e-6, max_iter=2000, if_rtol=False) -> List[float]`

零初始解复现 BiCGStab 逐迭代残差历史 `[||r0||, ||r1||, ...]`（整合自 logs/dev78_2 与
examples/qcu/dev73 的同名函数）。用途：C++ 求解路径只输出收敛点，用同一 matvec 在
torch 上数学等价复现参考收敛曲线（零 C++ 改动）。breakdown 时返回已收集部分并打印
提示（不抛异常，画图友好）。

### `fgmres(b, matvec, tol=1e-6, max_iter=1000, restart=30, x0=None, precond=None, if_rtol=False, verbose=True, history=None)`

FGMRES(m)：右预条件广义极小残量（flexible GMRES），`precond` 可逐迭代变化（SAP/MG
smoother 等非常数预条件）；`precond=None` 退化为标准 GMRES(m)。算法：Arnoldi MGS 正交化 +
复 Givens 旋转 + 上三角回代 + 重启（DDalphaAMG-SM 移植）。`history` 语义：内迭代追加残差
估计 `|g[j+1]|`，每重启周期末追加真实残差 `||b-Ax||`。Krylov 基存储为展平向量，
matvec/precond 调用前自动 reshape 回输入形状（任意布局兼容）。

实测（RTX 4060 / CPU，4x4x4x8 Wilson D，kappa=0.125，rtol 1e-6）：FGMRES 与 BiStabCG 解
一致（相对差 ~8e-6），CPU/CUDA 残差一致 9.4e-7。

### `mr(b, matvec, tol=1e-6, max_iter=1000, x0=None, matvec_dag=None, omega=1.0, if_rtol=False, verbose=True, history=None)`

MR 最小残差迭代（quda inv_mr 思想）：`p=A†r; α=ω·<p,p>/<Ap,Ap>; x+=αp; r-=αAp`。
每迭代一次 A + 一次 A†，内存 O(1) 向量数，作 MG smoother 或粗略预求解。
**`matvec_dag=None` 默认取 A 自共轭——仅当算子 Hermitian 时正确**（如奇偶 Schur 补；
Wilson 全算子是 γ₅-Hermitian 非厄米，必须显式传 `matvec_dag = γ₅·A·γ₅`
（spin 维 diag(1,1,-1,-1)），否则步长错误可发散）。
c128 实测严格单调收敛（10274 迭代 rel 2.5e-10）；c64 有精度地板（~1e-4 相对），
平滑器级精度预期。

### `multigrid` 类

自适应多重网格 V-cycle，可配置多层级。

**构造参数**：`dtype_list`/`device_list`（每层）、`U`/`clover_term`/`kappa`/`u_0`、`clover_ee_inv`+`clover_oo_inv`（二者都给出 → `with_cuda_qcu=True` 最细层 C++ 后端）、`min_size=4`、`max_level=4`、`mg_grid_size=[2,2,2,2]`、`dof_list=[12,24,24,...]`、`tol`/`max_iter`/`num_restart=5`、`num_convergence_sample=50`、`support_parity=False`

**关键方法**：`init()`（逆迭代建 null-space 向量、local-orthogonalize、Galerkin 粗网格算子）、`solve(b, x0)`、`cycle(level)`（递归 V-cycle）、`adaptive(iter)`（收敛停滞 ≥3 次则降到最粗层）、`levels_back()`、`plot(save_path)`（仅 root rank）

**执行层**：
- 层 0（最细）：`with_cuda_qcu=True` 时 C++ CUDA 后端做 BiStabCG 平滑
- 层 1：C++ CUDA 后端粗网格 dslash（`_coarse_dslash_cuda()`）
- 层 2+：纯 Python einsum

**C++ 后端集成（层 0）**：`applyInitQcu`/`applyEndQcu` 管理 scratch 生命周期；`applyCloverBistabCgQcu` 全 BiStabCG 求解；`applyCloverBistabCgDslashQcu` 单次奇偶预处理 dslash；**调用间必须递增 `_SET_INDEX_`**。

**C++ 后端集成（层 1）**：`applyMultigridRestrictQcu`/`applyMultigridProLongQcu`（层间转移）、`applyMultigridCoarseDslashQcu`（粗网格算子）。Hopping 矩阵打包为 `[2, 4, E, E, Xc, Yc, Zc, Tc]`（pm dir Eout Ein XYZT）。

**粗网格校正后 BiCGStab 状态重置（R3 fix）**：粗网格校正 `x = x + e_fine` 后残差 r 已变，必须重置全部 BiCGStab 状态：`r_tilde = r.clone()`，p/v/s/t 归零，rho_prev/alpha/omega 复位为 1.0。否则 `rho = vdot(r_tilde_old, r_new)` 无意义。

**收敛跟踪**：每轮记录两次 `r_norm`（粗网格校正前后）。plot 显示两者。

**调试工具**：`_verify_coarse_dslash(level, tol)` 对比 CUDA 粗网格 dslash 与 Python einsum 参考。

### 多线程多卡驱动

- `MultiGpuMultigrid`/`verify_multi_gpu_mg`：延迟导入（`__getattr__`，避免 tools._multigrid ↔ solver 循环依赖），
  单进程多线程一线程一卡运行 C++ Clover Multigrid；要求单 MPI rank（mpirun -np 1）。

### CUDA 混合路径同步与 breakdown 重启（2026-08-14）

- **同步**：`_matvec`/`_restrict_cuda`/`_prolong_cuda`/`_coarse_dslash_cuda` 的 C++ 调用后
  必须 `torch.cuda.synchronize()` —— C++ 私有流写固定输出缓冲，与 torch 默认流无
  跨流同步 → iter 0 breakdown（vdot(r_tilde,v)≈0）。此前依赖隐式时序，偶发失败。
- **BiCGStab 自动重启**：breakdown（rho/rtv/tts < 1e-20 或 alpha/omega 非有限）时
  保留当前 x/r，重置影子向量与系数后继续（不再抛 RuntimeError）；收敛不受影响
  （8x8x8x8 1L residual 8.7e-7、8x8x8x16 2L 7.8e-7 实测）。

### 多线程实例并行（Python multigrid）

- 每线程独立 `solver.multigrid` 实例（各持独立 params/set_ptrs/状态）可并行求解
  （2 线程 8x8x8x8 1L 实测 6.3e-7/9.5e-7 均收敛）。
- **前提**：线程池启动前主线程须预热 torch lazy 初始化（clover inverse 等，
  如 `torch.linalg.inv(torch.randn([4,4], device='cuda'))`），否则 worker 并发
  首次触发报 "lazy wrapper should be called at most once"。
  `MultiGpuMultigrid.solve()` 已内置该预热。

### 多实例资源管理（2026-08-14）

- `multigrid.end()` 释放 init 创建的 C++ LatticeSet（槽位 0.._max_set_index）；
  `__del__` 自动调用（先设备同步）。16³×32 大格子单实例 init 构建仍可能 OOM
  （null 向量逆迭代每向量一个 LatticeSet 的 pre-existing 设计，小格子正常）。
- **多实例循环必须显式**：`mg.end(); del mg; gc.collect()` —— 依赖隐式 GC 释放
  的时机不可控，会与后续实例的 C++ 操作并发干扰（实测 3 实例循环隐式 GC 时
  第 2 个实例残差 ~O(1)；显式 end+del+gc 后 3 实例全 9e-7 收敛）。
