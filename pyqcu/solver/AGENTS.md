# AGENTS.md — pyqcu.solver

Dirac 方程 D ψ = η 的迭代求解器。

## 文件

| 文件 | 用途 |
|---|---|
| `_bistabcg.py` | BiCGStab(l) 求解器 |
| `_multigrid.py` | 自适应多重网格 (AMG) V-cycle 求解器，最细层 CUDA 加速 |
| `_gmres.py` | GMRES 求解器 — **占位 stub，未实现** |

## 导出 API

### `bistabcg(b, matvec, tol=1e-6, max_iter=1000, x0=None, if_rtol=False, verbose=True)`

标准 BiCGStab。`matvec` 为可调用对象 `matvec(src) → dest`。

**Breakdown 检测（R2 增加）**：`rho ≈ 0`、`vdot(r_tilde, v) ≈ 0`（pivot breakdown）、`vdot(t, t) ≈ 0` 时抛 `RuntimeError`。

**容差**：`if_rtol=True` 用 `tol * ||b||`；否则绝对 `tol`。

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
