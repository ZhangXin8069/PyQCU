# AGENTS.md — pyqcu.solver

Dirac 方程 D ψ = η 的迭代求解器。

## 文件

| 文件 | 用途 |
|---|---|
| `_bistabcg.py` | BiCGStab(l) 求解器 |
| `_multigrid.py` | 自适应多重网格（AMG）V-cycle，最细层 CUDA 加速 |
| `_gmres.py` | GMRES — **占位 stub，未实现** |

## bistabcg

`bistabcg(b, matvec, tol=1e-6, max_iter=1000, x0=None, if_rtol=False, verbose=True)` — 接受 callable `matvec(src) → dest`。

**Breakdown 检测**（R2）：`rho≈0`（r_tilde 与 r 正交）、`vdot(r_tilde,v)≈0`（pivot）、`vdot(t,t)≈0`（t 为零）时抛 `RuntimeError`。`if_rtol=True` 用 `tol·‖b‖`。

## multigrid 类

构造：`dtype_list`/`device_list`（逐层）、`U`/`clover_term`/`kappa`/`u_0`、`clover_ee_inv`+`clover_oo_inv` 齐备时启用 `with_cuda_qcu=True`、`min_size=4`、`max_level=4`、`mg_grid_size=[2,2,2,2]`、`dof_list=[12,24,24,...]`、`tol`/`max_iter`/`num_restart=5`、`num_convergence_sample=50`、`support_parity`。

| 方法 | 用途 |
|---|---|
| `init()` | 逆迭代生成 null 向量 → 局部正交化 → Galerkin 粗网格算子 |
| `solve(b, x0)` | 返回 `[4,3,Lx,Ly,Lz,Lt]` |
| `cycle(level)` | 递归 V-cycle：BiCGStab 平滑 → 限制残差 → 递归 → 延长校正 → 再平滑 |
| `adaptive(iter)` | 收敛停滞时降层（采样窗口内 ≥3 次） |
| `levels_back()` | 重置自适应状态 |
| `plot(save_path)` | 收敛历史绘图（仅 root rank） |

**执行层**：L0 可用 `applyCloverBistabCgQcu`/`applyCloverBistabCgDslashQcu`（`_SET_INDEX_` 必须在调用间递增）；L1 用 `applyMultigridRestrictQcu`/`ProLongQcu`/`CoarseDslashQcu`（跳跃矩阵打包为 `[2,4,E,E,Xc,Yc,Zc,Tc]`）；L2+ 纯 Python einsum。

**粗网格校正后必须重置 BiCGStab 状态（R3 fix）**：`r_tilde = r.clone()`、`p/v/s/t` 置零、`rho_prev/alpha/omega` 重置 1.0。

收敛追踪每迭代记两次 `r_norm`（校正前后）；`_verify_coarse_dslash(level, tol)` 用 Python einsum 参考交叉验证 CUDA 粗 dslash。
