import torch
from time import perf_counter
from typing import Callable, Optional, List
from pyqcu import tools
import pyqcu.cann as _torch


def mr(b: torch.Tensor, matvec: Callable[[torch.Tensor], torch.Tensor],
       tol: float = 1e-6, max_iter: int = 1000, x0: Optional[torch.Tensor] = None,
       matvec_dag: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
       omega: float = 1.0, if_rtol: bool = False, verbose: bool = True,
       history: Optional[List[float]] = None) -> torch.Tensor:
    """MR（最小残差）求解器 — 非对称算子的非 Krylov 平滑/求解迭代。

    算法参考 quda lib/inv_mr_quda.cpp 思想（正规方程方向 + 步长阻尼）：
        r = b - A·x
        p  = A†·r                    （matvec_dag=None 时取 A 自共轭，如 Schur 奇偶算子）
        Ap = A·p
        α  = ω · <p,p>/<Ap,Ap>       （最小化 ||r - α·A·p||²，ω≤1 防过冲）
        x += α·p ;  r -= α·Ap

    每迭代一次 A 与一次 A†；无全局正交化，内存 O(1) 向量数，
    适合作为多重网格 smoother 或粗略预求解。

    Args:
        b: 右端项; matvec: A·v; matvec_dag: A†·v（None → 取 matvec）
        omega: 步长阻尼因子（quda mr_parameter，默认 1.0）
        tol/if_rtol/verbose/history: 同 solver.bistabcg 约定
    Returns:
        解张量（形状同 b）

    Breakdown：`<p,p> ≈ 0`（残差已零）或 `<Ap,Ap> ≈ 0` 时抛 RuntimeError。
    """
    dag = matvec if matvec_dag is None else matvec_dag
    x = x0.clone() if x0 is not None else _torch.zeros_like(b)
    r = b - matvec(x)
    r_norm = tools.norm(r)
    b_norm = tools.norm(b)
    if history is not None:
        history.append(float(r_norm))
    _tol = b_norm * tol if if_rtol else tol
    if verbose:
        print(f"PYQCU::SOLVER::MR:\n Norm of b:{b_norm}")
        print(f"PYQCU::SOLVER::MR:\n Norm of r:{r_norm}")
    if r_norm < _tol:
        print("PYQCU::SOLVER::MR:\n x0 is just right!")
        return x
    start_time = perf_counter()
    converged = False
    for i in range(max_iter):
        p = dag(r)
        pp = tools.vdot(p, p)
        if abs(pp) < 1e-30:
            raise RuntimeError(
                f"MR breakdown at iter {i}: <p,p> ≈ 0 (residual already zero).")
        Ap = matvec(p)
        ApAp = tools.vdot(Ap, Ap)
        if abs(ApAp) < 1e-30:
            raise RuntimeError(
                f"MR breakdown at iter {i}: <Ap,Ap> ≈ 0 (A·p is zero).")
        alpha = omega * (pp / ApAp).real
        x = x + alpha * p
        r = r - alpha * Ap
        r_norm = tools.norm(r)
        if history is not None:
            history.append(float(r_norm))
        if verbose:
            print(f"PYQCU::SOLVER::MR:\n Iteration {i}: Residual = {r_norm:.6e}")
        if r_norm < _tol:
            if verbose:
                print(f"PYQCU::SOLVER::MR:\n Converged at iteration {i} with residual {r_norm:.6e}")
            converged = True
            break
    total_time = perf_counter() - start_time
    if verbose:
        if not converged:
            print("PYQCU::SOLVER::MR:\n Warning: Maximum iterations reached, may not have converged")
        print(f"PYQCU::SOLVER::MR:\n Total time: {total_time:.6f} seconds")
        print(f"PYQCU::SOLVER::MR:\n Final residual: {r_norm:.2e}")
    return x
