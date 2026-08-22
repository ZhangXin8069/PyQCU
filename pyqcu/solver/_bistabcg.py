import torch
from time import perf_counter
from typing import Callable, Optional, List
from pyqcu import tools
import pyqcu.cann as _torch


def bistabcg(b: torch.Tensor, matvec: Callable[[torch.Tensor], torch.Tensor], tol: float = 1e-6, max_iter: int = 1000, x0:  Optional[torch.Tensor] = None, if_rtol: bool = False, verbose: bool = True, history: Optional[List[float]] = None) -> torch.Tensor:
    x = x0.clone() if x0 is not None else _torch.randn_like(b)
    r = b - matvec(x)
    r_norm = tools.norm(r)
    b_norm = tools.norm(b)
    if history is not None:
        history.append(float(r_norm))
    if if_rtol:
        _tol = b_norm*tol
    else:
        _tol = tol
    if verbose:
        print(f"PYQCU::SOLVER::BISTABCG:\n Norm of b:{b_norm}")
        print(f"PYQCU::SOLVER::BISTABCG:\n Norm of r:{r_norm}")
        print(f"PYQCU::SOLVER::BISTABCG:\n Norm of x0:{tools.norm(x)}")
    if r_norm < _tol:
        print("PYQCU::SOLVER::BISTABCG:\n x0 is just right!")
        return x
    r_tilde = r.clone()
    p = torch.zeros_like(b)
    v = torch.zeros_like(b)
    s = torch.zeros_like(b)
    t = torch.zeros_like(b)
    rho = torch.tensor(1.0, dtype=b.dtype, device=b.device)
    rho_prev = torch.tensor(1.0, dtype=b.dtype, device=b.device)
    alpha = torch.tensor(1.0, dtype=b.dtype, device=b.device)
    omega = torch.tensor(1.0, dtype=b.dtype, device=b.device)
    start_time = perf_counter()
    # BUGFIX 2026-07-28: always track iter_times to avoid ZeroDivisionError when verbose=False
    iter_times = []
    for i in range(max_iter):
        iter_start_time = perf_counter()
        # BUGFIX 2026-07-28 R2: BiCGStab breakdown detection.
        # rho ≈ 0 means r_tilde ⟂ r (method has lost orthogonality).
        # vdot(r_tilde, v) ≈ 0 is a pivot breakdown (division by zero).
        # vdot(t, t) ≈ 0 means t ≈ 0 (rare lucky breakdown, or stagnation).
        rho = tools.vdot(r_tilde, r)
        if abs(rho) < 1e-30:
            raise RuntimeError(
                f"BiCGStab breakdown at iter {i}: rho ≈ 0 "
                f"(r_tilde orthogonal to r). The method cannot continue.")
        beta = (rho / rho_prev) * (alpha / omega)
        rho_prev = rho
        p = r + beta * (p - omega * v)
        v = matvec(p)
        rtv = tools.vdot(r_tilde, v)
        if abs(rtv) < 1e-30:
            raise RuntimeError(
                f"BiCGStab breakdown at iter {i}: vdot(r_tilde, v) ≈ 0 "
                f"(pivot breakdown). The method cannot continue.")
        alpha = rho / rtv
        s = r - alpha * v
        t = matvec(s)
        tts = tools.vdot(t, t)
        if abs(tts) < 1e-30:
            raise RuntimeError(
                f"BiCGStab breakdown at iter {i}: vdot(t, t) ≈ 0 "
                f"(t is zero or near-zero). The method cannot continue.")
        omega = tools.vdot(t, s) / tts
        x = x + alpha * p + omega * s
        r = s - omega * t
        r_norm = tools.norm(r)
        iter_time = perf_counter() - iter_start_time
        iter_times.append(iter_time)
        if history is not None:
            history.append(float(r_norm))
        if verbose:
            # print(f"alpha,beta,omega:{alpha,beta,omega}\n")
            print(
                f"PYQCU::SOLVER::BISTABCG:\n Iteration {i}: Residual = {r_norm:.6e}, Time = {iter_time:.6f} s")
        if r_norm < _tol:
            if verbose:
                print(
                    f"PYQCU::SOLVER::BISTABCG:\n Converged at iteration {i} with residual {r_norm:.6e}")
            break
    else:
        print("PYQCU::SOLVER::BISTABCG:\n Warning: Maximum iterations reached, may not have converged")
    total_time = perf_counter() - start_time
    # BUGFIX 2026-07-28: guard against empty iter_times and only print stats when verbose
    if verbose and len(iter_times) > 0:
        avg_iter_time = sum(iter_times) / len(iter_times)
        print(f"PYQCU::SOLVER::BISTABCG:\n Performance Statistics:")
        print(f"PYQCU::SOLVER::BISTABCG:\n Total iterations: {len(iter_times)}")
        print(f"PYQCU::SOLVER::BISTABCG:\n Total time: {total_time:.6f} seconds")
        print(
            f"PYQCU::SOLVER::BISTABCG:\n Average time per iteration: {avg_iter_time:.6f} s")
        print(f"PYQCU::SOLVER::BISTABCG:\n Final residual: {r_norm:.2e}")
    return x


def bistabcg_history(b: torch.Tensor, matvec: Callable[[torch.Tensor], torch.Tensor], tol: float = 1e-6, max_iter: int = 2000, if_rtol: bool = False) -> List[float]:
    """零初始解复现 BiCGStab 逐迭代残差历史（返回 [||r0||, ||r1||, ...]）。

    整合自 logs/dev78_2/main.py::_bistabcg_history 与
    examples/qcu/dev73/mg_dev73_5_bench.py::bistabcg_history：
    用于给只输出收敛点的 C++ 求解路径补参考收敛曲线（同一 matvec、
    同一 BiCGStab 算法在 torch 上数学等价复现，零 C++ 改动）。
    breakdown 时打印提示并返回已收集的部分历史（不抛异常，画图友好）。
    """
    hist: List[float] = []
    try:
        bistabcg(b, matvec, tol=tol, max_iter=max_iter,
                 x0=_torch.zeros_like(b), if_rtol=if_rtol,
                 verbose=False, history=hist)
    except RuntimeError as e:
        print(f"PYQCU::SOLVER::BISTABCG_HISTORY:\n breakdown: partial history returned ({e})")
    return hist
