import torch
from time import perf_counter
from typing import Callable
from pyqcu import _torch, tools


def bistabcg(b: torch.Tensor, matvec: Callable[[torch.Tensor], torch.Tensor], tol: float = 1e-6, max_iter: int = 1000, x0: torch.Tensor = None, if_rtol: bool = False, verbose: bool = True) -> torch.Tensor:
    x = x0.clone() if x0 is not None else _torch.randn_like(b)
    r = b - matvec(x)
    r_norm = tools.norm(r)
    if if_rtol:
        _tol = tools.norm(b)*tol
    else:
        _tol = tol
    if verbose:
        print(f"PYQCU::SOLVER::BISTABCG:\n Norm of b:{tools.norm(b)}")
        print(f"PYQCU::SOLVER::BISTABCG:\n Norm of r:{r_norm}")
        print(f"PYQCU::SOLVER::BISTABCG:\n Norm of x0:{tools.norm(x)}")
    if r_norm < _tol:
        print("PYQCU::SOLVER::BISTABCG:\n x0 is just right!")
        return x.clone()
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
    iter_times = []
    for i in range(max_iter):
        iter_start_time = perf_counter()
        rho = tools.vdot(r_tilde, r)
        beta = (rho / rho_prev) * (alpha / omega)
        rho_prev = rho
        p = r + beta * (p - omega * v)
        v = matvec(p)
        alpha = rho / tools.vdot(r_tilde, v)
        s = r - alpha * v
        t = matvec(s)
        omega = tools.vdot(t, s) / \
            tools.vdot(t, t)
        x = x + alpha * p + omega * s
        r = s - omega * t
        r_norm = tools.norm(r)
        iter_time = perf_counter() - iter_start_time
        iter_times.append(iter_time)
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
    avg_iter_time = sum(iter_times) / len(iter_times)
    print(f"PYQCU::SOLVER::BISTABCG:\n Performance Statistics:")
    print(f"PYQCU::SOLVER::BISTABCG:\n Total iterations: {len(iter_times)}")
    print(f"PYQCU::SOLVER::BISTABCG:\n Total time: {total_time:.6f} seconds")
    print(
        f"PYQCU::SOLVER::BISTABCG:\n Average time per iteration: {avg_iter_time:.6f} s")
    print(f"PYQCU::SOLVER::BISTABCG:\n Final residual: {r_norm:.2e}")
    return x.clone()
