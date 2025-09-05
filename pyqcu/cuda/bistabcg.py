import cupy as cp
from pyqcu.cuda.linalg import dot, initialize_random_vector
from time import perf_counter
from typing import Callable
from pyqcu.cuda.define import cp_ndarray


def solver(b: cp_ndarray, matvec: Callable[[cp_ndarray], cp_ndarray], tol: float = 1e-6, max_iter: int = 1000, x0: cp_ndarray = None) -> cp_ndarray:
    n = b.size
    dtype = b.dtype
    buffers = {key: cp.zeros(n, dtype=dtype)
               for key in ['r', 'r_tilde', 'p', 'v', 's', 't', 'x']}
    x0 = None if x0 is None else x0.copy()
    x, r, r_tilde, p, v, s, t = buffers['x'], buffers['r'], buffers[
        'r_tilde'], buffers['p'], buffers['v'], buffers['s'], buffers['t']
    if x0 is not None:
        cp.copyto(x, x0)
    else:
        initialize_random_vector(x)
    r = b - matvec(x)
    cp.copyto(r_tilde, r)
    rho_prev = 1.0
    alpha = 1.0
    omega = 1.0
    start_time = perf_counter()
    iter_times = []
    for i in range(max_iter):
        iter_start_time = perf_counter()
        rho = dot(r_tilde, r)
        beta = (rho/rho_prev)*(alpha/omega)
        rho_prev = rho
        p = r+(p-v*omega)*beta
        r_norm2 = dot(r, r)
        v = matvec(p)
        alpha = rho / dot(r_tilde, v)
        s = r-v*alpha
        t = matvec(s)
        omega = dot(t, s)/dot(t, t)
        r = s-t*omega
        x = x+p*alpha+s*omega
        iter_time = perf_counter() - iter_start_time
        print(
            f"Iteration {i}: Residual = {r_norm2.real:.6e}, Time = {iter_time:.6f} s")
        iter_times.append(iter_time)
        if r_norm2.real < tol:
            print(
                f"Converged at iteration {i} with residual {r_norm2.real:.6e}")
            break
    total_time = perf_counter() - start_time
    avg_iter_time = sum(iter_times) / len(iter_times)
    print("\nPerformance Statistics:")
    print(f"Total time: {total_time:.6f} s")
    print(f"Average time per iteration: {avg_iter_time:.6f} s")
    cp.clear_memo()
    return x.copy()
