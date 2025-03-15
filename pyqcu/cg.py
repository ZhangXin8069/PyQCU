import cupy as cp
from time import perf_counter


def slover(b, matvec, max_iter=1000, tol=1e-9, x0=None):
    n = b.size
    dtype = b.dtype
    buffers = {key: cp.zeros(n, dtype=dtype) for key in ['r', 'p', 'v', 'x']}
    x0 = None if x0 is None else x0.copy()
    def initialize_random_vector(v):
        v.real, v.imag = cp.random.randn(n).astype(
            v.real.dtype), cp.random.randn(n).astype(v.imag.dtype)
        norm = cp.linalg.norm(v)
        if norm > 0:
            cp.divide(v, norm, out=v)
        return v
    def dot(x, y):
        return cp.sum(x.conj() * y)
    x, r, p, v = buffers['x'], buffers['r'], buffers['p'], buffers['v']
    if x0 is not None:
        cp.copyto(x, x0)
    else:
        initialize_random_vector(x)
    r = b - matvec(x)
    cp.copyto(p, r)
    rho = dot(r, r)
    rho_prev = 1.0
    start_time = perf_counter()
    iter_times = []
    for i in range(max_iter):
        iter_start_time = perf_counter()
        v = matvec(p)
        rho_prev = rho
        alpha = rho / dot(p, v)
        r -= alpha * v
        x += alpha * p
        rho = dot(r, r)
        beta = rho / rho_prev
        p = r + beta * p
        iter_time = perf_counter() - iter_start_time
        print(
            f"Iteration {i}: Residual = {rho.real:.6e}, Time = {iter_time:.6f} s")
        iter_times.append(iter_time)
        if rho.real < tol:
            print(
                f"Converged at iteration {i} with residual {rho.real:.6e}")
            break
    total_time = perf_counter() - start_time
    avg_iter_time = sum(iter_times) / len(iter_times)
    print("\nPerformance Statistics:")
    print(f"Total time: {total_time:.6f} s")
    print(f"Average time per iteration: {avg_iter_time:.6f} s")
    return x.copy()
