import cupy as cp
from time import perf_counter


class slover:
    def __init__(self, b, matvec, max_iter=1000, tol=1e-9):
        self.b = b
        self.n = b.size
        self.dtype = b.dtype
        self.matvec = matvec
        self.max_iter = max_iter
        self.tol = tol
        self.buffers = {
            'r': cp.zeros(self.n, dtype=self.dtype),
            'p': cp.zeros(self.n, dtype=self.dtype),
            'v': cp.zeros(self.n, dtype=self.dtype),
            'x': cp.zeros(self.n, dtype=self.dtype),
        }
        self.memory_pool = cp.cuda.MemoryPool()
        cp.cuda.set_allocator(self.memory_pool.malloc)

    def initialize_random_vector(self, v):
        v.real = cp.random.randn(self.n).astype(v.real.dtype)
        v.imag = cp.random.randn(self.n).astype(v.imag.dtype)
        norm = cp.linalg.norm(v)
        if norm > 0:
            cp.divide(v, norm, out=v)
        return v

    def dot(self, x, y):
        """Dot product of two vectors."""
        return cp.sum(x.conj() * y)

    def run(self):
        """Solve the linear system Ax = b using the Conjugate Gradient method."""
        # Initialize variables
        x = self.buffers['x']
        self.initialize_random_vector(x)
        r = self.buffers['r']
        p = self.buffers['p']
        v = self.buffers['v']
        r = self.b - self.matvec(x)
        cp.copyto(p, r)
        rho = self.dot(r, r)
        rho_prev = 1.0
        start_time = perf_counter()
        iter_times = []
        for i in range(self.max_iter):
            iter_start_time = perf_counter()
            v = self.matvec(p)
            rho_prev = rho
            alpha = rho / self.dot(p, v)
            r -= alpha * v
            x += alpha * p
            rho = self.dot(r, r)
            beta = rho / rho_prev
            p = r + beta * p
            iter_time = perf_counter() - iter_start_time
            print(
                f"Iteration {i}: Residual = {rho.real:.6e}, Time = {iter_time:.6f} s")
            iter_times.append(iter_time)
            if rho.real < self.tol:
                print(
                    f"Converged at iteration {i} with residual {rho.real:.6e}")
                break
        total_time = perf_counter() - start_time
        avg_iter_time = sum(iter_times) / len(iter_times)
        print("\nPerformance Statistics:")
        print(f"Total time: {total_time:.6f} s")
        print(f"Average time per iteration: {avg_iter_time:.6f} s")
        return x.copy()