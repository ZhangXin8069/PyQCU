import cupy as cp
from time import perf_counter


class slover:
    def __init__(self, b, matvec, max_iter=1000, tol=1e-9, x0=None):
        print("Just use this like a function, not a class.")
        self.b = b.copy()
        self.n = b.size
        self.dtype = b.dtype
        self.matvec = matvec
        self.max_iter = max_iter
        self.tol = tol
        self.x0 = None if x0 is None else x0.copy()
        self.buffers = {
            'r': cp.zeros(self.n, dtype=self.dtype),
            'r_tilde': cp.zeros(self.n, dtype=self.dtype),
            'p': cp.zeros(self.n, dtype=self.dtype),
            'v': cp.zeros(self.n, dtype=self.dtype),
            's': cp.zeros(self.n, dtype=self.dtype),
            't': cp.zeros(self.n, dtype=self.dtype),
            'x': cp.zeros(self.n, dtype=self.dtype),
        }
        self.memory_pool = cp.cuda.MemoryPool()
        cp.cuda.set_allocator(self.memory_pool.malloc)

    def initialize_random_vector(self, v):
        v.real = cp.random.randn(self.n).astype(v.real.dtype)
        v.imag = cp.random.randn(self.n).astype(v.imag.dtype)
        return v

    def dot(self, x, y):
        """Dot product of two vectors."""
        return cp.sum(x.conj() * y)

    def run(self):
        """Solve the linear system Ax = b using the Conjugate Gradient method."""
        # Initialize variables
        x = self.buffers['x']
        if self.x0 is not None:
            cp.copyto(x, self.x0)
        else:
            self.initialize_random_vector(x)
        r = self.buffers['r']
        r_tilde = self.buffers['r_tilde']
        p = self.buffers['p']
        v = self.buffers['v']
        s = self.buffers['s']
        t = self.buffers['t']
        r = self.b - self.matvec(x)
        cp.copyto(r_tilde, r)
        rho_prev = 1.0
        alpha = 1.0
        omega = 1.0
        start_time = perf_counter()
        iter_times = []
        for i in range(self.max_iter):
            iter_start_time = perf_counter()
            rho = self.dot(r_tilde, r)
            beta = (rho/rho_prev)*(alpha/omega)
            rho_prev = rho
            p = r+(p-v*omega)*beta
            r_norm2 = self.dot(r, r)
            v = self.matvec(p)
            alpha = rho / self.dot(r_tilde, v)
            s = r-v*alpha
            t = self.matvec(s)
            omega = self.dot(t, s)/self.dot(t, t)
            r = s-t*omega
            x = x+p*alpha+s*omega
            iter_time = perf_counter() - iter_start_time
            print(
                f"Iteration {i}: Residual = {r_norm2.real:.6e}, Time = {iter_time:.6f} s")
            iter_times.append(iter_time)
            if r_norm2.real < self.tol:
                print(
                    f"Converged at iteration {i} with residual {r_norm2.real:.6e}")
                break
        total_time = perf_counter() - start_time
        avg_iter_time = sum(iter_times) / len(iter_times)
        print("\nPerformance Statistics:")
        print(f"Total time: {total_time:.6f} s")
        print(f"Average time per iteration: {avg_iter_time:.6f} s")
        dest = x.copy()
        self.buffers = None
        print("Memory released.")
        return dest
