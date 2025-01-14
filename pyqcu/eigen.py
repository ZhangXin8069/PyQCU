import cupy as cp
from time import perf_counter


class solver:
    def __init__(self, n, k, matvec, dtype, plan='small', degree=20, max_iter=200, tol=1e-6, min_eigen_value=0.0, max_eigen_value=1.0):
        self.n = n
        self.k = k
        self.matvec = matvec
        self.dtype = dtype
        self.plan = plan
        self.degree = degree
        self.max_iter = max_iter
        self.tol = tol
        self.min_eigen_value = min_eigen_value
        self.max_eigen_value = max_eigen_value
        self.buffers = {
            'v': cp.zeros(n, dtype=dtype),
            'w': cp.zeros(n, dtype=dtype),
            'temp': cp.zeros(n, dtype=dtype),
            'matvec_result': cp.zeros(n, dtype=dtype),
            'projection': cp.zeros(n, dtype=dtype)
        }
        self.memory_pool = cp.cuda.MemoryPool()
        cp.cuda.set_allocator(self.memory_pool.malloc)
        self.min_degree = 10
        self.max_degree = 100
        self.growth_factor = 1.5
        self.shrink_factor = 0.5

    def initialize_random_vector(self, v):
        v.real = cp.random.randn(self.n).astype(v.real.dtype)
        v.imag = cp.random.randn(self.n).astype(v.imag.dtype)
        norm = cp.linalg.norm(v)
        if norm > 0:
            cp.divide(v, norm, out=v)
        return v

    def chebyshev_filter(self, src, alpha, beta):
        """Modified Chebyshev filter to emphasize smallest eigenvalues"""
        buffers = self.buffers
        t_prev = buffers['temp']
        cp.copyto(t_prev, src)
        t_curr = buffers['w']
        # Modified coefficients to target small eigenvalues
        c = (beta + alpha) / 2
        e = (beta - alpha) / 2
        # First iteration
        mv_result = self.matvec(src)
        cp.subtract(mv_result, c * src, out=t_curr)
        cp.divide(t_curr, e, out=t_curr)
        for _ in range(1, self.degree):
            cp.copyto(t_prev, t_curr)
            t_next = buffers['v']
            mv_result = self.matvec(t_curr)
            cp.subtract(mv_result, c * t_curr, out=t_curr)
            cp.divide(t_curr, e, out=t_curr)
            cp.subtract(2*t_curr, t_prev, out=t_next)
            norm = float(cp.sqrt(cp.real(cp.vdot(t_next, t_next))))
            if norm > 1e-14:
                cp.divide(t_next, norm, out=t_next)
            cp.copyto(t_curr, t_next)
        return t_curr

    def orthogonalize(self, v, eigenvectors):
        if not eigenvectors:
            norm = cp.linalg.norm(v)
            if norm > 0:
                cp.divide(v, norm, out=v)
            return v
        projection = self.buffers['projection']
        projection.fill(0)
        chunk_size = 20
        for i in range(0, len(eigenvectors), chunk_size):
            chunk = eigenvectors[i:i+chunk_size]
            Q = cp.column_stack(chunk)
            chunk_proj = Q @ (Q.conj().T @ v)
            cp.add(projection, chunk_proj, out=projection)
            del Q, chunk_proj
            self.memory_pool.free_all_blocks()
        cp.subtract(v, projection, out=v)
        norm = cp.linalg.norm(v)
        if norm > 0:
            cp.divide(v, norm, out=v)
        return v

    def run(self):
        eigenvalues = []
        eigenvectors = []
        # Initial spectral range estimation
        alpha, beta = self.min_eigen_value, self.max_eigen_value
        for eigen_index in range(self.k):
            t0 = perf_counter()
            v = self.buffers['v']
            if eigen_index == 0:
                self.initialize_random_vector(v)
            else:
                # Initialize with previous eigenvectors plus perturbation
                cp.copyto(v, cp.zeros_like(v))
                for i in range(max(0, eigen_index-2), eigen_index):
                    rand_coeff = complex(cp.random.randn(), cp.random.randn())
                    cp.add(v, rand_coeff * eigenvectors[i], out=v)
                perturbation = self.buffers['temp']
                self.initialize_random_vector(perturbation)
                cp.add(v, 0.1 * perturbation, out=v)
                cp.divide(v, cp.linalg.norm(v), out=v)
            self.orthogonalize(v, eigenvectors)
            lambda_prev = float('inf')
            last_improvement = float('inf')
            for iter in range(self.max_iter):
                w = self.chebyshev_filter(v, alpha, beta)
                self.orthogonalize(w, eigenvectors)
                # Compute Rayleigh quotient
                lambda_curr = float(cp.real(cp.vdot(w, self.matvec(w))))
                rel_tol = abs(lambda_curr - lambda_prev) / abs(lambda_curr)
                if rel_tol < last_improvement:
                    last_improvement = rel_tol
                print(f"eigen_index: {eigen_index}, iter: {iter}, alpha: {alpha:.9f}, "
                      f"beta: {beta:.9f}, tol: {rel_tol:.6e}, lambda: {lambda_curr:.9f}, "
                      f"degree: {self.degree}")
                if rel_tol < self.tol:
                    break
                cp.copyto(v, w)
                lambda_prev = lambda_curr
                # Adaptive updates
                if iter % 5 == 0:
                    if rel_tol > 0.1:
                        self.degree = min(self.max_degree, int(
                            self.degree * self.growth_factor))
                    elif rel_tol < 0.01:
                        self.degree = max(self.min_degree, int(
                            self.degree * self.shrink_factor))
                    if self.plan == 'small':
                        # Update bounds to focus on remaining small eigenvalues
                        alpha = max(alpha, lambda_curr * 0.5)
                    if self.plan == 'large':
                        # Update bounds to focus on remaining large eigenvalues
                        beta = min(beta, lambda_curr * 2.0)
            if self.plan == 'small':
                # Update bounds to focus on remaining small eigenvalues
                beta = alpha * 2.0
            if self.plan == 'large':
                # Update bounds to focus on remaining large eigenvalues
                alpha = beta * 0.5
            # Sort eigenvalues and eigenvectors
            eigenvalues.append(lambda_curr)
            eigenvectors.append(w.copy())
            print(
                f"eigen_index: {eigen_index}, time: {perf_counter()-t0:.2f}s")
        return cp.array(eigenvalues), cp.array(eigenvectors)
