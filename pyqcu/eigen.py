import cupy as cp
from time import perf_counter

def solver(n, k, matvec, dtype, plan='small', degree=20, max_iter=200, tol=1e-6, min_eigen_value=0.0, max_eigen_value=1.0):
    buffers = {key: cp.zeros(n, dtype=dtype) for key in ['v', 'w', 'temp', 'matvec_result', 'projection']}
    min_degree, max_degree = 10, 100
    growth_factor, shrink_factor = 1.5, 0.5
    
    def initialize_random_vector(v):
        v.real, v.imag = cp.random.randn(n).astype(v.real.dtype), cp.random.randn(n).astype(v.imag.dtype)
        norm = cp.linalg.norm(v)
        if norm > 0:
            cp.divide(v, norm, out=v)
        return v
    
    def chebyshev_filter(src, alpha, beta):
        t_prev, t_curr, t_next = buffers['temp'], buffers['w'], buffers['v']
        c, e = (beta + alpha) / 2, (beta - alpha) / 2
        cp.copyto(t_prev, src)
        cp.subtract(matvec(src), c * src, out=t_curr)
        cp.divide(t_curr, e, out=t_curr)
        for _ in range(1, degree):
            cp.copyto(t_prev, t_curr)
            cp.subtract(matvec(t_curr), c * t_curr, out=t_curr)
            cp.divide(t_curr, e, out=t_curr)
            cp.subtract(2 * t_curr, t_prev, out=t_next)
            norm = float(cp.linalg.norm(t_next))
            if norm > 1e-14:
                cp.divide(t_next, norm, out=t_next)
            cp.copyto(t_curr, t_next)
        return t_curr
    
    def orthogonalize(v, eigenvectors):
        if not eigenvectors:
            norm = cp.linalg.norm(v)
            if norm > 0:
                cp.divide(v, norm, out=v)
            return v
        projection = buffers['projection']
        projection.fill(0)
        chunk_size = 20
        for i in range(0, len(eigenvectors), chunk_size):
            Q = cp.column_stack(eigenvectors[i:i+chunk_size])
            cp.add(projection, Q @ (Q.conj().T @ v), out=projection)
        cp.subtract(v, projection, out=v)
        norm = cp.linalg.norm(v)
        if norm > 0:
            cp.divide(v, norm, out=v)
        return v
    
    eigenvalues, eigenvectors = [], []
    alpha, beta = min_eigen_value, max_eigen_value
    for eigen_index in range(k):
        t0, v = perf_counter(), buffers['v']
        initialize_random_vector(v) if eigen_index == 0 else cp.copyto(v, sum(
            complex(cp.random.randn(), cp.random.randn()) * ev for ev in eigenvectors[max(0, eigen_index-2):eigen_index]) +
            0.1 * initialize_random_vector(buffers['temp']))
        cp.divide(v, cp.linalg.norm(v), out=v)
        orthogonalize(v, eigenvectors)
        lambda_prev, last_improvement = float('inf'), float('inf')
        for iter in range(max_iter):
            w = chebyshev_filter(v, alpha, beta)
            orthogonalize(w, eigenvectors)
            lambda_curr = float(cp.real(cp.vdot(w, matvec(w))))
            rel_tol = abs(lambda_curr - lambda_prev) / abs(lambda_curr)
            last_improvement = min(last_improvement, rel_tol)
            print(f"eigen_index: {eigen_index}, iter: {iter}, alpha: {alpha:.9f}, beta: {beta:.9f}, tol: {rel_tol:.6e}, lambda: {lambda_curr:.9f}, degree: {degree}")
            if rel_tol < tol:
                break
            cp.copyto(v, w)
            lambda_prev = lambda_curr
            if iter % 5 == 0:
                degree = min(max_degree, int(degree * growth_factor)) if rel_tol > 0.1 else max(min_degree, int(degree * shrink_factor))
                alpha, beta = (max(alpha, lambda_curr * 0.5), beta) if plan == 'small' else (alpha, min(beta, lambda_curr * 2.0))
        beta = alpha * 2.0 if plan == 'small' else beta
        alpha = beta * 0.5 if plan == 'large' else alpha
        eigenvalues.append(lambda_curr)
        eigenvectors.append(w.copy())
        print(f"eigen_index: {eigen_index}, time: {perf_counter()-t0:.2f}s")
    return cp.array(eigenvalues, dtype=dtype), cp.array(eigenvectors, dtype=dtype)
