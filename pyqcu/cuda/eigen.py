import cupy as cp
from pyqcu.cuda.linalg import initialize_random_vector, orthogonalize_against_vectors, chebyshev_filter
from time import perf_counter
from typing import Callable, Tuple
from pyqcu.cuda.define import cp_ndarray, cp_dtype


def solver(n: int, k: int, matvec: Callable[[cp_ndarray], cp_ndarray], dtype: cp_dtype, plan: str = 'small', degree: int = 20, max_iter: int = 200, tol: float = 1e-6, min_eigen_value: float = 0.0, max_eigen_value: float = 1.0) -> Tuple[cp_ndarray, cp_ndarray]:
    print("This function is just for positive definite matrix.")
    temp = cp.zeros(n, dtype=dtype)
    min_degree, max_degree = 10, 100
    growth_factor, shrink_factor = 1.5, 0.5
    eigenvalues, eigenvectors = [], []
    alpha, beta = min_eigen_value, max_eigen_value
    for eigen_index in range(k):
        t0, v = perf_counter(), cp.zeros(n, dtype=dtype)
        initialize_random_vector(v) if eigen_index == 0 else cp.copyto(v, sum(
            complex(cp.random.randn(), cp.random.randn()) * ev for ev in eigenvectors[max(0, eigen_index-2):eigen_index]) +
            0.1 * initialize_random_vector(temp))
        cp.divide(v, cp.linalg.norm(v), out=v)
        if eigen_index == 0:
            pass
        else:
            orthogonalize_against_vectors(v, eigenvectors)
        lambda_prev, last_improvement = float('inf'), float('inf')
        for iter in range(max_iter):
            w = chebyshev_filter(v, alpha, beta, matvec,
                                 degree=degree, tol=tol)
            if eigen_index == 0:
                pass
            else:
                orthogonalize_against_vectors(w, eigenvectors)
            lambda_curr = float(cp.real(cp.vdot(w, matvec(w))))
            rel_tol = abs(lambda_curr - lambda_prev) / abs(lambda_curr)
            last_improvement = min(last_improvement, rel_tol)
            print(
                f"eigen_index: {eigen_index}, iter: {iter}, alpha: {alpha:.9f}, beta: {beta:.9f}, tol: {rel_tol:.6e}, lambda: {lambda_curr:.9f}, degree: {degree}")
            if rel_tol < tol:
                break
            cp.copyto(v, w)
            lambda_prev = lambda_curr
            if iter % 5 == 0:
                degree = min(max_degree, int(degree * growth_factor)
                             ) if rel_tol > 0.1 else max(min_degree, int(degree * shrink_factor))
                alpha, beta = (max(alpha, lambda_curr * 0.5),
                               beta) if plan == 'small' else (alpha, min(beta, lambda_curr * 2.0))
        beta = alpha * 2.0 if plan == 'small' else beta
        alpha = beta * 0.5 if plan == 'large' else alpha
        eigenvalues = cp.append(eigenvalues, lambda_curr).astype(dtype)
        eigenvectors = cp.append(eigenvectors, w).astype(dtype).reshape(
            eigen_index+1, n)
        print(f"eigen_index: {eigen_index}, time: {perf_counter()-t0:.2f}s")
    cp.clear_memo()
    return eigenvalues.copy(), eigenvectors.copy()


def cupyx_solver(n: int, k: int, matvec: Callable[[cp_ndarray], cp_ndarray], dtype: cp_dtype, plan: str = 'SA', max_iter: int = 1e3, tol: float = 1e-6, v0: cp_ndarray = None) -> Tuple[cp_ndarray, cp_ndarray]:
    import cupyx.scipy.sparse.linalg as linalg
    print(f"dtype: {dtype}, plan: {plan}, max_iter: {max_iter}, tol: {tol}")
    eigenvalues, eigenvectors = linalg.eigsh(a=linalg.LinearOperator(
        (n, n), matvec=matvec, dtype=dtype), k=k, which=plan, tol=tol, maxiter=max_iter, v0=v0, return_eigenvectors=True)
    return cp.asarray(eigenvalues).copy(), cp.asarray(eigenvectors.T).copy()


def scipy_solver(n: int, k: int, matvec: Callable[[cp_ndarray], cp_ndarray], dtype: cp_dtype, plan: str = 'SM', max_iter=1e3, tol: float = 1e-6, v0: cp_ndarray = None) -> Tuple[cp_ndarray, cp_ndarray]:
    from scipy.sparse import linalg
    print(f"dtype: {dtype}, plan: {plan}, max_iter: {max_iter}, tol: {tol}")

    def _matvec(src):
        return matvec(cp.asarray(src)).get()
    eigenvalues, eigenvectors = linalg.eigs(A=linalg.LinearOperator(
        (n, n), matvec=_matvec, dtype=dtype), k=k, which=plan, tol=tol, maxiter=max_iter, v0=v0, return_eigenvectors=True)
    return cp.asarray(eigenvalues).copy(), cp.asarray(eigenvectors.T).copy()
